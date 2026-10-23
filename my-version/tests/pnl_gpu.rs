//! #305: exact type-7 threshold selection and the shared valid-row mask.
//!
//! The threshold is a discrete input. These tests pin the properties that make
//! a reported position independent of the implementation that selected `theta`:
//!
//! * exact radix select is bit-identical to the CPU sort on 1e6 scores with
//!   duplicates, NaNs and `±0`;
//! * `score == theta` is flat (strict `>`);
//! * the shared mask drops a row whose score **or** target is non-finite, and
//!   reproduces the six-row NaN-target difference;
//! * a screening-only (sketch) threshold cannot place reported positions;
//! * batch composition and simulated thread count do not change `theta` or the
//!   position vector.
//!
//! Device tests skip cleanly when the crate was built without CUDA or no device
//! is present.

use rcsr::data;
use rcsr::trading_gpu::{
    long_position, long_positions, type7_quantile_f32, type7_quantile_f32_parts,
    type7_quantile_f32_sorted_reference, type7_quantile_f64, Threshold, ValidMask,
};
use rcsr::walkforward::{self, EvalSpec, Window};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Deterministic LCG; no dependency on a rand crate.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 24 bits of mantissa, mapped into [-1, 1).
        let u = (self.0 >> 40) as u32;
        (u as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn sample_scores(n: usize) -> Vec<f32> {
    let mut rng = Lcg(0x305_0305_0305_0305);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = rng.next_f32();
        // A quantised grid creates exact duplicates; every 10 007th row is NaN
        // and every 20 011th row exercises signed zero.
        let v = if i % 10_007 == 3 {
            f32::NAN
        } else if i % 20_011 == 7 {
            if i % 40_022 == 7 {
                -0.0
            } else {
                0.0
            }
        } else if i % 50_003 == 11 {
            (v * 8.0).round() / 8.0
        } else {
            v
        };
        out.push(v);
    }
    out
}

fn table(rows: &[(f64, &str, f64, f64)]) -> data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    for (ts, sym, x, y) in rows {
        s.push_str(&format!("{},{},{},{}\n", ts, sym, x, y));
    }
    data::parse_table(&s).unwrap()
}

fn window(label: &str, a: f64, b: f64, c: f64, d: f64) -> Window {
    Window {
        label: label.to_string(),
        train_start: a,
        train_end: b,
        test_start: c,
        test_end: d,
    }
}

fn spec(expr: &str, windows: Vec<Window>) -> EvalSpec {
    EvalSpec {
        expression: expr.to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: None,
        windows,
        horizon: 0.0,
        embargo: None,
        cost_bps: 0.0,
        threshold_quantile: 0.0,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(4.0),
        candidates: Vec::new(),
        pool_reset_flat: false,
        allow_lockbox: false,
    }
}

fn positions_of(scores: &[f32], theta: f64) -> Vec<u8> {
    scores
        .iter()
        .map(|&v| long_position(v as f64, theta))
        .collect()
}

// ---------------------------------------------------------------------------
// Exactness
// ---------------------------------------------------------------------------

/// Report 03 §8 test 8: 1e6 scores with duplicates, NaNs and signed zero.
#[test]
fn exact_type7_is_bit_identical_to_the_cpu_sort_at_one_million_rows() {
    let scores = sample_scores(1_000_000);
    for &q in &[0.0, 0.01, 0.25, 0.5, 0.7000001, 0.7, 0.9, 0.999, 1.0] {
        let exact = type7_quantile_f32(&scores, q);
        let sorted = type7_quantile_f32_sorted_reference(&scores, q);
        assert_eq!(
            exact.theta.to_bits(),
            sorted.theta.to_bits(),
            "theta differs at q={q}: {} vs {}",
            exact.theta,
            sorted.theta
        );
        assert_eq!(exact.n, sorted.n);
        assert_eq!(exact.lo, sorted.lo);
        assert_eq!(exact.hi, sorted.hi);
        assert_eq!(
            positions_of(&scores, exact.theta),
            positions_of(&scores, sorted.theta),
            "position vector differs at q={q}"
        );
    }
}

#[test]
fn score_equal_to_theta_is_flat() {
    let scores = [1.0f32, 2.0, 3.0];
    let theta = type7_quantile_f32(&scores, 0.5);
    assert_eq!(theta.theta, 2.0);
    // Row 1 is equal to theta and therefore flat; row 2 is strictly above.
    assert_eq!(positions_of(&scores, theta.theta), vec![0, 0, 1]);

    let f64_scores = [1.0f64, 2.0, 3.0];
    let theta = type7_quantile_f64(&f64_scores, 0.5);
    assert_eq!(theta.theta, 2.0);
    assert_eq!(long_position(2.0, theta.theta), 0);
    assert_eq!(long_position(2.000_000_000_000_001, theta.theta), 1);
}

#[test]
fn signed_zero_is_canonicalised_without_changing_the_order_statistic() {
    let scores = [-0.0f32, 0.0, -0.0, 1.0, 0.0];
    for q in [0.0, 0.2, 0.5, 1.0] {
        let exact = type7_quantile_f32(&scores, q);
        let sorted = type7_quantile_f32_sorted_reference(&scores, q);
        assert_eq!(exact.theta.to_bits(), sorted.theta.to_bits(), "q={q}");
    }
    // The minimum is a zero; it is reported as `+0.0`, not `-0.0`, so the
    // canonical bits do not depend on which signed zero happened to sort first.
    let exact = type7_quantile_f32(&scores, 0.0);
    assert_eq!(exact.theta.to_bits(), 0.0f64.to_bits());
}

// ---------------------------------------------------------------------------
// Shared valid-row mask
// ---------------------------------------------------------------------------

#[test]
fn shared_mask_drops_score_or_target_non_finites() {
    let scores = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let targets = [0.1, f64::NAN, 0.3, f64::INFINITY, -0.2];
    let mask = ValidMask::from_finites(&scores, &targets);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.valid_count(), 3);
    assert_eq!(mask.dropped_count(), 2);
    assert_eq!(mask.indices().collect::<Vec<_>>(), vec![0, 2, 4]);
    // The score-only mask is the weaker `position_for` contract.
    assert_eq!(ValidMask::from_scores(&scores).valid_count(), 5);
}

/// The pinned-cell difference: six rows have a finite score and a NaN target.
/// `position_for` would score them; the shared mask drops exactly those rows.
#[test]
fn walkforward_reproduces_the_six_row_nan_target_difference() {
    let mut rows: Vec<(f64, &str, f64, f64)> = Vec::new();
    for i in 0..8 {
        rows.push((i as f64, "A", i as f64, 0.0));
    }
    // Twelve test rows: x = 1..12, six of them with a NaN target.
    for i in 0..12 {
        let target = if i % 2 == 1 { f64::NAN } else { 0.01 };
        rows.push((8.0 + i as f64, "A", (i + 1) as f64, target));
    }
    let table = table(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 8.0, 8.0, 20.0)]);
    s.threshold_quantile = 0.0;
    let report = walkforward::run_table(&table, &s, "test").unwrap();
    let w = &report.windows[0];
    assert_eq!(w.threshold, 0.0);
    assert_eq!(w.threshold_source, "train_type7");
    assert_eq!(w.dropped_rows, 6, "exactly the NaN-target rows are dropped");
    assert_eq!(w.test_rows, 6, "P&L denominator excludes the six rows");
    assert!((w.metrics.exposure - 1.0).abs() < 1e-15);

    let test_scores: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let test_targets: Vec<f64> = (0..12)
        .map(|i| if i % 2 == 1 { f64::NAN } else { 0.01 })
        .collect();
    let shared = ValidMask::from_finites(&test_scores, &test_targets);
    let score_only = ValidMask::from_scores(&test_scores);
    assert_eq!(shared.valid_count(), 6);
    assert_eq!(score_only.valid_count(), 12);
    assert_eq!(score_only.valid_count() - shared.valid_count(), 6);
}

// ---------------------------------------------------------------------------
// Provenance: exact vs fixed vs screening
// ---------------------------------------------------------------------------

#[test]
fn fixed_threshold_is_reported_and_score_equal_is_flat() {
    let rows: Vec<(f64, &str, f64, f64)> = (0..8)
        .map(|i| (i as f64, "A", i as f64, if i >= 4 { 0.01 } else { 0.0 }))
        .collect();
    let table = table(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    s.threshold_fixed = Some(5.0); // test x = 4,5,6,7: equality at row 5
    let report = walkforward::run_table(&table, &s, "test").unwrap();
    let w = &report.windows[0];
    assert_eq!(w.threshold, 5.0);
    assert_eq!(w.threshold_source, "fixed");
    // 4 and 5 are flat; 6 and 7 are long.
    assert!((w.metrics.exposure - 0.5).abs() < 1e-15, "{:?}", w.metrics);
    let json = report.json();
    assert!(json.contains("\"source\":\"fixed\""), "{json}");
    assert!(json.contains("\"fixed\":5"), "{json}");
}

#[test]
fn screening_threshold_cannot_place_reported_positions() {
    let mut mask = ValidMask::new(4);
    for i in 0..4 {
        mask.set(i, true);
    }
    let scores = [0.5, 1.0, 1.5, 2.0];
    let sketch = Threshold::screening_only(1.0);
    assert!(sketch.is_screening_only());
    assert!(!sketch.is_report_eligible());
    assert!(long_positions(&scores, &mask, &sketch).is_err());

    let exact = Threshold::train_type7(&scores, 0.5);
    assert!(exact.is_report_eligible());
    assert_eq!(exact.source().as_str(), "train_type7");
    assert_eq!(
        long_positions(&scores, &mask, &exact).unwrap(),
        vec![0, 0, 1, 1]
    );
}

// ---------------------------------------------------------------------------
// Batch composition and thread count
// ---------------------------------------------------------------------------

#[test]
fn theta_and_positions_do_not_depend_on_batch_or_thread_partition() {
    let scores = sample_scores(200_000);
    let whole = type7_quantile_f32(&scores, 0.7);
    let want = positions_of(&scores, whole.theta);

    // Chunk boundaries: 4 k, 64 k and an odd size.
    for chunk in [4096usize, 65_536, 7777] {
        let parts: Vec<&[f32]> = scores.chunks(chunk).collect();
        let got = type7_quantile_f32_parts(&parts, 0.7);
        assert_eq!(got.theta.to_bits(), whole.theta.to_bits(), "chunk={chunk}");
        assert_eq!(positions_of(&scores, got.theta), want);
    }

    // Simulated thread striping: thread t owns rows t, t+T, t+2T, ...
    for threads in [128usize, 256, 512] {
        let owned: Vec<Vec<f32>> = (0..threads)
            .map(|t| {
                scores
                    .iter()
                    .skip(t)
                    .step_by(threads)
                    .copied()
                    .collect::<Vec<f32>>()
            })
            .collect();
        let parts: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
        let got = type7_quantile_f32_parts(&parts, 0.7);
        assert_eq!(
            got.theta.to_bits(),
            whole.theta.to_bits(),
            "threads={threads}"
        );
        assert_eq!(positions_of(&scores, got.theta), want);
    }

    // Part order must be irrelevant too: rank selection is a multiset
    // operation.
    let mut parts: Vec<&[f32]> = scores.chunks(10_000).collect();
    parts.reverse();
    let got = type7_quantile_f32_parts(&parts, 0.7);
    assert_eq!(got.theta.to_bits(), whole.theta.to_bits());
}

// ---------------------------------------------------------------------------
// Device parity (skip when CUDA is absent)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn device_threshold_and_mask_match_the_cpu_bit_for_bit() {
    use rcsr::trading_gpu::{
        type7_quantile_f32_device, type7_quantile_f32_device_threads, valid_mask_f32_device,
    };

    if !rcsr::trading_gpu::device_available() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let scores = sample_scores(1_000_000);
    for &q in &[0.0, 0.25, 0.5, 0.7, 0.73, 1.0] {
        let cpu = type7_quantile_f32(&scores, q);
        let dev = type7_quantile_f32_device(&scores, q).expect("device threshold");
        assert_eq!(cpu.theta.to_bits(), dev.theta.to_bits(), "q={q}");
        assert_eq!(cpu.n, dev.n);
        assert_eq!(
            positions_of(&scores, cpu.theta),
            positions_of(&scores, dev.theta)
        );
        // 128/256/512 threads must select the same two ranks.
        for threads in [128, 256, 512] {
            let t = type7_quantile_f32_device_threads(&scores, q, threads)
                .expect("device threshold");
            assert_eq!(
                cpu.theta.to_bits(),
                t.theta.to_bits(),
                "q={q} threads={threads}"
            );
        }
    }

    // Batch composition: reversing the input cannot change the multiset.
    let reversed: Vec<f32> = scores.iter().rev().copied().collect();
    let cpu = type7_quantile_f32(&scores, 0.7);
    let rev = type7_quantile_f32_device(&reversed, 0.7).expect("device threshold");
    assert_eq!(cpu.theta.to_bits(), rev.theta.to_bits(), "reversed input");

    // Mask parity: same packed words, same count.
    let n = 100_000usize;
    let scores = sample_scores(n);
    let targets: Vec<f32> = (0..n)
        .map(|i| {
            if i % 97 == 5 {
                f32::NAN
            } else if i % 193 == 7 {
                f32::INFINITY
            } else {
                (i as f32).sin()
            }
        })
        .collect();
    let cpu_scores: Vec<f64> = scores.iter().map(|&v| v as f64).collect();
    let cpu_targets: Vec<f64> = targets.iter().map(|&v| v as f64).collect();
    let cpu_mask = ValidMask::from_finites(&cpu_scores, &cpu_targets);
    let dev_words = valid_mask_f32_device(&scores, &targets).expect("device mask");
    assert_eq!(dev_words, cpu_mask.words());
    assert_eq!(
        ValidMask::from_words(&dev_words, n).valid_count(),
        cpu_mask.valid_count()
    );
}

// ---------------------------------------------------------------------------
// Paired select measurement (run with `cargo test --release -- --ignored`)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn threshold_select_throughput_at_one_and_three_point_nine_million_rows() {
    use std::time::Instant;
    for n in [1_000_000usize, 3_943_856] {
        let scores = sample_scores(n);
        let t0 = Instant::now();
        let exact = type7_quantile_f32(&scores, 0.7);
        let exact_ns = t0.elapsed().as_nanos();
        let t0 = Instant::now();
        let sorted = type7_quantile_f32_sorted_reference(&scores, 0.7);
        let sort_ns = t0.elapsed().as_nanos();
        assert_eq!(exact.theta.to_bits(), sorted.theta.to_bits());
        eprintln!(
            "threshold n={n}: radix {:.2} ns/row, sort {:.2} ns/row, ratio {:.2}x",
            exact_ns as f64 / n as f64,
            sort_ns as f64 / n as f64,
            sort_ns as f64 / exact_ns as f64
        );
    }
}
