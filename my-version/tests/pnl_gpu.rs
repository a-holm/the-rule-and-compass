//! #304: fused long-only `{0, +1}` P&L reduction.
//!
//! The arithmetic fixture, round-trip/final-exit rules, long-only invariant,
//! cash/buy-and-hold benchmarks, canonical chunk invariance and — with
//! `--features cuda` — device/CPU parity and thread/chunk/padding invariance.
//!
//! Run the device tests with:
//!
//! ```sh
//! cargo test --release --features cuda --test pnl_gpu
//! ```
//!
//! The performance measurement is ignored by default:
//!
//! ```sh
//! cargo test --release --features cuda --test pnl_gpu -- --ignored --nocapture bench_pnl_vs_cpu
//! ```

use rcsr::trading_gpu::{
    band_positions, band_scan, benchmarks, check_group_contiguous, check_long_only,
    cooldown_scan, dp_bound, dp_scan, reduce, reduce_chunked, threshold_positions, BandParams,
    Carry2, DpCarry, PnlParams, PnlTotals, MAX_TILE_CANDIDATES,
};
use rcsr::data;
use rcsr::trading_gpu::{
    long_position, long_positions, type7_quantile_f32, type7_quantile_f32_parts,
    type7_quantile_f32_sorted_reference, type7_quantile_f64, Threshold, ValidMask,
};
use rcsr::walkforward::{self, EvalSpec, Window};

fn fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    (
        vec![0.5, -0.1, 0.2, -0.3, 0.4],
        vec![0.01, -0.02, 0.03, 0.0, 0.015],
        vec![0, 0, 0, 0, 0],
    )
}

#[test]
fn five_row_fixture_to_1e12() {
    let (scores, returns, groups) = fixture();
    let params = PnlParams {
        threshold: 0.0,
        per_side_bps: 10.0,
        ..PnlParams::default()
    };
    let r = reduce(&scores, &returns, &groups, &params).unwrap();
    // net = 0.055 gross - 0.005 round-trip cost - 0.001 final exit.
    assert!((r.strategy.net - 0.049).abs() <= 1e-12, "net {}", r.strategy.net);
    assert!((r.strategy.gross - 0.055).abs() <= 1e-12);
    assert!((r.strategy.cost - 0.006).abs() <= 1e-12);
    assert_eq!(r.strategy.rows, 5);
    assert_eq!(r.strategy.trades, 5);
    assert_eq!(r.strategy.entries, 3);
    assert_eq!(r.strategy.exits, 3);
    assert_eq!(r.strategy.final_exits, 1);
    assert_eq!(r.strategy.long_rows, 3);
    assert!((r.strategy.exposure() - 3.0 / 5.0).abs() <= 1e-15);
}

#[test]
fn round_trip_is_two_sides_and_final_exit_is_one_extra_side() {
    let params = PnlParams {
        threshold: 0.0,
        per_side_bps: 10.0,
        ..PnlParams::default()
    };
    // Enter on row 0, leave on row 1: entry + rule exit = 2s.
    let r = reduce(&[1.0, -1.0], &[0.0, 0.0], &[0, 0], &params).unwrap();
    assert!((r.strategy.cost - 0.002).abs() <= 1e-15);
    assert_eq!(r.strategy.entries, 1);
    assert_eq!(r.strategy.exits, 1);
    assert_eq!(r.strategy.final_exits, 0);

    // Hold to the group end: entry + final exit = 2s, exactly one extra side
    // on the last row compared with the rule-exit charge.
    let held = reduce(&[1.0, 1.0], &[0.0, 0.0], &[0, 0], &params).unwrap();
    assert!((held.strategy.cost - 0.002).abs() <= 1e-15);
    assert_eq!(held.strategy.entries, 1);
    assert_eq!(held.strategy.exits, 1);
    assert_eq!(held.strategy.final_exits, 1);

    // Flat is free.
    let flat = reduce(&[-1.0, -1.0], &[0.0, 0.0], &[0, 0], &params).unwrap();
    assert_eq!(flat.strategy.cost, 0.0);
    assert_eq!(flat.strategy.net, 0.0);
}

#[test]
fn positions_are_long_only_and_negative_is_rejected() {
    let (scores, returns, groups) = fixture();
    let params = PnlParams::default();
    let terms = rcsr::trading_gpu::row_terms(&scores, &returns, &groups, &params).unwrap();
    for t in &terms {
        assert!(t.position == 0.0 || t.position == 1.0);
    }
    check_long_only(&terms.iter().map(|t| t.position).collect::<Vec<_>>()).unwrap();
    assert!(check_long_only(&[0.0, -1.0]).is_err());
    assert!(check_long_only(&[0.5]).is_err());
}

#[test]
fn groups_must_be_contiguous() {
    assert!(check_group_contiguous(&[0, 0, 1, 1]).is_ok());
    assert!(check_group_contiguous(&[0, 1, 0]).is_err());
    let params = PnlParams::default();
    let err = reduce(&[1.0, 1.0, 1.0], &[0.0; 3], &[0, 1, 0], &params).unwrap_err();
    assert!(err.contains("reappears"));
}

#[test]
fn cash_and_buy_and_hold_benchmarks() {
    let returns = [0.01, -0.02, 0.03, 0.0, 0.015];
    let groups = [0u64, 0, 0, 1, 1];
    let (cash, bh) = benchmarks(&returns, &groups, 0.001, 0.0);
    assert_eq!(cash.rows, 5);
    assert_eq!(cash.net, 0.0);
    assert_eq!(cash.cost, 0.0);
    assert_eq!(cash.long_rows, 0);
    // Buy-and-hold: sum(r) - 2s per group.
    let expect = returns.iter().sum::<f64>() - 0.004;
    assert!((bh.net - expect).abs() <= 1e-15, "bh {}", bh.net);
    assert_eq!(bh.entries, 2);
    assert_eq!(bh.exits, 2);
    assert_eq!(bh.final_exits, 2);
    // Turnover is sum |position change|; the final exit is not a `d`.
    assert_eq!(bh.turnover, 2.0);

    let params = PnlParams {
        threshold: 0.0,
        per_side_bps: 10.0,
        ..PnlParams::default()
    };
    let r = reduce(&[1.0, 1.0, 1.0, 1.0, 1.0], &returns, &groups, &params).unwrap();
    assert!((r.strategy.net - r.buy_and_hold.net).abs() <= 1e-15);
    assert_eq!(r.strategy.entries, r.buy_and_hold.entries);
    assert_eq!(r.strategy.exits, r.buy_and_hold.exits);
    assert_eq!(r.strategy.turnover, r.buy_and_hold.turnover);
}

#[test]
fn canonical_cpu_chunks_are_bit_identical() {
    let n = 40_000usize;
    let mut scores = Vec::with_capacity(n);
    let mut returns = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for i in 0..n {
        scores.push(((i * 37) % 101) as f64 / 50.0 - 1.0);
        returns.push(((i * 13) % 29) as f64 / 1000.0 - 0.01);
        groups.push((i / 7000) as u64);
    }
    let params = PnlParams::default();
    let base = reduce(&scores, &returns, &groups, &params).unwrap();
    for chunk in [1usize, 4_096, 65_536, n] {
        let got = reduce_chunked(&scores, &returns, &groups, &params, chunk).unwrap();
        assert_eq!(base.strategy.bits(), got.strategy.bits(), "chunk {}", chunk);
        assert_eq!(base.buy_and_hold.bits(), got.buy_and_hold.bits());
    }
}

#[test]
fn declared_tile_never_scales_with_the_population() {
    // The tile is a compile-time constant, not a function of P.
    assert_eq!(MAX_TILE_CANDIDATES, 8);
}

// ---------------------------------------------------------------------------
// #306: band / cooldown / hindsight DP
// ---------------------------------------------------------------------------

fn scan_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    let mut scores = Vec::with_capacity(n);
    let mut returns = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for i in 0..n {
        scores.push(((i * 37) % 101) as f64 / 50.0 - 1.0);
        returns.push(((i * 13) % 29) as f64 / 1000.0 - 0.01);
        groups.push((i / 7) as u64);
    }
    (scores, returns, groups)
}

fn serial_band_net(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
) -> f64 {
    let positions = band_positions(scores, groups, params).unwrap();
    let mut prev = 0.0f64;
    let mut current: Option<u64> = None;
    let mut net = 0.0f64;
    for i in 0..returns.len() {
        if current != Some(groups[i]) {
            prev = 0.0;
            current = Some(groups[i]);
        }
        let p = positions[i];
        let d = p - prev;
        net += p * returns[i] - side * d.abs();
        let last = i + 1 == returns.len() || groups[i + 1] != groups[i];
        if last && p != 0.0 {
            net -= side;
        }
        prev = p;
    }
    net
}

fn brute_force_best(returns: &[f64], groups: &[u64], side: f64) -> f64 {
    let n = returns.len();
    assert!(n <= 20, "brute force is only for n <= 20");
    let mut best = f64::NEG_INFINITY;
    for mask in 0u32..(1u32 << n) {
        let mut net = 0.0f64;
        for i in 0..n {
            let p = ((mask >> i) & 1) as f64;
            let prev = if i > 0 && groups[i] == groups[i - 1] {
                ((mask >> (i - 1)) & 1) as f64
            } else {
                0.0
            };
            net += p * returns[i] - side * (p - prev).abs();
            let last = i + 1 == n || groups[i + 1] != groups[i];
            if last && p != 0.0 {
                net -= side;
            }
        }
        if net > best {
            best = net;
        }
    }
    best
}

#[test]
fn band_beta_zero_is_the_threshold_rule_bit_for_bit() {
    let scores = vec![
        f64::NAN,
        0.5,
        0.2,
        0.2,
        -0.3,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.7,
        0.0,
    ];
    let groups = vec![0u64, 0, 0, 0, 1, 1, 1, 2, 2];
    let threshold = 0.2;
    let params = BandParams::new(threshold, 0.0, 1).unwrap();
    let plain = threshold_positions(&scores, threshold);
    let band = band_positions(&scores, &groups, &params).unwrap();
    assert_eq!(plain.len(), band.len());
    for (i, (a, b)) in plain.iter().zip(&band).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "row {}: {} vs {}", i, a, b);
    }
    check_long_only(&band).unwrap();
}

#[test]
fn dp_matches_brute_force_over_all_paths_for_n_le_20() {
    // Several 2-group fixtures, including a mid-series entry and a final exit.
    let cases: Vec<(Vec<f64>, Vec<f64>, Vec<u64>)> = vec![
        (
            vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
            vec![0.01, -0.02, 0.03, 0.0, 0.015, -0.004],
            vec![0, 0, 0, 0, 0, 0],
        ),
        (
            vec![1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
            vec![0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.04, -0.03],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ),
        (
            (0..16)
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect(),
            (0..16)
                .map(|i| ((i * 7 + 3) % 11) as f64 / 100.0 - 0.05)
                .collect(),
            (0..16).map(|i| (i / 8) as u64).collect(),
        ),
    ];
    for (case, (scores, returns, groups)) in cases.iter().enumerate() {
        let side = 0.001;
        let best = brute_force_best(returns, groups, side);
        let dp = dp_scan(returns, groups, side, 3).unwrap();
        let bound = dp.bound(side);
        assert!(
            (bound - best).abs() <= 1e-12,
            "case {}: dp {} vs brute force {}",
            case,
            bound,
            best
        );

        // The band reference is feasible, so it can never beat the DP bound.
        let params = BandParams::new(0.0, 0.2, 1).unwrap();
        let scan = band_scan(scores, returns, groups, &params, side, 5).unwrap();
        assert!(
            scan.net() <= bound + 1e-12,
            "case {}: band {} > dp {}",
            case,
            scan.net(),
            bound
        );
        let serial = serial_band_net(scores, returns, groups, &params, side);
        assert!((scan.net() - serial).abs() <= 1e-15);
    }
}

#[test]
fn dp_and_band_carries_are_chunk_size_invariant() {
    let n = 8_000usize;
    let (scores, returns, groups) = scan_data(n);
    let side = 0.001;

    let params = BandParams::new(0.0, 0.1, 1).unwrap();
    let base = band_scan(&scores, &returns, &groups, &params, side, n).unwrap();
    let dp_base = dp_scan(&returns, &groups, side, n).unwrap();
    for chunk in [1usize, 2, 7, 63, 4_096, 65_536] {
        let got = band_scan(&scores, &returns, &groups, &params, side, chunk).unwrap();
        assert_eq!(base.positions, got.positions, "band chunk {}", chunk);
        assert_eq!(base.carries.len(), got.carries.len());
        for (a, b) in base.carries.iter().zip(&got.carries) {
            assert_eq!(a.bits(), b.bits(), "band carry chunk {}", chunk);
        }
        assert_eq!(base.net().to_bits(), got.net().to_bits(), "band net chunk {}", chunk);

        let dp = dp_scan(&returns, &groups, side, chunk).unwrap();
        assert_eq!(dp_base.carries.len(), dp.carries.len());
        for (a, b) in dp_base.carries.iter().zip(&dp.carries) {
            assert_eq!(a.bits(), b.bits(), "dp carry chunk {}", chunk);
        }
        assert_eq!(
            dp_base.bound(side).to_bits(),
            dp.bound(side).to_bits(),
            "dp bound chunk {}",
            chunk
        );
    }
}

#[test]
fn cooldown_carry_is_chunk_invariant_and_holds_the_minimum() {
    let (scores, returns, groups) = scan_data(2_000);
    let side = 0.001;
    let params = BandParams::new(0.0, 0.05, 4).unwrap();
    let base = cooldown_scan(&scores, &returns, &groups, &params, side, 2_000).unwrap();
    for chunk in [1usize, 3, 17, 511, 65_536] {
        let got = cooldown_scan(&scores, &returns, &groups, &params, side, chunk).unwrap();
        assert_eq!(base.positions, got.positions);
        for (a, b) in base.carries.iter().zip(&got.carries) {
            assert_eq!(a.bits(), b.bits(), "cooldown chunk {}", chunk);
        }
        assert_eq!(base.net().to_bits(), got.net().to_bits());
    }

    // A high score enters; the next rows fall below the exit band but the
    // minim-hold keeps the book long until the counter reaches 4.
    let scores = vec![1.0, -1.0, -1.0, -1.0, -1.0];
    let groups = vec![0u64; 5];
    let params = BandParams::new(0.0, 0.0, 4).unwrap();
    let positions = band_positions(&scores, &groups, &params).unwrap();
    assert_eq!(positions, vec![1.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn group_seams_reset_and_mid_series_entry_and_final_exit_are_handled() {
    // Two groups; the first row enters immediately (mid-series entry).
    let scores = vec![1.0, 1.0, -1.0, 1.0, 1.0];
    let returns = vec![0.01, 0.02, -0.01, 0.03, 0.04];
    let groups = vec![0u64, 0, 0, 1, 1];
    let side = 0.001;

    let params = BandParams::new(0.0, 0.0, 1).unwrap();
    let all = band_positions(&scores, &groups, &params).unwrap();
    assert_eq!(all, vec![1.0, 1.0, 0.0, 1.0, 1.0]);

    // Split at the seam: each group starts flat, and the two carry nets add to
    // the combined net (each group's final exit is charged once).
    let a = band_scan(&scores[..3], &returns[..3], &groups[..3], &params, side, 2).unwrap();
    let b = band_scan(&scores[3..], &returns[3..], &groups[3..], &params, side, 2).unwrap();
    let whole = band_scan(&scores, &returns, &groups, &params, side, 1).unwrap();
    assert_eq!(whole.carries.len(), 2);
    assert!((whole.net() - (a.net() + b.net())).abs() <= 1e-15);

    // DP also resets at the seam: the combined bound is the sum of the parts.
    let da = dp_scan(&returns[..3], &groups[..3], side, 1).unwrap();
    let db = dp_scan(&returns[3..], &groups[3..], side, 1).unwrap();
    let dw = dp_scan(&returns, &groups, side, 4).unwrap();
    assert_eq!(dw.carries.len(), 2);
    assert!((dw.bound(side) - (da.bound(side) + db.bound(side))).abs() <= 1e-15);
}

#[test]
fn threshold_and_band_rule_nets_never_exceed_the_dp_bound() {
    let n = 5_000usize;
    let (scores, returns, groups) = scan_data(n);
    let side = 0.001;
    let threshold = 0.0;
    let params = BandParams::new(threshold, 0.15, 1).unwrap();

    let dp = dp_bound(&returns, &groups, side).unwrap();
    let plain = reduce(
        &scores,
        &returns,
        &groups,
        &PnlParams {
            threshold,
            per_side_bps: side * 10_000.0,
            ..PnlParams::default()
        },
    )
    .unwrap();
    assert!(plain.strategy.net <= dp + 1e-12);
    let band = band_scan(&scores, &returns, &groups, &params, side, 65_536).unwrap();
    assert!(band.net() <= dp + 1e-12);
}

#[test]
fn carry2_composition_is_associative_and_has_the_identity() {
    let rows = [
        Carry2::from_row([0, 1], [0.25, 0.5]),
        Carry2::from_row([1, 1], [0.125, -0.5]),
        Carry2::from_row([0, 1], [0.75, 0.375]),
    ];
    let left = rows[0].then(&rows[1]).then(&rows[2]);
    let right = rows[0].then(&rows[1].then(&rows[2]));
    assert_eq!(left, right);
    assert_eq!(Carry2::IDENTITY.then(&rows[0]), rows[0]);
    assert_eq!(rows[0].then(&Carry2::IDENTITY), rows[0]);

    let dp_rows = [
        DpCarry::row(0.25, 0.125),
        DpCarry::row(-0.5, 0.125),
        DpCarry::row(0.75, 0.125),
    ];
    let a = dp_rows[0].then(&dp_rows[1]).then(&dp_rows[2]);
    let b = dp_rows[0].then(&dp_rows[1].then(&dp_rows[2]));
    assert_eq!(a.bits(), b.bits());
    assert_eq!(DpCarry::IDENTITY.then(&dp_rows[0]).bits(), dp_rows[0].bits());
    assert_eq!(dp_rows[0].then(&DpCarry::IDENTITY).bits(), dp_rows[0].bits());
}

#[test]
fn no_short_positions_in_band_or_dp_inputs() {
    let scores = vec![1.0, -1.0, 1.0];
    let groups = vec![0u64, 0, 0];
    let params = BandParams::new(0.0, 0.0, 1).unwrap();
    let band = band_positions(&scores, &groups, &params).unwrap();
    check_long_only(&band).unwrap();
    // The band can only produce 0 or +1.
    for p in band {
        assert!(p == 0.0 || p == 1.0);
    }
    // A negative band width is rejected rather than silently opening shorts.
    assert!(BandParams::new(0.0, -0.1, 1).is_err());
}

#[test]
#[ignore]
fn bench_band_dp_scan() {
    let n: usize = std::env::var("RCSR_BAND_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_943_856);
    let (scores, returns, groups) = scan_data(n);
    let side = 0.001;
    let params = BandParams::new(0.0, 0.1, 1).unwrap();

    let t0 = std::time::Instant::now();
    let band = band_scan(&scores, &returns, &groups, &params, side, 65_536).unwrap();
    let band_s = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let dp = dp_scan(&returns, &groups, side, 65_536).unwrap();
    let dp_s = t1.elapsed().as_secs_f64();

    println!(
        "n={} groups={} band={:.3}s ({:.3e} rows/s) dp={:.3}s ({:.3e} rows/s) \
         band_carry={}B dp_carry={}B band_net={:.6} dp_bound={:.6}",
        n,
        band.carries.len(),
        band_s,
        n as f64 / band_s,
        dp_s,
        n as f64 / dp_s,
        std::mem::size_of::<Carry2>(),
        std::mem::size_of::<DpCarry>(),
        band.net(),
        dp.bound(side)
    );
    assert!(band.net() <= dp.bound(side) + 1e-9);
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::trading_gpu::{
        device_available, reduce_device, reduce_device_batch,
    };

    fn data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let mut scores = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);
        for i in 0..n {
            scores.push(((i * 37) % 1009) as f64 / 500.0 - 1.0);
            returns.push(((i * 13) % 101) as f64 / 2_000.0 - 0.02);
            groups.push((i / 20_000) as u64);
        }
        (scores, returns, groups)
    }

    fn f32_scores(scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&v| v as f32 as f64).collect()
    }

    #[test]
    fn device_matches_canonical_cpu_on_100k_rows() {
        if !device_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n = 100_000usize;
        let (scores, returns, groups) = data(n);
        let cast = f32_scores(&scores);
        let params = PnlParams {
            threshold: 0.0,
            per_side_bps: 10.0,
            ..PnlParams::default()
        };
        let cpu = reduce(&cast, &returns, &groups, &params).unwrap();
        let dev = reduce_device(&scores, &returns, &groups, &params, 256, 65_536)
            .expect("device P&L");
        let rel = (dev.strategy.net - cpu.strategy.net).abs()
            / cpu.strategy.net.abs().max(1e-30);
        assert!(rel <= 1e-9, "relative net error {}", rel);
        assert_eq!(dev.strategy.rows, cpu.strategy.rows);
        assert_eq!(dev.strategy.entries, cpu.strategy.entries);
        assert_eq!(dev.strategy.exits, cpu.strategy.exits);
        assert_eq!(dev.strategy.final_exits, cpu.strategy.final_exits);
        // The two canonical segment sums are added in the same order; in
        // practice this is bit-identical, not merely within the gate.
        assert_eq!(dev.strategy.bits(), cpu.strategy.bits());
    }

    #[test]
    fn repeat_runs_and_chunks_and_threads_are_bit_identical() {
        if !device_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n = 70_000usize;
        let (scores, returns, groups) = data(n);
        // Short segments make the segment count exceed every thread count, so
        // thread ownership really differs between 128/256/512.
        let params = PnlParams {
            segment_rows: 64,
            ..PnlParams::default()
        };
        let base = reduce_device(&scores, &returns, &groups, &params, 256, 65_536)
            .expect("device P&L");
        let repeat = reduce_device(&scores, &returns, &groups, &params, 256, 65_536)
            .expect("device P&L");
        assert_eq!(base.strategy.bits(), repeat.strategy.bits());
        for threads in [128usize, 256, 512] {
            for chunk in [1_000usize, 4_096, 65_536, n] {
                let got = reduce_device(&scores, &returns, &groups, &params, threads, chunk)
                    .expect("device P&L");
                assert_eq!(
                    base.strategy.bits(),
                    got.strategy.bits(),
                    "threads {} chunk {}",
                    threads,
                    chunk
                );
            }
        }
    }

    #[test]
    fn padded_batch_does_not_change_a_candidate() {
        if !device_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n = 30_000usize;
        let (scores, returns, groups) = data(n);
        let params = PnlParams::default();
        let alone = reduce_device(&scores, &returns, &groups, &params, 256, 65_536)
            .expect("device P&L");

        let mut candidates: Vec<Vec<f64>> = Vec::new();
        candidates.push(scores.clone());
        for k in 1..MAX_TILE_CANDIDATES {
            candidates.push(
                scores
                    .iter()
                    .map(|v| v - k as f64 * 0.25)
                    .collect::<Vec<_>>(),
            );
        }
        let refs: Vec<&[f64]> = candidates.iter().map(|v| v.as_slice()).collect();
        let thresholds = vec![0.0; refs.len()];
        let padded = reduce_device_batch(
            &refs,
            &returns,
            &groups,
            &thresholds,
            &params,
            256,
            65_536,
        )
        .expect("padded device batch");
        assert_eq!(alone.strategy.bits(), padded[0].strategy.bits());
    }

    #[test]
    #[ignore]
    fn bench_pnl_vs_cpu() {
        if !device_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n: usize = std::env::var("RCSR_PNL_BENCH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        let population = 256usize;
        let params = PnlParams::default();

        // Score generation is identical for both arms and excluded from the
        // timed section, so the comparison is position/cost/count work plus the
        // device H2D transfer of the declared tile.
        let mut score_sets: Vec<Vec<f64>> = Vec::with_capacity(population);
        for p in 0..population {
            let mut s = Vec::with_capacity(n);
            for i in 0..n {
                s.push((((i * 37 + p * 17) % 1009) as f64 / 500.0) - 1.0);
            }
            score_sets.push(s);
        }
        let mut returns = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);
        for i in 0..n {
            returns.push(((i * 13) % 101) as f64 / 2_000.0 - 0.02);
            groups.push((i / 20_000) as u64);
        }

        let cpu_start = std::time::Instant::now();
        let mut cpu_net = 0.0f64;
        for s in &score_sets {
            let r = reduce(s, &returns, &groups, &params).unwrap();
            cpu_net += r.strategy.net;
        }
        let cpu_s = cpu_start.elapsed().as_secs_f64();

        let tile = MAX_TILE_CANDIDATES;
        let mut dev = rcsr::trading_gpu::DevicePnl::new(&returns, &groups, 8_192)
            .expect("device P&L context");
        let thresholds = vec![0.0f64; tile];
        let gpu_start = std::time::Instant::now();
        let mut gpu_net = 0.0f64;
        for base in (0..population).step_by(tile) {
            let count = tile.min(population - base);
            let mut packed: Vec<f32> = Vec::with_capacity(count * n);
            for s in &score_sets[base..base + count] {
                packed.extend(s.iter().map(|&v| v as f32));
            }
            let totals = dev
                .eval(&packed, &thresholds[..count], params.per_side_bps, 256, 65_536)
                .expect("device P&L");
            gpu_net += totals.iter().map(|t| t.net).sum::<f64>();
        }
        let gpu_s = gpu_start.elapsed().as_secs_f64();
        let row_evals = (population * n) as f64;
        let speedup = cpu_s / gpu_s;
        println!(
            "n={} P={} tile={} cpu={:.3}s gpu={:.3}s speedup={:.2}x \
             candidates/s={:.1} row-evals/s={:.3e} cpu_net={:.6} gpu_net={:.6}",
            n,
            population,
            tile,
            cpu_s,
            gpu_s,
            speedup,
            population as f64 / gpu_s,
            row_evals / gpu_s,
            cpu_net,
            gpu_net
        );
        assert!(speedup >= 5.0, "device speed-up {:.2}x < 5x", speedup);
    }
}

// Silence an unused-import warning in non-CUDA builds.
#[allow(dead_code)]
fn _totals_is_used(t: &PnlTotals) -> [u64; 4] {
    t.bits()
}

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
        device_pnl: false,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        validation_frac: 0.0,
        periods_per_year: Some(4.0),
        candidates: Vec::new(),
        band_select: false,
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

// ---------------------------------------------------------------------------
// #308: banded finalist selection, tie-break and provenance
// ---------------------------------------------------------------------------

use rcsr::selection::{
    exact_cmp, rescore_exact, select_band, BandCandidate, BandOptions, BandRows, ExactScore,
};

/// Deterministic synthetic panel for the 200-candidate band fixture.
fn band_fixture(count: usize, rows: usize) -> (Vec<f64>, Vec<f64>, Vec<u64>, Vec<Vec<f64>>) {
    let returns: Vec<f64> = (0..rows)
        .map(|i| {
            let x = i as f64;
            (x * 0.37).sin() * 0.01 + if i % 7 == 0 { -0.004 } else { 0.001 }
        })
        .collect();
    let times: Vec<f64> = (0..rows).map(|i| i as f64).collect();
    let groups: Vec<u64> = vec![0; rows];
    let scores: Vec<Vec<f64>> = (0..count)
        .map(|c| {
            (0..rows)
                .map(|i| {
                    let x = i as f64;
                    (x * 0.013 * (c as f64 + 1.0)).sin()
                })
                .collect()
        })
        .collect();
    (returns, times, groups, scores)
}

fn exhaustive_winner(
    rows: &BandRows<'_>,
    scores: &[Vec<f64>],
    options: &BandOptions,
) -> (usize, ExactScore) {
    let mut best: Option<(usize, ExactScore)> = None;
    for c in 0..scores.len() {
        let exact = rescore_exact(rows, &scores[c], 0.0, options).unwrap();
        let better = match &best {
            None => true,
            Some((bi, be)) => {
                exact_cmp(&exact, be) == std::cmp::Ordering::Less
                    || (exact_cmp(&exact, be) == std::cmp::Ordering::Equal && c < *bi)
            }
        };
        if better {
            best = Some((c, exact));
        }
    }
    best.unwrap()
}

#[test]
fn band_recall_is_one_on_two_hundred_candidates_and_matches_exhaustive_cpu() {
    let (returns, times, groups, scores) = band_fixture(200, 400);
    let rows = BandRows {
        targets: &returns,
        groups: &groups,
        times: &times,
    };
    let options = BandOptions::default();
    let complexity: Vec<usize> = (0..scores.len()).collect();
    let (exact_winner, exact_best) = exhaustive_winner(&rows, &scores, &options);

    // Put the exact winner just below the cut on the surrogate. A top-K would
    // drop it; the epsilon band must admit it and the exact rescore must find it.
    let mut candidate = Vec::with_capacity(scores.len());
    for c in 0..scores.len() {
        candidate.push(BandCandidate {
            index: c,
            surrogate: 0.5 - c as f64 * 0.0001,
            complexity: complexity[c],
            scores: &scores[c],
            threshold: 0.0,
        });
    }
    let mut sorted_surrogate: Vec<f64> = candidate.iter().map(|c| c.surrogate).collect();
    sorted_surrogate.sort_by(|a, b| b.total_cmp(a));
    let cut = sorted_surrogate[11];
    candidate[exact_winner].surrogate = cut - 0.0005;

    let selection = select_band(&rows, &candidate, &options).unwrap();
    assert_eq!(selection.recall_band, Some(1.0));
    assert_eq!(selection.selected_index, exact_winner);
    assert_eq!(selection.selected_exact, exact_best);
    assert_eq!(selection.exact_rank, 0);
    assert_eq!(selection.scored_by, "cpu_f64");
    assert!(selection.finalists_considered >= 12);
    let expected_epsilon = (2e-3f64).max(0.005 * selection.cut_surrogate.abs());
    assert_eq!(selection.epsilon.to_bits(), expected_epsilon.to_bits());
    // A second run is bit-identical.
    let again = select_band(&rows, &candidate, &options).unwrap();
    assert_eq!(selection.selected_index, again.selected_index);
    assert_eq!(
        selection.selected_exact.net.to_bits(),
        again.selected_exact.net.to_bits()
    );
    assert_eq!(
        selection.selected_exact.turnover.to_bits(),
        again.selected_exact.turnover.to_bits()
    );
}

#[test]
fn a_band_that_misses_the_exact_winner_fails_loudly() {
    let (returns, times, groups, scores) = band_fixture(200, 400);
    let rows = BandRows {
        targets: &returns,
        groups: &groups,
        times: &times,
    };
    let options = BandOptions::default();
    let (exact_winner, _) = exhaustive_winner(&rows, &scores, &options);

    let mut candidate: Vec<BandCandidate<'_>> = (0..scores.len())
        .map(|c| BandCandidate {
            index: c,
            surrogate: 0.5 - c as f64 * 0.0001,
            complexity: c,
            scores: &scores[c],
            threshold: 0.0,
        })
        .collect();
    let mut sorted_surrogate: Vec<f64> = candidate.iter().map(|c| c.surrogate).collect();
    sorted_surrogate.sort_by(|a, b| b.total_cmp(a));
    let cut = sorted_surrogate[11];
    candidate[exact_winner].surrogate = cut - 0.1;

    let err = select_band(&rows, &candidate, &options).unwrap_err();
    assert!(err.contains("band recall failure"), "unexpected error: {err}");

    let lenient = BandOptions {
        fail_on_miss: false,
        ..options
    };
    let selection = select_band(&rows, &candidate, &lenient).unwrap();
    assert_eq!(selection.recall_band, Some(0.0));
    assert_ne!(selection.selected_index, exact_winner);
}

#[test]
fn exact_tie_break_is_total_deterministic_and_prefers_complexity_then_index() {
    // Direct contract for non-finite placement.
    let finite = ExactScore {
        net: 1.0,
        turnover: 1.0,
    };
    let plus_inf = ExactScore {
        net: f64::INFINITY,
        turnover: 1.0,
    };
    let nan_net = ExactScore {
        net: f64::NAN,
        turnover: 1.0,
    };
    let minus_inf = ExactScore {
        net: f64::NEG_INFINITY,
        turnover: 1.0,
    };
    assert_eq!(exact_cmp(&plus_inf, &finite), std::cmp::Ordering::Less);
    assert_eq!(exact_cmp(&nan_net, &finite), std::cmp::Ordering::Greater);
    assert_eq!(
        exact_cmp(&minus_inf, &finite),
        std::cmp::Ordering::Greater
    );
    assert_eq!(
        exact_cmp(&nan_net, &minus_inf),
        std::cmp::Ordering::Equal,
        "both net classes are the same worst class"
    );
    let zero_turnover = ExactScore {
        net: 1.0,
        turnover: 0.0,
    };
    let nan_turnover = ExactScore {
        net: 1.0,
        turnover: f64::NAN,
    };
    let inf_turnover = ExactScore {
        net: 1.0,
        turnover: f64::INFINITY,
    };
    assert_eq!(
        exact_cmp(&zero_turnover, &finite),
        std::cmp::Ordering::Less
    );
    assert_eq!(
        exact_cmp(&nan_turnover, &finite),
        std::cmp::Ordering::Greater
    );
    assert_eq!(
        exact_cmp(&inf_turnover, &finite),
        std::cmp::Ordering::Greater
    );
    // Transitivity on a mixed list.
    let mixed = [
        nan_net, finite, plus_inf, zero_turnover, minus_inf, inf_turnover,
    ];
    for a in &mixed {
        for b in &mixed {
            for c in &mixed {
                let ab = exact_cmp(a, b);
                let bc = exact_cmp(b, c);
                if ab == std::cmp::Ordering::Less && bc == std::cmp::Ordering::Less {
                    assert_eq!(exact_cmp(a, c), std::cmp::Ordering::Less);
                }
            }
        }
    }

    // Equal net and turnover: lower complexity wins, then lower index.
    let (returns, _times, groups, _scores) = band_fixture(3, 5);
    let times: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let rows = BandRows {
        targets: &returns[..5],
        groups: &groups[..5],
        times: &times,
    };
    let options = BandOptions::default();
    let score_a = vec![0.5, -0.1, 0.2, -0.3, 0.4];
    let score_b = vec![0.6, -0.5, 0.3, -0.7, 0.8];
    let candidates = [
        BandCandidate {
            index: 0,
            surrogate: 0.0,
            complexity: 5,
            scores: &score_a,
            threshold: 0.0,
        },
        BandCandidate {
            index: 1,
            surrogate: 0.0,
            complexity: 1,
            scores: &score_b,
            threshold: 0.0,
        },
        BandCandidate {
            index: 2,
            surrogate: 0.0,
            complexity: 1,
            scores: &score_b,
            threshold: 0.0,
        },
    ];
    let selection = select_band(&rows, &candidates, &options).unwrap();
    assert_eq!(selection.selected_index, 1, "complexity then index");
    assert_eq!(selection.exact_rank, 0);
}

#[test]
fn select_band_uses_validation_and_ignores_a_shuffled_or_resigned_test_block() {
    let mut base = String::from("ts,sym,x,target\n");
    for i in 0..40 {
        let target = ((i as f64) * 0.21).sin() * 0.01 + if i % 3 == 0 { 0.002 } else { 0.0 };
        base.push_str(&format!("{},A,{},{}\n", i, i, target));
    }
    let train = data::parse_table(&base).unwrap();
    let mut spec = spec("x", vec![window("w", 0.0, 20.0, 20.0, 40.0)]);
    spec.validation_frac = 0.5;
    spec.candidates = vec!["x".to_string(), "0 - x".to_string()];
    spec.band_select = true;

    let first = walkforward::run_table(&train, &spec, "test").unwrap();
    let w0 = &first.windows[0];
    assert_eq!(w0.band_selection.as_ref().unwrap().recall_band, Some(1.0));
    assert_eq!(first.band_select, true);
    assert_eq!(first.test_used_for_selection, false);
    let chosen = w0.candidate;
    let chosen_expr = w0.expression.clone();
    let chosen_threshold = w0.threshold.to_bits();

    // Reverse the test rows and flip the sign of every test target. Selection
    // uses the validation fibre only, so neither the candidate nor its
    // constants (here the threshold) may move.
    let mut shuffled = String::from("ts,sym,x,target\n");
    for i in (0..20).rev() {
        let target = ((i as f64) * 0.21).sin() * 0.01 + if i % 3 == 0 { 0.002 } else { 0.0 };
        shuffled.push_str(&format!("{},A,{},{}\n", i, i, target));
    }
    for i in (20..40).rev() {
        let target = -(((i as f64) * 0.21).sin() * 0.01 + if i % 3 == 0 { 0.002 } else { 0.0 });
        shuffled.push_str(&format!("{},A,{},{}\n", i, i, target));
    }
    let shuffled = data::parse_table(&shuffled).unwrap();
    let second = walkforward::run_table(&shuffled, &spec, "test").unwrap();
    let w1 = &second.windows[0];
    assert_eq!(w1.candidate, chosen);
    assert_eq!(w1.expression, chosen_expr);
    assert_eq!(w1.threshold.to_bits(), chosen_threshold);
    assert_eq!(w1.band_selection.as_ref().unwrap().selected_index, chosen);
}

#[test]
fn select_band_json_carries_the_provenance_fields() {
    let (returns, times, groups, scores) = band_fixture(20, 40);
    let rows = BandRows {
        targets: &returns,
        groups: &groups,
        times: &times,
    };
    let candidate: Vec<BandCandidate<'_>> = (0..scores.len())
        .map(|c| BandCandidate {
            index: c,
            surrogate: 0.0,
            complexity: c,
            scores: &scores[c],
            threshold: 0.0,
        })
        .collect();
    let selection = select_band(&rows, &candidate, &BandOptions::default()).unwrap();
    let json = selection.json();
    for field in [
        "surrogate_rank",
        "exact_rank",
        "band",
        "epsilon",
        "finalists_considered",
        "recall_band",
        "scored_by",
    ] {
        assert!(json.contains(&format!("\"{field}\"")), "missing {field}: {json}");
    }
    assert!(json.contains("\"scored_by\":\"cpu_f64\""));
}
