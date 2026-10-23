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
    benchmarks, check_group_contiguous, check_long_only, reduce, reduce_chunked, PnlParams,
    PnlTotals, MAX_TILE_CANDIDATES,
};

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
// Device tests
// ---------------------------------------------------------------------------

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
