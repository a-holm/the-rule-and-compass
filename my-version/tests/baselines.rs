//! #310: long-only baseline ladder.
//!
//! Cash, buy-and-hold, constant long/flat and matched-exposure random long:
//! closed forms, exposure/entry/exit preservation, seed determinism, CPU/device
//! parity and the walk-forward report surface.
//!
//! ```sh
//! cargo test --release --features cuda --test baselines
//! cargo test --release --features cuda --test baselines -- --ignored --nocapture bench_ladder
//! ```

use rcsr::baselines::{
    constant_is_long, control_positions, evaluate_controls, matched_exposure_positions,
    net_rows, position_totals, ControlKind,
};
use rcsr::data;
use rcsr::trading_gpu::{buy_and_hold_totals, cash_totals, PnlParams, PnlTotals};
use rcsr::walkforward::{self, EvalSpec, Window};

const SIDE_BPS: f64 = 10.0;

fn params() -> PnlParams {
    PnlParams {
        per_side_bps: SIDE_BPS,
        ..PnlParams::default()
    }
}

fn returns_of(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i * 13) % 101) as f64 - 50.0) / 20_000.0)
        .collect()
}

fn groups_of(n: usize, per_group: usize) -> Vec<u64> {
    (0..n).map(|i| (i / per_group) as u64).collect()
}

/// A strategy-like book with runs of varying length.
fn runs_book(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if (i / 3 + i / 17 + (i / 41) % 2) % 2 == 0 { 1.0 } else { 0.0 })
        .collect()
}

fn same_counts(a: &PnlTotals, b: &PnlTotals) {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.long_rows, b.long_rows);
    assert_eq!(a.entries, b.entries);
    assert_eq!(a.exits, b.exits);
    assert_eq!(a.trades, b.trades);
    assert_eq!(a.final_exits, b.final_exits);
}

#[test]
fn cash_is_zero_net_zero_cost_and_nan_sharpe() {
    let n = 400;
    let r = returns_of(n);
    let g = groups_of(n, 100);
    let strat = runs_book(n);
    let set = evaluate_controls(
        &strat, &r, &g, 0.001, 0.0, 1, 365.0, false,
        &|net: &[f64]| net.chunks(4).map(|c| c.iter().sum::<f64>() / 4.0).collect(),
    )
    .unwrap();
    let cash = set.get(ControlKind::Cash);
    assert_eq!(cash.totals.net, 0.0);
    assert_eq!(cash.totals.cost, 0.0);
    assert_eq!(cash.totals.long_rows, 0);
    assert!(cash.sharpe.is_nan());
    assert_eq!(cash.excess_over_cash, 0.0);
    assert_eq!(cash.beta, 0.0);
    assert_eq!(cash.exposure, 0.0);
}

#[test]
fn buy_and_hold_is_sum_minus_two_sides_at_1_5_and_1e6_rows() {
    for &n in &[1usize, 5, 1_000_000] {
        let r = returns_of(n);
        let g = groups_of(n, n.max(1));
        let side = SIDE_BPS / 10_000.0;
        let want = r.iter().sum::<f64>() - 2.0 * side;
        let bh = buy_and_hold_totals(&r, &g, side);
        assert!((bh.net - want).abs() <= 1e-9, "n={n}: {} vs {want}", bh.net);
        assert_eq!(bh.entries, 1);
        assert_eq!(bh.exits, 1);
        assert_eq!(bh.final_exits, 1);
        // The canonical reduction of p = 1 agrees.
        let red = position_totals(&vec![1.0; n], &r, &g, &params()).unwrap();
        assert!((red.net - want).abs() <= 1e-9, "n={n} reduce {}", red.net);
        assert_eq!(red.entries, 1);
        assert_eq!(red.exits, 1);
        assert_eq!(red.final_exits, 1);
        assert_eq!(red.long_rows, n as u64);
    }
}

#[test]
fn cash_totals_match_the_reduction_of_the_flat_book() {
    let n = 5_000;
    let r = returns_of(n);
    let g = groups_of(n, 700);
    let cash = cash_totals(&r, 0.0);
    let red = position_totals(&vec![0.0; n], &r, &g, &params()).unwrap();
    assert_eq!(cash.net, red.net);
    assert_eq!(cash.cost, red.cost);
    assert_eq!(red.trades, 0);
}

#[test]
fn matched_exposure_preserves_rows_entries_and_exits_exactly() {
    let n = 50_000;
    let r = returns_of(n);
    let g = groups_of(n, 1_234);
    let strat = runs_book(n);
    let base = position_totals(&strat, &r, &g, &params()).unwrap();
    for seed in [0u64, 1, 42, u64::MAX] {
        let m = matched_exposure_positions(&strat, &g, seed).unwrap();
        assert!(m.iter().all(|&p| p == 0.0 || p == 1.0));
        let t = position_totals(&m, &r, &g, &params()).unwrap();
        same_counts(&base, &t);
        assert_eq!(base.turnover, t.turnover);
        // Cost is the same multiset of side charges, summed in a different order.
        assert!((base.cost - t.cost).abs() <= 1e-12);
        // Per group, too.
        let mut lo = 0;
        while lo < n {
            let hi = (lo + 1_234).min(n);
            let a: f64 = strat[lo..hi].iter().sum();
            let b: f64 = m[lo..hi].iter().sum();
            assert_eq!(a, b);
            lo = hi;
        }
    }
    assert_ne!(
        matched_exposure_positions(&strat, &g, 1).unwrap(),
        matched_exposure_positions(&strat, &g, 2).unwrap()
    );
    assert_ne!(matched_exposure_positions(&strat, &g, 1).unwrap(), strat);
}

#[test]
fn matched_exposure_is_seed_deterministic_and_group_local() {
    let n = 20_000;
    let g = groups_of(n, 500);
    let strat = runs_book(n);
    let a = matched_exposure_positions(&strat, &g, 99).unwrap();
    for _ in 0..3 {
        assert_eq!(a, matched_exposure_positions(&strat, &g, 99).unwrap());
    }
    // Dropping the first group leaves every other group's draw unchanged.
    let tail = matched_exposure_positions(&strat[500..], &g[500..], 99).unwrap();
    assert_eq!(&a[500..], &tail[..]);
}

#[test]
fn matched_exposure_refuses_short_and_non_contiguous_input() {
    assert!(matched_exposure_positions(&[0.0, -1.0], &[0, 0], 0).is_err());
    assert!(matched_exposure_positions(&[0.0, 1.0, 1.0], &[0, 1, 0], 0).is_err());
}

#[test]
fn constant_control_follows_the_training_mean_only() {
    assert!(constant_is_long(0.001));
    assert!(!constant_is_long(0.0));
    assert!(!constant_is_long(-0.5));
    assert!(!constant_is_long(f64::NAN));
    let g = groups_of(10, 5);
    let strat = vec![1.0; 10];
    let long = control_positions(&strat, &g, true, 0).unwrap();
    let flat = control_positions(&strat, &g, false, 0).unwrap();
    assert!(long[2].iter().all(|&p| p == 1.0));
    assert!(flat[2].iter().all(|&p| p == 0.0));
}

#[test]
fn net_rows_sum_to_the_reduction_net() {
    let n = 9_000;
    let r = returns_of(n);
    let g = groups_of(n, 450);
    let strat = runs_book(n);
    let side = SIDE_BPS / 10_000.0;
    let rows = net_rows(&strat, &r, &g, side);
    let red = position_totals(&strat, &r, &g, &params()).unwrap();
    assert!((rows.iter().sum::<f64>() - red.net).abs() <= 1e-12);
}

#[test]
fn control_set_reports_beta_excess_exposure_turnover() {
    let n = 2_000;
    let r = returns_of(n);
    let g = groups_of(n, 2_000);
    let strat = runs_book(n);
    let set = evaluate_controls(
        &strat, &r, &g, 0.001, 1.0, 7, 365.0, true,
        &|net: &[f64]| net.to_vec(),
    )
    .unwrap();
    let bh = set.get(ControlKind::BuyAndHold);
    assert!((bh.beta - 1.0).abs() <= 1e-12);
    assert_eq!(bh.exposure, 1.0);
    assert_eq!(bh.turnover, 1.0);
    assert!((bh.excess_over_cash - bh.totals.net).abs() <= 0.0);
    let c = set.get(ControlKind::Constant);
    assert!(set.constant_long);
    assert_eq!(c.totals.bits(), bh.totals.bits());
    let m = set.get(ControlKind::MatchedRandomLong);
    let s = position_totals(&strat, &r, &g, &params()).unwrap();
    assert_eq!(m.totals.long_rows, s.long_rows);
    assert!(m.beta.is_finite());
    assert_eq!(set.seed, 7);
    let d = set.device.as_ref().expect("device check requested");
    assert!(d.counts_equal);
    assert!(d.max_abs_delta <= 1e-9, "delta {}", d.max_abs_delta);
}

fn walk_table() -> data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    let mut rng = 17u64;
    for sym in ["A", "B"] {
        for t in 0..120 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((rng >> 33) % 2001) as f64 / 100_000.0 - 0.01;
            let x = noise * 3.0 + if (t / 5) % 2 == 0 { 0.004 } else { -0.004 };
            s.push_str(&format!("{},{},{},{}\n", t, sym, x, noise));
        }
    }
    data::parse_table(&s).unwrap()
}

fn walk_spec(seed: u64, device: bool) -> EvalSpec {
    EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: Some("sym".to_string()),
        windows: vec![Window {
            label: "w".to_string(),
            train_start: 0.0,
            train_end: 60.0,
            test_start: 60.0,
            test_end: 120.0,
        }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.5,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(365.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: device,
        control_seed: seed,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
    }
}

#[test]
fn every_walkforward_report_includes_the_controls() {
    let t = walk_table();
    let r = walkforward::run_table(&t, &walk_spec(5, false), "test").unwrap();
    let c = r.windows[0].controls.as_ref().expect("controls");
    assert_eq!(c.seed, 5);
    assert_eq!(c.rows.len(), 4);
    let m = &r.windows[0].metrics;
    let matched = c.get(ControlKind::MatchedRandomLong);
    assert_eq!(matched.totals.long_rows, m.active_rows as u64);
    assert_eq!(matched.totals.entries, m.entries as u64);
    assert_eq!(c.get(ControlKind::Cash).totals.net, 0.0);
    assert!(c.get(ControlKind::Cash).sharpe.is_nan());
    let json = r.json();
    for key in [
        "\"controls\":{\"seed\":5",
        "\"control\":\"cash\"",
        "\"control\":\"buy_and_hold\"",
        "\"control\":\"constant\"",
        "\"control\":\"matched_exposure_random_long\"",
        "\"excess_over_cash\"",
        "\"beta\"",
    ] {
        assert!(json.contains(key), "missing {key}");
    }
    assert!(r.markdown().contains("## Controls"));
    assert!(r.text().contains("controls"));
    // Same seed, same report; another seed changes the matched control only.
    let again = walkforward::run_table(&t, &walk_spec(5, false), "test").unwrap();
    assert_eq!(json, again.json());
    let other = walkforward::run_table(&t, &walk_spec(6, false), "test").unwrap();
    let o = other.windows[0].controls.as_ref().unwrap();
    assert_eq!(
        o.get(ControlKind::BuyAndHold).totals.bits(),
        c.get(ControlKind::BuyAndHold).totals.bits()
    );
    assert_ne!(
        o.get(ControlKind::MatchedRandomLong).totals.bits(),
        matched.totals.bits()
    );
}

#[test]
fn device_flag_adds_the_cross_check() {
    let t = walk_table();
    let r = walkforward::run_table(&t, &walk_spec(1, true), "test").unwrap();
    let c = r.windows[0].controls.as_ref().unwrap();
    let d = c.device.as_ref().expect("device block");
    assert!(d.counts_equal);
    assert!(d.max_abs_delta <= 1e-9);
    let plain = walkforward::run_table(&t, &walk_spec(1, false), "test").unwrap();
    assert!(plain.windows[0].controls.as_ref().unwrap().device.is_none());
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::baselines::device_totals;

    #[test]
    fn device_controls_match_the_cpu_reference_to_1e_9() {
        if !rcsr::trading_gpu::device_available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        for &(n, per_group) in &[(1usize, 1usize), (5, 5), (1_000_000, 20_000), (300_001, 7_777)] {
            let r = returns_of(n);
            let g = groups_of(n, per_group);
            let strat = runs_book(n);
            let pos = control_positions(&strat, &g, true, 123).unwrap();
            let dev = device_totals(&pos, &r, &g, &params()).expect("device");
            for (k, p) in pos.iter().enumerate() {
                let cpu = position_totals(p, &r, &g, &params()).unwrap();
                same_counts(&cpu, &dev[k]);
                for (a, b) in [
                    (cpu.net, dev[k].net),
                    (cpu.gross, dev[k].gross),
                    (cpu.cost, dev[k].cost),
                    (cpu.turnover, dev[k].turnover),
                ] {
                    assert!((a - b).abs() <= 1e-9, "n={n} control {k}: {a} vs {b}");
                }
            }
            // Device buy-and-hold is the closed form.
            let side = SIDE_BPS / 10_000.0;
            let want = r.iter().sum::<f64>() - 2.0 * side * (n.div_ceil(per_group)) as f64;
            assert!((dev[1].net - want).abs() <= 1e-9, "n={n}");
            assert_eq!(dev[0].net, 0.0);
            assert_eq!(dev[0].cost, 0.0);
            // Repeats are bit-identical.
            let again = device_totals(&pos, &r, &g, &params()).unwrap();
            for (a, b) in dev.iter().zip(&again) {
                assert_eq!(a.bits(), b.bits());
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_ladder() {
        if !rcsr::trading_gpu::device_available() {
            eprintln!("skipped: no CUDA device");
            return;
        }
        let n: usize = std::env::var("RCSR_LADDER_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3_900_000);
        let r = returns_of(n);
        let g = groups_of(n, 30_000);
        let strat = runs_book(n);
        let side = SIDE_BPS / 10_000.0;
        let p = params();

        let t0 = std::time::Instant::now();
        let pos = control_positions(&strat, &g, true, 2026).unwrap();
        let gen_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let cpu_cash = cash_totals(&r, 0.0);
        let cpu_bh = buy_and_hold_totals(&r, &g, side);
        let cpu_const = cpu_bh;
        let cpu_matched = position_totals(&pos[3], &r, &g, &p).unwrap();
        let cpu_s = t1.elapsed().as_secs_f64();

        let t2 = std::time::Instant::now();
        let dev = device_totals(&pos, &r, &g, &p).expect("device");
        let dev_s = t2.elapsed().as_secs_f64();

        let cpu = [cpu_cash, cpu_bh, cpu_const, cpu_matched];
        let mut worst = 0.0f64;
        for (k, name) in ["cash", "buy_and_hold", "constant", "matched"].iter().enumerate() {
            let d = (cpu[k].net - dev[k].net).abs();
            worst = worst.max(d);
            println!(
                "{:<13} cpu_net={:+.9} dev_net={:+.9} delta={:.3e} long_rows={} entries={} exits={}",
                name, cpu[k].net, dev[k].net, d, cpu[k].long_rows, cpu[k].entries, cpu[k].exits
            );
        }
        println!(
            "n={} perm_gen={:.3}s cpu_ladder={:.3}s dev_ladder={:.3}s (4 controls, incl. H2D) max_delta={:.3e}",
            n, gen_s, cpu_s, dev_s, worst
        );
        assert!(worst <= 1e-9);
    }
}
