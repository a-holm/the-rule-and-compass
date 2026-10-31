//! Long-only evaluation protocol: purge, economics, benchmarks, block
//! inference and nulls on the walk-forward report.

use rcsr::data;
use rcsr::rng::Mt;
use rcsr::walkforward::{self, EvalSpec, ProtocolOptions, Window};

fn table(rows: &[(f64, &str, f64, f64)]) -> data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    for (ts, sym, x, y) in rows {
        s.push_str(&format!("{},{},{},{}\n", ts, sym, x, y));
    }
    data::parse_table(&s).unwrap()
}

fn single(rows: &[(f64, f64, f64)]) -> data::Table {
    let rows: Vec<(f64, &str, f64, f64)> = rows.iter().map(|&(ts, x, y)| (ts, "A", x, y)).collect();
    table(&rows)
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
        cost_bps: 10.0,
        threshold_quantile: 0.70,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(4.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn purge_must_cover_entry_offset_horizon_and_delay() {
    let t = single(&[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0), (20.0, 3.0, 0.0), (21.0, 4.0, 0.0)]);
    let mut s = spec("x", vec![window("w", 0.0, 10.0, 12.0, 30.0)]);
    s.horizon = 1.0;
    let mut o = ProtocolOptions::default();
    o.entry_offset = 1.0;
    o.delay = 1.0;
    // gap 2 < entry 1 + horizon 1 + delay 1
    let err = walkforward::run_table_with(&t, &s, "t", &o).unwrap_err();
    assert!(err.contains("purge gap is 2"), "{}", err);
    assert!(err.contains("needs 3"), "{}", err);
    // the plain horizon alone would have passed
    assert!(walkforward::run_table(&t, &s, "t").is_ok());
    s.windows = vec![window("w", 0.0, 9.0, 12.0, 30.0)];
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    assert_eq!(r.required_purge_seconds, 3.0);
    assert_eq!(r.min_purge_seconds, 3.0);
    assert!(r.json().contains("\"split\":{\"required_purge_seconds\":3,\"purge_seconds\":3"));
}

#[test]
fn no_training_label_reaches_a_test_window_on_a_ragged_panel() {
    // A: every hour. B: late listing, first row after the training range.
    // C: a hole in the middle of the training range.
    let mut rows: Vec<(f64, &str, f64, f64)> = Vec::new();
    for t in 0..30 {
        let ts = t as f64;
        rows.push((ts, "A", ts, 0.001));
        if t >= 14 {
            rows.push((ts, "B", ts, 0.001));
        }
        if !(4..8).contains(&t) {
            rows.push((ts, "C", ts, 0.001));
        }
    }
    let t = table(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 12.0, 15.0, 30.0)]);
    s.group = Some("sym".to_string());
    s.horizon = 3.0;
    let r = walkforward::run_table(&t, &s, "t").unwrap();
    let w = &r.windows[0];
    // train rows: A 0..12 (12) + C 0..12 without 4..8 (8); B has none
    assert_eq!(w.train_rows, 20);
    let last = w.train_last_time.unwrap();
    assert_eq!(last, 11.0);
    assert!(last + r.required_purge_seconds <= w.window.test_start);
    // a window that would let the label of the last training row reach the test is refused
    s.windows = vec![window("late", 0.0, 13.0, 15.0, 30.0)];
    assert!(walkforward::run_table(&t, &s, "t").unwrap_err().contains("purge gap"));
}

#[test]
fn shuffling_the_test_block_leaves_selection_and_constants_unchanged() {
    let train: Vec<(f64, f64, f64)> = (0..40)
        .map(|i| (i as f64, (i % 7) as f64 - 3.0, ((i * 5) % 11) as f64 * 0.001 - 0.005))
        .collect();
    let test: Vec<(f64, f64)> = (0..20).map(|i| ((i % 5) as f64, (i as f64 - 10.0) * 0.001)).collect();
    let build = |order: &[usize]| {
        let mut rows = train.clone();
        for (k, &i) in order.iter().enumerate() {
            rows.push((40.0 + k as f64, test[i].0, test[i].1));
        }
        single(&rows)
    };
    let id: Vec<usize> = (0..20).collect();
    let mut shuffled = id.clone();
    let mut rng = Mt::new(5);
    for i in (1..20).rev() {
        shuffled.swap(i, rng.usize_below(i + 1));
    }
    let mut s = spec("x", vec![window("w", 0.0, 40.0, 40.0, 60.0)]);
    s.validation_frac = 0.4;
    s.candidates = vec!["x".to_string(), "-x".to_string(), "x*x".to_string()];
    let a = walkforward::run_table(&build(&id), &s, "t").unwrap();
    let b = walkforward::run_table(&build(&shuffled), &s, "t").unwrap();
    assert_eq!(a.windows[0].candidate, b.windows[0].candidate);
    assert_eq!(a.windows[0].threshold.to_bits(), b.windows[0].threshold.to_bits());
    assert_eq!(a.windows[0].validation_error, b.windows[0].validation_error);
    assert!(!a.test_used_for_selection);
}

fn hand_fixture() -> walkforward::Report {
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, -0.02),
        (6.0, 6.0, 0.03),
        (7.0, 7.0, -0.01),
    ]);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    s.cost_bps = 100.0;
    s.threshold_quantile = 0.5;
    walkforward::run_table(&t, &s, "test").unwrap()
}

#[test]
fn economics_match_a_hand_computed_fixture() {
    // Net per period: 0, -0.02, 0.03, -0.02. Gross: 0.01, -0.02, 0.03, -0.01.
    // Cost per period: 0.01, 0, 0, 0.01 (entry, final exit).
    let r = hand_fixture();
    let e = &r.economics;
    let equity: f64 = 0.98 * 1.03 * 0.98;
    assert!(close(e.calmar, (equity - 1.0) / 0.02, 1e-12), "{}", e.calmar);
    assert!(close(e.cvar95, -0.02, 1e-15) && close(e.cvar99, -0.02, 1e-15));
    // mean gross 0.0025 / mean cost 0.005 * 100 bp
    assert!(close(e.break_even_bps_per_side, 50.0, 1e-9));
    let means: Vec<f64> = e.ladder.iter().map(|l| l.mean_return).collect();
    for (got, want) in means.iter().zip([0.0, -0.0025, -0.0075, -0.0125]) {
        assert!(close(*got, want, 1e-15), "{:?}", means);
    }
    let mult: Vec<f64> = e.ladder.iter().map(|l| l.multiplier).collect();
    assert_eq!(mult, vec![0.5, 1.0, 2.0, 3.0]);
    assert!(close(e.ladder[3].cost_bps_per_side, 300.0, 1e-12));
    assert!(close(e.sortino, r.pooled.sortino, 1e-15));
}

#[test]
fn every_table_carries_cash_buy_and_hold_universe_and_best_baseline() {
    let r = hand_fixture();
    let names: Vec<&str> = r.economics.benchmarks.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "strategy",
            "cash",
            "buy_and_hold",
            "equal_weight_universe",
            "constant_long_flat",
            "best_long_flat_baseline"
        ]
    );
    let get = |n: &str| r.economics.benchmarks.iter().find(|b| b.name == n).unwrap();
    assert_eq!(get("cash").mean_return, 0.0);
    assert!(get("cash").sharpe.is_nan());
    assert_eq!(get("cash").exposure, 0.0);
    // The universe is the costless market: 0.01, -0.02, 0.03, -0.01.
    assert!(close(get("equal_weight_universe").mean_return, 0.0025, 1e-15));
    assert_eq!(get("equal_weight_universe").exposure, 1.0);
    assert!(close(get("equal_weight_universe").beta, 1.0, 1e-12));
    // Buy and hold pays one entry and one exit: 0.0025 - 0.02/4.
    assert!(close(get("buy_and_hold").mean_return, -0.0025, 1e-15));
    // The training mean is zero, so the constant control is flat like cash;
    // the best baseline is the earlier of the tied entries, cash.
    assert_eq!(r.economics.best_baseline, "cash");
    assert_eq!(get("best_long_flat_baseline").mean_return, 0.0);
    // Strategy: fully exposed, beta on the universe 1 - cov(m, cost) / var(m).
    assert_eq!(get("strategy").exposure, 1.0);
    let beta = 1.0 + (5.0e-5 / 3.0) / (1.475e-3 / 3.0);
    assert!(close(get("strategy").beta, beta, 1e-9), "{}", get("strategy").beta);
    let json = r.json();
    for key in ["\"cash\"", "\"buy_and_hold\"", "\"equal_weight_universe\"", "\"best_long_flat_baseline\"", "\"cost_ladder\"", "\"cvar_95\"", "\"calmar\""] {
        assert!(json.contains(key), "{}", key);
    }
    assert!(r.markdown().contains("Long-only economics"));
}

/// Three symbols, hourly stamps from a real epoch, a weak signal.
fn panel(hours: usize, seed: u32) -> data::Table {
    let mut rng = Mt::new(seed);
    let mut rows: Vec<(f64, String, f64, f64)> = Vec::new();
    for h in 0..hours {
        let ts = 1_700_000_000.0 + h as f64 * 3600.0;
        for s in ["A", "B", "C"] {
            let x = rng.next_f64() - 0.5;
            let y = 0.0004 * x + (rng.next_f64() - 0.5) * 0.004;
            rows.push((ts, s.to_string(), x, y));
        }
    }
    let mut csv = String::from("ts,sym,x,target\n");
    for (ts, s, x, y) in &rows {
        csv.push_str(&format!("{},{},{},{}\n", ts, s, x, y));
    }
    data::parse_table(&csv).unwrap()
}

fn panel_spec(hours: usize) -> EvalSpec {
    let t0 = 1_700_000_000.0;
    let h = |n: usize| t0 + n as f64 * 3600.0;
    let mut s = spec(
        "x",
        vec![window("w1", h(0), h(hours / 2 - 10), h(hours / 2), h(hours - 1))],
    );
    s.group = Some("sym".to_string());
    s.horizon = 5.0 * 3600.0;
    s.periods_per_year = None;
    s.cost_bps = 5.0;
    s.candidates = vec!["x".to_string(), "x*x".to_string(), "0-x".to_string()];
    s.validation_frac = 0.3;
    s
}

#[test]
fn bootstrap_nulls_and_multiple_testing_are_reported_and_reproducible() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.bootstrap_draws = 300;
    o.null_draws = 40;
    o.trials = Some(20);
    o.pbo_blocks = 8;
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    let inf = r.inference.as_ref().unwrap();
    let b = inf.bootstrap.as_ref().unwrap();
    assert!(b.block_len >= 5, "block length must cover the 5 h label: {}", b.block_len);
    assert!(b.mean_ci.lo <= b.mean && b.mean <= b.mean_ci.hi);
    assert!(b.design_effect > 0.0 && b.effective_n > 0.0);
    let m = inf.multiple.as_ref().unwrap();
    assert_eq!(m.candidates, 3);
    assert_eq!(m.trials, 20);
    assert!(m.participation_ratio >= 1.0 && m.participation_ratio <= 3.0 + 1e-9);
    assert!(m.n_eff <= 20.0 + 1e-9 && m.n_eff >= 1.0);
    assert!(m.dsr_10_n_eff.sr0 >= m.dsr_n_eff.sr0);
    assert_eq!(m.dsr_n_eff.t, m.t_used);
    assert!(m.pbo.is_some(), "{:?}", m.pbo_error);
    assert_eq!(m.bh.adjusted.len(), 3);
    let (name, spa) = inf.spa.as_ref().unwrap();
    assert_eq!(name, &r.economics.best_baseline);
    assert!(spa.p_consistent > 0.0 && spa.p_consistent <= 1.0);
    let n = inf.nulls.as_ref().unwrap();
    for p in [n.block_permutation.p_mean, n.matched_exposure_random.p_mean, n.block_permutation.p_sharpe] {
        let count = p * (n.draws as f64 + 1.0) - 1.0;
        assert!(close(count, count.round(), 1e-9) && count >= 0.0 && count <= n.draws as f64, "p {}", p);
    }
    let json = r.json();
    for key in ["\"design_effect\"", "\"dsr_at_n_eff\"", "\"dsr_at_10_n_eff\"", "\"pbo\"", "\"bh_fdr\"", "\"spa\"", "\"block_permutation\"", "\"matched_exposure_random\"", "\"always_long\"", "\"p_formula\":\"(1+count)/(B+1)\""] {
        assert!(json.contains(key), "{}", key);
    }
    let again = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    assert_eq!(json, again.json());
    assert!(r.markdown().contains("Block uncertainty and multiple testing"));
}

#[test]
fn a_block_shorter_than_the_label_overlap_is_refused() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.bootstrap_draws = 200;
    o.block_len = Some(2);
    let err = walkforward::run_table_with(&t, &s, "t", &o).unwrap_err();
    assert!(err.contains("max(horizon, hold)"), "{}", err);
    o.block_len = Some(40);
    assert!(walkforward::run_table_with(&t, &s, "t", &o).is_ok());
}

#[test]
fn explicit_validation_starts_replace_the_fraction() {
    let rows: Vec<(f64, f64, f64)> = (0..30).map(|i| (i as f64, i as f64, 0.0)).collect();
    let t = single(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 20.0, 22.0, 30.0)]);
    let mut o = ProtocolOptions::default();
    o.validation_starts = vec![("w".to_string(), 15.0)];
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    assert_eq!(r.windows[0].validation_rows, 5);
    assert_eq!(r.windows[0].train_rows, 20);
    assert_eq!(r.windows[0].threshold_source, "validation_type7");
    s.validation_frac = 0.2;
    assert!(walkforward::run_table_with(&t, &s, "t", &o).unwrap_err().contains("mutually exclusive"));
}
