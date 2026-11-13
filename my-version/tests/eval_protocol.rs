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
    s.horizon = 1.0;
    let r = walkforward::run_table(&t, &s, "t").unwrap();
    let w = &r.windows[0];
    // train rows: A 0..12 (12) + C 0..12 without 4..8 (8); B has none
    assert_eq!(w.train_rows, 20);
    let last = w.train_last_time.unwrap();
    assert_eq!(last, 11.0);
    assert!(last + r.required_purge_seconds <= w.window.test_start);
    // a window that would let the label of the last training row reach the test is refused
    s.windows = vec![window("late", 0.0, 14.5, 15.0, 30.0)];
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
        // One row per symbol every fifth hour: the 5 h label horizon never overlaps within a symbol.
        for (k, s) in ["A", "B", "C"].into_iter().enumerate() {
            if h % 5 != k {
                continue;
            }
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
    assert_eq!(b.n, r.pooled.periods);
    let hold = if r.pooled.entries > 0 { (r.pooled.active_rows as f64 / r.pooled.entries as f64).ceil() as usize } else { 1 };
    let cube = (b.n as f64).cbrt().ceil() as usize;
    let want = 1usize.max(hold).max(cube).min(b.n);
    assert!(want < 5, "fixture must separate the one-period horizon from the old 5-period rule: {want}");
    assert_eq!(b.block_len, want, "max(ceil(5h/5h)=1, hold {hold}, ceil(n^(1/3)) {cube})");
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
fn a_block_shorter_than_the_holding_floor_is_refused() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.bootstrap_draws = 200;
    let free = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    let hold = if free.pooled.entries > 0 { (free.pooled.active_rows as f64 / free.pooled.entries as f64).ceil() as usize } else { 1 };
    let floor = hold.max(1);
    assert!(floor >= 2, "fixture must make the holding period the floor: {floor}");
    o.block_len = Some(floor - 1);
    let err = walkforward::run_table_with(&t, &s, "t", &o).unwrap_err();
    assert!(err.contains(&format!("max(horizon, hold) = {floor} periods")), "{}", err);
    o.block_len = Some(floor);
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

fn candidate_rows(n: usize) -> data::Table {
    let rows: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| {
            let x = (i % 17) as f64 - 8.0;
            (i as f64, x, 0.01 * x + 0.002 * ((i * 7 % 5) as f64 - 2.0))
        })
        .collect();
    single(&rows)
}

#[test]
fn table_validation_starts_select_among_candidates() {
    let t = candidate_rows(60);
    let mut by_table = spec("x", vec![window("w", 0.0, 40.0, 40.0, 60.0)]);
    by_table.candidates = vec!["-x".to_string(), "x".to_string()];
    let mut o = ProtocolOptions::default();
    o.validation_starts = vec![("w".to_string(), 30.0)];
    let a = walkforward::run_table_with(&t, &by_table, "t", &o).unwrap();
    assert_eq!(a.windows[0].validation_rows, 10);
    assert_eq!(a.windows[0].candidate, 1);
    assert_eq!(a.windows[0].threshold_source, "validation_type7");
    assert_eq!(a.selected_on, "validation");

    let mut by_frac = by_table.clone();
    by_frac.validation_frac = 0.25;
    let b = walkforward::run_table(&t, &by_frac, "t").unwrap();
    assert_eq!(a.windows[0].validation_rows, b.windows[0].validation_rows);
    assert_eq!(a.windows[0].candidate, b.windows[0].candidate);

    let none = walkforward::run_table(&t, &by_table, "t").unwrap_err();
    assert!(none.contains("need a validation slice"), "{none}");

    by_table.windows.push(window("w2", 0.0, 40.0, 40.0, 60.0));
    let partial = walkforward::run_table_with(&t, &by_table, "t", &o).unwrap_err();
    assert!(partial.contains("need a validation slice"), "{partial}");
}

#[test]
fn select_band_accepts_table_validation_starts() {
    let t = candidate_rows(400);
    let mut s = spec("x", vec![window("w", 0.0, 300.0, 300.0, 400.0)]);
    s.candidates = vec!["-x".to_string(), "x".to_string()];
    s.band_select = true;
    let mut o = ProtocolOptions::default();
    o.validation_starts = vec![("w".to_string(), 200.0)];
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    assert!(r.windows[0].band_selection.is_some());
    assert_eq!(r.windows[0].validation_rows, 100);
    let none = walkforward::run_table(&t, &s, "t").unwrap_err();
    assert!(none.contains("validation"), "{none}");
}

#[test]
fn pooled_period_series_golden() {
    // #926: pooled.period_net and pooled.period_time carry one value per pooled period.
    let r = hand_fixture();
    let want_net = [0.0, -0.02, 0.03, -0.02];
    let want_time = [4.0, 5.0, 6.0, 7.0];
    assert_eq!(r.period_net.len(), want_net.len());
    for (got, want) in r.period_net.iter().zip(want_net) {
        assert!(close(*got, want, 1e-15), "{:?}", r.period_net);
    }
    assert_eq!(r.period_time, want_time);
    let j = rcsr::devlog::Json::parse(&r.json()).unwrap();
    let pooled = j.get("pooled").unwrap();
    let list = |k: &str| -> Vec<f64> {
        match pooled.get(k) {
            Some(rcsr::devlog::Json::Arr(a)) => a.iter().map(|v| v.as_f64().unwrap()).collect(),
            _ => panic!("pooled.{k} missing"),
        }
    };
    assert_eq!(list("period_net"), r.period_net);
    assert_eq!(list("period_time"), want_time);
    assert_eq!(list("period_net").len() as f64, pooled.get("periods").unwrap().as_f64().unwrap());
}

fn book_with_test_rows(test: &[(f64, f64)]) -> walkforward::Report {
    let mut rows = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0), (3.0, 3.0, 0.0)];
    for (k, &(x, y)) in test.iter().enumerate() {
        rows.push((4.0 + k as f64, x, y));
    }
    let end = 4.0 + test.len() as f64;
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, end)]);
    s.cost_bps = 100.0;
    s.threshold_quantile = 0.5;
    walkforward::run_table(&single(&rows), &s, "test").unwrap()
}

#[test]
fn wf_break_even_null_for_losing_book() {
    let r = book_with_test_rows(&[(4.0, -0.01), (5.0, -0.02), (6.0, -0.01), (7.0, -0.01)]);
    assert!(r.economics.break_even_bps_per_side.is_nan());
    let json = r.json();
    assert!(json.contains("\"break_even_bps_per_side\":null"), "{json}");
    assert!(json.contains("\"break_even_status\":\"gross_nonpositive\""), "{json}");
}

#[test]
fn wf_break_even_flat_book() {
    let r = book_with_test_rows(&[(-1.0, 0.01), (-2.0, 0.02), (-3.0, 0.01), (-4.0, 0.01)]);
    assert!(r.economics.break_even_bps_per_side.is_nan());
    let json = r.json();
    assert!(json.contains("\"break_even_bps_per_side\":null"), "{json}");
    assert!(json.contains("\"break_even_status\":\"no_cost\""), "{json}");
}

#[test]
fn wf_break_even_defined_status() {
    let json = hand_fixture().json();
    assert!(json.contains("\"break_even_status\":\"defined\""), "{json}");
}

#[test]
fn wf_markdown_break_even_na() {
    let r = book_with_test_rows(&[(4.0, -0.01), (5.0, -0.02), (6.0, -0.01), (7.0, -0.01)]);
    assert!(r.markdown().contains("break-even n/a (gross <= 0) bp/side"), "{}", r.markdown());
}

fn normal(rng: &mut Mt) -> f64 {
    let u = rng.next_f64().max(1e-12);
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// `groups` symbols over `periods` periods, returns with a common share of 0.5, one
/// long/flat path for every symbol that is independent of the returns.
fn synchronised_book(groups: usize, periods: usize, seed: u32) -> rcsr::wf_protocol::NullBook {
    let mut rng = Mt::new(seed);
    let mut path = vec![0.0f64; periods];
    let mut on = rng.next_f64() < 0.5;
    for p in path.iter_mut() {
        if rng.next_f64() < 0.03 {
            on = !on;
        }
        *p = if on { 1.0 } else { 0.0 };
    }
    let common: Vec<f64> = (0..periods).map(|_| normal(&mut rng)).collect();
    let mut positions = Vec::new();
    let mut returns = Vec::new();
    let mut ids = Vec::new();
    let mut period = Vec::new();
    for g in 0..groups {
        for t in 0..periods {
            positions.push(path[t]);
            returns.push(0.01 * (0.5f64.sqrt() * common[t] + 0.5f64.sqrt() * normal(&mut rng)));
            ids.push(g as u64 + 1);
            period.push(t as u32);
        }
    }
    rcsr::wf_protocol::NullBook {
        positions,
        returns,
        groups: ids,
        period,
        period_count: vec![groups as u32; periods],
        period_cost: vec![0.0; periods],
        side: 0.0005,
    }
}

fn period_means(book: &rcsr::wf_protocol::NullBook, positions: &[f64]) -> Vec<f64> {
    let net = rcsr::baselines::net_rows(positions, &book.returns, &book.groups, book.side);
    let mut bins = vec![0.0f64; book.period_count.len()];
    for (row, v) in net.iter().enumerate() {
        bins[book.period[row] as usize] += v;
    }
    bins.iter().zip(&book.period_count).map(|(s, &c)| s / c as f64).collect()
}

/// Rejections of a true null at 10 % over 200 seeded panels. The new null is exact at
/// 10 % up to the discreteness of 99 draws, so a count above 34 has a binomial tail
/// below 1e-3 at p = 0.10 and below 2e-2 at p = 0.12; the per-group shuffle of the
/// old null rejects about 0.4 of the time.
#[test]
fn matched_null_keeps_its_size_under_synchronised_positions() {
    let (panels, draws) = (200usize, 99usize);
    let (mut new_rejects, mut old_rejects) = (0usize, 0usize);
    for k in 0..panels {
        let book = synchronised_book(20, 400, 1000 + k as u32);
        let observed = period_means(&book, &book.positions);
        let obs_mean = observed.iter().sum::<f64>() / observed.len() as f64;
        let control_seed = 77 + k as u64;
        let input = rcsr::wf_protocol::InferenceInput {
            draws: 0,
            null_draws: draws,
            block_len: 5,
            seed: 11,
            control_seed,
            periods_per_year: 365.0,
            pooled: &observed,
            fresh: &observed,
            candidates: &[],
            trials: 1,
            trials_source: "flag",
            pbo_blocks: 4,
            fdr_q: 0.1,
            best_baseline: "cash",
            baseline_series: &[],
            buy_and_hold: &[],
            books: std::slice::from_ref(&book),
        };
        let inf = rcsr::wf_protocol::infer(&input).unwrap();
        let p_new = inf.nulls.as_ref().unwrap().matched_exposure_random.p_mean;
        if p_new <= 0.10 {
            new_rejects += 1;
        }
        let old: Vec<f64> = (0..draws)
            .map(|b| {
                let seed = rcsr::baselines::counter_draw(control_seed, 0, 0, b as u64);
                let pos = rcsr::baselines::matched_exposure_positions(&book.positions, &book.groups, seed).unwrap();
                let m = period_means(&book, &pos);
                m.iter().sum::<f64>() / m.len() as f64
            })
            .collect();
        if rcsr::inference::null_p(obs_mean, &old) <= 0.10 {
            old_rejects += 1;
        }
    }
    assert!(new_rejects <= 34, "shifted null rejected {new_rejects} of {panels} at 10 %");
    assert!(old_rejects >= 60, "per-group shuffle rejected {old_rejects} of {panels} at 10 %");
}

#[test]
fn matched_null_withholds_p_with_too_few_shifts() {
    let book = synchronised_book(2, 4, 5);
    let observed = period_means(&book, &book.positions);
    let input = rcsr::wf_protocol::InferenceInput {
        draws: 0,
        null_draws: 50,
        block_len: 5,
        seed: 11,
        control_seed: 3,
        periods_per_year: 365.0,
        pooled: &observed,
        fresh: &observed,
        candidates: &[],
        trials: 1,
        trials_source: "flag",
        pbo_blocks: 4,
        fdr_q: 0.1,
        best_baseline: "cash",
        baseline_series: &[],
        buy_and_hold: &[],
        books: std::slice::from_ref(&book),
    };
    let n = rcsr::wf_protocol::infer(&input).unwrap().nulls.unwrap();
    assert_eq!(n.matched_distinct_shifts, 3);
    assert_eq!(n.matched_scheme, "shared_circular_shift");
    assert!(n.matched_exposure_random.p_mean.is_nan());
    assert!(n.matched_exposure_random.p_sharpe.is_nan());
    assert!(n.matched_exposure_random.null_mean_q95.is_finite());
    let json = rcsr::wf_protocol::Inference { bootstrap: None, multiple: None, spa: None, nulls: Some(n) }.json();
    assert!(
        json.contains("\"matched_exposure_random\":{\"scheme\":\"shared_circular_shift\",\"distinct_shifts\":3,\"p_mean\":null,\"p_sharpe\":null,\"null_mean_q95\":"),
        "{json}"
    );
}

#[test]
fn walkforward_report_carries_the_shift_scheme() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.null_draws = 40;
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    let n = r.inference.as_ref().unwrap().nulls.as_ref().unwrap();
    assert!(n.matched_distinct_shifts >= 20);
    let json = r.json();
    assert!(json.contains("\"matched_exposure_random\":{\"scheme\":\"shared_circular_shift\",\"distinct_shifts\":"), "{json}");
    assert!(r.markdown().contains("matched-exposure shifted long/flat"));
}

#[test]
fn block_permutation_null_is_pinned() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.bootstrap_draws = 300;
    o.null_draws = 40;
    o.trials = Some(20);
    o.pbo_blocks = 8;
    let r = walkforward::run_table_with(&t, &s, "t", &o).unwrap();
    let json = r.json();
    let at = json.find("\"block_permutation\":").unwrap();
    let end = json[at..].find('}').unwrap();
    assert_eq!(
        &json[at..at + end + 1],
        "\"block_permutation\":{\"p_mean\":0.8536585365853658,\"p_sharpe\":0.7073170731707317,\"null_mean_q95\":-0.00008283531167358163,\"null_sharpe_q95\":-0.1942474845702363}"
    );
}

#[test]
fn wf_markdown_psr_row() {
    let t = panel(600, 3);
    let s = panel_spec(600);
    let mut o = ProtocolOptions::default();
    o.bootstrap_draws = 100;
    o.trials = Some(1);
    o.trials_source = "out_of_sample_book";
    let md = walkforward::run_table_with(&t, &s, "t", &o).unwrap().markdown();
    assert!(md.contains("Trials N = 1 (source out_of_sample_book) over"), "{md}");
    assert!(md.contains("Deflate across runs with rcsr trials."), "{md}");
    let psr = md.find("| PSR (N = 1) | 1 | 0 |").expect("PSR row");
    let dsr = md.find("| N_eff |").expect("DSR row");
    assert!(psr < dsr);
    o.trials_source = "flag";
    let md = walkforward::run_table_with(&t, &s, "t", &o).unwrap().markdown();
    assert!(md.contains("(source flag)") && !md.contains("rcsr trials."), "{md}");
}
