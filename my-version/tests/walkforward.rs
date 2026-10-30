//! Walk-forward evaluation (issue #124).
//!
//! The tests use a tiny table whose scores, threshold and forward returns are
//! known by hand. They pin the long/flat rule, the per-side cost, the
//! train-only threshold, the purge check, pooled aggregation and the direct
//! `.rcsb` path.

use rcsr::data;
use rcsr::store;
use rcsr::walkforward::{self, EvalSpec, Window};

fn table(rows: &[(f64, &str, f64, f64)]) -> data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    for (ts, sym, x, y) in rows {
        s.push_str(&format!("{},{},{},{}\n", ts, sym, x, y));
    }
    data::parse_table(&s).unwrap()
}

fn single(rows: &[(f64, f64, f64)]) -> data::Table {
    let rows: Vec<(f64, &str, f64, f64)> = rows
        .iter()
        .map(|&(ts, x, y)| (ts, "A", x, y))
        .collect();
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
    }
}

#[test]
fn device_pnl_check_matches_the_scalar_book() {
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
    s.device_pnl = true;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let d = r.windows[0].device_pnl.as_ref().expect("device P&L block");
    let m = &r.windows[0].metrics;
    assert_eq!(d.strategy.rows, m.rows as u64);
    assert_eq!(d.strategy.entries, m.entries as u64);
    assert_eq!(d.strategy.trades, m.trades as u64);
    assert!((d.strategy.net - d.cpu_net).abs() <= 1e-12);
    assert!((d.strategy.turnover - m.total_abs_delta).abs() <= 1e-12);
    assert_eq!(d.cash.net, 0.0);
    assert_eq!(d.cash.cost, 0.0);
    let summary = r.device_pnl.as_ref().expect("report summary");
    assert_eq!(summary.windows, 1);
    assert_eq!(summary.rows, m.rows as u64);
    assert!(summary.max_abs_net_delta <= 1e-12);
    assert!(r.json().contains("\"device_pnl\""));
    assert!(r.json().contains("\"buy_and_hold\""));
}

#[test]
fn band_and_dp_policies_are_reported_and_dp_is_only_a_bound() {
    // Training x = 0..3 => q50 threshold 1.5. Test scores 1.8 are above the
    // threshold but below threshold+beta, so the band stays flat while the
    // threshold rule is long.
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 1.8, 0.01),
        (5.0, 0.0, -0.02),
        (6.0, 1.8, 0.03),
        (7.0, 0.0, -0.01),
    ]);

    let mut threshold = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    threshold.threshold_quantile = 0.5;
    let tr = walkforward::run_table(&t, &threshold, "test").unwrap();
    assert_eq!(tr.windows[0].metrics.active_rows, 2);

    let mut band_zero = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    band_zero.threshold_quantile = 0.5;
    band_zero.policy = walkforward::Policy::Band;
    band_zero.band_width = 0.0;
    let bz = walkforward::run_table(&t, &band_zero, "test").unwrap();
    // beta=0 reproduces the threshold book bit for bit.
    assert_eq!(bz.windows[0].metrics.active_rows, 2);
    assert_eq!(bz.windows[0].metrics.trades, tr.windows[0].metrics.trades);
    assert_eq!(
        bz.windows[0].metrics.total_abs_delta.to_bits(),
        tr.windows[0].metrics.total_abs_delta.to_bits()
    );
    assert!(bz.json().contains("\"policy\":{\"name\":\"band\""));

    let mut band = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    band.threshold_quantile = 0.5;
    band.policy = walkforward::Policy::Band;
    band.band_width = 0.6;
    let br = walkforward::run_table(&t, &band, "test").unwrap();
    assert_eq!(br.windows[0].metrics.active_rows, 0);
    let bc = br.windows[0].band.as_ref().expect("band carry check");
    assert_eq!(bc.carry, "carry2");
    assert_eq!(bc.groups, 1);

    let mut dp = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    dp.threshold_quantile = 0.5;
    dp.policy = walkforward::Policy::Dp;
    let dr = walkforward::run_table(&t, &dp, "test").unwrap();
    let h = dr.windows[0]
        .hindsight_bound
        .as_ref()
        .expect("hindsight bound");
    assert_eq!(h.label, "hindsight_bound");
    assert!(h.rule_le_bound);
    assert!(h.net + 1e-12 >= h.rule_net);
    // The reported book is still the threshold strategy, not the DP path.
    assert_eq!(dr.windows[0].metrics.active_rows, 2);
    let json = dr.json();
    assert!(json.contains("\"hindsight_bound\":{\"label\":\"hindsight_bound\""));
    assert!(json.contains("\"is_strategy\":false"));
    assert!(json.contains("\"policy\":{\"name\":\"dp\",\"hindsight_bound\":true"));
}

#[test]
fn known_expression_threshold_costs_and_metrics() {
    // Training x = 0..3; q50 threshold = 1.5. Test x = 4..7, so every test row
    // is long. Cost 100 bp/side; only the first row changes position.
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
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(r.windows.len(), 1);
    let w = &r.windows[0];
    assert!((w.threshold - 1.5).abs() < 1e-15, "threshold {}", w.threshold);
    assert_eq!(w.train_rows, 4);
    assert_eq!(w.test_rows, 4);
    assert_eq!(w.metrics.rows, 4);
    assert_eq!(w.metrics.active_rows, 4);
    assert_eq!(w.metrics.trades, 1);
    assert_eq!(w.metrics.entries, 1);
    assert!((w.metrics.exposure - 1.0).abs() < 1e-15);
    assert!((w.metrics.hit_rate - 0.5).abs() < 1e-15);
    // #145: entry 100 bp + final exit 100 bp = 0.02 over four active rows
    // = 50 bp per long row.
    assert!((w.metrics.avg_cost_bps - 50.0).abs() < 1e-12);
    // First net return is 0.01 - 0.01 = 0; the last row pays the final exit.
    // equity = 1 * 0.98 * 1.03 * 0.98
    let equity = 1.0 * 0.98 * 1.03 * 0.98;
    assert!((w.metrics.total_return - (equity - 1.0)).abs() < 1e-15);
    assert!((w.metrics.cagr - (equity - 1.0)).abs() < 1e-15);
    // The first test period is -2% from the start; the final exit makes the
    // last period -2% as well, so the peak-to-trough is still -2%.
    assert!((w.metrics.max_drawdown - (-0.02)).abs() < 1e-15);
    assert_eq!(w.threshold_source, "train_type7");
}

#[test]
fn every_position_change_is_charged() {
    // Test scores 3,0,3,0 against threshold 1.5: four changes, two entries.
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 3.0, 0.0),
        (5.0, 0.0, 0.0),
        (6.0, 3.0, 0.0),
        (7.0, 0.0, 0.0),
    ]);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    s.cost_bps = 100.0;
    s.threshold_quantile = 0.5;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let w = &r.windows[0];
    assert_eq!(w.metrics.trades, 4);
    assert_eq!(w.metrics.entries, 2);
    assert!((w.metrics.exposure - 0.5).abs() < 1e-15);
    assert!((w.metrics.total_return - (0.99f64.powi(4) - 1.0)).abs() < 1e-15);
    // Four periods, each -0.01 (target zero, cost on every change).
    assert!((r.pooled.mean_return + 0.01).abs() < 1e-15);
}

#[test]
fn negative_score_never_opens_a_short() {
    assert_eq!(walkforward::position_for(-5.0, 0.0), 0.0);
    assert_eq!(walkforward::position_for(-5.0, -6.0), 1.0);
    assert_eq!(walkforward::position_for(f64::NAN, -6.0), 0.0);

    let t = single(&[
        (0.0, -3.0, 0.0),
        (1.0, -2.0, 0.0),
        (2.0, -1.0, 0.0),
        (3.0, -0.5, 0.0),
        (4.0, -5.0, 0.01),
        (5.0, -4.0, 0.02),
    ]);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 6.0)]);
    s.threshold_quantile = 0.0; // threshold = -3
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let w = &r.windows[0];
    // The test scores are below the training minimum, so the rule is flat.
    assert_eq!(w.metrics.active_rows, 0);
    assert_eq!(w.metrics.trades, 0);
    assert!((w.metrics.exposure - 0.0).abs() < 1e-15);
}

#[test]
fn threshold_uses_only_train_or_validation_rows() {
    let mut rows = vec![
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
    ];
    // The test scores are enormous; a look-ahead threshold would move with them.
    rows.push((4.0, 1.0e9, 0.01));
    rows.push((5.0, 2.0e9, 0.02));
    let t = single(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 6.0)]);
    s.threshold_quantile = 0.5;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert!((r.windows[0].threshold - 1.5).abs() < 1e-15);
    assert_eq!(r.windows[0].threshold_source, "train_type7");

    // With a validation slice, the threshold comes from the trailing half.
    s.validation_frac = 0.5;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(r.windows[0].threshold_source, "validation_type7");
    assert_eq!(r.windows[0].validation_rows, 2);
    assert!((r.windows[0].threshold - 2.5).abs() < 1e-15);
}

#[test]
fn purge_gap_shorter_than_horizon_is_refused() {
    let t = single(&[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]);
    let mut s = spec("x", vec![window("w", 0.0, 2.0, 2.5, 3.0)]);
    s.horizon = 1.0;
    let err = walkforward::run_table(&t, &s, "test").unwrap_err();
    assert!(err.contains("purge gap"), "{}", err);
}

#[test]
fn pooled_covers_every_window_and_is_deterministic() {
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 3.0, 0.01),
        (5.0, 0.0, -0.01),
        (6.0, 0.0, 0.0),
        (7.0, 1.0, 0.0),
        (8.0, 2.0, 0.0),
        (9.0, 3.0, 0.0),
        (10.0, 3.0, 0.02),
        (11.0, 0.0, -0.02),
    ]);
    let mut s = spec(
        "x",
        vec![
            window("a", 0.0, 4.0, 4.0, 6.0),
            window("b", 6.0, 10.0, 10.0, 12.0),
        ],
    );
    s.threshold_quantile = 0.5;
    let r1 = walkforward::run_table(&t, &s, "test").unwrap();
    let r2 = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(r1.json(), r2.json());
    assert_eq!(r1.windows.len(), 2);
    assert_eq!(r1.pooled.rows, 4);
    assert_eq!(r1.pooled.periods, 4);
    assert_eq!(r1.pooled.trades, 4);
    assert_eq!(r1.windows.iter().map(|w| w.metrics.rows).sum::<usize>(), 4);
}

#[test]
fn store_path_matches_the_resident_table() {
    let rows: Vec<(f64, &str, f64, f64)> = (0..8)
        .map(|i| {
            let i = i as f64;
            (i, "A", i, if i < 4.0 { 0.0 } else { (i - 5.0) * 0.01 })
        })
        .collect();
    let t = table(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    s.threshold_quantile = 0.5;
    s.cost_bps = 5.0;
    let resident = walkforward::run_table(&t, &s, "test").unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "rcsr-walkforward-{}-{}.rcsb",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let path = path.to_string_lossy().into_owned();
    store::write_store(&path, &t, 3).unwrap();
    let streamed = walkforward::run_path(&path, &s).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(resident.source, "test");
    assert_eq!(streamed.source, "store-stream");
    assert!((resident.windows[0].threshold - streamed.windows[0].threshold).abs() < 1e-15);
    assert_eq!(resident.windows[0].test_rows, streamed.windows[0].test_rows);
    assert!((resident.pooled.sharpe - streamed.pooled.sharpe).abs() < 1e-15);
    assert!((resident.pooled.cagr - streamed.pooled.cagr).abs() < 1e-15);
    assert!((resident.pooled.max_drawdown - streamed.pooled.max_drawdown).abs() < 1e-15);
    assert!((resident.pooled.turnover - streamed.pooled.turnover).abs() < 1e-15);
}

#[test]
fn costs_and_positions_are_per_group() {
    // Two groups with one test row each. Both go long from flat, so there are
    // two entries and two costs, not one shared flat-to-long transition.
    let rows = [
        (0.0, "A", 0.0, 0.0),
        (1.0, "A", 1.0, 0.0),
        (2.0, "A", 2.0, 0.01),
        (0.0, "B", 1.0, 0.0),
        (1.0, "B", 2.0, 0.0),
        (2.0, "B", 3.0, 0.01),
    ];
    let t = table(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 2.0, 2.0, 3.0)]);
    s.group = Some("sym".to_string());
    s.threshold_quantile = 0.0; // every score above the training minimum
    s.cost_bps = 100.0;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let w = &r.windows[0];
    assert_eq!(w.metrics.entries, 2);
    assert_eq!(w.metrics.trades, 2);
    assert_eq!(w.metrics.active_rows, 2);
    // One timestamp: each group pays entry 100 bp and final exit 100 bp, so
    // the mean net return is 0.01 - 0.02 = -0.01.
    assert!((w.metrics.mean_return + 0.01).abs() < 1e-15);
    assert!((w.metrics.avg_cost_bps - 200.0).abs() < 1e-12);
}

#[test]
fn json_and_markdown_carry_the_rules_and_metrics() {
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, 0.02),
    ]);
    let s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 6.0)]);
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let json = r.json();
    for key in [
        "\"window_definition\"",
        "\"cost_model\"",
        "\"position_rule\"",
        "\"pooled\"",
        "\"break_even_hit_rate\"",
        "\"memory_bound\"",
        "\"pooled_seam\"",
        "\"sign_test\"",
        "\"selection\"",
    ] {
        assert!(json.contains(key), "missing {}", key);
    }
    let md = r.markdown();
    assert!(md.contains("Long/flat only"));
    assert!(md.contains("## Pooled"));
    assert!(md.contains("break-even"));
}

#[test]
fn carried_seam_charges_the_boundary_term() {
    // Training x=0..3, q0 threshold=0. Window 1 is long throughout and ends
    // long. Window 2 starts flat at t=6 and re-enters at t=7.
    let rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, 0.01),
        (6.0, -1.0, 0.0),
        (7.0, 7.0, 0.01),
    ];
    let t = single(&rows);
    let mut s = spec(
        "x",
        vec![
            window("a", 0.0, 4.0, 4.0, 6.0),
            window("b", 0.0, 4.0, 6.0, 8.0),
        ],
    );
    s.threshold_quantile = 0.0;
    s.cost_bps = 100.0;
    let carried = walkforward::run_table(&t, &s, "test").unwrap();
    // Carry: entry at 4, exit at 6 (the seam |1-0|), re-entry at 7 = 3 trades.
    assert_eq!(carried.pooled.trades, 3);
    assert_eq!(carried.seam_trades, 1);
    assert!((carried.seam_cost - 0.01).abs() < 1e-15, "seam {}", carried.seam_cost);
    assert!(!carried.pool_reset_flat);

    s.pool_reset_flat = true;
    let reset = walkforward::run_table(&t, &s, "test").unwrap();
    // Reset: entry in each window = 2 trades; the seam term is absent.
    assert_eq!(reset.pooled.trades, 2);
    assert_eq!(reset.seam_trades, 0);
    assert!((reset.seam_cost - 0.0).abs() < 1e-15);
    assert!(carried.pooled.total_abs_delta > reset.pooled.total_abs_delta);
}

#[test]
fn carried_seam_does_not_recharge_an_unchanged_position() {
    let rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, 0.01),
        (6.0, 6.0, 0.01),
        (7.0, 7.0, 0.01),
    ];
    let t = single(&rows);
    let mut s = spec(
        "x",
        vec![
            window("a", 0.0, 4.0, 4.0, 6.0),
            window("b", 0.0, 4.0, 6.0, 8.0),
        ],
    );
    s.threshold_quantile = 0.0;
    let carried = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(carried.pooled.trades, 1, "one entry for the whole pooled path");
    assert_eq!(carried.seam_trades, 0);
    assert!((carried.seam_cost - 0.0).abs() < 1e-15);
    s.pool_reset_flat = true;
    let reset = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(reset.pooled.trades, 2, "one entry per window when reset");
}

#[test]
fn cross_section_merge_carries_sum_and_count() {
    // Symbol A is present at t0 and t1; symbol B only at t0. A pre-divided
    // mean cannot be merged, because the denominators differ.
    let mut sequential = walkforward::CrossSection::default();
    sequential.add(0.0, 0.1, 0.1);
    sequential.add(0.0, 0.3, 0.3);
    sequential.add(1.0, 0.2, 0.2);
    let (seq_net, seq_gross) = sequential.series();

    let mut a = walkforward::CrossSection::default();
    a.add(0.0, 0.1, 0.1);
    a.add(1.0, 0.2, 0.2);
    let mut b = walkforward::CrossSection::default();
    b.add(0.0, 0.3, 0.3);
    let mut merged = a.clone();
    merged.merge(&b);
    let (net, gross) = merged.series();
    assert_eq!(net, seq_net);
    assert_eq!(gross, seq_gross);
    assert_eq!(merged.periods[0].count, 2, "t0 has two groups");
    assert_eq!(merged.periods[1].count, 1, "t1 has one group");
}

#[test]
fn expanding_folds_report_the_sign_test_as_not_applicable() {
    let rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.01),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, 0.01),
        (6.0, 6.0, 0.01),
    ];
    let t = single(&rows);
    let mut s = spec(
        "x",
        vec![
            window("early", 0.0, 3.0, 3.0, 4.0),
            // Expanding: train_2 contains test_1 [3,4).
            window("late", 0.0, 5.0, 5.0, 6.0),
        ],
    );
    s.threshold_quantile = 0.0;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert!(!r.sign_test.applicable);
    assert_eq!(r.sign_test.p_value, None);
    assert!(r.sign_test.method.contains("expanding"));
    assert!(r.json().contains("\"applicable\":false"));
}

#[test]
fn disjoint_rolling_folds_compute_a_binomial_sign_test() {
    let rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.01),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, 0.0),
        (6.0, 6.0, 0.01),
    ];
    let t = single(&rows);
    let mut s = spec(
        "x",
        vec![
            window("a", 0.0, 2.0, 2.0, 3.0),
            window("b", 3.0, 4.0, 4.0, 5.0),
            window("c", 5.0, 6.0, 6.0, 7.0),
        ],
    );
    s.threshold_quantile = 0.0;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert!(r.sign_test.applicable, "{}", r.sign_test.method);
    assert_eq!(r.sign_test.positive, 3);
    assert_eq!(r.sign_test.total, 3);
    assert!((r.sign_test.p_value.unwrap() - 0.125).abs() < 1e-15);
}

#[test]
fn candidates_are_selected_on_validation_never_on_test() {
    // Validation MSE prefers `x`; the test block would prefer `-x` (which
    // stays flat). Changing the test targets must not move the choice.
    let rows = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),  // validation
        (3.0, 3.0, 10.0), // validation
        (4.0, 4.0, -0.02),
        (5.0, 5.0, -0.03),
    ];
    let t = single(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 6.0)]);
    s.validation_frac = 0.5;
    s.threshold_quantile = 0.5;
    s.candidates = vec!["x".to_string(), "-x".to_string()];
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    assert_eq!(r.candidates, vec!["x".to_string(), "-x".to_string()]);
    assert_eq!(r.windows[0].candidate, 0, "validation picks x");
    assert_eq!(r.selected_on, "validation");
    assert!(!r.test_used_for_selection);
    let err = r.windows[0].validation_error.unwrap();
    assert!((err - 26.5).abs() < 1e-12, "validation MSE {}", err);
    assert_eq!(r.windows[0].expression, "x");

    // Flip the test target signs: the test block would now prefer -x, but the
    // report still chooses x because selection happened on validation.
    let flipped: Vec<(f64, &str, f64, f64)> = rows
        .iter()
        .map(|&(ts, x, y)| (ts, "A", x, if ts >= 4.0 { -y } else { y }))
        .collect();
    let t2 = table(&flipped);
    let r2 = walkforward::run_table(&t2, &s, "test").unwrap();
    assert_eq!(r2.windows[0].candidate, 0);
    assert_eq!(r2.windows[0].validation_error, Some(err));

    // Multiple candidates without a validation slice are refused.
    s.validation_frac = 0.0;
    let e = walkforward::run_table(&t, &s, "test").unwrap_err();
    assert!(e.contains("validation"), "{}", e);
}
