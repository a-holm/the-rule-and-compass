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
        gpu: false,
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

#[test]
fn final_exit_is_charged_per_group_when_rows_are_dropped() {
    // Group 1 has a NaN-target row first, group 2 is the single last row.
    // Both groups end held, so both pay an entry and a final exit.
    let time = [0.0, 1.0, 2.0, 3.0, 0.0];
    let group = [1u64, 1, 1, 1, 2];
    let target = [f64::NAN, 0.01, 0.02, 0.03, 0.04];
    let scores = [1.0; 5];
    let mut previous = std::collections::HashMap::new();
    let t = walkforward::reference_cell(&time, &group, &target, &scores, 0.0, 0.001, &mut previous);
    assert_eq!(t.rows, 4);
    assert_eq!(t.final_exits, 2);
    assert_eq!(t.entries, 2);
    assert!((t.gross - 0.10).abs() < 1e-15);
    assert!((t.cost - 0.004).abs() < 1e-15);
    assert!((t.net - 0.096).abs() < 1e-15);
}

#[test]
fn overlapping_test_rows_are_refused_and_the_spacing_is_reported() {
    let rows: Vec<(f64, f64, f64)> = (0..8).map(|i| (3600.0 * i as f64, i as f64, 0.001)).collect();
    let t = single(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 2.0 * 3600.0, 4.0 * 3600.0, 8.0 * 3600.0)]);
    s.horizon = 7200.0;
    let err = walkforward::run_table(&t, &s, "t").unwrap_err();
    assert!(err.contains("overlapping returns"), "{err}");
    s.horizon = 3600.0;
    let r = walkforward::run_table(&t, &s, "t").unwrap();
    assert_eq!(r.min_group_step, Some(3600.0));
    assert!(r.json().contains("\"min_group_step_seconds\":3600"));
}

#[test]
fn rows_without_a_return_do_not_count_toward_the_spacing() {
    let rows: Vec<(f64, f64, f64)> = (0..8)
        .map(|i| (3600.0 * i as f64, i as f64, if i % 2 == 0 { 0.001 } else { f64::NAN }))
        .collect();
    let t = single(&rows);
    let mut s = spec("x", vec![window("w", 0.0, 2.0 * 3600.0, 4.0 * 3600.0, 8.0 * 3600.0)]);
    s.horizon = 7200.0;
    let r = walkforward::run_table(&t, &s, "t").unwrap();
    assert_eq!(r.min_group_step, Some(7200.0));
}

#[test]
fn the_cli_exits_2_on_overlapping_returns() {
    let dir = std::env::temp_dir().join(format!("rcsr-wf-overlap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,target\n");
    for i in 0..8 {
        csv.push_str(&format!("{},{},0.001\n", 3600 * i, i));
    }
    std::fs::write(&data, csv).unwrap();
    let run = |horizon: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "walkforward", "--data", data.to_str().unwrap(), "--target", "target", "--expr", "x",
                "--time", "ts", "--horizon", horizon, "--cost-bps", "5", "--periods-per-year", "365",
                "--window", "w,0,7200,14400,28800", "--json",
            ])
            .output()
            .unwrap()
    };
    let bad = run("7200");
    assert_eq!(bad.status.code(), Some(2), "{}", String::from_utf8_lossy(&bad.stderr));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("overlapping returns"));
    let ok = run("3600");
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

mod front {
    use rcsr::devlog::Json;
    use std::path::PathBuf;
    use std::process::Command;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rcsr-front-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const MEMBERS: &str = r#"[{"expression":"x"},{"expression":"x * 2 + 0.1"},{"expression":"0.5 - x"}]"#;
    const RUN: &str = r#"{"tool":"rcsr","variables":["x"],"target_spec":{"return_column":"target"},
        "split":{"train_time":[0,119],"validation_time":[120,179],"required_purge_seconds":5,"purge_seconds":9},
        "search":{"candidates_evaluated":4321},
        "objective":{"spec":{"threshold":0,"per_side_bps":5}},
        "pareto":[{"expression":"x"},{"expression":"x * 2 + 0.1"},{"expression":"0.5 - x"}]}"#;

    fn write_data(dir: &std::path::Path) -> String {
        let mut csv = String::from("ts,x,target\n");
        for i in 0..400 {
            let x = ((i * 37) % 17) as f64 - 8.0;
            let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
            csv.push_str(&format!("{},{},{}\n", i, x, y));
        }
        let path = dir.join("d.csv");
        std::fs::write(&path, csv).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn write_run(dir: &std::path::Path, text: &str) -> String {
        let path = dir.join("run.json");
        std::fs::write(&path, text).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn rcsr(args: &[&str]) -> (i32, String, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    fn wf(data: &str, source: &[&str], window: &str, extra: &[&str]) -> (i32, String, String) {
        let mut a = vec![
            "walkforward", "--data", data, "--time", "ts", "--horizon", "1", "--periods-per-year", "365",
            "--validation-frac", "0.3", "--window", window, "--json",
        ];
        a.extend_from_slice(source);
        a.extend_from_slice(extra);
        rcsr(&a)
    }

    /// last seen 179 + required purge 5
    const OK_WINDOW: &str = "late,0,150,184,300";

    #[test]
    fn front_scores_every_member_and_matches_explicit_expr_flags_bit_for_bit() {
        let dir = scratch("equal");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, a, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--batch-pnl"]);
        assert_eq!(code, 0, "{err}");
        let flags = [
            "--expr", "x", "--expr", "x * 2 + 0.1", "--expr", "0.5 - x", "--batch-pnl", "--threshold-fixed", "0",
            "--cost-bps", "5", "--target", "target", "--vars", "x", "--trials", "4321",
        ];
        let (code, b, err) = wf(&data, &[], OK_WINDOW, &flags);
        assert_eq!(code, 0, "{err}");
        let (ja, jb) = (Json::parse(a.trim()).unwrap(), Json::parse(b.trim()).unwrap());
        let field = |j: &Json, k: &str| format!("{:?}", j.get(k).unwrap_or_else(|| panic!("missing {k}")));
        for k in ["candidates", "windows", "pooled"] {
            assert_eq!(field(&ja, k), field(&jb, k), "{k}");
        }
        let (pa, pb) = (ja.get("batch_pnl").unwrap(), jb.get("batch_pnl").unwrap());
        for (k, sub) in [("finalists", None), ("per_window", Some("cells")), ("pooled", Some("cells"))] {
            let pick = |p: &Json| {
                let v = p.get(k).unwrap_or_else(|| panic!("missing {k}"));
                format!("{:?}", sub.map_or(v, |s| v.get(s).unwrap_or_else(|| panic!("missing {k}.{s}"))))
            };
            assert_eq!(pick(pa), pick(pb), "{k}");
        }
        assert!(matches!(ja.get("batch_pnl"), Some(Json::Obj(_))));
        assert!(matches!(ja.get("candidates"), Some(Json::Arr(c)) if c.len() == 3));
        let sel = ja.get("selection").unwrap();
        assert_eq!(sel.get("source").and_then(Json::as_str), Some("front"));
        assert_eq!(sel.get("run").and_then(Json::as_str), Some(run.as_str()));
        assert_eq!(sel.get("run_sha256").and_then(Json::as_str).map(str::len), Some(64));
        assert!(matches!(sel.get("candidate_labels"), Some(Json::Arr(l)) if l.len() == 3));
    }

    #[test]
    fn front_trials_default_to_candidates_evaluated_and_can_be_overridden() {
        let dir = scratch("trials");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, a, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        assert!(a.contains("\"trials\":4321"), "{a}");
        let (code, b, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--trials", "77", "--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        assert!(b.contains("\"trials\":77"), "{b}");
    }

    #[test]
    fn a_window_inside_the_validation_slice_is_refused() {
        let dir = scratch("inside");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, _, err) = wf(&data, &["--front", &run], "bad,0,100,150,300", &[]);
        assert_eq!(code, 2, "{err}");
        assert!(
            err.contains("window 'bad' tests rows the run trained or selected on (test_start 150 < last seen 179 + purge 5)"),
            "{err}"
        );
        let (code, _, err) = wf(&data, &["--front", &run], "bad,0,100,183,300", &[]);
        assert_eq!(code, 2, "{err}");
    }

    #[test]
    fn a_window_starting_at_seen_plus_purge_passes() {
        let dir = scratch("edge");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &[]);
        assert_eq!(code, 0, "{err}");
        assert!(!out.contains("in_sample"), "{out}");
    }

    #[test]
    fn the_override_reports_the_window_as_in_sample() {
        let dir = scratch("override");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], "bad,0,100,150,300", &["--allow-in-sample-windows"]);
        assert_eq!(code, 0, "{err}");
        assert!(out.contains("\"in_sample\":true"), "{out}");
    }

    #[test]
    fn the_run_position_rule_is_carried_and_explicit_flags_win() {
        let dir = scratch("rule");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &[]);
        assert_eq!(code, 0, "{err}");
        assert!(out.contains("\"per_side_bps\":5,"), "{out}");
        assert!(out.contains("\"threshold_source\":\"fixed\""), "{out}");
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--cost-override", "12", "--threshold-quantile", "0.5"]);
        assert_eq!(code, 0, "{err}");
        assert!(out.contains("\"per_side_bps\":12,"), "{out}");
        assert!(!out.contains("\"threshold_source\":\"fixed\""), "{out}");
    }

    #[test]
    fn refusals() {
        let dir = scratch("refuse");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, _, err) = wf(&data, &["--front", &run, "--expr", "x"], OK_WINDOW, &[]);
        assert_eq!(code, 2, "{err}");
        assert!(err.contains("alternatives"), "{err}");

        let empty = write_run(&dir, &RUN.replace(MEMBERS, "[]"));
        let (code, _, err) = wf(&data, &["--front", &empty], OK_WINDOW, &[]);
        assert_eq!(code, 3, "{err}");
        assert!(err.contains("front is empty"), "{err}");

        let no_time = write_run(&dir, &RUN.replace(r#""train_time":[0,119],"validation_time":[120,179],"#, ""));
        let (code, _, err) = wf(&data, &["--front", &no_time], OK_WINDOW, &[]);
        assert_eq!(code, 3, "{err}");
        assert!(err.contains("no time column"), "{err}");

        let no_count = write_run(&dir, &RUN.replace(r#""search":{"candidates_evaluated":4321},"#, ""));
        let (code, _, err) = wf(&data, &["--front", &no_count], OK_WINDOW, &[]);
        assert_eq!(code, 2, "{err}");
        assert!(err.contains("--trials"), "{err}");
        let (code, _, err) = wf(&data, &["--front", &no_count], OK_WINDOW, &["--trials", "10"]);
        assert_eq!(code, 0, "{err}");

        let missing_var = write_run(&dir, &RUN.replace(r#"{"expression":"x"}"#, r#"{"expression":"zz"}"#));
        let (code, _, err) = wf(&data, &["--front", &missing_var], OK_WINDOW, &[]);
        assert_eq!(code, 3, "{err}");
        assert!(err.contains("zz"), "{err}");
    }
}

#[path = "common/ledger.rs"]
mod ledger;

#[test]
fn the_printed_json_carries_a_phase_ledger_outside_the_reproduction_hash() {
    let mut csv = String::from("ts,x,target\n");
    for t in 0..64 {
        csv.push_str(&format!("{},{},{}\n", t, (t % 7) as f64, ((t * 3) % 5) as f64 * 0.001 - 0.002));
    }
    let path = std::env::temp_dir().join(format!("rcsr-wf-ledger-{}.csv", std::process::id()));
    std::fs::write(&path, csv).unwrap();
    let s = spec("x", vec![window("w", 0.0, 32.0, 32.0, 64.0)]);
    let r = walkforward::run_path_with(path.to_str().unwrap(), &s, &walkforward::ProtocolOptions::default()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!r.json().contains("\"ledger\""), "the hashed report must stay deterministic");
    let l = ledger::check(&r.json_with_ledger(None), "walkforward", &["io", "pnl"]);
    assert_eq!(l.busy, 0.0, "the scalar evaluator has no device intervals");
}
