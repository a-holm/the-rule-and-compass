//! Walk-forward evaluation (issue #124).
//!
//! The tests use a tiny table whose scores, threshold and forward returns are
//! known by hand. They pin the long/flat rule, the per-side cost, the
//! train-only threshold, the purge check, pooled aggregation and the direct
//! `.rcsb` path.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::data;
use rcsr::store;
use rcsr::walkforward::{self, EvalSpec, Window};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

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

    let scratch = Scratch::new("walkforward");
    let path = scratch.join("table.rcsb").to_string_lossy().into_owned();
    store::write_store(&path, &t, 3).unwrap();
    let streamed = walkforward::run_path(&path, &s).unwrap();

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
    let scratch = Scratch::new("wf-overlap");
    let dir = scratch.path().to_path_buf();
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,target\n");
    for i in 0..8 {
        csv.push_str(&format!("{},{},0.001\n", 3600 * i, i));
    }
    std::fs::write(&data, csv).unwrap();
    let run = |horizon: &str| {
        rcsr_cmd::rcsr()
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
}

#[test]
fn device_pnl_is_refused_with_policy_band() {
    let scratch = Scratch::new("wf-band-device");
    let dir = scratch.path().to_path_buf();
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,target\n");
    for i in 0..8 {
        csv.push_str(&format!("{},{},0.001\n", 3600 * i, i));
    }
    std::fs::write(&data, csv).unwrap();
    let run = |extra: &[&str]| {
        rcsr_cmd::rcsr()
            .args([
                "walkforward", "--data", data.to_str().unwrap(), "--target", "target", "--expr", "x",
                "--time", "ts", "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365",
                "--window", "w,0,7200,14400,28800", "--json", "--policy", "band",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    for band in ["0", "0.001"] {
        let bad = run(&["--band", band, "--device-pnl"]);
        let stderr = String::from_utf8_lossy(&bad.stderr);
        assert_eq!(bad.status.code(), Some(2), "--band {band}: {stderr}");
        assert!(
            stderr.contains("--device-pnl is not supported with --policy band: the device arm prices the threshold book"),
            "--band {band}: {stderr}"
        );
    }
    let ok = run(&["--band", "0.001"]);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));

    let t = single(&[(0.0, 1.0, 0.001), (3600.0, 2.0, 0.001)]);
    let mut s = spec("x", vec![window("w", 0.0, 1.0, 1.0, 2.0)]);
    s.policy = walkforward::Policy::Band;
    s.device_pnl = true;
    let e = walkforward::run_table(&t, &s, "t").err().expect("band with device_pnl is refused");
    assert!(e.contains("--device-pnl is not supported with --policy band"), "{e}");
}

mod front {
    use rcsr::devlog::Json;
    use std::path::PathBuf;

    use super::scratch::Scratch;

    fn scratch(name: &str) -> (Scratch, PathBuf) {
        let scratch = Scratch::new(name);
        let path = scratch.path().to_path_buf();
        (scratch, path)
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
        let out = super::rcsr_cmd::rcsr().args(args).output().unwrap();
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

    /// Like `wf` for a run that declares its target spec: the spec sets the period width.
    fn wf_book(data: &str, source: &[&str], window: &str, extra: &[&str]) -> (i32, String, String) {
        let mut a = vec!["walkforward", "--data", data, "--time", "ts", "--validation-frac", "0.3", "--window", window, "--json"];
        a.extend_from_slice(source);
        a.extend_from_slice(extra);
        rcsr(&a)
    }

    /// last seen 179 + required purge 5
    const OK_WINDOW: &str = "late,0,150,184,300";

    #[test]
    fn front_scores_every_member_and_matches_explicit_expr_flags_bit_for_bit() {
        let (_scratch, dir) = scratch("equal");
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
    fn batch_pnl_report_hash_is_reproducible_and_timings_stay_outside_it() {
        let (_scratch, dir) = scratch("batchhash");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let sha = |o: &str| o.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        let (code, a, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--batch-pnl"]);
        assert_eq!(code, 0, "{err}");
        let (code, b, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--batch-pnl"]);
        assert_eq!(code, 0, "{err}");
        assert_eq!(sha(&a), sha(&b));
        let ja = Json::parse(a.trim()).unwrap();
        let timings = ja.get("batch_timings").expect("batch_timings object");
        for seam in ["per_window", "pooled"] {
            let t = timings.get(seam).unwrap_or_else(|| panic!("missing {seam}"));
            let stats = ja.get("batch_pnl").unwrap().get(seam).unwrap().get("stats").unwrap();
            for k in [
                "wall_s", "device_s", "score_s", "layout_s", "reference_s", "candidates_per_s", "row_evals_per_s",
                "device_fraction",
            ] {
                assert!(t.get(k).is_some(), "{seam}.{k}");
                assert!(stats.get(k).is_none(), "{seam}.stats.{k} is hashed");
            }
        }
    }

    const BOOK_RUN: &str = r#"{"tool":"rcsr","variables":["x"],
        "target_spec":{"column":"target","name":"target","kind":"forward_return","deprecated":false,
            "return_column":"target","return_units":"log","horizon_seconds":1,"entry_offset_seconds":0,
            "delay_seconds":0,"entry_price":"next open","exit_price":"close","offset":0.0,
            "scale":{"kind":"constant","value":1.0},"clip":null,"units":"log_return","cost_bps":null,
            "cost_in_label":false,"construction_columns":[],"note":null},
        "split":{"train_time":[0,119],"validation_time":[120,179],"required_purge_seconds":5,"purge_seconds":9},
        "search":{"candidates_evaluated":1000},
        "objective":{"spec":{"threshold":0,"per_side_bps":5,"max_position":1,"cash_return":0},
            "finalist":{"expression":"x"}},
        "pareto":[{"expression":"x"},{"expression":"x * 2 + 0.1"},{"expression":"0.5 - x"}]}"#;

    const IN_SAMPLE_WINDOW: &str = "bad,0,100,150,300";

    fn num(j: &Json, path: &[&str]) -> f64 {
        let mut cur = j;
        for k in path {
            cur = cur.get(k).unwrap_or_else(|| panic!("missing {k} in {path:?}"));
        }
        cur.as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
    }

    fn multiple(out: &str) -> Json {
        let j = Json::parse(out.trim()).unwrap();
        j.get("inference").and_then(|i| i.get("multiple_testing")).expect("multiple_testing").clone()
    }

    fn source(m: &Json) -> String {
        m.get("n_trials_source").and_then(Json::as_str).expect("n_trials_source").to_string()
    }

    fn mt(m: &Json, path: &[&str]) -> f64 {
        num(m, path)
    }

    #[test]
    fn from_run_trials_are_one_out_of_sample_book() {
        let (_scratch, dir) = scratch("fromrun1");
        let data = write_data(&dir);
        let run = write_run(&dir, BOOK_RUN);
        let (code, a, err) = wf_book(&data, &["--from-run", &run], OK_WINDOW, &["--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&a);
        assert_eq!(mt(&m, &["trials"]), 1.0);
        assert_eq!(source(&m), "out_of_sample_book");
        let (code, b, err) = wf_book(&data, &["--from-run", &run], OK_WINDOW, &["--bootstrap", "100", "--trials", "1"]);
        assert_eq!(code, 0, "{err}");
        let mb = multiple(&b);
        assert_eq!(source(&mb), "flag");
        for k in ["dsr_at_n_eff", "dsr_at_10_n_eff"] {
            assert_eq!(format!("{:?}", m.get(k)), format!("{:?}", mb.get(k)), "{k}");
        }
        assert_eq!(format!("{:?}", m.get("psr")), format!("{:?}", m.get("dsr_at_n_eff")));
        let (code, c, err) = wf_book(&data, &["--from-run", &run], OK_WINDOW, &["--bootstrap", "100", "--trials", "77"]);
        assert_eq!(code, 0, "{err}");
        let mc = multiple(&c);
        assert_eq!(mt(&mc, &["trials"]), 77.0);
        assert_eq!(source(&mc), "flag");
    }

    #[test]
    fn in_sample_window_deflates_by_the_search_trials() {
        let (_scratch, dir) = scratch("fromrun2");
        let data = write_data(&dir);
        let run = write_run(&dir, BOOK_RUN);
        let extra = ["--bootstrap", "100", "--allow-in-sample-windows"];
        let (code, a, err) = wf_book(&data, &["--from-run", &run], IN_SAMPLE_WINDOW, &extra);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&a);
        assert_eq!(mt(&m, &["trials"]), 1000.0);
        assert_eq!(source(&m), "run.candidates_evaluated");
        assert!(mt(&m, &["dsr_at_n_eff", "sr0"]) > 0.0);
        assert!(mt(&m, &["dsr_at_n_eff", "dsr"]) < mt(&m, &["psr", "dsr"]));
        assert_eq!(mt(&m, &["psr", "sr0"]), 0.0);
    }

    #[test]
    fn front_trials_follow_what_selected_on_the_scored_rows() {
        let (_scratch, dir) = scratch("trials");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, a, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&a);
        assert_eq!(mt(&m, &["trials"]), 3.0);
        assert_eq!(source(&m), "front_size");
        let (code, a, err) = wf(&data, &["--front", &run], IN_SAMPLE_WINDOW, &["--bootstrap", "100", "--allow-in-sample-windows"]);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&a);
        assert_eq!(mt(&m, &["trials"]), 4321.0);
        assert_eq!(source(&m), "run.candidates_evaluated");
        let (code, b, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--trials", "77", "--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&b);
        assert_eq!(mt(&m, &["trials"]), 77.0);
        assert_eq!(source(&m), "flag");
    }

    #[test]
    fn trials_fall_back_to_front_size() {
        let (_scratch, dir) = scratch("fallback");
        let data = write_data(&dir);
        let no_count = [
            ("--front", RUN.replace(r#""search":{"candidates_evaluated":4321},"#, "")),
            ("--from-run", BOOK_RUN.replace(r#""search":{"candidates_evaluated":1000},"#, "")),
        ];
        for (flag, text) in &no_count {
            let path = write_run(&dir, text);
            for (window, extra) in [
                (OK_WINDOW, vec!["--bootstrap", "100"]),
                (IN_SAMPLE_WINDOW, vec!["--bootstrap", "100", "--allow-in-sample-windows"]),
            ] {
                let (code, out, err) = if *flag == "--from-run" { wf_book(&data, &[*flag, &path], window, &extra) } else { wf(&data, &[*flag, &path], window, &extra) };
                assert_eq!(code, 0, "{flag} {window}: {err}");
                let m = multiple(&out);
                let in_sample = window == IN_SAMPLE_WINDOW;
                let (n, s) = match (*flag, in_sample) {
                    ("--from-run", false) => (1.0, "out_of_sample_book"),
                    _ => (3.0, "front_size"),
                };
                assert_eq!(mt(&m, &["trials"]), n, "{flag} {window}");
                assert_eq!(source(&m), s, "{flag} {window}");
            }
        }
    }

    #[test]
    fn min_btl_years_in_report() {
        let (_scratch, dir) = scratch("minbtl");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        let j = Json::parse(out.trim()).unwrap();
        let m = multiple(&out);
        let sharpe_annual = num(&j, &["inference", "bootstrap", "sharpe_annual"]);
        let want = rcsr::inference::min_btl_years(sharpe_annual, mt(&m, &["n_eff"]));
        let got = m.get("min_btl_years").and_then(Json::as_f64);
        if want.is_nan() {
            assert!(got.is_none(), "{got:?}");
        } else {
            assert!((got.expect("min_btl_years") - want).abs() < 1e-12, "{got:?} vs {want}");
        }
    }

    #[test]
    fn front_identical_columns_dsr_equals_psr() {
        let (_scratch, dir) = scratch("identical");
        let data = write_data(&dir);
        let same = RUN.replace(r#"{"expression":"x * 2 + 0.1"},{"expression":"0.5 - x"}"#, r#"{"expression":"x"}"#);
        let run = write_run(&dir, &same);
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &["--bootstrap", "100"]);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&out);
        assert_eq!(mt(&m, &["trials"]), 2.0);
        assert!(mt(&m, &["n_eff"]) <= 1.0 + 1e-9, "{m:?}");
        assert_eq!(format!("{:?}", m.get("psr").and_then(|p| p.get("dsr"))), format!("{:?}", m.get("dsr_at_n_eff").and_then(|p| p.get("dsr"))));
        assert_eq!(mt(&m, &["dsr_at_n_eff", "sr0"]), 0.0);
        assert_eq!(m.get("min_btl_years").and_then(Json::as_f64).unwrap_or(0.0), 0.0);
    }

    #[test]
    fn trials_source_labels_for_vote_and_a_raised_flag() {
        let (_scratch, dir) = scratch("labels");
        let data = write_data(&dir);
        let run = write_run(&dir, BOOK_RUN);
        let vote = ["--bootstrap", "100", "--policy", "vote", "--quorum", "0.5"];
        let (code, a, err) = wf_book(&data, &["--from-run", &run], OK_WINDOW, &vote);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&a);
        assert_eq!(source(&m), "out_of_sample_book", "{m:?}");
        let mut flagged = vote.to_vec();
        flagged.extend(["--trials", "5"]);
        let (code, b, err) = wf_book(&data, &["--from-run", &run], OK_WINDOW, &flagged);
        assert_eq!(code, 0, "{err}");
        assert_eq!(source(&multiple(&b)), "flag");
        assert_eq!(mt(&multiple(&b), &["trials"]), 5.0);
        let exprs = ["--target", "target", "--vars", "x", "--expr", "x", "--expr", "x * 2 + 0.1", "--expr", "0.5 - x", "--expr", "x - 0.1", "--expr", "x + 0.2"];
        let mut raised = exprs.to_vec();
        raised.extend(["--bootstrap", "100", "--trials", "2"]);
        let (code, c, err) = wf(&data, &[], OK_WINDOW, &raised);
        assert_eq!(code, 0, "{err}");
        let m = multiple(&c);
        assert_eq!(mt(&m, &["trials"]), 5.0);
        assert_eq!(source(&m), "candidate_count");
        let mut kept = exprs.to_vec();
        kept.extend(["--bootstrap", "100", "--trials", "9"]);
        let (code, d, err) = wf(&data, &[], OK_WINDOW, &kept);
        assert_eq!(code, 0, "{err}");
        assert_eq!(mt(&multiple(&d), &["trials"]), 9.0);
        assert_eq!(source(&multiple(&d)), "flag");
    }

    #[test]
    fn a_window_inside_the_validation_slice_is_refused() {
        let (_scratch, dir) = scratch("inside");
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
        let (_scratch, dir) = scratch("edge");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], OK_WINDOW, &[]);
        assert_eq!(code, 0, "{err}");
        assert!(!out.contains("in_sample"), "{out}");
    }

    #[test]
    fn the_override_reports_the_window_as_in_sample() {
        let (_scratch, dir) = scratch("override");
        let data = write_data(&dir);
        let run = write_run(&dir, RUN);
        let (code, out, err) = wf(&data, &["--front", &run], "bad,0,100,150,300", &["--allow-in-sample-windows"]);
        assert_eq!(code, 0, "{err}");
        assert!(out.contains("\"in_sample\":true"), "{out}");
    }

    #[test]
    fn the_run_position_rule_is_carried_and_explicit_flags_win() {
        let (_scratch, dir) = scratch("rule");
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
        let (_scratch, dir) = scratch("refuse");
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
        assert_eq!(code, 0, "{err}");
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
    let scratch = Scratch::new("wf-ledger");
    let path = scratch.join("data.csv");
    std::fs::write(&path, csv).unwrap();
    let s = spec("x", vec![window("w", 0.0, 32.0, 32.0, 64.0)]);
    let r = walkforward::run_path_with(path.to_str().unwrap(), &s, &walkforward::ProtocolOptions::default()).unwrap();
    assert!(!r.json().contains("\"ledger\""), "the hashed report must stay deterministic");
    let l = ledger::check(&r.json_with_ledger(None), "walkforward", &["io", "pnl"]);
    assert_eq!(l.busy, 0.0, "the scalar evaluator has no device intervals");
}


// ---- #925: the band in units of the training-score sd ----

fn sd_band_table(test_scale: f64, validation_scale: f64) -> data::Table {
    let mut rows = Vec::new();
    let mut v = 0.0f64;
    for i in 0..300 {
        let f = i as f64;
        v = 0.8 * v + ((f * 0.618_033_988_749_894_9).fract() - 0.5);
        let y = 0.01 * ((f * 0.754_877_666_246_692_7).fract() - 0.5);
        let (x, y) = if i >= 200 {
            (v * test_scale + 3.0, y * test_scale)
        } else if i >= 150 {
            (v * validation_scale - 2.0, y)
        } else {
            (v, y)
        };
        rows.push((f, x, y));
    }
    single(&rows)
}

fn sd_band_report(t: &data::Table, units: rcsr::trading::BandUnits) -> walkforward::Report {
    let mut s = spec("x", vec![window("w", 0.0, 200.0, 200.0, 300.0)]);
    s.policy = walkforward::Policy::Band;
    s.band_width = 1.0;
    s.min_hold = 2;
    s.threshold_fixed = Some(0.0);
    s.validation_frac = 0.25;
    let opts = walkforward::ProtocolOptions { band_units: units, ..walkforward::ProtocolOptions::default() };
    walkforward::run_table_with(t, &s, "test", &opts).unwrap()
}

#[test]
fn walkforward_band_sd_ignores_test_rows() {
    use rcsr::trading::BandUnits;
    let base = sd_band_report(&sd_band_table(1.0, 1.0), BandUnits::Sd);
    let band = |r: &walkforward::Report| r.windows[0].band.as_ref().expect("the band book").clone();
    let b0 = band(&base);
    let spec0 = b0.spec.clone().expect("the band spec");
    let sd = spec0.sd.expect("a finite training sd");
    assert_eq!(spec0.units, BandUnits::Sd);
    assert_eq!(spec0.sd_rows, Some(150));
    assert_eq!(spec0.sd_max_time, Some(149.0));
    assert_eq!(b0.band_width.to_bits(), (1.0 * sd).to_bits());
    assert_eq!(spec0.resolved.to_bits(), b0.band_width.to_bits());
    // Rows at or after the training end, and validation rows, never enter the sd.
    for (test_scale, validation_scale) in [(1000.0, 1.0), (1.0, 50.0), (1000.0, 50.0)] {
        let other = sd_band_report(&sd_band_table(test_scale, validation_scale), BandUnits::Sd);
        let s = band(&other).spec.expect("the band spec");
        assert_eq!(s.sd.map(f64::to_bits), Some(sd.to_bits()), "test x{test_scale}, validation x{validation_scale}");
        assert_eq!(s.resolved.to_bits(), spec0.resolved.to_bits());
        assert_eq!(s.sd_max_time, Some(149.0));
    }
    let json = base.json();
    assert!(json.contains("\"band_units\":\"sd\""), "policy band_units");
    assert!(json.contains(&format!("\"band_sd\":{}", sd)), "windows[].band.band_sd");
    // Score units keep the multiple as the half-width.
    let score = sd_band_report(&sd_band_table(1.0, 1.0), BandUnits::Score);
    let s = band(&score);
    assert_eq!(s.band_width, 1.0);
    assert_eq!(s.spec.expect("the band spec").sd, None);
}

// ---------------------------------------------------------------------------
// #931: the purge at 72, 168 and 336 hours.
// ---------------------------------------------------------------------------

#[path = "common/long_horizon.rs"]
mod lh;

/// 2020-09-13T12:00:00Z.
const LH_T0: i64 = 1_600_000_000 - 1_600_000_000 % 3600;

/// Hourly rows `ts,x,target` with the stride-`h` forward return of real bars;
/// rows off the stride carry no return.
fn long_label_table(h: usize, n: usize) -> (Vec<rcsr::ohlcv::Bar>, rcsr::labels::LongLabels, data::Table) {
    let bars = lh::bars(LH_T0, n, 0.0);
    let l = lh::build(&bars, &lh::src(h, &format!(",stride={h}")));
    let mut csv = String::from("ts,x,target\n");
    for t in 0..n {
        csv.push_str(&format!("{},{},{}\n", bars[t].time, ((t * 7) % 11) as f64, lh::fmt(l.fwd[t])));
    }
    (bars, l, data::parse_table(&csv).unwrap())
}

fn no_training_label_reaches_test_block(h: usize) {
    let (bars, l, table) = long_label_table(h, 30 * h);
    let spec_l = lh::spec(&lh::src(h, ""));
    let purge = spec_l.purge_seconds() as f64;
    let opts = walkforward::ProtocolOptions { entry_offset: 3600.0, ..walkforward::ProtocolOptions::default() };
    let test_start = bars[20 * h].time as f64;
    let test_end = bars[26 * h].time as f64;
    let run = |train_end: f64| {
        let mut s = spec("x", vec![window("w", LH_T0 as f64, train_end, test_start, test_end)]);
        s.horizon = spec_l.horizon_seconds() as f64;
        s.periods_per_year = Some(8760.0 / h as f64);
        walkforward::run_table_with(&table, &s, "t", &opts)
    };
    for short in [test_start - lh::DEFAULT_PURGE, test_start - purge + 1.0] {
        let err = run(short).unwrap_err();
        assert!(err.contains("purge gap") && err.contains(&format!("needs {purge}")), "h{h}: {err}");
    }
    let train_end = test_start - purge;
    let r = run(train_end).unwrap();
    let last = r.windows[0].train_last_time.expect("training rows");
    assert!(last < train_end && test_start - last >= purge, "h{h}: last training stamp {last}");
    let mut trained = 0;
    for t in 0..bars.len() {
        let time = bars[t].time as f64;
        if time < train_end && l.fwd[t].is_finite() {
            assert!(l.t1[t] < test_start, "h{h} row {t}: label end {} at or after the test start {test_start}", l.t1[t]);
            trained += 1;
        }
    }
    assert!(trained >= 18, "h{h}: {trained}");
    // The stride row one horizon before the test start has a vertical label end after it and is not trained on.
    let near = (bars[19 * h].time) as f64;
    assert!(near + purge > test_start && near >= train_end, "h{h}");
}

#[test]
fn no_training_label_reaches_test_block_h72() {
    no_training_label_reaches_test_block(72);
}

#[test]
fn no_training_label_reaches_test_block_h168() {
    no_training_label_reaches_test_block(168);
}

#[test]
fn no_training_label_reaches_test_block_h336() {
    no_training_label_reaches_test_block(336);
}

#[test]
fn walkforward_purge_floor_at_168h() {
    let scratch = Scratch::new("wf-h168-floor");
    let bars = lh::bars(LH_T0, 168 * 40, 0.0);
    let l = lh::build(&bars, &lh::src(168, ",stride=168"));
    let mut csv = String::from("ts,x,lbl_h168_fwd\n");
    for (t, b) in bars.iter().enumerate() {
        csv.push_str(&format!("{},{},{}\n", b.time, ((t * 7) % 11) as f64, lh::fmt(l.fwd[t])));
    }
    let data = scratch.join("d.csv");
    std::fs::write(&data, csv).unwrap();
    let spec_path = scratch.join("spec.json");
    std::fs::write(&spec_path, lh::spec_file(&lh::spec(&lh::src(168, ",stride=168")))).unwrap();
    let train_end = bars[20 * 168].time;
    let run = |gap: i64| {
        let w = format!("w,{},{},{},{}", LH_T0, train_end, train_end + gap, train_end + gap + 10 * 168 * 3600);
        rcsr_cmd::rcsr()
            .args([
                "walkforward", "--data", data.to_str().unwrap(), "--target", "lbl_h168_y", "--target-spec",
                spec_path.to_str().unwrap(), "--expr", "x", "--time", "ts", "--window", &w, "--json",
            ])
            .output()
            .unwrap()
    };
    let bad = run(608_399);
    let err = String::from_utf8_lossy(&bad.stderr);
    assert_eq!(bad.status.code(), Some(2), "{err}");
    assert!(err.contains("608400"), "{err}");
    let ok = run(608_400);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let out = String::from_utf8_lossy(&ok.stdout);
    assert!(out.contains("\"required_purge_seconds\":608400"), "{out}");
}

// ---------------------------------------------------------------------------
// #930: one book from K expressions (`--policy vote`).
// ---------------------------------------------------------------------------

/// Three members `x - 1`, `x - 2`, `x - 3` with theta 0: the votes of a row
/// with score `x` are 0 for x <= 1, 1 for 2, 2 for 3 and 3 for x >= 4.
const VOTE_X: [f64; 8] = [0.0, 2.0, 3.0, 4.0, 3.0, 2.0, 5.0, 1.0];
const VOTE_Y: [f64; 8] = [0.01, 0.02, 0.03, -0.01, 0.02, 0.04, 0.01, 0.05];

fn vote_table(x: &[f64; 8], y: &[f64; 8]) -> data::Table {
    let mut rows = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)];
    for i in 0..8 {
        rows.push((10.0 + i as f64, x[i], y[i]));
    }
    single(&rows)
}

fn vote_spec(quorum_members: &[&str]) -> EvalSpec {
    let mut s = spec(quorum_members[0], vec![window("w", 0.0, 2.0, 10.0, 18.0)]);
    s.candidates = quorum_members.iter().map(|m| m.to_string()).collect();
    s.policy = walkforward::Policy::Vote;
    s.threshold_fixed = Some(0.0);
    s
}

fn vote_opts(q: f64) -> walkforward::ProtocolOptions {
    walkforward::ProtocolOptions { quorum: Some(q), ..walkforward::ProtocolOptions::default() }
}

/// Equity of net returns `pos * y - cost * |delta pos|` (10 bp per side).
fn book_return(pos: &[f64], y: &[f64; 8]) -> f64 {
    let mut prev = 0.0;
    let mut equity = 1.0;
    for (i, (p, r)) in pos.iter().zip(y).enumerate() {
        // The last row of a long book also pays the final exit.
        let exit = if i + 1 == pos.len() { 0.001 * p } else { 0.0 };
        equity *= 1.0 + p * r - 0.001 * (p - prev).abs() - exit;
        prev = *p;
    }
    equity - 1.0
}

#[test]
fn vote_majority_book() {
    let t = vote_table(&VOTE_X, &VOTE_Y);
    let s = vote_spec(&["x - 1", "x - 2", "x - 3"]);
    let r = walkforward::run_table_with(&t, &s, "test", &vote_opts(0.66)).unwrap();
    assert_eq!(r.votes_required, 2);
    assert_eq!(r.candidates.len(), 3);
    let w = &r.windows[0];
    let pos = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    assert_eq!(w.metrics.rows, 8);
    assert_eq!(w.metrics.active_rows, 4);
    assert_eq!(w.metrics.entries, 2);
    assert_eq!(w.metrics.trades, 4);
    assert!((w.metrics.total_abs_delta - 4.0).abs() < 1e-15);
    assert!((w.metrics.total_return - book_return(&pos, &VOTE_Y)).abs() < 1e-12);
    assert_eq!(w.member_thresholds, vec![0.0, 0.0, 0.0]);
    let json = r.json();
    assert!(json.contains("\"name\":\"vote\""), "{json}");
    assert!(json.contains("\"vote\":{\"quorum\":0.66,\"votes_required\":2,\"members\":[\"x - 1\",\"x - 2\",\"x - 3\"]}"), "{json}");

    // Unanimity: only 4, 5 score 3 votes.
    let r = walkforward::run_table_with(&t, &s, "test", &vote_opts(1.0)).unwrap();
    assert_eq!(r.votes_required, 3);
    let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    assert!((r.windows[0].metrics.total_return - book_return(&pos, &VOTE_Y)).abs() < 1e-12);
}

#[test]
fn vote_one_member_is_threshold() {
    let t = vote_table(&VOTE_X, &VOTE_Y);
    for (fixed, validation) in [(Some(1.5), 0.0), (None, 0.0)] {
        let mut v = vote_spec(&["x"]);
        v.candidates = Vec::new();
        v.threshold_fixed = fixed;
        v.threshold_quantile = 0.5;
        v.validation_frac = validation;
        let mut th = v.clone();
        th.policy = walkforward::Policy::Threshold;
        let a = walkforward::run_table_with(&t, &v, "test", &vote_opts(1.0)).unwrap();
        let b = walkforward::run_table(&t, &th, "test").unwrap();
        assert_eq!(a.period_net.len(), b.period_net.len());
        for (x, y) in a.period_net.iter().zip(&b.period_net) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(a.period_time, b.period_time);
        let (wa, wb) = (&a.windows[0], &b.windows[0]);
        assert_eq!(
            (wa.metrics.rows, wa.metrics.active_rows, wa.metrics.trades, wa.metrics.entries),
            (wb.metrics.rows, wb.metrics.active_rows, wb.metrics.trades, wb.metrics.entries)
        );
        for (x, y) in [
            (wa.metrics.total_return, wb.metrics.total_return),
            (wa.metrics.mean_return, wb.metrics.mean_return),
            (wa.metrics.total_cost, wb.metrics.total_cost),
            (wa.metrics.total_abs_delta, wb.metrics.total_abs_delta),
        ] {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(a.pooled.total_return.to_bits(), b.pooled.total_return.to_bits());
        assert_eq!(a.seam_cost.to_bits(), b.seam_cost.to_bits());
    }
}

#[test]
fn vote_band_applies_after_voting() {
    let t = vote_table(&VOTE_X, &VOTE_Y);
    let mut s = vote_spec(&["x - 1", "x - 2", "x - 3"]);
    // Need 2 votes. Votes per row: 0 1 2 3 2 1 3 0. Band 1: enter at 2, stay
    // while votes >= 1, exit at 0.
    s.band_width = 1.0;
    let r = walkforward::run_table_with(&t, &s, "test", &vote_opts(0.66)).unwrap();
    let pos = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let w = &r.windows[0];
    assert_eq!(w.metrics.active_rows, 5);
    assert_eq!(w.metrics.trades, 2);
    assert!((w.metrics.total_return - book_return(&pos, &VOTE_Y)).abs() < 1e-12);
    assert!(w.band.is_some(), "the band scan priced the vote book");

    // Minimum hold 2 with no band: the exit at row 5 is allowed after two
    // long rows, the re-entry at row 6 is held through the exit signal at row 7.
    s.band_width = 0.0;
    s.min_hold = 2;
    let r = walkforward::run_table_with(&t, &s, "test", &vote_opts(0.66)).unwrap();
    let pos = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    assert!((r.windows[0].metrics.total_return - book_return(&pos, &VOTE_Y)).abs() < 1e-12);
}

#[test]
fn vote_refusals_exit_before_data_work() {
    let t = vote_table(&VOTE_X, &VOTE_Y);
    let base = vote_spec(&["x - 1", "x - 2"]);
    let refused = |s: &EvalSpec, o: &walkforward::ProtocolOptions, needle: &str| {
        let e = walkforward::run_table_with(&t, s, "t", o).err().expect("refused");
        assert!(e.contains(needle), "{e}");
    };
    let mut s = base.clone();
    s.device_pnl = true;
    refused(&s, &vote_opts(0.5), "--device-pnl is not supported with --policy vote");
    for q in [0.0, -0.5, 1.5, f64::NAN] {
        refused(&base, &vote_opts(q), "--quorum in (0, 1]");
    }
    refused(&base, &walkforward::ProtocolOptions::default(), "--quorum in (0, 1]");
    let mut s = base.clone();
    s.band_width = 0.5;
    refused(&s, &vote_opts(0.5), "whole number");
    let mut o = vote_opts(0.5);
    o.batch_pnl = true;
    refused(&base, &o, "--batch-pnl is not supported with --policy vote");

    let dir = Scratch::new("wf-vote-cli");
    let data = dir.path().join("d.csv");
    let mut csv = String::from("ts,x,target\n");
    for i in 0..8 {
        csv.push_str(&format!("{},{},0.001\n", 3600 * i, i));
    }
    std::fs::write(&data, csv).unwrap();
    let run = |extra: &[&str]| {
        rcsr_cmd::rcsr()
            .args([
                "walkforward", "--data", data.to_str().unwrap(), "--target", "target", "--expr", "x", "--expr", "x - 1",
                "--time", "ts", "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365",
                "--window", "w,0,7200,14400,28800", "--json", "--threshold-fixed", "1.5",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let bad = run(&["--policy", "vote", "--quorum", "0.5", "--device-pnl"]);
    assert_eq!(bad.status.code(), Some(2), "{}", String::from_utf8_lossy(&bad.stderr));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--device-pnl is not supported with --policy vote"));
    let bad = run(&["--policy", "vote", "--quorum", "1.5"]);
    assert_eq!(bad.status.code(), Some(2));
    let bad = run(&["--policy", "vote"]);
    assert_eq!(bad.status.code(), Some(2));
    let bad = run(&["--quorum", "0.5"]);
    assert_eq!(bad.status.code(), Some(2));
    let ok = run(&["--policy", "vote", "--quorum", "0.5"]);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    assert!(String::from_utf8_lossy(&ok.stdout).contains("\"votes_required\":1"));
}

#[test]
fn vote_members_and_thresholds_ignore_test_rows() {
    // Quantile thresholds per member from the validation slice (rows 6..8 of
    // the training range). Scaling and flipping the test block must move
    // neither the members nor the member thresholds nor the vote threshold.
    let build = |scale: f64, flip: f64| {
        let mut rows: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64, i as f64, 0.0)).collect();
        for i in 0..8 {
            rows.push((20.0 + i as f64, VOTE_X[i] * scale, flip * VOTE_Y[i]));
        }
        single(&rows)
    };
    let mut s = spec("x", vec![window("w", 0.0, 10.0, 20.0, 28.0)]);
    s.candidates = vec!["x".to_string(), "2 * x".to_string(), "x + 3".to_string()];
    s.policy = walkforward::Policy::Vote;
    s.validation_frac = 0.4;
    s.threshold_quantile = 0.5;
    let base = walkforward::run_table_with(&build(1.0, 1.0), &s, "t", &vote_opts(0.66)).unwrap();
    let w0 = &base.windows[0];
    assert_eq!(w0.member_thresholds.len(), 3);
    assert!(w0.member_thresholds.iter().all(|t| t.is_finite()));
    assert_eq!(w0.threshold, 1.5);
    for (scale, flip) in [(1000.0, 1.0), (1.0, -1.0), (-50.0, -3.0)] {
        let other = walkforward::run_table_with(&build(scale, flip), &s, "t", &vote_opts(0.66)).unwrap();
        assert_eq!(other.candidates, base.candidates);
        let w = &other.windows[0];
        assert_eq!(w.member_thresholds, w0.member_thresholds, "scale {scale}, flip {flip}");
        assert_eq!(w.threshold.to_bits(), w0.threshold.to_bits());
        assert_eq!(w.expression, w0.expression);
        assert!(!other.test_used_for_selection);
    }
}

// ---------------------------------------------------------------------------
// Projected store scan (issue #1034): the scan reads only the columns the run uses.
// ---------------------------------------------------------------------------

const WIDE_FEATURES: usize = 6;
const WIDE_GARBAGE: usize = 40;
const WIDE_STEPS: usize = 60;
const WIDE_SYMBOLS: usize = 3;

/// Rows in time order, `WIDE_SYMBOLS` per step. Columns: a time (`ts`, whole
/// numbers, or `ts` with a fraction when `fractional_time`), a text key `sym`,
/// a numeric key `gid`, features `f0..`, unused numeric columns `g0..` with NaN
/// and huge values, an unused text column `note`, and the target `y`.
fn wide_table(fractional_time: bool) -> data::Table {
    let mut csv = String::from("ts,sym,gid");
    for j in 0..WIDE_FEATURES {
        csv.push_str(&format!(",f{j}"));
    }
    for j in 0..WIDE_GARBAGE {
        csv.push_str(&format!(",g{j}"));
    }
    csv.push_str(",note,y\n");
    for k in 0..WIDE_STEPS * WIDE_SYMBOLS {
        let (step, sym) = (k / WIDE_SYMBOLS, k % WIDE_SYMBOLS);
        let ts = if fractional_time { format!("{}.25", step) } else { step.to_string() };
        csv.push_str(&format!("{ts},s{sym},{sym}"));
        for j in 0..WIDE_FEATURES {
            let v = ((k * (7 + j) + 3 * j) % 17) as f64 / 8.0 - 1.0;
            if j == 4 && k % 23 == 5 {
                csv.push_str(",NaN");
            } else {
                csv.push_str(&format!(",{v}"));
            }
        }
        for j in 0..WIDE_GARBAGE {
            match (k + j) % 9 {
                0 => csv.push_str(",NaN"),
                1 => csv.push_str(",1e300"),
                _ => csv.push_str(&format!(",{}", (k * j) as f64 - 50.0)),
            }
        }
        let y = (((k * 13) % 11) as f64 - 5.0) * 0.001 + (((k * 7) % 17) as f64 / 8.0 - 1.0) * 0.0004;
        csv.push_str(&format!(",n{},{y}\n", k % 7));
    }
    data::parse_table(&csv).unwrap()
}

fn wide_spec(expr: &str, windows: Vec<Window>, group: Option<&str>) -> EvalSpec {
    let mut s = spec(expr, windows);
    s.vars = (0..WIDE_FEATURES).map(|j| format!("f{j}")).collect();
    s.target = "y".to_string();
    s.group = group.map(str::to_string);
    s.cost_bps = 20.0;
    s.threshold_quantile = 0.5;
    s
}

fn write_wide(scratch: &Scratch, name: &str, table: &data::Table, group_rows: usize) -> String {
    let path = scratch.join(name).to_string_lossy().into_owned();
    store::write_store(&path, table, group_rows).unwrap();
    path
}

/// The report without the members that name the source of the rows.
fn without_source(json: &str) -> rcsr::devlog::Json {
    use rcsr::devlog::Json;
    match Json::parse(json).unwrap() {
        Json::Obj(items) => Json::Obj(items.into_iter().filter(|(k, _)| k != "source" && k != "memory_bound").collect()),
        other => panic!("not an object: {other:?}"),
    }
}

/// Equal structure and strings; numbers equal to rounding, since the resident
/// table path adds the same terms in a different order than the store scan.
fn close(a: &rcsr::devlog::Json, b: &rcsr::devlog::Json, at: &str) {
    use rcsr::devlog::Json;
    match (a, b) {
        (Json::Num(x), Json::Num(y)) => assert!((x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1e-6), "{at}: {x} vs {y}"),
        (Json::Arr(x), Json::Arr(y)) => {
            assert_eq!(x.len(), y.len(), "{at}");
            x.iter().zip(y).enumerate().for_each(|(i, (p, q))| close(p, q, &format!("{at}[{i}]")));
        }
        (Json::Obj(x), Json::Obj(y)) => {
            assert_eq!(x.iter().map(|m| &m.0).collect::<Vec<_>>(), y.iter().map(|m| &m.0).collect::<Vec<_>>(), "{at}");
            x.iter().zip(y).for_each(|(p, q)| close(&p.1, &q.1, &format!("{at}.{}", p.0)));
        }
        _ => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{at}"),
    }
}

/// Projected store scan against the full-decode reference: the resident table
/// the store was written from, which holds every column.
fn assert_matches_full_decode(path: &str, table: &data::Table, spec: &EvalSpec, opts: &walkforward::ProtocolOptions) -> walkforward::Report {
    let streamed = walkforward::run_path_with(path, spec, opts).unwrap();
    let resident = walkforward::run_table_with(table, spec, "test", opts).unwrap();
    close(&without_source(&streamed.json()), &without_source(&resident.json()), "report");
    streamed
}

#[test]
fn store_scan_reads_only_the_named_columns() {
    let scratch = Scratch::new("wf-scan-columns");
    let w = vec![window("w", 0.0, 24.0, 24.0, 40.0)];
    // (fractional time, group column, expected projected columns): 2 named + target
    // + 1 for an f64 time + 1 for a numeric group.
    for (fractional, group, extra) in [(false, None, 0), (false, Some("sym"), 0), (false, Some("gid"), 1), (true, Some("gid"), 2), (true, None, 1)] {
        let table = wide_table(fractional);
        let path = write_wide(&scratch, &format!("wide-{fractional}-{group:?}.rcsb"), &table, 36);
        let s = wide_spec("f1 + f4 * f1", w.clone(), group);
        let r = walkforward::run_path(&path, &s).unwrap();
        let scan = r.scan.clone().expect("the store scan reports its reads");
        assert_eq!(scan.store_columns, 3 + WIDE_FEATURES + WIDE_GARBAGE + 2, "{fractional} {group:?}");
        assert_eq!(scan.named_variables, 2, "{fractional} {group:?}");
        assert_eq!(scan.projected_columns, 2 + 1 + extra, "{fractional} {group:?}");
        // An f64 time is part of the value read, so no group is skipped before it.
        let (read, skipped) = if fractional { (5, 0) } else { (4, 1) };
        assert_eq!((scan.groups_read, scan.groups_skipped), (read, skipped), "{fractional} {group:?}");
        assert_eq!(scan.value_bytes, scan.projected_columns as u64 * read as u64 * 36 * 8, "{fractional} {group:?}");
        assert!(r.json_with_ledger(None).contains(&format!("\"scan\":{{\"source\":\"store\",\"store_columns\":{}", scan.store_columns)));
        assert!(!r.json().contains("\"scan\""), "the hashed report must stay deterministic");
        assert_matches_full_decode(&path, &table, &s, &walkforward::ProtocolOptions::default());
    }
}

#[test]
fn a_constant_candidate_projects_only_the_target() {
    let scratch = Scratch::new("wf-scan-constant");
    let table = wide_table(false);
    let path = write_wide(&scratch, "wide.rcsb", &table, 36);
    let s = wide_spec("0.5", vec![window("w", 0.0, 24.0, 24.0, 40.0)], None);
    let scan = assert_matches_full_decode(&path, &table, &s, &walkforward::ProtocolOptions::default()).scan.unwrap();
    assert_eq!((scan.named_variables, scan.projected_columns), (0, 1));
}

#[test]
fn store_scan_skips_groups_outside_the_windows() {
    let scratch = Scratch::new("wf-scan-skip");
    let table = wide_table(false);
    let path = write_wide(&scratch, "wide.rcsb", &table, 36);
    // 180 rows in 5 groups of 36 rows (12 time steps); the windows end inside group 2.
    let s = wide_spec("f1 + f4", vec![window("w", 0.0, 24.0, 24.0, 30.0)], Some("sym"));
    let r = assert_matches_full_decode(&path, &table, &s, &walkforward::ProtocolOptions::default());
    let scan = r.scan.unwrap();
    assert_eq!((scan.groups_read, scan.groups_skipped), (3, 2));
    assert_eq!(scan.value_bytes, scan.projected_columns as u64 * 3 * 36 * 8);
}

#[test]
fn store_scan_matches_the_full_decode_with_a_final_read_and_a_locked_range() {
    let scratch = Scratch::new("wf-scan-final");
    let table = wide_table(false);
    let path = write_wide(&scratch, "wide.rcsb", &table, 20);
    let mut s = wide_spec("f1 + f4", vec![window("w", 0.0, 24.0, 40.0, 56.0)], Some("gid"));
    s.candidates = vec!["f1 + f4".to_string(), "f2 - f0 * f5".to_string(), "0.25".to_string()];
    s.validation_frac = 0.25;
    for (final_read, locked) in [(false, vec![]), (true, vec![]), (false, vec![(30, 34)]), (true, vec![(30, 34)])] {
        let opts = walkforward::ProtocolOptions { final_read, locked_ranges: locked, locked_embargo: 2, ..walkforward::ProtocolOptions::default() };
        let r = assert_matches_full_decode(&path, &table, &s, &opts);
        let scan = r.scan.unwrap();
        // union of f0, f1, f2, f4, f5 plus the target and the numeric group key
        assert_eq!((scan.named_variables, scan.projected_columns), (5, 7), "{final_read} {:?}", opts.locked_ranges);
        assert_eq!(scan.groups_read + scan.groups_skipped, if final_read { 18 } else { 9 });
    }
}

// ---------------------------------------------------------------------------
// #1149: one group id per numeric key value on the table path and the store path.
// ---------------------------------------------------------------------------

/// The `controls` member of every window, as printed.
fn window_controls(json: &str) -> Vec<String> {
    use rcsr::devlog::Json;
    let report = Json::parse(json).unwrap();
    match report.get("windows") {
        Some(Json::Arr(ws)) => ws.iter().map(|w| format!("{:?}", w.get("controls").expect("controls"))).collect(),
        other => panic!("no windows: {other:?}"),
    }
}

fn assert_csv_and_store_agree(table: &data::Table, scratch: &Scratch, name: &str, group: &str) {
    let path = scratch.join(name).to_string_lossy().into_owned();
    store::write_store(&path, table, 36).unwrap();
    let mut s = spec("x", vec![window("w", 0.0, 6.0, 6.0, 12.0)]);
    s.target = "y".to_string();
    s.group = Some(group.to_string());
    s.control_seed = 7;
    s.cost_bps = 20.0;
    s.threshold_quantile = 0.5;
    let opts = walkforward::ProtocolOptions::default();
    let resident = walkforward::run_table_with(table, &s, "test", &opts).unwrap();
    let streamed = walkforward::run_path_with(&path, &s, &opts).unwrap();
    assert_eq!(window_controls(&resident.json()), window_controls(&streamed.json()), "controls");
    close(&without_source(&streamed.json()), &without_source(&resident.json()), "report");
}

/// `keys` distinct numeric keys, 12 steps, one row per key and step. Each key
/// alternates between two spellings of the same number (`1` and `1.0`, `0` and
/// `-0`); the last key is NaN.
fn keyed_table(keys: usize) -> data::Table {
    let mut vals: Vec<f64> = (0..keys - 2).map(|i| (i as f64 - keys as f64 / 3.0) * 0.25).filter(|v| *v != 0.0).collect();
    if !vals.contains(&1.0) {
        vals.push(1.0);
    }
    let mut csv = String::from("ts,gid,x,y\n");
    for step in 0..12usize {
        let rows = vals.iter().map(|v| if step % 2 == 0 { format!("{v}") } else { format!("{v:?}") });
        let zero = if step % 2 == 0 { "0".to_string() } else { "-0".to_string() };
        for (k, key) in rows.chain([zero, "NaN".to_string()]).enumerate() {
            let x = ((k * 7 + step * 3) % 13) as f64 / 6.0 - 1.0;
            let y = (((k * 5 + step * 11) % 17) as f64 - 8.0) * 0.001;
            csv.push_str(&format!("{step},{key},{x},{y}\n"));
        }
    }
    data::parse_table(&csv).unwrap()
}

#[test]
fn numeric_group_ids_match_between_csv_and_store() {
    let scratch = Scratch::new("wf-numeric-group");
    let table = wide_table(false);
    let path = write_wide(&scratch, "wide.rcsb", &table, 36);
    let mut s = wide_spec("f1 + f4 * f1", vec![window("w", 0.0, 24.0, 24.0, 40.0)], Some("gid"));
    s.control_seed = 7;
    let opts = walkforward::ProtocolOptions::default();
    let resident = walkforward::run_table_with(&table, &s, "test", &opts).unwrap();
    let streamed = walkforward::run_path_with(&path, &s, &opts).unwrap();
    assert_eq!(window_controls(&resident.json()), window_controls(&streamed.json()));
    close(&without_source(&streamed.json()), &without_source(&resident.json()), "report");
}

#[test]
fn numeric_group_keys_agree_across_spellings_negative_zero_nan_and_cardinality() {
    let scratch = Scratch::new("wf-numeric-group-keys");
    // 6 keys keep a dictionary on the table path; 4200 keys drop it.
    for keys in [6, 4200] {
        assert_csv_and_store_agree(&keyed_table(keys), &scratch, &format!("keys-{keys}.rcsb"), "gid");
    }
}

#[test]
fn the_cli_prints_the_same_controls_for_a_numeric_group_as_csv_and_store() {
    let scratch = Scratch::new("wf-numeric-group-cli");
    let csv_path = scratch.join("d.csv");
    let store_path = scratch.join("d.rcsb");
    let mut csv = String::from("ts,gid,x,target\n");
    for step in 0..12 {
        for (k, gid) in ["-1", "0.5", "2", "7"].iter().enumerate() {
            let x = ((k * 7 + step * 3) % 13) as f64 / 6.0 - 1.0;
            let y = (((k * 5 + step * 11) % 17) as f64 - 8.0) * 0.001;
            csv.push_str(&format!("{},{gid},{x},{y}\n", 3600 * step));
        }
    }
    std::fs::write(&csv_path, csv).unwrap();
    let import = rcsr_cmd::rcsr().args(["import", csv_path.to_str().unwrap(), store_path.to_str().unwrap(), "--time", "ts"]).output().unwrap();
    assert_eq!(import.status.code(), Some(0), "{}", String::from_utf8_lossy(&import.stderr));
    let run = |data: &std::path::Path| {
        let out = rcsr_cmd::rcsr()
            .args([
                "walkforward", "--data", data.to_str().unwrap(), "--target", "target", "--expr", "x", "--time", "ts",
                "--group-by", "gid", "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365",
                "--window", "w,0,18000,25200,43200", "--control-seed", "7", "--json",
            ])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        window_controls(&String::from_utf8_lossy(&out.stdout))
    };
    assert_eq!(run(&csv_path), run(&store_path));
}
