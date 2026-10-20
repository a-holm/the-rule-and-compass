//! Differential leakage-detector suite (issue #129).
//!
//! Every test here uses a synthetic fixture with a known answer: a **clean**
//! feature pipeline must pass and a deliberately **leaked** one must fail
//! loudly, naming the offending column. `Report::assert_clean` panics on a
//! leak, so a real leak cannot be reduced to a log line.
//!
//! To run the same suite on a user panel, see `docs/leakage.md`.

use rcsr::leakage::{
    as_of_join, audit_panel, canary_leakage, cut_points, effective_n, future_shift, synthetic_panel,
    truncation_replay, wall_time_gap, JoinRule, Panel,
};

/// An honest feature pipeline: each output row is a function of rows at or
/// before it (a lag and an expanding mean). Truncating the panel must not
/// change any visible row.
fn honest(panel: &Panel) -> Panel {
    let x = panel.column("x").expect("fixture has x");
    let n = panel.len();
    let mut lag = vec![f64::NAN; n];
    let mut mean = vec![f64::NAN; n];
    let mut acc = 0.0;
    for i in 0..n {
        if i > 0 {
            lag[i] = x[i - 1];
        }
        acc += x[i];
        mean[i] = acc / (i + 1) as f64;
    }
    panel
        .with_column("x_lag1", lag)
        .unwrap()
        .with_column("x_mean_so_far", mean)
        .unwrap()
}

/// A leaky pipeline: standardise the feature by the mean of *all* rows,
/// including the future. This is the classic whole-panel normalisation bug.
fn leaky_global_mean(panel: &Panel) -> Panel {
    let x = panel.column("x").expect("fixture has x");
    let mean = x.iter().sum::<f64>() / panel.len() as f64;
    let scaled: Vec<f64> = x.iter().map(|v| v - mean).collect();
    panel.with_column("x_centred", scaled).unwrap()
}

#[test]
fn truncation_at_twenty_cuts_is_byte_identical_for_a_clean_pipeline() {
    let panel = synthetic_panel(400, 11, false);
    let cuts = cut_points(&panel, 20);
    assert_eq!(cuts.len(), 20, "the fixture must exercise 20 cut points");
    let report = truncation_replay(&panel, &honest, &cuts);
    assert_eq!(report.checks, 20);
    report.assert_clean();
}

#[test]
fn truncation_replay_fails_loudly_on_a_global_normaliser_and_names_the_column() {
    let panel = synthetic_panel(200, 12, false);
    let report = truncation_replay(&panel, &leaky_global_mean, &cut_points(&panel, 10));
    assert!(!report.is_clean(), "a whole-panel mean must be caught");
    let failure = report.result().expect_err("must be a failure");
    assert!(
        failure
            .to_string()
            .contains("D4 truncation-replay") && failure.to_string().contains("x_centred"),
        "unhelpful failure: {}",
        failure
    );
}

#[test]
fn a_deliberate_future_column_fails_and_is_named() {
    let panel = synthetic_panel(200, 13, true);
    let report = future_shift(&panel, "y", 1);
    assert!(!report.is_clean());
    assert!(
        report
            .leaks
            .iter()
            .any(|l| l.column.as_deref() == Some("future_y")
                && l.detector == "D5 future-shift"),
        "leaks: {:?}",
        report.leaks
    );
}

#[test]
fn canary_names_are_flagged_but_iid_noise_is_not() {
    let clean = synthetic_panel(200, 14, false);
    canary_leakage(&clean, "y", 1).assert_clean();

    let leaky = synthetic_panel(200, 14, true);
    let report = canary_leakage(&leaky, "y", 1);
    assert!(
        report
            .leaks
            .iter()
            .any(|l| l.column.as_deref() == Some("future_y")),
        "the future column must be named: {:?}",
        report.leaks
    );

    // i.i.d. noise next to the target is a canary that must *not* fire.
    let noisy = clean
        .with_column("iid_noise_canary", clean.column("noise").unwrap().to_vec())
        .unwrap();
    let report = future_shift(&noisy, "y", 1);
    assert!(
        report
            .leaks
            .iter()
            .all(|l| l.column.as_deref() != Some("iid_noise_canary")),
        "i.i.d. noise was mistaken for a leak: {:?}",
        report.leaks
    );
}

#[test]
fn synthetic_fixtures_test_the_tests() {
    // Clean fixture: every detector passes.
    let clean = synthetic_panel(300, 21, false);
    let report = audit_panel(&clean, "y", 1);
    report.assert_clean();

    // Leaky fixture: the suite must fail, not merely warn.
    let leaky = synthetic_panel(300, 21, true);
    let report = audit_panel(&leaky, "y", 1);
    assert!(!report.is_clean(), "the planted leak was not detected");
    let text = report.result().unwrap_err().to_string();
    assert!(text.contains("future_y"), "leak not named: {}", text);
}

#[test]
fn an_open_time_higher_timeframe_join_is_rejected_and_a_close_time_join_passes() {
    let bar_open = [0.0, 10.0];
    let probe = [11.0, 15.0, 20.0, 25.0];

    let err = as_of_join(&probe, &bar_open, 10.0, JoinRule::Timestamp)
        .expect_err("ts <= t on a higher timeframe must be refused");
    assert_eq!(err.detector, "D3 as-of");
    assert!(err.message.contains("bar_close"));

    let asof = as_of_join(&probe, &bar_open, 10.0, JoinRule::BarClose).unwrap();
    // close times are 10 and 20: probe 11/15 use bar 0, 20/25 use bar 1.
    assert_eq!(asof, vec![Some(0), Some(0), Some(1), Some(1)]);
}

#[test]
fn wall_time_gap_is_converted_to_rows_and_overlap_reduces_n_eff() {
    // 90 minutes at a 5-minute median spacing -> 18 rows.
    assert_eq!(wall_time_gap(90.0 * 60.0, 5.0 * 60.0), Some(18));
    // A 90-minute gap at 7-minute spacing rounds up, never down.
    assert_eq!(wall_time_gap(90.0 * 60.0, 7.0 * 60.0), Some(13));
    assert_eq!(wall_time_gap(0.0, 300.0), None);
    assert_eq!(effective_n(1000, 24), 1000.0 / 24.0);
    assert_eq!(effective_n(1000, 0), 1000.0);
}
