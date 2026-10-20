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

// ---------------------------------------------------------------------------
// Tier-A auditors and the D5 differential learner (#129).
// ---------------------------------------------------------------------------

use rcsr::data::TimeUnit;
use rcsr::leakage::{
    adjust_corporate_actions, audit_determinism, audit_join_cardinality, audit_spec,
    audit_universe, corporate_action_replay, differential_learner, differential_suite,
    reconcile_resample, resample_bars, spec_hash, time_normalizer, CorporateAction,
    CorporateActionKind, Ohlcv, UniverseChange, UniverseEvent, UniverseRow,
};

#[test]
fn d1_time_normaliser_detects_units_duplicates_and_backwards_rows() {
    // 0, 1000, 2000 ms is monotone and 1 s apart after normalisation.
    let (secs, audit, report) = time_normalizer(&[0.0, 1000.0, 2000.0], TimeUnit::Millis);
    assert_eq!(secs, vec![0.0, 1.0, 2.0]);
    assert_eq!(audit.median_dt, 1.0);
    report.assert_clean();

    // A duplicate instant is a numeric duplicate, not a tolerated tie.
    let (_s, audit, report) = time_normalizer(&[0.0, 1.0, 1.0], TimeUnit::Seconds);
    assert_eq!(audit.duplicate_instants, 1);
    assert!(!report.is_clean());

    // Going backwards names the row.
    let (_s, audit, report) = time_normalizer(&[0.0, 2.0, 1.0], TimeUnit::Seconds);
    assert_eq!(audit.non_monotonic_rows, 1);
    assert!(report.leaks.iter().any(|l| l.row == Some(2)));
}

fn minute_bars(n: usize) -> Vec<Ohlcv> {
    (0..n)
        .map(|i| {
            let p = 100.0 + i as f64;
            Ohlcv {
                time: i as f64 * 60.0,
                open: p,
                high: p + 2.0,
                low: p - 1.0,
                close: p + 1.0,
                volume: 10.0 + i as f64,
                vwap: p + 0.5,
            }
        })
        .collect()
}

#[test]
fn d2_one_minute_to_one_hour_equals_the_fifteen_minute_cascade() {
    let bars = minute_bars(240);
    reconcile_resample(&bars, 15, 60, None).assert_clean();
    let direct = resample_bars(&bars, 60, None);
    let staged = resample_bars(&resample_bars(&bars, 15, None), 4, None);
    assert_eq!(direct, staged);
    // OHLC and volume conventions, checked on the first hour.
    assert_eq!(direct[0].open, bars[0].open);
    assert_eq!(direct[0].close, bars[59].close);
    assert_eq!(direct[0].high, bars.iter().take(60).map(|b| b.high).fold(f64::NEG_INFINITY, f64::max));
    assert_eq!(direct[0].volume, bars.iter().take(60).map(|b| b.volume).sum::<f64>());
}

#[test]
fn d8_point_in_time_universe_rejects_rows_before_listing_and_after_delisting() {
    let events = vec![
        UniverseEvent { time: 10.0, symbol: "AAA".into(), change: UniverseChange::Listed, to: None },
        UniverseEvent { time: 20.0, symbol: "AAA".into(), change: UniverseChange::Delisted, to: None },
    ];
    let rows = vec![
        UniverseRow { time: 9.0, symbol: "AAA".into() },   // before listing
        UniverseRow { time: 15.0, symbol: "AAA".into() },  // live
        UniverseRow { time: 21.0, symbol: "AAA".into() },  // after delisting
        UniverseRow { time: 15.0, symbol: "BBB".into() },  // no event
    ];
    let report = audit_universe(&rows, &events);
    let flagged: Vec<usize> = report.leaks.iter().filter_map(|l| l.row).collect();
    assert_eq!(flagged, vec![0, 2, 3]);
    assert!(report.leaks.iter().all(|l| l.detector == "D8 universe"));
}

#[test]
fn d9_a_split_is_known_only_from_its_instant_and_both_series_are_returned() {
    let times = vec![0.0, 1.0, 2.0, 3.0];
    let prices = vec![100.0, 100.0, 100.0, 100.0];
    let actions = vec![CorporateAction {
        time: 2.0,
        symbol: "AAA".into(),
        kind: CorporateActionKind::Split(2.0),
    }];
    let s = adjust_corporate_actions(&times, &prices, &actions, "AAA");
    // Rows 0 and 1 predate the action and must never see its factor.
    assert_eq!(&s.adjusted[..2], &[100.0, 100.0]);
    assert_eq!(&s.adjusted[2..], &[50.0, 50.0]);
    assert_eq!(s.unadjusted, prices);
    // A replay at every cut is stable: no future factor leaks backwards.
    corporate_action_replay(&times, &prices, &actions, "AAA", &[1.0, 2.0, 3.0])
        .assert_clean();
}

#[test]
fn d10_duplicate_right_keys_are_rejected_as_row_multiplication() {
    let left = [1u64, 2, 3];
    let right = [1u64, 2, 2];
    let (matched, report) = audit_join_cardinality(&left, &right);
    assert_eq!(matched, vec![Some(0), Some(2), None]);
    assert!(!report.is_clean());
    assert!(report.leaks.iter().any(|l| l.detector == "D10 join" && l.row == Some(2)));
    let (_, clean) = audit_join_cardinality(&left, &[1u64, 2, 3]);
    clean.assert_clean();
}

#[test]
fn d0_and_d11_resolve_the_spec_and_hash_it_deterministically() {
    let spec = [("target", "y"), ("tzdb_version", "2025a"), ("rows", "100")];
    audit_spec(&spec, &[("target", "y")], &[("horizon", "1")]).assert_clean();
    let missing = [("target", "y")];
    assert!(!audit_spec(&missing, &[("tzdb_version", "2025a")], &[]).is_clean());
    assert_eq!(spec_hash(&spec), spec_hash(&spec));
    audit_determinism(&spec, &spec).assert_clean();
    let other = [("target", "y"), ("tzdb_version", "2024a"), ("rows", "100")];
    assert!(!audit_determinism(&spec, &other).is_clean());
}

#[test]
fn d5_differential_learner_flags_the_future_column_not_iid_noise() {
    let leaky = synthetic_panel(400, 31, true);
    let probe = differential_learner(&leaky, "y", "future_y", 1);
    assert!(probe.flagged, "{probe:?}");
    let report = differential_suite(&leaky, "y", 1);
    assert!(report.leaks.iter().any(|l| l.column.as_deref() == Some("future_y")));

    let clean = synthetic_panel(400, 31, false);
    let report = differential_suite(&clean, "y", 1);
    assert!(report.leaks.is_empty(), "{:?}", report.leaks);
}
