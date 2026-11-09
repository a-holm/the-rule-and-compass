//! Long-only horizon, action and triple-barrier labels on small synthetic bars.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::data::{self, SplitMode, SplitSpec};
use rcsr::features::LabelSpec;
use rcsr::labels::{
    self, build, funding_charge, meta_labels, LongLabelSpec, EXIT_PROFIT, EXIT_STOP, EXIT_VERTICAL,
};
use rcsr::ohlcv::{Bar, Timeframe};
use rcsr::rng::Mt;
use rcsr::target;

const T0: i64 = 1_600_000_000 - 1_600_000_000 % 3600;
const HOUR: i64 = 3600;

fn bar(i: usize, o: f64, h: f64, l: f64, c: f64) -> Bar {
    Bar::new(T0 + i as i64 * HOUR, o, h, l, c, 1.0)
}

fn flat(i: usize, p: f64) -> Bar {
    bar(i, p, p, p, p)
}

/// Distinct open and close at every bar, so a wrong index changes the value.
fn wavy(n: usize) -> Vec<Bar> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            let o = 100.0 + 3.0 * (x * 0.37).sin() + 0.05 * x;
            let c = 100.0 + 3.0 * (x * 0.37 + 0.4).sin() + 0.05 * x;
            bar(i, o, o.max(c) + 0.3, o.min(c) - 0.3, c)
        })
        .collect()
}

fn spec(name: &str, h: usize) -> LongLabelSpec {
    let mut s = LongLabelSpec::new(name, HOUR, h);
    s.sigma_window = 5;
    s
}

fn population_sd(r: &[f64]) -> f64 {
    let m = r.iter().sum::<f64>() / r.len() as f64;
    (r.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / r.len() as f64).sqrt()
}

#[test]
fn horizon_windows_enter_next_open_and_exit_at_close_exactly() {
    let bars = wavy(200);
    for (name, h) in [("y_60m", 1usize), ("y_4h", 4), ("y_24h", 24), ("y_60h", 60)] {
        let s = spec(name, h);
        let l = build(&bars, &s, None).unwrap();
        for t in 0..bars.len() {
            if t + h < bars.len() {
                let want = (bars[t + h].close / bars[t + 1].open).ln();
                assert_eq!(l.fwd[t].to_bits(), want.to_bits(), "{name} t={t}");
            } else {
                assert!(l.fwd[t].is_nan() && l.y[t].is_nan(), "{name} t={t} reads past t+h");
            }
        }
        // y = clip(fwd / (max(s_W, 5bp) * sqrt(h)), -3, 3), s_W the population
        // sd of the last 5 one-bar log returns.
        let t = 100;
        let r: Vec<f64> = (t - 4..=t).map(|k| (bars[k].close / bars[k - 1].close).ln()).collect();
        let sigma_h = population_sd(&r).max(5e-4) * (h as f64).sqrt();
        let want_y = (l.fwd[t] / sigma_h).max(-3.0).min(3.0);
        assert_eq!(l.y[t].to_bits(), want_y.to_bits(), "{name}");
        assert_eq!(l.sigma_h[t].to_bits(), sigma_h.to_bits(), "{name}");
    }
}

#[test]
fn no_value_after_the_horizon_is_read() {
    let bars = wavy(80);
    let s = spec("y_4h", 4);
    let base = build(&bars, &s, None).unwrap();
    let mut changed = bars.clone();
    for (k, b) in changed.iter_mut().enumerate().skip(41) {
        b.open *= 3.0 + k as f64;
        b.high *= 5.0;
        b.low *= 0.1;
        b.close *= 7.0;
    }
    let l = build(&changed, &s, None).unwrap();
    // Row 36 ends at bar 40; bars from 41 on do not enter it.
    for t in 0..=36 {
        assert_eq!(l.fwd[t].to_bits(), base.fwd[t].to_bits(), "t={t}");
        assert_eq!(l.y[t].to_bits(), base.y[t].to_bits(), "t={t}");
        assert_eq!(l.exit_ret[t].to_bits(), base.exit_ret[t].to_bits(), "t={t}");
    }
}

#[test]
fn action_is_zero_or_plus_one_and_follows_the_cost() {
    let bars = wavy(200);
    let mut s = spec("y_4h", 4);
    s.cost.fee_bps = 5.0;
    s.cost.spread_bps = 4.0;
    s.cost.slippage_bps = 6.0;
    let c = s.cost.round_trip();
    assert_eq!(s.cost.round_trip_bps(), 20.0);
    let l = build(&bars, &s, None).unwrap();
    let (mut ones, mut zeros, mut negative) = (0, 0, 0);
    for t in 0..bars.len() {
        if !l.fwd[t].is_finite() {
            assert!(l.action[t].is_nan());
            continue;
        }
        assert!(l.action[t] == 0.0 || l.action[t] == 1.0, "t={t}: {}", l.action[t]);
        assert_eq!(l.action[t] == 1.0, l.fwd[t] > c, "t={t}");
        assert_eq!((l.long_net[t] - (l.fwd[t] - c)).abs(), 0.0);
        if l.fwd[t] < 0.0 {
            negative += 1;
            assert_eq!(l.action[t], 0.0, "a negative forward return maps to 0");
        }
        if l.action[t] == 1.0 {
            ones += 1
        } else {
            zeros += 1
        }
    }
    assert!(ones > 0 && zeros > 0 && negative > 0, "{ones} {zeros} {negative}");
}

/// Ten flat bars, then `tail`. The zero history volatility makes `sigma_H` the
/// floor: 100 bp * sqrt(3) with a 3-bar horizon.
fn barrier_series(tail: &[(f64, f64, f64, f64)]) -> Vec<Bar> {
    let mut v: Vec<Bar> = (0..10).map(|i| flat(i, 100.0)).collect();
    for (k, &(o, h, l, c)) in tail.iter().enumerate() {
        v.push(bar(10 + k, o, h, l, c));
    }
    v
}

fn barrier_spec() -> LongLabelSpec {
    let mut s = spec("y_3h", 3);
    s.sigma_floor_bps = 100.0;
    s.cost.fee_bps = 5.0;
    s
}

const ROW: usize = 9;

#[test]
fn barrier_hits_profit_stop_and_vertical_in_the_expected_order() {
    let s = barrier_spec();
    let sigma_h = 0.01 * 3f64.sqrt();
    let profit = 100.0 * sigma_h.exp();
    let stop = 100.0 * (-sigma_h).exp();
    let quiet = (100.0, 100.5, 99.5, 100.0);

    // Profit in the second bar of the window, stop never touched.
    let a = barrier_series(&[quiet, (100.0, profit + 0.5, 99.9, 101.0), quiet, quiet]);
    let la = build(&a, &s, None).unwrap();
    assert_eq!(la.exit_kind[ROW], EXIT_PROFIT);
    assert_eq!(la.m_barrier[ROW], 1.0);
    assert_eq!(la.m_net[ROW], 1.0);
    assert!((la.exit_ret[ROW] - sigma_h).abs() < 1e-15);
    assert_eq!(la.t1[ROW], (a[11].time + HOUR) as f64);

    // Stop in the third bar after a quiet start.
    let b = barrier_series(&[quiet, quiet, (100.0, 100.5, stop - 0.5, 99.0), quiet]);
    let lb = build(&b, &s, None).unwrap();
    assert_eq!(lb.exit_kind[ROW], EXIT_STOP);
    assert_eq!(lb.m_barrier[ROW], 0.0);
    assert_eq!(lb.m_net[ROW], 0.0);
    assert!((lb.exit_ret[ROW] + sigma_h).abs() < 1e-15);
    assert_eq!(lb.t1[ROW], (b[12].time + HOUR) as f64);

    // Neither barrier: exit at the vertical close.
    let c = barrier_series(&[quiet, quiet, (100.0, 100.6, 99.6, 100.05), quiet]);
    let lc = build(&c, &s, None).unwrap();
    assert_eq!(lc.exit_kind[ROW], EXIT_VERTICAL);
    assert_eq!(lc.m_barrier[ROW], 0.0);
    assert!((lc.exit_ret[ROW] - (100.05f64 / 100.0).ln()).abs() < 1e-15);
    assert_eq!(lc.exit_ret[ROW].to_bits(), lc.fwd[ROW].to_bits());
    // The gross return is below the round trip cost.
    assert_eq!(lc.m_net[ROW], 0.0);
    assert_eq!(lc.action[ROW], 0.0);

    for l in [&la, &lb, &lc] {
        let bound = (T0 + ROW as i64 * HOUR) + (1 + 3) * HOUR;
        assert!(l.t1[ROW] <= bound as f64, "t1 {} after t+h {}", l.t1[ROW], bound);
    }
    assert_eq!(lc.t1[ROW], (T0 + ROW as i64 * HOUR + 4 * HOUR) as f64);
}

#[test]
fn both_barriers_inside_one_bar_assume_the_stop() {
    let s = barrier_spec();
    let quiet = (100.0, 100.5, 99.5, 100.0);
    let wide = (100.0, 105.0, 95.0, 104.0);
    let bars = barrier_series(&[wide, quiet, quiet, quiet]);
    let l = build(&bars, &s, None).unwrap();
    assert_eq!(l.exit_kind[ROW], EXIT_STOP);
    assert_eq!(l.m_barrier[ROW], 0.0);
    assert!(l.exit_ret[ROW] < 0.0);
    assert_eq!(l.t1[ROW], (bars[10].time + HOUR) as f64);
}

#[test]
fn meta_label_marks_primary_long_rows_only() {
    let s = barrier_spec();
    let profit = 100.0 * (0.01 * 3f64.sqrt()).exp();
    let quiet = (100.0, 100.5, 99.5, 100.0);
    let bars = barrier_series(&[(100.0, profit + 1.0, 99.9, 101.0), quiet, quiet, quiet]);
    let l = build(&bars, &s, None).unwrap();
    let mut primary = vec![0.0; bars.len()];
    let m = meta_labels(&primary, &l).unwrap();
    assert!(m.iter().all(|v| v.is_nan()));
    primary[ROW] = 1.0;
    let m = meta_labels(&primary, &l).unwrap();
    assert_eq!(m[ROW], 1.0);
    assert_eq!(m.iter().filter(|v| v.is_finite()).count(), 1);
}

#[test]
fn short_is_refused_everywhere() {
    assert!(labels::check_side("long").is_ok());
    let e = labels::check_side("short").unwrap_err();
    assert!(e.contains("long-only"), "{e}");
    assert!(labels::refuse_allow_short().starts_with("--allow-short is refused"));
    let e = LabelSpec::parse("fwd:1h,side=short", Timeframe::H1).unwrap_err();
    assert!(e.contains("long-only"), "{e}");
    assert!(LabelSpec::parse("fwd:1h,side=long", Timeframe::H1).is_ok());

    for flag in [["enrich", "--allow-short"], ["prep", "--allow-short"]] {
        let out = rcsr_cmd::rcsr().args(flag).output().unwrap();
        assert!(!out.status.success(), "{flag:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("long-only"), "{flag:?}");
    }
    let out = rcsr_cmd::rcsr()
        .args(["enrich", "--side", "short"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("long-only"));
}

#[test]
fn long_perp_funding_is_charged_only_on_a_long_position() {
    assert_eq!(funding_charge(1, 0.0003), 0.0003);
    assert_eq!(funding_charge(0, 0.0003), 0.0);
    assert_eq!(funding_charge(-1, 0.0003), 0.0);

    // A 2 bp rise per bar over 4 bars is 8 bp gross; 3 bp funding per held bar
    // makes the long trade lose.
    let bars: Vec<Bar> = (0..12)
        .map(|i| {
            let p = 100.0 * (1.0002f64).powi(i as i32);
            bar(i, p, p * 1.00001, p * 0.99999, p * 1.0002)
        })
        .collect();
    let s = spec("y_4h", 4);
    let plain = build(&bars, &s, None).unwrap();
    let funding = vec![0.0003; bars.len()];
    let charged = build(&bars, &s, Some(&funding)).unwrap();
    assert_eq!(plain.action[2], 1.0);
    assert!((charged.long_net[2] - (plain.long_net[2] - 4.0 * 0.0003)).abs() < 1e-15);
    assert_eq!(charged.action[2], 0.0);
    let mut nan_funding = funding.clone();
    nan_funding[4] = f64::NAN;
    let unverified = build(&bars, &s, Some(&nan_funding)).unwrap();
    assert!(unverified.fwd[2].is_nan(), "a span with unverified funding is missing");
    assert!(build(&bars, &s, Some(&funding[..5])).is_err());
}

#[test]
fn missing_partial_and_gap_crossing_windows_are_nan() {
    let mut bars: Vec<Bar> = (0..30).map(|i| flat(i, 100.0 + i as f64)).collect();
    // A two-bar hole after index 14.
    for b in bars.iter_mut().skip(15) {
        b.time += 2 * HOUR;
    }
    bars[22].complete = false;
    bars[26].close = f64::NAN;
    let mut s = spec("y_3h", 3);
    s.sigma_window = 2;
    let l = build(&bars, &s, None).unwrap();
    let mut table = Vec::new();
    for t in 0..bars.len() {
        let valid = l.fwd[t].is_finite();
        table.push((t, valid));
    }
    let expect_nan = |t: usize| -> bool {
        // The window [t, t+3] spans the hole or the partial bar, the entry or exit bars hold a bad price, or it runs past the end.
        let (a, b) = (t, t + 3);
        b >= 30 || (a <= 14 && b >= 15) || (a..=b).contains(&22) || (a + 1..=b).contains(&26)
    };
    for (t, valid) in table {
        assert_eq!(valid, !expect_nan(t), "t={t}");
    }
    assert!(l.fwd[11].is_finite() && l.fwd[12].is_nan());
    assert!(l.action[12].is_nan());
}

#[test]
fn purge_is_derived_from_the_spec_not_from_a_name() {
    // (bar seconds, horizon bars, delay bars, expected purge seconds)
    let table = [
        (900, 4, 0, 4500),
        (3600, 1, 0, 7200),
        (3600, 4, 0, 18_000),
        (3600, 24, 0, 90_000),
        (3600, 60, 0, 219_600),
        (3600, 4, 2, 25_200),
    ];
    for (bar_s, h, d, want) in table {
        let mut s = LongLabelSpec::new("y60", bar_s, h);
        s.delay_bars = d;
        assert_eq!(s.purge_seconds(), want, "bar {bar_s} h {h} d {d}");
        let t = s.target_spec("y60", "fwd");
        assert_eq!(t.required_purge_seconds(), want as f64);
        assert_eq!(t.entry_offset_seconds + t.delay_seconds + t.horizon_seconds, want as f64);
    }
    // A 60 minute label named after 60 hours keeps its own purge.
    let sixty_minutes = LongLabelSpec::new("y60", 900, 4);
    assert_ne!(sixty_minutes.purge_seconds(), 60 * 3600);
    assert_eq!(sixty_minutes.entry_offset_seconds(), 900);
    assert_eq!(sixty_minutes.horizon_seconds(), 3600);
}

#[test]
fn t1_and_the_derived_spec_drive_the_split_purge() {
    let n = 120;
    let bars = wavy(n);
    let s = spec("y_4h", 4);
    let l = build(&bars, &s, None).unwrap();
    let ts = s.target_spec("a_4h", "fwd_4h");
    let mut csv = String::from("ts,x,a_4h\n");
    for t in 0..n {
        csv.push_str(&format!("{},{},{}\n", bars[t].time, t, t % 2));
    }
    let table = data::parse_table(&csv).unwrap();
    let time_col = table.index_of("ts").unwrap();
    let spec = SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(time_col),
        test_frac: 0.25,
        purge_seconds: Some(ts.required_purge_seconds()),
        ..SplitSpec::default()
    };
    let mut rng = Mt::new(1);
    let p = data::prepare_full_targets_weighted_with_indices(
        &table,
        &["x".to_string()],
        &["a_4h"],
        None,
        &spec,
        &mut rng,
        &data::PrepSpec::default(),
    )
    .unwrap();
    assert_eq!(p.report.required_purge_seconds, Some(s.purge_seconds() as f64));
    assert!(p.report.purged_test_boundary_rows >= 1);
    let first_test = 90;
    assert_eq!(p.report.test_rows, 30);
    let first_test_stamp = bars[first_test].time as f64;
    assert!(!p.train_indices.is_empty());
    for &i in &p.train_indices {
        // The label of every kept training row ends before the test region.
        let t1 = l.t1[i];
        assert!(t1 <= first_test_stamp, "row {i}: t1 {t1} after the first test stamp {first_test_stamp}");
    }
    // Without the purge the last training labels reach into the test region.
    assert!(bars[first_test - 1].time as f64 + s.purge_seconds() as f64 > first_test_stamp);
}

#[test]
fn overlap_is_guarded_and_uniqueness_weights_are_stored() {
    let mut s = LongLabelSpec::new("y_4h", HOUR, 4);
    assert!(s.check_overlap(false).unwrap_err().contains("--allow-overlap"));
    assert!(s.check_overlap(true).is_ok());
    s.stride = 4;
    assert!(s.check_overlap(false).is_ok());
    let hourly = LongLabelSpec::new("y_60m", HOUR, 1);
    assert!(hourly.check_overlap(false).is_ok());
    let two = LongLabelSpec::new("y_2h", HOUR, 2);
    assert!(two.check_overlap(false).is_ok(), "below 4h the weights cover the overlap");

    let l4 = LabelSpec::parse("fwd:4h,cost=0", Timeframe::H1).unwrap();
    assert!(l4.check_overlap(Timeframe::H1, false).unwrap_err().contains("--allow-overlap"));
    assert!(l4.check_overlap(Timeframe::H1, true).is_ok());
    let l4s = LabelSpec::parse("fwd:4h,cost=0,stride=4", Timeframe::H1).unwrap();
    assert!(l4s.check_overlap(Timeframe::H1, false).is_ok());
    let l1 = LabelSpec::parse("fwd:1h", Timeframe::H1).unwrap();
    assert!(l1.check_overlap(Timeframe::H1, false).is_ok());

    // Ten flat bars, 2-bar horizon, stride 1: rows 0..=7 span bars [t+1, t+2].
    let bars: Vec<Bar> = (0..10).map(|i| flat(i, 100.0)).collect();
    let l = build(&bars, &spec("y_2h", 2), None).unwrap();
    let r = l.report();
    assert_eq!(r.valid_rows, 8);
    assert!((r.effective_n - 4.5).abs() < 1e-12, "{}", r.effective_n);
    assert!((r.effective_n_hours - 9.0).abs() < 1e-12, "{}", r.effective_n_hours);
    assert!((l.uniqueness[0] - 0.75).abs() < 1e-12 && (l.uniqueness[3] - 0.5).abs() < 1e-12);
    let w_sum: f64 = l.weight.iter().filter(|v| v.is_finite()).sum();
    assert!((w_sum - 8.0).abs() < 1e-12);
    assert!((l.weight[0] - 0.75 * 8.0 / 4.5).abs() < 1e-12);

    let mut strided = spec("y_2h", 2);
    strided.stride = 2;
    let l = build(&bars, &strided, None).unwrap();
    let r = l.report();
    assert_eq!(r.valid_rows, 4);
    assert!((r.effective_n - 4.0).abs() < 1e-12);
    assert!((r.effective_n_hours - 8.0).abs() < 1e-12);
    assert!(l.weight.iter().filter(|v| v.is_finite()).all(|w| (w - 1.0).abs() < 1e-12));
}

#[test]
fn target_spec_links_to_the_construction_column_canary() {
    let s = spec("y_4h", 4);
    let ts = s.target_spec("y_4h", "fwd_4h");
    assert!(ts.long_only);
    assert!(!ts.structural_term());
    assert_eq!(ts.construction_columns, ["fwd_4h", "vol5"]);
    let clean = vec!["ret1".to_string(), "rsi14".to_string()];
    assert!(target::check(&ts, &clean, false).is_ok());
    for bad in ["fwd_4h", "y_4h"] {
        let vars = vec!["ret1".to_string(), bad.to_string()];
        let report = target::check(&ts, &vars, false).unwrap_err();
        assert!(report.leaks.iter().any(|l| l.detector == "D13a"), "{bad}");
    }
    // The spec round-trips through the file contract with long_only echoed.
    let json = format!("{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{{}}}]}}", ts.fields_json());
    let back = target::parse_file(&json).unwrap();
    assert_eq!(back[0].sha256(), ts.sha256());
    assert!(ts.fields_json().contains("\"long_only\":true"));
    let off = json.replace("\"long_only\":true", "\"long_only\":false");
    assert!(target::parse_file(&off).unwrap_err().contains("long_only"));
}

#[test]
fn report_carries_the_diagnostics_and_a_stable_hash() {
    let bars = wavy(300);
    let mut s = spec("y_4h", 4);
    s.cost.fee_bps = 5.0;
    let l = build(&bars, &s, None).unwrap();
    let r = l.report();
    assert_eq!(r.purge_seconds, 18_000);
    assert!((r.sign_pos_share + r.sign_neg_share + r.sign_zero_share - 1.0).abs() < 1e-12);
    assert!(r.fwd_quantiles.windows(2).all(|w| w[0] <= w[1]));
    assert!(r.clipped_share >= 0.0 && r.clipped_share <= 1.0);
    assert!(r.cost_to_move > 0.0);
    assert_eq!(r.profit_exits + r.stop_exits + r.vertical_exits, r.barrier_rows);
    let json = r.json();
    for key in [
        "\"long_only\":true", "\"effective_n_hours\"", "\"fwd_quantiles\"", "\"p99\"",
        "\"cost_to_move\"", "\"clipped_share\"", "\"purge_seconds\":18000", "\"label_hash\"",
    ] {
        assert!(json.contains(key), "{key} missing in {json}");
    }
    assert_eq!(build(&bars, &s, None).unwrap().label_hash(), r.label_hash);
    let mut other = bars.clone();
    other[150].close += 0.01;
    assert_ne!(build(&other, &s, None).unwrap().label_hash(), r.label_hash);
    let mut dearer = s.clone();
    dearer.cost.fee_bps = 6.0;
    assert_ne!(build(&bars, &dearer, None).unwrap().label_hash(), r.label_hash);
}

#[test]
fn spec_parses_and_refuses_what_it_cannot_label() {
    let s = LongLabelSpec::parse("long:4h,name=a_4h,delay=1,fee=5,spread=2,slip=1,vol=48,floor=3,clip=2,up=1.5,down=0.5,stride=2", HOUR).unwrap();
    assert_eq!((s.name.as_str(), s.horizon_bars, s.delay_bars, s.sigma_window, s.stride), ("a_4h", 4, 1, 48, 2));
    assert_eq!((s.cost.fee_bps, s.cost.spread_bps, s.cost.slippage_bps), (5.0, 2.0, 1.0));
    assert_eq!((s.sigma_floor_bps, s.clip, s.barrier_up, s.barrier_down), (3.0, 2.0, 1.5, 0.5));
    assert_eq!(LongLabelSpec::parse("long:6,fee=0", HOUR).unwrap().name, "h6");
    let missing = LongLabelSpec::parse("long:6", HOUR).unwrap_err();
    assert!(missing.contains("fee=") && missing.contains("fee=0"), "{missing}");
    for bad in ["long:90m,fee=0", "long:4,fee=0,side=short", "long:4,fee=0,vol=1", "long:4,fee=-1", "long:4,fee=0,x=1", "name=a,fee=0", "long:4,fee=0,name=A"] {
        assert!(LongLabelSpec::parse(bad, HOUR).is_err(), "{bad}");
    }
    assert_eq!(labels::parse_column("lbl_a_4h_m_net"), Some(("a_4h", "m_net")));
    assert_eq!(labels::parse_column("lbl_x_long_net"), Some(("x", "long_net")));
    assert_eq!(labels::parse_column("fwd16"), None);
    assert_eq!(labels::parse_column("lbl__y"), None);
}

#[test]
fn top_moves_are_the_largest_absolute_forward_moves_with_stamps() {
    let bars = wavy(300);
    let l = build(&bars, &spec("y_2h", 2), None).unwrap();
    let r = l.report();
    assert_eq!(r.top_moves.len(), labels::TOP_MOVES);
    let mut all: Vec<(f64, i64)> = (0..bars.len())
        .filter(|&t| l.fwd[t].is_finite())
        .map(|t| (l.fwd[t].abs(), bars[t].time))
        .collect();
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    for (m, w) in r.top_moves.iter().zip(&all) {
        assert_eq!(m.fwd.abs().to_bits(), w.0.to_bits());
        assert_eq!(m.time, w.1);
        assert!(m.symbol.is_none());
    }
    assert!(r.json().contains("\"top_abs_moves\":{\"n\":10,\"moves\":[{\"symbol\":null,\"time\":"));
}

#[test]
fn accumulator_merges_symbols_and_hash_follows_every_series() {
    let a = wavy(120);
    let mut b = wavy(120);
    for bar in &mut b {
        bar.close *= 1.01;
        bar.high *= 1.01;
    }
    let s = spec("y_3h", 3);
    let (la, lb) = (build(&a, &s, None).unwrap(), build(&b, &s, None).unwrap());
    let feed = |first: &labels::LongLabels, second: &labels::LongLabels| {
        let mut acc = labels::LabelAccumulator::new(&s);
        acc.add("AAA", first);
        acc.add("BBB", second);
        acc.finish()
    };
    let r = feed(&la, &lb);
    assert_eq!(r.rows, 240);
    assert_eq!(r.valid_rows, la.report().valid_rows + lb.report().valid_rows);
    assert!(r.top_moves.iter().any(|m| m.symbol.as_deref() == Some("AAA")) || r.top_moves.iter().any(|m| m.symbol.as_deref() == Some("BBB")));
    assert_eq!(feed(&la, &lb).label_hash, r.label_hash, "stable");
    assert_ne!(feed(&la, &la).label_hash, r.label_hash, "changes with one symbol's labels");
    assert_ne!(feed(&lb, &la).label_hash, r.label_hash, "depends on the series order");
}

#[test]
fn label_family_columns_are_refused_as_inputs_by_the_d13a_canary() {
    let ts = spec("y_4h", 4).target_spec("lbl_y_4h_y", "lbl_y_4h_fwd");
    for kind in ["fwd", "y", "long_net", "action", "t1", "exit_ret", "m_net", "m_barrier", "weight"] {
        let col = format!("lbl_other_{kind}");
        let report = target::check(&ts, &["ret1".to_string(), col.clone()], false).unwrap_err();
        assert!(report.leaks.iter().any(|l| l.detector == "D13a" && l.column.as_deref() == Some(col.as_str())), "{col}");
    }
    assert!(target::check(&ts, &["ret1".to_string(), "fwd16".to_string()], false).is_ok());
    let view = rcsr::registry::select_view(
        &["ret1".to_string(), "lbl_a_fwd".to_string()],
        24,
        1,
        0,
    );
    assert!(view.map(|v| !v.columns.iter().any(|c| c.starts_with("lbl_"))).unwrap_or(true));
}

fn noisy(n: usize, seed: u32, locked: std::ops::Range<usize>) -> Vec<Bar> {
    let mut rng = Mt::new(seed);
    let mut p = 100.0;
    (0..n)
        .map(|i| {
            let amp = if locked.contains(&i) { 0.02 } else { 0.004 };
            let o = p;
            p *= 1.0 + amp * (rng.next_f64() - 0.5);
            let c = p;
            bar(i, o, o.max(c) * (1.0 + 0.5 * amp * rng.next_f64()), o.min(c) * (1.0 - 0.5 * amp * rng.next_f64()), c)
        })
        .collect()
}

#[test]
fn uniqueness_weights_ignore_locked_prices() {
    let n = 300;
    let (a, b) = (150usize, 200usize);
    let mut s = spec("h3", 3);
    s.stride = 1;
    // Two series that agree on every bar outside the locked range.
    let base = noisy(n, 1, a..b);
    let mut other = base.clone();
    let alt = noisy(n, 2, a..b);
    for i in a..b {
        other[i] = alt[i];
    }
    let range = rcsr::lockbox::Lockbox { start: T0 + a as i64 * HOUR, end: T0 + b as i64 * HOUR };
    let embargo = 10 * HOUR;
    let acc = labels::LabelAccumulator::with_range(&s, Some(range)).with_embargo(embargo);
    let (x, y) = (acc.build(&base, None).unwrap(), acc.build(&other, None).unwrap());
    let (px, py) = (build(&base, &s, None).unwrap(), build(&other, &s, None).unwrap());
    let purge = s.purge_seconds();
    let (mut outside, mut plain_differs) = (0, 0);
    for t in 0..n {
        let time = T0 + t as i64 * HOUR;
        let outside_span = time + purge <= range.start || time >= range.end + embargo;
        if outside_span && x.weight[t].is_finite() {
            outside += 1;
            assert_eq!(x.weight[t].to_bits(), y.weight[t].to_bits(), "row {t}");
            plain_differs += (px.weight[t].to_bits() != py.weight[t].to_bits()) as usize;
        }
    }
    assert!(outside > 200, "{outside}");
    assert!(plain_differs > 100, "plain weights must depend on locked prices: {plain_differs}");
    let norm = x.weight_norm_rows;
    assert_eq!(norm, outside);
    let mean: f64 = (0..n)
        .filter(|&t| x.weight[t].is_finite() && (T0 + t as i64 * HOUR + purge <= range.start || T0 + t as i64 * HOUR >= range.end + embargo))
        .map(|t| x.weight[t])
        .sum::<f64>()
        / norm as f64;
    assert!((mean - 1.0).abs() < 1e-12, "{mean}");
}

#[test]
fn uniqueness_weights_without_a_range_match_the_plain_build() {
    let bars = noisy(120, 3, 0..0);
    let s = spec("h3", 3);
    let plain = build(&bars, &s, None).unwrap();
    let acc = labels::LabelAccumulator::new(&s).with_embargo(10 * HOUR);
    let kept = acc.build(&bars, None).unwrap();
    assert_eq!(plain.label_hash(), kept.label_hash());
    assert_eq!(plain.weight_norm_rows, plain.report().valid_rows);
}

#[test]
fn uniqueness_weights_are_nan_when_every_row_is_refused() {
    let bars = noisy(60, 4, 0..0);
    let s = spec("h3", 3);
    let range = rcsr::lockbox::Lockbox { start: T0, end: T0 + 100 * HOUR };
    let l = labels::LabelAccumulator::with_range(&s, Some(range)).build(&bars, None).unwrap();
    assert_eq!(l.weight_norm_rows, 0);
    assert!(l.weight.iter().all(|w| w.is_nan()));
}

// ---------------------------------------------------------------------------
// #931: labels at 72, 168 and 336 hours, time-anchored stride, purge and embargo.
// ---------------------------------------------------------------------------

#[path = "common/long_horizon.rs"]
mod lh;
#[path = "common/scratch.rs"]
mod scratch;

use lh::{ANCHOR, HORIZONS};
use rcsr::lockbox::Lockbox;

fn path_str(p: &std::path::Path) -> String {
    p.to_string_lossy().into_owned()
}

fn run_rcsr(args: &[&str]) -> (i32, String, String) {
    let out = rcsr_cmd::rcsr().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Builds `out.rcsb` from `bars` (one series per symbol) through `rcsr enrich`
/// with `ret1` and `labels`; `extra` adds flags such as `--locked-test`.
fn enrich_cli(dir: &std::path::Path, series: &[(&str, &[Bar])], labels: &[String], extra: &[&str]) -> String {
    let src = path_str(&dir.join("in.rcsb"));
    let out = path_str(&dir.join("out.rcsb"));
    lh::write_store(&src, series);
    let mut a: Vec<String> = ["enrich", "--data", &src, "--out", &out, "--cpu", "--features", "ret1"].iter().map(|s| s.to_string()).collect();
    for l in labels {
        a.push("--label".to_string());
        a.push(l.clone());
    }
    a.extend(extra.iter().map(|s| s.to_string()));
    let args: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, _, err) = run_rcsr(&args);
    assert_eq!(code, 0, "{err}");
    out
}

#[test]
fn long_horizon_specs_and_purge() {
    let s = scratch::Scratch::new("lh-specs");
    let a = lh::bars(ANCHOR - 8 * HOUR, 1200, 0.0);
    let labels: Vec<String> = HORIZONS.iter().map(|(h, _)| lh::src(*h, &format!(",stride={h},anchor=2021-01-04"))).collect();
    let out = enrich_cli(s.path(), &[("AAA", &a)], &labels, &[]);
    let store = rcsr::store::FileStore::open(&out).unwrap();
    let names = store.header().names();
    assert_eq!(names.iter().filter(|n| n.starts_with("lbl_h")).count(), 27, "{names:?}");
    for (h, _) in HORIZONS {
        for (kind, _) in labels::COLUMN_KINDS {
            assert!(names.contains(&format!("lbl_h{h}_{kind}")), "lbl_h{h}_{kind}");
        }
    }
    let tpath = target::sidecar_path(&out).unwrap();
    let specs = target::parse_file(&std::fs::read_to_string(&tpath).unwrap()).unwrap();
    for (h, purge) in HORIZONS {
        let e = target::entry_for(&specs, &format!("lbl_h{h}_y")).unwrap_or_else(|| panic!("lbl_h{h}_y"));
        assert_eq!(e.horizon_seconds, h as f64 * HOUR as f64, "h{h}");
        assert_eq!(e.required_purge_seconds(), purge as f64, "h{h}");
        assert_eq!(e.return_column.as_deref(), Some(format!("lbl_h{h}_fwd").as_str()));
        assert_eq!(e.cost_bps, Some(20.0));
        assert_eq!(lh::spec(&lh::src(h, "")).purge_seconds(), purge);
    }
    // An anchor needs a stride above one bar: exit 2.
    let bad = path_str(&s.path().join("bad.rcsb"));
    let src = path_str(&s.path().join("in.rcsb"));
    for label in ["long:168,fee=0,anchor=2021-01-04", "long:168,fee=0,stride=1,anchor=2021-01-04", "long:168,fee=0,stride=168,anchor=soon"] {
        let (code, _, err) = run_rcsr(&["enrich", "--data", &src, "--out", &bad, "--cpu", "--label", label]);
        assert_eq!(code, 2, "{label}: {err}");
        assert!(err.contains("anchor"), "{label}: {err}");
    }
}

/// `sum_i mean_{k in span_i} 1 / c_k`, with `c_k` counted row by row.
fn brute_force_effective_n(l: &labels::LongLabels, bars: &[Bar]) -> f64 {
    let rows: Vec<(usize, usize)> = (0..bars.len())
        .filter(|&t| l.fwd[t].is_finite())
        .map(|t| (t + 1, lh::row_of(bars, l.t1[t] - HOUR as f64)))
        .collect();
    let mut c = vec![0usize; bars.len()];
    for &(a, b) in &rows {
        for slot in &mut c[a..=b] {
            *slot += 1;
        }
    }
    rows.iter().map(|&(a, b)| (a..=b).map(|k| 1.0 / c[k] as f64).sum::<f64>() / (b - a + 1) as f64).sum()
}

#[test]
fn long_horizon_valid_rows_and_uniqueness() {
    let bars = lh::bars(T0, 2000, 0.0);
    let strided = lh::build(&bars, "long:168,fee=0,stride=168");
    let valid: Vec<usize> = (0..bars.len()).filter(|&t| strided.fwd[t].is_finite()).collect();
    assert_eq!(valid, (0..=10).map(|k| k * 168).collect::<Vec<_>>());
    let r = strided.report();
    assert_eq!(r.valid_rows, 11);
    assert_eq!(r.average_uniqueness, 1.0);
    let dense = lh::build(&bars, "long:168,fee=0");
    assert_eq!(dense.report().valid_rows, 1832);
    for l in [&strided, &dense] {
        let want = brute_force_effective_n(l, &bars);
        let got = l.report().effective_n;
        assert!((got - want).abs() < 1e-9, "effective_n {got} vs brute force {want}");
    }
}

fn kept_stamps(bars: &[Bar], src: &str) -> Vec<i64> {
    let l = lh::build(bars, src);
    (0..bars.len()).filter(|&t| l.fwd[t].is_finite()).map(|t| bars[t].time).collect()
}

#[test]
fn anchored_stride_aligns_symbols() {
    let a = lh::bars(ANCHOR - 8 * HOUR, 2000, 0.0);
    let b = lh::bars(ANCHOR - 3 * HOUR, 2000, 1.3);
    let anchored = "long:168,fee=0,stride=168,anchor=2021-01-04";
    let (ka, kb) = (kept_stamps(&a, anchored), kept_stamps(&b, anchored));
    assert_eq!(ka.len(), 11);
    assert!(ka.iter().chain(&kb).all(|t| (t - ANCHOR).rem_euclid(168 * HOUR) == 0));
    assert_eq!(ka, kb, "both symbols decide at the same stamps");
    assert_eq!(kept_stamps(&a, "long:168,fee=0,stride=168,anchor=1609718400"), ka, "epoch spelling");
    assert_eq!(kept_stamps(&a, "long:168,fee=0,stride=168,anchor=2021-02-01"), ka, "an anchor after the first bar");
    let plain = "long:168,fee=0,stride=168";
    assert_ne!(kept_stamps(&a, plain), kept_stamps(&b, plain), "row-index stride differs by symbol");
    let r = lh::build(&a, anchored).report();
    assert_eq!(r.stride_rows, 2000 - (0..2000).filter(|t| (a[*t].time - ANCHOR).rem_euclid(168 * HOUR) == 0).count());
    assert!(r.json().contains("\"stride\":168,\"anchor\":1609718400"), "{}", r.json());
    for bad in ["long:168,fee=0,anchor=2021-01-04", "long:168,fee=0,stride=1,anchor=2021-01-04", "long:168,fee=0,stride=168,anchor=soon", "long:168,fee=0,stride=168,anchor=1609718400.5"] {
        assert!(LongLabelSpec::parse(bad, HOUR).is_err(), "{bad}");
    }
}

#[test]
fn existing_label_hashes_are_unchanged() {
    let bars = wavy(600);
    let pinned = [
        ("long:1,name=h1,fee=10", "c651db18041c1fbe9b5b177ab27c633eac8f6488368ed8d9d31798e01045fed0"),
        ("long:4,name=h4,fee=10,stride=4", "f7be5ce9530d5218ebfcde56afcbae9694adaf0c96d29312aecf78d6a276d3ca"),
        ("long:24,name=h24,fee=10,stride=24", "9a66d623773e9795f42503a36ca021564609ec22b768708ea6f626f1d66f2c6d"),
    ];
    for (src, want) in pinned {
        let got = build(&bars, &LongLabelSpec::parse(src, HOUR).unwrap(), None).unwrap().label_hash();
        assert_eq!(got, want, "{src}");
    }
}

#[test]
fn lock_boundary_rows_per_horizon() {
    let bars = lh::bars(T0, 3000, 0.0);
    let range = Lockbox { start: T0 + 2000 * HOUR, end: T0 + 2500 * HOUR };
    for ((h, purge), norm) in HORIZONS.into_iter().zip([1928usize, 1832, 1664]) {
        let s = lh::spec(&lh::src(h, ""));
        assert_eq!(s.purge_seconds(), purge);
        let mut acc = labels::LabelAccumulator::with_range(&s, Some(range)).with_embargo(1000 * HOUR);
        let l = acc.build(&bars, None).unwrap();
        assert_eq!(l.weight_norm_rows, norm, "h{h}");
        acc.add("AAA", &l);
        let r = acc.finish();
        assert_eq!(r.locked_test.unwrap().rows_left_out, 1000, "h{h}");
        assert_eq!(r.weight_norm_rows, norm, "h{h}");
        let json = r.json();
        assert!(json.contains(&format!("\"purge_rows_per_symbol\":{}", h + 1)), "h{h}: {json}");
        assert!(json.contains(&format!("\"tail_rows_without_label\":{h}")), "h{h}: {json}");
    }
}

#[test]
fn label_scale_is_causal_under_future_edits() {
    let bars = lh::bars(T0, 1500, 0.0);
    let cut = 700;
    let mut edited = bars.clone();
    for (k, b) in edited.iter_mut().enumerate().skip(cut + 1) {
        b.open *= 1.5 + 0.001 * k as f64;
        b.high *= 2.0;
        b.low *= 0.5;
        b.close *= 1.7;
    }
    for (h, _) in HORIZONS {
        let src = lh::src(h, "");
        let (x, y) = (lh::build(&bars, &src), lh::build(&edited, &src));
        let mut compared = 0;
        for t in 0..=cut {
            assert_eq!(x.sigma_h[t].to_bits(), y.sigma_h[t].to_bits(), "h{h} row {t}");
            compared += x.sigma_h[t].is_finite() as usize;
        }
        assert!(compared > 500, "h{h}: {compared}");
        // The edit reaches the labels of rows whose window crosses the cut, so the test has teeth.
        assert_ne!(x.fwd[cut].to_bits(), y.fwd[cut].to_bits());
    }
}

#[test]
fn weights_ignore_locked_rows() {
    let n = 3000;
    let (a, b) = (1500usize, 1800usize);
    let base = noisy(n, 1, a..b);
    let alt = noisy(n, 2, a..b);
    let mut other = base.clone();
    other[a..b].copy_from_slice(&alt[a..b]);
    let range = Lockbox { start: T0 + a as i64 * HOUR, end: T0 + b as i64 * HOUR };
    let embargo = 400 * HOUR;
    for (h, purge) in HORIZONS {
        let s = lh::spec(&lh::src(h, ""));
        let acc = labels::LabelAccumulator::with_range(&s, Some(range)).with_embargo(embargo);
        let (x, y) = (acc.build(&base, None).unwrap(), acc.build(&other, None).unwrap());
        let mut compared = 0;
        for t in 0..n {
            let time = T0 + t as i64 * HOUR;
            if (time + purge <= range.start || time >= range.end + embargo) && x.weight[t].is_finite() {
                assert_eq!(x.weight[t].to_bits(), y.weight[t].to_bits(), "h{h} row {t}");
                compared += 1;
            }
        }
        assert_eq!(compared, x.weight_norm_rows, "h{h}");
        assert!(compared > 1500, "h{h}: {compared}");
    }
}

/// A one-symbol store with stride-1 labels at the three horizons and a declared
/// locked test `[S, S + 100 h)`, `S` at bar 1600.
fn locked_store(dir: &std::path::Path) -> (String, i64) {
    let start = ANCHOR - 8 * HOUR;
    let a = lh::bars(start, 2600, 0.0);
    let s = start + 1600 * HOUR;
    let range = format!("{},{}", s, s + 100 * HOUR);
    let labels: Vec<String> = HORIZONS.iter().map(|(h, _)| lh::src(*h, "")).collect();
    let out = enrich_cli(dir, &[("AAA", &a)], &labels, &["--allow-overlap", "--locked-test", &range]);
    (out, s)
}

fn search_args<'a>(data: &'a str, target: &'a str, vars: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec![
        "--data", data, "--target", target, "--vars", vars, "--time", "ts", "--split", "temporal", "--cpu", "--no-dev-log",
        "--seed", "3", "--generations", "2", "--pop", "8", "--quiet",
    ];
    a.extend_from_slice(extra);
    a
}

#[test]
fn new_label_columns_are_refused_as_inputs() {
    let s = scratch::Scratch::new("lh-vars");
    let (data, start) = locked_store(s.path());
    let exclude = format!("ts >= {}", start - 400 * HOUR);
    for (h, _) in HORIZONS {
        for kind in ["fwd", "y", "long_net", "action", "t1", "exit_ret", "m_net", "m_barrier", "weight"] {
            let col = format!("lbl_h{h}_{kind}");
            let vars = format!("ret1,{col}");
            let (code, _, err) = run_rcsr(&search_args(&data, "lbl_h72_y", &vars, &["--exclude", &exclude]));
            assert_eq!(code, 5, "{col}: {err}");
            assert!(err.contains(&col), "{col}: {err}");
        }
    }
    // The scale column is a construction column of every y target: refused under the clean-input gate.
    let (code, _, err) = run_rcsr(&search_args(&data, "lbl_h336_y", "ret1,vol96", &["--exclude", &exclude, "--require-clean-inputs"]));
    assert_eq!(code, 5, "{err}");
    assert!(err.contains("vol96"), "{err}");
}

#[test]
fn lockbox_counts_purge_rows_per_horizon() {
    let s = scratch::Scratch::new("lh-lockbox");
    let (data, start) = locked_store(s.path());
    let exclude = format!("ts >= {start}");
    for (h, _) in HORIZONS {
        let target = format!("lbl_h{h}_y");
        let (code, _, err) = run_rcsr(&search_args(&data, &target, "ret1", &["--exclude", &exclude, "--test-frac", "0.5"]));
        assert_eq!(code, 3, "h{h}: {err}");
        assert!(err.contains(&format!("{} row(s) whose label window reaches the declared locked test", h + 1)), "h{h}: {err}");
    }
}

/// Two symbols of hourly bars with h168 labels at an anchored weekly stride,
/// and a windows table with one expanding fold.
fn h168_store(dir: &std::path::Path) -> (String, String) {
    let start = ANCHOR - 8 * HOUR;
    let a = lh::bars(start, 9000, 0.0);
    let b = lh::bars(start, 9000, 1.3);
    let out = enrich_cli(dir, &[("AAA", &a), ("BBB", &b)], &[lh::src(168, ",stride=168,anchor=2021-01-04")], &[]);
    let w = |weeks: i64| rcsr::protocol::iso((ANCHOR + weeks * 168 * HOUR) as f64);
    let windows = path_str(&dir.join("windows.csv"));
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f0,{},{},{},{},{}\n",
            w(0),
            w(30),
            w(20),
            w(32),
            w(52)
        ),
    )
    .unwrap();
    (out, windows)
}

#[test]
fn fit_label_effective_n_in_reports() {
    let s = scratch::Scratch::new("lh-fit-label");
    let (data, windows) = h168_store(s.path());
    let report = rcsr::target::label_report_for(&data, "lbl_h168_y").expect("label report");
    let j = rcsr::devlog::Json::parse(&report).unwrap();
    assert_eq!(j.get("average_uniqueness").and_then(|v| v.as_f64()), Some(1.0));
    let want = format!(
        "\"fit_label\":{{\"name\":\"h168\",\"stride\":168,\"average_uniqueness\":{},\"effective_n\":{},\"effective_n_hours\":{}}}",
        rcsr::devlog::num(j.get("average_uniqueness").unwrap().as_f64().unwrap()),
        rcsr::devlog::num(j.get("effective_n").unwrap().as_f64().unwrap()),
        rcsr::devlog::num(j.get("effective_n_hours").unwrap().as_f64().unwrap())
    );
    let spec = rcsr::target::sidecar_path(&data).unwrap();
    let w = |weeks: i64| ANCHOR + weeks * 168 * HOUR;
    let window = format!("f0,{},{},{},{}", w(0), w(30), w(32), w(52));
    let (code, out, err) = run_rcsr(&[
        "walkforward", "--data", &data, "--target", "lbl_h168_y", "--target-spec", &spec, "--expr", "ret1", "--vars", "ret1",
        "--time", "ts", "--group-by", "symbol", "--periods-per-year", "52", "--window", &window, "--json",
    ]);
    assert_eq!(code, 0, "{err}");
    let split = rcsr::manifest::object(&out, "split").expect("split");
    assert!(split.contains(&want), "walkforward split {split} lacks {want}");
    let (code, out, err) = run_rcsr(&[
        "ladder", "--data", &data, "--target", "lbl_h168_y", "--time", "ts", "--group", "symbol", "--features", "ret1",
        "--windows", &windows, "--target-spec", &spec, "--draws", "100", "--gbm-trees", "20", "--json",
    ]);
    assert_eq!(code, 0, "{err}");
    let split = rcsr::manifest::object(&out, "split").expect("ladder split");
    assert!(split.contains(&want), "ladder split {split} lacks {want}");
}

#[test]
fn embargo_after_test_block_covers_h336() {
    // Library guards: a declared embargo below the label purge still keeps every
    // training row at least the purge after the locked end.
    let purge = 1_213_200.0;
    let (ls, le) = (T0 as f64 + 1000.0 * HOUR as f64, T0 as f64 + 1100.0 * HOUR as f64);
    let bars = lh::bars(T0, 4000, 0.0);
    let l = lh::build(&bars, &lh::src(336, ",stride=336"));
    let n = bars.len();
    let panel = rcsr::ladder::Panel::new(
        vec!["x".to_string()],
        vec![(0..n).map(|i| (i % 5) as f64).collect()],
        bars.iter().map(|b| b.time as f64).collect(),
        vec![0; n],
        l.y.clone(),
        l.fwd.clone(),
        vec![],
    )
    .unwrap();
    let fold = |train_start: f64| rcsr::protocol::WindowRow {
        scheme: "expanding".to_string(),
        window: rcsr::walkforward::Window {
            label: "after".to_string(),
            train_start,
            train_end: train_start + 1500.0 * HOUR as f64,
            test_start: train_start + 1500.0 * HOUR as f64 + purge,
            test_end: train_start + 2500.0 * HOUR as f64 + purge,
        },
        validation_start: train_start + 1000.0 * HOUR as f64,
    };
    for embargo in [0.0, 345_600.0] {
        let e = rcsr::ladder::make_folds(&panel, &[fold(le + purge - HOUR as f64)], purge, &[(ls, le)], embargo).unwrap_err();
        assert!(e.contains("reaches the declared locked test"), "embargo {embargo}: {e}");
        let folds = rcsr::ladder::make_folds(&panel, &[fold(le + purge)], purge, &[(ls, le)], embargo).unwrap();
        let first = folds[0].fit.iter().map(|&i| panel.time[i]).fold(f64::INFINITY, f64::min);
        assert!(first >= le + purge, "embargo {embargo}: first fit row {first}");
    }
    let mut spec = rcsr::walkforward::EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: None,
        windows: Vec::new(),
        horizon: 336.0 * HOUR as f64,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.5,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(26.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: rcsr::walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    };
    spec.windows = vec![fold(le + purge).window];
    let opts = rcsr::walkforward::ProtocolOptions {
        entry_offset: HOUR as f64,
        locked_ranges: vec![(ls as i64, le as i64)],
        locked_embargo: 0,
        ..rcsr::walkforward::ProtocolOptions::default()
    };
    assert_eq!(opts.locked_reach(&spec), vec![(ls, le, ls - purge, le + purge)]);

    // The store path: a ret1-only store with an h336 label declares the label purge as its embargo.
    let s = scratch::Scratch::new("lh-embargo");
    let start = ANCHOR - 8 * HOUR;
    let a = lh::bars(start, 9000, 0.0);
    let (s0, e0) = (start + 2000 * HOUR, start + 2100 * HOUR);
    let range = format!("{s0},{e0}");
    let data = enrich_cli(s.path(), &[("AAA", &a)], &[lh::src(336, ",stride=336,anchor=2021-01-04")], &["--locked-test", &range]);
    let enrich_json = std::fs::read_to_string(format!("{data}.enrich.json")).unwrap();
    let embargo = rcsr::manifest::object(&enrich_json, "lockbox_embargo").unwrap();
    assert!(embargo.contains("\"seconds\":1213200") && embargo.contains("\"column\":\"lbl_h336_"), "{embargo}");
    let spec_path = rcsr::target::sidecar_path(&data).unwrap();
    let wf = |train_start: i64| {
        let window = format!(
            "after,{},{},{},{}",
            train_start,
            train_start + 2016 * HOUR,
            train_start + 2016 * HOUR + 1_213_200,
            train_start + 4032 * HOUR + 1_213_200
        );
        run_rcsr(&[
            "walkforward", "--data", &data, "--target", "lbl_h336_y", "--target-spec", &spec_path, "--expr", "ret1", "--vars",
            "ret1", "--time", "ts", "--periods-per-year", "26", "--window", &window, "--json",
        ])
    };
    let (code, _, err) = wf(e0 + 1_213_200 - HOUR);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("reaches the declared locked test"), "{err}");
    let (code, out, err) = wf(e0 + 1_213_200);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"train_rows\":"), "{out}");
}
