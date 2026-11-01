//! Long-only horizon, action and triple-barrier labels on small synthetic bars.

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

    let bin = env!("CARGO_BIN_EXE_rcsr");
    for flag in [["enrich", "--allow-short"], ["prep", "--allow-short"]] {
        let out = std::process::Command::new(bin).args(flag).output().unwrap();
        assert!(!out.status.success(), "{flag:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("long-only"), "{flag:?}");
    }
    let out = std::process::Command::new(bin)
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
