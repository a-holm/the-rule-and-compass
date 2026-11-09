//! Baseline ladder on a small synthetic panel: the controls are always
//! present, every rung shares rows, folds and costs, and the equal-cost and
//! multiple-testing outputs are filled.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::ladder::{self, Fold, Options, Panel, Report};
use rcsr::protocol::{self, WindowRow};
use rcsr::rng::Mt;
use rcsr::target;
use rcsr::walkforward::Window;
use std::path::PathBuf;

const T0: f64 = 1_640_995_200.0;
const STEP: f64 = 3600.0;
const SYMBOLS: usize = 6;
const STEPS: usize = 420;

fn gauss(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

struct Fixture {
    panel: Panel,
    folds: Vec<Fold>,
    rows: Vec<WindowRow>,
    spec: target::TargetSpec,
}

fn spec() -> target::TargetSpec {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"label\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"ret\",\"return_units\":\"log\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\"exit_price\":\"close\",\
\"offset\":0.002,\"scale\":{\"kind\":\"column\",\"column\":\"vol\",\"multiplier\":6.0,\"window_seconds\":3600,\"ends_at_offset_seconds\":0},\
\"clip\":[-1.0,1.0],\"units\":\"log_return/(6*vol)\",\"cost_bps\":null,\"cost_in_label\":false,\
\"construction_columns\":[\"vol\"],\"note\":null}]}";
    target::parse_file(text).unwrap().remove(0)
}

fn windows() -> Vec<WindowRow> {
    (0..3)
        .map(|k| {
            let test_start = T0 + (240 + 60 * k) as f64 * STEP;
            let train_end = test_start - 2.0 * STEP;
            WindowRow {
                scheme: "expanding".to_string(),
                window: Window {
                    label: format!("f{}", k),
                    train_start: T0,
                    train_end,
                    test_start,
                    test_end: test_start + 60.0 * STEP,
                },
                validation_start: train_end - 60.0 * STEP,
            }
        })
        .collect()
}

fn fixture(seed: u32) -> Fixture {
    let mut rng = Mt::new(seed);
    let n = SYMBOLS * STEPS;
    let (mut time, mut group) = (Vec::new(), Vec::new());
    let (mut signal, mut vol, mut n1, mut n2, mut lag) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut ret, mut label) = (vec![], vec![]);
    let spec = spec();
    for g in 0..SYMBOLS {
        let mut prev = 0.0;
        for t in 0..STEPS {
            let s = gauss(&mut rng);
            let v = 0.004 + 0.002 * rng.next_f64();
            let r = 0.0035 * s + 0.002 * gauss(&mut rng);
            time.push(T0 + t as f64 * STEP);
            group.push(g as u64);
            signal.push(s);
            vol.push(v);
            n1.push(gauss(&mut rng));
            n2.push(gauss(&mut rng));
            lag.push(prev);
            prev = r;
            ret.push(r);
            label.push(spec.target_from_return(r, v));
        }
    }
    assert_eq!(time.len(), n);
    let panel = Panel::new(
        vec!["signal".into(), "vol".into(), "n1".into(), "n2".into(), "lag".into()],
        vec![signal.clone(), vol.clone(), n1, n2, lag],
        time,
        group,
        label,
        ret,
        vec![],
    )
    .unwrap();
    let rows = windows();
    let folds = ladder::make_folds(&panel, &rows, 2.0 * STEP, &[], 0.0).unwrap();
    Fixture { panel, folds, rows, spec }
}

fn options(f: &Fixture) -> Options {
    Options {
        spec: Some(f.spec.clone()),
        vol_col: Some("vol".into()),
        reversal_col: Some("lag".into()),
        signal_col: Some("signal".into()),
        draws: 200,
        gbm_trees: 60,
        purge_seconds: 2.0 * STEP,
        ..Options::default()
    }
}

fn run(f: &Fixture, o: &Options) -> Report {
    ladder::run_ladder(&f.panel, &f.folds, o).unwrap()
}

#[test]
fn folds_keep_fit_validation_and_test_apart() {
    let f = fixture(1);
    for (fold, w) in f.folds.iter().zip(&f.rows) {
        let fit_end = *fold.fit.iter().map(|&i| &f.panel.time[i]).max_by(|a, b| a.total_cmp(b)).unwrap();
        let valid_start = *fold.valid.iter().map(|&i| &f.panel.time[i]).min_by(|a, b| a.total_cmp(b)).unwrap();
        let valid_end = *fold.valid.iter().map(|&i| &f.panel.time[i]).max_by(|a, b| a.total_cmp(b)).unwrap();
        let test_start = *fold.test.iter().map(|&i| &f.panel.time[i]).min_by(|a, b| a.total_cmp(b)).unwrap();
        assert!(valid_start - fit_end >= 2.0 * STEP, "purge between fit and validation");
        assert!(test_start - valid_end >= 2.0 * STEP, "purge between validation and test");
        assert_eq!(w.window.label, fold.label);
    }
}

#[test]
fn controls_are_in_every_table_with_exposure_and_beta() {
    let f = fixture(2);
    let r = run(&f, &options(&f));
    let cash = r.row("cash").unwrap();
    assert_eq!(cash.exposure, 0.0);
    assert_eq!(cash.excess_over_cash, 0.0);
    for (_, net) in &cash.net_by_cost {
        assert_eq!(*net, 0.0);
    }
    let bh = r.row("buy_and_hold").unwrap();
    assert_eq!(bh.exposure, 1.0);
    assert!(bh.beta.is_finite());
    let ew = r.row("equal_weight_universe").unwrap();
    assert_eq!(ew.exposure, 1.0);
    assert!((ew.beta - 1.0).abs() < 1e-9, "the universe has beta 1 on itself, got {}", ew.beta);
    assert!(bh.excess_over_cash <= ew.excess_over_cash, "buy-and-hold pays entry and exit costs");
    let md = r.markdown();
    for needle in ["cash", "buy_and_hold", "equal_weight_universe", "exposure", "beta"] {
        assert!(md.contains(needle), "markdown table lacks {needle}");
    }
    let js = r.json();
    for needle in ["\"cash\"", "\"buy_and_hold\"", "\"equal_weight_universe\"", "\"exposure\"", "\"beta\""] {
        assert!(js.contains(needle), "json lacks {needle}");
    }
}

#[test]
fn every_rung_runs_and_is_long_or_flat() {
    let f = fixture(3);
    let r = run(&f, &options(&f));
    for name in [
        "constant_long_flat",
        "zero_param",
        "ols_ridge",
        "best_single_feature",
        "vol_binned20",
        "naive_reversal",
        "top_tercile_long",
        "feature_threshold",
        "gbm",
        "matched_exposure_random_long",
    ] {
        let row = r.row(name).unwrap_or_else(|| panic!("no rung {name}"));
        assert_eq!(row.status, "ok", "{name}: {:?}", row.note);
        assert!((0.0..=1.0).contains(&row.exposure), "{name} exposure {}", row.exposure);
        assert_eq!(row.net_by_cost.len(), 3, "{name}: one net per cost in the grid");
    }
    let tercile = r.row("top_tercile_long").unwrap();
    assert!(tercile.exposure > 0.2 && tercile.exposure < 0.4, "top tercile of 6 symbols holds 2, got {}", tercile.exposure);
}

#[test]
fn informative_models_beat_cash_and_report_rank_ic() {
    let f = fixture(4);
    let r = run(&f, &options(&f));
    for name in ["ols_ridge", "gbm", "best_single_feature"] {
        let row = r.row(name).unwrap();
        assert!(row.excess_over_cash > 0.0, "{name} net over cash {}", row.excess_over_cash);
        assert!(row.rank_ic > 0.2, "{name} rank IC {}", row.rank_ic);
        assert!(row.rank_ic_ci.0 > 0.0, "{name} IC interval {:?}", row.rank_ic_ci);
    }
    assert!(r.row("ols_ridge").unwrap().return_mse.is_finite());
    assert_eq!(r.row("best_single_feature").unwrap().params.contains("signal"), true);
}

#[test]
fn equal_cost_fields_and_multiple_testing_inputs_are_recorded() {
    let f = fixture(5);
    let mut o = options(&f);
    o.sr_evals = Some(12_345);
    o.sr_wall_seconds = Some(61.5);
    let r = run(&f, &o);
    let gbm = r.row("gbm").unwrap();
    assert!(gbm.fits > 0 && gbm.wall_seconds >= 0.0);
    let ols = r.row("ols_ridge").unwrap();
    assert!(ols.fits > 0 && ols.trials > 1);
    assert_eq!(r.sr_evals, Some(12_345));
    assert_eq!(r.sr_wall_seconds, Some(61.5));
    let total: u64 = r.rows.iter().filter(|x| x.status == "ok").map(|x| x.trials).sum();
    assert_eq!(r.n_trials, total);
    assert!(r.bh.is_some() && r.spa.is_some() && r.dsr.is_some() && r.pbo.is_some());
    assert_eq!(r.dsr.as_ref().unwrap().1.n_trials, r.n_trials as f64);
    let js = r.json();
    for needle in ["\"n_trials\"", "\"fdr\"", "\"spa\"", "\"dsr\"", "\"pbo\"", "\"candidate_evaluations\"", "\"wall_seconds\""] {
        assert!(js.contains(needle), "json lacks {needle}");
    }
    assert!(r.notes.iter().any(|n| n.contains("repeated inside every fold")));
}

#[test]
fn a_missing_input_is_not_run_with_a_reason() {
    let f = fixture(6);
    let r = run(&f, &Options { draws: 100, gbm_trees: 30, purge_seconds: 2.0 * STEP, ..Options::default() });
    for (name, why) in [
        ("zero_param", "--target-spec"),
        ("vol_binned20", "--vol-col"),
        ("naive_reversal", "--reversal-col"),
        ("feature_threshold", "--signal-col"),
        ("symbolic_regression", "--sr-score"),
    ] {
        let row = r.row(name).unwrap();
        assert_eq!(row.status, "not_run", "{name}");
        assert!(row.note.as_deref().unwrap().contains(why), "{name}: {:?}", row.note);
        assert!(row.excess_over_cash.is_nan());
    }
}

#[test]
fn a_searched_member_that_loses_to_the_zero_param_rung_is_flagged() {
    let f = fixture(7);
    let vol = f.panel.col("vol").unwrap().to_vec();
    let zero: Vec<f64> = vol.iter().map(|&v| f.spec.null_prediction(v)).collect();
    let sig = f.panel.col("signal").unwrap().to_vec();
    let with = |col: Vec<f64>| -> Report {
        let mut p = f.panel.clone();
        p.extra.push(("sr".into(), col));
        let mut o = options(&f);
        o.sr_score_col = Some("sr".into());
        ladder::run_ladder(&p, &f.folds, &o).unwrap()
    };
    let same = with(zero);
    assert_eq!(same.row("symbolic_regression").unwrap().beats_zero_param, Some(false));
    assert_eq!(same.row("zero_param").unwrap().beats_zero_param, None);
    let good = with(sig);
    assert_eq!(good.row("symbolic_regression").unwrap().beats_zero_param, Some(true));
}

#[test]
fn symbolic_regression_is_interpretability_only_when_gbm_wins() {
    let f = fixture(8);
    let mut rng = Mt::new(99);
    let noise: Vec<f64> = (0..f.panel.len()).map(|_| gauss(&mut rng)).collect();
    let mut p = f.panel.clone();
    p.extra.push(("sr".into(), noise));
    let mut o = options(&f);
    o.sr_score_col = Some("sr".into());
    let r = ladder::run_ladder(&p, &f.folds, &o).unwrap();
    assert_eq!(r.gbm_beats_sr, Some(true));
    assert_eq!(r.row("symbolic_regression").unwrap().role, "interpretability_only");
    assert!(r.markdown().contains("interpretability only"));
}

#[test]
fn the_report_is_deterministic() {
    let f = fixture(9);
    let o = options(&f);
    let a = run(&f, &o).json();
    let mut b = run(&f, &o).json();
    let strip = |s: &str| {
        let s = &s[..s.find(",\"ledger\":").unwrap()];
        let mut out = String::new();
        let mut rest = s;
        while let Some(i) = rest.find("\"wall_seconds\":") {
            out.push_str(&rest[..i]);
            let tail = &rest[i..];
            let end = tail.find(',').unwrap();
            rest = &tail[end..];
        }
        out.push_str(rest);
        out
    };
    b = strip(&b);
    assert_eq!(strip(&a), b);
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(name);
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

#[test]
fn one_command_reproduces_the_table() {
    let f = fixture(10);
    let (_scratch, dir) = scratch("cli");
    let mut csv = String::from("ts,sym,signal,vol,n1,n2,lag,y,ret\n");
    for i in 0..f.panel.len() {
        csv.push_str(&format!(
            "{},s{},{},{},{},{},{},{},{}\n",
            f.panel.time[i] as i64,
            f.panel.group[i],
            f.panel.col("signal").unwrap()[i],
            f.panel.col("vol").unwrap()[i],
            f.panel.col("n1").unwrap()[i],
            f.panel.col("n2").unwrap()[i],
            f.panel.col("lag").unwrap()[i],
            f.panel.y[i],
            f.panel.ret[i]
        ));
    }
    let data = dir.join("panel.csv");
    std::fs::write(&data, csv).unwrap();
    let mut win = String::from("scheme,label,train_start,train_end,validation_start,test_start,test_end\n");
    for w in &f.rows {
        win.push_str(&format!(
            "expanding,{},{},{},{},{},{}\n",
            w.window.label,
            protocol::iso(w.window.train_start),
            protocol::iso(w.window.train_end),
            protocol::iso(w.validation_start),
            protocol::iso(w.window.test_start),
            protocol::iso(w.window.test_end)
        ));
    }
    let windows = dir.join("windows.csv");
    std::fs::write(&windows, win).unwrap();
    let spec_text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"label\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"ret\",\"return_units\":\"log\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\"exit_price\":\"close\",\
\"offset\":0.002,\"scale\":{\"kind\":\"column\",\"column\":\"vol\",\"multiplier\":6.0,\"window_seconds\":3600,\"ends_at_offset_seconds\":0},\
\"clip\":[-1.0,1.0],\"units\":\"log_return/(6*vol)\",\"cost_bps\":null,\"cost_in_label\":false,\
\"construction_columns\":[\"vol\"],\"note\":null}]}";
    let spec = dir.join("spec.json");
    std::fs::write(&spec, spec_text).unwrap();
    let out = dir.join("ladder.json");
    let md = dir.join("ladder.md");
    let run_cli = || {
        rcsr_cmd::rcsr()
            .args([
                "ladder", "--data", data.to_str().unwrap(), "--target", "y", "--return-col", "ret", "--time", "ts",
                "--group", "sym", "--features", "signal,vol,n1,n2,lag", "--windows", windows.to_str().unwrap(),
                "--purge-seconds", "7200", "--vol-col", "vol", "--reversal-col", "lag", "--signal-col", "signal",
                "--target-spec", spec.to_str().unwrap(), "--draws", "100", "--gbm-trees", "30",
                "--out", out.to_str().unwrap(), "--markdown", md.to_str().unwrap(),
            ])
            .output()
            .expect("run rcsr ladder")
    };
    let first = run_cli();
    assert_eq!(first.status.code(), Some(0), "{}", String::from_utf8_lossy(&first.stderr));
    let json = std::fs::read_to_string(&out).unwrap();
    ledger::check(&json, "ladder", &["io", "pnl"]);
    assert!(json.contains("\"schema\":\"rcsr-ladder/1\""));
    for name in ["cash", "buy_and_hold", "equal_weight_universe", "zero_param", "ols_ridge", "gbm", "top_tercile_long"] {
        assert!(json.contains(&format!("\"name\":\"{}\"", name)), "{name} missing");
    }
    assert!(std::fs::read_to_string(&md).unwrap().contains("| cash |"));
    assert!(json.contains("\"min_group_step_seconds\":3600"), "the report carries the measured step");

    let overlap = dir.join("spec_overlap.json");
    std::fs::write(&overlap, spec_text.replace("\"horizon_seconds\":3600", "\"horizon_seconds\":10800")).unwrap();
    let refused = rcsr_cmd::rcsr()
        .args([
            "ladder", "--data", data.to_str().unwrap(), "--target", "y", "--return-col", "ret", "--time", "ts",
            "--group", "sym", "--features", "signal,vol,n1,n2,lag", "--windows", windows.to_str().unwrap(),
            "--purge-seconds", "7200", "--target-spec", overlap.to_str().unwrap(), "--draws", "100", "--gbm-trees", "30",
        ])
        .output()
        .unwrap();
    assert_eq!(refused.status.code(), Some(2));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("overlapping returns") && err.contains("10800-second label horizon"), "{err}");

    let missing = rcsr_cmd::rcsr().args(["ladder", "--data", "x.csv"]).output().unwrap();
    assert_eq!(missing.status.code(), Some(2));
}

#[path = "common/ledger.rs"]
mod ledger;

#[test]
fn the_json_carries_a_phase_ledger_with_the_pnl_stage() {
    let f = fixture(11);
    let report = run(&f, &options(&f));
    let l = ledger::check(&report.json(), "ladder", &["pnl"]);
    assert_eq!(l.busy, 0.0, "the ladder has no device work");
}

#[test]
fn fold_spacing_refuses_a_label_horizon_longer_than_the_row_step() {
    let f = fixture(12);
    assert_eq!(ladder::check_fold_spacing(&f.panel, &f.folds, 3600.0), Ok(Some(3600.0)));
    assert_eq!(ladder::check_fold_spacing(&f.panel, &f.folds, 0.0), Ok(Some(3600.0)));
    let e = ladder::check_fold_spacing(&f.panel, &f.folds, 3.0 * STEP).unwrap_err();
    assert!(e.contains("overlapping returns") && e.contains("window 'f0'"), "{e}");
    let mut o = options(&f);
    o.horizon_seconds = 3.0 * STEP;
    let e = ladder::run_ladder(&f.panel, &f.folds, &o).unwrap_err();
    assert!(e.contains("overlapping returns"), "{e}");
    assert_eq!(ladder::spacing_exit(&e, 3), 2);
    o.horizon_seconds = STEP;
    assert!(run(&f, &o).json().contains("\"min_group_step_seconds\":3600"));
}

// ---------------------------------------------------------------------------
// #931: the purge at 72, 168 and 336 hours.
// ---------------------------------------------------------------------------

#[path = "common/long_horizon.rs"]
mod lh;

/// Two symbols of hourly bars with stride-`h` labels: the panel (return =
/// `lbl_fwd`, target = `lbl_y`) and the label end `t1` of every panel row.
fn long_label_panel(h: usize, n: usize) -> (Panel, Vec<f64>) {
    let (mut time, mut group, mut x, mut y, mut ret, mut t1) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for (g, phase) in [0.0, 1.3].into_iter().enumerate() {
        let bars = lh::bars(T0 as i64, n, phase);
        let l = lh::build(&bars, &lh::src(h, &format!(",stride={h}")));
        for t in 0..n {
            time.push(bars[t].time as f64);
            group.push(g as u64);
            x.push(((t * 7 + g) % 11) as f64);
            y.push(l.y[t]);
            ret.push(l.fwd[t]);
            t1.push(l.t1[t]);
        }
    }
    (Panel::new(vec!["x".into()], vec![x], time, group, y, ret, vec![]).unwrap(), t1)
}

/// One fold of a windows table written with the protocol default purge.
fn default_purge_fold(h: usize) -> WindowRow {
    let test_start = T0 + (20 * h) as f64 * STEP;
    let train_end = test_start - lh::DEFAULT_PURGE;
    WindowRow {
        scheme: "expanding".to_string(),
        window: Window { label: "f0".to_string(), train_start: T0, train_end, test_start, test_end: test_start + (6 * h) as f64 * STEP },
        validation_start: train_end - (8 * h) as f64 * STEP,
    }
}

fn no_training_label_reaches_test_block(h: usize) {
    let (panel, t1) = long_label_panel(h, 30 * h);
    let spec = lh::spec(&lh::src(h, ""));
    let purge = (spec.purge_seconds() as f64).max(lh::DEFAULT_PURGE);
    assert_eq!(purge, spec.target_spec("y", "fwd").required_purge_seconds());
    let row = default_purge_fold(h);
    let folds = ladder::make_folds(&panel, &[row.clone()], purge, &[], 0.0).unwrap();
    let f = &folds[0];
    let test_start = row.window.test_start;
    assert!(!f.fit.is_empty() && !f.valid.is_empty() && !f.test.is_empty());
    for &i in f.fit.iter().chain(&f.valid) {
        assert!(t1[i] < test_start, "h{h} row {i}: label end {} at or after the test start", t1[i]);
        assert!(test_start - panel.time[i] >= purge, "h{h} row {i}: stamp {} within the purge of the test start", panel.time[i]);
    }
    let fit_end = f.fit.iter().map(|&i| panel.time[i]).fold(f64::MIN, f64::max);
    assert!(row.validation_start - fit_end >= purge, "h{h}: fit rows reach the validation slice");
    // The table's own gap is shorter than the label: the stride row one horizon
    // before the test start lies inside [validation_start, train_end).
    let near = test_start - (h as f64) * STEP;
    assert!(near < row.window.train_end && near >= row.validation_start && near + purge > test_start);
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
fn ladder_takes_the_larger_purge_at_168h() {
    let (_scratch, dir) = scratch("ladder-h168");
    let h = 168usize;
    let (panel, _) = long_label_panel(h, 60 * h);
    let mut csv = String::from("ts,sym,x,vol96,lbl_h168_y,lbl_h168_fwd\n");
    for i in 0..panel.len() {
        if panel.ret[i].is_finite() && panel.y[i].is_finite() {
            csv.push_str(&format!(
                "{},s{},{},0.01,{},{}\n",
                panel.time[i] as i64,
                panel.group[i],
                panel.col("x").unwrap()[i],
                panel.y[i],
                panel.ret[i]
            ));
        }
    }
    let data = dir.join("panel.csv");
    std::fs::write(&data, csv).unwrap();
    let wk = |k: usize| protocol::iso(T0 + (k * h) as f64 * STEP);
    let windows = dir.join("windows.csv");
    std::fs::write(
        &windows,
        format!("scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f0,{},{},{},{},{}\n", wk(0), wk(30), wk(20), wk(32), wk(52)),
    )
    .unwrap();
    let spec = dir.join("spec.json");
    std::fs::write(&spec, lh::spec_file(&lh::spec(&lh::src(h, ",stride=168")))).unwrap();
    let out = rcsr_cmd::rcsr()
        .args([
            "ladder", "--data", data.to_str().unwrap(), "--target", "lbl_h168_y", "--time", "ts", "--group", "sym", "--features", "x",
            "--windows", windows.to_str().unwrap(), "--purge-seconds", "600000", "--target-spec", spec.to_str().unwrap(),
            "--draws", "100", "--gbm-trees", "20", "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let json = String::from_utf8_lossy(&out.stdout);
    assert!(json.contains("\"purge_seconds\":608400,\"purge_source\":\"target_spec\""), "{json}");
}

// ---------------------------------------------------------------------------
// #927: trend_ma and trend_donchian
// ---------------------------------------------------------------------------

const TREND_COLS: [&str; 8] = [
    "ma20_ratio",
    "ma60_ratio",
    "mtf_4h_ma_ratio",
    "mtf_1d_ma_ratio",
    "don20",
    "don96",
    "mtf_4h_donchian_pos",
    "mtf_1d_donchian_pos",
];

struct Trend {
    panel: Panel,
    folds: Vec<Fold>,
}

/// Rows in `[cut.0, cut.1)` are replaced by a different draw of every column
/// and return, after the rest of the panel has been generated.
fn trend_fixture(seed: u32, trending: bool, drop: &[&str], windows_used: usize, cut: Option<(f64, f64)>) -> Trend {
    let mut rng = Mt::new(seed);
    let mut other = Mt::new(seed ^ 0x5eed);
    let (mut time, mut group, mut ret) = (vec![], vec![], vec![]);
    let mut cols: Vec<Vec<f64>> = vec![vec![]; TREND_COLS.len()];
    for g in 0..SYMBOLS {
        let mut s = 0.0;
        for t in 0..STEPS {
            s = 0.97 * s + 0.25 * gauss(&mut rng);
            let mut r = (if trending { 0.002 * s } else { 0.0 }) + 0.004 * gauss(&mut rng);
            let mut xs: Vec<f64> = (0..TREND_COLS.len()).map(|k| s * (1.0 + 0.1 * k as f64) + 0.5 * gauss(&mut rng)).collect();
            let at = T0 + t as f64 * STEP;
            if cut.is_some_and(|(a, b)| at >= a && at < b) {
                r = 0.05 * other.next_f64() - 0.02;
                for x in &mut xs {
                    *x = 4.0 * gauss(&mut other) + 3.0;
                }
            }
            time.push(at);
            group.push(g as u64);
            ret.push(r);
            for (c, x) in cols.iter_mut().zip(xs) {
                c.push(x);
            }
        }
    }
    let keep: Vec<usize> = (0..TREND_COLS.len()).filter(|&k| !drop.contains(&TREND_COLS[k])).collect();
    let panel = Panel::new(
        keep.iter().map(|&k| TREND_COLS[k].to_string()).collect(),
        keep.iter().map(|&k| cols[k].clone()).collect(),
        time,
        group,
        ret.clone(),
        ret,
        vec![],
    )
    .unwrap();
    let rows: Vec<WindowRow> = windows().into_iter().take(windows_used).collect();
    let folds = ladder::make_folds(&panel, &rows, 2.0 * STEP, &[], 0.0).unwrap();
    Trend { panel, folds }
}

fn trend_options(on: bool) -> Options {
    Options {
        trend_rungs: on,
        draws: 200,
        gbm_trees: 60,
        purge_seconds: 2.0 * STEP,
        ..Options::default()
    }
}

fn run_trend(t: &Trend, on: bool) -> Report {
    ladder::run_ladder(&t.panel, &t.folds, &trend_options(on)).unwrap()
}

#[test]
fn trend_rung_finds_trend() {
    let t = trend_fixture(11, true, &[], 3, None);
    let r = run_trend(&t, true);
    for name in ["trend_ma", "trend_donchian"] {
        let row = r.row(name).unwrap();
        assert_eq!(row.status, "ok", "{name}");
        assert!(row.excess_over_cash > 0.0, "{name} net {}", row.excess_over_cash);
        assert!(row.params.contains("column@band_sd_multiple_by_fold=["), "{}", row.params);
    }
}

#[test]
fn trend_rung_noise_is_null() {
    let t = trend_fixture(12, false, &[], 3, None);
    let r = run_trend(&t, true);
    for name in ["trend_ma", "trend_donchian"] {
        let row = r.row(name).unwrap();
        let matched = r.row("matched_exposure_random_long").unwrap();
        assert!(row.mean_ci.0 <= 0.0 && 0.0 <= row.mean_ci.1, "{name} ci {:?}", row.mean_ci);
        let sd = {
            let m = row.series.iter().sum::<f64>() / row.series.len() as f64;
            (row.series.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (row.series.len() - 1) as f64).sqrt()
        };
        let band = 3.0 * sd * (row.series.len() as f64).sqrt();
        assert!(
            (row.excess_over_cash - matched.excess_over_cash).abs() <= band,
            "{name} net {} matched {} band {}",
            row.excess_over_cash,
            matched.excess_over_cash,
            band
        );
    }
}

#[test]
fn trend_rungs_off_are_not_run_and_n_trials_grows_by_32() {
    let t = trend_fixture(13, true, &[], 3, None);
    let off = run_trend(&t, false);
    let on = run_trend(&t, true);
    for name in ["trend_ma", "trend_donchian"] {
        assert_eq!(off.row(name).unwrap().status, "not_run");
        assert_eq!(on.row(name).unwrap().trials, 16);
    }
    assert_eq!(on.n_trials, off.n_trials + 32);
    let (so, sn) = (off.spa.as_ref().unwrap(), on.spa.as_ref().unwrap());
    assert_eq!(sn.models, so.models + 2);
    assert_eq!(on.pbo.as_ref().unwrap().candidates, off.pbo.as_ref().unwrap().candidates + 2);
    for name in ["trend_ma", "trend_donchian"] {
        assert!(on.row(name).unwrap().bh_rejected.is_some());
    }
}

#[test]
fn trend_rung_drops_a_missing_column_with_a_note() {
    let t = trend_fixture(14, true, &["don96"], 3, None);
    let on = run_trend(&t, true);
    let row = on.row("trend_donchian").unwrap();
    assert_eq!(row.trials, 12);
    assert!(row.note.as_deref().unwrap().contains("dropped (not in the panel): don96"));
    assert_eq!(on.row("trend_ma").unwrap().trials, 16);
}

fn chosen_by_fold(params: &str) -> Vec<String> {
    let start = params.find('[').unwrap() + 1;
    let end = params.find(']').unwrap();
    params[start..end].split(',').map(|s| s.trim_matches('"').to_string()).collect()
}

#[test]
fn trend_rung_ignores_test_block_rows() {
    let base = trend_fixture(15, true, &[], 3, None);
    let first = &base.folds[0];
    let test_times: Vec<f64> = first.test.iter().map(|&i| base.panel.time[i]).collect();
    let (lo, hi) = (
        test_times.iter().cloned().fold(f64::INFINITY, f64::min),
        test_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0,
    );
    let mid = lo + 30.0 * STEP;
    let a = run_trend(&base, true);
    // every test row of fold 0 changed: the fold's choice comes from fit and validation rows only
    let all = trend_fixture(15, true, &[], 3, Some((lo, hi)));
    let b = run_trend(&all, true);
    // only the second half of the block changed: earlier periods are unchanged
    let late = trend_fixture(15, true, &[], 3, Some((mid, hi)));
    let c = run_trend(&late, true);
    assert_ne!(a.row("buy_and_hold").unwrap().series, b.row("buy_and_hold").unwrap().series);
    for name in ["trend_ma", "trend_donchian"] {
        let (ra, rb, rc) = (a.row(name).unwrap(), b.row(name).unwrap(), c.row(name).unwrap());
        assert_eq!(chosen_by_fold(&ra.params)[0], chosen_by_fold(&rb.params)[0], "{name} choice");
        assert_eq!(chosen_by_fold(&ra.params)[0], chosen_by_fold(&rc.params)[0], "{name} choice");
        let before = a.period_time.iter().filter(|&&t| t < mid).count();
        assert!(before >= 30);
        assert_eq!(ra.series[..before], rc.series[..before], "{name} earlier periods");
        assert_ne!(ra.series, rc.series, "{name} later periods differ");
    }
}

#[test]
fn trend_rung_matches_walkforward_band() {
    let t = trend_fixture(16, true, &[], 1, None);
    let r = run_trend(&t, true);
    let row = r.row("trend_ma").unwrap();
    let pick = chosen_by_fold(&row.params).remove(0);
    let (col, mult) = pick.split_once('@').unwrap();
    let mult: f64 = mult.parse().unwrap();
    let fold = &t.folds[0];
    let x = t.panel.col(col).unwrap();
    let fit: Vec<f64> = fold.fit.iter().map(|&i| x[i]).collect();
    let theta = rcsr::walkforward::quantile(&fit, 0.5);
    let band = mult * rcsr::trading::score_sd(&fit).sd.unwrap();
    let w = windows().remove(0).window;
    let mut csv = String::from("ts,sym,x,target\n");
    for i in 0..t.panel.len() {
        csv.push_str(&format!("{},{},{:e},{:e}\n", t.panel.time[i], t.panel.group[i], x[i], t.panel.ret[i]));
    }
    let table = rcsr::data::parse_table(&csv).unwrap();
    let spec = rcsr::walkforward::EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: Some("sym".to_string()),
        windows: vec![w],
        horizon: 0.0,
        embargo: None,
        cost_bps: 12.0,
        threshold_quantile: 0.5,
        threshold_fixed: Some(theta),
        validation_frac: 0.0,
        periods_per_year: Some(8760.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: rcsr::walkforward::Policy::Band,
        band_width: band,
        min_hold: 1,
        gpu: false,
    };
    let wf = rcsr::walkforward::run_table(&table, &spec, "t").unwrap();
    let m = &wf.windows[0].metrics;
    let wf_net = m.mean_return * m.periods as f64;
    assert!((wf_net - row.excess_over_cash).abs() <= 1e-12, "walkforward {} ladder {}", wf_net, row.excess_over_cash);
}
