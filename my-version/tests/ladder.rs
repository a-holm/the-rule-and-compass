//! Baseline ladder on a small synthetic panel: the controls are always
//! present, every rung shares rows, folds and costs, and the equal-cost and
//! multiple-testing outputs are filled.

use rcsr::ladder::{self, Fold, Options, Panel, Report};
use rcsr::protocol::{self, WindowRow};
use rcsr::rng::Mt;
use rcsr::target;
use rcsr::walkforward::Window;
use std::path::PathBuf;
use std::process::Command;

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
    let folds = ladder::make_folds(&panel, &rows, 2.0 * STEP, None).unwrap();
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

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-ladder-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn one_command_reproduces_the_table() {
    let f = fixture(10);
    let dir = scratch("cli");
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
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
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
    assert!(json.contains("\"schema\":\"rcsr-ladder/1\""));
    for name in ["cash", "buy_and_hold", "equal_weight_universe", "zero_param", "ols_ridge", "gbm", "top_tercile_long"] {
        assert!(json.contains(&format!("\"name\":\"{}\"", name)), "{name} missing");
    }
    assert!(std::fs::read_to_string(&md).unwrap().contains("| cash |"));

    let missing = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(["ladder", "--data", "x.csv"]).output().unwrap();
    assert_eq!(missing.status.code(), Some(2));
}
