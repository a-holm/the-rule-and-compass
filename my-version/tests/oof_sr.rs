//! Out-of-fold symbolic-regression scores (#492): one search run scored on
//! later folds by the ladder and by `ablation score`, and the in-sample guard.

use rcsr::devlog::Json;
use rcsr::protocol;
use rcsr::run_json::RunFile;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const T0: f64 = 1_600_000_000.0;
const STEP: f64 = 900.0;
const ROWS: usize = 600;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-oof-sr-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run_json_text(train_end: f64, validation_end: Option<f64>, purge: f64) -> String {
    let validation = validation_end.map_or(String::new(), |v| format!(",\"validation_time\":[{},{}]", T0, v));
    format!(
        "{{\"variables\":[\"a\"],\"pareto\":[],\"split\":{{\"train_time\":[{},{}],\"required_purge_seconds\":{}{}}}}}",
        T0, train_end, purge, validation
    )
}

#[test]
fn in_sample_has_one_boundary_shared_by_every_caller() {
    let run = RunFile::parse(&run_json_text(1000.0, Some(2000.0), 300.0)).unwrap();
    assert!(run.in_sample(2299.0), "one second before last seen + purge");
    assert!(!run.in_sample(2300.0), "equal to last seen + purge is out of fold");
    assert!(!run.in_sample(5000.0));
    let train_only = RunFile::parse(&run_json_text(1000.0, None, 300.0)).unwrap();
    assert!(train_only.in_sample(1299.0));
    assert!(!train_only.in_sample(1300.0));
    let no_time = RunFile::parse("{\"variables\":[],\"pareto\":[]}").unwrap();
    assert!(no_time.in_sample(f64::MAX), "a run without a time range cannot prove out-of-fold");
    let folds = vec![("early".to_string(), 1500.0), ("late".to_string(), 2300.0)];
    let err = run.oof_flags(&folds, false).unwrap_err();
    assert!(err.contains("early") && !err.contains("late"), "{err}");
    assert_eq!(run.oof_flags(&folds, true).unwrap(), vec![false, true]);
}

struct Fixture {
    data: String,
    spec: String,
    run: PathBuf,
    windows: PathBuf,
    dir: PathBuf,
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().expect("run rcsr")
}

fn fixture(name: &str) -> Fixture {
    let dir = scratch(name);
    let frac = |v: f64| v - v.floor();
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..ROWS {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", (T0 as i64) + (STEP as i64) * i as i64, x1, x2, r, r));
    }
    let data = dir.join("d.csv");
    std::fs::write(&data, text).unwrap();
    let spec_text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
\"horizon_seconds\":900,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
\"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let spec = dir.join("spec.json");
    std::fs::write(&spec, spec_text).unwrap();
    let run = dir.join("A0-s3.run.json");
    let out = rcsr(&[
        "--data", data.to_str().unwrap(), "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec.to_str().unwrap(), "--cpu", "--no-dev-log", "--seed", "3", "--generations", "15",
        "--pop", "32", "--metric", "squared", "--quiet", "--objective", "pnl", "--cost-bps", "5",
        "--validation-frac", "0.3", "--arm", "A0", "--json-out", run.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let facts = RunFile::load(run.to_str().unwrap()).unwrap();
    let (_, seen, purge) = facts.earliest_test_start().expect("the run records its time range");
    let seen_row = ((seen - T0) / STEP).round() as usize;
    let purge_rows = (purge / STEP).ceil() as usize;
    let late_start = seen_row + purge_rows + 1;
    assert!(late_start + 20 < ROWS, "the search leaves rows after what it saw: {late_start}");
    let iso = |i: usize| protocol::iso(T0 + STEP * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\n\
expanding,early,{},{},{},{},{}\nexpanding,late,{},{},{},{},{}\n",
            iso(0),
            iso(100),
            iso(60),
            iso(100),
            iso(150),
            iso(0),
            iso(late_start),
            iso(late_start / 2),
            iso(late_start),
            iso(ROWS),
        ),
    )
    .unwrap();
    Fixture { data: data.to_string_lossy().into_owned(), spec: spec.to_string_lossy().into_owned(), run, windows, dir }
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn json_of(path: &Path) -> Json {
    Json::parse(std::fs::read_to_string(path).unwrap().trim()).unwrap()
}

fn ladder(f: &Fixture, windows: &Path, extra: &[&str], out: &Path) -> Output {
    let mut args = vec![
        "ladder", "--data", &f.data, "--target", "y", "--return-col", "r", "--time", "t", "--features", "x1,x2",
        "--windows", p(windows), "--purge-seconds", "1800", "--draws", "100", "--gbm-trees", "30", "--out", p(out),
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

#[test]
fn ladder_sr_run_refuses_an_in_sample_fold_and_labels_it_with_the_flag() {
    let f = fixture("ladder");
    let out = f.dir.join("ladder.json");
    let refused = ladder(&f, &f.windows, &["--sr-run", p(&f.run)], &out);
    assert_eq!(refused.status.code(), Some(3), "{}", String::from_utf8_lossy(&refused.stderr));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("early") && !err.contains("late"), "{err}");

    let ok = ladder(&f, &f.windows, &["--sr-run", p(&f.run), "--allow-in-sample"], &out);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let j = json_of(&out);
    let sr = j.get("sr_run").expect("sr_run object");
    let folds = match sr.get("folds") {
        Some(Json::Arr(a)) => a.clone(),
        _ => panic!("folds"),
    };
    let flags: Vec<(String, bool)> = folds
        .iter()
        .map(|x| (x.get("fold").unwrap().as_str().unwrap().to_string(), matches!(x.get("oof"), Some(Json::Bool(true)))))
        .collect();
    assert_eq!(flags, vec![("early".to_string(), false), ("late".to_string(), true)]);
    match j.get("folds") {
        Some(Json::Arr(a)) => assert_eq!(a.len(), 1, "the in-sample fold is out of every pooled series"),
        _ => panic!("folds"),
    }
}

#[test]
fn ladder_sr_run_reports_an_ok_rung_with_fits_and_wall_from_the_run() {
    let f = fixture("rung");
    let late_only = f.dir.join("late.csv");
    let text = std::fs::read_to_string(&f.windows).unwrap();
    let kept: Vec<&str> = text.lines().filter(|l| !l.starts_with("expanding,early")).collect();
    std::fs::write(&late_only, kept.join("\n") + "\n").unwrap();
    let out = f.dir.join("ladder.json");
    let ok = ladder(&f, &late_only, &["--sr-run", p(&f.run)], &out);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let j = json_of(&out);
    let rungs = match j.get("rungs") {
        Some(Json::Arr(a)) => a.clone(),
        _ => panic!("rungs"),
    };
    let sr = rungs.iter().find(|r| r.get("name").and_then(Json::as_str) == Some("symbolic_regression")).unwrap();
    assert_eq!(sr.get("status").and_then(Json::as_str), Some("ok"));
    let run = RunFile::load(p(&f.run)).unwrap();
    assert_eq!(sr.get("candidate_evaluations").and_then(Json::as_f64), run.candidates_evaluated.map(|v| v as f64));
    assert_eq!(sr.get("wall_seconds").and_then(Json::as_f64), run.elapsed_seconds);
    let meta = j.get("sr_run").unwrap();
    assert_eq!(meta.get("threshold_source").and_then(Json::as_str), Some("run"));
    assert_eq!(meta.get("run_sha256").and_then(Json::as_str), Some(run.sha256.as_str()));

    let both = ladder(&f, &late_only, &["--sr-run", p(&f.run), "--sr-score", "x1"], &out);
    assert_eq!(both.status.code(), Some(2));
}

#[test]
fn ablation_score_refuses_an_in_sample_fold_and_drops_it_with_the_flag() {
    let f = fixture("ablation");
    let res = f.dir.join("res.json");
    let base = |extra: &[&str]| {
        let mut args = vec![
            "ablation", "score", "--run", p(&f.run), "--data", f.data.as_str(), "--target", "y", "--return-col", "r", "--time", "t",
            "--windows", p(&f.windows), "--purge-seconds", "1800", "--out", p(&res),
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    let refused = base(&[]);
    assert_eq!(refused.status.code(), Some(3), "{}", String::from_utf8_lossy(&refused.stderr));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("early") && !err.contains("late"), "{err}");
    assert!(!res.exists(), "no result file for a refused run");

    let ok = base(&["--allow-in-sample"]);
    assert_eq!(ok.status.code(), Some(0), "{}", String::from_utf8_lossy(&ok.stderr));
    let j = json_of(&res);
    match j.get("folds") {
        Some(Json::Arr(a)) => assert_eq!(a.len(), 1),
        _ => panic!("folds"),
    }
    let oof: Vec<bool> = match j.get("fold_oof") {
        Some(Json::Arr(a)) => a.iter().map(|x| matches!(x.get("oof"), Some(Json::Bool(true)))).collect(),
        _ => panic!("fold_oof"),
    };
    assert_eq!(oof, vec![false, true]);
    let _ = &f.spec;
}
