//! #487: one cost, one return unit and one position rule from the search to
//! walk-forward, ablation score and the ladder.

use rcsr::devlog::Json;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;
const N: usize = 400;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-cost-contract-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// `(x1, x2, r)` of row `i`; `r` is a log return that depends on `x1`.
fn row(i: usize) -> (f64, f64, f64) {
    let f = i as f64;
    let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
    let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
    let noise = frac(f * 0.324_717_957_244_746) - 0.5;
    (x1, x2, 0.01 * x1 + 0.0005 * noise)
}

fn write_csv(dir: &Path, name: &str, ret: impl Fn(usize) -> f64) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
        let (x1, x2, _) = row(i);
        let r = ret(i);
        text.push_str(&format!("{},{},{},{},{}\n", T0 + STEP * i as i64, x1, x2, r, r));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path, name: &str, units: &str, round_trip: &str) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"{units}\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":{round_trip},\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}]}}"
    );
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// One expanding fold whose test block is rows 210..300, the search's validation slice.
fn write_windows(dir: &Path) -> String {
    let text = format!(
        "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f1,{},{},{},{},{}\n",
        T0,
        T0 + STEP * 150,
        T0 + STEP * 100,
        T0 + STEP * 210,
        T0 + STEP * 300
    );
    let path = dir.join("windows.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn parse(text: &str) -> Json {
    Json::parse(text.trim()).unwrap_or_else(|| panic!("not json: {text}"))
}

fn read_json(path: &str) -> Json {
    parse(&std::fs::read_to_string(path).unwrap())
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}"));
    }
    cur
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
}

fn first_window(j: &Json) -> &Json {
    match at(j, &["windows"]) {
        Json::Arr(a) => &a[0],
        _ => panic!("windows is not an array"),
    }
}

/// Runs the search on `data` and writes its run.json next to it.
fn search(dir: &Path, data: &str, spec: &str, extra: &[&str]) -> String {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "15",
        "--pop", "32", "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
        "--validation-frac", "0.3", "--arm", "A0",
    ];
    args.extend_from_slice(extra);
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let path = dir.join("run.json");
    std::fs::write(&path, &out.stdout).unwrap();
    path.to_string_lossy().into_owned()
}

fn window_arg() -> String {
    format!("v,{},{},{},{}", T0, T0 + STEP * 209, T0 + STEP * 210, T0 + STEP * 300)
}

#[test]
fn one_cost_one_book_across_evaluators() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let run = search(&dir, &data, &spec, &[]);
    let run_json = read_json(&run);

    let book = at(&run_json, &["objective", "validation"]);
    assert_eq!(num(&run_json, &["objective", "cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&run_json, &["objective", "cost", "source"]).as_str(), Some("target_spec"));
    assert_eq!(at(&run_json, &["objective", "cost", "return_units"]).as_str(), Some("log"));
    assert_eq!(num(&run_json, &["objective", "spec", "per_side_bps"]), 10.0);

    // walk-forward: no --cost-bps, no --expr, no --target
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let w = window_arg();
    let wf = rcsr(&["walkforward", "--data", &data, "--time", "t", "--from-run", &run, "--window", &w, "--out", &wf_path]);
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let wf_json = read_json(&wf_path);
    assert_eq!(num(&wf_json, &["cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&wf_json, &["cost", "source"]).as_str(), Some("target_spec"));
    assert_eq!(at(&wf_json, &["cost", "return_units"]).as_str(), Some("log"));
    assert_eq!(at(&wf_json, &["threshold", "source"]).as_str(), Some("fixed"));
    let m = at(first_window(&wf_json), &["metrics"]);
    for (wf_key, obj_key) in [("total_cost", "total_cost"), ("exposure", "exposure"), ("trades", "trades"), ("rows", "rows"), ("entries", "entries")] {
        let (a, b) = (num(m, &[wf_key]), num(book, &[obj_key]));
        assert!((a - b).abs() <= 1e-12, "{wf_key}: walkforward {a} objective {b}");
    }
    let wf_net = num(m, &["mean_return"]) * num(m, &["periods"]);
    assert!((wf_net - num(book, &["net_pnl"])).abs() <= 1e-12, "net {wf_net} vs {}", num(book, &["net_pnl"]));

    // ablation score: same book
    let ab_path = dir.join("ab.json").to_string_lossy().into_owned();
    let ab = rcsr(&[
        "ablation", "score", "--from-run", &run, "--data", &data, "--time", "t", "--windows", &windows,
        "--purge-seconds", "900", "--out", &ab_path,
    ]);
    assert_eq!(code(&ab), 0, "{}", stderr(&ab));
    let ab_json = read_json(&ab_path);
    assert_eq!(num(&ab_json, &["cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&ab_json, &["cost", "source"]).as_str(), Some("target_spec"));
    assert_eq!(at(&ab_json, &["threshold_rule", "kind"]).as_str(), Some("fixed"));
    let ab_net = match at(&ab_json, &["net_by_fold_threshold"]) {
        Json::Arr(a) => match &a[0] {
            Json::Arr(b) => b[0].as_f64().unwrap(),
            _ => panic!(),
        },
        _ => panic!(),
    };
    assert!((ab_net - num(book, &["net_pnl"])).abs() <= 1e-12, "ablation {ab_net}");

    // ladder: every rung at the spec cost and units; buy-and-hold is a rung of the same book
    let lad_path = dir.join("lad.json").to_string_lossy().into_owned();
    let lad = rcsr(&[
        "ladder", "--data", &data, "--target", "y", "--time", "t", "--windows", &windows, "--features", "x1,x2",
        "--purge-seconds", "900", "--target-spec", &spec, "--draws", "50", "--gbm-trees", "10", "--out", &lad_path,
    ]);
    assert_eq!(code(&lad), 0, "{}", stderr(&lad));
    let lad_json = read_json(&lad_path);
    assert_eq!(num(&lad_json, &["cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&lad_json, &["cost", "source"]).as_str(), Some("target_spec"));
    assert_eq!(at(&lad_json, &["cost", "return_units"]).as_str(), Some("log"));
    let Json::Arr(rungs) = at(&lad_json, &["rungs"]) else { panic!() };
    let mut priced = 0;
    for r in rungs {
        let Some(Json::Arr(by_cost)) = r.get("net_by_cost") else { continue };
        if by_cost.is_empty() {
            continue;
        }
        let costs: Vec<f64> = by_cost.iter().map(|c| c.get("cost_bps").unwrap().as_f64().unwrap()).collect();
        assert_eq!(costs, vec![10.0, 20.0, 33.3]);
        if at(r, &["name"]).as_str() == Some("buy_and_hold") {
            let net = by_cost[0].get("net").unwrap().as_f64().unwrap();
            let want = num(book, &["buy_and_hold", "net_pnl"]);
            assert!((net - want).abs() <= 1e-12, "buy and hold {net} vs {want}");
            priced += 1;
        }
    }
    assert_eq!(priced, 1);
}

#[test]
fn walkforward_from_run_prices_the_selected_book() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let run = search(&dir, &data, &spec, &[]);
    let text = std::fs::read_to_string(&run).unwrap();
    let finalist = at(&parse(&text), &["objective", "finalist", "expression"]).as_str().unwrap().to_string();
    let constant = text
        .replacen(&format!("\"expression\":\"{finalist}\""), "\"expression\":\"0.0226\"", 1)
        .replacen("\"constant\":false", "\"constant\":true", 1);
    assert_ne!(constant, text);
    let run_const = dir.join("const.json");
    std::fs::write(&run_const, &constant).unwrap();

    // reference: the objective's own book function on the validation rows
    let rows: Vec<usize> = (210..300).collect();
    let ret: Vec<f64> = rows.iter().map(|&i| row(i).2.exp_m1()).collect();
    let tr = rcsr::trading::TradingRows {
        time: rows.iter().map(|&i| (T0 + STEP * i as i64) as f64).collect(),
        group: vec![0; rows.len()],
        funding: Vec::new(),
    };
    let tspec = rcsr::trading::TradingSpec { cost_bps: 10.0, ..Default::default() };
    let want = rcsr::trading::evaluate(&vec![0.0226; rows.len()], &ret, &tr, &tspec).unwrap();
    assert_eq!(want.exposure, 1.0);

    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let w = window_arg();
    let out = rcsr(&[
        "walkforward", "--data", &data, "--time", "t", "--from-run", run_const.to_str().unwrap(),
        "--window", &w, "--out", &wf_path,
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(!stderr(&out).contains("is constant"), "{}", stderr(&out));
    let wf = read_json(&wf_path);
    let m = at(first_window(&wf), &["metrics"]);
    assert_eq!(num(m, &["exposure"]), 1.0);
    assert!((num(m, &["total_cost"]) - want.total_cost).abs() <= 1e-12);
    let net = num(m, &["mean_return"]) * num(m, &["periods"]);
    assert!((net - want.net_pnl).abs() <= 1e-12, "walkforward {net} objective {}", want.net_pnl);

    // the quantile rule on the same constant is flat and says so
    let flat = rcsr(&[
        "walkforward", "--data", &data, "--time", "t", "--target", "r", "--expr", "0.0226", "--vars", "x1,x2",
        "--window", &w, "--horizon", "900", "--cost-bps", "10", "--out", &wf_path,
    ]);
    assert_eq!(code(&flat), 0, "{}", stderr(&flat));
    assert!(stderr(&flat).contains("is constant"), "{}", stderr(&flat));
    assert_eq!(num(at(first_window(&read_json(&wf_path)), &["metrics"]), &["exposure"]), 0.0);
}

#[test]
fn a_log_return_column_is_converted_with_exp_m1_in_every_evaluator() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |_| 0.1);
    let log_spec = write_spec(&dir, "log.json", "log", "0.0");
    let simple_spec = write_spec(&dir, "simple.json", "simple", "0.0");
    let want = 0.1f64.exp_m1();
    assert!((want - 0.105_170_918_075_647_6).abs() < 1e-15);

    let w = window_arg();
    for (spec, expect) in [(&log_spec, want), (&simple_spec, 0.1)] {
        let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
        let out = rcsr(&[
            "walkforward", "--data", &data, "--time", "t", "--target", "y", "--target-spec", spec, "--expr", "1",
            "--threshold-fixed", "0", "--vars", "x1,x2", "--window", &w, "--out", &wf_path,
        ]);
        assert_eq!(code(&out), 0, "{}", stderr(&out));
        let wf = read_json(&wf_path);
        let m = at(first_window(&wf), &["metrics"]);
        // the first row pays the entry and the window's last row the exit; the rest earn the return
        let gross = num(m, &["mean_return"]) * num(m, &["periods"]) + num(m, &["total_cost"]);
        assert!((gross - expect * 90.0).abs() <= 1e-9, "walkforward gross {gross} for {expect}");
    }

    let windows = write_windows(&dir);
    for (spec, expect) in [(&log_spec, want), (&simple_spec, 0.1)] {
        let lad_path = dir.join("lad.json").to_string_lossy().into_owned();
        let out = rcsr(&[
            "ladder", "--data", &data, "--target", "y", "--time", "t", "--windows", &windows, "--features", "x1,x2",
            "--purge-seconds", "900", "--target-spec", spec, "--draws", "20", "--gbm-trees", "5", "--out", &lad_path,
        ]);
        assert_eq!(code(&out), 0, "{}", stderr(&out));
        let lad = read_json(&lad_path);
        let Json::Arr(rungs) = at(&lad, &["rungs"]) else { panic!() };
        let bh = rungs.iter().find(|r| r.get("name").and_then(Json::as_str) == Some("buy_and_hold")).expect("buy_and_hold rung");
        let net = match at(bh, &["net_by_cost"]) {
            Json::Arr(a) => a[0].get("net").unwrap().as_f64().unwrap(),
            _ => panic!(),
        };
        assert!((net - expect * 90.0).abs() <= 1e-9, "ladder buy and hold {net} for {expect}");
    }

    let a_path = dir.join("ab.json").to_string_lossy().into_owned();
    let run = search(&dir, &data, &log_spec, &[]);
    let text = std::fs::read_to_string(&run).unwrap();
    let finalist = at(&parse(&text), &["objective", "finalist", "expression"]).as_str().unwrap().to_string();
    let run = dir.join("x1.json").to_string_lossy().into_owned();
    std::fs::write(&run, text.replacen(&format!("\"expression\":\"{finalist}\""), "\"expression\":\"x1\"", 1)).unwrap();
    let out = rcsr(&[
        "ablation", "score", "--run", &run, "--data", &data, "--target", "y", "--time", "t", "--windows", &windows,
        "--purge-seconds", "900", "--target-spec", &log_spec, "--cost-bps", "0", "--thresholds", "0.5", "--headline-threshold", "0.5",
        "--out", &a_path,
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let ab = read_json(&a_path);
    let Json::Arr(periods) = at(&ab, &["net_period"]) else { panic!() };
    let sum: f64 = periods.iter().map(|p| p.as_f64().unwrap()).sum();
    let longs = periods.iter().filter(|p| p.as_f64().unwrap() != 0.0).count();
    assert!(longs > 0 && (sum - longs as f64 * want).abs() <= 1e-9, "ablation net {sum} over {longs} long periods");
}

#[test]
fn the_cost_flag_must_agree_with_the_target_spec() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let run = search(&dir, &data, &spec, &[]);
    let w = window_arg();
    let same = "differs from the target spec";

    let wf = |extra: &[&str]| {
        let mut a = vec!["walkforward", "--data", &data, "--time", "t", "--target", "y", "--target-spec", &spec, "--expr", "x1", "--vars", "x1,x2", "--window", &w, "--out", "/dev/null"];
        a.extend_from_slice(extra);
        rcsr(&a)
    };
    let bad = wf(&["--cost-bps", "12"]);
    assert_eq!(code(&bad), 2);
    assert!(stderr(&bad).contains(same) && stderr(&bad).contains("--cost-override"), "{}", stderr(&bad));
    assert_eq!(code(&wf(&["--cost-bps", "10"])), 0);
    let over = wf(&["--cost-override", "30"]);
    assert_eq!(code(&over), 0, "{}", stderr(&over));
    let bad_horizon = wf(&["--horizon", "3600"]);
    assert_eq!(code(&bad_horizon), 2);
    assert!(stderr(&bad_horizon).contains("--horizon"), "{}", stderr(&bad_horizon));

    let lad = |extra: &[&str]| {
        let mut a = vec![
            "ladder", "--data", &data, "--target", "y", "--time", "t", "--windows", &windows, "--features", "x1,x2",
            "--purge-seconds", "900", "--target-spec", &spec, "--draws", "20", "--gbm-trees", "5", "--out", "/dev/null",
        ];
        a.extend_from_slice(extra);
        rcsr(&a)
    };
    let bad = lad(&["--cost-bps", "12"]);
    assert_eq!(code(&bad), 2);
    assert!(stderr(&bad).contains(same), "{}", stderr(&bad));
    assert_eq!(code(&lad(&["--cost-override", "12"])), 0);

    let ab = |extra: &[&str]| {
        let mut a = vec![
            "ablation", "score", "--run", &run, "--data", &data, "--target", "y", "--time", "t", "--windows", &windows,
            "--purge-seconds", "900", "--target-spec", &spec, "--out", "/dev/null",
        ];
        a.extend_from_slice(extra);
        rcsr(&a)
    };
    let bad = ab(&["--cost-bps", "12"]);
    assert_eq!(code(&bad), 2);
    assert!(stderr(&bad).contains(same), "{}", stderr(&bad));

    // the search refuses the same mismatch
    let args = [
        "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal", "--target-spec", &spec,
        "--cpu", "--no-dev-log", "--generations", "1", "--pop", "8", "--json", "--quiet", "--objective", "pnl",
        "--validation-frac", "0.3", "--cost-bps", "12",
    ];
    let bad = rcsr(&args);
    assert_eq!(code(&bad), 2);
    assert!(stderr(&bad).contains(same), "{}", stderr(&bad));
}

#[test]
fn the_cost_source_is_reported() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let w = window_arg();
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let run = |extra: &[&str]| {
        let mut a = vec!["walkforward", "--data", &data, "--time", "t", "--target", "r", "--expr", "x1", "--vars", "x1,x2", "--window", &w, "--out", &wf_path];
        a.extend_from_slice(extra);
        let o = rcsr(&a);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        read_json(&wf_path)
    };
    let d = run(&[]);
    assert_eq!(at(&d, &["cost", "source"]).as_str(), Some("default"));
    assert_eq!(num(&d, &["cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&d, &["cost", "return_units"]).as_str(), Some("simple"));
    let f = run(&["--cost-bps", "7"]);
    assert_eq!(at(&f, &["cost", "source"]).as_str(), Some("flag"));
    assert_eq!(num(&f, &["cost", "per_side_bps"]), 7.0);
}

#[test]
fn from_run_refuses_a_run_without_an_objective_and_competing_rules() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let w = window_arg();
    let plain = dir.join("plain.json");
    std::fs::write(&plain, "{\"tool\":\"rcsr\",\"variables\":[\"x1\"],\"pareto\":[],\"target_spec\":null}").unwrap();
    let out = rcsr(&["walkforward", "--data", &data, "--time", "t", "--from-run", plain.to_str().unwrap(), "--window", &w]);
    assert_eq!(code(&out), 3);
    assert!(stderr(&out).contains("no objective block"), "{}", stderr(&out));

    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let run = search(&dir, &data, &spec, &[]);
    for extra in [vec!["--expr", "x1"], vec!["--threshold-quantile", "0.7"], vec!["--threshold-fixed", "0"]] {
        let mut a = vec!["walkforward", "--data", &data, "--time", "t", "--from-run", &run, "--window", &w];
        a.extend(extra);
        let out = rcsr(&a);
        assert_eq!(code(&out), 2, "{}", stderr(&out));
    }
    let out = rcsr(&["ablation", "score", "--from-run", &run, "--run", &run, "--data", &data, "--time", "t", "--windows", "x", "--out", "x"]);
    assert_eq!(code(&out), 2);
}

#[test]
fn a_long_label_needs_an_explicit_fee() {
    let out = rcsr(&["enrich", "--data", "/nonexistent.rcsb", "--out", "/nonexistent-out.rcsb", "--label", "long:1h"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    let e = stderr(&out);
    assert!(e.contains("fee=") && e.contains("fee=0"), "{e}");
    let ok = rcsr(&["enrich", "--data", "/nonexistent.rcsb", "--out", "/nonexistent-out.rcsb", "--label", "long:1h,fee=0"]);
    assert_eq!(code(&ok), 3, "fee=0 passes argument parsing and then fails on the missing input: {}", stderr(&ok));
}
