//! #487: one cost, one return unit and one position rule from the search to
//! walk-forward, ablation score and the ladder.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::devlog::Json;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;
const N: usize = 400;

#[path = "common/shape.rs"]
mod shape;
#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("cost-contract");
    let path = scratch.path().to_path_buf();
    (scratch, path)
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

/// One expanding fold whose test block is rows 310..400, the search's test block (`--test-frac 0.225`), so the fold starts at the rows the run saw plus the purge.
fn write_windows(dir: &Path) -> String {
    let text = format!(
        "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f1,{},{},{},{},{}\n",
        T0,
        T0 + STEP * 150,
        T0 + STEP * 100,
        T0 + STEP * 310,
        T0 + STEP * 400
    );
    let path = dir.join("windows.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn rcsr(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
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
        "--validation-frac", "0.3", "--test-frac", "0.225", "--arm", "A0",
    ];
    args.extend_from_slice(extra);
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let path = dir.join("run.json");
    std::fs::write(&path, &out.stdout).unwrap();
    path.to_string_lossy().into_owned()
}

fn window_arg() -> String {
    format!("v,{},{},{},{}", T0, T0 + STEP * 309, T0 + STEP * 310, T0 + STEP * 400)
}

#[test]
fn one_cost_one_book_across_evaluators() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let run = search(&dir, &data, &spec, &[]);
    let run_json = read_json(&run);

    let book = at(&run_json, &["objective", "test"]);
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
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let run = search(&dir, &data, &spec, &[]);
    let text = std::fs::read_to_string(&run).unwrap();
    let finalist = at(&parse(&text), &["objective", "finalist", "expression"]).as_str().unwrap().to_string();
    let constant = without_exact(&text)
        .replacen(&format!("\"expression\":\"{finalist}\""), "\"expression\":\"0.0226\"", 1)
        .replacen("\"constant\":false", "\"constant\":true", 1);
    assert_ne!(constant, text);
    let run_const = dir.join("const.json");
    std::fs::write(&run_const, &constant).unwrap();

    // reference: the objective's own book function on the test rows
    let rows: Vec<usize> = (310..400).collect();
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
    let (_scratch, dir) = scratch();
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
    std::fs::write(&run, without_exact(&text).replacen(&format!("\"expression\":\"{finalist}\""), "\"expression\":\"x1\"", 1)).unwrap();
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
    let (_scratch, dir) = scratch();
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
    let (_scratch, dir) = scratch();
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
    let (_scratch, dir) = scratch();
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

const TINY_MEAN: f64 = 3.2e-5;
const TINY_SCALE: f64 = 0.006;

/// Training rows of the search below: 400 rows, test 0.225, validation 0.3.
const TINY_TRAIN_ROWS: usize = 217;

/// Table whose `x1` has training mean `TINY_MEAN` and scale about `TINY_SCALE`,
/// so the display form rounds the centring to a signed zero. Every fourth row
/// sits between 0 and the mean and pays a large return; the sign of the other
/// returns follows `x1`.
fn write_tiny_csv(dir: &Path) -> String {
    let cluster = |i: usize| i % 4 == 3;
    let base: Vec<f64> = (0..N).map(|i| if cluster(i) { 0.0 } else { row(i).0 * 3.4641 * TINY_SCALE }).collect();
    let moving = (0..TINY_TRAIN_ROWS).filter(|&i| !cluster(i)).count() as f64;
    let fixed = (0..TINY_TRAIN_ROWS).filter(|&i| cluster(i)).count() as f64;
    let held: f64 = (0..TINY_TRAIN_ROWS).map(|i| base[i]).sum();
    let shift = (TINY_MEAN * TINY_TRAIN_ROWS as f64 - held - fixed * TINY_MEAN / 2.0) / moving;
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
        let (u, x2, noise) = row(i);
        let (x1, r) = if cluster(i) {
            (TINY_MEAN / 2.0, 0.02)
        } else {
            (base[i] + shift, if u > 0.0 { 0.01 } else { -0.01 } + 0.002 * noise)
        };
        text.push_str(&format!("{},{:?},{:?},{:?},{:?}\n", T0 + STEP * i as i64, x1, x2, r, r));
    }
    let path = dir.join("tiny.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}
/// `run.json` text with every `expression_exact` member removed.
fn without_exact(text: &str) -> String {
    let key = ",\"expression_exact\":\"";
    let mut out = text.to_string();
    while let Some(at) = out.find(key) {
        let start = at + key.len();
        let end = start + out[start..].find('"').expect("closing quote");
        out.replace_range(at..=end, "");
    }
    out
}

#[test]
fn the_evaluated_book_is_the_selected_book() {
    let (_scratch, dir) = scratch();
    let data = write_tiny_csv(&dir);
    let spec = write_spec(&dir, "s.json", "simple", "20.0");
    let run = search(&dir, &data, &spec, &[]);
    let doc = read_json(&run);
    let shown = at(&doc, &["objective", "finalist", "expression"]).as_str().unwrap().to_string();
    assert!(shown.contains("-0 + x1"), "the display must round the centring to zero: {shown}");
    let want_net = num(&doc, &["objective", "validation", "net_pnl"]);
    let want_trades = num(&doc, &["objective", "validation", "trades"]);

    let t = |k: i64| T0 + STEP * k;
    let w = format!("v,{},{},{},{}", t(0), t(215), t(217), t(310));
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let out = rcsr(&[
        "walkforward", "--data", &data, "--time", "t", "--from-run", &run, "--window", &w,
        "--allow-in-sample-windows", "--out", &wf_path,
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let wf = read_json(&wf_path);
    let m = at(first_window(&wf), &["metrics"]);
    let net = num(m, &["mean_return"]) * num(m, &["periods"]);
    assert!((net - want_net).abs() <= 1e-9, "walkforward {net} search {want_net}");
    assert_eq!(num(m, &["trades"]), want_trades);
    assert_eq!(at(&wf, &["selection", "expression_source"]).as_str(), Some("exact"));
}

#[test]
fn a_rounded_source_is_refused_by_the_sealed_read() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let run = search(&dir, &data, &spec, &[]);
    let text = std::fs::read_to_string(&run).unwrap();
    assert!(text.contains("\"expression_exact\":"));
    let old = dir.join("old.run.json");
    std::fs::write(&old, without_exact(&text)).unwrap();
    let old = old.to_string_lossy().into_owned();

    let t = |k: i64| T0 + STEP * k;
    let w = format!("v,{},{},{},{}", t(0), t(215), t(217), t(310));
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let dev = rcsr(&[
        "walkforward", "--data", &data, "--time", "t", "--from-run", &old, "--window", &w,
        "--allow-in-sample-windows", "--out", &wf_path,
    ]);
    assert_eq!(code(&dev), 0, "{}", stderr(&dev));
    assert_eq!(at(&read_json(&wf_path), &["selection", "expression_source"]).as_str(), Some("rounded"));

    let ledger = dir.join("ledger.jsonl").to_string_lossy().into_owned();
    let sealed = dir.join("sealed.json").to_string_lossy().into_owned();
    let sealed_read = rcsr(&[
        "walkforward", "--data", &data, "--time", "t", "--from-run", &old, "--window", &w,
        "--locked-test", "--ledger", &ledger, "--sealed", &sealed,
    ]);
    assert_eq!(code(&sealed_read), 3, "{}", stderr(&sealed_read));
    assert!(stderr(&sealed_read).contains("expression_exact"), "{}", stderr(&sealed_read));
}

const SCORE_BAND: [&str; 6] = ["--fitness", "pnl", "--band", "0.05", "--min-holding", "4"];
const SD_BAND: [&str; 8] = ["--fitness", "pnl", "--band", "1.0", "--band-units", "sd", "--min-holding", "3"];

/// `search` with the run.json kept under `name`.
fn search_as(dir: &Path, data: &str, spec: &str, extra: &[&str], name: &str) -> String {
    let made = search(dir, data, spec, extra);
    let kept = dir.join(name);
    std::fs::rename(made, &kept).unwrap();
    kept.to_string_lossy().into_owned()
}

/// `run.json` text as a run written before #925: no band keys.
fn without_band_keys(text: &str) -> String {
    let mut old = text.to_string();
    for key in ["band", "band_units", "band_sd", "band_resolved", "band_sd_rows", "band_sd_max_time", "band_sd_degenerate"] {
        let needle = format!(",\"{key}\":");
        while let Some(at) = old.find(&needle) {
            let start = at + needle.len();
            let end = start + old[start..].find(|c| c == ',' || c == '}').expect("end of value");
            old.replace_range(at..end, "");
        }
    }
    old
}

/// The window of the run's validation rows; it overlaps the run's own rows.
fn validation_window() -> String {
    let t = |k: i64| T0 + STEP * k;
    format!("v,{},{},{},{}", t(0), t(215), t(217), t(310))
}

fn walk(dir: &Path, data: &str, extra: &[&str]) -> Json {
    let out_path = dir.join("walk.json").to_string_lossy().into_owned();
    let mut args = vec!["walkforward", "--data", data, "--time", "t", "--allow-in-sample-windows", "--out", &out_path];
    args.extend_from_slice(extra);
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    read_json(&out_path)
}

/// The report without timing members, numbers masked.
fn digits_stripped(j: &Json) -> String {
    assert!(matches!(j, Json::Obj(_)), "report is not an object");
    shape::timing_free_shape(j)
}

#[test]
fn from_run_prices_the_band_book_of_the_search() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let w = validation_window();
    for (name, extra) in [("score.json", SCORE_BAND.to_vec()), ("sd.json", SD_BAND.to_vec())] {
        let run = search_as(&dir, &data, &spec, &extra, name);
        let doc = read_json(&run);
        let want = at(&doc, &["objective", "validation"]);
        let s = at(&doc, &["objective", "spec"]);
        assert!(num(s, &["band_resolved"]) > 0.0, "{name}: the band must be live");
        assert!(num(want, &["trades"]) > 0.0, "{name}: the book must trade");
        let wf = walk(&dir, &data, &["--from-run", &run, "--window", &w]);
        assert_eq!(at(&wf, &["policy", "name"]).as_str(), Some("band"), "{name}");
        assert_eq!(num(&wf, &["policy", "min_hold"]), num(s, &["min_holding"]), "{name}");
        let m = at(first_window(&wf), &["metrics"]);
        assert_eq!(num(m, &["trades"]), num(want, &["trades"]), "{name}: trades");
        for (a, b) in [("exposure", "exposure"), ("total_cost", "total_cost")] {
            assert!((num(m, &[a]) - num(want, &[b])).abs() <= 1e-12, "{name}: {a} {} vs {}", num(m, &[a]), num(want, &[b]));
        }
        let net = num(m, &["mean_return"]) * num(m, &["periods"]);
        assert!((net - num(want, &["net_pnl"])).abs() <= 1e-12, "{name}: net {net} vs {}", num(want, &["net_pnl"]));
        let band = at(first_window(&wf), &["band"]);
        assert_eq!(num(band, &["band_width"]).to_bits(), num(s, &["band_resolved"]).to_bits(), "{name}: priced as recorded");
    }
}

#[test]
fn from_run_without_a_band_keeps_the_threshold_book() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let w = validation_window();
    let plain = search_as(&dir, &data, &spec, &[], "plain.json");
    let text = std::fs::read_to_string(&plain).unwrap();
    let old = without_band_keys(&text);
    assert_ne!(old, text);
    assert!(!old.contains("band_resolved"));
    let old_path = dir.join("old.json").to_string_lossy().into_owned();
    std::fs::write(&old_path, &old).unwrap();
    let a = walk(&dir, &data, &["--from-run", &plain, "--window", &w]);
    let b = walk(&dir, &data, &["--from-run", &old_path, "--window", &w]);
    assert_eq!(at(&a, &["policy", "name"]).as_str(), Some("threshold"));
    assert_eq!(num(&a, &["policy", "min_hold"]), 1.0);
    assert_eq!(digits_stripped(&a), digits_stripped(&b));
    assert_eq!(at(&a, &["windows"]), at(&b, &["windows"]));
}

#[test]
fn from_run_refuses_a_conflicting_band_flag() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let w = validation_window();
    let run = search_as(&dir, &data, &spec, &SCORE_BAND, "band.json");
    let base = ["walkforward", "--data", &data, "--time", "t", "--from-run", &run, "--window", &w, "--allow-in-sample-windows", "--out", "/dev/null"];
    let cases: [(&[&str], &str); 5] = [
        (&["--policy", "threshold"], "band book"),
        (&["--policy", "dp"], "band book"),
        (&["--band", "0.1"], "band 0.05 score"),
        (&["--band-units", "sd"], "band 0.05 score"),
        (&["--min-hold", "2"], "minimum hold of 4"),
    ];
    for (flags, named) in cases {
        let mut a = base.to_vec();
        a.extend_from_slice(flags);
        let out = rcsr(&a);
        assert_eq!(code(&out), 2, "{flags:?}: {}", stderr(&out));
        assert!(stderr(&out).contains(named), "{flags:?}: {}", stderr(&out));
    }
    let free = walk(&dir, &data, &["--from-run", &run, "--window", &w]);
    let agree = walk(
        &dir,
        &data,
        &["--from-run", &run, "--window", &w, "--policy", "band", "--band", "0.05", "--band-units", "score", "--min-hold", "4"],
    );
    assert_eq!(digits_stripped(&free), digits_stripped(&agree));
    assert_eq!(at(&free, &["windows"]), at(&agree, &["windows"]));
}

#[test]
fn front_candidates_take_their_own_band() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let run = search_as(&dir, &data, &spec, &SD_BAND, "sd.json");
    let doc = read_json(&run);
    let Json::Arr(front) = at(&doc, &["pareto"]) else { panic!("pareto") };
    let resolved: Vec<f64> = front.iter().map(|m| num(m, &["validation_book", "band_resolved"])).collect();
    assert!(resolved.len() >= 2, "the front needs two members: {resolved:?}");
    assert!(resolved.windows(2).any(|p| p[0] != p[1]), "members share one band: {resolved:?}");
    let t = |k: i64| T0 + STEP * k;
    let windows = dir.join("front-windows.csv");
    std::fs::write(
        &windows,
        format!("scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f,{},{},{},{},{}\n", t(0), t(150), t(100), t(217), t(310)),
    )
    .unwrap();
    let windows = windows.to_string_lossy().into_owned();
    let wf = walk(&dir, &data, &["--front", &run, "--windows", &windows]);
    let win = first_window(&wf);
    let chosen = num(win, &["candidate"]) as usize;
    let want = resolved[chosen];
    assert_eq!(num(at(win, &["band"]), &["band_width"]).to_bits(), want.to_bits(), "selected candidate {chosen}");
    assert_eq!(at(&wf, &["policy", "band_source"]).as_str(), Some("run"));

    let expr = at(&front[chosen], &["expression_exact"]).as_str().unwrap().to_string();
    let theta = format!("{}", num(&doc, &["objective", "spec", "threshold"]));
    let want_band = format!("{want:?}");
    let by_hand = walk(
        &dir,
        &data,
        &[
            "--target", "y", "--target-spec", &spec, "--expr", &expr, "--vars", "x1,x2", "--threshold-fixed", &theta,
            "--policy", "band", "--band", &want_band, "--min-hold", "3", "--windows", &windows,
        ],
    );
    let (a, b) = (at(win, &["metrics"]), at(first_window(&by_hand), &["metrics"]));
    for k in ["trades", "exposure", "total_cost", "mean_return", "periods"] {
        assert!((num(a, &[k]) - num(b, &[k])).abs() <= 1e-12, "{k}: front {} by hand {}", num(a, &[k]), num(b, &[k]));
    }
}

#[test]
fn from_run_records_the_band_source() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let w = validation_window();
    let run = search_as(&dir, &data, &spec, &SD_BAND, "sd.json");
    let doc = read_json(&run);
    let s = at(&doc, &["objective", "spec"]);
    let wf = walk(&dir, &data, &["--from-run", &run, "--window", &w]);
    assert_eq!(at(&wf, &["policy", "band_source"]).as_str(), Some("run"));
    assert_eq!(at(&wf, &["policy", "band_units"]).as_str(), Some("sd"));
    let band = at(first_window(&wf), &["band"]);
    assert_eq!(at(band, &["band_units"]).as_str(), Some("sd"));
    for k in ["band", "band_sd", "band_resolved", "band_sd_max_time"] {
        assert_eq!(num(band, &[k]).to_bits(), num(s, &[k]).to_bits(), "{k}");
    }
    let flag = walk(
        &dir,
        &data,
        &["--target", "y", "--target-spec", &spec, "--expr", "x1", "--vars", "x1,x2", "--threshold-fixed", "0", "--policy", "band", "--band", "0.05", "--window", &w],
    );
    assert_eq!(at(&flag, &["policy", "band_source"]).as_str(), Some("flag"));
}

fn ladder_sr(dir: &Path, data: &str, spec: &str, windows: &str, run: &str, tag: &str) -> Json {
    let out_path = dir.join(format!("ladder-{tag}.json")).to_string_lossy().into_owned();
    let out = rcsr(&[
        "ladder", "--data", data, "--target", "y", "--time", "t", "--windows", windows, "--features", "x1,x2",
        "--purge-seconds", "900", "--target-spec", spec, "--draws", "50", "--gbm-trees", "10", "--sr-run", run, "--out", &out_path,
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    read_json(&out_path)
}

fn ablation_score(dir: &Path, data: &str, windows: &str, mode: &str, run: &str, tag: &str) -> Json {
    let out_path = dir.join(format!("ablation-{tag}.json")).to_string_lossy().into_owned();
    let mut args = vec!["ablation", "score", mode, run, "--data", data, "--target", "y", "--time", "t", "--windows", windows, "--purge-seconds", "900", "--out", &out_path];
    if mode == "--run" {
        args.extend(["--thresholds", "0.5", "--headline-threshold", "0.5"]);
    }
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    read_json(&out_path)
}

fn sr_row(ladder: &Json) -> &Json {
    let Json::Arr(rungs) = at(ladder, &["rungs"]) else { panic!("rungs") };
    rungs.iter().find(|r| at(r, &["name"]).as_str() == Some("symbolic_regression")).expect("symbolic_regression rung")
}

fn ladder_net(row: &Json) -> f64 {
    let Json::Arr(by_cost) = at(row, &["net_by_cost"]) else { panic!("net_by_cost") };
    num(&by_cost[0], &["net"])
}

fn ablation_net(j: &Json) -> f64 {
    let Json::Arr(by_fold) = at(j, &["net_by_fold_threshold"]) else { panic!("net_by_fold_threshold") };
    let Json::Arr(by_threshold) = &by_fold[0] else { panic!("by threshold") };
    by_threshold[0].as_f64().unwrap()
}

#[test]
fn ladder_sr_rung_prices_the_run_band_book() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    for (name, extra) in [("score.json", SCORE_BAND.to_vec()), ("sd.json", SD_BAND.to_vec())] {
        let run = search_as(&dir, &data, &spec, &extra, name);
        let doc = read_json(&run);
        let want = at(&doc, &["objective", "test"]);
        let s = at(&doc, &["objective", "spec"]);
        assert!(num(s, &["band_resolved"]) > 0.0, "{name}: the band must be live");
        assert!(num(want, &["trades"]) > 0.0, "{name}: the book must trade");
        let lad = ladder_sr(&dir, &data, &spec, &windows, &run, name);
        let row = sr_row(&lad);
        assert_eq!(at(row, &["status"]).as_str(), Some("ok"), "{name}");
        assert!(at(row, &["params"]).as_str().unwrap().contains("as recorded"), "{name}");
        for (a, b) in [("exposure", "exposure"), ("entries", "entries")] {
            assert!((num(row, &[a]) - num(want, &[b])).abs() <= 1e-12, "{name}: {a} {} vs {}", num(row, &[a]), num(want, &[b]));
        }
        assert!((ladder_net(row) - num(want, &["net_pnl"])).abs() <= 1e-12, "{name}: net {} vs {}", ladder_net(row), num(want, &["net_pnl"]));
        let report = at(&lad, &["sr_run"]);
        assert_eq!(at(report, &["policy"]).as_str(), Some("band"), "{name}");
        assert_eq!(num(report, &["min_holding"]), num(s, &["min_holding"]), "{name}");
        assert_eq!(num(report, &["band_resolved"]).to_bits(), num(s, &["band_resolved"]).to_bits(), "{name}: priced as recorded");
    }
}

#[test]
fn ablation_score_prices_the_run_band_book() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    for (name, extra) in [("score.json", SCORE_BAND.to_vec()), ("sd.json", SD_BAND.to_vec())] {
        let run = search_as(&dir, &data, &spec, &extra, name);
        let doc = read_json(&run);
        let want = at(&doc, &["objective", "test"]);
        let s = at(&doc, &["objective", "spec"]);
        let from = ablation_score(&dir, &data, &windows, "--from-run", &run, &format!("from-{name}"));
        assert!((ablation_net(&from) - num(want, &["net_pnl"])).abs() <= 1e-12, "{name}: net {} vs {}", ablation_net(&from), num(want, &["net_pnl"]));
        let rule = at(&from, &["band_book"]);
        assert_eq!(num(rule, &["half_width"]).to_bits(), num(s, &["band_resolved"]).to_bits(), "{name}: priced as recorded");
        assert_eq!(num(rule, &["min_holding"]), num(s, &["min_holding"]), "{name}");

        let grid = ablation_score(&dir, &data, &windows, "--run", &run, &format!("run-{name}"));
        let rule = at(&grid, &["band_book"]);
        assert_eq!(num(rule, &["half_width"]).to_bits(), num(s, &["band_resolved"]).to_bits(), "{name}: --run band");
        assert_eq!(num(rule, &["min_holding"]), num(s, &["min_holding"]), "{name}: --run hold");
        let plain_run = dir.join(format!("plain-{name}"));
        let text = std::fs::read_to_string(&run).unwrap();
        std::fs::write(&plain_run, without_band_keys(&text).replace("\"min_holding\":3", "\"min_holding\":1").replace("\"min_holding\":4", "\"min_holding\":1")).unwrap();
        let plain = ablation_score(&dir, &data, &windows, "--run", &plain_run.to_string_lossy(), &format!("plain-{name}"));
        assert!(plain.get("band_book").is_none(), "{name}");
        assert_ne!(ablation_net(&grid).to_bits(), ablation_net(&plain).to_bits(), "{name}: the band changes the --run book");
    }
}

#[test]
fn band_does_not_depend_on_test_rows() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let other = write_csv(&dir, "d2.csv", |i| if i >= 310 { -row(i).2 } else { row(i).2 });
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let run = search_as(&dir, &data, &spec, &SD_BAND, "sd.json");
    let doc = read_json(&run);
    let resolved = num(at(&doc, &["objective", "spec"]), &["band_resolved"]);
    assert!(resolved > 0.0);
    let (a, b) = (ladder_sr(&dir, &data, &spec, &windows, &run, "a"), ladder_sr(&dir, &other, &spec, &windows, &run, "b"));
    assert_ne!(ladder_net(sr_row(&a)).to_bits(), ladder_net(sr_row(&b)).to_bits(), "the test rows must matter to the net");
    for lad in [&a, &b] {
        assert_eq!(num(at(lad, &["sr_run"]), &["band_resolved"]).to_bits(), resolved.to_bits());
        assert!(at(sr_row(lad), &["params"]).as_str().unwrap().contains(&format!("half-width {}", rcsr::devlog::num(resolved))));
    }
    let (c, d) = (
        ablation_score(&dir, &data, &windows, "--from-run", &run, "a"),
        ablation_score(&dir, &other, &windows, "--from-run", &run, "b"),
    );
    for abl in [&c, &d] {
        assert_eq!(num(at(abl, &["band_book"]), &["half_width"]).to_bits(), resolved.to_bits());
    }
}

#[test]
fn old_runs_keep_the_threshold_book_in_ladder_and_ablation() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let plain = search_as(&dir, &data, &spec, &[], "plain.json");
    let old_path = dir.join("old.json").to_string_lossy().into_owned();
    std::fs::write(&old_path, without_band_keys(&std::fs::read_to_string(&plain).unwrap())).unwrap();
    let (a, b) = (ladder_sr(&dir, &data, &spec, &windows, &plain, "plain"), ladder_sr(&dir, &data, &spec, &windows, &old_path, "old"));
    assert!(at(&a, &["sr_run"]).get("policy").is_none());
    assert_eq!(shape::timing_free_shape(at(&a, &["rungs"])), shape::timing_free_shape(at(&b, &["rungs"])));
    let (a, b) = (
        ablation_score(&dir, &data, &windows, "--from-run", &plain, "plain"),
        ablation_score(&dir, &data, &windows, "--from-run", &old_path, "old"),
    );
    assert!(a.get("band_book").is_none());
    assert_eq!(digits_stripped(&a), digits_stripped(&b));
    let (a, b) = (
        ablation_score(&dir, &data, &windows, "--run", &plain, "plain-run"),
        ablation_score(&dir, &data, &windows, "--run", &old_path, "old-run"),
    );
    assert_eq!(digits_stripped(&a), digits_stripped(&b));
}

#[test]
fn malformed_band_block_is_still_refused() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |i| row(i).2);
    let spec = write_spec(&dir, "s.json", "log", "20.0");
    let windows = write_windows(&dir);
    let run = search_as(&dir, &data, &spec, &SD_BAND, "sd.json");
    let broken = dir.join("broken.json").to_string_lossy().into_owned();
    let text = std::fs::read_to_string(&run).unwrap();
    std::fs::write(&broken, text.replace("\"band_resolved\":", "\"band_resolvd\":")).unwrap();
    let ladder = rcsr(&[
        "ladder", "--data", &data, "--target", "y", "--time", "t", "--windows", &windows, "--features", "x1,x2",
        "--purge-seconds", "900", "--target-spec", &spec, "--draws", "50", "--gbm-trees", "10", "--sr-run", &broken, "--out", "/dev/null",
    ]);
    assert_eq!(code(&ladder), 3, "{}", stderr(&ladder));
    assert!(stderr(&ladder).contains("band_resolved"), "{}", stderr(&ladder));
    for mode in ["--from-run", "--run"] {
        let out = rcsr(&[
            "ablation", "score", mode, &broken, "--data", &data, "--target", "y", "--time", "t", "--windows", &windows,
            "--purge-seconds", "900", "--out", "/dev/null",
        ]);
        assert_eq!(code(&out), 3, "{mode}: {}", stderr(&out));
        assert!(stderr(&out).contains("band_resolved"), "{mode}: {}", stderr(&out));
    }
}
