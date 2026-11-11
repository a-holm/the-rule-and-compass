//! #1016: the walk-forward period is one label horizon wide; the annualisation
//! follows from it, not from the spacing of unlabelled rows.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::data;
use rcsr::devlog::Json;
use rcsr::walkforward::{self, EvalSpec, Window};
use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const T0: f64 = 1_609_459_200.0;
const HOUR: f64 = 3600.0;
const DAY: f64 = 86_400.0;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn feature(i: usize, g: usize) -> f64 {
    frac(i as f64 * 0.618_033_988_749_894_9 + g as f64 * 0.37)
}

fn ret(i: usize, g: usize) -> f64 {
    0.01 * (frac(i as f64 * 0.754_877_666_246_692_7 + g as f64 * 0.31) - 0.5)
}

fn window(a: f64, b: f64, c: f64, d: f64) -> Window {
    Window { label: "w".to_string(), train_start: a, train_end: b, test_start: c, test_end: d }
}

fn spec(horizon: f64, windows: Vec<Window>) -> EvalSpec {
    EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "y".to_string(),
        time: "ts".to_string(),
        group: Some("sym".to_string()),
        windows,
        horizon,
        embargo: None,
        cost_bps: 5.0,
        threshold_quantile: 0.5,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: None,
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

/// `(time, group, x, y)` rows as a table; `y` of `None` is written as `nan`.
fn table(rows: &[(f64, usize, f64, Option<f64>)]) -> data::Table {
    let mut s = String::from("ts,sym,x,y\n");
    for (t, g, x, y) in rows {
        let y = y.map_or("nan".to_string(), |v| format!("{v}"));
        s.push_str(&format!("{t},s{g},{x},{y}\n"));
    }
    data::parse_table(&s).unwrap()
}

fn pooled(report: &walkforward::Report) -> &walkforward::Metrics {
    &report.pooled
}

fn json_of(report: &walkforward::Report) -> Json {
    Json::parse(report.json().trim()).expect("report json")
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

/// Two groups, every group on every hour, shared timestamps.
fn hourly_panel() -> data::Table {
    let mut rows = Vec::new();
    for i in 0..160usize {
        for g in 0..2usize {
            rows.push((T0 + HOUR * i as f64, g, feature(i, g), Some(ret(i, g))));
        }
    }
    table(&rows)
}

/// Three groups, hourly rows, a finite target only on the shared day stamps.
fn anchored_day_panel() -> data::Table {
    let mut rows = Vec::new();
    for i in 0..(24 * 40usize) {
        for g in 0..3usize {
            let y = (i % 24 == 0).then(|| ret(i / 24, g));
            rows.push((T0 + HOUR * i as f64, g, feature(i / 24, g), y));
        }
    }
    table(&rows)
}

/// Three groups, one row per day each, at phases 0 h, 7 h and 15 h.
fn staggered_day_panel() -> data::Table {
    let mut rows = Vec::new();
    for d in 0..30usize {
        for (g, phase) in [0.0, 7.0, 15.0].into_iter().enumerate() {
            rows.push((T0 + DAY * d as f64 + HOUR * phase, g, feature(d, g), Some(ret(d, g))));
        }
    }
    table(&rows)
}

#[test]
fn wf_h1_unchanged() {
    let t = hourly_panel();
    let s = spec(HOUR, vec![window(T0, T0 + 79.0 * HOUR, T0 + 80.0 * HOUR, T0 + 160.0 * HOUR)]);
    let report = walkforward::run_table(&t, &s, "test").unwrap();
    let p = pooled(&report);
    assert_eq!(p.mean_return.to_bits(), 0xbf3c9c6e80daf6a6);
    assert_eq!(p.total_return.to_bits(), 0xbfa19e1a8f4aa020);
    assert_eq!(p.max_drawdown.to_bits(), 0xbfa3a05ec3aa6560);
    assert_eq!(p.sharpe.to_bits(), 0xc03c5c22ca523a05);
    assert_eq!(p.periods, 80);
    assert_eq!(p.periods_per_year, 8760.0);
    let j = json_of(&report);
    assert_eq!(at(&j, &["annualization", "source"]).as_str(), Some("horizon_flag"));
    assert_eq!(num(&j, &["annualization", "period_seconds"]), 3600.0);
}

#[test]
fn wf_ppy_anchored_day_panel() {
    let t = anchored_day_panel();
    let s = spec(DAY, vec![window(T0, T0 + 14.0 * DAY, T0 + 15.0 * DAY, T0 + 40.0 * DAY)]);
    let report = walkforward::run_table(&t, &s, "test").unwrap();
    let p = pooled(&report);
    assert_eq!(p.mean_return.to_bits(), 0xbf31b5273b20948b);
    assert_eq!(p.total_return.to_bits(), 0xbf7ba3d7e7ecf800);
    assert_eq!(p.max_drawdown.to_bits(), 0xbf8030cda78cc040);
    assert_eq!(p.periods, 25);
    assert_eq!(p.periods_per_year, 365.0);
    let j = json_of(&report);
    assert_eq!(num(&j, &["pooled", "periods_per_year"]), 365.0);
    assert_eq!(at(&j, &["annualization", "source"]).as_str(), Some("horizon_flag"));
}

#[test]
fn wf_stagger_buckets_are_daily() {
    let t = staggered_day_panel();
    let mut s = spec(DAY, vec![window(T0, T0 + 9.0 * DAY, T0 + 10.0 * DAY, T0 + 30.0 * DAY)]);
    s.cost_bps = 0.0;
    s.threshold_fixed = Some(-1.0e9);
    let report = walkforward::run_table(&t, &s, "test").unwrap();
    let p = pooled(&report);
    let mut want = 1.0f64;
    for d in 10..30usize {
        let mean: f64 = (0..3usize).map(|g| ret(d, g)).sum::<f64>() / 3.0;
        want *= 1.0 + mean;
    }
    assert_eq!(p.periods, 20, "one period per UTC day");
    assert_eq!(p.periods_per_year, 365.0);
    assert!((p.total_return - (want - 1.0)).abs() < 1e-12, "total_return {} want {}", p.total_return, want - 1.0);
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

fn write_spec(dir: &Path, horizon_seconds: u64) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"y\",\"return_units\":\"simple\",\
         \"horizon_seconds\":{horizon_seconds},\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"simple_return\",\
         \"cost_bps\":0.0,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}]}}"
    );
    let path = dir.join(format!("spec{horizon_seconds}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// Two groups, `n` rows each, spaced by `step` seconds, shared timestamps.
fn write_csv(dir: &Path, step: u64, n: usize) -> String {
    let mut text = String::from("t,sym,x,y\n");
    for i in 0..n {
        for g in 0..2usize {
            let t = T0 as u64 + step * i as u64;
            text.push_str(&format!("{t},s{g},{},{}\n", feature(i, g), ret(i, g)));
        }
    }
    let path = dir.join(format!("d{step}.csv"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run_cli(dir: &Path, step: u64, extra: &[&str]) -> (Output, String) {
    let n = 60usize;
    let data = write_csv(dir, step, n);
    let spec = write_spec(dir, step);
    let out_path = dir.join(format!("wf{step}.json")).to_string_lossy().into_owned();
    let w = format!(
        "w,{},{},{},{}",
        T0 as u64,
        T0 as u64 + step * 19,
        T0 as u64 + step * 20,
        T0 as u64 + step * n as u64
    );
    let mut args = vec![
        "walkforward", "--data", &data, "--time", "t", "--group-by", "sym", "--target", "y", "--target-spec", &spec,
        "--expr", "x", "--vars", "x", "--window", &w, "--threshold-quantile", "0.5", "--out", &out_path,
    ];
    args.extend_from_slice(extra);
    let out = rcsr(&args);
    (out, out_path)
}

fn read_json(path: &str) -> Json {
    Json::parse(std::fs::read_to_string(path).unwrap().trim()).expect("report json")
}

#[test]
fn wf_ppy_anchored_horizons() {
    let scratch = Scratch::new("wf-period-anchored");
    let dir = scratch.path().to_path_buf();
    for (step, want) in [(86_400u64, 365.0), (259_200, 365.0 * 86_400.0 / 259_200.0), (604_800, 52.142_857_142_857_146), (1_209_600, 26.071_428_571_428_573)] {
        let (out, path) = run_cli(&dir, step, &[]);
        assert_eq!(code(&out), 0, "{}", stderr(&out));
        let j = read_json(&path);
        assert!((num(&j, &["pooled", "periods_per_year"]) - want).abs() < 1e-9, "step {step}");
        assert_eq!(at(&j, &["annualization", "source"]).as_str(), Some("target_spec"));
        assert_eq!(num(&j, &["annualization", "period_seconds"]), step as f64);
    }
}

#[test]
fn wf_ppy_explicit_mismatch_refused() {
    let scratch = Scratch::new("wf-period-mismatch");
    let (out, _) = run_cli(scratch.path(), 604_800, &["--periods-per-year", "8760"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("--periods-per-year 8760"), "{}", stderr(&out));
}

#[test]
fn wf_ppy_explicit_match_ok() {
    let scratch = Scratch::new("wf-period-match");
    let (out, path) = run_cli(scratch.path(), 604_800, &["--periods-per-year", "52.142857142857146"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = read_json(&path);
    assert_eq!(at(&j, &["annualization", "source"]).as_str(), Some("target_spec"));
}

#[test]
fn wf_block_len_one_period() {
    let scratch = Scratch::new("wf-period-block");
    let dir = scratch.path().to_path_buf();
    let (out, path) = run_cli(&dir, 604_800, &["--bootstrap", "200"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = read_json(&path);
    let periods = num(&j, &["pooled", "periods"]);
    let block = num(&j, &["inference", "bootstrap", "block_len"]);
    assert!(block < periods / 2.0, "block {block} of {periods} periods");
    let ci = at(&j, &["inference", "bootstrap", "mean_ci"]);
    assert!(num(ci, &["hi"]) - num(ci, &["lo"]) > 0.0, "degenerate interval");
    // the horizon is one period, so a block of the holding period is accepted
    let (out, path) = run_cli(&dir, 604_800, &["--bootstrap", "200", "--block-len", "2"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert_eq!(num(&read_json(&path), &["inference", "bootstrap", "block_len"]), 2.0);
}

#[test]
fn wf_daily_block_by_period() {
    let scratch = Scratch::new("wf-period-daily");
    let dir = scratch.path().to_path_buf();
    for step in [604_800u64, 86_400] {
        let (out, path) = run_cli(&dir, step, &[]);
        assert_eq!(code(&out), 0, "{}", stderr(&out));
        let j = read_json(&path);
        assert!(matches!(at(&j, &["economics", "daily"]), Json::Null), "step {step}");
        assert_eq!(at(&j, &["economics", "daily_unavailable"]).as_str(), Some("period >= 1 day"));
    }
    let (out, path) = run_cli(&dir, 3600, &[]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = read_json(&path);
    assert_eq!(num(&j, &["economics", "daily", "periods_per_year"]), 365.0);
}
