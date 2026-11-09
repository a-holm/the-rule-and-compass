//! `--book-target`: fit on one label, price the long/flat book on another (#929).

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::data::TradingSlice;
use rcsr::devlog::Json;
use rcsr::trading::{self, FitnessBook, TradingRows, TradingSpec};
use scratch::Scratch;
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 3600;
const N: usize = 700;
const DAY: usize = 24;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn x1(i: usize) -> f64 {
    frac(i as f64 * 0.618_033_988_749_894_9) - 0.5
}

fn r1(i: usize) -> f64 {
    0.004 * x1(i) + 0.0005 * (frac(i as f64 * 0.324_717_957_244_746) - 0.5)
}

/// Hourly rows; `y24` is the sum of the next 24 hourly returns (shorter at the tail).
fn write_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,x1,x2,r1,y1,r24,y24\n");
    for i in 0..N {
        let x2 = frac(i as f64 * 0.754_877_666_246_692_7) - 0.5;
        let r24: f64 = (i + 1..(i + 1 + DAY).min(N)).map(r1).sum();
        text.push_str(&format!("{},{},{},{},{},{},{}\n", T0 + STEP * i as i64, x1(i), x2, r1(i), r1(i), r24, r24));
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn entry(column: &str, return_column: &str, horizon: i64) -> String {
    format!(
        "{{\"column\":\"{column}\",\"name\":\"{column}\",\"kind\":\"forward_return\",\"deprecated\":false,\
         \"return_column\":\"{return_column}\",\"return_units\":\"log\",\"horizon_seconds\":{horizon},\
         \"entry_offset_seconds\":{STEP},\"delay_seconds\":0,\"entry_price\":\"next open\",\"exit_price\":\"close\",\
         \"offset\":0.0,\"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}"
    )
}

fn write_spec(dir: &std::path::Path) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{},{}]}}",
        entry("y1", "r1", STEP),
        entry("y24", "r24", STEP * DAY as i64)
    );
    let path = dir.join("s.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

const PNL: &[&str] = &[
    "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--test-frac", "0.2", "--threshold", "1e-30",
    "--min-trades", "0", "--fitness", "pnl",
];

fn run(data: &str, spec: &str, target: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", target, "--vars", "x1,x2", "--time", "t", "--split", "temporal", "--target-spec", spec,
        "--cpu", "--no-dev-log", "--seed", "3", "--generations", "12", "--pop", "32", "--metric", "squared", "--json",
        "--quiet",
    ];
    args.extend_from_slice(PNL);
    args.extend_from_slice(extra);
    rcsr_cmd::rcsr().args(&args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json(o: &Output) -> Json {
    let text = String::from_utf8_lossy(&o.stdout);
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}\n{}", stderr(o)))
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

fn ok(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    json(o)
}

fn usage_error(o: &Output, needle: &str) {
    assert_eq!(code(o), 2, "{}", stderr(o));
    assert!(stderr(o).contains(needle), "want '{needle}' in: {}", stderr(o));
}

/// The report without the members that vary from run to run.
fn stable(j: &Json) -> Json {
    match j {
        Json::Obj(members) => Json::Obj(
            members
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "elapsed_seconds" | "seconds" | "timing" | "gpu"))
                .map(|(k, v)| (k.clone(), stable(v)))
                .collect(),
        ),
        Json::Arr(items) => Json::Arr(items.iter().map(stable).collect()),
        other => other.clone(),
    }
}

#[test]
fn book_target_same_column_is_identity() {
    let scratch = Scratch::new("book_target_identity");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let plain = ok(&run(&data, &spec, "y1", &[]));
    let named = ok(&run(&data, &spec, "y1", &["--book-target", "y1"]));
    assert_eq!(stable(&plain), stable(&named));
    assert!(at(&named, &["objective"]).get("book_target").is_none());
    assert!(at(&named, &["search_fitness"]).get("book_target").is_none());
}

#[test]
fn book_target_allows_hourly_book_on_daily_fit() {
    let scratch = Scratch::new("book_target_hourly");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    usage_error(&run(&data, &spec, "y24", &[]), "overlapping returns");
    let j = ok(&run(&data, &spec, "y24", &["--book-target", "y1"]));
    assert_eq!(at(&j, &["objective", "book_target"]).as_str(), Some("y1"));
    assert_eq!(at(&j, &["search_fitness", "book_target"]).as_str(), Some("y1"));
    assert_eq!(num(&j, &["objective", "spec", "horizon_seconds"]), STEP as f64);
    assert_eq!(num(&j, &["objective", "spec", "min_group_step_seconds"]), STEP as f64);
    assert_eq!(at(&j, &["target"]).as_str(), Some("y24"));
}

#[test]
fn book_target_purge_is_the_larger_of_the_two_specs() {
    let scratch = Scratch::new("book_target_purge");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let day_purge = (STEP + STEP * DAY as i64) as f64;
    let both = ok(&run(&data, &spec, "y24", &["--book-target", "y1"]));
    assert_eq!(num(&both, &["split", "required_purge_seconds"]), day_purge);
    assert!(num(&both, &["split", "validation_purge_seconds"]) >= day_purge);
    assert!(num(&both, &["split", "purge_seconds"]) >= day_purge);
    let hourly = ok(&run(&data, &spec, "y1", &[]));
    assert_eq!(num(&hourly, &["split", "required_purge_seconds"]), (2 * STEP) as f64);
    assert!(num(&hourly, &["split", "validation_purge_seconds"]) < day_purge);
}

#[test]
fn no_training_label_window_reaches_the_test_block() {
    let scratch = Scratch::new("book_target_leakage");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let j = ok(&run(&data, &spec, "y24", &["--book-target", "y1"]));
    let train = num(&j, &["train_rows"]) as usize;
    let test = num(&j, &["test_rows"]) as usize;
    assert!(train > 0 && test > 0);
    let test_start = T0 + STEP * (N - test) as i64;
    let last_train = T0 + STEP * (train - 1) as i64;
    for (label, horizon) in [("fit y24", STEP * DAY as i64), ("book y1", STEP)] {
        let window_end = last_train + STEP + horizon;
        assert!(window_end < test_start, "{label}: window ends {window_end}, test starts {test_start}");
    }
    let book_rows = num(&j, &["search_fitness", "rows"]) as usize;
    assert_eq!(book_rows, train);
}

#[test]
fn the_fitness_book_equals_the_exact_book_on_the_book_returns() {
    let scratch = Scratch::new("book_target_exact");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let j = ok(&run(&data, &spec, "y24", &["--book-target", "y1"]));
    let rows = num(&j, &["search_fitness", "rows"]) as usize;
    let mut slice = TradingSlice { rows: TradingRows::default(), ret: Vec::new() };
    for i in 0..rows {
        slice.rows.group.push(0);
        slice.rows.time.push((T0 + STEP * i as i64) as f64);
        slice.ret.push(r1(i).exp_m1());
    }
    let sp = TradingSpec { cost_bps: 5.0, threshold: 1e-30, min_trades: 0, ..Default::default() };
    let book = FitnessBook::new(&slice, &sp, STEP as f64).unwrap();
    assert_eq!(at(&j, &["search_fitness", "book_hash"]).as_str(), Some(format!("{:016x}", book.hash()).as_str()));
    let pred: Vec<f64> = (0..rows).map(|i| frac(i as f64 * 0.754_877_666_246_692_7) - 0.4).collect();
    let totals = book.totals(&pred);
    let exact = trading::evaluate(&pred, &slice.ret, &slice.rows, &sp).unwrap();
    let net = book.net_of(&totals);
    assert!((net - exact.net_pnl).abs() <= 1e-12 * exact.net_pnl.abs().max(1.0), "{net} vs {}", exact.net_pnl);
}

#[test]
fn book_target_refusals_are_usage_errors() {
    let scratch = Scratch::new("book_target_refusals");
    let data = write_csv(scratch.path());
    let spec = write_spec(scratch.path());
    usage_error(&run(&data, &spec, "y24", &["--book-target", "nope"]), "no entry for target column 'nope'");
    let no_objective = rcsr_cmd::rcsr()
        .args([
            "--data", data.as_str(), "--target", "y24", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", spec.as_str(), "--book-target", "y1", "--cpu", "--no-dev-log", "--generations", "1",
            "--pop", "8", "--json", "--quiet",
        ])
        .output()
        .unwrap();
    usage_error(&no_objective, "--book-target needs --objective");
}

const STOP: usize = 200;

fn r1_group(i: usize, g: usize) -> f64 {
    r1(i) + 0.0003 * g as f64
}

/// Two groups on one hourly clock. Group 1 ends at `STOP`. The last `DAY` rows of each group have a
/// NaN `y24` and `r24` (the daily window is incomplete) and a finite `y1` and `r1`.
fn write_panel_csv(dir: &std::path::Path, drop_nan_label_rows: bool) -> String {
    let mut text = String::from("t,g,x1,x2,r1,y1,r24,y24\n");
    for i in 0..N {
        for g in 0..2usize {
            let len = if g == 0 { N } else { STOP };
            if i >= len {
                continue;
            }
            let complete = i + DAY < len;
            if drop_nan_label_rows && !complete {
                continue;
            }
            let x2 = frac(i as f64 * 0.754_877_666_246_692_7) - 0.5;
            let r24: f64 = (i + 1..(i + 1 + DAY).min(len)).map(|j| r1_group(j, g)).sum();
            let (r24, y24) = if complete { (r24.to_string(), r24.to_string()) } else { ("nan".to_string(), "nan".to_string()) };
            let r = r1_group(i, g);
            text.push_str(&format!("{},g{},{},{},{},{},{},{}\n", T0 + STEP * i as i64, g, x1(i), x2, r, r, r24, y24));
        }
    }
    let path = dir.join(if drop_nan_label_rows { "p_fit.csv" } else { "p.csv" });
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run_panel(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec!["--group-by", "g", "--fill", "drop"];
    args.extend_from_slice(extra);
    run(data, spec, "y24", &args)
}

#[test]
fn book_target_keeps_rows_with_nan_fit_label() {
    let scratch = Scratch::new("book_target_nan_fit_rows");
    let data = write_panel_csv(scratch.path(), false);
    let spec = write_spec(scratch.path());
    let j = ok(&run_panel(&data, &spec, &["--book-target", "y1"]));
    let fit_rows = num(&j, &["train_rows"]) as usize;
    let book_rows = num(&j, &["search_fitness", "rows"]) as usize;
    assert_eq!(num(&j, &["search_fitness", "groups"]) as usize, 2);
    assert!(fit_rows > STOP, "group 1 ends inside the training block");
    assert_eq!(book_rows, fit_rows + DAY, "the DAY NaN-label rows of group 1 stay in the book");
}

#[test]
fn book_target_book_rows_include_nan_fit_rows() {
    let scratch = Scratch::new("book_target_nan_fit_book");
    let data = write_panel_csv(scratch.path(), false);
    let spec = write_spec(scratch.path());
    let j = ok(&run_panel(&data, &spec, &["--book-target", "y1"]));
    let rows = num(&j, &["search_fitness", "rows"]) as usize;
    let long = rows - STOP;
    let mut slice = TradingSlice { rows: TradingRows::default(), ret: Vec::new() };
    for (g, len) in [(0usize, long), (1, STOP)] {
        for i in 0..len {
            slice.rows.group.push(g as u64);
            slice.rows.time.push((T0 + STEP * i as i64) as f64);
            slice.ret.push(r1_group(i, g).exp_m1());
        }
    }
    let sp = TradingSpec { cost_bps: 5.0, threshold: 1e-30, min_trades: 0, ..Default::default() };
    let book = FitnessBook::new(&slice, &sp, STEP as f64).unwrap();
    assert_eq!(at(&j, &["search_fitness", "book_hash"]).as_str(), Some(format!("{:016x}", book.hash()).as_str()));
    let pred: Vec<f64> = (0..slice.ret.len()).map(|i| frac(i as f64 * 0.754_877_666_246_692_7) - 0.4).collect();
    let net = book.net_of(&book.totals(&pred));
    let exact = trading::evaluate(&pred, &slice.ret, &slice.rows, &sp).unwrap();
    assert!((net - exact.net_pnl).abs() <= 1e-12 * exact.net_pnl.abs().max(1.0), "{net} vs {}", exact.net_pnl);
}

#[test]
fn no_training_label_window_reaches_the_test_block_with_nan_fit_tails() {
    let scratch = Scratch::new("book_target_nan_leakage");
    let data = write_panel_csv(scratch.path(), false);
    let spec = write_spec(scratch.path());
    let j = ok(&run_panel(&data, &spec, &["--book-target", "y1"]));
    let day_purge = (STEP + STEP * DAY as i64) as f64;
    assert_eq!(num(&j, &["split", "required_purge_seconds"]), day_purge);
    assert!(num(&j, &["split", "purge_seconds"]) >= day_purge, "every training row, with or without a fit label");
    assert!(num(&j, &["split", "validation_purge_seconds"]) >= day_purge);
}

fn fit_split_flags() -> Vec<String> {
    let train = format!("t < {}", T0 + STEP * 400);
    let test = format!("t >= {}", T0 + STEP * 450);
    ["--validation-frac", "0.0105", "--train-select", train.as_str(), "--test-select", test.as_str()]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The panel with NaN-label rows, and the same panel with those rows deleted from the file.
fn run_nan_and_prefiltered(extra: &[&str], manifest_name: &str) -> (Json, Json, Json, Json) {
    let scratch = Scratch::new(manifest_name);
    let spec = write_spec(scratch.path());
    let with_nan = write_panel_csv(scratch.path(), false);
    let prefiltered = write_panel_csv(scratch.path(), true);
    let mut flags = fit_split_flags();
    flags.extend(extra.iter().map(|s| s.to_string()));
    let mut results = Vec::new();
    for (name, data) in [("nan", &with_nan), ("fit", &prefiltered)] {
        let manifest = scratch.path().join(format!("{name}.manifest.json")).to_string_lossy().into_owned();
        let mut args: Vec<&str> = vec!["--book-target", "y1", "--run-manifest", manifest.as_str()];
        args.extend(flags.iter().map(|s| s.as_str()));
        let j = ok(&run_panel(data, &spec, &args));
        let m = Json::parse(std::fs::read_to_string(&manifest).unwrap().trim()).expect("manifest is json");
        results.push((j, m));
    }
    let (fit_run, fit_manifest) = results.pop().unwrap();
    let (nan_run, nan_manifest) = results.pop().unwrap();
    (nan_run, nan_manifest, fit_run, fit_manifest)
}

#[test]
fn book_target_fit_uses_finite_fit_rows_only() {
    let (nan, _, fit, _) = run_nan_and_prefiltered(
        &["--generations", "0", "--pop", "1", "--seed-expr", "0.5 * x1 + 0.1 * x2 + 0.01"],
        "book_target_fit_rows_constants",
    );
    for path in [&["train_rows"][..], &["split", "validation_rows"][..], &["test_rows"][..]] {
        assert_eq!(num(&nan, path), num(&fit, path), "{path:?}: the fit rows are the same");
    }
    assert_eq!(
        num(&nan, &["search_fitness", "rows"]) as usize,
        num(&fit, &["search_fitness", "rows"]) as usize + DAY,
        "the book keeps the NaN-label rows"
    );
    let expression = |j: &Json| at(j, &["objective", "finalist", "expression_exact"]).as_str().unwrap().to_string();
    assert!(!expression(&nan).contains("NaN"), "{}", expression(&nan));
    assert_eq!(expression(&nan), expression(&fit), "constants equal those of the pre-filtered fit panel");
}

#[test]
fn book_target_fit_rows_subsample_is_finite_only() {
    let (nan, nan_manifest, fit, fit_manifest) = run_nan_and_prefiltered(
        &["--generations", "6", "--pop", "16", "--fit-rows", "100"],
        "book_target_fit_rows_subsample",
    );
    let sub = |m: &Json| at(m, &["fit_subsample"]).clone();
    assert_eq!(num(&sub(&nan_manifest), &["fit_rows"]), 100.0);
    assert_eq!(
        at(&sub(&nan_manifest), &["hash"]).as_str(),
        at(&sub(&fit_manifest), &["hash"]).as_str(),
        "the draw is over the finite-label rows, the same indices as on the pre-filtered panel"
    );
    assert!(num(&nan, &["search_fitness", "rows"]) > num(&fit, &["search_fitness", "rows"]));
}

#[test]
fn book_target_without_return_column_is_refused() {
    let scratch = Scratch::new("book_target_no_return_column");
    let data = write_csv(scratch.path());
    let no_return = entry("y24", "r24", STEP * DAY as i64).replace("\"return_column\":\"r24\"", "\"return_column\":null");
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{},{}]}}",
        entry("y1", "r1", STEP),
        no_return
    );
    let spec = scratch.path().join("no_return.json");
    std::fs::write(&spec, text).unwrap();
    let spec = spec.to_string_lossy().into_owned();
    let o = run(&data, &spec, "y1", &["--book-target", "y24"]);
    usage_error(&o, "book target 'y24'");
    assert!(stderr(&o).contains("return_column"), "{}", stderr(&o));
}
