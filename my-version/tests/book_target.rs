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
    let no_return = entry("y1", "r1", STEP).replace("\"return_column\":\"r1\"", "\"return_column\":null");
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{},{}]}}",
        no_return,
        entry("y24", "r24", STEP * DAY as i64)
    );
    let spec = scratch.path().join("no_return.json");
    std::fs::write(&spec, text).unwrap();
    let spec = spec.to_string_lossy().into_owned();
    let o = run(&data, &spec, "y24", &["--book-target", "y1"]);
    usage_error(&o, "book target 'y1'");
    assert!(stderr(&o).contains("return_column"), "{}", stderr(&o));
}

/// #971: the device search with book rows that have no fit label.
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::checkpoint::Backend;
    use rcsr::data::Sample;
    use rcsr::metrics::Metric;
    use rcsr::search::{self, Config, Fitness};
    use std::sync::Arc;

    const GROUPS: usize = 4;
    const LEN: usize = 600;

    fn device_present() -> bool {
        let present = rcsr::gpu::probe().available();
        if !present {
            rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        }
        present
    }

    /// Four hourly groups. The last `DAY` rows of each group and every 37th row have a NaN fit label
    /// and a finite book return. Returns the book sample, its fit rows and the book.
    fn book_rows() -> (Sample, Sample, Arc<FitnessBook>) {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        let (mut fx, mut fy) = (Vec::new(), Vec::new());
        let mut slice = TradingSlice { rows: TradingRows::default(), ret: Vec::new() };
        for g in 0..GROUPS {
            for i in 0..LEN {
                let k = g * LEN + i;
                let row = vec![x1(k), frac(k as f64 * 0.754_877_666_246_692_7) - 0.5];
                let label = 0.6 * row[0] - 0.2 * row[1] + 0.01 * (k as f64 * 1.3).sin();
                if i + DAY < LEN && k % 37 != 0 {
                    fx.push(row.clone());
                    fy.push(label);
                    y.push(label);
                } else {
                    y.push(f64::NAN);
                }
                x.push(row);
                slice.rows.group.push(g as u64);
                slice.rows.time.push((T0 + STEP * i as i64) as f64);
                slice.ret.push(r1(k) + 0.0003 * g as f64);
            }
        }
        let sp = TradingSpec { cost_bps: 5.0, threshold: 1e-30, min_trades: 0, ..Default::default() };
        let book = FitnessBook::new(&slice, &sp, STEP as f64).expect("a profitable book");
        let sample = |x, y| Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None };
        (sample(x, y), sample(fx, fy), Arc::new(book))
    }

    fn config(fit: &Sample, book: &Arc<FitnessBook>, pipeline_depth: usize, fit_rows: usize) -> Config {
        Config {
            generations: 4,
            pop: 32,
            islands: 2,
            seed: 7,
            metric: Metric::Squared,
            fitness: Fitness::LongPnl(book.clone()),
            fit_view: Some(Arc::new(fit.clone())),
            gpu: true,
            gpu_notes: false,
            dispatch: rcsr::dispatch::DispatchMode::Device,
            fit_strategy: rcsr::fit::FitStrategy::Gpu,
            fit_admission: rcsr::fit::FitAdmission::All,
            pipeline_depth,
            fit_rows,
            ..Default::default()
        }
    }

    /// The device fitted constants and read no row without a fit label: such a row would
    /// make the device fit non-finite.
    fn assert_device_fit(out: &search::Outcome, fit_rows: usize, score_rows: usize) {
        assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{:?}", out.backend_reason);
        let g = out.gpu.as_ref().expect("the GPU report");
        assert!(g.fit_gpu > 0, "the device fitted constants: {:?}", g.fit_fallback_by_cause);
        assert!(!g.fit_fallback_by_cause.contains_key("non_finite"), "{:?}", g.fit_fallback_by_cause);
        assert_eq!(g.fit_rows, fit_rows, "rows of the device fit");
        assert_eq!(g.score_rows, score_rows, "the device scores every book row");
    }

    #[test]
    fn label_free_book_rows_keep_the_device_search() {
        let (train, fit, book) = book_rows();
        assert!(fit.len() < train.len());
        let cfg = config(&fit, &book, 0, 0);
        assert_eq!(search::backend_blocker_reason(&cfg, &train), None);
        assert_eq!(search::effective_backend(&cfg, &train), Backend::Gpu);
        if !device_present() {
            return;
        }
        for depth in [0, 2] {
            let out = search::run(&config(&fit, &book, depth, 0), &train, &fit);
            assert_device_fit(&out, fit.len(), train.len());
            assert_eq!(out.gpu.as_ref().unwrap().fit_subsample_hash, None, "depth {depth}");
            for s in &out.archive.items {
                let pred: Vec<f64> = train.x.iter().map(|r| rcsr::expr::eval(&s.tree, r)).collect();
                assert_eq!(s.error.to_bits(), book.regret(&pred).to_bits(), "depth {depth}: the book prices every book row");
            }
        }
    }

    #[test]
    fn device_fit_rows_draw_only_fit_rows_as_the_cpu_draw() {
        let (train, fit, book) = book_rows();
        let cfg = config(&fit, &book, 0, 300);
        let record = search::fit_subsample_record(&cfg, &train).expect("a subsample");
        let on_fit_rows = search::fit_subsample_record(&Config { fit_view: None, ..cfg.clone() }, &fit).expect("a subsample");
        assert_eq!((record.0, record.1), (on_fit_rows.0, on_fit_rows.1), "the draw is over the fit rows");
        assert!(record.2, "the device fits the same drawn rows for the whole run");
        if !device_present() {
            return;
        }
        let out = search::run(&cfg, &train, &fit);
        assert_device_fit(&out, 300, train.len());
        assert_eq!(out.gpu.as_ref().unwrap().fit_subsample_hash, Some(record.1));
    }

    fn run_device(data: &str, spec: &str, extra: &[&str]) -> Output {
        let mut args = vec![
            "--data", data, "--target", "y24", "--book-target", "y1", "--vars", "x1,x2", "--time", "t", "--split",
            "temporal", "--target-spec", spec, "--group-by", "g", "--fill", "drop", "--no-dev-log", "--seed", "3",
            "--metric", "squared", "--json", "--quiet",
        ];
        args.extend_from_slice(PNL);
        args.extend_from_slice(extra);
        rcsr_cmd::rcsr_for(&args).args(&args).output().expect("run rcsr")
    }

    fn constants(j: &Json) -> Vec<f64> {
        let exact = at(j, &["objective", "finalist", "expression_exact"]).as_str().unwrap();
        let tree = rcsr::parse::parse(exact, &["x1".to_string(), "x2".to_string()]).unwrap_or_else(|e| panic!("{exact}: {e:?}"));
        fn walk(n: &rcsr::expr::Node, out: &mut Vec<f64>) {
            if n.op == rcsr::expr::Op::Const {
                out.push(n.val);
            }
            n.kids.iter().for_each(|k| walk(k, out));
        }
        let mut out = Vec::new();
        walk(&tree, &mut out);
        out
    }

    #[test]
    fn cli_device_search_prices_label_free_book_rows() {
        if !device_present() {
            return;
        }
        let scratch = Scratch::new("book_target_device_cli");
        let data = write_panel_csv(scratch.path(), false);
        let spec = write_spec(scratch.path());
        let seed = ["--generations", "0", "--pop", "1", "--seed-expr", "0.5 * x1 + 0.1 * x2 + 0.01"];
        let cpu_out = run_device(&data, &spec, &[&["--cpu"][..], &seed].concat());
        let device_out = run_device(&data, &spec, &[&["--gpu", "--dispatch", "device", "--fit-strategy", "gpu"][..], &seed].concat());
        for o in [&cpu_out, &device_out] {
            assert!(!stderr(o).contains("without a fit label"), "{}", stderr(o));
            assert!(!String::from_utf8_lossy(&o.stdout).contains("without a fit label"));
        }
        let cpu = ok(&cpu_out);
        let device = ok(&device_out);
        assert_eq!(at(&device, &["backend"]).as_str(), Some("gpu"), "{}", stderr(&device_out));
        let rows = num(&device, &["search_fitness", "rows"]);
        assert_eq!(rows, num(&cpu, &["search_fitness", "rows"]));
        assert_eq!(rows as usize, num(&device, &["train_rows"]) as usize + DAY, "the book keeps the NaN-label rows");
        assert_eq!(at(&device, &["search_fitness", "book_hash"]).as_str(), at(&cpu, &["search_fitness", "book_hash"]).as_str());
        assert_eq!(num(&device, &["gpu", "fit", "fit_rows"]), num(&device, &["train_rows"]), "the device fits the fit rows");
        assert_eq!(num(&device, &["gpu", "fit", "score_rows"]), rows, "the device scores the book rows");
        assert!(num(&device, &["gpu", "fit", "attempts"]) >= 1.0, "the seed copy reached the device fit stage");
        let (c, d) = (constants(&cpu), constants(&device));
        assert_eq!(c.len(), d.len());
        for (a, b) in d.iter().zip(&c) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "device constant {a} vs CPU {b}");
        }
        for block in ["validation", "test"] {
            for field in ["rows", "net_pnl", "gross_pnl", "trades", "exposure"] {
                assert_eq!(num(&device, &["objective", block, field]), num(&cpu, &["objective", block, field]), "{block}.{field}");
            }
        }
        assert!(num(&device, &["objective", "test", "rows"]) > num(&device, &["test_rows"]), "the test book keeps the NaN-label rows");

        // A search: the device fits constants on the fit rows; a fit that read a row without a
        // fit label would fail for every candidate.
        let search = ok(&run_device(&data, &spec, &["--gpu", "--dispatch", "device", "--fit-strategy", "gpu", "--generations", "3", "--pop", "24"]));
        assert_eq!(at(&search, &["backend"]).as_str(), Some("gpu"));
        assert!(num(&search, &["gpu", "fit", "gpu"]) > 0.0, "the device fitted constants");
        assert_eq!(num(&search, &["gpu", "fit", "fit_rows"]), num(&search, &["train_rows"]));
        assert_eq!(num(&search, &["gpu", "fit", "score_rows"]), num(&search, &["search_fitness", "rows"]));
    }
}

const STRIDE_N: usize = 2400;

/// `y24` is finite on every `DAY`-th row only; `y1` is finite on every row.
fn write_stride_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,x1,x2,r1,y1,r24,y24\n");
    for i in 0..STRIDE_N {
        let x2 = frac(i as f64 * 0.754_877_666_246_692_7) - 0.5;
        let r24: f64 = (i + 1..i + DAY).take_while(|&j| j < STRIDE_N).map(r1).sum();
        let t = T0 + STEP * i as i64;
        if i % DAY == 0 && i + DAY < STRIDE_N {
            text.push_str(&format!("{t},{},{x2},{},{},{r24},{r24}\n", x1(i), r1(i), r1(i)));
        } else {
            text.push_str(&format!("{t},{},{x2},{},{},nan,nan\n", x1(i), r1(i), r1(i)));
        }
    }
    let path = dir.join("stride.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run_stride(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec!["--fill", "drop", "--book-target", "y1"];
    args.extend_from_slice(extra);
    run(data, spec, "y24", &args)
}

fn pareto(j: &Json) -> &[Json] {
    match at(j, &["pareto"]) {
        Json::Arr(items) => items,
        _ => panic!("pareto is not an array"),
    }
}

#[test]
fn book_target_member_fitness_is_on_the_book_rows() {
    let scratch = Scratch::new("book_target_member_fitness");
    let data = write_stride_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let j = ok(&run_stride(&data, &spec, &[]));
    let rows = num(&j, &["search_fitness", "rows"]) as usize;
    assert!(rows > num(&j, &["train_rows"]) as usize, "the book holds rows without a fit label");
    let mut slice = TradingSlice { rows: TradingRows::default(), ret: Vec::new() };
    for i in 0..rows {
        slice.rows.group.push(0);
        slice.rows.time.push((T0 + STEP * i as i64) as f64);
        slice.ret.push(r1(i).exp_m1());
    }
    let sp = TradingSpec { cost_bps: 5.0, threshold: 1e-30, min_trades: 0, ..Default::default() };
    let book = FitnessBook::new(&slice, &sp, STEP as f64).unwrap();
    let vars = vec!["x1".to_string(), "x2".to_string()];
    let members = pareto(&j);
    assert!(!members.is_empty());
    for m in members {
        let tree = rcsr::parse::parse(at(m, &["expression_exact"]).as_str().unwrap(), &vars).unwrap();
        let pred: Vec<f64> = (0..rows)
            .map(|i| rcsr::expr::eval(&tree, &[x1(i), frac(i as f64 * 0.754_877_666_246_692_7) - 0.5]))
            .collect();
        let totals = book.totals(&pred);
        let regret = num(m, &["fitness", "train_regret"]);
        let want = book.regret(&pred);
        assert!((regret - want).abs() <= 1e-9, "train_regret {regret} vs book {want}");
        assert_eq!(num(m, &["fitness", "train_flips"]) as u64, totals.flips);
        assert_eq!((num(m, &["fitness", "train_exposure"]) * rows as f64).round() as u64, totals.long_rows);
    }
}

#[test]
fn require_edge_reads_the_book_rows_under_book_target() {
    let scratch = Scratch::new("book_target_require_edge");
    let data = write_stride_csv(scratch.path());
    let spec = write_spec(scratch.path());
    let j = ok(&run_stride(&data, &spec, &["--select-require-edge"]));
    assert_eq!(num(&j, &["objective", "edge_rejected"]), 0.0);
    assert_eq!(num(&j, &["objective", "usable_candidates"]), 3.0);
    assert!(at(&j, &["objective", "selection_rule"]).get("verdict").is_none());
}
