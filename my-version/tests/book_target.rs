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
        let r24: f64 = (i + 1..i + 1 + DAY).take_while(|&j| j < STRIDE_N).map(r1).sum();
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

// #1212: `rcsr walkforward` prices the book label of a formula fitted on another label, out of fold.

/// 2021-01-04T00:00:00Z.
const WF_T0: i64 = 1_609_718_400;
const WF_DAY: i64 = 86_400;
const WF_WEEK: i64 = 7 * WF_DAY;
const WF_DAYS: i64 = 364;

fn wf_return(d: i64, j: i64) -> f64 {
    let i = (2 * d + j) as usize;
    0.01 * x1(i) + 0.0005 * (frac(i as f64 * 0.324_717_957_244_746) - 0.5)
}

/// Daily rows of symbols a and b. The daily label `lbl_h24_y` and its return `lbl_h24_fwd` are
/// finite on every day; the weekly label `lbl_h168_y` and its return `lbl_h168_fwd`, the sum of the
/// next seven daily returns, only on every seventh day with a full week ahead. `d.labels.json`
/// holds the label reports `rcsr enrich` writes, the source of `split.fit_label`.
fn wf_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,s,x1,x2,lbl_h24_y,lbl_h24_fwd,lbl_h168_y,lbl_h168_fwd\n");
    for d in 0..WF_DAYS {
        for (j, sym) in ["a", "b"].iter().enumerate() {
            let j = j as i64;
            let i = (2 * d + j) as usize;
            let x2 = frac(i as f64 * 0.754_877_666_246_692_7) - 0.5;
            let r = wf_return(d, j);
            let week = if d % 7 == 0 && d + 7 <= WF_DAYS {
                let w: f64 = (d..d + 7).map(|k| wf_return(k, j)).sum();
                format!("{w},{w}")
            } else {
                ",".to_string()
            };
            text.push_str(&format!("{},{sym},{},{x2},{r},{r},{week}\n", WF_T0 + WF_DAY * d, x1(i)));
        }
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    let report = |column: &str, name: &str, stride: i64| {
        format!(
            "{{\"target\":\"{column}\",\"report\":{{\"name\":\"{name}\",\"stride\":{stride},\"average_uniqueness\":1,\"effective_n\":104,\"effective_n_hours\":17472}}}}"
        )
    };
    std::fs::write(dir.join("d.labels.json"), format!("[\n{},\n{}\n]\n", report("lbl_h168_y", "h168", 168), report("lbl_h24_y", "h24", 24))).unwrap();
    path.to_string_lossy().into_owned()
}

fn wf_entry(column: &str, return_column: Option<&str>, horizon: i64, stride: Option<i64>) -> String {
    format!(
        "{{\"column\":\"{column}\",\"name\":\"{column}\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":{},\
         \"return_units\":\"log\",\"horizon_seconds\":{horizon},{}\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\
         \"exit_price\":\"close\",\"offset\":0.0,\"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":10,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}",
        return_column.map_or("null".to_string(), |c| format!("\"{c}\"")),
        stride.map_or(String::new(), |s| format!("\"stride_seconds\":{s},"))
    )
}

/// The weekly fit label, the daily book label, a second label on the daily return, a label without
/// a return column and a label whose return column the data lacks.
fn wf_spec(dir: &std::path::Path) -> String {
    let entries = [
        wf_entry("lbl_h168_y", Some("lbl_h168_fwd"), WF_WEEK, Some(WF_WEEK)),
        wf_entry("lbl_h24_y", Some("lbl_h24_fwd"), WF_DAY, None),
        wf_entry("lbl_d1_y", Some("lbl_h24_fwd"), WF_DAY, None),
        wf_entry("lbl_noret_y", None, WF_DAY, None),
        wf_entry("lbl_h48_y", Some("lbl_h48_fwd"), 2 * WF_DAY, None),
    ];
    let path = dir.join("wf-spec.json");
    std::fs::write(&path, format!("{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{}]}}", entries.join(","))).unwrap();
    path.to_string_lossy().into_owned()
}

fn wf(data: &str, args: &[&str]) -> Output {
    let mut a = vec!["walkforward", "--data", data, "--time", "t", "--group-by", "s", "--json"];
    a.extend_from_slice(args);
    rcsr_cmd::rcsr().args(&a).output().expect("run rcsr walkforward")
}

/// A window in days from `WF_T0`; training starts at day 0.
fn wf_window(label: &str, train_end: i64, test_start: i64, test_end: i64) -> String {
    format!("{label},{WF_T0},{},{},{}", WF_T0 + WF_DAY * train_end, WF_T0 + WF_DAY * test_start, WF_T0 + WF_DAY * test_end)
}

/// `x1` priced with the weekly fit label picked by `--target`.
fn wf_expr<'a>(spec: &'a str, window: &'a str) -> Vec<&'a str> {
    vec!["--target-spec", spec, "--target", "lbl_h168_fwd", "--expr", "x1", "--vars", "x1", "--window", window]
}

fn text_at<'a>(j: &'a Json, path: &[&str]) -> &'a str {
    at(j, path).as_str().unwrap_or_else(|| panic!("{path:?} is not a string"))
}

fn first_window(j: &Json) -> &Json {
    match at(j, &["windows"]) {
        Json::Arr(w) => &w[0],
        _ => panic!("windows is not an array"),
    }
}

/// The printed report without the phase ledger, which holds wall-clock seconds.
fn stable_report(o: &Output) -> String {
    assert_eq!(code(o), 0, "{}", stderr(o));
    rcsr::verdict::remove_member(String::from_utf8_lossy(&o.stdout).trim(), &["ledger"]).unwrap()
}

/// A search fitted on the weekly label; `extra` adds `--book-target`. Returns the run.json path.
fn wf_search(dir: &std::path::Path, data: &str, spec: &str, name: &str, extra: &[&str]) -> String {
    let run = dir.join(name).to_string_lossy().into_owned();
    let mut args = vec![
        "--data", data, "--target-spec", spec, "--target", "lbl_h168_y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--fill", "drop",
        "--split", "temporal", "--objective", "pnl", "--validation-frac", "0.3", "--test-frac", "0.2", "--min-trades", "0", "--cpu", "--no-dev-log",
        "--seed", "3", "--generations", "12", "--pop", "32", "--quiet", "--json", "--json-out", run.as_str(),
    ];
    args.extend_from_slice(extra);
    let o = rcsr_cmd::rcsr().args(&args).output().expect("run rcsr");
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    run
}

/// #1212 criteria 1 and 2: the formula fitted on the weekly label, traded on the daily label, is
/// priced on the daily label: one period per test day, daily annualisation, the report names both.
#[test]
fn wf_book_target_prices_daily_over_weekly_fit() {
    let scratch = Scratch::new("wf_book_target_daily");
    let (data, spec) = (wf_csv(scratch.path()), wf_spec(scratch.path()));
    let window = wf_window("w", 200, 207, 300);
    let weekly = ok(&wf(&data, &wf_expr(&spec, &window)));
    assert_eq!(num(&weekly, &["annualization", "period_seconds"]), WF_WEEK as f64, "without the flag the fit label prices the book");
    assert!(at(&weekly, &["book"]).get("target").is_none());
    let mut args = wf_expr(&spec, &window);
    args.extend_from_slice(&["--book-target", "lbl_h24_fwd"]);
    let daily = ok(&wf(&data, &args));
    assert_eq!(num(&daily, &["pooled", "periods_per_year"]), 365.0);
    assert_eq!(num(&daily, &["annualization", "period_seconds"]), 86400.0);
    assert_eq!(text_at(&daily, &["target"]), "lbl_h24_fwd");
    assert_eq!(num(&daily, &["pooled", "periods"]), 93.0, "one period per test day");
    assert_eq!(num(first_window(&daily), &["metrics", "rows"]), 186.0, "both symbols on every test day");
    assert_eq!(text_at(&daily, &["split", "fit_label", "name"]), "h168");
    assert_eq!(text_at(&daily, &["book", "target", "column"]), "lbl_h24_y");
    assert_eq!(text_at(&daily, &["book", "target", "return_column"]), "lbl_h24_fwd");
    assert_eq!(num(&daily, &["book", "target", "horizon_seconds"]), 86400.0);
    assert_eq!(text_at(&daily, &["book", "fit_target", "column"]), "lbl_h168_y");
    assert_eq!(text_at(&daily, &["book", "source"]), "flag");
    assert_eq!(num(&daily, &["split", "required_purge_seconds"]), WF_WEEK as f64, "the fit label's purge binds");
    assert_eq!(text_at(&daily, &["split", "labelled_rows", "rule"]), "book_target");
}

/// #1212 criterion 6: a book label equal to the fit label is the report without the flag.
#[test]
fn wf_book_target_same_label_identical() {
    let scratch = Scratch::new("wf_book_target_same");
    let (data, spec) = (wf_csv(scratch.path()), wf_spec(scratch.path()));
    let window = wf_window("w", 200, 207, 300);
    let plain = wf(&data, &wf_expr(&spec, &window));
    let want = stable_report(&plain);
    for same in ["lbl_h168_fwd", "lbl_h168_y"] {
        let mut args = wf_expr(&spec, &window);
        args.extend_from_slice(&["--book-target", same]);
        let o = wf(&data, &args);
        assert_eq!(stable_report(&o), want, "--book-target {same}");
        assert_eq!(text_at(&json(&o), &["protocol", "report_sha256"]), text_at(&json(&plain), &["protocol", "report_sha256"]));
    }
}

/// #1212 criterion 3: the purge covers the longer fit horizon; a 24 h gap names the weekly fit label.
#[test]
fn wf_book_target_purge_covers_fit_horizon() {
    let scratch = Scratch::new("wf_book_target_purge");
    let (data, spec) = (wf_csv(scratch.path()), wf_spec(scratch.path()));
    let window = wf_window("w", 200, 201, 300);
    ok(&wf(&data, &["--target-spec", &spec, "--target", "lbl_h24_fwd", "--expr", "x1", "--vars", "x1", "--window", &window]));
    let mut args = wf_expr(&spec, &window);
    args.extend_from_slice(&["--book-target", "lbl_h24_fwd"]);
    let o = wf(&data, &args);
    usage_error(&o, "fit label 'lbl_h168_y'");
    assert!(stderr(&o).contains("purge gap is 86400 seconds") && stderr(&o).contains("604800"), "{}", stderr(&o));
}

/// #1212 criterion 4: `--from-run`, `--front` and `--front --all-members` price the run's book label.
#[test]
fn wf_from_run_honours_book_target() {
    let scratch = Scratch::new("wf_book_target_from_run");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y"]);
    let j = Json::parse(std::fs::read_to_string(&run).unwrap().trim()).expect("run.json");
    assert_eq!(text_at(&j, &["objective", "book_target"]), "lbl_h24_y");
    let window = wf_window("t", 293, 300, WF_DAYS);
    let from_run = |extra: &[&str]| {
        let mut a = vec!["--from-run", run.as_str(), "--window", window.as_str(), "--allow-in-sample-windows"];
        a.extend_from_slice(extra);
        wf(&data, &a)
    };
    let r = ok(&from_run(&[]));
    assert_eq!(text_at(&r, &["target"]), "lbl_h24_fwd");
    assert_eq!(text_at(&r, &["book", "source"]), "run");
    assert_eq!(text_at(&r, &["book", "target", "column"]), "lbl_h24_y");
    assert_eq!(text_at(&r, &["split", "fit_label", "name"]), "h168");
    assert_eq!(num(&r, &["annualization", "period_seconds"]), 86400.0);
    assert_eq!(num(&r, &["pooled", "periods_per_year"]), 365.0);
    assert_eq!(num(&r, &["pooled", "periods"]), 64.0, "one period per test day");
    assert_eq!(num(&r, &["split", "required_purge_seconds"]), WF_WEEK as f64);
    // Naming the run's book label changes nothing; another label exits 2.
    assert_eq!(stable_report(&from_run(&["--book-target", "lbl_h24_fwd"])), stable_report(&from_run(&[])));
    usage_error(&from_run(&["--book-target", "lbl_h168_y"]), "differs from the book label of");
    for flag in ["--device-pnl", "--batch-pnl"] {
        usage_error(&from_run(&[flag]), "is priced on the CPU");
    }
    let front = |extra: &[&str]| {
        let mut a = vec!["--front", run.as_str(), "--window", window.as_str(), "--allow-in-sample-windows", "--validation-frac", "0.2"];
        a.extend_from_slice(extra);
        ok(&wf(&data, &a))
    };
    let f = front(&[]);
    assert_eq!(text_at(&f, &["target"]), "lbl_h24_fwd");
    assert_eq!(text_at(&f, &["book", "source"]), "run");
    assert_eq!(num(&f, &["annualization", "period_seconds"]), 86400.0);
    assert_eq!(num(first_window(&f), &["metrics", "rows"]), 128.0);
    let m = front(&["--all-members"]);
    let Json::Arr(members) = at(&m, &["members"]) else { panic!("members is not an array") };
    assert!(!members.is_empty());
    for member in members {
        assert_eq!(num(member, &["pooled", "periods_per_year"]), 365.0);
        assert_eq!(num(first_window(member), &["test_rows"]), num(first_window(&f), &["test_rows"]));
    }
}

/// #1212 criterion 5: `--from-run` of a run without `--book-target` keeps the report of the build
/// before #1212, byte for byte (the hashed report without the run file's own hash).
#[test]
fn wf_from_run_without_book_target_unchanged() {
    let scratch = Scratch::new("wf_book_target_plain_run");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "fit-run.json", &[]);
    let window = wf_window("t", 293, 300, WF_DAYS);
    let o = wf(&data, &["--from-run", &run, "--window", &window, "--allow-in-sample-windows"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = String::from_utf8_lossy(&o.stdout).into_owned();
    let hashed = rcsr::verdict::hashed_report_text(&text).unwrap();
    let before = rcsr::verdict::remove_member(&hashed, &["selection", "run_sha256"]).unwrap();
    assert_eq!(rcsr::manifest::sha256_hex(before.as_bytes()), PINNED_FIT_LABEL_FROM_RUN, "{before}");
    let r = json(&o);
    assert!(at(&r, &["book"]).get("target").is_none() && at(&r, &["book"]).get("source").is_none());
    usage_error(&wf(&data, &["--from-run", &run, "--window", &window, "--allow-in-sample-windows", "--book-target", "lbl_h24_y"]), "differs from the book label of");
}

const PINNED_FIT_LABEL_FROM_RUN: &str = "efe1d444de2ba21827804f82f4f2c9fd538754f33e9aca8f4f508d015bd02ae0";

/// #1212 criterion 7 and the refusals of `--book-target`: exit 2.
#[test]
fn wf_book_target_refusals_are_usage_errors() {
    let scratch = Scratch::new("wf_book_target_refusals");
    let (data, spec) = (wf_csv(scratch.path()), wf_spec(scratch.path()));
    let window = wf_window("w", 200, 207, 300);
    let with = |extra: &[&str]| {
        let mut a = wf_expr(&spec, &window);
        a.extend_from_slice(extra);
        wf(&data, &a)
    };
    for flag in ["--device-pnl", "--batch-pnl"] {
        usage_error(&with(&["--book-target", "lbl_h24_fwd", flag]), "is priced on the CPU");
    }
    usage_error(&with(&["--book-target", "nope"]), "has no entry whose column, name or return_column is 'nope'");
    usage_error(&with(&["--book-target", "lbl_noret_y"]), "book target 'lbl_noret_y' declares no return_column");
    usage_error(&with(&["--book-target", "lbl_h48_y"]), "'lbl_h48_fwd'");
    usage_error(&wf(&data, &["--target", "lbl_h24_fwd", "--expr", "x1", "--vars", "x1", "--window", &window, "--book-target", "lbl_h24_fwd"]), "--book-target needs --target-spec");
}

/// #1212 criterion 8: the candidate id binds the book label, so a locked read of another book label
/// is not the sealed candidate (exit 3), even on the same return column.
#[test]
fn wf_book_target_binds_the_sealed_candidate() {
    let scratch = Scratch::new("wf_book_target_seal");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    rcsr::lockbox::write_sidecar(&data, &rcsr::lockbox::Lockbox { start: WF_T0 + 300 * WF_DAY, end: WF_T0 + WF_DAYS * WF_DAY }, "test").unwrap();
    let ledger = dir.join("ledger.jsonl").to_string_lossy().into_owned();
    let sealed = dir.join("sealed.json").to_string_lossy().into_owned();
    let (dev_window, locked_window) = (wf_window("dev", 200, 207, 290), wf_window("locked", 290, 300, WF_DAYS));
    let books: [&[&str]; 3] = [&["--book-target", "lbl_h24_fwd"], &["--book-target", "lbl_d1_y"], &[]];
    let ids: Vec<String> = books
        .iter()
        .map(|b| {
            let mut a = wf_expr(&spec, &dev_window);
            a.extend_from_slice(&["--ledger", ledger.as_str()]);
            a.extend_from_slice(b);
            text_at(&ok(&wf(&data, &a)), &["protocol", "candidate_id"]).to_string()
        })
        .collect();
    assert!(ids[0] != ids[1] && ids[0] != ids[2] && ids[1] != ids[2], "{ids:?}");
    let o = rcsr_cmd::rcsr().args(["protocol", "seal", "--ledger", &ledger, "--out", &sealed, "--id", &ids[0]]).output().unwrap();
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let read = |b: &[&str]| {
        let mut a = wf_expr(&spec, &locked_window);
        a.extend_from_slice(&["--ledger", ledger.as_str(), "--locked-test", "--sealed", sealed.as_str()]);
        a.extend_from_slice(b);
        wf(&data, &a)
    };
    for b in &books[1..] {
        let o = read(b);
        assert_eq!(code(&o), 3, "{b:?}: {}", stderr(&o));
        assert!(stderr(&o).contains("is not in the sealed list"), "{b:?}: {}", stderr(&o));
    }
    ok(&read(books[0]));
}

/// One expanding ladder fold: fit from day 0, validation from day 200, test days 300..364 (the test
/// window of `wf_from_run_honours_book_target`).
fn ladder_windows(dir: &std::path::Path) -> String {
    let iso = |d: i64| rcsr::protocol::iso((WF_T0 + WF_DAY * d) as f64);
    let path = dir.join("ladder-windows.csv");
    std::fs::write(
        &path,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,t,{},{},{},{},{}\n",
            iso(0),
            iso(293),
            iso(200),
            iso(300),
            iso(WF_DAYS)
        ),
    )
    .unwrap();
    path.to_string_lossy().into_owned()
}

fn ladder_sr(data: &str, windows: &str, run: &str, extra: &[&str]) -> Output {
    let mut a = vec![
        "ladder", "--data", data, "--time", "t", "--group", "s", "--features", "x1,x2", "--windows", windows, "--sr-run", run, "--draws", "50",
        "--gbm-trees", "10", "--json",
    ];
    a.extend_from_slice(extra);
    rcsr_cmd::rcsr().args(&a).output().expect("run rcsr ladder")
}

/// The printed ladder report without what differs between two runs of one build: the phase ledger,
/// the rungs' wall seconds, the run file's path and hash (the run records its own wall) and the
/// search's wall.
fn ladder_stable(o: &Output) -> String {
    assert_eq!(code(o), 0, "{}", stderr(o));
    let mut t = String::from_utf8_lossy(&o.stdout).trim().to_string();
    for path in [&["ledger"][..], &["sr_run", "run"], &["sr_run", "run_sha256"], &["equal_cost", "sr_wall_seconds"]] {
        t = rcsr::verdict::remove_member(&t, path).unwrap();
    }
    let key = "\"wall_seconds\":";
    while let Some(k) = t.find(key) {
        let end = t[k..].find(',').map_or(t.len(), |e| k + e + 1);
        t.replace_range(k..end, "");
    }
    t
}

/// The walk-forward of the run on the ladder's test days, the reference book.
fn wf_reference(data: &str, run: &str) -> Json {
    let window = wf_window("t", 293, 300, WF_DAYS);
    ok(&wf(data, &["--from-run", run, "--window", &window, "--allow-in-sample-windows"]))
}

/// The run without `objective.book_target_spec`, as a run written before #1291.
fn without_book_spec(dir: &std::path::Path, run: &str) -> String {
    let text = std::fs::read_to_string(run).unwrap();
    let bare = rcsr::verdict::remove_member(text.trim(), &["objective", "book_target_spec"]).unwrap();
    assert!(bare.contains("\"book_target\":") && !bare.contains("book_target_spec"));
    let path = dir.join("no-book-spec.run.json");
    std::fs::write(&path, bare).unwrap();
    path.to_string_lossy().into_owned()
}

fn sr_rung(j: &Json) -> &Json {
    let Json::Arr(rungs) = at(j, &["rungs"]) else { panic!("rungs is not an array") };
    rungs.iter().find(|r| r.get("name").and_then(Json::as_str) == Some("symbolic_regression")).expect("symbolic_regression rung")
}

/// #1311 criteria 1 and 5: `ladder --sr-run` of a run searched with `--book-target` prices the daily
/// book label whichever of the two labels `--target` names, as `walkforward --from-run` does.
#[test]
fn ladder_sr_run_prices_book_label() {
    let scratch = Scratch::new("ladder_book_label");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y"]);
    let windows = ladder_windows(dir);
    let fit = ladder_sr(&data, &windows, &run, &["--target-spec", &spec, "--target", "lbl_h168_y"]);
    let j = ok(&fit);
    assert_eq!(num(&j, &["annualization", "period_seconds"]), 86400.0, "the daily book label sets the period");
    assert_eq!(num(&j, &["periods"]), 64.0, "one period per test day");
    assert_eq!(num(&j, &["rows_scored"]), 128.0, "both symbols on every test day");
    assert_eq!(text_at(sr_rung(&j), &["status"]), "ok");
    assert_eq!(text_at(&j, &["book", "target", "column"]), "lbl_h24_y");
    assert_eq!(text_at(&j, &["book", "target", "return_column"]), "lbl_h24_fwd");
    assert_eq!(text_at(&j, &["book", "fit_target", "column"]), "lbl_h168_y");
    assert_eq!(text_at(&j, &["book", "source"]), "run");
    assert_eq!(text_at(&j, &["split", "fit_label", "name"]), "h168");
    assert_eq!(text_at(&j, &["split", "labelled_rows", "rule"]), "book_target");
    assert_eq!(text_at(&j, &["split", "labelled_rows", "column"]), "lbl_h24_y");
    let r = wf_reference(&data, &run);
    assert_eq!(num(&j, &["periods"]), num(&r, &["pooled", "periods"]));
    assert_eq!(num(&j, &["annualization", "period_seconds"]), num(&r, &["annualization", "period_seconds"]));
    assert_eq!(num(&j, &["purge_seconds"]), num(&r, &["split", "required_purge_seconds"]));
    // The book label by name or return column, with or without the spec file, is the same report.
    let want = ladder_stable(&fit);
    for target in [&["--target", "lbl_h24_y"][..], &["--target", "lbl_h24_fwd"], &["--target-spec", &spec, "--target", "lbl_h24_y"]] {
        assert_eq!(ladder_stable(&ladder_sr(&data, &windows, &run, target)), want, "{target:?}");
    }
}

/// #1311 criterion 2: the purge covers the weekly fit label although the book label is daily.
#[test]
fn ladder_sr_run_book_label_purge() {
    let scratch = Scratch::new("ladder_book_purge");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y"]);
    let windows = ladder_windows(dir);
    let j = ok(&ladder_sr(&data, &windows, &run, &["--target", "lbl_h24_y"]));
    assert_eq!(num(&j, &["purge_seconds"]), WF_WEEK as f64);
    assert_eq!(text_at(&j, &["purge_source"]), "target_spec");
    let wide = ok(&ladder_sr(&data, &windows, &run, &["--target", "lbl_h24_y", "--purge-seconds", "700000"]));
    assert_eq!(num(&wide, &["purge_seconds"]), 700000.0, "a wider flag still wins");
    assert_eq!(text_at(&wide, &["purge_source"]), "flag");
}

/// #1311 criterion 3: a third label, another return column and a run without the book label's spec exit 2.
#[test]
fn ladder_sr_run_book_label_refusals() {
    let scratch = Scratch::new("ladder_book_refusals");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y"]);
    let windows = ladder_windows(dir);
    let third = "is neither the fit label 'lbl_h168_y' nor the book label 'lbl_h24_y'";
    usage_error(&ladder_sr(&data, &windows, &run, &["--target", "lbl_d1_y"]), third);
    usage_error(&ladder_sr(&data, &windows, &run, &["--target-spec", &spec, "--target", "lbl_d1_y"]), third);
    usage_error(&ladder_sr(&data, &windows, &run, &["--target", "lbl_h24_y", "--return-col", "lbl_h168_fwd"]), "--return-col lbl_h168_fwd differs from the book label");
    let bare = without_book_spec(dir, &run);
    usage_error(&ladder_sr(&data, &windows, &bare, &["--target", "lbl_h24_y"]), "records no objective.book_target_spec");
}

/// #1311 criterion 4: `ladder --sr-run` of a run without `--book-target` keeps the report of the
/// build before #1311, byte for byte (without what differs between two runs of one build).
#[test]
fn ladder_sr_run_plain_unchanged() {
    let scratch = Scratch::new("ladder_plain_run");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "fit-run.json", &[]);
    let windows = ladder_windows(dir);
    let o = ladder_sr(&data, &windows, &run, &["--target-spec", &spec, "--target", "lbl_h168_y"]);
    let text = ladder_stable(&o);
    assert_eq!(rcsr::manifest::sha256_hex(text.as_bytes()), PINNED_FIT_LABEL_LADDER, "{text}");
    assert!(json(&o).get("book").is_none());
}

const PINNED_FIT_LABEL_LADDER: &str = "2efe5dc3e3a654c0c7c6c4dd226a4ccacd94f80e30b44180f30041fbefe549d5";

fn ablation_score(dir: &std::path::Path, name: &str, data: &str, windows: &str, extra: &[&str]) -> (Output, Option<Json>) {
    let out = dir.join(name).to_string_lossy().into_owned();
    let mut a = vec!["ablation", "score", "--data", data, "--time", "t", "--group", "s", "--windows", windows, "--out", out.as_str()];
    a.extend_from_slice(extra);
    let o = rcsr_cmd::rcsr().args(&a).output().expect("run rcsr ablation score");
    let j = (code(&o) == 0).then(|| Json::parse(std::fs::read_to_string(&out).unwrap().trim()).expect("ablation result"));
    (o, j)
}

fn net_periods(j: &Json) -> usize {
    match at(j, &["net_period"]) {
        Json::Arr(a) => a.len(),
        _ => panic!("net_period is not an array"),
    }
}

/// #1311: `ablation score` of a run searched with `--book-target` prices the daily book label with the
/// weekly fit label's purge, from `--from-run` and from `--run`, as `walkforward --from-run` does.
#[test]
fn ablation_score_prices_book_label() {
    let scratch = Scratch::new("ablation_book_label");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y", "--arm", "A0"]);
    let windows = ladder_windows(dir);
    let r = wf_reference(&data, &run);
    let runs: [&[&str]; 2] = [&["--from-run", &run], &["--run", &run, "--target", "lbl_h168_y"]];
    for (k, args) in runs.iter().enumerate() {
        let (o, j) = ablation_score(dir, &format!("score{k}.json"), &data, &windows, args);
        let j = j.unwrap_or_else(|| panic!("{args:?}: {}", stderr(&o)));
        assert_eq!(net_periods(&j), 64, "{args:?}: one period per test day");
        assert_eq!(net_periods(&j) as f64, num(&r, &["pooled", "periods"]), "{args:?}");
        assert_eq!(num(&j, &["purge_seconds"]), num(&r, &["split", "required_purge_seconds"]), "{args:?}");
        assert_eq!(text_at(&j, &["purge_source"]), "target_spec", "{args:?}");
        assert_eq!(text_at(&j, &["labelled_rows", "column"]), "lbl_h24_y", "{args:?}");
        assert_eq!(text_at(&j, &["book", "target", "column"]), "lbl_h24_y", "{args:?}");
        assert_eq!(text_at(&j, &["book", "fit_target", "column"]), "lbl_h168_y", "{args:?}");
        assert_eq!(text_at(&j, &["book", "source"]), "run", "{args:?}");
    }
}

/// #1311: a run without the book label's spec, a third label and another return column exit 2.
#[test]
fn ablation_score_book_label_refusals() {
    let scratch = Scratch::new("ablation_book_refusals");
    let dir = scratch.path();
    let (data, spec) = (wf_csv(dir), wf_spec(dir));
    let run = wf_search(dir, &data, &spec, "book-run.json", &["--book-target", "lbl_h24_y", "--arm", "A0"]);
    let windows = ladder_windows(dir);
    let bare = without_book_spec(dir, &run);
    for mode in ["--from-run", "--run"] {
        let (o, _) = ablation_score(dir, "bare.json", &data, &windows, &[mode, &bare, "--target", "lbl_h24_y"]);
        usage_error(&o, "records no objective.book_target_spec");
    }
    let (o, _) = ablation_score(dir, "third.json", &data, &windows, &["--from-run", &run, "--target", "lbl_d1_y"]);
    usage_error(&o, "is neither the fit label 'lbl_h168_y' nor the book label 'lbl_h24_y'");
    let (o, _) = ablation_score(dir, "ret.json", &data, &windows, &["--from-run", &run, "--return-col", "lbl_h168_fwd"]);
    usage_error(&o, "--return-col lbl_h168_fwd differs from the book label");
}

/// #1208: `rcsr trials` reads the daily book report of a weekly fit and the weekly report without it.
#[test]
fn trials_reads_book_target_reports() {
    let scratch = Scratch::new("wf_book_target_trials");
    let (data, spec) = (wf_csv(scratch.path()), wf_spec(scratch.path()));
    let window = wf_window("w", 200, 207, 300);
    for (name, extra) in [("weekly", vec![]), ("daily", vec!["--book-target", "lbl_h24_fwd"])] {
        let mut args = wf_expr(&spec, &window);
        args.extend_from_slice(&extra);
        let o = wf(&data, &args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        let path = scratch.path().join(format!("{name}.json"));
        std::fs::write(&path, &o.stdout).unwrap();
        let t = rcsr_cmd::rcsr().args(["trials", "--wf", path.to_str().unwrap(), "--draws", "50", "--block-len", "2"]).output().expect("run rcsr trials");
        assert_eq!(code(&t), 0, "{name}: {}", stderr(&t));
        assert!(!stderr(&t).contains("annualization"), "{name}: {}", stderr(&t));
    }
}
