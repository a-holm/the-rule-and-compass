//! `--objective pnl|sharpe|log_growth`: refusals, validation-only selection
//! and the `--json` block.

use rcsr::devlog::Json;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;
const N: usize = 400;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-objective-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// `(x1, x2, r)` of row `i`; `r` depends on `x1`, so a long book can win.
fn row(i: usize) -> (f64, f64, f64) {
    let f = i as f64;
    let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
    let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
    let noise = frac(f * 0.324_717_957_244_746) - 0.5;
    (x1, x2, 0.01 * x1 + 0.0005 * noise)
}

/// `rotate_from..N` has its `r` (and `y`) rotated by 7 rows.
fn write_csv(dir: &std::path::Path, name: &str, rotate_from: Option<usize>) -> String {
    let mut r: Vec<f64> = (0..N).map(|i| row(i).2).collect();
    if let Some(from) = rotate_from {
        r[from..].rotate_left(7);
    }
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
        let (x1, x2, _) = row(i);
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r[i], r[i]));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &std::path::Path, name: &str, offset: f64, cost: &str, in_label: bool) -> String {
    write_spec_horizon(dir, name, offset, cost, in_label, 900)
}

fn write_spec_horizon(
    dir: &std::path::Path,
    name: &str,
    offset: f64,
    cost: &str,
    in_label: bool,
    horizon: u32,
) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":{horizon},\"entry_offset_seconds\":900,\"delay_seconds\":0,\
         \"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":{offset:?},\
         \"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":{cost},\"cost_in_label\":{in_label},\"construction_columns\":[],\"note\":null}}]}}"
    );
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "15",
        "--pop", "32", "--metric", "squared", "--json", "--quiet",
    ];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().expect("run rcsr")
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

fn usage_error(o: &Output, needle: &str) {
    assert_eq!(code(o), 2, "{}", stderr(o));
    assert!(stderr(o).contains(needle), "want '{needle}' in: {}", stderr(o));
}

const PNL: &[&str] = &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3"];

#[test]
fn refusals_are_usage_errors() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);

    usage_error(&run(&data, &spec, &["--cost-bps", "5"]), "--cost-bps needs --objective");
    usage_error(
        &run(&data, &spec, &["--objective", "pnl", "--validation-frac", "0.3"]),
        "needs a cost assumption",
    );
    usage_error(
        &run(&data, &spec, &["--objective", "pnl", "--cost-bps", "5"]),
        "needs a validation slice",
    );
    let mut short = PNL.to_vec();
    short.extend_from_slice(&["--position", "short"]);
    usage_error(&run(&data, &spec, &short), "long-only");
    let mut allow = PNL.to_vec();
    allow.push("--allow-short");
    usage_error(&run(&data, &spec, &allow), "long-only");
    usage_error(&run(&data, &spec, &["--objective", "alpha"]), "unknown --objective");

    let labelled = write_spec(&dir, "net.json", 0.002, "20.0", true);
    usage_error(&run(&data, &labelled, PNL), "cost_already_in_label");

    let no_spec = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", data.as_str(), "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal", "--cpu", "--no-dev-log", "--generations", "1", "--pop", "8", "--json", "--quiet"])
        .args(PNL)
        .output()
        .unwrap();
    usage_error(&no_spec, "forward-return target");
}

#[test]
fn validation_selection_ignores_the_test_block_and_json_has_the_book() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let plain = write_csv(&dir, "a.csv", None);
    let shuffled = write_csv(&dir, "b.csv", Some(330));

    let a = run(&plain, &spec, PNL);
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    let b = run(&shuffled, &spec, PNL);
    assert_eq!(code(&b), 0, "{}", stderr(&b));
    let (ja, jb) = (json(&a), json(&b));

    let o = at(&ja, &["objective"]);
    assert_eq!(at(o, &["name"]).as_str(), Some("pnl"));
    assert_eq!(at(o, &["exact"]).as_str(), Some("cpu_f64"));
    assert_eq!(at(o, &["position"]).as_str(), Some("long_only"));
    assert!(num(o, &["candidates"]) >= 1.0);
    assert!(num(o, &["usable_candidates"]) >= 1.0);
    assert!(num(o, &["finalist", "fit_error"]).is_finite());
    assert!(at(o, &["backend_reason"]).as_str().is_some());
    assert_eq!(num(o, &["spec", "per_side_bps"]), 5.0);
    for book in ["validation", "test"] {
        let p = at(o, &[book]);
        assert!(num(p, &["exposure"]).is_finite(), "{book}");
        assert!(num(p, &["rows"]) > 10.0, "{book}");
        assert!(at(p, &["cash"]).get("net_pnl").is_some(), "{book}");
        assert!(num(p, &["buy_and_hold", "net_pnl"]).is_finite(), "{book}");
        assert!(matches!(at(p, &["cost_ladder"]), Json::Arr(a) if a.len() == 5), "{book}");
    }
    assert_eq!(num(o, &["validation", "cash", "net_pnl"]), 0.0);

    // The selected candidate, its constants and its validation book do not
    // depend on the test block; only the reported test book does.
    let ob = at(&jb, &["objective"]);
    assert_eq!(
        at(o, &["finalist", "expression"]).as_str(),
        at(ob, &["finalist", "expression"]).as_str()
    );
    assert_eq!(num(o, &["finalist", "validation_objective"]), num(ob, &["finalist", "validation_objective"]));
    assert_eq!(num(o, &["validation", "net_pnl"]), num(ob, &["validation", "net_pnl"]));
    assert_ne!(num(o, &["test", "net_pnl"]), num(ob, &["test", "net_pnl"]));
}

#[test]
fn sharpe_and_log_growth_objectives_run() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let data = write_csv(&dir, "d.csv", None);
    for name in ["sharpe", "log_growth"] {
        let out = run(&data, &spec, &["--objective", name, "--cost-bps", "5", "--validation-frac", "0.3"]);
        assert_eq!(code(&out), 0, "{name}: {}", stderr(&out));
        assert_eq!(at(&json(&out), &["objective", "name"]).as_str(), Some(name));
    }
}

#[test]
fn the_default_run_has_no_objective_block() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let data = write_csv(&dir, "d.csv", None);
    let out = run(&data, &spec, &["--validation-frac", "0.3"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(json(&out).get("objective").is_none());
}

/// Imports `csv` to a `.rcsb` store; `extra` columns are not added, so callers
/// write the CSV they want.
fn import_store(dir: &std::path::Path, csv: &str) -> String {
    let store = dir.join("d.rcsb").to_string_lossy().into_owned();
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["import", "--data", csv, "--out", &store, "--time", "t", "--quiet"])
        .output()
        .expect("run rcsr import");
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    store
}

fn write_csv_with_funding(dir: &std::path::Path) -> String {
    let base = std::fs::read_to_string(write_csv(dir, "base.csv", None)).unwrap();
    let mut text = String::new();
    for (i, line) in base.lines().enumerate() {
        if i == 0 {
            text.push_str(&format!("{},f\n", line));
        } else {
            text.push_str(&format!("{},0.0001\n", line));
        }
    }
    let path = dir.join("fund.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn projection_columns(dir: &std::path::Path, data: &str, spec: &str, extra: &[&str]) -> (Output, Vec<String>) {
    let path = dir.join("manifest.json");
    let mut args = extra.to_vec();
    args.extend_from_slice(&["--run-manifest", path.to_str().unwrap()]);
    let o = run(data, spec, &args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = Json::parse(std::fs::read_to_string(&path).unwrap().trim()).expect("manifest parses");
    let cols = at(&j, &["projection", "columns"]);
    let Json::Arr(items) = cols else { panic!("projection.columns is not an array") };
    let names = items.iter().map(|c| c.as_str().unwrap().to_string()).collect();
    (o, names)
}

#[test]
fn a_projected_store_read_keeps_the_return_column() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv(&dir, "d.csv", None));
    let (out, cols) = projection_columns(&dir, &store, &spec, PNL);
    assert!(json(&out).get("objective").is_some());
    assert!(cols.contains(&"r".to_string()), "{cols:?}");
}

#[test]
fn a_projected_store_read_keeps_the_funding_column() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv_with_funding(&dir));
    let mut extra = PNL.to_vec();
    extra.extend_from_slice(&["--funding", "f"]);
    let (out, cols) = projection_columns(&dir, &store, &spec, &extra);
    let paid = num(&json(&out), &["objective", "validation", "funding_paid"]);
    assert!(paid.is_finite(), "{paid}");
    assert!(cols.contains(&"f".to_string()), "{cols:?}");
}

#[test]
fn the_projection_without_an_objective_is_unchanged() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv(&dir, "d.csv", None));
    let (_, cols) = projection_columns(&dir, &store, &spec, &["--validation-frac", "0.3"]);
    assert_eq!(cols, vec!["t", "x1", "x2", "y"]);
}

#[test]
fn a_missing_return_column_names_the_target_spec() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let text = std::fs::read_to_string(&spec).unwrap().replace("\"return_column\":\"r\"", "\"return_column\":\"gone\"");
    std::fs::write(&spec, text).unwrap();
    let store = import_store(&dir, &write_csv(&dir, "d.csv", None));
    let out = run(&store, &spec, PNL);
    assert_eq!(code(&out), 3, "{}", stderr(&out));
    assert!(stderr(&out).contains("declares return_column 'gone'"), "{}", stderr(&out));
}

#[test]
fn overlapping_rows_are_refused_and_the_spacing_is_reported() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let overlapping = write_spec_horizon(&dir, "o.json", 0.0, "null", false, 3600);
    usage_error(&run(&data, &overlapping, PNL), "overlapping returns");

    let spaced = write_spec_horizon(&dir, "p.json", 0.0, "null", false, 900);
    let o = run(&data, &spaced, PNL);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = json(&o);
    assert_eq!(num(&j, &["objective", "spec", "min_group_step_seconds"]), 900.0);
    assert_eq!(num(&j, &["objective", "spec", "horizon_seconds"]), 900.0);
}

fn write_csv_with_spare_columns(dir: &std::path::Path) -> String {
    let base = std::fs::read_to_string(write_csv(dir, "base.csv", None)).unwrap();
    let mut text = String::new();
    for (i, line) in base.lines().enumerate() {
        if i == 0 {
            text.push_str(&format!("{},z1,z2,z3\n", line));
        } else {
            text.push_str(&format!("{},{},{},{}\n", line, i, 2 * i, 3 * i));
        }
    }
    let path = dir.join("spare.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn store_projection_note(j: &Json) -> String {
    at(j, &["preparation", "store_memory", "projection"]).as_str().unwrap().to_string()
}

fn projected_columns(j: &Json) -> f64 {
    num(j, &["preparation", "store_memory", "projected_columns"])
}

fn row_expression_runs() -> Vec<Vec<String>> {
    let cut = T0 + 900 * 300;
    let early = T0 + 900 * 280;
    vec![
        vec!["--exclude".to_string(), format!("t >= {cut}")],
        vec![
            "--train-select".to_string(),
            format!("t < {early}"),
            "--test-select".to_string(),
            format!("t >= {early}"),
        ],
        vec!["--filter".to_string(), "z1 > 5 and z2 > 0".to_string()],
    ]
}

fn run_args(data: &str, spec: &str, expr: &[String], no_project: bool) -> Output {
    let mut extra: Vec<&str> = PNL.to_vec();
    extra.extend(expr.iter().map(String::as_str));
    if no_project {
        extra.push("--no-project");
    }
    run(data, spec, &extra)
}

#[test]
fn row_expressions_keep_the_projection() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv_with_spare_columns(&dir));
    for expr in row_expression_runs() {
        let out = run_args(&store, &spec, &expr, false);
        assert_eq!(code(&out), 0, "{expr:?}: {}", stderr(&out));
        let j = json(&out);
        let total = num(&j, &["preparation", "store_memory", "columns"]);
        let used = projected_columns(&j);
        assert!(used < total, "{expr:?}: projected {used} of {total}");
        assert_eq!(store_projection_note(&j), "on", "{expr:?}");
    }
}

#[test]
fn the_projection_includes_the_columns_an_expression_reads() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv_with_spare_columns(&dir));
    let expr = vec!["--filter".to_string(), "z1 > 5 and z2 > 0".to_string()];
    let plain = projected_columns(&json(&run_args(&store, &spec, &[], false)));
    let out = run_args(&store, &spec, &expr, false);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert_eq!(projected_columns(&json(&out)), plain + 2.0);
}

#[test]
fn projected_and_full_reads_agree_with_row_expressions() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv_with_spare_columns(&dir));
    for expr in row_expression_runs() {
        let on = run_args(&store, &spec, &expr, false);
        let off = run_args(&store, &spec, &expr, true);
        assert_eq!(code(&on), 0, "{expr:?}: {}", stderr(&on));
        assert_eq!(code(&off), 0, "{expr:?}: {}", stderr(&off));
        let (a, b) = (json(&on), json(&off));
        assert_eq!(store_projection_note(&a), "on");
        assert_eq!(store_projection_note(&b), "--no-project");
        assert_eq!(at(&a, &["objective"]), at(&b, &["objective"]), "{expr:?}");
        assert_eq!(at(&a, &["pareto"]), at(&b, &["pareto"]), "{expr:?}");
    }
}

#[test]
fn an_unknown_column_in_a_row_expression_fails_before_the_read() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let store = import_store(&dir, &write_csv_with_spare_columns(&dir));
    let out = run_args(&store, &spec, &["--exclude".to_string(), "nope > 1".to_string()], false);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("cannot parse --exclude"), "{}", stderr(&out));
}

#[test]
fn run_json_carries_the_seen_time_ranges_and_the_candidate_count() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let data = write_csv(&dir, "a.csv", None);
    let out = run(&data, &spec, PNL);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    let pair = |path: &[&str]| match at(&j, path) {
        Json::Arr(a) if a.len() == 2 => (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()),
        other => panic!("{path:?} is not a pair: {other:?}"),
    };
    let (t0, t1) = pair(&["split", "train_time"]);
    let (v0, v1) = pair(&["split", "validation_time"]);
    assert_eq!(t0, T0 as f64);
    assert!(t1 < v0 && v0 < v1, "train {t1} validation {v0}..{v1}");
    assert!(num(&j, &["search", "candidates_evaluated"]) > 0.0);
    assert_eq!(at(&j, &["target_spec", "return_column"]).as_str(), Some("r"));
}

#[test]
fn walkforward_front_of_a_real_run_guards_the_seen_rows() {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let data = write_csv(&dir, "a.csv", None);
    let out = run(&data, &spec, PNL);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let j = json(&out);
    let run_path = dir.join("run.json");
    std::fs::write(&run_path, text.trim()).unwrap();
    let run_path = run_path.to_string_lossy().into_owned();
    let seen = match at(&j, &["split", "validation_time"]) {
        Json::Arr(a) => a[1].as_f64().unwrap(),
        _ => unreachable!(),
    };
    let purge = num(&j, &["split", "required_purge_seconds"]);
    let end = (T0 + 900 * N as i64) as f64;
    let walk = |test_start: f64| {
        let window = format!("w,{},{},{},{}", T0, test_start - purge, test_start, end);
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "walkforward", "--data", &data, "--time", "t", "--front", &run_path, "--window", &window,
                "--horizon", "900", "--periods-per-year", "365", "--validation-frac", "0.3", "--json",
            ])
            .output()
            .unwrap()
    };
    let inside = walk(seen);
    assert_eq!(code(&inside), 2, "{}", stderr(&inside));
    assert!(stderr(&inside).contains("tests rows the run trained or selected on"), "{}", stderr(&inside));
    let outside = walk(seen + purge);
    assert_eq!(code(&outside), 0, "{}", stderr(&outside));
    let report = json(&outside);
    let selection = at(&report, &["selection"]);
    assert_eq!(at(selection, &["source"]).as_str(), Some("front"));
    assert!(matches!(at(&report, &["candidates"]), Json::Arr(c) if !c.is_empty()));
}

// ---- #604: an explicit --gpu never completes on the CPU ----

#[cfg(feature = "cuda")]
fn run_gpu(data: &str, spec: &str, flag: &[&str], fault: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--no-dev-log", "--seed", "3", "--generations", "6", "--pop", "32",
        "--metric", "squared", "--json", "--quiet", "--fitness", "pnl",
    ];
    args.extend_from_slice(PNL);
    args.extend_from_slice(flag);
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(&args)
        .env("RCSR_FAULT", fault)
        .output()
        .expect("run rcsr")
}

#[cfg(feature = "cuda")]
fn has_device(o: &Output) -> bool {
    !(code(o) == 2 && stderr(o).contains("no GPU is available"))
}

#[cfg(feature = "cuda")]
#[test]
fn an_explicit_gpu_run_exits_non_zero_when_the_book_is_refused() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let o = run_gpu(&data, &spec, &["--gpu"], "pnl_book", &[]);
    if !has_device(&o) {
        return;
    }
    assert_eq!(code(&o), 6, "{}", stderr(&o));
    assert!(stderr(&o).contains("--gpu was requested but the device search failed"), "{}", stderr(&o));
    assert!(stderr(&o).contains("long_pnl fitness book"), "{}", stderr(&o));
    assert!(o.stdout.is_empty(), "no result is written: {}", String::from_utf8_lossy(&o.stdout));
}

#[cfg(feature = "cuda")]
#[test]
fn the_auto_mode_labels_the_cpu_fallback_when_the_book_is_refused() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let o = run_gpu(&data, &spec, &[], "pnl_book", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = json(&o);
    assert_eq!(at(&j, &["backend"]).as_str(), Some("cpu"));
    let why = at(&j, &["backend_reason"]).as_str().unwrap_or_default().to_string();
    assert!(why.contains("long_pnl fitness book"), "{why}");
}

#[cfg(feature = "cuda")]
#[test]
fn an_explicit_gpu_run_exits_non_zero_on_a_mid_run_batch_failure() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let o = run_gpu(&data, &spec, &["--gpu"], "batch", &["--dispatch", "device"]);
    if !has_device(&o) {
        return;
    }
    assert_eq!(code(&o), 6, "{}", stderr(&o));
    assert!(stderr(&o).contains("generation batch"), "{}", stderr(&o));
    assert!(o.stdout.is_empty(), "no result is written");
}

fn from_run_fixture() -> (String, String, f64, f64) {
    let dir = scratch();
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let data = write_csv(&dir, "a.csv", None);
    let out = run(&data, &spec, PNL);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    let run_path = dir.join("run.json");
    std::fs::write(&run_path, String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    let seen = match at(&j, &["split", "validation_time"]) {
        Json::Arr(a) => a[1].as_f64().unwrap(),
        _ => unreachable!(),
    };
    let purge = num(&j, &["split", "required_purge_seconds"]);
    (data, run_path.to_string_lossy().into_owned(), seen, purge)
}

fn walk_from_run(data: &str, run_path: &str, test_start: f64, purge: f64, extra: &[&str]) -> Output {
    let end = (T0 + 900 * N as i64) as f64;
    let window = format!("w,{},{},{},{}", T0, test_start - purge, test_start, end);
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "walkforward", "--data", data, "--time", "t", "--from-run", run_path, "--window", &window,
            "--horizon", "900", "--periods-per-year", "365", "--json",
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn from_run_refuses_in_sample_windows() {
    let (data, run_path, seen, purge) = from_run_fixture();
    let inside = walk_from_run(&data, &run_path, seen, purge, &[]);
    assert_eq!(code(&inside), 2, "{}", stderr(&inside));
    assert!(stderr(&inside).contains("tests rows the run trained or selected on"), "{}", stderr(&inside));
    let outside = walk_from_run(&data, &run_path, seen + purge, purge, &[]);
    assert_eq!(code(&outside), 0, "{}", stderr(&outside));
}

#[test]
fn from_run_labels_allowed_in_sample_windows() {
    let (data, run_path, seen, purge) = from_run_fixture();
    let inside = walk_from_run(&data, &run_path, seen, purge, &["--allow-in-sample-windows"]);
    assert_eq!(code(&inside), 0, "{}", stderr(&inside));
    let text = String::from_utf8_lossy(&inside.stdout).into_owned();
    assert!(text.contains("\"in_sample\":true"), "{text}");
    let outside = walk_from_run(&data, &run_path, seen + purge, purge, &["--allow-in-sample-windows"]);
    assert_eq!(code(&outside), 0, "{}", stderr(&outside));
    assert!(!String::from_utf8_lossy(&outside.stdout).contains("\"in_sample\":true"));
}

fn walk_table(extra: &[&str]) -> Output {
    let dir = scratch();
    let data = write_csv(&dir, "a.csv", None);
    let at_row = |i: usize| T0 + 900 * i as i64;
    let table = format!(
        "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f1,{},{},{},{},{}\n",
        at_row(0),
        at_row(250),
        at_row(180),
        at_row(260),
        at_row(399)
    );
    let table_path = dir.join("windows.csv");
    std::fs::write(&table_path, table).unwrap();
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "walkforward", "--data", &data, "--time", "t", "--target", "r", "--vars", "x1,x2",
            "--windows", &table_path.to_string_lossy(), "--windows-scheme", "expanding", "--horizon", "900",
            "--periods-per-year", "365", "--json",
        ])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn windows_table_validation_start_selects_among_repeated_expr() {
    let out = walk_table(&["--expr", "x1", "--expr", "x2"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["selection", "selected_on"]).as_str(), Some("validation"));
    let windows = match at(&j, &["windows"]) {
        Json::Arr(a) => a,
        _ => unreachable!(),
    };
    assert!(num(&windows[0], &["validation_rows"]) > 0.0);
}

#[test]
fn windows_table_accepts_select_band() {
    let out = walk_table(&["--expr", "x1", "--expr", "x2", "--select-band"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let windows = match at(&json(&out), &["windows"]) {
        Json::Arr(a) => a.clone(),
        _ => unreachable!(),
    };
    assert!(!matches!(at(&windows[0], &["band_selection"]), Json::Null));
}

#[test]
fn windows_table_still_refuses_validation_frac_and_missing_slice() {
    let both = walk_table(&["--expr", "x1", "--expr", "x2", "--validation-frac", "0.2"]);
    assert_ne!(code(&both), 0);
    assert!(stderr(&both).contains("mutually exclusive"), "{}", stderr(&both));
    let dir = scratch();
    let data = write_csv(&dir, "a.csv", None);
    let none = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "walkforward", "--data", &data, "--time", "t", "--target", "r", "--vars", "x1,x2", "--expr", "x1",
            "--expr", "x2", "--window", &format!("w,{},{},{},{}", T0, T0 + 900 * 250, T0 + 900 * 260, T0 + 900 * 399),
            "--horizon", "900", "--periods-per-year", "365",
        ])
        .output()
        .unwrap();
    assert_eq!(code(&none), 3, "{}", stderr(&none));
    assert!(stderr(&none).contains("need a validation slice"), "{}", stderr(&none));
}

// ---- #619: the extra searches of one invocation fail closed too ----

/// The second search of the process hits an injected batch fault.
#[cfg(feature = "cuda")]
fn run_second_search_fault(extra: &[&str], keep_dev_log: bool) -> Output {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir, "s.json", 0.0, "null", false);
    let mut args = vec![
        "--data", data.as_str(), "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec.as_str(), "--seed", "3", "--generations", "6", "--pop", "32",
        "--metric", "squared", "--json", "--quiet", "--fitness", "pnl", "--gpu", "--dispatch", "device",
    ];
    if !keep_dev_log {
        args.push("--no-dev-log");
    }
    args.extend_from_slice(PNL);
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(&args)
        .current_dir(&dir)
        .env("RCSR_FAULT", "batch@2")
        .output()
        .expect("run rcsr")
}

#[cfg(feature = "cuda")]
fn assert_exits_6_with_message(o: &Output) {
    assert_eq!(code(o), 6, "{}", stderr(o));
    assert!(stderr(o).contains("--gpu was requested but the device search failed"), "{}", stderr(o));
    assert!(!stderr(o).contains("panicked"), "{}", stderr(o));
}

#[cfg(feature = "cuda")]
#[test]
fn permutation_with_an_injected_fault_exits_6() {
    let o = run_second_search_fault(&["--null-runs", "1"], false);
    if !has_device(&o) {
        return;
    }
    assert_exits_6_with_message(&o);
}

#[cfg(feature = "cuda")]
#[test]
fn benchmark_with_an_injected_fault_exits_6() {
    let o = run_second_search_fault(&["--report", "benchmark", "--report-repeats", "2"], true);
    if !has_device(&o) {
        return;
    }
    assert_exits_6_with_message(&o);
}
