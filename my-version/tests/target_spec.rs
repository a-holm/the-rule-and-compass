//! Declared target spec (#254): refusal, spec-derived purge, rung, contract.

use rcsr::devlog::Json;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-target-spec-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// One synthetic row; every value is deterministic in the row index.
struct Row {
    t: i64,
    g: usize,
    sigma: f64,
    r: f64,
    x: f64,
}

impl Row {
    fn new(i: usize, t: i64, g: usize) -> Row {
        let i = i as f64;
        Row {
            t,
            g,
            sigma: 0.002 + 0.009 * frac(i * 0.618_033_988_749_894_9),
            r: 0.01 * (frac(i * 0.754_877_666_246_692_7) - 0.5),
            x: frac(i * 0.324_717_957_244_746),
        }
    }
    fn y(&self) -> f64 {
        ((self.r - 0.002) / (6.0 * self.sigma)).clamp(-1.0, 1.0)
    }
    fn yg(&self) -> f64 {
        (self.r / (6.0 * self.sigma)).clamp(-1.0, 1.0)
    }
}

/// `per_stamp` rows on every stamp of a regular grid.
fn grid(step: i64, stamps: usize, per_stamp: usize) -> Vec<Row> {
    let mut rows = Vec::new();
    for k in 0..stamps {
        for s in 0..per_stamp {
            rows.push(Row::new(rows.len(), T0 + step * k as i64, s));
        }
    }
    rows
}

fn write_csv(dir: &std::path::Path, name: &str, rows: &[Row], time_as_index: bool) -> String {
    let mut text = String::from("t,g,sigma,r,y,yg,y60,x\n");
    for (i, r) in rows.iter().enumerate() {
        let t = if time_as_index { i as i64 } else { r.t };
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            t,
            r.g,
            r.sigma,
            r.r,
            r.y(),
            r.yg(),
            r.y(),
            r.x
        ));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

struct Spec<'a> {
    column: &'a str,
    name: &'a str,
    entry: f64,
    offset: f64,
    /// `(scale column, ends_at_offset_seconds)`; `None` is the constant 1.
    scale: Option<(&'a str, f64)>,
    clip: Option<(f64, f64)>,
    return_column: Option<&'a str>,
}

impl<'a> Spec<'a> {
    fn net(column: &'a str) -> Spec<'a> {
        Spec {
            column,
            name: "net_target",
            entry: 900.0,
            offset: 0.002,
            scale: Some(("sigma", 900.0)),
            clip: Some((-1.0, 1.0)),
            return_column: Some("r"),
        }
    }
    fn gross(column: &'a str) -> Spec<'a> {
        Spec {
            offset: 0.0,
            name: "gross_target",
            ..Spec::net(column)
        }
    }
    fn raw(column: &'a str) -> Spec<'a> {
        Spec {
            column,
            name: "raw_target",
            entry: 900.0,
            offset: 0.0,
            scale: None,
            clip: None,
            return_column: Some("r"),
        }
    }
    fn json(&self) -> String {
        let scale = match self.scale {
            Some((c, ends)) => format!(
                "{{\"kind\":\"column\",\"column\":\"{c}\",\"multiplier\":6.0,\"window_seconds\":86400,\"ends_at_offset_seconds\":{ends}}}"
            ),
            None => "{\"kind\":\"constant\",\"value\":1.0}".to_string(),
        };
        let clip = match self.clip {
            Some((lo, hi)) => format!("[{lo:?},{hi:?}]"),
            None => "null".into(),
        };
        let (cost, in_label) = if self.offset != 0.0 {
            (format!("{:?}", 20.0), "true")
        } else {
            ("null".into(), "false")
        };
        let cols = match self.scale {
            Some((c, _)) => format!("[\"{c}\"]"),
            None => "[]".into(),
        };
        let ret = match self.return_column {
            Some(r) => format!("\"{r}\""),
            None => "null".into(),
        };
        format!(
            "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"{}\",\"name\":\"{}\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":{ret},\"return_units\":\"log\",\"horizon_seconds\":3600,\"entry_offset_seconds\":{:?},\"delay_seconds\":0,\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":{:?},\"scale\":{scale},\"clip\":{clip},\"units\":\"log_return\",\"cost_bps\":{cost},\"cost_in_label\":{in_label},\"construction_columns\":{cols},\"note\":null}}]}}",
            self.column, self.name, self.entry, self.offset
        )
    }
    fn write(&self, dir: &std::path::Path, file: &str) -> String {
        let p = dir.join(file);
        std::fs::write(&p, self.json()).unwrap();
        p.to_string_lossy().into_owned()
    }
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
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

/// A tiny search run with a spec and `t >= B` selectors.
fn run(data: &str, spec: &str, target: &str, vars: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", target, "--vars", vars, "--time", "t", "--target-spec", spec,
        "--cpu", "--no-dev-log", "--seed", "1", "--generations", "1", "--pop", "8", "--metric",
        "squared", "--json", "--quiet",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

fn select(train_before: i64, test_from: i64) -> [String; 4] {
    [
        "--train-select".into(),
        format!("t < {train_before}"),
        "--test-select".into(),
        format!("t >= {test_from}"),
    ]
}

fn with_select<'a>(sel: &'a [String; 4], more: &[&'a str]) -> Vec<&'a str> {
    let mut v: Vec<&str> = sel.iter().map(|s| s.as_str()).collect();
    v.extend_from_slice(more);
    v
}

fn m0(sigma: f64) -> f64 {
    (-0.002 / (6.0 * sigma)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Refusal
// ---------------------------------------------------------------------------

#[test]
fn structural_target_is_refused_with_the_scale_in_vars() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let spec = Spec::net("y").write(&dir, "s.json");
    let out = run(&data, &spec, "y", "sigma,x", &[]);
    assert_eq!(code(&out), 5, "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("D13c"), "{err}");
    assert!(err.contains("clip(-0.002/(6*sigma),-1,1)"), "{err}");
}

#[test]
fn structural_target_is_refused_without_the_scale_in_vars() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let spec = Spec::net("y").write(&dir, "s.json");
    let out = run(&data, &spec, "y", "x", &[]);
    assert_eq!(code(&out), 5, "{}", stderr(&out));
    assert!(stderr(&out).contains("D13c"));
}

#[test]
fn override_runs_and_reports_the_rung() {
    let dir = scratch();
    let rows = grid(900, 120, 1);
    let data = write_csv(&dir, "d.csv", &rows, false);
    let spec = Spec::net("y").write(&dir, "s.json");
    let a = T0 + 900 * 80;
    let b = T0 + 900 * 90;
    let sel = select(a, b);
    let out = run(
        &data,
        &spec,
        "y",
        "sigma,x",
        &with_select(&sel, &["--allow-target-structure"]),
    );
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "check", "status"]).as_str(), Some("overridden"));
    assert_eq!(at(&j, &["target_spec", "structural_term"]), &Json::Bool(true));
    assert_eq!(
        at(&j, &["target_spec", "null_formula"]).as_str(),
        Some("clip(-0.002/(6*sigma),-1,1)")
    );
    // The same formula on the test rows, computed here.
    let test: Vec<&Row> = rows.iter().filter(|r| r.t >= b).collect();
    let mse = test.iter().map(|r| (r.y() - m0(r.sigma)).powi(2)).sum::<f64>() / test.len() as f64;
    let got = num(&j, &["baselines", "target_structure", "test_error"]);
    assert!((got - mse).abs() <= 1e-12, "{got} vs {mse}");
    assert_eq!(num(&j, &["baselines", "target_structure", "params"]), 0.0);
    match at(&j, &["target_spec", "check", "proxies"]) {
        Json::Arr(p) => assert!(!p.is_empty()),
        other => panic!("proxies: {other:?}"),
    }
}

#[test]
fn future_return_in_vars_is_refused_even_with_the_override() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let spec = Spec::net("y").write(&dir, "s.json");
    let out = run(&data, &spec, "y", "r,x", &["--allow-target-structure"]);
    assert_eq!(code(&out), 5, "{}", stderr(&out));
    assert!(stderr(&out).contains("D13a"), "{}", stderr(&out));
}

#[test]
fn look_ahead_scale_is_refused() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let s = Spec {
        scale: Some(("sigma", 4500.0)),
        ..Spec::gross("yg")
    };
    let spec = s.write(&dir, "s.json");
    let out = run(&data, &spec, "yg", "x", &["--allow-target-structure"]);
    assert_eq!(code(&out), 5, "{}", stderr(&out));
    assert!(stderr(&out).contains("D13b"), "{}", stderr(&out));
}

#[test]
fn gross_scaled_target_with_the_scale_in_vars_is_allowed() {
    let dir = scratch();
    let rows = grid(900, 120, 1);
    let data = write_csv(&dir, "d.csv", &rows, false);
    let spec = Spec::gross("yg").write(&dir, "s.json");
    let b = T0 + 900 * 90;
    let sel = select(T0 + 900 * 80, b);
    let out = run(&data, &spec, "yg", "sigma,x", &with_select(&sel, &[]));
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "structural_term"]), &Json::Bool(false));
    assert_eq!(at(&j, &["target_spec", "check", "status"]).as_str(), Some("ok"));
    match at(&j, &["target_spec", "check", "construction_in_vars"]) {
        Json::Arr(a) => assert_eq!(a.iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>(), ["sigma"]),
        other => panic!("{other:?}"),
    }
    let test: Vec<&Row> = rows.iter().filter(|r| r.t >= b).collect();
    let zero = test.iter().map(|r| r.yg().powi(2)).sum::<f64>() / test.len() as f64;
    let got = num(&j, &["baselines", "target_structure", "test_error"]);
    assert!((got - zero).abs() <= 1e-12, "{got} vs {zero}");
}

// ---------------------------------------------------------------------------
// Purge
// ---------------------------------------------------------------------------

#[test]
fn purge_is_derived_from_the_spec_not_the_name() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(3600, 200, 1), false);
    let spec = Spec {
        name: "y60m",
        ..Spec::gross("y60")
    }
    .write(&dir, "s.json");
    let a = T0 + 3600 * 100;
    let sel = select(a, a + 216_000);
    let out = run(&data, &spec, "y60", "x", &with_select(&sel, &[]));
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(num(&j, &["split", "required_purge_seconds"]), 4500.0);
    assert_eq!(num(&j, &["split", "purge_seconds"]), 219_600.0);
    assert_eq!(num(&j, &["split", "purged_rows", "test_boundary"]), 0.0);
    assert_eq!(at(&j, &["split", "purge_source"]).as_str(), Some("target_spec"));
}

fn adjacent(entry: f64, expect_steps: usize, required: f64) {
    let per_stamp = 2;
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 100, per_stamp), false);
    let spec = Spec {
        entry,
        ..Spec::raw("r")
    }
    .write(&dir, "s.json");
    let a = T0 + 900 * 60;
    let sel = select(a, a);
    let out = run(&data, &spec, "r", "x", &with_select(&sel, &[]));
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(num(&j, &["split", "required_purge_seconds"]), required);
    assert_eq!(
        num(&j, &["split", "purged_rows", "test_boundary"]),
        (expect_steps * per_stamp) as f64
    );
    assert_eq!(num(&j, &["split", "purge_seconds"]), required);
    assert_eq!(num(&j, &["train_rows"]), ((60 - expect_steps) * per_stamp) as f64);
}

#[test]
fn adjacent_selectors_are_purged_to_the_label_end() {
    adjacent(900.0, 4, 4500.0);
}

#[test]
fn entry_offset_comes_from_the_spec() {
    adjacent(3600.0, 7, 7200.0);
}

#[test]
fn validation_boundary_is_purged_on_a_panel() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 200, 3), false);
    let spec = Spec::raw("r").write(&dir, "s.json");
    let out = run(
        &data,
        &spec,
        "r",
        "x",
        &["--test-frac", "0.25", "--validation-frac", "0.2", "--gap", "0"],
    );
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(num(&j, &["split", "validation_purge_seconds"]), 4500.0);
    assert!(num(&j, &["split", "purged_rows", "validation_boundary"]) >= 12.0);
    assert!(num(&j, &["split", "purge_seconds"]) >= 4500.0);
}

#[test]
fn misaligned_temporal_group_series_are_purged_across_symbols() {
    let dir = scratch();
    // Group 0 covers steps 0..100, group 1 covers steps 40..140: the cut
    // times differ by 40 steps (10 hours).
    let mut rows = Vec::new();
    for k in 0..100usize {
        let mut r = Row::new(rows.len(), T0 + 900 * k as i64, 0);
        r.g = 0;
        rows.push(r);
    }
    for k in 40..140usize {
        let mut r = Row::new(rows.len(), T0 + 900 * k as i64, 1);
        r.g = 1;
        rows.push(r);
    }
    let data = write_csv(&dir, "d.csv", &rows, false);
    let spec = Spec::raw("r").write(&dir, "s.json");
    let out = run(
        &data,
        &spec,
        "r",
        "x",
        &["--split", "temporal-group", "--group-by", "g", "--test-frac", "0.25"],
    );
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    // Each group tests its last 25 rows; train is the first 75.
    let train: Vec<i64> = (0..75)
        .map(|k| k as i64)
        .chain((40..115).map(|k| k as i64))
        .collect();
    let test: Vec<i64> = (75..100).chain(115..140).collect();
    let dropped = train
        .iter()
        .filter(|&&s| test.iter().any(|&u| (s - u).abs() * 900 < 4500))
        .count();
    let within_group_only = 4 + 4;
    assert!(dropped > within_group_only);
    assert_eq!(
        num(&j, &["split", "purged_rows", "test_boundary"]),
        dropped as f64
    );
}

#[test]
fn a_spec_without_an_epoch_time_column_is_a_usage_error() {
    let dir = scratch();
    let rows = grid(900, 80, 1);
    let spec = Spec::raw("r").write(&dir, "s.json");
    // The time column is a row index.
    let data = write_csv(&dir, "idx.csv", &rows, true);
    let out = run(&data, &spec, "r", "x", &[]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("target spec"), "{}", stderr(&out));
    // No time column at all.
    let data = write_csv(&dir, "d.csv", &rows, false);
    let out = rcsr(&[
        "--data", &data, "--target", "r", "--vars", "x", "--target-spec", &spec, "--cpu",
        "--no-dev-log", "--generations", "1", "--pop", "8", "--quiet",
    ]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
}

// ---------------------------------------------------------------------------
// Contract and reports
// ---------------------------------------------------------------------------

const SINE_PARETO: &str = r#""pareto":[{"rank":1,"complexity":1,"k_constants":0,"train_error":0.4355307063142365,"test_error":0.352025818822857,"aicc":-43.04796241506922,"bic":-43.04796241506922,"durbin_watson":2.2869337818583646,"overfit_index":0.8082686564213671,"fold_errors":[0.352025818822857],"bootstrap_error":null,"constant_errors":null,"stability":null,"expression":"(0.5627 * (-0.1096 + x))"},{"rank":2,"complexity":2,"k_constants":0,"train_error":0.2859967384426627,"test_error":0.24457430839771904,"aicc":-95.11309556998992,"bic":-95.11309556998992,"durbin_watson":2.016633531387803,"overfit_index":0.855164676805403,"fold_errors":[0.24457430839771904],"bootstrap_error":null,"constant_errors":null,"stability":null,"expression":"tanh((0.5627 * (-0.1096 + x)))"},{"rank":3,"complexity":3,"k_constants":0,"train_error":0.27134953322397803,"test_error":0.23532615071551435,"aicc":-101.96680418403614,"bic":-101.96680418403614,"durbin_watson":1.969483305432346,"overfit_index":0.867243617188281,"fold_errors":[0.23532615071551435],"bootstrap_error":null,"constant_errors":null,"stability":null,"expression":"sin(sin((0.5627 * (-0.1096 + x))))"},{"rank":4,"complexity":4,"k_constants":1,"train_error":0.06659842155744641,"test_error":0.05969917606905527,"aicc":-219.07984822505742,"bic":-217.36620899110105,"durbin_watson":2.5255396627050257,"overfit_index":0.8964052701693538,"fold_errors":[0.05969917606905527],"bootstrap_error":null,"constant_errors":{"dof":44,"ill_conditioned":false,"constants":[{"value":0.5825087953039945,"std_error":0.006087389058424975,"relative":0.010450295527723566}]},"stability":null,"expression":"sin((0.966 * (-0.1096 + x)))"}]}"#;

#[test]
fn no_spec_keeps_the_default_run_bit_identical() {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sine.csv");
    let out = rcsr(&[
        "--data", data, "--target", "y", "--vars", "x", "--cpu", "--no-dev-log", "--seed", "7",
        "--generations", "15", "--pop", "40", "--json", "--quiet",
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let i = text.find("\"pareto\":").expect("pareto");
    assert_eq!(text[i..].trim_end(), SINE_PARETO);
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec"]), &Json::Null);
    assert_eq!(at(&j, &["split", "required_purge_seconds"]), &Json::Null);
    assert_eq!(at(&j, &["baselines", "target_structure"]), &Json::Null);
}

#[test]
fn sidecar_is_loaded_and_the_flag_wins() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let sidecar = Spec::gross("yg");
    let sidecar_text = sidecar.json();
    std::fs::write(dir.join("d.targets.json"), &sidecar_text).unwrap();
    let args = [
        "--data", &data, "--target", "yg", "--vars", "x", "--time", "t", "--cpu", "--no-dev-log",
        "--seed", "1", "--generations", "1", "--pop", "8", "--json", "--quiet",
    ];
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "source"]).as_str(), Some("sidecar"));
    assert_eq!(
        at(&j, &["target_spec", "file_sha256"]).as_str(),
        Some(rcsr::manifest::sha256_hex(sidecar_text.as_bytes()).as_str())
    );
    assert_eq!(num(&j, &["target_spec", "required_purge_seconds"]), 4500.0);
    assert_eq!(num(&j, &["target_spec", "horizon_seconds"]), 3600.0);
    assert_eq!(
        at(&j, &["target_spec", "window_seconds"]),
        &Json::Arr(vec![Json::Num(900.0), Json::Num(4500.0)])
    );
    assert_eq!(at(&j, &["target_spec", "cost_in_label"]), &Json::Bool(false));
    assert_eq!(at(&j, &["target_spec", "long_only"]), &Json::Bool(true));
    assert_eq!(at(&j, &["target_spec", "units"]).as_str(), Some("log_return"));

    let flag = Spec {
        entry: 3600.0,
        ..Spec::gross("yg")
    }
    .write(&dir, "other.json");
    let mut with_flag = args.to_vec();
    with_flag.extend(["--target-spec", &flag]);
    let out = rcsr(&with_flag);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "source"]).as_str(), Some("file"));
    assert_eq!(num(&j, &["target_spec", "required_purge_seconds"]), 7200.0);
}

#[test]
fn a_spec_without_an_entry_for_the_target_is_a_usage_error() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let spec = Spec::gross("yg").write(&dir, "s.json");
    let out = run(&data, &spec, "y", "x", &[]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    let bad = dir.join("bad.json");
    std::fs::write(&bad, "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{}]}").unwrap();
    let out = run(&data, bad.to_str().unwrap(), "y", "x", &[]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    let out = run(&data, "/nonexistent/spec.json", "y", "x", &[]);
    assert_eq!(code(&out), 3, "{}", stderr(&out));
}

#[test]
fn text_run_prints_the_target_purge_and_rung_lines() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(3600, 200, 1), false);
    let spec = Spec {
        name: "y60m",
        ..Spec::net("y60")
    }
    .write(&dir, "s.json");
    let a = T0 + 3600 * 100;
    let sel = select(a, a + 216_000);
    let mut args = vec![
        "--data", &data, "--target", "y60", "--vars", "x", "--time", "t", "--target-spec", &spec,
        "--allow-target-structure", "--cpu", "--no-dev-log", "--generations", "1", "--pop", "8",
        "--metric", "squared",
    ];
    args.extend(sel.iter().map(|s| s.as_str()));
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("# target: y60 is y60m | window [t+900s, t+4500s] | horizon 3600s | units log_return | cost 20bp in label | spec "), "{err}");
    assert!(err.contains("# purge: required 4500s (entry 900s + delay 0s + horizon 3600s) | realised train->test 219600s | fit->validation n/a | purged 0+0 row(s)"), "{err}");
    assert!(err.contains("# target structure (0 params): clip(-0.002/(6*sigma),-1,1) train="), "{err}");
    assert!(err.contains("# WARNING: structural target term clip(-0.002/(6*sigma),-1,1) allowed by --allow-target-structure"), "{err}");
}

#[test]
fn eval_expr_carries_the_rung_and_spec() {
    let dir = scratch();
    let rows = grid(900, 120, 1);
    let data = write_csv(&dir, "d.csv", &rows, false);
    let spec = Spec::gross("yg").write(&dir, "s.json");
    let b = T0 + 900 * 90;
    let sel = select(T0 + 900 * 80, b);
    let mut args = vec![
        "--data", &data, "--target", "yg", "--vars", "sigma,x", "--time", "t", "--target-spec",
        &spec, "--cpu", "--no-dev-log", "--metric", "squared", "--eval-expr", "0", "--json",
        "--quiet",
    ];
    args.extend(sel.iter().map(|s| s.as_str()));
    let out = rcsr(&args);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    let test: Vec<&Row> = rows.iter().filter(|r| r.t >= b).collect();
    let zero = test.iter().map(|r| r.yg().powi(2)).sum::<f64>() / test.len() as f64;
    assert!((num(&j, &["target_structure", "test_error"]) - zero).abs() <= 1e-12);
    assert!((num(&j, &["eval", "test_error"]) - zero).abs() <= 1e-12);
    assert_eq!(at(&j, &["target_spec", "name"]).as_str(), Some("gross_target"));
}

#[test]
fn schema_documents_the_target_spec_contract() {
    let out = rcsr(&["schema"]);
    assert_eq!(code(&out), 0);
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    for needle in [
        "--target-spec",
        "--allow-target-structure",
        "rcsr-target-spec/1",
        "D13a",
        "D13b",
        "D13c",
        "required_purge_seconds",
        "purged_rows",
        "target_structure",
    ] {
        assert!(text.contains(needle), "schema lacks {needle}");
    }
    let j = Json::parse(text.trim()).expect("schema is json");
    let fields = at(&j, &["target_spec", "fields"]);
    let names: Vec<String> = match fields {
        Json::Arr(a) => a
            .iter()
            .map(|f| f.get("name").and_then(|n| n.as_str()).unwrap().to_string())
            .collect(),
        other => panic!("{other:?}"),
    };
    for f in [
        "column", "name", "kind", "deprecated", "return_column", "return_units", "horizon_seconds",
        "entry_offset_seconds", "delay_seconds", "entry_price", "exit_price", "offset", "scale",
        "clip", "units", "cost_bps", "cost_in_label", "construction_columns", "note",
    ] {
        assert!(names.iter().any(|n| n == f), "field {f} missing: {names:?}");
    }
}

#[test]
fn audit_runs_the_target_structure_detector() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let spec = Spec::net("y").write(&dir, "s.json");
    let out = rcsr(&[
        "audit", "--data", &data, "--target", "y", "--time", "t", "--vars", "sigma,x",
        "--target-spec", &spec,
    ]);
    assert_eq!(code(&out), 5, "{}{}", stderr(&out), String::from_utf8_lossy(&out.stdout));
    let all = format!("{}{}", stderr(&out), String::from_utf8_lossy(&out.stdout));
    assert!(all.contains("D13c"), "{all}");
    let ok = Spec::gross("yg").write(&dir, "g.json");
    let out = rcsr(&[
        "audit", "--data", &data, "--target", "yg", "--time", "t", "--vars", "x", "--target-spec",
        &ok,
    ]);
    assert_eq!(code(&out), 0, "{}{}", stderr(&out), String::from_utf8_lossy(&out.stdout));
}

// ---------------------------------------------------------------------------
// #345: provenance hardening
// ---------------------------------------------------------------------------

#[test]
fn the_spec_note_reaches_run_json() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 80, 1), false);
    let note = "survivor-biased universe, not point-in-time; \"quoted\"";
    let text = Spec::gross("yg")
        .json()
        .replace("\"note\":null", &format!("\"note\":\"{}\"", note.replace('"', "\\\"")));
    assert!(text.contains("survivor-biased"), "fixture must carry the note");
    let spec = dir.join("noted.json");
    std::fs::write(&spec, text).unwrap();
    let out = run(&data, spec.to_str().unwrap(), "yg", "x", &[]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "note"]).as_str(), Some(note));
}

/// Daily epoch rows: fit rows end before the lockbox, the validation slice
/// (last 20% of the training period) sits inside it, the test block follows it.
fn lockbox_validation_csv(dir: &std::path::Path) -> String {
    let l = rcsr::lockbox::Lockbox::crypto_2026h2();
    let day = 86_400i64;
    let mut text = String::from("t,x,y\n");
    for i in 0..100i64 {
        let t = if i < 56 {
            l.start - (56 - i) * day
        } else if i < 70 {
            l.start + (i - 56) * day
        } else {
            l.end + (i - 70) * day
        };
        text.push_str(&format!("{},{},{}\n", t, (i as f64).sin(), (i as f64 * 0.3).cos()));
    }
    let p = dir.join("lockbox_validation.csv");
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn lockbox_run(data: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x", "--time", "t", "--cpu", "--no-dev-log",
        "--seed", "1", "--generations", "1", "--pop", "8", "--quiet", "--json",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

#[test]
fn a_validation_slice_in_the_lockbox_is_refused_like_a_test_block() {
    let dir = scratch();
    let data = lockbox_validation_csv(&dir);
    let l = rcsr::lockbox::Lockbox::crypto_2026h2();

    // Reference: a test block that overlaps the lockbox.
    let block = lockbox_run(&data, &["--train-select", &format!("t < {}", l.start), "--test-select", &format!("t >= {}", l.start)]);
    assert_eq!(code(&block), 3, "{}", stderr(&block));
    assert!(stderr(&block).contains("test split: "), "{}", stderr(&block));

    let out = lockbox_run(&data, &["--test-frac", "0.3", "--validation-frac", "0.2"]);
    assert_eq!(code(&out), code(&block), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("validation split: 14 row(s) in [")
            && err.contains(") overlap the lockbox [")
            && err.contains("--allow-lockbox"),
        "{err}"
    );
    assert!(!err.contains("training split") && !err.contains("test split: "), "{err}");

    let allowed = lockbox_run(
        &data,
        &["--test-frac", "0.3", "--validation-frac", "0.2", "--allow-lockbox", "--lockbox", "none"],
    );
    assert_eq!(code(&allowed), 0, "{}", stderr(&allowed));

    // Without a validation slice the same rows are fit rows, refused as such.
    let none = lockbox_run(&data, &["--test-frac", "0.3"]);
    assert_eq!(code(&none), 3, "{}", stderr(&none));
}

fn resume_args<'a>(data: &'a str, spec: Option<&'a str>, cp: &'a str, sel: &'a [String; 4]) -> Vec<&'a str> {
    let mut a = vec![
        "--data", data, "--target", "yg", "--vars", "x", "--time", "t", "--cpu", "--no-dev-log",
        "--seed", "1", "--generations", "2", "--pop", "8", "--metric", "squared", "--json",
        "--quiet", "--resume", cp,
    ];
    if let Some(s) = spec {
        a.extend(["--target-spec", s]);
    }
    a.extend(sel.iter().map(|s| s.as_str()));
    a
}

#[test]
fn resume_with_a_different_target_spec_is_refused_naming_both_hashes() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 120, 1), false);
    let a = Spec::gross("yg").write(&dir, "a.json");
    let b = Spec {
        entry: 3600.0,
        ..Spec::gross("yg")
    }
    .write(&dir, "b.json");
    let cp = dir.join("cp.txt").to_string_lossy().into_owned();
    let sel = select(T0 + 900 * 60, T0 + 900 * 90);

    let saved = run(&data, &a, "yg", "x", &with_select(&sel, &["--save", &cp]));
    assert_eq!(code(&saved), 0, "{}", stderr(&saved));
    let sha_a = at(&json(&saved), &["target_spec", "sha256"]).as_str().unwrap().to_string();
    let other = run(&data, &b, "yg", "x", &with_select(&sel, &[]));
    assert_eq!(code(&other), 0, "{}", stderr(&other));
    let sha_b = at(&json(&other), &["target_spec", "sha256"]).as_str().unwrap().to_string();
    assert_ne!(sha_a, sha_b);
    let text = std::fs::read_to_string(&cp).unwrap();
    assert!(text.contains(&format!("\ntarget_spec {sha_a}\n")), "{text}");

    // The purge differs (4500 s vs 7200 s) but removes no row: the data
    // fingerprints are equal, so only the explicit spec check can refuse.
    let refused = rcsr(&resume_args(&data, Some(&b), &cp, &sel));
    assert_eq!(code(&refused), 2, "{}", stderr(&refused));
    let err = stderr(&refused);
    assert!(err.contains(&sha_a) && err.contains(&sha_b), "{err}");
    assert!(err.contains("target spec mismatch"), "{err}");

    let dropped = rcsr(&resume_args(&data, None, &cp, &sel));
    assert_eq!(code(&dropped), 2, "{}", stderr(&dropped));
    assert!(stderr(&dropped).contains(&sha_a) && stderr(&dropped).contains("none"), "{}", stderr(&dropped));

    let same = rcsr(&resume_args(&data, Some(&a), &cp, &sel));
    assert_eq!(code(&same), 0, "{}", stderr(&same));
}

#[test]
fn a_run_without_a_spec_writes_no_target_spec_record() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &grid(900, 120, 1), false);
    let cp = dir.join("cp.txt").to_string_lossy().into_owned();
    let out = rcsr(&[
        "--data", &data, "--target", "yg", "--vars", "x", "--time", "t", "--cpu", "--no-dev-log",
        "--seed", "1", "--generations", "2", "--pop", "8", "--json", "--quiet", "--save", &cp,
    ]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(!std::fs::read_to_string(&cp).unwrap().contains("target_spec"));
    let again = rcsr(&[
        "--data", &data, "--target", "yg", "--vars", "x", "--time", "t", "--cpu", "--no-dev-log",
        "--seed", "1", "--generations", "2", "--pop", "8", "--json", "--quiet", "--resume", &cp,
    ]);
    assert_eq!(code(&again), 0, "{}", stderr(&again));
}

// ---------------------------------------------------------------------------
// Long-only label columns from `rcsr enrich`
// ---------------------------------------------------------------------------

/// A labelled store from `rcsr enrich --label long:3`; returns its path.
fn labelled_store(dir: &std::path::Path) -> String {
    let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
    for sym in ["AAA", "BBB"] {
        for i in 0..260usize {
            let x = i as f64;
            let o = 100.0 + 4.0 * (x * 0.31).sin() + 0.02 * x;
            let c = 100.0 + 4.0 * (x * 0.31 + 0.5).sin() + 0.02 * x;
            text.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                T0 + 3600 * i as i64,
                sym,
                o,
                o.max(c) + 0.4,
                o.min(c) - 0.4,
                c,
                10.0 + (i % 7) as f64
            ));
        }
    }
    let csv = dir.join("bars.csv");
    std::fs::write(&csv, text).unwrap();
    let out = dir.join("bars.rcsb").to_string_lossy().into_owned();
    let o = rcsr(&[
        "enrich", "--data", csv.to_str().unwrap(), "--out", &out, "--features", "r1",
        "--label", "long:3,name=h3,vol=20,fee=5", "--bar-seconds", "3600", "--time", "ts",
        "--symbol", "symbol", "--cpu",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    out
}

fn run_labelled(data: &str, vars: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "lbl_h3_y", "--time", "ts", "--fill", "drop", "--cpu",
        "--no-dev-log", "--seed", "1", "--generations", "1", "--pop", "8", "--metric", "squared",
        "--json", "--quiet",
    ];
    args.extend_from_slice(vars);
    rcsr(&args)
}

#[test]
fn run_json_echoes_the_label_report_of_an_enriched_target() {
    let dir = scratch();
    let data = labelled_store(&dir);
    let o = run_labelled(&data, &["--vars", "r1"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = json(&o);
    let lr = at(&j, &["target_spec", "label_report"]);
    assert_eq!(lr.get("name").and_then(|v| v.as_str()), Some("h3"));
    let hash = lr.get("label_hash").and_then(|v| v.as_str()).unwrap().to_string();
    assert_eq!(hash.len(), 64);
    let sidecar = std::fs::read_to_string(dir.join("bars.labels.json")).unwrap();
    assert!(sidecar.contains(&hash), "run.json hash matches the enrich sidecar");
    assert_eq!(num(lr, &["top_abs_moves", "n"]), 10.0);
    match at(lr, &["top_abs_moves", "moves"]) {
        Json::Arr(a) => {
            assert_eq!(a.len(), 10);
            assert!(a[0].get("symbol").and_then(|v| v.as_str()).is_some());
            assert!(a[0].get("time").and_then(|v| v.as_f64()).is_some());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn label_columns_are_never_inputs_named_or_default() {
    let dir = scratch();
    let data = labelled_store(&dir);
    for bad in ["r1,lbl_h3_fwd", "r1,lbl_h3_weight", "r1,lbl_h3_action"] {
        let o = run_labelled(&data, &["--vars", bad]);
        assert_eq!(code(&o), 5, "{bad}: {}", stderr(&o));
        assert!(stderr(&o).contains("D13a"), "{bad}: {}", stderr(&o));
    }
    let o = run_labelled(&data, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = String::from_utf8_lossy(&o.stdout).into_owned();
    let j = Json::parse(text.trim()).unwrap();
    match at(&j, &["variables"]) {
        Json::Arr(a) => assert!(a.iter().all(|v| !v.as_str().unwrap().starts_with("lbl_")), "{a:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn require_clean_inputs_refuses_a_construction_column_in_vars() {
    let dir = scratch();
    let rows = grid(900, 120, 1);
    let data = write_csv(&dir, "d.csv", &rows, false);
    let spec = Spec::gross("yg").write(&dir, "s.json");
    let b = T0 + 900 * 90;
    let sel = select(T0 + 900 * 80, b);
    let out = run(&data, &spec, "yg", "sigma,x", &with_select(&sel, &["--require-clean-inputs"]));
    assert_eq!(code(&out), 5, "{}", stderr(&out));
    assert!(stderr(&out).contains("sigma"), "{}", stderr(&out));
    let clean = run(&data, &spec, "yg", "x", &with_select(&sel, &["--require-clean-inputs"]));
    assert_eq!(code(&clean), 0, "{}", stderr(&clean));
    let off = run(&data, &spec, "yg", "sigma,x", &with_select(&sel, &[]));
    assert_eq!(code(&off), 0, "{}", stderr(&off));
}
