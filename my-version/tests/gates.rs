//! Validation gates (#257): selection, verdict, flags, determinism.

use rcsr::devlog::Json;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-gates-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

#[derive(Clone)]
struct Row {
    t: i64,
    sigma: f64,
    r: f64,
    x: f64,
    y: f64,
}

/// `per` rows on each of `stamps` timestamps. `y = 0.8 x + noise` when
/// `signal`, else the constructed label `clip((r - 0.002) / (6 sigma))` with
/// `r` independent of `sigma`.
fn panel(stamps: usize, per: usize, signal: bool) -> Vec<Row> {
    let mut rows = Vec::new();
    for k in 0..stamps {
        for _ in 0..per {
            let i = rows.len() as f64;
            let sigma = 0.002 + 0.009 * frac(i * 0.618_033_988_749_894_9);
            let r = 0.01 * (frac(i * 0.754_877_666_246_692_7) - 0.5);
            let x = frac(i * 0.324_717_957_244_746);
            let y = if signal {
                0.8 * x + 0.1 * (frac(i * 0.569_840_290_998_053_3) - 0.5)
            } else {
                ((r - 0.002) / (6.0 * sigma)).clamp(-1.0, 1.0)
            };
            rows.push(Row { t: T0 + STEP * k as i64, sigma, r, x, y });
        }
    }
    rows
}

fn write_csv(dir: &std::path::Path, name: &str, rows: &[Row]) -> String {
    let mut text = String::from("t,sigma,r,x,y\n");
    for r in rows {
        text.push_str(&format!("{},{},{},{},{}\n", r.t, r.sigma, r.r, r.x, r.y));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn spec_file(dir: &std::path::Path) -> String {
    let json = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"net_target\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\"horizon_seconds\":3600,\"entry_offset_seconds\":900.0,\"delay_seconds\":0,\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.002,\"scale\":{\"kind\":\"column\",\"column\":\"sigma\",\"multiplier\":6.0,\"window_seconds\":86400,\"ends_at_offset_seconds\":900},\"clip\":[-1.0,1.0],\"units\":\"log_return\",\"cost_bps\":20.0,\"cost_in_label\":true,\"construction_columns\":[\"sigma\"],\"note\":null}]}";
    let p = dir.join("spec.json");
    std::fs::write(&p, json).unwrap();
    p.to_string_lossy().into_owned()
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

fn pareto(j: &Json) -> &Vec<Json> {
    match at(j, &["pareto"]) {
        Json::Arr(a) => a,
        other => panic!("pareto: {other:?}"),
    }
}

fn search(data: &str, vars: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", vars, "--cpu", "--no-dev-log", "--seed", "1",
        "--generations", "6", "--pop", "40", "--metric", "squared", "--json", "--quiet",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

const TEMPORAL: &[&str] = &["--time", "t", "--split", "temporal", "--test-frac", "0.25"];

#[test]
fn require_validation_refuses_without_a_slice() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(120, 1, true));
    let mut extra = TEMPORAL.to_vec();
    extra.push("--require-validation");
    let out = search(&data, "x", &extra);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("--validation-frac"), "{}", stderr(&out));
}

#[test]
fn time_ordered_run_without_a_slice_is_unvalidated() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(120, 1, true));
    let out = search(&data, "x", TEMPORAL);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["selection", "verdict"]).as_str(), Some("unvalidated"));
    assert_eq!(at(&j, &["selection", "selected_rank"]), &Json::Null);
    let warnings = stderr(&out).lines().filter(|l| l.starts_with("warning: time-ordered")).count();
    assert_eq!(warnings, 1, "{}", stderr(&out));
    for m in pareto(&j) {
        assert_eq!(at(m, &["validated"]), &Json::Bool(false));
        assert!(m.get("beats_mean_test").is_some());
    }
}

#[test]
fn random_split_default_run_has_no_gate_fields() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(120, 1, true));
    let out = search(&data, "x", &["--split", "random"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert!(j.get("selection").is_none());
    assert!(j.get("noise").is_none());
    for m in pareto(&j) {
        assert!(m.get("gates").is_none());
        assert!(m.get("comparisons").is_none());
        let ce = m.get("constant_errors").unwrap();
        if ce != &Json::Null {
            assert!(ce.get("rank").is_none());
        }
    }
    assert!(at(&j, &["validation_hv"]).get("complexity_min").is_some());
    assert!(at(&j, &["validation_hv"]).get("complexity_max").is_some());
}

#[test]
fn shuffling_the_test_block_leaves_the_selection_unchanged() {
    let dir = scratch();
    let rows = panel(160, 2, true);
    let test_from = T0 + STEP * 120;
    let mut shuffled = rows.clone();
    let idx: Vec<usize> = (0..shuffled.len()).filter(|&i| shuffled[i].t >= test_from).collect();
    let ys: Vec<f64> = idx.iter().map(|&i| rows[i].y).collect();
    for (k, &i) in idx.iter().enumerate() {
        shuffled[i].y = ys[(k + 7) % ys.len()];
    }
    let a = write_csv(&dir, "a.csv", &rows);
    let b = write_csv(&dir, "b.csv", &shuffled);
    let train_before = format!("t < {}", T0 + STEP * 120);
    let test_expr = format!("t >= {test_from}");
    let extra = [
        "--time", "t", "--train-select", train_before.as_str(), "--test-select", test_expr.as_str(),
        "--validation-frac", "0.25",
    ];
    let ja = json(&search(&a, "x", &extra));
    let jb = json(&search(&b, "x", &extra));
    assert_eq!(at(&ja, &["selection"]), at(&jb, &["selection"]));
    assert_eq!(at(&ja, &["noise"]), at(&jb, &["noise"]));
    assert_ne!(at(&ja, &["selection", "verdict"]).as_str(), Some("unvalidated"));
    let (pa, pb) = (pareto(&ja), pareto(&jb));
    assert_eq!(pa.len(), pb.len());
    let mut test_differs = false;
    for (ma, mb) in pa.iter().zip(pb) {
        assert_eq!(at(ma, &["gates"]), at(mb, &["gates"]));
        assert_eq!(at(ma, &["comparisons", "validation"]), at(mb, &["comparisons", "validation"]));
        test_differs |= at(ma, &["comparisons", "test"]) != at(mb, &["comparisons", "test"]);
    }
    assert!(test_differs, "the shuffled test block must change the test comparisons");
}

#[test]
fn structure_columns_are_not_search_inputs() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(200, 1, false));
    let spec = spec_file(&dir);
    let mut extra = TEMPORAL.to_vec();
    extra.extend_from_slice(&["--validation-frac", "0.2", "--target-spec", spec.as_str(), "--allow-target-structure"]);
    let out = search(&data, "x", &extra);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    match at(&j, &["variables"]) {
        Json::Arr(v) => assert_eq!(v.iter().map(|s| s.as_str().unwrap()).collect::<Vec<_>>(), ["x"]),
        other => panic!("{other:?}"),
    }
    let rungs = at(&j, &["baselines", "target_structure", "rungs"]);
    for name in ["zero_param", "affine", "binned20", "construction_ols"] {
        assert!(rungs.get(name).is_some(), "{name}");
    }
    assert_eq!(at(&j, &["baselines", "target_structure", "columns"]), &Json::Arr(vec![Json::Str("sigma".into())]));
}

#[test]
fn synthetic_constructed_target_reports_no_signal() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(400, 2, false));
    let spec = spec_file(&dir);
    let mut extra = TEMPORAL.to_vec();
    extra.extend_from_slice(&["--validation-frac", "0.2", "--target-spec", spec.as_str(), "--allow-target-structure"]);
    let out = search(&data, "sigma,x", &extra);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    assert_eq!(at(&j, &["selection", "verdict"]).as_str(), Some("no_signal"), "{}", stderr(&out));
    assert_eq!(at(&j, &["selection", "reference"]).as_str(), Some("target_structure.affine"));
    assert_eq!(at(&j, &["selection", "selected_rank"]), &Json::Null);
    for m in pareto(&j) {
        assert_eq!(at(m, &["beats_zero_param_test"]), &Json::Bool(false));
    }
}

#[test]
fn validation_cut_respects_timestamp_boundaries() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(100, 3, true));
    let mut extra = TEMPORAL.to_vec();
    extra[5] = "0.2";
    extra.extend_from_slice(&["--validation-frac", "0.26"]);
    let out = search(&data, "x", &extra);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let j = json(&out);
    let rows = at(&j, &["split", "validation_rows"]).as_f64().unwrap();
    assert_eq!(rows % 3.0, 0.0, "validation rows {rows}");
    assert_eq!(at(&j, &["split", "validation_purge_seconds"]).as_f64(), Some(STEP as f64));
}

#[test]
fn gate_json_is_deterministic_across_runs() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(160, 2, true));
    let mut extra = TEMPORAL.to_vec();
    extra.extend_from_slice(&["--validation-frac", "0.25"]);
    let a = json(&search(&data, "x", &extra));
    let b = json(&search(&data, "x", &extra));
    assert_eq!(at(&a, &["selection"]), at(&b, &["selection"]));
    assert_eq!(at(&a, &["noise"]), at(&b, &["noise"]));
    for (x, y) in pareto(&a).iter().zip(pareto(&b)) {
        assert_eq!(at(x, &["gates"]), at(y, &["gates"]));
        assert_eq!(at(x, &["comparisons"]), at(y, &["comparisons"]));
        assert_eq!(at(x, &["constant_errors"]), at(y, &["constant_errors"]));
    }
}

#[test]
fn invalid_gate_flags_are_usage_errors() {
    let dir = scratch();
    let data = write_csv(&dir, "d.csv", &panel(60, 1, true));
    for bad in [["--equivalence-z", "0"], ["--equivalence-z", "abc"], ["--noise-lag", "-1"]] {
        let mut extra = TEMPORAL.to_vec();
        extra.extend_from_slice(&bad);
        let out = search(&data, "x", &extra);
        assert_eq!(code(&out), 2, "{bad:?}: {}", stderr(&out));
    }
}
