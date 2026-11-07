//! #827: an explicit `--cpu` does not probe the device, so a report says "not probed",
//! never "no device". The calibration device check is skipped under `--cpu` because a
//! CPU-only plan reads CPU rates only.

use std::path::Path;
use std::process::Output;

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const T0: i64 = 1_704_067_200;
const H: i64 = 3600;

fn rcsr(args: &[&str], gpu_env: Option<&str>) -> Output {
    // Each call states its own `RCSR_GPU`: unset means the live default.
    let mut c = rcsr_cmd::rcsr_on_device();
    c.args(args);
    match gpu_env {
        Some(v) => c.env("RCSR_GPU", v),
        None => c.env_remove("RCSR_GPU"),
    };
    c.output().unwrap()
}

fn out(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn store(dir: &Path) -> String {
    let mut ohlcv = String::from("ts,symbol,open,high,low,close,volume\n");
    for g in 0..2i64 {
        let mut c = 100.0f64;
        for i in 0..200i64 {
            c *= 1.0 + (((i * 7 + g * 3) % 11) as f64 - 5.0) / 500.0;
            ohlcv.push_str(&format!("{},s{},{},{},{},{},{}\n", T0 + i * H, g, c, c * 1.01, c * 0.99, c, 1000 + i));
        }
    }
    let csv = s(&dir.join("o.csv"));
    std::fs::write(&csv, ohlcv).unwrap();
    let store = s(&dir.join("o.rcsb"));
    let o = rcsr(&["import", &csv, &store, "--time", "ts", "--quiet"], Some("0"));
    assert!(o.status.success(), "{}", out(&o));
    store
}

fn enrich(store: &str, dir: &Path, name: &str, extra: &[&str], gpu_env: Option<&str>) -> Output {
    let target = s(&dir.join(name));
    let mut a = vec!["enrich", "--data", store, "--out", target.as_str(), "--features", "returns", "--json"];
    a.extend_from_slice(extra);
    rcsr(&a, gpu_env)
}

fn plan_of(json: &str) -> &str {
    &json[json.find("\"plan\":[").expect("report has a plan")..]
}

#[test]
fn enrich_cpu_says_not_probed_and_a_probed_run_says_what_it_found() {
    let scratch = Scratch::new("cpu-not-probed-enrich");
    let dir = scratch.path();
    let st = store(dir);

    let o = enrich(&st, dir, "a.rcsb", &["--cpu"], None);
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"gpu_probed\":false,\"gpu_available\":null,\"gpu_device\":null"), "{j}");

    let o = enrich(&st, dir, "b.rcsb", &["--no-gpu"], None);
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"gpu_probed\":false,\"gpu_available\":null"), "{j}");

    let o = rcsr_cmd::rcsr_on_device()
        .args(["enrich", "--data", &st, "--out", &s(&dir.join("d.rcsb")), "--features", "returns", "--cpu"])
        .env_remove("RCSR_GPU")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", out(&o));
    let text = out(&o);
    assert!(text.contains("gpu_available=not probed device=not probed"), "{text}");
    assert!(!text.contains("gpu_available=false"), "{text}");

    // The probe ran and found nothing: a finding, not an absence of one.
    let o = enrich(&st, dir, "e.rcsb", &[], Some("0"));
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"gpu_probed\":true,\"gpu_available\":false,\"gpu_device\":null"), "{j}");
}

/// A default run on a machine with a device: the probe ran and found it. Skips through the
/// shared helper, so `RCSR_REQUIRE_DEVICE=1` fails it on a machine without one.
#[cfg(feature = "cuda")]
#[test]
fn a_live_device_default_run_is_probed_and_names_the_device() {
    if !rcsr::backend::probe().available() {
        rcsr::backend::skip_without_device("skipping: no usable CUDA device");
        return;
    }
    let scratch = Scratch::new("cpu-not-probed-live");
    let dir = scratch.path();
    let st = store(dir);
    let o = enrich(&st, dir, "live.rcsb", &[], None);
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"gpu_probed\":true,\"gpu_available\":true,\"gpu_device\":\""), "{j}");
    assert!(!j.contains("\"gpu_device\":null"), "{j}");
    let text = out(&enrich_text(&st, dir, "live-text.rcsb"));
    assert!(text.contains("gpu_available=true"), "{text}");
    assert!(!text.contains("not probed"), "{text}");

    let mut csv = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        csv.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    let d = s(&dir.join("d.csv"));
    std::fs::write(&d, csv).unwrap();
    let o = rcsr(&["--data", d.as_str(), "--target", "y", "--generations", "1", "--pop", "20", "--json"], None);
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"device_probed\":true"), "{j}");
    assert!(!j.contains("\"device_probed\":false"), "{j}");
}

#[test]
fn search_cpu_reports_the_device_as_not_probed() {
    let scratch = Scratch::new("cpu-not-probed-search");
    let dir = scratch.path();
    let mut csv = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        csv.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    let d = s(&dir.join("d.csv"));
    std::fs::write(&d, csv).unwrap();
    let base = ["--data", d.as_str(), "--target", "y", "--generations", "1", "--pop", "20", "--json"];

    let mut a = base.to_vec();
    a.push("--cpu");
    let o = rcsr(&a, None);
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"device\":null,\"device_probed\":false"), "{j}");
    assert!(String::from_utf8_lossy(&o.stderr).contains("device=not probed"), "{}", out(&o));

    let o = rcsr(&base, Some("0"));
    assert!(o.status.success(), "{}", out(&o));
    let j = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(j.contains("\"device\":null,\"device_probed\":true"), "{j}");
}

/// A calibration file measured on another device. `--cpu` skips the device check
/// because a CPU-only plan uses CPU rates only: the plan equals the plan of a host-only file.
#[test]
fn enrich_cpu_with_a_calibration_of_another_device_plans_with_cpu_rates_only() {
    let scratch = Scratch::new("cpu-not-probed-calibration");
    let dir = scratch.path();
    let st = store(dir);
    let host = s(&dir.join("host.json"));
    let o = rcsr(&["calibrate", "--cpu", "--out", &host], Some("0"));
    assert!(o.status.success(), "{}", out(&o));
    let text = std::fs::read_to_string(&host).unwrap();
    assert!(text.contains("\"device\":null"), "{text}");
    let foreign = s(&dir.join("foreign.json"));
    std::fs::write(&foreign, text.replace("\"device\":null", "\"device\":\"Other Device (sm_00)\"")).unwrap();

    let a = enrich(&st, dir, "a.rcsb", &["--cpu", "--calibration", &host], None);
    let b = enrich(&st, dir, "b.rcsb", &["--cpu", "--calibration", &foreign], None);
    assert!(a.status.success() && b.status.success(), "{}{}", out(&a), out(&b));
    let (ja, jb) = (String::from_utf8_lossy(&a.stdout).to_string(), String::from_utf8_lossy(&b.stdout).to_string());
    assert!(jb.contains("\"source\":\"file\""), "{jb}");
    assert_eq!(plan_of(&ja), plan_of(&jb));

    // With a probe the other-device check applies and refuses the same file.
    if cfg!(feature = "cuda") && rcsr::backend::probe().available() {
        let o = enrich(&st, dir, "c.rcsb", &["--calibration", &foreign], None);
        assert_eq!(o.status.code(), Some(2), "{}", out(&o));
        assert!(out(&o).contains("was measured on 'Other Device (sm_00)'"), "{}", out(&o));
    }
}

fn enrich_text(store: &str, dir: &Path, name: &str) -> Output {
    let target = s(&dir.join(name));
    rcsr(&["enrich", "--data", store, "--out", target.as_str(), "--features", "returns"], None)
}

/// `rcsr schema` names the probe fields in the backend reporting text and in the enrich
/// `--json` description, not only inside the `--no-gpu` description.
#[test]
fn the_schema_lists_the_probe_fields() {
    let o = rcsr(&["schema"], Some("0"));
    let schema = String::from_utf8_lossy(&o.stdout).to_string();
    let reporting = &schema[schema.find("\"reporting\":\"").expect("backend reporting")..];
    let reporting = &reporting[..reporting.find("\",\"").unwrap()];
    assert!(reporting.contains("device_probed"), "{reporting}");
    assert!(reporting.contains("gpu_probed"), "{reporting}");
    let enrich = &schema[schema.find("\"enrich\":{\"command\"").expect("enrich in schema")..];
    let json_flag = &enrich[enrich.find("\"name\":\"--json\"").expect("enrich --json")..];
    let json_flag = &json_flag[..json_flag.find("\"}").unwrap()];
    assert!(json_flag.contains("gpu_probed"), "{json_flag}");
}
