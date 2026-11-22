//! #1258: `rcsr enumerate --gpu`, the device arm, on the planted panel of
//! `tests/enumerate.rs`: the report, the refusals and the fail-closed exit.

#![cfg(feature = "cuda")]

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use scratch::Scratch;
use std::path::Path;
use std::process::{Command, Output};

const OPS10: &str = "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss";

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Four symbols on an hourly grid; the return pays where `x1 - x2 < -0.3`.
fn write_csv(dir: &Path) -> String {
    let mut text = String::from("t,s,x1,x2,x3,r,y\n");
    for k in 0..600i64 {
        for (j, sym) in ["a", "b", "c", "d"].iter().enumerate() {
            let i = (k + 40) as f64 * 4.0 + j as f64;
            let (x1, x2) = (frac(i * 0.618_033_988_749_894_9) - 0.5, frac(i * 0.754_877_666_246_692_7) - 0.5);
            let x3 = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let r = (if x1 - x2 < -0.3 { 0.004 } else { -0.004 }) + 0.002 * noise;
            text.push_str(&format!("{},{},{},{},{},{},{}\n", 1_600_000_000 + 3600 * k, sym, x1, x2, x3, r, r));
        }
    }
    let path = dir.join("planted.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("s.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn device_available() -> bool {
    if rcsr::gpu::probe().available() {
        true
    } else {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        false
    }
}

fn enumerate(mut cmd: Command, data: &str, spec: &str, extra: &[&str]) -> Output {
    cmd.args(["enumerate", "--data", data, "--target-spec", spec, "--target", "y", "--vars", "x1,x2,x3", "--time", "t", "--group-by", "s"]);
    cmd.args(["--split", "temporal", "--validation-frac", "0.3", "--test-frac", "0.2", "--objective", "pnl", "--cost-bps", "5"]);
    cmd.args(["--center", "none", "--scale", "none", "--no-dev-log", "--quiet"]);
    cmd.args(extra);
    cmd.output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    let text = String::from_utf8_lossy(&o.stdout).into_owned();
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
}

/// The text of member `key` of a one-line JSON report.
fn member<'a>(text: &'a str, key: &str) -> &'a str {
    let needle = format!(",\"{key}\":{{");
    let at = text.find(&needle).unwrap_or_else(|| panic!("no {key} member")) + needle.len() - 1;
    let mut depth = 0usize;
    for (i, ch) in text[at..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[at..at + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced {key} member");
}

fn num(j: &Json, path: &[&str]) -> f64 {
    path.iter().fold(j, |v, k| v.get(k).unwrap_or_else(|| panic!("no {k} in {path:?}"))).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
}

fn arr<'a>(j: &'a Json, key: &str) -> &'a [Json] {
    match j.get(key) {
        Some(Json::Arr(v)) => v,
        _ => panic!("{key} is not an array"),
    }
}

/// The device report states its arm and device, gives every member a device
/// fitness, holds the planted rule at size 3 and certifies it with the family
/// null, and its class object is the CPU arm's.
#[test]
fn enumerate_gpu_reports_the_device_arm_and_the_cpu_class() {
    if !device_available() {
        return;
    }
    let s = Scratch::new("enumerate-gpu");
    let (data, spec) = (write_csv(s.path()), write_spec(s.path()));
    let base = ["--ops", OPS10, "--max-size", "3", "--null-shifts", "19", "--seed", "11", "--json"];
    let cpu = enumerate(rcsr_cmd::rcsr(), &data, &spec, &base);
    let mut extra = base.to_vec();
    extra.push("--gpu");
    let dev = enumerate(rcsr_cmd::rcsr_on_device(), &data, &spec, &extra);
    let (cj, dj) = (json(&cpu), json(&dev));
    let (ct, dt) = (String::from_utf8_lossy(&cpu.stdout).into_owned(), String::from_utf8_lossy(&dev.stdout).into_owned());
    assert_eq!(cj.get("arm").and_then(Json::as_str), Some("cpu"));
    assert_eq!(dj.get("arm").and_then(Json::as_str), Some("device"));
    assert!(ct.find(",\"device\":{").is_none(), "the CPU report has no device member");
    assert_eq!(member(&dt, "class"), member(&ct, "class"), "the device arm scores the CPU arm's class");
    assert!(num(&dj, &["device", "batch"]) >= 1.0);
    assert_eq!(num(&dj, &["device", "book_batch"]), 20.0);
    assert!(num(&dj, &["device", "bytes"]) > 0.0);
    let best3 = &arr(&dj, "best")[2];
    let tree = best3.get("tree").and_then(Json::as_str).unwrap();
    let side = best3.get("side").and_then(Json::as_str).unwrap();
    assert!(matches!((tree, side), ("(x1 - x2)", "below") | ("(x2 - x1)", "above")), "{tree} {side}");
    let null = dj.get("null").unwrap();
    assert_eq!(arr(null, "best").len(), 19);
    let observed = num(null, &["observed"]);
    assert!(num(null, &["p_family"]) <= 0.05, "p_family {}", num(null, &["p_family"]));
    let members: Vec<&Json> = arr(&dj, "best").iter().chain(arr(&dj, "top")).filter(|m| m.get("expression").is_some()).collect();
    assert!(members.iter().all(|m| m.get("device_fitness").and_then(Json::as_f64).is_some()), "every device member has a device fitness");
    assert!(members.iter().any(|m| m.get("device_fitness").and_then(Json::as_f64).map(f64::to_bits) == Some(observed.to_bits())), "the device class best is a member");
    // top[].p_family compares the member's device fitness with the device null.
    for m in arr(&dj, "top") {
        let f = num(m, &["device_fitness"]);
        let below = arr(null, "best").iter().filter(|b| b.as_f64().is_some_and(|b| b <= f)).count();
        assert_eq!(num(m, &["p_family"]), (1 + below) as f64 / 20.0);
    }
    assert!(arr(&cj, "best").iter().all(|m| m.get("device_fitness").is_none()));
}

/// The refusals of #1183 hold with --gpu, and --gpu with --cpu is refused.
#[test]
fn enumerate_gpu_refusals_are_unchanged() {
    if !device_available() {
        return;
    }
    let s = Scratch::new("enumerate-gpu-refusals");
    let (data, spec) = (write_csv(s.path()), write_spec(s.path()));
    for (extra, needle) in [
        (vec!["--band", "0.1"], "--band"),
        (vec!["--min-holding", "3"], "--min-holding"),
        (vec!["--pnl-mask", "x3"], "--pnl-mask"),
        (vec!["--band-units", "sd"], "--band-units"),
        (vec!["--sample-rows", "100"], "--sample-rows"),
        (vec!["--cash-return", "0.0001"], "--cash-return"),
        (vec!["--cpu"], "--cpu"),
    ] {
        let mut args = vec!["--ops", OPS10, "--gpu"];
        args.extend(extra.iter().copied());
        let o = enumerate(rcsr_cmd::rcsr_on_device(), &data, &spec, &args);
        assert_eq!(code(&o), 2, "{args:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{args:?}: {}", stderr(&o));
        assert!(o.stdout.is_empty(), "{args:?}");
    }
}

/// A refused kernel launch exits 1 naming the launch, prints no report and
/// writes no --out file: no CPU fallback.
#[test]
fn enumerate_gpu_fails_closed_on_a_refused_launch() {
    if !device_available() {
        return;
    }
    let s = Scratch::new("enumerate-gpu-refused");
    let (data, spec) = (write_csv(s.path()), write_spec(s.path()));
    let out = s.path().join("e.json");
    let out_s = out.to_string_lossy().into_owned();
    let mut cmd = rcsr_cmd::rcsr_on_device();
    cmd.env("RCSR_FORCE_LAUNCH_REFUSAL", "rc_enum_score");
    let o = enumerate(cmd, &data, &spec, &["--ops", "add,sub,abs", "--max-size", "2", "--gpu", "--json", "--out", &out_s]);
    let err = stderr(&o);
    assert_eq!(code(&o), 1, "{err}");
    assert!(err.contains("rc_enum_score_kernel") && err.contains("launch guard refused"), "{err}");
    assert!(o.stdout.is_empty());
    assert!(!out.exists(), "a failed device arm writes no report");
}
