//! `--fit-rows` parsing: `0` and integers >= 2 are accepted, anything else exits 2.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use std::path::PathBuf;
use std::process::Output;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn csv() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("small");
    let path = scratch.join("data.csv");
    let mut text = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    (scratch, path)
}

fn run(value: &str) -> Output {
    let (_scratch, data) = csv();
    let out = rcsr_cmd::rcsr()
        .args(["--data"])
        .arg(&data)
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--fit-rows", value])
        .output()
        .expect("run rcsr");
    out
}

#[test]
fn fit_rows_refuses_unparsable_values() {
    for value in ["none", "abc", "-3", "1.5", "1", ""] {
        let out = run(value);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "value '{value}': {err}");
        assert!(err.contains("--fit-rows"), "value '{value}': {err}");
    }
}

#[test]
fn fit_rows_names_the_bad_value() {
    let out = run("none");
    assert!(String::from_utf8_lossy(&out.stderr).contains("'none'"));
}

#[test]
fn fit_rows_accepts_zero_and_integers() {
    for value in ["0", "10"] {
        let out = run(value);
        assert_eq!(out.status.code(), Some(0), "value '{value}': {}", String::from_utf8_lossy(&out.stderr));
    }
}

fn big_csv(tag: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(tag);
    let path = scratch.join("big.csv");
    let mut text = String::from("x,y\n");
    for i in 0..360_000u32 {
        let x = (i % 1000) as f64 * 0.01;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    (scratch, path)
}

fn run_big(tag: &str, extra: &[&str]) -> Output {
    let (_scratch, data) = big_csv(tag);
    rcsr_cmd::rcsr_for(extra)
        .args(["--data"])
        .arg(&data)
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--fit-iters", "3"])
        .args(extra)
        .output()
        .expect("run rcsr")
}

#[cfg(feature = "cuda")]
#[test]
fn fit_rows_default_applies_on_the_gpu_and_zero_turns_it_off() {
    if !rcsr::gpu::probe().available() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let on = run_big("on", &["--gpu", "--fit-strategy", "gpu"]);
    let on_err = String::from_utf8_lossy(&on.stderr);
    assert_eq!(on.status.code(), Some(0), "{on_err}");
    assert!(on_err.contains("# fit-rows: 65536 (default"), "{on_err}");
    let off = run_big("off", &["--gpu", "--fit-strategy", "gpu", "--fit-rows", "0"]);
    let off_err = String::from_utf8_lossy(&off.stderr);
    assert_eq!(off.status.code(), Some(0), "{off_err}");
    assert!(!off_err.contains("# fit-rows:"), "{off_err}");
}

#[test]
fn fit_rows_default_does_not_apply_on_the_cpu_path() {
    let out = run_big("cpu", &["--cpu"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    assert!(!err.contains("# fit-rows:"), "{err}");
}
