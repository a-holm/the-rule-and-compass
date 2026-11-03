//! `--fit-rows` parsing: `0` and integers >= 2 are accepted, anything else exits 2.

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static CSV_SEQ: AtomicUsize = AtomicUsize::new(0);

fn csv() -> PathBuf {
    let n = CSV_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("rcsr-fit-rows-{}-{}.csv", std::process::id(), n));
    let mut text = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn run(value: &str) -> Output {
    let data = csv();
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data"])
        .arg(&data)
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--fit-rows", value])
        .output()
        .expect("run rcsr");
    let _ = std::fs::remove_file(&data);
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

fn big_csv(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("rcsr-fit-rows-big-{}-{}.csv", std::process::id(), tag));
    let mut text = String::from("x,y\n");
    for i in 0..360_000u32 {
        let x = (i % 1000) as f64 * 0.01;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn run_big(tag: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data"])
        .arg(big_csv(tag))
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--fit-iters", "3"])
        .args(extra)
        .output()
        .expect("run rcsr")
}

#[cfg(feature = "cuda")]
#[test]
fn fit_rows_default_applies_on_the_gpu_and_zero_turns_it_off() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
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
