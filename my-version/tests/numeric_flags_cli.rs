//! Numeric flags: an unparsable value exits 2 naming the flag and the value;
//! 0 stays valid where it means "off" or "all".

use std::path::PathBuf;
use std::process::{Command, Output};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn csv() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("numeric-flags");
    let path = scratch.join("data.csv");
    let mut text = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    (scratch, path)
}

/// The children set `RCSR_GPU=0`: no device call, not even the startup probe that `--cpu` still runs.
fn search(flag: &str, value: &str) -> Output {
    let (_scratch, data) = csv();
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .env("RCSR_GPU", "0")
        .arg("--data")
        .arg(&data)
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--quiet"])
        .args([flag, value])
        .output()
        .expect("run rcsr")
}

fn audit(flag: &str, value: &str) -> Output {
    let (_scratch, data) = csv();
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .env("RCSR_GPU", "0")
        .arg("audit")
        .arg("--data")
        .arg(&data)
        .args(["--target", "y"])
        .args([flag, value])
        .output()
        .expect("run rcsr")
}

fn refused(out: &Output, flag: &str, value: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{flag} '{value}': {err}");
    assert!(err.contains(&format!("error: {flag} needs ")), "{flag} '{value}': {err}");
    assert!(err.contains(&format!("got '{value}'")), "{flag} '{value}': {err}");
}

const COUNT_FLAGS: &[&str] = &[
    "--generations", "--islands", "--pop", "--max-complexity", "--seed", "--folds", "--lags",
    "--view", "--null-runs", "--null-block", "--baseline-samples", "--bootstrap-error",
    "--stability", "--cv-k", "--fit-iters", "--fit-fallback-max", "--patience", "--top",
];

const FLOAT_FLAGS: &[&str] = &[
    "--test-frac", "--validation-frac", "--history-frac", "--gap-break", "--time-budget",
];

#[test]
fn count_flags_refuse_unparsable_values() {
    for flag in COUNT_FLAGS {
        for value in ["abc", "-1", "1.5", ""] {
            refused(&search(flag, value), flag, value);
        }
    }
}

#[test]
fn float_flags_refuse_unparsable_values() {
    for flag in FLOAT_FLAGS {
        for value in ["abc", "nan", "inf", "-0.5", ""] {
            refused(&search(flag, value), flag, value);
        }
    }
}

#[test]
fn zero_stays_valid_where_it_means_off_or_all() {
    for flag in [
        "--lags", "--view", "--null-runs", "--bootstrap-error", "--stability", "--fit-iters",
        "--fit-fallback-max", "--baseline-samples", "--top", "--time-budget", "--validation-frac",
        "--history-frac", "--gap-break",
    ] {
        let out = search(flag, "0");
        assert_eq!(out.status.code(), Some(0), "{flag} 0: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn clamped_flags_keep_clamping() {
    for (flag, value) in [("--cv-k", "1"), ("--folds", "0")] {
        let out = search(flag, value);
        assert_ne!(out.status.code(), Some(2), "{flag} {value}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn audit_flags_refuse_unparsable_values() {
    for flag in ["--horizon", "--label-horizon", "--resample-small", "--resample-large"] {
        refused(&audit(flag, "abc"), flag, "abc");
        refused(&audit(flag, "-1"), flag, "-1");
    }
    refused(&audit("--session-gap", "abc"), "--session-gap", "abc");
    refused(&audit("--session-gap", "nan"), "--session-gap", "nan");
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_bench_flags_refuse_unparsable_values() {
    for flag in ["--batch", "--reps", "--depth", "--seed"] {
        let out = Command::new(env!("CARGO_BIN_EXE_rcsr-gpu-bench"))
            .args([flag, "abc"])
            .output()
            .expect("run rcsr-gpu-bench");
        refused(&out, flag, "abc");
    }
}
