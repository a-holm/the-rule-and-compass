//! `--fit-rows` parsing: `0` and integers >= 2 are accepted, anything else exits 2.

use std::path::PathBuf;
use std::process::{Command, Output};

fn csv() -> PathBuf {
    let path = std::env::temp_dir().join(format!("rcsr-fit-rows-{}.csv", std::process::id()));
    let mut text = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn run(value: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data"])
        .arg(csv())
        .args(["--target", "y", "--generations", "1", "--pop", "20", "--fit-rows", value])
        .output()
        .expect("run rcsr")
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
