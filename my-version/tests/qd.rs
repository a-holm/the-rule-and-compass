//! Quality-diversity coverage layer (#120).
//!
//! The archive is passive: enabling it records descriptor cells but must not
//! change the Pareto front. These tests pin the off default, determinism and
//! the presence of a real coverage report.

use std::process::{Command, Output};

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn base(extra: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = [
        "--data",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv"),
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--islands",
        "1",
        "--pop",
        "40",
        "--generations",
        "8",
        "--no-front-parents",
        "--quiet",
        "--json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

fn json(extra: &[&str]) -> String {
    let args = base(extra);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let out = rcsr(&refs);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 json")
}

fn pareto(json: &str) -> &str {
    let start = json.find("\"pareto\":").expect("pareto field");
    &json[start..]
}

fn qd_object(json: &str) -> String {
    json.split("\"qd\":")
        .nth(1)
        .and_then(|s| s.split(",\"seeds\"").next())
        .expect("qd field")
        .to_string()
}

#[test]
fn qd_is_off_by_default_and_does_not_change_the_front() {
    let off = json(&[]);
    assert!(off.contains("\"qd\":null"), "{off}");
    let on = json(&["--qd"]);
    assert!(!on.contains("\"qd\":null"), "{on}");
    assert_eq!(
        pareto(&off),
        pareto(&on),
        "the passive QD layer changed the reported Pareto front"
    );
}

#[test]
fn coverage_is_reported_and_deterministic() {
    let a = json(&["--qd", "--qd-cells", "8"]);
    let b = json(&["--qd", "--qd-cells", "8"]);
    assert_eq!(qd_object(&a), qd_object(&b), "QD coverage is not deterministic");
    let q = qd_object(&a);
    assert!(q.contains("\"bins\":8"), "{q}");
    assert!(q.contains("\"cells\":64"), "{q}");
    // A real search must occupy more than the empty or singleton grid.
    let occupied: usize = q
        .split("\"occupied\":")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.parse().ok())
        .expect("occupied");
    assert!(occupied > 1, "coverage did not rise above one cell: {q}");
    assert_eq!(pareto(&a), pareto(&b), "QD changed the front between runs");
}

#[test]
fn an_invalid_grid_is_a_usage_error() {
    let args = base(&["--qd-cells", "1"]);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let out = rcsr(&refs);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--qd-cells"), "{err}");
}
