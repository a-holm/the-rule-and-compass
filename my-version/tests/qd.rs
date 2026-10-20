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

#[test]
fn cvt_mode_is_reported_deterministic_and_updates_centroids() {
    let a = json(&["--qd-cvt", "--qd-cells", "16"]);
    let b = json(&["--qd-cvt", "--qd-cells", "16"]);
    assert_eq!(qd_object(&a), qd_object(&b), "CVT archive is not deterministic");
    let q = qd_object(&a);
    assert!(q.contains("\"mode\":\"cvt\""), "{q}");
    assert!(q.contains("\"cells\":16"), "{q}");
    assert!(q.contains("\"hypervolume\""), "{q}");
    assert!(q.contains("\"evaluations\""), "{q}");
    // CVT updates need observations; a real search must have offered some.
    let offered: usize = q
        .split("\"evaluations\":")
        .nth(1)
        .and_then(|s| s.split('}').next())
        .and_then(|s| s.parse().ok())
        .expect("evaluations");
    assert!(offered > 0, "{q}");
}

#[test]
fn the_qd_policy_is_opt_in_and_does_not_change_the_disabled_trajectory() {
    let off = json(&[]);
    let passive = json(&["--qd"]);
    assert_eq!(pareto(&off), pareto(&passive));
    let policy = json(&["--qd-policy", "--qd-cells", "16"]);
    assert!(!policy.contains("\"qd\":null"), "{policy}");
    let q = qd_object(&policy);
    assert!(q.contains("\"mode\":\"cvt\""), "{q}");
    // The policy is deterministic for a seed and reports its own front.
    let again = json(&["--qd-policy", "--qd-cells", "16"]);
    assert_eq!(qd_object(&policy), qd_object(&again));
    assert_eq!(pareto(&policy), pareto(&again));
}

#[test]
fn cvt_accepts_the_k512_design_size() {
    let q = qd_object(&json(&["--qd-cvt", "--qd-cells", "512"]));
    assert!(q.contains("\"cells\":512"), "{q}");
}

#[test]
fn the_two_controls_are_available_and_deterministic() {
    let uniform = json(&["--qd-uniform", "--qd-cells", "16"]);
    let again = json(&["--qd-uniform", "--qd-cells", "16"]);
    assert_eq!(qd_object(&uniform), qd_object(&again));
    assert!(qd_object(&uniform).contains("\"mode\":\"cvt\""));

    let random = json(&["--qd-random", "--qd-cells", "16"]);
    assert!(qd_object(&random).contains("\"mode\":\"random\""), "{}", qd_object(&random));
    let again = json(&["--qd-random", "--qd-cells", "16"]);
    assert_eq!(qd_object(&random), qd_object(&again));
}
