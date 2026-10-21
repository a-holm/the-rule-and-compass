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
    // #120: coverage has a reachable denominator, entropy and quality pairing.
    assert!(q.contains("\"reachable\""), "{q}");
    assert!(q.contains("\"reachable_coverage\""), "{q}");
    assert!(q.contains("\"occupancy_entropy\""), "{q}");
    assert!(q.contains("\"qd_score\""), "{q}");
    let reachable: usize = q
        .split("\"reachable\":")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.parse().ok())
        .expect("reachable");
    assert!(reachable > 0 && reachable <= 64, "{q}");
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

/// #120: every run reports a fixed-reference validation hypervolume computed
/// from its own archive, so QD arms and the baseline are comparable. The
/// reference is the held-out target variance, not an arm's observed extremes.
#[test]
fn fixed_reference_validation_hypervolume_is_reported_for_every_arm() {
    let baseline = json(&[]);
    let qd = json(&["--qd", "--qd-cells", "8"]);
    for (label, out) in [("baseline", &baseline), ("qd", &qd)] {
        assert!(
            out.contains("\"validation_hv\":{"),
            "{label} has no validation_hv: {out}"
        );
        assert!(out.contains("\"reference_error\":"), "{label}: {out}");
        let hv: f64 = out
            .split("\"validation_hv\":{\"sample\":")
            .nth(1)
            .and_then(|s| s.split("\"hypervolume\":").nth(1))
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
            .expect("validation hypervolume");
        assert!(hv.is_finite() && hv >= 0.0, "{label}: {hv}");
    }
    // Baseline and QD arms differ in the archive they produce but the reference
    // is the same data constant; both numbers are directly comparable.
    let reference = |out: &str| -> f64 {
        out.split("\"validation_hv\":{\"sample\":")
            .nth(1)
            .and_then(|s| s.split("\"reference_error\":").nth(1))
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
            .expect("reference error")
    };
    assert_eq!(reference(&baseline), reference(&qd));
    // #120: the CPU passive path classifies every candidate from the scoring
    // pass; the archive must not fall back to a second tree evaluation.
    let fallbacks: usize = qd_object(&qd)
        .split("\"descriptor_fallbacks\":")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.parse().ok())
        .expect("descriptor_fallbacks");
    assert_eq!(fallbacks, 0, "QD re-evaluated candidates for descriptors");
    // Deterministic for a seed (timing fields are not).
    let repeat = json(&["--qd", "--qd-cells", "8"]);
    assert_eq!(qd_object(&repeat), qd_object(&qd));
}

/// The reachable denominator is bounded by the descriptor, not by `K`. With a
/// single input (`--vars x`) usage is either 0 or 1, so a K=64 CVT cannot have
/// 64 reachable cells even if the map has 64 centroids.
#[test]
fn reachable_denominator_is_bounded_by_the_usage_lattice() {
    let q = qd_object(&json(&["--qd-cvt", "--qd-cells", "64"]));
    let reachable: usize = q
        .split("\"reachable\":")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.parse().ok())
        .expect("reachable");
    assert!(reachable > 0, "{q}");
    assert!(reachable <= 64, "{q}");
    assert!(q.contains("\"cells\":64"), "{q}");
}

/// #120: the passive layer must not change the trajectory even on the
/// squared-metric path that reuses the fitter's SSE. The descriptor may cost a
/// prediction pass there; the reported front must stay bit-identical.
#[test]
fn passive_qd_does_not_change_the_squared_metric_front() {
    let squared = ["--metric", "squared"];
    let mut off_args = squared.to_vec();
    off_args.extend(["--fit-iters", "4"]);
    let off = json(&off_args);
    let mut on_args = off_args.clone();
    on_args.extend(["--qd", "--qd-cells", "8"]);
    let on = json(&on_args);
    assert_eq!(
        pareto(&off),
        pareto(&on),
        "QD changed the squared-metric front through the SSE fast path"
    );
    assert!(qd_object(&on).contains("\"reachable\":"));
}
