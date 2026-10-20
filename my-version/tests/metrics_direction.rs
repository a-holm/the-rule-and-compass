//! Direction and class semantics at the CLI boundary (issue #113).
//!
//! The library tests in `metrics.rs` pin the arithmetic. These tests pin what a
//! user sees: a three-valued target is refused by the binary metrics, the
//! signed metric can be selected by name, and a fitted `step` produces the
//! zero-adjoint warning.

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], csv: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rcsr");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(csv.as_bytes())
        .expect("write csv");
    child.wait_with_output().expect("wait for rcsr")
}

fn csv(rows: &[(f64, f64)]) -> String {
    let mut s = String::from("x,y\n");
    for &(x, y) in rows {
        s.push_str(&format!("{x},{y}\n"));
    }
    s
}

#[test]
fn binary_metrics_refuse_a_three_valued_target() {
    let data = csv(&[
        (0.0, -1.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (3.0, -1.0),
    ]);
    for metric in ["auc", "log_loss", "hinge"] {
        let out = run(
            &[
                "--data",
                "-",
                "--target",
                "y",
                "--vars",
                "x",
                "--metric",
                metric,
                "--generations",
                "1",
                "--pop",
                "8",
            ],
            &data,
        );
        assert_eq!(
            out.status.code(),
            Some(3),
            "{metric} accepted a three-valued target"
        );
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("3 distinct values") && err.contains("binary"),
            "{metric}: {err}"
        );
    }
}

#[test]
fn binary_metrics_still_accept_two_classes() {
    let data = csv(&[(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]);
    let out = run(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--metric",
            "log_loss",
            "--generations",
            "2",
            "--pop",
            "8",
        ],
        &data,
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn signed_correlation_is_a_selectable_metric() {
    let data = csv(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]);
    let out = run(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--metric",
            "correlation_signed",
            "--generations",
            "1",
            "--pop",
            "8",
            "--json",
        ],
        &data,
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8_lossy(&out.stdout);
    assert!(
        json.contains("\"metric\":\"correlation_signed\""),
        "{json}"
    );
}

#[test]
fn a_fitted_step_warns_that_the_threshold_is_not_optimised() {
    let data = csv(&[(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]);
    let out = run(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--seed-expr",
            "step(2*x - 1)",
            "--cpu",
            "--generations",
            "1",
            "--pop",
            "8",
        ],
        &data,
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("warning") && err.contains("step") && err.contains("adjoint"),
        "stderr: {err}"
    );
}
