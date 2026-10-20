//! Constants are fitted to the configured metric (issue #117).
//!
//! The arithmetic and the "never worse than squared" property are pinned in
//! `fit.rs`; this test pins the user-visible contract: `--json` names the
//! fitting objective, and an L1 run can lower the reported absolute error at
//! the same shape while a squared run keeps its own optimum.

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

/// A line with one gross outlier: squared and absolute optima differ.
fn outlier_csv() -> String {
    let mut s = String::from("x,y\n");
    for (i, y) in [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 40.0]
        .into_iter()
        .enumerate()
    {
        s.push_str(&format!("{},{}\n", -3.0 + i as f64, y));
    }
    s
}

fn seed_args(metric: &str) -> Vec<String> {
    vec![
        "--data".into(),
        "-".into(),
        "--target".into(),
        "y".into(),
        "--vars".into(),
        "x".into(),
        "--metric".into(),
        metric.into(),
        "--seed-expr".into(),
        "0.5 * x + 0.0".into(),
        "--generations".into(),
        "2".into(),
        "--pop".into(),
        "12".into(),
        "--json".into(),
    ]
}

fn run_metric(metric: &str) -> (f64, String) {
    let args = seed_args(metric);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let out = run(&refs, &outlier_csv());
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8(out.stdout).expect("utf8 json");
    // The fitted seed is the first archive entry whose train_error we can read.
    let value = json
        .split("\"fitting\":{\"loss\":\"")
        .nth(1)
        .and_then(|s| s.split("\"value\":").nth(1))
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.parse::<f64>().ok())
        .expect("fitting.value in json");
    (value, json)
}

#[test]
fn json_names_the_fitting_objective() {
    let (_, absolute) = run_metric("absolute");
    assert!(
        absolute.contains("\"fitting\":{\"loss\":\"absolute\""),
        "{absolute}"
    );
    let (_, squared) = run_metric("squared");
    assert!(
        squared.contains("\"fitting\":{\"loss\":\"squared\""),
        "{squared}"
    );
}

#[test]
fn the_l1_run_fits_a_lower_absolute_loss_than_the_squared_run() {
    let (l1, _) = run_metric("absolute");
    let (l2, _) = run_metric("squared");
    assert!(
        l1 <= l2,
        "the absolute-fitted constants should not report a worse absolute loss: \
         {l1} vs {l2}"
    );
}
