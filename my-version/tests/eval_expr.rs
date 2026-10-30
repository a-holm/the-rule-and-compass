//! `--eval-expr` output contract (#254): one test block, `--json-out`.

use rcsr::devlog::Json;
use std::process::{Command, Output};

fn sine() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sine.csv").to_string()
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn eval_args(data: &str) -> Vec<&str> {
    vec![
        "--data", data, "--target", "y", "--vars", "x", "--cpu", "--no-dev-log", "--eval-expr", "sin(x)", "--json",
    ]
}

#[test]
fn eval_expr_reports_the_test_error_with_one_fold() {
    let data = sine();
    let out = rcsr(&eval_args(&data));
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let doc = Json::parse(String::from_utf8_lossy(&out.stdout).trim()).expect("json");
    let eval = doc.get("eval").expect("eval");
    let test = eval.get("test_error").and_then(|v| v.as_f64()).expect("finite test_error");
    assert!((test - 3.621871760723742e-7).abs() < 1e-18, "test_error {test:e}");
    let train = eval.get("train_error").and_then(|v| v.as_f64()).expect("train_error");
    assert!(train.is_finite());
    match eval.get("fold_errors") {
        Some(Json::Arr(f)) => {
            assert_eq!(f.len(), 1);
            assert_eq!(f[0].as_f64(), Some(test));
        }
        other => panic!("fold_errors: {other:?}"),
    }
}
