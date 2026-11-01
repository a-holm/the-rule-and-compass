//! `--dispatch auto|device` (#438): usage errors and documentation.

use std::process::Command;

fn rcsr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn dispatch_device_without_gpu_is_a_usage_error() {
    for args in [
        vec!["--dispatch", "device", "--data", "missing.csv"],
        vec!["--dispatch", "device", "--cpu", "--data", "missing.csv"],
    ] {
        let out = rcsr(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("--dispatch device needs --gpu"),
            "{args:?}: {stderr}"
        );
    }
}

#[test]
fn dispatch_rejects_an_unknown_mode() {
    let out = rcsr(&["--dispatch", "fast", "--data", "missing.csv"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--dispatch needs auto or device"));
}

#[test]
fn dispatch_is_in_help_and_schema() {
    let help = String::from_utf8_lossy(&rcsr(&["--help"]).stdout).to_string();
    assert!(help.contains("--dispatch <auto|device>"));
    let schema = String::from_utf8_lossy(&rcsr(&["schema"]).stdout).to_string();
    assert!(schema.contains("\"--dispatch\""));
}
