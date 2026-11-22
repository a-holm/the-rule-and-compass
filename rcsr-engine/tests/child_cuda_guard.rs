//! #1349: the guard that makes a CUDA test executable refuse a child `rcsr` built without CUDA.

use std::os::unix::fs::PermissionsExt;
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

fn stub(tag: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-guard-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("rcsr");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[test]
fn only_a_version_text_with_a_cuda_line_reports_cuda() {
    assert!(!rcsr_cmd::child_reports_cuda("rcsr 0.0.0"));
    assert!(!rcsr_cmd::child_reports_cuda("rcsr 0.0.0\nnot cuda: x"));
    assert!(rcsr_cmd::child_reports_cuda("rcsr 0.0.0\ncuda: toolkit 13.0"));
}

#[test]
fn a_cpu_child_is_refused_with_the_guard_message() {
    let program = stub("cpu", "echo 'rcsr 0.0.0'");
    let cell = OnceLock::new();
    let probes = AtomicUsize::new(0);
    let err = catch_unwind(|| rcsr_cmd::assert_child_has_cuda_once(program.to_str().unwrap(), &cell, &probes))
        .expect_err("a CPU child must be refused");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("child rcsr built without cuda"), "message: {msg}");
    let _ = std::fs::remove_dir_all(program.parent().unwrap());
}

#[test]
fn a_cuda_child_passes_and_is_probed_once() {
    let program = stub("cuda", "echo 'rcsr 0.0.0'; echo 'cuda: toolkit 13.0'");
    let cell = OnceLock::new();
    let probes = AtomicUsize::new(0);
    for _ in 0..3 {
        rcsr_cmd::assert_child_has_cuda_once(program.to_str().unwrap(), &cell, &probes);
    }
    assert_eq!(probes.load(Ordering::SeqCst), 1);
    let _ = std::fs::remove_dir_all(program.parent().unwrap());
}
