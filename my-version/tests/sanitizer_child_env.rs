//! #910: an rcsr child of a sanitized test runs without the compute-sanitizer injection.

use std::ffi::OsString;
use std::process::Command;

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

fn vars(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
    pairs.iter().map(|(k, v)| (OsString::from(k), OsString::from(v))).collect()
}

fn env_of(cmd: &Command) -> Vec<(String, Option<String>)> {
    cmd.get_envs()
        .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned())))
        .collect()
}

const SANITIZED: &[(&str, &str)] = &[
    ("RCSR_UNDER_SANITIZER", "memcheck"),
    ("NV_SANITIZER_INJECTION_TRANSPORT_TYPE", "uds"),
    ("NV_SANITIZER_INJECTION_PORT_BASE", "49152"),
    ("NVTX_INJECTION64_PATH", "libsanitizer-collection.so"),
    ("NVIDIA_PROCESS_INJECTION_CRASH_REPORTING", "0"),
    ("CUDA_INJECTION64_PATH", "x.so"),
    ("LD_PRELOAD", "x.so"),
    ("LD_LIBRARY_PATH", "/usr/local/cuda-13.0/compute-sanitizer:/opt/lib"),
    ("PATH", "/usr/bin"),
];

#[test]
fn a_sanitized_child_loses_the_injection_environment() {
    let cmd = rcsr_cmd::without_injection(Command::new("rcsr"), vars(SANITIZED));
    let env = env_of(&cmd);
    for name in [
        "NV_SANITIZER_INJECTION_TRANSPORT_TYPE",
        "NV_SANITIZER_INJECTION_PORT_BASE",
        "NVTX_INJECTION64_PATH",
        "NVIDIA_PROCESS_INJECTION_CRASH_REPORTING",
        "CUDA_INJECTION64_PATH",
        "LD_PRELOAD",
    ] {
        assert!(env.contains(&(name.to_string(), None)), "{name} is not removed: {env:?}");
    }
    assert!(env.contains(&("LD_LIBRARY_PATH".to_string(), Some("/opt/lib".to_string()))), "{env:?}");
    assert!(!env.iter().any(|(k, _)| k == "PATH" || k == "RCSR_UNDER_SANITIZER"), "{env:?}");
}

#[test]
fn a_library_path_with_only_the_sanitizer_directory_is_removed() {
    let cmd = rcsr_cmd::without_injection(
        Command::new("rcsr"),
        vars(&[("RCSR_UNDER_SANITIZER", "memcheck"), ("LD_LIBRARY_PATH", "/usr/local/cuda-13.0/compute-sanitizer")]),
    );
    assert_eq!(env_of(&cmd), vec![("LD_LIBRARY_PATH".to_string(), None)]);
}

#[test]
fn an_unsanitized_child_keeps_its_environment() {
    let mut pairs = SANITIZED.to_vec();
    pairs.retain(|(k, _)| *k != "RCSR_UNDER_SANITIZER");
    let cmd = rcsr_cmd::without_injection(Command::new("rcsr"), vars(&pairs));
    assert!(env_of(&cmd).is_empty());
}
