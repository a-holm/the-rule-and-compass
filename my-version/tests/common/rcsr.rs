//! #822: the one place a test builds an `rcsr` child. A test with no device purpose gets
//! `RCSR_GPU=0`, so its child makes no CUDA call and a compute-sanitizer pass over the
//! test binary traces no device work it does not mean to test. A test that exists to run
//! on the device says so by calling `rcsr_on_device`. `tests/rcsr_command_lint.rs`
//! refuses any other spawn of `CARGO_BIN_EXE_rcsr`.
#![allow(dead_code)]

use std::ffi::{OsStr, OsString};
use std::process::Command;

const INJECTION_PREFIXES: &[&str] = &["NV_SANITIZER_INJECTION_", "NVIDIA_PROCESS_INJECTION_"];
const INJECTION_NAMES: &[&str] = &["NVTX_INJECTION64_PATH", "CUDA_INJECTION64_PATH", "LD_PRELOAD"];

/// #910: under a sanitized test run (`RCSR_UNDER_SANITIZER` in `vars`) the child must not load the
/// compute-sanitizer injection: it would block in `libsanitizer-collection.so` or add a second tracked
/// process. Removes the injection variables and the sanitizer directory from `LD_LIBRARY_PATH`.
pub fn without_injection<I>(mut cmd: Command, vars: I) -> Command
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    let vars: Vec<(OsString, OsString)> = vars.into_iter().collect();
    if !vars.iter().any(|(k, _)| k == "RCSR_UNDER_SANITIZER") {
        return cmd;
    }
    for (key, value) in &vars {
        let name = key.to_string_lossy();
        if INJECTION_PREFIXES.iter().any(|p| name.starts_with(p)) || INJECTION_NAMES.contains(&&*name) {
            cmd.env_remove(key);
        } else if key == OsStr::new("LD_LIBRARY_PATH") {
            let kept: Vec<_> = std::env::split_paths(value)
                .filter(|p| !p.components().any(|c| c.as_os_str() == "compute-sanitizer"))
                .collect();
            if kept.is_empty() {
                cmd.env_remove(key);
            } else {
                cmd.env(key, std::env::join_paths(kept).expect("join library path"));
            }
        }
    }
    cmd
}

fn child(program: &str) -> Command {
    without_injection(Command::new(program), std::env::vars_os())
}

/// `rcsr` with `RCSR_GPU=0`: no device probe, no CUDA context.
pub fn rcsr() -> Command {
    let mut cmd = child(env!("CARGO_BIN_EXE_rcsr"));
    cmd.env("RCSR_GPU", "0");
    cmd
}

/// `rcsr-bench` with `RCSR_GPU=0`, for the same reason as [`rcsr`].
pub fn rcsr_bench() -> Command {
    let mut cmd = child(env!("CARGO_BIN_EXE_rcsr-bench"));
    cmd.env("RCSR_GPU", "0");
    cmd
}

/// `rcsr` that keeps the caller's environment: the default backend and `--gpu` run on the
/// device. The test names a device purpose by calling this.
pub fn rcsr_on_device() -> Command {
    child(env!("CARGO_BIN_EXE_rcsr"))
}

/// Flags whose purpose is the device: `rcsr_for` keeps the device when any of them is present.
const DEVICE_FLAGS: &[&str] = &["--gpu", "--dispatch", "--batch-pnl", "--device-pnl", "--gpu-memory"];

/// `rcsr_on_device` when `args` carry a device flag, else `rcsr`. For a helper that serves
/// device and CPU tests alike and takes the arguments from the test.
pub fn rcsr_for(args: &[&str]) -> Command {
    if args.iter().any(|a| DEVICE_FLAGS.contains(a) || a.starts_with("--gpu=")) {
        rcsr_on_device()
    } else {
        rcsr()
    }
}

/// Write `input` to the piped stdin of `child` and close it. A child that exits before it
/// reads (a refused argument or file) closes the pipe: `BrokenPipe` then means "already
/// exited", and the caller asserts on the exit status and output. Any other error panics.
pub fn feed_stdin(child: &mut std::process::Child, input: &[u8]) {
    use std::io::Write as _;
    let mut stdin = child.stdin.take().expect("the child has a piped stdin");
    if let Err(e) = stdin.write_all(input) {
        assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe, "cannot write the child's stdin: {e}");
    }
}
