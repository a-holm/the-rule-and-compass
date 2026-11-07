//! #822: the one place a test builds an `rcsr` child. A test with no device purpose gets
//! `RCSR_GPU=0`, so its child makes no CUDA call and a compute-sanitizer pass over the
//! test binary traces no device work it does not mean to test. A test that exists to run
//! on the device says so by calling `rcsr_on_device`. `tests/rcsr_command_lint.rs`
//! refuses any other spawn of `CARGO_BIN_EXE_rcsr`.
#![allow(dead_code)]

use std::process::Command;

/// `rcsr` with `RCSR_GPU=0`: no device probe, no CUDA context.
pub fn rcsr() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr"));
    cmd.env("RCSR_GPU", "0");
    cmd
}

/// `rcsr-bench` with `RCSR_GPU=0`, for the same reason as [`rcsr`].
pub fn rcsr_bench() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr-bench"));
    cmd.env("RCSR_GPU", "0");
    cmd
}

/// `rcsr` that keeps the caller's environment: the default backend and `--gpu` run on the
/// device. The test names a device purpose by calling this.
pub fn rcsr_on_device() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
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
