//! #744: `rc_launch` refuses a kernel launch whose pointer argument is not
//! NULL, device or managed memory. A refused kernel never runs.

#![cfg(feature = "cuda")]

use std::os::raw::{c_char, c_int};
use std::process::Command;

extern "C" {
    fn rc_test_launch_guard(kind: c_int, out: *mut c_int, kernel: *mut c_char, len: c_int) -> c_int;
}

const DEVICE: c_int = 0;
const MALLOC: c_int = 1;
const STACK: c_int = 2;
const PINNED: c_int = 3;
const NULL: c_int = 4;
const MANAGED: c_int = 5;
const STRUCT_HOST: c_int = 6;
const STRUCT_DEVICE: c_int = 7;

const REFUSED: c_int = 1;

struct Outcome {
    launch: c_int,
    first_error: c_int,
    second_error: c_int,
    ran: c_int,
    arg: c_int,
    kernel: String,
}

fn launch(kind: c_int) -> Option<Outcome> {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let mut out = [0 as c_int; 5];
    let mut name = [0 as c_char; 96];
    let rc = unsafe { rc_test_launch_guard(kind, out.as_mut_ptr(), name.as_mut_ptr(), 96) };
    assert_eq!(rc, 0, "test setup failed for kind {kind}");
    let kernel = name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8 as char)
        .collect();
    Some(Outcome {
        launch: out[0],
        first_error: out[1],
        second_error: out[2],
        ran: out[3],
        arg: out[4],
        kernel,
    })
}

fn assert_refused(kind: c_int, kernel: &str, arg: c_int) {
    let Some(o) = launch(kind) else { return };
    assert_eq!(o.launch, REFUSED, "kind {kind}");
    assert_eq!(o.ran, 0, "kind {kind}: the kernel ran");
    assert_eq!(o.first_error, REFUSED, "kind {kind}");
    assert_eq!(o.second_error, 0, "kind {kind}: the error is reported once");
    assert_eq!(o.kernel, kernel);
    assert_eq!(o.arg, arg);
}

fn assert_launched(kind: c_int) {
    let Some(o) = launch(kind) else { return };
    assert_eq!(o.launch, 0, "kind {kind}");
    assert_eq!(o.ran, 1, "kind {kind}: the kernel did not run");
    assert_eq!(o.first_error, 0, "kind {kind}");
    assert_eq!(o.arg, -1, "kind {kind}: no refusal is recorded");
}

#[test]
fn device_argument_launches() {
    assert_launched(DEVICE);
}

#[test]
fn malloc_argument_is_refused_before_the_kernel_runs() {
    assert_refused(MALLOC, "k_guard_touch", 1);
}

#[test]
fn stack_argument_is_refused() {
    assert_refused(STACK, "k_guard_touch", 1);
}

#[test]
fn pinned_argument_is_refused() {
    assert_refused(PINNED, "k_guard_touch", 1);
}

#[test]
fn null_argument_launches() {
    assert_launched(NULL);
}

#[test]
fn managed_argument_launches() {
    assert_launched(MANAGED);
}

#[test]
fn by_value_struct_with_a_host_member_is_refused() {
    assert_refused(STRUCT_HOST, "k_guard_touch_struct", 0);
}

#[test]
fn by_value_struct_with_device_members_launches() {
    assert_launched(STRUCT_DEVICE);
}

/// The environment is read once per process, so each setting runs in a child
/// process that executes `child_probe`.
fn child(env: Option<&str>) -> String {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "child_probe", "--nocapture", "--test-threads=1"]);
    cmd.env("RCSR_GUARD_CHILD", "1");
    cmd.env_remove("RCSR_DEVICE_PTR_CHECK");
    if let Some(v) = env {
        cmd.env("RCSR_DEVICE_PTR_CHECK", v);
    }
    let out = cmd.output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn child_probe() {
    if std::env::var("RCSR_GUARD_CHILD").is_err() {
        return;
    }
    let Some(o) = launch(MALLOC) else {
        println!("probe: skipped");
        return;
    };
    println!("probe: launch={} ran={}", o.launch, o.ran);
}

#[test]
fn opt_out_disables_the_check() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    // The test kernel never reads the argument, so it is safe to let it run.
    assert!(child(None).contains("probe: launch=1 ran=0"));
    assert!(child(Some("1")).contains("probe: launch=1 ran=0"));
    assert!(child(Some("0")).contains("probe: launch=0 ran=1"));
}
