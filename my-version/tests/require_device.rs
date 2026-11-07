//! #829: `rcsr::backend::skip_without_device` lets a device test skip, and panics when
//! `RCSR_REQUIRE_DEVICE=1`. The checks run the helper in a child process of this test
//! binary, so the environment of the parallel tests of this process is never changed.

use std::process::Command;

const CHILD: &str = "RCSR_REQUIRE_DEVICE_TEST_CHILD";
const REASON: &str = "no usable CUDA device (child)";

/// Runs only as the child of the tests below; on its own it calls nothing.
#[test]
fn helper_child() {
    if std::env::var_os(CHILD).is_some() {
        rcsr::backend::skip_without_device(REASON);
        println!("helper_child returned");
    }
}

fn run_child(require: Option<&str>) -> std::process::Output {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["helper_child", "--exact", "--nocapture", "--test-threads", "1"]).env(CHILD, "1");
    match require {
        Some(v) => c.env("RCSR_REQUIRE_DEVICE", v),
        None => c.env_remove("RCSR_REQUIRE_DEVICE"),
    };
    c.output().unwrap()
}

fn text(o: &std::process::Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn a_required_device_that_is_missing_panics() {
    let o = run_child(Some("1"));
    let t = text(&o);
    assert!(!o.status.success(), "{t}");
    assert!(t.contains("RCSR_REQUIRE_DEVICE=1 but the device test cannot run"), "{t}");
    assert!(t.contains(REASON), "{t}");
    assert!(!t.contains("helper_child returned"), "{t}");
}

#[test]
fn without_the_variable_the_test_skips_and_says_why() {
    for v in [None, Some("0"), Some("")] {
        let o = run_child(v);
        let t = text(&o);
        assert!(o.status.success(), "{v:?}: {t}");
        assert!(t.contains(REASON), "{v:?}: {t}");
        assert!(t.contains("helper_child returned"), "{v:?}: {t}");
    }
}
