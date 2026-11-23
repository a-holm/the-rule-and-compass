use std::process::Command;

#[test]
#[ignore = "needs RCSR_ATTACHABLE=1 and gdb"]
fn test_binaries_are_attachable_when_asked() {
    if std::env::var("RCSR_ATTACHABLE").as_deref() != Ok("1") {
        eprintln!("RCSR_ATTACHABLE is not 1; skipping");
        return;
    }
    let _ = rcsr::attach::linked();
    let out = Command::new("gdb")
        .args(["-batch", "-p", &std::process::id().to_string()])
        .args(["-ex", "set pagination off", "-ex", "thread apply all bt"])
        .output()
        .expect("gdb must be installed");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("test_binaries_are_attachable_when_asked"),
        "no own test frame in backtrace:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
