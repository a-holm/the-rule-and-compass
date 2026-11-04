//! The workshop scripts parse and their hermetic tests pass (#667).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const LIMIT: Duration = Duration::from_secs(60);

fn tools_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../verksted/_tools")
}

fn shell_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> =
        fs::read_dir(dir).expect("read dir").map(|e| e.expect("dir entry").path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if path.file_name().map(|n| n != "target").unwrap_or(true) {
                shell_files(&path, out);
            }
        } else if path.extension().map(|e| e == "sh").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[test]
fn tools_parse() {
    let mut files = Vec::new();
    shell_files(&tools_dir(), &mut files);
    assert!(!files.is_empty(), "no scripts under {}", tools_dir().display());
    let mut failures = Vec::new();
    for file in &files {
        let out = Command::new("bash").arg("-n").arg(file).output().expect("run bash -n");
        if !out.status.success() {
            failures.push(format!("{}:\n{}", file.display(), String::from_utf8_lossy(&out.stderr)));
        }
    }
    assert!(failures.is_empty(), "bash -n failed:\n{}", failures.join("\n"));
}

#[test]
fn tools_tests_pass() {
    let mut tests: Vec<PathBuf> = fs::read_dir(tools_dir().join("tests"))
        .map(|rd| rd.map(|e| e.expect("dir entry").path()).collect())
        .unwrap_or_default();
    tests.retain(|p| p.extension().map(|e| e == "sh").unwrap_or(false));
    tests.sort();
    let scratch = Scratch::new("logs");
    let log_dir = scratch.path();
    let mut failures = Vec::new();
    for test in &tests {
        let name = test.file_name().and_then(|n| n.to_str()).unwrap_or("test").to_string();
        let log = log_dir.join(format!("{name}.log"));
        let file = fs::File::create(&log).expect("create log");
        let mut cmd = Command::new("bash");
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("RCSR_") {
                cmd.env_remove(key);
            }
        }
        let mut child = cmd
            .arg(test)
            .stdin(Stdio::null())
            .stdout(file.try_clone().expect("clone log handle"))
            .stderr(file)
            .spawn()
            .expect("spawn bash");
        let start = Instant::now();
        let verdict = loop {
            if let Some(status) = child.try_wait().expect("poll child") {
                break if status.success() { None } else { Some(format!("exit {status}")) };
            }
            if start.elapsed() > LIMIT {
                let _ = child.kill();
                let _ = child.wait();
                break Some(format!("killed after {} s", LIMIT.as_secs()));
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        if let Some(why) = verdict {
            let output = fs::read_to_string(&log).unwrap_or_default();
            failures.push(format!("{name}: {why}\n{}", tail(&output, 20)));
        }
    }
    assert!(failures.is_empty(), "script tests failed:\n{}", failures.join("\n\n"));
}
