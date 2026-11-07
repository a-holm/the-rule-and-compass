//! A test that feeds a child's stdin must not panic when the child has already exited:
//! a refused argument or file closes the pipe before the write, and `BrokenPipe` is then
//! the expected outcome. `common/rcsr.rs::feed_stdin` is the one place that writes; the
//! lint refuses a bare `unwrap` or `expect` on a stdin write anywhere else.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[test]
fn a_child_that_exited_before_reading_is_not_a_write_error() {
    let mut child = Command::new("sh")
        .args(["-c", "exit 3"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // `wait` closes our end of the pipe, so keep the handle and put it back: the child has
    // exited and its read end is closed before the first write.
    let stdin = child.stdin.take();
    assert_eq!(child.wait().unwrap().code(), Some(3));
    child.stdin = stdin;
    rcsr_cmd::feed_stdin(&mut child, &vec![b'x'; 1 << 20]);
    assert_eq!(child.wait_with_output().unwrap().status.code(), Some(3));
}

/// Lines that write a stdin and unwrap the result: `write_all` with `stdin` in the six
/// lines above it and `.unwrap()` or `.expect(` in the two lines from it on.
fn bare_stdin_writes(text: &str) -> Vec<usize> {
    let lines: Vec<&str> = text.lines().collect();
    let mut hits = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("write_all(") || line.trim_start().starts_with("//") {
            continue;
        }
        let before = lines[i.saturating_sub(6)..=i].iter().any(|l| l.contains("stdin"));
        let after = lines[i..lines.len().min(i + 3)]
            .iter()
            .any(|l| l.contains(".unwrap()") || l.contains(".expect("));
        let ignored = lines[i.saturating_sub(6)..=i].iter().any(|l| l.contains("let _ ="));
        if before && after && !ignored {
            hits.push(i + 1);
        }
    }
    hits
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_matcher_flags_a_bare_stdin_write_and_not_the_helper() {
    let bare = "child\n    .stdin\n    .as_mut()\n    .unwrap()\n    .write_all(b\"x\")\n    .unwrap();\n";
    assert_eq!(bare_stdin_writes(bare), vec![5]);
    assert!(bare_stdin_writes("rcsr_cmd::feed_stdin(&mut child, b\"x\");\n").is_empty());
    assert!(bare_stdin_writes("let _ = child\n    .stdin\n    .unwrap()\n    .write_all(b\"x\");\n").is_empty());
    assert!(bare_stdin_writes("file.write_all(b\"x\").unwrap();\n").is_empty());
}

#[test]
fn no_test_unwraps_a_child_stdin_write() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("tests"), &mut files);
    files.sort();
    let mut problems = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if rel == "tests/stdin_write_lint.rs" || rel == "tests/common/rcsr.rs" {
            continue;
        }
        let hits = bare_stdin_writes(&fs::read_to_string(&file).unwrap());
        if !hits.is_empty() {
            problems.push(format!("{rel}: line(s) {hits:?} unwrap a stdin write; use rcsr_cmd::feed_stdin"));
        }
    }
    assert!(problems.is_empty(), "bare stdin writes:\n{}", problems.join("\n"));
}
