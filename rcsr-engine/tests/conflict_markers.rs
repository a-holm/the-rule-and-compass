//! Tracked docs and sources must not contain merge conflict markers.

use std::path::PathBuf;
use std::process::Command;

const ROOTS: [&str; 4] = ["docs", "README.md", "my-version/src", "my-version/tests"];

#[test]
fn tracked_files_hold_no_conflict_markers() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let out = Command::new("git")
        .current_dir(&repo)
        .args(["ls-files", "-z", "--"])
        .args(ROOTS)
        .output()
        .expect("run git ls-files");
    assert!(out.status.success(), "git ls-files failed");
    let starts = [["<<<<<<", "<"].concat() + " ", [">>>>>>", ">"].concat() + " "];
    let mut hits = Vec::new();
    for name in String::from_utf8(out.stdout).unwrap().split('\0') {
        if name.is_empty() {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(repo.join(name)) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            if starts.iter().any(|s| line.starts_with(s.as_str())) {
                hits.push(format!("{}:{}: {}", name, i + 1, line));
            }
        }
    }
    assert!(hits.is_empty(), "conflict markers found:\n{}", hits.join("\n"));
}
