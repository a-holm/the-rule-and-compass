//! Tracked docs and sources must not contain merge conflict markers.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn tracked_files_hold_no_conflict_markers() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_name = manifest.file_name().expect("crate directory name").to_str().expect("utf-8 name");
    let roots = ["docs".to_string(), "README.md".to_string(), format!("{crate_name}/src"), format!("{crate_name}/tests")];
    let repo = manifest.join("..");
    let out = Command::new("git")
        .current_dir(&repo)
        .args(["ls-files", "-z", "--"])
        .args(&roots)
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
