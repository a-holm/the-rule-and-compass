//! The repository is Rust only (#366): no tracked Python file, no `python`
//! shebang in any tracked file.

use std::path::Path;
use std::process::Command;

fn tracked_files() -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let output = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(&root)
        .output()
        .expect("run git ls-files");
    assert!(output.status.success(), "git ls-files failed: {:?}", output);
    String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn no_python_file_or_shebang_is_tracked() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut offenders = Vec::new();
    for rel in tracked_files() {
        if rel.ends_with(".py") {
            offenders.push(format!("{rel}: .py file"));
            continue;
        }
        let Ok(bytes) = std::fs::read(root.join(&rel)) else {
            continue;
        };
        let first = bytes.split(|b| *b == b'\n').next().unwrap_or(&[]);
        if first.starts_with(b"#!") && String::from_utf8_lossy(first).contains("python") {
            offenders.push(format!("{rel}: python shebang"));
        }
    }
    assert!(offenders.is_empty(), "Python is not allowed:\n{}", offenders.join("\n"));
}
