//! #822: a test builds its `rcsr` child through `tests/common/rcsr.rs`, which sets
//! `RCSR_GPU=0` unless the test calls `rcsr_on_device`. A bare
//! `Command::new(env!("CARGO_BIN_EXE_rcsr"))` makes a device call from the child in a CUDA
//! build, which crashes or hangs a compute-sanitizer pass (#805). Any use of the binary
//! path outside the helper fails here, except the entries below.

use std::fs;
use std::path::{Path, PathBuf};

const HELPER: &str = "tests/common/rcsr.rs";

/// `(file, reason)`: files that use the binary path for something other than a plain child.
const ALLOWED: &[(&str, &str)] = &[
    ("tests/report_integrity.rs", "hashes and copies the binary; the child it starts goes through the manifest verifier"),
    ("tests/launchers.rs", "gives the path to rcsr-launch scripts that run with RCSR_GPU=0"),
    ("tests/cpu_makes_no_device_call.rs", "counts /dev/nvidia opens of rcsr under strace and must not set RCSR_GPU=0"),
    ("tests/rcsr_command_lint.rs", "names the token it searches for"),
];

fn uses_the_binary_path(line: &str) -> bool {
    let t = line.trim_start();
    !t.starts_with("//") && (t.contains("CARGO_BIN_EXE_rcsr\"") || t.contains("CARGO_BIN_EXE_rcsr-bench\""))
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
fn no_test_spawns_rcsr_outside_the_shared_helper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("tests"), &mut files);
    files.sort();
    let mut problems = Vec::new();
    let mut used = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if rel == HELPER {
            continue;
        }
        let hits: Vec<usize> = fs::read_to_string(&file)
            .unwrap()
            .lines()
            .enumerate()
            .filter(|(_, l)| uses_the_binary_path(l))
            .map(|(n, _)| n + 1)
            .collect();
        if hits.is_empty() {
            continue;
        }
        if ALLOWED.iter().any(|(path, _)| *path == rel) {
            used.push(rel);
        } else {
            problems.push(format!("{rel}: line(s) {hits:?} use CARGO_BIN_EXE_rcsr; build the child with common/rcsr.rs (rcsr() or rcsr_on_device())"));
        }
    }
    for (path, reason) in ALLOWED {
        assert!(!reason.is_empty(), "{path} needs a reason");
        if *path != "tests/rcsr_command_lint.rs" && !used.iter().any(|u| u == path) {
            problems.push(format!("stale allowed entry {path}"));
        }
    }
    assert!(problems.is_empty(), "bare rcsr spawns:\n{}", problems.join("\n"));
}

#[test]
fn the_matcher_flags_a_bare_spawn_and_not_the_other_binaries() {
    assert!(uses_the_binary_path("    Command::new(env!(\"CARGO_BIN_EXE_rcsr\"))"));
    assert!(uses_the_binary_path("let bin = env!(\"CARGO_BIN_EXE_rcsr\");"));
    assert!(uses_the_binary_path("std::process::Command::new(env!(\"CARGO_BIN_EXE_rcsr\"))"));
    assert!(uses_the_binary_path("Command::new(env!(\"CARGO_BIN_EXE_rcsr-bench\"))"));
    assert!(!uses_the_binary_path("Command::new(env!(\"CARGO_BIN_EXE_rcsr-gpu-bench\"))"));
    assert!(!uses_the_binary_path("// Command::new(env!(\"CARGO_BIN_EXE_rcsr\"))"));
    assert!(!uses_the_binary_path("rcsr_cmd::rcsr()"));
}

/// The body of `fn child` in the helper source: from its signature to the next top-level `}`.
fn child_body(helper: &str) -> Option<&str> {
    let start = helper.find("fn child(")?;
    let rest = &helper[start..];
    let end = rest.find("\n}\n")?;
    Some(&rest[..end])
}

/// #1262: every spawn goes through `child()`, so a `current_dir` call there gives every child a
/// working directory that holds no stray file named like a relative argument.
#[test]
fn the_helper_child_sets_a_working_directory() {
    let helper = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(HELPER)).unwrap();
    let body = child_body(&helper).unwrap_or_else(|| panic!("{HELPER}: fn child not found"));
    assert!(body.contains(".current_dir("), "{HELPER}: fn child must call .current_dir( (#1262)");
}

#[test]
fn the_child_matcher_finds_the_body_and_its_cwd_call() {
    let with = "fn child(p: &str) -> Command {\n    let mut c = Command::new(p);\n    c.current_dir(d);\n    c\n}\n\nfn other() {}\n";
    let without = "fn child(p: &str) -> Command {\n    Command::new(p)\n}\n\nfn other() { x.current_dir(d); }\n";
    assert!(child_body(with).unwrap().contains(".current_dir("));
    assert!(!child_body(without).unwrap().contains(".current_dir("));
    assert!(child_body("fn other() {}\n").is_none());
}
