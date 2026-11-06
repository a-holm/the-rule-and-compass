//! #829: a device test that finds no device calls `rcsr::backend::skip_without_device`.
//! A bare `eprintln!("skipping ... device")` followed by a return passes silently in a
//! gate whose device went missing, so it is refused in `tests/` and `src/`.

use std::fs;
use std::path::{Path, PathBuf};

/// Messages that name a skip for a reason other than a missing device.
const OTHER_REASONS: &[&str] = &["sm_121", "a device is usable here", "RCSR_ENRICH_PARITY_CPU", "not found"];

fn is_device_skip_print(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("//") || !t.contains("eprintln!(") {
        return false;
    }
    let l = t.to_lowercase();
    l.contains("skip") && (l.contains("device") || l.contains("gpu") || l.contains("cuda")) && !OTHER_REASONS.iter().any(|r| t.contains(r))
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
fn no_device_skip_prints_its_own_message() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("tests"), &mut files);
    rust_files(&root.join("src"), &mut files);
    files.sort();
    let mut problems = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if rel == "tests/device_skip_lint.rs" || rel == "src/backend.rs" {
            continue;
        }
        for (n, line) in fs::read_to_string(&file).unwrap().lines().enumerate() {
            if is_device_skip_print(line) {
                problems.push(format!("{rel}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "use rcsr::backend::skip_without_device(reason) (crate::backend in src) so RCSR_REQUIRE_DEVICE=1 fails the test:\n{}",
        problems.join("\n")
    );
}

#[test]
fn the_lint_flags_device_skips_and_nothing_else() {
    assert!(is_device_skip_print("        eprintln!(\"skipping GPU test: no usable CUDA device\");"));
    assert!(is_device_skip_print("eprintln!(\"skipped: no CUDA device\");"));
    assert!(is_device_skip_print("eprintln!(\"skip: no CUDA device\")"));
    assert!(!is_device_skip_print("// eprintln!(\"skipping: no CUDA device\");"));
    assert!(!is_device_skip_print("rcsr::backend::skip_without_device(\"skipping: no CUDA device\");"));
    assert!(!is_device_skip_print("eprintln!(\"skipping: RCSR_ENRICH_PARITY_CPU is not set\");"));
    assert!(!is_device_skip_print("eprintln!(\"skipping: a device is usable here\");"));
    assert!(!is_device_skip_print("eprintln!(\"rows={}\", n);"));
}
