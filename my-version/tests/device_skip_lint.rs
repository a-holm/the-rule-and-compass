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

/// `RCSR_REQUIRE_DEVICE` is read only inside `backend::skip_without_device`, and no production
/// code calls the helper: it may appear in `src/` only after the `#[cfg(test)]` marker.
#[test]
fn the_helper_and_its_variable_stay_out_of_production_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    files.sort();
    let mut problems = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if rel == "src/backend.rs" {
            continue;
        }
        let text = fs::read_to_string(&file).unwrap();
        let test_part = text.find("\n#[cfg(test)]").unwrap_or(text.len());
        let whole_file_is_tests = rel.ends_with("_tests.rs");
        let mut offset = 0usize;
        for (n, line) in text.lines().enumerate() {
            let line_start = offset;
            offset += line.len() + 1;
            let in_tests = whole_file_is_tests || line_start >= test_part;
            if line.contains("RCSR_REQUIRE_DEVICE") || line.contains("REQUIRE_DEVICE_ENV") {
                problems.push(format!("{rel}:{}: reads the require-device variable outside backend.rs", n + 1));
            }
            if !in_tests && line.contains("skip_without_device") {
                problems.push(format!("{rel}:{}: production code calls skip_without_device", n + 1));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
