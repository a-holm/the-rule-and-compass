//! #866: on a CUDA build `search::Config::default()` has `gpu: true`, so a test that runs
//! `search::run` on a default config uses the device in the test process. A test with no device
//! purpose states `gpu: false`; a device test states `gpu: true` (and `dispatch: Device`).
//! This lint fails on a `Config { .. ..Default::default() }` literal or a bare
//! `Config::default()` in a test file that imports `rcsr::search::Config` and never states `gpu`
//! for that config. A literal that spreads another helper (`..config(5)`) inherits its choice.

use std::fs;
use std::path::{Path, PathBuf};

/// `(file, reason)`: tests that run the default backend on purpose.
const ALLOWED: &[(&str, &str)] = &[(
    "tests/refit_band_default.rs",
    "reads a field of the default config and runs no search",
)];

/// Problems of one file: the 1-based line of each `Config` literal or default call that does not
/// state `gpu`.
fn unstated_gpu(src: &str) -> Vec<usize> {
    let imports_config = !src.contains("feature = \"cuda\"") && (src.contains("search::Config") || (src.contains("search::{") && src.contains("Config")));
    if !imports_config {
        return Vec::new();
    }
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut from = 0;
    while let Some(at) = src[from..].find("Config") {
        let i = from + at;
        from = i + "Config".len();
        let before_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !before_ok || src[..i].trim_end().ends_with("..") {
            continue;
        }
        let rest = &src[from..];
        let line = src[..i].matches('\n').count() + 1;
        if rest.starts_with("::default()") {
            // `Config { gpu: false, ..Config::default() }` is a spread inside a literal that states gpu
            let stmt_start = src[..i].rfind(';').map_or(0, |p| p + 1);
            if !src[stmt_start..i].contains("gpu:") {
                out.push(line);
            }
        } else if rest.starts_with(" {") {
            let body_start = from + 2;
            let mut depth = 1;
            let mut j = body_start;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let body = &src[body_start..j.saturating_sub(1)];
            let spreads_default = body.contains("..Default::default()") || body.contains("..Config::default()");
            if spreads_default && !body.contains("gpu:") {
                out.push(line);
            }
        }
    }
    out
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
fn no_test_runs_a_default_search_config_without_stating_gpu() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("tests"), &mut files);
    files.sort();
    let mut problems = Vec::new();
    let mut used = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if rel == "tests/cpu_config_lint.rs" {
            continue;
        }
        let hits = unstated_gpu(&fs::read_to_string(&file).unwrap());
        if hits.is_empty() {
            continue;
        }
        if ALLOWED.iter().any(|(p, _)| *p == rel) {
            used.push(rel);
        } else {
            problems.push(format!("{rel}: line(s) {hits:?} build a search Config from the default without `gpu:`"));
        }
    }
    for (path, reason) in ALLOWED {
        assert!(!reason.is_empty(), "{path} needs a reason");
        if !used.iter().any(|u| u == path) {
            problems.push(format!("stale allowed entry {path}"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn the_matcher_flags_defaults_and_accepts_stated_or_inherited_gpu() {
    let head = "use rcsr::search::{self, Config};\n";
    assert_eq!(unstated_gpu(&format!("{head}let c = Config {{ pop: 1, ..Default::default() }};")), [2]);
    assert_eq!(unstated_gpu(&format!("{head}let mut c = Config::default();")), [2]);
    assert!(unstated_gpu(&format!("{head}let c = Config {{ gpu: false, ..Default::default() }};")).is_empty());
    assert!(unstated_gpu(&format!("{head}let c = Config {{ gpu: true, ..config(1) }};")).is_empty());
    assert!(unstated_gpu(&format!("{head}let c = Config {{ pop: 2, ..config(1) }};")).is_empty());
    assert!(unstated_gpu(&format!("{head}let mut c = Config {{ gpu: false, ..Config::default() }};")).is_empty());
    assert!(unstated_gpu("let c = PrepConfig::default();").is_empty());
    assert!(unstated_gpu("#[cfg(feature = \"cuda\")]\nuse rcsr::search::Config;\nlet c = Config::default();").is_empty(), "CUDA-gated files carry device tests");
    assert!(unstated_gpu("let c = Config::default();").is_empty(), "files that do not import the search config are skipped");
}
