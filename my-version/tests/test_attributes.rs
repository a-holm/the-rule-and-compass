//! Guard: a function in a `#[cfg(test)]` module that is neither a test nor
//! referenced anywhere in its file is a test that lost its `#[test]`.

use std::fs;
use std::path::{Path, PathBuf};

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

fn fn_name(line: &str) -> Option<&str> {
    let mut rest = line.strip_prefix("    ")?;
    if rest.starts_with(' ') {
        return None;
    }
    for prefix in ["pub(crate) ", "pub ", "const ", "unsafe ", "async "] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    let rest = rest.strip_prefix("fn ")?;
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    Some(&rest[..end])
}

fn mentions(text: &str, name: &str) -> usize {
    text.match_indices(name)
        .filter(|(i, _)| {
            let before = text[..*i].chars().next_back();
            let after = text[i + name.len()..].chars().next();
            let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
            !word(before) && !word(after)
        })
        .count()
}

fn scan(text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let mut in_test_code = false;
    let mut attrs: Vec<&str> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(test)]" {
            in_test_code = true;
        }
        if trimmed.starts_with("#[") || trimmed.starts_with("///") {
            attrs.push(trimmed);
            continue;
        }
        if in_test_code {
            let tests = attrs.iter().filter(|a| **a == "#[test]").count();
            if tests > 1 {
                problems.push(format!("line {}: duplicated #[test]", n + 1));
            }
            if let Some(name) = fn_name(line) {
                let runs = tests > 0 || attrs.iter().any(|a| a.starts_with("#[ignore"));
                if !runs && mentions(text, name) < 2 {
                    problems.push(format!("line {}: `{name}` is never run or called", n + 1));
                }
            }
        }
        attrs.clear();
    }
    problems
}

#[test]
fn every_function_in_a_test_module_is_a_test_or_is_used() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    let mut problems = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file).unwrap();
        for p in scan(&text) {
            problems.push(format!("{}: {p}", file.display()));
        }
    }
    assert!(problems.is_empty(), "lost #[test] attributes:\n{}", problems.join("\n"));
}

#[test]
fn the_scan_flags_the_known_defects() {
    let lost = "#[cfg(test)]\nmod tests {\n    /// doc\n    fn a_test() {}\n}\n";
    assert_eq!(scan(lost).len(), 1);
    let joined = "#[cfg(test)]\nmod tests {\n    /// doc    #[test]\n    fn a_test() {}\n}\n";
    assert_eq!(scan(joined).len(), 1);
    let dup = "#[cfg(test)]\nmod tests {\n    #[test]\n    /// doc\n    #[test]\n    fn a_test() {}\n}\n";
    assert_eq!(scan(dup).len(), 1);
    let ok = "#[cfg(test)]\nmod tests {\n    fn helper() {}\n    #[test]\n    fn a_test() { helper(); }\n}\n";
    assert!(scan(ok).is_empty());
}
