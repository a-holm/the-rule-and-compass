//! Guard: a function directly inside a `#[cfg(test)]` module that is neither a
//! test nor referenced anywhere in its file is a test that lost its `#[test]`.
//! A `#[test]` joined onto a comment line, or repeated, is caught too.

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

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

fn fn_name(item: &str) -> Option<&str> {
    let mut rest = item;
    for prefix in ["pub(crate) ", "pub ", "const ", "unsafe ", "async "] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    let rest = rest.strip_prefix("fn ")?;
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    Some(&rest[..end])
}

fn mentions(text: &str, name: &str) -> usize {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    text.match_indices(name)
        .filter(|(i, _)| {
            !word(text[..*i].chars().next_back()) && !word(text[i + name.len()..].chars().next())
        })
        .count()
}

/// `whole_file`: the file is itself a `#[cfg(test)]` module (declared with
/// `mod name;`), so its items sit at indent 0.
fn scan(text: &str, whole_file: bool) -> Vec<String> {
    let mut problems = Vec::new();
    // Indent of the items inside the current test module.
    let mut item_indent: Option<usize> = whole_file.then_some(0);
    let mut module_indent = 0;
    let mut cfg_test_pending = false;
    let mut attrs: Vec<&str> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(test)]" {
            cfg_test_pending = true;
        }
        if trimmed.starts_with("#[") || trimmed.starts_with("///") {
            attrs.push(trimmed);
            continue;
        }
        if trimmed.is_empty() {
            attrs.clear();
            continue;
        }
        if let Some(ind) = item_indent {
            if !whole_file && indent(line) == module_indent && trimmed == "}" {
                item_indent = None;
            } else if indent(line) == ind {
                let tests = attrs.iter().filter(|a| **a == "#[test]").count();
                if tests > 1 {
                    problems.push(format!("line {}: duplicated #[test]", n + 1));
                }
                if let Some(name) = fn_name(trimmed) {
                    let runs = tests > 0 || attrs.iter().any(|a| a.starts_with("#[ignore"));
                    if !runs && mentions(text, name) < 2 {
                        problems.push(format!("line {}: `{name}` is never run or called", n + 1));
                    }
                }
            }
        } else if cfg_test_pending && trimmed.starts_with("mod ") && trimmed.ends_with('{') {
            module_indent = indent(line);
            item_indent = Some(module_indent + 4);
        }
        cfg_test_pending = false;
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
        let stem = file.file_stem().unwrap().to_string_lossy();
        for p in scan(&text, stem.ends_with("_tests")) {
            problems.push(format!("{}: {p}", file.display()));
        }
    }
    assert!(problems.is_empty(), "lost #[test] attributes:\n{}", problems.join("\n"));
}

#[test]
fn the_scan_flags_the_known_defects() {
    let lost = "#[cfg(test)]\nmod tests {\n    /// doc\n    fn a_test() {}\n}\n";
    assert_eq!(scan(lost, false).len(), 1);
    let joined = "#[cfg(test)]\nmod tests {\n    /// doc    #[test]\n    fn a_test() {}\n}\n";
    assert_eq!(scan(joined, false).len(), 1);
    let dup = "#[cfg(test)]\nmod tests {\n    #[test]\n    /// doc\n    #[test]\n    fn a_test() {}\n}\n";
    assert_eq!(scan(dup, false).len(), 1);
    let ok = "#[cfg(test)]\nmod tests {\n    fn helper() {}\n    #[test]\n    fn a_test() { helper(); }\n}\n";
    assert!(scan(ok, false).is_empty());
    let production = "#[cfg(test)]\nmod tests {\n}\nimpl X {\n    fn unused() {}\n}\n";
    assert!(scan(production, false).is_empty());
    assert_eq!(scan("fn a_test() {}\n", true).len(), 1);
}
