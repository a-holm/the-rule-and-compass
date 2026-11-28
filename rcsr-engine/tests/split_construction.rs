//! #1467: production code builds an `IntervalSplit` only through `IntervalSplit::checked`,
//! which validates the shape and the bound. The lint reads every `.rs` file under `src/` up to
//! its first line that is exactly `#[cfg(test)]`.

use std::fs;
use std::path::{Path, PathBuf};

const PATTERNS: [&str; 2] = ["IntervalSplit::new(", "IntervalSplit {"];

/// `(file, trimmed line, reason)`: the lines that may hold a pattern.
const ALLOWED: &[(&str, &str, &str)] = &[
    ("src/intervals.rs", "pub struct IntervalSplit {", "the type definition"),
    ("src/intervals.rs", "impl IntervalSplit {", "the methods"),
    (
        "src/intervals.rs",
        "pub fn new(declared: Declared, purge: i64, embargo: i64, embargo_source: EmbargoSource, embargo_column: Option<String>) -> IntervalSplit {",
        "the unchecked constructor of tests",
    ),
    (
        "src/intervals.rs",
        "IntervalSplit { declared, purge: purge.max(0), embargo: embargo.max(0), embargo_source, embargo_column }",
        "the body of new",
    ),
    ("src/intervals.rs", "Ok(IntervalSplit { declared, purge, embargo, embargo_source, embargo_column })", "the body of checked"),
    (
        "src/formal_vectors/split.rs",
        "IntervalSplit::new(d, self.purge, self.embargo, EmbargoSource::Flag, None)",
        "the vector generator builds splits that are not well-formed on purpose, to compare role and effective on them (#1512)",
    ),
];

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

/// `(line number, trimmed line)` of every line before the first `#[cfg(test)]` line that holds a pattern.
fn constructions(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .take_while(|l| l.trim_end() != "#[cfg(test)]")
        .enumerate()
        .filter(|(_, l)| PATTERNS.iter().any(|p| l.contains(p)))
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect()
}

#[test]
fn the_matcher_stops_at_the_test_module() {
    let text = "fn a() {\n    let s = IntervalSplit::new(d, 1, 2, EmbargoSource::Flag, None);\n}\n#[cfg(test)]\nmod tests {\n    IntervalSplit::new(d, 0, 0, s, None);\n}\n";
    assert_eq!(constructions(text), vec![(2, "let s = IntervalSplit::new(d, 1, 2, EmbargoSource::Flag, None);".to_string())]);
    assert_eq!(constructions("let s = IntervalSplit { declared, purge, embargo, embargo_source, embargo_column };\n").len(), 1);
    assert!(constructions("let s = IntervalSplit::checked(d, 1, 2, EmbargoSource::Flag, None)?;\n").is_empty());
}

#[test]
fn no_unchecked_split_construction_in_src() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    files.sort();
    let mut used = vec![false; ALLOWED.len()];
    let mut unchecked = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let text = fs::read_to_string(&file).unwrap();
        for (no, line) in constructions(&text) {
            match ALLOWED.iter().position(|(f, l, _)| *f == rel && *l == line) {
                Some(k) => used[k] = true,
                None => unchecked.push(format!("{rel}:{no}: {line}")),
            }
        }
    }
    let new_calls = unchecked.iter().filter(|u| u.contains(PATTERNS[0])).count();
    assert!(
        unchecked.is_empty(),
        "{new_calls} use(s) of IntervalSplit::new( and {} of a struct literal outside test code; build the split with \
         IntervalSplit::checked:\n{}",
        unchecked.len() - new_calls,
        unchecked.join("\n")
    );
    let stale: Vec<String> = ALLOWED
        .iter()
        .zip(&used)
        .filter(|((f, _, _), u)| !**u && root.join(f).exists())
        .map(|((f, l, why), _)| format!("{f}: {l} ({why})"))
        .collect();
    assert!(stale.is_empty(), "allow-list lines that match nothing:\n{}", stale.join("\n"));
}
