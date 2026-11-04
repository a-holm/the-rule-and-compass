//! Guard against two sources of unstable tests (#743).
//!
//! R1: no `set_var` or `remove_var` call in `tests/` or `src/`. A process-wide
//!     environment change races with every other test thread and leaks into
//!     spawned children.
//! R2: no `Instant::now` and no `.elapsed()` inside a `#[test]` without
//!     `#[ignore`. Measurements belong in ignored tests or `rcsr-bench`.
//!
//! `tests/hygiene-allow.txt` lists exemptions as `<path>:<fn> <reason>`. An
//! entry that matches no hit fails the test.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
struct Hit {
    line: usize,
    rule: &'static str,
    what: &'static str,
    func: String,
}

/// Replaces comments, string literals and char literals with spaces. Newlines
/// stay, so line numbers survive.
fn strip(src: &str) -> Vec<char> {
    let c: Vec<char> = src.chars().collect();
    let mut out = c.clone();
    let blank = |out: &mut Vec<char>, from: usize, to: usize| {
        for ch in &mut out[from..to] {
            if *ch != '\n' {
                *ch = ' ';
            }
        }
    };
    let mut i = 0;
    while i < c.len() {
        let rest = &c[i..];
        if rest.starts_with(&['/', '/']) {
            let mut j = i;
            while j < c.len() && c[j] != '\n' {
                j += 1;
            }
            blank(&mut out, i, j);
            i = j;
        } else if rest.starts_with(&['/', '*']) {
            let (mut j, mut depth) = (i + 2, 1);
            while j < c.len() && depth > 0 {
                if c[j..].starts_with(&['/', '*']) {
                    depth += 1;
                    j += 2;
                } else if c[j..].starts_with(&['*', '/']) {
                    depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            blank(&mut out, i, j);
            i = j;
        } else if c[i] == 'r' && raw_open(&c, i).is_some() {
            let (hashes, start) = raw_open(&c, i).unwrap();
            let close: Vec<char> = std::iter::once('"').chain(std::iter::repeat_n('#', hashes)).collect();
            let mut j = start;
            while j < c.len() && !c[j..].starts_with(&close) {
                j += 1;
            }
            let end = (j + close.len()).min(c.len());
            blank(&mut out, i, end);
            i = end;
        } else if c[i] == '"' {
            let mut j = i + 1;
            while j < c.len() && c[j] != '"' {
                j += if c[j] == '\\' { 2 } else { 1 };
            }
            let end = (j + 1).min(c.len());
            blank(&mut out, i, end);
            i = end;
        } else if c[i] == '\'' {
            let end = if c.get(i + 1) == Some(&'\\') {
                let mut j = i + 2;
                while j < c.len() && c[j] != '\'' {
                    j += 1;
                }
                Some(j + 1)
            } else if c.get(i + 2) == Some(&'\'') {
                Some(i + 3)
            } else {
                None
            };
            match end {
                Some(end) => {
                    let end = end.min(c.len());
                    blank(&mut out, i, end);
                    i = end;
                }
                None => i += 1,
            }
        } else {
            i += 1;
        }
    }
    out
}

/// At `r`: the `#` count and the index after the opening quote of a raw string.
fn raw_open(c: &[char], i: usize) -> Option<(usize, usize)> {
    if i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_') {
        return None;
    }
    let mut j = i + 1;
    while c.get(j) == Some(&'#') {
        j += 1;
    }
    (c.get(j) == Some(&'"')).then_some((j - i - 1, j + 1))
}

fn is_ident(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

struct Frame {
    name: String,
    gate: bool,
    depth: usize,
}

fn scan(src: &str) -> Vec<Hit> {
    let c = strip(src);
    let mut hits = Vec::new();
    let (mut line, mut depth, mut parens) = (1, 0usize, 0usize);
    let mut attrs: Vec<String> = Vec::new();
    let mut pending: Option<(String, bool)> = None;
    let mut frame: Option<Frame> = None;
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        match ch {
            '\n' => line += 1,
            '#' if c.get(i + 1) == Some(&'[') || (c.get(i + 1) == Some(&'!') && c.get(i + 2) == Some(&'[')) => {
                let mut j = i + 1;
                while c[j] != '[' {
                    j += 1;
                }
                let (start, mut d) = (j + 1, 1);
                j += 1;
                while j < c.len() && d > 0 {
                    match c[j] {
                        '[' => d += 1,
                        ']' => d -= 1,
                        '\n' => line += 1,
                        _ => {}
                    }
                    j += 1;
                }
                let text: String = c[start..j - 1].iter().collect();
                attrs.push(text.split_whitespace().collect());
                i = j;
                continue;
            }
            '(' | '[' => parens += 1,
            ')' | ']' => parens = parens.saturating_sub(1),
            ';' if parens == 0 => {
                pending = None;
                attrs.clear();
            }
            '{' => {
                depth += 1;
                if let Some((name, gate)) = pending.take() {
                    frame = Some(Frame { name, gate, depth });
                }
                attrs.clear();
            }
            '}' => {
                if frame.as_ref().is_some_and(|f| f.depth == depth) {
                    frame = None;
                }
                depth = depth.saturating_sub(1);
                attrs.clear();
            }
            _ if is_ident(ch) && (i == 0 || !is_ident(c[i - 1])) => {
                let mut j = i;
                while j < c.len() && is_ident(c[j]) {
                    j += 1;
                }
                let word: String = c[i..j].iter().collect();
                let mut k = j;
                while k < c.len() && c[k].is_whitespace() {
                    k += 1;
                }
                let next = c.get(k).copied();
                let prev_dot = c[..i].iter().rev().find(|x| !x.is_whitespace()) == Some(&'.');
                let func = || frame.as_ref().map_or("<file>".to_string(), |f| f.name.clone());
                match word.as_str() {
                    "fn" if frame.is_none() && next.is_some_and(is_ident) => {
                        let mut e = k;
                        while e < c.len() && is_ident(c[e]) {
                            e += 1;
                        }
                        let gate = attrs.iter().any(|a| a == "test")
                            && !attrs.iter().any(|a| a.starts_with("ignore"));
                        pending = Some((c[k..e].iter().collect(), gate));
                    }
                    "set_var" | "remove_var" if next == Some('(') => hits.push(Hit {
                        line,
                        rule: "R1",
                        what: if word == "set_var" { "set_var" } else { "remove_var" },
                        func: func(),
                    }),
                    "Instant" if c[k..].starts_with(&[':', ':']) && ident_after(&c, k + 2) == "now" => {
                        if frame.as_ref().is_some_and(|f| f.gate) {
                            hits.push(Hit { line, rule: "R2", what: "Instant::now", func: func() });
                        }
                    }
                    "elapsed" if prev_dot && next == Some('(') => {
                        if frame.as_ref().is_some_and(|f| f.gate) {
                            hits.push(Hit { line, rule: "R2", what: ".elapsed()", func: func() });
                        }
                    }
                    _ => {}
                }
                i = j;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    hits
}

fn ident_after(c: &[char], from: usize) -> String {
    let mut j = from;
    while j < c.len() && c[j].is_whitespace() {
        j += 1;
    }
    let mut e = j;
    while e < c.len() && is_ident(c[e]) {
        e += 1;
    }
    c[j..e].iter().collect()
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

/// Entries as `(path, fn)`; every entry needs a reason.
fn parse_allow(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut entries = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, reason) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let (path, func) = key.split_once(':').ok_or(format!("line {}: expected <path>:<fn>", n + 1))?;
        if reason.trim().is_empty() {
            return Err(format!("line {}: `{key}` has no reason", n + 1));
        }
        let entry = (path.to_string(), func.to_string());
        if entries.contains(&entry) {
            return Err(format!("line {}: duplicate entry `{key}`", n + 1));
        }
        entries.push(entry);
    }
    Ok(entries)
}

#[test]
fn no_env_mutation_and_no_clock_read_in_gate_tests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let allow = parse_allow(&fs::read_to_string(root.join("tests/hygiene-allow.txt")).unwrap())
        .unwrap_or_else(|e| panic!("tests/hygiene-allow.txt: {e}"));
    let mut files = Vec::new();
    rust_files(&root.join("tests"), &mut files);
    rust_files(&root.join("src"), &mut files);
    files.sort();
    let mut used = BTreeSet::new();
    let mut problems = Vec::new();
    for file in files {
        let rel = file.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let text = fs::read_to_string(&file).unwrap();
        for hit in scan(&text) {
            let entry = (rel.clone(), hit.func.clone());
            if allow.contains(&entry) {
                used.insert(entry);
            } else {
                problems.push(format!("{rel}:{}: {} {} in {}", hit.line, hit.rule, hit.what, hit.func));
            }
        }
    }
    for (path, func) in &allow {
        if !used.contains(&(path.clone(), func.clone())) {
            problems.push(format!("tests/hygiene-allow.txt: stale entry {path}:{func}"));
        }
    }
    assert!(problems.is_empty(), "test hygiene violations:\n{}", problems.join("\n"));
}

#[cfg(test)]
mod scanner {
    use super::*;

    fn rules(src: &str) -> Vec<(usize, &'static str, String)> {
        scan(src).into_iter().map(|h| (h.line, h.rule, h.func)).collect()
    }

    #[test]
    fn flags_env_mutation_anywhere() {
        let src = "fn helper() {\n    std::env::set_var(\"X\", \"1\");\n}\n#[test]\n#[ignore = \"m\"]\nfn t() { std::env::remove_var(\"X\"); }\n";
        assert_eq!(rules(src), [(2, "R1", "helper".into()), (6, "R1", "t".into())]);
    }

    #[test]
    fn flags_clock_in_gate_test_only() {
        let src = "#[test]\nfn gate() {\n    let t = std::time::Instant::now();\n    assert!(t.elapsed() < d);\n}\n";
        assert_eq!(rules(src), [(3, "R2", "gate".into()), (4, "R2", "gate".into())]);
        let ignored = "#[test]\n#[ignore = \"measure\"]\nfn bench() { let t = Instant::now(); t.elapsed(); }\n";
        assert!(rules(ignored).is_empty());
        let helper = "fn helper() { Instant::now(); }\n#[test]\nfn t() { helper(); }\n";
        assert!(rules(helper).is_empty());
    }

    #[test]
    fn ignores_comments_and_literals() {
        let src = "#[test]\nfn t() {\n    // Instant::now()\n    /* set_var( /* nested */ */\n    let s = \"Instant::now() set_var(\";\n    let r = r#\"x.elapsed() \" set_var(\"#;\n    let c = '\"'; let l: &'static str = \"\";\n    let d = std::time::Duration::from_secs(1);\n}\n";
        assert!(rules(src).is_empty());
    }

    #[test]
    fn nested_fn_belongs_to_the_outer_test() {
        let src = "#[test]\nfn outer() {\n    fn inner() { let _ = Instant::now(); }\n    inner();\n}\n";
        assert_eq!(rules(src), [(3, "R2", "outer".into())]);
    }

    #[test]
    fn test_inside_module_is_found() {
        let src = "#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn a() { x.elapsed(); }\n}\n";
        assert_eq!(rules(src), [(5, "R2", "a".into())]);
    }

    #[test]
    fn bodiless_fn_does_not_capture_next_block() {
        let src = "trait T { fn f(&self); }\nfn g() { Instant::now(); }\n";
        assert!(rules(src).is_empty());
    }

    #[test]
    fn allow_file_needs_reason_and_rejects_duplicates() {
        assert_eq!(parse_allow("# c\n\na.rs:f because\n").unwrap(), [("a.rs".into(), "f".into())]);
        assert!(parse_allow("a.rs:f\n").is_err());
        assert!(parse_allow("a.rs:f r\na.rs:f r\n").is_err());
        assert!(parse_allow("a.rs r\n").is_err());
    }
}
