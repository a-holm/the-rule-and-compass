//! The workshop in `verksted/` stays navigable (#368): every relative link and
//! every backticked input path resolves, every pipeline and stage holds the
//! four contract headings, and the entry file routes every pipeline.

use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DIRS: [&str; 3] = ["node_modules", "_archive", "raw"];
const CONTRACT: [&str; 4] = ["Inputs", "Process", "Outputs", "Check"];

#[derive(Default)]
struct Report {
    files: usize,
    checked: usize,
    broken: Vec<String>,
}

fn collect_markdown(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if !name.starts_with('.') && !SKIP_DIRS.contains(&name) {
                collect_markdown(&path, out);
            }
        } else if name.ends_with(".md") {
            out.push(path);
        }
    }
}

fn has_scheme(s: &str) -> bool {
    match s.find(':') {
        Some(i) if i > 0 => s[..i].chars().all(|c| c.is_ascii_lowercase()),
        _ => false,
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn fence_marker(trimmed: &str) -> Option<&'static str> {
    if trimmed.starts_with("```") {
        Some("```")
    } else if trimmed.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    }
}

/// Targets of `[text](target)` links in one line.
fn link_targets(line: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let mut from = 0;
    while let Some(rel) = line[from..].find("](") {
        let close = from + rel;
        let label_ok = line[..close]
            .rfind('[')
            .map(|open| !line[open + 1..close].contains(']'))
            .unwrap_or(false);
        let start = close + 2;
        from = start;
        if !label_ok {
            continue;
        }
        let rest = &line[start..];
        let target = if let Some(inner) = rest.strip_prefix('<') {
            match inner.find('>') {
                Some(end) if inner[end + 1..].starts_with(')') => Some(inner[..end].to_string()),
                _ => None,
            }
        } else {
            let end = rest.find(|c: char| c == ')' || c.is_whitespace());
            match end {
                Some(e) if e > 0 && rest[e..].starts_with(')') => Some(rest[..e].to_string()),
                _ => None,
            }
        };
        if let Some(t) = target {
            targets.push(t);
        }
    }
    targets
}

fn looks_like_path(span: &str) -> bool {
    if span.is_empty()
        || span.chars().any(|c| c.is_whitespace() || "{}<>*?|".contains(c))
        || has_scheme(span)
        || span.starts_with('/')
        || span.starts_with('~')
    {
        return false;
    }
    if span.contains('/') {
        return true;
    }
    match span.rfind('.') {
        Some(i) if i + 1 < span.len() => {
            let ext = &span[i + 1..];
            ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        }
        _ => false,
    }
}

fn backtick_spans(line: &str) -> Vec<&str> {
    let parts: Vec<&str> = line.split('`').collect();
    if parts.len() % 2 == 0 {
        return Vec::new();
    }
    parts.iter().skip(1).step_by(2).copied().collect()
}

fn check_file(root: &Path, file: &Path, report: &mut Report) {
    let text = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let body = match text.strip_prefix("---") {
        Some(rest) => match rest.find("\n---") {
            Some(end) => &rest[end + 4..],
            None => text,
        },
        None => text,
    };
    let rel = file.strip_prefix(root).unwrap_or(file).display().to_string();
    let dir = file.parent().expect("file has a parent");
    let mut fence: Option<&str> = None;
    let mut in_inputs = false;
    let mut strict = false;

    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            }
            continue;
        }
        if let Some(marker) = fence_marker(trimmed) {
            fence = Some(marker);
            continue;
        }

        let hashes = line.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) && line[hashes..].starts_with(char::is_whitespace) {
            let title = line[hashes..].trim().trim_end_matches('#').trim();
            in_inputs = title.eq_ignore_ascii_case("inputs");
            strict = false;
        }

        let lower = trimmed.to_lowercase();
        if let Some(label) = lower.strip_prefix("**") {
            let label = label.trim_start();
            if label.starts_with("this run") {
                strict = false;
            } else if label.starts_with("reference") {
                strict = true;
            }
        }

        let mut targets = link_targets(line);
        if let Some(import) = trimmed.strip_prefix('@') {
            if import.ends_with(".md")
                && !import.contains(char::is_whitespace)
                && !import.starts_with('~')
                && !import.starts_with('/')
            {
                targets.push(import.to_string());
            }
        }
        for target in targets {
            if has_scheme(&target) || target.starts_with('#') || target.contains('{') {
                continue;
            }
            let path_part = target.split('#').next().unwrap_or("");
            if path_part.is_empty() {
                continue;
            }
            report.checked += 1;
            if !dir.join(percent_decode(path_part)).exists() {
                report.broken.push(format!("{rel}: link {target}"));
            }
        }

        let is_bullet = trimmed.starts_with("- ") || trimmed.starts_with("* ");
        if in_inputs && is_bullet {
            let line_strict = strict || lower.contains("reference");
            for span in backtick_spans(line) {
                if !looks_like_path(span) {
                    continue;
                }
                report.checked += 1;
                let candidates = [dir.join(span), root.join(span)];
                if candidates.iter().any(|c| c.exists()) {
                    continue;
                }
                if !line_strict {
                    let under_output = span.split('/').any(|seg| seg == "output");
                    let parent_exists = candidates
                        .iter()
                        .any(|c| c.parent().map(|p| p.exists()).unwrap_or(false));
                    if under_output || parent_exists {
                        continue;
                    }
                }
                report.broken.push(format!("{rel}: input `{span}`"));
            }
        }
    }
}

fn check_tree(root: &Path) -> Report {
    let mut files = Vec::new();
    collect_markdown(root, &mut files);
    let mut report = Report::default();
    for file in &files {
        report.files += 1;
        check_file(root, file, &mut report);
    }
    report
}

/// The `##` headings of a contract file, fenced blocks excluded.
fn level_two_headings(text: &str) -> Vec<String> {
    let mut fence: Option<&str> = None;
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            }
            continue;
        }
        if let Some(marker) = fence_marker(trimmed) {
            fence = Some(marker);
            continue;
        }
        if let Some(title) = line.strip_prefix("## ") {
            out.push(title.trim().to_string());
        }
    }
    out
}

fn contract_problems(root: &Path, file: &Path, expect_prefix: &str) -> Vec<String> {
    let rel = file.strip_prefix(root).unwrap_or(file).display().to_string();
    let text = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
    let mut problems = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    if !lines.first().map(|l| l.starts_with(expect_prefix)).unwrap_or(false) {
        problems.push(format!("{rel}: first line must start with `{expect_prefix}`"));
    }
    if !lines.get(2).map(|l| l.starts_with("One job: ")).unwrap_or(false) {
        problems.push(format!("{rel}: line 3 must start with `One job: `"));
    }
    if !lines.iter().any(|l| l.starts_with("Do NOT load: ")) {
        problems.push(format!("{rel}: missing a `Do NOT load: ` line"));
    }
    let headings = level_two_headings(&text);
    if headings != CONTRACT {
        problems.push(format!("{rel}: ## headings are {headings:?}, expected {CONTRACT:?}"));
    }
    problems
}

fn numbered_subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.len() > 2 && n.as_bytes()[..2].iter().all(u8::is_ascii_digit))
                    .unwrap_or(false)
        })
        .collect();
    dirs.sort();
    dirs
}

fn name_of(path: &Path) -> String {
    path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()
}

fn structure_problems(root: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    let agents = fs::read_to_string(root.join("AGENTS.md")).unwrap_or_default();
    if agents.lines().count() > 60 {
        problems.push("AGENTS.md: more than 60 lines; it routes and holds nothing else".to_string());
    }
    for pipeline in numbered_subdirs(root) {
        let pname = name_of(&pipeline);
        let context = pipeline.join("CONTEXT.md");
        if !context.exists() {
            problems.push(format!("{pname}: missing CONTEXT.md"));
            continue;
        }
        problems.extend(contract_problems(root, &context, &format!("# {pname}: ")));
        if !agents.contains(&format!("{pname}/CONTEXT.md")) {
            problems.push(format!("AGENTS.md: no route to {pname}/CONTEXT.md"));
        }
        let pipeline_text = fs::read_to_string(&context).unwrap_or_default();
        let stages_dir = pipeline.join("stages");
        if !stages_dir.is_dir() {
            problems.push(format!("{pname}: missing stages/"));
            continue;
        }
        for stage in numbered_subdirs(&stages_dir) {
            let sname = name_of(&stage);
            let file = stage.join("CONTEXT.md");
            if !file.exists() {
                problems.push(format!("{pname}/stages/{sname}: missing CONTEXT.md"));
                continue;
            }
            problems.extend(contract_problems(root, &file, &format!("# {sname}: ")));
            if !pipeline_text.contains(&format!("stages/{sname}/")) {
                problems.push(format!("{pname}/CONTEXT.md: does not name stages/{sname}/"));
            }
        }
    }
    problems
}

fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read dir").map(|e| e.expect("dir entry")) {
        let path = entry.path();
        if path.is_dir() {
            python_files(&path, out);
        } else if path.extension().map(|e| e == "py").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn workshop_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../verksted")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("verksted_links_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("create dirs");
    fs::write(path, text).expect("write file");
}

#[test]
fn workshop_links_and_inputs_resolve() {
    let root = workshop_root();
    let report = check_tree(&root);
    assert!(report.files > 0, "no markdown files under {}", root.display());
    assert!(
        report.broken.is_empty(),
        "{} files, {} links and inputs checked, {} broken:\n{}",
        report.files,
        report.checked,
        report.broken.len(),
        report.broken.join("\n")
    );
}

#[test]
fn workshop_contracts_and_routes_are_complete() {
    let problems = structure_problems(&workshop_root());
    assert!(problems.is_empty(), "contract problems:\n{}", problems.join("\n"));
}

#[test]
fn workshop_holds_no_python() {
    let mut found = Vec::new();
    python_files(&workshop_root(), &mut found);
    assert!(found.is_empty(), "Python is not allowed: {found:?}");
}

#[test]
fn a_broken_relative_link_is_reported() {
    let root = scratch("broken_link");
    write(&root.join("a.md"), "# a\n\nSee [b](missing.md) and [c](a.md#top).\n");
    let report = check_tree(&root);
    assert_eq!(report.broken, vec!["a.md: link missing.md".to_string()]);
    assert_eq!(report.checked, 2);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_missing_reference_input_fails_and_a_missing_working_input_under_output_passes() {
    let root = scratch("inputs");
    write(&root.join("rules.md"), "# rules\n");
    write(
        &root.join("s/CONTEXT.md"),
        "# s\n\n## Inputs\n\n**This run:**\n- `../prev/output/x.md`\n\n**Reference (every run):**\n- `../rules.md`\n- `../gone.md`\n",
    );
    let report = check_tree(&root);
    assert_eq!(report.broken, vec!["s/CONTEXT.md: input `../gone.md`".to_string()]);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fenced_blocks_and_placeholders_are_ignored() {
    let root = scratch("fenced");
    write(
        &root.join("a.md"),
        "# a\n\n```sh\n[x](nowhere.md)\n```\n\n## Inputs\n\n- `<path>/x.md`\n- `{dir}/y.md`\n- see [u](https://example.com/a.md)\n",
    );
    let report = check_tree(&root);
    assert!(report.broken.is_empty(), "{:?}", report.broken);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_stage_without_the_four_headings_is_reported() {
    let root = scratch("contract");
    write(&root.join("AGENTS.md"), "# AGENTS.md\n\n- `01_p/CONTEXT.md`\n");
    write(
        &root.join("01_p/CONTEXT.md"),
        "# 01_p: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n\n`stages/01_s/`\n",
    );
    write(&root.join("01_p/stages/01_s/CONTEXT.md"), "# 01_s: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Check\n");
    let problems = structure_problems(&root);
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].contains("## headings"), "{problems:?}");
    let _ = fs::remove_dir_all(&root);
}
