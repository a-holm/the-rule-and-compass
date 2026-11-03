//! Check logic of the verksted workshop. Pure std; no `crate::` paths, so the file compiles as a
//! crate module and through `#[path]` in `my-version/tests/verksted_links.rs`.

use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DIRS: [&str; 4] = ["node_modules", "_archive", "raw", "target"];
const CONTRACT: [&str; 4] = ["Inputs", "Process", "Outputs", "Check"];

#[derive(Default)]
pub struct Report {
    pub files: usize,
    pub checked: usize,
    pub broken: Vec<String>,
}

pub fn collect_markdown(dir: &Path, out: &mut Vec<PathBuf>) {
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

pub fn check_tree(root: &Path) -> Report {
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

pub fn structure_problems(root: &Path) -> Vec<String> {
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

pub fn python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read dir").map(|e| e.expect("dir entry")) {
        let path = entry.path();
        if path.is_dir() {
            if name_of(&path) != "target" {
                python_files(&path, out);
            }
        } else if path.extension().map(|e| e == "py").unwrap_or(false) {
            out.push(path);
        }
    }
}


fn read_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path).unwrap_or_default().lines().map(str::to_string).collect()
}

fn rel_of(root: &Path, file: &Path) -> String {
    file.strip_prefix(root).unwrap_or(file).display().to_string()
}

/// Top-level `DIR/_tools/*.sh`, sorted.
pub fn tool_scripts(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(root.join("_tools"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    out.retain(|p| p.is_file() && p.extension().map(|e| e == "sh").unwrap_or(false));
    out.sort();
    out
}

/// Directories `DIR/_tools/<name>/` that hold a `Cargo.toml`, sorted.
pub fn tool_crates(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(root.join("_tools"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    out.retain(|p| p.join("Cargo.toml").is_file());
    out.sort();
    out
}

/// Header rule of every top-level tool script.
pub fn header_problems(root: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    for script in tool_scripts(root) {
        let rel = rel_of(root, &script);
        let lines = read_lines(&script);
        if lines.first().map(String::as_str) != Some("#!/usr/bin/env bash") {
            problems.push(format!("{rel}: line 1 must be `#!/usr/bin/env bash`"));
        }
        let second_ok = lines
            .get(1)
            .map(|l| l.starts_with("# Usage:") || l.starts_with("# Sourced"))
            .unwrap_or(false);
        if !second_ok {
            problems.push(format!("{rel}: line 2 must start with `# Usage:` or `# Sourced`"));
        }
        if !lines.iter().take(15).any(|l| l.starts_with("# Writes:")) {
            problems.push(format!("{rel}: no `# Writes:` line in the first 15 lines"));
        }
    }
    problems
}

/// `DIR/CLAUDE.md` is exactly `@AGENTS.md` and a newline.
pub fn claude_problems(root: &Path) -> Vec<String> {
    match fs::read_to_string(root.join("CLAUDE.md")) {
        Ok(text) if text == "@AGENTS.md\n" => Vec::new(),
        Ok(_) => vec!["CLAUDE.md: must be exactly `@AGENTS.md` and a newline".to_string()],
        Err(_) => vec!["CLAUDE.md: missing".to_string()],
    }
}

fn is_ref_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_./-".contains(c)
}

/// Mentions of `_tools/<path>` in one line, each with its `verksted/` prefix flag.
fn tool_mentions(line: &str) -> Vec<(bool, String)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = line[from..].find("_tools/") {
        let at = from + rel;
        from = at + "_tools/".len();
        let before = &line[..at];
        let prev = before.chars().next_back();
        let prefixed = before.ends_with("verksted/");
        let word_before = match prev {
            Some(c) if c.is_ascii_alphanumeric() || c == '_' => true,
            _ => false,
        };
        if word_before {
            continue;
        }
        if prefixed && before[..before.len() - "verksted/".len()]
            .chars()
            .next_back()
            .map(|c| c.is_ascii_alphanumeric() || c == '_')
            .unwrap_or(false)
        {
            continue;
        }
        let rest = &line[at..];
        let len = rest.find(|c: char| !is_ref_char(c)).unwrap_or(rest.len());
        if rest[len..].starts_with(|c: char| "*<{[$".contains(c)) {
            continue;
        }
        let token = rest[..len].trim_end_matches(|c: char| ".,):".contains(c));
        let token = if rest[..len].ends_with('/') { &rest[..len] } else { token };
        out.push((prefixed, token.to_string()));
    }
    out
}

/// Every `_tools/<path>` mention in a workshop markdown file resolves.
pub fn refs_problems(root: &Path) -> Vec<String> {
    let repo = root
        .canonicalize()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let mut files = Vec::new();
    collect_markdown(root, &mut files);
    let mut problems = Vec::new();
    for file in files {
        if name_of(&file) == "_index.md" {
            continue;
        }
        let rel = rel_of(root, &file);
        for line in read_lines(&file) {
            for (prefixed, path) in tool_mentions(&line) {
                let target = if prefixed { repo.join("verksted").join(&path) } else { root.join(&path) };
                let wants_dir = path.ends_with('/');
                let ok = if wants_dir { target.is_dir() } else { target.exists() };
                if !ok {
                    let shown = if prefixed { format!("verksted/{path}") } else { path };
                    problems.push(format!("{rel}: reference {shown} does not exist"));
                }
            }
        }
    }
    problems
}

fn heading_text(text: &str) -> String {
    text.lines()
        .find(|l| l.starts_with('#'))
        .map(|l| l.trim_start_matches('#').trim().to_string())
        .unwrap_or_default()
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|")
}

fn after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix).map(str::trim)
}

fn crate_field(path: &Path, key: &str) -> Option<String> {
    for line in read_lines(&path.join("Cargo.toml")) {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(v) = rest.strip_prefix('=') {
                return Some(v.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

/// The generated `_index.md`, deterministic.
pub fn render_index(root: &Path) -> String {
    let mut out = String::from(
        "# Index\n\nGenerated by `verksted-check index`. Do not edit by hand.\n\n## Pipelines\n\n",
    );
    for pipeline in numbered_subdirs(root) {
        let pname = name_of(&pipeline);
        let lines = read_lines(&pipeline.join("CONTEXT.md"));
        let title = lines
            .first()
            .and_then(|l| after(l, &format!("# {pname}: ")))
            .unwrap_or("")
            .to_string();
        out.push_str(&format!("- [{pname}]({pname}/CONTEXT.md): {title}\n"));
        let stages_dir = pipeline.join("stages");
        if !stages_dir.is_dir() {
            continue;
        }
        for stage in numbered_subdirs(&stages_dir) {
            let sname = name_of(&stage);
            let job = read_lines(&stage.join("CONTEXT.md"))
                .get(2)
                .and_then(|l| after(l, "One job: ").map(str::to_string))
                .unwrap_or_default();
            out.push_str(&format!("  - [{sname}]({pname}/stages/{sname}/CONTEXT.md): {job}\n"));
        }
    }
    out.push_str("\n## Tools\n\n| Tool | Usage | Writes |\n|---|---|---|\n");
    let mut rows: Vec<(String, String)> = Vec::new();
    for script in tool_scripts(root) {
        let name = name_of(&script);
        let lines = read_lines(&script);
        let second = lines.get(1).cloned().unwrap_or_default();
        let usage = if second.starts_with("# Sourced") {
            "sourced".to_string()
        } else {
            format!("`{}`", cell(after(&second, "# Usage:").unwrap_or("")))
        };
        let writes = lines
            .iter()
            .take(15)
            .find_map(|l| after(l, "# Writes:"))
            .unwrap_or("")
            .trim_end_matches('.')
            .to_string();
        rows.push((name.clone(), format!("| [{name}](_tools/{name}) | {usage} | {} |\n", cell(&writes))));
    }
    for dir in tool_crates(root) {
        let name = name_of(&dir);
        let usage = crate_field(&dir, "description").unwrap_or_default();
        let writes = read_lines(&dir.join("Cargo.toml"))
            .iter()
            .find_map(|l| after(l, "# Writes:").map(str::to_string))
            .unwrap_or_else(|| "nothing".to_string());
        rows.push((
            name.clone(),
            format!("| [{name}](_tools/{name}/) | {} | {} |\n", cell(&usage), cell(&writes)),
        ));
    }
    rows.sort();
    for (_, row) in rows {
        out.push_str(&row);
    }
    out.push_str("\n## Templates\n\n");
    let mut templates: Vec<PathBuf> = fs::read_dir(root.join("_templates"))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    templates.retain(|p| p.is_file() && name_of(p).ends_with(".md"));
    templates.sort();
    for t in templates {
        let name = name_of(&t);
        let title = heading_text(&fs::read_to_string(&t).unwrap_or_default());
        out.push_str(&format!("- [{name}](_templates/{name}): {title}\n"));
    }
    out
}

/// `_index.md` matches the tree.
pub fn index_problems(root: &Path) -> Vec<String> {
    match fs::read_to_string(root.join("_index.md")) {
        Ok(text) if text == render_index(root) => Vec::new(),
        Ok(_) => vec!["_index.md: differs from the tree; run `verksted-check index`".to_string()],
        Err(_) => vec!["_index.md: missing; run `verksted-check index`".to_string()],
    }
}

/// Every non-sourced tool script and every tool crate is mentioned in a workshop markdown file
/// other than `_index.md`.
pub fn orphan_problems(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    collect_markdown(root, &mut files);
    let text: String = files
        .iter()
        .filter(|f| name_of(f) != "_index.md")
        .map(|f| fs::read_to_string(f).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    let mut problems = Vec::new();
    for script in tool_scripts(root) {
        let sourced = read_lines(&script).get(1).map(|l| l.starts_with("# Sourced")).unwrap_or(false);
        let name = name_of(&script);
        if !sourced && !text.contains(&name) {
            problems.push(format!("_tools/{name}: not mentioned in any workshop markdown file"));
        }
    }
    for dir in tool_crates(root) {
        let name = name_of(&dir);
        if !text.contains(&name) {
            problems.push(format!("_tools/{name}/: not mentioned in any workshop markdown file"));
        }
    }
    problems
}

/// The repository `AGENTS.md` links the workshop entry file.
pub fn root_route_problems(root: &Path) -> Vec<String> {
    let repo = root
        .canonicalize()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    let text = fs::read_to_string(repo.join("AGENTS.md")).unwrap_or_default();
    if text.contains("verksted/AGENTS.md") {
        Vec::new()
    } else {
        vec!["../AGENTS.md: does not link verksted/AGENTS.md".to_string()]
    }
}

/// All `walk` rules; `strict` adds the index, orphan and root route rules.
pub fn walk_problems(root: &Path, strict: bool) -> Vec<String> {
    let mut problems = structure_problems(root);
    let mut py = Vec::new();
    python_files(root, &mut py);
    for p in py {
        problems.push(format!("{}: Python is not allowed", rel_of(root, &p)));
    }
    problems.extend(claude_problems(root));
    problems.extend(header_problems(root));
    problems.extend(refs_problems(root));
    if strict {
        problems.extend(index_problems(root));
        problems.extend(orphan_problems(root));
        problems.extend(root_route_problems(root));
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("verksted_check_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().expect("parent")).expect("create dirs");
        fs::write(path, text).expect("write file");
    }

    const GOOD_SCRIPT: &str = "#!/usr/bin/env bash\n# Usage: a.sh\n# Writes: nothing.\n";

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
        write(
            &root.join("01_p/stages/01_s/CONTEXT.md"),
            "# 01_s: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Check\n",
        );
        let problems = structure_problems(&root);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("## headings"), "{problems:?}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn refs_red_a_missing_tool_mention_is_reported_in_text_and_fences() {
        let base = scratch("refs_red");
        let root = base.join("verksted");
        write(&root.join("_tools/a.sh"), GOOD_SCRIPT);
        write(
            &root.join("x.md"),
            "# x\n\nRun `_tools/nope.sh`, then:\n\n```sh\nverksted/_tools/gone.sh --x\n```\n\n| a | ../_tools/cell.sh. |\n",
        );
        let problems = refs_problems(&root);
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("_tools/nope.sh")));
        assert!(problems.iter().any(|p| p.contains("verksted/_tools/gone.sh")));
        assert!(problems.iter().any(|p| p.contains("_tools/cell.sh")));
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn refs_green_existing_directories_files_and_patterns_pass() {
        let base = scratch("refs_green");
        let root = base.join("verksted");
        write(&root.join("_tools/a.sh"), GOOD_SCRIPT);
        write(&root.join("_tools/sub/b.sh"), "x\n");
        write(&root.join("_index.md"), "[z](_tools/missing.sh)\n");
        write(
            &root.join("x.md"),
            "# x\n\n`_tools/a.sh`, `verksted/_tools/a.sh:` `_tools/` `_tools/sub/` `_tools/*.sh` `_tools/<x>` (`../_tools/sub/b.sh`).\n",
        );
        let problems = refs_problems(&root);
        assert!(problems.is_empty(), "{problems:?}");
        write(&root.join("y.md"), "`_tools/a.sh/` is not a directory\n");
        assert_eq!(refs_problems(&root).len(), 1);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn headers_red_each_rule_is_reported() {
        let root = scratch("headers_red");
        write(&root.join("_tools/no_shebang.sh"), "# Usage: x\n# Writes: nothing.\n");
        write(&root.join("_tools/no_usage.sh"), "#!/usr/bin/env bash\nset -u\n# Writes: nothing.\n");
        write(&root.join("_tools/no_writes.sh"), "#!/usr/bin/env bash\n# Usage: x\n");
        write(&root.join("_tools/tests/test-x.sh"), "echo no header\n");
        let problems = header_problems(&root);
        assert_eq!(problems.len(), 4, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("no_shebang.sh: line 1")));
        assert!(problems.iter().any(|p| p.contains("no_usage.sh: line 2")));
        assert!(problems.iter().any(|p| p.contains("no_writes.sh: no `# Writes:`")));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn headers_green_usage_sourced_and_late_writes_pass() {
        let root = scratch("headers_green");
        write(&root.join("_tools/a.sh"), GOOD_SCRIPT);
        write(&root.join("_tools/lib.sh"), "#!/usr/bin/env bash\n# Sourced by x; not run.\n# Writes: nothing.\n");
        let mut late = String::from("#!/usr/bin/env bash\n# Usage: l.sh\n");
        late.push_str(&"# more\n".repeat(11));
        late.push_str("# Writes: log\n");
        write(&root.join("_tools/late.sh"), &late);
        assert!(header_problems(&root).is_empty());
        let mut too_late = String::from("#!/usr/bin/env bash\n# Usage: l.sh\n");
        too_late.push_str(&"# more\n".repeat(14));
        too_late.push_str("# Writes: log\n");
        write(&root.join("_tools/late.sh"), &too_late);
        assert_eq!(header_problems(&root).len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn claude_red_missing_or_different_text_is_reported() {
        let root = scratch("claude_red");
        assert_eq!(claude_problems(&root), vec!["CLAUDE.md: missing".to_string()]);
        write(&root.join("CLAUDE.md"), "@AGENTS.md");
        assert_eq!(claude_problems(&root).len(), 1);
        write(&root.join("CLAUDE.md"), "@AGENTS.md\n\n");
        assert_eq!(claude_problems(&root).len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn claude_green_exact_text_passes() {
        let root = scratch("claude_green");
        write(&root.join("CLAUDE.md"), "@AGENTS.md\n");
        assert!(claude_problems(&root).is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    fn index_tree(name: &str) -> PathBuf {
        let root = scratch(name);
        write(&root.join("01_p/CONTEXT.md"), "# 01_p: the job | here\n\nOne job: x.\n");
        write(&root.join("01_p/stages/01_s/CONTEXT.md"), "# 01_s: job\n\nOne job: do s.\n");
        write(&root.join("_tools/a.sh"), "#!/usr/bin/env bash\n# Usage: a.sh <x> | y\n# Writes: out.md.\n");
        write(&root.join("_tools/lib.sh"), "#!/usr/bin/env bash\n# Sourced by a.sh; not run.\n# Writes: nothing.\n");
        write(
            &root.join("_tools/c/Cargo.toml"),
            "[package]\nname = \"c\"\n# Writes: `_index.md` with `index`\ndescription = \"A crate\"\n",
        );
        write(&root.join("_templates/t.md"), "---\nx: y\n---\n# Template title\n");
        root
    }

    #[test]
    fn index_renders_pipelines_tools_and_templates() {
        let root = index_tree("index_render");
        let expected = "# Index\n\nGenerated by `verksted-check index`. Do not edit by hand.\n\n## Pipelines\n\n\
- [01_p](01_p/CONTEXT.md): the job | here\n  - [01_s](01_p/stages/01_s/CONTEXT.md): do s.\n\n## Tools\n\n\
| Tool | Usage | Writes |\n|---|---|---|\n\
| [a.sh](_tools/a.sh) | `a.sh <x> \\| y` | out.md |\n\
| [c](_tools/c/) | A crate | `_index.md` with `index` |\n\
| [lib.sh](_tools/lib.sh) | sourced | nothing |\n\n## Templates\n\n- [t.md](_templates/t.md): Template title\n";
        assert_eq!(render_index(&root), expected);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn index_red_missing_or_stale_is_reported_and_green_when_fresh() {
        let root = index_tree("index_fresh");
        assert_eq!(index_problems(&root).len(), 1);
        write(&root.join("_index.md"), &render_index(&root));
        assert!(index_problems(&root).is_empty());
        let mut text = fs::read_to_string(root.join("_index.md")).expect("read");
        text.push_str("extra\n");
        write(&root.join("_index.md"), &text);
        assert_eq!(index_problems(&root).len(), 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn orphans_red_an_unmentioned_script_and_crate_are_reported() {
        let root = index_tree("orphans_red");
        write(&root.join("_index.md"), "a.sh c lib.sh\n");
        let problems = orphan_problems(&root);
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("a.sh")));
        assert!(problems.iter().any(|p| p.contains("_tools/c/")));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn orphans_green_mentions_outside_the_index_clear_them_and_sourced_files_are_exempt() {
        let root = index_tree("orphans_green");
        write(&root.join("note.md"), "Run `a.sh`. The crate c builds it.\n");
        assert!(orphan_problems(&root).is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn strict_walk_adds_the_root_route() {
        let base = scratch("root_route");
        let root = base.join("verksted");
        write(&root.join("AGENTS.md"), "# AGENTS.md\n");
        write(&root.join("CLAUDE.md"), "@AGENTS.md\n");
        write(&root.join("_index.md"), &render_index(&root));
        assert_eq!(root_route_problems(&root).len(), 1);
        write(&base.join("AGENTS.md"), "[w](verksted/AGENTS.md)\n");
        assert!(root_route_problems(&root).is_empty());
        assert!(walk_problems(&root, true).is_empty(), "{:?}", walk_problems(&root, true));
        let _ = fs::remove_dir_all(&base);
    }
}
