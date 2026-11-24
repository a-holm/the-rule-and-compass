//! Guards of the standard rules: worktrees are created inside `$WT` (rule 59) and the general briefs
//! and contracts name machine keys, never machine values (plan section 12, decision 4).

use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn workshop_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> =
        fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Tracked sources of the workshop: markdown and shell files, outside generated or local folders, the script
/// tests (they build fixture repositories in temp directories) and the guard crate itself.
fn tracked_sources(root: &Path) -> Vec<PathBuf> {
    let mut all = Vec::new();
    walk(root, &mut all);
    all.into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "md" || e == "sh"))
        .filter(|p| {
            let rel = p.strip_prefix(root).unwrap_or(p).to_string_lossy().into_owned();
            !(rel.starts_with("_local/")
                || rel.starts_with("_tools/tests/")
                || rel.starts_with("_tools/verksted-check/")
                || rel.split('/').any(|c| c == "output" || c == "target"))
        })
        .collect()
}

const WORKTREE_PREFIXES: [&str; 6] = ["worktrees/", "$ROOT/worktrees/", "${ROOT}/worktrees/", "$WT_DIR/", "$WT/", "${WT}/"];
const TEMP_WORDS: [&str; 3] = ["/tmp", "$TMPDIR", "mktemp"];

/// Why a command span that holds `worktree add` breaks rule 59, or None. The path argument is the first token after the
/// options; a token without a slash, a dollar sign or a quote is prose ("worktree add and remove"), not a path.
fn span_problem(span: &str) -> Option<String> {
    let at = span.find("worktree add")?;
    if let Some(word) = TEMP_WORDS.iter().find(|w| span.contains(**w)) {
        return Some(format!("`{word}` on a `worktree add` line"));
    }
    let mut words = span[at + "worktree add".len()..].split_whitespace();
    let mut path = None;
    while let Some(word) = words.next() {
        if matches!(word, "-b" | "-B") {
            words.next();
            continue;
        }
        if word.starts_with('-') || word.starts_with("[-") {
            continue;
        }
        path = Some(word);
        break;
    }
    let path = path?;
    let bare = path.trim_matches(|c| matches!(c, '"' | '\'' | ',' | ';' | ')' | '`'));
    if !(path.contains('/') || path.contains('$') || path.starts_with(['"', '\''])) {
        return None;
    }
    if WORKTREE_PREFIXES.iter().any(|p| bare.starts_with(p)) {
        None
    } else {
        Some(format!("path `{bare}` does not start with worktrees/, $ROOT/worktrees/, $WT_DIR/ or $WT/"))
    }
}

/// Command spans of one line: the whole line of a shell file or a fenced block, the code spans of markdown prose.
fn spans(line: &str, whole_line: bool) -> Vec<&str> {
    if whole_line {
        vec![line]
    } else {
        line.split('`').skip(1).step_by(2).collect()
    }
}

/// `file:line: reason: text` for every `worktree add` in the tracked sources under `root` that breaks rule 59.
fn worktree_path_problems(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for file in tracked_sources(root) {
        let rel = file.strip_prefix(root).unwrap_or(&file).display().to_string();
        let shell = file.extension().is_some_and(|e| e == "sh");
        let mut fenced = false;
        for (n, line) in fs::read_to_string(&file).unwrap_or_default().lines().enumerate() {
            if !shell && line.trim_start().starts_with("```") {
                fenced = !fenced;
                continue;
            }
            for span in spans(line, shell || fenced) {
                if let Some(why) = span_problem(span) {
                    out.push(format!("{rel}:{}: {why}: {}", n + 1, line.trim()));
                }
            }
        }
    }
    out
}

#[test]
fn worktree_paths() {
    let problems = worktree_path_problems(&workshop_root());
    assert!(problems.is_empty(), "worktrees are created inside $WT (rule 59):\n{}", problems.join("\n"));
}

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().expect("parent")).expect("create dir");
    fs::write(path, text).expect("write fixture");
}

#[test]
fn worktree_paths_fail_on_a_contract_that_adds_a_worktree_outside_wt() {
    let dir = Scratch::new("wt-bad");
    write(dir.path(), "01_a/CONTEXT.md", "# a\n\n```sh\ngit worktree add /tmp/x main\n```\n");
    let problems = worktree_path_problems(dir.path());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].starts_with("01_a/CONTEXT.md:4: "), "{}", problems[0]);
    assert!(problems[0].contains("/tmp"), "{}", problems[0]);
}

#[test]
fn worktree_paths_flag_every_bad_spelling() {
    for bad in [
        "git worktree add /tmp/x main",
        "git -C \"$R\" worktree add -q \"$(mktemp -d)\" HEAD",
        "git -C \"$R\" worktree add -q \"$TMPDIR/w\" ref",
        "git worktree add ../x",
        "git worktree add -b b \"$G/scratch-w\" ref",
        "git worktree add -b b \"$w\" ref",
        "git worktree add /mnt/gates/w HEAD",
        "git worktree add worktrees-x/y HEAD",
    ] {
        assert!(span_problem(bad).is_some(), "{bad}");
    }
}

#[test]
fn worktree_paths_pass_every_good_spelling() {
    for good in [
        "git -C \"$ROOT\" worktree add -q --no-track -b \"$branch\" \"$WT/$name\" \"origin/$base\"",
        "git worktree add worktrees/issue-1-x main",
        "git -C \"$ROOT\" worktree add --detach \"$ROOT/worktrees/r\" HEAD",
        "git worktree add \"$WT_DIR/x\" HEAD",
        "git -C <an existing worktree> worktree add [--detach] \"$WT/<purpose>-<id>\" <ref>",
        "fetch, worktree add and remove, branch delete",
        "worktree add",
        "no worktree here",
    ] {
        assert_eq!(span_problem(good), None, "{good}");
    }
}

#[test]
fn worktree_paths_read_markdown_code_spans_and_fences_only() {
    let dir = Scratch::new("wt-md");
    write(
        dir.path(),
        "a.md",
        "Prose may say worktree add /tmp/x outside a code span.\nNot `/tmp` here, and `git worktree add \"$WT/ok\" HEAD` is fine.\n",
    );
    assert_eq!(worktree_path_problems(dir.path()), Vec::<String>::new());
    write(dir.path(), "b.md", "Run `git worktree add /tmp/x HEAD` now.\n");
    let problems = worktree_path_problems(dir.path());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].starts_with("b.md:1: "), "{}", problems[0]);
}

#[test]
fn worktree_paths_scan_shell_files_line_by_line() {
    let dir = Scratch::new("wt-sh");
    write(dir.path(), "_tools/a.sh", "#!/usr/bin/env bash\ngit -C \"$ROOT\" worktree add -q --detach \"$WT/$name\" HEAD\n");
    assert_eq!(worktree_path_problems(dir.path()), Vec::<String>::new());
    write(dir.path(), "_tools/b.sh", "#!/usr/bin/env bash\ngit -C \"$ROOT\" worktree add -q --detach \"$(mktemp -d)\" HEAD\n");
    let problems = worktree_path_problems(dir.path());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(problems[0].starts_with("_tools/b.sh:2: "), "{}", problems[0]);
    write(dir.path(), "_tools/tests/test-x.sh", "git worktree add /tmp/fixture HEAD\n");
    assert_eq!(worktree_path_problems(dir.path()).len(), 1, "script tests are not scanned");
}

/// Tracked files that may hold machine values: the machine scripts, until they read the profile keys (#1413).
/// `tracked_sources` already leaves out the script tests and this crate.
const MACHINE_VALUE_ALLOWED: [&str; 1] = ["_tools/machine/"];

/// Lines whose number list or size is a tool parameter, not a machine value: (file, text the line contains).
const MACHINE_VALUE_LINE_ALLOWED: [(&str, &str); 6] = [
    ("_tools/ab-pair.sh", "seeds 1,2,3"),
    ("_tools/ab-pair.sh", "seeds=1,2,3"),
    ("_index.md", "seeds 1,2,3"),
    ("_shared/commands.md", "seeds 1,2,3"),
    ("_tools/run-record.sh", "2 GiB"),
    ("_tools/resources.sh", "KiB 1024"),
];

/// Campaign literals: a branch, a work-folder name or a fixed job count belongs in the local area.
const CAMPAIGN_LITERALS: [&str; 5] =
    ["integration/gpu-hardening", "integration-gpu-hardening", "arbeidsmappe", "crypto-rcsr", "CARGO_BUILD_JOBS=2"];

/// Every tracked `.md` and `.sh` file of the workshop that the allow list does not exempt, as paths relative to `root`.
fn machine_free_files(root: &Path) -> Vec<String> {
    tracked_sources(root)
        .iter()
        .map(|p| p.strip_prefix(root).unwrap_or(p).to_string_lossy().into_owned())
        .filter(|rel| !MACHINE_VALUE_ALLOWED.iter().any(|a| rel.starts_with(a)))
        .collect()
}

const VOLUME_PREFIXES: [&str; 7] = ["/mnt/", "/media/", "/home/", "/usr/local/", "/opt/", "/srv/", "/Volumes/"];
const SIZE_UNITS: [&str; 8] = ["KiB", "MiB", "GiB", "TiB", "KB", "MB", "GB", "TB"];

/// A number followed by a size unit, with at most one space between: `500 GB`, `8GiB`.
fn has_size(line: &str) -> bool {
    let bytes = line.as_bytes();
    SIZE_UNITS.iter().any(|unit| {
        line.match_indices(unit).any(|(i, _)| {
            let before = &line[..i];
            let before = before.strip_suffix(' ').unwrap_or(before);
            let digit = before.as_bytes().last().is_some_and(u8::is_ascii_digit);
            let after = bytes.get(i + unit.len()).copied();
            digit && !after.is_some_and(|c| c.is_ascii_alphanumeric())
        })
    })
}

/// Three or more comma-separated numbers without a space: a core list such as `2,3,6,7`.
fn has_core_list(line: &str) -> bool {
    let b = line.as_bytes();
    (0..b.len()).any(|i| {
        let digits = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
        let d1 = digits(i);
        if d1 == 0 || (i > 0 && b[i - 1].is_ascii_digit()) {
            return false;
        }
        let j = i + d1;
        if b.get(j) != Some(&b',') {
            return false;
        }
        let d2 = digits(j + 1);
        d2 > 0 && b.get(j + 1 + d2) == Some(&b',') && digits(j + 2 + d2) > 0
    })
}

/// An architecture token `sm_<digit>` or a driver version `driver <digit>`.
fn has_arch_or_driver(line: &str) -> bool {
    let lower = line.to_lowercase();
    let after_digit = |needle: &str| {
        lower.match_indices(needle).any(|(i, _)| lower.as_bytes().get(i + needle.len()).is_some_and(u8::is_ascii_digit))
    };
    after_digit("sm_") || after_digit("driver ")
}

fn machine_value(line: &str) -> Option<&'static str> {
    if VOLUME_PREFIXES.iter().any(|p| line.contains(p)) {
        Some("absolute volume path")
    } else if has_size(line) {
        Some("size with a unit")
    } else if has_core_list(line) {
        Some("core list")
    } else if has_arch_or_driver(line) {
        Some("architecture token or driver version")
    } else if CAMPAIGN_LITERALS.iter().any(|l| line.contains(l)) {
        Some("campaign value")
    } else {
        None
    }
}

/// `file:line: kind: text` for every machine value in the listed files under `root`; a listed file must exist.
fn machine_value_problems(root: &Path, files: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for rel in files {
        let Ok(text) = fs::read_to_string(root.join(rel)) else {
            out.push(format!("{rel}: file missing"));
            continue;
        };
        for (n, line) in text.lines().enumerate() {
            if MACHINE_VALUE_LINE_ALLOWED.iter().any(|(f, s)| f == rel && line.contains(s)) {
                continue;
            }
            if let Some(kind) = machine_value(line) {
                out.push(format!("{rel}:{}: {kind}: {}", n + 1, line.trim()));
            }
        }
    }
    out
}

#[test]
fn no_machine_values() {
    let root = workshop_root();
    let files = machine_free_files(&root);
    let files: Vec<&str> = files.iter().map(String::as_str).collect();
    let problems = machine_value_problems(&root, &files);
    assert!(
        problems.is_empty(),
        "machine values belong in machine.md or rules-local.md, tracked text names the keys:\n{}",
        problems.join("\n")
    );
}

#[test]
fn no_machine_values_fail_on_a_fixture_with_each_kind_of_value() {
    for (bad, kind) in [
        ("builds target sm_86 only", "architecture token or driver version"),
        ("run `/mnt/volume/build-slot.sh`", "absolute volume path"),
        ("at least 500 GB free", "size with a unit"),
        ("needs 8GiB", "size with a unit"),
        ("pin to 2,3,6,7", "core list"),
        ("works on driver 580", "architecture token or driver version"),
        ("the sanitizer lives in /usr/local/tools", "absolute volume path"),
        ("branches from integration/gpu-hardening", "campaign value"),
        ("notes in arbeidsmappe/x", "campaign value"),
        ("env CARGO_BUILD_JOBS=2 cargo", "campaign value"),
    ] {
        let dir = Scratch::new("mv-bad");
        write(dir.path(), "_shared/rules.md", &format!("# Rules\n\n{bad}\n"));
        let problems = machine_value_problems(dir.path(), &["_shared/rules.md"]);
        assert_eq!(problems.len(), 1, "{bad}: {problems:?}");
        assert!(problems[0].starts_with("_shared/rules.md:3: "), "{}", problems[0]);
        assert!(problems[0].contains(kind), "{bad}: {}", problems[0]);
    }
}

#[test]
fn no_machine_values_pass_keys_and_plain_numbers() {
    let dir = Scratch::new("mv-good");
    write(
        dir.path(),
        "_shared/rules.md",
        "Pin to `PIN_CORES_BUILD`; free space `DISK_MIN_FREE` on `BUILD_VOLUME`.\nRules 12, 23, 34 and 59 to 66. Rule 44 of 72.\nAt most 10 agents, 3 tickets, a hold under 10 minutes, nice 19.\nThe NVIDIA driver can freeze the machine; sm is not a token. Use `$SANITIZER`.\n",
    );
    assert_eq!(machine_value_problems(dir.path(), &["_shared/rules.md"]), Vec::<String>::new());
}

#[test]
fn no_machine_values_report_a_missing_file() {
    let dir = Scratch::new("mv-missing");
    assert_eq!(machine_value_problems(dir.path(), &["_shared/brief.md"]), ["_shared/brief.md: file missing"]);
}

/// `file:line: text` for every absolute volume path in the tracked markdown and shell sources under `root`.
fn volume_path_problems(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for file in tracked_sources(root) {
        let rel = file.strip_prefix(root).unwrap_or(&file).to_string_lossy().into_owned();
        let text = fs::read_to_string(&file).unwrap_or_default();
        for (n, line) in text.lines().enumerate() {
            if VOLUME_PREFIXES.iter().any(|p| line.contains(p)) && !line.trim_start().starts_with("#!") {
                out.push(format!("{rel}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    out
}

#[test]
fn tracked_workshop_sources_name_no_absolute_volume_path() {
    let problems = volume_path_problems(&workshop_root());
    assert!(
        problems.is_empty(),
        "absolute volume paths belong in machine.md; tracked text uses the keys or variables of lib.sh:\n{}",
        problems.join("\n")
    );
}

#[test]
fn volume_path_guard_flags_a_script_and_a_note_and_spares_a_shebang() {
    let dir = Scratch::new("vp");
    write(dir.path(), "_tools/a.sh", "#!/usr/bin/env bash\nDIR=/mnt/lager/tmp/x\n");
    write(dir.path(), "_shared/n.md", "Run /home/someone/tool\n");
    write(dir.path(), "_tools/tests/t.sh", "X=/mnt/fixture\n");
    write(dir.path(), "_shared/ok.md", "Use `$GATES`.\n");
    let problems = volume_path_problems(dir.path());
    assert_eq!(problems, ["_shared/n.md:1: Run /home/someone/tool", "_tools/a.sh:2: DIR=/mnt/lager/tmp/x"]);
}

#[test]
fn machine_value_guard_flags_a_lib_default() {
    let dir = Scratch::new("mv-lib");
    write(dir.path(), "_tools/lib.sh", "GATES=${RCSR_GATES:-/mnt/vol/gates}\nCUDA_ARCH=sm_80\nSANITIZER=${RCSR_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}\n");
    let problems = machine_value_problems(dir.path(), &["_tools/lib.sh"]);
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems[0].contains("absolute volume path"));
    assert!(problems[1].contains("architecture token"));
    assert!(problems[2].contains("absolute volume path"));
}

#[test]
fn machine_value_scan_covers_every_tracked_markdown_and_shell_file() {
    let root = workshop_root();
    let tracked: Vec<String> = tracked_sources(&root)
        .iter()
        .map(|p| p.strip_prefix(&root).unwrap_or(p).to_string_lossy().into_owned())
        .collect();
    let scanned = machine_free_files(&root);
    assert!(scanned.len() > 100, "the scan lists {} files", scanned.len());
    for rel in &tracked {
        let allowed = MACHINE_VALUE_ALLOWED.iter().any(|a| rel.starts_with(a));
        assert!(allowed || scanned.contains(rel), "{rel} is neither scanned nor allow-listed");
    }
    for allowed in MACHINE_VALUE_ALLOWED {
        assert!(tracked.iter().any(|rel| rel.starts_with(allowed)), "allow-list entry {allowed} matches no file");
    }
    for (file, text) in MACHINE_VALUE_LINE_ALLOWED {
        let body = fs::read_to_string(root.join(file)).unwrap_or_default();
        assert!(body.lines().any(|l| l.contains(text)), "line allowance {file}: {text} matches no line");
    }
    for must in ["_shared/rules.md", "_tools/lib.sh", "_templates/runbook.sh", "07_longrun/stages/02_launch/CONTEXT.md"] {
        assert!(scanned.iter().any(|s| s == must), "{must} is not scanned");
    }
}

/// Keys of the `## Profile keys` section of the machine.md template.
fn template_keys(root: &Path) -> Vec<String> {
    let text = fs::read_to_string(root.join("_templates/local/machine.md")).unwrap_or_default();
    let mut in_keys = false;
    let mut keys = Vec::new();
    for line in text.lines() {
        if line.starts_with("## ") {
            in_keys = line == "## Profile keys";
        } else if in_keys {
            if let Some((key, _)) = line.strip_prefix("- ").and_then(|l| l.split_once(": ")) {
                keys.push(key.to_string());
            }
        }
    }
    keys
}

/// True when `key` stands as a whole word in `text`.
fn names_word(text: &str, key: &str) -> bool {
    text.match_indices(key).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + key.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        !word(before) && !word(after)
    })
}

/// Template keys that no script of `_tools` reads and no contract (`CONTEXT.md`, the shared rules and commands, the
/// templates outside the local starters) names.
fn unused_template_keys(root: &Path) -> Vec<String> {
    let mut readers = String::new();
    for file in tracked_sources(root) {
        let rel = file.strip_prefix(root).unwrap_or(&file).to_string_lossy().into_owned();
        let is_script = rel.starts_with("_tools/") && rel.ends_with(".sh");
        let is_contract = rel.ends_with("CONTEXT.md")
            || rel == "_shared/rules.md"
            || rel == "_shared/commands.md"
            || rel == "_shared/models.md"
            || (rel.starts_with("_templates/") && !rel.starts_with("_templates/local/"));
        if is_script || is_contract {
            readers.push_str(&fs::read_to_string(&file).unwrap_or_default());
            readers.push('\n');
        }
    }
    template_keys(root).into_iter().filter(|k| !names_word(&readers, k)).collect()
}

#[test]
fn every_machine_template_key_is_read_by_a_script_or_named_by_a_contract() {
    let root = workshop_root();
    assert!(template_keys(&root).len() >= 17, "the template lists too few keys");
    let unused = unused_template_keys(&root);
    assert!(unused.is_empty(), "template keys that nothing reads or names: {unused:?}");
}

#[test]
fn an_unused_template_key_is_reported() {
    let dir = Scratch::new("keys");
    write(
        dir.path(),
        "_templates/local/machine.md",
        "# Machine profile\n\n## Profile keys\n\n- READ_KEY: <set me>\n- NAMED_KEY: <set me>\n- LOST_KEY: <set me>\n\n## Hardware\n\n- CPU model: <model>\n",
    );
    write(dir.path(), "_tools/a.sh", "x=$(machine_key READ_KEY)\n");
    write(dir.path(), "01/CONTEXT.md", "The key `NAMED_KEY` is used. LOST_KEY_2 is another word.\n");
    write(dir.path(), "_shared/local.md", "| `LOST_KEY` | documented only |\n");
    assert_eq!(unused_template_keys(dir.path()), ["LOST_KEY"]);
}
