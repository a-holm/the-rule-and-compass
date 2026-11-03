//! Exit codes and output of each subcommand on scratch trees.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("verksted_cli_{}_{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("create dirs");
    fs::write(path, text).expect("write file");
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_verksted-check")).args(args).output().expect("run verksted-check")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exit code")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn workshop(name: &str) -> PathBuf {
    let base = scratch(name);
    let root = base.join("verksted");
    write(&base.join("AGENTS.md"), "[w](verksted/AGENTS.md)\n");
    write(&root.join("AGENTS.md"), "# AGENTS.md\n\n- `01_p/CONTEXT.md`\n");
    write(&root.join("CLAUDE.md"), "@AGENTS.md\n");
    write(
        &root.join("01_p/CONTEXT.md"),
        "# 01_p: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n\n`stages/01_s/` and `_tools/a.sh`\n",
    );
    write(
        &root.join("01_p/stages/01_s/CONTEXT.md"),
        "# 01_s: job\n\nOne job: s.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n",
    );
    write(&root.join("_tools/a.sh"), "#!/usr/bin/env bash\n# Usage: a.sh\n# Writes: nothing.\n");
    write(&root.join("_templates/t.md"), "# T\n");
    root
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf8 path")
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(code(&run(&[])), 2);
    assert_eq!(code(&run(&["nope"])), 2);
    assert_eq!(code(&run(&["links", "--strict"])), 2);
    assert_eq!(code(&run(&["links", "/no/such/dir"])), 2);
}

#[test]
fn links_exit_1_with_the_broken_link_on_stdout() {
    let root = workshop("links");
    assert_eq!(code(&run(&["links", s(&root)])), 0);
    write(&root.join("x.md"), "[b](missing.md)\n");
    let o = run(&["links", s(&root)]);
    assert_eq!(code(&o), 1);
    assert_eq!(stdout(&o).trim(), "x.md: link missing.md");
}

#[test]
fn refs_exit_1_naming_the_missing_script() {
    let root = workshop("refs");
    assert_eq!(code(&run(&["refs", s(&root)])), 0);
    write(&root.join("x.md"), "see `verksted/_tools/nope.sh`\n");
    let o = run(&["refs", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("verksted/_tools/nope.sh"), "{}", stdout(&o));
}

#[test]
fn walk_non_strict_passes_and_strict_needs_the_index() {
    let root = workshop("walk");
    assert_eq!(code(&run(&["walk", s(&root)])), 0);
    let o = run(&["walk", "--strict", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("_index.md"), "{}", stdout(&o));
    assert_eq!(code(&run(&["index", s(&root)])), 0);
    write(&root.join("note.md"), "`a.sh` runs.\n");
    assert_eq!(code(&run(&["index", s(&root)])), 0);
    assert_eq!(code(&run(&["walk", "--strict", s(&root)])), 0);
}

#[test]
fn walk_reports_python_and_a_wrong_claude_file() {
    let root = workshop("walk_red");
    write(&root.join("x.py"), "pass\n");
    write(&root.join("CLAUDE.md"), "other\n");
    let o = run(&["walk", s(&root)]);
    assert_eq!(code(&o), 1);
    let out = stdout(&o);
    assert!(out.contains("x.py"), "{out}");
    assert!(out.contains("CLAUDE.md"), "{out}");
}

#[test]
fn index_check_exits_1_after_a_hand_edit() {
    let root = workshop("index");
    assert_eq!(code(&run(&["index", "--check", s(&root)])), 1);
    assert_eq!(code(&run(&["index", s(&root)])), 0);
    assert_eq!(code(&run(&["index", "--check", s(&root)])), 0);
    let mut text = fs::read_to_string(root.join("_index.md")).expect("read");
    text.push_str("hand edit\n");
    fs::write(root.join("_index.md"), text).expect("write");
    assert_eq!(code(&run(&["index", "--check", s(&root)])), 1);
}
