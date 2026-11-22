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
        "# 01_p: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n\n`stages/01_s/` and `_tools/a.sh`\n\n[s](stages/01_s/CONTEXT.md)\n",
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

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/icm")
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create dir");
    for entry in fs::read_dir(from).expect("read dir") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

/// The frozen fixture tree copied next to a scratch output folder, as `fixture-tree`.
fn fixture_copy(name: &str) -> (PathBuf, PathBuf) {
    let base = scratch(name);
    let tree = base.join("fixture-tree");
    copy_tree(&fixture_dir().join("tree"), &tree);
    (base, tree)
}

fn expected(name: &str) -> String {
    fs::read_to_string(fixture_dir().join(name)).expect("read expected file")
}

fn sorted_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text.lines().collect();
    lines.sort();
    lines
}

#[test]
fn icm_links_reproduce_the_frozen_output() {
    let (_base, tree) = fixture_copy("icm_links");
    let o = run(&["links", "--icm", s(&tree)]);
    assert_eq!(code(&o), 1);
    let want = expected("links.expected.txt");
    assert_eq!(sorted_lines(&stdout(&o)), sorted_lines(&want));
    assert_eq!(stdout(&o), want);
}

#[test]
fn graph_reproduces_the_frozen_jsonl() {
    let (base, tree) = fixture_copy("icm_graph");
    let out = base.join("graph.jsonl");
    let o = run(&["graph", "--out", s(&out), s(&tree)]);
    assert_eq!(code(&o), 0);
    let got = fs::read_to_string(&out).expect("read graph");
    let want = expected("graph.expected.jsonl");
    assert_eq!(sorted_lines(&got), sorted_lines(&want));
    assert_eq!(got, want);
}

#[test]
fn index_lists_the_references_of_the_fixture() {
    let (_base, tree) = fixture_copy("icm_index");
    assert_eq!(code(&run(&["index", s(&tree)])), 0);
    let got = fs::read_to_string(tree.join("_index.md")).expect("read index");
    assert_eq!(got, expected("index.expected.md"));
}

#[test]
fn graph_dot_prints_one_node_per_checked_file() {
    let root = workshop("dot");
    let o = run(&["graph", "--dot", s(&root)]);
    assert_eq!(code(&o), 0);
    let out = stdout(&o);
    assert!(out.starts_with("digraph "), "{out}");
    assert_eq!(out.lines().filter(|l| l.ends_with(';') && !l.contains("->")).count(), 4, "{out}");
    assert!(out.contains("\"01_p/CONTEXT.md\" -> \"01_p/stages/01_s/CONTEXT.md\";"), "{out}");
}

#[test]
fn the_real_walk_ignores_the_broken_links_of_the_fixture() {
    let verksted = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(code(&run(&["links", s(&verksted)])), 0);
    let o = run(&["links", "--icm", s(&fixture_dir().join("tree"))]);
    assert_eq!(code(&o), 1, "the fixture itself still holds broken links");
}

#[test]
fn a_local_folder_is_skipped_by_default_and_checked_by_icm() {
    let root = workshop("local");
    write(&root.join("_local/x.md"), "[b](missing.md)\n");
    assert_eq!(code(&run(&["links", s(&root)])), 0);
    let o = run(&["links", "--icm", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("_local/x.md: missing.md"), "{}", stdout(&o));
    assert_eq!(code(&run(&["graph", "--dot", s(&root)])), 0);
    assert!(!stdout(&run(&["graph", "--dot", s(&root)])).contains("_local"));
}

#[test]
fn a_consumes_path_to_a_missing_note_is_reported() {
    let root = workshop("consumes");
    write(&root.join("x.md"), "---\nconsumes: [a.md, gone.md]\nproduces: out.md\n---\n# x\n");
    write(&root.join("a.md"), "# a\n");
    let o = run(&["links", s(&root)]);
    assert_eq!(code(&o), 1);
    assert_eq!(stdout(&o).lines().collect::<Vec<_>>(), ["x.md: link gone.md", "x.md: link out.md"]);
}

#[test]
fn stages_with_one_number_need_the_alternatives_sentence() {
    let root = workshop("alternatives");
    write(
        &root.join("01_p/stages/01_t/CONTEXT.md"),
        "# 01_t: job\n\nOne job: s.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n",
    );
    let pipeline = fs::read_to_string(root.join("01_p/CONTEXT.md")).expect("read");
    write(&root.join("01_p/CONTEXT.md"), &format!("{pipeline}\n`stages/01_t/` [t](stages/01_t/CONTEXT.md)\n"));
    let o = run(&["walk", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("01_s, 01_t share number 01"), "{}", stdout(&o));
    write(
        &root.join("01_p/CONTEXT.md"),
        &format!("{pipeline}\n`stages/01_t/` [t](stages/01_t/CONTEXT.md)\n\nStages with the same number are alternatives.\n"),
    );
    assert_eq!(code(&run(&["walk", s(&root)])), 0);
}

#[test]
fn strict_walk_needs_stage_links_and_reference_descriptions() {
    let root = workshop("strict_rules");
    assert_eq!(code(&run(&["index", s(&root)])), 0);
    assert_eq!(code(&run(&["walk", "--strict", s(&root)])), 0);
    write(&root.join("_shared/r.md"), "# r\n");
    write(&root.join("01_p/references/q.md"), "---\ndescription: q\n---\n# q\n");
    write(&root.join("01_p/CONTEXT.md"), "# 01_p: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n## Process\n## Outputs\n## Check\n\n`stages/01_s/` and `_tools/a.sh`\n");
    assert_eq!(code(&run(&["index", s(&root)])), 0);
    let o = run(&["walk", "--strict", s(&root)]);
    assert_eq!(code(&o), 1);
    let out = stdout(&o);
    assert!(out.contains("_shared/r.md: reference has no description"), "{out}");
    assert!(!out.contains("q.md"), "{out}");
    assert!(out.contains("01_p/CONTEXT.md: no link to stages/01_s/CONTEXT.md"), "{out}");
}

#[test]
fn graph_usage_errors_exit_2() {
    assert_eq!(code(&run(&["graph", "--out"])), 2);
    assert_eq!(code(&run(&["graph", "--icm"])), 2);
    assert_eq!(code(&run(&["links", "--dot"])), 2);
}

#[test]
fn links_external_checks_outside_references_strictly() {
    let root = workshop("external");
    let base = root.parent().expect("base").to_path_buf();
    write(&base.join("pkg/Cargo.toml"), "[package]\nname = \"rule-and-compass\"\n");
    write(
        &root.join("01_p/CONTEXT.md"),
        "# 01_p: job\n\nOne job: x.\n\nDo NOT load: y.\n\n## Inputs\n\n**Reference:**\n- `../../<crate>/src/gone.rs`\n\n## Process\n## Outputs\n## Check\n\n`stages/01_s/` and `_tools/a.sh`\n\n[s](stages/01_s/CONTEXT.md)\n",
    );
    assert_eq!(code(&run(&["links", s(&root)])), 0);
    let o = run(&["links", "--external", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("<crate>/src/gone.rs"), "{}", stdout(&o));
    write(&base.join("pkg/src/gone.rs"), "");
    assert_eq!(code(&run(&["links", "--external", s(&root)])), 0);
    fs::remove_file(base.join("pkg/Cargo.toml")).expect("remove manifest");
    let o = run(&["links", "--external", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(String::from_utf8_lossy(&o.stderr).contains("no engine crate directory"));
    assert_eq!(code(&run(&["links", "--icm", "--external", s(&root)])), 2);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn walk_flags_the_crate_directory_name() {
    let root = workshop("crate_name");
    let base = root.parent().expect("base").to_path_buf();
    write(&base.join("pkg/Cargo.toml"), "[package]\nname = \"rule-and-compass\"\n");
    assert_eq!(code(&run(&["walk", s(&root)])), 0);
    write(&root.join("_templates/t.md"), "# T\n\nSee `pkg/src/x.rs`.\n");
    let o = run(&["walk", s(&root)]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("_templates/t.md:3"), "{}", stdout(&o));
    let _ = fs::remove_dir_all(&base);
}
