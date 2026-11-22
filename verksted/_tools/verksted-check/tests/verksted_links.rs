//! The workshop stays navigable. The logic lives in verksted/_tools/verksted-check/src/check.rs.

#[allow(dead_code)]
#[path = "../src/check.rs"]
mod check;

use std::path::{Path, PathBuf};

fn workshop_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn workshop_links_and_inputs_resolve() {
    let root = workshop_root();
    let report = check::check_tree(&root);
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
    let problems = check::walk_problems(&workshop_root(), false);
    assert!(problems.is_empty(), "contract problems:\n{}", problems.join("\n"));
}

#[test]
fn workshop_holds_no_python() {
    let mut found = Vec::new();
    check::python_files(&workshop_root(), &mut found);
    assert!(found.is_empty(), "Python is not allowed: {found:?}");
}

#[test]
#[ignore = "enabled by the wire-up package"]
fn workshop_graph_links_every_pipeline_to_its_stages() {
    let problems = check::graph_stage_problems(&workshop_root());
    assert!(problems.is_empty(), "pipeline contracts miss stage links:\n{}", problems.join("\n"));
}

#[test]
#[ignore = "enabled by the wire-up package: the shared notes carry no description yet"]
fn workshop_references_carry_a_description() {
    let problems = check::reference_description_problems(&workshop_root());
    assert!(problems.is_empty(), "references without a description:\n{}", problems.join("\n"));
}

#[test]
#[ignore = "enabled by the local-area package (#1352): verksted/.gitignore is not there yet"]
fn local_area_is_not_tracked() {
    let root = workshop_root();
    let ignore = std::fs::read_to_string(root.join(".gitignore")).unwrap_or_default();
    assert!(ignore.lines().any(|l| l.trim() == "/_local/"), "verksted/.gitignore must hold /_local/");
    let Ok(output) = std::process::Command::new("git")
        .args(["ls-files", "-z", "--", "_local"])
        .current_dir(&root)
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let tracked = String::from_utf8_lossy(&output.stdout).replace('\0', "\n");
    assert!(tracked.trim().is_empty(), "tracked files under verksted/_local/:\n{tracked}");
}

const MACHINE_PATH_FILES: [&str; 3] = ["_shared/rules.md", "_shared/commands.md", "setup/questionnaire.md"];

#[test]
#[ignore = "enabled when the packages that move machine values into the local area land"]
fn tracked_workshop_names_no_machine_path_outside_lib() {
    let root = workshop_root();
    let mut files = Vec::new();
    check::collect_markdown(&root, &mut files);
    let mut offenders = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&root).expect("under root").to_string_lossy().into_owned();
        if MACHINE_PATH_FILES.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        if text.contains("/mnt/") || text.contains("/home/") {
            offenders.push(rel);
        }
    }
    assert!(offenders.is_empty(), "machine paths in tracked files:\n{}", offenders.join("\n"));
}
