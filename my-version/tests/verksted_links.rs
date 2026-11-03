//! The workshop stays navigable. The logic lives in verksted/_tools/verksted-check/src/check.rs.

#[allow(dead_code)]
#[path = "../../verksted/_tools/verksted-check/src/check.rs"]
mod check;

use std::path::{Path, PathBuf};

fn workshop_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../verksted")
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
