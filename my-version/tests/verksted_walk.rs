//! The workshop passes the strict walk: fresh index, no orphan tool, root route.
//! The logic lives in verksted/_tools/verksted-check/src/check.rs.

#[allow(dead_code)]
#[path = "../../verksted/_tools/verksted-check/src/check.rs"]
mod check;

use std::path::{Path, PathBuf};

fn workshop_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../verksted")
}

#[test]
fn workshop_strict_walk_is_clean() {
    let problems = check::walk_problems(&workshop_root(), true);
    assert!(problems.is_empty(), "strict walk problems:\n{}", problems.join("\n"));
}
