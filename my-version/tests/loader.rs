//! The loader refuses a row whose field count differs from the header instead
//! of silently dropping it (issue #72). `--skip-malformed` is the explicit,
//! counted opt-out.

use rcsr::data::{self, PrepSpec, SplitSpec};
use rcsr::rng::Mt;
use std::io::Write;
use std::process::{Command, Stdio};

const RAGGED: &str = "x,y\n1,2\n2,3,EXTRA\n3,6\n4,8\n";

#[test]
fn a_row_with_the_wrong_field_count_is_refused_not_dropped() {
    // 4 lines total: header + 3 data rows, the middle one short.
    let err = data::parse_table("x,y\n1,2\n3\n4,5\n").unwrap_err();
    assert!(err.contains("line 3"), "{err}"); // 1-based file line
    assert!(err.contains("expected 2"), "{err}");
    assert!(err.contains("found 1"), "{err}");

    // Extra field: same refusal, no truncation of the row.
    let err = data::parse_table("x,y\n1,2\n3,4,5\n").unwrap_err();
    assert!(err.contains("line 3") && err.contains("found 3"), "{err}");

    // The well-formed control is unchanged: all three rows survive in order.
    let t = data::parse_table("x,y\n1,2\n3,4\n5,6\n").unwrap();
    assert_eq!(t.rows.len(), 3);
    assert_eq!(t.rows[1], vec![3.0, 4.0]);
    assert_eq!(t.skipped_malformed, 0);
}

#[test]
fn blank_and_comment_lines_are_not_data_rows() {
    let t = data::parse_table("# exported by a sheet\n\nx,y\n1,2\n\n# a note\n3,4\n").unwrap();
    assert_eq!(t.rows.len(), 2);
    assert_eq!(t.rows[1], vec![3.0, 4.0]);
    assert_eq!(t.skipped_malformed, 0);
}

#[test]
fn skip_malformed_drops_exactly_one_row_and_reports_it() {
    let t = data::parse_table_with(RAGGED, true).unwrap();
    assert_eq!(t.rows.len(), 3, "one of four data rows is ragged");
    assert_eq!(t.skipped_malformed, 1);

    let vars = ["x".to_string()];
    let mut rng = Mt::new(1);
    let prepared = data::prepare_full_with(
        &t,
        &vars,
        "y",
        &SplitSpec::default(),
        &mut rng,
        &PrepSpec::default(),
    )
    .unwrap();
    assert_eq!(prepared.prep.rows_skipped_malformed, 1);
    let text = prepared.prep.text_lines(None, 0).join("\n");
    assert!(
        text.contains("--skip-malformed dropped 1 ragged row(s)"),
        "{text}"
    );
    assert!(
        prepared.prep.json().contains("\"rows_skipped_malformed\":1"),
        "{}",
        prepared.prep.json()
    );
}

#[test]
fn the_cli_refuses_a_ragged_file_with_exit_3() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", "-", "--target", "y", "--generations", "1", "--pop", "8"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(RAGGED.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(3));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("line 3"), "stderr: {err}");
    assert!(err.contains("expected 2"), "stderr: {err}");
    assert!(err.contains("found 3"), "stderr: {err}");
    // No result table was printed for the truncated dataset.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("y ="), "stdout: {stdout}");
}

#[test]
fn the_cli_skip_malformed_flag_reports_the_skipped_count() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            "-",
            "--target",
            "y",
            "--skip-malformed",
            "--generations",
            "1",
            "--pop",
            "8",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(RAGGED.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--skip-malformed dropped 1 ragged row(s)"),
        "stderr: {err}"
    );
}
