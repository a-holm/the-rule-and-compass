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
    assert_eq!(t.len(), 3);
    assert_eq!((0..t.n_cols()).map(|j| t.value(1, j)).collect::<Vec<_>>(), vec![3.0, 4.0]);
    assert_eq!(t.skipped_malformed, 0);
}

#[test]
fn blank_and_comment_lines_are_not_data_rows() {
    let t = data::parse_table("# exported by a sheet\n\nx,y\n1,2\n\n# a note\n3,4\n").unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!((0..t.n_cols()).map(|j| t.value(1, j)).collect::<Vec<_>>(), vec![3.0, 4.0]);
    assert_eq!(t.skipped_malformed, 0);
}

#[test]
fn skip_malformed_drops_exactly_one_row_and_reports_it() {
    let t = data::parse_table_with(RAGGED, true).unwrap();
    assert_eq!(t.len(), 3, "one of four data rows is ragged");
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

/// Issue #70: the parser fills one contiguous column-major image, and text
/// columns are dictionary codes rather than a `String` per cell.
#[test]
fn column_major_layout_matches_the_parser() {
    let t = data::parse_table("x,label,y\n1,A,10\n2,B,20\n3,A,30\n").unwrap();
    assert_eq!(t.numeric, vec![true, false, true]);
    assert_eq!(t.col(0), &[1.0, 2.0, 3.0]);
    assert_eq!(t.col(2), &[10.0, 20.0, 30.0]);
    assert!(t.col(1).is_empty(), "a text column keeps no f64 copy");
    assert!(t.value(0, 1).is_nan());
    let dict = t.dict(1).expect("text column is dictionary encoded");
    assert_eq!(dict.levels, vec!["A", "B"]);
    assert_eq!(dict.codes, vec![0, 1, 0]);
    assert_eq!(t.text(2, 1), Some("A"));

    // A low-cardinality numeric column keeps the exact source text; a
    // high-cardinality one does not (the text would dominate memory).
    let lo = data::parse_table("flag,y\n1,0\n0,1\n1,1\n").unwrap();
    assert_eq!(lo.dict(0).unwrap().levels, vec!["1", "0"]);
    let mut csv = String::from("x,y\n");
    for i in 0..(data::NUMERIC_DICT_MAX + 10) {
        csv.push_str(&format!("{},{}\n", i, i));
    }
    let hi = data::parse_table(&csv).unwrap();
    assert!(hi.dict(0).is_none(), "high-cardinality numeric text is dropped");
    assert_eq!(hi.col(0).len(), hi.len());
}

// ---------------------------------------------------------------------------
// #126: quoted fields, quote-aware delimiter sniffing and streaming errors
// ---------------------------------------------------------------------------

#[test]
fn quoted_commas_tabs_newlines_escaped_quotes_crlf_and_empty_field() {
    // One fixture with every hazard in the issue. The last row exercises a
    // quoted empty field; the file uses CRLF record separators.
    let csv = concat!(
        "x,y,label\r\n",
        "1,2,\"a,b\"\r\n",
        "3,4,\"c\td\"\r\n",
        "5,6,\"line1\nline2\"\r\n",
        "7,8,\"he said \"\"hi\"\"\"\r\n",
        "9,10,\"\"\r\n",
    );
    let t = data::parse_table(csv).unwrap();
    assert_eq!(t.len(), 5);
    assert_eq!(t.names, vec!["x", "y", "label"]);
    assert!(t.numeric[0] && t.numeric[1], "x and y stay numeric");
    assert!(!t.numeric[2], "the label column is text");
    assert_eq!(t.text(0, 2), Some("a,b"));
    assert_eq!(t.text(1, 2), Some("c\td"));
    assert_eq!(t.text(2, 2), Some("line1\nline2"));
    assert_eq!(t.text(3, 2), Some("he said \"hi\""));
    assert_eq!(t.text(4, 2), Some(""));
    assert_eq!((0..5).map(|r| t.value(r, 0)).collect::<Vec<_>>(), vec![1.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn delimiter_sniffing_ignores_delimiters_inside_quotes() {
    // Tab file whose header quotes a comma: the comma count (3 in header)
    // must not compete with the tab count.
    let tabbed = "time\tvalue\tnote\n1\t2\t\"a,b\"\n3\t4\t\"c,d\"\n";
    let t = data::parse_table(tabbed).unwrap();
    assert_eq!(t.n_cols(), 3);
    assert_eq!(t.text(0, 2), Some("a,b"));

    // Comma file whose header quotes a tab.
    let comma = "time,value,note\n1,2,\"a\tb\"\n3,4,\"c\td\"\n";
    let t = data::parse_table(comma).unwrap();
    assert_eq!(t.n_cols(), 3);
    assert_eq!(t.text(0, 2), Some("a\tb"));

    // Semicolon file with quoted commas and an unquoted tab in a value.
    let semi = "time;value;note\n1;2;\"a,b\"\n3;4;\"c,d\"\n";
    let t = data::parse_table(semi).unwrap();
    assert_eq!(t.n_cols(), 3);
    assert_eq!(t.text(1, 2), Some("c,d"));
}

#[test]
fn a_quoted_comma_is_not_a_ragged_row_and_real_ragged_rows_still_name_the_line() {
    // Quoted commas must not look like extra fields.
    let t = data::parse_table("x,y,label\n1,2,\"a,b\"\n3,4,\"c,d\"\n").unwrap();
    assert_eq!(t.len(), 2);

    // A genuinely short row still fails at its 1-based physical line, even
    // after a record that spans two lines inside quotes.
    let err = data::parse_table("x,y,label\n1,2,\"a\nb\"\n3,4\n5\n").unwrap_err();
    assert!(err.contains("line 4"), "{err}");
    assert!(err.contains("expected 3") && err.contains("found 2"), "{err}");
}

#[test]
fn quoted_field_at_end_of_file_without_a_trailing_newline() {
    let csv = "x,label\n1,\"a,b\"\n2,\"the end\"";
    let t = data::parse_table(csv).unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(t.text(1, 1), Some("the end"));
}

#[test]
fn unterminated_quote_names_the_opening_line() {
    let err = data::parse_table("x,y\n1,\"open\n2,3\n").unwrap_err();
    assert!(err.contains("line 2") && err.contains("unterminated"), "{err}");
}
