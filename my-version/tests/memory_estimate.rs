//! Memory-guard calibration (issue #112): the estimate comes from the table's
//! rows x columns shape, not from a multiple of the file size. The committed
//! fixtures are tiny; the 1M-row and 90-column peak-RSS measurements are in
//! `docs/data-format.md`.

use rcsr::data::{self, CsvShape, PrepReport};
use std::process::Command;

const WIDE: &str = include_str!("fixtures/wide90_small.csv");
const NARROW: &str = include_str!("fixtures/narrow2_large.csv");

fn shape(text: &str) -> CsvShape {
    data::csv_shape(text)
}

#[test]
fn wide_csv_with_a_small_file_size_is_estimated_from_its_shape() {
    let s = shape(WIDE);
    assert_eq!(s.cols, 90);
    assert_eq!(s.rows, 10);
    assert!(s.bytes < 4_000, "the wide fixture is deliberately small on disk");

    let search = data::csv_estimate(s, 0);
    // The table and preparation terms dominate the text term: this is the old
    // 33x-file-size model's blind spot.
    assert!(
        search.table_bytes + search.prep_bytes > 10 * search.text_bytes,
        "{search:?}"
    );
    assert_eq!(search.table_bytes, 10 * 90 * data::CSV_TABLE_BYTES_PER_CELL);
    assert_eq!(search.used_cols, 90);
    assert!(search.total_bytes >= search.table_bytes + search.prep_bytes);
    assert_eq!(
        search.slack_bytes,
        search.total_bytes - (search.text_bytes + search.table_bytes + search.prep_bytes)
    );

    let load = data::csv_load_estimate(s);
    assert_eq!(load.prep_bytes, 0, "import does not prepare a search sample");
    assert!(load.total_bytes < search.total_bytes);
    assert!(load.total_bytes >= load.text_bytes + load.table_bytes);
}

#[test]
fn narrow_csv_with_a_large_file_size_does_not_dominate_the_estimate() {
    let s = shape(NARROW);
    assert_eq!(s.cols, 2);
    assert_eq!(s.rows, 10_000);
    assert!(s.bytes > 100_000, "the narrow fixture is the larger file on disk");

    let search = data::csv_estimate(s, 2);
    // Over 90 % of the model is the shape (table + preparation), not the text.
    assert!(search.text_bytes * 10 < search.total_bytes, "{search:?}");
    assert!(search.table_bytes > 2 * search.text_bytes, "{search:?}");
    assert_eq!(search.rows, 10_000);
    assert_eq!(search.used_cols, 2);

    // The old "33 x file size" number for this fixture is 3.8 MB; the new
    // estimate is the same order, while for the wide fixture it is far from it.
    let old = s.bytes * 33;
    assert!(
        search.total_bytes * 2 > old && search.total_bytes < old * 2,
        "narrow estimate {} vs old 33x {}",
        search.total_bytes,
        old
    );
}

#[test]
fn the_preparation_block_reports_the_rows_by_columns_basis() {
    let s = shape(WIDE);
    let mut report = PrepReport::default();
    report.memory_estimate = Some(data::csv_estimate(s, 90));
    let lines = report.text_lines(None, 0).join("\n");
    assert!(lines.contains("memory estimate"), "{lines}");
    assert!(lines.contains("10 rows x 90 columns"), "{lines}");
    assert!(lines.contains("basis: text"), "{lines}");
    assert!(lines.contains("table"), "{lines}");
    assert!(lines.contains("preparation"), "{lines}");

    let json = report.json();
    assert!(json.contains("\"memory\":{"), "{json}");
    assert!(json.contains("\"rows\":10"), "{json}");
    assert!(json.contains("\"columns\":90"), "{json}");
    assert!(json.contains("\"basis\":\"rows x columns\""), "{json}");
}

#[test]
fn the_cli_json_preparation_block_carries_the_estimate_for_a_wide_csv() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wide90_small.csv"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path,
            "--target",
            "y",
            "--generations",
            "1",
            "--pop",
            "4",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"memory\":{"), "{stdout}");
    assert!(stdout.contains("\"columns\":90"), "{stdout}");
    assert!(stdout.contains("\"basis\":\"rows x columns\""), "{stdout}");
}
