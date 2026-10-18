//! Row selection by expression (issue #3).
//!
//! `--train-select` / `--test-select` assign rows to a side; `--exclude` drops
//! rows before anything else. The ambiguous cases have explicit rules and are
//! counted in the split report, so nothing is silently discarded.

use rcsr::data::{self, Filter, PrepSpec, SplitMode, SplitSpec};
use rcsr::expr;
use rcsr::rng::Mt;

/// Rows `1..=10`, with `x_ = i` and `y = i`.
fn table() -> data::Table {
    let mut s = String::from("x,y\n");
    for i in 1..=10 {
        s.push_str(&format!("{},{}\n", i, i));
    }
    data::parse_table(&s).unwrap()
}

fn filter(src: &str) -> Filter {
    let vars = vec!["x".to_string(), "y".to_string()];
    let tree = expr::parse(src, &vars).expect("test expression parses");
    Filter {
        tree,
        vars,
        expression: src.to_string(),
    }
}

fn prep(spec: SplitSpec) -> data::Prepared {
    data::prepare_full_with(
        &table(),
        &["x".to_string()],
        "y",
        &spec,
        &mut Mt::new(0),
        &PrepSpec::default(),
    )
    .expect("preparation succeeds")
}

/// The original `x` values on a side, recovered from the normalised sample.
fn raw_x(p: &data::Prepared, s: &data::Sample) -> Vec<f64> {
    let m = p.means[0];
    let sd = p.stds[0];
    let mut v: Vec<f64> = s.x.iter().map(|r| r[0] * sd + m).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn train_select_partitions_the_rows() {
    let spec = SplitSpec {
        train_select: Some(filter("x < 6")),
        // Only one side is given: the other side is the complement.
        ..Default::default()
    };
    let p = prep(spec);
    assert_eq!(p.report.mode, "select");
    assert_eq!(p.train.len(), 5);
    assert_eq!(p.test.len(), 5);
    assert_eq!(raw_x(&p, &p.train), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(raw_x(&p, &p.test), vec![6.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn a_row_matching_both_selections_goes_to_training_and_is_counted() {
    let spec = SplitSpec {
        train_select: Some(filter("x < 6")),
        test_select: Some(filter("x > 3")),
        ..Default::default()
    };
    let p = prep(spec);
    // x in 1..=5 train, x in 6..=10 test, x in 4..=5 claimed by both -> train.
    assert_eq!(p.report.overlap_rows, 2);
    assert_eq!(p.report.unassigned_rows, 0);
    assert_eq!(raw_x(&p, &p.train), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(raw_x(&p, &p.test), vec![6.0, 7.0, 8.0, 9.0, 10.0]);
    assert!(
        p.report
            .notes
            .iter()
            .any(|n| n.contains("both selections") && n.contains('2')),
        "the report must explain the overlap: {:?}",
        p.report.notes
    );
}

#[test]
fn a_row_matching_neither_selection_is_reported_unused() {
    let spec = SplitSpec {
        train_select: Some(filter("x < 3")),
        test_select: Some(filter("x > 8")),
        ..Default::default()
    };
    let p = prep(spec);
    // 1..=2 train, 9..=10 test, 3..=8 unused (six rows).
    assert_eq!(p.report.unassigned_rows, 6);
    assert_eq!(raw_x(&p, &p.train), vec![1.0, 2.0]);
    assert_eq!(raw_x(&p, &p.test), vec![9.0, 10.0]);
    assert!(
        p.report
            .notes
            .iter()
            .any(|n| n.contains("neither") && n.contains('6')),
        "the report must explain the unused rows: {:?}",
        p.report.notes
    );
}

#[test]
fn test_select_alone_gives_the_complement_to_training() {
    let spec = SplitSpec {
        test_select: Some(filter("x >= 8")),
        ..Default::default()
    };
    let p = prep(spec);
    assert_eq!(raw_x(&p, &p.test), vec![8.0, 9.0, 10.0]);
    assert_eq!(raw_x(&p, &p.train).len(), 7);
}

#[test]
fn exclude_drops_rows_before_the_split() {
    let spec = SplitSpec {
        exclude: Some(filter("x == 5")),
        train_select: Some(filter("x < 6")),
        ..Default::default()
    };
    let p = prep(spec);
    assert_eq!(p.report.excluded_rows, 1);
    assert_eq!(
        raw_x(&p, &p.train),
        vec![1.0, 2.0, 3.0, 4.0],
        "x = 5 must not appear on either side"
    );
    // x == 5 is on the training side of `x < 6`, so test is unaffected.
    assert_eq!(raw_x(&p, &p.test), vec![6.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn selection_overrides_a_temporal_spec_and_says_so() {
    // A temporal split would put the future in test; the explicit selection
    // must win, and the report must record that it replaced the mode.
    let spec = SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        test_select: Some(filter("x >= 8")),
        ..Default::default()
    };
    let p = prep(spec);
    assert_eq!(p.report.mode, "select");
    assert_eq!(raw_x(&p, &p.test), vec![8.0, 9.0, 10.0]);
    assert!(
        p.report
            .notes
            .iter()
            .any(|n| n.contains("replaced the temporal split")),
        "the report must say the mode was replaced: {:?}",
        p.report.notes
    );
}

#[test]
fn selection_is_deterministic() {
    let build = || {
        prep(SplitSpec {
            train_select: Some(filter("x < 6")),
            test_select: Some(filter("x > 8")),
            ..Default::default()
        })
    };
    let a = build();
    let b = build();
    assert_eq!(raw_x(&a, &a.train), raw_x(&b, &b.train));
    assert_eq!(raw_x(&a, &a.test), raw_x(&b, &b.test));
    assert_eq!(a.report.unassigned_rows, b.report.unassigned_rows);
}

#[test]
fn a_bad_selection_expression_is_a_usage_error() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            "-",
            "--target",
            "y",
            "--train-select",
            "x >",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"x,y\n1,2\n2,3\n4,5\n").unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--train-select"), "stderr: {err}");
    assert!(err.contains("column 4"), "stderr: {err}");
}
