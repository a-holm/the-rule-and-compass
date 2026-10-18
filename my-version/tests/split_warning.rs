//! The split warning must describe what was observed, not what it decided
//! (issue #49).
//!
//! `suggest_split`'s notes are used by two callers: `--split auto`, which
//! adopts the suggestion, and the warning path, which does not. When the notes
//! said "using a chronological split", the warning contradicted the
//! `split=random` line printed beside it.

use rcsr::data;
use std::io::Write;
use std::process::{Command, Stdio};

/// `t` is a perfectly spaced 1..n axis, `x = t`, `y = 2t + 1`.
fn time_ordered_csv(n: usize) -> String {
    let mut s = String::from("t,x,y\n");
    for i in 1..=n {
        s.push_str(&format!("{},{},{}\n", i, i, 2 * i + 1));
    }
    s
}

fn run(args: &[&str]) -> (Option<i32>, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rcsr");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(time_ordered_csv(40).as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn the_random_warning_does_not_claim_a_split_it_did_not_use() {
    let (code, stdout, stderr) =
        run(&["--data", "-", "--target", "y", "--vars", "x", "--generations", "5", "--pop", "20"]);

    assert_eq!(code, Some(0));
    // The human-readable report (including the split line) is on stderr;
    // stdout carries the solutions.
    assert!(!stdout.is_empty(), "solutions go to stdout");
    // It still warns, and it still says a random split was used.
    assert!(stderr.contains("the rows do not look exchangeable"), "stderr: {stderr}");
    assert!(stderr.contains("found a time-like column 't'"), "stderr: {stderr}");
    assert!(stderr.contains("split=random"), "stderr: {stderr}");

    // The contradiction: claiming the split *is* chronological while reporting
    // a random one.
    assert!(
        !stderr.contains("using a chronological split"),
        "the warning must not claim a split it did not use: {stderr}"
    );
    assert!(
        !stderr.contains("holding out whole groups"),
        "same for groups: {stderr}"
    );
    // And it must be clear that the suggestion is a suggestion.
    assert!(stderr.contains("it would"), "stderr: {stderr}");
}

#[test]
fn auto_still_explains_and_uses_the_suggested_split() {
    let (code, stdout, stderr) = run(&[
        "--data",
        "-",
        "--target",
        "y",
        "--vars",
        "x",
        "--split",
        "auto",
        "--generations",
        "5",
        "--pop",
        "20",
    ]);

    assert_eq!(code, Some(0));
    let _ = stdout;
    // The observation and the decision are both stated, consistently.
    assert!(stderr.contains("found a time-like column 't'"), "stderr: {stderr}");
    assert!(stderr.contains("using the temporal split"), "stderr: {stderr}");
    assert!(stderr.contains("split=temporal"), "stderr: {stderr}");
}

/// The invariant behind both tests: a note may describe the data, never the
/// mode in force.
#[test]
fn suggest_split_notes_are_observations_only() {
    let t = data::parse_table(&time_ordered_csv(20)).unwrap();
    // target `y` (index 2), input `x` (index 1); `t` (index 0) is left over.
    let (mode, time_col, _group, notes) = data::suggest_split(&t, 2, &[1]);
    assert_eq!(mode, data::SplitMode::Temporal);
    assert_eq!(time_col, Some(0));
    for n in &notes {
        assert!(
            !n.contains("using") && !n.contains("holding out"),
            "a note must not assert the mode in force: {n}"
        );
    }
}

/// A table with neither a time axis nor groups gets the random suggestion, and
/// the wording must be just as neutral there.
#[test]
fn no_structure_notes_are_neutral_too() {
    let mut s = String::from("a,b\n");
    for i in 0..20 {
        let v = if i % 2 == 0 { 0.317 } else { -1.921 };
        s.push_str(&format!("{},{}\n", v, v * 2.0 + 1.0));
    }
    let t = data::parse_table(&s).unwrap();
    let (mode, _, _, notes) = data::suggest_split(&t, 1, &[0]);
    assert_eq!(mode, data::SplitMode::Random);
    assert!(notes.iter().all(|n| !n.contains("using")), "{notes:?}");
}
