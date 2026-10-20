//! Validation slice and lockbox selection (issue #111).
//!
//! The training period is split chronologically into fit-train and validation;
//! the test block is only scored. These tests pin the sizes, the purge, the
//! selection API and the optimism of ranking by the test block.

use rcsr::data::{self, PrepSpec, SplitMode, SplitSpec};
use rcsr::rng::Mt;
use rcsr::selection::{select, Candidate, SelectBy};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Output};

fn table(n: usize) -> data::Table {
    let mut s = String::from("t,x,y\n");
    for i in 0..n {
        let x = i as f64;
        let _ = writeln!(s, "{},{},{}", i, x, 2.0 * x + 1.0);
    }
    data::parse_table(&s).unwrap()
}

fn spec(validation_frac: f64, gap: usize) -> SplitSpec {
    SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        test_frac: 0.25,
        gap,
        validation_frac,
        ..Default::default()
    }
}

fn prep(n: usize, validation_frac: f64, gap: usize) -> data::Prepared {
    data::prepare_full_with(
        &table(n),
        &["x".to_string()],
        "y",
        &spec(validation_frac, gap),
        &mut Mt::new(0),
        &PrepSpec::default(),
    )
    .expect("preparation succeeds")
}

fn raw(sample: &data::Sample, p: &data::Prepared) -> Vec<f64> {
    let m = p.means[0];
    let sd = p.stds[0];
    sample.x.iter().map(|r| r[0] * sd + m).collect()
}

#[test]
fn validation_frac_holds_out_the_last_twenty_percent_of_training() {
    let p = prep(100, 0.2, 0);
    // 100 rows -> 75 training / 25 test; 20% of 75 = 15 validation.
    assert_eq!(p.report.train_rows, 60);
    assert_eq!(p.report.validation_rows, 15);
    assert_eq!(p.report.test_rows, 25);
    let v = p.validation.as_ref().expect("validation slice present");
    assert_eq!(v.len(), 15);

    let tr = raw(&p.train, &p);
    let va = raw(v, &p);
    let te = raw(&p.test, &p);
    assert!(tr.iter().cloned().fold(f64::MIN, f64::max) < va[0]);
    assert!(va.iter().cloned().fold(f64::MIN, f64::max) < te[0]);
    // Both errors are available: the fit train error and the validation error.
    assert!(tr.len() + va.len() == p.report.train_rows + p.report.validation_rows);
}

#[test]
fn the_purge_is_in_time_between_fit_train_validation_and_test() {
    let p = prep(100, 0.2, 5);
    let tr = raw(&p.train, &p);
    let va = raw(p.validation.as_ref().unwrap(), &p);
    let te = raw(&p.test, &p);
    let gap = 5usize;
    assert_eq!(p.report.purged_validation_rows, gap);
    // Five complete rows are missing between the last fit row and the first
    // validation row (and the split's own embargo before the test block).
    assert_eq!(va[0] - tr.last().copied().unwrap(), gap as f64 + 1.0);
    assert!(te[0] - va.last().copied().unwrap() >= gap as f64 + 1.0);
}

#[test]
fn no_validation_slice_means_no_validation_and_the_old_default() {
    let p = prep(100, 0.0, 0);
    assert!(p.validation.is_none());
    assert_eq!(p.report.validation_rows, 0);
    assert_eq!(p.report.train_rows, 75);
}

#[test]
fn validation_selection_never_reads_the_test_block() {
    let candidates = [
        Candidate {
            train: 0.10,
            validation: Some(0.50),
            test: Some(0.10),
            aicc: 1.0,
        },
        Candidate {
            train: 0.12,
            validation: Some(0.30),
            test: Some(0.20),
            aicc: 0.9,
        },
    ];
    let by_val = select(&candidates, SelectBy::Validation).unwrap();
    // Rewriting every test score (the act of peeking) must not move the choice.
    let mut peeked = candidates;
    for (i, c) in peeked.iter_mut().enumerate() {
        c.test = Some(if i == 0 { 0.001 } else { 99.0 });
    }
    assert_eq!(select(&peeked, SelectBy::Validation), Some(by_val));
}

#[test]
fn ranking_by_test_looks_better_but_validation_picks_the_honest_candidate() {
    let candidates = [
        Candidate {
            train: 0.10,
            validation: Some(0.50),
            test: Some(0.10),
            aicc: 1.0,
        },
        Candidate {
            train: 0.12,
            validation: Some(0.30),
            test: Some(0.20),
            aicc: 0.9,
        },
    ];
    // The test-ranked winner reports the smaller test error...
    let by_test = candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.test.unwrap().partial_cmp(&b.test.unwrap()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let by_val = select(&candidates, SelectBy::Validation).unwrap();
    assert_eq!(by_test, 0);
    assert_eq!(by_val, 1);
    assert!(candidates[by_test].test.unwrap() < candidates[by_val].test.unwrap());
    // ...but it is the worse model on the validation slice. Selection on the
    // lockbox spends the very block it reports, so its advantage is optimism.
    assert!(candidates[by_test].validation.unwrap() > candidates[by_val].validation.unwrap());
}

struct TempCsv(PathBuf);

impl TempCsv {
    fn new(contents: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsr-validation-{}-{}.csv",
            std::process::id(),
            n
        ));
        std::fs::write(&path, contents).expect("write temp csv");
        TempCsv(path)
    }
    fn as_str(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn json_reports_the_validation_slice_and_error() {
    let mut text = String::from("t,x,y\n");
    for i in 0..80 {
        let x = i as f64 * 0.1;
        let _ = writeln!(text, "{},{:.6},{:.6}", i, x, 2.0 * x + 1.0);
    }
    let csv = TempCsv::new(&text);
    let out = run(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--split",
        "temporal",
        "--time",
        "t",
        "--validation-frac",
        "0.2",
        "--select-by",
        "validation",
        "--generations",
        "2",
        "--pop",
        "20",
        "--format",
        "json",
    ]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"validation_rows\":"), "json: {}", stdout);
    assert!(stdout.contains("\"validation_error\":"), "json: {}", stdout);
    assert!(stdout.contains("\"archive_rank\":"), "json: {}", stdout);
}
