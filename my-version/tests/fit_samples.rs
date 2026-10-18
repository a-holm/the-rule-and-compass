//! Constant-fitting screen tests (issue #85).
//!
//! `--fit-samples N` changes only the rows constants are fitted on. Candidates
//! are still scored on the generation's sample, and the reported archive is
//! refit on every training row. These tests pin down the off path (bit for
//! bit), the full-data guarantee, the CLI note and a small recovery problem.

use rcsr::data::Sample;
use rcsr::expr;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};

/// A two-variable linear problem with a little deterministic noise.
fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b + 1.0 + noise);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

fn config(seed: u32) -> Config {
    Config {
        generations: 12,
        islands: 1,
        pop: 30,
        max_complexity: 12,
        fit_iters: 5,
        patience: 0,
        seed,
        ..Default::default()
    }
}

fn describe(out: &Outcome) -> Vec<(u64, usize, String)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.complexity,
                expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]),
            )
        })
        .collect()
}

#[test]
fn fit_samples_zero_is_bit_identical_to_the_default() {
    let train = sample(400);
    let test = sample(100);
    let plain = search::run(&config(5), &train, &test);
    let off = search::run(
        &Config {
            fit_samples: 0,
            ..config(5)
        },
        &train,
        &test,
    );
    assert_eq!(describe(&plain), describe(&off));
    assert_eq!(plain.test_errors, off.test_errors);
    assert_eq!(plain.fit_samples, 0);
    assert!(!plain.archive_refit);
}

#[test]
fn fit_samples_at_least_the_training_size_is_an_exact_no_op() {
    let train = sample(300);
    let test = sample(100);
    let plain = search::run(&config(7), &train, &test);
    for n in [train.len(), train.len() + 1, 10_000] {
        let out = search::run(
            &Config {
                fit_samples: n,
                ..config(7)
            },
            &train,
            &test,
        );
        assert_eq!(describe(&plain), describe(&out), "N = {} changed the archive", n);
        assert_eq!(plain.test_errors, out.test_errors);
    }
}

#[test]
fn archive_errors_are_full_data_when_fitting_is_screened() {
    let train = sample(800);
    let test = sample(200);
    let cfg = Config {
        fit_samples: 25,
        ..config(9)
    };
    let out = search::run(&cfg, &train, &test);
    assert_eq!(out.fit_samples, 25);
    assert!(out.archive_refit, "the survivors must be refit on full data");
    assert!(!out.archive.items.is_empty(), "the search must return a front");
    for sol in &out.archive.items {
        let full = search::eval_error(&sol.tree, &train, cfg.metric);
        assert_eq!(
            sol.error.to_bits(),
            full.to_bits(),
            "every archived error must be the full-training error"
        );
    }
}

#[test]
fn recovers_a_law_with_fit_samples() {
    // y = 2*x + 1 on 200 rows; the constants are fitted on 40 rows only.
    let n = 200;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        xs.push(vec![x]);
        ys.push(2.0 * x + 1.0);
    }
    let train = Sample {
        x: xs.clone(),
        y: ys.clone(),
        w: Vec::new(),
        ys: Vec::new(),
    };
    let test = Sample {
        x: xs,
        y: ys,
        w: Vec::new(),
        ys: Vec::new(),
    };
    let cfg = Config {
        generations: 40,
        islands: 1,
        pop: 80,
        max_complexity: 12,
        fit_iters: 8,
        patience: 0,
        seed: 1,
        metric: Metric::Squared,
        ops: vec![expr::Op::Add, expr::Op::Sub, expr::Op::Mul],
        fit_samples: 40,
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    let best = out
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-6, "best test error = {}", best);
}

/// A temporary CSV that removes itself when the test ends.
struct TempCsv {
    path: std::path::PathBuf,
}

impl TempCsv {
    fn new(contents: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsr-fit-samples-{}-{}.csv",
            std::process::id(),
            n
        ));
        std::fs::write(&path, contents).expect("write temp csv");
        TempCsv { path }
    }

    fn as_str(&self) -> &str {
        self.path.to_str().expect("utf-8 path")
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn line_csv(n: usize) -> String {
    use std::fmt::Write as _;
    let mut text = String::from("x,y\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        let _ = writeln!(text, "{:.6},{:.6}", x, 2.0 * x + 1.0);
    }
    text
}

#[test]
fn cli_reports_the_screening_size_and_refit() {
    let csv = TempCsv::new(&line_csv(200));
    let text = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            csv.as_str(),
            "--target",
            "y",
            "--vars",
            "x",
            "--fit-samples",
            "20",
            "--generations",
            "3",
            "--pop",
            "20",
            "--seed",
            "1",
        ])
        .output()
        .expect("run rcsr");
    assert!(text.status.success(), "stderr: {}", String::from_utf8_lossy(&text.stderr));
    let err = String::from_utf8_lossy(&text.stderr);
    assert!(
        err.contains("constants screened on 20 of 150 training rows per generation"),
        "the text header must state the screening size: {}",
        err
    );
    assert!(
        err.contains("survivors refit on all 150 rows"),
        "the text header must state the refit: {}",
        err
    );

    let json = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            csv.as_str(),
            "--target",
            "y",
            "--vars",
            "x",
            "--fit-samples",
            "20",
            "--generations",
            "3",
            "--pop",
            "20",
            "--seed",
            "1",
            "--json",
            "--quiet",
        ])
        .output()
        .expect("run rcsr --json");
    assert!(json.status.success());
    let stdout = String::from_utf8_lossy(&json.stdout);
    assert!(
        stdout.contains("\"fit\":{\"samples\":20,\"train_rows\":150,\"refit\":true}"),
        "the JSON must state the screening size and refit: {}",
        stdout
    );
}

#[test]
fn fit_samples_one_is_a_usage_error() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", "/nonexistent", "--target", "y"])
        .args(["--fit-samples", "1"])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--fit-samples"));
}
