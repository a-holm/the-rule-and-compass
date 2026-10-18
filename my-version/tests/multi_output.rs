//! Multi-output regression (issue #15).
//!
//! Two outputs share the term `x1*x2`. The shared-shape search is compared with
//! two independent single-target searches at the same generation/population
//! budget, the JSON contract is checked, a single target is unchanged, and a
//! missing target column is a clear input error.

use rcsr::data::Sample;
use rcsr::metrics::Metric;
use rcsr::multi;
use rcsr::search::{self, Config};
use std::path::PathBuf;
use std::process::{Command, Output};

fn config(seed: u32, generations: usize) -> Config {
    Config {
        generations,
        islands: 1,
        pop: 160,
        max_complexity: 9,
        fit_iters: 12,
        patience: 0,
        seed,
        metric: Metric::Absolute,
        ..Default::default()
    }
}

/// `y1 = 2*x1*x2`, `y2 = 3*x1*x2 + 1` on a deterministic grid. The two outputs
/// share one shape, `c0*x1*x2 + c1` (with `c1 = 0` for the first).
fn shared() -> (Sample, Sample) {
    let mut rows = Vec::new();
    for i in 0..7 {
        for j in 0..7 {
            let a = -1.5 + 3.0 * i as f64 / 6.0;
            let b = -1.0 + 2.0 * j as f64 / 6.0;
            rows.push((a, b));
        }
    }
    // Interleave so a sequential split still covers the grid.
    let split = rows.len() * 3 / 4;
    let make = |slice: &[(f64, f64)]| {
        let x: Vec<Vec<f64>> = slice.iter().map(|&(a, b)| vec![a, b]).collect();
        let y1: Vec<f64> = slice.iter().map(|&(a, b)| 2.0 * a * b).collect();
        let y2: Vec<f64> = slice.iter().map(|&(a, b)| 3.0 * a * b + 1.0).collect();
        let mut s = Sample::new(x, y1.clone());
        s.ys = vec![y1, y2];
        s
    };
    (make(&rows[..split]), make(&rows[split..]))
}

fn single(train: &Sample, test: &Sample, target: usize) -> (Sample, Sample) {
    let pick = |s: &Sample| {
        let y = s.target(target).to_vec();
        let mut out = Sample::new(s.x.clone(), y);
        out.w = s.w.clone();
        out
    };
    (pick(train), pick(test))
}

fn best_error(out: &search::Outcome) -> f64 {
    out.archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn shared_shape_recovery_beats_two_independent_runs() {
    let (train, test) = shared();
    let cfg = config(7, 120);

    let shared_out = search::run(&cfg, &train, &test);
    let shared_error = best_error(&shared_out);
    let shared_best = shared_out
        .archive
        .sorted()
        .into_iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
        .expect("a shared solution");

    // The same structure serves both outputs; only the constants differ.
    let trees = multi::fitted_outputs(&shared_best.tree, &train, cfg.fit_iters);
    assert_eq!(
        rcsr::stability::shape(&trees[0]),
        rcsr::stability::shape(&trees[1]),
        "the shared candidate must keep one structure for every output"
    );

    // Two independent searches at the same budget. Their combined error is the
    // mean of their own best training errors.
    let mut independent = Vec::new();
    for target in 0..2 {
        let (tr, te) = single(&train, &test, target);
        let out = search::run(&cfg, &tr, &te);
        independent.push(best_error(&out));
    }
    let independent_mean = (independent[0] + independent[1]) / 2.0;

    eprintln!(
        "shared best shape: {} (complexity {})",
        rcsr::expr::to_string(&shared_best.tree, &["x1".to_string(), "x2".to_string()]),
        shared_best.cost
    );

    for (j, t) in trees.iter().enumerate() {
        eprintln!(
            "  target {} fitted: {}",
            j,
            rcsr::expr::to_string(t, &["x1".to_string(), "x2".to_string()])
        );
    }
    eprintln!(
        "shared combined error = {:.9}; independent errors = {:.9}, {:.9}; \
         independent mean = {:.9}",
        shared_error, independent[0], independent[1], independent_mean
    );

    // Both paths can land at the floating-point noise floor (one exactly 0.0,
    // the other ~1e-31), so an exact `<=` comparison is a coin flip. The claim
    // is "not measurably worse", which needs an absolute tolerance.
    const NOISE: f64 = 1e-12;
    assert!(
        shared_error <= independent_mean + NOISE,
        "shared combined error {shared_error:.9} must not be worse than the mean \
         independent error {independent_mean:.9} (independent: {independent:?})"
    );
    // The shared front must actually contain a computable model.
    assert!(shared_error.is_finite(), "shared error = {shared_error}");
}

struct TempPath(PathBuf);

impl TempPath {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("rcsr-multi-{}-{}-{}", std::process::id(), n, name));
        TempPath(path)
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn write_shared_csv() -> TempPath {
    let path = TempPath::new("shared.csv");
    let mut text = String::from("x1,x2,y1,y2\n");
    for i in 0..7 {
        for j in 0..7 {
            let a = -1.5 + 3.0 * i as f64 / 6.0;
            let b = -1.0 + 2.0 * j as f64 / 6.0;
            text.push_str(&format!(
                "{},{},{},{}\n",
                a,
                b,
                2.0 * a * b,
                3.0 * a * b + 1.0
            ));
        }
    }
    std::fs::write(&path.0, text).unwrap();
    path
}

#[test]
fn json_has_one_entry_per_target_and_a_combined_error() {
    let csv = write_shared_csv();
    let out = rcsr(&[
        "--data",
        csv.0.to_str().unwrap(),
        "--target",
        "y1,y2",
        "--vars",
        "x1,x2",
        "--json",
        "--generations",
        "3",
        "--pop",
        "24",
        "--seed",
        "1",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8(out.stdout).unwrap();
    assert!(json.contains("\"multi_output\":true"), "{json}");
    assert!(json.contains("\"targets\":[\"y1\",\"y2\"]"), "{json}");
    // Every Pareto row carries both target entries, each with its own
    // expression, constants and error.
    let pareto = &json[json.find("\"pareto\":[").expect("pareto")..];
    assert!(pareto.contains("\"name\":\"y1\""), "{pareto}");
    assert!(pareto.contains("\"name\":\"y2\""), "{pareto}");
    assert!(pareto.contains("\"constants\":["), "{pareto}");
    assert!(pareto.contains("\"train_error\":"), "{pareto}");
    assert!(pareto.contains("\"k_constants\":"), "{pareto}");
}

#[test]
fn a_single_target_keeps_the_historical_shape() {
    let csv = write_shared_csv();
    let out = rcsr(&[
        "--data",
        csv.0.to_str().unwrap(),
        "--target",
        "y1",
        "--vars",
        "x1,x2",
        "--json",
        "--generations",
        "2",
        "--pop",
        "16",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8(out.stdout).unwrap();
    assert!(!json.contains("\"multi_output\""), "{json}");
    assert!(json.contains("\"target\":\"y1\""), "{json}");
    assert!(json.contains("\"pareto\":["), "{json}");
}

#[test]
fn a_missing_target_column_is_a_clear_input_error() {
    let csv = write_shared_csv();
    let out = rcsr(&[
        "--data",
        csv.0.to_str().unwrap(),
        "--target",
        "y1,nope",
        "--vars",
        "x1,x2",
        "--json",
    ]);
    assert_eq!(out.status.code(), Some(3), "input errors exit 3");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("target column 'nope' not found"),
        "stderr: {stderr}"
    );
}
