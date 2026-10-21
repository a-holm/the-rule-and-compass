//! Promotion-refit budget tests (issue #201).
//!
//! `--refit-iters N` changes only the full-data refit of candidates promoted
//! by `--fit-samples`. The default (0) must stay bit-identical to the
//! historical full `--fit-iters` path; the opt-in path must be deterministic
//! and pass the pre-registered quality gate on a small fixture.

use rcsr::data::Sample;
use rcsr::expr;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};
use std::process::Command;

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

fn config(seed: u32, refit_iters: usize) -> Config {
    Config {
        generations: 12,
        islands: 1,
        pop: 30,
        max_complexity: 12,
        metric: Metric::Squared,
        fit_iters: 6,
        fit_samples: 40,
        refit_iters,
        patience: 0,
        seed,
        ..Default::default()
    }
}

fn describe(out: &Outcome) -> Vec<(u64, usize, u64, String)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.complexity,
                expr::structural_hash(&s.tree),
                expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]),
            )
        })
        .collect()
}

fn best_error(out: &Outcome) -> f64 {
    out.archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min)
}

fn hypervolume(out: &Outcome) -> f64 {
    let points: Vec<(f64, f64)> = out
        .archive
        .items
        .iter()
        .map(|s| (s.cost, s.error))
        .collect();
    // Fixed reference, exactly like the CLI's comparable validation HV.
    rcsr::qd::fixed_reference_hypervolume(&points, 1.0)
}

/// #201 acceptance: `--refit-iters` unset (0) is the historical path, bit for
/// bit, including the archive and its structural hashes.
#[test]
fn refit_iters_zero_is_bit_identical_to_the_default() {
    let train = sample(120);
    let test = sample(40);
    let default = config(1, 0);
    let explicit_zero = config(1, 0);
    let a = search::run(&default, &train, &test);
    let b = search::run(&explicit_zero, &train, &test);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(
        a.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
        b.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
    );
}

/// The opt-in budget must not introduce run-to-run nondeterminism.
#[test]
fn refit_iters_one_is_deterministic() {
    let train = sample(120);
    let test = sample(40);
    let cfg = config(7, 1);
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(
        a.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
        b.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
    );
}

/// Pre-registered quality gate on a small deterministic fixture: the opt-in
/// one-iteration promotion refit must not lose best error (5% per seed, 20%
/// worst case) and must keep fixed-reference hypervolume within 2% of the
/// full-budget promotion refit at equal evaluations.
#[test]
fn refit_iters_one_meets_the_quality_gate() {
    let train = sample(120);
    let test = sample(40);
    for seed in [1, 7, 42] {
        let control = search::run(&config(seed, 0), &train, &test);
        let cheap = search::run(&config(seed, 1), &train, &test);
        let control_best = best_error(&control);
        let cheap_best = best_error(&cheap);
        let control_hv = hypervolume(&control);
        let cheap_hv = hypervolume(&cheap);
        assert!(
            cheap_best <= control_best * 1.05 + 1e-12,
            "seed {seed}: best error regressed {control_best} -> {cheap_best} (gate 1.05x)"
        );
        assert!(
            cheap_hv + 1e-12 >= control_hv * 0.98,
            "seed {seed}: hypervolume regressed {control_hv} -> {cheap_hv} (gate 0.98x)"
        );
    }
}

/// The flag is documented in `--help`/`schema`, accepts an explicit 0 (the
/// historical path), and is never reported as an unknown argument.
#[test]
fn cli_accepts_and_documents_refit_iters() {
    let bin = env!("CARGO_BIN_EXE_rcsr");
    let help = Command::new(bin).arg("--help").output().expect("run --help");
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--refit-iters"));

    let schema = Command::new(bin).arg("schema").output().expect("run schema");
    assert!(schema.status.success());
    let doc = String::from_utf8_lossy(&schema.stdout);
    assert!(doc.contains("--refit-iters"));

    for value in ["0", "1"] {
        let out = Command::new(bin)
            .args(["--refit-iters", value])
            .output()
            .expect("run rcsr");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("unknown argument"),
            "--refit-iters {value} was rejected as unknown: {stderr}"
        );
        assert_eq!(out.status.code(), Some(2), "expected a usage error: {stderr}");
    }
}

/// The flag is a budget, not a new objective: a no-fit run stays a no-op even
/// when a promotion budget is requested.
#[test]
fn refit_iters_with_no_fit_is_a_no_op() {
    let train = sample(120);
    let test = sample(40);
    let mut cfg = config(3, 1);
    cfg.fit_iters = 0;
    let a = search::run(&cfg, &train, &test);
    cfg.refit_iters = 0;
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b));
}
