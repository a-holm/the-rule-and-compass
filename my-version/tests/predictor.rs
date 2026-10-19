//! Predictor-sample tests (issue #4).
//!
//! Subsampling changes only how candidates are scored and fitted. These tests
//! pin down the sampling contract, determinism, the exact no-op when the request
//! covers the training set, and the full-data guarantee for the archive (which
//! is what the report uses for `train_error`).

use rcsr::expr;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};
use rcsr::data::Sample;

/// A two-variable, noisy linear problem: `y = 2*x0 + 3*x1 + 1`.
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
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
}

fn config(seed: u32) -> Config {
    Config {
        generations: 15,
        islands: 2,
        pop: 40,
        max_complexity: 12,
        fit_iters: 5,
        patience: 0,
        seed,
        ..Default::default()
    }
}

/// A comparable view of the returned front: every error bit pattern, the
/// complexity and the expression text, in archive order.
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
fn subset_is_distinct_and_deterministic_per_seed_and_generation() {
    let rows = 1_000;
    let n = 37;
    let a = search::generation_subset(rows, n, 42, 3);
    assert_eq!(a.len(), n, "the subset must contain exactly N indices");
    assert!(a.iter().all(|&i| i < rows), "indices must be in range");
    let mut distinct = a.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), n, "indices must be drawn without replacement");

    assert_eq!(
        a,
        search::generation_subset(rows, n, 42, 3),
        "the same (seed, generation) must give the same subset"
    );
    assert_ne!(
        a,
        search::generation_subset(rows, n, 42, 4),
        "the next generation must draw a fresh subset"
    );
    assert_ne!(
        a,
        search::generation_subset(rows, n, 43, 3),
        "a different seed must draw a different subset"
    );

    // A request larger than the table is capped at the table size.
    let all = search::generation_subset(rows, rows + 50, 7, 0);
    assert_eq!(all.len(), rows);
    assert_eq!(all, (0..rows).collect::<Vec<_>>());
}

#[test]
fn same_seed_and_predictor_samples_are_deterministic() {
    let train = sample(600);
    let test = sample(200);
    let cfg = Config {
        predictor_samples: Some(50),
        ..config(11)
    };
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(a.test_errors, b.test_errors);
}

#[test]
fn predictor_samples_at_least_the_training_size_is_an_exact_no_op() {
    let train = sample(300);
    let test = sample(100);
    let plain = search::run(&config(5), &train, &test);
    // #86: a request above the measured break-even is widened to the full
    // sample, so it is an exact no-op too (271 of 300 here).
    for n in [
        train.len() * 9 / 10 + 1,
        train.len(),
        train.len() + 1,
        10_000,
    ] {
        let cfg = Config {
            predictor_samples: Some(n),
            ..config(5)
        };
        let out = search::run(&cfg, &train, &test);
        assert_eq!(
            describe(&plain),
            describe(&out),
            "N = {} must not change the archive",
            n
        );
        assert_eq!(plain.test_errors, out.test_errors);
    }
}

/// #86: the break-even keeps the measured 750/1500 win and widens near-full
/// requests (1400/1500) to the full sample.
#[test]
fn effective_subset_accepts_750_and_rejects_near_full_requests() {
    let train = sample(1500);
    let mut cfg = Config::default();

    for n in [750, 900] {
        cfg.predictor_samples = Some(n);
        assert_eq!(
            search::effective_predictor_samples(&cfg, &train),
            Some(n),
            "{} of 1500 rows is at or below the break-even",
            n
        );
    }

    for n in [0, 1, 901, 1400, 1401, 1499, 1500, 1501] {
        cfg.predictor_samples = Some(n);
        assert_eq!(
            search::effective_predictor_samples(&cfg, &train),
            None,
            "{} of 1500 rows must widen to the full sample",
            n
        );
    }

    cfg.predictor_samples = None;
    assert_eq!(search::effective_predictor_samples(&cfg, &train), None);
}

#[test]
fn archive_errors_are_full_data_even_when_fitness_is_subsampled() {
    let train = sample(800);
    let test = sample(200);
    let cfg = Config {
        predictor_samples: Some(25),
        ..config(9)
    };
    let out = search::run(&cfg, &train, &test);
    assert!(!out.archive.items.is_empty(), "the search must return a front");
    for sol in &out.archive.items {
        let full = search::eval_error(&sol.tree, &train, cfg.metric);
        assert_eq!(
            sol.error.to_bits(),
            full.to_bits(),
            "every archived error must be the full-training error"
        );
    }
    // The best archived solution is the number the report would quote; it is
    // full-data even though no generation scored more than 25 rows.
    let best = out
        .archive
        .items
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
        .unwrap();
    assert_eq!(
        best.error.to_bits(),
        search::eval_error(&best.tree, &train, Metric::Absolute).to_bits()
    );
}

#[test]
fn predictor_samples_below_two_is_a_usage_error() {
    for bad in ["0", "1", "-3", "abc"] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args(["--data", "/nonexistent", "--target", "y"])
            .arg("--predictor-samples")
            .arg(bad)
            .output()
            .expect("run rcsr");
        assert_eq!(
            out.status.code(),
            Some(2),
            "--predictor-samples {} must be a usage error, got {:?}",
            bad,
            out.status
        );
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("--predictor-samples"),
            "the error must name the flag: {}",
            err
        );
    }
}
