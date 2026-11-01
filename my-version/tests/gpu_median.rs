//! #226: median absolute error fitting on the GPU.
//!
//! The device-level selection tests live in `src/gpu.rs` (they need the
//! crate-private fit plans and the raw median stage). These integration tests
//! pin the search contract:
//!
//! * from `MEDIAN_GPU_MIN_ROWS` (#288) `--metric median` is **scored** on the
//!   device by the exact `f64` order statistic, so the backend is `gpu`; below
//!   the floor (or with `--fit-strategy exact` on a small sample) the CPU scores
//!   exactly and the reason is named. The population is never ranked by MSE;
//! * `--fit-strategy gpu` additionally moves constants through the exact median
//!   polish (fit-only below the scoring floor);
//! * reported errors are the exact CPU median of the stored tree;
//! * fixed seed/device is deterministic, and `--no-fit` is a true no-op;
//! * every fallback is labelled and the ledger adds up;
//! * the measured cost model keeps a small sample on the CPU with a reason.
//!
//! Device-dependent tests skip cleanly on a machine without CUDA.

use rcsr::checkpoint::Backend;
use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};
#[cfg(feature = "cuda")]
use rcsr::search::Outcome;

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        // Exact duplicate targets and gross outliers on purpose, so ties in
        // `|r|` and the median's rank structure are both exercised.
        let v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
        y.push(if i % 7 == 3 { 0.0 } else if i % 97 == 5 { 12.0 } else { v });
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

#[cfg(feature = "cuda")]
fn median_config(seed: u32) -> Config {
    Config {
        generations: 3,
        islands: 1,
        pop: 12,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Median,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

#[cfg(feature = "cuda")]
fn describe(out: &Outcome) -> Vec<(u64, usize, String)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.complexity,
                rcsr::expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]),
            )
        })
        .collect()
}

/// #288: below the measured scoring floor the exact CPU median runs; at or
/// above it the exact device order statistic scores the population even with
/// the default `exact` fit strategy.
#[test]
fn median_gpu_scoring_respects_the_measured_row_floor() {
    let small = sample(96, 3);
    let cfg = Config {
        gpu: true,
        metric: Metric::Median,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &small), Backend::Cpu);
    let cfg = Config {
        gpu: true,
        metric: Metric::Median,
        fit_strategy: FitStrategy::Gpu,
        fit_iters: 4,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &small), Backend::Cpu);
    let cfg = Config {
        fit_iters: 0,
        ..cfg
    };
    assert_eq!(search::effective_backend(&cfg, &small), Backend::Cpu);
    let big = sample(rcsr::fit::MEDIAN_GPU_MIN_ROWS + 32, 3);
    let cfg = Config {
        gpu: true,
        metric: Metric::Median,
        ..Default::default()
    };
    let expected = if cfg!(feature = "cuda") {
        Backend::Gpu
    } else {
        Backend::Cpu
    };
    assert_eq!(search::effective_backend(&cfg, &big), expected);
}

/// #226: the cost model keeps a small sample on the exact CPU median arm and
/// records the labelled reason; nothing is silently fitted on the device.
#[cfg(feature = "cuda")]
#[test]
fn a_small_sample_keeps_the_labelled_cpu_median_arm() {
    let train = sample(2_000, 4);
    let test = sample(200, 5);
    assert!(
        train.len() < rcsr::fit::MEDIAN_GPU_MIN_ROWS,
        "the fixture must sit below the measured crossover"
    );
    let out = search::run(&median_config(7), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Cpu);
    assert!(
        out.gpu.is_none(),
        "below the crossover the device fit arm must not be created"
    );
    let why = out.backend_reason.unwrap_or_default();
    assert!(
        why.contains("median fit arm")
            || why.contains("not scored natively")
            || why.contains("4096-row device floor"),
        "the CPU decision must be labelled: {why:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_median_run_is_deterministic_and_reports_the_exact_objective() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let n = rcsr::fit::MEDIAN_GPU_MIN_ROWS + 64;
    let train = sample(n, 5);
    let test = sample(300, 6);
    let a = search::run(&median_config(11), &train, &test);
    let b = search::run(&median_config(11), &train, &test);
    // #288: the device moves constants *and* scores the exact median, so the
    // backend is `gpu`; the archive is still rebuilt in CPU f64.
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
    assert!(a.device.is_some(), "the device scored the population");
    assert_eq!(describe(&a), describe(&b), "repeated seed differs");
    assert_eq!(a.test_errors, b.test_errors);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the median fit arm must run on the device: {:?}",
        g.fit_fallback_by_cause
    );
    // The reported error of every archive row is the exact CPU median of the
    // stored tree, never an MSE proxy.
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Median);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU median"
        );
    }
    // Ledger: every admitted fit is one GPU success or one labelled cause.
    assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback, "{:?}", g);
    let known: Vec<&str> = search::FitFallbackCause::ALL
        .iter()
        .map(|c| c.label())
        .collect();
    let mut labelled = 0usize;
    for (cause, count) in &g.fit_fallback_by_cause {
        assert!(known.contains(&cause.as_str()), "unknown cause {cause:?}");
        labelled += count;
    }
    assert_eq!(labelled, g.fit_fallback, "{:?}", g);
}

#[cfg(feature = "cuda")]
#[test]
fn no_fit_is_a_true_noop_for_the_median_gpu_arm() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    // Below the #288 scoring floor: with `--no-fit` there is neither a fit arm
    // nor a native scorer, so the exact CPU median is the only arm.
    let train = sample(rcsr::fit::MEDIAN_GPU_MIN_ROWS - 32, 7);
    let test = sample(200, 8);
    let base = Config {
        fit_iters: 0,
        ..median_config(13)
    };
    let gpu = search::run(
        &Config {
            fit_strategy: FitStrategy::Gpu,
            gpu: true,
            ..base.clone()
        },
        &train,
        &test,
    );
    let cpu = search::run(
        &Config {
            fit_strategy: FitStrategy::Exact,
            gpu: false,
            ..base
        },
        &train,
        &test,
    );
    // `--no-fit` means "do not move constants": both arms must produce the
    // same archive, bit for bit, and the device must not be entered at all.
    assert_eq!(describe(&gpu), describe(&cpu));
    assert!(gpu.gpu.is_none(), "--no-fit must not spin up the fit evaluator");
    assert_eq!(gpu.backend, rcsr::backend::Kind::Cpu);
}
