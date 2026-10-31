//! #232: interquartile absolute error fitting on the GPU.
//!
//! The device-level selection tests live in `src/gpu.rs` (they need the
//! crate-private fit plans and the raw order-statistic stage) and in
//! `src/fit/interquartile.rs` (the ladder and the rank window). These
//! integration tests pin the search contract:
//!
//! * the exact window mean also **scores** the population on the device from
//!   `INTERQUARTILE_GPU_MIN_ROWS` (#288), so the backend is `gpu` at or above
//!   the floor; below it the exact CPU window mean runs with a named reason,
//!   and the population is never ranked by SSE;
//! * `--fit-strategy gpu` additionally moves constants through the exact polish;
//! * reported errors are the exact CPU window mean of the stored tree;
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
        // `|r|` and the rank structure of the window are both exercised.
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
fn iqr_config(seed: u32) -> Config {
    Config {
        generations: 3,
        islands: 1,
        pop: 12,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Interquartile,
        gpu: true,
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

/// #232/#288: an explicit `--fit-strategy gpu` with `--metric interquartile`
/// is a valid combination, and the metric now has a native *scoring* entry
/// point as well as the fit-only arm.
#[test]
fn an_explicit_gpu_strategy_may_fit_interquartile() {
    assert!(
        rcsr::fit::gpu_fit_metric_error(Metric::Interquartile).is_none(),
        "the interquartile arm must be an accepted GPU fit"
    );
    assert!(rcsr::fit::interquartile_fit_only(Metric::Interquartile));
    assert!(rcsr::fit::order_stat_fit_only(Metric::Interquartile));
    assert!(!rcsr::fit::interquartile_fit_only(Metric::Median));
    // The objective-loss table still has no interquartile entry: the exact
    // order-statistic batch scorer is a separate scoring shape (#288), so a
    // run is never ranked by squared error.
    assert_eq!(
        rcsr::fit::DeviceLoss::for_metric(Metric::Interquartile),
        None
    );
}

/// #288: below the scoring floor the exact CPU window mean runs; above it the
/// device statistic scores the population.
#[test]
fn interquartile_gpu_scoring_respects_the_measured_row_floor() {
    let s = sample(96, 3);
    let cfg = Config {
        gpu: true,
        metric: Metric::Interquartile,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    let cfg = Config {
        gpu: true,
        metric: Metric::Interquartile,
        fit_strategy: FitStrategy::Gpu,
        fit_iters: 4,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    let cfg = Config {
        fit_iters: 0,
        ..cfg
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    let big = sample(rcsr::fit::INTERQUARTILE_GPU_MIN_ROWS + 32, 3);
    let cfg = Config {
        gpu: true,
        metric: Metric::Interquartile,
        ..Default::default()
    };
    let expected = if cfg!(feature = "cuda") {
        Backend::Gpu
    } else {
        Backend::Cpu
    };
    assert_eq!(search::effective_backend(&cfg, &big), expected);
}

/// #232: the cost model keeps a small sample on the exact CPU interquartile
/// arm and records the labelled reason; nothing is silently fitted on the
/// device.
#[cfg(feature = "cuda")]
#[test]
fn a_small_sample_keeps_the_labelled_cpu_interquartile_arm() {
    let train = sample(2_000, 4);
    let test = sample(200, 5);
    assert!(
        train.len() < rcsr::fit::INTERQUARTILE_GPU_MIN_ROWS,
        "the fixture must sit below the measured crossover"
    );
    let out = search::run(&iqr_config(7), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Cpu);
    assert!(
        out.gpu.is_none(),
        "below the crossover the device fit arm must not be created"
    );
    let why = out.backend_reason.unwrap_or_default();
    assert!(
        why.contains("interquartile fit arm")
            || why.contains("not scored natively")
            || why.contains("4096-row device floor"),
        "the CPU decision must be labelled: {why:?}"
    );
}

/// #232: a full interquartile run is deterministic, moves constants on the
/// device, reports the exact CPU window mean and labels every fallback.
#[cfg(feature = "cuda")]
#[test]
fn gpu_interquartile_run_is_deterministic_and_reports_the_exact_objective() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let n = rcsr::fit::INTERQUARTILE_GPU_MIN_ROWS + 64;
    let train = sample(n, 5);
    let test = sample(300, 6);
    let a = search::run(&iqr_config(11), &train, &test);
    let b = search::run(&iqr_config(11), &train, &test);
    // #288: the device moves constants *and* scores the exact window mean, so
    // the backend is `gpu`; the archive is still rebuilt in CPU f64.
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
    assert!(a.device.is_some(), "the device scored the population");
    assert_eq!(describe(&a), describe(&b), "repeated seed differs");
    assert_eq!(a.test_errors, b.test_errors);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the interquartile fit arm must run on the device: {:?}",
        g.fit_fallback_by_cause
    );
    // The reported error of every archive row is the exact CPU window mean of
    // the stored tree, never an MSE proxy.
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Interquartile);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU interquartile mean"
        );
    }
    // Ledger: every admitted fit is one GPU success or one labelled cause.
    assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback, "{:?}", g);
    let known: Vec<&str> = search::FitFallbackCause::ALL
        .iter()
        .map(|c| c.label())
        .collect();
    assert!(
        known.contains(&"interquartile_declined"),
        "the interquartile decline must have its own label: {known:?}"
    );
    let mut labelled = 0usize;
    for (cause, count) in &g.fit_fallback_by_cause {
        assert!(known.contains(&cause.as_str()), "unknown cause {cause:?}");
        labelled += count;
    }
    assert_eq!(labelled, g.fit_fallback, "{:?}", g);
}

/// #232: `--no-fit` is a true no-op for the interquartile arm: the device is
/// never entered and the archive is identical to the exact CPU run, bit for
/// bit.
#[cfg(feature = "cuda")]
#[test]
fn no_fit_is_a_true_noop_for_the_interquartile_gpu_arm() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    // Below the #288 scoring floor: `--no-fit` leaves neither a fit arm nor a
    // native scorer, so the exact CPU window mean is the only arm.
    let train = sample(rcsr::fit::INTERQUARTILE_GPU_MIN_ROWS - 32, 7);
    let test = sample(200, 8);
    let base = Config {
        fit_iters: 0,
        ..iqr_config(13)
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
    assert_eq!(describe(&gpu), describe(&cpu));
    assert!(gpu.gpu.is_none(), "--no-fit must not spin up the fit evaluator");
    assert_eq!(gpu.backend, rcsr::backend::Kind::Cpu);
    // And the run must not blame the device for a fit the user switched off:
    // `--no-fit` has nothing to fit, so the refusal label is the scoring gate,
    // never "no usable device".
    let why = gpu.backend_reason.unwrap_or_default();
    assert!(
        !why.contains("no usable device"),
        "--no-fit must not report a missing device: {why:?}"
    );
}
