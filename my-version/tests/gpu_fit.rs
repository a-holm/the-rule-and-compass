//! #169: batched GPU constant fitting with exact CPU f64 verification.
//!
//! The kernel tests live in `src/gpu.rs` (they can reach the crate-private fit
//! plans). These integration tests pin the properties the search itself must
//! keep: the opt-in strategy, determinism, exact reporting and checkpoint
//! round-trips. Device-dependent tests skip cleanly on a machine without CUDA.

use rcsr::checkpoint::{Backend, Checkpoint};
use rcsr::data::Sample;
use rcsr::expr;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2 + (((i * 37) % 11) as f64 - 5.0) * 1e-3);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn gpu_config(seed: u32) -> Config {
    Config {
        generations: 5,
        islands: 1,
        pop: 24,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Squared,
        gpu: true,
        ..Default::default()
    }
}

/// #236 / #229 / #287 / #288: a 64-row sample is below every measured native
/// scoring floor, so the exact order/rank family keeps the CPU here, and a
/// metric the table refuses is never handed to the evaluator at any size.
#[test]
fn metrics_without_an_admitted_scorer_never_use_the_gpu_evaluator() {
    let s = sample(64, 7);
    let metrics = [
        Metric::Median,
        Metric::RankCorrelation,
        Metric::RankCorrelationSigned,
        Metric::Interquartile,
        Metric::Auc,
        Metric::Slope,
        Metric::ImplicitDerivative,
    ];
    for metric in metrics {
        let cfg = Config {
            gpu: true,
            metric,
            ..Default::default()
        };
        assert_eq!(
            search::effective_backend(&cfg, &s),
            Backend::Cpu,
            "{metric:?} must not be scored by a GPU evaluator without its objective"
        );
        match rcsr::metrics::ScoreStat::for_metric(metric) {
            Some(stat) => assert!(
                s.len() < stat.min_rows(),
                "{metric:?} is on the CPU here only because of its row floor"
            ),
            None => assert!(
                rcsr::metrics::score_refusal(metric).is_some(),
                "{metric:?} must carry a named score refusal"
            ),
        }
    }
}

/// #287: every native score objective is in the table, independent of whether
/// this build has a CUDA device. The device-side arithmetic is covered by the
/// `src/gpu.rs` tests.
#[test]
fn native_score_objectives_are_table_driven() {
    for metric in [
        Metric::Squared,
        Metric::Absolute,
        Metric::ImplicitResidual,
        Metric::LogSquashed,
        Metric::Signed,
        Metric::Maximum,
        Metric::Correlation,
        Metric::CorrelationSigned,
        Metric::Aic,
        Metric::Aicc,
        Metric::Bic,
        Metric::LogLoss,
        Metric::Brier,
        Metric::Hinge,
        Metric::HingeOrdinary,
        Metric::HingeSquared,
    ] {
        assert!(
            rcsr::metrics::native_score_objective(metric).is_some(),
            "{metric:?} must have a native score objective"
        );
    }
}

/// #229: `log_squashed` has a native device objective, so the fit kernel may
/// minimise it. The evaluator gate itself is CUDA-only and tested below.
#[test]
fn log_squashed_has_a_native_objective() {
    assert_eq!(
        rcsr::fit::DeviceLoss::for_metric(Metric::LogSquashed),
        Some(rcsr::fit::DeviceLoss::LogSquashed)
    );
    assert_eq!(
        rcsr::fit::fit_loss(Metric::LogSquashed),
        Some(rcsr::fit::DeviceLoss::LogSquashed)
    );
    // The information criteria fit by least squares and stay on the squared
    // *fit* objective; their native score is the separate AIC-family reducer
    // (#287).
    assert_eq!(
        rcsr::fit::fit_loss(Metric::Aic),
        Some(rcsr::fit::DeviceLoss::Squared)
    );
    assert_eq!(
        rcsr::fit::DeviceLoss::for_metric(Metric::Aic),
        None,
        "AIC is not a squared objective for ranking"
    );
    // Weighted rows are scored natively on the device (the evaluator holds the
    // per-row weights).
    #[cfg(feature = "cuda")]
    {
        let mut weighted = sample(64, 7);
        weighted.w = vec![1.0; weighted.len()];
        let wcfg = Config {
            gpu: true,
            metric: Metric::LogSquashed,
            ..Default::default()
        };
        assert_eq!(search::backend_blocker_reason(&wcfg, &weighted), None);
    }
}

/// #222: `absolute` has a native `Σ|r|` fit and scoring objective with its own
/// code, so a squared-only backend can never answer a MAE request.
#[test]
fn absolute_has_a_native_objective() {
    assert_eq!(
        rcsr::fit::DeviceLoss::for_metric(Metric::Absolute),
        Some(rcsr::fit::DeviceLoss::Absolute)
    );
    assert_eq!(
        rcsr::fit::fit_loss(Metric::Absolute),
        Some(rcsr::fit::DeviceLoss::Absolute)
    );
    assert_ne!(
        rcsr::fit::DeviceLoss::Absolute.code(),
        rcsr::fit::DeviceLoss::Squared.code()
    );
    assert!(
        rcsr::fit::gpu_fit_metric_error(Metric::Absolute).is_none(),
        "an explicit GPU strategy may fit absolute"
    );
    #[cfg(feature = "cuda")]
    {
        let mut weighted = sample(64, 7);
        weighted.w = vec![1.0; weighted.len()];
        let wcfg = Config {
            gpu: true,
            metric: Metric::Absolute,
            ..Default::default()
        };
        assert_eq!(search::backend_blocker_reason(&wcfg, &weighted), None);
    }
}

/// #222: with the native MAE scorer, an `absolute` run is admitted to the
/// device; it was refused before because the evaluator would rank it by MSE.
#[cfg(feature = "cuda")]
#[test]
fn absolute_is_admitted_to_the_gpu_evaluator() {
    let s = sample(64, 7);
    let cfg = Config {
        gpu: true,
        metric: Metric::Absolute,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Gpu);
}

/// #229: with the native `Σ ln(1+r²)` scorer in the evaluator, a
/// `log_squashed` run is admitted to the device; before #229 it was refused
/// because the evaluator would have ranked it by MSE.
#[cfg(feature = "cuda")]
#[test]
fn log_squashed_is_admitted_to_the_gpu_evaluator() {
    let s = sample(64, 7);
    let cfg = Config {
        gpu: true,
        metric: Metric::LogSquashed,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Gpu);
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
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

/// #169: the admission ledger must be internally exact and use only the
/// documented fallback labels.
#[cfg(feature = "cuda")]
fn assert_fit_ledger(g: &rcsr::search::GpuRunStats) {
    assert!(
        g.fit_screened >= g.fit_attempts,
        "admission cannot fit more candidates than it screened: {:?}",
        g
    );
    assert_eq!(
        g.fit_attempts,
        g.fit_gpu + g.fit_fallback,
        "every admitted fit is exactly one GPU success or one labelled fallback: {:?}",
        g
    );
    let known: Vec<&str> = rcsr::search::FitFallbackCause::ALL
        .iter()
        .map(|c| c.label())
        .collect();
    let mut labelled = 0usize;
    for (cause, n) in &g.fit_fallback_by_cause {
        assert!(
            known.contains(&cause.as_str()),
            "unknown fallback cause {:?} in {:?}",
            cause,
            g
        );
        labelled += n;
    }
    assert_eq!(
        labelled, g.fit_fallback,
        "fallback causes must add up to the total: {:?}",
        g
    );
    // #284: the same counters and the raw kernel/fallback seconds are in the
    // run ledger, which must also close and never report duty > 1.
    assert_eq!(g.ledger.admitted() as usize, g.fit_attempts, "{:?}", g.ledger);
    assert_eq!(g.ledger.device_fits() as usize, g.fit_gpu, "{:?}", g.ledger);
    assert_eq!(g.ledger.fallback_fits() as usize, g.fit_fallback, "{:?}", g.ledger);
    assert_eq!(
        g.ledger.fallback_by_cause(),
        &g.fit_fallback_by_cause,
        "the ledger must carry every labelled cause"
    );
    assert!(g.ledger.wall_s() > 0.0);
    assert!(
        g.ledger.closure_residual().value.abs() < 0.01,
        "run ledger does not close: {:?}",
        g.ledger
    );
    assert!(g.ledger.duty_run().value <= 1.0);
    assert!(g.ledger.device_busy_s() <= g.ledger.device_busy_sum_s() + 1e-12);
    if g.fit_gpu > 0 {
        assert!(g.ledger.fit_kernel_s() > 0.0, "a device fit must be recorded");
        assert!(g.ledger.fit_host_s() >= g.ledger.fit_kernel_s());
        assert!(g.ledger.fit_host_s() >= g.ledger.fit_fallback_s());
    }
}

#[test]
fn exact_is_still_the_default_and_unchanged() {
    assert_eq!(Config::default().fit_strategy, FitStrategy::Exact);
    let train = sample(160, 1);
    let test = sample(40, 2);
    let mut base = Config {
        generations: 3,
        islands: 1,
        pop: 16,
        max_complexity: 10,
        fit_iters: 3,
        patience: 0,
        seed: 3,
        gpu: false,
        ..Default::default()
    };
    base.fit_strategy = FitStrategy::Exact;
    let a = search::run(&base, &train, &test);
    // The default is `exact`: an explicit exact must be bit-identical.
    let mut default = base.clone();
    default.fit_strategy = Config::default().fit_strategy;
    let b = search::run(&default, &train, &test);
    assert_eq!(describe(&a), describe(&b));
}

#[test]
fn checkpoint_round_trip_preserves_the_gpu_strategy() {
    let train = sample(160, 1);
    let test = sample(40, 2);
    let cfg = Config {
        generations: 2,
        islands: 1,
        pop: 12,
        max_complexity: 8,
        fit_iters: 2,
        patience: 0,
        seed: 4,
        fit_strategy: FitStrategy::Gpu,
        gpu: false, // strategy round-trip does not need a device
        ..Default::default()
    };
    let mut text: Option<String> = None;
    search::run_resumable(&cfg, &train, &test, None, 0, |cp| {
        text = Some(cp.to_text());
    })
    .expect("fresh run");
    let text = text.expect("a checkpoint callback");
    let cp = Checkpoint::from_text(&text).expect("parse checkpoint");
    assert_eq!(cp.config.fit_strategy, FitStrategy::Gpu);
    let again = Checkpoint::from_text(&cp.to_text()).expect("reparse checkpoint");
    assert_eq!(again.config.fit_strategy, FitStrategy::Gpu);
    assert_eq!(
        rcsr::checkpoint::config_hash(&again.config),
        rcsr::checkpoint::config_hash(&cfg)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_fit_is_deterministic_for_repeated_seeds_and_island_counts() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 5);
    let test = sample(60, 6);
    for islands in [1usize, 4] {
        let cfg = Config {
            islands,
            pop: 24,
            ..gpu_config(11)
        };
        let a = search::run(&cfg, &train, &test);
        let b = search::run(&cfg, &train, &test);
        assert_eq!(describe(&a), describe(&b), "islands={}", islands);
        assert_eq!(a.test_errors, b.test_errors);
        assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
        let g = a.gpu.as_ref().expect("a GPU report");
        assert!(g.fit_gpu > 0, "the fit kernel must have run: {:?}", g);
        assert_fit_ledger(g);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn reported_finalists_are_refit_exactly_on_the_cpu() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 7);
    let test = sample(60, 8);
    let out = search::run(&gpu_config(13), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert!(g.fit_gpu > 0, "batched GPU fit ran: {:?}", g);
    assert!(g.fit_refits > 0, "finalists were refit: {:?}", g);
    assert_fit_ledger(g);
    // Every reported training error is the exact CPU f64 score of the stored
    // tree, bit for bit (the archive was rebuilt after the exact refit).
    for s in &out.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Squared);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU score: {} vs {}",
            s.error,
            cpu
        );
    }
}

/// #287 AC6: for every metric the device scores natively, the reported score of
/// each finalist after the exact refit is the CPU `score_weighted` of the stored
/// tree on the same data, bit for bit. Metrics the device refuses are not run.
#[cfg(feature = "cuda")]
#[test]
fn reported_finalists_equal_cpu_score_for_every_native_metric() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let metrics = [
        Metric::Squared,
        Metric::Absolute,
        Metric::LogSquashed,
        Metric::Signed,
        Metric::Maximum,
        Metric::Correlation,
        Metric::CorrelationSigned,
        Metric::Aic,
        Metric::Aicc,
        Metric::Bic,
        Metric::LogLoss,
        Metric::Brier,
        Metric::Hinge,
        Metric::HingeOrdinary,
        Metric::HingeSquared,
    ];
    for metric in metrics {
        let objective = rcsr::metrics::native_score_objective(metric)
            .unwrap_or_else(|| panic!("{metric:?} has no native objective"));
        let mut train = sample(600, 7);
        let mut test = sample(100, 8);
        if objective.is_classification() {
            for s in [&mut train, &mut test] {
                for y in s.y.iter_mut() {
                    *y = if *y > 0.0 { 1.0 } else { 0.0 };
                }
            }
        }
        let cfg = Config {
            metric,
            ..gpu_config(13)
        };
        let out = search::run(&cfg, &train, &test);
        assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{metric:?} backend");
        let g = out.gpu.as_ref().expect("a GPU report");
        assert!(g.fit_refits > 0, "{metric:?}: finalists were refit: {:?}", g);
        assert_fit_ledger(g);
        assert!(!out.archive.items.is_empty(), "{metric:?}: empty archive");
        for s in &out.archive.items {
            let pred: Vec<f64> = train.x.iter().map(|row| expr::eval(&s.tree, row)).collect();
            let cpu = rcsr::metrics::score_weighted(
                &pred,
                &train.y,
                &train.w,
                metric,
                expr::count_constants(&s.tree),
            );
            assert_eq!(
                s.error.to_bits(),
                cpu.to_bits(),
                "{metric:?}: archive row is not the exact CPU score: {} vs {}",
                s.error,
                cpu
            );
        }
    }
}

/// Weighted regression on the device: every native regression metric with a
/// weighted form is admitted, `maximum` keeps a labelled refusal, and a metric
/// without a native scorer keeps its own refusal.
#[cfg(feature = "cuda")]
#[test]
fn the_device_gate_admits_weighted_regression_metrics_with_a_weighted_form() {
    let mut s = sample(64, 7);
    s.w = (0..s.len()).map(|i| 0.5 + (i % 5) as f64).collect();
    let cfg = |m: Metric| Config {
        gpu: true,
        metric: m,
        ..Default::default()
    };
    for m in [
        Metric::Squared,
        Metric::Absolute,
        Metric::LogSquashed,
        Metric::Signed,
        Metric::Correlation,
        Metric::CorrelationSigned,
        Metric::Aic,
        Metric::Aicc,
        Metric::Bic,
    ] {
        assert!(rcsr::metrics::weights_supported(m), "{m:?}");
        assert_eq!(search::backend_blocker_reason(&cfg(m), &s), None, "{m:?}");
    }
    let why = search::backend_blocker_reason(&cfg(Metric::Maximum), &s).expect("maximum is refused");
    assert!(why.contains("maximum"), "{why}");
    assert!(!rcsr::metrics::weights_supported(Metric::Maximum));
}

/// Weighted regression: the reported score of each finalist after the exact
/// refit is the CPU `score_weighted` of the stored tree under the same
/// weights, bit for bit, for every admitted metric. The run claims the device.
#[cfg(feature = "cuda")]
#[test]
fn weighted_reported_finalists_equal_cpu_score_for_every_admitted_metric() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let metrics = [
        Metric::Squared,
        Metric::Absolute,
        Metric::LogSquashed,
        Metric::Signed,
        Metric::Correlation,
        Metric::CorrelationSigned,
        Metric::Aic,
        Metric::Aicc,
        Metric::Bic,
    ];
    let weights = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| if i % 7 == 0 { 0.0 } else { 0.5 + (i % 5) as f64 * 0.75 })
            .collect()
    };
    for metric in metrics {
        let mut train = sample(600, 7);
        let mut test = sample(100, 8);
        train.w = weights(train.len());
        test.w = weights(test.len());
        let cfg = Config {
            metric,
            ..gpu_config(13)
        };
        let out = search::run(&cfg, &train, &test);
        assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{metric:?} backend");
        let g = out.gpu.as_ref().expect("a GPU report");
        assert!(g.fit_refits > 0, "{metric:?}: finalists were refit: {:?}", g);
        assert_fit_ledger(g);
        assert!(!out.archive.items.is_empty(), "{metric:?}: empty archive");
        for s in &out.archive.items {
            let pred: Vec<f64> = train.x.iter().map(|row| expr::eval(&s.tree, row)).collect();
            let cpu = rcsr::metrics::score_weighted(
                &pred,
                &train.y,
                &train.w,
                metric,
                expr::count_constants(&s.tree),
            );
            assert_eq!(
                s.error.to_bits(),
                cpu.to_bits(),
                "{metric:?}: archive row is not the exact weighted CPU score: {} vs {}",
                s.error,
                cpu
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn unsupported_trees_fall_back_and_the_kernel_is_not_a_noop() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 9);
    let test = sample(60, 10);
    let cfg = gpu_config(17);
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_attempts >= g.fit_gpu,
        "attempts {} < gpu {}",
        g.fit_attempts,
        g.fit_gpu
    );
    assert!(
        g.fit_gpu > 0 || g.fit_fallback > 0,
        "the GPU-fit arm must do work: {:?}",
        g
    );
    assert_fit_ledger(g);
}

#[cfg(feature = "cuda")]
#[test]
fn admission_fits_only_the_promoted_fraction() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 31);
    let test = sample(60, 32);
    let cfg = Config {
        generations: 2,
        islands: 1,
        pop: 64,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed: 33,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Squared,
        gpu: true,
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_screened > g.fit_attempts,
        "admission must screen more offspring than it fits: {:?}",
        g
    );
    assert_fit_ledger(g);
}

/// #229: rows with gross outliers, so `log_squashed` and `squared` disagree
/// about which model is good.
fn sample_with_outliers(n: usize, seed: u32) -> Sample {
    let mut s = sample(n, seed);
    for i in (4..n).step_by(10) {
        s.y[i] += 30.0 * (1.0 + (i % 3) as f64);
    }
    s
}

/// #229: a `log_squashed` run uses the batched IRLS fit kernel, ranks on
/// `mean ln(1+r²)` natively, is bit-deterministic for a fixed seed, and every
/// reported number is the exact CPU `f64` score of the stored tree.
#[cfg(feature = "cuda")]
#[test]
fn log_squashed_gpu_fit_runs_natively_and_reports_exact_f64() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample_with_outliers(240, 41);
    let test = sample(60, 42);
    let cfg = Config {
        generations: 4,
        metric: Metric::LogSquashed,
        ..gpu_config(43)
    };
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b), "a fixed seed must be deterministic");
    assert_eq!(a.test_errors, b.test_errors);
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the log-squashed fit kernel must actually run: {:?}",
        g
    );
    assert_fit_ledger(g);
    // Reported train errors are the exact CPU f64 configured loss, bit for bit.
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::LogSquashed);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU log_squashed score: {} vs {}",
            s.error,
            cpu
        );
        assert!(s.error.is_finite());
    }
    // The archive is rebuilt and res-cored on the CPU in f64; a robust run is
    // therefore never reported from f32 kernel values.
    assert!(a
        .archive
        .items
        .iter()
        .all(|s| s.error.is_finite() && s.cost >= 1.0));
}

/// #229: `--no-fit` (`--fit-iters 0`) must not dispatch the fit kernel at all,
/// for a robust run exactly as for a squared one.
#[cfg(feature = "cuda")]
#[test]
fn log_squashed_gpu_no_fit_is_a_true_noop() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample_with_outliers(240, 51);
    let test = sample(60, 52);
    let cfg = Config {
        fit_iters: 0,
        metric: Metric::LogSquashed,
        ..gpu_config(53)
    };
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert_eq!(g.fit_attempts, 0, "no fit may be attempted: {:?}", g);
    assert_eq!(g.fit_gpu, 0);
    assert_eq!(g.fit_screened, 0, "admission must not run without a fit: {:?}", g);
    for s in &out.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::LogSquashed);
        assert_eq!(s.error.to_bits(), cpu.to_bits());
    }
}

/// #222: an `absolute` run uses the batched L1/IRLS fit kernel, ranks on
/// `mean |r|` natively, is bit-deterministic for a fixed seed, and every
/// reported number is the exact CPU `f64` MAE of the stored tree.
#[cfg(feature = "cuda")]
#[test]
fn absolute_gpu_fit_runs_natively_and_reports_exact_f64() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample_with_outliers(240, 61);
    let test = sample(60, 62);
    let cfg = Config {
        generations: 4,
        metric: Metric::Absolute,
        ..gpu_config(63)
    };
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b), "a fixed seed must be deterministic");
    assert_eq!(a.test_errors, b.test_errors);
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the absolute fit kernel must actually run: {:?}",
        g
    );
    assert_fit_ledger(g);
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Absolute);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU MAE: {} vs {}",
            s.error,
            cpu
        );
        assert!(s.error.is_finite());
    }
}

/// #222: `--no-fit` (`--fit-iters 0`) must not dispatch the L1 fit kernel.
#[cfg(feature = "cuda")]
#[test]
fn absolute_gpu_no_fit_is_a_true_noop() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample_with_outliers(240, 71);
    let test = sample(60, 72);
    let cfg = Config {
        fit_iters: 0,
        metric: Metric::Absolute,
        ..gpu_config(73)
    };
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert_eq!(g.fit_attempts, 0, "no fit may be attempted: {:?}", g);
    assert_eq!(g.fit_gpu, 0);
    assert_eq!(g.fit_screened, 0, "admission must not run without a fit: {:?}", g);
    for s in &out.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Absolute);
        assert_eq!(s.error.to_bits(), cpu.to_bits());
    }
}
/// #228: the CUDA LM kernel is a warm start for `--metric signed`; the exact
/// `f64` constrained-LS projection completed on the host is what decides. The
/// arm must reach the same constants as the CPU arm (both are on the
/// zero-bias manifold) and must be deterministic.
///
/// A projection moves constants onto `Σ wᵢ rᵢ = 0` and therefore *raises* SSE,
/// so this also proves the gate is metric-aware: a squared-SSE gate would
/// reject the projection by construction.
#[cfg(feature = "cuda")]
#[test]
fn signed_arm_projects_the_device_proposal_exactly() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    use rcsr::fit::{fit_batch_verified, fit_weighted_metric, BatchStatus, BatchedGpu};
    use rcsr::metrics::score_weighted;

    // Two variables with an intercept: the squared LM gets close, the
    // projection finishes the job in exact f64.
    let s = sample(512, 5);
    let x = s.x.clone();
    let y = s.y.clone();
    let n = y.len();
    let nv = x[0].len();
    let cols: Vec<f32> = (0..nv)
        .flat_map(|v| x.iter().map(move |r| r[v] as f32))
        .collect();
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut cols64 = vec![0.0f64; n * nv];
    for (r, row) in x.iter().enumerate() {
        for (v, &val) in row.iter().enumerate().take(nv) {
            cols64[v * n + r] = val;
        }
    }
    let mut ev = match rcsr::gpu::GpuEvaluator::new(&cols, n, nv, &y32) {
        Some(ev) => ev,
        None => {
            eprintln!("skipping GPU test: evaluator construction failed");
            return;
        }
    };
    assert!(
        ev.set_fit_data(&cols64, &y, &[]),
        "f64 fit rows uploaded"
    );

    let model = || {
        rcsr::expr::node(
            rcsr::expr::Op::Add,
            vec![
                rcsr::expr::node(
                    rcsr::expr::Op::Mul,
                    vec![rcsr::expr::constant(1.0), rcsr::expr::input(0)],
                ),
                rcsr::expr::constant(0.0),
            ],
        )
    };
    let objective = |t: &rcsr::expr::Node| {
        let p: Vec<f64> = x.iter().map(|row| rcsr::expr::eval(t, row)).collect();
        score_weighted(&p, &y, &[], Metric::Signed, rcsr::expr::count_constants(t))
    };
    let constants = |t: &rcsr::expr::Node| {
        let mut out = Vec::new();
        fn walk(n: &rcsr::expr::Node, out: &mut Vec<f64>) {
            if n.op == rcsr::expr::Op::Const {
                out.push(n.val);
            }
            for k in &n.kids {
                walk(k, out);
            }
        }
        walk(t, &mut out);
        out
    };

    let mut batch = vec![model(), model()];
    let backend = BatchedGpu::new(&mut ev);
    let out = fit_batch_verified(&backend, &mut batch, &x, &y, &[], 8, Metric::Signed);
    assert_eq!(
        out.status,
        vec![BatchStatus::Gpu, BatchStatus::Gpu],
        "the device proposal must pass the signed gate"
    );
    // Each reported objective is the exact metric value at the final constants,
    // at the f64 rounding scale of the bias: the intercept makes the optimum
    // exactly 0 in exact arithmetic, not bit-identically in f64.
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for (i, t) in batch.iter().enumerate() {
        assert_eq!(out.score[i].to_bits(), objective(t).to_bits());
        assert!(
            out.score[i] <= 1e-14 * scale,
            "bias {:e} must be at the rounding scale",
            out.score[i]
        );
    }
    // Determinism: a second batch over fresh trees yields identical bits.
    let mut again = vec![model(), model()];
    let backend = BatchedGpu::new(&mut ev);
    let out2 = fit_batch_verified(&backend, &mut again, &x, &y, &[], 8, Metric::Signed);
    assert_eq!(
        out.score.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out2.score.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(constants(&batch[0]), constants(&again[0]));

    // CPU parity: the exact arm from the same start reaches the same objective
    // (the two constant sets may differ by f32 row rounding of the warm start,
    // which the exact projection then removes).
    let mut cpu = model();
    let cpu_value = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Signed);
    assert!(cpu_value <= 1e-14 * scale, "cpu bias {cpu_value:e}");
    let (a, b) = (constants(&batch[0]), constants(&cpu));
    for (ca, cb) in a.iter().zip(&b) {
        assert!(
            (ca - cb).abs() <= 1e-5 * (1.0 + cb.abs()),
            "constant {ca} vs CPU {cb}"
        );
    }
}

/// #169 measurement (ignored): end-to-end wall time of a fit-heavy
/// configuration on the CPU exact arm versus the GPU-fit arm.
/// Run with `cargo test --release --features cuda --test gpu_fit bench_end_to_end -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
/// #169: normalized 2-D hypervolume of `(cost, error)` points. The extrema are
/// passed in so both arms are measured on the same grid.
fn hypervolume(points: &[(f64, f64)], cmin: f64, cmax: f64, emin: f64, emax: f64) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let cspan = (cmax - cmin).max(1e-12);
    let espan = (emax - emin).max(1e-12);
    const G: usize = 64;
    let normalized: Vec<(f64, f64)> = points
        .iter()
        .map(|&(c, e)| ((c - cmin) / cspan, (e - emin) / espan))
        .collect();
    let mut dominated = 0usize;
    for gy in 0..G {
        let y = (gy as f64 + 0.5) / G as f64;
        for gx in 0..G {
            let x = (gx as f64 + 0.5) / G as f64;
            if normalized.iter().any(|&(px, py)| px <= x && py <= y) {
                dominated += 1;
            }
        }
    }
    dominated as f64 / (G * G) as f64
}

fn front_points(out: &Outcome) -> Vec<(f64, f64)> {
    out.archive
        .items
        .iter()
        .filter(|s| s.error.is_finite())
        .map(|s| (s.cost, s.error))
        .collect()
}

/// #169 measurement (ignored): the pinned fit-heavy 154k cell, equal
/// evaluations, exact CPU versus GPU admission. Reports end-to-end wall, the
/// fit stage's share, GPU duty, every fallback cause, final best error,
/// hypervolume and generation-to-target.
/// Run with `cargo test --release --features cuda --test gpu_fit bench_end_to_end -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_end_to_end() {
    if !device_available() {
        eprintln!("skipping GPU benchmark: no usable CUDA device");
        return;
    }
    let n = std::env::var("RCSR_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(154_000);
    let pop = std::env::var("RCSR_BENCH_POP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let islands = std::env::var("RCSR_BENCH_ISLANDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let generations = std::env::var("RCSR_BENCH_GENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let fit_iters = std::env::var("RCSR_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    // #229/#222: the same paired measurement for the robust arms. The metric
    // decides both the objective the kernel fits and the one it ranks on.
    let metric = match std::env::var("RCSR_BENCH_METRIC").as_deref() {
        Ok("log_squashed") | Ok("log-squashed") | Ok("log") => Metric::LogSquashed,
        Ok("absolute") | Ok("mae") | Ok("l1") => Metric::Absolute,
        // #232: the fit-only order-statistic arms, so the same paired
        // measurement covers them.
        Ok("median") => Metric::Median,
        Ok("interquartile") | Ok("iqr") | Ok("iqmae") => Metric::Interquartile,
        _ => Metric::Squared,
    };
    let train = if matches!(
        metric,
        Metric::LogSquashed | Metric::Absolute | Metric::Median | Metric::Interquartile
    ) {
        sample_with_outliers(n, 21)
    } else {
        sample(n, 21)
    };
    let test = sample(1_000, 22);
    // The feasibility note's pinned cell: fit-heavy, no screening, a smooth
    // always-finite operator set so most random candidates are eligible for
    // the kernel. Both arms run the same generations, population and islands,
    // so the comparison is at equal evaluations.
    use rcsr::expr::Op;
    let base = Config {
        generations,
        islands,
        pop,
        max_complexity: 14,
        fit_iters,
        metric,
        fit_samples: 0,
        predictor_samples: None,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Sin, Op::Cos, Op::Tanh, Op::Gauss, Op::Square, Op::Cube, Op::Neg],
        seed: 9,
        patience: 0,
        gpu_notes: false,
        ..Default::default()
    };
    // `RCSR_BENCH_ARMS=exact,gpu` (the default) runs both arms; a diagnostic
    // can select one (`exact` or `gpu`) to repeat an arm without paying for
    // the other. The equal-evaluation comparison always uses both.
    let arms = std::env::var("RCSR_BENCH_ARMS").unwrap_or_else(|_| "exact,gpu".to_string());
    let wanted = |name: &str| {
        arms.split(',').map(str::trim).any(|a| {
            a.eq_ignore_ascii_case(name)
                || (a.eq_ignore_ascii_case("exact") && name.starts_with("exact"))
                || (a.eq_ignore_ascii_case("gpu") && name.starts_with("gpu"))
        })
    };
    // (name, strategy, gpu, wall, per-generation best, outcome)
    let mut runs = Vec::new();
    for (name, strategy, gpu) in [
        ("exact/cpu", FitStrategy::Exact, false),
        ("gpu/admission", FitStrategy::Gpu, true),
    ] {
        if !wanted(name) {
            continue;
        }
        let cfg = Config {
            fit_strategy: strategy,
            gpu,
            ..base.clone()
        };
        let t0 = std::time::Instant::now();
        let mut history: Vec<(usize, f64, f64, String)> = Vec::new();
        let out = search::run_resumable(&cfg, &train, &test, None, 1, |cp| {
            let best = cp
                .archive
                .items
                .iter()
                .min_by(|a, b| {
                    a.error
                        .partial_cmp(&b.error)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let best = best.map(|s| s.error).unwrap_or(f64::INFINITY);
            if history.last().map(|h| h.0) != Some(cp.generations) {
                let text = cp
                    .archive
                    .items
                    .iter()
                    .min_by(|a, b| {
                        a.error
                            .partial_cmp(&b.error)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|s| {
                        expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()])
                    })
                    .unwrap_or_default();
                history.push((cp.generations, best, t0.elapsed().as_secs_f64(), text));
            }
        })
        .expect("benchmark search");
        let wall = t0.elapsed().as_secs_f64();
        let best = out
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "arm={} metric={} N={} evals={} wall={:.2}s best={:.6e} archive_hash={:016x}",
            name,
            metric.name(),
            n,
            base.generations * base.islands * base.pop,
            wall,
            best,
            out.archive.hash()
        );
        // Per-generation archive best, so a quality regression can be located
        // in the trajectory rather than only at the end.
        let trajectory: Vec<String> = history
            .iter()
            .map(|(g, b, t, text)| format!("g{}={:.6e}@{:.1}s[{}]", g, b, t, text))
            .collect();
        eprintln!("  history: {}", trajectory.join(" "));
        if let Some(g) = &out.gpu {
            let causes = rcsr::search::FitFallbackCause::ALL
                .iter()
                .map(|c| {
                    format!(
                        "{}={}",
                        c.label(),
                        g.fit_fallback_by_cause.get(c.label()).copied().unwrap_or(0)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let fallback_rate = if g.fit_attempts == 0 {
                0.0
            } else {
                g.fit_fallback as f64 / g.fit_attempts as f64
            };
            eprintln!(
                "  fit: screened={} admitted={} gpu={} fallback={} rate={:.1}% causes=[{}] refits={}",
                g.fit_screened,
                g.fit_attempts,
                g.fit_gpu,
                g.fit_fallback,
                100.0 * fallback_rate,
                causes,
                g.fit_refits
            );
            // #284: a CUDA fit run must leave a fit record in the run ledger,
            // with kernel seconds from raw device events and any CPU fallback
            // seconds measured separately.
            if g.fit_gpu > 0 {
                assert!(
                    g.ledger.fit_kernel_s() > 0.0,
                    "a device fit must record fit kernel seconds: {:?}",
                    g.ledger
                );
                assert_eq!(g.ledger.device_fits(), g.fit_gpu as u64);
                assert_eq!(g.ledger.fallback_fits(), g.fit_fallback as u64);
                assert_eq!(
                    g.ledger.admitted() as usize,
                    g.fit_attempts,
                    "the ledger must count every admitted fit once"
                );
                assert!(g.ledger.fit_host_s() >= g.ledger.fit_kernel_s());
                assert!(g.ledger.fit_host_s() >= g.ledger.fit_fallback_s());
            }
            eprintln!(
"  time: fit_host={:.2}s ({:.1}% of wall) kernel={:.2}s fallback={:.2}s bounded={}/{} workers={} duty={:.1}% run_duty={:.1}%",
                g.fit_host_ms / 1000.0,
                100.0 * (g.fit_host_ms / 1000.0) / wall,
                g.fit_ms / 1000.0,
                g.fit_fallback_ms / 1000.0,
                g.fit_fallback_bounded,
                g.fit_fallback_max,
                g.fit_fallback_workers,
                100.0 * g.accounting.duty_cycle(),
                100.0 * g.ledger.duty_run().value
            );
        }
        runs.push((name, wall, history, out));
    }

    // Equal-evaluation quality control: both arms are normalized on the same
    // (cost, error) extrema. The time-to-target control uses the exact arm's
    // final best × 1.02, so the baseline reaches the target at its last
    // generation by construction and the GPU arm must reach it no later.
    if runs.is_empty() {
        eprintln!("no arm selected by RCSR_BENCH_ARMS={}", arms);
        return;
    }
    let all: Vec<(f64, f64)> = runs.iter().flat_map(|r| front_points(&r.3)).collect();
    let cmin = all.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let cmax = all.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let emin = all.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let emax = all.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let baseline = runs
        .iter()
        .find(|r| r.0.starts_with("exact"))
        .unwrap_or(&runs[0]);
    let exact_best = baseline
        .3
        .archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min);
    let target = exact_best * 1.02;
    eprintln!(
        "control: target={:.6e} ({} final best × 1.02)",
        target,
        if baseline.0.starts_with("exact") {
            "exact"
        } else {
            "this arm's"
        }
    );
    for (name, _, history, out) in &runs {
        let hv = hypervolume(&front_points(out), cmin, cmax, emin, emax);
        let ttt = history
            .iter()
            .find(|h| h.1 <= target)
            .map(|h| format!("gen {} / {:.1}s", h.0, h.2))
            .unwrap_or_else(|| "not reached".to_string());
        eprintln!("control: arm={} hypervolume={:.6} ttt={}", name, hv, ttt);
    }
}

// ---------------------------------------------------------------- #223 ------
//
// The GPU LM kernel minimises one configured objective: least squares, the
// robust `log_squashed` loss, or #222's mean absolute error. An explicit
// `--fit-strategy gpu` with any other metric must be a usage error naming the
// cause and the ways out, never a silent CPU fallback that optimises a
// different objective. These tests run the real binary so the exit code and the
// message are what a user sees.

fn rcsr(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("run rcsr")
}

/// Every metric without a device fit objective is refused with exit 2 and a
/// message that names what the GPU arms minimise and every way out. #229 admits
/// `log_squashed` because the kernel now fits it, #222 admits `absolute`, and
/// #230 admits the rank family through the exact rank polish (a fit-only arm),
/// so the refused set is the complement of `fit_loss ∪ rank_fit_only`.
/// Iterating over `Metric::all()` means a future metric cannot inherit the old
/// silent path.
#[test]
fn explicit_gpu_outside_the_native_objectives_is_a_usage_error() {
    let refused: Vec<&str> = Metric::all()
        .iter()
        .filter(|m| {
            rcsr::fit::fit_loss(**m).is_none()
                && rcsr::fit::rank_fit_only(**m).is_none()
                && !rcsr::fit::order_stat_fit_only(**m)
                && !rcsr::fit::maximum_fit_only(**m)
        })
        .map(|m| m.name())
        .collect();
    assert!(
        !refused.contains(&"absolute") && !refused.contains(&"rank_correlation"),
        "the device-fit family changed unexpectedly: {refused:?}"
    );
    assert!(
        !refused.contains(&"log_squashed"),
        "log_squashed is fitted by the kernel since #229 and must not be refused"
    );
    assert!(
        !refused.contains(&"rank_correlation") && !refused.contains(&"rank_correlation_signed"),
        "the rank family is fitted by the exact polish since #230 and must not be refused"
    );
    assert!(
        !refused.contains(&"median"),
        "median is fitted by the exact median polish since #226 and must not be refused"
    );
    assert!(
        !refused.contains(&"interquartile"),
        "interquartile is fitted by the same device polish since #232 and must not be refused"
    );
    assert!(
        !refused.contains(&"log_loss"),
        "#234: log_loss has a native device fit objective and must not be refused"
    );
    assert!(
        !refused.contains(&"maximum"),
        "maximum is fitted by the exact maximum-error polish since #227 and must not be refused"
    );
    for metric in refused {
        // `--implicit` metrics are refused earlier (by #14) when it is missing;
        // with it, the GPU refusal is the one that must fire.
        let implicit = Metric::parse(metric).is_implicit();
        let mut args = vec![
            "--data", "-", "--target", "y", "--vars", "x", "--fit-strategy", "gpu",
            "--metric", metric,
        ];
        if implicit {
            args.push("--implicit");
        }
        let out = rcsr(&args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "--metric {metric}: {stderr}");
        assert!(
            stderr.contains(
                "--fit-strategy gpu minimises squared error, log-squashed loss, mean"
            ),
            "--metric {metric} must say what the GPU arms minimise: {stderr}"
        );
        assert!(
            stderr.contains(&format!("--metric {metric}")),
            "--metric {metric} must name the metric it refused: {stderr}"
        );
        assert!(
            stderr.contains("--metric squared"),
            "--metric {metric} must offer --metric squared: {stderr}"
        );
        assert!(
            stderr.contains("--fit-strategy exact, auto, linear, varpro or stream"),
            "--metric {metric} must offer the CPU strategies: {stderr}"
        );
        assert!(
            out.stdout.is_empty(),
            "--metric {metric}: a refused run prints no solutions"
        );
    }
}

/// The gate must not over-fire. The four least-squares metrics,
/// `log_squashed`, `absolute` (#222), the rank family and `median` pass the
/// usage check with `gpu`, and every metric keeps every other strategy: each
/// reaches the data loader, so the run fails on the missing file (exit 3)
/// rather than on the fit strategy.
#[test]
fn the_usage_gate_only_refuses_explicit_gpu_outside_the_family() {
    for metric in [
        "squared",
        "aic",
        "aicc",
        "bic",
        "log_squashed",
        "absolute",
        "rank_correlation",
        "rank_correlation_signed",
        "median",
        "interquartile",
        "log_loss",
        "maximum",
    ] {
        let out = rcsr(&[
            "--data", "missing-223.csv", "--target", "y", "--vars", "x",
            "--fit-strategy", "gpu", "--metric", metric,
        ]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(3), "--metric {metric}: {stderr}");
        assert!(
            !stderr.contains("minimises squared error, log-squashed loss, mean"),
            "--metric {metric} is fitted by the device and must be accepted: {stderr}"
        );
    }
    for strategy in ["exact", "auto", "linear", "varpro", "stream"] {
        let out = rcsr(&[
            "--data", "missing-223.csv", "--target", "y", "--vars", "x",
            "--fit-strategy", strategy, "--metric", "absolute",
        ]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(3), "--fit-strategy {strategy}: {stderr}");
        assert!(
            !stderr.contains("minimises squared error, log-squashed loss, mean"),
            "--fit-strategy {strategy} keeps metric-aware CPU fitting: {stderr}"
        );
    }
}

/// #399: an exactly zero normal-equation diagonal freezes that slot; the other
/// constants are still fitted on the device and match the CPU fit.
#[cfg(feature = "cuda")]
#[test]
fn zero_adjoint_constant_is_frozen_and_the_rest_are_fitted() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    use rcsr::expr::{constant, input, node, Op};
    use rcsr::fit::{fit_weighted_metric, BatchedFitter, BatchedGpu};

    let n = 256;
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64 / (n - 1) as f64 * 4.0 - 2.0, 0.0])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 1.3 * r[0] + 0.5).collect();
    let nv = 2;
    let cols: Vec<f32> = (0..nv)
        .flat_map(|v| x.iter().map(move |r| r[v] as f32))
        .collect();
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut cols64 = vec![0.0f64; n * nv];
    for (r, row) in x.iter().enumerate() {
        for (v, &val) in row.iter().enumerate() {
            cols64[v * n + r] = val;
        }
    }
    let mut ev = match rcsr::gpu::GpuEvaluator::new(&cols, n, nv, &y32) {
        Some(ev) => ev,
        None => {
            eprintln!("skipping GPU test: evaluator construction failed");
            return;
        }
    };
    assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

    // c1*x0 + c0 + c2*x1, where x1 is identically zero: c2 has a zero adjoint.
    let model = || {
        node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(1.0), input(0)]),
                constant(0.0),
                node(Op::Mul, vec![constant(0.7), input(1)]),
            ],
        )
    };
    let consts = |t: &rcsr::expr::Node| {
        fn walk(n: &rcsr::expr::Node, out: &mut Vec<f64>) {
            if n.op == Op::Const {
                out.push(n.val);
            }
            for k in &n.kids {
                walk(k, out);
            }
        }
        let mut out = Vec::new();
        walk(t, &mut out);
        out
    };

    let trees = vec![model()];
    let backend = BatchedGpu::new(&mut ev);
    let proposals = backend.fit_batch_metric(&trees, &x, &y, &[], 8, Metric::Squared);
    let device = proposals[0]
        .clone()
        .expect("a zero diagonal must not drop the candidate");
    let mut cpu = model();
    fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Squared);
    let cpu_c = consts(&cpu);
    assert_eq!(device.len(), 3);
    for (d, c) in device.iter().zip(&cpu_c) {
        assert!(
            (d - c).abs() <= 1e-4 * (1.0 + c.abs()),
            "device {d} vs CPU {c}"
        );
    }
    assert!((device[0] - 1.3).abs() < 1e-3, "slope {}", device[0]);
    assert!((device[1] - 0.5).abs() < 1e-3, "intercept {}", device[1]);
    assert_eq!(device[2], 0.7, "the frozen slot does not move");
}

#[test]
fn checkpoint_round_trip_preserves_the_admission() {
    let train = sample(160, 1);
    let test = sample(40, 2);
    let base = Config {
        generations: 2,
        islands: 1,
        pop: 12,
        max_complexity: 8,
        fit_iters: 2,
        patience: 0,
        seed: 4,
        fit_strategy: FitStrategy::Gpu,
        gpu: false,
        ..Default::default()
    };
    assert_eq!(base.fit_admission, rcsr::fit::FitAdmission::Screened);
    let save = |cfg: &Config| {
        let mut text: Option<String> = None;
        search::run_resumable(cfg, &train, &test, None, 0, |cp| {
            text = Some(cp.to_text());
        })
        .expect("fresh run");
        text.expect("a checkpoint callback")
    };
    let screened = save(&base);
    assert!(
        !screened.contains("fit_admission"),
        "the default admission leaves the checkpoint bytes unchanged"
    );
    let cp = Checkpoint::from_text(&screened).expect("parse screened checkpoint");
    assert_eq!(cp.config.fit_admission, rcsr::fit::FitAdmission::Screened);

    let all = Config { fit_admission: rcsr::fit::FitAdmission::All, ..base.clone() };
    let text = save(&all);
    assert!(text.contains("config fit_admission all"));
    let cp = Checkpoint::from_text(&text).expect("parse all checkpoint");
    assert_eq!(cp.config.fit_strategy, FitStrategy::Gpu);
    assert_eq!(cp.config.fit_admission, rcsr::fit::FitAdmission::All);
    let again = Checkpoint::from_text(&cp.to_text()).expect("reparse");
    assert_eq!(again.config.fit_admission, rcsr::fit::FitAdmission::All);
    assert_ne!(
        rcsr::checkpoint::config_hash(&again.config),
        rcsr::checkpoint::config_hash(&base),
        "the admission is part of the config hash"
    );
}

fn cli_csv() -> String {
    let mut text = String::from("x0,x1,y\n");
    for i in 0..300 {
        let a = ((i * 37) % 101) as f64 / 25.0 - 2.0;
        let b = ((i * 53) % 97) as f64 / 24.0 - 2.0;
        let noise = (((i * 29) % 13) as f64 - 6.0) / 100.0;
        text.push_str(&format!("{a},{b},{}\n", 1.5 * (0.8 * a).sin() + 0.3 * b + noise));
    }
    text
}

/// The `fitting` object and the Pareto array of one `--json` run: everything
/// the fit arm can change, without the wall-clock fields.
fn cli_fit_and_front(extra: &[&str]) -> (String, String) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", "-", "--target", "y", "--vars", "x0,x1", "--islands", "1", "--pop", "30",
            "--generations", "3", "--seed", "1", "--patience", "0", "--json", "--no-dev-log",
        ])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(cli_csv().as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let json = String::from_utf8(out.stdout).unwrap();
    let fitting = json.split("\"fitting\":").nth(1).unwrap().split('}').next().unwrap().to_string();
    let front = json.split("\"pareto\":").nth(1).unwrap().to_string();
    (fitting, front)
}

/// A CPU run keeps today's archive and `fitting` object whether the arm is
/// unset or explicitly `exact`.
#[test]
fn cpu_default_run_is_byte_identical_to_explicit_exact() {
    let default = cli_fit_and_front(&["--cpu", "--metric", "squared"]);
    let exact = cli_fit_and_front(&["--cpu", "--metric", "squared", "--fit-strategy", "exact"]);
    assert_eq!(default, exact);
    assert!(!default.0.contains("strategy_source"), "{}", default.0);
}

#[cfg(feature = "cuda")]
#[test]
fn default_run_resolves_to_the_gpu_arm_and_labels_every_other_resolution() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (fitting, front) = cli_fit_and_front(&["--metric", "squared"]);
    assert!(fitting.contains("\"strategy\":\"gpu\""), "{fitting}");
    assert!(fitting.contains("\"strategy_source\":\"default\""), "{fitting}");
    assert!(fitting.contains("\"strategy_reason\":\"admissible\""), "{fitting}");
    let again = cli_fit_and_front(&["--metric", "squared"]);
    assert_eq!(front, again.1, "two default-resolved runs give one front");

    let (fitting, _) = cli_fit_and_front(&["--metric", "squared", "--no-fit"]);
    assert!(fitting.contains("\"strategy\":\"exact\""), "{fitting}");
    assert!(fitting.contains("\"strategy_reason\":\"no_fit\""), "{fitting}");

    let (fitting, _) = cli_fit_and_front(&["--metric", "absolute"]);
    assert!(fitting.contains("\"strategy\":\"exact\""), "{fitting}");
    assert!(fitting.contains("\"strategy_reason\":\"metric_not_least_squares\""), "{fitting}");

    let (fitting, _) = cli_fit_and_front(&["--metric", "squared", "--fit-strategy", "auto"]);
    assert!(fitting.contains("\"strategy\":\"gpu\""), "{fitting}");
    assert!(fitting.contains("\"strategy_source\":\"auto\""), "{fitting}");

    let (fitting, _) = cli_fit_and_front(&["--metric", "squared", "--fit-strategy", "gpu"]);
    assert!(!fitting.contains("strategy_source"), "an explicit arm is never resolved: {fitting}");
}

#[cfg(feature = "cuda")]
fn correlation_run(metric: Metric, seed: u32, fit_iters: usize) -> (Outcome, Sample) {
    let train = sample(2000, seed);
    let test = sample(200, seed + 1);
    let cfg = Config {
        generations: 4,
        pop: 32,
        fit_iters,
        metric,
        ..gpu_config(seed + 2)
    };
    (search::run(&cfg, &train, &test), train)
}

/// The correlation family fits on the device through the exact gate: two runs
/// are bit-identical and every archive error is the exact CPU `f64` metric.
#[cfg(feature = "cuda")]
fn assert_correlation_gpu_fit_is_deterministic_and_exact(metric: Metric, seed: u32) {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (a, train) = correlation_run(metric, seed, 4);
    let (b, _) = correlation_run(metric, seed, 4);
    assert_eq!(describe(&a), describe(&b), "a fixed seed must be deterministic");
    assert_eq!(a.test_errors, b.test_errors);
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu, "{:?}", a.backend_reason);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(g.fit_gpu > 0, "the correlation fit kernel must run: {:?}", g);
    assert_fit_ledger(g);
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, metric);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU {} score: {} vs {}",
            metric.name(),
            s.error,
            cpu
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn correlation_gpu_fit_run_is_deterministic_and_exact() {
    assert_correlation_gpu_fit_is_deterministic_and_exact(Metric::Correlation, 81);
}

#[cfg(feature = "cuda")]
#[test]
fn signed_gpu_fit_run_is_deterministic_and_exact() {
    assert_correlation_gpu_fit_is_deterministic_and_exact(Metric::CorrelationSigned, 91);
}

/// `--fit-iters 0` attempts no fit and moves no constant for either metric.
#[cfg(feature = "cuda")]
#[test]
fn correlation_gpu_no_fit_is_a_true_noop() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    for metric in [Metric::Correlation, Metric::CorrelationSigned] {
        let (out, train) = correlation_run(metric, 101, 0);
        let g = out.gpu.as_ref().expect("a GPU report");
        assert_eq!(g.fit_attempts, 0, "no fit may be attempted: {:?}", g);
        assert_eq!(g.fit_gpu, 0);
        for s in &out.archive.items {
            let cpu = search::eval_error(&s.tree, &train, metric);
            assert_eq!(s.error.to_bits(), cpu.to_bits());
        }
    }
}

/// Affine-in-constants correlation trees take the CPU closed form, labelled
/// `affine_closed_form`, and the ledger still closes.
#[cfg(feature = "cuda")]
#[test]
fn an_affine_correlation_tree_takes_the_closed_form() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (out, _) = correlation_run(Metric::CorrelationSigned, 111, 4);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert_fit_ledger(g);
    let affine = g
        .fit_fallback_by_cause
        .get("affine_closed_form")
        .copied()
        .unwrap_or(0);
    assert!(affine > 0, "no affine tree was routed: {:?}", g.fit_fallback_by_cause);
}

fn implicit_circle(n: usize) -> Sample {
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let r = 1.0 + 0.04 * (((i * 37) % 11) as f64 / 11.0 - 0.5);
            vec![r * a.cos(), r * a.sin()]
        })
        .collect();
    Sample {
        x,
        y: vec![0.0; n],
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

/// `implicit_residual` is the absolute objective on the all-zero implicit
/// target: the dispatch table, the fit objective and the weights gate follow
/// `absolute`.
#[test]
fn implicit_residual_dispatch_is_absolute() {
    use rcsr::fit::DeviceLoss;
    use rcsr::metrics::ScoreObjective;
    assert_eq!(
        rcsr::metrics::native_score_objective(Metric::ImplicitResidual),
        Some(ScoreObjective::Absolute)
    );
    assert_eq!(rcsr::fit::fit_loss(Metric::ImplicitResidual), Some(DeviceLoss::Absolute));
    assert!(rcsr::fit::gpu_fit_metric_error(Metric::ImplicitResidual).is_none());
    assert!(rcsr::metrics::score_refusal(Metric::ImplicitResidual).is_none());
    assert!(rcsr::metrics::native_score_objective(Metric::ImplicitDerivative).is_none());
}

/// A GPU `implicit_residual` run on an implicit circle sample defers scoring and
/// the constant fit to the device, repeats exactly, and reports the exact CPU
/// metric for every archived tree.
#[cfg(feature = "cuda")]
#[test]
fn implicit_residual_gpu_search_defers_and_reports_the_exact_metric() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = implicit_circle(4096);
    let test = implicit_circle(512);
    let cfg = Config {
        generations: 4,
        islands: 1,
        pop: 64,
        max_complexity: 12,
        fit_iters: 6,
        patience: 0,
        seed: 3,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::ImplicitResidual,
        ops: vec![expr::Op::Add, expr::Op::Sub, expr::Op::Mul, expr::Op::Square],
        gpu: true,
        ..Default::default()
    };
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu, "{:?}", a.backend_reason);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(g.fit_gpu > 0, "the fit kernel must have run: {:?}", g);
    assert_eq!(describe(&a), describe(&b));
    assert!(!a.archive.items.is_empty());
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::ImplicitResidual);
        assert_eq!(s.error.to_bits(), cpu.to_bits(), "{}", s.error);
    }
}
