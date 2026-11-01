//! #288: native order-statistic and rank **scoring** in the search.
//!
//! These integration tests pin the production path, not just the device entry
//! point:
//!
//! * from the measured row floor, `median`, `interquartile`,
//!   `rank_correlation`, `rank_correlation_signed` and `auc` score the whole
//!   population on the device, so `--no-fit` still uses the exact statistic
//!   (the constants are untouched, but the fitness is not a CPU fallback);
//! * every reported archive error is the exact CPU metric of the stored tree,
//!   never an SSE proxy;
//! * a metric without a native scorer keeps its named CPU refusal.
//!
//! Device-dependent tests skip cleanly on a machine without CUDA.

use rcsr::checkpoint::Backend;
use rcsr::data::Sample;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

fn sample(n: usize, seed: u32, binary: bool) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        if binary {
            // Exactly two target values: the AUC midpoint rule is the larger
            // one and the device's rank-derived label mask is exact.
            y.push(if i % 3 == 0 { 1.0 } else { -1.0 });
        } else {
            // Duplicates and outliers on purpose, so ties in |r| and in the
            // rank statistic are exercised.
            let v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
            y.push(if i % 7 == 3 { 0.0 } else { v });
        }
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
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

#[cfg(feature = "cuda")]
fn no_fit_config(metric: Metric, seed: u32) -> Config {
    Config {
        generations: 2,
        islands: 1,
        pop: 16,
        max_complexity: 8,
        fit_iters: 0,
        patience: 0,
        seed,
        metric,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        ..Default::default()
    }
}

/// #288: `--no-fit` is no longer a synonym for "CPU scoring". At or above the
/// floor the exact order/rank statistic scores the population on the device
/// and every reported archive error is still the exact CPU value.
#[cfg(feature = "cuda")]
#[test]
fn no_fit_scores_order_and_rank_metrics_on_the_device() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(rcsr::fit::MEDIAN_GPU_MIN_ROWS + 64, 7, false);
    let test = sample(120, 8, false);
    for metric in [
        Metric::Median,
        Metric::Interquartile,
        Metric::RankCorrelation,
        Metric::RankCorrelationSigned,
    ] {
        let out = search::run(&no_fit_config(metric, 11), &train, &test);
        assert_eq!(
            out.backend,
            rcsr::backend::Kind::Gpu,
            "{metric:?}: the exact statistic must score on the device: {:?}",
            out.backend_reason
        );
        assert!(out.device.is_some(), "{metric:?}: no device recorded");
        let g = out.gpu.as_ref().expect("a GPU report");
        assert!(
            g.generations.iter().any(|gen| gen.programs > 0),
            "{metric:?}: no program was scored"
        );
        assert_eq!(g.fit_attempts, 0, "{metric:?}: --no-fit moved constants");
        for row in &out.archive.items {
            let cpu = search::eval_error(&row.tree, &train, metric);
            assert_eq!(
                row.error.to_bits(),
                cpu.to_bits(),
                "{metric:?}: archive row is not the exact CPU statistic"
            );
        }
    }
}

/// #287: maximum is an objective-kernel row of the table, so it scores on the
/// device without the order/rank floor and its archive is still exact.
#[cfg(feature = "cuda")]
#[test]
fn maximum_scores_natively_through_the_objective_kernel() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(rcsr::fit::MAXIMUM_GPU_MIN_ROWS + 64, 9, false);
    let test = sample(120, 10, false);
    let out = search::run(&no_fit_config(Metric::Maximum, 13), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{:?}", out.backend_reason);
    for row in &out.archive.items {
        assert_eq!(
            row.error.to_bits(),
            search::eval_error(&row.tree, &train, Metric::Maximum).to_bits()
        );
    }
}

/// #288: AUC uses the exact integer rank statistic with the CPU's midpoint
/// label rule; a two-valued target is admitted and the archive is exact.
#[cfg(feature = "cuda")]
#[test]
fn auc_scores_natively_on_a_binary_target() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(rcsr::fit::MEDIAN_GPU_MIN_ROWS + 64, 11, true);
    let test = sample(120, 12, true);
    let out = search::run(&no_fit_config(Metric::Auc, 17), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{:?}", out.backend_reason);
    for row in &out.archive.items {
        assert_eq!(
            row.error.to_bits(),
            search::eval_error(&row.tree, &train, Metric::Auc).to_bits(),
            "AUC archive row is not the exact CPU statistic"
        );
    }
}

/// #288: a metric without a native scorer keeps the named CPU refusal; it is
/// never silently ranked by MSE.
#[test]
fn a_metric_outside_the_table_keeps_its_named_refusal() {
    let s = sample(96, 3, false);
    let cfg = Config {
        gpu: true,
        metric: Metric::ImplicitDerivative,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    #[cfg(feature = "cuda")]
    assert_eq!(
        search::backend_blocker_reason(&cfg, &s),
        rcsr::metrics::score_refusal(Metric::ImplicitDerivative)
    );
    assert_eq!(
        rcsr::metrics::score_refusal(Metric::ImplicitDerivative),
        Some("implicit needs an expression-tree gradient pass")
    );
}