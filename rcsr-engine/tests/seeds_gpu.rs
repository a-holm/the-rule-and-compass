//! #236: the seed path on the CPU and the device (#16, #197).
//!
//! A seed is fitted and scored on the CPU before either backend runs. This
//! file runs the same seeded search on both backends, so the pure `seeds`
//! binary never opens the device.

use rcsr::bench;
use rcsr::data::Sample;
use rcsr::dispatch::DispatchMode;
use rcsr::expr;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

/// The spring-period law `p = sqrt(m / k)` at zero noise, over a fixed grid.
fn spring() -> (Vec<String>, Sample, Sample) {
    let p = bench::known_truth()
        .into_iter()
        .find(|p| p.name == "spring_period")
        .expect("the spring_period problem");
    let (train, test) = p.generate(0.0);
    (p.vars.clone(), train, test)
}

fn config(seed: u32, generations: usize) -> Config {
    Config {
        generations,
        islands: 1,
        pop: 60,
        max_complexity: 30,
        fit_iters: 8,
        patience: 0,
        seed,
        metric: Metric::Absolute,
        gpu: false,
        ..Default::default()
    }
}

/// Regression for #197: a seed written with neutral terms must reach the same
/// canonical tree on the CPU and GPU paths, and the archive entry must be the
/// CPU-refitted seed on both.
///
/// GPU fitness is f32 and may therefore choose a different final front, but it
/// must never change the seed's structure or its reported error: the seed is
/// fitted and scored on the CPU in f64 before either backend runs. Before the
/// fix, a domain-safe simplifier left `0 + sqrt(m / k)` in the seed; the GPU
/// trajectory then found the smaller `sqrt(m / k)` with the same zero error,
/// which legitimately dominated and removed the seed from the Pareto front.
///
/// #236: the device scores squared error only, so the A/B uses `squared` for
/// both arms; with the default `absolute` the "GPU" arm would silently be a
/// second CPU run. The absolute-metric seed path is covered by the tests above.
#[test]
fn seed_is_canonical_and_present_on_cpu_and_gpu_paths() {
    #[cfg(feature = "cuda")]
    if !rcsr::gpu::probe().available() {
        rcsr::backend::skip_without_device("skipping: no usable CUDA device");
        return;
    }
    let (vars, train, test) = spring();
    let seed = expr::parse("2 * sqrt(m / k) + 0", &vars).expect("parse seed");
    let canonical = expr::simplify(&expr::parse("sqrt(m / k)", &vars).expect("parse canonical"));
    let expected_hash = expr::structural_hash(&canonical);

    let mut reported = Vec::new();
    for gpu in [false, true] {
        let mut cfg = config(1, 8);
        cfg.gpu = gpu;
        if gpu {
            cfg.dispatch = DispatchMode::Device;
        }
        cfg.metric = Metric::Squared;
        let out = search::run_seeded(&cfg, &train, &test, &[seed.clone()]);

        assert_eq!(out.seeds.len(), 1, "gpu={}: one accepted seed", gpu);
        let fitted = &out.seeds[0];
        assert_eq!(
            expr::structural_hash(&fitted.tree),
            expected_hash,
            "gpu={}: the neutral terms must canonicalise to sqrt(m / k), got {}",
            gpu,
            expr::to_string(&fitted.tree, &vars),
        );
        assert!(
            fitted.error < 1e-10,
            "gpu={}: the fitted seed should explain the data, error = {}",
            gpu,
            fitted.error,
        );

        let archived = out
            .archive
            .items
            .iter()
            .find(|s| expr::structural_hash(&s.tree) == expected_hash)
            .unwrap_or_else(|| {
                panic!(
                    "gpu={}: the seed must survive in the final archive; front is {:?}",
                    gpu,
                    out.archive
                        .items
                        .iter()
                        .map(|s| (expr::to_string(&s.tree, &vars), s.error, s.cost))
                        .collect::<Vec<_>>(),
                )
            });
        assert_eq!(
            archived.error.to_bits(),
            fitted.error.to_bits(),
            "gpu={}: the archived seed must carry the CPU-refitted error",
            gpu,
        );
        reported.push((
            expr::structural_hash(&fitted.tree),
            fitted.error.to_bits(),
            expr::to_string(&fitted.tree, &vars),
        ));
    }

    assert_eq!(
        reported[0], reported[1],
        "the CPU and GPU paths must report the same fitted seed",
    );
}
