//! #169: batched GPU constant fitting with exact CPU f64 verification.
//!
//! The kernel tests live in `src/gpu.rs` (they can reach the crate-private fit
//! plans). These integration tests pin the properties the search itself must
//! keep: the opt-in strategy, determinism, exact reporting and checkpoint
//! round-trips. Device-dependent tests skip cleanly on a machine without CUDA.

use rcsr::checkpoint::Checkpoint;
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
}

/// #169 measurement (ignored): end-to-end wall time of a fit-heavy
/// configuration on the CPU exact arm versus the GPU-fit arm.
/// Run with `cargo test --release --features cuda --test gpu_fit bench_end_to_end -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_end_to_end() {
    if !device_available() {
        eprintln!("skipping GPU benchmark: no usable CUDA device");
        return;
    }
    let n = 154_000;
    let train = sample(n, 21);
    let test = sample(1_000, 22);
    // A fit-heavy cell: a smooth, always-finite operator set so most random
    // candidates are eligible for the kernel (the default set mixes in
    // non-finite and piecewise-constant operators, which correctly fall back).
    use rcsr::expr::Op;
    let base = Config {
        generations: 2,
        islands: 4,
        pop: 64,
        max_complexity: 14,
        fit_iters: 8,
        metric: Metric::Squared,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Sin, Op::Cos, Op::Tanh, Op::Gauss, Op::Square, Op::Cube, Op::Neg],
        seed: 9,
        patience: 0,
        gpu_notes: false,
        ..Default::default()
    };
    for (name, strategy, gpu) in [
        ("exact/cpu", FitStrategy::Exact, false),
        ("gpu", FitStrategy::Gpu, true),
    ] {
        let cfg = Config {
            fit_strategy: strategy,
            gpu,
            ..base.clone()
        };
        let t0 = std::time::Instant::now();
        let out = search::run(&cfg, &train, &test);
        let wall = t0.elapsed().as_secs_f64();
        let fit = out
            .gpu
            .as_ref()
            .map(|g| (g.fit_gpu, g.fit_fallback, g.fit_ms, g.fit_refits))
            .unwrap_or((0, 0, 0.0, 0));
        eprintln!(
            "arm={} N={} wall={:.2}s gpu_fit={} fallback={} kernel={:.2}s refits={}",
            name,
            n,
            wall,
            fit.0,
            fit.1,
            fit.2 / 1000.0,
            fit.3
        );
    }
}
