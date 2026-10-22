//! #230: rank-correlation fitting on the GPU.
//!
//! The device-level parity tests live in `src/gpu.rs` (they need the
//! crate-private fit plans and the raw rank-stage hook). These integration
//! tests pin the search contract:
//!
//! * `rank_correlation`/`rank_correlation_signed` may use the GPU **fit** arm
//!   only when `--fit-strategy gpu` is requested; it is a fit-only arm: the
//!   backend stays `cpu`, the fitness is scored by the CPU evaluator, and the
//!   population is never ranked by MSE;
//! * reported errors are the exact CPU metric of the stored tree;
//! * fixed seed/device is deterministic, and `--no-fit` is a true no-op;
//! * every fallback is labelled and the ledger adds up.
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
        // Ties on purpose: every seventh row repeats the previous target, so
        // the average-rank rule and the tie correction are exercised.
        let v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
        y.push(if i % 7 == 3 { 0.0 } else { v });
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

#[cfg(feature = "cuda")]
fn rank_config(seed: u32, metric: Metric) -> Config {
    Config {
        generations: 4,
        islands: 1,
        pop: 20,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric,
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

/// #230: the rank family is offered to the fit arm only when the user asked
/// for the GPU fit strategy. Scoring stays on the CPU in every case, so the
/// backend itself is reported as `cpu` even though the fit arm uses the device.
#[test]
fn rank_gpu_fit_is_opt_in_and_never_the_default() {
    let s = sample(96, 3);
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        // Default strategy (`exact`): the CPU, regardless of `--gpu`.
        let cfg = Config {
            gpu: true,
            metric,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
        // `--fit-strategy gpu` with fitting enabled offers the fit arm, but the
        // evaluator has no native rank scorer yet, so the reported backend is
        // still the CPU (native rank evaluation is open, #230).
        let cfg = Config {
            gpu: true,
            metric,
            fit_strategy: FitStrategy::Gpu,
            fit_iters: 4,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
        // `--no-fit` has nothing to fit, so it stays on the CPU.
        let cfg = Config {
            gpu: true,
            metric,
            fit_strategy: FitStrategy::Gpu,
            fit_iters: 0,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_rank_run_is_deterministic_and_reports_the_exact_objective() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 5);
    let test = sample(60, 6);
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        let cfg = rank_config(11, metric);
        let a = search::run(&cfg, &train, &test);
        let b = search::run(&cfg, &train, &test);
        // Fit-only: the device moves constants, the CPU scores and reports.
        assert_eq!(a.backend, rcsr::backend::Kind::Cpu);
        assert!(a.device.is_some(), "the fit arm used a device");
        assert_eq!(describe(&a), describe(&b), "{metric:?}: repeated seed differs");
        assert_eq!(a.test_errors, b.test_errors);
        let g = a.gpu.as_ref().expect("a GPU report");
        assert!(g.fit_gpu > 0, "{metric:?}: the rank fit arm must run: {:?}", g);
        // The reported error of every archive row is the exact CPU score of
        // the configured metric, not an MSE proxy.
        for s in &a.archive.items {
            let cpu = search::eval_error(&s.tree, &train, metric);
            assert_eq!(
                s.error.to_bits(),
                cpu.to_bits(),
                "{metric:?}: archive row is not the exact CPU rank score"
            );
            assert!(
                (0.0..=2.0).contains(&s.error),
                "{metric:?}: rank score out of range: {}",
                s.error
            );
        }
        // Ledger: every admitted fit is one GPU success or one labelled cause.
        assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback, "{:?}", g);
        let known: Vec<&str> = search::FitFallbackCause::ALL
            .iter()
            .map(|c| c.label())
            .collect();
        let mut labelled = 0usize;
        for (cause, n) in &g.fit_fallback_by_cause {
            assert!(known.contains(&cause.as_str()), "unknown cause {cause:?}");
            labelled += n;
        }
        assert_eq!(labelled, g.fit_fallback, "{:?}", g);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn no_fit_is_a_true_noop_for_the_rank_gpu_arm() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(160, 7);
    let test = sample(40, 8);
    let base = Config {
        fit_iters: 0,
        metric: Metric::RankCorrelation,
        ..rank_config(13, Metric::RankCorrelation)
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
    // same archive, bit for bit, and the GPU path must not be entered at all.
    assert_eq!(describe(&gpu), describe(&cpu));
    // With nothing to fit the evaluator is not even created: the rank arm is
    // never a silent no-op that still costs a device context.
    assert!(gpu.gpu.is_none(), "--no-fit must not spin up the fit evaluator");
    assert_eq!(gpu.backend, rcsr::backend::Kind::Cpu);
}

/// #230 measurement (ignored): paired CPU-exact versus GPU rank fit on the
/// pinned cell. Reports wall clock, fit-stage share, GPU duty and quality.
/// Run with
/// `cargo test --release --features cuda --test gpu_rank bench_rank_end_to_end -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_rank_end_to_end() {
    if !device_available() {
        eprintln!("skipping GPU benchmark: no usable CUDA device");
        return;
    }
    let n: usize = std::env::var("RCSR_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let pop: usize = std::env::var("RCSR_BENCH_POP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let generations: usize = std::env::var("RCSR_BENCH_GENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let islands: usize = std::env::var("RCSR_BENCH_ISLANDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let train = sample(n, 21);
    let test = sample(1_000, 22);
    use rcsr::expr::Op;
    let base = Config {
        generations,
        islands,
        pop,
        max_complexity: 12,
        fit_iters: 8,
        metric: Metric::RankCorrelation,
        ops: vec![
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Sin,
            Op::Cos,
            Op::Tanh,
            Op::Square,
            Op::Neg,
            Op::Div,
        ],
        seed: 9,
        patience: 0,
        gpu_notes: false,
        ..Default::default()
    };
    for (name, strategy, gpu) in [
        ("exact/cpu", FitStrategy::Exact, false),
        ("gpu/rank", FitStrategy::Gpu, true),
    ] {
        let cfg = Config {
            fit_strategy: strategy,
            gpu,
            ..base.clone()
        };
        let t0 = std::time::Instant::now();
        let out = search::run(&cfg, &train, &test);
        let wall = t0.elapsed().as_secs_f64();
        let best = out
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "arm={} N={} evals={} wall={:.2}s best={:.6e}",
            name,
            n,
            base.generations * base.islands * base.pop,
            wall,
            best
        );
        if let Some(g) = &out.gpu {
            let rate = if g.fit_attempts == 0 {
                0.0
            } else {
                100.0 * g.fit_fallback as f64 / g.fit_attempts as f64
            };
            eprintln!(
                "  fit: screened={} admitted={} gpu={} fallback={} rate={:.1}% refits={}",
                g.fit_screened, g.fit_attempts, g.fit_gpu, g.fit_fallback, rate, g.fit_refits
            );
            eprintln!(
                "  time: fit_host={:.2}s ({:.1}% of wall) kernel={:.2}s fit_duty={:.1}% eval_duty={:.1}%",
                g.fit_host_ms / 1000.0,
                100.0 * (g.fit_host_ms / 1000.0) / wall.max(1e-9),
                g.fit_ms / 1000.0,
                if g.fit_host_ms > 0.0 {
                    100.0 * g.fit_ms / g.fit_host_ms
                } else {
                    0.0
                },
                100.0 * g.accounting.duty_cycle()
            );
        }
    }
}
