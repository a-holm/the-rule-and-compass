//! #230: rank-correlation fitting on the GPU.
//!
//! The device-level parity tests live in `src/gpu.rs` (they need the
//! crate-private fit plans and the raw rank-stage hook). These integration
//! tests pin the search contract:
//!
//! * the exact integer rank statistic also **scores** the population on the
//!   device from `MEDIAN_GPU_MIN_ROWS` (#288), so the backend is `gpu` there;
//!   below the floor the exact CPU rank score runs with a named reason. The
//!   population is never ranked by MSE;
//! * `--fit-strategy gpu` additionally moves constants through the exact rank
//!   polish (fit-only below the scoring floor);
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
        pairs: None,
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

/// #288: below the measured scoring floor the exact CPU rank score runs (the
/// fit-only arm may still use the device); above it the exact integer statistic
/// scores the population and the backend is `gpu`.
#[test]
fn rank_gpu_scoring_respects_the_measured_row_floor() {
    let s = sample(96, 3);
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        // Default strategy (`exact`): the CPU, regardless of `--gpu`.
        let cfg = Config {
            gpu: true,
            metric,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
        // `--fit-strategy gpu` with fitting enabled offers the fit arm; below
        // the scoring floor the exact CPU rank score remains, so the reported
        // backend is still the CPU.
        let cfg = Config {
            gpu: true,
            metric,
            fit_strategy: FitStrategy::Gpu,
            fit_iters: 4,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
        // `--no-fit` below the floor leaves nothing for the device.
        let cfg = Config {
            gpu: true,
            metric,
            fit_strategy: FitStrategy::Gpu,
            fit_iters: 0,
            ..Default::default()
        };
        assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
        // At or above the floor the exact device statistic scores the run even
        // with the default `exact` fit strategy (#288).
        let big = sample(rcsr::fit::MEDIAN_GPU_MIN_ROWS + 32, 3);
        let cfg = Config {
            gpu: true,
            metric,
            ..Default::default()
        };
        let expected = if cfg!(feature = "cuda") {
        Backend::Gpu
    } else {
        Backend::Cpu
    };
    assert_eq!(search::effective_backend(&cfg, &big), expected);
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
    // #231: the signed orientation has its own arm; `--no-fit` must be a no-op
    // for it too, not only for `1 - |rho|`.
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        let base = Config {
            fit_iters: 0,
            metric,
            ..rank_config(13, metric)
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
        // same archive, bit for bit, and the GPU path must not be entered.
        assert_eq!(describe(&gpu), describe(&cpu), "{metric:?}");
        // With nothing to fit the evaluator is not even created: the rank arm is
        // never a silent no-op that still costs a device context.
        assert!(gpu.gpu.is_none(), "{metric:?}: --no-fit must not spin up the fit evaluator");
        assert_eq!(gpu.backend, rcsr::backend::Kind::Cpu);
    }
}

/// #231: the signed arm is deterministic for a fixed seed and device, whatever
/// the island count, and every archive row is the exact CPU signed score.
#[cfg(feature = "cuda")]
#[test]
fn signed_rank_gpu_fit_is_deterministic_across_runs_and_island_counts() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(240, 41);
    let test = sample(60, 42);
    let one = rank_config(19, Metric::RankCorrelationSigned);
    let four = Config {
        islands: 4,
        ..rank_config(19, Metric::RankCorrelationSigned)
    };
    let a = search::run(&one, &train, &test);
    let b = search::run(&one, &train, &test);
    assert_eq!(describe(&a), describe(&b), "repeated seed differs");
    let c = search::run(&four, &train, &test);
    let d = search::run(&four, &train, &test);
    assert_eq!(describe(&c), describe(&d), "repeated 4-island seed differs");
    assert_eq!(c.test_errors, d.test_errors);
    for out in [&a, &c] {
        assert!(out.device.is_some());
        let g = out.gpu.as_ref().expect("a GPU report");
        assert!(g.fit_gpu > 0, "the signed arm must run on the device: {g:?}");
        for s in &out.archive.items {
            let cpu = search::eval_error(&s.tree, &train, Metric::RankCorrelationSigned);
            assert_eq!(
                s.error.to_bits(),
                cpu.to_bits(),
                "an archive row is not the exact CPU signed score"
            );
        }
    }
}

/// #231: the public batched verifier (`BatchedGpu` + `fit_batch_verified`) is
/// rank-native and judges the device proposal on the **exact** configured rank
/// objective. Before #231 it declined every rank metric, so a rank run could
/// only ever reach the device through the search's own arm.
#[cfg(feature = "cuda")]
#[test]
fn the_batched_verifier_is_rank_native_and_exact() {
    use rcsr::expr::{constant, input, node, Op};
    use rcsr::fit::{fit_batch_verified, fit_weighted_metric, BatchStatus, BatchedGpu};
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(400, 51);
    let x = train.x.clone();
    let y = train.y.clone();
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
    assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
    let model = || {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)],
        )
    };
    let k = rcsr::expr::count_constants(&model());
    let score = |t: &rcsr::expr::Node, metric: Metric| {
        let pred: Vec<f64> = x.iter().map(|row| rcsr::expr::eval(t, row)).collect();
        rcsr::metrics::score_weighted(&pred, &y, &[], metric, k)
    };
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        let mut batch = vec![model()];
        let backend = BatchedGpu::new(&mut ev);
        let out = fit_batch_verified(&backend, &mut batch, &x, &y, &[], 8, metric);
        assert_eq!(
            out.status[0],
            BatchStatus::Gpu,
            "{metric:?}: the rank proposal must come from the device arm"
        );
        // The reported number is the exact configured rank score of the final
        // constants, not an SSE and not the other orientation.
        assert_eq!(out.score[0].to_bits(), score(&batch[0], metric).to_bits());
        // Monotone against the incoming constants...
        assert!(out.score[0] <= score(&model(), metric));
        // ... and no worse than the exact CPU arm from the same start.
        let mut cpu = model();
        let cpu_score = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, metric);
        assert!(
            out.score[0] <= cpu_score + 1e-12,
            "{metric:?}: device {} is worse than the CPU arm {cpu_score}",
            out.score[0]
        );
    }
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
    // #231: both rank orientations, paired CPU exact versus GPU rank fit on the
    // same seed and rows. `1 - rho` and `1 - |rho|` share the device work.
    for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
        let base = Config {
            generations,
            islands,
            pop,
            max_complexity: 12,
            fit_iters: 8,
            metric,
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
                "metric={} arm={} N={} evals={} wall={:.2}s best={:.6e}",
                metric.name(),
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
}
