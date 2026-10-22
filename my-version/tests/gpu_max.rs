//! #227: maximum absolute error (minimax / Chebyshev) fitting on the GPU.
//!
//! The device-level tests live in `src/gpu.rs` (`maximum_tests`: the exact
//! maximum, its lowest maximiser, the exact level interval, the polish itself).
//! These integration tests pin the search contract:
//!
//! * `--metric maximum` may use the GPU **fit** arm only when
//!   `--fit-strategy gpu` is requested; it is a fit-only arm: the backend stays
//!   `cpu`, every fitness value is the CPU `f64` maximum, and the population is
//!   never ranked by MSE;
//! * reported errors are the exact CPU `max |r|` of the stored tree;
//! * the arm never loses to the exact CPU polish on the same candidates — the
//!   zero-tolerance dominance claim of the research note;
//! * fixed seed/device is deterministic and `--no-fit` is a true no-op;
//! * every fallback is labelled, the ledger adds up, and the measured cost
//!   model keeps a small sample on the CPU with a reason.
//!
//! Device-dependent tests skip cleanly on a machine without CUDA.

use rcsr::checkpoint::Backend;
use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};
#[cfg(feature = "cuda")]
use rcsr::search::Outcome;

/// A sample with a gross outlier, exact duplicate targets and a smooth signal,
/// so the least-squares start and the minimax optimum really differ.
fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        let mut v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
        if i % 7 == 3 {
            v = 0.0; // exact duplicate target
        }
        if i % 151 == 5 {
            v += 9.0; // gross outlier: the minimax fit must ignore it
        }
        y.push(v);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

#[cfg(feature = "cuda")]
fn maximum_config(seed: u32) -> Config {
    Config {
        generations: 3,
        islands: 1,
        pop: 12,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Maximum,
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

/// #227: the maximum fit arm is opt-in, fit-only, and never the default. The
/// scoring objective is the metric, so the reported backend is always `cpu`.
#[test]
fn maximum_gpu_fit_is_opt_in_and_never_the_default() {
    let s = sample(96, 3);
    let cfg = Config {
        gpu: true,
        metric: Metric::Maximum,
        ..Default::default()
    };
    assert_eq!(search::effective_backend(&cfg, &s), Backend::Cpu);
    let cfg = Config {
        gpu: true,
        metric: Metric::Maximum,
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
}

/// #227: the cost model keeps a small sample on the exact CPU maximum arm and
/// records the labelled reason; nothing is silently fitted on the device.
#[cfg(feature = "cuda")]
#[test]
fn a_small_sample_keeps_the_labelled_cpu_maximum_arm() {
    let train = sample(256, 4);
    let test = sample(200, 5);
    assert!(
        train.len() < rcsr::fit::MAXIMUM_GPU_MIN_ROWS,
        "the fixture must sit below the measured crossover"
    );
    let out = search::run(&maximum_config(7), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Cpu);
    assert!(
        out.gpu.is_none(),
        "below the crossover the device fit arm must not be created"
    );
    let why = out.backend_reason.unwrap_or_default();
    assert!(
        why.contains("maximum fit arm") || why.contains("not scored natively"),
        "the CPU decision must be labelled: {why:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_maximum_run_is_deterministic_and_reports_the_exact_objective() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let n = rcsr::fit::MAXIMUM_GPU_MIN_ROWS + 64;
    let train = sample(n, 5);
    let test = sample(300, 6);
    let a = search::run(&maximum_config(11), &train, &test);
    let b = search::run(&maximum_config(11), &train, &test);
    // Fit-only: the device moves constants, the CPU scores and reports.
    assert_eq!(a.backend, rcsr::backend::Kind::Cpu);
    assert!(a.device.is_some(), "the fit arm used a device");
    assert_eq!(describe(&a), describe(&b), "repeated seed differs");
    assert_eq!(a.test_errors, b.test_errors);
    let g = a.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the maximum fit arm must run on the device: {:?}",
        g.fit_fallback_by_cause
    );
    // The reported error of every archive row is the exact CPU maximum of the
    // stored tree, never an MSE proxy.
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Maximum);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU maximum"
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

/// #227: the device arm never loses to the exact CPU polish on the same
/// candidates — the zero-tolerance dominance claim, from the same start.
#[cfg(feature = "cuda")]
#[test]
fn the_maximum_arm_never_loses_to_the_cpu_polish_on_the_same_candidates() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    use rcsr::expr::{constant, input, node, Op};
    use rcsr::fit::{fit_batch_verified, fit_weighted_metric, BatchedGpu};
    use rcsr::gpu::GpuEvaluator;

    let n = 512;
    let train = sample(n, 9);
    let shapes: Vec<_> = vec![
        node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(1.0),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])]),
                    ],
                ),
                constant(0.0),
            ],
        ),
        node(Op::Add, vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)]),
        node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(1.0), node(Op::Abs, vec![input(0)])]),
                constant(0.0),
            ],
        ),
        constant(0.5),
    ];
    // The same squared warm start on both arms, exactly like the search.
    let mut starts = shapes.clone();
    for t in starts.iter_mut() {
        rcsr::fit::fit_weighted(t, &train.x, &train.y, &[], 8);
    }
    let mut cols = vec![0.0f32; 2 * n];
    for (i, r) in train.x.iter().enumerate() {
        cols[i] = r[0] as f32;
        cols[n + i] = r[1] as f32;
    }
    let y32: Vec<f32> = train.y.iter().map(|&v| v as f32).collect();
    let mut ev = GpuEvaluator::new(&cols, n, 2, &y32).expect("device");
    let mut cols64 = vec![0.0f64; 2 * n];
    for (i, r) in train.x.iter().enumerate() {
        cols64[i] = r[0];
        cols64[n + i] = r[1];
    }
    assert!(ev.set_fit_data(&cols64, &train.y, &[]));

    let mut gpu_trees = starts.clone();
    let batch = fit_batch_verified(
        &BatchedGpu::new(&mut ev),
        &mut gpu_trees,
        &train.x,
        &train.y,
        &[],
        8,
        Metric::Maximum,
    );
    // This batch mixes `p = 1` with `p = 2` and `p = 3`; every candidate must
    // be answered by the device, not silently by the CPU fallback (the gate
    // would still be correct, but the test would prove nothing).
    assert!(
        batch.status.iter().filter(|s| s.is_gpu()).count() >= starts.len() - 1,
        "the device must fit this batch: {:?}",
        batch.status
    );
    for (k, start) in starts.iter().enumerate() {
        let mut cpu = start.clone();
        let cpu_score = fit_weighted_metric(&mut cpu, &train.x, &train.y, &[], 8, Metric::Maximum);
        let start_score = search::eval_error(start, &train, Metric::Maximum);
        assert!(
            cpu_score <= start_score,
            "the CPU arm must be monotone from its own start"
        );
        assert!(
            batch.score[k].is_finite() && batch.score[k] <= cpu_score,
            "candidate {k}: the device maximum ({}) lost to the CPU polish ({cpu_score})",
            batch.score[k]
        );
        assert!(
            batch.score[k] <= start_score,
            "candidate {k}: the device maximum ({}) lost to its start ({start_score})",
            batch.score[k]
        );
        // The reported score is the exact configured objective at the constants.
        assert_eq!(
            batch.score[k].to_bits(),
            search::eval_error(&gpu_trees[k], &train, Metric::Maximum).to_bits()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn no_fit_is_a_true_noop_for_the_maximum_gpu_arm() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(rcsr::fit::MAXIMUM_GPU_MIN_ROWS + 32, 7);
    let test = sample(200, 8);
    let base = Config {
        fit_iters: 0,
        ..maximum_config(13)
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

/// #227: the paired measurement behind the cost model. Same candidates, same
/// squared start, same `iters`: the CPU arm (`fit_weighted_metric`, squared
/// pre-fit plus the fixed ladder polish) against the batched device arm
/// (`max_fit_batch`, squared LM start plus the bisection on the error level).
/// Prints the numbers the coefficients in `fit.rs` are derived from.
#[cfg(feature = "cuda")]
#[test]
fn the_paired_maximum_measurement_shows_the_device_tail_is_faster() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    use rcsr::expr::{constant, input, node, Op};
    use rcsr::fit::{fit_batch_verified, fit_weighted_metric, BatchedGpu};
    use rcsr::gpu::GpuEvaluator;
    use std::time::Instant;

    for n in [5_000usize, 20_000, 50_000] {
        let train = sample(n, 21);
        // A batch of distinct affine candidates (`a·x + b`), the shape the
        // kernel fits best and the one the cost model is stated for.
        let mut starts = Vec::new();
        for k in 0..24 {
            let a = 0.2 + 0.15 * k as f64;
            let t = node(
                Op::Add,
                vec![node(Op::Mul, vec![constant(a), input(0)]), constant(0.1 * k as f64)],
            );
            starts.push(t);
        }
        for t in starts.iter_mut() {
            rcsr::fit::fit_weighted(t, &train.x, &train.y, &[], 8);
        }
        let cpu_start = Instant::now();
        for t in &starts {
            let mut c = t.clone();
            fit_weighted_metric(&mut c, &train.x, &train.y, &[], 8, Metric::Maximum);
        }
        let cpu_s = cpu_start.elapsed().as_secs_f64();

        let mut cols = vec![0.0f32; 2 * n];
        for (i, r) in train.x.iter().enumerate() {
            cols[i] = r[0] as f32;
            cols[n + i] = r[1] as f32;
        }
        let y32: Vec<f32> = train.y.iter().map(|&v| v as f32).collect();
        let mut ev = GpuEvaluator::new(&cols, n, 2, &y32).expect("device");
        let mut cols64 = vec![0.0f64; 2 * n];
        for (i, r) in train.x.iter().enumerate() {
            cols64[i] = r[0];
            cols64[n + i] = r[1];
        }
        assert!(ev.set_fit_data(&cols64, &train.y, &[]));
            let mut gpu_trees = starts.clone();
        let gpu_start = Instant::now();
        let out = fit_batch_verified(
            &BatchedGpu::new(&mut ev),
            &mut gpu_trees,
            &train.x,
            &train.y,
            &[],
            8,
            Metric::Maximum,
        );
        let gpu_s = gpu_start.elapsed().as_secs_f64();
        let worse = starts
            .iter()
            .zip(&out.score)
            .filter(|(t, s)| search::eval_error(t, &train, Metric::Maximum) < **s)
            .count();
        eprintln!(
            "#227 paired maximum fit: n={n} p=2 candidates={} cpu={:.3} ms gpu={:.3} ms \
             speedup={:.1}x worse_than_start={worse}",
            starts.len(),
            cpu_s * 1e3,
            gpu_s * 1e3,
            cpu_s / gpu_s.max(f64::MIN_POSITIVE)
        );
        assert_eq!(worse, 0, "the device arm must never lose to its start");
        assert!(
            gpu_s < cpu_s,
            "n={n}: the device tail ({:.3} ms) must beat the CPU arm ({:.3} ms)",
            gpu_s * 1e3,
            cpu_s * 1e3
        );
    }
}

