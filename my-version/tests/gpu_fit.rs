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
    let train = sample(n, 21);
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
        metric: Metric::Squared,
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
            "arm={} N={} evals={} wall={:.2}s best={:.6e}",
            name,
            n,
            base.generations * base.islands * base.pop,
            wall,
            best
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
            eprintln!(
                "  time: fit_host={:.2}s ({:.1}% of wall) kernel={:.2}s duty={:.1}%",
                g.fit_host_ms / 1000.0,
                100.0 * (g.fit_host_ms / 1000.0) / wall,
                g.fit_ms / 1000.0,
                100.0 * g.accounting.duty_cycle()
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
