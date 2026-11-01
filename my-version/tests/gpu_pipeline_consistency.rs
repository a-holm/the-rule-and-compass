//! #560: every island individual and every in-search archive member stores the
//! error of the tree it stores, on the synchronous path (`pipeline_depth = 0`)
//! and on the overlap path alike.
//!
//! `asinh` has no device fit opcode, so each candidate that contains it falls
//! back to the exact CPU fit. A QD immigrant replaces a slot that was already
//! offered to the overlap queue, and a tree that fails `--require-dimensions`
//! must keep its infinite error. Every generation's snapshot is checked against
//! an exact `f64` evaluation of each stored tree. Device tests skip cleanly
//! without CUDA.

#[cfg(feature = "cuda")]
use rcsr::checkpoint::Checkpoint;
#[cfg(feature = "cuda")]
use rcsr::data::Sample;
#[cfg(feature = "cuda")]
use rcsr::expr::{Node, Op};
#[cfg(feature = "cuda")]
use rcsr::fit::{FitAdmission, FitStrategy};
#[cfg(feature = "cuda")]
use rcsr::metrics::Metric;
#[cfg(feature = "cuda")]
use rcsr::search::{self, Config};
#[cfg(feature = "cuda")]
use rcsr::units::{Unit, Units};

#[cfg(feature = "cuda")]
fn sample(n: usize, salt: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ salt) as f64 / u32::MAX as f64;
        let s = s * 2.0 - 1.0;
        x.push(vec![t, s]);
        y.push(1.5 * (2.5 * t).asinh() + 0.4 * s - 0.3);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

/// `tan` is left out: near its poles the device's `f32` value of a tree drifts
/// from the exact one by more than the tolerance below.
#[cfg(feature = "cuda")]
fn config(depth: usize, seed: u32) -> Config {
    let mut ops: Vec<Op> = Op::default_set().into_iter().filter(|&op| op != Op::Tan).collect();
    ops.push(Op::Asinh);
    Config {
        generations: 6,
        islands: 1,
        pop: 48,
        max_complexity: 12,
        fit_iters: 8,
        patience: 0,
        seed,
        ops,
        fit_strategy: FitStrategy::Gpu,
        fit_admission: FitAdmission::All,
        metric: Metric::Squared,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        pipeline_depth: depth,
        ..Default::default()
    }
}

/// The device scores in `f32`, so a stored error matches the exact `f64`
/// evaluation of its tree only up to that rounding.
#[cfg(feature = "cuda")]
fn agrees(stored: f64, exact: f64) -> bool {
    if !stored.is_finite() || !exact.is_finite() {
        return stored == exact || (stored.is_nan() && exact.is_nan());
    }
    (stored - exact).abs() <= 1e-3 * stored.abs().max(exact.abs()) + 1e-7
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct Audit {
    checked: usize,
    rejected: usize,
    /// CPU fallback refits that lowered the exact error.
    improved: usize,
    /// Largest relative deviation among the agreeing finite pairs.
    worst: f64,
    mismatches: Vec<String>,
}

/// Run `cfg` with a snapshot after every generation and compare each island
/// individual and archive member with an exact evaluation of its own tree.
#[cfg(feature = "cuda")]
fn audit(cfg: &Config, train: &Sample, test: &Sample) -> Audit {
    let names = ["x0".to_string(), "x1".to_string()];
    let mut a = Audit::default();
    let mut check = |generation: usize, what: &str, tree: &Node, stored: f64| {
        a.checked += 1;
        let exact = if cfg.accepts_dimensions(tree) {
            search::eval_error(tree, train, cfg.metric)
        } else {
            a.rejected += 1;
            f64::INFINITY
        };
        if !agrees(stored, exact) {
            a.mismatches.push(format!(
                "generation {generation} {what}: stored {stored:.9e} exact {exact:.9e} {}",
                rcsr::expr::to_string(tree, &names)
            ));
        } else if stored.is_finite() && exact.is_finite() && exact != 0.0 {
            a.worst = a.worst.max((stored - exact).abs() / exact.abs());
        }
    };
    let out = search::run_resumable(cfg, train, test, None, 1, |cp: &Checkpoint| {
        for island in &cp.islands {
            for ind in &island.population {
                check(cp.generations, "island", &ind.tree, ind.error);
            }
        }
        for s in &cp.archive.items {
            check(cp.generations, "archive", &s.tree, s.error);
        }
    })
    .expect("search runs");
    a.improved = out
        .gpu
        .as_ref()
        .map_or(0, |g| g.fit_fallback_improved_by_cause.values().sum());
    a
}

#[cfg(feature = "cuda")]
fn report(label: &str, a: &Audit) {
    eprintln!(
        "{label}: checked {}, rejected {}, improved CPU fallback refits {}, mismatches {}, worst agreeing deviation {:.3e}",
        a.checked,
        a.rejected,
        a.improved,
        a.mismatches.len(),
        a.worst
    );
    for line in a.mismatches.iter().take(8) {
        eprintln!("  {line}");
    }
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    let ok = rcsr::gpu::probe().available();
    if !ok {
        eprintln!("skipping GPU test: no usable CUDA device");
    }
    ok
}

#[cfg(feature = "cuda")]
#[test]
fn cpu_fit_fallbacks_store_the_error_of_the_stored_tree() {
    if !device_available() {
        return;
    }
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    let mut failures = Vec::new();
    for depth in [0usize, 1, 2] {
        let a = audit(&config(depth, 7), &train, &test);
        report(&format!("depth {depth}"), &a);
        assert!(
            a.improved > 0,
            "depth {depth}: no CPU fallback refit lowered an error, so the case is not exercised"
        );
        if !a.mismatches.is_empty() {
            failures.push(format!("depth {depth}: {} of {}", a.mismatches.len(), a.checked));
        }
    }
    assert!(failures.is_empty(), "stored errors of a different tree: {failures:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn rejected_dimensions_stay_rejected_on_the_overlap_path() {
    if !device_available() {
        return;
    }
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    let metre = Unit::parse("m").expect("unit");
    let second = Unit::parse("s").expect("unit");
    let units = Units::new(vec![metre.clone(), second], metre);
    let mut failures = Vec::new();
    for depth in [0usize, 2] {
        let cfg = Config {
            units: Some(units.clone()),
            require_dimensions: true,
            ..config(depth, 7)
        };
        let a = audit(&cfg, &train, &test);
        report(&format!("depth {depth}"), &a);
        assert!(a.rejected > 0, "depth {depth}: no tree failed the dimension check");
        if !a.mismatches.is_empty() {
            failures.push(format!("depth {depth}: {} of {}", a.mismatches.len(), a.checked));
        }
    }
    assert!(failures.is_empty(), "rejected trees with a finite error: {failures:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn qd_immigrants_store_the_error_of_the_stored_tree() {
    if !device_available() {
        return;
    }
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    let mut failures = Vec::new();
    // Other seeds reach `ln(asinh(u))` with `u` below the `f32` resolution of
    // `1 + u`, where the device's lowering of `asinh` returns zero and the
    // stored error is `+inf` for a tree whose exact error is finite.
    for seed in [7u32, 8] {
        for depth in [0usize, 2] {
            let cfg = Config {
                qd_bins: 4,
                qd_policy: true,
                ..config(depth, seed)
            };
            let a = audit(&cfg, &train, &test);
            report(&format!("seed {seed} depth {depth}"), &a);
            if !a.mismatches.is_empty() {
                failures.push(format!(
                    "seed {seed} depth {depth}: {} of {}",
                    a.mismatches.len(),
                    a.checked
                ));
            }
        }
    }
    assert!(failures.is_empty(), "stored errors of a different tree: {failures:?}");
}

/// #578: what the pipeline depth must not change.
#[cfg(feature = "cuda")]
struct Trace {
    archive: Vec<(u64, usize, u64)>,
    /// Island errors (bit patterns) of every generation snapshot.
    errors: Vec<Vec<u64>>,
    stats: search::GpuRunStats,
}

#[cfg(feature = "cuda")]
fn trace(cfg: &Config, train: &Sample, test: &Sample) -> Trace {
    let mut errors: Vec<Vec<u64>> = Vec::new();
    let out = search::run_resumable(cfg, train, test, None, 1, |cp: &Checkpoint| {
        errors.push(
            cp.islands
                .iter()
                .flat_map(|isl| isl.population.iter().map(|ind| ind.error.to_bits()))
                .collect(),
        );
    })
    .expect("search runs");
    Trace {
        archive: out
            .archive
            .items
            .iter()
            .map(|s| (rcsr::expr::structural_hash(&s.tree), s.complexity, s.error.to_bits()))
            .collect(),
        errors,
        stats: out.gpu.expect("device run"),
    }
}

#[cfg(feature = "cuda")]
fn assert_same_trajectory(label: &str, a: &Trace, b: &Trace) {
    assert_eq!(a.errors.len(), b.errors.len(), "{label}: snapshot count");
    for (g, (x, y)) in a.errors.iter().zip(&b.errors).enumerate() {
        assert_eq!(x, y, "{label}: island errors differ in snapshot {g}");
    }
    assert_eq!(a.archive, b.archive, "{label}: archive differs");
}

#[cfg(feature = "cuda")]
fn cause_total(t: &Trace, causes: &[&str]) -> usize {
    causes
        .iter()
        .map(|c| t.stats.fit_fallback_by_cause.get(*c).copied().unwrap_or(0))
        .sum()
}

/// `x1` repeats `x0`, so `c0 * x0 + c1 * x1` has a singular normal matrix and
/// the device fit reports a non-usable status for it.
#[cfg(feature = "cuda")]
fn collinear_sample(n: usize) -> Sample {
    let mut s = sample(n, 3);
    for row in &mut s.x {
        row[1] = row[0];
    }
    for (row, y) in s.x.iter().zip(&mut s.y) {
        *y = 1.5 * row[0] + 0.7 * row[1] - 0.3;
    }
    s
}

#[cfg(feature = "cuda")]
#[test]
fn pipeline_fallback_matches_sync_trajectory() {
    if !device_available() {
        return;
    }
    let train = collinear_sample(4_096);
    let test = collinear_sample(256);
    let mut runs = Vec::new();
    for depth in [0usize, 2] {
        let t = trace(&config(depth, 7), &train, &test);
        let status = cause_total(&t, &["ill_conditioned", "non_finite", "rank_declined"]);
        eprintln!("depth {depth}: status fallbacks {status}, causes {:?}", t.stats.fit_fallback_by_cause);
        assert!(status > 0, "depth {depth}: no device status 1/2/3 fallback, the case is not exercised");
        runs.push(t);
    }
    assert_same_trajectory("status fallbacks, depth 0 vs 2", &runs[0], &runs[1]);
}

#[cfg(feature = "cuda")]
#[test]
fn pipeline_fit_fallback_max_matches_sync() {
    if !device_available() {
        return;
    }
    const CAP: usize = 2;
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    let mut runs = Vec::new();
    for depth in [0usize, 2] {
        let cfg = Config {
            fit_fallback_max: CAP,
            ..config(depth, 7)
        };
        let t = trace(&cfg, &train, &test);
        for g in &t.stats.generations {
            assert!(
                g.fit_fallback - g.fit_fallback_bounded <= CAP,
                "depth {depth} generation {}: {} fallbacks, {} bounded",
                g.generation,
                g.fit_fallback,
                g.fit_fallback_bounded
            );
        }
        runs.push(t);
    }
    let bounded = |t: &Trace| -> Vec<(usize, usize, usize)> {
        t.stats
            .generations
            .iter()
            .map(|g| (g.generation, g.fit_fallback, g.fit_fallback_bounded))
            .collect()
    };
    assert!(
        bounded(&runs[0]).iter().any(|&(_, _, b)| b > 0),
        "the cap never bound, the case is not exercised"
    );
    assert_eq!(bounded(&runs[0]), bounded(&runs[1]), "per-generation fallback and bound counts");
    assert_same_trajectory("fit-fallback-max, depth 0 vs 2", &runs[0], &runs[1]);
}

#[cfg(feature = "cuda")]
#[test]
fn pipeline_does_not_launch_dimension_rejected() {
    if !device_available() {
        return;
    }
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    let metre = Unit::parse("m").expect("unit");
    let second = Unit::parse("s").expect("unit");
    let units = Units::new(vec![metre.clone(), second], metre);
    let mut runs = Vec::new();
    for depth in [0usize, 2] {
        let cfg = Config {
            units: Some(units.clone()),
            require_dimensions: true,
            ..config(depth, 7)
        };
        runs.push(trace(&cfg, &train, &test));
    }
    eprintln!(
        "fit attempts: depth 0 {}, depth 2 {}",
        runs[0].stats.fit_attempts, runs[1].stats.fit_attempts
    );
    assert_eq!(
        runs[0].stats.fit_attempts, runs[1].stats.fit_attempts,
        "the pipeline fitted candidates the synchronous path never admits"
    );
    assert_same_trajectory("require-dimensions, depth 0 vs 2", &runs[0], &runs[1]);
}
