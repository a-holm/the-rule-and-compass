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
    for seed in [7u32, 8, 9] {
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
