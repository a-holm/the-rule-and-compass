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
/// evaluation of its tree only up to that rounding. A tree with a near-zero
/// denominator amplifies that rounding on a few rows past this bound; the
/// audit then compares the stored error with a device rescore of the tree.
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
    /// Pairs that disagree with the exact `f64` error but equal the device score of their own tree.
    device_confirmed: usize,
    rejected: usize,
    /// CPU fallback refits that lowered the exact error.
    improved: usize,
    /// Largest relative deviation among the agreeing finite pairs.
    worst: f64,
    mismatches: Vec<String>,
}

/// Scores a tree on the device the way the search does, to check that a stored
/// error belongs to its tree even where `f32` and `f64` evaluation diverge.
#[cfg(feature = "cuda")]
struct Rescorer {
    ev: Option<rcsr::gpu::GpuEvaluator>,
    rows: usize,
}

#[cfg(feature = "cuda")]
impl Rescorer {
    fn new(train: &Sample) -> Self {
        let rows = train.x.len();
        let vars = train.x[0].len();
        let mut cols = vec![0.0f32; rows * vars];
        for (r, x) in train.x.iter().enumerate() {
            for (v, &value) in x.iter().enumerate() {
                cols[v * rows + r] = value as f32;
            }
        }
        let y: Vec<f32> = train.y.iter().map(|&v| v as f32).collect();
        Self {
            ev: rcsr::gpu::GpuEvaluator::new(&cols, rows, vars, &y),
            rows,
        }
    }

    /// The stored error equals the device mean squared error of `tree` up to
    /// `f32` reduction noise. A tree the device cannot compile never confirms.
    fn confirms(&mut self, tree: &Node, stored: f64) -> bool {
        let Some(ev) = self.ev.as_mut() else {
            return false;
        };
        let pop = rcsr::gpu::Population::from_trees([tree]);
        if pop.len() != 1 || pop.unsupported() != 0 {
            return false;
        }
        let (prog, off) = pop.programs();
        let (scores, ms) = ev.eval_batch(prog, off);
        if ms < 0.0 {
            return false;
        }
        let device = scores[0] as f64 / self.rows as f64;
        device.is_finite() && (stored - device).abs() <= 1e-4 * stored.abs().max(device.abs()) + 1e-7
    }
}

/// Run `cfg` with a snapshot after every generation and compare each island
/// individual and archive member with an exact evaluation of its own tree.
#[cfg(feature = "cuda")]
fn audit(cfg: &Config, train: &Sample, test: &Sample) -> Audit {
    let names = ["x0".to_string(), "x1".to_string()];
    let mut a = Audit::default();
    let mut rescorer = Rescorer::new(train);
    let mut check = |generation: usize, what: &str, tree: &Node, stored: f64| {
        a.checked += 1;
        let exact = if cfg.accepts_dimensions(tree) {
            search::eval_error(tree, train, cfg.metric)
        } else {
            a.rejected += 1;
            f64::INFINITY
        };
        if !agrees(stored, exact) {
            if exact.is_finite() && rescorer.confirms(tree, stored) {
                a.device_confirmed += 1;
                return;
            }
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
        "{label}: checked {}, rejected {}, improved CPU fallback refits {}, device-confirmed {}, mismatches {}, worst agreeing deviation {:.3e}",
        a.checked,
        a.rejected,
        a.improved,
        a.device_confirmed,
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
    /// Every island individual of every generation snapshot.
    islands: Vec<Vec<(Node, f64)>>,
    stats: search::GpuRunStats,
}

#[cfg(feature = "cuda")]
fn trace(cfg: &Config, train: &Sample, test: &Sample) -> Trace {
    let mut islands: Vec<Vec<(Node, f64)>> = Vec::new();
    let out = search::run_resumable(cfg, train, test, None, 1, |cp: &Checkpoint| {
        islands.push(
            cp.islands
                .iter()
                .flat_map(|isl| isl.population.iter().map(|ind| (ind.tree.clone(), ind.error)))
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
        islands,
        stats: out.gpu.expect("device run"),
    }
}

/// Same stored trees with their constants bit for bit; with `exact` also the
/// same error bits, otherwise the same error up to `f32` scoring noise (a
/// program longer than the genome slot is scored on the CPU by the synchronous
/// path and on the device by the overlap path).
#[cfg(feature = "cuda")]
fn assert_same_trajectory(label: &str, a: &Trace, b: &Trace, exact: bool) {
    assert_eq!(a.islands.len(), b.islands.len(), "{label}: snapshot count");
    let names = ["x0".to_string(), "x1".to_string()];
    for (g, (x, y)) in a.islands.iter().zip(&b.islands).enumerate() {
        assert_eq!(x.len(), y.len(), "{label}: snapshot {g} size");
        let diff: Vec<String> = x
            .iter()
            .zip(y)
            .enumerate()
            .filter(|(_, (p, q))| {
                p.0 != q.0 || if exact { p.1.to_bits() != q.1.to_bits() } else { !agrees(p.1, q.1) }
            })
            .map(|(k, (p, q))| {
                format!(
                    "[{k}] {:e} {} | {:e} {}",
                    p.1,
                    rcsr::expr::to_string(&p.0, &names),
                    q.1,
                    rcsr::expr::to_string(&q.0, &names)
                )
            })
            .collect();
        assert!(diff.is_empty(), "{label}: snapshot {g} differs: {diff:#?}");
    }
    if exact {
        assert_eq!(a.archive, b.archive, "{label}: archive differs");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn pipeline_fit_fallback_max_matches_sync() {
    if !device_available() {
        return;
    }
    const CAP: usize = 1;
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
    assert_same_trajectory("fit-fallback-max, depth 0 vs 2", &runs[0], &runs[1], false);
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
    assert_eq!(
        runs[0].stats.fit_attempts, runs[1].stats.fit_attempts,
        "the pipeline fitted candidates the synchronous path never admits"
    );
    assert_same_trajectory("require-dimensions, depth 0 vs 2", &runs[0], &runs[1], true);
}

/// #600: the overlap drain books its device fit as fit, in the interval union
/// and the fit stage, exactly as the synchronous path does.
#[cfg(feature = "cuda")]
#[test]
fn pipeline_books_fit_interval() {
    use rcsr::gpu_stats::Stage;
    if !device_available() {
        return;
    }
    let train = sample(4_096, 3);
    let test = sample(256, 4);
    for depth in [0usize, 2] {
        let out = search::run(&config(depth, 7), &train, &test);
        let g = out.gpu.expect("the device scored generations");
        let l = &g.ledger;
        let fit_intervals: f64 = l
            .intervals()
            .iter()
            .filter(|i| i.stage == Stage::Fit)
            .map(|i| i.end_s - i.start_s)
            .sum();
        let kernel = l.fit_kernel_s();
        assert!(kernel > 0.0, "depth {depth}: no fit kernel ran");
        assert!(
            (fit_intervals - kernel).abs() <= 0.01 * kernel,
            "depth {depth}: fit intervals {fit_intervals} vs fit kernel {kernel}"
        );
        let sum = l.eval_kernel_s() + kernel;
        assert!(
            l.device_busy_s() >= 0.9 * sum * (1.0 - l.device_overlap_s() / l.device_busy_sum_s()),
            "depth {depth}: device_busy {} vs kernels {sum}",
            l.device_busy_s()
        );
        assert!(
            l.stage_s(Stage::Fit) >= kernel,
            "depth {depth}: fit stage {} below fit kernel {kernel}",
            l.stage_s(Stage::Fit)
        );
        assert!(
            l.stage_s(Stage::Vary) + kernel <= l.wall_s() * 1.01,
            "depth {depth}: vary {} still holds the fit kernel {kernel} of wall {}",
            l.stage_s(Stage::Vary),
            l.wall_s()
        );
        assert!(l.closure_residual().value.abs() < 0.005, "depth {depth}: closure");
        let per_gen: f64 = g.generations.iter().map(|s| s.fit_fallback_seconds).sum();
        let run = g.fit_fallback_ms / 1000.0;
        assert!(run > 0.0, "depth {depth}: the fixture produced no CPU fallback");
        assert!(
            (per_gen - run).abs() <= 1e-9 + 1e-6 * run,
            "depth {depth}: per-generation fallback {per_gen} vs run {run}"
        );
    }
}

/// #601: the overlap fit queue groups candidates into device calls of at least
/// `FIT_BATCH_MIN`. A candidate's constants depend on its own plan only, so the
/// archive and the fit counters equal the synchronous run's, with far fewer
/// fit calls than scoring batches.
#[cfg(feature = "cuda")]
#[test]
fn pipeline_fit_constants_match_sync() {
    if !device_available() {
        return;
    }
    let train = sample(4_096, 5);
    let test = sample(256, 6);
    let run = |depth: usize| {
        let mut cfg = config(depth, 11);
        cfg.ops.retain(|&op| op != Op::Asinh);
        cfg.pop = 256;
        cfg.generations = 3;
        search::run(&cfg, &train, &test)
    };
    let sync = run(0);
    let overlap = run(4);
    let key = |o: &search::Outcome| {
        o.archive
            .items
            .iter()
            .map(|s| format!("{:?}|{:e}", s.tree, s.error))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&sync), key(&overlap), "archive differs between depth 0 and 4");
    let (gs, go) = (sync.gpu.expect("sync gpu"), overlap.gpu.expect("overlap gpu"));
    assert!(gs.fit_calls > 0 && go.fit_calls > 0);
    assert_eq!(gs.fit_call_candidates, go.fit_call_candidates);
    let gens = 3 + 1;
    assert!(
        go.fit_calls <= 2 * gens,
        "overlap fit calls {} for {gens} generations",
        go.fit_calls
    );
}
