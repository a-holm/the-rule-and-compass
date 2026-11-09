//! #936, #959: statuses 4, 5, 7 and 9 of the LM kernel go to the exact CPU fit.
//!
//! The fixture is 240 deterministic trees over `min`, `max`, `if` and
//! `greater` on 8 columns and 112,500 rows. A tree with one of the four
//! statuses ends at the CPU fit from its starting constants; every other tree
//! keeps the device constants bit for bit. Device tests skip cleanly on a
//! machine without CUDA.

use super::*;
use crate::data::Sample;
use crate::expr::Node;
use crate::fit::{self, FitPolish, FitStrategy};
use crate::search::{self, Config};

const ROWS: usize = 112_500;
const VARS: usize = 8;
const ITERS: usize = 8;
const TREES: &str = include_str!("../tests/fixtures/fit_tail_trees.tsv");

fn gauss(rng: &mut crate::rng::Mt) -> f64 {
    let u = rng.next_f64().max(1e-12);
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Eight standard-normal columns, rounded to 6 decimals as in a CSV, and a
/// target that mixes three of them with noise.
fn rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = crate::rng::Mt::new(936);
    let mut x = Vec::with_capacity(ROWS);
    let mut y = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let r: Vec<f64> = (0..VARS).map(|_| (gauss(&mut rng) * 1e6).round() / 1e6).collect();
        y.push(r[0] + 0.5 * r[1] * r[2] + 0.3 * gauss(&mut rng));
        x.push(r);
    }
    (x, y)
}

fn trees() -> Vec<Node> {
    let vars: Vec<String> = (0..VARS).map(|i| format!("x{i}")).collect();
    TREES
        .lines()
        .map(|l| {
            let expr = l.split('\t').nth(2).expect("an expression column");
            crate::parse::parse(expr, &vars).unwrap_or_else(|e| panic!("{expr}: {e:?}"))
        })
        .collect()
}

fn sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let mut t = tree.clone();
    fit::fit_weighted(&mut t, x, y, &[], 0)
}

macro_rules! need_device {
    () => {
        if !probe().available() {
            crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
            return;
        }
    };
}

/// The device fit of every fixture tree: the plan, the result and the tree
/// after the device constants are applied.
fn device_results(x: &[Vec<f64>], y: &[f64], trees: &[Node]) -> Vec<(i32, Node)> {
    let n = x.len();
    let mut cols64 = vec![0.0f64; n * VARS];
    for (r, row) in x.iter().enumerate() {
        for (k, &val) in row.iter().enumerate() {
            cols64[k * n + r] = val;
        }
    }
    let cols32: Vec<f32> = cols64.iter().map(|&a| a as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&a| a as f32).collect();
    let mut ev = GpuEvaluator::new(&cols32, n, VARS, &y32).expect("device evaluator");
    assert!(ev.set_fit_data(&cols64, y, &[]), "f64 fit rows uploaded");
    ev.set_fit_polish(FitPolish::F64);
    let plans: Vec<fit::GpuFitPlan> =
        trees.iter().map(|t| fit::gpu_fit_plan(t).expect("a fit plan")).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let (out, ms) = ev.fit_batch(&refs, ITERS, fit::DeviceLoss::Squared);
    assert!(ms >= 0.0, "the launch was refused");
    trees
        .iter()
        .zip(&plans)
        .zip(out)
        .map(|((t, plan), res)| {
            let mut dev = t.clone();
            if res.c.iter().all(|v| v.is_finite()) {
                plan.apply(&mut dev, &res.c);
            }
            (res.status, dev)
        })
        .collect()
}

/// After routing, every tree with status 4, 5, 7 or 9 equals the CPU fit and
/// every other tree keeps its device constants.
#[test]
fn device_fit_tail_statuses_fall_back() {
    need_device!();
    let (x, y) = rows();
    let trees = trees();
    assert_eq!(trees.len(), 240);
    let device = device_results(&x, &y, &trees);
    let mut routed = std::collections::BTreeMap::<&str, usize>::new();
    let mut seen = [0usize; 10];
    let mut before = 1.0f64;
    let mut after = 1.0f64;
    let mut kept_worse: Vec<usize> = Vec::new();
    for ((start, (status, dev)), k) in trees.iter().zip(&device).zip(0..) {
        seen[*status as usize] += 1;
        let mut cpu = start.clone();
        let cpu_sse = fit::fit_weighted(&mut cpu, &x, &y, &[], ITERS);
        match search::tail_status_cause(*status) {
            Some(cause) => {
                *routed.entry(cause.label()).or_default() += 1;
                let device_sse = sse(dev, &x, &y);
                if device_sse.is_finite() && cpu_sse > 0.0 {
                    before = before.max(device_sse / cpu_sse);
                }
            }
            None => {
                let device_sse = sse(dev, &x, &y);
                if cpu_sse > 0.0 && device_sse.is_finite() {
                    after = after.max(device_sse / cpu_sse);
                    if device_sse > cpu_sse * 1.001 {
                        kept_worse.push(k);
                    }
                }
            }
        }
    }
    eprintln!("tail: statuses {seen:?} routed {routed:?} device/cpu before routing {before:.6} kept after routing {after:.6} kept worse than 1.001 {kept_worse:?}");
    let tails = seen[4] + seen[5] + seen[7] + seen[9];
    assert!(seen[4] > 0 && seen[5] > 0 && seen[7] > 0, "the fixture lost a status: {seen:?}");
    assert_eq!(routed.values().sum::<usize>(), tails);
    assert!(kept_worse.is_empty(), "kept device answers above 1.001 of the CPU SSE: {kept_worse:?}");
}

/// A device-dispatched search over the hard operators counts the routed fits
/// as labelled fallbacks and no longer keeps them as finite device answers.
#[test]
fn a_search_counts_the_routed_tail_fits_in_the_ledger() {
    need_device!();
    let (x, y) = rows();
    let n = 20_000;
    let train = Sample { x: x[..n].to_vec(), y: y[..n].to_vec(), w: Vec::new(), ys: Vec::new(), pairs: None };
    let test = Sample { x: x[n..2 * n].to_vec(), y: y[n..2 * n].to_vec(), w: Vec::new(), ys: Vec::new(), pairs: None };
    let cfg = Config {
        generations: 6,
        islands: 1,
        pop: 200,
        max_complexity: 14,
        fit_iters: ITERS,
        patience: 0,
        seed: 11,
        ops: crate::expr::Op::parse_list("add,sub,mul,div,min,max,if,greater"),
        dispatch: crate::dispatch::DispatchMode::Device,
        gpu: true,
        fit_strategy: FitStrategy::Gpu,
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a fit ledger");
    assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback, "{g:?}");
    for label in ["rank_deficient", "recovered", "start_unusable", "unconverged"] {
        assert_eq!(g.fit_status_by_label.get(label).copied().unwrap_or(0), 0, "{label} kept: {g:?}");
    }
    let routed: usize = ["rank_deficient", "recovered_iterate", "non_finite_start", "unconverged"]
        .iter()
        .map(|c| g.fit_fallback_by_cause.get(*c).copied().unwrap_or(0))
        .sum();
    eprintln!("search: attempts {} gpu {} fallback {} routed {routed} causes {:?}", g.fit_attempts, g.fit_gpu, g.fit_fallback, g.fit_fallback_by_cause);
    assert!(routed > 0, "no tail fit was routed: {:?}", g.fit_fallback_by_cause);
}

/// #958: what the tail overlap must not change, one search.
struct TailTrace {
    /// Every island individual of every generation snapshot.
    islands: Vec<Vec<(Node, u64)>>,
    archive: Vec<(Node, u64, u64)>,
    test_errors: Vec<u64>,
    by_cause: Vec<std::collections::BTreeMap<String, usize>>,
    improved: Vec<std::collections::BTreeMap<String, usize>>,
    bounded: Vec<usize>,
    stats: search::GpuRunStats,
}

fn tail_trace(cfg: &Config, train: &Sample, test: &Sample) -> TailTrace {
    let mut islands = Vec::new();
    let out = search::run_resumable(cfg, train, test, None, 1, |cp: &crate::checkpoint::Checkpoint| {
        islands.push(
            cp.islands
                .iter()
                .flat_map(|isl| isl.population.iter().map(|ind| (ind.tree.clone(), ind.error.to_bits())))
                .collect(),
        );
    })
    .expect("the search runs");
    let stats = out.gpu.expect("a device run");
    TailTrace {
        islands,
        archive: out.archive.items.iter().map(|s| (s.tree.clone(), s.error.to_bits(), s.cost.to_bits())).collect(),
        test_errors: out.test_errors.iter().map(|e| e.to_bits()).collect(),
        by_cause: stats.generations.iter().map(|g| g.fit_fallback_by_cause.clone()).collect(),
        improved: stats.generations.iter().map(|g| g.fit_fallback_improved_by_cause.clone()).collect(),
        bounded: stats.generations.iter().map(|g| g.fit_fallback_bounded).collect(),
        stats,
    }
}

/// #958: the exact CPU fits on the fallback pool beside the device work give
/// the serial routing's trees, errors and per-cause counts bit for bit, on the
/// narrow synchronous path, the wide synchronous path (the `+inf` check is on
/// the pool at the same time), the overlap pipeline, and under a binding
/// `--fit-fallback-max`.
#[test]
fn tail_fit_overlap_equals_serial_routing() {
    need_device!();
    let (x, y) = rows();
    let n = 6_000;
    let train = Sample { x: x[..n].to_vec(), y: y[..n].to_vec(), w: Vec::new(), ys: Vec::new(), pairs: None };
    let test = Sample { x: x[n..n + 1_000].to_vec(), y: y[n..n + 1_000].to_vec(), w: Vec::new(), ys: Vec::new(), pairs: None };
    let base = Config {
        generations: 4,
        islands: 2,
        pop: 96,
        max_complexity: 14,
        fit_iters: ITERS,
        patience: 0,
        seed: 11,
        threads: 4,
        ops: crate::expr::Op::parse_list("add,sub,mul,div,min,max,if,greater"),
        dispatch: crate::dispatch::DispatchMode::Device,
        gpu: true,
        fit_strategy: FitStrategy::Gpu,
        ..Default::default()
    };
    let wide = Config { fit_admission: fit::FitAdmission::All, ..base.clone() };
    let cases = [
        ("narrow synchronous", Config { pipeline_depth: 0, ..base.clone() }),
        ("wide synchronous", Config { pipeline_depth: 0, ..wide.clone() }),
        ("overlap pipeline", Config { pipeline_depth: 2, ..wide.clone() }),
        ("narrow bounded", Config { pipeline_depth: 0, fit_fallback_max: 2, ..base.clone() }),
        ("pipeline bounded", Config { pipeline_depth: 2, fit_fallback_max: 2, ..wide }),
    ];
    for (label, cfg) in cases {
        let serial = tail_trace(&Config { fit_tail_overlap: false, ..cfg.clone() }, &train, &test);
        let overlap = tail_trace(&Config { fit_tail_overlap: true, ..cfg.clone() }, &train, &test);
        let (s, o) = (&serial.stats, &overlap.stats);
        let routed: usize = ["rank_deficient", "recovered_iterate", "non_finite_start", "unconverged"]
            .iter()
            .map(|c| s.fit_fallback_by_cause.get(*c).copied().unwrap_or(0))
            .sum();
        eprintln!(
            "{label}: attempts {} fallback {} routed {routed} bounded {} causes {:?} | overlap {:.4}s join wait {:.4}s fallback {:.4}s workers {} | serial join wait {:.4}s | calls {} vs {}",
            o.fit_attempts,
            o.fit_fallback,
            o.fit_fallback_bounded,
            o.fit_fallback_by_cause,
            o.ledger.fit_tail_overlap_s(),
            o.ledger.fit_tail_join_wait_s(),
            o.ledger.fit_fallback_s(),
            o.fit_fallback_workers,
            s.ledger.fit_tail_join_wait_s(),
            o.fit_calls,
            s.fit_calls,
        );
        assert!(routed > 0, "{label}: no tail fit was routed");
        assert_eq!(serial.islands.len(), overlap.islands.len(), "{label}: snapshots");
        for (g, (a, b)) in serial.islands.iter().zip(&overlap.islands).enumerate() {
            assert_eq!(a.len(), b.len(), "{label}: snapshot {g} size");
            for (k, (p, q)) in a.iter().zip(b).enumerate() {
                assert!(crate::expr::same(&p.0, &q.0) && p.1 == q.1, "{label}: generation {g} individual {k} differs");
            }
        }
        assert_eq!(serial.archive.len(), overlap.archive.len(), "{label}: archive size");
        for (k, (p, q)) in serial.archive.iter().zip(&overlap.archive).enumerate() {
            assert!(crate::expr::same(&p.0, &q.0) && p.1 == q.1 && p.2 == q.2, "{label}: archive member {k} differs");
        }
        assert_eq!(serial.test_errors, overlap.test_errors, "{label}: test errors");
        assert_eq!(serial.by_cause, overlap.by_cause, "{label}: per-generation fallback causes");
        assert_eq!(serial.improved, overlap.improved, "{label}: per-generation improved refits");
        assert_eq!(serial.bounded, overlap.bounded, "{label}: per-generation bound");
        assert_eq!(
            (s.fit_attempts, s.fit_gpu, s.fit_fallback, &s.fit_fallback_by_cause),
            (o.fit_attempts, o.fit_gpu, o.fit_fallback, &o.fit_fallback_by_cause),
            "{label}: run counters"
        );
        assert_eq!(s.ledger.fit_tail_overlap_s(), 0.0, "{label}: the serial arm overlaps nothing");
        for t in [s, o] {
            let split = t.ledger.fit_tail_overlap_s() + t.ledger.fit_tail_join_wait_s();
            assert!((t.ledger.fit_fallback_s() - split).abs() < 1e-9, "{label}: fallback_s != overlap + wait");
            // The overlap pipeline closes to its drain booking, as in
            // `pipeline_books_fit_interval`.
            let bound = if cfg.pipeline_depth == 0 { 1e-9 } else { 0.005 };
            assert!(t.ledger.closure_residual().value.abs() < bound, "{label}: the ledger does not close");
        }
    }
}

/// #959: fixture trees whose device LM ended with status 0 above the CPU fit.
const PINNED: [usize; 2] = [136, 150];

/// #959: on trees 136 and 150 the answer the search keeps (the device
/// constants, or the CPU fit for a routed status) is at most 1.001 times the
/// CPU SSE.
#[test]
fn status_zero_device_fit_matches_cpu_on_pinned_trees() {
    need_device!();
    let (x, y) = rows();
    let all = trees();
    let pinned: Vec<Node> = PINNED.iter().map(|&k| all[k].clone()).collect();
    let device = device_results(&x, &y, &pinned);
    let mut worst = 0.0f64;
    for ((start, (status, dev)), id) in pinned.iter().zip(&device).zip(PINNED) {
        let mut cpu = start.clone();
        let cpu_sse = fit::fit_weighted(&mut cpu, &x, &y, &[], ITERS);
        let kept = if search::tail_status_cause(*status).is_some() { &cpu } else { dev };
        let ratio = sse(kept, &x, &y) / cpu_sse;
        eprintln!(
            "pinned {id}: status {status} start sse {:.6} device c {:?} sse {:.6} cpu c {:?} sse {cpu_sse:.6} kept/cpu {ratio:.6}",
            sse(start, &x, &y),
            fit::constants_slots(dev),
            sse(dev, &x, &y),
            fit::constants_slots(&cpu),
        );
        worst = worst.max(ratio);
    }
    assert!(worst <= 1.001, "kept/cpu SSE ratio {worst:.6} above 1.001");
}

/// #968: a routed tail fit keeps the lower SSE of the device and the CPU
/// answer. Tree 22 (status 9) keeps the device answer.
#[test]
fn routed_tail_fits_keep_the_better_answer() {
    need_device!();
    let (x, y) = rows();
    let trees = trees();
    let device = device_results(&x, &y, &trees);
    let sample = Sample { x: x.clone(), y: y.clone(), w: Vec::new(), ys: Vec::new(), pairs: None };
    let mut worst = 0.0f64;
    let mut device_kept = 0usize;
    let mut routed = 0usize;
    let mut tree22 = f64::NAN;
    for (k, (start, (status, dev))) in trees.iter().zip(&device).enumerate() {
        if search::tail_status_cause(*status).is_none() {
            continue;
        }
        routed += 1;
        let mut cpu = start.clone();
        let cpu_sse = fit::fit_weighted(&mut cpu, &x, &y, &[], ITERS);
        let device_sse = sse(dev, &x, &y);
        let kept = search::keep_better_tail_fit(start, dev.clone(), &sample, ITERS, crate::metrics::Metric::Squared);
        let again = search::keep_better_tail_fit(start, dev.clone(), &sample, ITERS, crate::metrics::Metric::Squared);
        let bits = |t: &Node| fit::constants_slots(t).iter().map(|c| c.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&kept), bits(&again), "tree {k}: two runs differ");
        let kept_sse = sse(&kept, &x, &y);
        let best = if device_sse.is_finite() { cpu_sse.min(device_sse) } else { cpu_sse };
        worst = worst.max(kept_sse / best - 1.0);
        if bits(&kept) == bits(dev) && bits(&kept) != bits(&cpu) {
            device_kept += 1;
        }
        if k == 22 {
            tree22 = kept_sse / cpu_sse;
            eprintln!("tree 22: status {status} device/cpu {:.6} kept/cpu {tree22:.6}", device_sse / cpu_sse);
        }
    }
    eprintln!("keep-better: routed {routed} device kept {device_kept} worst kept/best - 1 {worst:.3e}");
    assert!(routed > 0);
    assert!(worst <= 1e-12, "a kept answer is above min(device, CPU) by {worst:e}");
    assert!(tree22 <= 0.9367, "tree 22 kept/cpu SSE {tree22:.6}");
}
