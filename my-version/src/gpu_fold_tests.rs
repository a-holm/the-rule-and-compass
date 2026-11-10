//! #982: the device fit stage folds redundant constants before the fit.
//!
//! A tree whose constants trade off against one another is rank deficient on
//! the device and goes to the exact CPU fit. The fold gives it one constant
//! per foldable direction, so the device fits it; a tree with nothing to fold
//! keeps the device fit it had. Device tests skip cleanly without CUDA.

use super::*;
use crate::expr::{constant, input, node, Op};

macro_rules! need_device {
    () => {
        if !crate::gpu::probe().available() {
            crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
            return;
        }
    };
}

/// Two columns and a target close to `0.7*x0 - 0.4`, deterministic.
pub(super) fn sample(n: usize) -> Sample {
    let x: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let t = r as f64;
            vec![1.5 * (0.37 * t).sin(), (0.11 * t).cos()]
        })
        .collect();
    let y = x.iter().map(|r| 0.7 * r[0] - 0.4 + 0.05 * (3.0 * r[1]).sin()).collect();
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

pub(super) fn config() -> Config {
    Config {
        metric: Metric::Squared,
        gpu: true,
        fit_strategy: crate::fit::FitStrategy::Gpu,
        dispatch: crate::dispatch::DispatchMode::Device,
        ..Default::default()
    }
}

fn device(train: &Sample) -> GpuSearch {
    GpuSearch::new(
        train,
        true,
        crate::fit::DeviceLoss::Squared,
        true,
        None,
        Some(crate::metrics::ScoreObjective::Squared),
        false,
        std::time::Instant::now(),
        &[],
        None,
        None,
        None,
        2,
    )
    .expect("a device search")
}

fn constants(t: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        n.kids.iter().for_each(|k| walk(k, out));
    }
    let mut out = Vec::new();
    walk(t, &mut out);
    out
}

pub(super) fn fit_on_device(trees: &[Node], train: &Sample, cfg: &Config) -> (GpuSearch, Vec<Node>) {
    let mut g = device(train);
    let mut islands = vec![trees.iter().map(|t| Ind::pending(t.clone(), cfg)).collect::<Vec<_>>()];
    let admitted: Vec<(usize, usize)> = (0..trees.len()).map(|j| (0, j)).collect();
    let mut dispatcher = crate::dispatch::Dispatcher::new(crate::dispatch::Calibration::default())
        .with_mode(crate::dispatch::DispatchMode::Device);
    g.fit_pending(&mut islands, &admitted, train, cfg, 0, &mut dispatcher);
    let fitted = islands.remove(0).into_iter().map(|ind| ind.tree).collect();
    (g, fitted)
}

/// `0.5*x + 0.1` written in original units on an input with mean `0.3` and
/// standard deviation `1.7` is `0.5*(z*1.7 + 0.3) + 0.1` on the standardised
/// input `z` (`--seed-expr`). Its four constants span two directions.
#[test]
fn a_standardised_seed_folds_and_fits_on_the_device() {
    need_device!();
    let train = sample(4096);
    let cfg = config();
    let affine = node(Op::Add, vec![node(Op::Mul, vec![input(0), constant(1.7)]), constant(0.3)]);
    let seed = node(Op::Add, vec![node(Op::Mul, vec![constant(0.5), affine]), constant(0.1)]);
    let (g, fitted) = fit_on_device(std::slice::from_ref(&seed), &train, &cfg);
    assert_eq!(g.fit_fallback_by_cause.get("rank_deficient"), None, "{:?}", g.fit_fallback_by_cause);
    assert_eq!((g.fit_gpu, g.fit_fallback), (1, 0), "{:?}", g.fit_fallback_by_cause);
    assert_eq!((g.fit_fold_count, g.fit_folded), (2, 1));
    let stats = g.run_stats();
    assert_eq!((stats.fit_fold_count, stats.fit_folded), (2, 1));
    // The device constants are those of the exact CPU fit of the folded tree.
    let mut cpu = crate::fold::fold_constants(&seed).0;
    fit_constants(&mut cpu, &cfg, &train, &train).expect("the CPU fit");
    let cpu = expr::simplify(&cpu);
    let (d, c) = (constants(&fitted[0]), constants(&cpu));
    assert_eq!(d.len(), 2, "{:?}", fitted[0]);
    assert_eq!(d.len(), c.len(), "{:?} vs {:?}", fitted[0], cpu);
    for (d, c) in d.iter().zip(&c) {
        assert!((d - c).abs() <= 1e-5 * (1.0 + c.abs()), "device {d} vs CPU {c}");
    }
}

/// A tree with nothing to fold keeps the device fit of its own plan, bit for
/// bit, and the counters stay at zero.
#[test]
fn a_tree_without_redundant_constants_keeps_its_device_fit() {
    need_device!();
    let train = sample(4096);
    let cfg = config();
    let trees = vec![
        node(
            Op::Add,
            vec![
                node(Op::Add, vec![node(Op::Mul, vec![constant(0.3), input(0)]), node(Op::Mul, vec![constant(0.2), input(1)])]),
                constant(0.1),
            ],
        ),
        node(
            Op::Sub,
            vec![node(Op::Mul, vec![constant(0.4), node(Op::Sin, vec![input(0)])]), node(Op::Mul, vec![constant(0.2), input(1)])],
        ),
    ];
    for t in &trees {
        assert_eq!(crate::fold::fold_constants(t).1, 0, "{t:?}");
    }
    let (mut g, fitted) = fit_on_device(&trees, &train, &cfg);
    assert_eq!((g.fit_fold_count, g.fit_folded), (0, 0));
    assert_eq!(g.fit_gpu, trees.len(), "{:?}", g.fit_fallback_by_cause);
    let plans: Vec<crate::fit::GpuFitPlan> =
        trees.iter().map(|t| crate::fit::gpu_fit_plan_reason(t).expect("a plan")).collect();
    let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
    g.ev.set_fit_polish(cfg.fit_polish);
    let (out, _) = g.ev.fit_batch(&refs, cfg.fit_iters, crate::fit::DeviceLoss::Squared);
    for (k, t) in trees.iter().enumerate() {
        let mut want = t.clone();
        plans[k].apply(&mut want, &out[k].c);
        let want = expr::simplify(&want);
        assert!(expr::same(&fitted[k], &want), "tree {k}: {:?} vs {:?}", fitted[k], want);
    }
}
