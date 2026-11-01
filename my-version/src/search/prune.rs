//! #466: data-driven prune of front members.
//!
//! `expr::simplify` keeps code that is neutral only on the data, because it is
//! not value-preserving for every IEEE input. This pass removes such code by
//! measurement: it replaces a subtree with one of its children or with a
//! constant, refits the constants and keeps the replacement when the training
//! error does not rise beyond the archive's epsilon-dominance tolerance. It
//! uses no RNG, so the result is a pure function of the tree and the data.

use super::*;

/// What one run's prune pass did. All zero when the pass is off.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PruneReport {
    /// Candidate trees fitted and scored.
    pub fits: usize,
    /// Pruned trees offered to the archive.
    pub accepted: usize,
    pub seconds: f64,
}

/// Salt of the row subsample the prune fits on; independent of every other
/// row stream.
const PRUNE_SUBSET_SALT: u32 = 0x4F1B_66A3;

/// The archive's epsilon-dominance tolerance on error.
fn tolerance(e0: f64) -> f64 {
    1e-6 * (1.0 + e0.abs())
}

/// Pre-order paths (child indices from the root) of every internal node that
/// may be replaced. Subtrees of aggregates and custom blocks are not entered:
/// an aggregate caches a dataset value and a block shares its body.
fn internal_paths(n: &Node, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if n.kids.is_empty() {
        return;
    }
    out.push(path.clone());
    if n.op.is_agg() || n.op.is_custom() {
        return;
    }
    for (i, k) in n.kids.iter().enumerate() {
        path.push(i);
        internal_paths(k, path, out);
        path.pop();
    }
}

fn node_at<'a>(n: &'a Node, path: &[usize]) -> &'a Node {
    path.iter().fold(n, |cur, &i| &cur.kids[i])
}

fn replace_at(n: &Node, path: &[usize], with: Node) -> Node {
    match path.split_first() {
        None => with,
        Some((&i, rest)) => {
            let mut out = n.clone();
            out.kids[i] = replace_at(&n.kids[i], rest, with);
            out
        }
    }
}

/// Mean of the finite values `t` takes on the rows; `None` when it has none.
fn mean_value(t: &Node, fit: &Sample) -> Option<f64> {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for row in &fit.x {
        let v = expr::eval(t, row);
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    (n > 0).then(|| sum / n as f64)
}

/// Every one-step replacement of `tree`, in a fixed order: for each internal
/// node in pre-order, each child, then the constant.
fn candidates(tree: &Node, fit: &Sample) -> Vec<Node> {
    let mut paths = Vec::new();
    internal_paths(tree, &mut Vec::new(), &mut paths);
    let mut out = Vec::new();
    for path in paths {
        let t = node_at(tree, &path);
        for k in &t.kids {
            out.push(replace_at(tree, &path, k.clone()));
        }
        if let Some(m) = mean_value(t, fit) {
            out.push(replace_at(tree, &path, expr::constant(m)));
        }
    }
    out
}

/// Refit the constants of `tree` on `fit` with the exact CPU arm.
fn refit(tree: &mut Node, cfg: &Config, fit: &Sample) {
    if cfg.fit_iters == 0 || fit.is_multi() {
        return;
    }
    if cfg.metric.is_implicit_derivative() {
        crate::implicit::fit(tree, &fit.x, &fit.w, cfg.fit_iters);
    } else if let (Metric::Slope, Some(pairs)) = (cfg.metric, &fit.pairs) {
        crate::fit::fit_slope_pairs(tree, &fit.x, &fit.y, pairs, cfg.fit_iters);
    } else {
        let strategy = match cfg.fit_strategy {
            crate::fit::FitStrategy::Gpu | crate::fit::FitStrategy::Stream => {
                crate::fit::FitStrategy::Exact
            }
            s => s,
        };
        crate::fit::fit_weighted_strategy_metric(
            tree,
            &fit.x,
            &fit.y,
            &fit.w,
            cfg.fit_iters,
            strategy,
            cfg.metric,
        );
    }
}

/// Greedy prune of one tree. Returns the accepted trees in order, each
/// smaller than the one before, and the number of fits spent.
fn prune_tree(
    tree: &Node,
    data: &EvalSample,
    fit: &Sample,
    cfg: &Config,
    budget: usize,
) -> (Vec<Node>, usize) {
    let e0 = score_tree(tree, data, cfg);
    if !e0.is_finite() {
        return (Vec::new(), 0);
    }
    let bound = e0 + tolerance(e0);
    let mut cur = tree.clone();
    let mut cur_cost = cfg.complexity_model.complexity(&cur);
    let mut accepted = Vec::new();
    let mut fits = 0usize;
    loop {
        let mut cands = candidates(&cur, fit);
        let left = budget.saturating_sub(fits);
        let truncated = cands.len() > left;
        cands.truncate(left);
        if cands.is_empty() {
            break;
        }
        fits += cands.len();
        let workers = batch_workers(cfg.threads, cands.len(), fit.len());
        let scored = par_map(workers, cands, |c| {
            let mut t = c;
            refit(&mut t, cfg, fit);
            let t = expr::simplify_equal_branches(&t);
            let cost = cfg.complexity_model.complexity(&t);
            let err = if cfg.accepts_dimensions(&t) {
                score_tree(&t, data, cfg)
            } else {
                f64::INFINITY
            };
            (t, cost, err)
        });
        let mut best: Option<usize> = None;
        for (i, (_, cost, err)) in scored.iter().enumerate() {
            if !(*err <= bound) || !(*cost < cur_cost) {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => {
                    let (_, bc, be) = &scored[b];
                    cost.total_cmp(bc).then(err.total_cmp(be)).is_lt()
                }
            };
            if better {
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                let (t, cost, _) = scored.into_iter().nth(i).expect("index in range");
                cur = t.clone();
                cur_cost = cost;
                accepted.push(t);
            }
            None => break,
        }
        if truncated {
            break;
        }
    }
    (accepted, fits)
}

/// Prune the members of `archive` in ascending error order until the budget
/// is spent, and rebuild the front from the old members and every accepted
/// tree scored on all training rows. The archive is returned unchanged when
/// nothing was accepted.
pub(super) fn prune_archive(
    archive: Archive,
    train: &Sample,
    train_data: &EvalSample,
    cfg: &Config,
) -> (Archive, PruneReport) {
    let mut report = PruneReport::default();
    if !cfg.prune || cfg.prune_budget == 0 || archive.items.is_empty() {
        return (archive, report);
    }
    let start = std::time::Instant::now();
    let sub;
    let sub_data;
    let (fit, data): (&Sample, &EvalSample) = if cfg.prune_rows >= 2 && train.len() > cfg.prune_rows {
        let idx = generation_subset(train.len(), cfg.prune_rows, cfg.seed ^ PRUNE_SUBSET_SALT, 0);
        sub = subset_sample(train, &idx);
        sub_data = EvalSample::new(&sub, cfg);
        (&sub, &sub_data)
    } else {
        (train, train_data)
    };
    let mut order: Vec<&Sol> = archive.items.iter().collect();
    order.sort_by(|a, b| a.error.total_cmp(&b.error).then(a.cost.total_cmp(&b.cost)));
    let mut budget = cfg.prune_budget;
    let mut pruned: Vec<Node> = Vec::new();
    for sol in order {
        if budget == 0 {
            break;
        }
        if sol.tree.kids.is_empty() {
            continue;
        }
        let (trees, fits) = prune_tree(&sol.tree, data, fit, cfg, budget);
        budget -= fits.min(budget);
        report.fits += fits;
        pruned.extend(trees);
    }
    report.accepted = pruned.len();
    if pruned.is_empty() {
        report.seconds = start.elapsed().as_secs_f64();
        return (archive, report);
    }
    let cap = archive.cap;
    let mut scored: Vec<Sol> = archive.items;
    let mut seen: HashSet<u64> = scored.iter().map(|s| expr::structural_hash(&s.tree)).collect();
    for tree in pruned {
        if !seen.insert(expr::structural_hash(&tree)) {
            continue;
        }
        let error = if cfg.accepts_dimensions(&tree) {
            score_tree(&tree, train_data, cfg)
        } else {
            f64::INFINITY
        };
        scored.push(Sol {
            complexity: expr::complexity(&tree),
            cost: cfg.complexity_model.complexity(&tree),
            tree,
            error,
        });
    }
    let out = canonical_front(cap, scored);
    report.seconds = start.elapsed().as_secs_f64();
    (out, report)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    fn coulomb_tree() -> Node {
        let q = |i| input(i);
        let tan_if = node(
            Op::Tan,
            vec![node(
                Op::If,
                vec![node(Op::Exp, vec![q(2)]), constant(-0.0), q(0)],
            )],
        );
        let left = node(
            Op::Mul,
            vec![
                node(Op::Mul, vec![q(0), q(1)]),
                node(Op::Pow, vec![q(0), tan_if]),
            ],
        );
        node(
            Op::Mul,
            vec![left, node(Op::Pow, vec![q(2), constant(-2.0)])],
        )
    }

    fn coulomb_rows() -> Sample {
        let mut rng = Mt::new(466);
        let x: Vec<Vec<f64>> = (0..60)
            .map(|_| vec![rng.range(1.0, 3.0), rng.range(1.0, 3.0), rng.range(1.0, 3.0)])
            .collect();
        let y = x.iter().map(|r| r[0] * r[1] / (r[2] * r[2])).collect();
        Sample::new(x, y)
    }

    fn cfg() -> Config {
        Config {
            metric: Metric::Squared,
            prune: true,
            ..Config::default()
        }
    }

    #[test]
    fn prune_removes_introns() {
        let cfg = cfg();
        let s = coulomb_rows();
        let data = EvalSample::new(&s, &cfg);
        let tree = coulomb_tree();
        assert_eq!(expr::complexity(&tree), 16);
        assert!(score_tree(&tree, &data, &cfg) <= 1e-24);
        let (steps, fits) = prune_tree(&tree, &data, &s, &cfg, 20_000);
        assert!(fits > 0);
        let last = steps.last().expect("at least one accepted tree");
        assert!(expr::complexity(last) <= 7, "{}", expr::complexity(last));
        assert!(score_tree(last, &data, &cfg) <= 1e-24);
    }

    #[test]
    fn prune_is_deterministic() {
        let cfg = cfg();
        let s = coulomb_rows();
        let data = EvalSample::new(&s, &cfg);
        let tree = coulomb_tree();
        let a = prune_tree(&tree, &data, &s, &cfg, 20_000);
        let b = prune_tree(&tree, &data, &s, &cfg, 20_000);
        assert_eq!(a.1, b.1);
        assert_eq!(a.0, b.0);
        let threaded = Config { threads: 4, ..cfg.clone() };
        let c = prune_tree(&tree, &data, &s, &threaded, 20_000);
        assert_eq!(a.0, c.0);
    }

    #[test]
    fn prune_respects_the_budget() {
        let cfg = cfg();
        let s = coulomb_rows();
        let data = EvalSample::new(&s, &cfg);
        let (_, fits) = prune_tree(&coulomb_tree(), &data, &s, &cfg, 5);
        assert!(fits <= 5);
    }

    #[test]
    fn prune_archive_replaces_the_bloated_member_and_off_is_identity() {
        let cfg = cfg();
        let s = coulomb_rows();
        let data = EvalSample::new(&s, &cfg);
        let tree = coulomb_tree();
        let mut archive = Archive::new(20);
        archive.try_insert(Sol {
            complexity: expr::complexity(&tree),
            cost: cfg.complexity_model.complexity(&tree),
            error: score_tree(&tree, &data, &cfg),
            tree,
        });
        let before = archive.hash();
        let off = Config { prune: false, ..cfg.clone() };
        let (same, rep) = prune_archive(
            Archive { items: archive.items.clone(), cap: archive.cap },
            &s,
            &data,
            &off,
        );
        assert_eq!(same.hash(), before);
        assert_eq!(rep, PruneReport::default());
        let (out, rep) = prune_archive(archive, &s, &data, &cfg);
        assert!(rep.accepted > 0 && rep.fits > 0);
        assert!(out.items.iter().all(|m| m.complexity <= 7), "{:?}", out.items.iter().map(|m| m.complexity).collect::<Vec<_>>());
    }
}
