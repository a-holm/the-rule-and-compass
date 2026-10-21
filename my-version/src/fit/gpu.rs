//! #116 / #169: the batched-GPU fit **integration point** and exact verifier.
//!
//! The GPU LM kernel itself lives in #169. This module defines the narrow Rust
//! contract the rest of the engine uses, so the search can batch a whole
//! generation without knowing CUDA and so every GPU result passes the same
//! exactness gate:
//!
//! 1. the backend proposes constants per candidate, or `None` when a candidate
//!    is unsupported, ill-conditioned or its arithmetic was not finite;
//! 2. every proposal is applied to a **clone** and scored in exact `f64` on the
//!    same rows; it is accepted only when the score is finite and no worse than
//!    the starting constants;
//! 3. every rejected candidate is refit by the exact CPU arm (the historical
//!    LM unless the tree is affine/separable), on the untouched tree;
//! 4. finalists are always refit exactly with [`refit_finalists_exact`].
//!
//! #229: the gate and the refit are **metric-aware**. A `log_squashed`
//! proposal that improves `mean ln(1 + r²)` can be worse in SSE; judging it by
//! SSE would silently discard exactly the fits the metric exists for (the bug
//! #236 closed the GPU gate over). Both steps therefore score the metric the
//! run is configured with, and the squared family keeps its historical
//! arithmetic bit for bit.
//!
//! A backend that returns `None` for every candidate is therefore a correct
//! (slow) implementation, and a buggy backend can waste work but cannot make a
//! reported fit worse than the CPU reference. The default build has no backend;
//! [`FitStrategy::Gpu`](super::FitStrategy::Gpu) then uses the exact CPU arms
//! and is never a silent no-op.

use super::*;

/// Outcome of one candidate in a GPU batch. Reported by `--fit-report`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchStatus {
    /// The proposal was finite and no worse than the starting constants.
    Gpu,
    /// The backend declined the candidate (unsupported or ill-conditioned).
    Unsupported,
    /// The proposal had the wrong shape or a non-finite constant.
    NonFinite,
    /// The proposal scored worse than the starting constants in exact `f64`.
    Worse,
    /// No backend is registered in this build.
    NoBackend,
}

/// A batched fit backend. #169 implements this for the CUDA LM kernel.
///
/// `trees` are in the same order as the returned vector. A backend must not
/// mutate the trees: it returns constants in the flattened constant-slot order
/// that [`fit_weighted`](super::fit_weighted) uses. Returning `None` for a
/// candidate is always valid and routes it to the exact CPU fallback.
pub trait BatchedFitter {
    fn fit_batch(
        &self,
        trees: &[Node],
        x: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        iters: usize,
    ) -> Vec<Option<Vec<f64>>>;
}

/// A verified batch result: exact `f64` objective and the outcome for each
/// candidate, in input order.
#[derive(Clone, Debug)]
pub struct VerifiedBatch {
    /// The exact objective at the constants now in the tree: the weighted sum
    /// of squares for the squared family (the historical return value of
    /// [`fit_weighted`]), the configured loss otherwise — the same convention
    /// as [`fit_weighted_metric`](super::fit_weighted_metric).
    pub score: Vec<f64>,
    pub status: Vec<BatchStatus>,
}

/// Run a batched backend through the exactness gate for `metric`.
///
/// The tree is modified only to the verified constants (or to the exact CPU
/// fallback's result). Each candidate's final score is the exact `f64` value of
/// the **configured objective** of the constants now in the tree, so a robust
/// proposal is never judged, or accepted, on squared error.
pub fn fit_batch_verified<B: BatchedFitter + ?Sized>(
    backend: &B,
    trees: &mut [Node],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> VerifiedBatch {
    let proposals = backend.fit_batch(trees, x, y, w, iters);
    let mut score = Vec::with_capacity(trees.len());
    let mut status = Vec::with_capacity(trees.len());
    for (i, tree) in trees.iter_mut().enumerate() {
        let proposal = proposals.get(i).and_then(|p| p.as_ref());
        let (value, state) = match proposal {
            None => (
                fit_weighted_metric(tree, x, y, w, iters, metric),
                BatchStatus::Unsupported,
            ),
            Some(c) => match verify_proposal(tree, c, x, y, w, metric) {
                Some(v) => (v, BatchStatus::Gpu),
                None => (
                    fit_weighted_metric(tree, x, y, w, iters, metric),
                    BatchStatus::Worse,
                ),
            },
        };
        note_gpu_fit(state == BatchStatus::Gpu);
        score.push(value);
        status.push(state);
    }
    VerifiedBatch { score, status }
}

/// Apply one proposal to a clone and return its exact score of `metric` only
/// when it is a legitimate, non-worse fit. The real tree is untouched on
/// failure.
///
/// `Squared` keeps the historical weighted-SSE comparison bit for bit, so every
/// existing squared trajectory is unchanged; every other metric is compared on
/// [`metrics::score_weighted`](super::metrics::score_weighted) of the same rows.
fn verify_proposal(
    tree: &mut Node,
    proposal: &[f64],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
) -> Option<f64> {
    let f = flatten(tree);
    if proposal.len() != f.nconst || !proposal.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut c0 = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c0[f.cslot[i] as usize] = f.val[i];
        }
    }
    if metric != Metric::Squared {
        // #229: the configured loss decides, never SSE.
        let start = metric_score(tree, x, y, w, metric);
        let mut trial = tree.clone();
        write_constants(&mut trial, &f, proposal);
        let candidate = metric_score(&trial, x, y, w, metric);
        if !candidate.is_finite() || !start.is_finite() || candidate > start {
            return None;
        }
        write_constants(tree, &f, proposal);
        return Some(candidate);
    }
    let mut v = vec![0.0; f.op.len()];
    let start = sse(&f, x, y, w, &c0, &mut v);
    let candidate = sse(&f, x, y, w, proposal, &mut v);
    if !candidate.is_finite() || !start.is_finite() || candidate > start {
        return None;
    }
    write_constants(tree, &f, proposal);
    Some(candidate)
}

/// #169: the exact `f64` refit every reported finalist receives, whatever arm
/// produced it.
///
/// For `squared`/AIC/AICc/BIC this is the historical LM path bit for bit. #229:
/// for every other metric it is [`fit_weighted_metric`], which starts from
/// `best = metric_score(tree)` and only writes improving trials, so the refit
/// can never make the finalist worse on the configured objective than the
/// constants the GPU arm produced.
pub fn refit_finalists_exact(
    trees: &mut [Node],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> Vec<f64> {
    trees
        .iter_mut()
        .map(|tree| {
            note_gpu_refit();
            if matches!(metric, Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic) {
                fit_lm(tree, x, y, w, iters)
            } else {
                fit_weighted_metric(tree, x, y, w, iters, metric)
            }
        })
        .collect()
}

/// #116: informational description of the integration point for `--fit-report`
/// and the docs. It names the dependency instead of pretending the kernel is
/// present.
pub fn backend_description() -> &'static str {
    if cfg!(feature = "cuda") {
        "gpu batch verifier ready; LM kernel supplied by #169"
    } else {
        "no CUDA feature: --fit-strategy gpu uses the exact CPU arms"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    fn model() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.7), input(0)]), constant(0.1)],
        )
    }

    fn fitted(tree: &Node) -> Vec<f64> {
        let f = flatten(tree);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        c
    }

    struct Backend {
        proposal: Option<Vec<f64>>,
    }

    impl BatchedFitter for Backend {
        fn fit_batch(&self, trees: &[Node], _: &[Vec<f64>], _: &[f64], _: &[f64], _: usize) -> Vec<Option<Vec<f64>>> {
            trees.iter().map(|_| self.proposal.clone()).collect()
        }
    }

    #[test]
    fn a_good_gpu_proposal_is_accepted() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();
        let mut trees = vec![model()];
        let backend = Backend { proposal: Some(vec![2.0, 3.0]) };
        let out = fit_batch_verified(&backend, &mut trees, &x, &y, &[], 8, Metric::Squared);
        assert_eq!(out.status, vec![BatchStatus::Gpu]);
        assert!(out.score[0] < 1e-20);
        let c = fitted(&trees[0]);
        assert!((c[0] - 2.0).abs() < 1e-12 && (c[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_and_wrong_length_proposals_fall_back_exactly() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();
        for proposal in [Some(vec![f64::NAN, 3.0]), Some(vec![2.0]), None] {
            let mut gpu = vec![model()];
            let out = fit_batch_verified(&Backend { proposal }, &mut gpu, &x, &y, &[], 8, Metric::Squared);
            let mut cpu = model();
            let reference = fit_weighted(&mut cpu, &x, &y, &[], 8);
            assert_eq!(out.score[0].to_bits(), reference.to_bits());
            let a: Vec<u64> = fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect();
            let b: Vec<u64> = fitted(&cpu).iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn a_worse_proposal_never_replaces_the_cpu_fit() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();
        let mut gpu = vec![model()];
        let out = fit_batch_verified(
            &Backend { proposal: Some(vec![0.0, 0.0]) },
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Squared,
        );
        assert_ne!(out.status[0], BatchStatus::Gpu);
        let mut cpu = model();
        let reference = fit_weighted(&mut cpu, &x, &y, &[], 8);
        assert_eq!(out.score[0].to_bits(), reference.to_bits());
    }

    #[test]
    fn finalists_are_refit_with_the_exact_lm_path() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![-2.0 + 0.1 * i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 1.5 * (0.8 * r[0]).sin() + 0.2).collect();
        let mut gpu = vec![node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(1.0), node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])])]),
                constant(0.0),
            ],
        )];
        let mut cpu = gpu[0].clone();
        let a = refit_finalists_exact(&mut gpu, &x, &y, &[], 40, Metric::Squared);
        let b = fit_weighted(&mut cpu, &x, &y, &[], 40);
        assert_eq!(a[0].to_bits(), b.to_bits());
        let ca: Vec<u64> = fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect();
        let cb: Vec<u64> = fitted(&cpu).iter().map(|v| v.to_bits()).collect();
        assert_eq!(ca, cb);
    }

    /// #229: a proposal that improves `ln(1+r²)` while making SSE worse must be
    /// accepted — the exact bug a squared-only gate causes. The fixture is an
    /// affine fit on rows with 5 % gross outliers, where the robust optimum and
    /// the least-squares optimum differ.
    #[test]
    fn a_robust_proposal_is_judged_by_its_own_metric() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64 / 4.0]).collect();
        let mut y: Vec<f64> = x.iter().map(|r| 1.0 + 0.5 * r[0]).collect();
        for i in (3..40).step_by(8) {
            y[i] += 40.0;
        }
        // The robust optimum on the same 40 rows (CPU IRLS, the reference arm).
        let mut reference = model();
        let robust = fit_weighted_metric(&mut reference, &x, &y, &[], 8, Metric::LogSquashed);
        let proposal = fitted(&reference);
        let sse_worse = sse(&flatten(&model()), &x, &y, &[], &proposal, &mut vec![0.0; 64])
            > sse(&flatten(&model()), &x, &y, &[], &fitted(&model()), &mut vec![0.0; 64]);
        assert!(
            sse_worse,
            "the fixture must make the robust fit worse in SSE, else it proves nothing"
        );
        let mut gpu = vec![model()];
        let out = fit_batch_verified(
            &Backend { proposal: Some(proposal.clone()) },
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::LogSquashed,
        );
        assert_eq!(
            out.status[0],
            BatchStatus::Gpu,
            "a better log_squashed proposal must not be rejected on SSE"
        );
        assert_eq!(out.score[0].to_bits(), robust.to_bits());
        let c = fitted(&gpu[0]);
        let want = fitted(&reference);
        assert_eq!(
            c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// #229: the squared family keeps its exact arithmetic and the log-squashed
    /// refit is idempotent — a second refit from its own endpoint changes
    /// nothing, which is what "monotone from the GPU endpoint" means in bits.
    #[test]
    fn a_robust_finalist_refit_is_idempotent() {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![-1.5 + 0.05 * i as f64]).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| 2.0 * (1.3 * r[0]).sin() + if i % 7 == 0 { 6.0 } else { 0.0 })
            .collect();
        let tree = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(1.0), node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])])]),
                constant(0.0),
            ],
        );
        let mut once = vec![tree.clone()];
        let first = refit_finalists_exact(&mut once, &x, &y, &[], 8, Metric::LogSquashed);
        let c1 = fitted(&once[0]);
        let mut twice = once.clone();
        let second = refit_finalists_exact(&mut twice, &x, &y, &[], 8, Metric::LogSquashed);
        let c2 = fitted(&twice[0]);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
        assert_eq!(
            c1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // And the reported value is the configured loss, not an SSE.
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&twice[0], r)).collect();
        let scored = crate::metrics::score_weighted(
            &pred,
            &y,
            &[],
            Metric::LogSquashed,
            crate::expr::count_constants(&twice[0]),
        );
        assert_eq!(second[0].to_bits(), scored.to_bits());
    }

    /// #229: `--no-fit` (`iters == 0`) is a true no-op for a robust run too:
    /// the proposal never runs, the constants stay bit-identical and the score
    /// is the metric at those constants.
    #[test]
    fn no_fit_keeps_a_robust_run_untouched() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 1.0 + 0.2 * r[0]).collect();
        let tree = model();
        let before = fitted(&tree);
        let mut trees = vec![tree];
        let out = fit_batch_verified(
            &Backend { proposal: Some(vec![9.0, 9.0]) },
            &mut trees,
            &x,
            &y,
            &[],
            0,
            Metric::LogSquashed,
        );
        assert_eq!(
            fitted(&trees[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "--no-fit must not move a constant"
        );
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&trees[0], r)).collect();
        let want = crate::metrics::score_weighted(&pred, &y, &[], Metric::LogSquashed, 2);
        assert_eq!(out.score[0].to_bits(), want.to_bits());
    }
}
