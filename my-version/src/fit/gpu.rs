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
//! #222: `absolute` (MAE) uses the same contract with the kernel's Lawson/IRLS
//! weights `1/max(|r|, 1e-6)` and true-`|r|` acceptance. A proposal is accepted
//! only when the exact `f64` MAE is no worse than the start, and the finalist
//! refit is the exact CPU L1 IRLS, never `fit_lm`.
//!
//! #228: the gate is also **signed-aware**. `|Σ wᵢ rᵢ| / Σ wᵢ` has a
//! codimension-1 optimal manifold at value exactly 0, so a proposal is
//! completed by the exact `f64` constrained-LS projection
//! ([`super::signed_projection`]) before it is compared, and a model that
//! cannot move the mean is labelled [`BatchStatus::NotIdentifiable`] instead of
//! being accepted on an equal objective. [`BatchedFitter::fit_batch_metric`]
//! additionally guarantees that a backend which only knows squared error can
//! never answer a non-squared request.
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
    /// #228: the configured objective is `signed`, the model cannot represent
    /// the weighted mean residual, and the proposal did not improve the exact
    /// objective. The starting constants are kept.
    NotIdentifiable,
    /// No backend is registered in this build.
    NoBackend,
}

impl BatchStatus {
    /// Stable label for the ledger and the docs.
    pub fn label(self) -> &'static str {
        match self {
            BatchStatus::Gpu => "gpu",
            BatchStatus::Unsupported => "unsupported",
            BatchStatus::NonFinite => "non_finite",
            BatchStatus::Worse => "worse",
            BatchStatus::NotIdentifiable => "not_identifiable",
            BatchStatus::NoBackend => "no_backend",
        }
    }

    /// Whether the candidate's constants came from the batch proposal.
    pub fn is_gpu(self) -> bool {
        self == BatchStatus::Gpu
    }
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

    /// #228: fit constants for the **configured** objective.
    ///
    /// The default implementation is metric-aware in the only safe direction:
    /// the squared family (and the information criteria, whose constants are
    /// least-squares constants) delegates to [`Self::fit_batch`], and every
    /// other metric returns `None` so the caller routes it to the exact CPU
    /// arm. A backend must never answer a non-squared metric with its squared
    /// proposal: that is a silent substitution of the loss the user chose.
    fn fit_batch_metric(
        &self,
        trees: &[Node],
        x: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        iters: usize,
        metric: Metric,
    ) -> Vec<Option<Vec<f64>>> {
        if is_squared_family(metric) {
            self.fit_batch(trees, x, y, w, iters)
        } else {
            vec![None; trees.len()]
        }
    }
}

/// The metrics whose fitted constants minimise weighted squared error. The
/// information criteria fit by least squares and add their parameter penalty at
/// scoring time, exactly like [`fit_weighted_metric`](super::fit_weighted_metric).
pub fn is_squared_family(metric: Metric) -> bool {
    matches!(
        metric,
        Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic
    )
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
    let proposals = backend.fit_batch_metric(trees, x, y, w, iters, metric);
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
                Verdict::Accepted(v) => (v, BatchStatus::Gpu),
                Verdict::Rejected => (
                    fit_weighted_metric(tree, x, y, w, iters, metric),
                    BatchStatus::Worse,
                ),
                Verdict::NotIdentifiable => (
                    fit_weighted_metric(tree, x, y, w, iters, metric),
                    BatchStatus::NotIdentifiable,
                ),
            },
        };
        note_gpu_fit(state == BatchStatus::Gpu);
        score.push(value);
        status.push(state);
    }
    VerifiedBatch { score, status }
}

/// The outcome of comparing one proposal with the starting constants.
enum Verdict {
    /// The proposal was accepted; carries its exact score of the metric.
    Accepted(f64),
    /// The proposal was not better (or was not finite): keep the start.
    Rejected,
    /// #228: `signed`, and no constant can move the mean, so the proposal has
    /// no justification on the bias. Keep the start and label it.
    NotIdentifiable,
}

/// Apply one proposal to a clone and return its exact score of `metric` only
/// when it is a legitimate, non-worse fit. The real tree is untouched on
/// failure.
///
/// `Squared` keeps the historical weighted-SSE comparison bit for bit, so every
/// existing squared trajectory is unchanged; every other metric is compared on
/// [`metrics::score_weighted`](super::metrics::score_weighted) of the same rows.
///
/// #228: `signed` is additionally *completed* first — the proposal is moved
/// onto the zero-mean-residual manifold by the exact `f64` constrained-LS
/// projection, and the comparison is on the resulting bias. A projection that
/// cannot exist means the model cannot represent the mean; then only a strict
/// objective improvement can justify the proposal, and otherwise the outcome is
/// [`Verdict::NotIdentifiable`] rather than a squared answer in disguise.
fn verify_proposal(
    tree: &mut Node,
    proposal: &[f64],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
) -> Verdict {
    let f = flatten(tree);
    if proposal.len() != f.nconst || !proposal.iter().all(|v| v.is_finite()) {
        return Verdict::Rejected;
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
        // #228: a signed proposal is projected before it is judged.
        let projected = if metric == Metric::Signed {
            Some(signed_projection(&mut trial, x, y, w, 2))
        } else {
            None
        };
        let candidate = metric_score(&trial, x, y, w, metric);
        let justified = match projected {
            Some(outcome) => outcome.is_projected() || candidate < start,
            None => true,
        };
        if !candidate.is_finite() || !start.is_finite() || candidate > start || !justified {
            return match projected {
                Some(SignedProjection::Unidentifiable) => Verdict::NotIdentifiable,
                _ => Verdict::Rejected,
            };
        }
        let constants = fitted_constants(&trial);
        write_constants(tree, &f, &constants);
        return Verdict::Accepted(candidate);
    }
    let mut v = vec![0.0; f.op.len()];
    let start = sse(&f, x, y, w, &c0, &mut v);
    let candidate = sse(&f, x, y, w, proposal, &mut v);
    if !candidate.is_finite() || !start.is_finite() || candidate > start {
        return Verdict::Rejected;
    }
    write_constants(tree, &f, proposal);
    Verdict::Accepted(candidate)
}

/// The constant values currently in `tree`, in flat slot order.
fn fitted_constants(tree: &Node) -> Vec<f64> {
    let f = flatten(tree);
    let mut c = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    c
}

/// #228: the exact `f64` finalist refit for `--metric signed`.
///
/// Runs the same constrained-LS projection the CPU arm uses on the (possibly
/// GPU-proposed) constants and returns the exact configured objective, which is
/// what the archive and the report consume. Like [`refit_finalists_exact`] this
/// is the step that makes the reported number independent of the device's
/// arithmetic.
pub fn signed_finalist(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
) -> (f64, SignedProjection) {
    note_gpu_refit();
    let outcome = signed_projection(tree, x, y, w, 2);
    let value = metric_score(tree, x, y, w, Metric::Signed);
    (value, outcome)
}

/// #169 + #228: the CUDA LM kernel as a [`BatchedFitter`], so a whole
/// generation can be driven through the metric-aware verifier.
///
/// The kernel fits on the evaluator's **resident** rows (uploaded once with
/// [`crate::gpu::GpuEvaluator::set_fit_data`] in exact `f64`), so `x`, `y` and
/// `w` are not uploaded here: the caller must have uploaded the same rows and
/// the same weights it will verify the proposals on. For a metric with a native
/// device loss (`squared`, `log_squashed`, `absolute`) the proposal is the
/// kernel's own optimum. For `signed` the squared LM proposal is returned as a
/// **warm start** and the verifier completes it with the exact `f64` projection
/// onto the zero-mean-residual manifold: the device supplies the row sweep, the
/// exact objective decides what is kept. Other metrics are declined.
#[cfg(feature = "cuda")]
pub struct BatchedGpu<'a> {
    ev: std::cell::RefCell<&'a mut crate::gpu::GpuEvaluator>,
}

#[cfg(feature = "cuda")]
impl<'a> BatchedGpu<'a> {
    pub fn new(ev: &'a mut crate::gpu::GpuEvaluator) -> Self {
        BatchedGpu {
            ev: std::cell::RefCell::new(ev),
        }
    }
}

#[cfg(feature = "cuda")]
impl BatchedGpu<'_> {
    fn propose(
        &self,
        trees: &[Node],
        iters: usize,
        loss: DeviceLoss,
    ) -> Vec<Option<Vec<f64>>> {
        // A tree the kernel cannot plan is declined and refit by the exact CPU
        // arm.
        let mut out: Vec<Option<Vec<f64>>> = trees.iter().map(|_| None).collect();
        let slots: Vec<(usize, GpuFitPlan)> = trees
            .iter()
            .enumerate()
            .filter_map(|(i, t)| gpu_fit_plan(t).map(|plan| (i, plan)))
            .collect();
        if slots.is_empty() {
            return out;
        }
        let plans: Vec<&GpuFitPlan> = slots.iter().map(|(_, p)| p).collect();
        let (results, ms) = self.ev.borrow_mut().fit_batch(&plans, iters, loss);
        if ms < 0.0 || results.len() != slots.len() {
            return out;
        }
        for ((i, _), res) in slots.iter().zip(results) {
            if res.status == 0 && res.c.iter().all(|v| v.is_finite()) {
                out[*i] = Some(res.c);
            }
        }
        out
    }
}

#[cfg(feature = "cuda")]
impl BatchedFitter for BatchedGpu<'_> {
    fn fit_batch(
        &self,
        trees: &[Node],
        _x: &[Vec<f64>],
        _y: &[f64],
        _w: &[f64],
        iters: usize,
    ) -> Vec<Option<Vec<f64>>> {
        self.propose(trees, iters, DeviceLoss::Squared)
    }

    fn fit_batch_metric(
        &self,
        trees: &[Node],
        _x: &[Vec<f64>],
        _y: &[f64],
        _w: &[f64],
        iters: usize,
        metric: Metric,
    ) -> Vec<Option<Vec<f64>>> {
        if let Some(loss) = fit_loss(metric) {
            return self.propose(trees, iters, loss);
        }
        if metric == Metric::Signed {
            // #228: warm start only; `verify_proposal` projects and judges.
            return self.propose(trees, iters, DeviceLoss::Squared);
        }
        // #226: `median` is fitted by the batched exact median polish (squared
        // LM start, then the same first-improver ladder on the exact `f64`
        // median). The verifier below judges it on the configured median, so
        // the proposal is never accepted on SSE.
        if median_fit_only(metric) && iters > 0 {
            let mut out: Vec<Option<Vec<f64>>> = trees.iter().map(|_| None).collect();
            let slots: Vec<(usize, GpuFitPlan)> = trees
                .iter()
                .enumerate()
                .filter_map(|(i, t)| gpu_fit_plan(t).map(|plan| (i, plan)))
                .collect();
            if slots.is_empty() {
                return out;
            }
            let plans: Vec<&GpuFitPlan> = slots.iter().map(|(_, p)| p).collect();
            let (results, ms) = self.ev.borrow_mut().median_fit_batch(&plans, iters);
            if ms < 0.0 || results.len() != slots.len() {
                return out;
            }
            for ((i, _), res) in slots.iter().zip(results) {
                if res.status == 0 && res.c.iter().all(|v| v.is_finite()) {
                    out[*i] = Some(res.c);
                }
            }
            return out;
        }
        vec![None; trees.len()]
    }
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

    /// The same tree with its constants replaced (plan slot order).
    fn with_constants(tree: &Node, c: &[f64]) -> Node {
        let f = flatten(tree);
        let mut out = tree.clone();
        write_constants(&mut out, &f, c);
        out
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

        /// #228: a backend that claims to answer a metric must answer it
        /// explicitly. The default declines every non-squared metric (pinned by
        /// `a_squared_proposal_is_never_accepted_for_signed_by_default`), so the
        /// test double overrides it exactly like a real device backend does.
        fn fit_batch_metric(
            &self,
            trees: &[Node],
            _: &[Vec<f64>],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: Metric,
        ) -> Vec<Option<Vec<f64>>> {
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

    /// #226: a proposal that improves the median while making SSE worse must
    /// be accepted for `--metric median` — the exact bug an SSE gate causes.
    ///
    /// Fixture: `y = 0` on 20 rows and `y = 100` on the last. The least-squares
    /// line (the starting constants) has a strictly lower SSE than the flat
    /// line, while the flat line drives the median of `|r|` to exactly 0 (the
    /// 11th of 21 residuals). An SSE gate would reject the only median answer.
    #[test]
    fn a_median_proposal_is_judged_by_the_median_not_by_sse() {
        let x: Vec<Vec<f64>> = (0..21).map(|i| vec![i as f64]).collect();
        let mut y = vec![0.0; 21];
        y[20] = 100.0;
        let mut start = model();
        fit_weighted(&mut start, &x, &y, &[], 8);
        let start_c = fitted(&start);
        let proposal = vec![0.0, 0.0];
        let sse_of = |c: &[f64]| {
            sse(&flatten(&model()), &x, &y, &[], c, &mut vec![0.0; 64])
        };
        let median_of = |c: &[f64]| metric_score(&with_constants(&model(), c), &x, &y, &[], Metric::Median);
        assert_eq!(median_of(&proposal), 0.0);
        assert!(
            median_of(&start_c) > 0.0,
            "the least-squares line must have a positive median"
        );
        assert!(
            sse_of(&proposal) > sse_of(&start_c),
            "the fixture must make the median answer worse in SSE"
        );

        let mut gpu = vec![start.clone()];
        let out = fit_batch_verified(
            &Backend { proposal: Some(proposal.clone()) },
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Median,
        );
        assert_eq!(
            out.status[0],
            BatchStatus::Gpu,
            "a better median proposal must not be rejected on SSE"
        );
        assert_eq!(out.score[0], 0.0);
        assert_eq!(
            fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            proposal.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // The same proposal through the squared gate is rejected: that is the
        // defect the metric-aware gate removes.
        assert!(
            sse_of(&proposal) > sse_of(&start_c),
            "sanity: the squared gate would reject this proposal"
        );
    }

    /// #226: a squared-only backend never answers a median request, and a
    /// declined median request falls back to the exact CPU median arm bit for
    /// bit (never to a squared answer in disguise).
    #[test]
    fn a_squared_proposal_is_never_answered_for_median() {
        let trees = vec![model()];
        assert_eq!(
            SquaredOnlyBackend(vec![2.0, 3.0]).fit_batch_metric(
                &trees,
                &[],
                &[],
                &[],
                8,
                Metric::Median
            ),
            vec![None],
            "the default must decline a median request"
        );
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
        let mut y: Vec<f64> = x.iter().map(|r| 1.0 + 0.2 * r[0]).collect();
        y[7] += 3.0;
        let mut gpu = vec![model()];
        let out = fit_batch_verified(
            &SquaredOnlyBackend(vec![2.0, 3.0]),
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Median,
        );
        assert_eq!(out.status[0], BatchStatus::Unsupported);
        let mut cpu = model();
        let want = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Median);
        assert_eq!(out.score[0].to_bits(), want.to_bits());
        assert_eq!(
            fitted(&gpu[0])
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            fitted(&cpu).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// #226: the finalist refit for a median run is the exact CPU median arm
    /// and cannot make the reported model worse on the median.
    #[test]
    fn a_median_finalist_refit_is_exact_and_never_worse() {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![-1.5 + 0.05 * i as f64]).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| 2.0 * (1.3 * r[0]).sin() + if i % 7 == 0 { 6.0 } else { 0.0 })
            .collect();
        let mut tree = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(1.0), node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])])]),
                constant(0.0),
            ],
        );
        let before = metric_score(&tree, &x, &y, &[], Metric::Median);
        let mut once = vec![tree.clone()];
        let first = refit_finalists_exact(&mut once, &x, &y, &[], 8, Metric::Median);
        assert!(first[0].is_finite());
        assert!(first[0] <= before, "refit {} > start {}", first[0], before);
        // Exactly what `fit_weighted_metric` produces for the same start...
        let mut cpu = tree.clone();
        let want = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Median);
        assert_eq!(first[0].to_bits(), want.to_bits());
        assert_eq!(
            fitted(&once[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fitted(&cpu).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // ... and idempotent from its own endpoint.
        let mut twice = once.clone();
        let second = refit_finalists_exact(&mut twice, &x, &y, &[], 8, Metric::Median);
        assert_eq!(first[0].to_bits(), second[0].to_bits());
        tree = twice[0].clone();
        assert!(metric_score(&tree, &x, &y, &[], Metric::Median).is_finite());
    }

    /// #226: `--no-fit` is a no-op for the median arm: the proposal never
    /// runs, the constants stay bit-identical and the score is the median at
    /// those constants.
    #[test]
    fn no_fit_keeps_a_median_run_untouched() {
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
            Metric::Median,
        );
        assert_eq!(
            fitted(&trees[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "--no-fit must not move a constant"
        );
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&trees[0], r)).collect();
        let want = crate::metrics::score_weighted(&pred, &y, &[], Metric::Median, 2);
        assert_eq!(out.score[0].to_bits(), want.to_bits());
    }

    // -- #228: the signed arm through the batch gate ------------------------

    /// A backend that answers the configured metric explicitly (the contract a
    /// real device backend must meet).
    struct ProposingBackend(Vec<f64>);

    impl BatchedFitter for ProposingBackend {
        fn fit_batch(
            &self,
            trees: &[Node],
            _: &[Vec<f64>],
            _: &[f64],
            _: &[f64],
            _: usize,
        ) -> Vec<Option<Vec<f64>>> {
            trees.iter().map(|_| Some(self.0.clone())).collect()
        }

        fn fit_batch_metric(
            &self,
            trees: &[Node],
            _: &[Vec<f64>],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: Metric,
        ) -> Vec<Option<Vec<f64>>> {
            trees.iter().map(|_| Some(self.0.clone())).collect()
        }
    }

    /// A backend that only knows squared error: every non-squared request must
    /// be declined by the default `fit_batch_metric`, never answered with SSE.
    struct SquaredOnlyBackend(Vec<f64>);

    impl BatchedFitter for SquaredOnlyBackend {
        fn fit_batch(
            &self,
            trees: &[Node],
            _: &[Vec<f64>],
            _: &[f64],
            _: &[f64],
            _: usize,
        ) -> Vec<Option<Vec<f64>>> {
            trees.iter().map(|_| Some(self.0.clone())).collect()
        }
    }

    #[test]
    fn a_squared_proposal_is_never_accepted_for_signed_by_default() {
        let trees = vec![model()];
        let out = SquaredOnlyBackend(vec![2.0, 3.0]).fit_batch_metric(
            &trees,
            &[],
            &[],
            &[],
            8,
            Metric::Signed,
        );
        assert_eq!(
            out,
            vec![None],
            "the default must decline, not answer with SSE"
        );
    }

    #[test]
    fn a_signed_proposal_is_judged_on_the_bias_not_on_sse() {
        // y = 2x + 4 on x = 1..20, model `a·x` (no intercept). The least-squares
        // optimum is biased (a·mean(x) != mean(y)); the only zero-bias point is
        // a = sum(y)/sum(x), and its SSE is strictly higher. The gate must accept
        // the projection even though SSE got worse: an SSE gate would have
        // rejected the only signed answer.
        let x: Vec<Vec<f64>> = (1..=20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 4.0).collect();
        let mut start = node(Op::Mul, vec![constant(1.0), input(0)]);
        fit_weighted(&mut start, &x, &y, &[], 8);
        let start_sse = metric_score(&start, &x, &y, &[], Metric::Squared);
        let start_bias = metric_score(&start, &x, &y, &[], Metric::Signed);
        assert!(start_bias > 0.5, "the squared optimum is biased: {start_bias}");

        let mut gpu = vec![start];
        let out = fit_batch_verified(
            &ProposingBackend(vec![2.0]),
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Signed,
        );
        assert_eq!(out.status, vec![BatchStatus::Gpu]);
        let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
        assert!(out.score[0] <= 1e-14 * scale, "bias {}", out.score[0]);
        // The accepted constants are *worse* in squared error: the gate is on
        // the configured objective, not on SSE.
        assert!(
            metric_score(&gpu[0], &x, &y, &[], Metric::Squared) > start_sse,
            "the zero-bias point must have a higher SSE for this test to mean anything"
        );
        // The reported objective is the exact metric value at the constants.
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&gpu[0], r)).collect();
        assert_eq!(
            out.score[0].to_bits(),
            crate::metrics::score_weighted(
                &pred,
                &y,
                &[],
                Metric::Signed,
                crate::expr::count_constants(&gpu[0])
            )
            .to_bits()
        );
    }

    #[test]
    fn a_non_identifiable_model_is_labelled_and_keeps_the_start() {
        // No intercept and a zero-mean input: no constant can move the mean, so
        // the bias is a constant of the model and the projection must decline.
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 - 9.5]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 4.0).collect();
        let bias_only = node(Op::Mul, vec![constant(1.0), input(0)]);
        let mut gpu = vec![bias_only];
        let out = fit_batch_verified(
            &ProposingBackend(vec![1.5]),
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Signed,
        );
        assert_eq!(out.status, vec![BatchStatus::NotIdentifiable]);
        // The reported objective is the exact bias of the model, not zero.
        assert!((out.score[0] - 4.0).abs() < 1e-12, "{:?}", out.score);
    }

    // -- #222: the absolute (MAE) arm through the batch gate ----------------

    /// #222: `absolute` has an explicit device objective, distinct from the
    /// squared code, and a squared-only backend must decline it rather than
    /// answer with an SSE proposal.
    #[test]
    fn absolute_has_its_own_device_objective() {
        assert_eq!(
            fit_loss(Metric::Absolute),
            Some(DeviceLoss::Absolute),
            "the fit loss must name the L1 arm"
        );
        assert_eq!(
            DeviceLoss::for_metric(Metric::Absolute),
            Some(DeviceLoss::Absolute),
            "the evaluator may score MAE natively"
        );
        assert_ne!(
            DeviceLoss::Absolute.code(),
            DeviceLoss::Squared.code(),
            "a MAE request must never share the SSE code"
        );
        let trees = vec![model()];
        let declined =
            SquaredOnlyBackend(vec![2.0, 3.0]).fit_batch_metric(&trees, &[], &[], &[], 8, Metric::Absolute);
        assert_eq!(
            declined,
            vec![None],
            "the default contract must decline, never substitute SSE"
        );
    }

    /// #222: a proposal that lowers MAE while raising SSE is accepted, and the
    /// reported score is the exact MAE at the accepted constants. The fixture
    /// is an affine fit with gross outliers, where the L1 and L2 optima differ.
    #[test]
    fn an_absolute_proposal_is_judged_by_mae_not_by_sse() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64 / 4.0]).collect();
        let mut y: Vec<f64> = x.iter().map(|r| 1.0 + 0.5 * r[0]).collect();
        for i in (3..40).step_by(8) {
            y[i] += 40.0;
        }
        let mut reference = model();
        let robust = fit_weighted_metric(&mut reference, &x, &y, &[], 8, Metric::Absolute);
        let proposal = fitted(&reference);
        let start = fitted(&model());
        let sse_at = |c: &[f64]| sse(&flatten(&model()), &x, &y, &[], c, &mut vec![0.0; 64]);
        assert!(
            sse_at(&proposal) > sse_at(&start),
            "the fixture must make the MAE optimum worse in SSE, else it proves nothing"
        );
        assert!(
            metric_score(&reference, &x, &y, &[], Metric::Absolute)
                < metric_score(&model(), &x, &y, &[], Metric::Absolute),
            "the fixture must separate the L1 and least-squares endpoints"
        );
        let mut gpu = vec![model()];
        let out = fit_batch_verified(
            &Backend { proposal: Some(proposal.clone()) },
            &mut gpu,
            &x,
            &y,
            &[],
            8,
            Metric::Absolute,
        );
        assert_eq!(
            out.status[0],
            BatchStatus::Gpu,
            "a better MAE proposal must not be rejected on SSE"
        );
        assert_eq!(out.score[0].to_bits(), robust.to_bits());
        let c = fitted(&gpu[0]);
        assert_eq!(
            c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            proposal.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // The reported value is the configured loss, not an SSE.
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&gpu[0], r)).collect();
        assert_eq!(
            out.score[0].to_bits(),
            crate::metrics::score_weighted(&pred, &y, &[], Metric::Absolute, 2).to_bits()
        );
    }

    /// #222: the exact finalist refit for `absolute` is the CPU L1 IRLS, bit
    /// for bit, and it is not the least-squares endpoint.
    #[test]
    fn absolute_finalists_are_refit_with_the_exact_cpu_irls() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![-2.0 + 0.08 * i as f64]).collect();
        let mut y: Vec<f64> = x.iter().map(|r| 0.4 + 1.7 * r[0]).collect();
        for i in (1..50).step_by(9) {
            y[i] += 30.0;
        }
        let start = model();
        let mut gpu = vec![start.clone()];
        let a = refit_finalists_exact(&mut gpu, &x, &y, &[], 8, Metric::Absolute);
        let mut cpu = start.clone();
        let b = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Absolute);
        assert_eq!(a[0].to_bits(), b.to_bits());
        assert_eq!(
            fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fitted(&cpu).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the finalist refit must be the exact CPU IRLS, not fit_lm"
        );
        let mut l2 = start.clone();
        fit_weighted(&mut l2, &x, &y, &[], 8);
        assert!(
            metric_score(&l2, &x, &y, &[], Metric::Absolute)
                > metric_score(&gpu[0], &x, &y, &[], Metric::Absolute) + 1e-3,
            "the fixture must separate the L1 and L2 endpoints"
        );
    }

    /// #222: `--no-fit` (`iters == 0`) is a bit-exact no-op for `absolute` too.
    #[test]
    fn no_fit_keeps_an_absolute_run_untouched() {
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
            Metric::Absolute,
        );
        assert_eq!(
            fitted(&trees[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "--no-fit must not move a constant"
        );
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&trees[0], r)).collect();
        let want = crate::metrics::score_weighted(&pred, &y, &[], Metric::Absolute, 2);
        assert_eq!(out.score[0].to_bits(), want.to_bits());
    }
}
