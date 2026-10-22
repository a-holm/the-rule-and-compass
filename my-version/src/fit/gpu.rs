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
///
/// #233: for `auc` the gate is [`gate_auc_proposal`] — exact AUC **and** the
/// class-balanced logistic surrogate — because exact AUC is a rank statistic the
/// kernel cannot step on.
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
        if metric == Metric::Auc {
            // #233: exact AUC plus the class-balanced in-loop objective; a
            // rejected proposal is refit by the exact CPU classification arm
            // from the untouched starting constants.
            let report = gate_auc_proposal(tree, proposal.map(|c| c.as_slice()), x, y, w);
            let (value, state) = if report.verdict.accepted() {
                (report.proposal, BatchStatus::Gpu)
            } else {
                let state = match report.verdict {
                    GateVerdict::NoProposal => BatchStatus::Unsupported,
                    GateVerdict::NonFinite | GateVerdict::WrongShape => BatchStatus::NonFinite,
                    _ => BatchStatus::Worse,
                };
                (fit_weighted_metric(tree, x, y, w, iters, metric), state)
            };
            note_gpu_fit(state == BatchStatus::Gpu);
            score.push(value);
            status.push(state);
            continue;
        }
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

// ---------------------------------------------------------------------------
// #233: the AUC arm — class-balanced logistic surrogate in the loop, exact AUC
// at candidate exit.
// ---------------------------------------------------------------------------

/// #233: the objective the **fit** kernel minimises for `metric`, including the
/// AUC arm's *proposal* loss.
///
/// [`fit_loss`](super::fit_loss) is the single source for metrics the kernel
/// minimises natively. `auc` has no native kernel objective — exact AUC is a
/// rank statistic — so the device proposes with the squared loss on the `{0, 1}`
/// label and every proposal is judged by [`gate_auc_proposal`]. Returning
/// `Some(DeviceLoss::Squared)` here is what admits the arm to the fit batch; it
/// is **never** the acceptance criterion, and the search keeps the AUC fitness
/// on the exact CPU evaluator (`GpuSearch::fit_only`).
pub fn gpu_proposal_loss(metric: Metric) -> Option<crate::fit::DeviceLoss> {
    if metric == Metric::Auc {
        return Some(crate::fit::DeviceLoss::Squared);
    }
    super::fit_loss(metric)
}

/// #233: whether a run may use the device **evaluator** for `metric`.
///
/// The CUDA evaluator accumulates a native kernel loss (squared, or
/// log-squashed since #229). That is the configured objective for the
/// least-squares family and `log_squashed`, and nothing else. `auc` is admitted
/// only when `--fit-strategy gpu` selects the fit arm, and even then it is a
/// *proposer*: [`super::GpuSearch`](crate) scores every candidate with the
/// exact CPU AUC, so the device never produces a fitness. Every other metric
/// keeps the CPU evaluator.
pub fn gpu_fitness_blocker(
    metric: Metric,
    fit_strategy: FitStrategy,
    fit_iters: usize,
) -> Option<&'static str> {
    if crate::fit::DeviceLoss::for_metric(metric).is_some() {
        return None;
    }
    match metric {
        Metric::Auc if fit_strategy.defers_to_gpu() && fit_iters > 0 => None,
        Metric::Auc => Some(
            "auc has no native device scorer; --fit-strategy gpu would use the device for the fit only",
        ),
        // #236: the historical, user-facing wording; a test and the backend
        // report quote it verbatim.
        _ => Some("metric is not scored natively on the GPU"),
    }
}

/// #233: the class-balanced logistic surrogate of `1 - auc`.
///
/// `y` is labelled by [`metrics::binary_labels`] and the model output is read as
/// a logit, exactly like [`Metric::LogLoss`]. The per-row weight is the
/// *balanced* one from the AUC literature: `n / (2 n_+)` for a positive and
/// `n / (2 n_-)` for a negative, so a 1 %-positive sample does not put 99 % of
/// the objective on the class AUC conditions away. A per-row base weight `w`
/// (empty = unweighted) multiplies it; a zero-weight row is invisible.
///
/// The loss is `Σ ω·softplus(-t·p) / Σ ω` with `t = ±1` and
/// `softplus(-u) = max(-u, 0) + ln(1 + e^{-|u|})`, the Lin–Lin–Weng form that
/// neither overflows nor loses the small-`u` term.
///
/// **This is not AUC.** It is AUC-consistent and is used as the in-loop
/// objective; the exact AUC at candidate exit is the acceptance test. It is
/// never reported as, or compared with, an AUC value, and a sample with an
/// empty class has no balanced weights at all, so it returns `+inf`.
pub fn balanced_logistic_loss(pred: &[f64], y: &[f64], w: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() || (!w.is_empty() && w.len() != pred.len()) {
        return f64::INFINITY;
    }
    let labels = metrics::binary_labels(y);
    // A zero-weight row does not exist: it counts for neither class and
    // contributes nothing, exactly as in `metrics::score_weighted`.
    let mut visible: Vec<(f64, f64, f64)> = Vec::with_capacity(pred.len());
    for (i, (&p, &l)) in pred.iter().zip(&labels).enumerate() {
        let base = w.get(i).copied().unwrap_or(1.0);
        if base == 0.0 {
            continue;
        }
        if !p.is_finite() || !base.is_finite() || base < 0.0 {
            return f64::INFINITY;
        }
        visible.push((p, l, base));
    }
    let n = visible.len() as f64;
    let npos = visible.iter().filter(|&&(_, l, _)| l > 0.5).count() as f64;
    let nneg = n - npos;
    if npos == 0.0 || nneg == 0.0 {
        return f64::INFINITY;
    }
    let (wp, wn) = (n / (2.0 * npos), n / (2.0 * nneg));
    let mut acc = 0.0;
    let mut total = 0.0;
    for (p, l, base) in visible {
        let omega = base * if l > 0.5 { wp } else { wn };
        let u = (2.0 * l - 1.0) * p;
        let softplus = if u >= 0.0 {
            (-u).exp().ln_1p()
        } else {
            -u + u.exp().ln_1p()
        };
        acc += omega * softplus;
        total += omega;
    }
    if !(total > 0.0) || !acc.is_finite() {
        return f64::INFINITY;
    }
    acc / total
}

/// #233: the verdict of the AUC gate for one device proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateVerdict {
    /// The proposal is no worse on the surrogate **and** no worse on exact AUC.
    Accept,
    /// The backend had nothing to offer for this candidate.
    NoProposal,
    /// The proposal had the wrong number of constants.
    WrongShape,
    /// A constant, prediction or score was not finite.
    NonFinite,
    /// The proposal is worse on exact AUC than the starting constants.
    MetricWorse,
    /// The proposal is worse on the class-balanced surrogate than the starting
    /// constants, although exact AUC may have improved. This is the measured
    /// surrogate-vs-exact disagreement; the candidate is reverted to `c0`.
    SurrogateWorse,
}

impl GateVerdict {
    /// Whether the device constants were written back to the tree.
    pub fn accepted(self) -> bool {
        matches!(self, GateVerdict::Accept)
    }

    /// Stable label for `--fit-report` and the run ledger.
    pub fn label(self) -> &'static str {
        match self {
            GateVerdict::Accept => "accept",
            GateVerdict::NoProposal => "no_proposal",
            GateVerdict::WrongShape => "wrong_shape",
            GateVerdict::NonFinite => "non_finite",
            GateVerdict::MetricWorse => "auc_worse",
            GateVerdict::SurrogateWorse => "auc_surrogate_worse",
        }
    }
}

/// #233: one gate decision with the numbers it was made from, all in exact
/// `f64` on the same rows.
#[derive(Clone, Debug)]
pub struct GateReport {
    pub verdict: GateVerdict,
    /// Exact `1 - auc` at the starting constants.
    pub start: f64,
    /// Exact `1 - auc` at the proposal (`NaN` when there was no usable one).
    pub proposal: f64,
    /// Class-balanced logistic surrogate at the starting constants.
    pub start_surrogate: f64,
    /// Class-balanced logistic surrogate at the proposal (`NaN` when there was
    /// no usable one).
    pub proposal_surrogate: f64,
}

/// #233: the exact-metric gate for one `auc` proposal.
///
/// The device LM minimises squared error on the `{0, 1}` label, so its
/// constants are a **proposal**. They are written back only when both of these
/// hold in exact CPU `f64` on the fitted rows:
///
/// ```text
/// 1 - auc(c)                <= 1 - auc(c0)                (the configured metric)
/// balanced_logistic(c)      <= balanced_logistic(c0)      (the in-loop objective)
/// ```
///
/// Otherwise the tree keeps its starting constants — the caller is free to run
/// the exact CPU arm from there, which only ever improves the same metric. The
/// invariant is therefore `metric(final) <= metric(start)` on every candidate,
/// whatever the device returned.
pub fn gate_auc_proposal(
    tree: &mut Node,
    proposal: Option<&[f64]>,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
) -> GateReport {
    let f = flatten(tree);
    let start = metric_score(tree, x, y, w, Metric::Auc);
    let start_surrogate = balanced_logistic_loss(&predictions(tree, x), y, w);
    let report = |verdict: GateVerdict, proposal: f64, proposal_surrogate: f64| GateReport {
        verdict,
        start,
        proposal,
        start_surrogate,
        proposal_surrogate,
    };
    let Some(c) = proposal else {
        return report(GateVerdict::NoProposal, f64::NAN, f64::NAN);
    };
    if c.len() != f.nconst {
        return report(GateVerdict::WrongShape, f64::NAN, f64::NAN);
    }
    if !c.iter().all(|v| v.is_finite()) {
        return report(GateVerdict::NonFinite, f64::NAN, f64::NAN);
    }
    let mut trial = tree.clone();
    write_constants(&mut trial, &f, c);
    let proposal_score = metric_score(&trial, x, y, w, Metric::Auc);
    let proposal_surrogate = balanced_logistic_loss(&predictions(&trial, x), y, w);
    // A single-class sample, a non-finite prediction or a zero total weight
    // makes the surrogate undefined; refuse rather than accept on `inf <= inf`.
    if !proposal_score.is_finite()
        || !proposal_surrogate.is_finite()
        || !start.is_finite()
        || !start_surrogate.is_finite()
    {
        return report(GateVerdict::NonFinite, proposal_score, proposal_surrogate);
    }
    if proposal_score > start {
        return report(GateVerdict::MetricWorse, proposal_score, proposal_surrogate);
    }
    if proposal_surrogate > start_surrogate {
        return report(GateVerdict::SurrogateWorse, proposal_score, proposal_surrogate);
    }
    write_constants(tree, &f, c);
    report(GateVerdict::Accept, proposal_score, proposal_surrogate)
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

    // ---- #233: the AUC arm -------------------------------------------------

    /// One constant per row: `c0·x0 + c1·x1 + c2·x2` on the one-hot rows, so a
    /// proposal can be any prediction vector of length three.
    fn one_hot_model() -> Node {
        node(
            Op::Add,
            vec![
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(1.0), input(0)]),
                        node(Op::Mul, vec![constant(1.0), input(1)]),
                    ],
                ),
                node(Op::Mul, vec![constant(1.0), input(2)]),
            ],
        )
    }

    fn one_hot_rows() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]
    }

    fn with_constants(tree: &mut Node, c: &[f64]) {
        let f = flatten(tree);
        crate::fit::write_constants(tree, &f, c);
    }

    fn predictions_of(tree: &Node, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| crate::expr::eval(tree, row)).collect()
    }

    /// The label rule is `y > (min+max)/2` on the observed target, so the same
    /// two values give the same labels whether they are `{0, 1}`, `{1, 2}` or
    /// normalised; a single-valued sample has no positive class at all.
    #[test]
    fn the_label_rule_is_the_target_midpoint() {
        assert_eq!(metrics::binary_labels(&[0.0, 1.0]), vec![0.0, 1.0]);
        assert_eq!(metrics::binary_labels(&[1.0, 2.0]), vec![0.0, 1.0]);
        assert_eq!(metrics::binary_labels(&[-1.0, 1.0]), vec![0.0, 1.0]);
        // The midpoint, not the median: an imbalanced sample.
        assert_eq!(
            metrics::binary_labels(&[0.0, 0.0, 0.0, 1.0]),
            vec![0.0, 0.0, 0.0, 1.0]
        );
        // Single-class: every row is negative, so AUC is chance (0.5).
        assert_eq!(metrics::binary_labels(&[3.0, 3.0]), vec![0.0, 0.0]);
        assert_eq!(metrics::class_counts(&[0.0, 0.0, 0.0, 1.0]), (1, 3));
        assert_eq!(metrics::class_counts(&[3.0, 3.0]), (0, 2));
        assert!(!balanced_logistic_loss(&[0.0, 0.0], &[3.0, 3.0], &[]).is_finite());
    }

    /// The balanced surrogate is the mean of the two per-class mean losses, so
    /// the class prior cannot dominate the step. The naive unweighted mean on
    /// the same margins is smaller, which is what the balance lifts.
    #[test]
    fn the_surrogate_balances_the_classes() {
        // 1 positive at a zero logit, 9 negatives correctly at -2.
        let mut pred = vec![0.0];
        pred.extend(std::iter::repeat(-2.0).take(9));
        let mut y = vec![1.0];
        y.extend(std::iter::repeat(0.0).take(9));
        let sp = |u: f64| u.exp().ln_1p();
        let expected = 0.5 * (sp(0.0) + sp(-2.0));
        let got = balanced_logistic_loss(&pred, &y, &[]);
        assert!((got - expected).abs() <= 1e-12, "{got} vs {expected}");
        let unweighted = (sp(0.0) + 9.0 * sp(-2.0)) / 10.0;
        assert!(
            got > unweighted + 0.2,
            "balancing must lift the minority class: {got} vs {unweighted}"
        );
        // Duplicating the majority class cannot change the balanced value.
        let mut pred9 = pred.clone();
        pred9.extend(std::iter::repeat(-2.0).take(9));
        let mut y9 = y.clone();
        y9.extend(std::iter::repeat(0.0).take(9));
        assert!((balanced_logistic_loss(&pred9, &y9, &[]) - got).abs() <= 1e-12);
        // A zero-weight row is invisible: adding a terrible one changes
        // nothing (both balanced weights and the total scale together).
        let mut pred_z = pred.clone();
        pred_z.push(100.0);
        let mut y_z = y.clone();
        y_z.push(0.0);
        let mut w_z = vec![1.0; 10];
        w_z.push(0.0);
        assert!((balanced_logistic_loss(&pred_z, &y_z, &w_z) - got).abs() <= 1e-12);
        // A negative base weight is not a weight.
        assert!(!balanced_logistic_loss(&pred, &y, &[-1.0; 10]).is_finite());
    }

    /// The surrogate is a margin loss and AUC is a rank statistic: the two
    /// disagree on a concrete prediction. A proposal that fixes the ranking but
    /// inflates the margins must be reverted, and one that only looks better in
    /// the smooth sum must never be accepted on that alone.
    #[test]
    fn the_surrogate_and_exact_auc_can_disagree() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        // c0 = (1.0, 1.5, 0.5): negative 1 outranks the positive, so exactly one
        // of the two pairs is right and the exact AUC is 0.5.
        let c0 = vec![1.0, 1.5, 0.5];
        let mut tree = one_hot_model();
        with_constants(&mut tree, &c0);
        let start = balanced_logistic_loss(&[1.0, 1.5, 0.5], &y, &[]);
        // The proposal fixes the ranking (AUC 1.0) by lifting the positive to 5
        // and negative 1 to 4.9: every pair is right, yet the second negative
        // now sits 4.5 logits below the positive and pays for it, so the
        // balanced smooth sum *rises*. Rank and margin disagree, and the
        // in-loop objective vetoes a metric improvement.
        let proposal = vec![5.0, 4.9, 0.5];
        let proposal_loss = balanced_logistic_loss(&[5.0, 4.9, 0.5], &y, &[]);
        assert!(proposal_loss > start, "{proposal_loss} vs {start}");
        assert!(
            metrics::auc(&[5.0, 4.9, 0.5], &y) > metrics::auc(&[1.0, 1.5, 0.5], &y),
            "the exact metric improves"
        );
        let report = gate_auc_proposal(&mut tree, Some(&proposal), &x, &y, &[]);
        assert_eq!(report.verdict, GateVerdict::SurrogateWorse);
        assert_eq!(report.start.to_bits(), 0.5f64.to_bits());
        assert_eq!(report.proposal.to_bits(), 0.0f64.to_bits());
        // Reverted to c0, bit for bit.
        assert_eq!(
            fitted(&tree).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// A proposal that is no worse on both objectives is accepted, and the
    /// reported score is the exact AUC, not the surrogate.
    #[test]
    fn the_gate_accepts_a_non_worse_proposal_on_the_exact_metric() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        let mut tree = one_hot_model();
        with_constants(&mut tree, &[1.0, 0.9, 0.8]);
        let report = gate_auc_proposal(&mut tree, Some(&[2.0, 0.9, 0.8]), &x, &y, &[]);
        assert_eq!(report.verdict, GateVerdict::Accept);
        assert_eq!(report.proposal.to_bits(), 0.0f64.to_bits(), "AUC 1 -> score 0");
        assert!(report.proposal_surrogate < report.start_surrogate);
        let pred = predictions_of(&tree, &x);
        assert_eq!(pred, vec![2.0, 0.9, 0.8]);
        assert_eq!(
            metrics::score(&pred, &y, Metric::Auc).to_bits(),
            report.proposal.to_bits()
        );
    }

    /// Exact AUC is a deterministic function of the rows: repeated calls and a
    /// permuted row order give the same bits (ranks, not row positions).
    #[test]
    fn the_exact_auc_at_exit_is_deterministic() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        let mut tree = one_hot_model();
        with_constants(&mut tree, &[1.0, 0.9, 0.8]);
        let a = gate_auc_proposal(&mut tree, Some(&[2.0, 0.9, 0.8]), &x, &y, &[]);
        let b = gate_auc_proposal(&mut tree, Some(&[2.0, 0.9, 0.8]), &x, &y, &[]);
        assert_eq!(a.proposal.to_bits(), b.proposal.to_bits());
        assert_eq!(a.proposal_surrogate.to_bits(), b.proposal_surrogate.to_bits());
        let order = [2, 0, 1];
        let xp: Vec<Vec<f64>> = order.iter().map(|&i| x[i].clone()).collect();
        let yp: Vec<f64> = order.iter().map(|&i| y[i]).collect();
        let mut tree2 = one_hot_model();
        with_constants(&mut tree2, &[1.0, 0.9, 0.8]);
        let c = gate_auc_proposal(&mut tree2, Some(&[2.0, 0.9, 0.8]), &xp, &yp, &[]);
        assert_eq!(a.proposal.to_bits(), c.proposal.to_bits());
    }

    /// Every unusable proposal is labelled and leaves `c0` untouched, so the
    /// caller can only improve it.
    #[test]
    fn bad_proposals_are_reverted_to_the_starting_constants() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        let c0 = [1.0, 0.9, 0.8];
        for (proposal, verdict) in [
            (None, GateVerdict::NoProposal),
            (Some(vec![1.0, 0.9]), GateVerdict::WrongShape),
            (Some(vec![f64::NAN, 0.9, 0.8]), GateVerdict::NonFinite),
            (Some(vec![1.0, f64::INFINITY, 0.8]), GateVerdict::NonFinite),
            // Both objectives worsen.
            (Some(vec![0.5, 0.9, 0.8]), GateVerdict::MetricWorse),
        ] {
            let mut tree = one_hot_model();
            with_constants(&mut tree, &c0);
            let report = gate_auc_proposal(&mut tree, proposal.as_deref(), &x, &y, &[]);
            assert_eq!(report.verdict, verdict, "{proposal:?}");
            assert!(!report.verdict.accepted());
            assert_eq!(
                fitted(&tree).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                c0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{proposal:?} moved the starting constants"
            );
        }
    }

    /// `fit_batch_verified` keeps the metric-aware contract: a rejected `auc`
    /// proposal is refit by the exact CPU arm on the untouched tree, so the
    /// returned score is bit-identical to that arm and never an SSE.
    #[test]
    fn a_rejected_auc_proposal_falls_back_to_the_exact_cpu_arm() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        let c0 = [1.0, 0.9, 0.8];
        for proposal in [None, Some(vec![f64::NAN, 0.9, 0.8]), Some(vec![1.0])] {
            let mut gpu = vec![one_hot_model()];
            with_constants(&mut gpu[0], &c0);
            let out = fit_batch_verified(
                &Backend { proposal },
                &mut gpu,
                &x,
                &y,
                &[],
                4,
                Metric::Auc,
            );
            assert_ne!(out.status[0], BatchStatus::Gpu);
            let mut cpu = one_hot_model();
            with_constants(&mut cpu, &c0);
            let reference = crate::fit::fit_weighted_metric(&mut cpu, &x, &y, &[], 4, Metric::Auc);
            assert_eq!(out.score[0].to_bits(), reference.to_bits());
            assert_eq!(
                fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                fitted(&cpu).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    /// A device proposal that passes the exact gate is written back and scored
    /// with the exact AUC, not with the kernel's squared proposal loss.
    #[test]
    fn an_accepted_auc_proposal_is_scored_with_the_exact_metric() {
        let x = one_hot_rows();
        let y = [1.0, 0.0, 0.0];
        let mut trees = vec![one_hot_model()];
        with_constants(&mut trees[0], &[1.0, 0.9, 0.8]);
        let out = fit_batch_verified(
            &Backend { proposal: Some(vec![2.0, 0.9, 0.8]) },
            &mut trees,
            &x,
            &y,
            &[],
            4,
            Metric::Auc,
        );
        assert_eq!(out.status[0], BatchStatus::Gpu);
        assert_eq!(out.score[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(fitted(&trees[0]), vec![2.0, 0.9, 0.8]);
    }

    /// #233: the arm is offered only for a native kernel loss or `auc`, and the
    /// AUC fitness never comes from the device.
    #[test]
    fn the_gate_admits_auc_only_as_a_proposal() {
        use crate::fit::FitStrategy;
        assert_eq!(gpu_proposal_loss(Metric::Auc), Some(crate::fit::DeviceLoss::Squared));
        assert_eq!(gpu_proposal_loss(Metric::Squared), Some(crate::fit::DeviceLoss::Squared));
        assert_eq!(gpu_proposal_loss(Metric::Absolute), None);
        assert!(gpu_fitness_blocker(Metric::Auc, FitStrategy::Exact, 8).is_some());
        assert!(gpu_fitness_blocker(Metric::Auc, FitStrategy::Gpu, 0).is_some());
        assert_eq!(gpu_fitness_blocker(Metric::Auc, FitStrategy::Gpu, 8), None);
        assert_eq!(gpu_fitness_blocker(Metric::Squared, FitStrategy::Exact, 8), None);
        assert_eq!(
            gpu_fitness_blocker(Metric::Absolute, FitStrategy::Gpu, 8),
            Some("metric is not scored natively on the GPU")
        );
    }

}
