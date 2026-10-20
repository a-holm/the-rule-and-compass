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

/// A verified batch result: exact `f64` score and the outcome for each
/// candidate, in input order.
#[derive(Clone, Debug)]
pub struct VerifiedBatch {
    pub sse: Vec<f64>,
    pub status: Vec<BatchStatus>,
}

/// Run a batched backend through the exactness gate.
///
/// The tree is modified only to the verified constants (or to the exact CPU
/// fallback's result). Each candidate's final SSE is the exact weighted SSE of
/// the constants now in the tree.
pub fn fit_batch_verified<B: BatchedFitter + ?Sized>(
    backend: &B,
    trees: &mut [Node],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
) -> VerifiedBatch {
    let proposals = backend.fit_batch(trees, x, y, w, iters);
    let mut sse = Vec::with_capacity(trees.len());
    let mut status = Vec::with_capacity(trees.len());
    for (i, tree) in trees.iter_mut().enumerate() {
        let proposal = proposals.get(i).and_then(|p| p.as_ref());
        let (value, state) = match proposal {
            None => (
                fit_weighted_strategy(tree, x, y, w, iters, FitStrategy::Exact),
                BatchStatus::Unsupported,
            ),
            Some(c) => match verify_proposal(tree, c, x, y, w) {
                Some(v) => (v, BatchStatus::Gpu),
                None => (
                    fit_weighted_strategy(tree, x, y, w, iters, FitStrategy::Exact),
                    BatchStatus::Worse,
                ),
            },
        };
        note_gpu_fit(state == BatchStatus::Gpu);
        sse.push(value);
        status.push(state);
    }
    VerifiedBatch { sse, status }
}

/// Apply one proposal to a clone and return its exact SSE only when it is a
/// legitimate, non-worse fit. The real tree is untouched on failure.
fn verify_proposal(
    tree: &mut Node,
    proposal: &[f64],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
) -> Option<f64> {
    let f = flatten(tree);
    if proposal.len() != f.nconst || !proposal.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut v = vec![0.0; f.op.len()];
    let mut c0 = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c0[f.cslot[i] as usize] = f.val[i];
        }
    }
    let start = sse(&f, x, y, w, &c0, &mut v);
    let candidate = sse(&f, x, y, w, proposal, &mut v);
    if !candidate.is_finite() || !start.is_finite() || candidate > start {
        return None;
    }
    write_constants(tree, &f, proposal);
    Some(candidate)
}

/// #169: the exact `f64` refit every reported finalist receives, whatever arm
/// produced it. This is the historical LM path, bit for bit.
pub fn refit_finalists_exact(trees: &mut [Node], x: &[Vec<f64>], y: &[f64], w: &[f64], iters: usize) -> Vec<f64> {
    trees
        .iter_mut()
        .map(|tree| {
            note_gpu_refit();
            fit_lm(tree, x, y, w, iters)
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
        let out = fit_batch_verified(&backend, &mut trees, &x, &y, &[], 8);
        assert_eq!(out.status, vec![BatchStatus::Gpu]);
        assert!(out.sse[0] < 1e-20);
        let c = fitted(&trees[0]);
        assert!((c[0] - 2.0).abs() < 1e-12 && (c[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_and_wrong_length_proposals_fall_back_exactly() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();
        for proposal in [Some(vec![f64::NAN, 3.0]), Some(vec![2.0]), None] {
            let mut gpu = vec![model()];
            let out = fit_batch_verified(&Backend { proposal }, &mut gpu, &x, &y, &[], 8);
            let mut cpu = model();
            let reference = fit_weighted(&mut cpu, &x, &y, &[], 8);
            assert_eq!(out.sse[0].to_bits(), reference.to_bits());
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
        );
        assert_ne!(out.status[0], BatchStatus::Gpu);
        let mut cpu = model();
        let reference = fit_weighted(&mut cpu, &x, &y, &[], 8);
        assert_eq!(out.sse[0].to_bits(), reference.to_bits());
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
        let a = refit_finalists_exact(&mut gpu, &x, &y, &[], 40);
        let b = fit_weighted(&mut cpu, &x, &y, &[], 40);
        assert_eq!(a[0].to_bits(), b.to_bits());
        let ca: Vec<u64> = fitted(&gpu[0]).iter().map(|v| v.to_bits()).collect();
        let cb: Vec<u64> = fitted(&cpu).iter().map(|v| v.to_bits()).collect();
        assert_eq!(ca, cb);
    }
}
