//! #230: exact rank-correlation fitting — the batched polish.
//!
//! `defers_fit_to_gpu` used to accept `squared | aic | aicc | bic` only,
//! because the CUDA evaluator sums squared error. The rank family
//! (`rank_correlation`, `rank_correlation_signed`) fitted by
//! `fit_weighted_metric` is a squared pre-fit followed by
//! [`coordinate_polish`](super::coordinate_polish): a deterministic pattern
//! search whose trial scores are exact integer ratios of average ranks. This
//! module moves that tail to the device without changing what it optimises.
//!
//! The contract is deliberately narrow:
//!
//! 1. the **same ladder** ([`LADDER`]), in the **same order**, committing the
//!    **first strict improver** — never the best of the six trials;
//! 2. the **same score**: [`metrics::RankTarget`]'s integer sums
//!    (`Σ A·B`, `Σ A²`, `Σ B²` in `i128`), so a tie, a duplicated prediction or
//!    a `-0.0`/`+0.0` pair is ranked exactly as on the CPU and the value does
//!    not depend on reduction order;
//! 3. a scorer may decline (`None`) at any step; the caller then refits that
//!    batch with the exact CPU arm and labels the cause. A buggy backend can
//!    waste work but can never change the reported objective.
//!
//! This is the "exact-rank batched pattern search" recommended by
//! `arbeidsmappe/crypto-rcsr/gpu-metrics/rank-correlation.md`: the sort, not a
//! gradient, is the cost centre, and integer partials make the device and the
//! CPU agree bit for bit.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::*;

/// One scorer call: exact rank sums for every `(candidate, trial)` pair.
///
/// `plans[k]` is the flattened program of `trials[k]`; `trials[k][t]` is the
/// constant vector (slot order) the device should predict and rank. The result
/// has the same shape; `None` marks a segment whose prediction was not finite
/// and `None` for the whole vector means the device declined the call.
pub(crate) trait RankTrialScorer {
    /// One `Vec` per candidate, one entry per trial in `trials[k]`.
    fn score_trials(
        &mut self,
        plans: &[&GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<metrics::RankSums>>>>;
}

/// Score one segment's sums against the cached target.
fn score_of(sums: Option<metrics::RankSums>, target: &metrics::RankTarget, signed: bool) -> f64 {
    // `B` is the target's, so the target's `Σ B²` is authoritative: a device
    // that returned a different value would be a bug, and the score must not
    // depend on it.
    let sums = sums.map(|s| metrics::RankSums {
        bb: target.sum_bb(),
        ..s
    });
    metrics::score_rank_sums(sums, signed)
}

/// The constants a batched polish settled on.
#[derive(Clone, Debug)]
pub(crate) struct PolishResult {
    /// Final constants per candidate, slot order.
    pub(crate) constants: Vec<Vec<f64>>,
    /// Exact score of those constants, per candidate.
    pub(crate) scores: Vec<f64>,
    /// True when at least one ladder step was committed for some candidate.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) improved: bool,
}

/// Run the CPU coordinate-polish ladder for a whole batch, scoring every
/// candidate's six trials in one call per `(sweep, constant)`.
///
/// `starts` must have one constant vector per plan, in `plan.c0` slot order.
/// Returns `None` when the scorer declined; callers then use the exact CPU arm.
pub(crate) fn polish_rank_batch<S: RankTrialScorer + ?Sized>(
    scorer: &mut S,
    plans: &[&GpuFitPlan],
    starts: &[Vec<f64>],
    target: &metrics::RankTarget,
    signed: bool,
    sweeps: usize,
) -> Option<PolishResult> {
    if plans.len() != starts.len() {
        return None;
    }
    let n = plans.len();
    let mut c: Vec<Vec<f64>> = starts.to_vec();
    let mut active = vec![true; n];
    let mut best = vec![f64::INFINITY; n];
    let maxp = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
    if n == 0 || maxp == 0 {
        return Some(PolishResult {
            constants: c,
            scores: best,
            improved: false,
        });
    }

    // Baseline: the starting constants' exact score, in one batched call. The
    // CPU polish returns immediately when this is not finite.
    let baseline: Vec<Vec<Vec<f64>>> = c.iter().map(|v| vec![v.clone()]).collect();
    let sums = scorer.score_trials(plans, &baseline)?;
    if sums.len() != n || sums.iter().any(|t| t.len() != 1) {
        return None;
    }
    for k in 0..n {
        let s = score_of(sums[k][0], target, signed);
        if !s.is_finite() {
            active[k] = false;
        }
        best[k] = s;
    }
    if !active.iter().any(|&a| a) {
        return Some(PolishResult {
            constants: c,
            scores: best,
            improved: false,
        });
    }

    let mut improved_any = false;
    for _ in 0..sweeps.min(12) {
        let mut improved = false;
        for pi in 0..maxp {
            let idx: Vec<usize> = (0..n)
                .filter(|&k| active[k] && pi < plans[k].nconst)
                .collect();
            if idx.is_empty() {
                continue;
            }
            // The trial values follow `coordinate_polish` bit for bit: the same
            // ladder table and the same `original + sign * mul * scale`.
            let mut trials: Vec<Vec<Vec<f64>>> = Vec::with_capacity(idx.len());
            for &k in &idx {
                let slot = plans[k].order[pi];
                let original = c[k][slot];
                let scale = original.abs().max(1.0);
                let mut list = Vec::with_capacity(LADDER.len());
                for &(mul, sign) in LADDER.iter() {
                    let mut t = c[k].clone();
                    t[slot] = original + sign * mul * scale;
                    list.push(t);
                }
                trials.push(list);
            }
            let sub: Vec<&GpuFitPlan> = idx.iter().map(|&k| plans[k]).collect();
            let sums = scorer.score_trials(&sub, &trials)?;
            if sums.len() != idx.len() || sums.iter().any(|t| t.len() != LADDER.len()) {
                return None;
            }
            for (j, &k) in idx.iter().enumerate() {
                // First strict improver in ladder order, exactly like the CPU.
                for (t, trial_sums) in sums[j].iter().enumerate() {
                    let s = score_of(*trial_sums, target, signed);
                    if s.is_finite() && s < best[k] {
                        best[k] = s;
                        c[k] = trials[j][t].clone();
                        improved = true;
                        break;
                    }
                }
            }
        }
        if !improved {
            break;
        }
        improved_any = true;
    }
    Some(PolishResult {
        constants: c,
        scores: best,
        improved: improved_any,
    })
}

/// #230: exact rank score of one tree at its current constants, on the CPU.
///
/// It is the per-tree reference the parity tests and diagnostics use; the fit
/// path itself scores through [`metrics::RankTarget`] directly.
#[cfg_attr(not(test), allow(dead_code))]
///
/// Used by the verification gate after a device polish and by the tests that
/// compare the two paths. `None` when the metric is not a rank metric, the
/// target is not finite, weights are in play (the rank family refuses them) or
/// a prediction is not finite.
pub(crate) fn exact_rank_score(
    tree: &Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
) -> Option<f64> {
    let signed = super::rank_metric_signed(metric)?;
    if !w.is_empty() || x.len() != y.len() {
        return None;
    }
    let target = metrics::RankTarget::new(y)?;
    if target.len() != x.len() {
        return None;
    }
    let pred = super::predictions(tree, x);
    let s = target.score(&pred, signed);
    if s.is_finite() {
        Some(s)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::metrics::{centred_ranks_doubled, RankSums, RankTarget};

    /// The reference scorer: evaluates the tree with `expr::eval` and ranks the
    /// prediction with the exact CPU path. This is what the device is compared
    /// against, so it deliberately does not share code with the kernel.
    struct CpuScorer<'a> {
        tree: &'a Node,
        x: &'a [Vec<f64>],
        target: RankTarget,
    }

    impl<'a> RankTrialScorer for CpuScorer<'a> {
        fn score_trials(
            &mut self,
            plans: &[&GpuFitPlan],
            trials: &[Vec<Vec<f64>>],
        ) -> Option<Vec<Vec<Option<RankSums>>>> {
            let mut out = Vec::with_capacity(plans.len());
            for (plan, list) in plans.iter().zip(trials) {
                let flat = super::super::flatten(self.tree);
                let mut per_trial = Vec::with_capacity(list.len());
                for c in list {
                    if c.len() != plan.nconst {
                        per_trial.push(None);
                        continue;
                    }
                    let mut t = self.tree.clone();
                    super::super::write_constants(&mut t, &flat, c);
                    let pred: Vec<f64> = self.x.iter().map(|r| crate::expr::eval(&t, r)).collect();
                    per_trial.push(self.target.sums(&pred));
                }
                out.push(per_trial);
            }
            Some(out)
        }

    }

    fn fixture(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![-2.0 + 4.0 * i as f64 / (n - 1) as f64])
            .collect();
        // A smooth monotone target with exact duplicate rows: ties have to be
        // handled by the average-rank rule on both sides.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 7 == 3 {
                    0.0
                } else {
                    1.5 * (0.8 * x[i][0]).sin() + 0.25 * x[i][0]
                }
            })
            .collect();
        (x, y)
    }

    fn tree() -> Node {
        node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(0.7),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(0.6), input(0)])]),
                    ],
                ),
                constant(0.1),
            ],
        )
    }

    fn plan_of(t: &Node) -> GpuFitPlan {
        super::super::gpu_fit_plan_reason(t).expect("plan")
    }

    fn constants_of(t: &Node) -> Vec<f64> {
        let f = super::super::flatten(t);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        c
    }

    /// The plan's `order` must be the slot sequence of the CPU polish's
    /// `const_positions` order, not the slot numbering itself.
    #[test]
    fn plan_order_matches_const_position_order() {
        let t = node(
            Op::Add,
            vec![
                constant(9.0),
                node(
                    Op::Mul,
                    vec![constant(2.0), node(Op::Add, vec![constant(-1.0), input(0)])],
                ),
            ],
        );
        let plan = plan_of(&t);
        let mut positions = Vec::new();
        let mut counter = 0;
        super::super::const_positions(&t, &mut positions, &mut counter);
        assert_eq!(plan.order.len(), positions.len());
        let mut sorted = plan.order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..plan.nconst).collect::<Vec<_>>());
        // Constants are leaves, so flatten's post-order slot assignment meets
        // them left to right exactly like `const_positions`; the mapping is the
        // identity here and must not reorder the ladder.
        assert_eq!(plan.order, vec![0, 1, 2]);
    }

    #[test]
    fn batched_polish_reproduces_the_cpu_ladder_from_the_same_start() {
        let (x, y) = fixture(64);
        let target = RankTarget::new(&y).unwrap();
        for signed in [false, true] {
            let metric = if signed {
                Metric::RankCorrelationSigned
            } else {
                Metric::RankCorrelation
            };
            let mut cpu = tree();
            super::super::coordinate_polish(&mut cpu, &x, &y, &[], metric, 8);
            let cpu_c = constants_of(&cpu);

            let t = tree();
            let plan = plan_of(&t);
            let mut scorer = CpuScorer {
                tree: &t,
                x: &x,
                target: target.clone(),
            };
            let out = polish_rank_batch(&mut scorer, &[&plan], &[plan.c0.clone()], &target, signed, 8)
                .expect("cpu scorer");
            assert_eq!(
                out.constants[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                cpu_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "signed={signed}: the batched polish must commit the CPU ladder"
            );
        }
    }

    #[test]
    fn first_improver_wins_even_when_a_later_trial_scores_better() {
        // `y` is `x` with the rows 5 and 6 swapped: the objective is improved by
        // both ladder directions, so the committed value has to be the *first*
        // improver, exactly as the CPU commits it.
        let n = 40;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let mut y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        y.swap(5, 6);
        let t = node(Op::Add, vec![constant(1.0), input(0)]);
        let target = RankTarget::new(&y).unwrap();
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            target: target.clone(),
        };
        let out = polish_rank_batch(&mut scorer, &[&plan], &[vec![1.0]], &target, false, 1)
            .expect("cpu scorer");
        let moved = out.constants[0][0];
        if moved != 1.0 {
            assert_eq!(moved, 1.0 + 0.1, "the first ladder entry must win");
        }
        assert_ne!(moved, 1.0 - 0.1, "a later trial was committed");
    }

    #[test]
    fn a_constant_prediction_polishes_to_itself_and_scores_one() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let target = RankTarget::new(&y).unwrap();
        // A bare constant model: every ladder trial is still a constant
        // prediction, so nothing can improve and the start is returned.
        let t = constant(5.0);
        let plan = plan_of(&t);
        // The score of a constant prediction is exactly 1.0 and every trial is
        // also constant, so nothing can improve and the start is returned.
        assert_eq!(target.score(&[0.0; 8], false), 1.0);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            target: target.clone(),
        };
        let out = polish_rank_batch(&mut scorer, &[&plan], &[plan.c0.clone()], &target, false, 4)
            .expect("cpu scorer");
        assert_eq!(out.constants[0], plan.c0);
        assert!(!out.improved);
    }

    #[test]
    fn a_scorer_that_declines_reports_failure_instead_of_a_wrong_answer() {
        struct Decline;
        impl RankTrialScorer for Decline {
            fn score_trials(
                &mut self,
                _: &[&GpuFitPlan],
                _: &[Vec<Vec<f64>>],
            ) -> Option<Vec<Vec<Option<RankSums>>>> {
                None
            }
        }
        let (_, y) = fixture(16);
        let target = RankTarget::new(&y).unwrap();
        let t = tree();
        let plan = plan_of(&t);
        assert!(polish_rank_batch(
            &mut Decline,
            &[&plan],
            &[plan.c0.clone()],
            &target,
            false,
            4
        )
        .is_none());
    }

    #[test]
    fn a_batch_polish_is_deterministic_across_candidates() {
        let (x, y) = fixture(48);
        let target = RankTarget::new(&y).unwrap();
        let t = tree();
        let plan = plan_of(&t);
        let run = || {
            let mut scorer = CpuScorer {
                tree: &t,
                x: &x,
                target: target.clone(),
            };
            // The same candidate twice in one batch: independent coordinates, so
            // both must reach the same constants.
            let plans: Vec<&GpuFitPlan> = vec![&plan; 3];
            let starts = vec![plan.c0.clone(); 3];
            polish_rank_batch(&mut scorer, &plans, &starts, &target, false, 8)
                .unwrap()
                .constants
                .iter()
                .map(|c| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let a = run();
        assert_eq!(a[0], a[1]);
        assert_eq!(a[0], a[2]);
        assert_eq!(a, run(), "repeated calls must agree bit for bit");
    }

    #[test]
    fn exact_rank_score_refuses_what_it_cannot_score() {
        let (x, y) = fixture(20);
        let t = tree();
        assert!(exact_rank_score(&t, &x, &y, &[], Metric::RankCorrelation).is_some());
        assert!(exact_rank_score(&t, &x, &y, &[], Metric::Squared).is_none());
        assert!(exact_rank_score(&t, &x, &y, &[1.0; 20], Metric::RankCorrelation).is_none());
        // A non-finite prediction is not a score.
        let broken = node(Op::Ln, vec![node(Op::Mul, vec![constant(-1.0), input(0)])]);
        assert!(exact_rank_score(&broken, &x, &y, &[], Metric::RankCorrelation).is_none());
    }

    #[test]
    fn ties_and_negative_zero_rank_like_the_cpu() {
        let a = centred_ranks_doubled(&[1.0, -0.0, 0.0, 2.0]).unwrap();
        // -0.0 and +0.0 tie at rank 1.5 (A = -2), 1.0 is rank 3 (A = 1) and
        // 2.0 is rank 4 (A = 3).
        assert_eq!(a, vec![1, -2, -2, 3]);
    }
}
