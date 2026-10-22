//! #226: median absolute error — the batched exact polish.
//!
//! `fit_weighted_metric` fits `median` as a squared pre-fit followed by
//! [`coordinate_polish`](super::coordinate_polish): a deterministic pattern
//! search whose trial score is `medianᵢ |rᵢ|`. Every trial costs one full
//! prediction sweep plus a full sort of the residuals, and a sweep runs `6·p`
//! trials, so the tail is the cost centre of the whole fit.
//!
//! This module moves that tail onto the device **without changing what it
//! optimises**, following the #230 rank polish exactly:
//!
//! 1. the **same ladder** ([`LADDER`]), in the same constant order, committing
//!    the **first strict improver** — never the best of the six trials;
//! 2. the **same score**: the exact `f64` median of `|pred − y|`, with the
//!    even-`n` mean of the two middle values and no tie-break that can change
//!    the value (`median` is a symmetric function of the multiset `{|rᵢ|}`);
//! 3. a scorer may decline (`None`) at any step; the caller then refits that
//!    batch with the exact CPU arm and labels the cause. A buggy backend can
//!    waste work but can never change the reported objective.
//!
//! The device computes the median as a **segmented sort of the `|r|` bit
//! patterns** (see `src/cuda/eval.cu`, `rc_median_score_batch`). Because
//! `|r| ≥ 0`, its IEEE-754 `u64` pattern orders exactly like the value, so the
//! sorted keys give the two middle values of the CPU's
//! [`metrics::score_weighted`] directly. The `f32` prediction is *not* used:
//! the arm predicts in `f64` from the resident exact rows, which is what makes
//! the device and the CPU scores agree to the last bit on arithmetic-only
//! trees (transcendental operators may differ by an ulp per row, exactly as in
//! the landed #169 arm).
//!
//! This is one slice of issue #226: the warm start is the squared LM kernel.
//! The Huber/IRLS continuation and the interquartile window mean need the
//! weighted assembly that the `absolute` arm will land (`l1-quantile.md` §4.2),
//! and stay open.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::rank::PolishResult;
use super::*;

/// One scorer call: the exact median absolute residual for every
/// `(candidate, trial)` pair.
///
/// `plans[k]` is the flattened program of candidate `k`; `trials[k][t]` is the
/// constant vector (slot order) to predict with. The result has the same shape:
/// `None` for a segment whose prediction was not finite, and `None` for the
/// whole vector when the device declined the call.
pub(crate) trait MedianTrialScorer {
    /// One `Vec` per candidate, one entry per trial in `trials[k]`.
    fn score_trials(
        &mut self,
        plans: &[&GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<f64>>>>;
}

/// Run the CPU coordinate-polish ladder for a whole batch, scoring every
/// candidate's six trials in one call per `(sweep, constant)`.
///
/// `starts` must have one constant vector per plan, in `plan.c0` slot order
/// (the caller passes the squared-LM warm start). Returns `None` when the
/// scorer declined; callers then use the exact CPU arm.
///
/// The rule is `coordinate_polish`'s: per constant, walk [`LADDER`] and commit
/// the **first** trial whose exact score is finite and strictly below the best
/// seen so far; stop a sweep when no constant improved; at most
/// `min(sweeps, 12)` sweeps.
pub(crate) fn polish_median_batch<S: MedianTrialScorer + ?Sized>(
    scorer: &mut S,
    plans: &[&GpuFitPlan],
    starts: &[Vec<f64>],
    sweeps: usize,
) -> Option<PolishResult> {
    let n = plans.len();
    if n != starts.len() {
        return None;
    }
    let maxp = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
    let mut c: Vec<Vec<f64>> = starts.to_vec();
    if n == 0 || maxp == 0 {
        return Some(PolishResult {
            constants: c,
            scores: vec![f64::INFINITY; n],
            improved: false,
        });
    }

    // Baseline: the warm start's exact median, in one batched call. A start
    // the scorer cannot score is inactive, exactly like the CPU polish
    // returning early on a non-finite score.
    let baseline: Vec<Vec<Vec<f64>>> = c.iter().map(|v| vec![v.clone()]).collect();
    let values = scorer.score_trials(plans, &baseline)?;
    if values.len() != n || values.iter().any(|t| t.len() != 1) {
        return None;
    }
    let mut best = vec![f64::INFINITY; n];
    let mut active = vec![true; n];
    for k in 0..n {
        match values[k][0] {
            Some(v) if v.is_finite() => best[k] = v,
            _ => active[k] = false,
        }
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
            let values = scorer.score_trials(&sub, &trials)?;
            if values.len() != idx.len() || values.iter().any(|t| t.len() != LADDER.len()) {
                return None;
            }
            for (j, &k) in idx.iter().enumerate() {
                // First strict improver in ladder order, exactly like the CPU.
                for (t, value) in values[j].iter().enumerate() {
                    if let Some(v) = *value {
                        if v.is_finite() && v < best[k] {
                            best[k] = v;
                            c[k] = trials[j][t].clone();
                            improved = true;
                            break;
                        }
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

/// Exact median of a `|r|` key multiset, the rule the device reproduces.
///
/// The keys are the `f64` bit patterns of the (non-negative) absolute residual,
/// so sorting them sorts the values. `n` odd: the middle value. `n` even: the
/// CPU's `0.5 * (a + b)` of the two middle values. Ties need no tie-break: the
/// median is a function of the multiset, so equal keys cannot change it.
#[cfg(test)]
pub(crate) fn median_of_keys(keys: &mut [u64]) -> Option<f64> {
    if keys.is_empty() {
        return None;
    }
    keys.sort_unstable();
    let n = keys.len();
    let v = |k: u64| f64::from_bits(k);
    Some(if n % 2 == 1 {
        v(keys[n / 2])
    } else {
        0.5 * (v(keys[n / 2 - 1]) + v(keys[n / 2]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::metrics::score_weighted;

    /// The reference scorer: evaluates the tree with `expr::eval` in `f64` and
    /// scores with the configured metric, exactly like `metric_score`. It
    /// deliberately shares no code with the device selection.
    struct CpuScorer<'a> {
        tree: &'a Node,
        x: &'a [Vec<f64>],
        y: &'a [f64],
    }

    impl MedianTrialScorer for CpuScorer<'_> {
        fn score_trials(
            &mut self,
            plans: &[&GpuFitPlan],
            trials: &[Vec<Vec<f64>>],
        ) -> Option<Vec<Vec<Option<f64>>>> {
            let flat = super::super::flatten(self.tree);
            let mut out = Vec::with_capacity(plans.len());
            for (plan, list) in plans.iter().zip(trials) {
                let mut per_trial = Vec::with_capacity(list.len());
                for c in list {
                    if c.len() != plan.nconst {
                        per_trial.push(None);
                        continue;
                    }
                    let mut t = self.tree.clone();
                    super::super::write_constants(&mut t, &flat, c);
                    let pred: Vec<f64> = self.x.iter().map(|r| crate::expr::eval(&t, r)).collect();
                    let v = score_weighted(&pred, self.y, &[], Metric::Median, plan.nconst);
                    per_trial.push(if v.is_finite() { Some(v) } else { None });
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
        // A smooth target with exact duplicates and a few gross outliers, so
        // ties and the median's rank structure both matter.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 5 == 2 {
                    1.0
                } else if i % 11 == 7 {
                    25.0
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

    /// The key rule (sort the `|r|` bit patterns, average the two middles for
    /// even `n`) is exactly `metrics::score_weighted(.., Median)`.
    #[test]
    fn the_key_median_is_the_configured_metric_for_odd_even_and_tied_rows() {
        let cases: [&[f64]; 9] = [
            &[],
            &[3.0],
            &[3.0, 1.0],
            &[3.0, 1.0, 2.0],
            &[4.0, 1.0, 3.0, 2.0],
            &[2.0, 2.0, 2.0, 2.0],
            &[0.0, 0.0, 5.0, 5.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            &[-0.0, 0.0, -0.0, 1.0, 1.0],
        ];
        for resid in cases {
            let y = vec![0.0; resid.len()];
            let pred: Vec<f64> = resid.to_vec();
            let mut keys: Vec<u64> = resid.iter().map(|r| r.abs().to_bits()).collect();
            let want = score_weighted(&pred, &y, &[], Metric::Median, 0);
            match median_of_keys(&mut keys) {
                None => assert!(resid.is_empty()),
                Some(v) => assert_eq!(v.to_bits(), want.to_bits(), "residuals {resid:?}"),
            }
        }
    }

    /// The batched driver commits the same constants the CPU polish commits
    /// from the same starting point, bit for bit.
    #[test]
    fn batched_polish_reproduces_the_cpu_ladder_from_the_same_start() {
        let (x, y) = fixture(96);
        let mut cpu = tree();
        super::super::coordinate_polish(&mut cpu, &x, &y, &[], Metric::Median, 8);
        let cpu_c = constants_of(&cpu);

        let t = tree();
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        let out = polish_median_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 8)
            .expect("cpu scorer");
        assert_eq!(
            out.constants[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            cpu_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the batched polish must commit the CPU ladder"
        );
        let want = score_weighted(
            &x.iter().map(|r| crate::expr::eval(&cpu, r)).collect::<Vec<_>>(),
            &y,
            &[],
            Metric::Median,
            3,
        );
        assert_eq!(out.scores[0].to_bits(), want.to_bits());
    }

    /// A constant prediction scores its own (constant) median and the ladder
    /// can only move it: the driver keeps the start when nothing improves.
    #[test]
    fn a_constant_prediction_polishes_to_itself_when_it_is_the_median() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let t = constant(3.5);
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        let out = polish_median_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 4)
            .expect("cpu scorer");
        // |pred − y| = 0.5, 2.5, 0.5, 2.5, 1.5, 5.5, 1.5, 2.5 -> sorted middles
        // 1.5 and 2.5 -> median 2.0. `3.5` is the median of the target, so no
        // ladder step can improve it.
        assert_eq!(out.scores[0], 2.0);
        assert!(!out.improved, "a single constant cannot improve its median here");
        assert_eq!(out.constants[0], plan.c0);
    }

    /// Even-row median: the average of the two middle residuals, at both an
    /// even and an odd row count.
    #[test]
    fn even_and_odd_row_counts_score_the_documented_median() {
        // n = 4 residuals 1, 2, 3, 10 -> middle 2 and 3 -> 2.5.
        let x = vec![vec![0.0]; 4];
        let y = vec![1.0, 2.0, 3.0, 10.0];
        let t = constant(0.0);
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        // `0` sweeps: the baseline score of the start, no ladder step.
        let out = polish_median_batch(&mut scorer, &[&plan], &[vec![0.0]], 0).unwrap();
        assert_eq!(out.scores[0], 0.5 * (2.0 + 3.0));
        // n = 5 residuals 1, 2, 3, 10, 11 -> middle 3.
        let x = vec![vec![0.0]; 5];
        let y = vec![1.0, 2.0, 3.0, 10.0, 11.0];
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        let out = polish_median_batch(&mut scorer, &[&plan], &[vec![0.0]], 0).unwrap();
        assert_eq!(out.scores[0], 3.0);
    }

    /// A scorer that declines is reported as such, never as a wrong answer.
    #[test]
    fn a_scorer_that_declines_reports_failure_instead_of_a_wrong_answer() {
        struct Decline;
        impl MedianTrialScorer for Decline {
            fn score_trials(
                &mut self,
                _: &[&GpuFitPlan],
                _: &[Vec<Vec<f64>>],
            ) -> Option<Vec<Vec<Option<f64>>>> {
                None
            }
        }
        let t = tree();
        let plan = plan_of(&t);
        assert!(polish_median_batch(&mut Decline, &[&plan], &[plan.c0.clone()], 4).is_none());
    }

    /// The same candidate three times in one batch reaches the same constants,
    /// and repeated calls agree bit for bit.
    #[test]
    fn a_batch_polish_is_deterministic_across_candidates() {
        let (x, y) = fixture(64);
        let t = tree();
        let plan = plan_of(&t);
        let run = || {
            let mut scorer = CpuScorer {
                tree: &t,
                x: &x,
                y: &y,
            };
            let plans: Vec<&GpuFitPlan> = vec![&plan; 3];
            let starts = vec![plan.c0.clone(); 3];
            polish_median_batch(&mut scorer, &plans, &starts, 8)
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

    /// A non-finite start is inactive, so the driver returns it untouched with
    /// a non-finite score and never a fabricated improvement.
    #[test]
    fn a_non_finite_start_is_left_inactive() {
        let x = vec![vec![0.0]; 8];
        let y = vec![0.0; 8];
        let t = constant(1.0);
        let _ = (&x, &y, &t);
        struct NonFinite;
        impl MedianTrialScorer for NonFinite {
            fn score_trials(
                &mut self,
                plans: &[&GpuFitPlan],
                trials: &[Vec<Vec<f64>>],
            ) -> Option<Vec<Vec<Option<f64>>>> {
                Some(
                    plans
                        .iter()
                        .zip(trials)
                        .map(|(_, list)| list.iter().map(|_| None).collect())
                        .collect(),
                )
            }
        }
        let plan = plan_of(&t);
        let out = polish_median_batch(&mut NonFinite, &[&plan], &[plan.c0.clone()], 4).unwrap();
        assert_eq!(out.constants[0], plan.c0);
        assert!(!out.scores[0].is_finite());
        assert!(!out.improved);
    }
}
