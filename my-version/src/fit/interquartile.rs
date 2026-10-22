//! #232: interquartile absolute error — the batched middle-50 % window mean.
//!
//! `--metric interquartile` scores the mean of the absolute residuals that fall
//! between the 25th and 75th percentile (the "interquartile mean absolute
//! error"): with `a = sorted |r|` and `n = a.len()`,
//!
//! ```text
//! n < 4 :   mean(a)                        (the CPU arm's small-`n` rule)
//! else  :   mean(a[n / 4 .. (3 * n) / 4])   (integer division, so ⌊·⌋)
//! ```
//!
//! which is exactly [`crate::metrics::score_weighted`]'s `Interquartile` branch
//! (`metrics::interquartile_mean_abs`, including its `hi.max(lo + 1)` widening,
//! which only fires below `n = 4`).
//!
//! The CPU arm is `fit_weighted_metric`'s order-statistic family: a squared
//! pre-fit followed by [`coordinate_polish`](super::coordinate_polish) on that
//! window mean. Every trial pays a full prediction sweep **and** a full sort of
//! the residuals, and a sweep is `6·p` trials. This module runs the *same*
//! ladder on the device through the shared #226 driver
//! ([`polish_order_stat_batch`](super::median::polish_order_stat_batch)) with
//! [`OrderStat::Interquartile`]:
//!
//! 1. the same [`LADDER`](super::LADDER) table, the same constant order, the
//!    **first strict improver** per constant, at most `min(sweeps, 12)` sweeps;
//! 2. the same objective, scored from the exact `f64` rows the evaluator
//!    uploaded (no `f32` round trip on the scoring path);
//! 3. `None` from the scorer means "declined": the caller refits the batch with
//!    the exact CPU arm and labels the cause, so a buggy backend can waste work
//!    but cannot change the reported objective.
//!
//! **Boundary and tie rules.** The window is defined by **rank**, never by a
//! value interval: `a = [1, 2, 3, 3]` (`n = 4`) has `lo = 1`, `hi = 3` and a
//! window mean of `2.5`, while the values inside `[a[lo], a[hi-1]] = [2, 3]`
//! average `8/3` because the top boundary value repeats. So the device selects
//! ranks (the same `u64` bit-pattern sort the median arm uses) and sums exactly
//! those ranks. Ties therefore cannot change the value: the window mean is a
//! function of the `|r|` multiset, and CUB's radix sort is stable, so equal keys
//! keep row order and the selected *set* is reproducible as well.
//!
//! **Exactness.** The window sum is a deterministic fixed-width tree reduction
//! over ranks `[lo, hi)` (`k_order_stat_pick` in `src/cuda/eval.cu`), so the
//! device value is deterministic but agrees with the CPU's left-to-right `f64`
//! sum only to rounding — measured at or below `1e-12` relative, which is the
//! tolerance the design note prescribes (`l1-quantile.md` §7). The median pick
//! is a single value and does agree bit for bit; the difference is a property
//! of averaging, not of the selection. Every accepted or rejected proposal is
//! re-scored by the exact CPU arm anyway, and every finalist is refit by
//! `fit_weighted_metric(.., Interquartile)`.
//!
//! **Not promised:** a global optimum. The interquartile mean is not convex in
//! the constants, and neither this arm nor the CPU ladder finds a global
//! optimum (`l1-quantile.md` §2, §9).

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::median::{polish_order_stat_batch, OrderStatTrialScorer};
use super::rank::PolishResult;
use super::*;

/// Run the CPU coordinate-polish ladder for a whole batch on the interquartile
/// window mean.
///
/// Identical to [`super::polish_median_batch`](super::median) except for the
/// statistic, so the two arms cannot drift: both go through
/// [`polish_order_stat_batch`] with the same ladder and the same rule.
pub(crate) fn polish_interquartile_batch<S: OrderStatTrialScorer + ?Sized>(
    scorer: &mut S,
    plans: &[&GpuFitPlan],
    starts: &[Vec<f64>],
    sweeps: usize,
) -> Option<PolishResult> {
    polish_order_stat_batch(scorer, plans, starts, sweeps, OrderStat::Interquartile)
}

/// The exact interquartile window mean of a `|r|` key multiset, the rule the
/// device reproduces.
///
/// The keys are the `f64` bit patterns of the (non-negative) absolute residual,
/// so sorting them sorts the values. The window is ranks `[n/4, 3n/4)` (all
/// rows below `n = 4`), summed in rank order exactly like
/// `metrics::interquartile_mean_abs` sums its sorted slice. Ties need no
/// tie-break: the mean is a function of the multiset.
#[cfg(test)]
pub(crate) fn interquartile_of_keys(keys: &mut [u64]) -> Option<f64> {
    if keys.is_empty() {
        return None;
    }
    keys.sort_unstable();
    let n = keys.len();
    let (lo, hi) = if n < 4 { (0, n) } else { (n / 4, (3 * n) / 4) };
    let mut sum = 0.0f64;
    for &k in &keys[lo..hi] {
        sum += f64::from_bits(k);
    }
    Some(sum / (hi - lo) as f64)
}

#[cfg(test)]
mod tests {
    use super::median::test_support::*;
    use super::*;
    use crate::expr::constant;
    use crate::metrics::score_weighted;

    fn keys_of(resid: &[f64]) -> Vec<u64> {
        resid.iter().map(|r| r.abs().to_bits()).collect()
    }

    /// The rank rule is `metrics::score_weighted(.., Interquartile)` for odd and
    /// even `n`, including the `n < 4` fallback to the plain mean.
    #[test]
    fn the_key_rule_is_the_configured_metric_for_odd_even_and_small_rows() {
        let cases: [&[f64]; 12] = [
            &[],
            &[3.0],
            &[3.0, 1.0],
            &[3.0, 1.0, 2.0],
            &[4.0, 1.0, 3.0, 2.0],
            &[1.0, 2.0, 3.0, 3.0],
            &[5.0, 5.0, 1.0, 5.0, 5.0],
            &[0.0, 0.0, 5.0, 5.0],
            &[2.0; 9],
            &[-0.0, 0.0, -0.0, 1.0, 1.0],
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            &[7.0, 1.0, 3.0, 9.0, 5.0, 11.0, 13.0],
        ];
        for resid in cases {
            let y = vec![0.0; resid.len()];
            let pred: Vec<f64> = resid.to_vec();
            let mut keys = keys_of(resid);
            let want = score_weighted(&pred, &y, &[], Metric::Interquartile, 0);
            match interquartile_of_keys(&mut keys) {
                None => assert!(resid.is_empty()),
                Some(v) => assert_eq!(v.to_bits(), want.to_bits(), "residuals {resid:?}"),
            }
        }
    }

    /// The window really is `[⌊n/4⌋, ⌊3n/4⌋)`: for `n = 4, 5` that is two rows.
    #[test]
    fn the_window_bounds_are_the_floored_quarter_indices() {
        for n in 4..=17usize {
            let resid: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let lo = n / 4;
            let hi = (3 * n) / 4;
            assert!(hi > lo, "n={n}: the window is never empty above 3");
            let want: f64 =
                resid[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            let mut keys = keys_of(&resid);
            assert_eq!(
                interquartile_of_keys(&mut keys).unwrap().to_bits(),
                want.to_bits(),
                "n={n}"
            );
            // And the same as the configured metric, which sorts the residuals.
            let y = vec![0.0; n];
            let pred = resid.clone();
            assert_eq!(
                score_weighted(&pred, &y, &[], Metric::Interquartile, 0).to_bits(),
                want.to_bits(),
                "n={n}"
            );
        }
    }

    /// A boundary tie cannot change the mean, and neither can the choice of
    /// which tied row sits at the edge of the window.
    ///
    /// `[1, 2, 3, 3]`: ranks 1..3 are `2` and `3`, mean `2.5`. The value-range
    /// shortcut ("everything in `[a[lo], a[hi-1]]`") would average `2, 3, 3` and
    /// report `8/3`; this pins the rank semantics.
    #[test]
    fn ties_at_either_boundary_cannot_change_the_window_mean() {
        let mut keys = keys_of(&[1.0, 2.0, 3.0, 3.0]);
        assert_eq!(interquartile_of_keys(&mut keys), Some(2.5));
        let value_range: f64 = [2.0, 3.0, 3.0].iter().sum::<f64>() / 3.0;
        assert!(
            (value_range - 8.0 / 3.0).abs() < 1e-15,
            "the shortcut really does differ"
        );
        assert_ne!(value_range, 2.5, "the fixture must separate the two rules");

        // All rows tied at a boundary value: every rule agrees, and the mean is
        // the tied value.
        let mut keys = keys_of(&[5.0; 8]);
        assert_eq!(interquartile_of_keys(&mut keys), Some(5.0));
        // A bottom-boundary tie: [1, 1, 2, 3] -> window [1, 2] -> 1.5, and
        // every tied `1` would give the same window multiset.
        let mut keys = keys_of(&[1.0, 1.0, 2.0, 3.0]);
        assert_eq!(interquartile_of_keys(&mut keys), Some(1.5));
        // Exactly half the rows tied at the top boundary.
        let mut keys = keys_of(&[0.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        // n = 8 -> ranks 2..6 -> 9, 9, 9, 9 -> 9.
        assert_eq!(interquartile_of_keys(&mut keys), Some(9.0));
    }

    /// A constant prediction scores the mean of the middle 50 % of its own
    /// residuals; the ladder may then move the constant, but it can only lower
    /// that number.
    #[test]
    fn a_constant_prediction_scores_its_own_window_mean() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let t = constant(3.5);
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        // |pred − y| = .5, 2.5, .5, 2.5, 1.5, 5.5, 1.5, 2.5 -> sorted
        // .5, .5, 1.5, 1.5, 2.5, 2.5, 2.5, 5.5 -> `n = 8`, ranks 2..6 ->
        // 1.5, 1.5, 2.5, 2.5 -> 2.0. `0` sweeps: the baseline only.
        let base = polish_interquartile_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 0)
            .expect("cpu scorer");
        assert_eq!(base.scores[0], 2.0);
        assert_eq!(base.constants[0], plan.c0);
        assert!(!base.improved);
        // With sweeps the ladder may move the single constant, strictly
        // lowering the window mean and never raising it.
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        let out = polish_interquartile_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 4)
            .expect("cpu scorer");
        assert!(out.scores[0] <= base.scores[0]);
        let want = score_weighted(
            &x.iter().map(|r| crate::expr::eval(&t, r)).collect::<Vec<_>>(),
            &y,
            &[],
            Metric::Interquartile,
            1,
        );
        assert_eq!(want, 2.0);
    }

    /// The batched driver commits exactly the constants the CPU
    /// `coordinate_polish` commits from the same start, and the score is the
    /// same number: the ladder, not the GPU, is what is being tested.    #[test]
    fn batched_polish_reproduces_the_cpu_ladder_from_the_same_start() {
        let (x, y) = fixture(96);
        let mut cpu = tree();
        super::super::coordinate_polish(&mut cpu, &x, &y, &[], Metric::Interquartile, 8);
        let cpu_c = constants_of(&cpu);
        let t = tree();
        let plan = plan_of(&t);
        let mut scorer = CpuScorer {
            tree: &t,
            x: &x,
            y: &y,
        };
        let out = polish_interquartile_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 8)
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
            &x.iter()
                .map(|r| crate::expr::eval(&cpu, r))
                .collect::<Vec<_>>(),
            &y,
            &[],
            Metric::Interquartile,
            3,
        );
        assert_eq!(out.scores[0].to_bits(), want.to_bits());
        // The ladder must have moved something, else the parity is vacuous.
        assert!(out.improved, "the fixture must exercise the ladder");
    }

    /// The first strict improver, not the best of the six: the first trial that
    /// beats the best score wins even though the sixth is strictly better.
    #[test]
    fn the_driver_commits_the_first_strict_improver_not_the_best() {
        struct Scripted {
            calls: usize,
        }
        impl OrderStatTrialScorer for Scripted {
            fn score_trials(
                &mut self,
                plans: &[&GpuFitPlan],
                trials: &[Vec<Vec<f64>>],
                want: OrderStat,
            ) -> Option<Vec<Vec<Option<f64>>>> {
                assert_eq!(want, OrderStat::Interquartile);
                // The first call of a session is the baseline; every later call
                // is one constant's six ladder trials, scored `10 − t` so the
                // first trial improves and the last is the best of the six.
                let baseline = self.calls == 0;
                self.calls += 1;
                Some(
                    plans
                        .iter()
                        .zip(trials)
                        .map(|(_, list)| {
                            list.iter()
                                .enumerate()
                                .map(|(t, _)| Some(if baseline { 20.0 } else { 10.0 - t as f64 }))
                                .collect()
                        })
                        .collect(),
                )
            }
        }
        let t = tree();
        let plan = plan_of(&t);
        let mut scorer = Scripted { calls: 0 };
        let out = polish_interquartile_batch(&mut scorer, &[&plan], &[plan.c0.clone()], 1)
            .expect("scripted scorer");
        // The first constant is committed at the first trial, not at the sixth:
        // `coordinate_polish` walks the ladder in order and breaks on the first
        // strict improver, so the committed value is the +10 % rung.
        let slot = plan.order[0];
        let scale = plan.c0[slot].abs().max(1.0);
        assert_eq!(out.constants[0][slot], plan.c0[slot] + 0.1 * scale);
        // Three constants, one sweep: 10 (first rung), 9 (second rung), 8.
        assert_eq!(out.scores[0], 8.0);
        assert!(out.improved);
    }

    /// A one-ulp change of a boundary row moves the window mean by at most one
    /// ulp of that row divided by the window width — the closed form of the
    /// "f32 boundary risk". Ties cannot change the mean (the previous test),
    /// but a boundary row selected differently by rounding moves it by this much.
    #[test]
    fn a_one_ulp_boundary_shift_moves_the_mean_by_at_most_one_ulp_over_the_window() {
        // n = 16 -> ranks 4..12, so the lower boundary is 1.0 and the upper one
        // 5.0. Rows outside the window are far away, so only the window matters.
        let mut resid: Vec<f64> = vec![0.25; 4];
        resid.extend([1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]);
        resid.extend([9.0; 4]);
        let mut keys = keys_of(&resid);
        let base = interquartile_of_keys(&mut keys).unwrap();
        let window = 8usize;
        for (rank, value) in [(4usize, 1.0f64), (11, 5.0)] {
            for step in [-1i64, 1] {
                let mut bumped = resid.clone();
                bumped[rank] = f64::from_bits((value.to_bits() as i64 + step) as u64);
                let mut keys = keys_of(&bumped);
                let moved = interquartile_of_keys(&mut keys).unwrap();
                let ulp = f64::from_bits(value.to_bits() + 1) - value;
                assert!(
                    (moved - base).abs() <= ulp / window as f64 * (1.0 + 1e-12),
                    "rank {rank} step {step}: {base} -> {moved} exceeds one ulp / {window}"
                );
            }
        }
        // And an f32-rounded boundary moves it far more than the arm's 1e-12
        // parity tolerance: that is why the arm predicts in f64 from the exact
        // resident rows and why the f32-twin fallback is not a parity path.
        let f32_ulp =
            f32::from_bits(2.0f32.to_bits() + 1) - 2.0f32; // ≈ 2.4e-7
        assert!(
            f32_ulp as f64 / window as f64 > 1e-9,
            "an f32 ulp at 2.0 is {f32_ulp}, which over {window} rows is well above 1e-12"
        );
    }

    /// A scorer that declines is reported as such, never as a wrong answer.
    #[test]
    fn a_scorer_that_declines_reports_failure_instead_of_a_wrong_answer() {
        struct Decline;
        impl OrderStatTrialScorer for Decline {
            fn score_trials(
                &mut self,
                _: &[&GpuFitPlan],
                _: &[Vec<Vec<f64>>],
                _: OrderStat,
            ) -> Option<Vec<Vec<Option<f64>>>> {
                None
            }
        }
        let t = tree();
        let plan = plan_of(&t);
        assert!(polish_interquartile_batch(
            &mut Decline,
            &[&plan],
            &[plan.c0.clone()],
            4
        )
        .is_none());
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
            polish_interquartile_batch(&mut scorer, &plans, &starts, 8)
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

    /// A non-finite start is inactive: untouched constants, no fabricated
    /// improvement.
    #[test]
    fn a_non_finite_start_is_left_inactive() {
        struct NonFinite;
        impl OrderStatTrialScorer for NonFinite {
            fn score_trials(
                &mut self,
                plans: &[&GpuFitPlan],
                trials: &[Vec<Vec<f64>>],
                _: OrderStat,
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
        let t = constant(1.0);
        let plan = plan_of(&t);
        let out = polish_interquartile_batch(&mut NonFinite, &[&plan], &[plan.c0.clone()], 4)
            .unwrap();
        assert_eq!(out.constants[0], plan.c0);
        assert!(!out.scores[0].is_finite());
        assert!(!out.improved);
    }
}
