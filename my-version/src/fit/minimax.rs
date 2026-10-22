//! #227: maximum absolute error (minimax / Chebyshev) — the batched exact polish.
//!
//! `fit_weighted_metric` fits `maximum` as a squared pre-fit followed by
//! [`coordinate_polish`](super::coordinate_polish): a deterministic pattern
//! search that tries six **fixed geometric steps** (`±10 %, ±3 %, ±1 %` of
//! `max(|c|, 1)`) per constant. That ladder cannot resolve a minimum between two
//! of its rungs, and the CPU pays one full prediction sweep per probe
//! (`6·p·sweeps`), which is the most expensive fit in the engine.
//!
//! This module replaces that tail with an **exact one-dimensional minimisation
//! per coordinate**, batched over a whole generation:
//!
//! 1. With the other constants fixed, a tree that is affine in the coordinate
//!    has `r_i(δ) = a_i + b_i·δ` with `a_i = r_i` and `b_i = ∂r_i/∂c` (the exact
//!    adjoint the batched kernel already forms). The level `t` is attainable
//!    along the coordinate exactly when the per-row intervals
//!    `[min(e1,e2), max(e1,e2)]`, `e1 = (−t−a_i)/b_i`, `e2 = (t−a_i)/b_i`,
//!    intersect. A row with `b_i == 0` constrains no `δ` and only vetoes the
//!    level when `|a_i| > t`.
//! 2. **Bisection on the error level** `t ∈ [0, J(c)]` finds the smallest
//!    attainable level, hence the exact coordinate minimiser; the committed
//!    step is the midpoint of the intersecting interval at the last feasible
//!    level. There is no step ladder and no unbounded optimiser: the bracket is
//!    closed, ordered and identical on every run.
//! 3. The proposal is then scored with the **exact maximum** `max_i |r_i|` on
//!    the same rows and committed only when it is *strictly* lower. The arm is
//!    therefore monotone by construction: it can never report a constant it
//!    made worse, whatever the device returned.
//!
//! For a tree that is *not* affine in the coordinate (a fitted exponent, a
//! threshold, `log(c·x)`) the residual is not affine in `δ`, so the interval
//! test is a **linearised feasibility test, not an exact one** — documented as a
//! heuristic in `docs/gpu-backend.md`. It still proposes a step, and the exact
//! accept test keeps the monotonicity claim true; the exactness claim is made
//! only where it holds.
//!
//! A scorer may decline (`None`) at any step; the caller then refits that batch
//! with the exact CPU arm and labels the cause. A buggy backend can waste work
//! but can never change the reported objective.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::*;

/// One level of the error, in units of `max |r|`.
///
/// `lo` is the largest per-row lower bound on the coordinate step and `hi` the
/// smallest upper bound; the level is attainable exactly when `lo <= hi`.
/// `+inf`/`-inf` (the empty intersection) is the infeasible encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LevelInterval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl LevelInterval {
    /// Whether some step attains this level.
    pub(crate) fn is_feasible(&self) -> bool {
        self.lo <= self.hi
    }

    /// The midpoint step, when it is a finite number. An unbounded or empty
    /// interval has no usable midpoint and is dropped by the driver.
    pub(crate) fn midpoint(&self) -> Option<f64> {
        let m = 0.5 * (self.lo + self.hi);
        if m.is_finite() {
            Some(m)
        } else {
            None
        }
    }
}

/// The exact maximum of `|pred − y|` over the rows and the row that attains it.
///
/// The device returns the **lowest** row index on a tie (`min` over the
/// maximisers), which is what makes the active set reproducible: a non-unique
/// maximiser is reported as its first row, never as whichever row the last
/// thread happened to write.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MaxStat {
    pub(crate) value: f64,
    pub(crate) row: usize,
}

impl MaxStat {
    /// The exact reduction: the larger `|r|` wins, and a tie goes to the lower
    /// row index. Associative, commutative and idempotent, so the result cannot
    /// depend on the reduction order.
    ///
    /// The device performs the same reduction in integers (see `k_max_scan`);
    /// this is the reference implementation the tests and the CPU scorer use.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn better(self, other: MaxStat) -> MaxStat {
        if other.value > self.value || (other.value == self.value && other.row < self.row) {
            other
        } else {
            self
        }
    }
}

/// One scorer call: the exact maximum, and the exact level intervals.
pub(crate) trait MinimaxTrialScorer {
    /// `(candidate, trial)` → exact `max |r|` and its lowest maximiser row.
    /// `None` marks a segment whose prediction was not finite (or an invalid
    /// constants vector); `None` for the whole vector means the device declined.
    fn score_trials(
        &mut self,
        plans: &[&GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<MaxStat>>>>;

    /// Per candidate, one level-interval test per coordinate: `constants[k]` is
    /// the candidate's current constants (slot order), `coords[k][j]` a constant
    /// slot and `levels[k][j]` the level to test.
    fn intervals(
        &mut self,
        plans: &[&GpuFitPlan],
        constants: &[Vec<f64>],
        coords: &[Vec<i32>],
        levels: &[Vec<f64>],
    ) -> Option<Vec<Vec<Option<LevelInterval>>>>;
}

/// Rounds of the bisection on the level, per coordinate and sweep.
///
/// `2^32 ≈ 4.3e9`, so a relative bracket of `1e-9` is reached long before the
/// cap; the constant is what bounds the work per sweep deterministically.
pub(crate) const MINIMAX_BISECT_ROUNDS: usize = 32;

/// Relative width at which a level bracket counts as converged: the exact
/// 1-D minimiser is then known to `1e-9` relative in the objective. The
/// *accepted* step is still judged by an exact objective comparison, so this
/// tolerance costs quality only at the level of the last bit of the search.
pub(crate) const MINIMAX_LEVEL_REL_TOL: f64 = 1e-9;

/// Largest step the bracket may propose. A bracket whose intersection is
/// unbounded has no finite midpoint and is dropped; this cap keeps a
/// near-zero derivative from proposing an absurd constant that the exact test
/// would reject anyway.
pub(crate) const MINIMAX_MAX_STEP: f64 = 1e6;

/// The constants a batched minimax polish settled on.
#[derive(Clone, Debug)]
pub(crate) struct MinimaxPolish {
    /// Final constants per candidate, slot order.
    pub(crate) constants: Vec<Vec<f64>>,
    /// Exact `max |r|` of those constants, per candidate.
    pub(crate) scores: Vec<f64>,
    /// Lowest row index attaining that maximum, per candidate (`None` when the
    /// candidate's start was not finite). This is the active set a diagnostic
    /// reports; the fit path itself only consumes `scores`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) maximisers: Vec<Option<usize>>,
    /// True when at least one coordinate step was committed.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) improved: bool,
}

/// The proposed step for one coordinate, if the bisection produced one.
fn coordinate_step(ivs: &[Option<LevelInterval>], ci: usize, current: f64) -> Option<f64> {
    let iv = (*ivs.get(ci)?)?;
    if !iv.is_feasible() {
        return None;
    }
    let delta = iv.midpoint()?.clamp(-MINIMAX_MAX_STEP, MINIMAX_MAX_STEP);
    if delta == 0.0 || !(current + delta).is_finite() {
        return None;
    }
    Some(delta)
}

/// Run the exact minimax polish for a whole batch, `sweeps` sweeps deep.
///
/// One sweep proposes, **in parallel from the same point**, one exact 1-D
/// minimiser per constant, scores all of them with the exact maximum in one
/// batched call and commits the best **strict** improver (ties go to the lowest
/// constant index). A sweep that commits nothing ends the polish, and at most
/// `min(sweeps, 12)` sweeps run — the same budget the CPU polish uses.
///
/// `starts` must have one constant vector per plan, in `plan.c0` slot order
/// (the caller passes the squared-LM warm start). Returns `None` when the
/// scorer declined or returned an inconsistent shape; callers then use the
/// exact CPU arm and label the fallback.
pub(crate) fn polish_minimax_batch<S: MinimaxTrialScorer + ?Sized>(
    scorer: &mut S,
    plans: &[&GpuFitPlan],
    starts: &[Vec<f64>],
    sweeps: usize,
) -> Option<MinimaxPolish> {
    let n = plans.len();
    if n != starts.len() {
        return None;
    }
    let maxp = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
    let mut c: Vec<Vec<f64>> = starts.to_vec();
    if n == 0 || maxp == 0 {
        return Some(MinimaxPolish {
            constants: c,
            scores: vec![f64::INFINITY; n],
            maximisers: vec![None; n],
            improved: false,
        });
    }
    if plans.iter().zip(&c).any(|(p, v)| p.nconst != v.len()) {
        return None;
    }

    // Baseline: the warm start's exact maximum, in one batched call. A start
    // the scorer cannot score is inactive, exactly like the CPU polish
    // returning early on a non-finite score.
    let baseline: Vec<Vec<Vec<f64>>> = c.iter().map(|v| vec![v.clone()]).collect();
    let base = scorer.score_trials(plans, &baseline)?;
    if base.len() != n || base.iter().any(|t| t.len() != 1) {
        return None;
    }
    let mut best = vec![f64::INFINITY; n];
    let mut maximisers: Vec<Option<usize>> = vec![None; n];
    let mut active = vec![true; n];
    for k in 0..n {
        match base[k][0] {
            Some(s) if s.value.is_finite() => {
                best[k] = s.value;
                maximisers[k] = Some(s.row);
            }
            _ => active[k] = false,
        }
    }
    if !active.iter().any(|&a| a) {
        return Some(MinimaxPolish {
            constants: c,
            scores: best,
            maximisers,
            improved: false,
        });
    }

    let coords: Vec<Vec<i32>> = plans
        .iter()
        .map(|p| p.order.iter().map(|&slot| slot as i32).collect())
        .collect();
    let mut improved_any = false;
    for _ in 0..sweeps.min(12) {
        // One closed bracket per (candidate, constant). `t = 0` is the lower
        // end (a level nothing can beat) and the candidate's current exact
        // objective the upper end, which is attained at `δ = 0` by definition.
        let mut lo = vec![vec![0.0f64; maxp]; n];
        let mut hi: Vec<Vec<f64>> = (0..n).map(|k| vec![best[k]; maxp]).collect();
        let mut ivs: Vec<Vec<Option<LevelInterval>>> = vec![vec![None; maxp]; n];
        let mut open: Vec<Vec<bool>> = (0..n)
            .map(|k| {
                let p = plans[k].nconst;
                (0..maxp).map(|ci| ci < p && active[k] && best[k] > 0.0).collect()
            })
            .collect();

        for _round in 0..MINIMAX_BISECT_ROUNDS {
            if !open.iter().any(|r| r.iter().any(|&o| o)) {
                break;
            }
            let levels: Vec<Vec<f64>> = (0..n)
                .map(|k| {
                    (0..maxp)
                        .map(|ci| {
                            if open[k][ci] {
                                0.5 * (lo[k][ci] + hi[k][ci])
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                })
                .collect();
            let res = scorer.intervals(plans, &c, &coords, &levels)?;
            if res.len() != n || res.iter().zip(&coords).any(|(r, c)| r.len() != c.len()) {
                return None;
            }
            for k in 0..n {
                for ci in 0..plans[k].nconst {
                    if !open[k][ci] {
                        continue;
                    }
                    let t = levels[k][ci];
                    match res[k][ci] {
                        Some(iv) if iv.is_feasible() => {
                            ivs[k][ci] = Some(iv);
                            hi[k][ci] = t;
                        }
                        Some(_) => lo[k][ci] = t,
                        // The segment had a non-finite row: this coordinate is
                        // dropped for this sweep, never guessed.
                        None => open[k][ci] = false,
                    }
                    if open[k][ci] {
                        let (l, h) = (lo[k][ci], hi[k][ci]);
                        if !(h > l) || (h - l) <= MINIMAX_LEVEL_REL_TOL * h.abs() {
                            open[k][ci] = false;
                        }
                    }
                }
            }
        }

        // One trial per constant, padded with the candidate's own constants so
        // every candidate has the same trial count (the batch shape the device
        // requires). A padded trial cannot strictly improve: it scores exactly
        // the current objective.
        let mut trials: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);
        let mut moved: Vec<Vec<bool>> = Vec::with_capacity(n);
        for k in 0..n {
            let mut list: Vec<Vec<f64>> = Vec::with_capacity(maxp);
            let mut flags: Vec<bool> = Vec::with_capacity(maxp);
            for ci in 0..maxp {
                let step = if ci < plans[k].nconst && active[k] {
                    coordinate_step(&ivs[k], ci, c[k][plans[k].order[ci]])
                } else {
                    None
                };
                match step {
                    Some(delta) => {
                        let mut t = c[k].clone();
                        t[plans[k].order[ci]] += delta;
                        list.push(t);
                        flags.push(true);
                    }
                    None => {
                        list.push(c[k].clone());
                        flags.push(false);
                    }
                }
            }
            trials.push(list);
            moved.push(flags);
        }
        let scored = scorer.score_trials(plans, &trials)?;
        if scored.len() != n || scored.iter().any(|t| t.len() != maxp) {
            return None;
        }
        let mut improved = false;
        for k in 0..n {
            if !active[k] {
                continue;
            }
            let mut choice: Option<(usize, MaxStat)> = None;
            for (t, stat) in scored[k].iter().enumerate() {
                let Some(stat) = *stat else { continue };
                if !moved[k][t] || !stat.value.is_finite() || !(stat.value < best[k]) {
                    continue;
                }
                match choice {
                    // Ties go to the lowest trial index, i.e. the lowest
                    // constant index in `plan.order`.
                    Some((_, m)) if stat.value >= m.value => {}
                    _ => choice = Some((t, stat)),
                }
            }
            if let Some((t, stat)) = choice {
                c[k] = trials[k][t].clone();
                best[k] = stat.value;
                maximisers[k] = Some(stat.row);
                improved = true;
            }
        }
        if !improved {
            break;
        }
        improved_any = true;
    }
    Some(MinimaxPolish {
        constants: c,
        scores: best,
        maximisers,
        improved: improved_any,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    /// The reference scorer: the same forward and adjoint sweeps the CPU fit
    /// uses (`eval_row`/`backward`), the same order-free reductions. It shares
    /// no code with the device kernel or with the driver's arithmetic.
    struct CpuScorer<'a> {
        x: &'a [Vec<f64>],
        y: &'a [f64],
    }

    impl CpuScorer<'_> {
        /// `(exact max |r|, per-row a_i and b_j)` for one segment.
        fn scan(
            &self,
            plan: &GpuFitPlan,
            c: &[f64],
            coord: i32,
        ) -> Option<(MaxStat, Vec<(f64, f64)>)> {
            let f = &plan.flat;
            let m = f.op.len();
            let mut v = vec![0.0f64; m];
            let mut adj = vec![0.0f64; m];
            let mut g = vec![0.0f64; plan.nconst];
            let mut stat = MaxStat {
                value: 0.0,
                row: 0,
            };
            let mut rows = Vec::with_capacity(self.x.len());
            for (row, xs) in self.x.iter().enumerate() {
                super::super::eval_row(f, xs, c, &mut v);
                let pred = v[f.root];
                let r = pred - self.y[row];
                if !pred.is_finite() || pred.abs() > 1e30 || !r.is_finite() {
                    return None;
                }
                stat = stat.better(MaxStat {
                    value: r.abs(),
                    row,
                });
                let b = if coord >= 0 {
                    super::super::backward(f, &v, &mut adj, &mut g);
                    g[coord as usize]
                } else {
                    0.0
                };
                rows.push((r, b));
            }
            Some((stat, rows))
        }
    }

    impl MinimaxTrialScorer for CpuScorer<'_> {
        fn score_trials(
            &mut self,
            plans: &[&GpuFitPlan],
            trials: &[Vec<Vec<f64>>],
        ) -> Option<Vec<Vec<Option<MaxStat>>>> {
            let mut out = Vec::with_capacity(plans.len());
            for (plan, list) in plans.iter().zip(trials) {
                let mut per_trial = Vec::with_capacity(list.len());
                for c in list {
                    if c.len() != plan.nconst {
                        per_trial.push(None);
                        continue;
                    }
                    per_trial.push(self.scan(plan, c, -1).map(|(s, _)| s));
                }
                out.push(per_trial);
            }
            Some(out)
        }

        fn intervals(
            &mut self,
            plans: &[&GpuFitPlan],
            constants: &[Vec<f64>],
            coords: &[Vec<i32>],
            levels: &[Vec<f64>],
        ) -> Option<Vec<Vec<Option<LevelInterval>>>> {
            let mut out = Vec::with_capacity(plans.len());
            for (k, plan) in plans.iter().enumerate() {
                let mut per = Vec::with_capacity(coords[k].len());
                for j in 0..coords[k].len() {
                    let coord = coords[k][j];
                    let level = levels[k][j];
                    if coord < 0 || !level.is_finite() || level < 0.0 {
                        per.push(None);
                        continue;
                    }
                    let Some((_, rows)) = self.scan(plan, &constants[k], coord) else {
                        per.push(None);
                        continue;
                    };
                    let mut lo = f64::NEG_INFINITY;
                    let mut hi = f64::INFINITY;
                    for (a, b) in rows {
                        if b == 0.0 {
                            if a.abs() > level {
                                lo = f64::INFINITY;
                                hi = f64::NEG_INFINITY;
                            }
                            continue;
                        }
                        let e1 = (-level - a) / b;
                        let e2 = (level - a) / b;
                        lo = lo.max(e1.min(e2));
                        hi = hi.min(e1.max(e2));
                    }
                    per.push(Some(LevelInterval { lo, hi }));
                }
                out.push(per);
            }
            Some(out)
        }
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

    fn scorer<'a>(x: &'a [Vec<f64>], y: &'a [f64]) -> CpuScorer<'a> {
        CpuScorer { x, y }
    }

    fn max_objective(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(tree, r)).collect();
        crate::metrics::score_weighted(&pred, y, &[], Metric::Maximum, 0)
    }

    /// A unique maximiser, a tied maximiser and an all-zero residual set reduce
    /// as documented: the largest `|r|`, the lowest row on a tie.
    #[test]
    fn the_reduction_breaks_ties_to_the_lowest_row() {
        let rows = [
            MaxStat {
                value: 1.0,
                row: 3,
            },
            MaxStat {
                value: 2.0,
                row: 5,
            },
            MaxStat {
                value: 0.5,
                row: 0,
            },
        ];
        let best = rows.iter().copied().fold(
            MaxStat {
                value: -1.0,
                row: usize::MAX,
            },
            MaxStat::better,
        );
        assert_eq!((best.value, best.row), (2.0, 5), "unique maximiser");

        // Non-unique: the same value at rows 7 and 2 must report row 2,
        // whatever the order the rows are reduced in.
        let tie = [
            MaxStat {
                value: 2.0,
                row: 7,
            },
            MaxStat {
                value: 2.0,
                row: 2,
            },
        ];
        assert_eq!(tie.iter().copied().fold(tie[0], MaxStat::better).row, 2);
        assert_eq!(tie.iter().copied().rev().fold(tie[0], MaxStat::better).row, 2);

        // All rows zero: the maximum is 0 and row 0 is its lowest maximiser.
        let zeros: Vec<MaxStat> = (0..5).map(|row| MaxStat { value: 0.0, row }).collect();
        assert_eq!(zeros.iter().copied().fold(zeros[0], MaxStat::better).row, 0);
    }

    /// The hand-solved 1-D Chebyshev fixture: one constant, targets 1, 2, 4.
    /// The minimiser of `max |c − y_i|` is the midrange `(1+4)/2 = 2.5` with
    /// objective `1.5`. The fixed CPU ladder cannot reach it from the
    /// least-squares start (`7/3`), so this is a strict dominance test too.
    #[test]
    fn the_exact_coordinate_minimiser_finds_the_hand_solved_midrange() {
        let x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64]).collect();
        let y = [1.0, 2.0, 4.0];
        let tree = constant(7.0 / 3.0);
        let plan = plan_of(&tree);
        let start = constants_of(&tree);
        assert_eq!(start, vec![7.0 / 3.0]);
        let mut s = scorer(&x, &y);
        let out = polish_minimax_batch(&mut s, &[&plan], &[start.clone()], 8).expect("cpu scorer");
        assert!(out.improved);
        let c = out.constants[0][0];
        assert!(
            (c - 2.5).abs() <= 1e-8 * 2.5,
            "the exact midrange is 2.5, got {c}"
        );
        assert!(
            (out.scores[0] - 1.5).abs() <= 1e-8 * 1.5,
            "the exact minimax is 1.5, got {}",
            out.scores[0]
        );
        // Dominance over the start, and over the CPU ladder from the same start.
        let mut cpu = tree.clone();
        super::super::coordinate_polish(&mut cpu, &x, &y, &[], Metric::Maximum, 8);
        let cpu_score = max_objective(&cpu, &x, &y);
        let start_score = max_objective(&tree, &x, &y);
        assert!(out.scores[0] < start_score, "{} !< {start_score}", out.scores[0]);
        assert!(
            out.scores[0] < cpu_score,
            "the exact minimiser ({}) must beat the CPU ladder ({cpu_score})",
            out.scores[0]
        );
    }

    /// A two-parameter affine fixture with a hand-solved optimum: `y = 0, 1, 0`
    /// at `x = -1, 0, 1`. The Chebyshev line is `a = 0, b = 0.5` with objective
    /// `0.5` (any `a != 0` raises the level to `0.5 + |a|/2`), while the
    /// least-squares start is `b = 1/3` with objective `2/3`.
    #[test]
    fn two_constants_reach_the_hand_solved_chebyshev_line() {
        let x: Vec<Vec<f64>> = [-1.0, 0.0, 1.0].iter().map(|v| vec![*v]).collect();
        let y = [0.0, 1.0, 0.0];
        let tree = node(Op::Add, vec![node(Op::Mul, vec![constant(0.0), input(0)]), constant(1.0 / 3.0)]);
        let plan = plan_of(&tree);
        let mut s = scorer(&x, &y);
        let out = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).expect("cpu scorer");
        assert!((out.scores[0] - 0.5).abs() <= 1e-8, "{}", out.scores[0]);
        let a = out.constants[0][0];
        let b = out.constants[0][1];
        assert!(a.abs() <= 1e-8, "the exact slope is 0, got {a}");
        assert!((b - 0.5).abs() <= 1e-8, "the exact intercept is 0.5, got {b}");
        assert!(out.scores[0] < max_objective(&tree, &x, &y));
    }

    /// A constant prediction (the same value on every row) is the 1-D Chebyshev
    /// centre of the target: the **midrange** `(min y + max y) / 2`, with level
    /// `(max y − min y) / 2` attained by two rows. The exact arm must find it
    /// from the least-squares start (the mean), report the *lowest* maximiser,
    /// and then stop: a second polish from the answer commits nothing.
    #[test]
    fn a_constant_prediction_polishes_to_its_midrange_and_stops_there() {
        let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
        let y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        // The least-squares start of a bare constant is the mean, 3.875.
        let tree = constant(3.875);
        let plan = plan_of(&tree);
        let mut s = scorer(&x, &y);
        let out = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).expect("cpu scorer");
        assert!(out.improved);
        // midrange = (1 + 9) / 2 = 5, level = 4.
        assert!((out.constants[0][0] - 5.0).abs() <= 1e-8, "{:?}", out.constants[0]);
        assert!((out.scores[0] - 4.0).abs() <= 1e-8, "{}", out.scores[0]);
        // |5 − y| = 2, 4, 1, 4, 0, 4, 3, 1: the maximum 4 is attained by rows
        // 1, 3 and 5, so the reported maximiser is the lowest of them.
        assert_eq!(out.maximisers[0], Some(1));
        // From its own answer nothing can improve: the midrange is the optimum.
        let mut s = scorer(&x, &y);
        let again = polish_minimax_batch(&mut s, &[&plan], &out.constants.clone(), 8)
            .expect("cpu scorer");
        assert!(!again.improved);
        assert_eq!(again.scores[0], out.scores[0]);
        assert_eq!(again.constants[0], out.constants[0]);
    }

    /// A tied maximiser (two rows at the same `|r|`) polishes to the same
    /// constants as the unique case and reports the *lowest* row.
    #[test]
    fn a_non_unique_maximiser_is_deterministic_and_reports_the_lowest_row() {
        // Symmetric targets: rows 0 and 4 are both 4 away from the start.
        let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let y = [0.0, 1.0, 2.0, 1.0, 0.0];
        let tree = constant(4.0);
        let plan = plan_of(&tree);
        let run = || {
            let mut s = scorer(&x, &y);
            polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).expect("cpu scorer")
        };
        let a = run();
        let b = run();
        assert_eq!(a.constants[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   b.constants[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        // The Chebyshev centre of 0,1,2,1,0 is the midrange 1 with level 1,
        // attained by rows 0 and 4 at once.
        assert!((a.scores[0] - 1.0).abs() <= 1e-8, "{}", a.scores[0]);
        assert!((a.constants[0][0] - 1.0).abs() <= 1e-8, "{:?}", a.constants[0]);
        assert_eq!(a.maximisers[0], Some(0), "the lowest maximiser is reported");
    }

    /// Batched and single-candidate runs agree bit for bit, and repeated calls
    /// are identical: the arm has no scheduling dependence.
    #[test]
    fn a_batch_polish_is_deterministic_across_candidates() {
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![-2.0 + 0.1 * i as f64]).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| 1.5 * (0.8 * r[0]).sin() + if i % 9 == 0 { 3.0 } else { 0.0 })
            .collect();
        let tree = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(0.7), node(Op::Sin, vec![node(Op::Mul, vec![constant(0.6), input(0)])])]),
                constant(0.1),
            ],
        );
        let plan = plan_of(&tree);
        let bits = |out: &MinimaxPolish| {
            out.constants
                .iter()
                .map(|c| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let mut s = scorer(&x, &y);
        let one = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).unwrap();
        let mut s = scorer(&x, &y);
        let three = polish_minimax_batch(
            &mut s,
            &[&plan, &plan, &plan],
            &[plan.c0.clone(), plan.c0.clone(), plan.c0.clone()],
            8,
        )
        .unwrap();
        assert_eq!(bits(&one), vec![bits(&three)[0].clone()]);
        assert_eq!(bits(&three)[0], bits(&three)[1]);
        assert_eq!(bits(&three)[0], bits(&three)[2]);
        let mut s = scorer(&x, &y);
        let again = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).unwrap();
        assert_eq!(bits(&one), bits(&again), "repeated calls must agree bit for bit");
        assert_eq!(one.scores[0].to_bits(), again.scores[0].to_bits());
    }

    /// The polish never makes a candidate worse than its start, and never worse
    /// than the CPU ladder from the same start, over a mixed batch.
    #[test]
    fn the_polish_dominates_the_start_and_the_cpu_ladder() {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![-3.0 + 0.1 * i as f64]).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| 2.0 * (1.3 * r[0]).sin() + if i % 11 == 4 { 8.0 } else { 0.0 })
            .collect();
        let trees = vec![
            node(Op::Add, vec![node(Op::Mul, vec![constant(1.0), node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])])]), constant(0.0)]),
            node(Op::Add, vec![node(Op::Mul, vec![constant(0.5), input(0)]), constant(0.2)]),
            constant(0.0),
        ];
        for tree in &trees {
            let plan = plan_of(tree);
            let mut s = scorer(&x, &y);
            let out = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 8).expect("cpu scorer");
            let start = max_objective(tree, &x, &y);
            assert!(out.scores[0] <= start, "{} > start {start}", out.scores[0]);
            let mut cpu = tree.clone();
            super::super::coordinate_polish(&mut cpu, &x, &y, &[], Metric::Maximum, 8);
            let cpu_score = max_objective(&cpu, &x, &y);
            assert!(
                out.scores[0] <= cpu_score,
                "the GPU arm ({}) must not lose to the CPU ladder ({cpu_score})",
                out.scores[0]
            );
        }
    }

    /// A scorer that declines is reported as such, never as a wrong answer.
    #[test]
    fn a_scorer_that_declines_reports_failure_instead_of_a_wrong_answer() {
        struct Decline;
        impl MinimaxTrialScorer for Decline {
            fn score_trials(
                &mut self,
                _: &[&GpuFitPlan],
                _: &[Vec<Vec<f64>>],
            ) -> Option<Vec<Vec<Option<MaxStat>>>> {
                None
            }
            fn intervals(
                &mut self,
                _: &[&GpuFitPlan],
                _: &[Vec<f64>],
                _: &[Vec<i32>],
                _: &[Vec<f64>],
            ) -> Option<Vec<Vec<Option<LevelInterval>>>> {
                None
            }
        }
        let tree = constant(1.0);
        let plan = plan_of(&tree);
        assert!(polish_minimax_batch(&mut Decline, &[&plan], &[plan.c0.clone()], 4).is_none());
    }

    /// A start whose prediction is not finite is left inactive: no constant is
    /// fabricated and the score stays non-finite.
    #[test]
    fn a_non_finite_start_is_left_inactive() {
        struct NonFinite;
        impl MinimaxTrialScorer for NonFinite {
            fn score_trials(
                &mut self,
                plans: &[&GpuFitPlan],
                trials: &[Vec<Vec<f64>>],
            ) -> Option<Vec<Vec<Option<MaxStat>>>> {
                Some(
                    plans
                        .iter()
                        .zip(trials)
                        .map(|(_, list)| list.iter().map(|_| None).collect())
                        .collect(),
                )
            }
            fn intervals(
                &mut self,
                plans: &[&GpuFitPlan],
                _: &[Vec<f64>],
                coords: &[Vec<i32>],
                _: &[Vec<f64>],
            ) -> Option<Vec<Vec<Option<LevelInterval>>>> {
                Some(plans.iter().zip(coords).map(|(_, c)| vec![None; c.len()]).collect())
            }
        }
        let tree = constant(1.0);
        let plan = plan_of(&tree);
        let out = polish_minimax_batch(&mut NonFinite, &[&plan], &[plan.c0.clone()], 4).unwrap();
        assert_eq!(out.constants[0], plan.c0);
        assert!(!out.scores[0].is_finite());
        assert!(!out.improved);
        assert_eq!(out.maximisers[0], None);
    }

    /// Zero sweeps is the baseline only: the start's exact maximum and its
    /// maximiser, with no constant moved.
    #[test]
    fn zero_sweeps_scores_the_start_and_moves_nothing() {
        let x: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        let y = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        let tree = constant(1.0);
        let plan = plan_of(&tree);
        let mut s = scorer(&x, &y);
        let out = polish_minimax_batch(&mut s, &[&plan], &[plan.c0.clone()], 0).unwrap();
        assert_eq!(out.constants[0], plan.c0);
        assert_eq!(out.scores[0], 4.0);
        assert_eq!(out.maximisers[0], Some(2));
        assert!(!out.improved);
    }
}
