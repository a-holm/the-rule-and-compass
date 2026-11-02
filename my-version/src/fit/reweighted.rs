//! #611: the opt-in reweighted L1 schedule for `absolute` and
//! `implicit_residual`, the CPU twin of the device L1 kernel.
//!
//! `--fit-irls nested` is today's IRLS (`irls_rounds`). `reweighted` runs one
//! Levenberg-Marquardt over the true L1 loss `Σ base·|r|`: the weights
//! `base / max(|r|, 1e-6)` are rebuilt at every assembly, a trial is judged on
//! the true loss with up to two halvings toward the incumbent, and the damping
//! is carried across rounds. `reweighted-final` then runs today's final
//! weighted solve at the weights of the end point.
//!
//! The schedule is read from a thread-local that only [`with_schedule`] sets,
//! so the exact paths (finalist refit, direct `fit::*` calls) never see it.

use super::{
    improves, metric_score_cols, solve, write_constants, FitCols, IrlsSolver, EXACT_FITS,
};
use super::tile::TileFit;
use crate::expr::{self, Node};
use crate::metrics::{self, Metric};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The schedule of the `absolute` and `implicit_residual` constant fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FitIrls {
    /// Today's IRLS: weighted least-squares rounds, then a final solve.
    #[default]
    Nested,
    /// One LM over the true L1 loss, weights rebuilt at every assembly.
    Reweighted,
    /// `Reweighted`, then today's final weighted solve at the end point.
    ReweightedFinal,
}

impl FitIrls {
    pub fn name(self) -> &'static str {
        match self {
            FitIrls::Nested => "nested",
            FitIrls::Reweighted => "reweighted",
            FitIrls::ReweightedFinal => "reweighted-final",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "nested" => Some(FitIrls::Nested),
            "reweighted" => Some(FitIrls::Reweighted),
            "reweighted-final" => Some(FitIrls::ReweightedFinal),
            _ => None,
        }
    }
}

thread_local! {
    static SCHEDULE: Cell<FitIrls> = const { Cell::new(FitIrls::Nested) };
}

#[cfg(test)]
thread_local! {
    static LOCAL_FITS: Cell<u64> = const { Cell::new(0) };
}

static REWEIGHTED_FITS: AtomicU64 = AtomicU64::new(0);

/// Restores the previous schedule on exit and on unwind.
struct Restore(FitIrls);

impl Drop for Restore {
    fn drop(&mut self) {
        SCHEDULE.with(|s| s.set(self.0));
    }
}

/// Run `f` with `schedule` as the L1 schedule of this thread.
pub fn with_schedule<R>(schedule: FitIrls, f: impl FnOnce() -> R) -> R {
    let _restore = Restore(SCHEDULE.with(|s| s.replace(schedule)));
    f()
}

pub(super) fn current() -> FitIrls {
    SCHEDULE.with(Cell::get)
}

/// Fits that ran on a reweighted schedule since process start.
pub(super) fn fits() -> u64 {
    REWEIGHTED_FITS.load(Ordering::Relaxed)
}

#[cfg(test)]
pub(super) fn thread_fits() -> u64 {
    LOCAL_FITS.with(Cell::get)
}

fn clamp_add(c: &[f64], step: &[f64]) -> Vec<f64> {
    c.iter().zip(step).map(|(a, d)| (a + d).clamp(-1e6, 1e6)).collect()
}

fn halve_toward(trial: &mut [f64], c: &[f64]) {
    for (t, &a) in trial.iter_mut().zip(c) {
        *t = a + 0.5 * (*t - a);
    }
}

/// The L1 fit of `tree` on `schedule` (`Reweighted` or `ReweightedFinal`).
/// Returns the configured metric at the kept constants; the tree changes only
/// after an accepted step, so the result is never worse than the start on
/// `Σ base·|r|`.
pub(super) fn l1_fit(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
    schedule: FitIrls,
) -> f64 {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    EXACT_FITS.fetch_add(1, Ordering::Relaxed);
    REWEIGHTED_FITS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    LOCAL_FITS.with(|n| n.set(n.get() + 1));
    let cols = FitCols::gather(tree, x);
    let mut solver = IrlsSolver::new(tree, x);
    let k = solver.flat.nconst;
    if x.is_empty() || k == 0 || iters == 0 {
        return metric_score_cols(tree, x, &cols, y, w, metric);
    }
    let mut c = solver.c.clone();
    let moved = {
        let mut pass = TileFit::new(&mut solver.core, y, w);
        reweighted_lm(&mut pass, &mut c, k, iters)
    };
    if moved {
        write_constants(tree, &solver.flat, &c);
    }
    let mut best = metric_score_cols(tree, x, &cols, y, w, metric);
    if schedule == FitIrls::ReweightedFinal && moved && best.is_finite() && best != 0.0 {
        best = final_solve(tree, x, y, w, metric, &cols, &mut solver, c, iters.max(4), best);
    }
    best
}

/// The device L1 schedule over `pass`: `c` holds the fitted constants on
/// return. True when `c` moved.
fn reweighted_lm(pass: &mut TileFit<'_>, c: &mut Vec<f64>, k: usize, iters: usize) -> bool {
    let rounds = iters.clamp(1, 4);
    let inner = (iters / rounds).max(1);
    let Some((mut ne, mut best)) = pass.normal_l1(c) else {
        return false;
    };
    if !best.is_finite() {
        return false;
    }
    let mut lambda = 1e-3;
    let mut assembled = true;
    let mut moved = false;
    let mut stop = false;
    for _ in 0..rounds {
        if stop || best == 0.0 {
            break;
        }
        for _ in 0..inner {
            if !assembled {
                match pass.normal_l1(c) {
                    Some((n, _)) => ne = n,
                    None => {
                        stop = true;
                        break;
                    }
                }
                assembled = true;
            }
            let mut rhs: Vec<f64> = ne.jtr.iter().map(|v| -v).collect();
            let Some(step) = solve(&ne.jtj, &mut rhs, lambda, k) else {
                stop = true;
                break;
            };
            let mut trial = clamp_add(c, &step);
            let mut loss = pass.loss_l1(&trial);
            for _ in 0..2 {
                if loss < best {
                    break;
                }
                halve_toward(&mut trial, c);
                loss = pass.loss_l1(&trial);
            }
            if loss < best {
                *c = trial;
                best = loss;
                lambda = (lambda / 8.0).max(1e-12);
                assembled = false;
                moved = true;
                if best == 0.0 {
                    stop = true;
                    break;
                }
            } else {
                lambda *= 6.0;
                if lambda > 1e9 {
                    break;
                }
            }
        }
    }
    // Undamped polish: the held assembly is the system at `c` when the last
    // damped trial was rejected.
    let mut held = if assembled { Some(ne) } else { None };
    for _ in 0..4 {
        if stop || best == 0.0 {
            break;
        }
        let n = match held.take() {
            Some(n) => n,
            None => match pass.normal_l1(c) {
                Some((n, _)) => n,
                None => break,
            },
        };
        let mut rhs: Vec<f64> = n.jtr.iter().map(|v| -v).collect();
        let Some(step) = solve(&n.jtj, &mut rhs, 0.0, k) else {
            break;
        };
        let mut trial = clamp_add(c, &step);
        let mut loss = pass.loss_l1(&trial);
        if !(loss < best) {
            halve_toward(&mut trial, c);
            loss = pass.loss_l1(&trial);
        }
        if loss < best {
            *c = trial;
            best = loss;
            moved = true;
        } else {
            break;
        }
    }
    moved
}

/// Today's final weighted solve at the weights of the end point `c`, kept only
/// on a strictly lower metric. `tree` already carries `c`.
#[allow(clippy::too_many_arguments)]
fn final_solve(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
    cols: &FitCols,
    solver: &mut IrlsSolver,
    c: Vec<f64>,
    last: usize,
    best: f64,
) -> f64 {
    let mut pred = Vec::new();
    cols.predictions_into(tree, x, &mut pred);
    let mut weights = vec![0.0; x.len()];
    for (i, &p) in pred.iter().enumerate() {
        let r = p - y[i];
        if !r.is_finite() {
            return best;
        }
        let base = if w.is_empty() { 1.0 } else { w[i] };
        weights[i] = if base > 0.0 { base * (1.0 / r.abs().max(1e-6)) } else { 0.0 };
    }
    solver.c = c;
    let (trial, _) = solver.solve(tree, y, &weights, last);
    cols.predictions_into(&trial, x, &mut pred);
    let score = metrics::score_weighted(&pred, y, w, metric, expr::count_constants(tree));
    if improves(score, best) {
        *tree = trial;
        return score;
    }
    best
}
