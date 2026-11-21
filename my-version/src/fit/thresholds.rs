//! #1024: exact 1-D fit of threshold constants.
//!
//! A constant that is an operand of a comparison, or sits in the condition of
//! `if`, has a zero adjoint almost everywhere, so the LM never moves it. With
//! `--fit-thresholds` this step searches such a constant on the fit rows after
//! the LM iterations. The objective is piecewise constant in the threshold and
//! changes only where it crosses an operand value, so the candidates are the
//! midpoints between consecutive sorted operand values.

use super::{metric_score_cols, FitCols};
use crate::expr::{Node, Op};
use crate::metrics::Metric;
use std::sync::atomic::{AtomicU64, Ordering};

/// Threshold constants fitted per tree; the rest keep their value.
pub(super) const MAX_THRESHOLDS: usize = 2;

/// Candidates per search level. Each level scores at most this many thresholds,
/// then the next level narrows the range around the best one, until every
/// midpoint of the range is scored.
const Q: usize = 64;

/// Rounds of threshold search followed by a refit of the other constants.
const ROUNDS: usize = 2;

static MOVES: AtomicU64 = AtomicU64::new(0);

/// Thresholds moved since process start.
pub fn threshold_moves() -> u64 {
    MOVES.load(Ordering::Relaxed)
}

/// One threshold constant and where its breakpoints come from.
///
/// The compared quantity changes sign where `c = bound_sign * k + scale * x`,
/// with `x` the operand and `k` the constant the other side of the comparison
/// holds (`0` for an `if` condition and for the direct form).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Threshold {
    constant: Vec<usize>,
    operand: Vec<usize>,
    bound: Option<Vec<usize>>,
    bound_sign: f64,
    scale: f64,
}

fn is_comparison(op: Op) -> bool {
    matches!(
        op,
        Op::Equal | Op::Less | Op::LessOrEqual | Op::Greater | Op::GreaterOrEqual
    )
}

fn has_input(n: &Node) -> bool {
    n.op == Op::Input || n.kids.iter().any(has_input)
}

fn at<'a>(n: &'a Node, path: &[usize]) -> &'a Node {
    path.iter().fold(n, |n, &k| &n.kids[k])
}

fn at_mut<'a>(n: &'a mut Node, path: &[usize]) -> &'a mut Node {
    path.iter().fold(n, |n, &k| &mut n.kids[k])
}

fn child(path: &[usize], k: usize) -> Vec<usize> {
    let mut p = path.to_vec();
    p.push(k);
    p
}

/// The constants of `tree` that only reach the root through a comparison or an
/// `if` condition, in pre-order. Constants under an aggregate, a custom block or
/// a meta operator never qualify.
pub(crate) fn threshold_constants(tree: &Node) -> Vec<Threshold> {
    let mut out = Vec::new();
    collect(tree, &mut Vec::new(), &mut out);
    out
}

fn sum_form(
    out: &mut Vec<Threshold>,
    u: &Node,
    u_path: &[usize],
    bound: Option<Vec<usize>>,
) {
    if !matches!(u.op, Op::Sub | Op::Add) || u.kids.len() != 2 {
        return;
    }
    for ci in 0..2 {
        let (c, x) = (&u.kids[ci], &u.kids[1 - ci]);
        if c.op != Op::Const || !has_input(x) {
            continue;
        }
        // sub(x, c) = k  =>  c = x - k;  sub(c, x) = k  =>  c = x + k;
        // add(x, c) = k  =>  c = k - x.
        let (bound_sign, scale) = match (u.op, ci) {
            (Op::Sub, 1) => (-1.0, 1.0),
            (Op::Sub, _) => (1.0, 1.0),
            _ => (1.0, -1.0),
        };
        out.push(Threshold {
            constant: child(u_path, ci),
            operand: child(u_path, 1 - ci),
            bound: bound.clone(),
            bound_sign,
            scale,
        });
    }
}

fn collect(n: &Node, path: &mut Vec<usize>, out: &mut Vec<Threshold>) {
    if n.op.is_agg() || n.op.is_meta() || n.op.is_custom() {
        return;
    }
    if is_comparison(n.op) && n.kids.len() == 2 {
        for j in 0..2 {
            let (kid, other) = (&n.kids[j], &n.kids[1 - j]);
            if kid.op == Op::Const && has_input(other) {
                out.push(Threshold {
                    constant: child(path, j),
                    operand: child(path, 1 - j),
                    bound: None,
                    bound_sign: 0.0,
                    scale: 1.0,
                });
            }
            if other.op == Op::Const {
                sum_form(out, kid, &child(path, j), Some(child(path, 1 - j)));
            }
        }
    }
    if n.op == Op::If && n.kids.len() == 3 {
        sum_form(out, &n.kids[0], &child(path, 0), None);
    }
    for (i, k) in n.kids.iter().enumerate() {
        path.push(i);
        collect(k, path, out);
        path.pop();
    }
}

struct Search<'a> {
    x: &'a [Vec<f64>],
    y: &'a [f64],
    w: &'a [f64],
    metric: Metric,
    squared: bool,
    cols: FitCols,
}

impl Search<'_> {
    /// The loss the constants are fitted on: the weighted residual sum of
    /// squares for the squared family, the configured metric otherwise. Lower
    /// is better; a non-finite loss is `+inf`.
    fn objective(&self, tree: &Node) -> f64 {
        if self.x.len() != self.y.len() || (!self.w.is_empty() && self.w.len() != self.x.len()) {
            return f64::INFINITY;
        }
        let v = if self.squared {
            self.cols.with_predictions(tree, self.x, |pred| {
                let mut s = 0.0;
                for (i, (&p, &t)) in pred.iter().zip(self.y).enumerate() {
                    let sw = self.w.get(i).copied().unwrap_or(1.0);
                    if sw == 0.0 {
                        continue;
                    }
                    let r = sw.sqrt() * (p - t);
                    s += r * r;
                }
                s
            })
        } else {
            metric_score_cols(tree, self.x, &self.cols, self.y, self.w, self.metric)
        };
        if v.is_finite() {
            v
        } else {
            f64::INFINITY
        }
    }

    /// Sorted distinct breakpoints of `th` on the fit rows.
    fn breakpoints(&self, tree: &Node, th: &Threshold) -> Vec<f64> {
        let operand = at(tree, &th.operand);
        let values = FitCols::gather(operand, self.x)
            .with_predictions(operand, self.x, |p| p.to_vec());
        let shift = match &th.bound {
            Some(path) => th.bound_sign * at(tree, path).val,
            None => 0.0,
        };
        let mut v: Vec<f64> = values
            .iter()
            .map(|&x| shift + th.scale * x)
            .filter(|b| b.is_finite())
            .collect();
        v.sort_by(f64::total_cmp);
        v.dedup();
        v
    }

    /// Search `th`; returns the new objective when the constant moved. The tree
    /// keeps the best constant found, or the start value when nothing is
    /// strictly better.
    fn fit_one(&self, tree: &mut Node, th: &Threshold, start_obj: f64) -> Option<f64> {
        let points = self.breakpoints(tree, th);
        if points.len() < 2 {
            return None;
        }
        let mids: Vec<f64> = points
            .windows(2)
            .map(|p| p[0] + (p[1] - p[0]) / 2.0)
            .zip(points.windows(2))
            .filter(|(m, p)| *m > p[0] && *m < p[1])
            .map(|(m, _)| m)
            .collect();
        if mids.is_empty() {
            return None;
        }
        let start = at(tree, &th.constant).val;
        let mut best = Best { obj: if start_obj.is_finite() { start_obj } else { f64::INFINITY }, t: start, current: true };
        let try_index = |tree: &mut Node, j: usize, best: &mut Best, best_mid: &mut Option<(Best, usize)>| {
            let t = mids[j];
            at_mut(tree, &th.constant).val = t;
            let cand = Best { obj: self.objective(tree), t, current: false };
            if cand.beats(best) {
                *best = cand;
            }
            if best_mid.as_ref().is_none_or(|(b, _)| cand.beats(b)) {
                *best_mid = Some((cand, j));
            }
        };
        let (mut lo, mut hi) = (0, mids.len());
        loop {
            let stride = (hi - lo).div_ceil(Q);
            let mut level_best: Option<(Best, usize)> = None;
            let mut j = lo;
            while j < hi {
                try_index(tree, j, &mut best, &mut level_best);
                j += stride;
            }
            let Some((_, jw)) = level_best else { break };
            if stride == 1 {
                break;
            }
            lo = jw.saturating_sub(stride - 1);
            hi = (jw + stride).min(mids.len());
        }
        if best.current {
            at_mut(tree, &th.constant).val = start;
            None
        } else {
            at_mut(tree, &th.constant).val = best.t;
            Some(best.obj)
        }
    }
}

#[derive(Clone, Copy)]
struct Best {
    obj: f64,
    t: f64,
    current: bool,
}

impl Best {
    /// Strictly lower objective; on a tie the current value stays, then the
    /// smaller absolute threshold wins.
    fn beats(&self, other: &Best) -> bool {
        self.obj < other.obj
            || (self.obj == other.obj
                && !other.current
                && self.obj.is_finite()
                && self.t.abs() < other.t.abs())
    }
}

/// The threshold step after the LM iterations of one fit.
///
/// `start` is the fitted value of the LM (squared error for the squared family,
/// the configured loss otherwise); it is returned unchanged, with the tree
/// untouched, when no threshold moves. After a move `refit` runs on a copy so
/// the other constants follow the new partition; the copy is kept only when it
/// is not worse. Two rounds at most.
pub(crate) fn fit_thresholds(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
    start: f64,
    mut refit: impl FnMut(&mut Node) -> f64,
) -> f64 {
    if iters == 0 || metric == Metric::ImplicitDerivative {
        return start;
    }
    let found = threshold_constants(tree);
    if found.is_empty() {
        return start;
    }
    let search = Search {
        x,
        y,
        w,
        metric,
        squared: metric.is_least_squares(),
        cols: FitCols::gather(tree, x),
    };
    let mut value = start;
    let mut current = search.objective(tree);
    for _ in 0..ROUNDS {
        let mut moved = false;
        for th in found.iter().take(MAX_THRESHOLDS) {
            if let Some(obj) = search.fit_one(tree, th, current) {
                current = obj;
                value = obj;
                moved = true;
                MOVES.fetch_add(1, Ordering::Relaxed);
            }
        }
        if !moved {
            break;
        }
        let mut trial = tree.clone();
        let v = refit(&mut trial);
        let obj = search.objective(&trial);
        if obj <= current {
            *tree = trial;
            current = obj;
            value = v;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{fit_weighted_strategy_metric, gpu_fit_plan_reason, FitStrategy, GpuFitReject};

    fn vars() -> Vec<String> {
        (0..3).map(|i| format!("x{i}")).collect()
    }

    fn tree(src: &str) -> Node {
        crate::parse::parse(src, &vars()).unwrap()
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn consts(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            consts(k, out);
        }
    }

    fn fit(t: &mut Node, x: &[Vec<f64>], y: &[f64], on: bool) -> f64 {
        fit_weighted_strategy_metric(t, x, y, &[], 8, FitStrategy::Exact, Metric::Squared, on)
    }

    fn sse(t: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
        x.iter().zip(y).map(|(r, &t0)| (crate::expr::eval(t, r) - t0).powi(2)).sum()
    }

    #[test]
    fn threshold_fit_finds_the_exact_gap() {
        let n = 200;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 + 0.37) / n as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| if r[0] > 0.3137 { 1.0 } else { 0.0 }).collect();
        let (below, above) = {
            let mut s: Vec<f64> = x.iter().map(|r| r[0]).collect();
            s.sort_by(f64::total_cmp);
            let k = s.iter().position(|&v| v > 0.3137).unwrap();
            (s[k - 1], s[k])
        };
        let start = "if(x0 - 0.5, 1, 0)";
        let mut lm = tree(start);
        fit(&mut lm, &x, &y, false);
        let mut c = Vec::new();
        consts(&lm, &mut c);
        assert_eq!(c[0], 0.5, "the LM leaves the threshold alone");
        assert!(sse(&lm, &x, &y) > 1.0);

        let mut t = tree(start);
        fit(&mut t, &x, &y, true);
        let mut c = Vec::new();
        consts(&t, &mut c);
        assert!(c[0] > below && c[0] < above, "threshold {} outside ({below}, {above})", c[0]);
        assert!(sse(&t, &x, &y) < 1e-18, "sse {}", sse(&t, &x, &y));
    }

    #[test]
    fn threshold_fit_hits_a_gap_finer_than_one_refinement() {
        let n = 65_536usize;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![((i * 40_503) % n) as f64 / n as f64 + 0.5 / n as f64])
            .collect();
        let y: Vec<f64> = x.iter().map(|r| if r[0] > 0.3137 { 1.0 } else { 0.0 }).collect();
        let mut s: Vec<f64> = x.iter().map(|r| r[0]).collect();
        s.sort_by(f64::total_cmp);
        let k = s.iter().position(|&v| v > 0.3137).unwrap();
        let (below, above) = (s[k - 1], s[k]);
        let mut t = tree("if(x0 - 0.9, 1, 0)");
        fit(&mut t, &x, &y, true);
        let mut c = Vec::new();
        consts(&t, &mut c);
        assert!(c[0] > below && c[0] < above, "threshold {} outside ({below}, {above})", c[0]);
        assert!(sse(&t, &x, &y) < 1e-18);
    }

    #[test]
    fn threshold_fit_never_worsens_the_objective() {
        let mut seed = 7u64;
        let x: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![lcg(&mut seed) * 4.0 - 2.0, lcg(&mut seed) * 4.0 - 2.0, lcg(&mut seed)])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| if r[0] > 0.4 { r[1] + 1.0 } else { -0.5 * r[1] })
            .collect();
        let mut moved = 0;
        for k in 0..50 {
            let c = lcg(&mut seed) * 4.0 - 2.0;
            let a = lcg(&mut seed) * 2.0 - 1.0;
            let src = match k % 5 {
                0 => format!("if(x0 - {c:.3}, {a:.3} * x1 + 1, x1)"),
                1 => format!("{a:.3} * x1 + greater(x0, {c:.3})"),
                2 => format!("if({c:.3} - x0, x1, {a:.3} + x1)"),
                3 => format!("x1 * less(x0 + {c:.3}, 0.5) + {a:.3}"),
                _ => format!("if(greater({c:.3}, x2) + x0, x1, 2 * x1)"),
            };
            let mut lm = tree(&src);
            let v_lm = fit(&mut lm, &x, &y, false);
            let mut th = tree(&src);
            let v_th = fit(&mut th, &x, &y, true);
            let (mut a_c, mut b_c) = (Vec::new(), Vec::new());
            consts(&lm, &mut a_c);
            consts(&th, &mut b_c);
            assert!(sse(&th, &x, &y) <= sse(&lm, &x, &y), "{src}");
            if a_c == b_c {
                assert_eq!(v_lm.to_bits(), v_th.to_bits(), "{src}");
            } else {
                moved += 1;
            }
        }
        assert!(moved > 0, "the fixture moves at least one threshold");
    }

    #[test]
    fn threshold_constants_are_routed_to_the_cpu_arm_under_the_flag() {
        let t = tree("greater(x0, 0.25) * x1 + 0.5");
        assert!(gpu_fit_plan_reason(&t, false).is_ok());
        let reason = gpu_fit_plan_reason(&t, true).err();
        assert_eq!(reason, Some(GpuFitReject::ThresholdConstant));
        assert_eq!(GpuFitReject::ThresholdConstant.label(), "threshold_constant");
        let plain = tree("x0 * 0.25 + 0.5");
        assert!(gpu_fit_plan_reason(&plain, true).is_ok());
    }

    #[test]
    fn threshold_plan_reject_needs_no_scope() {
        let t = tree("greater(x0, 0.25) * x1 + 0.5");
        std::thread::scope(|s| {
            s.spawn(|| {
                assert_eq!(gpu_fit_plan_reason(&t, true).err(), Some(GpuFitReject::ThresholdConstant));
                assert!(gpu_fit_plan_reason(&t, false).is_ok());
            });
        });
    }

    #[test]
    fn threshold_constants_select_only_comparison_operands() {
        assert_eq!(threshold_constants(&tree("greater(x0, 0.5)")).len(), 1);
        assert_eq!(threshold_constants(&tree("greater(0.5, x0)")).len(), 1);
        assert_eq!(threshold_constants(&tree("if(x0 - 0.5, x1, 0.2)")).len(), 1);
        assert_eq!(threshold_constants(&tree("if(0.5 + x0, x1, 0.2)")).len(), 1);
        assert!(threshold_constants(&tree("if(x0, x1, 0.2)")).is_empty());
        assert!(threshold_constants(&tree("x0 * 0.5 + 0.2")).is_empty());
        assert!(threshold_constants(&tree("greater(0.5, 0.2) * x0")).is_empty());
    }

    #[test]
    fn threshold_fit_is_deterministic() {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64 / 60.0, 1.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| if r[0] > 0.7 { 2.0 } else { 0.0 }).collect();
        let run = || {
            let mut t = tree("2 * greater(x0, 0.2)");
            let v = fit(&mut t, &x, &y, true);
            (t, v.to_bits())
        };
        assert_eq!(run(), run());
    }
}
