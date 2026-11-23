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

#[cfg(test)]
thread_local! {
    static REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PREDICTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OBJECTIVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LOCAL_MOVES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy)]
enum Rule {
    /// `cmp(c, x)` when `c_first`, else `cmp(x, c)`.
    Direct { cmp: Op, c_first: bool },
    /// `cmp(s, k)`, or `cmp(k, s)` when `k_first`; `s = sum(c, x)` when
    /// `c_first`, else `sum(x, c)`.
    Bounded { sum: Op, c_first: bool, cmp: Op, k: f64, k_first: bool },
    /// `if(s, a, b)`: branch `a` when `s > 0`.
    Branch { sum: Op, c_first: bool },
}

fn sum(op: Op, a: f64, b: f64) -> f64 {
    if op == Op::Sub {
        a - b
    } else {
        a + b
    }
}

/// The comparison of `expr::eval` on non-NaN operands. A NaN operand gives
/// `false` here; such a row has equal terms on both sides.
#[inline(always)]
fn holds(op: Op, a: f64, b: f64) -> bool {
    match op {
        Op::Greater => a > b,
        Op::Less => a < b,
        Op::GreaterOrEqual => a >= b,
        Op::LessOrEqual => a <= b,
        _ => a == b,
    }
}

impl Rule {
    /// The rule of `th` and the path of the decision node it controls, or
    /// `None` for a shape outside the three forms.
    fn of(tree: &Node, th: &Threshold) -> Option<(Rule, Vec<usize>)> {
        let (&ci, parent) = th.constant.split_last()?;
        let p = at(tree, parent);
        let c_first = ci == 0;
        if is_comparison(p.op) {
            return Some((Rule::Direct { cmp: p.op, c_first }, parent.to_vec()));
        }
        if !matches!(p.op, Op::Sub | Op::Add) {
            return None;
        }
        let (&ui, gp) = parent.split_last()?;
        let g = at(tree, gp);
        match (&th.bound, g.op) {
            (Some(b), op) if is_comparison(op) => {
                let k = at(tree, b);
                (k.op == Op::Const).then(|| {
                    (
                        Rule::Bounded { sum: p.op, c_first, cmp: op, k: k.val, k_first: ui == 1 },
                        gp.to_vec(),
                    )
                })
            }
            (None, Op::If) if ui == 0 => Some((Rule::Branch { sum: p.op, c_first }, gp.to_vec())),
            _ => None,
        }
    }
}

/// The two possible per-row terms of the objective, one for each side of a
/// threshold's decision.
struct Sides {
    rule: Rule,
    x: Vec<f64>,
    on: Vec<f64>,
    off: Vec<f64>,
}

impl Sides {
    #[inline(always)]
    fn total(&self, side: impl Fn(f64) -> bool) -> f64 {
        let mut s = 0.0;
        for i in 0..self.x.len() {
            s += if side(self.x[i]) { self.on[i] } else { self.off[i] };
        }
        s
    }

    /// `Search::objective` of the tree with the constant at `t`: the same terms
    /// added in the same order from `0.0`.
    fn objective(&self, t: f64) -> f64 {
        let s = match self.rule {
            Rule::Direct { cmp, c_first: true } => self.total(|x| holds(cmp, t, x)),
            Rule::Direct { cmp, c_first: false } => self.total(|x| holds(cmp, x, t)),
            Rule::Bounded { sum: op, c_first, cmp, k, k_first } => self.total(|x| {
                let v = if c_first { sum(op, t, x) } else { sum(op, x, t) };
                if k_first {
                    holds(cmp, k, v)
                } else {
                    holds(cmp, v, k)
                }
            }),
            Rule::Branch { sum: op, c_first } => {
                self.total(|x| (if c_first { sum(op, t, x) } else { sum(op, x, t) }) > 0.0)
            }
        };
        if s.is_finite() {
            s
        } else {
            f64::INFINITY
        }
    }
}

/// Midpoints between consecutive distinct breakpoints that lie strictly between
/// them.
fn midpoints(points: &[f64]) -> Vec<f64> {
    points
        .windows(2)
        .map(|p| p[0] + (p[1] - p[0]) / 2.0)
        .zip(points.windows(2))
        .filter(|(m, p)| *m > p[0] && *m < p[1])
        .map(|(m, _)| m)
        .collect()
}

impl Search<'_> {
    /// The loss the constants are fitted on: the weighted residual sum of
    /// squares for the squared family, the configured metric otherwise. Lower
    /// is better; a non-finite loss is `+inf`.
    fn objective(&self, tree: &Node) -> f64 {
        #[cfg(test)]
        OBJECTIVES.with(|c| c.set(c.get() + 1));
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

    /// Predictions of `tree` on the fit rows, on columns gathered for this
    /// tree: a tree that compiles never reads the columns of another tree.
    fn predict(&self, tree: &Node) -> Vec<f64> {
        #[cfg(test)]
        PREDICTS.with(|c| c.set(c.get() + 1));
        FitCols::gather(tree, self.x).with_predictions(tree, self.x, |p| p.to_vec())
    }

    /// The operand of `th` on the fit rows.
    fn operand_values(&self, tree: &Node, th: &Threshold) -> Vec<f64> {
        let operand = at(tree, &th.operand);
        FitCols::gather(operand, self.x).with_predictions(operand, self.x, |p| p.to_vec())
    }

    /// Sorted distinct breakpoints of `th` from its operand `values`.
    fn breakpoints(&self, tree: &Node, th: &Threshold, values: &[f64]) -> Vec<f64> {
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

    /// The two-state scorer of `th`, or `None` when the reference loop must
    /// score the cuts by evaluating the tree: a metric outside the least-squares
    /// family, any length mismatch (rows, weights, operand) or a shape outside
    /// the three forms. Per row, the unforced prediction stands in on rows with a
    /// NaN operand under a comparison, and on every row when the bound is NaN.
    /// Each forced tree gets its own `FitCols`, never `self.cols`.
    fn sides(&self, tree: &Node, th: &Threshold, operand: &[f64]) -> Option<Sides> {
        #[cfg(test)]
        if REFERENCE.with(|c| c.get()) {
            return None;
        }
        if !self.squared
            || self.x.len() != self.y.len()
            || operand.len() != self.x.len()
            || (!self.w.is_empty() && self.w.len() != self.x.len())
        {
            return None;
        }
        let (rule, node) = Rule::of(tree, th)?;
        let forced = |on: bool| {
            let mut t = tree.clone();
            let n = at_mut(&mut t, &node);
            *n = if n.op == Op::If {
                n.kids[if on { 1 } else { 2 }].clone()
            } else {
                crate::expr::constant(if on { 1.0 } else { 0.0 })
            };
            t
        };
        let (p_on, p_off) = (self.predict(&forced(true)), self.predict(&forced(false)));
        let k_nan = matches!(rule, Rule::Bounded { k, .. } if k.is_nan());
        let nan_rows = !matches!(rule, Rule::Branch { .. })
            && (k_nan || operand.iter().any(|v| v.is_nan()));
        // The comparison is NaN on these rows for every cut: the prediction is
        // the one of the unforced tree.
        let p_nan = nan_rows.then(|| self.predict(tree));
        let mut s = Sides { rule, x: Vec::new(), on: Vec::new(), off: Vec::new() };
        for i in 0..self.x.len() {
            let sw = self.w.get(i).copied().unwrap_or(1.0);
            if sw == 0.0 {
                continue;
            }
            let term = |p: f64| {
                let r = sw.sqrt() * (p - self.y[i]);
                r * r
            };
            let (a, b) = match &p_nan {
                Some(p) if k_nan || operand[i].is_nan() => (term(p[i]), term(p[i])),
                _ => (term(p_on[i]), term(p_off[i])),
            };
            s.x.push(operand[i]);
            s.on.push(a);
            s.off.push(b);
        }
        Some(s)
    }

    /// Search `th`; returns the new objective when the constant moved. The tree
    /// keeps the best constant found, or the start value when nothing is
    /// strictly better.
    fn fit_one(&self, tree: &mut Node, th: &Threshold, start_obj: f64) -> Option<f64> {
        let values = self.operand_values(tree, th);
        let points = self.breakpoints(tree, th, &values);
        if points.len() < 2 {
            return None;
        }
        let mids = midpoints(&points);
        if mids.is_empty() {
            return None;
        }
        let sides = self.sides(tree, th, &values);
        let start = at(tree, &th.constant).val;
        let mut best = Best { obj: if start_obj.is_finite() { start_obj } else { f64::INFINITY }, t: start, current: true };
        let try_index = |tree: &mut Node, j: usize, best: &mut Best, best_mid: &mut Option<(Best, usize)>| {
            let t = mids[j];
            let obj = match &sides {
                Some(s) => s.objective(t),
                None => {
                    at_mut(tree, &th.constant).val = t;
                    self.objective(tree)
                }
            };
            let cand = Best { obj, t, current: false };
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
                #[cfg(test)]
                LOCAL_MOVES.with(|c| c.set(c.get() + 1));
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

    const CMPS: [&str; 5] = ["greater", "less", "greater_or_equal", "less_or_equal", "equal"];
    const SUMS: [&str; 4] = ["x0 - 0.3", "0.3 - x0", "x0 + 0.3", "0.3 + x0"];

    const UNARY: [&str; 6] = ["abs", "exp", "square", "tanh", "logistic", "gauss"];
    const BINARY: [&str; 6] = ["add", "sub", "mul", "div", "min", "max"];

    /// Every operator once above the decision node and once inside the
    /// comparison operand, the anchor shapes and nested comparisons.
    fn operator_sources() -> Vec<String> {
        let mut out = Vec::new();
        for u in UNARY {
            out.push(format!("{u}(greater(x0, 0.3) + x1)"));
            out.push(format!("x1 * greater({u}(x0), 0.3) + 0.5"));
            out.push(format!("if({u}(x0) - 0.3, x1, x2)"));
        }
        for b in BINARY {
            out.push(format!("{b}(greater(x0, 0.3), x1)"));
            out.push(format!("x2 + {b}(less(x0, 0.2), x1 * x2)"));
            out.push(format!("x1 * less({b}(x0, x1), 0.3) + 0.5"));
            out.push(format!("if({b}(x0, x1) - 0.3, x1, x2)"));
        }
        out.push("x1 * greater(0.3, 0.8 * (0.5 + x0))".to_string());
        out.push("x1 * greater(0.8 * (0.5 + x0), 0.3)".to_string());
        out.push("x1 * less(0.3, 0.8 * (0.5 + x0)) + x2".to_string());
        out.push("greater(greater(x0, 0.2) + x1, 0.3) * x2".to_string());
        out.push("if(greater(less(x0, 0.4), 0.5), x1, x2)".to_string());
        out.push("less(x1 * greater(x0, 0.1), 0.4) + x2".to_string());
        out
    }

    fn form_sources() -> Vec<String> {
        let mut out = Vec::new();
        for cmp in CMPS {
            out.push(format!("x1 * {cmp}(x0, 0.3) + 0.5"));
            out.push(format!("x1 * {cmp}(0.3, x0) + 0.5"));
            out.push(format!("x1 * {cmp}(x0 * x1, 0.3) + greater(x2, 0.1)"));
            out.push(format!("if({cmp}(x0, 0.3), x1 * 2, x1 - 1)"));
            for s in SUMS {
                out.push(format!("x1 * {cmp}({s}, 0.2) + 0.5"));
                out.push(format!("x1 * {cmp}(0.2, {s}) + 0.5"));
                out.push(format!("x1 * {cmp}({s}, 0.2) + less(x2, 0.1)"));
                out.push(format!("if({cmp}({s}, 0.2), x1 * 2, x1 - 1)"));
            }
        }
        for s in SUMS {
            out.push(format!("if({s}, x1 * 2, x1 - 1)"));
            out.push(format!("if({s}, greater(x1, 0.2), x2) + x1"));
            out.push(format!("if({s}, x2, 0.7) * x1"));
        }
        out.extend(operator_sources());
        out
    }

    fn special_rows(n: usize, seed: u64, with_nan: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut s = seed;
        let mut specials = vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            5e-324,
            -5e-324,
            0.25,
            0.25,
            -0.5,
            0.3,
            0.2,
            0.5,
        ];
        if with_nan {
            specials.push(f64::NAN);
        }
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..n {
            let a = if lcg(&mut s) < 0.4 {
                specials[(lcg(&mut s) * specials.len() as f64) as usize % specials.len()]
            } else {
                lcg(&mut s) * 2.0 - 1.0
            };
            let b = lcg(&mut s) * 2.0 - 1.0;
            let c = if lcg(&mut s) < 0.1 { 0.0 } else { lcg(&mut s) * 2.0 - 1.0 };
            x.push(vec![a, b, c]);
            y.push(lcg(&mut s) * 2.0 - 1.0);
        }
        (x, y)
    }

    /// Every midpoint of every threshold of `src`: two-state bits against the tree.
    fn check_sides(src: &str, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> usize {
        let t = tree(src);
        let found = threshold_constants(&t);
        assert!(!found.is_empty(), "{src}");
        let mut checked = 0;
        for th in &found {
            let search = Search {
                x,
                y,
                w,
                metric: Metric::Squared,
                squared: true,
                cols: FitCols::gather(&t, x),
            };
            let values = search.operand_values(&t, th);
            let mids = midpoints(&search.breakpoints(&t, th, &values));
            let sides = search
                .sides(&t, th, &values)
                .unwrap_or_else(|| panic!("no two-state rule for {src}"));
            for &m in &mids {
                let mut at_m = t.clone();
                at_mut(&mut at_m, &th.constant).val = m;
                assert_eq!(
                    sides.objective(m).to_bits(),
                    search.objective(&at_m).to_bits(),
                    "{src}, cut {m:?}, weights {}",
                    w.len()
                );
                checked += 1;
            }
        }
        checked
    }

    #[test]
    fn two_state_objective_is_the_tree_objective_bit_for_bit() {
        let mut checked = 0;
        let mut finite = 0usize;
        for (set, with_nan) in [(1u64, false), (2, true)] {
            let (x, y) = special_rows(500, set, with_nan);
            let mut s = 99u64;
            let zeros: Vec<f64> = (0..x.len())
                .map(|_| if lcg(&mut s) < 0.2 { 0.0 } else { lcg(&mut s) * 2.0 })
                .collect();
            let negative: Vec<f64> = (0..x.len()).map(|i| if i % 7 == 0 { -1.0 } else { 1.0 }).collect();
            let nan_weights: Vec<f64> = (0..x.len()).map(|i| if i % 11 == 0 { f64::NAN } else { 1.0 }).collect();
            let zero_at_nan: Vec<f64> = x
                .iter()
                .map(|r| if r[0].is_nan() { 0.0 } else { 1.0 })
                .collect();
            for src in form_sources() {
                for w in [&[][..], &zeros[..], &zero_at_nan[..], &negative[..], &nan_weights[..]] {
                    checked += check_sides(&src, &x, &y, w);
                    let t = tree(&src);
                    let search = Search {
                        x: &x,
                        y: &y,
                        w,
                        metric: Metric::Squared,
                        squared: true,
                        cols: FitCols::gather(&t, &x),
                    };
                    finite += usize::from(search.objective(&t).is_finite());
                }
            }
        }
        assert!(checked > 100_000, "{checked} cuts checked");
        assert!(finite > 100, "{finite} trees with a finite objective on a NaN-free row set");
    }

    fn random_source(k: usize, s: &mut u64) -> String {
        let a = format!("{:.3}", lcg(s) * 2.0);
        let b = format!("{:.3}", lcg(s) * 1.0);
        let c = format!("{:.3}", lcg(s) * 2.0);
        match k % 16 {
            0 => format!("if(x0 - {a}, {b} * x1 + 1, x1)"),
            1 => format!("{b} * x1 + greater(x0, {a})"),
            2 => format!("if({a} - x0, x1, {b} + x1)"),
            3 => format!("x1 * less(x0 + {a}, 0.5) + {b}"),
            4 => format!("if(greater({a}, x2) + x0, x1, 2 * x1)"),
            5 => format!("if(x0 - {a}, x1, 2) + greater(x2, {c}) * x1"),
            6 => format!("{b} * greater_or_equal(x0 * x1, {a}) + less_or_equal({c}, x2)"),
            7 => format!("if(x0 + {a}, if(x1 - {c}, x2, 1), {b} * x1)"),
            8 => format!("equal(x0, {a}) * x1 + {b} * x2"),
            9 => format!("greater(x0 - {a}, {c}) * x1 + {b}"),
            10 => format!("{b} * (x1 + 0.5) * greater(x2, {c}) + less(x0, {a})"),
            11 => format!("if(x0 - {a}, gauss(x1) * {b}, x2) + if(x2 - {c}, 1, x0)"),
            12 => format!("{}(greater(x0, {a}) + x1) + {b}", UNARY[k / 16 % 6]),
            13 => format!("x1 * greater({}(x0), {a}) + {b}", UNARY[k / 16 % 6]),
            14 => format!("{}(greater(x0, {a}), x1) + {b} * x2", BINARY[k / 16 % 6]),
            15 => format!("x1 * greater({b} * (0.5 + x0), {a}) + less({c}, {b} * (0.5 + x2))"),
            _ => unreachable!(),
        }
    }

    fn dataset(n: usize, cols: usize, discrete: bool, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut s = seed;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let mut r: Vec<f64> = (0..cols).map(|_| lcg(&mut s) * 4.0 - 2.0).collect();
            if discrete {
                r[0] = (lcg(&mut s) * 40.0).floor() / 10.0 - 2.0;
            }
            let v = if r[0] > 0.4 { r[1] + 1.0 } else { -0.5 * r[1] };
            y.push(v + if r[2] > -0.3 { 0.6 } else { 0.0 } + (lcg(&mut s) - 0.5) * 0.05);
            x.push(r);
        }
        (x, y)
    }

    type FitRun = (Vec<u64>, u64, u64);

    /// The tree of `src` with its threshold constants negated when `negate`: the
    /// parser reads a negative literal as `neg(c)`, which is not a constant.
    fn built(src: &str, negate: bool) -> Node {
        let mut t = tree(src);
        if negate {
            for th in threshold_constants(&t) {
                let c = at_mut(&mut t, &th.constant);
                c.val = -c.val;
            }
        }
        t
    }

    fn fit_run(src: &str, negate: bool, x: &[Vec<f64>], y: &[f64], reference: bool) -> FitRun {
        REFERENCE.with(|c| c.set(reference));
        let before = LOCAL_MOVES.with(|c| c.get());
        let mut t = built(src, negate);
        let v = fit(&mut t, x, y, true);
        REFERENCE.with(|c| c.set(false));
        let mut c = Vec::new();
        consts(&t, &mut c);
        (
            c.iter().map(|v| v.to_bits()).collect(),
            v.to_bits(),
            LOCAL_MOVES.with(|c| c.get()) - before,
        )
    }

    #[test]
    fn two_state_fit_equals_the_reference_fit() {
        let mut seed = 5u64;
        let mut moves = 0;
        let mut trees = 0;
        for (n, cols, discrete, count) in
            [(200usize, 3usize, false, 300usize), (5_000, 3, true, 90), (57_760, 8, false, 10)]
        {
            let (x, y) = dataset(n, cols, discrete, n as u64);
            for k in 0..count {
                let src = random_source(k, &mut seed);
                PREDICTS.with(|c| c.set(0));
                let negate = k % 2 == 1;
                let fast = fit_run(&src, negate, &x, &y, false);
                assert!(
                    threshold_constants(&tree(&src)).is_empty() || PREDICTS.with(|c| c.get()) > 0,
                    "the two-state scorer ran for {src} on {n} rows"
                );
                let reference = fit_run(&src, negate, &x, &y, true);
                assert_eq!(fast, reference, "{src} on {n} rows");
                moves += fast.2;
                trees += 1;
            }
        }
        assert_eq!(trees, 400);
        assert!(moves > 100, "{moves} thresholds moved");
    }

    #[test]
    fn two_state_makes_no_tree_evaluation_per_cut() {
        let n = 57_760usize;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(((i * 40_503) % n) as f64 + 0.5) / n as f64, ((i * 7_919) % n) as f64 / n as f64])
            .collect();
        let y: Vec<f64> = x.iter().map(|r| if r[0] > 0.3137 { r[1] } else { 0.0 }).collect();
        let t0 = tree("greater(x0, 0.9) * x1 + 0.2");
        let th = threshold_constants(&t0).remove(0);
        let run = |reference: bool| {
            REFERENCE.with(|c| c.set(reference));
            let search = Search {
                x: &x,
                y: &y,
                w: &[],
                metric: Metric::Squared,
                squared: true,
                cols: FitCols::gather(&t0, &x),
            };
            let start = search.objective(&t0);
            PREDICTS.with(|c| c.set(0));
            OBJECTIVES.with(|c| c.set(0));
            let mut t = t0.clone();
            let got = search.fit_one(&mut t, &th, start);
            REFERENCE.with(|c| c.set(false));
            (
                OBJECTIVES.with(|c| c.get()),
                PREDICTS.with(|c| c.get()),
                at(&t, &th.constant).val.to_bits(),
                got.map(f64::to_bits),
            )
        };
        let reference = run(true);
        let fast = run(false);
        assert_eq!(reference.0, 184, "the reference scores one tree per candidate");
        assert_eq!(fast.0, 0, "no objective call inside the level loop");
        assert!(fast.1 <= 3, "{} predictions", fast.1);
        assert_eq!((fast.2, fast.3), (reference.2, reference.3));
        assert!(fast.3.is_some(), "the threshold moves");
    }

    fn weighted_run(
        src: &str,
        x: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        metric: Metric,
        reference: bool,
    ) -> (Vec<u64>, u64, usize) {
        REFERENCE.with(|c| c.set(reference));
        PREDICTS.with(|c| c.set(0));
        let mut t = tree(src);
        let v = fit_weighted_strategy_metric(&mut t, x, y, w, 8, FitStrategy::Exact, metric, true);
        REFERENCE.with(|c| c.set(false));
        let mut c = Vec::new();
        consts(&t, &mut c);
        (c.iter().map(|v| v.to_bits()).collect(), v.to_bits(), PREDICTS.with(|c| c.get()))
    }

    #[test]
    fn a_metric_outside_least_squares_is_scored_by_the_reference_loop() {
        let (x, y) = dataset(300, 3, false, 1);
        for src in ["x1 * greater(x0, 0.3) + 0.5", "if(x0 - 0.3, x1, 2)", "greater(0.4 - x0, 0.3) * x2"] {
            let fast = weighted_run(src, &x, &y, &[], Metric::Absolute, false);
            let reference = weighted_run(src, &x, &y, &[], Metric::Absolute, true);
            assert_eq!(fast.2, 0, "{src}: no two-state prediction");
            assert_eq!((fast.0, fast.1), (reference.0, reference.1), "{src}");
        }
    }

    #[test]
    fn odd_weights_give_the_reference_bits() {
        let (x, y) = dataset(200, 3, false, 2);
        let mut s = 17u64;
        let negative: Vec<f64> = (0..x.len()).map(|_| if lcg(&mut s) < 0.1 { -1.0 } else { 1.0 }).collect();
        let nan: Vec<f64> = (0..x.len()).map(|_| if lcg(&mut s) < 0.1 { f64::NAN } else { 1.0 }).collect();
        let zeros: Vec<f64> = (0..x.len()).map(|_| if lcg(&mut s) < 0.3 { 0.0 } else { 0.5 + lcg(&mut s) }).collect();
        let short: Vec<f64> = vec![1.0; x.len() - 1];
        for src in ["x1 * greater(x0, 0.3) + 0.5", "if(x0 - 0.3, x1, 2)", "greater(x0 - 0.1, 0.3) * x2 + x1"] {
            for w in [&negative, &nan, &zeros, &short] {
                let fast = weighted_run(src, &x, &y, w, Metric::Squared, false);
                let reference = weighted_run(src, &x, &y, w, Metric::Squared, true);
                assert_eq!((fast.0, fast.1), (reference.0, reference.1), "{src}");
            }
        }
    }

    #[test]
    #[ignore = "timing bench: cargo test --release bench_threshold_fit_two_state -- --ignored --nocapture"]
    fn bench_threshold_fit_two_state() {
        let names: Vec<String> = (0..32).map(|i| format!("x{i}")).collect();
        let n = 57_760;
        let (x, y) = dataset(n, 32, false, 3);
        let srcs = [
            "x1 * greater(x0, 0.4) + 0.5",
            "x3 * (x1 + 0.7) * greater(0.1, x2) + 0.2",
            "x1 * less(x0 + 0.3, 0.2) + x5",
            "0.8 * (x4 + 1.1) * greater_or_equal(x0 - 0.5, 0.1) + x1",
            "if(x0 - 0.4, x1 * 1.5 + 1, x2 * 0.3)",
            "if(0.2 + x2, gauss(x1) * 0.4, x3)",
            "greater(x0, 0.4) * x1 + less(x2, -0.3) * x4",
            "if(x0 - 0.4, x1, 2) + greater(x2, -0.3) * x1",
            "0.9 * (x1 + 0.4) * greater(x0 * x2, 0.1) + if(x3 - 0.2, x4, 1)",
            "x2 * less_or_equal(x7 - 0.4, 0.3) + if(x0 - 0.1, 0.5, x1) * x3",
            "(x1 + 0.3) * tanh(x4) * greater(x0, 0.3) + x2",
            "if(x0 - 0.4, if(x2 - 0.3, x1, x3), 0.5 * x4) + 0.1",
        ];
        let (mut sum_reference, mut sum_fast) = (0.0, 0.0);
        for src in srcs {
            let time = |reference: bool| {
                REFERENCE.with(|c| c.set(reference));
                let mut t = crate::parse::parse(src, &names).unwrap();
                let start = std::time::Instant::now();
                let v = fit(&mut t, &x, &y, true);
                REFERENCE.with(|c| c.set(false));
                (start.elapsed().as_secs_f64() * 1e3, v)
            };
            let (reference_ms, a) = time(true);
            let (fast_ms, b) = time(false);
            assert_eq!(a.to_bits(), b.to_bits(), "{src}");
            sum_reference += reference_ms;
            sum_fast += fast_ms;
            println!("{src}\treference {reference_ms:.1} ms\tfast {fast_ms:.1} ms");
        }
        println!("sum\treference {sum_reference:.1} ms\tfast {sum_fast:.1} ms\tratio {:.3}", sum_fast / sum_reference);
    }
}
