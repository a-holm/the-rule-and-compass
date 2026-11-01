//! Linear scaling (#470): a free affine output map `a + b * f(x)` around every
//! candidate shape `f`.
//!
//! The wrapper is an ordinary tree, `add(const a, mul(const b, f))`, so every
//! evaluator, fitter and device path sees a plain expression. Variation edits
//! the bare shape: [`peel`] removes the wrapper from a parent and [`wrap`]
//! puts a wrapper on the child.

use crate::expr::{self, Node, Op};

/// `add(a, mul(b, f))`.
pub fn wrap(a: f64, b: f64, f: Node) -> Node {
    expr::node(
        Op::Add,
        vec![expr::constant(a), expr::node(Op::Mul, vec![expr::constant(b), f])],
    )
}

/// The affine map and shape of a wrapped tree, accepting either child order of
/// both operators. Simplification may reorder them.
pub fn peel(tree: &Node) -> Option<(f64, f64, &Node)> {
    if tree.op != Op::Add || tree.kids.len() != 2 {
        return None;
    }
    for (c, m) in [(0, 1), (1, 0)] {
        let (cn, mn) = (&tree.kids[c], &tree.kids[m]);
        if cn.op != Op::Const || mn.op != Op::Mul || mn.kids.len() != 2 {
            continue;
        }
        for (bi, fi) in [(0, 1), (1, 0)] {
            if mn.kids[bi].op == Op::Const && mn.kids[fi].op != Op::Const {
                return Some((cn.val, mn.kids[bi].val, &mn.kids[fi]));
            }
        }
    }
    None
}

/// The bare shape of a tree and its map, or the tree itself with the identity
/// map when it carries no wrapper.
pub fn split(tree: &Node) -> (f64, f64, Node) {
    match peel(tree) {
        Some((a, b, f)) => (a, b, f.clone()),
        None => (0.0, 1.0, tree.clone()),
    }
}

/// Wrap `tree` unless it already carries a wrapper.
pub fn ensure(tree: Node, a: f64, b: f64) -> Node {
    if peel(&tree).is_some() {
        tree
    } else {
        wrap(a, b, tree)
    }
}

/// Complexity the wrapper adds, in the configured model: the same for every
/// shape because the wrapper adds two operators, two constants and two levels.
pub fn overhead(model: &crate::complexity::ComplexityModel) -> f64 {
    let leaf = expr::input(0);
    model.complexity(&wrap(0.0, 1.0, leaf.clone())) - model.complexity(&leaf)
}

/// Weighted least squares `min_{a,b} sum w (y - a - b p)^2` from centred
/// moments. The deviations of `p` are rescaled by their largest magnitude so
/// tiny or huge predictions neither underflow nor overflow. A constant or
/// non-finite `p` gives `b = 0` and `a = mean(y)`.
pub fn closed_form(pred: &[f64], y: &[f64], w: &[f64]) -> (f64, f64) {
    let n = pred.len().min(y.len());
    let weight = |i: usize| if w.is_empty() { 1.0 } else { w[i] };
    let (mut sw, mut mp, mut my) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let wi = weight(i);
        sw += wi;
        mp += wi * pred[i];
        my += wi * y[i];
    }
    if !(sw > 0.0) || !sw.is_finite() {
        return (0.0, 1.0);
    }
    mp /= sw;
    my /= sw;
    if !my.is_finite() {
        return (0.0, 1.0);
    }
    if !mp.is_finite() {
        return (my, 0.0);
    }
    let scale = (0..n).map(|i| (pred[i] - mp).abs()).fold(0.0f64, f64::max);
    if !(scale > 0.0) || !scale.is_finite() {
        return (my, 0.0);
    }
    let (mut var, mut cov) = (0.0, 0.0);
    for i in 0..n {
        let wi = weight(i);
        let d = (pred[i] - mp) / scale;
        var += wi * d * d;
        cov += wi * d * (y[i] - my);
    }
    if !(var > 0.0) || !var.is_finite() || !cov.is_finite() {
        return (my, 0.0);
    }
    let b = cov / var / scale;
    if !b.is_finite() {
        return (my, 0.0);
    }
    (my - b * mp, b)
}

/// Wrap `f` with the closed-form map fitted to its predictions on `x`, `y`.
pub fn wrap_closed_form(f: Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Node {
    let pred: Vec<f64> = x.iter().map(|row| expr::eval(&f, row)).collect();
    let (a, b) = closed_form(&pred, y, w);
    wrap(a, b, f)
}

/// Replace the map of a wrapped tree by the closed-form map of its shape on the
/// given rows. A tree without a wrapper is left as it is.
pub fn restart(tree: &mut Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) {
    let Some((_, _, f)) = peel(tree) else { return };
    let pred: Vec<f64> = x.iter().map(|row| expr::eval(f, row)).collect();
    let (a, b) = closed_form(&pred, y, w);
    let f = f.clone();
    *tree = wrap(a, b, f);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peel_accepts_both_child_orders() {
        let f = expr::node(Op::Sin, vec![expr::input(0)]);
        let w = wrap(1.5, 2.5, f.clone());
        assert_eq!(peel(&w).map(|(a, b, g)| (a, b, g.clone())), Some((1.5, 2.5, f.clone())));
        let swapped = expr::node(
            Op::Add,
            vec![
                expr::node(Op::Mul, vec![f.clone(), expr::constant(2.5)]),
                expr::constant(1.5),
            ],
        );
        assert_eq!(peel(&swapped).map(|(a, b, g)| (a, b, g.clone())), Some((1.5, 2.5, f)));
        assert!(peel(&expr::input(0)).is_none());
    }

    #[test]
    fn closed_form_recovers_the_map() {
        let p: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = p.iter().map(|v| 2.0 * v + 1.0).collect();
        let (a, b) = closed_form(&p, &y, &[]);
        assert!((a - 1.0).abs() < 1e-12 && (b - 2.0).abs() < 1e-12);
        let (a, b) = closed_form(&[3.0; 5], &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert_eq!((a, b), (3.0, 0.0));
    }

    #[test]
    fn closed_form_survives_tiny_predictions() {
        let p: Vec<f64> = (0..20).map(|i| 1e-170 * i as f64).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (_, b) = closed_form(&p, &y, &[]);
        assert!((b * 1e-170 - 1.0).abs() < 1e-12);
    }
}
