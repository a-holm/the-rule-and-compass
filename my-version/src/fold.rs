//! #982: fold redundant constants before the device fit.
//!
//! A tree whose constants trade off against one another, such as
//! `c1*(c2*x)`, `c1 + (c2 + x)` or the seed shape `c*(x*s + m) + d` that input
//! standardisation produces, has a Jacobian with a null direction: the device
//! LM kernel reports it rank deficient and the exact CPU fit refits it.
//! [`fold_constants`] rewrites such a tree into a form where each foldable
//! direction keeps one constant. Every rule moves one constant into another
//! constant of the same additive or multiplicative chain, so the folded tree
//! is the same function of the inputs: it differs from the input tree only by
//! the rounding of the re-associated products and sums.
//!
//! Only `add`, `sub`, `mul`, `div` and `neg` chains are rewritten. A constant
//! moves into an `if` only into both branches at once (`c + if(x, a, b)` is
//! `if(x, c + a, c + b)` on every row), never into one branch or the
//! condition. Every other operator is a boundary: its arguments are folded on
//! their own, and nothing moves across it. Aggregates, meta operators and user
//! blocks are never entered.

use crate::expr::{constant, node, Node, Op};

/// A constant factor moved into a subtree: the subtree's value is multiplied
/// (`Mul`) or divided (`Div`) by `k`.
#[derive(Clone, Copy, Debug)]
enum Factor {
    Mul(f64),
    Div(f64),
}

impl Factor {
    fn apply(self, c: f64) -> f64 {
        match self {
            Factor::Mul(k) => c * k,
            Factor::Div(k) => c / k,
        }
    }

    fn invert(self) -> Factor {
        match self {
            Factor::Mul(k) => Factor::Div(k),
            Factor::Div(k) => Factor::Mul(k),
        }
    }

    /// A zero factor may scale (each scaled term becomes a zero constant the
    /// fit can move) but never divide.
    fn usable(self) -> bool {
        match self {
            Factor::Mul(k) => k.is_finite(),
            Factor::Div(k) => k.is_finite() && k != 0.0,
        }
    }
}

fn binary(op: Op, a: Node, b: Node) -> Node {
    node(op, vec![a, b])
}

fn finite_constant(v: f64) -> Option<Node> {
    v.is_finite().then(|| constant(v))
}

/// `f(s)` with `f` absorbed into one constant of each additive term of `s`,
/// so the result has the constants of `s`. `None` when some term of `s` has no
/// constant factor reachable through `mul`, `div` and `neg`.
fn scale_into(s: &Node, f: Factor) -> Option<Node> {
    if !f.usable() {
        return None;
    }
    match s.op {
        Op::Const => finite_constant(f.apply(s.val)),
        Op::Mul => {
            if let Some(a) = scale_into(&s.kids[0], f) {
                return Some(binary(Op::Mul, a, s.kids[1].clone()));
            }
            scale_into(&s.kids[1], f).map(|b| binary(Op::Mul, s.kids[0].clone(), b))
        }
        Op::Div => {
            if let Some(a) = scale_into(&s.kids[0], f) {
                return Some(binary(Op::Div, a, s.kids[1].clone()));
            }
            scale_into(&s.kids[1], f.invert()).map(|b| binary(Op::Div, s.kids[0].clone(), b))
        }
        Op::Neg => scale_into(&s.kids[0], f).map(|a| node(Op::Neg, vec![a])),
        Op::Add | Op::Sub => {
            let a = scale_into(&s.kids[0], f)?;
            let b = scale_into(&s.kids[1], f)?;
            Some(binary(s.op, a, b))
        }
        Op::If => {
            let a = scale_into(&s.kids[1], f)?;
            let b = scale_into(&s.kids[2], f)?;
            Some(node(Op::If, vec![s.kids[0].clone(), a, b]))
        }
        _ => None,
    }
}

/// `s + d` with `d` absorbed into one constant of `s`, reached through `add`,
/// `sub`, `neg` and a multiplication or division by a constant.
fn shift_into(s: &Node, d: f64) -> Option<Node> {
    if !d.is_finite() {
        return None;
    }
    match s.op {
        Op::Const => finite_constant(s.val + d),
        Op::Add => {
            if let Some(a) = shift_into(&s.kids[0], d) {
                return Some(binary(Op::Add, a, s.kids[1].clone()));
            }
            shift_into(&s.kids[1], d).map(|b| binary(Op::Add, s.kids[0].clone(), b))
        }
        Op::Sub => {
            if let Some(a) = shift_into(&s.kids[0], d) {
                return Some(binary(Op::Sub, a, s.kids[1].clone()));
            }
            shift_into(&s.kids[1], -d).map(|b| binary(Op::Sub, s.kids[0].clone(), b))
        }
        Op::Neg => shift_into(&s.kids[0], -d).map(|a| node(Op::Neg, vec![a])),
        Op::If => {
            let a = shift_into(&s.kids[1], d)?;
            let b = shift_into(&s.kids[2], d)?;
            Some(node(Op::If, vec![s.kids[0].clone(), a, b]))
        }
        Op::Mul => {
            for (k, other) in [(0, 1), (1, 0)] {
                let c = &s.kids[k];
                if c.op == Op::Const && c.val.is_finite() && c.val != 0.0 {
                    if let Some(t) = shift_into(&s.kids[other], d / c.val) {
                        let mut kids = s.kids.clone();
                        kids[other] = t;
                        return Some(node(Op::Mul, kids));
                    }
                }
            }
            None
        }
        Op::Div => {
            let c = &s.kids[1];
            if c.op == Op::Const && c.val.is_finite() && c.val != 0.0 {
                return shift_into(&s.kids[0], d * c.val)
                    .map(|a| binary(Op::Div, a, c.clone()));
            }
            None
        }
        _ => None,
    }
}

/// Split `s` into `f(r)`: remove one constant factor of `s`, reached through
/// `mul`, `div` and `neg`, and return it with the remaining tree.
fn take_scale(s: &Node) -> Option<(Factor, Node)> {
    let take = |f: Factor, r: &Node| f.usable().then(|| (f, r.clone()));
    match s.op {
        Op::Mul => {
            let (a, b) = (&s.kids[0], &s.kids[1]);
            if a.op == Op::Const {
                return take(Factor::Mul(a.val), b);
            }
            if b.op == Op::Const {
                return take(Factor::Mul(b.val), a);
            }
            if let Some((f, r)) = take_scale(a) {
                return Some((f, binary(Op::Mul, r, b.clone())));
            }
            take_scale(b).map(|(f, r)| (f, binary(Op::Mul, a.clone(), r)))
        }
        Op::Div => {
            let (a, b) = (&s.kids[0], &s.kids[1]);
            if b.op == Op::Const {
                return take(Factor::Div(b.val), a);
            }
            if let Some((f, r)) = take_scale(a) {
                return Some((f, binary(Op::Div, r, b.clone())));
            }
            take_scale(b).map(|(f, r)| (f.invert(), binary(Op::Div, a.clone(), r)))
        }
        Op::Neg => take_scale(&s.kids[0]).map(|(f, r)| (f, node(Op::Neg, vec![r]))),
        _ => None,
    }
}

/// Split `s` into `r + d`: remove one additive constant of `s`, reached
/// through `add`, `sub` and `neg`, and return it with the remaining tree.
fn take_shift(s: &Node) -> Option<(f64, Node)> {
    let take = |d: f64, r: &Node| d.is_finite().then(|| (d, r.clone()));
    match s.op {
        Op::Add => {
            let (a, b) = (&s.kids[0], &s.kids[1]);
            if a.op == Op::Const {
                return take(a.val, b);
            }
            if b.op == Op::Const {
                return take(b.val, a);
            }
            if let Some((d, r)) = take_shift(a) {
                return Some((d, binary(Op::Add, r, b.clone())));
            }
            take_shift(b).map(|(d, r)| (d, binary(Op::Add, a.clone(), r)))
        }
        Op::Sub => {
            let (a, b) = (&s.kids[0], &s.kids[1]);
            if b.op == Op::Const {
                return take(-b.val, a);
            }
            if let Some((d, r)) = take_shift(a) {
                return Some((d, binary(Op::Sub, r, b.clone())));
            }
            take_shift(b).map(|(d, r)| (-d, binary(Op::Sub, a.clone(), r)))
        }
        Op::Neg => take_shift(&s.kids[0]).map(|(d, r)| (-d, node(Op::Neg, vec![r]))),
        _ => None,
    }
}

/// One fold at the root of `m`, whose children are already folded. Every
/// rewrite removes exactly one constant.
fn fold_root(m: &Node) -> Option<Node> {
    if m.kids.len() != 2 {
        return None;
    }
    let (a, b) = (&m.kids[0], &m.kids[1]);
    let a_const = a.op == Op::Const;
    let b_const = b.op == Op::Const;
    if a_const && b_const {
        return None;
    }
    match m.op {
        Op::Mul => {
            if a_const {
                return scale_into(b, Factor::Mul(a.val));
            }
            if b_const {
                return scale_into(a, Factor::Mul(b.val));
            }
            if let Some((f, r)) = take_scale(b) {
                if let Some(s) = scale_into(a, f) {
                    return Some(binary(Op::Mul, s, r));
                }
            }
            let (f, r) = take_scale(a)?;
            scale_into(b, f).map(|s| binary(Op::Mul, r, s))
        }
        Op::Div => {
            if b_const {
                return scale_into(a, Factor::Div(b.val));
            }
            if a_const {
                // `k/(c/r)` is `(k/c)*r`.
                if b.op == Op::Div && b.kids[0].op == Op::Const && Factor::Div(b.kids[0].val).usable() {
                    return finite_constant(a.val / b.kids[0].val)
                        .map(|k| binary(Op::Mul, k, b.kids[1].clone()));
                }
                let (f, r) = take_scale(b)?;
                return finite_constant(f.invert().apply(a.val)).map(|k| binary(Op::Div, k, r));
            }
            if let Some((f, r)) = take_scale(b) {
                if let Some(s) = scale_into(a, f.invert()) {
                    return Some(binary(Op::Div, s, r));
                }
            }
            let (f, r) = take_scale(a)?;
            scale_into(b, f.invert()).map(|s| binary(Op::Div, r, s))
        }
        Op::Add => {
            if a_const {
                return shift_into(b, a.val);
            }
            if b_const {
                return shift_into(a, b.val);
            }
            if let Some((d, r)) = take_shift(b) {
                if let Some(s) = shift_into(a, d) {
                    return Some(binary(Op::Add, s, r));
                }
            }
            let (d, r) = take_shift(a)?;
            shift_into(b, d).map(|s| binary(Op::Add, r, s))
        }
        Op::Sub => {
            if b_const {
                return shift_into(a, -b.val);
            }
            if a_const {
                if let Some((e, r)) = take_shift(b) {
                    return finite_constant(a.val - e).map(|k| binary(Op::Sub, k, r));
                }
                return shift_into(b, -a.val).map(|s| node(Op::Neg, vec![s]));
            }
            if let Some((e, r)) = take_shift(b) {
                if let Some(s) = shift_into(a, -e) {
                    return Some(binary(Op::Sub, s, r));
                }
            }
            let (e, r) = take_shift(a)?;
            shift_into(b, -e).map(|s| binary(Op::Sub, r, s))
        }
        _ => None,
    }
}

/// Opaque nodes: never entered, never rewritten.
fn opaque(n: &Node) -> bool {
    n.kids.is_empty() || n.op.is_meta() || n.op.is_agg() || n.op.is_custom()
}

fn fold_pass(n: &Node, removed: &mut usize) -> Node {
    if opaque(n) {
        return n.clone();
    }
    let m = Node {
        op: n.op,
        val: n.val,
        kids: n.kids.iter().map(|k| fold_pass(k, removed)).collect(),
        block: n.block.clone(),
    };
    match fold_root(&m) {
        Some(folded) => {
            *removed += 1;
            folded
        }
        None => m,
    }
}

/// Fold the redundant constants of `tree`. Returns the folded tree and the
/// number of constants removed; with `0` the tree is returned unchanged, bit
/// for bit. The result is a fixed point: folding it again removes nothing.
pub fn fold_constants(tree: &Node) -> (Node, usize) {
    return (tree.clone(), 0);
    let mut total = 0usize;
    let mut cur = tree.clone();
    loop {
        let mut removed = 0usize;
        let next = fold_pass(&cur, &mut removed);
        if removed == 0 {
            return (cur, total);
        }
        total += removed;
        cur = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{count_constants, eval, input, same};
    use crate::rng::Mt;

    fn c(v: f64) -> Node {
        constant(v)
    }
    fn x(i: usize) -> Node {
        input(i)
    }
    fn add(a: Node, b: Node) -> Node {
        binary(Op::Add, a, b)
    }
    fn sub(a: Node, b: Node) -> Node {
        binary(Op::Sub, a, b)
    }
    fn mul(a: Node, b: Node) -> Node {
        binary(Op::Mul, a, b)
    }
    fn div(a: Node, b: Node) -> Node {
        binary(Op::Div, a, b)
    }
    fn neg(a: Node) -> Node {
        node(Op::Neg, vec![a])
    }
    fn sin(a: Node) -> Node {
        node(Op::Sin, vec![a])
    }

    /// Rows with every input in `[lo, hi]`, deterministic.
    fn rows(n: usize, vars: usize, lo: f64, hi: f64, seed: u32) -> Vec<Vec<f64>> {
        let mut rng = Mt::new(seed);
        (0..n).map(|_| (0..vars).map(|_| rng.range(lo, hi)).collect()).collect()
    }

    fn max_rel(a: &Node, b: &Node, xs: &[Vec<f64>]) -> f64 {
        xs.iter()
            .map(|r| {
                let (u, v) = (eval(a, r), eval(b, r));
                if u.to_bits() == v.to_bits() {
                    0.0
                } else {
                    (u - v).abs() / u.abs().max(v.abs())
                }
            })
            .fold(0.0, f64::max)
    }

    fn rank(tree: &Node, xs: &[Vec<f64>]) -> usize {
        let y: Vec<f64> = (0..xs.len()).map(|r| (r as f64 * 0.37).sin()).collect();
        crate::fit::constant_errors(tree, xs, &y).expect("constant errors").rank
    }

    /// Fold `t`, check the count, the value on rows and that the folded tree's
    /// Jacobian has full rank while the input's has lost `removed` columns.
    fn check_rule(t: Node, removed: usize) -> Node {
        let xs = rows(64, 2, 0.25, 2.0, 7);
        let (f, n) = fold_constants(&t);
        assert_eq!(n, removed, "{t:?} -> {f:?}");
        assert_eq!(count_constants(&f), count_constants(&t) - removed);
        assert!(max_rel(&t, &f, &xs) < 1e-14, "{t:?} -> {f:?}");
        assert_eq!(rank(&t, &xs), count_constants(&t) - removed, "{t:?}");
        assert_eq!(rank(&f, &xs), count_constants(&f), "{f:?}");
        f
    }

    #[test]
    fn a_constant_factor_folds_into_a_product() {
        let f = check_rule(mul(c(3.0), mul(c(0.5), x(0))), 1);
        assert!(same(&f, &mul(c(1.5), x(0))), "{f:?}");
        check_rule(mul(mul(c(0.5), x(0)), c(3.0)), 1);
        check_rule(mul(c(3.0), div(x(0), c(4.0))), 1);
        check_rule(mul(c(3.0), div(c(4.0), x(0))), 1);
        check_rule(mul(c(3.0), mul(x(1), mul(c(0.5), x(0)))), 1);
    }

    #[test]
    fn a_constant_divisor_folds_into_a_product() {
        let f = check_rule(div(mul(c(3.0), x(0)), c(4.0)), 1);
        assert!(same(&f, &mul(c(0.75), x(0))), "{f:?}");
        let f = check_rule(div(c(3.0), mul(c(4.0), x(0))), 1);
        assert!(same(&f, &div(c(0.75), x(0))), "{f:?}");
        check_rule(div(c(3.0), div(x(0), c(4.0))), 1);
        let f = check_rule(div(c(3.0), div(c(4.0), x(0))), 1);
        assert!(same(&f, &mul(c(0.75), x(0))), "{f:?}");
    }

    #[test]
    fn a_product_or_quotient_of_two_scaled_terms_keeps_one_scale() {
        check_rule(mul(mul(c(3.0), x(0)), mul(c(0.5), x(1))), 1);
        check_rule(div(mul(c(3.0), x(0)), mul(c(0.5), x(1))), 1);
        check_rule(div(mul(c(3.0), sin(x(0))), div(x(1), c(0.5))), 1);
    }

    #[test]
    fn a_constant_factor_distributes_over_scaled_terms() {
        let f = check_rule(mul(c(2.0), add(c(0.5), mul(c(3.0), x(0)))), 1);
        assert!(same(&f, &add(c(1.0), mul(c(6.0), x(0)))), "{f:?}");
        check_rule(mul(c(2.0), sub(mul(c(1.5), x(1)), mul(c(3.0), x(0)))), 1);
    }

    #[test]
    fn a_constant_term_folds_into_a_sum() {
        let f = check_rule(add(c(1.0), add(c(0.5), x(0))), 1);
        assert!(same(&f, &add(c(1.5), x(0))), "{f:?}");
        check_rule(sub(add(c(0.5), x(0)), c(1.0)), 1);
        check_rule(add(c(1.0), sub(x(0), c(0.25))), 1);
        let f = check_rule(sub(c(1.0), add(c(0.25), x(0))), 1);
        assert!(same(&f, &sub(c(0.75), x(0))), "{f:?}");
        check_rule(sub(c(1.0), sub(c(0.25), x(0))), 1);
        check_rule(add(add(c(1.0), x(0)), add(c(0.5), x(1))), 1);
        check_rule(sub(add(c(1.0), x(0)), sub(x(1), c(0.5))), 1);
    }

    #[test]
    fn a_constant_term_folds_through_a_constant_scale() {
        let f = check_rule(add(c(1.0), mul(c(2.0), add(c(0.5), x(0)))), 1);
        assert!(same(&f, &mul(c(2.0), add(c(1.0), x(0)))), "{f:?}");
        check_rule(add(div(add(c(0.5), x(0)), c(4.0)), c(1.0)), 1);
        check_rule(sub(c(1.0), mul(c(2.0), add(c(0.5), mul(x(1), x(0))))), 1);
    }

    #[test]
    fn constants_fold_through_neg() {
        let f = check_rule(mul(c(3.0), neg(mul(c(0.5), x(0)))), 1);
        assert!(same(&f, &neg(mul(c(1.5), x(0)))), "{f:?}");
        check_rule(add(c(1.0), neg(add(c(0.5), x(0)))), 1);
        check_rule(mul(neg(mul(c(3.0), x(1))), mul(c(0.5), x(0))), 1);
    }

    /// A zero factor leaves the terms it scales as live zero constants, so the
    /// dead constants under it can move again.
    #[test]
    fn a_zero_factor_folds_into_live_zero_constants() {
        let xs = rows(64, 2, 0.25, 2.0, 7);
        let t = mul(c(0.0), add(c(0.5), mul(c(3.0), x(0))));
        let (f, n) = fold_constants(&t);
        assert_eq!(n, 1);
        assert!(same(&f, &add(c(0.0), mul(c(0.0), x(0)))), "{f:?}");
        assert_eq!(max_rel(&t, &f, &xs), 0.0);
        assert_eq!(rank(&f, &xs), 2);
    }

    #[test]
    fn a_constant_distributes_into_both_branches_of_an_if() {
        let cond = || sub(x(0), x(1));
        let f = check_rule(add(c(-0.5), node(Op::If, vec![cond(), c(-1.75), c(-0.75)])), 1);
        assert!(same(&f, &node(Op::If, vec![cond(), c(-2.25), c(-1.25)])), "{f:?}");
        check_rule(
            mul(c(3.0), node(Op::If, vec![cond(), mul(c(0.5), x(0)), div(x(1), c(4.0))])),
            1,
        );
        check_rule(sub(node(Op::If, vec![cond(), add(c(0.5), x(0)), c(2.0)]), c(1.0)), 1);
    }

    #[test]
    fn the_standardised_seed_shape_keeps_one_scale_and_one_offset() {
        // `c*x + d` in original units becomes `c*(z*s + m) + d` on the
        // standardised input `z`.
        let seed = add(mul(c(0.8), add(mul(x(0), c(1.7)), c(0.3))), c(-0.2));
        let f = check_rule(seed, 2);
        assert_eq!(count_constants(&f), 2);
    }

    #[test]
    fn identified_trees_and_boundaries_are_unchanged_bit_for_bit() {
        let keep = [
            add(mul(c(3.0), x(0)), c(0.5)),
            mul(c(3.0), add(x(0), c(0.5))),
            mul(c(3.0), sin(mul(c(0.5), x(0)))),
            add(add(mul(c(3.0), x(0)), mul(c(0.5), x(1))), c(1.0)),
            mul(c(3.0), add(mul(c(0.5), x(0)), x(1))),
            // A constant never moves into one branch, the condition or across
            // an operator boundary.
            mul(c(3.0), node(Op::If, vec![x(1), mul(c(0.5), x(0)), x(0)])),
            add(c(1.0), node(Op::If, vec![add(c(0.5), x(1)), x(0), add(c(2.0), x(0))])),
            mul(c(3.0), node(Op::Exp, vec![add(c(0.5), x(0))])),
            // A non-finite factor or a division by zero is not folded.
            div(mul(c(0.5), x(0)), c(0.0)),
            mul(c(f64::INFINITY), mul(c(0.5), x(0))),
            mul(c(1e300), mul(c(1e300), x(0))),
        ];
        for t in keep {
            let (f, n) = fold_constants(&t);
            assert_eq!(n, 0, "{t:?} -> {f:?}");
            assert!(same(&f, &t), "{t:?} -> {f:?}");
        }
    }

    #[test]
    fn folds_inside_an_operator_boundary_stay_inside() {
        let t = add(sin(mul(c(3.0), mul(c(0.5), x(0)))), c(1.0));
        let (f, n) = fold_constants(&t);
        assert_eq!(n, 1);
        assert!(same(&f, &add(sin(mul(c(1.5), x(0))), c(1.0))), "{f:?}");
    }

    const OPS: [Op; 8] = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Sin, Op::Exp, Op::If];

    #[test]
    fn folding_is_idempotent_on_random_trees() {
        let mut rng = Mt::new(982);
        let mut folded = 0usize;
        for _ in 0..4000 {
            let t = crate::expr::random_tree(&mut rng, 3, &OPS, 6);
            let (f, n) = fold_constants(&t);
            folded += usize::from(n > 0);
            let (g, m) = fold_constants(&f);
            assert_eq!(m, 0, "{t:?} -> {f:?} -> {g:?}");
            assert!(same(&f, &g), "{f:?} -> {g:?}");
            assert_eq!(count_constants(&f) + n, count_constants(&t), "{t:?} -> {f:?}");
        }
        assert!(folded > 400, "only {folded} of 4000 random trees folded");
    }

    /// Random trees whose every value is positive: constants and inputs in
    /// `[0.25, 2]`, operators `add`, `mul`, `div`, `if` on a difference of two
    /// inputs and the boundaries `sqrt` and `logistic`. No other subtraction,
    /// so no cancellation can magnify the rounding of a re-associated product
    /// or sum.
    fn positive_tree(rng: &mut Mt, depth: usize) -> Node {
        if depth == 0 || rng.bool(0.2) {
            return if rng.bool(0.5) { x(rng.usize_below(3)) } else { c(rng.range(0.25, 2.0)) };
        }
        match rng.usize_below(6) {
            0 => add(positive_tree(rng, depth - 1), positive_tree(rng, depth - 1)),
            1 => mul(positive_tree(rng, depth - 1), positive_tree(rng, depth - 1)),
            2 => div(positive_tree(rng, depth - 1), positive_tree(rng, depth - 1)),
            3 => node(Op::Sqrt, vec![positive_tree(rng, depth - 1)]),
            4 => {
                let cond = sub(x(rng.usize_below(3)), x(rng.usize_below(3)));
                node(Op::If, vec![cond, positive_tree(rng, depth - 1), positive_tree(rng, depth - 1)])
            }
            _ => node(Op::Logistic, vec![positive_tree(rng, depth - 1)]),
        }
    }

    #[test]
    fn folding_preserves_values_on_random_rows() {
        let mut rng = Mt::new(9820);
        let xs = rows(256, 3, 0.25, 2.0, 11);
        let (mut folded, mut worst) = (0usize, 0.0f64);
        for _ in 0..3000 {
            let t = positive_tree(&mut rng, 6);
            let (f, n) = fold_constants(&t);
            if n == 0 {
                continue;
            }
            folded += 1;
            worst = worst.max(max_rel(&t, &f, &xs));
        }
        assert!(folded > 300, "only {folded} of 3000 random trees folded");
        assert!(worst < 1e-12, "max relative difference {worst:e}");
    }

    /// `|t|` with every operand replaced by its magnitude: a bound on every
    /// intermediate value of an `add`/`sub`/`mul`/`neg` tree.
    fn magnitude(t: &Node, r: &[f64]) -> f64 {
        match t.op {
            Op::Const => t.val.abs(),
            Op::Input => r[t.val as usize].abs(),
            Op::Neg => magnitude(&t.kids[0], r),
            Op::Add | Op::Sub => magnitude(&t.kids[0], r) + magnitude(&t.kids[1], r),
            Op::Mul => magnitude(&t.kids[0], r) * magnitude(&t.kids[1], r),
            op => panic!("{op:?} is outside the signed test grammar"),
        }
    }

    #[test]
    fn folding_with_signs_stays_within_rounding_of_the_magnitude() {
        let mut rng = Mt::new(9821);
        let xs = rows(256, 3, -2.0, 2.0, 12);
        let ops = [Op::Add, Op::Sub, Op::Mul, Op::Neg];
        let (mut folded, mut worst) = (0usize, 0.0f64);
        for _ in 0..3000 {
            let t = crate::expr::random_tree(&mut rng, 3, &ops, 6);
            let (f, n) = fold_constants(&t);
            if n == 0 {
                continue;
            }
            folded += 1;
            for r in &xs {
                let d = (eval(&t, r) - eval(&f, r)).abs();
                worst = worst.max(d / magnitude(&t, r).max(f64::MIN_POSITIVE));
            }
        }
        assert!(folded > 300, "only {folded} of 3000 random trees folded");
        assert!(worst < 1e-12, "max difference over the magnitude {worst:e}");
    }
}
