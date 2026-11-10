//! #997: `expr::eval` and the fit's flat evaluator give the same finiteness
//! and, where finite, the same value.

use super::*;
use crate::expr::{constant, input, node};
use crate::rng::Mt;

/// The flat evaluator on one row at the tree's own constants: every node's
/// value in flat order, with its operator.
pub(crate) fn flat_trace(tree: &Node, row: &[f64]) -> (Vec<(Op, f64)>, f64) {
    let f = flatten(tree);
    let c = constants_slots(tree);
    let mut v = vec![0.0; f.op.len()];
    eval_row(&f, row, &c, &mut v);
    (f.op.iter().copied().zip(v.iter().copied()).collect(), v[f.root])
}

fn flat_value(tree: &Node, row: &[f64]) -> f64 {
    flat_trace(tree, row).1
}

fn agree(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn pow_of(a: f64, b: f64) -> Node {
    node(Op::Pow, vec![constant(a), constant(b)])
}

/// The documented `pow` table (docs/operator-semantics.md), on both evaluators.
#[test]
fn pow_semantics_match_the_documented_table() {
    let inf = f64::INFINITY;
    let nan = f64::NAN;
    let table: &[(f64, f64, f64)] = &[
        (-8.0, 3.0, -512.0),
        (-8.0, 2.0, 64.0),
        (-8.0, -3.0, -1.0 / 512.0),
        (-8.0, 0.0, 1.0),
        (-8.0, 1.0 / 3.0, nan),
        (-8.0, 0.5, nan),
        (-8.0, -2.5, nan),
        (-2.0, 3.0 + 5e-10, nan),
        (-2.0, 3.0 + 2e-9, nan),
        (-inf, 0.5, nan),
        (-inf, 3.0, -inf),
        (-inf, 2.0, inf),
        (0.0, 0.0, 1.0),
        (-0.0, 0.0, 1.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -1.0, inf),
        (0.0, -0.5, inf),
        (-0.0, -1.0, -inf),
        (0.278, 0.5, 0.278f64.sqrt()),
        (10.0, 400.0, inf),
        (-10.0, 401.0, -inf),
        (-10.0, 400.0, inf),
        (1e-3, 400.0, 0.0),
        (nan, 0.0, 1.0),
        (1.0, nan, 1.0),
        (nan, nan, nan),
        (2.0, inf, inf),
        (0.5, inf, 0.0),
        (2.0, -inf, 0.0),
    ];
    for &(a, b, want) in table {
        let t = pow_of(a, b);
        let ev = expr::eval(&t, &[]);
        let fl = flat_value(&t, &[]);
        assert!(agree(ev, want), "eval {a:e} ^ {b:e} = {ev:e}, want {want:e}");
        assert!(agree(fl, want), "flat {a:e} ^ {b:e} = {fl:e}, want {want:e}");
    }
}

/// The documented `ln` table on both evaluators.
#[test]
fn ln_semantics_match_the_documented_table() {
    let table: &[(f64, f64)] = &[
        (1.0, 0.0),
        (std::f64::consts::E, 1.0),
        (f64::INFINITY, f64::INFINITY),
        (0.0, f64::NAN),
        (-0.0, f64::NAN),
        (-1.0, f64::NAN),
        (f64::NEG_INFINITY, f64::NAN),
        (f64::NAN, f64::NAN),
        (1e-310, (1e-310f64).ln()),
    ];
    for &(a, want) in table {
        let t = node(Op::Ln, vec![constant(a)]);
        let ev = expr::eval(&t, &[]);
        let fl = flat_value(&t, &[]);
        assert!(agree(ev, want), "eval ln({a:e}) = {ev:e}, want {want:e}");
        assert!(agree(fl, want), "flat ln({a:e}) = {fl:e}, want {want:e}");
    }
}

const UNARY: [Op; 28] = [
    Op::Neg, Op::Exp, Op::Ln, Op::Abs, Op::Sqrt, Op::Sin, Op::Cos, Op::Tan, Op::Sinh, Op::Cosh,
    Op::Tanh, Op::Logistic, Op::Step, Op::Gauss, Op::Square, Op::Cube, Op::Floor, Op::Ceil,
    Op::Round, Op::Sign, Op::Erf, Op::Erfc, Op::Asinh, Op::Acosh, Op::Atanh, Op::Not,
    Op::Require, Op::Constrain,
];
const BINARY: [Op; 17] = [
    Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Pow, Op::Mod, Op::Min, Op::Max, Op::Atan2,
    Op::Equal, Op::Less, Op::LessOrEqual, Op::Greater, Op::GreaterOrEqual, Op::And, Op::Or,
    Op::Xor,
];

const SPECIAL: [f64; 22] = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, 0.278, -2.5, 1e-310, -1e-310, 1e-13, 700.5, -701.0,
    800.0, 1e300, -1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 3.0000000005,
];

fn pick(rng: &mut Mt, n: usize) -> usize {
    (rng.next_u32() as usize) % n
}

/// A constant or input value: half special values, half log-uniform over
/// 1e-20..1e20 with a random sign.
fn value(rng: &mut Mt) -> f64 {
    if pick(rng, 2) == 0 {
        SPECIAL[pick(rng, SPECIAL.len())]
    } else {
        let mag = 10f64.powf(rng.next_f64() * 40.0 - 20.0);
        if pick(rng, 2) == 0 {
            mag
        } else {
            -mag
        }
    }
}

fn random(rng: &mut Mt, depth: usize) -> Node {
    if depth == 0 || pick(rng, 7) == 0 {
        return if pick(rng, 2) == 0 { constant(value(rng)) } else { input(pick(rng, 2)) };
    }
    match pick(rng, 8) {
        0 | 1 => node(UNARY[pick(rng, UNARY.len())], vec![random(rng, depth - 1)]),
        2..=5 => node(
            BINARY[pick(rng, BINARY.len())],
            vec![random(rng, depth - 1), random(rng, depth - 1)],
        ),
        _ => node(
            if pick(rng, 2) == 0 { Op::If } else { Op::Between },
            vec![random(rng, depth - 1), random(rng, depth - 1), random(rng, depth - 1)],
        ),
    }
}

/// 20000 seeded trees of depth up to 5, each on 6 sampled rows.
#[test]
fn evaluators_agree_on_finiteness_for_random_trees() {
    let mut rng = Mt::new(997);
    let mut finite = 0usize;
    let mut points = 0usize;
    for case in 0..20_000 {
        let tree = random(&mut rng, 5);
        for _ in 0..6 {
            let row = [value(&mut rng), value(&mut rng)];
            let ev = expr::eval(&tree, &row);
            let fl = flat_value(&tree, &row);
            points += 1;
            assert_eq!(ev.is_finite(), fl.is_finite(), "case {case} row {row:?} eval {ev:e} flat {fl:e} {tree:?}");
            if ev.is_finite() {
                finite += 1;
                assert!(agree(ev, fl), "case {case} row {row:?} eval {ev:e} flat {fl:e} {tree:?}");
            } else {
                assert!(agree(ev, fl) || (!ev.is_nan() && !fl.is_nan()), "case {case} row {row:?} eval {ev:e} flat {fl:e} {tree:?}");
            }
        }
    }
    assert!(finite > points / 10, "only {finite} of {points} points were finite");
}
