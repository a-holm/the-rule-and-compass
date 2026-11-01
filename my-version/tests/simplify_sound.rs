//! Differential and property tests for issue #143.
//!
//! The simplifier is a rewrite on a partial algebra: `eval` returns `NaN` at
//! failed guards and propagates non-finite values. These tests generate random
//! trees over `NaN`, `±inf`, `±0`, subnormals and guard boundaries, then check
//! that the sound normal form preserves the value bit for bit (NaNs compare
//! equal to NaNs) and preserves undefinedness.

use rcsr::expr::{self, constant, eval, input, node, structural_hash, Node, Op};
use rcsr::fit;
use rcsr::rng::Mt;
use std::time::Instant;

const EDGE: [f64; 18] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    1e-320,
    -1e-320,
    1e-13,
    -1e-13,
    1e-12,
    -1e-12,
    1e308,
    -1e308,
    2.0,
];

fn same_value(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn finite_rows(rng: &mut Mt, n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| vec![rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)])
        .collect()
}

fn edge_rows() -> Vec<Vec<f64>> {
    let mut rows = Vec::new();
    for &a in &EDGE {
        for b in [0.0, 1.0, -0.0, f64::NAN, f64::INFINITY, 1e-13] {
            rows.push(vec![a, b]);
        }
    }
    rows.push(vec![1e-12, 1e-12]);
    rows.push(vec![1e-12, -1e-12]);
    rows
}

fn gen_tree(rng: &mut Mt, depth: usize, nvars: usize) -> Node {
    use Op::*;
    if depth == 0 || rng.bool(0.22) {
        return match rng.usize_below(8) {
            0 | 1 => input(rng.usize_below(nvars)),
            2 => constant(EDGE[rng.usize_below(EDGE.len())]),
            3 => constant(rng.range(-2.0, 2.0)),
            4 => constant(rng.range(-1.0, 1.0)),
            5 => constant(if rng.bool(0.5) { 0.0 } else { -0.0 }),
            6 => constant(1.0 + (rng.range(-2.0, 2.0)) * f64::EPSILON),
            _ => constant(rng.range(-2.0, 2.0)),
        };
    }
    let ops = [
        Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Step, Gauss, Square,
        Cube, Floor, Ceil, Round, Sign, Erf, Erfc, Asinh, Acosh, Atanh, Add, Sub, Mul, Div, Pow,
        Mod, Min, Max, Atan2, If, Equal, Less, LessOrEqual, Greater, GreaterOrEqual, And, Or, Xor,
        Not, Require, Between, Constrain,
    ];
    let op = *rng.pick(&ops);
    if op == If {
        return node(
            If,
            vec![
                gen_tree(rng, depth - 1, nvars),
                gen_tree(rng, depth - 1, nvars),
                gen_tree(rng, depth - 1, nvars),
            ],
        );
    }
    if op == Between {
        return node(
            Between,
            vec![
                gen_tree(rng, depth - 1, nvars),
                gen_tree(rng, depth - 1, nvars),
                gen_tree(rng, depth - 1, nvars),
            ],
        );
    }
    let kids = (0..op.arity())
        .map(|_| gen_tree(rng, depth - 1, nvars))
        .collect();
    node(op, kids)
}

#[test]
fn differential_simplify_preserves_values_on_edge_inputs() {
    let mut rng = Mt::new(0x0143_ED6E);
    let rows = edge_rows();
    let mut checked = 0usize;
    for _ in 0..3000 {
        let tree = gen_tree(&mut rng, 4, 2);
        let simple = expr::simplify(&tree);
        // Idempotence and a deterministic normal form.
        let again = expr::simplify(&simple);
        assert_eq!(structural_hash(&simple), structural_hash(&again));
        for row in &rows {
            let before = eval(&tree, row);
            let after = eval(&simple, row);
            assert!(
                same_value(before, after),
                "{} at {:?}: {} -> {}",
                expr::to_string(&tree, &["x".to_string(), "y".to_string()]),
                row,
                before,
                after
            );
            checked += 1;
        }
    }
    assert!(checked > 100_000, "checked {checked} tree/row pairs");
}

#[test]
fn differential_simplify_preserves_finite_values_and_undefinedness() {
    let mut rng = Mt::new(0x0143_F17E);
    for _ in 0..10_000 {
        let nvars = 1 + rng.usize_below(2);
        let tree = gen_tree(&mut rng, 4, nvars);
        let simple = expr::simplify(&tree);
        let rows = finite_rows(&mut rng, 8);
        for row in &rows {
            let before = eval(&tree, &row[..nvars]);
            let after = eval(&simple, &row[..nvars]);
            assert!(
                same_value(before, after),
                "{} at {:?}: {} -> {}",
                expr::to_string(&tree, &["x".to_string(), "y".to_string()]),
                row,
                before,
                after
            );
        }
    }
}

#[test]
fn abstract_analysis_is_sound_and_precise_on_10k_trees() {
    let mut rng = Mt::new(0x0143_A9A1);
    let bounds = [(-2.0, 2.0), (-2.0, 2.0)];
    let rows = finite_rows(&mut rng, 8);
    let mut total_trees = 0usize;
    let mut proved_total = 0usize;
    let mut soundness_failures = 0usize;
    for _ in 0..10_000 {
        let tree = gen_tree(&mut rng, 4, 2);
        let proved = expr::analyze_total(&tree, Some(&bounds));
        let actually_total = rows
            .iter()
            .all(|row| eval(&tree, row).is_finite());
        if proved && !actually_total {
            soundness_failures += 1;
        }
        if actually_total {
            total_trees += 1;
            if proved {
                proved_total += 1;
            }
        }
    }
    assert_eq!(
        soundness_failures, 0,
        "analysis marked a non-finite tree total"
    );
    assert!(total_trees > 0);
    let precision = proved_total as f64 / total_trees as f64;
    assert!(
        precision >= 0.80,
        "precision {:.3} on {total_trees} total trees (target >= 0.80)",
        precision
    );
}

#[test]
fn simplification_cost_is_well_under_two_percent_of_scoring() {
    let mut rng = Mt::new(0x0143_C057);
    let trees: Vec<Node> = (0..300).map(|_| gen_tree(&mut rng, 4, 2)).collect();
    let rows: Vec<Vec<f64>> = (0..4096)
        .map(|_| vec![rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)])
        .collect();

    let simplify_run = || {
        let t = Instant::now();
        let mut sum = 0usize;
        for tree in &trees {
            sum = sum.wrapping_add(expr::complexity(&expr::simplify(tree)));
        }
        (t.elapsed(), sum)
    };
    // The previous aggressive relation is still available as `simplify_display`,
    // so its cost is the "before" number on the same corpus.
    let display_run = || {
        let t = Instant::now();
        let mut sum = 0usize;
        for tree in &trees {
            sum = sum.wrapping_add(expr::complexity(&expr::simplify_display(tree)));
        }
        (t.elapsed(), sum)
    };
    let score_run = || {
        let t = Instant::now();
        let mut sum = 0.0f64;
        for tree in &trees {
            for row in &rows {
                sum += eval(tree, row);
            }
        }
        (t.elapsed(), sum)
    };

    let (mut best_simplify, mut sink) = simplify_run();
    let (mut best_display, _) = display_run();
    let (mut best_score, _) = score_run();
    for _ in 0..2 {
        let (s, v) = simplify_run();
        best_simplify = best_simplify.min(s);
        sink = sink.wrapping_add(v);
        let (s, _) = display_run();
        best_display = best_display.min(s);
        let (s, _) = score_run();
        best_score = best_score.min(s);
    }
    assert!(sink > 0);
    let ratio = best_simplify.as_secs_f64() / best_score.as_secs_f64();
    eprintln!(
        "sound simplify {:.3} ms, previous display relation {:.3} ms, scoring {:.3} ms \
         over 300 trees x 4096 rows: after {:.3}% of scoring, before {:.3}%, change {:+.1}%",
        best_simplify.as_secs_f64() * 1e3,
        best_display.as_secs_f64() * 1e3,
        best_score.as_secs_f64() * 1e3,
        ratio * 100.0,
        best_display.as_secs_f64() / best_score.as_secs_f64() * 100.0,
        (best_simplify.as_secs_f64() / best_display.as_secs_f64() - 1.0) * 100.0
    );
    assert!(
        ratio < 0.02,
        "simplification cost {:.3}% of scoring (target < 2%)",
        ratio * 100.0
    );
}

#[test]
fn sound_simplification_leaves_fitted_constants_and_predictions_unchanged() {
    let rng = &mut Mt::new(0x0143_F17E);
    let xs: Vec<Vec<f64>> = (0..40)
        .map(|_| vec![rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)])
        .collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|r| 3.0 * r[0].sin() + 0.5 * r[1] + 2.0)
        .collect();
    // The two trees are structurally different but canonically equal: the
    // model, its parameter slots and therefore its fitted constants are the
    // same. Both must fit to the same predictions.
    let c = || constant(1.0);
    let a = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![c(), node(Op::Sin, vec![input(0)])]),
            node(Op::Mul, vec![c(), input(1)]),
        ],
    );
    let b = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![c(), input(1)]),
            node(Op::Mul, vec![node(Op::Sin, vec![input(0)]), c()]),
        ],
    );
    let mut fitted_a = a.clone();
    let mut fitted_b = b.clone();
    let sse_a = fit::fit(&mut fitted_a, &xs, &ys, 25);
    let sse_b = fit::fit(&mut fitted_b, &xs, &ys, 25);
    assert!(sse_a.is_finite() && sse_b.is_finite());
    assert!(
        (sse_a - sse_b).abs() <= 1e-9 * (1.0 + sse_a.abs()),
        "canonical forms fitted to different SSE: {} vs {}",
        sse_a,
        sse_b
    );
    assert_eq!(
        structural_hash(&expr::simplify(&a)),
        structural_hash(&expr::simplify(&b)),
        "the canonical forms must agree before fitting"
    );
    for x in &xs {
        let pa = eval(&fitted_a, x);
        let pb = eval(&fitted_b, x);
        assert!(
            (pa - pb).abs() <= 1e-9 * (1.0 + pa.abs()),
            "fitted predictions changed at {:?}: {} vs {}",
            x,
            pa,
            pb
        );
    }
    // A safe rewrite does not change the prediction of the same fitted tree.
    let simplified = expr::simplify(&fitted_a);
    for x in &xs {
        assert!(same_value(eval(&fitted_a, x), eval(&simplified, x)));
    }
}

#[test]
fn dedup_keys_are_canonical_and_value_preserving() {
    let x = input(0);
    let y = input(1);
    let forms = [
        node(Op::Add, vec![x.clone(), y.clone()]),
        node(Op::Add, vec![y.clone(), x.clone()]),
        node(Op::Add, vec![x.clone(), node(Op::Add, vec![y.clone(), constant(-0.0)])]),
    ];
    let keys: Vec<u64> = forms
        .iter()
        .map(|t| structural_hash(&expr::simplify(t)))
        .collect();
    assert_eq!(keys[0], keys[1], "Add(x,y) and Add(y,x) must share a key");
    assert_eq!(keys[1], keys[2], "Add(x,y) and Add(x,y+-0) must share a key");

    // The sound key never merges unsafe forms: x/x is not 1, x-x is not 0 and
    // 0*x is not 0 without proofs.
    assert_ne!(
        structural_hash(&expr::simplify(&node(Op::Div, vec![x.clone(), x.clone()]))),
        structural_hash(&expr::simplify(&constant(1.0)))
    );
    assert_ne!(
        structural_hash(&expr::simplify(&node(Op::Sub, vec![x.clone(), x.clone()]))),
        structural_hash(&expr::simplify(&constant(0.0)))
    );
    assert_ne!(
        structural_hash(&expr::simplify(&node(Op::Mul, vec![constant(0.0), x]))),
        structural_hash(&expr::simplify(&constant(0.0)))
    );
}
/// #466: `if(c, a, a)` is `a` for every condition, including a NaN or
/// infinite one, and for a branch that is itself NaN or infinite.
#[test]
fn simplify_if_equal_branches() {
    let mut rng = Mt::new(0x0466_1F00);
    let mut rows = edge_rows();
    while rows.len() < 10_000 {
        rows.push(vec![rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)]);
    }
    let vars = ["x".to_string(), "y".to_string()];
    let mut removed = 0usize;
    for _ in 0..200 {
        let cond = gen_tree(&mut rng, 3, 2);
        let branch = gen_tree(&mut rng, 3, 2);
        let tree = node(Op::If, vec![cond, branch.clone(), branch.clone()]);
        let simple = expr::simplify_equal_branches(&tree);
        assert_eq!(
            structural_hash(&simple),
            structural_hash(&expr::simplify(&branch)),
            "{}",
            expr::to_string(&tree, &vars)
        );
        removed += 1;
        for row in &rows {
            let (before, after) = (eval(&tree, row), eval(&simple, row));
            assert!(
                same_value(before, after),
                "{} at {:?}: {} -> {}",
                expr::to_string(&tree, &vars),
                row,
                before,
                after
            );
        }
    }
    assert_eq!(removed, 200);
}
