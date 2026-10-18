//! Agreement and edge-case tests for the structure-of-arrays evaluator.
//!
//! Stage 2 of issue #27 wires the evaluator into the search; these tests pin
//! the kernel itself against [`rcsr::expr::eval`], the reference scalar walk
//! that `--no-simd` still uses. [`rcsr::eval_soa::columns`] is the transpose the
//! search pays once per candidate.

use rcsr::eval_soa::{self, compile};
use rcsr::expr::{self, Node, Op};
use rcsr::rng::Mt;

/// Supported trees are expected to agree to machine precision: the SoA paths
/// use the same `f64` operations and the same domain guards as `expr::eval`.
/// This tolerance still leaves room for a future vectorised transcendental that
/// rounds differently.
const REL_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    if a == b {
        return true; // also covers ±inf and ±0
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= REL_TOL * (1.0 + a.abs().max(b.abs()))
}

/// Deterministic data with negatives, exact zeros (so the `ln`/`div`/`sqrt`
/// guards are exercised), and values large enough to overflow `exp`.
fn sample(rows: usize, nvars: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut cols = vec![vec![0.0; rows]; nvars];
    for (v, col) in cols.iter_mut().enumerate() {
        for (r, cell) in col.iter_mut().enumerate() {
            *cell = if r % 11 == 0 {
                0.0
            } else {
                ((r as f64 * 0.37 + v as f64 * 1.7).sin()) * 3.0
            };
        }
    }
    let mut aos = vec![vec![0.0; nvars]; rows];
    for (r, row) in aos.iter_mut().enumerate() {
        for (v, cell) in row.iter_mut().enumerate() {
            *cell = cols[v][r];
        }
    }
    (cols, aos)
}

fn soa(tree: &Node, cols: &[Vec<f64>], rows: usize) -> Vec<f64> {
    let program = compile(tree).expect("tree should be supported");
    let mut out = vec![0.0; rows];
    eval_soa::eval_batch(&program, cols, rows, &mut out);
    out
}

fn assert_agrees(tree: &Node, rows: usize) {
    let program = compile(tree).expect("a supported tree must compile");
    let (cols, aos) = sample(rows, 4);
    let mut scalar = vec![0.0; rows];
    let mut simd = vec![0.0; rows];
    eval_soa::eval_batch_scalar(&program, &cols, rows, &mut scalar);
    eval_soa::eval_batch(&program, &cols, rows, &mut simd);

    for r in 0..rows {
        let want = expr::eval(tree, &aos[r]);
        assert!(
            close(want, scalar[r]),
            "SoA scalar row {r}: expr::eval = {want:?}, evaluator = {:?}",
            scalar[r]
        );
        assert!(
            close(want, simd[r]),
            "SoA AVX row {r}: expr::eval = {want:?}, evaluator = {:?}",
            simd[r]
        );
        assert!(
            close(scalar[r], simd[r]),
            "AVX vs scalar row {r}: {:?} vs {:?}",
            scalar[r],
            simd[r]
        );
    }
}

/// The operator set the evaluator claims to support, one representative tree
/// per operator.
fn supported_ops() -> Vec<Op> {
    use Op::*;
    vec![
        Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Step, Gauss, Square,
        Cube, Floor, Ceil, Round, Sign, Add, Sub, Mul, Div, Pow, Min, Max, If,
    ]
}

fn op_tree(op: Op) -> Node {
    let kids = (0..op.arity())
        .map(|i| if i == 0 { expr::input(0) } else { expr::constant(1.0) })
        .collect();
    expr::node(op, kids)
}

#[test]
fn supports_exactly_the_documented_set() {
    for op in supported_ops() {
        assert!(eval_soa::supports(op), "{op:?} should be supported");
        assert!(compile(&op_tree(op)).is_some(), "{op:?} should compile");
    }
    assert!(eval_soa::supports(Op::Const));
    assert!(eval_soa::supports(Op::Input));
}

#[test]
fn unsupported_operators_return_none() {
    use Op::*;
    let unsupported = [
        Mod, Atan2, Erf, Erfc, Asinh, Acosh, Atanh, Equal, Less, LessOrEqual, Greater,
        GreaterOrEqual, And, Or, Xor, Not, Require, Between, Constrain, Complexity, Variables,
        Constants, Contains, ContainsOp,
    ];
    for op in unsupported {
        assert!(!eval_soa::supports(op), "{op:?} must not be supported");
        assert!(
            compile(&op_tree(op)).is_none(),
            "{op:?} must return None, not a wrong number"
        );
    }

    // An unsupported leaf deep in an otherwise supported tree poisons the whole
    // compilation instead of silently dropping it.
    let nested = expr::node(
        Op::Add,
        vec![expr::input(0), expr::node(Op::Erf, vec![expr::input(1)])],
    );
    assert!(compile(&nested).is_none());
}

#[test]
fn generated_corpus_agrees_with_expr_eval() {
    let ops = supported_ops();
    let mut rng = Mt::new(27_027);
    let mut trees = Vec::new();
    // A spread of depths gives leaves, unary chains, arithmetic and `if`.
    for &depth in &[1usize, 2, 3, 4] {
        for _ in 0..12 {
            trees.push(expr::random_tree(&mut rng, 4, &ops, depth));
        }
    }
    // Hand-built cases that a random draw may miss.
    trees.push(expr::node(
        Op::If,
        vec![
            expr::input(0),
            expr::node(Op::Ln, vec![expr::input(1)]),
            expr::constant(-3.5),
        ],
    ));
    trees.push(expr::node(
        Op::Div,
        vec![
            expr::constant(1.0),
            expr::node(Op::Add, vec![expr::input(0), expr::input(1)]),
        ],
    ));
    trees.push(expr::node(
        Op::Pow,
        vec![expr::input(2), expr::constant(0.5)],
    ));

    for tree in &trees {
        assert_agrees(tree, 67);
    }
}

#[test]
fn domain_guards_match_expr_eval() {
    // ln of a non-positive value, at -1, 0 and 1.
    let ln = expr::node(Op::Ln, vec![expr::input(0)]);
    let out = soa(&ln, &[vec![-1.0, 0.0, 1.0]], 3);
    assert!(out[0].is_nan());
    assert!(out[1].is_nan());
    assert_eq!(out[2], 0.0);

    // Division by zero (and near zero) is NaN, not infinity.
    let div = expr::node(Op::Div, vec![expr::constant(1.0), expr::input(0)]);
    let out = soa(&div, &[vec![-2.0, 0.0, 1e-13, 2.0]], 4);
    assert_eq!(out[0], -0.5);
    assert!(out[1].is_nan());
    assert!(out[2].is_nan());
    assert_eq!(out[3], 0.5);

    // sqrt of a negative value is NaN.
    let sqrt = expr::node(Op::Sqrt, vec![expr::input(0)]);
    let out = soa(&sqrt, &[vec![-1.0, 0.0, 4.0]], 3);
    assert!(out[0].is_nan());
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 2.0);

    // `if` is `cond > 0`, so zero takes the else branch.
    let if_tree = expr::node(
        Op::If,
        vec![
            expr::input(0),
            expr::constant(10.0),
            expr::constant(-10.0),
        ],
    );
    let out = soa(&if_tree, &[vec![-1.0, 0.0, 1.0]], 3);
    assert_eq!(out, [-10.0, -10.0, 10.0]);

    // The exp guard: exactly 700 is finite, 701 is +inf.
    let exp = expr::node(Op::Exp, vec![expr::input(0)]);
    let out = soa(&exp, &[vec![0.0, 700.0, 701.0]], 3);
    assert_eq!(out[0], 1.0);
    assert!(out[1].is_finite());
    assert_eq!(out[2], f64::INFINITY);

    // pow(negative, fractional) is NaN; pow(negative, integer) is fine.
    let pow = expr::node(Op::Pow, vec![expr::input(0), expr::constant(0.5)]);
    let out = soa(&pow, &[vec![-1.0, 4.0]], 2);
    assert!(out[0].is_nan());
    assert_eq!(out[1], 2.0);

    // min/max propagate NaN rather than ignoring it.
    let nan_col = vec![vec![f64::NAN, 2.0]];
    for op in [Op::Min, Op::Max] {
        let tree = expr::node(op, vec![expr::input(0), expr::constant(1.0)]);
        let out = soa(&tree, &nan_col, 2);
        assert!(out[0].is_nan(), "{op:?} must propagate NaN");
    }
}

#[test]
fn every_supported_operator_matches_expr_eval() {
    let (cols, aos) = sample(41, 4);
    for op in supported_ops() {
        let tree = match op.arity() {
            1 => expr::node(op, vec![expr::input(0)]),
            2 => expr::node(op, vec![expr::input(0), expr::input(1)]),
            3 => expr::node(op, vec![expr::input(0), expr::input(1), expr::input(2)]),
            _ => unreachable!("supported operators have arity 1..=3, got {op:?}"),
        };
        let program = compile(&tree).expect("supported operator must compile");
        let mut simd = vec![0.0; 41];
        let mut scalar = vec![0.0; 41];
        eval_soa::eval_batch(&program, &cols, 41, &mut simd);
        eval_soa::eval_batch_scalar(&program, &cols, 41, &mut scalar);
        for r in 0..41 {
            let want = expr::eval(&tree, &aos[r]);
            assert!(
                close(want, simd[r]),
                "{op:?} AVX row {r}: expr::eval = {want:?}, evaluator = {:?}",
                simd[r]
            );
            assert!(
                close(want, scalar[r]),
                "{op:?} scalar row {r}: expr::eval = {want:?}, evaluator = {:?}",
                scalar[r]
            );
        }
    }
}

#[test]
fn all_row_counts_including_tails_agree() {
    // Covers 0..4, 4..8 and a tail of one row when the SIMD width is four.
    let ops = supported_ops();
    let mut rng = Mt::new(0x51A);
    let tree = expr::random_tree(&mut rng, 3, &ops, 3);
    for rows in [1usize, 2, 3, 4, 5, 7, 8, 101] {
        assert_agrees(&tree, rows);
    }
}

#[test]
fn empty_input_and_zero_rows_do_not_panic() {
    // Zero rows with an empty output buffer.
    let p = compile(&expr::constant(2.5)).unwrap();
    let mut none: [f64; 0] = [];
    eval_soa::eval_batch(&p, &[], 0, &mut none);
    eval_soa::eval_batch_scalar(&p, &[], 0, &mut none);

    // A constant does not need any input columns.
    let mut out = [0.0; 3];
    eval_soa::eval_batch(&p, &[], 3, &mut out);
    assert_eq!(out, [2.5; 3]);

    // An input with no columns is NaN, matching `expr::eval` on an empty row.
    let p = compile(&expr::input(3)).unwrap();
    let mut out = [0.0; 2];
    eval_soa::eval_batch(&p, &[], 2, &mut out);
    assert!(out.iter().all(|v| v.is_nan()));

    // A short column is padded with NaN for the missing rows.
    let p = compile(&expr::input(0)).unwrap();
    let mut out = [0.0; 5];
    eval_soa::eval_batch(&p, &[vec![1.0]], 5, &mut out);
    assert_eq!(out[0], 1.0);
    assert!(out[1..].iter().all(|v| v.is_nan()));
}

#[test]
fn columns_transpose_row_major_inputs() {
    let rows = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        // A short row is padded with NaN, like `expr::eval`'s Input rule.
        vec![7.0],
    ];
    let cols = eval_soa::columns(&rows);
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0], [1.0, 4.0, 7.0]);
    assert_eq!(&cols[1][..2], [2.0, 5.0]);
    assert_eq!(&cols[2][..2], [3.0, 6.0]);
    assert!(cols[1][2].is_nan());
    assert!(cols[2][2].is_nan());

    // Empty input has no columns and is still a valid (zero-width) dataset.
    assert!(eval_soa::columns(&[]).is_empty());

    // A tree evaluated over the transposed columns agrees row for row.
    let tree = expr::node(
        Op::Add,
        vec![
            expr::node(Op::Mul, vec![expr::input(0), expr::constant(2.0)]),
            expr::input(1),
        ],
    );
    let program = compile(&tree).unwrap();
    let full = vec![
        vec![1.0, 2.0],
        vec![4.0, 5.0],
        vec![7.0, 8.0],
    ];
    let cols = eval_soa::columns(&full);
    let mut out = vec![0.0; 3];
    eval_soa::eval_batch(&program, &cols, 3, &mut out);
    for (r, row) in full.iter().enumerate() {
        assert_eq!(out[r], expr::eval(&tree, row));
    }
}

/// On a corpus of supported trees, both SoA paths are not merely close to the
/// scalar walk but bit-identical (NaN payloads excepted). This is what lets the
/// search switch paths without moving a single archive entry.
#[test]
fn supported_corpus_is_bit_identical_to_expr_eval() {
    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    let ops = supported_ops();
    let mut rng = Mt::new(27_002);
    let (cols, aos) = sample(73, 4);
    for &depth in &[1usize, 2, 3, 4] {
        for _ in 0..8 {
            let tree = expr::random_tree(&mut rng, 4, &ops, depth);
            let program = match compile(&tree) {
                Some(p) => p,
                None => panic!("a supported tree must compile: {tree:?}"),
            };
            let mut scalar = vec![0.0; 73];
            let mut simd = vec![0.0; 73];
            eval_soa::eval_batch_scalar(&program, &cols, 73, &mut scalar);
            eval_soa::eval_batch(&program, &cols, 73, &mut simd);
            for r in 0..73 {
                let want = expr::eval(&tree, &aos[r]);
                assert!(
                    same(want, scalar[r]),
                    "SoA scalar row {r}: {want:?} vs {:?}",
                    scalar[r]
                );
                assert!(
                    same(want, simd[r]),
                    "SoA AVX row {r}: {want:?} vs {:?}",
                    simd[r]
                );
            }
        }
    }
}

#[test]
fn sse_matches_a_manual_sum() {
    let tree = expr::node(
        Op::Add,
        vec![
            expr::node(Op::Mul, vec![expr::input(0), expr::constant(2.0)]),
            expr::input(1),
        ],
    );
    let (cols, aos) = sample(97, 2);
    let y: Vec<f64> = (0..97).map(|i| (i as f64 * 0.01).cos()).collect();
    let manual: f64 = (0..97)
        .map(|r| {
            let pred = expr::eval(&tree, &aos[r]);
            let d = pred - y[r];
            d * d
        })
        .sum();
    let got = eval_soa::sse(&compile(&tree).unwrap(), &cols, 97, &y);
    assert!(close(manual, got), "manual {manual:?} vs sse {got:?}");
}
