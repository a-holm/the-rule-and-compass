//! End-to-end and unit tests. Run with `cargo test`.

use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Op};
use rcsr::metrics::{score, Metric};
use rcsr::search::{run, Config, Outcome};

fn sample(xs: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample { x: xs, y }
}

fn best_test_error(out: &Outcome) -> f64 {
    out.test_errors.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn grid(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|k| lo + (hi - lo) * (k as f64) / ((n - 1) as f64))
        .collect()
}

#[test]
fn recovers_identity() {
    let g = grid(31, -1.5, 1.5);
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let train = sample(xs.clone(), g.clone());
    let test = sample(xs, g);

    let mut cfg = Config::default();
    cfg.generations = 60;
    cfg.pop = 120;
    cfg.islands = 1;
    cfg.seed = 1;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-9,
        "best test error = {}",
        best_test_error(&out)
    );
}

#[test]
fn recovers_x_squared() {
    let g = grid(41, -2.0, 2.0);
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let ys: Vec<f64> = g.iter().map(|v| v * v).collect();
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 300;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 7;
    cfg.metric = Metric::Squared;
    cfg.ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-6,
        "best test error = {}",
        best_test_error(&out)
    );
}

#[test]
fn recovers_product_of_two_vars() {
    let g = grid(15, -1.0, 1.0);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for a in &g {
        for b in &g {
            xs.push(vec![*a, *b]);
            ys.push(a * b);
        }
    }
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 300;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 11;
    cfg.metric = Metric::Squared;
    cfg.ops = vec![Op::Add, Op::Sub, Op::Mul];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-6,
        "best test error = {}",
        best_test_error(&out)
    );
}

#[test]
fn metric_values() {
    let y = vec![1.0, 2.0, 3.0];
    assert!(score(&y, &y, Metric::Absolute).abs() < 1e-12);
    assert!(score(&y, &y, Metric::Squared).abs() < 1e-12);

    let p = vec![2.0, 3.0, 4.0];
    assert!((score(&p, &y, Metric::Absolute) - 1.0).abs() < 1e-12);
    assert!((score(&p, &y, Metric::Squared) - 1.0).abs() < 1e-12);

    // A non-finite prediction is a hard failure.
    assert!(score(&[f64::NAN, 1.0, 2.0], &y, Metric::Absolute).is_infinite());
}

#[test]
fn domain_guards() {
    assert!(expr::eval(&node(Op::Ln, vec![constant(-1.0)]), &[0.0]).is_nan());
    assert!(expr::eval(&node(Op::Sqrt, vec![constant(-4.0)]), &[0.0]).is_nan());
    assert!(expr::eval(&node(Op::Div, vec![constant(1.0), constant(0.0)]), &[0.0]).is_nan());
    assert!(expr::eval(&node(Op::Ln, vec![constant(std::f64::consts::E)]), &[0.0]) - 1.0 < 1e-12);
}

#[test]
fn simplify_folds() {
    let e = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![input(0), constant(1.0)]),
            constant(0.0),
        ],
    );
    let s = expr::simplify(&e);
    assert_eq!(expr::to_string(&s, &["x".to_string()]), "x");

    let e2 = node(Op::Mul, vec![constant(0.0), input(0)]);
    assert_eq!(expr::to_string(&expr::simplify(&e2), &["x".to_string()]), "0");

    let e3 = node(
        Op::Add,
        vec![constant(2.0), node(Op::Mul, vec![constant(3.0), constant(4.0)])],
    );
    assert_eq!(expr::to_string(&expr::simplify(&e3), &["x".to_string()]), "14");
}

#[test]
fn fit_refines_constants_to_machine_precision() {
    // y = sin(1.7 x): start from sin(1 x) and let Levenberg–Marquardt find 1.7.
    let g = grid(41, -2.0, 2.0);
    let ys: Vec<f64> = g.iter().map(|v| (1.7 * v).sin()).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let mut t = node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])]);
    rcsr::fit::fit(&mut t, &xs, &ys, 25);
    let pred: Vec<f64> = g.iter().map(|v| expr::eval(&t, &[*v])).collect();
    let err = score(&pred, &ys, Metric::Squared);
    assert!(err < 1e-20, "squared error after fit = {}", err);
}

#[test]
fn complexity_counts_nodes() {
    // x + 1 -> 3 nodes
    let e = node(Op::Add, vec![input(0), constant(1.0)]);
    assert_eq!(expr::complexity(&e), 3);
}

// ---- extended primitive set (issue #6) ----

/// Every canonical name must round-trip through `parse_list`, and the extended
/// operators must be distinct from the defaults.
#[test]
fn extended_ops_parse_and_are_distinct() {
    for name in Op::names() {
        assert!(
            !Op::parse_list(name).is_empty(),
            "parse_list dropped the name `{name}`"
        );
    }
    // A representative command line using every new operator.
    let parsed = Op::parse_list(
        "floor,ceil,round,sign,erf,erfc,asinh,acosh,atanh,mod,min,max,atan2",
    );
    let expected = vec![
        Op::Floor,
        Op::Ceil,
        Op::Round,
        Op::Sign,
        Op::Erf,
        Op::Erfc,
        Op::Asinh,
        Op::Acosh,
        Op::Atanh,
        Op::Mod,
        Op::Min,
        Op::Max,
        Op::Atan2,
    ];
    assert_eq!(parsed, expected);

    // Correct arity, and the binary operators print as f(a, b).
    for op in [
        Op::Floor,
        Op::Ceil,
        Op::Round,
        Op::Sign,
        Op::Erf,
        Op::Erfc,
        Op::Asinh,
        Op::Acosh,
        Op::Atanh,
    ] {
        assert_eq!(op.arity(), 1, "{op:?} should be unary");
    }
    for op in [Op::Mod, Op::Min, Op::Max, Op::Atan2] {
        assert_eq!(op.arity(), 2, "{op:?} should be binary");
    }
    let t = node(Op::Max, vec![input(0), constant(2.0)]);
    assert_eq!(expr::to_string(&t, &["x".to_string()]), "max(x, 2)");
    let t = node(Op::Atan2, vec![input(0), constant(1.0)]);
    assert_eq!(expr::to_string(&t, &["x".to_string()]), "atan2(x, 1)");
}

#[test]
fn extended_op_values() {
    let unary = |op: Op, v: f64| expr::eval(&node(op, vec![constant(v)]), &[0.0]);
    let binary = |op: Op, a: f64, b: f64| {
        expr::eval(&node(op, vec![constant(a), constant(b)]), &[0.0])
    };

    assert_eq!(unary(Op::Floor, 2.7), 2.0);
    assert_eq!(unary(Op::Floor, -2.1), -3.0);
    assert_eq!(unary(Op::Ceil, 2.1), 3.0);
    assert_eq!(unary(Op::Ceil, -2.7), -2.0);
    assert_eq!(unary(Op::Round, 2.5), 3.0);
    assert_eq!(unary(Op::Round, -2.5), -3.0);
    assert_eq!(unary(Op::Sign, -3.0), -1.0);
    assert_eq!(unary(Op::Sign, 0.0), 0.0);
    assert_eq!(unary(Op::Sign, 4.0), 1.0);
    assert_eq!(unary(Op::Step, 2.0), 1.0);
    assert_eq!(unary(Op::Step, -2.0), 0.0);
    assert_eq!(unary(Op::Step, 0.0), 0.0);

    assert!((binary(Op::Mod, 7.5, 2.0) - 1.5).abs() < 1e-12);
    // Floor-mod has the sign of the divisor: -1 mod 3 = 2.
    assert!((binary(Op::Mod, -1.0, 3.0) - 2.0).abs() < 1e-12);
    assert_eq!(binary(Op::Min, 3.0, 5.0), 3.0);
    assert_eq!(binary(Op::Min, -1.0, -4.0), -4.0);
    assert_eq!(binary(Op::Max, 3.0, 5.0), 5.0);

    assert!((unary(Op::Asinh, 1.0f64.sinh()) - 1.0).abs() < 1e-12);
    assert!((unary(Op::Acosh, 2.0f64.cosh()) - 2.0).abs() < 1e-12);
    assert!((unary(Op::Atanh, 1.0f64.tanh()) - 1.0).abs() < 1e-12);

    assert!((binary(Op::Atan2, 1.0, 1.0) - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert!((binary(Op::Atan2, 1.0, 0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert_eq!(binary(Op::Atan2, 0.0, 0.0), 0.0);

    // `erf` saturates to ±1.
    assert_eq!(unary(Op::Erf, 0.0), 0.0);
    assert_eq!(
        expr::eval(&node(Op::Erf, vec![constant(f64::INFINITY)]), &[0.0]),
        1.0
    );
}

#[test]
fn erf_erfc_are_accurate() {
    let erf = |x: f64| expr::eval(&node(Op::Erf, vec![constant(x)]), &[0.0]);
    let erfc = |x: f64| expr::eval(&node(Op::Erfc, vec![constant(x)]), &[0.0]);

    // Reference values (DLMF 7.2 / standard tables).
    assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-14);
    assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-14);
    assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-14);
    assert!((erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-14);
    assert!((erfc(3.0) - 2.209_049_699_858_544e-5).abs() < 1e-18);

    // Complementary pair, including the far tail that would cancel as 1 - erf.
    for x in [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 6.0] {
        assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14, "erf+erfc at {x}");
    }
    assert!((erfc(10.0) - 2.088_487_583_762_545e-45).abs() / 2.088_487_583_762_545e-45 < 1e-9);

    // Saturation and infinities stay finite.
    assert_eq!(erf(f64::INFINITY), 1.0);
    assert_eq!(erf(f64::NEG_INFINITY), -1.0);
    assert_eq!(erfc(f64::INFINITY), 0.0);
    assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    assert!(erf(f64::NAN).is_nan());
    assert!(erfc(f64::NAN).is_nan());
}

#[test]
fn extended_op_domain_guards_and_finiteness() {
    let unary = |op: Op, v: f64| expr::eval(&node(op, vec![constant(v)]), &[0.0]);
    let binary = |op: Op, a: f64, b: f64| {
        expr::eval(&node(op, vec![constant(a), constant(b)]), &[0.0])
    };

    // acosh: only [1, inf).
    assert!(unary(Op::Acosh, 0.5).is_nan());
    assert!(unary(Op::Acosh, 0.999).is_nan());
    assert_eq!(unary(Op::Acosh, 1.0), 0.0);
    assert!((unary(Op::Acosh, 2.0) - 1.316_957_896_924_816_6).abs() < 1e-12);

    // atanh: only (-1, 1).
    assert!(unary(Op::Atanh, 1.0).is_nan());
    assert!(unary(Op::Atanh, -1.0).is_nan());
    assert!(unary(Op::Atanh, 2.0).is_nan());
    assert_eq!(unary(Op::Atanh, 0.0), 0.0);

    // mod by (near) zero.
    assert!(binary(Op::Mod, 1.0, 0.0).is_nan());
    assert!(binary(Op::Mod, 1.0, 1e-13).is_nan());

    // NaN propagates instead of being swallowed by min/max/atan2.
    let nan = node(Op::Ln, vec![constant(-1.0)]);
    assert!(expr::eval(&node(Op::Min, vec![nan.clone(), constant(1.0)]), &[0.0]).is_nan());
    assert!(expr::eval(&node(Op::Max, vec![nan.clone(), constant(1.0)]), &[0.0]).is_nan());
    assert!(expr::eval(&node(Op::Atan2, vec![nan, constant(1.0)]), &[0.0]).is_nan());

    // Over a valid grid every new operator stays finite (no NaN/Inf escapes).
    for i in -40..=40 {
        let v = i as f64 / 8.0; // -5.0 .. 5.0
        for op in [Op::Floor, Op::Ceil, Op::Round, Op::Sign, Op::Erf, Op::Erfc, Op::Asinh] {
            assert!(unary(op, v).is_finite(), "{op:?} not finite at {v}");
        }
        // acosh/atanh domains.
        if v >= 1.0 {
            assert!(unary(Op::Acosh, v).is_finite());
        }
        if v.abs() < 1.0 {
            assert!(unary(Op::Atanh, v).is_finite());
        }
        for op in [Op::Min, Op::Max, Op::Atan2] {
            assert!(binary(op, v, 0.3).is_finite(), "{op:?} not finite at {v}");
        }
        if v.abs() > 0.05 {
            assert!(binary(Op::Mod, 3.0, v).is_finite());
        }
    }
}

#[test]
fn recovers_floor() {
    // y = floor(x): exact at every sample, including the integer breakpoints.
    let g: Vec<f64> = (-25..=25).map(|k| k as f64 / 10.0).collect();
    let ys: Vec<f64> = g.iter().map(|v| v.floor()).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 300;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 3;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![Op::Floor, Op::Add, Op::Sub, Op::Mul, Op::Div];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-9,
        "best test error = {}",
        best_test_error(&out)
    );
}

#[test]
fn recovers_max() {
    // y = max(x, 0.5): the constant has to be fitted, so this also exercises
    // the `max` subgradient through the adjoint pass.
    let g = grid(41, -2.0, 3.0);
    let ys: Vec<f64> = g.iter().map(|v| v.max(0.5)).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 300;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 5;
    cfg.metric = Metric::Squared;
    cfg.ops = vec![Op::Max, Op::Add, Op::Sub, Op::Mul, Op::Div];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-6,
        "best test error = {}",
        best_test_error(&out)
    );
}

// ---- conditional, logical and constraint operators (issue #7) ----

/// Evaluate a unary / binary / ternary operator on constant arguments.
fn u1(op: Op, a: f64) -> f64 {
    expr::eval(&node(op, vec![constant(a)]), &[0.0])
}

fn u2(op: Op, a: f64, b: f64) -> f64 {
    expr::eval(&node(op, vec![constant(a), constant(b)]), &[0.0])
}

fn u3(op: Op, a: f64, b: f64, c: f64) -> f64 {
    expr::eval(
        &node(op, vec![constant(a), constant(b), constant(c)]),
        &[0.0],
    )
}

#[test]
fn comparison_operator_values() {
    // `==` is exact equality.
    assert_eq!(u2(Op::Equal, 2.0, 2.0), 1.0);
    assert_eq!(u2(Op::Equal, 2.0, 3.0), 0.0);
    assert_eq!(u2(Op::Equal, 0.0, 0.0), 1.0);

    assert_eq!(u2(Op::Less, 1.0, 2.0), 1.0);
    assert_eq!(u2(Op::Less, 2.0, 2.0), 0.0); // boundary: not strictly less
    assert_eq!(u2(Op::LessOrEqual, 2.0, 2.0), 1.0); // boundary: less-or-equal

    assert_eq!(u2(Op::Greater, 3.0, 2.0), 1.0);
    assert_eq!(u2(Op::Greater, 2.0, 2.0), 0.0); // boundary: not strictly greater
    assert_eq!(u2(Op::GreaterOrEqual, 2.0, 2.0), 1.0);

    // NaN in either operand is a hard failure, not a comparison result.
    assert!(u2(Op::Greater, f64::NAN, 0.0).is_nan());
    assert!(u2(Op::Equal, 0.0, f64::NAN).is_nan());
}

#[test]
fn logical_operator_values_and_boundary() {
    // Truthy means strictly positive; exactly 0 is false.
    assert_eq!(u1(Op::Not, 0.0), 1.0); // boundary: 0 is false
    assert_eq!(u1(Op::Not, 1.0), 0.0);
    assert_eq!(u1(Op::Not, -1.0), 1.0); // negative is false, so `not` is true
    assert_eq!(u1(Op::Not, 1e-300), 0.0);
    assert!(u1(Op::Not, f64::NAN).is_nan());

    assert_eq!(u2(Op::And, 1.0, 1.0), 1.0);
    assert_eq!(u2(Op::And, 1.0, 0.0), 0.0); // boundary
    assert_eq!(u2(Op::And, 0.0, 0.0), 0.0);
    assert_eq!(u2(Op::And, -1.0, 1.0), 0.0);

    assert_eq!(u2(Op::Or, 1.0, 0.0), 1.0);
    assert_eq!(u2(Op::Or, 0.0, 0.0), 0.0); // boundary
    assert_eq!(u2(Op::Or, -1.0, 1.0), 1.0);

    assert_eq!(u2(Op::Xor, 1.0, -1.0), 1.0);
    assert_eq!(u2(Op::Xor, 1.0, 1.0), 0.0);
    assert_eq!(u2(Op::Xor, 0.0, 1.0), 1.0); // boundary: 0 is false
    assert_eq!(u2(Op::Xor, 0.0, 0.0), 0.0);

    assert!(u2(Op::And, f64::NAN, 1.0).is_nan());
    assert!(u2(Op::Or, 1.0, f64::NAN).is_nan());
    assert!(u2(Op::Xor, f64::NAN, 1.0).is_nan());
}

#[test]
fn constraint_operators_return_their_argument() {
    // require(cond) -> cond when cond > 0, else NaN.
    assert_eq!(u1(Op::Require, 2.5), 2.5);
    assert!(u1(Op::Require, 0.0).is_nan()); // boundary: not > 0
    assert!(u1(Op::Require, -1.0).is_nan());

    // constrain(x) is the unary spelling of require.
    assert_eq!(u1(Op::Constrain, 2.5), 2.5);
    assert!(u1(Op::Constrain, 0.0).is_nan());
    assert!(u1(Op::Constrain, -3.0).is_nan());

    // between(x, lo, hi) -> x on the closed interval, else NaN.
    assert_eq!(u3(Op::Between, 0.5, -1.0, 1.0), 0.5);
    assert_eq!(u3(Op::Between, -1.0, -1.0, 1.0), -1.0); // inclusive lower
    assert_eq!(u3(Op::Between, 1.0, -1.0, 1.0), 1.0); // inclusive upper
    assert!(u3(Op::Between, 1.5, -1.0, 1.0).is_nan());
    assert!(u3(Op::Between, -2.0, -1.0, 1.0).is_nan());
}

#[test]
fn violated_constraints_are_a_hard_failure() {
    // A violated constraint evaluates to NaN, so the error metric is +inf, not
    // a merely large finite error: the candidate is infeasible.
    let requires_positive =
        node(Op::Require, vec![node(Op::Sub, vec![input(0), constant(1.0)])]);
    assert_eq!(expr::eval(&requires_positive, &[3.0]), 2.0);
    assert!(expr::eval(&requires_positive, &[0.5]).is_nan());

    let y = vec![1.0, 2.0, 3.0];
    let good: Vec<f64> = [3.0f64, 4.0, 5.0]
        .iter()
        .map(|v| expr::eval(&requires_positive, &[*v]))
        .collect();
    assert!(score(&good, &y, Metric::Absolute).is_finite());

    // One violated row poisons the whole candidate.
    let bad: Vec<f64> = [3.0f64, 0.0, 5.0]
        .iter()
        .map(|v| expr::eval(&requires_positive, &[*v]))
        .collect();
    assert!(score(&bad, &y, Metric::Absolute).is_infinite());

    // The same holds for between and constrain.
    let bounded = node(
        Op::Between,
        vec![input(0), constant(-1.0), constant(1.0)],
    );
    assert!(expr::eval(&bounded, &[2.0]).is_nan());
    assert!(expr::eval(&node(Op::Constrain, vec![input(0)]), &[-1.0]).is_nan());

    // A constraint with no movable constant can never be made feasible: the
    // fitter reports an infinite sum of squared errors (a rejected candidate).
    let infeasible = node(Op::Require, vec![node(Op::Neg, vec![input(0)])]);
    let xs: Vec<Vec<f64>> = [1.0f64, 2.0, 3.0].iter().map(|v| vec![*v]).collect();
    let mut tree = infeasible;
    let sse = rcsr::fit::fit(&mut tree, &xs, &y, 10);
    assert!(!sse.is_finite(), "infeasible candidate must not be fit: {sse}");
}

#[test]
fn recovers_piecewise_x_times_positive() {
    // y = x * (x > 0), a piecewise target that needs the `greater` comparison.
    let g = grid(41, -2.0, 2.0);
    let ys: Vec<f64> = g.iter().map(|v| v * (*v > 0.0) as i32 as f64).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 400;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 9;
    cfg.metric = Metric::Squared;
    cfg.ops = vec![
        Op::Mul,
        Op::Div,
        Op::Add,
        Op::Sub,
        Op::Greater,
        Op::GreaterOrEqual,
        Op::If,
    ];

    let out = run(&cfg, &train, &test);
    assert!(
        best_test_error(&out) < 1e-6,
        "best test error = {}",
        best_test_error(&out)
    );
}
