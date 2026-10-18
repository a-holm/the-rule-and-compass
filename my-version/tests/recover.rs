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
fn complexity_counts_nodes() {
    // x + 1 -> 3 nodes
    let e = node(Op::Add, vec![input(0), constant(1.0)]);
    assert_eq!(expr::complexity(&e), 3);
}
