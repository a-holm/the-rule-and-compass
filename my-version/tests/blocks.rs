//! User-defined building blocks (issue #10).
//!
//! These tests cover the public surface: `--function`-style definitions,
//! parameter inference, evaluation, printing/parsing, per-block complexity and
//! shared constants, plus an end-to-end search that can only express the law
//! through the supplied block.

use rcsr::complexity::ComplexityModel;
use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Op};
use rcsr::metrics::Metric;
use rcsr::pareto::{Archive, Sol};
use rcsr::search::{run, Config};

fn vars() -> Vec<String> {
    vec!["x1".to_string(), "x2".to_string()]
}

fn sample(xs: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample {
        x: xs,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

#[test]
fn definitions_parse_and_infer_parameters() {
    let blocks = expr::parse_blocks(&[
        "f(a,b)=sin(a)*cos(b)".to_string(),
        "g=2*x + y".to_string(),
    ])
    .expect("definitions parse");

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].name, "f");
    assert_eq!(blocks[0].params, ["a", "b"]);
    // `name=expr`: free variables in order of first appearance.
    assert_eq!(blocks[1].name, "g");
    assert_eq!(blocks[1].params, ["x", "y"]);
}

#[test]
fn calls_evaluate_and_print_with_the_user_name() {
    let blocks = expr::parse_blocks(&["f(a,b)=sin(a)*cos(b)".to_string()]).unwrap();
    expr::set_blocks(blocks.clone());

    let t = expr::parse_with("f(x1, x2)", &vars(), &blocks).expect("call parses");
    assert_eq!(t.op, Op::Custom(0));
    assert_eq!(expr::to_string(&t, &vars()), "f(x1, x2)");
    let want = 0.5f64.sin() * 0.25f64.cos();
    assert!((expr::eval(&t, &[0.5, 0.25]) - want).abs() < 1e-15);

    // The canonical slot name is accepted too and prints as the user name.
    let canonical = expr::parse_with("custom0(x1, x2)", &vars(), &blocks).unwrap();
    assert!(expr::same(&t, &canonical));
    assert_eq!(expr::to_string(&canonical, &vars()), "f(x1, x2)");
}

#[test]
fn printing_round_trips_through_the_parser() {
    let blocks = expr::parse_blocks(&["f(a,b)=sin(a)*cos(b)".to_string()]).unwrap();
    let t = expr::parse_with("f(x1 + 1, x2)", &vars(), &blocks).unwrap();
    let printed = expr::to_string(&t, &vars());
    let again = expr::parse_with(&printed, &vars(), &blocks)
        .unwrap_or_else(|e| panic!("re-parsing '{printed}': {e}"));
    assert!(expr::same(&t, &again), "{printed} did not round-trip");
}

#[test]
fn arity_mismatch_is_rejected() {
    let blocks = expr::parse_blocks(&["f(a,b)=a+b".to_string()]).unwrap();
    let err = expr::parse_with("f(x1)", &vars(), &blocks).unwrap_err();
    assert!(err.message.contains("takes 2 arguments"), "{}", err.message);
    let err = expr::parse_with("f(x1, x2, x1)", &vars(), &blocks).unwrap_err();
    assert!(err.message.contains("takes 2 arguments"), "{}", err.message);
}

#[test]
fn recursion_is_rejected() {
    let err = expr::parse_blocks(&["f(a)=f(a)".to_string()]).unwrap_err();
    assert!(err.contains("unknown function 'f'"), "{}", err);
}

#[test]
fn duplicate_names_are_rejected() {
    let err = expr::parse_blocks(&["f(a)=a".to_string(), "f(b)=b".to_string()]).unwrap_err();
    assert!(err.contains("duplicate block name 'f'"), "{}", err);
}

#[test]
fn at_most_four_definitions_and_four_parameters() {
    let err = expr::parse_blocks(&[
        "a(x)=x".to_string(),
        "b(x)=x".to_string(),
        "c(x)=x".to_string(),
        "d(x)=x".to_string(),
        "e(x)=x".to_string(),
    ])
    .unwrap_err();
    assert!(err.contains("at most four"), "{}", err);

    let err = expr::parse_blocks(&["f(a,b,c,d,e)=a".to_string()]).unwrap_err();
    assert!(err.contains("at most four parameters"), "{}", err);
}

#[test]
fn block_cost_participates_in_the_complexity_model() {
    let blocks = expr::parse_blocks(&["f(a,b)=sin(a)*cos(b)".to_string()]).unwrap();
    expr::set_blocks(blocks);
    let tree = node(Op::Custom(0), vec![input(0), input(1)]);

    let mut model = ComplexityModel::default();
    // Default cost is 1: the call and its two inputs.
    assert_eq!(model.complexity(&tree), 3.0);

    model.apply_op_costs("custom0=10").unwrap();
    assert_eq!(model.complexity(&tree), 12.0);

    // The user's name is resolved to its slot by the CLI; the model itself
    // takes the canonical slot name.
    let mut named = ComplexityModel::default();
    named.apply_op_costs("custom0=4").unwrap();
    assert_eq!(named.complexity(&tree), 6.0);

    // A costlier block now ranks above a same-size primitive tree.
    let plain = node(Op::Sin, vec![input(0)]);
    assert!(model.complexity(&tree) > model.complexity(&plain));

    // That cost, not the node count, orders the Pareto front.
    let mut front = Archive::new(10);
    front.try_insert(Sol {
        tree: tree.clone(),
        error: 0.02,
        complexity: expr::complexity(&tree),
        cost: model.complexity(&tree),
    });
    front.try_insert(Sol {
        tree: plain.clone(),
        error: 0.03,
        complexity: expr::complexity(&plain),
        cost: model.complexity(&plain),
    });
    let sorted = front.sorted();
    assert_eq!(sorted.len(), 2);
    assert_eq!(sorted[0].tree.op, Op::Sin, "cheaper primitive comes first");
    assert_eq!(sorted[1].tree.op, Op::Custom(0));
}

#[test]
fn body_constants_are_shared_by_every_call() {
    // f(a) = c*a with a body constant c. Two calls must use ONE fitted c.
    let blocks = expr::parse_blocks(&["f(a)=2*a".to_string()]).unwrap();
    expr::set_blocks(blocks);

    let one = node(Op::Custom(0), vec![input(0)]);
    let tree = node(Op::Add, vec![one.clone(), one]);
    assert_eq!(
        expr::count_constants(&tree),
        1,
        "the shared body constant must be counted once, not once per call"
    );

    let g: Vec<f64> = (0..21).map(|i| -2.0 + 4.0 * i as f64 / 20.0).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    // y = 2 * (3.25 x): the block is called twice with the same c.
    let ys: Vec<f64> = g.iter().map(|v| 2.0 * 3.25 * v).collect();

    let mut fitted = expr::simplify(&tree);
    let sse = rcsr::fit::fit(&mut fitted, &xs, &ys, 25);
    assert!(sse < 1e-18, "squared error after fit = {}", sse);

    let pred: Vec<f64> = g
        .iter()
        .map(|v| expr::eval(&fitted, &[*v]))
        .collect();
    for (p, y) in pred.iter().zip(&ys) {
        assert!((p - y).abs() < 1e-9, "prediction {} vs {}", p, y);
    }

    // Both calls carry the same fitted constant.
    let mut cs = Vec::new();
    fn body_const(n: &expr::Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            body_const(k, out);
        }
    }
    fn collect(n: &expr::Node, out: &mut Vec<f64>) {
        if let Some(b) = &n.block {
            body_const(&b.body, out);
        }
        for k in &n.kids {
            collect(k, out);
        }
    }
    collect(&fitted, &mut cs);
    assert_eq!(cs.len(), 2);
    assert!((cs[0] - 3.25).abs() < 1e-9, "first use c = {}", cs[0]);
    assert_eq!(cs[0].to_bits(), cs[1].to_bits(), "shared c diverged");
}

#[test]
fn a_law_expressed_with_a_supplied_block_is_recovered() {
    let blocks = expr::parse_blocks(&["f(a,b)=sin(a)*cos(b)".to_string()]).unwrap();
    expr::set_blocks(blocks);

    let g: Vec<f64> = (0..15).map(|i| -1.5 + 3.0 * i as f64 / 14.0).collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for a in &g {
        for b in &g {
            xs.push(vec![*a, *b]);
            ys.push(a.sin() * b.cos());
        }
    }
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs, ys);

    let mut cfg = Config::default();
    cfg.generations = 25;
    cfg.pop = 120;
    cfg.islands = 1;
    cfg.seed = 4;
    cfg.metric = Metric::Squared;
    // The only way to express the law is the supplied block.
    cfg.ops = vec![Op::Custom(0)];
    cfg.init_depth = 1;

    let out = run(&cfg, &train, &test);
    let best = out
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-12, "best test error = {}", best);

    let shown = expr::to_string(
        &expr::simplify(
            &out.archive
                .items
                .iter()
                .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
                .expect("a solution")
                .tree,
        ),
        &["x1".to_string(), "x2".to_string()],
    );
    assert!(shown.starts_with("f("), "recovered form is {}", shown);
}

/// Without `--function` the registry is empty and the search keeps its
/// historical result for a fixed seed.
#[test]
fn no_functions_keeps_the_baseline_behaviour() {
    expr::set_blocks(Vec::new());
    assert!(expr::current_blocks().is_empty());

    let g: Vec<f64> = (0..31).map(|i| -1.5 + 3.0 * i as f64 / 30.0).collect();
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let train = sample(xs.clone(), g.clone());
    let test = sample(xs, g);

    let mut cfg = Config::default();
    cfg.generations = 50;
    cfg.pop = 100;
    cfg.islands = 1;
    cfg.seed = 3;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div];

    let a = run(&cfg, &train, &test);
    let b = run(&cfg, &train, &test);
    let a_best = a
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let b_best = b
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(a_best < 1e-9, "identity is not recovered: {}", a_best);
    assert_eq!(a_best.to_bits(), b_best.to_bits());
    assert_eq!(a.archive.items.len(), b.archive.items.len());
    assert_eq!(
        expr::to_string(&a.archive.sorted()[0].tree, &["x".to_string()]),
        expr::to_string(&b.archive.sorted()[0].tree, &["x".to_string()])
    );

    // The primitive vocabulary and default set are unchanged.
    assert_eq!(Op::default_set().len(), 21);
    assert!(!Op::default_set().iter().any(|o| o.is_custom()));
    assert!(!Op::names().iter().any(|n| n.starts_with("custom")));
    let probe = node(Op::Add, vec![input(0), constant(1.0)]);
    assert!(expr::same(&probe, &probe));
}

#[test]
fn block_definitions_may_use_earlier_blocks() {
    let blocks = expr::parse_blocks(&[
        "f(a)=sin(a)".to_string(),
        "g(a,b)=f(a)*b".to_string(),
    ])
    .unwrap();
    expr::set_blocks(blocks.clone());
    let t = expr::parse_with("g(x1, x2)", &vars(), &blocks).unwrap();
    let want = 0.5f64.sin() * 0.25;
    assert!((expr::eval(&t, &[0.5, 0.25]) - want).abs() < 1e-15);
}

#[test]
fn forward_references_are_rejected() {
    let err = expr::parse_blocks(&[
        "f(a)=g(a)".to_string(),
        "g(a)=sin(a)".to_string(),
    ])
    .unwrap_err();
    assert!(err.contains("unknown function 'g'"), "{}", err);
}

#[test]
fn a_block_named_after_a_primitive_is_rejected() {
    let err = expr::parse_blocks(&["sin(a)=a".to_string()]).unwrap_err();
    assert!(err.contains("reserved"), "{}", err);
}

#[test]
fn block_with_no_parameters_is_a_named_constant() {
    let blocks = expr::parse_blocks(&["k=2.5".to_string()]).unwrap();
    assert!(blocks[0].params.is_empty());
    expr::set_blocks(blocks);
    let t = node(Op::Custom(0), Vec::new());
    assert_eq!(expr::eval(&t, &[]), 2.5);
}
