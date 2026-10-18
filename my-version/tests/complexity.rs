//! Configurable complexity (issue #13): defaults, per-operator costs, terminal
//! weights, nesting depth, parsing errors, front ordering and a search run.

use rcsr::complexity::ComplexityModel;
use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Node, Op};
use rcsr::metrics::Metric;
use rcsr::pareto::{Archive, Sol};
use rcsr::search::{run, Config};
use std::process::Command;

fn representative_trees() -> Vec<Node> {
    vec![
        constant(2.5),
        input(0),
        node(Op::Neg, vec![input(0)]),
        node(Op::Sin, vec![node(Op::Add, vec![input(0), constant(1.0)])]),
        node(Op::Pow, vec![input(0), constant(2.0)]),
        node(
            Op::If,
            vec![
                node(Op::Greater, vec![input(0), constant(0.0)]),
                node(Op::Mul, vec![input(0), constant(2.0)]),
                constant(-1.0),
            ],
        ),
        node(
            Op::Add,
            vec![
                node(Op::Mul, vec![input(0), input(0)]),
                node(Op::Div, vec![constant(3.0), input(0)]),
            ],
        ),
    ]
}

/// The defaults are the historical node count, exactly, for every tree.
#[test]
fn default_model_is_the_node_count() {
    let model = ComplexityModel::default();
    for tree in representative_trees() {
        let expected = expr::complexity(&tree) as f64;
        let got = model.complexity(&tree);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "default complexity must equal the node count"
        );
    }
}

fn sol(tree: Node, error: f64, model: &ComplexityModel) -> Sol {
    Sol {
        complexity: expr::complexity(&tree),
        cost: model.complexity(&tree),
        tree,
        error,
    }
}

/// A costlier operator makes the same-shaped tree strictly more complex, and
/// the front is ordered by the configured cost rather than the node count.
#[test]
fn per_operator_cost_changes_front_order() {
    let pow_tree = node(Op::Pow, vec![input(0), constant(2.0)]);
    let add_tree = node(
        Op::Add,
        vec![
            node(Op::Add, vec![input(0), constant(1.0)]),
            constant(2.0),
        ],
    );

    let mut model = ComplexityModel::default();
    model.apply_op_costs("pow=10").unwrap();
    assert_eq!(model.complexity(&pow_tree), 12.0, "10 + input + constant");
    assert_eq!(model.complexity(&add_tree), 5.0, "three operators + two terminals");
    assert!(model.complexity(&pow_tree) > model.complexity(&add_tree));

    // pow: 3 nodes, error 0.01. add-chain: 5 nodes, error 0.05.
    // Default: pow first (smaller node count). Custom pow=10: add-chain first.
    let default = ComplexityModel::default();
    let mut default_front = Archive::new(10);
    default_front.try_insert(sol(pow_tree.clone(), 0.01, &default));
    default_front.try_insert(sol(add_tree.clone(), 0.05, &default));
    let order: Vec<String> = default_front
        .sorted()
        .iter()
        .map(|s| expr::to_string(&s.tree, &["x".to_string()]))
        .collect();
    assert!(order[0].contains('^'), "default front starts with pow: {order:?}");

    let mut custom_front = Archive::new(10);
    let custom = model;
    custom_front.try_insert(sol(pow_tree, 0.01, &custom));
    custom_front.try_insert(sol(add_tree, 0.05, &custom));
    let order: Vec<String> = custom_front
        .sorted()
        .iter()
        .map(|s| expr::to_string(&s.tree, &["x".to_string()]))
        .collect();
    assert!(
        !order[0].contains('^'),
        "with pow=10 the add-chain must come first: {order:?}"
    );
}

/// Variable and constant weights price the terminals.
#[test]
fn terminal_weights_change_the_count() {
    let tree = node(Op::Add, vec![input(0), constant(2.0)]);
    let default = ComplexityModel::default();
    assert_eq!(default.complexity(&tree), 3.0);

    let mut model = ComplexityModel::default();
    model.apply_weights("var=2,const=3").unwrap();
    assert_eq!(model.complexity(&tree), 6.0, "1 + 2 + 3");

    // Aliases are accepted.
    let mut model = ComplexityModel::default();
    model.apply_weights("variable=2,constant=0.5").unwrap();
    assert_eq!(model.complexity(&tree), 3.5, "1 + 2 + 0.5");
}

/// Equal node counts, different nesting: only the depth term can tell them
/// apart.
#[test]
fn depth_weight_penalises_nesting() {
    // 7 nodes each: three binary operators and four inputs.
    let balanced = node(
        Op::Mul,
        vec![
            node(Op::Mul, vec![input(0), input(0)]),
            node(Op::Mul, vec![input(0), input(0)]),
        ],
    );
    let deep = node(
        Op::Mul,
        vec![
            input(0),
            node(
                Op::Mul,
                vec![input(0), node(Op::Mul, vec![input(0), input(0)])],
            ),
        ],
    );
    assert_eq!(expr::complexity(&balanced), 7);
    assert_eq!(expr::complexity(&deep), 7);

    let default = ComplexityModel::default();
    assert_eq!(default.complexity(&balanced), 7.0);
    assert_eq!(default.complexity(&deep), 7.0);

    let mut model = ComplexityModel::default();
    model.set_depth_weight(0.5).unwrap();
    assert_eq!(model.complexity(&balanced), 8.0, "depth 2");
    assert_eq!(model.complexity(&deep), 8.5, "depth 3");
}

fn rcsr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

/// Unknown names, malformed pairs and non-positive weights are usage errors
/// (exit 2) naming the bad token. Argument parsing happens before the data is
/// opened, so a nonexistent table is fine here.
#[test]
fn malformed_specs_are_usage_errors() {
    let cases: &[(&str, &str, &str)] = &[
        ("--op-complexity", "pow=2,bogus=1", "bogus"),
        ("--op-complexity", "pow", "pow"),
        ("--op-complexity", "pow=0", "pow=0"),
        ("--op-complexity", "pow=nan", "pow=nan"),
        ("--complexity-weights", "var=0", "var=0"),
        ("--complexity-weights", "bogus=1", "bogus"),
        ("--depth-weight", "-1", "-1"),
    ];
    for (flag, value, bad) in cases {
        let out = rcsr(&["--data", "/nonexistent", "--target", "y", flag, value]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{flag} {value} must be a usage error"
        );
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains(bad),
            "the error must name '{bad}', got: {err}"
        );
    }
}

/// Schema and help advertise the new options.
#[test]
fn help_and_schema_document_the_model() {
    let help = rcsr(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    for flag in ["--op-complexity", "--complexity-weights", "--depth-weight"] {
        assert!(text.contains(flag), "--help is missing {flag}");
    }

    let schema = rcsr(&["schema"]);
    assert!(schema.status.success());
    let text = String::from_utf8_lossy(&schema.stdout);
    for flag in ["--op-complexity", "--complexity-weights", "--depth-weight"] {
        assert!(text.contains(flag), "schema is missing {flag}");
    }
    assert!(text.contains("\"complexity\""));
}

fn grid(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n)
        .map(|k| lo + (hi - lo) * (k as f64) / ((n - 1) as f64))
        .collect()
}

/// A custom model must not break the search: it still recovers `y = 2x + 1`.
#[test]
fn a_custom_model_still_recovers_a_known_law() {
    let g = grid(31, -1.5, 1.5);
    let xs: Vec<Vec<f64>> = g.iter().map(|v| vec![*v]).collect();
    let ys: Vec<f64> = g.iter().map(|v| 2.0 * v + 1.0).collect();
    let train = Sample {
        x: xs.clone(),
        y: ys.clone(),
        w: Vec::new(),
    };
    let test = Sample {
        x: xs,
        y: ys,
        w: Vec::new(),
    };

    let mut cfg = Config::default();
    cfg.generations = 120;
    cfg.pop = 160;
    cfg.islands = 2;
    cfg.seed = 5;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div];
    cfg.complexity_model.apply_op_costs("mul=2,add=1.5").unwrap();
    cfg.complexity_model.apply_weights("var=1.25,const=0.75").unwrap();
    cfg.complexity_model.set_depth_weight(0.25).unwrap();
    cfg.max_complexity = 40;

    let out = run(&cfg, &train, &test);
    let best = out
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-6, "best test error = {}", best);
}
