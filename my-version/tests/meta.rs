//! Integration tests for the meta operators (issue #9).
//!
//! A meta operator inspects the *shape* of its argument, so every value it can
//! return is a whole-tree quantity, not a per-row one. These tests pin the
//! semantics, the construction-time cache, the fitter's behaviour and the text
//! round-trip, then exercise the operators through the search.

use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, meta, node, Node, Op};
use rcsr::fit;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};
use std::process::Command;

fn sample(xs: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample {
        x: xs,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn rows2() -> Vec<Vec<f64>> {
    (0..9)
        .map(|i| vec![i as f64 - 4.0, (i as f64) * 0.5 - 1.0])
        .collect()
}

/// Evaluate a meta operator on two different rows: the value must not depend
/// on the row.
fn same_on_every_row(kind: Op, child: Node) -> f64 {
    let first = expr::eval(&meta(kind, child.clone()), &[0.0, 0.0]);
    let second = expr::eval(&meta(kind, child), &[3.5, -7.25]);
    assert_eq!(first, second, "{kind:?} must be row-independent");
    first
}

#[test]
fn each_operator_reports_the_shape_of_hand_built_trees() {
    // (complexity, variables, constants, contains, contains_op)
    let cases: Vec<(Node, [f64; 5])> = vec![
        // a bare input
        (input(0), [1.0, 1.0, 0.0, 1.0, 0.0]),
        // a bare constant
        (constant(7.5), [1.0, 0.0, 1.0, 0.0, 0.0]),
        // x1 * x1: repeated variables count once
        (
            node(Op::Mul, vec![input(0), input(0)]),
            [3.0, 1.0, 0.0, 1.0, 1.0],
        ),
        // (x1 * x2) + 3.5
        (
            node(
                Op::Add,
                vec![node(Op::Mul, vec![input(0), input(1)]), constant(3.5)],
            ),
            [5.0, 2.0, 1.0, 1.0, 1.0],
        ),
        // sin(x1) ^ 2
        (
            node(
                Op::Pow,
                vec![node(Op::Sin, vec![input(0)]), constant(2.0)],
            ),
            [4.0, 1.0, 1.0, 1.0, 1.0],
        ),
        // a meta operator nested inside another counts as an operator node
        (
            meta(
                Op::Complexity,
                meta(Op::Complexity, node(Op::Mul, vec![input(0), input(1)])),
            ),
            [5.0, 2.0, 0.0, 1.0, 1.0],
        ),
    ];
    let ops = [
        Op::Complexity,
        Op::Variables,
        Op::Constants,
        Op::Contains,
        Op::ContainsOp,
    ];
    for (child, want) in cases {
        for (i, kind) in ops.iter().enumerate() {
            assert_eq!(
                same_on_every_row(*kind, child.clone()),
                want[i],
                "{kind:?} on {child:?}"
            );
        }
    }
}

#[test]
fn wrapping_a_subtree_does_not_change_its_constant_count() {
    let subtree = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(1.0), input(0)]),
            constant(2.0),
        ],
    );
    let before = expr::count_constants(&subtree);
    assert_eq!(before, 2);
    for kind in [
        Op::Complexity,
        Op::Variables,
        Op::Constants,
        Op::Contains,
        Op::ContainsOp,
    ] {
        let wrapped = meta(kind, subtree.clone());
        assert_eq!(
            expr::count_constants(&wrapped),
            before,
            "{kind:?} added a constant slot"
        );
    }
    // The operator itself is never counted as a constant, even when its whole
    // argument is one.
    assert_eq!(expr::count_constants(&meta(Op::Constants, constant(1.0))), 1);
}

#[test]
fn fitting_constants_inside_a_meta_argument_leaves_the_meta_value_alone() {
    // The meta term is the intercept (3) and the second branch is the slope.
    // The fitter may only move the slope and the intercept's sibling; the
    // constant *inside* the meta argument has no gradient.
    let meta_tree = meta(
        Op::Complexity,
        node(Op::Mul, vec![constant(1.0), input(0)]),
    );
    let before = meta_tree.val;
    assert_eq!(before, 3.0);

    let mut tree = node(
        Op::Add,
        vec![
            meta_tree,
            node(Op::Mul, vec![constant(0.0), input(0)]),
        ],
    );
    let x = rows2();
    let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();
    fit::fit(&mut tree, &x, &y, 40);

    assert_eq!(
        tree.kids[0].val, before,
        "the cached structural value must survive the fit"
    );
    // The constant inside the meta argument received no gradient.
    assert_eq!(tree.kids[0].kids[0].kids[0].val, 1.0);
    // ... and the model still evaluates to the fitted line.
    for row in &x {
        let got = expr::eval(&tree, row);
        let want = 2.0 * row[0] + 3.0;
        assert!(
            (got - want).abs() < 1e-6,
            "row {row:?}: {got} vs {want}"
        );
    }
}

#[test]
fn moving_a_constant_changes_its_value_but_not_the_meta_cache() {
    // A constant inside the argument is a fitted parameter in principle; the
    // meta value depends on the leaf count, not its value, so the cache must
    // not be recomputed (or invalidated).
    let mut tree = meta(
        Op::Constants,
        node(Op::Mul, vec![constant(2.0), input(0)]),
    );
    assert_eq!(tree.val, 1.0);
    let c = expr::nth_mut(&mut tree, 1).expect("the argument's constant");
    c.val = 9.75;
    assert_eq!(tree.val, 1.0);
    // Simplification keeps the same shape here, so the cache stays 1.
    let simple = expr::simplify(&tree);
    assert_eq!(simple.val, 1.0);
}

#[test]
fn every_meta_operator_round_trips_through_text() {
    let vars: Vec<String> = ["x", "y"].iter().map(|s| s.to_string()).collect();
    let cases = [
        (Op::Complexity, node(Op::Mul, vec![input(0), input(1)])),
        (Op::Variables, node(Op::Add, vec![input(0), input(1)])),
        (Op::Constants, node(Op::Add, vec![input(0), constant(2.0)])),
        (Op::Contains, node(Op::Sin, vec![input(1)])),
        (Op::ContainsOp, input(0)),
        (Op::Complexity, constant(4.0)),
        (
            Op::Variables,
            meta(Op::Complexity, node(Op::Mul, vec![input(0), input(1)])),
        ),
    ];
    for (kind, child) in cases {
        let tree = meta(kind, child);
        let printed = expr::to_string(&tree, &vars);
        assert!(
            printed.starts_with(kind.symbol()),
            "{printed} should start with {}",
            kind.symbol()
        );
        let again = expr::parse(&printed, &vars)
            .unwrap_or_else(|e| panic!("re-parsing '{printed}': {e}"));
        assert!(expr::same(&tree, &again), "'{printed}' did not round-trip");
        assert_eq!(again.val, tree.val);
    }
}

/// Recovery test from the issue: the target column is
/// `k = complexity(x1 * x2) = 3` on every row, computed by hand, and the search
/// is allowed to use the meta operators.
///
/// What this proves: the operators are wired through the parser, evaluator,
/// constant fitter and search, and a zero-error model is recovered; the exact
/// expression `complexity(x1 * x2)` scores zero error on the same data.
///
/// What it does **not** prove: any predictive advantage. A meta operator
/// returns one value per tree, so the target it matches is necessarily a
/// constant column. The parsimony pressure correctly prefers the one-node
/// constant `3` over the four-node meta expression; the two are numerically
/// equivalent here. The meta operators are a description of the model, not a
/// new source of row-level signal.
#[test]
fn search_recovers_a_target_built_from_the_models_complexity() {
    let xs = rows2();
    let k = expr::eval(
        &meta(Op::Complexity, node(Op::Mul, vec![input(0), input(1)])),
        &xs[0],
    );
    assert_eq!(k, 3.0, "hand-computed complexity(x1 * x2)");

    let ys: Vec<f64> = xs.iter().map(|_| k).collect();
    let train = sample(xs.clone(), ys.clone());
    let test = sample(xs.clone(), ys.clone());

    let mut cfg = Config::default();
    cfg.generations = 30;
    cfg.pop = 80;
    cfg.islands = 1;
    cfg.seed = 9;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Complexity,
        Op::Variables,
        Op::Constants,
        Op::Contains,
        Op::ContainsOp,
    ];

    let out = search::run(&cfg, &train, &test);
    let best = out
        .archive
        .sorted()
        .into_iter()
        .min_by(|a, b| a.error.total_cmp(&b.error))
        .expect("the search always archives a finite candidate");
    assert!(
        best.error < 1e-6,
        "best archive error = {} ({})",
        best.error,
        expr::to_string(&best.tree, &["x1".to_string(), "x2".to_string()])
    );

    // The documented equivalent: a constant `3` model. Its predictions match
    // the target column exactly.
    for row in &xs {
        assert!((expr::eval(&best.tree, row) - k).abs() < 1e-6);
    }

    // The exact meta expression is also a zero-error model; this is the form
    // the issue asked to recover.
    let intended = meta(Op::Complexity, node(Op::Mul, vec![input(0), input(1)]));
    assert!(search::eval_error(&intended, &train, Metric::Absolute) < 1e-12);
    assert!(search::eval_error(&intended, &test, Metric::Absolute) < 1e-12);
}

#[test]
fn ops_rejects_an_unknown_meta_name_with_exit_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            "/nonexistent",
            "--target",
            "y",
            "--ops",
            "complexity,complexityy",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("complexityy"), "stderr: {err}");
}

#[test]
fn ops_accepts_every_meta_name_and_none_is_default() {
    for name in ["complexity", "variables", "constants", "contains", "contains_op"] {
        let op = Op::from_name(name).unwrap_or_else(|| panic!("from_name({name})"));
        assert!(op.is_meta(), "{name}");
        assert!(!Op::default_set().contains(&op), "{name} must be opt-in");
    }
    // Argument parsing succeeds, so the run proceeds to the (missing) data and
    // fails with the input exit code, not the usage code.
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            "/nonexistent",
            "--target",
            "y",
            "--ops",
            "complexity variables constants contains contains_op",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(3));
}
