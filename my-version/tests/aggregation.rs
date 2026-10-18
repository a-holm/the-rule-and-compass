//! Dataset aggregation operators (issue #8).
//!
//! An aggregate is a reduction over one input column of the **training** rows.
//! Its value is computed once into an `AggTable`, cached in `Node::val`, and
//! never recomputed per row or from the test split. These tests pin down the
//! hand-computed values, the cache rule, the absence of a fitted constant, the
//! printed form, feasibility, and a recovery run on a target whose exact slope
//! is a dataset statistic.

use rcsr::data::Sample;
use rcsr::expr::{
    self, aggregate, bind_aggregates, constant, eval, input, meta, node, AggKind, AggTable, Op,
};
use rcsr::search::{eval_error, run, Config};
use rcsr::{fit, metrics::Metric};

fn sample(xs: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample {
        x: xs,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

/// A hand-computed one-column table: 1, 2, 3, 4, 5.
fn column() -> (Sample, AggTable) {
    let xs: Vec<Vec<f64>> = [1.0, 2.0, 3.0, 4.0, 5.0]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let y = xs.iter().map(|r| 2.0 * r[0]).collect();
    let s = sample(xs, y);
    let t = AggTable::from_sample(&s);
    (s, t)
}

fn contains_agg(n: &expr::Node) -> bool {
    n.op.is_agg() || n.kids.iter().any(contains_agg)
}

// ---------------------------------------------------------------------------
// Exact values
// ---------------------------------------------------------------------------

/// mean/median = 3, population std = sqrt(2), q1 = 2, q3 = 4, sum = 15,
/// min = 1, max = 5, count = 5.
#[test]
fn every_aggregate_matches_a_hand_computed_column() {
    let (_, t) = column();
    let expect = [
        (AggKind::Mean, 3.0),
        (AggKind::Std, 2.0f64.sqrt()),
        (AggKind::Median, 3.0),
        (AggKind::Iqr, 2.0),
        (AggKind::Sum, 15.0),
        (AggKind::Min, 1.0),
        (AggKind::Max, 5.0),
    ];
    for (kind, want) in expect {
        assert!(
            (t.get(kind, 0) - want).abs() < 1e-12,
            "{kind:?}: {} != {want}",
            t.get(kind, 0)
        );
        let n = aggregate(kind, Some(0), t.get(kind, 0));
        // The same value on every row, including one far outside the column.
        for row in [-1000.0, 0.0, 3.0, 1e9] {
            assert!((eval(&n, &[row]) - want).abs() < 1e-12, "{kind:?}");
        }
    }
    assert_eq!(t.get(AggKind::Count, 0), 5.0);
    assert_eq!(t.rows(), 5);
    let c = aggregate(AggKind::Count, None, t.get(AggKind::Count, 0));
    assert!(c.kids.is_empty(), "count() is zero-arity");
    assert_eq!(eval(&c, &[]), 5.0);
}

#[test]
fn aggregate_values_come_from_the_training_rows_only() {
    let (train, t) = column();
    // A test half chosen to move every statistic as far as possible.
    let test = sample(
        vec![vec![1000.0], vec![2000.0], vec![3000.0], vec![-500.0]],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let test_table = AggTable::from_sample(&test);
    for kind in AggKind::ALL {
        let a = t.get(kind, 0);
        // The cached value is exactly the training value...
        let n = aggregate(kind, Some(0), a);
        assert_eq!(eval(&n, &test.x[0]), a, "{kind:?}");
        // ...not something the test half would suggest.
        if kind != AggKind::Count {
            assert_ne!(a, test_table.get(kind, 0), "{kind:?} would leak");
        }
    }
    // `count()` is the training row count, not the test row count.
    let c = aggregate(AggKind::Count, None, t.get(AggKind::Count, 0));
    assert_eq!(eval(&c, &[]), train.len() as f64);
    assert_ne!(train.len(), test.len());
    // A bound tree evaluates identically on every test row: the evaluator never
    // recomputes a reduction.
    let mut law = node(
        Op::Div,
        vec![
            input(0),
            aggregate(AggKind::Mean, Some(0), t.get(AggKind::Mean, 0)),
        ],
    );
    bind_aggregates(&mut law, &t).unwrap();
    let v: Vec<f64> = test.x.iter().map(|r| eval(&law, r)).collect();
    for (row, got) in test.x.iter().zip(&v) {
        assert!((got - row[0] / 3.0).abs() < 1e-12);
    }
}

/// A column with no finite value cannot invent a statistic; the existing
/// domain guard turns the `NaN` into an infeasible candidate.
#[test]
fn a_column_without_finite_values_is_infeasible() {
    let s = sample(
        vec![vec![f64::NAN], vec![f64::INFINITY], vec![f64::NEG_INFINITY]],
        vec![1.0, 2.0, 3.0],
    );
    let t = AggTable::from_sample(&s);
    for kind in [
        AggKind::Mean,
        AggKind::Std,
        AggKind::Median,
        AggKind::Iqr,
        AggKind::Sum,
        AggKind::Min,
        AggKind::Max,
    ] {
        assert!(t.get(kind, 0).is_nan(), "{kind:?} must be NaN");
        let n = aggregate(kind, Some(0), t.get(kind, 0));
        assert!(eval(&n, &[1.0]).is_nan(), "{kind:?}");
        assert_eq!(
            eval_error(&n, &s, Metric::Absolute),
            f64::INFINITY,
            "{kind:?} must make the candidate infeasible"
        );
    }
    // count() is still defined: it is the number of rows, not a value.
    assert_eq!(t.get(AggKind::Count, 0), 3.0);
}

// ---------------------------------------------------------------------------
// A dataset fact is not a fitted parameter
// ---------------------------------------------------------------------------

#[test]
fn an_aggregate_has_no_fitted_constant_slot() {
    let (s, t) = column();
    let mean = t.get(AggKind::Mean, 0);

    let with_agg = node(
        Op::Mul,
        vec![
            constant(1.0),
            aggregate(AggKind::Mean, Some(0), mean),
        ],
    );
    let with_const = node(Op::Mul, vec![constant(1.0), constant(mean)]);
    // The aggregate contributes nothing to `k`; the plain constant in the same
    // slot is one free parameter. (A dataset fact is not a fitted parameter, so
    // the aggregate form has exactly one fewer parameter.)
    assert_eq!(expr::count_constants(&with_agg), 1);
    assert_eq!(expr::count_constants(&with_const), 2);
    assert_eq!(
        expr::count_constants(&with_const) - expr::count_constants(&with_agg),
        1,
        "an aggregate must not add a parameter"
    );
    // The aggregate itself has no constant leaf at all, not even a hidden one.
    assert_eq!(expr::count_constants(&aggregate(AggKind::Mean, Some(0), mean)), 0);
    // `constants(e)` (the meta operator) agrees.
    assert_eq!(meta(Op::Constants, aggregate(AggKind::Mean, Some(0), mean)).val, 0.0);

    // Fit a target that wants the outer multiplier `2.5 * mean`: the cached
    // dataset value must not move by a single bit.
    let y: Vec<f64> = s.x.iter().map(|r| 2.5 * r[0]).collect();
    let mut law = node(
        Op::Mul,
        vec![
            constant(1.0),
            node(
                Op::Div,
                vec![input(0), aggregate(AggKind::Mean, Some(0), mean)],
            ),
        ],
    );
    let before = law.kids[1].kids[1].val.to_bits();
    fit::fit(&mut law, &s.x, &y, 60);
    assert_eq!(
        law.kids[1].kids[1].val.to_bits(),
        before,
        "the fitter must not move an aggregate"
    );
    let predicted: Vec<f64> = s.x.iter().map(|r| eval(&law, r)).collect();
    for (p, target) in predicted.iter().zip(&y) {
        assert!((p - target).abs() < 1e-6, "{p} vs {target}");
    }
}

// ---------------------------------------------------------------------------
// Printing and parsing
// ---------------------------------------------------------------------------

#[test]
fn aggregates_print_and_parse_round_trip() {
    let v = vec!["x1".to_string()];
    let (_, t) = column();
    for src in [
        "mean(x1)",
        "std(x1)",
        "median(x1)",
        "iqr(x1)",
        "sum(x1)",
        "amin(x1)",
        "amax(x1)",
        "min(x1)",
        "max(x1)",
        "count()",
        "x1 / mean(x1)",
        "(x1 - median(x1)) / iqr(x1)",
    ] {
        let parsed = expr::parse(src, &v).unwrap_or_else(|e| panic!("{src}: {e}"));
        let printed = expr::to_string(&parsed, &v);
        let again = expr::parse(&printed, &v)
            .unwrap_or_else(|e| panic!("re-parsing '{printed}' from '{src}': {e}"));
        // A parsed aggregate is unresolved until bound to the data.
        let mut a = parsed;
        let mut b = again;
        bind_aggregates(&mut a, &t).unwrap_or_else(|e| panic!("{src}: {e}"));
        bind_aggregates(&mut b, &t).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert!(expr::same(&a, &b), "'{src}' printed as '{printed}' did not round-trip");
    }
    // The canonical printed spellings and the column requirement.
    let t = expr::parse("mean(x1)", &v).unwrap();
    assert_eq!(expr::to_string(&t, &v), "mean(x1)");
    assert_eq!(expr::complexity(&t), 2);
    let c = expr::parse("count()", &v).unwrap();
    assert_eq!(expr::to_string(&c, &v), "count()");
    assert_eq!(expr::complexity(&c), 1);
    assert!(expr::parse("mean(x1 + 1)", &v).is_err());
    assert!(expr::parse("mean(x1, x1)", &v).is_err());
    // `min`/`max` keep their binary meaning with two arguments.
    assert_eq!(expr::parse("min(x1, x1)", &v).unwrap().op, Op::Min);
}

#[test]
fn aggregates_are_opt_in_and_discoverable() {
    for kind in AggKind::ALL {
        let op = Op::Agg(kind);
        assert!(!Op::default_set().contains(&op), "{kind:?}");
        assert!(Op::names().contains(&op.symbol()), "{kind:?}");
    }
    assert_eq!(
        Op::parse_list_checked("add,div,mean,count").unwrap(),
        vec![
            Op::Add,
            Op::Div,
            Op::Agg(AggKind::Mean),
            Op::Agg(AggKind::Count)
        ]
    );
}

/// The sampler emits well-formed aggregate nodes whose value comes from the
/// table, and mutation keeps them consistent when it moves the column.
#[test]
fn the_sampler_and_mutation_read_the_cached_table() {
    use rcsr::expr::{mutate_with, random_tree_with};
    use rcsr::rng::Mt;

    let (_, t) = column();
    let ops = vec![
        Op::Add,
        Op::Mul,
        Op::Agg(AggKind::Mean),
        Op::Agg(AggKind::Median),
        Op::Agg(AggKind::Count),
    ];
    let mut rng = Mt::new(42);
    let check = |n: &expr::Node| {
        fn go(n: &expr::Node, t: &AggTable) {
            if let Op::Agg(kind) = n.op {
                if kind.arity() == 0 {
                    assert!(n.kids.is_empty(), "count() must stay zero-arity");
                } else {
                    assert_eq!(n.kids.len(), 1);
                    assert_eq!(n.kids[0].op, Op::Input, "the child is the column");
                    let col = n.kids[0].val as usize;
                    assert_eq!(n.val, t.get(kind, col), "value from the table");
                }
            }
            for k in &n.kids {
                go(k, t);
            }
        }
        go(n, &t);
    };
    for _ in 0..200 {
        check(&random_tree_with(&mut rng, 1, &ops, 3, Some(&t)));
    }
    for _ in 0..500 {
        let mut n = random_tree_with(&mut rng, 1, &ops, 3, Some(&t));
        mutate_with(&mut n, &mut rng, 1, &ops, Some(&t));
        check(&n);
    }
}

#[test]
fn aggregates_survive_a_checkpoint_round_trip() {
    let v = vec!["x1".to_string()];
    let (_, t) = column();
    let mut tree = expr::parse("x1 / mean(x1)", &v).unwrap();
    bind_aggregates(&mut tree, &t).unwrap();
    let text = rcsr::checkpoint::encode_tree(&tree);
    let back = rcsr::checkpoint::decode_tree(&text).unwrap();
    assert!(expr::same(&tree, &back), "{text}");
    assert!(contains_agg(&back));
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

/// `y = x1 / mean(x1)`: the exact slope is the dataset's mean.
///
/// Constant fitting is disabled, so a plain `x1 / c` cannot land on the exact
/// slope (sampled constants are rounded to three decimals); only the aggregate
/// supplies it. The run is unseeded, so this is what the search itself found.
#[test]
fn recovery_needs_a_dataset_aggregate() {
    let xs: Vec<f64> = vec![0.7, 1.9, 2.3, 3.1, 5.4, 6.6, 8.2, 9.7];
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!((mean - 4.7375).abs() < 1e-12);
    let rows: Vec<Vec<f64>> = xs.iter().map(|&v| vec![v]).collect();
    let y: Vec<f64> = xs.iter().map(|v| v / mean).collect();
    let train = sample(rows.clone(), y.clone());
    let test = sample(rows, y);

    let cfg = Config {
        generations: 300,
        islands: 4,
        pop: 250,
        max_complexity: 12,
        fit_iters: 0,
        patience: 0,
        seed: 7,
        metric: Metric::Absolute,
        ops: vec![
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Neg,
            Op::Agg(AggKind::Mean),
            Op::Agg(AggKind::Median),
            Op::Agg(AggKind::Count),
        ],
        ..Default::default()
    };
    let out = run(&cfg, &train, &test);
    let best = out
        .archive
        .items
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).expect("finite errors"))
        .expect("a Pareto member");
    let shown = expr::to_string(&best.tree, &["x1".to_string()]);
    eprintln!(
        "recovery: error={:.3e} complexity={} agg={} law={}",
        best.error,
        best.complexity,
        contains_agg(&best.tree),
        shown
    );
    // The exact law is `x1 / mean(x1)`; anything less than machine precision
    // means the search did not use the dataset statistic.
    assert!(
        best.error < 1e-9,
        "the search did not reach the exact dataset statistic: {shown} (error {})",
        best.error
    );
    assert!(
        contains_agg(&best.tree),
        "the exact law should contain an aggregate: {shown}"
    );
    // Independently: the law with a three-decimal constant is measurably worse.
    let rounded = node(
        Op::Div,
        vec![input(0), constant((mean * 1000.0).round() / 1000.0)],
    );
    let rounded_error = eval_error(&rounded, &train, Metric::Absolute);
    eprintln!("recovery: x1/{} error={rounded_error:.3e}", (mean * 1000.0).round() / 1000.0);
    assert!(rounded_error > 1e-6);
}

/// The acceptance-criterion target `y = (x1 - mean(x1)) / std(x1)`, again with
/// constant fitting disabled so ordinary constants cannot supply the exact
/// location and scale.
#[test]
fn recovery_of_the_standardisation_law() {
    let xs: Vec<f64> = vec![0.7, 1.9, 2.3, 3.1, 5.4, 6.6, 8.2, 9.7];
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let std = (xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let rows: Vec<Vec<f64>> = xs.iter().map(|&v| vec![v]).collect();
    let y: Vec<f64> = xs.iter().map(|v| (v - mean) / std).collect();
    let train = sample(rows.clone(), y.clone());
    let test = sample(rows, y);

    let cfg = Config {
        generations: 400,
        islands: 4,
        pop: 300,
        max_complexity: 12,
        fit_iters: 0,
        patience: 0,
        seed: 11,
        metric: Metric::Absolute,
        ops: vec![
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Neg,
            Op::Agg(AggKind::Mean),
            Op::Agg(AggKind::Std),
            Op::Agg(AggKind::Median),
        ],
        ..Default::default()
    };
    let out = run(&cfg, &train, &test);
    let best = out
        .archive
        .items
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).expect("finite errors"))
        .expect("a Pareto member");
    let shown = expr::to_string(&best.tree, &["x1".to_string()]);
    eprintln!(
        "recovery (standardise): error={:.3e} complexity={} agg={} law={}",
        best.error,
        best.complexity,
        contains_agg(&best.tree),
        shown
    );
    assert!(
        best.error < 1e-9,
        "the search did not reach the exact dataset statistics: {shown} (error {})",
        best.error
    );
    assert!(
        contains_agg(&best.tree),
        "the exact law should contain aggregates: {shown}"
    );
}
