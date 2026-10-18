//! Per-row weights and weighted metrics (issue #1).
//!
//! The central property is that weights are **frequency-like**: doubling one
//! row's weight must give exactly the score and the fitted constants of a table
//! in which that row appears twice. That is only true when the weighted mean is
//! normalised by `Σw`, which these tests pin down.

use rcsr::data::{self, Prepared, Sample, SplitSpec};
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::fit;
use rcsr::metrics::{self, score, score_weighted, Metric};
use rcsr::rng::Mt;
use rcsr::search::{self, Config};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// A straight line with two free constants: `a*x + b`.
fn line() -> Node {
    node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(0.5), input(0)]),
            constant(0.0),
        ],
    )
}

/// Fitted constants in left-to-right order, for comparing two fits.
fn constants(tree: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

fn rows(xs: &[f64]) -> Vec<Vec<f64>> {
    xs.iter().map(|&v| vec![v]).collect()
}

fn cli(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The process may refuse the arguments before reading stdin; a broken pipe
    // is therefore expected in those cases.
    let _ = child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes());
    child.wait_with_output().unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

// ---------------------------------------------------------------------------
// Doubling a weight == duplicating the row
// ---------------------------------------------------------------------------

#[test]
fn doubling_a_weight_equals_duplicating_the_row_for_the_metrics() {
    let pred = [0.0, 1.0, 2.5, 4.0];
    let y = [1.0, 2.0, 4.0, 5.0];
    let w = [1.0, 2.0, 1.0, 1.0];

    // The same table with row 1 (weight 2) written twice.
    let dup_pred = [0.0, 1.0, 1.0, 2.5, 4.0];
    let dup_y = [1.0, 2.0, 2.0, 4.0, 5.0];

    for m in [Metric::Absolute, Metric::Squared, Metric::Signed, Metric::LogSquashed] {
        let weighted = score_weighted(&pred, &y, &w, m, 0);
        let duplicated = score(&dup_pred, &dup_y, m);
        assert!(
            approx(weighted, duplicated, 1e-12),
            "{}: weighted {weighted} vs duplicated {duplicated}",
            m.name()
        );
    }

    // Correlation has the same frequency interpretation.
    let weighted = score_weighted(&pred, &y, &w, Metric::Correlation, 0);
    let duplicated = score(&dup_pred, &dup_y, Metric::Correlation);
    assert!(
        approx(weighted, duplicated, 1e-12),
        "correlation: weighted {weighted} vs duplicated {duplicated}"
    );
}

#[test]
fn doubling_a_weight_equals_duplicating_the_row_for_the_fit() {
    // y = 2x + 1; row 1 gets weight 2.
    let x = rows(&[0.0, 1.0, 2.0, 3.0]);
    let y = vec![1.0, 3.0, 5.0, 7.0];
    let w = vec![1.0, 2.0, 1.0, 1.0];

    // The duplicated table: x = [0,1,1,2,3], y = [1,3,3,5,7].
    let dup_x = rows(&[0.0, 1.0, 1.0, 2.0, 3.0]);
    let dup_y = vec![1.0, 3.0, 3.0, 5.0, 7.0];

    let mut weighted = line();
    let mut duplicated = line();
    let weighted_sse = fit::fit_weighted(&mut weighted, &x, &y, &w, 40);
    let duplicated_sse = fit::fit(&mut duplicated, &dup_x, &dup_y, 40);

    let a = constants(&weighted);
    let b = constants(&duplicated);
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    for (p, q) in a.iter().zip(&b) {
        assert!(approx(*p, *q, 1e-8), "constants {a:?} vs {b:?}");
    }
    // The weighted fit returns Σw·e²; the duplicated fit returns the same sum
    // because each copy contributes one more e².
    assert!(
        approx(weighted_sse, duplicated_sse, 1e-8),
        "weighted sse {weighted_sse} vs duplicated sse {duplicated_sse}"
    );
}

// ---------------------------------------------------------------------------
// A zero-weight row has no influence
// ---------------------------------------------------------------------------

#[test]
fn a_zero_weight_row_has_no_influence_on_fit_error_or_constants() {
    let x = rows(&[0.0, 1.0, 2.0, 3.0]);
    let y = vec![1.0, 3.0, 5.0, 7.0];
    let w = vec![1.0, 1.0, 0.0, 1.0];

    let mut clean = line();
    fit::fit_weighted(&mut clean, &x, &y, &w, 40);

    let sample_clean = Sample::weighted(x.clone(), y.clone(), w.clone());
    let e_clean = search::eval_error(&clean, &sample_clean, Metric::Absolute);
    let c_clean = fit::constant_errors_weighted(&clean, &x, &sample_clean.y, &w)
        .expect("two constants and three positive-weight rows");
    assert_eq!(c_clean.dof, 3 - 2);

    // A large finite value, then the two non-finite values that used to slip
    // through the scorer's finiteness guard before the weights were read.
    for wild in [1.0e9, f64::INFINITY, f64::NAN] {
        let mut y_wild = y.clone();
        y_wild[2] = wild;

        let mut wild_tree = line();
        fit::fit_weighted(&mut wild_tree, &x, &y_wild, &w, 40);
        assert_eq!(constants(&clean), constants(&wild_tree), "target {wild}");

        let sample_wild = Sample::weighted(x.clone(), y_wild, w.clone());
        let e_wild = search::eval_error(&wild_tree, &sample_wild, Metric::Absolute);
        assert_eq!(
            e_clean, e_wild,
            "a zero-weight target of {wild} moved the error"
        );

        let c_wild = fit::constant_errors_weighted(&wild_tree, &x, &sample_wild.y, &w)
            .expect("same information");
        assert_eq!(c_clean.values, c_wild.values, "target {wild}");
        assert_eq!(c_wild.dof, 3 - 2);
    }
}

/// A masked row is exactly a deleted row, even when the model's prediction or
/// the target there is undefined (`+inf` or `NaN`). This is the scorer-side
/// counterpart of `fit.rs` skipping `w == 0`.
#[test]
fn a_zero_weight_row_is_invisible_to_every_weighted_metric() {
    let kept_pred = [1.0, 2.0];
    let kept_y = [1.0, 2.5];
    let w = [0.0, 1.0, 1.0];

    for bad in [f64::INFINITY, f64::NAN] {
        let cases = [
            ([bad, 1.0, 2.0], [9.0, 1.0, 2.5], "prediction"),
            ([1.0, 1.0, 2.0], [bad, 1.0, 2.5], "target"),
        ];
        for (pred, y, what) in cases {
            for name in metrics::weighted_metric_names() {
                let m = Metric::parse(name);
                assert_eq!(
                    score_weighted(&pred, &y, &w, m, 0).to_bits(),
                    score(&kept_pred, &kept_y, m).to_bits(),
                    "{name}: a masked {what} of {bad} changed the score"
                );
            }
        }
    }
}

/// The same non-finite prediction on a row that counts must still reject the
/// candidate; only the zero weight can hide it.
#[test]
fn a_non_finite_prediction_on_a_positive_weight_row_is_infeasible() {
    let y = [1.0, 2.5, 4.0];
    let w = [0.0, 1.0, 1.0];

    for bad in [f64::INFINITY, f64::NAN] {
        let pred = [0.5, bad, 4.0];
        for name in metrics::weighted_metric_names() {
            let m = Metric::parse(name);
            assert_eq!(
                score_weighted(&pred, &y, &w, m, 0),
                f64::INFINITY,
                "{name}: a positive-weight prediction of {bad} must be infeasible"
            );
        }
    }
}

/// The fitter already skips the row; pin the equivalence so the two paths
/// cannot drift.
#[test]
fn a_zero_weight_row_is_invisible_to_the_fit_as_a_deleted_row() {
    let x = rows(&[0.0, 1.0, 2.0]);
    let y = [f64::INFINITY, 1.0, 2.5];
    let w = [0.0, 1.0, 1.0];

    let kept_x = rows(&[1.0, 2.0]);
    let kept_y = [1.0, 2.5];

    let mut masked = line();
    let mut kept = line();
    fit::fit_weighted(&mut masked, &x, &y, &w, 40);
    fit::fit(&mut kept, &kept_x, &kept_y, 40);

    assert_eq!(constants(&masked), constants(&kept));
}

// ---------------------------------------------------------------------------
// Validation through the CLI
// ---------------------------------------------------------------------------

const TABLE: &str = "x,y,w\n0,1,1\n1,3,2\n2,5,1\n3,7,1\n";

#[test]
fn a_missing_weight_column_is_an_input_error() {
    let out = cli(
        &["--data", "-", "--target", "y", "--vars", "x", "--weight", "nope"],
        TABLE,
    );
    assert_eq!(out.status.code(), Some(3));
    let err = stderr(&out);
    assert!(err.contains("weight column 'nope'"), "stderr: {err}");
    assert!(err.contains("not found"), "stderr: {err}");
}

#[test]
fn a_non_numeric_weight_column_is_an_input_error() {
    let out = cli(
        &["--data", "-", "--target", "y", "--vars", "x", "--weight", "w"],
        "x,y,w\n0,1,1\n1,3,2\n2,5,abc\n3,7,1\n",
    );
    assert_eq!(out.status.code(), Some(3));
    let err = stderr(&out);
    assert!(err.contains("weight column 'w'"), "stderr: {err}");
    assert!(err.contains("not numeric"), "stderr: {err}");
}

#[test]
fn a_negative_weight_names_the_column_and_the_count() {
    let out = cli(
        &["--data", "-", "--target", "y", "--vars", "x", "--weight", "w"],
        "x,y,w\n0,1,1\n1,3,-2\n2,5,1\n3,7,-1\n",
    );
    assert_eq!(out.status.code(), Some(3));
    let err = stderr(&out);
    assert!(err.contains("weight column 'w'"), "stderr: {err}");
    assert!(err.contains("2 negative"), "stderr: {err}");
}

#[test]
fn a_non_finite_weight_names_the_column_and_the_count() {
    let out = cli(
        &["--data", "-", "--target", "y", "--vars", "x", "--weight", "w"],
        "x,y,w\n0,1,1\n1,3,inf\n2,5,1\n3,7,1\n",
    );
    assert_eq!(out.status.code(), Some(3));
    let err = stderr(&out);
    assert!(err.contains("weight column 'w'"), "stderr: {err}");
    assert!(err.contains("1 non-finite"), "stderr: {err}");
}

#[test]
fn an_all_zero_weight_column_is_an_input_error() {
    let out = cli(
        &["--data", "-", "--target", "y", "--vars", "x", "--weight", "w"],
        "x,y,w\n0,1,0\n1,3,0\n2,5,0\n3,7,0\n",
    );
    assert_eq!(out.status.code(), Some(3));
    let err = stderr(&out);
    assert!(err.contains("weight column 'w'"), "stderr: {err}");
    assert!(err.contains("all zero"), "stderr: {err}");
}

#[test]
fn an_unsupported_metric_refuses_weights_and_lists_the_supported_ones() {
    let out = cli(
        &[
            "--data", "-", "--target", "y", "--vars", "x", "--weight", "w",
            "--metric", "median",
        ],
        TABLE,
    );
    assert_eq!(out.status.code(), Some(2));
    let err = stderr(&out);
    assert!(err.contains("median"), "stderr: {err}");
    assert!(err.contains("supported weighted metrics"), "stderr: {err}");
    assert!(err.contains("absolute"), "stderr: {err}");
    assert!(err.contains("squared"), "stderr: {err}");
    assert!(err.contains("correlation"), "stderr: {err}");
}

// ---------------------------------------------------------------------------
// The weight column reaches the engine and is reported
// ---------------------------------------------------------------------------

fn weighted_table() -> data::Table {
    let mut csv = String::from("t,x,w,y\n");
    for i in 0..31 {
        let x = i as f64;
        let w = 1 + (i % 3); // a non-trivial but clean weighting
        csv.push_str(&format!("{},{},{},{}\n", i, x, w, 2.0 * x + 1.0));
    }
    data::parse_table(&csv).unwrap()
}

fn prepare_weighted(t: &data::Table) -> Prepared {
    let spec = SplitSpec {
        mode: data::SplitMode::Temporal,
        time_col: Some(0),
        test_frac: 0.25,
        ..Default::default()
    };
    data::prepare_full_weighted_with(
        t,
        &["x".into()],
        "y",
        Some("w"),
        &spec,
        &mut Mt::new(1),
        &data::PrepSpec {
            center: data::Center::None,
            scale: data::Scale::None,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn a_weighted_search_still_recovers_a_clean_law() {
    let p = prepare_weighted(&weighted_table());
    assert_eq!(p.train.w.len(), p.train.len());
    assert!(p.train.w.iter().all(|&v| v > 0.0));

    let cfg = Config {
        generations: 80,
        islands: 1,
        pop: 120,
        seed: 1,
        metric: Metric::Absolute,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Div],
        ..Default::default()
    };
    let out = search::run(&cfg, &p.train, &p.test);
    let best = out
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-9, "best weighted test error = {best}");
}

#[test]
fn json_echoes_the_weight_column_and_its_summary() {
    let out = cli(
        &[
            "--data", "-", "--target", "y", "--vars", "x", "--weight", "w",
            "--generations", "1", "--pop", "4", "--fit-iters", "0", "--json",
            "--test-frac", "0",
        ],
        TABLE,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"weight_column\":\"w\""),
        "stdout: {stdout}"
    );
    assert!(stdout.contains("\"weight_sum\":5"), "stdout: {stdout}");
    assert!(stdout.contains("\"weight_rows\":4"), "stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// The unweighted path is unchanged
// ---------------------------------------------------------------------------

#[test]
fn unweighted_scores_keep_their_known_values() {
    let p = [0.0, 3.0, 0.0];
    let y = [1.0, 1.0, 3.0];
    assert!(approx(score(&p, &y, Metric::Absolute), 2.0, 1e-12));
    assert!(approx(score(&p, &y, Metric::Squared), 14.0 / 3.0, 1e-12));
    // And the weighted entry point with no weights is the same function.
    assert_eq!(
        score_weighted(&p, &y, &[], Metric::Squared, 0),
        score(&p, &y, Metric::Squared)
    );
    assert!(metrics::weights_supported(Metric::Squared));
}
