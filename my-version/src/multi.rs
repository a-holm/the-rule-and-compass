//! Multi-output regression with one shared shape (issue #15).
//!
//! The search evolves a single **structure**. Each output gets its own vector
//! of constants, fitted independently on that output's target column; the
//! structure itself is shared. A candidate's combined error is the arithmetic
//! mean of its per-output errors under the configured metric.
//!
//! This is deliberately not a single constant vector serving every output: a
//! shape such as `c0*x1*x2 + c1` can fit `y1 = a*x1*x2` with `c1 = 0` and
//! `y2 = b*x1*x2 + c` with both constants non-zero. Sharing the *shape* is what
//! buys statistical strength; sharing the constants would forbid that.
//!
//! Single-output samples take none of these paths (`Sample::is_multi` is
//! false), so their search remains bit-for-bit unchanged.

use crate::data::Sample;
use crate::expr::{self, Node};
use crate::fit;
use crate::metrics::{score_weighted, Metric};

/// Fit a copy of `shape` to every target of `s`, returning one tree per output
/// in target order. The input tree is never modified.
pub fn fitted_outputs(shape: &Node, s: &Sample, fit_iters: usize) -> Vec<Node> {
    (0..s.target_count())
        .map(|j| {
            let mut tree = shape.clone();
            if fit_iters > 0 {
                // Deliberately **not** simplified afterwards: simplifying a
                // fitted output can delete a constant term that happens to be
                // zero for that output (`c + 0` -> `c`), which would make the
                // outputs look like different structures. The structural
                // identity of the shared shape is the point of this mode.
                fit::fit_weighted(&mut tree, &s.x, s.target(j), &s.w, fit_iters);
            }
            tree
        })
        .collect()
}

/// Score one already-fitted tree against target `j`.
pub fn eval_target(tree: &Node, s: &Sample, metric: Metric, j: usize) -> f64 {
    if s.is_empty() {
        return f64::INFINITY;
    }
    let target = s.target(j);
    if target.len() != s.x.len() {
        return f64::INFINITY;
    }
    let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(tree, row)).collect();
    score_weighted(
        &pred,
        target,
        &s.w,
        metric,
        expr::count_constants(tree),
    )
}

/// Fitted constant values in pre-order, matching [`crate::expr::nth`].
pub fn constants(tree: &Node) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..expr::complexity(tree) {
        if let Some(n) = expr::nth(tree, i) {
            if n.op == expr::Op::Const {
                out.push(n.val);
            }
        }
    }
    out
}

/// Per-output errors of already-fitted trees.
pub fn output_errors(trees: &[Node], s: &Sample, metric: Metric) -> Vec<f64> {
    trees
        .iter()
        .enumerate()
        .map(|(j, tree)| eval_target(tree, s, metric, j))
        .collect()
}

/// The combined error: the mean of the per-output errors.
///
/// An empty front entry (no outputs) or a non-finite output error is reported
/// as `+inf`, so a shape that fails one target can never look good on the mean.
pub fn mean_error(errors: &[f64]) -> f64 {
    if errors.is_empty() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for e in errors {
        if !e.is_finite() {
            return f64::INFINITY;
        }
        sum += e;
    }
    sum / errors.len() as f64
}

/// Fit the shared shape per output on `s` and return the combined training
/// error. This is the value the search and the Pareto archive rank on.
pub fn combined_error(shape: &Node, s: &Sample, metric: Metric, fit_iters: usize) -> f64 {
    let trees = fitted_outputs(shape, s, fit_iters);
    mean_error(&output_errors(&trees, s, metric))
}

/// Fit the shape per output on `train` and score those fitted trees on `test`.
///
/// Constants are **never** fitted on the test rows: that would leak the
/// held-out data into the reported error.
pub fn combined_test_error(
    shape: &Node,
    train: &Sample,
    test: &Sample,
    metric: Metric,
    fit_iters: usize,
) -> f64 {
    let trees = fitted_outputs(shape, train, fit_iters);
    mean_error(&output_errors(&trees, test, metric))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Sample {
        let mut s = Sample::default();
        for i in 0..11 {
            let a = i as f64 - 5.0;
            s.x.push(vec![a, 2.0 * a + 1.0]);
            s.y.push(3.0 * a);
        }
        s.ys = vec![
            s.y.clone(),
            s.y.iter().map(|v| 2.0 * v + 7.0).collect(),
        ];
        s
    }

    #[test]
    fn per_output_constants_share_one_shape() {
        // Shape `c0*x0 + c1` must fit both outputs with different constants.
        let shape = expr::node(
            expr::Op::Add,
            vec![
                expr::node(expr::Op::Mul, vec![expr::constant(1.0), expr::input(0)]),
                expr::constant(0.0),
            ],
        );
        let s = sample();
        let trees = fitted_outputs(&shape, &s, 20);
        assert_eq!(trees.len(), 2);
        let errors = output_errors(&trees, &s, Metric::Absolute);
        assert!(errors.iter().all(|e| *e < 1e-12), "errors: {:?}", errors);
        // Structures are identical; only the constants differ.
        let shape_of = |t: &Node| crate::stability::shape(t);
        assert_eq!(shape_of(&trees[0]), shape_of(&trees[1]));
        assert!(combined_error(&shape, &s, Metric::Absolute, 20) < 1e-12);
    }

    #[test]
    fn a_failing_output_makes_the_mean_infinite() {
        assert_eq!(mean_error(&[0.0, f64::INFINITY]), f64::INFINITY);
        assert_eq!(mean_error(&[]), f64::INFINITY);
        assert_eq!(mean_error(&[1.0, 3.0]), 2.0);
    }
}
