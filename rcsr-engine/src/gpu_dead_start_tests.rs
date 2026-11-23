//! #988: a status-7 fit whose exact f64 start loss is non-finite skips its CPU
//! refit. Device tests skip cleanly without CUDA.

use super::gpu_fold_tests::{config, fit_on_device, sample};
use super::*;
use crate::expr::{constant, input, node, Op};

macro_rules! need_device {
    () => {
        if !crate::gpu::probe().available() {
            crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
            return;
        }
    };
}

/// `ln(-1) + 0.5*x`, `(-0.5)^x` and `cosh(100*x)`: the first two have a
/// non-finite exact start, the third overflows in f32 only.
fn trees() -> Vec<Node> {
    vec![
        node(Op::Add, vec![node(Op::Mul, vec![constant(0.5), input(0)]), node(Op::Ln, vec![constant(-1.0)])]),
        node(Op::Add, vec![node(Op::Pow, vec![constant(-0.5), input(0)]), node(Op::Mul, vec![constant(0.3), input(1)])]),
        node(Op::Add, vec![node(Op::Cosh, vec![node(Op::Mul, vec![constant(100.0), input(0)])]), node(Op::Mul, vec![constant(0.3), input(1)])]),
    ]
}

#[test]
fn skipped_refits_equal_the_status_seven_fits_with_a_non_finite_exact_start() {
    need_device!();
    let train = sample(4096);
    let cfg = config();
    let trees = trees();
    let (g, fitted) = fit_on_device(&trees, &train, &cfg);
    eprintln!("causes {:?} skipped {:?}", g.fit_fallback_by_cause, g.fit_fallback_skipped_by_cause);
    let start_finite: Vec<bool> =
        trees.iter().map(|t| exact_start_finite(t, &train.x, &train.y, &train.w, cfg.metric)).collect();
    assert_eq!(start_finite, vec![false, false, true]);
    let dead = start_finite.iter().filter(|f| !**f).count();
    assert_eq!(g.fit_fallback_skipped_by_cause.get("non_finite_start").copied().unwrap_or(0), dead);
    assert_eq!(g.run_stats().fit_fallback_skipped_by_cause.get("non_finite_start").copied().unwrap_or(0), dead);
    assert_eq!(g.fit_fallback_by_cause.get("non_finite_start").copied().unwrap_or(0), trees.len());
    assert_eq!(format!("{:?}", fitted[0]), format!("{:?}", expr::simplify(&trees[0])));
    assert_eq!(format!("{:?}", fitted[1]), format!("{:?}", expr::simplify(&trees[1])));
}
