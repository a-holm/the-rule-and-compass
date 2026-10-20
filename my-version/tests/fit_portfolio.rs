//! #116: integration tests for the exact constant-fitting portfolio.
//!
//! These exercise the public surface the search and the CLI use: the default
//! arm's bit-identical contract, structure-adaptive dispatch, VarPro, the
//! streaming sufficient-statistics merger, rank/non-finite fallbacks, and the
//! #169 GPU integration verifier.

use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::fit::{
    fit_batch_verified, fit_stream_with, fit_weighted, fit_weighted_strategy, separable_partition,
    BatchStatus, BatchedFitter, FitStrategy, RowSource,
};

fn fitted(tree: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
        if let Some(b) = &n.block {
            walk(&b.body, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

fn separable() -> Node {
    // a*sin(b*x) + c
    node(
        Op::Add,
        vec![
            node(
                Op::Mul,
                vec![
                    constant(0.4),
                    node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                ],
            ),
            constant(0.0),
        ],
    )
}

fn affine() -> Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)],
    )
}

#[test]
fn exact_default_is_bit_identical_to_fit_weighted() {
    let x: Vec<Vec<f64>> = (0..50).map(|i| vec![-2.0 + 0.1 * i as f64]).collect();
    let y: Vec<f64> = x.iter().map(|r| 1.7 * (0.9 * r[0]).sin() + 0.2).collect();
    let mut a = separable();
    let mut b = separable();
    let sa = fit_weighted(&mut a, &x, &y, &[], 8);
    let sb = fit_weighted_strategy(&mut b, &x, &y, &[], 8, FitStrategy::Exact);
    assert_eq!(sa.to_bits(), sb.to_bits());
    let ca: Vec<u64> = fitted(&a).iter().map(|v| v.to_bits()).collect();
    let cb: Vec<u64> = fitted(&b).iter().map(|v| v.to_bits()).collect();
    assert_eq!(ca, cb);
}

#[test]
fn auto_dispatches_linear_and_varpro_without_losing_to_lm() {
    let x: Vec<Vec<f64>> = (0..64).map(|i| vec![-3.0 + 6.0 * i as f64 / 63.0]).collect();
    let y_aff: Vec<f64> = x.iter().map(|r| 3.0 * r[0] + 2.0).collect();
    let mut lin = affine();
    let auto = fit_weighted_strategy(&mut lin, &x, &y_aff, &[], 16, FitStrategy::Auto);
    assert!(auto < 1e-20, "affine law must be solved in one QR: {auto}");

    let y_sep: Vec<f64> = x.iter().map(|r| 2.5 * (1.3 * r[0]).sin() + 0.7).collect();
    assert!(separable_partition(&separable()).is_some());
    let mut vp = separable();
    let a = fit_weighted_strategy(&mut vp, &x, &y_sep, &[], 40, FitStrategy::Auto);
    let mut lm = separable();
    let b = fit_weighted(&mut lm, &x, &y_sep, &[], 40);
    assert!(a <= b * (1.0 + 1e-9) + 1e-12, "auto {a} vs exact {b}");
    let c = fitted(&vp);
    assert!((c[0] - 2.5).abs() < 1e-4 && (c[1] - 1.3).abs() < 1e-4 && (c[2] - 0.7).abs() < 1e-4, "{c:?}");
}

#[test]
fn non_finite_rows_fall_back_and_never_improve() {
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64]).collect();
    let mut y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 1.0).collect();
    y[7] = f64::NAN;
    let mut tree = affine();
    let a = fit_weighted_strategy(&mut tree, &x, &y, &[], 8, FitStrategy::Auto);
    assert!(!a.is_finite(), "a non-finite row must reject the candidate");
    // The tree is left at its carrier constants, not at a half-solved proposal.
    assert_eq!(fitted(&tree), vec![1.0, 0.0]);
}

#[test]
fn rank_deficient_affine_system_uses_exact_fallback() {
    let x: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64]).collect();
    let y = vec![3.0, 2.8, 3.1, 2.9, 3.2, 3.0, 3.1, 2.95, 3.05, 3.0, 3.1, 2.9, 3.0, 3.05, 2.98];
    let tree = node(Op::Add, vec![constant(2.0), constant(1.0)]);
    let mut a = tree.clone();
    let mut b = tree.clone();
    let lm = fit_weighted(&mut a, &x, &y, &[], 8);
    let auto = fit_weighted_strategy(&mut b, &x, &y, &[], 8, FitStrategy::Auto);
    assert_eq!(lm.to_bits(), auto.to_bits());
    assert_eq!(fitted(&a), fitted(&b));
}

struct Generated {
    n: usize,
}

impl RowSource for Generated {
    fn rows(&self) -> usize {
        self.n
    }

    fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
        for i in 0..self.n {
            let x = -3.0 + 6.0 * i as f64 / (self.n - 1) as f64;
            if !f(&[x], 2.0 * (0.9 * x).sin() + 0.3, 1.0) {
                return false;
            }
        }
        true
    }
}

#[test]
fn out_of_core_source_never_materialises_a_design_matrix() {
    // 200k rows arrive one at a time from a source that holds only the count.
    let source = Generated { n: 200_000 };
    let mut tree = separable();
    let sse = fit_stream_with(&mut tree, &source, 12).expect("streaming fit");
    assert!(sse.is_finite() && sse < 1e-12, "sse={sse}");
    let c = fitted(&tree);
    assert!((c[0] - 2.0).abs() < 1e-5 && (c[1] - 0.9).abs() < 1e-5 && (c[2] - 0.3).abs() < 1e-5, "{c:?}");
}

struct FakeGpu(Option<Vec<f64>>);

impl BatchedFitter for FakeGpu {
    fn fit_batch(
        &self,
        trees: &[Node],
        _: &[Vec<f64>],
        _: &[f64],
        _: &[f64],
        _: usize,
    ) -> Vec<Option<Vec<f64>>> {
        trees.iter().map(|_| self.0.clone()).collect()
    }
}

#[test]
fn gpu_verifier_accepts_good_and_rejects_bad_proposals() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0).collect();

    let mut good = vec![affine()];
    let out = fit_batch_verified(&FakeGpu(Some(vec![2.0, 3.0])), &mut good, &x, &y, &[], 8);
    assert_eq!(out.status, vec![BatchStatus::Gpu]);

    for bad in [Some(vec![f64::NAN, 3.0]), Some(vec![2.0]), None] {
        let mut gpu = vec![affine()];
        let out = fit_batch_verified(&FakeGpu(bad), &mut gpu, &x, &y, &[], 8);
        let mut cpu = affine();
        let reference = fit_weighted(&mut cpu, &x, &y, &[], 8);
        assert_eq!(out.sse[0].to_bits(), reference.to_bits());
        assert_eq!(fitted(&gpu[0]), fitted(&cpu));
    }
}