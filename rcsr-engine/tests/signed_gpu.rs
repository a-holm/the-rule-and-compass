//! #228: signed bias (`|mean residual|`) — the exact arm, its degeneracy and
//! the metric-aware GPU-fit gate.
//!
//! The objective is `J0(c) = |Σ wᵢ (f(xᵢ;c) − yᵢ)| / Σ wᵢ`. Its optimal set is a
//! codimension-1 manifold at value exactly 0 wherever the weighted mean
//! residual is representable in the model's constants, so the arm is a
//! **constrained least-squares projection** onto that manifold, not a descent
//! rule: `min ||W^{1/2} r||² s.t. Σ wᵢ rᵢ = 0`, solved through its KKT system
//! with two right-hand sides on the same normal matrix. These tests pin the
//! contract the issue asks for: the exact zero-bias case, the non-identifiable
//! model, determinism, every labelled fallback, CPU parity on fixtures, the
//! `--no-fit` no-op, and the metric-aware batch gate.

use rcsr::expr::{self, constant, input, node, Node, Op};
use rcsr::fit::{
    fit_batch_verified, fit_weighted, fit_weighted_metric, signed_finalist, signed_projection,
    BatchStatus, BatchedFitter, SignedProjection,
};
use rcsr::metrics::{score_weighted, Metric};

/// The constant values of `tree`, in the order the fitter numbers them.
fn fitted(tree: &Node) -> Vec<f64> {
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

fn walk(n: &Node, out: &mut Vec<f64>) {
    if n.op == Op::Const {
        out.push(n.val);
    }
    for k in &n.kids {
        walk(k, out);
    }
}

fn preds(tree: &Node, x: &[Vec<f64>]) -> Vec<f64> {
    x.iter().map(|row| expr::eval(tree, row)).collect()
}

fn objective(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
    let p = preds(tree, x);
    score_weighted(&p, y, w, Metric::Signed, expr::count_constants(tree))
}

fn sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let p = preds(tree, x);
    p.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f64>() / x.len() as f64
}

/// `y = 2x + 4 + small deterministic noise`. The intercept makes the weighted
/// mean residual representable, so the manifold exists and the squared fit
/// already sits on it.
fn affine_fixture(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        let noise = ((i * 37) % 7) as f64 * 1e-3;
        x.push(vec![xi]);
        y.push(2.0 * xi + 4.0 + noise);
    }
    (x, y)
}

fn line() -> Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(0.7), input(0)]), constant(0.1)],
    )
}

/// A backend that answers the configured metric explicitly (what a real device
/// backend must do), with one fixed proposal.
struct ProposingBackend(Option<Vec<f64>>);

impl BatchedFitter for ProposingBackend {
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

    fn fit_batch_metric(
        &self,
        trees: &[Node],
        _: &[Vec<f64>],
        _: &[f64],
        _: &[f64],
        _: usize,
        _: Metric,
    ) -> Vec<Option<Vec<f64>>> {
        trees.iter().map(|_| self.0.clone()).collect()
    }
}

/// A backend that only knows squared error and inherits the default
/// `fit_batch_metric`.
struct SquaredOnlyBackend(Vec<f64>);

impl BatchedFitter for SquaredOnlyBackend {
    fn fit_batch(
        &self,
        trees: &[Node],
        _: &[Vec<f64>],
        _: &[f64],
        _: &[f64],
        _: usize,
    ) -> Vec<Option<Vec<f64>>> {
        trees.iter().map(|_| Some(self.0.clone())).collect()
    }
}

/// --- 1. the exact zero-bias case -------------------------------------------

#[test]
fn a_point_on_the_manifold_is_recognised_and_left_untouched() {
    // Exact data, exact intercept: every residual is zero, so `Σ w r` is zero
    // by inspection and the arm must not touch an already feasible point.
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 - 9.5]).collect();
    let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 4.0).collect();
    let mut tree = node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(2.0), input(0)]), constant(4.0)],
    );
    let before = fitted(&tree);
    assert_eq!(objective(&tree, &x, &y, &[]), 0.0);
    assert_eq!(signed_projection(&mut tree, &x, &y, &[], 2), SignedProjection::Projected);
    assert_eq!(fitted(&tree), before, "a feasible start must not be perturbed");
    // And the full arm reports exactly 0.0 for it, bit for bit.
    let value = fit_weighted_metric(&mut tree, &x, &y, &[], 8, Metric::Signed);
    assert_eq!(value, 0.0);
    assert_eq!(value.to_bits(), objective(&tree, &x, &y, &[]).to_bits());
}

#[test]
fn an_intercept_model_reaches_zero_bias_within_the_f64_rounding_scale() {
    let (x, y) = affine_fixture(40);
    let mut tree = line();
    let start = fitted(&tree);
    let value = fit_weighted_metric(&mut tree, &x, &y, &[], 8, Metric::Signed);
    // The reported objective is exactly the metric function's value: never a
    // substituted loss, never a different rounding path.
    assert_eq!(value.to_bits(), objective(&tree, &x, &y, &[]).to_bits());
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(value <= 1e-14 * scale, "bias {value:e} must be at rounding scale");
    assert_ne!(fitted(&tree), start, "the squared warm start moves the constants");
    // Repeated projection never makes the exact objective worse.
    let mut again = tree.clone();
    let before = objective(&again, &x, &y, &[]);
    signed_projection(&mut again, &x, &y, &[], 2);
    assert!(objective(&again, &x, &y, &[]) <= before);
}

#[test]
fn one_projection_step_is_exact_and_is_the_minimum_sse_zero_bias_point() {
    // The report's acceptance test: one KKT step reaches
    // |Σ w r| <= 1e-15 · W · scale on a linear model, from a start far from the
    // manifold. For an affine tree the step is the exact constrained
    // minimiser, and because the least-squares optimum is itself feasible it
    // coincides with the least-squares optimum, which is the documented
    // tie-break on the zero-bias manifold.
    let (x, y) = affine_fixture(64);
    let mut tree = node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(5.0), input(0)]), constant(-9.0)],
    );
    assert!(objective(&tree, &x, &y, &[]) > 1.0, "the start must be off the manifold");
    assert_eq!(
        signed_projection(&mut tree, &x, &y, &[], 1),
        SignedProjection::Projected
    );
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    let mean = objective(&tree, &x, &y, &[]);
    assert!(mean <= 1e-15 * scale, "|mean residual| = {mean:e} must be machine precision");

    let mut squared = line();
    fit_weighted(&mut squared, &x, &y, &[], 8);
    let (a, b) = (sse(&tree, &x, &y), sse(&squared, &x, &y));
    assert!(
        (a - b).abs() <= 1e-9 * (1.0 + b),
        "projection SSE {a} must equal the least-squares optimum {b}"
    );
}

/// --- 2. the non-identifiable model -----------------------------------------

#[test]
fn a_model_that_cannot_move_the_mean_is_labelled_and_kept() {
    // Zero-mean input, no intercept: no constant changes the mean residual, so
    // the bias belongs to the shape and no projection exists.
    let n = 41;
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64 - (n as f64 - 1.0) / 2.0])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] + 7.0).collect();
    let mut tree = node(Op::Mul, vec![constant(1.0), input(0)]);
    let before = fitted(&tree);
    assert_eq!(
        signed_projection(&mut tree, &x, &y, &[], 2),
        SignedProjection::Unidentifiable
    );
    assert_eq!(fitted(&tree), before, "a declined projection must not move constants");
    // The objective stays the data mean: the arm reports the truth instead of
    // pretending to have solved it.
    let value = fit_weighted_metric(&mut tree, &x, &y, &[], 8, Metric::Signed);
    assert!((value - 7.0).abs() < 1e-9, "bias = {value}");
    assert_eq!(value.to_bits(), objective(&tree, &x, &y, &[]).to_bits());
}

#[test]
fn the_weighted_manifold_puts_the_weighted_mean_at_zero() {
    // Weights change the constraint (`Σ w r = 0`), not only the loss: the arm
    // must land on the *weighted* manifold, where the unweighted mean is not
    // zero. A row with weight 0 must not enter the constraint at all.
    let x: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
    let y: Vec<f64> = x.iter().map(|r| 0.5 * r[0] + 2.0 + (r[0] * 0.7).sin()).collect();
    let w: Vec<f64> = (0..12).map(|i| [1.0, 3.0, 0.0, 2.0][i % 4]).collect();
    let mut tree = line();
    let value = fit_weighted_metric(&mut tree, &x, &y, &w, 8, Metric::Signed);
    assert_eq!(value.to_bits(), objective(&tree, &x, &y, &w).to_bits());
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    assert!(value <= 1e-14 * scale, "weighted mean residual {value:e} must vanish");
    let unweighted = objective(&tree, &x, &y, &[]);
    assert!(
        unweighted > 1e-3,
        "the unweighted mean is a different manifold, got {unweighted:e}"
    );
    // A zero-weight row is invisible: changing its target changes nothing.
    let mut muted = y.clone();
    muted[2] = 1e6;
    let mut other = tree.clone();
    let same = fit_weighted_metric(&mut other, &x, &muted, &w, 8, Metric::Signed);
    assert!(same <= 1e-14 * scale, "the muted row leaked into the fit: {same:e}");
}

/// --- 3. determinism --------------------------------------------------------

#[test]
fn the_projection_is_bit_deterministic() {
    let (x, y) = affine_fixture(50);
    let run = || {
        let mut t = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(-3.0), input(0)]), constant(11.0)],
        );
        let value = fit_weighted_metric(&mut t, &x, &y, &[], 8, Metric::Signed);
        (
            fitted(&t).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            value.to_bits(),
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn a_batched_proposal_reaches_the_same_constants_as_the_cpu_arm() {
    let (x, y) = affine_fixture(30);
    let mut batch = vec![line()];
    let out = fit_batch_verified(
        &ProposingBackend(Some(vec![0.5, -1.0])),
        &mut batch,
        &x,
        &y,
        &[],
        8,
        Metric::Signed,
    );
    let mut cpu = line();
    let reference = fit_weighted_metric(&mut cpu, &x, &y, &[], 8, Metric::Signed);
    assert_eq!(out.status, vec![BatchStatus::Gpu]);
    assert_eq!(out.score[0].to_bits(), objective(&batch[0], &x, &y, &[]).to_bits());
    // Same zero-bias point as the CPU arm: both are the constrained
    // least-squares optimum of the same rows.
    for (a, b) in fitted(&batch[0]).iter().zip(&fitted(&cpu)) {
        assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{a} vs {b}");
    }
    assert!(reference <= 1e-13 * y.iter().fold(0.0f64, |m, v| m.max(v.abs())));
}

/// --- 4. fallbacks, each labelled ------------------------------------------

#[test]
fn every_declined_projection_carries_its_label() {
    let (x, y) = affine_fixture(20);
    let mut bare = node(Op::Mul, vec![input(0), input(0)]);
    assert_eq!(
        signed_projection(&mut bare, &x, &y, &[], 2),
        SignedProjection::NoConstants
    );
    let mut bad_y = y.clone();
    bad_y[3] = f64::NAN;
    assert_eq!(
        signed_projection(&mut line(), &x, &bad_y, &[], 2),
        SignedProjection::NonFinite
    );
    let bad_w = vec![-1.0; x.len()];
    assert_eq!(
        signed_projection(&mut line(), &x, &y, &bad_w, 2),
        SignedProjection::NonFinite
    );
    assert_eq!(
        signed_projection(&mut line(), &x, &y[..5], &[], 2),
        SignedProjection::NonFinite
    );
    // Two constants that are the same column: the normal matrix is singular.
    let mut collinear = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(1.0), input(0)]),
            node(Op::Mul, vec![constant(1.0), input(0)]),
        ],
    );
    assert_eq!(
        signed_projection(&mut collinear, &x, &y, &[], 2),
        SignedProjection::IllConditioned
    );
    let labels: Vec<&str> = SignedProjection::ALL.iter().map(|s| s.label()).collect();
    let mut sorted = labels.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), labels.len(), "labels must be unique: {labels:?}");
}

/// --- 5. CPU parity on fixtures --------------------------------------------

#[test]
fn the_reported_objective_is_always_the_metric_function_value() {
    let (x, y) = affine_fixture(33);
    let w: Vec<f64> = (0..x.len())
        .map(|i| if i % 5 == 0 { 0.0 } else { 0.5 + i as f64 })
        .collect();
    for weights in [vec![], w] {
        for tree in [
            line(),
            node(Op::Mul, vec![constant(1.0), input(0)]),
        ] {
            let mut tree = tree;
            let value = fit_weighted_metric(&mut tree, &x, &y, &weights, 8, Metric::Signed);
            assert_eq!(value.to_bits(), objective(&tree, &x, &y, &weights).to_bits());
            assert!(value >= 0.0 && value.is_finite());
        }
    }
}

#[test]
fn the_signed_finalist_refit_returns_the_exact_objective() {
    let (x, y) = affine_fixture(25);
    let mut tree = node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(2.5), input(0)]), constant(0.0)],
    );
    let (value, outcome) = signed_finalist(&mut tree, &x, &y, &[]);
    assert_eq!(outcome, SignedProjection::Projected);
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    assert!(value <= 1e-14 * scale, "bias {value:e}");
    assert_eq!(value.to_bits(), objective(&tree, &x, &y, &[]).to_bits());
}

#[test]
fn the_batch_gate_never_accepts_a_squared_answer_for_signed() {
    // The default `fit_batch_metric` declines every non-squared metric, so a
    // backend that only knows squared error can never answer a signed request.
    let (x, y) = affine_fixture(20);
    let mut tree = vec![line()];
    let out = fit_batch_verified(
        &SquaredOnlyBackend(vec![2.0, 4.0]),
        &mut tree,
        &x,
        &y,
        &[],
        8,
        Metric::Signed,
    );
    assert_eq!(out.status, vec![BatchStatus::Unsupported]);
    assert_eq!(out.score[0].to_bits(), objective(&tree[0], &x, &y, &[]).to_bits());
    let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    assert!(out.score[0] <= 1e-14 * scale, "bias {}", out.score[0]);
}

#[test]
fn a_non_identifiable_proposal_is_labelled_not_identifiable() {
    // Zero-mean input, no intercept: the proposal cannot remove the mean, so
    // the gate must say so instead of accepting it on squared error.
    let n = 21;
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64 - (n as f64 - 1.0) / 2.0])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 4.0).collect();
    let mut tree = vec![node(Op::Mul, vec![constant(1.0), input(0)])];
    let out = fit_batch_verified(
        &ProposingBackend(Some(vec![1.5])),
        &mut tree,
        &x,
        &y,
        &[],
        8,
        Metric::Signed,
    );
    assert_eq!(out.status, vec![BatchStatus::NotIdentifiable]);
    assert!((out.score[0] - 4.0).abs() < 1e-9, "{:?}", out.score);
}

/// --- 7. paired measurement and the cost model ------------------------------
///
/// Ignored: it reports timings rather than asserting them, and the machine is
/// shared with other agents' runs. Run with
/// `cargo test --release --test signed_gpu signed_cost_model -- --ignored --nocapture`.
///
/// The model is per candidate, with `n` rows and `p` constants:
///
/// ```text
/// pre-#228 CPU signed  = LM(iters) + polish(min(iters,12)·p·6 probes)
///                        ≈ (iters + 48p) · n · c_sweep
/// #228 projection arm  = LM(iters) + 1 round (assemble, factor, 2 solves)
///                        ≈ (iters + 1) · n · c_sweep + O(p³)
/// ```
#[test]
#[ignore]
fn signed_cost_model() {
    use std::time::Instant;
    let n = 20_000;
    // A well-conditioned design of `p` affine-in-terminals constants on
    // x in [-pi, pi]: an intercept plus {sin(kx)} and {cos(kx)}, k = 1..3, and
    // x. Every column has a healthy norm, so the projection's rank gate passes
    // and the timing measures the arm rather than a labelled decline.
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![-std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64])
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|r| {
            let v = r[0];
            1.5 * v.sin() - 0.8 * (2.0 * v).sin() + 0.4 * (3.0 * v).cos() + 0.3 * v + 0.7
                + ((v * 100.0) as i64 % 7) as f64 * 1e-4
        })
        .collect();
    fn model_with(p: usize) -> Node {
        let x = input(0);
        let scale = |k: f64| node(Op::Mul, vec![constant(k), x.clone()]);
        let basis: Vec<Node> = vec![
            node(Op::Sin, vec![x.clone()]),
            node(Op::Sin, vec![scale(2.0)]),
            node(Op::Cos, vec![scale(3.0)]),
            x.clone(),
            node(Op::Cos, vec![x.clone()]),
            node(Op::Sin, vec![scale(3.0)]),
            node(Op::Cos, vec![scale(2.0)]),
        ];
        let mut tree = constant(1.0);
        for b in basis.into_iter().take(p - 1) {
            tree = node(Op::Add, vec![tree, node(Op::Mul, vec![constant(1.0), b])]);
        }
        tree
    }
    let cases: [(usize, Node); 4] = [
        (2, model_with(2)),
        (3, model_with(3)),
        (4, model_with(4)),
        (8, model_with(8)),
    ];
    eprintln!("# signed cost model: n={n} rows, fit-iters=8");
    eprintln!("# p  squared_ms  signed_ms  projection_ms  polish_proxy_ms  probes_removed");
    for (p, model) in cases {
        let time = |f: &dyn Fn() -> f64| {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let squared = time(&|| {
            let mut t = model.clone();
            fit_weighted(&mut t, &x, &y, &[], 8)
        });
        let signed = time(&|| {
            let mut t = model.clone();
            fit_weighted_metric(&mut t, &x, &y, &[], 8, Metric::Signed)
        });
        // The removed path's cost, as a proxy: a metric that still takes the
        // coordinate polish pays the same `min(iters,12)·p·6` full-row probes on
        // the same tree. `maximum` is O(n) per probe like `signed`.
        let polished = time(&|| {
            let mut t = model.clone();
            fit_weighted_metric(&mut t, &x, &y, &[], 8, Metric::Maximum)
        });
        // The projection alone, from a start that is already fitted.
        let mut warm = model.clone();
        fit_weighted(&mut warm, &x, &y, &[], 8);
        let projection = time(&|| {
            let mut t = warm.clone();
            signed_projection(&mut t, &x, &y, &[], 2);
            objective(&t, &x, &y, &[])
        });
        eprintln!(
            "# {p:<2} {squared:>11.2} {signed:>10.2} {projection:>14.2} {polished:>16.2} {:>14}",
            48 * p
        );
        // Dominance over the starting point is exact: the arm never reports a
        // worse bias than the squared fit it started from. Print both so the
        // residual (non-zero only when the design is not certified) is visible.
        let mut squared_only = model.clone();
        fit_weighted(&mut squared_only, &x, &y, &[], 8);
        let before = objective(&squared_only, &x, &y, &[]);
        let mut final_tree = model.clone();
        let v = fit_weighted_metric(&mut final_tree, &x, &y, &[], 8, Metric::Signed);
        eprintln!("# {p:<3} bias_before={before:e} bias_after={v:e}");
        assert!(v <= before, "p={p}: the arm must never worsen the bias");
    }
}

/// --- 6. `--no-fit` ---------------------------------------------------------

#[test]
fn no_fit_is_a_true_no_op_for_signed() {
    let (x, y) = affine_fixture(20);
    for weights in [vec![], vec![1.0; 20]] {
        let mut tree = line();
        let before = fitted(&tree);
        let value = fit_weighted_metric(&mut tree, &x, &y, &weights, 0, Metric::Signed);
        assert_eq!(fitted(&tree), before, "--no-fit must not move a constant");
        assert_eq!(value.to_bits(), objective(&tree, &x, &y, &weights).to_bits());
        assert!(value > 0.0, "the generated constants carry a bias: {value}");
    }
}
