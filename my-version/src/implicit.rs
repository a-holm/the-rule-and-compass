//! Implicit relations `f(x1, …, xn, y) = 0` (issue #14).
//!
//! In an explicit fit the search models `y = g(x)`. An implicit relation is a
//! level set: the law is `f(p) = 0`, and the data are points `p` that should lie
//! on it. A circle `x² + y² = 1` is the canonical example — no single-valued
//! `y = g(x)` can express it, while `f(x, y) = x² + y² − 1` can.
//!
//! The CLI implements the model by **extending the input list with the target
//! column** (`data::make_implicit`) and setting the target to the constant 0.
//! The expression tree, evaluator, mutation, crossover and archive are
//! untouched; the search simply fits `f(all inputs) = 0`.
//!
//! Two metrics score such a candidate:
//!
//! | metric | value |
//! |---|---|
//! | `implicit` (alias `implicit_derivative`) | mean over rows of `|f(p)| / ||∇f(p)||` |
//! | `implicit_residual` | mean over rows of `|f(p)|` |
//!
//! The first is the **first-order distance from a point to the level set**:
//! moving from `p` along the gradient by `t` changes `f` by about `t·||∇f(p)||`,
//! so the step that reaches `f = 0` is `t = |f(p)| / ||∇f(p)||` (the
//! Newton step). It is scale-free: multiplying `f` by any non-zero constant
//! leaves the metric unchanged, and it prefers a steep, well-defined level set
//! to a flat function that happens to be small at the data.
//!
//! `∇f` comes from the existing adjoint (reverse-mode) sweep in
//! [`crate::fit::input_gradients`]: one backward pass at each row, seeded at the
//! root and accumulated at every `Input` leaf, so all partial derivatives cost
//! `O(nodes)` per row and no finite differences are involved.
//!
//! # The guard
//!
//! If `||∇f(p)||` is zero, not finite, or the row evaluates to a non-finite
//! value, the distance is undefined. Such a row makes the whole candidate
//! **infeasible** (`+inf`), never a lucky zero: `f ≡ 0` has `|f| = 0` but no
//! level set to speak of, and would otherwise score a perfect fit.
//!
//! # Constant fitting
//!
//! The implicit-derivative metric changes how constants are fitted. The
//! ordinary least-squares fit of `f` against the zero target would shrink every
//! multiplicative constant towards zero (the trivial `f ≡ 0`), collapsing the
//! level set. Instead [`fit`] minimises the **signed** geometric residual
//! `f(p) / ||∇f(p)||` in a small Levenberg–Marquardt loop, the same quantity
//! the metric scores (squared instead of absolute, as least squares requires).
//! The plain `implicit_residual` metric keeps the historical fit against the
//! zero target, i.e. `Σ w·f²`; it has the trivial solution `f ≡ 0` by
//! construction, which is why the derivative metric is the recommended one.

use crate::data::Sample;
use crate::expr::{self, Node};
use crate::fit;
use crate::metrics::Metric;

/// A level-set distance is rejected when this many evaluations/derivatives are
/// non-finite, or when the gradient norm is not strictly positive.
#[inline]
fn finite_positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// The signed first-order distance from `p` to the level set `f = 0`:
/// `f(p) / ||∇f(p)||`. `None` when the candidate is undefined or the gradient
/// vanishes at `p` (the guard). The sign is the direction in which `f` grows,
/// which is what the least-squares fitter needs; the metric uses the absolute
/// value.
pub fn signed_distance(tree: &Node, row: &[f64]) -> Option<f64> {
    signed_distance_with(tree, &mut fit::InputGradients::new(tree), row)
}

/// [`signed_distance`] on a prepared `grads` that was built from `tree` (with
/// the same constants).
fn signed_distance_with(
    tree: &Node,
    grads: &mut fit::InputGradients,
    row: &[f64],
) -> Option<f64> {
    let f = expr::eval(tree, row);
    if !f.is_finite() {
        return None;
    }
    let grad = grads.gradient(row);
    let mut norm_sq = 0.0;
    for g in grad {
        if !g.is_finite() {
            return None;
        }
        norm_sq += g * g;
    }
    let norm = norm_sq.sqrt();
    // Guard ||∇f|| = 0 (and the non-finite cases above): a constant candidate
    // has no level set, so it must be infeasible rather than perfect.
    if !finite_positive(norm) {
        return None;
    }
    let d = f / norm;
    if d.is_finite() {
        Some(d)
    } else {
        None
    }
}

/// The distance from `p` to the level set `f = 0`: `|f(p)| / ||∇f(p)||`.
///
/// `+inf` when the candidate is undefined or the guard rejects the row.
pub fn distance(tree: &Node, row: &[f64]) -> f64 {
    distance_with(tree, &mut fit::InputGradients::new(tree), row)
}

fn distance_with(tree: &Node, grads: &mut fit::InputGradients, row: &[f64]) -> f64 {
    match signed_distance_with(tree, grads, row) {
        Some(d) => d.abs(),
        None => f64::INFINITY,
    }
}

/// The plain implicit residual `|f(p)|`; `+inf` when `f` is not finite.
pub fn residual(tree: &Node, row: &[f64]) -> f64 {
    let f = expr::eval(tree, row);
    if f.is_finite() {
        f.abs()
    } else {
        f64::INFINITY
    }
}

/// Score a candidate on an implicit sample.
///
/// `metric` must be one of the implicit metrics. The result is the weighted
/// mean `Σ wᵢ·dᵢ / Σ wᵢ` of the per-row distance `dᵢ` (an empty weight vector
/// means every row weighs 1, exactly as elsewhere in the engine). A
/// non-finite distance on any positive-weight row makes the whole candidate
/// `+inf`.
pub fn score(tree: &Node, s: &Sample, metric: Metric) -> f64 {
    if s.is_empty() {
        return f64::INFINITY;
    }
    score_rows(tree, &s.x, &s.w, metric)
}

/// [`score`] on bare rows and weights (empty `w` means every row weighs 1).
pub fn score_rows(tree: &Node, x: &[Vec<f64>], w: &[f64], metric: Metric) -> f64 {
    if x.is_empty() {
        return f64::INFINITY;
    }
    let mut grads = fit::InputGradients::new(tree);
    let mut total = 0.0;
    let mut sum = 0.0;
    for (i, row) in x.iter().enumerate() {
        let w = w.get(i).copied().unwrap_or(1.0);
        if !w.is_finite() || w < 0.0 {
            return f64::INFINITY;
        }
        if w == 0.0 {
            continue;
        }
        let d = match metric {
            Metric::ImplicitDerivative => distance_with(tree, &mut grads, row),
            Metric::ImplicitResidual => residual(tree, row),
            _ => return f64::INFINITY,
        };
        if !d.is_finite() {
            return f64::INFINITY;
        }
        total += w;
        sum += w * d;
    }
    if !finite_positive(total) {
        // Every row had weight zero: no information, not a free zero.
        return f64::INFINITY;
    }
    sum / total
}

/// The per-row signed residuals used by [`fit`]; a zero-weight row contributes
/// `0` and never influences the constants.
fn residuals(
    tree: &Node,
    grads: &mut fit::InputGradients,
    x: &[Vec<f64>],
    w: &[f64],
) -> Option<Vec<f64>> {
    if !w.is_empty() && w.len() != x.len() {
        return None;
    }
    let mut out = Vec::with_capacity(x.len());
    for (i, row) in x.iter().enumerate() {
        let wi = w.get(i).copied().unwrap_or(1.0);
        if !wi.is_finite() || wi < 0.0 {
            return None;
        }
        if wi == 0.0 {
            out.push(0.0);
            continue;
        }
        let d = signed_distance_with(tree, grads, row)?;
        out.push(wi.sqrt() * d);
    }
    Some(out)
}

fn cost(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}

/// Pre-order node indices of the constant leaves, matching [`expr::nth`].
fn const_positions(n: &Node, out: &mut Vec<usize>, counter: &mut usize) {
    let mine = *counter;
    *counter += 1;
    if n.op == expr::Op::Const {
        out.push(mine);
    }
    for k in &n.kids {
        const_positions(k, out, counter);
    }
}

fn constants(tree: &Node) -> Vec<f64> {
    let mut positions = Vec::new();
    let mut counter = 0;
    const_positions(tree, &mut positions, &mut counter);
    positions
        .iter()
        .filter_map(|&pos| expr::nth(tree, pos).map(|n| n.val))
        .collect()
}

fn set_constants(tree: &mut Node, values: &[f64]) {
    let mut positions = Vec::new();
    let mut counter = 0;
    const_positions(tree, &mut positions, &mut counter);
    for (slot, &pos) in positions.iter().enumerate() {
        if let Some(n) = expr::nth_mut(tree, pos) {
            n.val = values[slot];
        }
    }
}

/// Solve `(A + λ·diag(A) + 1e-12·I) δ = b` by Gaussian elimination with partial
/// pivoting. `None` when the system is numerically singular.
fn solve(a: &[Vec<f64>], b: &[f64], lambda: f64, k: usize) -> Option<Vec<f64>> {
    let mut m = vec![vec![0.0; k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            m[i][j] = a[i][j];
        }
        m[i][i] = m[i][i] * (1.0 + lambda) + 1e-12;
        m[i][k] = b[i];
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        for r in (col + 1)..k {
            let factor = m[r][col] / m[col][col];
            if factor != 0.0 {
                for c in col..=k {
                    m[r][c] -= factor * m[col][c];
                }
            }
        }
    }
    let mut d = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = m[i][k];
        for j in (i + 1)..k {
            s -= m[i][j] * d[j];
        }
        d[i] = s / m[i][i];
    }
    Some(d)
}

/// A tree prepared for repeated residual evaluation: one working copy whose
/// constants are swapped in place, and one flat form built once.
struct Level {
    tree: Node,
    grads: fit::InputGradients,
}

impl Level {
    fn new(tree: &Node) -> Self {
        Level {
            grads: fit::InputGradients::new(tree),
            tree: tree.clone(),
        }
    }

    fn set(&mut self, c: &[f64]) {
        set_constants(&mut self.tree, c);
        self.grads.set_constants(c);
    }

    fn residuals(&mut self, c: &[f64], x: &[Vec<f64>], w: &[f64]) -> Option<Vec<f64>> {
        self.set(c);
        residuals(&self.tree, &mut self.grads, x, w)
    }
}

fn rss_of(tree: &Node, x: &[Vec<f64>], w: &[f64]) -> f64 {
    match residuals(tree, &mut fit::InputGradients::new(tree), x, w) {
        Some(r) => cost(&r),
        None => f64::INFINITY,
    }
}

/// The residuals and the Jacobian at `c` from three sweeps per row.
///
/// `∂rᵢ/∂cⱼ = (√wᵢ/N)·[aⱼ − d·Σₖ eₖ·hₖⱼ]` with `a = ∂f/∂c`, `N = ||∇f||`,
/// `d = f/N`, `e = ∇f/N` and `h = ∂²f/∂p∂c`. The sum is the derivative of `a`
/// along the unit normal `e`, taken by a central difference of the constant
/// adjoint at `p ± ε·e`, so the cost does not grow with the constant count.
/// `None` when any row, or any shifted point, is undefined.
fn directional_jacobian(
    level: &mut Level,
    c: &[f64],
    x: &[Vec<f64>],
    w: &[f64],
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    if !w.is_empty() && w.len() != x.len() {
        return None;
    }
    level.set(c);
    let k = c.len();
    let mut r = Vec::with_capacity(x.len());
    let mut jac = Vec::with_capacity(x.len());
    let mut a = vec![0.0; k];
    let mut ap = vec![0.0; k];
    let mut am = vec![0.0; k];
    let mut unit = Vec::new();
    let mut shifted = Vec::new();
    for (i, row) in x.iter().enumerate() {
        let wi = w.get(i).copied().unwrap_or(1.0);
        if !wi.is_finite() || wi < 0.0 {
            return None;
        }
        if wi == 0.0 {
            r.push(0.0);
            jac.push(vec![0.0; k]);
            continue;
        }
        let f = expr::eval(&level.tree, row);
        if !f.is_finite() {
            return None;
        }
        let grad = level.grads.both_gradients(row, &mut a);
        let mut norm_sq = 0.0;
        for g in grad {
            if !g.is_finite() {
                return None;
            }
            norm_sq += g * g;
        }
        let norm = norm_sq.sqrt();
        if !finite_positive(norm) {
            return None;
        }
        let d = f / norm;
        if !d.is_finite() {
            return None;
        }
        unit.clear();
        unit.extend(grad.iter().map(|g| g / norm));
        let scale = row.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let eps = 1e-6 * (1.0 + scale);
        shifted.clear();
        shifted.extend(row.iter().zip(&unit).map(|(p, e)| p + eps * e));
        level.grads.constant_gradient(&shifted, &mut ap);
        shifted.clear();
        shifted.extend(row.iter().zip(&unit).map(|(p, e)| p - eps * e));
        level.grads.constant_gradient(&shifted, &mut am);
        let sw = wi.sqrt();
        r.push(sw * d);
        let mut jr = Vec::with_capacity(k);
        for j in 0..k {
            let h = (ap[j] - am[j]) / (2.0 * eps);
            jr.push(sw / norm * (a[j] - d * h));
        }
        jac.push(jr);
    }
    Some((r, jac))
}

/// Fit the constants of `tree` to minimise `Σ wᵢ·(f(pᵢ)/||∇f(pᵢ)||)²`.
///
/// This is the constant-fitting companion of the implicit-derivative metric:
/// the same signed geometric residual the metric measures, squared so that the
/// existing Levenberg–Marquardt idea applies. The Jacobian with respect to the
/// constants comes from [`directional_jacobian`]: three sweeps per row, whatever
/// the number of constants, with the derivative of `||∇f||` covered by a
/// directional difference instead of a second-order adjoint.
///
/// The fit is scale-free: multiplying the whole expression by a constant does
/// not change the residual, so the fitter does not shrink the level set to the
/// trivial `f ≡ 0`. A candidate that is infeasible under the guard in
/// [`signed_distance`] is left unchanged and reports `+inf`.
///
/// Returns the final sum of squared geometric residuals.
pub fn fit(tree: &mut Node, x: &[Vec<f64>], w: &[f64], iters: usize) -> f64 {
    fit_with(tree, x, w, iters, directional_jacobian)
}

/// Residuals and their Jacobian with respect to the constants, at `c`.
type Jacobian =
    fn(&mut Level, &[f64], &[Vec<f64>], &[f64]) -> Option<(Vec<f64>, Vec<Vec<f64>>)>;

fn fit_with(
    tree: &mut Node,
    x: &[Vec<f64>],
    w: &[f64],
    iters: usize,
    jacobian: Jacobian,
) -> f64 {
    if x.is_empty() {
        return f64::INFINITY;
    }
    let initial = constants(tree);
    let k = initial.len();
    if k == 0 {
        return rss_of(tree, x, w);
    }
    // #135: `--no-fit` / `--fit-iters 0` means no fitting at all, including
    // the implicit-derivative path.
    if iters == 0 {
        return rss_of(tree, x, w);
    }
    let mut level = Level::new(tree);
    let base_r = match level.residuals(&initial, x, w) {
        Some(r) => r,
        None => return f64::INFINITY,
    };
    let mut best_c = initial;
    let initial_sse = cost(&base_r);
    let mut best_sse = initial_sse;
    let mut c = best_c.clone();
    let mut lambda = 1e-3;
    let n = x.len();

    for _ in 0..iters.max(1) {
        // A row that is undefined under a perturbation gives no usable step:
        // stop rather than fit a discontinuous model.
        let (r, jac) = match jacobian(&mut level, &c, x, w) {
            Some(v) => v,
            None => break,
        };
        let mut ok = true;
        let mut jtj = vec![vec![0.0; k]; k];
        let mut jtr = vec![0.0; k];
        for i in 0..n {
            for a in 0..k {
                let ga = jac[i][a];
                if !ga.is_finite() {
                    ok = false;
                    break;
                }
                jtr[a] += ga * r[i];
                for b in 0..=a {
                    jtj[a][b] += ga * jac[i][b];
                }
            }
            if !ok {
                break;
            }
        }
        if !ok {
            break;
        }
        for a in 0..k {
            for b in 0..a {
                jtj[b][a] = jtj[a][b];
            }
        }
        let rhs: Vec<f64> = jtr.iter().map(|v| -v).collect();
        let step = match solve(&jtj, &rhs, lambda, k) {
            Some(s) => s,
            None => {
                lambda *= 6.0;
                if lambda > 1e9 {
                    break;
                }
                continue;
            }
        };
        let trial: Vec<f64> = c
            .iter()
            .zip(&step)
            .map(|(v, d)| (v + d).clamp(-1e6, 1e6))
            .collect();
        match level.residuals(&trial, x, w) {
            Some(rt) => {
                let sse = cost(&rt);
                if sse < best_sse {
                    best_sse = sse;
                    best_c = trial.clone();
                    c = trial;
                    lambda = (lambda / 8.0).max(1e-12);
                    if best_sse == 0.0 {
                        break;
                    }
                } else {
                    lambda *= 6.0;
                    if lambda > 1e9 {
                        break;
                    }
                }
            }
            None => {
                lambda *= 6.0;
                if lambda > 1e9 {
                    break;
                }
            }
        }
    }

    if best_sse < initial_sse {
        set_constants(tree, &best_c);
    }
    best_sse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    fn circle() -> Sample {
        let mut x = Vec::new();
        for i in 0..16 {
            let a = 2.0 * std::f64::consts::PI * i as f64 / 16.0;
            x.push(vec![a.cos(), a.sin()]);
        }
        Sample {
            x,
            y: vec![0.0; 16],
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        }
    }

    #[test]
    fn level_set_distance_is_the_newton_step_and_scale_free() {
        let s = circle();
        let f = node(
            Op::Sub,
            vec![
                node(
                    Op::Add,
                    vec![
                        node(Op::Square, vec![input(0)]),
                        node(Op::Square, vec![input(1)]),
                    ],
                ),
                constant(1.0),
            ],
        );
        for row in &s.x {
            assert!(distance(&f, row) < 1e-12, "on the circle");
        }
        let scaled = node(Op::Mul, vec![constant(7.0), f.clone()]);
        for row in &s.x {
            assert_eq!(
                distance(&f, row).to_bits(),
                distance(&scaled, row).to_bits(),
                "the metric is scale-free"
            );
        }
    }

    #[test]
    fn zero_gradient_is_infeasible_not_perfect() {
        let s = circle();
        let zero = constant(0.0);
        assert_eq!(score(&zero, &s, Metric::ImplicitDerivative), f64::INFINITY);
        assert_eq!(score(&constant(3.0), &s, Metric::ImplicitDerivative), f64::INFINITY);
        // The plain residual does call f = 0 perfect: documented, and why the
        // derivative metric is the recommended choice.
        assert_eq!(score(&zero, &s, Metric::ImplicitResidual), 0.0);
    }

    #[test]
    fn fit_recovers_the_radius_without_collapsing_the_scale() {
        let s = circle();
        let mut f = node(
            Op::Sub,
            vec![
                node(
                    Op::Add,
                    vec![
                        node(Op::Square, vec![input(0)]),
                        node(Op::Square, vec![input(1)]),
                    ],
                ),
                constant(4.0),
            ],
        );
        fit(&mut f, &s.x, &s.w, 20);
        let c = -expr::eval(&f, &[0.0, 0.0]);
        assert!((c - 1.0).abs() < 1e-6, "radius = {c}");
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Points near the curve `((x - cx) / a)^2 + ((y - cy) / b)^2 = 1`.
    fn ellipse_rows(n: usize, cx: f64, cy: f64, a: f64, b: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let t = 2.0 * std::f64::consts::PI * lcg(&mut s);
                let k = 1.0 + 0.01 * (lcg(&mut s) - 0.5);
                vec![cx + k * a * t.cos(), cy + k * b * t.sin()]
            })
            .collect()
    }

    fn sq(i: usize) -> Node {
        node(Op::Square, vec![input(i)])
    }

    fn weighted_sum(terms: Vec<Node>, c: f64) -> Node {
        let mut it = terms.into_iter();
        let first = it.next().unwrap();
        let sum = it.fold(first, |acc, t| node(Op::Add, vec![acc, t]));
        node(Op::Sub, vec![sum, constant(c)])
    }

    /// Fixture trees and their rows: circle, ellipse, shifted circle, and a
    /// transcendental form, each with and without weights.
    fn fixtures() -> Vec<(Node, Vec<Vec<f64>>, Vec<f64>)> {
        let mul = |c: f64, t: Node| node(Op::Mul, vec![constant(c), t]);
        let shifted = |c: f64, i: usize| {
            node(Op::Square, vec![node(Op::Sub, vec![input(i), constant(c)])])
        };
        let trees = vec![
            (
                weighted_sum(vec![sq(0), sq(1)], 4.0),
                ellipse_rows(120, 0.0, 0.0, 1.0, 1.0, 1),
            ),
            (
                weighted_sum(vec![mul(1.0, sq(0)), mul(2.0, sq(1))], 3.0),
                ellipse_rows(120, 0.0, 0.0, 2.0, 1.0, 2),
            ),
            (
                weighted_sum(vec![shifted(0.5, 0), shifted(0.5, 1)], 2.0),
                ellipse_rows(150, 1.0, 0.0, 2.0, 2.0, 3),
            ),
            (
                node(
                    Op::Sub,
                    vec![
                        node(
                            Op::Add,
                            vec![
                                node(Op::Sin, vec![mul(0.9, input(0))]),
                                mul(1.2, input(1)),
                            ],
                        ),
                        constant(0.3),
                    ],
                ),
                ellipse_rows(100, 0.0, 0.0, 1.5, 1.5, 4),
            ),
        ];
        let mut out = Vec::new();
        for (tree, x) in trees {
            let mut s = 99;
            let w: Vec<f64> = x.iter().map(|_| 0.5 + lcg(&mut s)).collect();
            out.push((tree.clone(), x.clone(), Vec::new()));
            out.push((tree, x, w));
        }
        out
    }

    fn fnv(h: &mut u64, v: f64) {
        for b in v.to_bits().to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    }

    /// Hash of every score, signed distance, fitted constant and fit error.
    fn fixture_hash() -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for (tree, x, w) in fixtures() {
            let s = Sample {
                x: x.clone(),
                y: vec![0.0; x.len()],
                w: w.clone(),
                ys: Vec::new(),
            };
            for m in [Metric::ImplicitDerivative, Metric::ImplicitResidual] {
                fnv(&mut h, score(&tree, &s, m));
            }
            for row in &x {
                fnv(&mut h, signed_distance(&tree, row).unwrap_or(f64::NAN));
            }
            for iters in [0, 1, 8, 20] {
                let mut t = tree.clone();
                fnv(&mut h, fit_with(&mut t, &x, &w, iters, central_jacobian));
                for c in constants(&t) {
                    fnv(&mut h, c);
                }
            }
        }
        h
    }

    #[test]
    fn implicit_fixture_bits_are_pinned() {
        assert_eq!(format!("{:016x}", fixture_hash()), "9926197398bf2e00");
    }

    #[test]
    fn implicit_derivative_refit_through_fit_weighted_metric_moves_constants() {
        let x = ellipse_rows(120, 0.0, 0.0, 1.0, 1.0, 1);
        let y = vec![0.0; x.len()];
        let before = weighted_sum(vec![sq(0), sq(1)], 4.0);
        let mut t = before.clone();
        let score = fit::fit_weighted_metric(&mut t, &x, &y, &[], 20, Metric::ImplicitDerivative);
        assert!(score.is_finite(), "score = {score}");
        let s = Sample { x: x.clone(), y, w: Vec::new(), ys: Vec::new() };
        assert!(score < super::score(&before, &s, Metric::ImplicitDerivative));
        let c = -expr::eval(&t, &[0.0, 0.0]);
        assert!((c - 1.0).abs() < 0.01, "radius squared = {c}");
    }

    #[test]
    #[ignore]
    fn implicit_fit_timing_report() {
        let x = ellipse_rows(1500, 1.0, 0.0, 2.0, 2.0, 7);
        let w: Vec<f64> = Vec::new();
        let tree = weighted_sum(
            vec![
                node(Op::Square, vec![node(Op::Sub, vec![input(0), constant(0.5)])]),
                node(Op::Square, vec![node(Op::Sub, vec![input(1), constant(0.5)])]),
            ],
            2.0,
        );
        let s = Sample { x: x.clone(), y: vec![0.0; x.len()], w: Vec::new(), ys: Vec::new() };
        let reps = 20;
        let t0 = std::time::Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += score(&tree, &s, Metric::ImplicitDerivative);
        }
        let score_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t1 = std::time::Instant::now();
        let mut last = Vec::new();
        for _ in 0..reps {
            let mut t = tree.clone();
            acc += fit(&mut t, &x, &w, 20);
            last = constants(&t);
        }
        let fit_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("TIMING score_ms={score_ms:.3} fit20_ms={fit_ms:.3} acc={acc:e} c={last:?}");
    }

    /// Central differences of the signed residual, `h = 1e-6·(1 + |cⱼ|)`.
    fn central_jacobian(
        level: &mut Level,
        c: &[f64],
        x: &[Vec<f64>],
        w: &[f64],
    ) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let r = level.residuals(c, x, w)?;
        let k = c.len();
        let mut jac = vec![vec![0.0; k]; x.len()];
        for j in 0..k {
            let h = 1e-6 * (1.0 + c[j].abs());
            let mut cp = c.to_vec();
            let mut cm = c.to_vec();
            cp[j] += h;
            cm[j] -= h;
            let rp = level.residuals(&cp, x, w)?;
            let rm = level.residuals(&cm, x, w)?;
            for i in 0..x.len() {
                jac[i][j] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        Some((r, jac))
    }

    /// Double-double arithmetic (about 32 significant digits) for the
    /// reference Jacobian.
    #[derive(Clone, Copy)]
    struct Dd(f64, f64);

    impl Dd {
        fn from(v: f64) -> Dd {
            Dd(v, 0.0)
        }
        fn two_sum(a: f64, b: f64) -> Dd {
            let s = a + b;
            let bb = s - a;
            Dd(s, (a - (s - bb)) + (b - bb))
        }
        fn add(self, o: Dd) -> Dd {
            let s = Dd::two_sum(self.0, o.0);
            let t = Dd::two_sum(self.1, o.1);
            let u = Dd::two_sum(s.0, s.1 + t.0);
            Dd::two_sum(u.0, u.1 + t.1)
        }
        fn neg(self) -> Dd {
            Dd(-self.0, -self.1)
        }
        fn sub(self, o: Dd) -> Dd {
            self.add(o.neg())
        }
        fn mul(self, o: Dd) -> Dd {
            let p = self.0 * o.0;
            let e = self.0.mul_add(o.0, -p);
            let e = e + (self.0 * o.1 + self.1 * o.0);
            Dd::two_sum(p, e)
        }
        fn div(self, o: Dd) -> Dd {
            let q1 = self.0 / o.0;
            let r = self.sub(o.mul(Dd::from(q1)));
            let q2 = r.0 / o.0;
            let r = r.sub(o.mul(Dd::from(q2)));
            let q3 = r.0 / o.0;
            Dd::two_sum(q1, q2).add(Dd::from(q3))
        }
        fn sqrt(self) -> Dd {
            let s = self.0.sqrt();
            let r = self.sub(Dd::from(s).mul(Dd::from(s)));
            Dd::two_sum(s, r.0 / (2.0 * s))
        }
    }

    /// Signed distance `f / ||∇f||` in double-double for the two reference
    /// models: shifted circle `(x-c0)² + (y-c1)² - c2` and ellipse
    /// `c0·x² + c1·y² + c2·x·y - c3`.
    fn reference_residual(model: usize, c: &[Dd], p: &[f64]) -> Dd {
        let (x, y) = (Dd::from(p[0]), Dd::from(p[1]));
        let two = Dd::from(2.0);
        let (f, fx, fy) = if model == 0 {
            let dx = x.sub(c[0]);
            let dy = y.sub(c[1]);
            (
                dx.mul(dx).add(dy.mul(dy)).sub(c[2]),
                two.mul(dx),
                two.mul(dy),
            )
        } else {
            (
                c[0].mul(x.mul(x))
                    .add(c[1].mul(y.mul(y)))
                    .add(c[2].mul(x.mul(y)))
                    .sub(c[3]),
                two.mul(c[0]).mul(x).add(c[2].mul(y)),
                two.mul(c[1]).mul(y).add(c[2].mul(x)),
            )
        };
        f.div(fx.mul(fx).add(fy.mul(fy)).sqrt())
    }

    fn reference_jacobian(model: usize, c: &[f64], x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = c.len();
        x.iter()
            .map(|p| {
                (0..k)
                    .map(|j| {
                        let h = 1e-10 * (1.0 + c[j].abs());
                        let mut cp: Vec<Dd> = c.iter().map(|&v| Dd::from(v)).collect();
                        let mut cm = cp.clone();
                        cp[j] = cp[j].add(Dd::from(h));
                        cm[j] = cm[j].sub(Dd::from(h));
                        let d = reference_residual(model, &cp, p)
                            .sub(reference_residual(model, &cm, p));
                        let q = d.div(Dd::from(2.0 * h));
                        q.0 + q.1
                    })
                    .collect()
            })
            .collect()
    }

    /// Largest `|J - Jref|` over the largest `|Jref|` of the same column.
    fn max_column_error(j: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
        let k = reference[0].len();
        let mut worst = 0.0_f64;
        for col in 0..k {
            let scale = reference.iter().fold(0.0_f64, |m, r| m.max(r[col].abs()));
            for (row, rref) in j.iter().zip(reference) {
                worst = worst.max((row[col] - rref[col]).abs() / scale);
            }
        }
        worst
    }

    #[test]
    fn directional_jacobian_matches_central_differences_and_the_reference() {
        let shifted = |c: f64, i: usize| {
            node(Op::Square, vec![node(Op::Sub, vec![input(i), constant(c)])])
        };
        let mul = |c: f64, t: Node| node(Op::Mul, vec![constant(c), t]);
        let ellipse = node(
            Op::Sub,
            vec![
                node(
                    Op::Add,
                    vec![
                        node(Op::Add, vec![mul(0.3, sq(0)), mul(0.9, sq(1))]),
                        mul(0.2, node(Op::Mul, vec![input(0), input(1)])),
                    ],
                ),
                constant(1.0),
            ],
        );
        let cases = vec![
            (
                0,
                weighted_sum(vec![shifted(0.8, 0), shifted(0.3, 1)], 3.5),
                vec![0.8, 0.3, 3.5],
                ellipse_rows(1000, 1.0, 0.0, 2.0, 2.0, 11),
            ),
            (
                1,
                ellipse,
                vec![0.3, 0.9, 0.2, 1.0],
                ellipse_rows(1000, 0.0, 0.0, 1.8, 1.0, 12),
            ),
        ];
        for (model, tree, c, x) in cases {
            let mut level = Level::new(&tree);
            let (r, jd) = directional_jacobian(&mut level, &c, &x, &[]).unwrap();
            let (rc, jc) = central_jacobian(&mut level, &c, &x, &[]).unwrap();
            for (a, b) in r.iter().zip(&rc) {
                assert_eq!(a.to_bits(), b.to_bits(), "residuals share one expression");
            }
            let reference = reference_jacobian(model, &c, &x);
            let ed = max_column_error(&jd, &reference);
            let ec = max_column_error(&jc, &reference);
            let ed_c = max_column_error(&jd, &jc);
            println!("JACOBIAN model={model} directional={ed:e} central={ec:e} directional_vs_central={ed_c:e}");
            assert!(ed <= 1e-6, "directional error {ed}");
            assert!(ed_c <= 1e-6, "directional vs central {ed_c}");
        }
    }

    #[test]
    fn directional_fit_recovers_the_shifted_circle() {
        let x = ellipse_rows(400, 1.0, 0.0, 2.0, 2.0, 5);
        let mut t = weighted_sum(
            vec![
                node(Op::Square, vec![node(Op::Sub, vec![input(0), constant(0.5)])]),
                node(Op::Square, vec![node(Op::Sub, vec![input(1), constant(0.5)])]),
            ],
            2.0,
        );
        fit(&mut t, &x, &[], 20);
        let c = constants(&t);
        assert!((c[0] - 1.0).abs() < 1e-3, "centre x = {}", c[0]);
        assert!(c[1].abs() < 1e-3, "centre y = {}", c[1]);
        assert!((c[2] - 4.0).abs() < 2e-2, "r^2 = {}", c[2]);
    }
}
