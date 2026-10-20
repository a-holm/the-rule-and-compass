//! #116: variable projection (Golub–Pereyra) for separable models.
//!
//! A tree is *separable* when its constants split into a linear block `beta`
//! and a nonlinear block `theta` and the value is affine in `beta` for every
//! fixed `theta`:
//!
//! ```text
//! f(x; beta, theta) = f(x; beta0, theta) + sum_j (beta_j - beta0_j) g_j(x; theta)
//! ```
//!
//! For a fixed `theta` the optimal `beta` is one linear least-squares solve
//! (column-pivoted QR). VarPro then optimises only the `s` nonlinear constants
//! on the *projected* residual: the columns of `J_theta` are orthogonalised
//! against the span of `J_beta`, so the step never fights the inner solve.
//! `a*sin(b*x) + c` therefore has `beta = {a, c}`, `theta = {b}` and a 1-D
//! outer problem instead of a 3-D one.
//!
//! This arm is exact for the separable class in the same sense as the linear
//! arm: at convergence the inner solve is the global minimiser over `beta`, and
//! the outer iteration is a damped Gauss–Newton solve. It is never allowed to
//! replace a better starting point: any rank deficiency, non-finite row,
//! failure to converge or non-improving result returns `None` and the caller
//! runs the historical LM path on the untouched tree.

use super::*;

/// Largest `rows × constants` dense system VarPro materialises. Same cap as the
/// linear arm (64 MiB of `f64`) so an accidental huge candidate falls back to
/// the streaming/exact paths rather than allocating without bound.
const VARPRO_MAX_CELLS: usize = 8_000_000;

/// Largest number of constants VarPro will partition.
const VARPRO_MAX_CONSTANTS: usize = 64;

/// #116: the structural partition used by VarPro. Returned for reporting and
/// tests; empty `beta`/`theta` means the tree is not separable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VarProPartition {
    pub beta: Vec<usize>,
    pub theta: Vec<usize>,
}

/// #116: find a maximal linear block `beta` such that the value is affine in
/// `beta` with every other constant held fixed.
///
/// Constants are visited in slot order and admitted to `beta` only when the
/// selected set remains affine. A product of two selected constants is
/// bilinear and is never admitted together, so `a*(b + x)` keeps one of them
/// nonlinear. At least one constant must remain nonlinear; otherwise the
/// closed-form linear arm owns the tree.
pub(super) fn partition(f: &Flat) -> Option<VarProPartition> {
    let k = f.nconst;
    if k < 2 || k > VARPRO_MAX_CONSTANTS {
        return None;
    }
    let mut selected = vec![false; k];
    let mut beta = Vec::new();
    for j in 0..k {
        selected[j] = true;
        if affine_in_selected(f, &selected) {
            beta.push(j);
        } else {
            selected[j] = false;
        }
    }
    if beta.is_empty() || beta.len() == k {
        return None;
    }
    let theta = (0..k).filter(|&j| !selected[j]).collect();
    Some(VarProPartition { beta, theta })
}

/// #116: is the flattened tree affine in the **selected** constants, treating
/// every other constant as a fixed number?
///
/// This is the same structural classifier as [`affine_in_constants`], with
/// `has_const` restricted to the selected slots. Nonlinear operators are affine
/// only when they read no selected constant at all; `Mul` rejects two selected
/// factors (bilinear); `Div` rejects a selected denominator.
fn affine_in_selected(f: &Flat, selected: &[bool]) -> bool {
    let n = f.op.len();
    let mut has = vec![false; n];
    let mut affine = vec![true; n];
    for i in 0..n {
        let ks = &f.kids[i];
        if f.cslot[i] >= 0 {
            has[i] = selected[f.cslot[i] as usize];
            affine[i] = true;
            continue;
        }
        if f.op[i] == Op::Input {
            continue;
        }
        has[i] = ks.iter().any(|&j| has[j]);
        affine[i] = match f.op[i] {
            Op::Neg => affine[ks[0]],
            Op::Add | Op::Sub => affine[ks[0]] && affine[ks[1]],
            Op::Mul => affine[ks[0]] && affine[ks[1]] && !(has[ks[0]] && has[ks[1]]),
            Op::Div => affine[ks[0]] && !has[ks[1]],
            Op::If => !has[ks[0]] && affine[ks[1]] && affine[ks[2]],
            _ => !has[i],
        };
    }
    affine[f.root]
}

/// #116: fit a separable tree by variable projection.
///
/// Returns the weighted SSE when the arm ran and improved (or matched) the
/// starting constants, and `None` when the caller must use the exact LM path.
pub(super) fn fit_varpro(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
) -> Option<f64> {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) || x.is_empty() {
        return None;
    }
    let f = flatten(tree);
    let k = f.nconst;
    if k < 2 || k > VARPRO_MAX_CONSTANTS || x.len().saturating_mul(k) > VARPRO_MAX_CELLS {
        return None;
    }
    let part = partition(&f)?;
    let (q, s) = (part.beta.len(), part.theta.len());
    if q == 0 || s == 0 {
        return None;
    }
    let mut c = constants(&f);
    let mut v = vec![0.0; f.op.len()];
    let start = sse(&f, x, y, w, &c, &mut v);
    if !start.is_finite() {
        return None;
    }

    // Eliminate beta once at the starting theta, so the outer iteration begins
    // from the profile optimum rather than from the carrier's arbitrary beta.
    if !solve_beta(&f, x, y, w, &part.beta, &mut c) {
        return None;
    }
    let mut best_c = c.clone();
    let mut best_sse = sse(&f, x, y, w, &best_c, &mut v);
    if !best_sse.is_finite() || best_sse > start {
        return None;
    }
    let (q, s) = (part.beta.len(), part.theta.len());
    // `iters` is the same budget LM receives; the outer problem has only `s`
    // parameters, so this is comparable work, not extra work. The projected
    // Gauss–Newton direction is a descent direction for the profile residual;
    // a short backtracking line search makes one accepted step per iteration
    // instead of spending the budget growing a damping parameter.
    for _ in 0..iters.max(1) {
        let (jb, jt, r) = jacobians(&f, x, y, w, &best_c, &part)?;
        let m = r.len();
        let projected = project_out(&jb, &jt, m, q, s)?;
        let mut rhs: Vec<f64> = r.iter().map(|v| -v).collect();
        let delta = solve_dense_ls(projected, &mut rhs, m, s)?;
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..20 {
            let mut trial = best_c.clone();
            for (i, &slot) in part.theta.iter().enumerate() {
                trial[slot] = (trial[slot] + alpha * delta[i]).clamp(-1e6, 1e6);
                if !trial[slot].is_finite() {
                    return None;
                }
            }
            if solve_beta(&f, x, y, w, &part.beta, &mut trial) {
                let trial_sse = sse(&f, x, y, w, &trial, &mut v);
                if trial_sse.is_finite() && trial_sse < best_sse {
                    let gain = best_sse - trial_sse;
                    best_c = trial;
                    best_sse = trial_sse;
                    accepted = true;
                    if best_sse == 0.0 || gain <= 1e-14 * (1.0 + best_sse) {
                        return finish(&f, tree, &best_c, start, best_sse);
                    }
                    break;
                }
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    finish(&f, tree, &best_c, start, best_sse)
}

/// Validate the final constants and write them back. `None` means fall back.
fn finish(f: &Flat, tree: &mut Node, best_c: &[f64], start: f64, best_sse: f64) -> Option<f64> {
    if !best_sse.is_finite() || best_sse > start || !best_c.iter().all(|v| v.is_finite()) {
        return None;
    }
    write_constants(tree, f, best_c);
    Some(best_sse)
}

/// Initial constants in slot order.
fn constants(f: &Flat) -> Vec<f64> {
    let mut c = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    c
}

/// The weighted design matrices for beta and theta at `c`, plus the weighted
/// residual. Rows with zero weight are skipped, exactly as in `sse`.
#[allow(clippy::type_complexity)]
fn jacobians(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    part: &VarProPartition,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let k = f.nconst;
    let mut v = vec![0.0; f.op.len()];
    let mut adj = vec![0.0; f.op.len()];
    let mut grad = vec![0.0; k];
    let mut jb = Vec::new();
    let mut jt = Vec::new();
    let mut r = Vec::new();
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue;
        }
        if !(sw.is_finite() && sw > 0.0) {
            return None;
        }
        eval_row(f, row, c, &mut v);
        let pred = v[f.root];
        if !pred.is_finite() {
            return None;
        }
        backward(f, &v, &mut adj, &mut grad);
        let root = sw.sqrt();
        for &j in &part.beta {
            let g = grad[j];
            if !g.is_finite() {
                return None;
            }
            jb.push(root * g);
        }
        for &j in &part.theta {
            let g = grad[j];
            if !g.is_finite() {
                return None;
            }
            jt.push(root * g);
        }
        let ri = root * (pred - t);
        if !ri.is_finite() {
            return None;
        }
        r.push(ri);
    }
    Some((jb, jt, r))
}

/// Solve the inner linear least-squares problem for beta at fixed theta and
/// update `c` in place. The tree is affine in beta, so this is exact.
fn solve_beta(f: &Flat, x: &[Vec<f64>], y: &[f64], w: &[f64], beta: &[usize], c: &mut [f64]) -> bool {
    let q = beta.len();
    let k = f.nconst;
    let mut v = vec![0.0; f.op.len()];
    let mut adj = vec![0.0; f.op.len()];
    let mut grad = vec![0.0; k];
    let mut a = Vec::new();
    let mut rhs = Vec::new();
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue;
        }
        if !(sw.is_finite() && sw > 0.0) {
            return false;
        }
        eval_row(f, row, c, &mut v);
        let pred = v[f.root];
        if !pred.is_finite() {
            return false;
        }
        backward(f, &v, &mut adj, &mut grad);
        let root = sw.sqrt();
        for &j in beta {
            let g = grad[j];
            if !g.is_finite() {
                return false;
            }
            a.push(root * g);
        }
        let ri = -root * (pred - t);
        if !ri.is_finite() {
            return false;
        }
        rhs.push(ri);
    }
    let m = rhs.len();
    if m < q {
        return false;
    }
    let Some(delta) = solve_dense_ls(a, &mut rhs, m, q) else {
        return false;
    };
    for (i, &slot) in beta.iter().enumerate() {
        c[slot] = (c[slot] + delta[i]).clamp(-1e6, 1e6);
        if !c[slot].is_finite() {
            return false;
        }
    }
    true
}

/// Column-scaled QR least squares for a row-major `m × n` matrix and `m`-vector.
fn solve_dense_ls(mut a: Vec<f64>, b: &mut [f64], m: usize, n: usize) -> Option<Vec<f64>> {
    if m < n || a.len() != m * n {
        return None;
    }
    let mut scale = vec![0.0; n];
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..m {
            let v = a[i * n + j];
            s += v * v;
        }
        let norm = s.sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        scale[j] = norm;
        for i in 0..m {
            a[i * n + j] /= norm;
        }
    }
    let z = qr_least_squares(&mut a, m, n, &mut b[..m])?;
    let mut out = vec![0.0; n];
    for j in 0..n {
        out[j] = z[j] / scale[j];
        if !out[j].is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Orthogonalise every `theta` column against the span of the `beta` columns.
///
/// Modified Gram–Schmidt with one reorthogonalisation pass. Columns are
/// normalised first so the rank test is independent of constant magnitude.
/// `None` means `J_beta` is rank-deficient (or non-finite), which routes the
/// candidate to the exact LM fallback.
fn project_out(jb: &[f64], jt: &[f64], m: usize, q: usize, s: usize) -> Option<Vec<f64>> {
    if jb.len() != m * q || jt.len() != m * s {
        return None;
    }
    let mut qb = vec![0.0; m * q];
    for j in 0..q {
        let mut norm = 0.0;
        for i in 0..m {
            norm += jb[i * q + j] * jb[i * q + j];
        }
        let norm = norm.sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        for i in 0..m {
            qb[i * q + j] = jb[i * q + j] / norm;
        }
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += qb[i * q + k] * qb[i * q + j];
            }
            for i in 0..m {
                qb[i * q + j] -= dot * qb[i * q + k];
            }
        }
        // Reorthogonalise once: the first pass can lose too much precision for
        // nearly collinear columns.
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += qb[i * q + k] * qb[i * q + j];
            }
            for i in 0..m {
                qb[i * q + j] -= dot * qb[i * q + k];
            }
        }
        let mut norm = 0.0;
        for i in 0..m {
            norm += qb[i * q + j] * qb[i * q + j];
        }
        if !(norm.sqrt() > super::LINEAR_RANK_TOL) {
            return None;
        }
    }

    let mut out = vec![0.0; m * s];
    for j in 0..s {
        for i in 0..m {
            out[i * s + j] = jt[i * s + j];
        }
        for k in 0..q {
            let mut dot = 0.0;
            for i in 0..m {
                dot += qb[i * q + k] * out[i * s + j];
            }
            for i in 0..m {
                out[i * s + j] -= dot * qb[i * q + k];
            }
        }
        for k in 0..q {
            let mut dot = 0.0;
            for i in 0..m {
                dot += qb[i * q + k] * out[i * s + j];
            }
            for i in 0..m {
                out[i * s + j] -= dot * qb[i * q + k];
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    fn fitted(tree: &Node) -> Vec<f64> {
        let mut out = Vec::new();
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
        walk(tree, &mut out);
        out
    }

    /// `a*sin(b*x) + c` is separable: {a,c} linear, {b} nonlinear.
    #[test]
    fn partition_finds_the_linear_block() {
        let tree = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(1.0),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])]),
                    ],
                ),
                constant(0.0),
            ],
        );
        let f = flatten(&tree);
        let p = partition(&f).expect("separable");
        assert_eq!(p.beta, vec![0, 2]);
        assert_eq!(p.theta, vec![1]);
    }

    /// Bilinear `a*(b + x)` cannot eliminate both factors together; one stays
    /// nonlinear so VarPro remains valid.
    #[test]
    fn bilinear_tree_keeps_one_nonlinear_constant() {
        let tree = node(
            Op::Mul,
            vec![constant(1.0), node(Op::Add, vec![constant(1.0), input(0)])],
        );
        let f = flatten(&tree);
        let p = partition(&f).expect("separable in one factor");
        assert_eq!(p.beta.len(), 1);
        assert_eq!(p.theta.len(), 1);
    }

    /// The classic separable law is recovered to high precision, and the
    /// projected objective is never worse than exact LM.
    #[test]
    fn varpro_recovers_a_separable_law() {
        let x: Vec<Vec<f64>> = (0..81)
            .map(|i| vec![-3.0 + 6.0 * i as f64 / 80.0])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 2.5 * (1.3 * r[0]).sin() + 0.7)
            .collect();
        let build = || {
            node(
                Op::Add,
                vec![
                    node(
                        Op::Mul,
                        vec![constant(0.4), node(Op::Sin, vec![node(Op::Mul, vec![constant(0.4), input(0)])])],
                    ),
                    constant(0.0),
                ],
            )
        };
        let mut vp = build();
        let sse = fit_varpro(&mut vp, &x, &y, &[], 40).expect("VarPro must run");
        let c = fitted(&vp);
        assert!((c[0] - 2.5).abs() < 1e-5, "amplitude {c:?}");
        assert!((c[1] - 1.3).abs() < 1e-5, "frequency {c:?}");
        assert!((c[2] - 0.7).abs() < 1e-5, "offset {c:?}");
        assert!(sse < 1e-8, "separable law should fit to rounding: {sse}");

        let mut lm = build();
        let lm_sse = fit_weighted(&mut lm, &x, &y, &[], 40);
        assert!(sse <= lm_sse * (1.0 + 1e-9) + 1e-12, "VarPro {sse} vs LM {lm_sse}");
    }

    /// A rank-deficient inner system (two indistinguishable beta columns) is
    /// rejected, not solved with a pseudo-inverse that hides the problem.
    #[test]
    fn rank_deficient_inner_system_falls_back() {
        let x: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..12).map(|i| 3.0 * i as f64).collect();
        // a*sin(b*x) + d*sin(b*x) + c: the two amplitudes are the same column,
        // so the inner beta system is rank-deficient.
        let tree = node(
            Op::Add,
            vec![
                node(
                    Op::Add,
                    vec![
                        node(
                            Op::Mul,
                            vec![
                                constant(1.0),
                                node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                            ],
                        ),
                        node(
                            Op::Mul,
                            vec![
                                constant(1.0),
                                node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                            ],
                        ),
                    ],
                ),
                constant(0.0),
            ],
        );
        let f = flatten(&tree);
        let p = partition(&f).expect("separable in b");
        assert!(p.theta.contains(&1), "b stays nonlinear: {p:?}");
        assert!(p.beta.len() >= 2, "the duplicate amplitudes are linear: {p:?}");
        let mut c = constants(&f);
        assert!(!solve_beta(&f, &x, &y, &[], &p.beta, &mut c));
    }
}
