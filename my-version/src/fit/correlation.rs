//! #224: exact correlation fitting for `--metric correlation` and
//! `--metric correlation_signed`.
//!
//! The reported objective is exactly `1 - |rho|` (unsigned) or `1 - rho`
//! (signed) with `rho` the weighted Pearson correlation:
//!
//! ```text
//! rho = S_fy / sqrt(S_ff * S_yy)
//! S_ff = sum w (f - fbar)^2   S_yy = sum w (y - ybar)^2
//! S_fy = sum w (f - fbar)(y - ybar)
//! ```
//!
//! The fit uses the analytic gradient and the Gauss-Newton (GN) curvature of
//! `rho`:
//!
//! ```text
//! A_j  = sum w (g_j - gbar_j)(f - fbar)
//! B_j  = sum w (g_j - gbar_j)(y - ybar)
//! C_jk = sum w (g_j - gbar_j)(g_k - gbar_k)
//! d rho / d c_j = B_j / sqrt(S_ff S_yy) - rho A_j / S_ff
//! H_jk          = (C_jk - A_j A_k / S_ff) / S_ff
//! ```
//!
//! `H` contains no `y`, so signed and unsigned differ only in the sign of the
//! gradient. `rho` is invariant under `f -> a f + b` with `a > 0`, so a tree
//! with a free scale/offset pair has an exact two-dimensional null space in
//! `H`. That null space is removed **structurally**:
//!
//! * a tree that is affine in *every* constant is a rank-one generalised
//!   eigenproblem. For a single target its optimum is the ordinary weighted
//!   least-squares fit of `y` on the design, and the optimal `|rho|` is the
//!   multiple correlation of that regression (closed form, global);
//! * a tree with an explicit outer `a*T + b` shape freezes `a` and `b` and
//!   renormalises them to `a = sign(a)`, `b = 0`. Because `rho` does not
//!   depend on the pair, no shape descent direction is lost;
//! * every other tree runs LM on the exact objective with the existing
//!   damping, and a rank deficiency is reported (the caller falls back).
//!
//! Degenerate variance mirrors [`crate::metrics::correlation`]: a constant
//! prediction (`S_ff <= 0`) or a constant target scores exactly `1.0`, never
//! `+inf`.
//!
//! The fit never compares raw constants: the objective depends on the
//! prediction only, and the tests compare `rho`/the metric, never the affine
//! pair.

use super::*;
use crate::metrics::pow2_floor;

/// The free affine pair of a tree, if it has one.
///
/// `scale` is the constant in the outer `a*T` factor and `offset` the constant
/// added to it. Either can be absent (`a*T` alone has only a scale gauge; an
/// addend with no scale is a shape parameter, not part of the gauge).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CorrGauge {
    pub(crate) scale: Option<usize>,
    pub(crate) offset: Option<usize>,
}

impl CorrGauge {
    /// The constant slots the LM must not move, as the kernel's `gauge` pair.
    pub(crate) fn slots(self) -> [i32; 2] {
        [
            self.scale.map(|s| s as i32).unwrap_or(-1),
            self.offset.map(|s| s as i32).unwrap_or(-1),
        ]
    }

    fn is_gauge(&self, slot: usize) -> bool {
        self.scale == Some(slot) || self.offset == Some(slot)
    }

    fn is_empty(&self) -> bool {
        self.scale.is_none() && self.offset.is_none()
    }
}

/// Locate the outer affine gauge of a flattened tree.
///
/// Recognises `Add(Mul(Const, T), Const)` in either operand order (after
/// simplification), plus the scale-only `Mul(Const, T)`. `T` may contain other
/// constants; only the outer pair is the exact gauge.
pub(super) fn gauge_of(f: &Flat) -> CorrGauge {
    let root = f.root;
    if f.op[root] == Op::Mul {
        let ks = &f.kids[root];
        if ks.len() == 2 {
            if f.cslot[ks[0]] >= 0 && f.cslot[ks[1]] < 0 {
                return CorrGauge {
                    scale: Some(f.cslot[ks[0]] as usize),
                    offset: None,
                };
            }
            if f.cslot[ks[1]] >= 0 && f.cslot[ks[0]] < 0 {
                return CorrGauge {
                    scale: Some(f.cslot[ks[1]] as usize),
                    offset: None,
                };
            }
        }
    }
    if f.op[root] == Op::Add {
        let ks = &f.kids[root];
        for (mul_side, add_side) in [(0usize, 1usize), (1, 0)] {
            let m = ks[mul_side];
            let b = ks[add_side];
            if f.op[m] != Op::Mul || f.cslot[b] < 0 {
                continue;
            }
            let mk = &f.kids[m];
            if mk.len() != 2 {
                continue;
            }
            let scale = if f.cslot[mk[0]] >= 0 && f.cslot[mk[1]] < 0 {
                f.cslot[mk[0]] as usize
            } else if f.cslot[mk[1]] >= 0 && f.cslot[mk[0]] < 0 {
                f.cslot[mk[1]] as usize
            } else {
                continue;
            };
            return CorrGauge {
                scale: Some(scale),
                offset: Some(f.cslot[b] as usize),
            };
        }
    }
    CorrGauge::default()
}

/// Weighted centred moments and the exact gradient/GN curvature of `rho`.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct CorrMoments {
    pub(crate) weight: f64,
    pub(crate) f_mean: f64,
    pub(crate) y_mean: f64,
    pub(crate) s_ff: f64,
    pub(crate) s_yy: f64,
    pub(crate) s_fy: f64,
    pub(crate) rho: f64,
    /// `d rho / d c_j`.
    pub(crate) grad_rho: Vec<f64>,
    /// Gauss-Newton curvature of `1 - rho` (the same matrix for signed).
    pub(crate) hess: Vec<Vec<f64>>,
}

/// The chordal form of the objective used by the fit.
///
/// `1 - rho = 0.5 ||u - v||^2` for the centred unit vectors, so a value close
/// to zero is an `O(theta^2)` quantity rather than a difference of two numbers
/// close to one. For unsigned correlation the objective is the minimum over the
/// reflection `u -> -u`, i.e. `1 - |rho|`.
pub(crate) fn objective(signed: bool, rho: f64) -> f64 {
    if !rho.is_finite() {
        return f64::INFINITY;
    }
    if signed {
        1.0 - rho
    } else {
        1.0 - rho.abs()
    }
}

/// The direction multiplier of the objective: `d(1 - |rho|)/d rho`.
fn sigma(signed: bool, rho: f64) -> f64 {
    if signed {
        1.0
    } else if rho >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Two passes over the rows at `c`: weighted centred moments, `d rho`, `H`.
///
/// Returns `None` when a prediction or derivative is non-finite, so the caller
/// can report `non_finite` and keep the incoming constants.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moments(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    v: &mut [f64],
    adj: &mut [f64],
    g: &mut [f64],
) -> Option<CorrMoments> {
    let p = f.nconst;
    let mut wsum = 0.0;
    let mut sf = 0.0;
    let mut sy = 0.0;
    let mut sg = vec![0.0; p];
    let mut rw: Vec<f64> = Vec::with_capacity(x.len());
    let mut rf: Vec<f64> = Vec::with_capacity(x.len());
    let mut rt: Vec<f64> = Vec::with_capacity(x.len());
    let mut rg: Vec<f64> = Vec::with_capacity(x.len() * p);
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue;
        }
        if !(sw.is_finite() && sw > 0.0) {
            return None;
        }
        eval_row(f, row, c, v);
        let pred = v[f.root];
        if !pred.is_finite() || !t.is_finite() {
            return None;
        }
        backward(f, v, adj, g);
        wsum += sw;
        sf += sw * pred;
        sy += sw * t;
        for j in 0..p {
            let gj = g[j];
            if !gj.is_finite() {
                return None;
            }
            sg[j] += sw * gj;
        }
        rw.push(sw);
        rf.push(pred);
        rt.push(t);
        rg.extend_from_slice(&g[..p]);
    }
    if !(wsum > 0.0) || !wsum.is_finite() {
        return None;
    }
    let f_mean = sf / wsum;
    let y_mean = sy / wsum;
    let gbar: Vec<f64> = sg.iter().map(|s| s / wsum).collect();
    // Centred two-pass sums in units of the largest deviation, so a prediction
    // of tiny magnitude or a large mean with a small spread keeps its digits.
    let mut scale_f = 0.0f64;
    let mut scale_y = 0.0f64;
    for (&pred, &t) in rf.iter().zip(&rt) {
        scale_f = scale_f.max((pred - f_mean).abs());
        scale_y = scale_y.max((t - y_mean).abs());
    }
    let mut grad = vec![0.0; p];
    let mut hess = vec![vec![0.0; p]; p];
    if !(scale_f.is_finite() && scale_y.is_finite()) {
        return None;
    }
    // Mirror `metrics::correlation`: a degenerate variance scores 0.0 there,
    // which is the metric's worst value `1.0`.
    if scale_f < f64::MIN_POSITIVE || scale_y < f64::MIN_POSITIVE {
        return Some(CorrMoments {
            weight: wsum,
            f_mean,
            y_mean,
            s_ff: 0.0,
            s_yy: 0.0,
            s_fy: 0.0,
            rho: 0.0,
            grad_rho: grad,
            hess,
        });
    }
    let (scale_f, scale_y) = (pow2_floor(scale_f), pow2_floor(scale_y));
    let (mut s_ff, mut s_yy, mut s_fy) = (0.0, 0.0, 0.0);
    let mut a = vec![0.0; p];
    let mut b = vec![0.0; p];
    let mut cjk = vec![0.0; p * (p + 1) / 2];
    let mut dg = vec![0.0; p];
    for (r, &sw) in rw.iter().enumerate() {
        let df = (rf[r] - f_mean) / scale_f;
        let dy = (rt[r] - y_mean) / scale_y;
        s_ff += sw * df * df;
        s_yy += sw * dy * dy;
        s_fy += sw * df * dy;
        for j in 0..p {
            dg[j] = (rg[r * p + j] - gbar[j]) / scale_f;
            a[j] += sw * dg[j] * df;
            b[j] += sw * dg[j] * dy;
        }
        let mut k = 0;
        for i in 0..p {
            for j in 0..=i {
                cjk[k] += sw * dg[i] * dg[j];
                k += 1;
            }
        }
    }
    if !(s_ff.is_finite() && s_yy.is_finite() && s_fy.is_finite()) {
        return None;
    }
    if s_ff <= 0.0 || s_yy <= 0.0 {
        return Some(CorrMoments {
            weight: wsum,
            f_mean,
            y_mean,
            s_ff: 0.0,
            s_yy: 0.0,
            s_fy: 0.0,
            rho: 0.0,
            grad_rho: grad,
            hess,
        });
    }
    let rho = (s_fy / (s_ff.sqrt() * s_yy.sqrt())).clamp(-1.0, 1.0);
    let root = s_ff.sqrt() * s_yy.sqrt();
    for j in 0..p {
        grad[j] = b[j] / root - rho * a[j] / s_ff;
        if !grad[j].is_finite() {
            return None;
        }
    }
    let mut k = 0;
    for i in 0..p {
        for j in 0..=i {
            hess[i][j] = (cjk[k] - a[i] * a[j] / s_ff) / s_ff;
            hess[j][i] = hess[i][j];
            k += 1;
        }
    }
    // The sums above are in scaled units; report them in data units so the
    // callers' relative-variance gate keeps its meaning. They may underflow to
    // zero for a tiny-magnitude prediction, which the fit treats as degenerate.
    Some(CorrMoments {
        weight: wsum,
        f_mean,
        y_mean,
        s_ff: s_ff * scale_f * scale_f,
        s_yy: s_yy * scale_y * scale_y,
        s_fy: s_fy * scale_f * scale_y,
        rho,
        grad_rho: grad,
        hess,
    })
}

/// Whether the flattened tree is affine in all constants (the closed-form arm).
pub(crate) fn is_affine(f: &Flat) -> bool {
    affine_in_constants(f)
}

/// Exact correlation fit. Returns the configured metric at the fitted
/// constants, exactly as `metrics::score_weighted` reports it.
///
/// `iters == 0` is a true no-op (`--no-fit`): the tree is untouched and its
/// current metric is returned.
pub(crate) fn fit(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> f64 {
    let signed = metric == Metric::CorrelationSigned;
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    let start_score = metric_score(tree, x, y, w, metric);
    if iters == 0 {
        return start_score;
    }
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || x.is_empty() {
        return start_score;
    }
    // Closed-form arm: for a tree affine in every constant the correlation
    // optimum is the weighted least-squares fit of y on the design. It is the
    // rank-one generalised eigenproblem with a single target, and the LS
    // solution also maximises the signed correlation (its residual is
    // orthogonal to the design). Never replace a better starting point.
    if is_affine(&f) {
        #[cfg(test)]
        FIT_STEPS.with(|n| n.set(n.get() + 1));
        let mut trial = tree.clone();
        if fit_linear(&mut trial, x, y, w).is_some() {
            let trial_score = metric_score(&trial, x, y, w, metric);
            if trial_score.is_finite() && trial_score < start_score {
                *tree = trial;
                return trial_score;
            }
        }
    }
    let gauge = gauge_of(&f);
    // Signed correlation has a genuine sign choice. A frozen scale preserves
    // the current orientation, so try both canonical orientations and keep the
    // better one; unsigned has no orientation to choose.
    let signs: &[Option<f64>] = if signed && gauge.scale.is_some() {
        &[None, Some(-1.0), Some(1.0)]
    } else {
        &[None]
    };
    let mut best_tree = tree.clone();
    let mut best_score = start_score;
    for &force in signs {
        let mut trial = tree.clone();
        let score = fit_lm(&mut trial, x, y, w, iters, signed, gauge, force);
        if score.is_finite() && score < best_score {
            best_score = score;
            best_tree = trial;
        }
    }
    if best_score < start_score {
        *tree = best_tree;
        best_score
    } else {
        // The exact gate: a numerical edge case never replaces a better start.
        start_score
    }
}

/// LM on the exact correlation objective with analytic gradient and curvature.
///
/// The affine gauge slots (if any) are renormalised once and then frozen: their
/// step is exactly zero, so the reduced system has no null direction.
#[allow(clippy::too_many_arguments)]
fn fit_lm(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    signed: bool,
    gauge: CorrGauge,
    force_scale: Option<f64>,
) -> f64 {
    let f = flatten(tree);
    let k = f.nconst;
    let n = f.op.len();
    let mut c = vec![0.0; k];
    for i in 0..n {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    if !gauge.is_empty() {
        // The gauge does not change rho; a canonical `(a,b) = (sign(a), 0)`
        // keeps both the proposal and the reported constants deterministic.
        if let Some(s) = gauge.scale {
            let sign = force_scale.unwrap_or_else(|| {
                if c[s].is_finite() && c[s] < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            });
            c[s] = sign;
        }
        if let Some(o) = gauge.offset {
            c[o] = 0.0;
        }
        write_constants(tree, &f, &c);
    }
    let mut v = vec![0.0; n];
    let mut adj = vec![0.0; n];
    let mut g = vec![0.0; k];
    let Some(mut m) = moments(&f, x, y, w, &c, &mut v, &mut adj, &mut g) else {
        return metric_score(tree, x, y, w, if signed { Metric::CorrelationSigned } else { Metric::Correlation });
    };
    let mut best_c = c.clone();
    let mut best = objective(signed, m.rho);
    let metric = if signed { Metric::CorrelationSigned } else { Metric::Correlation };
    if !best.is_finite() {
        return metric_score(tree, x, y, w, metric);
    }
    // A degenerate variance is exactly 1.0, matching the CPU metric. No step
    // can improve it, and the incoming constants are kept.
    if m.s_ff <= 0.0 || m.s_yy <= 0.0 {
        return metric_score(tree, x, y, w, metric);
    }
    // A near-constant prediction is a conditioning signal, not a known answer:
    // keep the start and let the caller decide (the exact gate still applies).
    if m.s_ff < 1e-24 * m.s_yy {
        return metric_score(tree, x, y, w, metric);
    }
    let free: Vec<usize> = (0..k).filter(|&j| !gauge.is_gauge(j)).collect();
    if free.is_empty() {
        return metric_score(tree, x, y, w, metric);
    }
    let mut lambda = 1e-3;
    for _ in 0..iters.max(1) {
        let sig = sigma(signed, m.rho);
        if sig == 0.0 {
            break;
        }
        // Reduced GN system over the free slots.
        let q = free.len();
        let mut h = vec![vec![0.0; q]; q];
        let mut rhs = vec![0.0; q];
        for (a, &ja) in free.iter().enumerate() {
            rhs[a] = sig * m.grad_rho[ja];
            for (b, &jb) in free.iter().enumerate() {
                h[a][b] = m.hess[ja][jb];
            }
        }
        #[cfg(test)]
        FIT_STEPS.with(|n| n.set(n.get() + 1));
        let Some(delta) = solve(&h, &mut rhs, lambda, q) else {
            // A singular reduced system is a labelled conditioning failure:
            // keep the best constants the exact objective has seen.
            break;
        };
        let mut trial = c.clone();
        for (a, &ja) in free.iter().enumerate() {
            trial[ja] = c[ja] + delta[a];
        }
        let Some(trial_m) = moments(&f, x, y, w, &trial, &mut v, &mut adj, &mut g) else {
            break;
        };
        let trial_obj = objective(signed, trial_m.rho);
        if trial_obj < best {
            c = trial;
            m = trial_m;
            best_c = c.clone();
            best = trial_obj;
            lambda = (lambda / 8.0).max(1e-12);
            if best <= 0.0 {
                break;
            }
        } else {
            lambda *= 6.0;
            if lambda > 1e9 {
                break;
            }
        }
    }
    // Undamped Gauss-Newton polish, the same shape as the SSE LM path.
    for _ in 0..4 {
        if best <= 0.0 {
            break;
        }
        let sig = sigma(signed, m.rho);
        if sig == 0.0 {
            break;
        }
        let q = free.len();
        let mut h = vec![vec![0.0; q]; q];
        let mut rhs = vec![0.0; q];
        for (a, &ja) in free.iter().enumerate() {
            rhs[a] = sig * m.grad_rho[ja];
            for (b, &jb) in free.iter().enumerate() {
                h[a][b] = m.hess[ja][jb];
            }
        }
        #[cfg(test)]
        FIT_STEPS.with(|n| n.set(n.get() + 1));
        let Some(delta) = solve(&h, &mut rhs, 0.0, q) else {
            break;
        };
        let mut trial = c.clone();
        for (a, &ja) in free.iter().enumerate() {
            trial[ja] = c[ja] + delta[a];
        }
        let Some(trial_m) = moments(&f, x, y, w, &trial, &mut v, &mut adj, &mut g) else {
            break;
        };
        let trial_obj = objective(signed, trial_m.rho);
        if trial_obj < best {
            c = trial;
            m = trial_m;
            best_c = c.clone();
            best = trial_obj;
        } else {
            break;
        }
    }
    write_constants(tree, &f, &best_c);
    metric_score(tree, x, y, w, metric)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    fn rows(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![-2.0 + 4.0 * i as f64 / (n - 1).max(1) as f64]).collect()
    }

    fn score(tree: &Node, x: &[Vec<f64>], y: &[f64], metric: Metric) -> f64 {
        metric_score(tree, x, y, &[], metric)
    }

    fn gauge_tree() -> Node {
        node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(0.7), node(Op::Sin, vec![node(Op::Mul, vec![constant(1.3), input(0)])])]),
                constant(0.1),
            ],
        )
    }

    fn rho_of(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
        let f = flatten(tree);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        let mut v = vec![0.0; f.op.len()];
        let mut adj = vec![0.0; f.op.len()];
        let mut g = vec![0.0; f.nconst];
        moments(&f, x, y, w, &c, &mut v, &mut adj, &mut g).expect("finite moments").rho
    }

    fn tiny_target(x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|r| (r[0] + 1.0).ln() + (r[0] * r[0] + 1.0).ln()).collect()
    }

    #[test]
    fn correlation_tiny_magnitude() {
        let x: Vec<Vec<f64>> = (0..200).map(|i| vec![2.0 * i as f64 / 199.0]).collect();
        let y = tiny_target(&x);
        let scaled = |k: f64| node(Op::Mul, vec![constant(k), node(Op::Exp, vec![input(0)])]);
        for w in [vec![], (0..200).map(|i| 1.0 + (i % 3) as f64).collect::<Vec<f64>>()] {
            let r = rho_of(&scaled(1e-162), &x, &y, &w);
            assert!(r.abs() <= 1.0, "r={r}");
            assert!((r - rho_of(&scaled(1.0), &x, &y, &w)).abs() < 1e-9, "r={r}");
        }
    }

    #[test]
    fn correlation_large_mean_small_spread() {
        let x = rows(101);
        let y: Vec<f64> = x.iter().map(|r| r[0] * r[0] + r[0]).collect();
        let big = node(Op::Add, vec![constant(1e8), node(Op::Mul, vec![constant(1e-4), input(0)])]);
        let unit = input(0);
        for w in [vec![], (0..101).map(|i| 1.0 + (i % 3) as f64).collect::<Vec<f64>>()] {
            let r = rho_of(&big, &x, &y, &w);
            assert!((r - rho_of(&unit, &x, &y, &w)).abs() < 1e-6, "r={r}");
        }
    }

    #[test]
    fn gradient_matches_central_differences() {
        for signed in [false, true] {
            let x = rows(61);
            let y: Vec<f64> = x.iter().map(|r| 1.7 * (0.9 * r[0]).sin() + 0.3).collect();
            let w: Vec<f64> = (0..x.len()).map(|i| 0.5 + (i % 7) as f64 / 7.0).collect();
            let tree = gauge_tree();
            let f = flatten(&tree);
            let mut c = vec![0.0; f.nconst];
            for i in 0..f.op.len() {
                if f.cslot[i] >= 0 {
                    c[f.cslot[i] as usize] = f.val[i];
                }
            }
            let mut v = vec![0.0; f.op.len()];
            let mut adj = vec![0.0; f.op.len()];
            let mut g = vec![0.0; f.nconst];
            let m = moments(&f, &x, &y, &w, &c, &mut v, &mut adj, &mut g).expect("finite moments");
            // The fixture is well-conditioned: rho must be finite and not 1.
            assert!(m.rho.abs() < 0.999, "rho={}", m.rho);
            for j in 0..f.nconst {
                let h = 1e-6 * c[j].abs().max(1.0);
                let mut cp = c.clone();
                cp[j] += h;
                let mp = moments(&f, &x, &y, &w, &cp, &mut v, &mut adj, &mut g).unwrap();
                let mut cm = c.clone();
                cm[j] -= h;
                let mm = moments(&f, &x, &y, &w, &cm, &mut v, &mut adj, &mut g).unwrap();
                let fd = (mp.rho - mm.rho) / (2.0 * h);
                assert!(
                    (fd - m.grad_rho[j]).abs() <= 1e-6 * (1.0 + m.grad_rho[j].abs()),
                    "signed={signed} j={j} fd={fd} analytic={}",
                    m.grad_rho[j]
                );
            }
        }
    }

    #[test]
    fn the_analytic_gauge_is_a_null_vector_of_the_curvature() {
        let x = rows(41);
        let y: Vec<f64> = x.iter().map(|r| 1.7 * (0.9 * r[0]).sin() + 0.3).collect();
        let tree = gauge_tree();
        let f = flatten(&tree);
        let gauge = gauge_of(&f);
        assert_eq!(gauge.scale, Some(0));
        assert_eq!(gauge.offset, Some(2));
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        let mut v = vec![0.0; f.op.len()];
        let mut adj = vec![0.0; f.op.len()];
        let mut g = vec![0.0; f.nconst];
        let m = moments(&f, &x, &y, &[], &c, &mut v, &mut adj, &mut g).unwrap();
        // The scale direction is f - b and the offset direction is 1.
        let h_scale = m.hess[0][0];
        let h_off = m.hess[2][2];
        assert!(h_scale.abs() < 1e-10, "H_aa={h_scale}");
        assert!(h_off.abs() < 1e-10, "H_bb={h_off}");
    }

    #[test]
    fn rho_is_invariant_under_the_affine_gauge() {
        let x = rows(37);
        let tree = gauge_tree();
        let mut y = Vec::new();
        for r in &x {
            y.push(expr::eval(&tree, r));
        }
        let base = score(&tree, &x, &y, Metric::Correlation);
        for a in [1e-3, 0.5, 1.0, 1e3] {
            for b in [-1e3, 0.0, 1e3] {
                let mut scaled = tree.clone();
                // `a*T + b` from the same shape: constants at preorder 2 and 7.
                if let Some(n) = expr::nth_mut(&mut scaled, 2) {
                    n.val = 0.7 * a;
                }
                if let Some(n) = expr::nth_mut(&mut scaled, 7) {
                    n.val = 0.1 * a + b;
                }
                assert!((score(&scaled, &x, &y, Metric::Correlation) - base).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_constant_prediction_scores_exactly_one() {
        let x = rows(20);
        let y: Vec<f64> = x.iter().map(|r| r[0] * 2.0).collect();
        // `sin(0*x)` is a constant prediction but not affine in the constant,
        // so the closed-form arm cannot rescue it.
        let mut tree = node(
            Op::Add,
            vec![
                node(Op::Sin, vec![node(Op::Mul, vec![constant(0.0), input(0)])]),
                constant(3.0),
            ],
        );
        let before = fitted(&tree);
        let score = fit(&mut tree, &x, &y, &[], 8, Metric::Correlation);
        assert_eq!(score, 1.0);
        assert_eq!(fitted(&tree), before);
    }

    #[test]
    fn zero_iterations_is_a_true_no_op() {
        let x = rows(20);
        let y: Vec<f64> = x.iter().map(|r| (1.3 * r[0]).sin()).collect();
        let mut tree = gauge_tree();
        let before = fitted(&tree);
        let start = score(&tree, &x, &y, Metric::CorrelationSigned);
        let got = fit(&mut tree, &x, &y, &[], 0, Metric::CorrelationSigned);
        assert_eq!(got.to_bits(), start.to_bits());
        assert_eq!(fitted(&tree), before);
    }

    #[test]
    fn correlation_fit_beats_the_historical_squared_plus_polish() {
        let x = rows(81);
        let y: Vec<f64> = x
            .iter()
            .map(|r| 1.5 * (0.8 * r[0]).sin() + 0.4 * r[0] + 0.2)
            .collect();
        let mut corr = gauge_tree();
        let got = fit(&mut corr, &x, &y, &[], 12, Metric::Correlation);
        let mut old = gauge_tree();
        fit_weighted(&mut old, &x, &y, &[], 12);
        coordinate_polish(&mut old, &x, &y, &[], Metric::Correlation, 12);
        let old_score = score(&old, &x, &y, Metric::Correlation);
        assert!(got <= old_score + 1e-12, "new={got} old={old_score}");
    }

    #[test]
    fn the_fit_is_deterministic_and_metric_monotone() {
        let x = rows(97);
        let y: Vec<f64> = x
            .iter()
            .map(|r| 1.2 * (1.7 * r[0]).cos() - 0.3 * r[0] + 0.1)
            .collect();
        let w: Vec<f64> = (0..x.len()).map(|i| 1.0 + (i % 5) as f64).collect();
        let mut a = gauge_tree();
        let mut b = gauge_tree();
        let start = score(&a, &x, &y, Metric::Correlation);
        let sa = fit(&mut a, &x, &y, &w, 9, Metric::Correlation);
        let sb = fit(&mut b, &x, &y, &w, 9, Metric::Correlation);
        assert_eq!(sa.to_bits(), sb.to_bits());
        assert_eq!(fitted(&a), fitted(&b));
        assert!(sa <= start, "fit {sa} worsened the start {start}");
    }

    #[test]
    fn an_affine_tree_gets_the_closed_form_correlation_optimum() {
        let x = rows(41);
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] - 1.0).collect();
        let mut tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.4), input(0)]), constant(2.0)],
        );
        let got = fit(&mut tree, &x, &y, &[], 6, Metric::Correlation);
        assert!(got < 1e-20, "closed-form rho optimum got {got}");
        let got_signed = fit(&mut tree, &x, &y, &[], 6, Metric::CorrelationSigned);
        assert!(got_signed < 1e-20, "signed closed-form got {got_signed}");
    }

    #[test]
    fn signed_fit_recovers_an_inverted_start() {
        // The shape is non-affine, so the frozen-affine LM (with both
        // orientations) decides the sign, not the closed-form arm.
        let x = rows(81);
        let y: Vec<f64> = x.iter().map(|r| -(1.4 * (0.9 * r[0]).sin() + 0.2)).collect();
        let mut tree = gauge_tree();
        let start = score(&tree, &x, &y, Metric::CorrelationSigned);
        let got = fit(&mut tree, &x, &y, &[], 12, Metric::CorrelationSigned);
        assert!(got < start + 1e-12, "signed start {start} -> {got}");
        assert!(got < 0.2, "signed fit did not recover the orientation: {got}");
    }

    #[test]
    fn weighted_fit_matches_the_duplicated_rows() {
        let x = rows(31);
        let y: Vec<f64> = x.iter().map(|r| 0.9 * (1.1 * r[0]).sin() + 0.05).collect();
        // Weight 2 on rows with an even index is exactly duplication.
        let mut weighted = gauge_tree();
        let ws = fit(&mut weighted, &x, &y, &vec![2.0; x.len()], 10, Metric::Correlation);
        let mut xx = Vec::new();
        let mut yy = Vec::new();
        for (row, &t) in x.iter().zip(&y) {
            xx.push(row.clone());
            yy.push(t);
            xx.push(row.clone());
            yy.push(t);
        }
        let mut plain = gauge_tree();
        let ps = fit(&mut plain, &xx, &yy, &[], 10, Metric::Correlation);
        assert!((ws - ps).abs() < 1e-12, "weighted {ws} vs duplicated {ps}");
    }

    #[test]
    fn a_non_finite_row_keeps_the_incoming_constants() {
        let mut x = rows(20);
        x[3][0] = f64::NAN;
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut tree = gauge_tree();
        let before = fitted(&tree);
        let score = fit(&mut tree, &x, &y, &[], 8, Metric::Correlation);
        assert!(!score.is_finite());
        assert_eq!(fitted(&tree), before);
    }

    fn fitted(tree: &Node) -> Vec<u64> {
        let mut out = Vec::new();
        let mut counter = 0;
        let mut positions = Vec::new();
        const_positions(tree, &mut positions, &mut counter);
        for pos in positions {
            if let Some(n) = expr::nth(tree, pos) {
                out.push(n.val.to_bits());
            }
        }
        out
    }

    // -- signed orientation --

    /// `1 − ρ` charges the reflection exactly `2ρ`, while `1 − |ρ|`
    /// charges it nothing. This is the difference the signed arm exists for.
    #[test]
    fn the_signed_objective_penalises_the_reflection() {
        let x = rows(41);
        let h: Vec<f64> = x.iter().map(|r| 1.3 * (0.9 * r[0]).sin() + 0.2).collect();
        let mirrored: Vec<f64> = h.iter().map(|v| -v).collect();
        let rho = crate::metrics::correlation(&h, &h);
        let rho_reflected = crate::metrics::correlation(&mirrored, &h);
        assert!(rho > 0.99, "rho={rho}");
        assert!((rho_reflected + rho).abs() < 1e-12, "rho(-h) = -rho(h)");
        assert_eq!(objective(true, rho), 1.0 - rho);
        assert_eq!(objective(true, rho_reflected), 1.0 - rho_reflected);
        // Reflected: 1 - (-rho) = 2 - (1 - rho).
        let gap = objective(true, rho_reflected) - objective(true, rho);
        assert!((gap - 2.0 * rho).abs() < 1e-12, "reflection penalty {gap}");
        // Unsigned scores both orientations identically.
        assert!((objective(false, rho) - objective(false, rho_reflected)).abs() < 1e-12);
    }

    /// the signed fit recovers an inverted start by choosing the sign of
    /// the frozen affine scale, and it never returns the reflected answer.
    #[test]
    fn the_signed_fit_returns_the_positive_orientation_not_the_reflection() {
        let x = rows(81);
        let y: Vec<f64> = x
            .iter()
            .map(|r| -(1.4 * (0.9 * r[0]).sin() + 0.15 * r[0] + 0.2))
            .collect();
        let mut tree = gauge_tree();
        let start = score(&tree, &x, &y, Metric::CorrelationSigned);
        assert!(start > 1.0, "the fixture must start inverted: {start}");
        let got = fit(&mut tree, &x, &y, &[], 12, Metric::CorrelationSigned);
        assert!(got < start, "signed fit {start} -> {got}");
        assert!(got < 0.05, "signed fit did not recover the direction: {got}");
        // The reported prediction is positively correlated with the target:
        // the fit selected the orientation instead of the reflection.
        let pred: Vec<f64> = x.iter().map(|r| expr::eval(&tree, r)).collect();
        assert!(
            crate::metrics::correlation(&pred, &y) > 0.9,
            "fitted rho must be positive"
        );
        // The gauge is frozen and renormalised: the scale is a unit sign and
        // the offset is exactly zero. Either sign is a valid optimum (the
        // inner constants absorb the sign), but the *prediction* must be
        // positively correlated with the target, never the reflection.
        let scale = expr::nth(&tree, 2).expect("scale constant").val;
        let offset = expr::nth(&tree, 7).expect("offset constant").val;
        assert!(scale == 1.0 || scale == -1.0, "scale={scale}");
        assert_eq!(offset, 0.0, "offset={offset}");
    }

    /// the same shape started from `-T` must reach the same signed
    /// optimum, because the orientation search is free and `ρ(a·T+b) = ρ(T)`.
    #[test]
    fn the_signed_fit_is_invariant_to_the_starting_orientation() {
        let x = rows(57);
        let y: Vec<f64> = x.iter().map(|r| 1.1 * (1.2 * r[0]).sin() + 0.3).collect();
        let mut positive = gauge_tree();
        let mut negative = gauge_tree();
        if let Some(n) = expr::nth_mut(&mut negative, 2) {
            n.val = -0.7;
        }
        let start_pos = score(&positive, &x, &y, Metric::CorrelationSigned);
        let start_neg = score(&negative, &x, &y, Metric::CorrelationSigned);
        assert!(start_neg > start_pos, "the negative start must be the worse one");
        let a = fit(&mut positive, &x, &y, &[], 12, Metric::CorrelationSigned);
        let b = fit(&mut negative, &x, &y, &[], 12, Metric::CorrelationSigned);
        assert!(a < 0.05, "positive start {start_pos} -> {a}");
        assert!(b < 0.05, "negative start {start_neg} -> {b}");
        assert!(
            (a - b).abs() < 1e-12,
            "the orientation of the start must not decide the optimum: {a} vs {b}"
        );
    }

    /// a degenerate variance under the signed objective is exactly `1.0`
    /// and the incoming constants are untouched.
    #[test]
    fn a_constant_prediction_scores_exactly_one_signed() {
        let x = rows(20);
        let y: Vec<f64> = x.iter().map(|r| r[0] * 2.0).collect();
        let mut tree = node(
            Op::Add,
            vec![
                node(Op::Sin, vec![node(Op::Mul, vec![constant(0.0), input(0)])]),
                constant(3.0),
            ],
        );
        let before = fitted(&tree);
        let score = fit(&mut tree, &x, &y, &[], 8, Metric::CorrelationSigned);
        assert_eq!(score, 1.0);
        assert_eq!(fitted(&tree), before);
    }

    /// the signed arm is deterministic bit for bit.
    #[test]
    fn the_signed_fit_is_deterministic() {
        let x = rows(97);
        let y: Vec<f64> = x
            .iter()
            .map(|r| -(1.2 * (1.7 * r[0]).cos() - 0.3 * r[0] + 0.1))
            .collect();
        let mut a = gauge_tree();
        let mut b = gauge_tree();
        let sa = fit(&mut a, &x, &y, &[], 9, Metric::CorrelationSigned);
        let sb = fit(&mut b, &x, &y, &[], 9, Metric::CorrelationSigned);
        assert_eq!(sa.to_bits(), sb.to_bits());
        assert_eq!(fitted(&a), fitted(&b));
    }

    /// `--no-fit` under the signed objective returns the current score
    /// bit for bit and moves nothing.
    #[test]
    fn signed_zero_iterations_is_a_true_no_op() {
        let x = rows(30);
        let y: Vec<f64> = x.iter().map(|r| -(0.8 * r[0]).sin() + 0.2).collect();
        let mut tree = gauge_tree();
        let before = fitted(&tree);
        let start = score(&tree, &x, &y, Metric::CorrelationSigned);
        let got = fit(&mut tree, &x, &y, &[], 0, Metric::CorrelationSigned);
        assert_eq!(got.to_bits(), start.to_bits());
        assert_eq!(fitted(&tree), before);
    }
}
