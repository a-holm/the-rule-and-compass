//! Analytic constant optimisation.
//!
//! Structures found by the evolutionary search are usually *almost* right: the
//! shape `a·sin(b·x) + c` is easy to find, but landing `a`, `b`, `c` on the
//! exact values by random perturbation is hopeless. Instead of perturbing
//! constants one at a time (coordinate descent), we treat the constants as the
//! parameters of a nonlinear least-squares problem and solve it with
//! Gauss–Newton / Levenberg–Marquardt.
//!
//! The Jacobian is obtained by **adjoint (backpropagation) differentiation** of
//! the expression tree, so all constants get their gradient in a single
//! backward sweep — cost O(nodes) per row, independent of how many constants
//! there are. This is the single biggest accuracy-per-unit-work win in the
//! engine.

use crate::expr::{self, Node, Op};

/// `1/sqrt(pi)`, the constant in the error-function derivative.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Flattened expression: children always precede parents, so index order is a
/// post-order and the backward (adjoint) sweep is a simple descending loop.
struct Flat {
    op: Vec<Op>,
    val: Vec<f64>,
    kids: Vec<Vec<usize>>,
    /// constant slot for `Const` nodes, `-1` otherwise
    cslot: Vec<i32>,
    root: usize,
    nconst: usize,
}

fn build(n: &Node, f: &mut Flat) -> usize {
    let mut kids = Vec::with_capacity(n.kids.len());
    for k in &n.kids {
        kids.push(build(k, f));
    }
    let slot = if n.op == Op::Const {
        let s = f.nconst as i32;
        f.nconst += 1;
        s
    } else {
        -1
    };
    f.op.push(n.op);
    f.val.push(n.val);
    f.kids.push(kids);
    f.cslot.push(slot);
    f.op.len() - 1
}

fn flatten(root: &Node) -> Flat {
    let mut f = Flat {
        op: Vec::new(),
        val: Vec::new(),
        kids: Vec::new(),
        cslot: Vec::new(),
        root: 0,
        nconst: 0,
    };
    f.root = build(root, &mut f);
    f
}

/// Pre-order node indices of `Const` nodes, matching `expr::nth` indexing.
fn const_positions(n: &Node, out: &mut Vec<usize>, counter: &mut usize) {
    let mine = *counter;
    *counter += 1;
    if n.op == Op::Const {
        out.push(mine);
    }
    for k in &n.kids {
        const_positions(k, out, counter);
    }
}

#[inline]
fn value_of(op: Op, kids: &[usize], v: &[f64], node_val: f64) -> f64 {
    use Op::*;
    let a = |j: usize| v[kids[j]];
    match op {
        Const => node_val,
        Input => node_val, // unused: inputs are folded in below
        Neg => -a(0),
        Exp => a(0).exp(),
        Ln => a(0).ln(),
        Abs => a(0).abs(),
        Sqrt => a(0).sqrt(),
        Sin => a(0).sin(),
        Cos => a(0).cos(),
        Tan => a(0).tan(),
        Sinh => a(0).sinh(),
        Cosh => a(0).cosh(),
        Tanh => a(0).tanh(),
        Logistic => {
            let x = a(0);
            1.0 / (1.0 + (-x).exp())
        }
        Step => {
            if a(0) > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Gauss => {
            let x = a(0);
            (-x * x).exp()
        }
        Square => a(0) * a(0),
        Cube => a(0) * a(0) * a(0),
        Floor => a(0).floor(),
        Ceil => a(0).ceil(),
        Round => a(0).round(),
        Sign => {
            let x = a(0);
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else if x == 0.0 {
                0.0
            } else {
                f64::NAN
            }
        }
        Erf => expr::erf(a(0)),
        Erfc => expr::erfc(a(0)),
        Asinh => a(0).asinh(),
        Acosh => {
            let x = a(0);
            if x >= 1.0 {
                x.acosh()
            } else {
                f64::NAN
            }
        }
        Atanh => {
            let x = a(0);
            if x.abs() < 1.0 {
                x.atanh()
            } else {
                f64::NAN
            }
        }
        Add => a(0) + a(1),
        Sub => a(0) - a(1),
        Mul => a(0) * a(1),
        Div => a(0) / a(1),
        Pow => a(0).powf(a(1)),
        Mod => {
            let (x, b) = (a(0), a(1));
            if !x.is_finite() || !b.is_finite() || b.abs() < 1e-12 {
                f64::NAN
            } else {
                x - b * (x / b).floor()
            }
        }
        Min => {
            let (x, y) = (a(0), a(1));
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.min(y)
            }
        }
        Max => {
            let (x, y) = (a(0), a(1));
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.max(y)
            }
        }
        Atan2 => a(0).atan2(a(1)),
        If => {
            if a(0) > 0.0 {
                a(1)
            } else {
                a(2)
            }
        }
    }
}

/// Evaluate one row into `v` (length = number of flat nodes). Constants read
/// from `c[slot]`. Inputs are read from `x`.
fn eval_row(f: &Flat, x: &[f64], c: &[f64], v: &mut [f64]) {
    for i in 0..f.op.len() {
        v[i] = if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize]
        } else if f.op[i] == Op::Input {
            let idx = f.val[i] as usize;
            if idx < x.len() {
                x[idx]
            } else {
                f64::NAN
            }
        } else {
            value_of(f.op[i], &f.kids[i], v, f.val[i])
        };
    }
}

/// Set the gradient of `out` w.r.t. the constants into `g`.
fn backward(f: &Flat, v: &[f64], adj: &mut [f64], g: &mut [f64]) {
    for a in adj.iter_mut() {
        *a = 0.0;
    }
    for x in g.iter_mut() {
        *x = 0.0;
    }
    adj[f.root] = 1.0;
    for i in (0..f.op.len()).rev() {
        let a = adj[i];
        if a == 0.0 {
            continue;
        }
        if f.op[i] == Op::Const {
            g[f.cslot[i] as usize] += a;
            continue;
        }
        let ks = &f.kids[i];
        if ks.is_empty() {
            continue;
        }
        let get = |j: usize| v[ks[j]];
        use Op::*;
        match f.op[i] {
            Neg => adj[ks[0]] += -a,
            Exp => adj[ks[0]] += a * get(0).exp(),
            Ln => adj[ks[0]] += a / get(0),
            Abs => adj[ks[0]] += a * if get(0) >= 0.0 { 1.0 } else { -1.0 },
            Sqrt => adj[ks[0]] += a / (2.0 * get(0).sqrt()),
            Sin => adj[ks[0]] += a * get(0).cos(),
            Cos => adj[ks[0]] += -a * get(0).sin(),
            Tan => {
                let t = get(0).cos();
                adj[ks[0]] += a / (t * t);
            }
            Sinh => adj[ks[0]] += a * get(0).cosh(),
            Cosh => adj[ks[0]] += a * get(0).sinh(),
            Tanh => {
                let t = get(0).tanh();
                adj[ks[0]] += a * (1.0 - t * t);
            }
            Logistic => {
                let s = 1.0 / (1.0 + (-get(0)).exp());
                adj[ks[0]] += a * s * (1.0 - s);
            }
            Step => {}
            Gauss => adj[ks[0]] += a * (-2.0 * get(0)) * (-get(0) * get(0)).exp(),
            Square => adj[ks[0]] += a * 2.0 * get(0),
            Cube => adj[ks[0]] += a * 3.0 * get(0) * get(0),
            // floor/ceil/round/sign are piecewise constant, derivative 0 a.e.
            Floor | Ceil | Round | Sign => {}
            Erf => adj[ks[0]] += a * 2.0 * INV_SQRT_PI * (-get(0) * get(0)).exp(),
            Erfc => adj[ks[0]] += -a * 2.0 * INV_SQRT_PI * (-get(0) * get(0)).exp(),
            Asinh => adj[ks[0]] += a / (get(0) * get(0) + 1.0).sqrt(),
            Acosh => {
                let x = get(0);
                if x > 1.0 {
                    adj[ks[0]] += a / (x * x - 1.0).sqrt();
                }
            }
            Atanh => {
                let x = get(0);
                if x.abs() < 1.0 {
                    adj[ks[0]] += a / (1.0 - x * x);
                }
            }
            Add => {
                adj[ks[0]] += a;
                adj[ks[1]] += a;
            }
            Sub => {
                adj[ks[0]] += a;
                adj[ks[1]] += -a;
            }
            Mul => {
                adj[ks[0]] += a * get(1);
                adj[ks[1]] += a * get(0);
            }
            Div => {
                let d = get(1);
                adj[ks[0]] += a / d;
                adj[ks[1]] += -a * get(0) / (d * d);
            }
            Pow => {
                let (b, e) = (get(0), get(1));
                if b > 0.0 {
                    adj[ks[0]] += a * e * b.powf(e - 1.0);
                    adj[ks[1]] += a * b.powf(e) * b.ln();
                }
            }
            // `mod(a,b) = a - b*floor(a/b)`; subgradient 1 for `a`, `-floor(a/b)`
            // for `b`, undefined (0) where `b = 0`.
            Mod => {
                let b = get(1);
                if b.abs() >= 1e-12 {
                    adj[ks[0]] += a;
                    adj[ks[1]] += -a * (get(0) / b).floor();
                }
            }
            // `min`/`max` subgradient: all flow to the active argument, split
            // evenly on a tie (a documented, symmetric choice).
            Min => {
                let (x, y) = (get(0), get(1));
                if x < y {
                    adj[ks[0]] += a;
                } else if y < x {
                    adj[ks[1]] += a;
                } else {
                    adj[ks[0]] += 0.5 * a;
                    adj[ks[1]] += 0.5 * a;
                }
            }
            Max => {
                let (x, y) = (get(0), get(1));
                if x > y {
                    adj[ks[0]] += a;
                } else if y > x {
                    adj[ks[1]] += a;
                } else {
                    adj[ks[0]] += 0.5 * a;
                    adj[ks[1]] += 0.5 * a;
                }
            }
            Atan2 => {
                let (y, x) = (get(0), get(1));
                let d = x * x + y * y;
                if d > 0.0 {
                    adj[ks[0]] += a * x / d;
                    adj[ks[1]] += -a * y / d;
                }
            }
            If => {
                // derivative flows only to the taken branch; condition is
                // piecewise-constant.
                if get(0) > 0.0 {
                    adj[ks[1]] += a;
                } else {
                    adj[ks[2]] += a;
                }
            }
            Const | Input => {}
        }
    }
}

/// Solve `(A + λ·diag(A)) δ = b` for small symmetric `A` (Cholesky).
fn solve(a: &mut [Vec<f64>], b: &mut [f64], lambda: f64, k: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    for i in 0..k {
        m[i][i] = m[i][i] * (1.0 + lambda) + 1e-12;
    }
    // Cholesky
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = m[i][j];
            for p in 0..j {
                s -= l[i][p] * l[j][p];
            }
            if i == j {
                if s <= 1e-18 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    // forward/back substitution
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut s = b[i];
        for p in 0..i {
            s -= l[i][p] * y[p];
        }
        y[i] = s / l[i][i];
    }
    let mut d = vec![0.0; k];
    for i in (0..k).rev() {
        let mut s = y[i];
        for p in (i + 1)..k {
            s -= l[p][i] * d[p];
        }
        d[i] = s / l[i][i];
    }
    Some(d)
}

fn sse(f: &Flat, x: &[Vec<f64>], y: &[f64], c: &[f64], v: &mut [f64]) -> f64 {
    let mut s = 0.0;
    for (row, &t) in x.iter().zip(y) {
        eval_row(f, row, c, v);
        let r = v[f.root] - t;
        if !r.is_finite() {
            return f64::INFINITY;
        }
        s += r * r;
    }
    s
}

/// One Gauss–Newton/Levenberg–Marquardt step. Returns the trial constants and
/// their sum of squared errors, or `None` if the normal equations are singular.
#[allow(clippy::too_many_arguments)]
fn gn_step(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    c: &[f64],
    lambda: f64,
    k: usize,
    v: &mut [f64],
    adj: &mut [f64],
    grad: &mut [f64],
) -> Option<(Vec<f64>, f64)> {
    let mut jtj = vec![vec![0.0; k]; k];
    let mut jtr = vec![0.0; k];
    for (row, &t) in x.iter().zip(y) {
        eval_row(f, row, c, v);
        let r = v[f.root] - t;
        if !r.is_finite() {
            return None;
        }
        backward(f, v, adj, grad);
        for i in 0..k {
            let gi = grad[i];
            if !gi.is_finite() {
                return None;
            }
            jtr[i] += gi * r;
            for j in 0..=i {
                jtj[i][j] += gi * grad[j];
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            jtj[j][i] = jtj[i][j];
        }
    }
    let mut rhs: Vec<f64> = jtr.iter().map(|v| -v).collect();
    let step = solve(&mut jtj, &mut rhs, lambda, k)?;
    let trial: Vec<f64> = (0..k)
        .map(|i| (c[i] + step[i]).clamp(-1e6, 1e6))
        .collect();
    let s = sse(f, x, y, &trial, v);
    Some((trial, s))
}

/// Fit all constants of `tree` to minimise squared error on `x`/`y`.
///
/// A damped Levenberg–Marquardt phase finds the basin, then a short undamped
/// Gauss–Newton polish drives the constants to machine precision on
/// well-conditioned problems. The tree is only modified if the fit improves it.
/// Returns the final sum of squared errors.
pub fn fit(tree: &mut Node, x: &[Vec<f64>], y: &[f64], iters: usize) -> f64 {
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || x.is_empty() {
        let mut v = vec![0.0; f.op.len().max(1)];
        return sse(&f, x, y, &[], &mut v);
    }
    let n = f.op.len();
    let mut c = vec![0.0; k];
    for i in 0..n {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    let mut v = vec![0.0; n];
    let mut adj = vec![0.0; n];
    let mut grad = vec![0.0; k];
    let mut best_c = c.clone();
    let mut best_sse = sse(&f, x, y, &c, &mut v);
    let mut lambda = 1e-3;

    for _ in 0..iters.max(1) {
        match gn_step(&f, x, y, &c, lambda, k, &mut v, &mut adj, &mut grad) {
            Some((trial, tsse)) => {
                if tsse < best_sse {
                    c = trial;
                    best_sse = tsse;
                    best_c = c.clone();
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
            None => break,
        }
    }

    // Undamped Gauss–Newton polish: quadratic convergence to the optimum.
    for _ in 0..4 {
        if best_sse == 0.0 {
            break;
        }
        match gn_step(&f, x, y, &best_c, 0.0, k, &mut v, &mut adj, &mut grad) {
            Some((trial, tsse)) if tsse < best_sse => {
                best_c = trial;
                best_sse = tsse;
            }
            _ => break,
        }
    }

    // write back
    let mut positions = Vec::new();
    let mut counter = 0;
    const_positions(tree, &mut positions, &mut counter);
    for (slot, &pos) in positions.iter().enumerate() {
        if let Some(t) = expr::nth_mut(tree, pos) {
            t.val = best_c[slot];
        }
    }
    best_sse
}

/// Analytic derivative of the whole expression w.r.t. one input variable,
/// evaluated over `x` (used for implicit/derivative objectives and for
/// checking candidate solutions). Kept simple: central finite differences.
pub fn partial(tree: &Node, x: &[f64], var: usize) -> f64 {
    let h = 1e-6;
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[var] += h;
    xm[var] -= h;
    (expr::eval(tree, &xp) - expr::eval(tree, &xm)) / (2.0 * h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    /// Gradient of the root w.r.t. every constant, via the adjoint sweep.
    fn analytic_grad(tree: &Node, x: &[f64]) -> Vec<f64> {
        let f = flatten(tree);
        let n = f.op.len();
        let mut c = vec![0.0; f.nconst];
        for i in 0..n {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        let mut v = vec![0.0; n];
        let mut adj = vec![0.0; n];
        let mut g = vec![0.0; f.nconst];
        eval_row(&f, x, &c, &mut v);
        backward(&f, &v, &mut adj, &mut g);
        g
    }

    /// Compare the adjoint gradient against central finite differences of the
    /// evaluator, perturbing one constant at a time.
    fn check_grad<F: Fn(&[f64]) -> Node>(build: F, c: &[f64], x: &[f64]) {
        let tree = build(c);
        let g = analytic_grad(&tree, x);
        assert_eq!(g.len(), c.len());
        let h = 1e-6;
        for k in 0..c.len() {
            let mut cp = c.to_vec();
            let mut cm = c.to_vec();
            cp[k] += h;
            cm[k] -= h;
            let fd = (expr::eval(&build(&cp), x) - expr::eval(&build(&cm), x)) / (2.0 * h);
            assert!(
                (g[k] - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
                "constant {k}: analytic {a} vs finite difference {fd}",
                a = g[k]
            );
        }
    }

    #[test]
    fn adjoint_matches_finite_differences() {
        let x = [0.4];
        // Modified Bessel-free special functions: erf, erfc, asinh.
        check_grad(
            |c| node(Op::Erf, vec![node(Op::Mul, vec![constant(c[0]), input(0)])]),
            &[1.3],
            &x,
        );
        check_grad(
            |c| node(Op::Erfc, vec![node(Op::Mul, vec![constant(c[0]), input(0)])]),
            &[1.3],
            &x,
        );
        check_grad(
            |c| node(Op::Asinh, vec![node(Op::Mul, vec![constant(c[0]), input(0)])]),
            &[0.7],
            &[0.5],
        );
        // acosh / atanh, evaluated strictly inside their domains.
        check_grad(
            |c| node(Op::Acosh, vec![node(Op::Add, vec![constant(c[0]), input(0)])]),
            &[2.0],
            &[0.5],
        );
        check_grad(
            |c| node(Op::Atanh, vec![node(Op::Mul, vec![constant(c[0]), input(0)])]),
            &[0.5],
            &[0.8],
        );
        // atan2(c, x): d/dc = x / (x^2 + c^2).
        check_grad(|c| node(Op::Atan2, vec![constant(c[0]), input(0)]), &[1.2], &[2.0]);
        // mod with both arguments movable: d/da = 1, d/db = -floor(a/b).
        check_grad(
            |c| {
                node(
                    Op::Mod,
                    vec![
                        node(Op::Mul, vec![constant(c[0]), input(0)]),
                        constant(c[1]),
                    ],
                )
            },
            &[3.0, 4.0],
            &[1.7],
        );
    }

    #[test]
    fn min_max_subgradient_flows_to_active_argument() {
        let x = [3.0];
        // c is the smaller argument: all of the gradient goes to it.
        check_grad(|c| node(Op::Min, vec![constant(c[0]), input(0)]), &[-1.0], &x);
        // c is the larger argument.
        check_grad(|c| node(Op::Max, vec![constant(c[0]), input(0)]), &[5.0], &x);

        // On a tie the subgradient is split evenly between the arguments.
        let tie = node(Op::Min, vec![constant(2.0), constant(2.0)]);
        assert_eq!(analytic_grad(&tie, &[0.0]), vec![0.5, 0.5]);
        let tie = node(Op::Max, vec![constant(2.0), constant(2.0)]);
        assert_eq!(analytic_grad(&tie, &[0.0]), vec![0.5, 0.5]);
    }

    #[test]
    fn floor_like_operators_have_zero_gradient() {
        // Piecewise-constant operators: derivative 0 away from the jumps.
        for op in [Op::Floor, Op::Ceil, Op::Round, Op::Sign, Op::Step] {
            let tree = node(op, vec![node(Op::Mul, vec![constant(1.3), input(0)])]);
            assert_eq!(analytic_grad(&tree, &[1.7]), vec![0.0]);
        }
    }
}
