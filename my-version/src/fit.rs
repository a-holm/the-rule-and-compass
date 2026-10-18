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
    /// Shared constant nodes inside block bodies, keyed by (block slot,
    /// constant ordinal in that body). Every use of a block maps its body
    /// constants to the same flat node, so one constant is fitted once and all
    /// uses follow.
    shared: std::collections::HashMap<(usize, usize), usize>,
    /// The constant slots of ordinary (non-body) constants, in the same order
    /// as `const_positions` visits them. Body constants are omitted.
    normal_order: Vec<usize>,
}

fn push_flat(f: &mut Flat, op: Op, val: f64, kids: Vec<usize>, cslot: i32) -> usize {
    f.op.push(op);
    f.val.push(val);
    f.kids.push(kids);
    f.cslot.push(cslot);
    f.op.len() - 1
}

fn build(n: &Node, f: &mut Flat) -> usize {
    // A block call is inlined for the fit: the body is built in parameter
    // space, its parameter leaves are replaced by the argument sub-trees, and
    // its constants are shared (see `Flat::shared`).
    if let Some(slot) = n.op.custom_slot() {
        let args: Vec<usize> = n.kids.iter().map(|k| build(k, f)).collect();
        if let Some(block) = &n.block {
            let mut ordinal = 0;
            return build_body(&block.body, f, slot, &args, &mut ordinal);
        }
        return push_flat(f, Op::Const, f64::NAN, Vec::new(), -1);
    }
    let mut kids = Vec::with_capacity(n.kids.len());
    for k in &n.kids {
        kids.push(build(k, f));
    }
    let slot = if n.op == Op::Const {
        let s = f.nconst as i32;
        f.nconst += 1;
        f.normal_order.push(s as usize);
        s
    } else {
        -1
    };
    push_flat(f, n.op, n.val, kids, slot)
}

/// Build one block body. `slot` identifies the block whose constants are
/// shared, `args` are the flat indices of the call's argument sub-trees (for
/// `Input(i)`) and `ordinal` counts the body's constants in traversal order.
fn build_body(
    n: &Node,
    f: &mut Flat,
    slot: usize,
    args: &[usize],
    ordinal: &mut usize,
) -> usize {
    if n.op == Op::Input {
        let i = n.val as usize;
        return args.get(i).copied().unwrap_or_else(|| {
            push_flat(f, Op::Const, f64::NAN, Vec::new(), -1)
        });
    }
    if n.op == Op::Const {
        let ord = *ordinal;
        *ordinal += 1;
        if let Some(&ix) = f.shared.get(&(slot, ord)) {
            return ix;
        }
        let s = f.nconst as i32;
        f.nconst += 1;
        let ix = push_flat(f, Op::Const, n.val, Vec::new(), s);
        f.shared.insert((slot, ord), ix);
        return ix;
    }
    if let Some(inner) = n.op.custom_slot() {
        // A block used inside another block: its arguments are in the outer
        // body's parameter space, then its own body gets a fresh ordinal.
        let arg_ix: Vec<usize> = n
            .kids
            .iter()
            .map(|k| build_body(k, f, slot, args, ordinal))
            .collect();
        if let Some(block) = &n.block {
            let mut inner_ord = 0;
            return build_body(&block.body, f, inner, &arg_ix, &mut inner_ord);
        }
        return push_flat(f, Op::Const, f64::NAN, Vec::new(), -1);
    }
    let kids: Vec<usize> = n
        .kids
        .iter()
        .map(|k| build_body(k, f, slot, args, ordinal))
        .collect();
    push_flat(f, n.op, n.val, kids, -1)
}

fn flatten(root: &Node) -> Flat {
    let mut f = Flat {
        op: Vec::new(),
        val: Vec::new(),
        kids: Vec::new(),
        cslot: Vec::new(),
        root: 0,
        nconst: 0,
        shared: std::collections::HashMap::new(),
        normal_order: Vec::new(),
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
        Equal => cmp_f(a(0), a(1), |x, y| x == y),
        Less => cmp_f(a(0), a(1), |x, y| x < y),
        LessOrEqual => cmp_f(a(0), a(1), |x, y| x <= y),
        Greater => cmp_f(a(0), a(1), |x, y| x > y),
        GreaterOrEqual => cmp_f(a(0), a(1), |x, y| x >= y),
        And => logical(a(0), a(1), |x, y| x && y),
        Or => logical(a(0), a(1), |x, y| x || y),
        Xor => logical(a(0), a(1), |x, y| x != y),
        Not => match truth(a(0)) {
            Some(t) => {
                if t {
                    0.0
                } else {
                    1.0
                }
            }
            None => f64::NAN,
        },
        Require | Constrain => {
            if a(0) > 0.0 {
                a(0)
            } else {
                f64::NAN
            }
        }
        Between => {
            let (v, lo, hi) = (a(0), a(1), a(2));
            if lo <= v && v <= hi {
                v
            } else {
                f64::NAN
            }
        }
        // Meta operators return the structural value cached on the node. It is
        // row-independent by construction; the flat copy keeps it so a fitted
        // constant inside the argument cannot change it.
        Complexity | Variables | Constants | Contains | ContainsOp => node_val,
        // An aggregate's cached dataset value, likewise copied into the flat
        // representation and never recomputed per row.
        Agg(_) => node_val,
        // Block calls are inlined by `build`, so no custom operator reaches
        // the flat representation.
        Custom0 | Custom1 | Custom2 | Custom3 => f64::NAN,
    }
}

/// Truthiness of a value for the logical operators: `None` for `NaN`, otherwise
/// whether it is strictly positive.
#[inline]
fn truth(v: f64) -> Option<bool> {
    if v.is_nan() {
        None
    } else {
        Some(v > 0.0)
    }
}

/// 0/1 comparison, `NaN` if an operand is `NaN` (mirrors `expr::eval`).
#[inline]
fn cmp_f(a: f64, b: f64, f: impl Fn(f64, f64) -> bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if f(a, b) {
        1.0
    } else {
        0.0
    }
}

/// 0/1 logical operator on the truthiness of its arguments.
#[inline]
fn logical(a: f64, b: f64, f: impl Fn(bool, bool) -> bool) -> f64 {
    match (truth(a), truth(b)) {
        (Some(x), Some(y)) => {
            if f(x, y) {
                1.0
            } else {
                0.0
            }
        }
        _ => f64::NAN,
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
    backward_impl(f, v, adj, Some(g), None);
}

/// The adjoint (reverse-mode) sweep shared by the constant fit and the
/// variable-sensitivity report.
///
/// Seeded with `adj[root] = 1`, it accumulates the derivative of the root into
/// either the constant slots (`const_grad`) or the input leaves (`input_grad`).
/// Passing only one of the two keeps the fitter's cost unchanged; passing both
/// would give the full Jacobian in one pass.
fn backward_impl(
    f: &Flat,
    v: &[f64],
    adj: &mut [f64],
    const_grad: Option<&mut [f64]>,
    input_grad: Option<&mut [f64]>,
) {
    let mut const_grad = const_grad;
    let mut input_grad = input_grad;
    for a in adj.iter_mut() {
        *a = 0.0;
    }
    if let Some(g) = const_grad.as_deref_mut() {
        for x in g.iter_mut() {
            *x = 0.0;
        }
    }
    if let Some(g) = input_grad.as_deref_mut() {
        for x in g.iter_mut() {
            *x = 0.0;
        }
    }
    adj[f.root] = 1.0;
    for i in (0..f.op.len()).rev() {
        let a = adj[i];
        if a == 0.0 {
            continue;
        }
        if f.op[i] == Op::Const {
            if let Some(g) = const_grad.as_deref_mut() {
                if let Some(slot) = g.get_mut(f.cslot[i] as usize) {
                    *slot += a;
                }
            }
            continue;
        }
        if f.op[i] == Op::Input {
            if let Some(g) = input_grad.as_deref_mut() {
                if let Some(slot) = g.get_mut(f.val[i] as usize) {
                    *slot += a;
                }
            }
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
            // Comparisons are piecewise constant (0/1), so flat a.e.: the
            // subgradient is 0, like `step`. Same for `not`.
            Equal | Less | LessOrEqual | Greater | GreaterOrEqual | Not => {}
            // `and`/`or`/`xor` are flat a.e. too. Their documented subgradients
            // are the partials of the product surrogate over the truth values
            // `[a>0]`, `[b>0]`:
            //   and: [a>0][b>0]            -> d/da = [b>0], d/db = [a>0]
            //   or:  [a>0]+[b>0]-[a>0][b>0] -> d/da = 1-[b>0], d/db = 1-[a>0]
            //   xor: [a>0]+[b>0]-2[a>0][b>0] -> d/da = 1-2[b>0], d/db = 1-2[a>0]
            // These are bounded (|g| <= 2) and symmetric in the two arguments,
            // matching the min/max convention above.
            And => {
                adj[ks[0]] += a * ind(get(1));
                adj[ks[1]] += a * ind(get(0));
            }
            Or => {
                adj[ks[0]] += a * (1.0 - ind(get(1)));
                adj[ks[1]] += a * (1.0 - ind(get(0)));
            }
            Xor => {
                adj[ks[0]] += a * (1.0 - 2.0 * ind(get(1)));
                adj[ks[1]] += a * (1.0 - 2.0 * ind(get(0)));
            }
            // A constraint returns its argument while feasible, so the gradient
            // passes through with weight 1; once violated it is `NaN` and the
            // row is already rejected, so nothing is added.
            Require | Constrain => {
                if get(0) > 0.0 {
                    adj[ks[0]] += a;
                }
            }
            // `between(v, lo, hi)` returns `v` when `lo <= v <= hi`; the bounds
            // are not part of the returned value, so they get no gradient.
            Between => {
                let (v, lo, hi) = (get(0), get(1), get(2));
                if lo <= v && v <= hi {
                    adj[ks[0]] += a;
                }
            }
            // Meta operators are structural constants: the fitter must not
            // move them, so their gradient is zero (like `floor`/`step`).
            // Nothing flows into the argument either.
            Complexity | Variables | Constants | Contains | ContainsOp => {}
            // An aggregate is a dataset fact, not a fitted parameter: zero
            // gradient, and nothing flows into its column child.
            Agg(_) => {}
            // Block calls are inlined by `build`, so this is unreachable.
            Custom0 | Custom1 | Custom2 | Custom3 => {}
            Const | Input => {}
        }
    }
}

/// Indicator `[x > 0]` as 0/1, `NaN` when `x` is `NaN` (a rejected row).
#[inline]
fn ind(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else {
        0.0
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

/// Weighted residual sum of squares. The residual is multiplied by `sqrt(wᵢ)`
/// so that minimising this is weighted least squares; an empty `w` leaves the
/// residual untouched (the unweighted path, bit for bit).
fn sse(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    v: &mut [f64],
) -> f64 {
    if !w.is_empty() && w.len() != x.len().min(y.len()) {
        return f64::INFINITY;
    }
    let mut s = 0.0;
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue; // a zero-weight row never influences the fit
        }
        eval_row(f, row, c, v);
        let r = sw.sqrt() * (v[f.root] - t);
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
    w: &[f64],
    c: &[f64],
    lambda: f64,
    k: usize,
    v: &mut [f64],
    adj: &mut [f64],
    grad: &mut [f64],
) -> Option<(Vec<f64>, f64)> {
    if !w.is_empty() && w.len() != x.len().min(y.len()) {
        return None;
    }
    let mut jtj = vec![vec![0.0; k]; k];
    let mut jtr = vec![0.0; k];
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue; // a zero-weight row contributes nothing to JᵀJ or Jᵀr
        }
        eval_row(f, row, c, v);
        // Scaling the residual by sqrt(w) makes JᵀJ and Jᵀr the weighted normal
        // equations: (JᵀWJ) δ = -JᵀWr.
        let r = sw.sqrt() * (v[f.root] - t);
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
    let s = sse(f, x, y, w, &trial, v);
    Some((trial, s))
}

/// Fit all constants of `tree` to minimise squared error on `x`/`y`.
///
/// A damped Levenberg–Marquardt phase finds the basin, then a short undamped
/// Gauss–Newton polish drives the constants to machine precision on
/// well-conditioned problems. The tree is only modified if the fit improves it.
/// Returns the final sum of squared errors.
///
/// This is the unweighted entry point; it delegates to [`fit_weighted`] with an
/// empty weight vector, so existing results are bit-for-bit unchanged.
pub fn fit(tree: &mut Node, x: &[Vec<f64>], y: &[f64], iters: usize) -> f64 {
    fit_weighted(tree, x, y, &[], iters)
}

/// Weighted least squares: fit `tree`'s constants to minimise
/// `Σ wᵢ·(f(xᵢ) − yᵢ)²`.
///
/// Each residual is multiplied by `sqrt(wᵢ)` before it enters `JᵀJ` and `Jᵀr`,
/// so the Gauss–Newton step solves the weighted normal equations
/// `(JᵀWJ) δ = −JᵀWr`. An empty `w` is exactly [`fit`]; a zero weight removes
/// the row from the objective (it cannot influence the constants). Returns the
/// final weighted sum of squared errors.
pub fn fit_weighted(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
) -> f64 {
    if !w.is_empty() && w.len() != x.len().min(y.len()) {
        return f64::INFINITY;
    }
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || x.is_empty() {
        let mut v = vec![0.0; f.op.len().max(1)];
        return sse(&f, x, y, w, &[], &mut v);
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
    let mut best_sse = sse(&f, x, y, w, &c, &mut v);
    let mut lambda = 1e-3;

    for _ in 0..iters.max(1) {
        match gn_step(&f, x, y, w, &c, lambda, k, &mut v, &mut adj, &mut grad) {
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
        match gn_step(
            &f,
            x,
            y,
            w,
            &best_c,
            0.0,
            k,
            &mut v,
            &mut adj,
            &mut grad,
        ) {
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
    for (pos, &slot) in positions.iter().zip(&f.normal_order) {
        if let Some(t) = expr::nth_mut(tree, *pos) {
            t.val = best_c[slot];
        }
    }
    // Shared constants live in block bodies, which `nth` does not traverse;
    // every use of a slot gets the same fitted value. `Flat::shared` holds the
    // reused *flat node* index, so translate it to the constant slot first.
    let shared_slots: std::collections::HashMap<(usize, usize), usize> = f
        .shared
        .iter()
        .map(|(key, &ix)| (*key, f.cslot[ix].max(0) as usize))
        .collect();
    apply_shared_constants(tree, &shared_slots, &best_c);
    best_sse
}

/// Write the fitted shared constants back into every block body in `tree`.
fn apply_shared_constants(
    n: &mut Node,
    shared: &std::collections::HashMap<(usize, usize), usize>,
    best: &[f64],
) {
    if let Some(slot) = n.op.custom_slot() {
        if let Some(block) = &mut n.block {
            let mut ordinal = 0;
            apply_shared_body(&mut block.body, slot, shared, best, &mut ordinal);
        }
    }
    for k in &mut n.kids {
        apply_shared_constants(k, shared, best);
    }
}

fn apply_shared_body(
    n: &mut Node,
    slot: usize,
    shared: &std::collections::HashMap<(usize, usize), usize>,
    best: &[f64],
    ordinal: &mut usize,
) {
    if n.op == Op::Input {
        return;
    }
    if n.op == Op::Const {
        let ord = *ordinal;
        *ordinal += 1;
        if let Some(&cs) = shared.get(&(slot, ord)) {
            n.val = best[cs];
        }
        return;
    }
    if let Some(inner) = n.op.custom_slot() {
        for k in &mut n.kids {
            apply_shared_body(k, slot, shared, best, ordinal);
        }
        if let Some(block) = &mut n.block {
            let mut inner_ord = 0;
            apply_shared_body(&mut block.body, inner, shared, best, &mut inner_ord);
        }
        return;
    }
    for k in &mut n.kids {
        apply_shared_body(k, slot, shared, best, ordinal);
    }
}

/// Standard errors of the fitted constants, from the Gauss–Newton Jacobian at
/// the solution.
///
/// Treating the fitted constants as the parameters of a nonlinear
/// least-squares problem, the parameter covariance is
///
/// ```text
/// cov = sigma^2 * (JᵀJ)^-1,     sigma^2 = RSS / (n - k)
/// ```
///
/// where `J` is the residual Jacobian w.r.t. the constants (one row per data
/// point, from the same adjoint sweep the fitter uses), `RSS` the residual sum
/// of squares and `n - k` the residual degrees of freedom. The square roots of
/// the diagonal of `cov` are the constants' standard errors; dividing by
/// `|value|` gives a unit-free relative uncertainty.
///
/// `JᵀJ` is scaled to a correlation-like matrix before inversion, so the
/// conditioning test does not depend on the magnitude of each constant, and
/// inverted by Gaussian elimination with partial pivoting. When the matrix is
/// (near-)singular the constants can trade off against one another:
/// `ill_conditioned` is set and the unidentified ones get infinite errors.
#[derive(Clone, Debug)]
pub struct ConstantErrors {
    /// Fitted value of each constant, left to right (the [`crate::expr::nth`]
    /// order).
    pub values: Vec<f64>,
    /// `sqrt(cov_ii)`; infinite when the constant is not identified.
    pub std_errors: Vec<f64>,
    /// `std_errors[i] / |values[i]|`; infinite for a zero or unidentified one.
    pub relative: Vec<f64>,
    /// Residual degrees of freedom `n - k`.
    pub dof: usize,
    /// `JᵀJ` is (near-)singular: the constants are not separately determined.
    pub ill_conditioned: bool,
}

/// Compute each fitted constant's standard error at `tree`'s current values.
///
/// Returns `None` when there is nothing to report: no constants, no rows, or no
/// residual degrees of freedom (`n <= k`, so `sigma^2` is undefined). The
/// returned values are in left-to-right constant order, matching
/// [`crate::expr::nth`].
pub fn constant_errors(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> Option<ConstantErrors> {
    constant_errors_weighted(tree, x, y, &[])
}

/// Weighted standard errors. `JᵀJ` and the RSS are weighted (`Σ wᵢ·…`), and the
/// residual degrees of freedom are `#rows with wᵢ > 0 − k`: a weight is treated
/// as a frequency, so a zero-weight row does not buy a degree of freedom. An
/// empty `w` reproduces [`constant_errors`] exactly.
pub fn constant_errors_weighted(
    tree: &Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
) -> Option<ConstantErrors> {
    if !w.is_empty() && w.len() != x.len().min(y.len()) {
        return None;
    }
    let f = flatten(tree);
    let k = f.nconst;
    let n = x.len().min(y.len());
    let npos = if w.is_empty() {
        n
    } else {
        w.iter().filter(|&&v| v > 0.0).count()
    };
    if k == 0 || n == 0 || npos <= k {
        return None;
    }
    let mut c = vec![0.0; k];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    let mut v = vec![0.0; f.op.len()];
    let mut adj = vec![0.0; f.op.len()];
    let mut grad = vec![0.0; k];
    let mut jtj = vec![vec![0.0; k]; k];
    let mut rss = 0.0;
    for (row_i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(row_i).copied().unwrap_or(1.0);
        if !sw.is_finite() || sw < 0.0 {
            return None;
        }
        if sw == 0.0 {
            continue;
        }
        eval_row(&f, row, &c, &mut v);
        let r = v[f.root] - t;
        if !r.is_finite() {
            return None;
        }
        backward(&f, &v, &mut adj, &mut grad);
        for i in 0..k {
            let gi = grad[i];
            if !gi.is_finite() {
                return None;
            }
            for j in 0..=i {
                jtj[i][j] += sw * gi * grad[j];
            }
        }
        rss += sw * r * r;
    }
    for i in 0..k {
        for j in 0..i {
            jtj[j][i] = jtj[i][j];
        }
    }
    let dof = npos - k;
    // Scale to a correlation-like matrix so collinearity is judged
    // independently of each constant's magnitude. A constant with no gradient
    // at all (e.g. inside an untaken branch) is never determined.
    let mut scale = vec![0.0; k];
    for i in 0..k {
        if jtj[i][i] > 0.0 && jtj[i][i].is_finite() {
            scale[i] = jtj[i][i].sqrt();
        } else {
            return Some(unidentified(c, dof));
        }
    }
    let mut corr = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            corr[i][j] = jtj[i][j] / (scale[i] * scale[j]);
        }
    }
    let (inv, pivot_ratio) = match invert(&corr, k) {
        Some(r) => r,
        None => return Some(unidentified(c, dof)),
    };
    // Below this pivot ratio the columns of `J` are nearly collinear: the
    // constants can trade off and the standard errors are not meaningful.
    const ILL_CONDITIONED: f64 = 1e-12;
    let ill_conditioned = pivot_ratio < ILL_CONDITIONED;
    let sigma2 = rss / dof as f64;
    let mut std_errors = vec![0.0; k];
    let mut relative = vec![0.0; k];
    for i in 0..k {
        let cov_ii = sigma2 * inv[i][i] / (scale[i] * scale[i]);
        let se = if cov_ii >= 0.0 {
            cov_ii.sqrt()
        } else {
            f64::INFINITY
        };
        std_errors[i] = se;
        relative[i] = if c[i] == 0.0 {
            f64::INFINITY
        } else {
            se / c[i].abs()
        };
    }
    Some(ConstantErrors {
        values: c,
        std_errors,
        relative,
        dof,
        ill_conditioned,
    })
}

/// Fallback for an unidentified system: values kept, every error infinite.
fn unidentified(values: Vec<f64>, dof: usize) -> ConstantErrors {
    let k = values.len();
    ConstantErrors {
        values,
        std_errors: vec![f64::INFINITY; k],
        relative: vec![f64::INFINITY; k],
        dof,
        ill_conditioned: true,
    }
}

/// Invert a `k x k` matrix by Gauss–Jordan elimination with partial pivoting.
///
/// Returns the inverse together with `min|pivot| / max|pivot|`, a cheap
/// conditioning heuristic (near zero means nearly singular), or `None` when a
/// pivot is numerically zero.
fn invert(a: &[Vec<f64>], k: usize) -> Option<(Vec<Vec<f64>>, f64)> {
    let mut m = vec![vec![0.0; 2 * k]; k];
    for i in 0..k {
        for j in 0..k {
            m[i][j] = a[i][j];
        }
        m[i][k + i] = 1.0;
    }
    let mut max_pivot = 0.0f64;
    let mut min_pivot = f64::INFINITY;
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        let p = m[piv][col].abs();
        if p < 1e-300 {
            return None;
        }
        max_pivot = max_pivot.max(p);
        min_pivot = min_pivot.min(p);
        m.swap(col, piv);
        let d = m[col][col];
        for j in 0..2 * k {
            m[col][j] /= d;
        }
        for r in 0..k {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor != 0.0 {
                for j in 0..2 * k {
                    m[r][j] -= factor * m[col][j];
                }
            }
        }
    }
    let ratio = if max_pivot > 0.0 {
        min_pivot / max_pivot
    } else {
        0.0
    };
    let inv: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| m[i][k + j]).collect())
        .collect();
    Some((inv, ratio))
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

/// Analytic derivative of the model output w.r.t. **every input** at `x`.
///
/// This is the same adjoint (reverse-mode) sweep the fitter uses for constants
/// ([`fit`], [`constant_errors`]), seeded at the root but accumulated at the
/// `Input` leaves instead. One backward pass gives every input's derivative at
/// once, so a sensitivity analysis over `n` rows costs `O(n · nodes)` rather
/// than `2 · vars · n · nodes` for central differences.
///
/// The returned vector has one entry per element of `x`. An input that does not
/// occur in `tree` receives `0`; an input index outside `x` is ignored (the
/// same row would evaluate to `NaN`).
pub fn input_gradients(tree: &Node, x: &[f64]) -> Vec<f64> {
    let f = flatten(tree);
    let n = f.op.len();
    let mut c = vec![0.0; f.nconst];
    for i in 0..n {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    let mut v = vec![0.0; n];
    eval_row(&f, x, &c, &mut v);
    let mut adj = vec![0.0; n];
    let mut grad = vec![0.0; x.len()];
    backward_impl(&f, &v, &mut adj, None, Some(&mut grad));
    grad
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

    #[test]
    fn comparisons_and_not_have_zero_gradient() {
        // Comparisons are piecewise constant (0/1) and `not` is too, so their
        // subgradient is 0 a.e., exactly like `step`.
        for op in [
            Op::Equal,
            Op::Less,
            Op::LessOrEqual,
            Op::Greater,
            Op::GreaterOrEqual,
        ] {
            let tree = node(op, vec![constant(1.3), input(0)]);
            assert_eq!(analytic_grad(&tree, &[1.7]), vec![0.0], "{op:?}");
        }
        let not = node(Op::Not, vec![node(Op::Mul, vec![constant(1.3), input(0)])]);
        assert_eq!(analytic_grad(&not, &[1.7]), vec![0.0]);
    }

    #[test]
    fn logical_subgradients_match_the_documented_surrogate() {
        let g = |op: Op, a: f64, b: f64| {
            analytic_grad(&node(op, vec![constant(a), constant(b)]), &[0.0])
        };
        // and: d/da = [b>0], d/db = [a>0].
        assert_eq!(g(Op::And, 1.0, 1.0), vec![1.0, 1.0]);
        assert_eq!(g(Op::And, 1.0, -1.0), vec![0.0, 1.0]);
        assert_eq!(g(Op::And, -1.0, -1.0), vec![0.0, 0.0]);
        // or: d/da = 1-[b>0], d/db = 1-[a>0].
        assert_eq!(g(Op::Or, 1.0, 1.0), vec![0.0, 0.0]);
        assert_eq!(g(Op::Or, 1.0, -1.0), vec![1.0, 0.0]);
        assert_eq!(g(Op::Or, -1.0, -1.0), vec![1.0, 1.0]);
        // xor: d/da = 1-2[b>0], d/db = 1-2[a>0].
        assert_eq!(g(Op::Xor, 1.0, 1.0), vec![-1.0, -1.0]);
        assert_eq!(g(Op::Xor, 1.0, -1.0), vec![1.0, -1.0]);
        assert_eq!(g(Op::Xor, -1.0, -1.0), vec![1.0, 1.0]);
    }

    #[test]
    fn constraint_gradients_pass_through_while_feasible() {
        // require/constrain return their argument, so d/da = 1 while feasible.
        assert_eq!(analytic_grad(&node(Op::Require, vec![constant(2.0)]), &[0.0]), vec![1.0]);
        assert_eq!(analytic_grad(&node(Op::Constrain, vec![constant(2.0)]), &[0.0]), vec![1.0]);
        // Violated: the value is NaN, so no gradient flows.
        assert_eq!(analytic_grad(&node(Op::Require, vec![constant(-1.0)]), &[0.0]), vec![0.0]);
        assert_eq!(analytic_grad(&node(Op::Constrain, vec![constant(0.0)]), &[0.0]), vec![0.0]);
        // between(v, lo, hi) returns v; the bounds get no gradient.
        let feasible =
            node(Op::Between, vec![constant(1.0), constant(0.0), constant(2.0)]);
        assert_eq!(analytic_grad(&feasible, &[0.0]), vec![1.0, 0.0, 0.0]);
        let violated =
            node(Op::Between, vec![constant(3.0), constant(0.0), constant(2.0)]);
        assert_eq!(analytic_grad(&violated, &[0.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn fit_value_of_matches_the_public_evaluator() {
        // The flat evaluator used by the fitter must agree with `expr::eval`
        // on every new primitive.
        let check = |tree: Node, x: &[f64]| {
            let f = flatten(&tree);
            let mut c = vec![0.0; f.nconst];
            for i in 0..f.op.len() {
                if f.cslot[i] >= 0 {
                    c[f.cslot[i] as usize] = f.val[i];
                }
            }
            let mut v = vec![0.0; f.op.len()];
            eval_row(&f, x, &c, &mut v);
            let got = v[f.root];
            let want = expr::eval(&tree, x);
            assert!(
                (got.is_nan() && want.is_nan()) || got == want,
                "flat {got} vs eval {want} for {tree:?}"
            );
        };
        check(node(Op::Equal, vec![constant(2.0), constant(2.0)]), &[0.0]);
        check(node(Op::Less, vec![constant(1.0), constant(2.0)]), &[0.0]);
        check(node(Op::LessOrEqual, vec![constant(2.0), constant(2.0)]), &[0.0]);
        check(node(Op::Greater, vec![constant(3.0), constant(2.0)]), &[0.0]);
        check(node(Op::GreaterOrEqual, vec![constant(2.0), constant(2.0)]), &[0.0]);
        check(node(Op::And, vec![constant(1.0), constant(-1.0)]), &[0.0]);
        check(node(Op::Or, vec![constant(1.0), constant(-1.0)]), &[0.0]);
        check(node(Op::Xor, vec![constant(1.0), constant(-1.0)]), &[0.0]);
        check(node(Op::Not, vec![constant(0.0)]), &[0.0]);
        check(node(Op::Require, vec![input(0)]), &[2.0]);
        check(node(Op::Require, vec![input(0)]), &[-2.0]);
        check(node(Op::Constrain, vec![input(0)]), &[2.0]);
        check(node(Op::Between, vec![input(0), constant(-1.0), constant(1.0)]), &[0.5]);
        check(node(Op::Between, vec![input(0), constant(-1.0), constant(1.0)]), &[2.0]);
    }

    /// Wrap scalars as single-input rows.
    fn rows(xs: &[f64]) -> Vec<Vec<f64>> {
        xs.iter().map(|&v| vec![v]).collect()
    }

    #[test]
    fn clean_law_has_small_relative_errors() {
        // y = a*x + b on well-spread x with a little noise: both constants are
        // pinned down tightly.
        let x = rows(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let noise = [0.01, -0.015, 0.008, -0.012, 0.02, -0.005, 0.006, -0.02, 0.011, -0.003];
        let y: Vec<f64> = x.iter().zip(noise).map(|(r, e)| 2.0 * r[0] + 1.0 + e).collect();

        let mut tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)],
        );
        fit(&mut tree, &x, &y, 25);

        let e = constant_errors(&tree, &x, &y).expect("errors for a definite fit");
        assert!(!e.ill_conditioned, "a linear law is well conditioned");
        assert_eq!(e.std_errors.len(), 2);
        assert_eq!(e.dof, y.len() - 2);
        for r in &e.relative {
            assert!(
                r.is_finite() && *r < 0.05,
                "relative uncertainty should be small: {:?}",
                e.relative
            );
        }
    }

    #[test]
    fn degenerate_constants_are_flagged() {
        // a + b: both gradients are identically 1, so the two constants can
        // trade off and the data cannot tell them apart.
        let x = rows(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let y = vec![3.0, 2.7, 3.2, 2.9, 3.1, 3.05, 2.95, 3.0];
        let tree = node(Op::Add, vec![constant(2.0), constant(1.0)]);

        let e = constant_errors(&tree, &x, &y).expect("still reports the values");
        assert!(e.ill_conditioned, "a + b is unidentified");
        assert!(
            e.relative.iter().all(|r| !r.is_finite() || *r > 1.0),
            "unidentified constants have large/infinite relative uncertainty: {:?}",
            e.relative
        );
    }

    #[test]
    fn nearly_degenerate_model_reports_large_uncertainty() {
        // y = a*x + b with x almost constant: the design is nearly singular, so
        // at least one constant is very poorly determined.
        let x = rows(&[1.0, 1.0 + 1e-6, 1.0 + 2e-6, 1.0 + 3e-6, 1.0 + 4e-6, 1.0 + 5e-6]);
        let y = vec![3.0, 2.9, 3.1, 3.0, 2.95, 3.05];
        let tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(2.0)],
        );

        let e = constant_errors(&tree, &x, &y).expect("reports the values");
        assert!(
            e.ill_conditioned || e.relative.iter().any(|r| *r > 1.0),
            "nearly collinear gradients must be flagged or show a large relative uncertainty: {:?}",
            e.relative
        );
    }

    #[test]
    fn constant_errors_returns_none_without_enough_information() {
        let x = rows(&[0.0, 1.0, 2.0, 3.0]);
        let y = vec![1.0, 2.0, 3.0, 4.0];

        // No constants at all.
        assert!(constant_errors(&input(0), &x, &y).is_none());
        // No data rows.
        let tree = node(Op::Add, vec![constant(1.0), input(0)]);
        assert!(constant_errors(&tree, &[], &[]).is_none());
        // No residual degrees of freedom (n <= k).
        let two = node(Op::Add, vec![constant(1.0), constant(2.0)]);
        assert!(constant_errors(&two, &rows(&[0.0]), &[1.0]).is_none());
    }
}
