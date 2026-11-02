//! #474: the column-tile passes of the exact Levenberg-Marquardt fitter.
//!
//! The row path evaluates one row at a time: a `match` per node per row over
//! `Vec<Vec<f64>>` rows. Here each node is evaluated over a tile of rows, the
//! adjoint sweep runs node by node over the same tile, and the `JᵀJ`/`Jᵀr`
//! sums are then accumulated row by row in the original row order. Every
//! per-row operation is the one the row path performs (`apply_op`,
//! `adj_terms`) and every accumulation keeps its order, so the SSE, `JᵀJ`,
//! `Jᵀr` and therefore every fitted constant are bit-identical.
//!
//! The loops are plain scalar Rust. The AVX entry points only change the code
//! generation of the same source, selected at runtime like `eval_soa`.

use super::{adj_terms, apply_op, Flat, NormalEquations};
use crate::expr::Op;

/// Rows per tile.
const TILE: usize = 256;

const NO_KID: u32 = u32::MAX;

#[derive(Clone, Copy)]
enum Src {
    /// A fitted constant read from `c[slot]`.
    Slot(usize),
    /// A node that carries a fixed value.
    Fill(f64),
    /// An input read from `cols[column]`.
    Column(usize),
    /// An input the rows do not provide: `NaN`.
    Missing,
    /// An operator node.
    Op,
}

struct Plan {
    op: Vec<Op>,
    src: Vec<Src>,
    val: Vec<f64>,
    kid: Vec<[u32; 3]>,
    nk: Vec<u8>,
    root: usize,
}

struct Scratch {
    v: Vec<f64>,
    adj: Vec<f64>,
    g: Vec<f64>,
    terms: Vec<f64>,
    mask: Vec<u8>,
    nan: Vec<f64>,
}

/// The plan, the gathered input columns and the scratch that every pass of a
/// fit reuses. It depends on the program and the rows, not on the target or
/// the weights, so an IRLS call builds it once.
pub(super) struct TileCore {
    plan: Plan,
    cols: Vec<Vec<f64>>,
    rows: usize,
    k: usize,
    avx: bool,
    s: Scratch,
}

/// The row passes of one fit: a core borrowed with one target and one weight
/// vector.
pub(super) struct TileFit<'a> {
    core: &'a mut TileCore,
    y: &'a [f64],
    w: &'a [f64],
    contract_ok: bool,
}

impl TileCore {
    pub(super) fn new(f: &Flat, x: &[Vec<f64>]) -> TileCore {
        let n = f.op.len();
        let mut cols: Vec<Vec<f64>> = Vec::new();
        let mut col_of: Vec<(usize, Option<usize>)> = Vec::new();
        let mut src = Vec::with_capacity(n);
        let mut kid = Vec::with_capacity(n);
        let mut nk = Vec::with_capacity(n);
        for i in 0..n {
            src.push(if f.cslot[i] >= 0 {
                Src::Slot(f.cslot[i] as usize)
            } else if f.op[i] == Op::Input {
                let var = f.val[i] as usize;
                let found = col_of.iter().find(|(v, _)| *v == var).map(|(_, c)| *c);
                let col = match found {
                    Some(c) => c,
                    None => {
                        let c = if x.iter().any(|row| var < row.len()) {
                            cols.push(
                                x.iter()
                                    .map(|row| row.get(var).copied().unwrap_or(f64::NAN))
                                    .collect(),
                            );
                            Some(cols.len() - 1)
                        } else {
                            None
                        };
                        col_of.push((var, c));
                        c
                    }
                };
                match col {
                    Some(c) => Src::Column(c),
                    None => Src::Missing,
                }
            } else if f.op[i] == Op::Const {
                Src::Fill(f.val[i])
            } else {
                Src::Op
            });
            let mut ks = [NO_KID; 3];
            for (j, &k) in f.kids[i].iter().take(3).enumerate() {
                ks[j] = k as u32;
            }
            kid.push(ks);
            nk.push(f.kids[i].len().min(3) as u8);
        }
        let k = f.nconst;
        TileCore {
            plan: Plan {
                op: f.op.clone(),
                src,
                val: f.val.clone(),
                kid,
                nk,
                root: f.root,
            },
            cols,
            rows: x.len(),
            k,
            avx: avx_available(),
            s: Scratch {
                v: vec![0.0; n * TILE],
                adj: vec![0.0; n * TILE],
                g: vec![0.0; k.max(1) * TILE],
                terms: vec![0.0; 3 * TILE],
                mask: vec![0; TILE],
                nan: vec![f64::NAN; TILE],
            },
        }
    }
}

impl<'a> TileFit<'a> {
    pub(super) fn new(core: &'a mut TileCore, y: &'a [f64], w: &'a [f64]) -> TileFit<'a> {
        let contract_ok = core.rows == y.len() && (w.is_empty() || w.len() == core.rows);
        TileFit { core, y, w, contract_ok }
    }

    /// The weighted residual sum of squares at `c`, as `sse` in `fit.rs`.
    pub(super) fn sse(&mut self, c: &[f64]) -> f64 {
        if !self.contract_ok {
            return f64::INFINITY;
        }
        let k = &mut *self.core;
        #[cfg(target_arch = "x86_64")]
        if k.avx {
            // SAFETY: AVX is detected at construction.
            return unsafe { sse_avx(&k.plan, &k.cols, self.y, self.w, k.rows, c, &mut k.s) };
        }
        sse_impl(&k.plan, &k.cols, self.y, self.w, k.rows, c, &mut k.s)
    }

    /// `JᵀWJ` and `JᵀWr` at `c`, as `normal_equations` in `fit.rs`.
    pub(super) fn normal(&mut self, c: &[f64]) -> Option<NormalEquations> {
        if !self.contract_ok {
            return None;
        }
        let k = &mut *self.core;
        #[cfg(target_arch = "x86_64")]
        if k.avx {
            // SAFETY: AVX is detected at construction.
            return unsafe { normal_avx(&k.plan, &k.cols, self.y, self.w, k.rows, k.k, c, &mut k.s) };
        }
        normal_impl(&k.plan, &k.cols, self.y, self.w, k.rows, k.k, c, &mut k.s)
    }

    /// The reweighted L1 assembly at `c`: `JᵀWJ` and `JᵀWr` with the row
    /// weight `base / max(|r|, 1e-6)` taken from the residual at `c`, and the
    /// loss `Σ base·|r|`. `self.w` holds the base weights.
    pub(super) fn normal_l1(&mut self, c: &[f64]) -> Option<(NormalEquations, f64)> {
        if !self.contract_ok {
            return None;
        }
        let k = &mut *self.core;
        #[cfg(target_arch = "x86_64")]
        if k.avx {
            // SAFETY: AVX is detected at construction.
            return unsafe {
                normal_l1_avx(&k.plan, &k.cols, self.y, self.w, k.rows, k.k, c, &mut k.s)
            };
        }
        normal_l1_impl(&k.plan, &k.cols, self.y, self.w, k.rows, k.k, c, &mut k.s)
    }

    /// `Σ base·|r|` at `c`; `+inf` on a non-finite residual.
    pub(super) fn loss_l1(&mut self, c: &[f64]) -> f64 {
        if !self.contract_ok {
            return f64::INFINITY;
        }
        let k = &mut *self.core;
        #[cfg(target_arch = "x86_64")]
        if k.avx {
            // SAFETY: AVX is detected at construction.
            return unsafe { loss_l1_avx(&k.plan, &k.cols, self.y, self.w, k.rows, c, &mut k.s) };
        }
        loss_l1_impl(&k.plan, &k.cols, self.y, self.w, k.rows, c, &mut k.s)
    }
}

fn avx_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sse_avx(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    c: &[f64],
    s: &mut Scratch,
) -> f64 {
    sse_impl(p, cols, y, w, rows, c, s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn normal_avx(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    k: usize,
    c: &[f64],
    s: &mut Scratch,
) -> Option<NormalEquations> {
    normal_impl(p, cols, y, w, rows, k, c, s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn loss_l1_avx(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    c: &[f64],
    s: &mut Scratch,
) -> f64 {
    loss_l1_impl(p, cols, y, w, rows, c, s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments)]
unsafe fn normal_l1_avx(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    k: usize,
    c: &[f64],
    s: &mut Scratch,
) -> Option<(NormalEquations, f64)> {
    normal_l1_impl(p, cols, y, w, rows, k, c, s)
}

#[inline(always)]
fn loss_l1_impl(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    c: &[f64],
    s: &mut Scratch,
) -> f64 {
    let mut sum = 0.0;
    let mut base = 0;
    while base < rows {
        let n = TILE.min(rows - base);
        forward(p, cols, c, base, n, &mut s.v, &s.nan);
        let root = &s.v[p.root * TILE..p.root * TILE + n];
        for r in 0..n {
            let bw = w.get(base + r).copied().unwrap_or(1.0);
            if bw == 0.0 {
                continue;
            }
            let e = root[r] - y[base + r];
            if !e.is_finite() {
                return f64::INFINITY;
            }
            sum += bw * e.abs();
        }
        base += n;
    }
    sum
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn normal_l1_impl(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    k: usize,
    c: &[f64],
    s: &mut Scratch,
) -> Option<(NormalEquations, f64)> {
    let mut jtj = vec![0.0; k * k];
    let mut jtr = vec![0.0; k];
    let mut gi = vec![0.0; k];
    let mut loss = 0.0;
    let mut base = 0;
    while base < rows {
        let n = TILE.min(rows - base);
        forward(p, cols, c, base, n, &mut s.v, &s.nan);
        adjoint(p, n, s);
        let root = &s.v[p.root * TILE..p.root * TILE + n];
        for r in 0..n {
            let bw = w.get(base + r).copied().unwrap_or(1.0);
            if bw == 0.0 {
                continue;
            }
            let e = root[r] - y[base + r];
            if !e.is_finite() {
                return None;
            }
            loss += bw * e.abs();
            let sw = bw / e.abs().max(1e-6);
            let res = sw.sqrt() * e;
            let jw = sw.sqrt();
            for i in 0..k {
                let v = jw * s.g[i * TILE + r];
                if !v.is_finite() {
                    return None;
                }
                gi[i] = v;
            }
            for i in 0..k {
                let gii = gi[i];
                jtr[i] += gii * res;
                let row = &mut jtj[i * k..i * k + i + 1];
                for (a, &gj) in row.iter_mut().zip(&gi[..=i]) {
                    *a += gii * gj;
                }
            }
        }
        base += n;
    }
    let mut full = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            full[i][j] = jtj[i * k + j];
            full[j][i] = jtj[i * k + j];
        }
    }
    Some((NormalEquations { jtj: full, jtr }, loss))
}

#[inline(always)]
fn sse_impl(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    c: &[f64],
    s: &mut Scratch,
) -> f64 {
    let mut sum = 0.0;
    let mut base = 0;
    while base < rows {
        let n = TILE.min(rows - base);
        forward(p, cols, c, base, n, &mut s.v, &s.nan);
        let root = &s.v[p.root * TILE..p.root * TILE + n];
        for r in 0..n {
            let sw = w.get(base + r).copied().unwrap_or(1.0);
            if sw == 0.0 {
                continue; // a zero-weight row never influences the fit
            }
            let e = sw.sqrt() * (root[r] - y[base + r]);
            if !e.is_finite() {
                return f64::INFINITY;
            }
            sum += e * e;
        }
        base += n;
    }
    sum
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn normal_impl(
    p: &Plan,
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rows: usize,
    k: usize,
    c: &[f64],
    s: &mut Scratch,
) -> Option<NormalEquations> {
    let mut jtj = vec![0.0; k * k];
    let mut jtr = vec![0.0; k];
    let mut gi = vec![0.0; k];
    let mut base = 0;
    while base < rows {
        let n = TILE.min(rows - base);
        forward(p, cols, c, base, n, &mut s.v, &s.nan);
        adjoint(p, n, s);
        let root = &s.v[p.root * TILE..p.root * TILE + n];
        for r in 0..n {
            let sw = w.get(base + r).copied().unwrap_or(1.0);
            if sw == 0.0 {
                continue; // a zero-weight row contributes nothing to JᵀJ or Jᵀr
            }
            let res = sw.sqrt() * (root[r] - y[base + r]);
            if !res.is_finite() {
                return None;
            }
            let jw = sw.sqrt();
            for i in 0..k {
                let v = jw * s.g[i * TILE + r];
                if !v.is_finite() {
                    return None;
                }
                gi[i] = v;
            }
            for i in 0..k {
                let gii = gi[i];
                jtr[i] += gii * res;
                let row = &mut jtj[i * k..i * k + i + 1];
                for (a, &gj) in row.iter_mut().zip(&gi[..=i]) {
                    *a += gii * gj;
                }
            }
        }
        base += n;
    }
    let mut full = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            full[i][j] = jtj[i * k + j];
            full[j][i] = jtj[i * k + j];
        }
    }
    Some(NormalEquations { jtj: full, jtr })
}

/// Operators with their own monomorphised loop. Every other operator runs the
/// same code with the operator as a runtime value.
macro_rules! specialise {
    ($op:expr, $call:ident($($arg:expr),*)) => {
        match $op {
            Op::Add => $call(Op::Add, $($arg),*),
            Op::Sub => $call(Op::Sub, $($arg),*),
            Op::Mul => $call(Op::Mul, $($arg),*),
            Op::Div => $call(Op::Div, $($arg),*),
            Op::Neg => $call(Op::Neg, $($arg),*),
            Op::Abs => $call(Op::Abs, $($arg),*),
            Op::Square => $call(Op::Square, $($arg),*),
            Op::Cube => $call(Op::Cube, $($arg),*),
            Op::Sqrt => $call(Op::Sqrt, $($arg),*),
            Op::Exp => $call(Op::Exp, $($arg),*),
            Op::Ln => $call(Op::Ln, $($arg),*),
            Op::Sin => $call(Op::Sin, $($arg),*),
            Op::Cos => $call(Op::Cos, $($arg),*),
            Op::Tanh => $call(Op::Tanh, $($arg),*),
            Op::Logistic => $call(Op::Logistic, $($arg),*),
            Op::Gauss => $call(Op::Gauss, $($arg),*),
            Op::Min => $call(Op::Min, $($arg),*),
            Op::Max => $call(Op::Max, $($arg),*),
            o => $call(o, $($arg),*),
        }
    };
}

#[inline(always)]
fn forward_node(op: Op, nv: f64, out: &mut [f64], a: &[f64], b: &[f64], c: &[f64]) {
    let n = out.len();
    let (a, b, c) = (&a[..n], &b[..n], &c[..n]);
    for r in 0..n {
        out[r] = apply_op(op, a[r], b[r], c[r], nv);
    }
}

/// Forward values of every node for rows `base..base + n`, node-major with a
/// stride of `TILE`.
#[inline(always)]
fn forward(p: &Plan, cols: &[Vec<f64>], c: &[f64], base: usize, n: usize, v: &mut [f64], nan: &[f64]) {
    for i in 0..p.op.len() {
        let (lo, hi) = v.split_at_mut(i * TILE);
        let out = &mut hi[..n];
        match p.src[i] {
            Src::Slot(s) => out.fill(c[s]),
            Src::Fill(x) => out.fill(x),
            Src::Column(col) => out.copy_from_slice(&cols[col][base..base + n]),
            Src::Missing => out.fill(f64::NAN),
            Src::Op => {
                let operand = |j: usize| -> &[f64] {
                    match p.kid[i][j] {
                        NO_KID => &nan[..n],
                        k => &lo[k as usize * TILE..k as usize * TILE + n],
                    }
                };
                let (a, b, d) = (operand(0), operand(1), operand(2));
                let nv = p.val[i];
                specialise!(p.op[i], forward_node(nv, out, a, b, d));
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn terms_node(
    op: Op,
    ai: &[f64],
    x0: &[f64],
    x1: &[f64],
    x2: &[f64],
    vi: &[f64],
    terms: &mut [f64],
    mask: &mut [u8],
) {
    let n = ai.len();
    let (x0, x1, x2, vi) = (&x0[..n], &x1[..n], &x2[..n], &vi[..n]);
    let (t0, rest) = terms.split_at_mut(TILE);
    let (t1, t2) = rest.split_at_mut(TILE);
    let (t0, t1, t2, mask) = (&mut t0[..n], &mut t1[..n], &mut t2[..n], &mut mask[..n]);
    for r in 0..n {
        let (t, m) = adj_terms(op, ai[r], x0[r], x1[r], x2[r], vi[r]);
        t0[r] = t[0];
        t1[r] = t[1];
        t2[r] = t[2];
        mask[r] = m;
    }
}

/// The adjoint sweep over one tile, descending node order, leaving the
/// constant gradients in `s.g` (stride `TILE`). A row whose node adjoint is
/// exactly zero is skipped, as in the row path.
#[inline(always)]
fn adjoint(p: &Plan, n: usize, s: &mut Scratch) {
    let nn = p.op.len();
    for i in 0..nn {
        s.adj[i * TILE..i * TILE + n].fill(0.0);
    }
    for gs in s.g.chunks_mut(TILE) {
        gs[..n].fill(0.0);
    }
    s.adj[p.root * TILE..p.root * TILE + n].fill(1.0);
    for i in (0..nn).rev() {
        let (lo, hi) = s.adj.split_at_mut(i * TILE);
        let ai = &hi[..n];
        match p.src[i] {
            Src::Slot(slot) => {
                let gs = &mut s.g[slot * TILE..slot * TILE + n];
                for r in 0..n {
                    gs[r] = if ai[r] == 0.0 { gs[r] } else { gs[r] + ai[r] };
                }
                continue;
            }
            Src::Fill(_) | Src::Column(_) | Src::Missing => continue,
            Src::Op => {}
        }
        let nk = p.nk[i] as usize;
        if nk == 0 {
            continue;
        }
        let operand = |j: usize| -> &[f64] {
            match p.kid[i][j] {
                NO_KID => &s.nan[..n],
                k => &s.v[k as usize * TILE..k as usize * TILE + n],
            }
        };
        let (x0, x1, x2) = (operand(0), operand(1), operand(2));
        let vi = &s.v[i * TILE..i * TILE + n];
        specialise!(p.op[i], terms_node(ai, x0, x1, x2, vi, &mut s.terms, &mut s.mask));
        for j in 0..nk {
            let k = p.kid[i][j] as usize;
            let dst = &mut lo[k * TILE..k * TILE + n];
            let t = &s.terms[j * TILE..j * TILE + n];
            let m = &s.mask[..n];
            for r in 0..n {
                let add = ai[r] != 0.0 && (m[r] >> j) & 1 == 1;
                dst[r] = if add { dst[r] + t[r] } else { dst[r] };
            }
        }
    }
}
