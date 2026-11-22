//! #895: an exact f64 interval bound on a tree over the training columns.
//!
//! The device `+inf` check settles a slot without a CPU pass when no row can
//! have a finite prediction above the book threshold (`--min-trades >= 1`: the
//! book then counts no flip, or a non-finite prediction scores `+inf` anyway).
//!
//! Soundness rests on the CPU evaluator ([`crate::expr::eval`] and the SoA
//! kernel of [`crate::eval_soa`]):
//!
//! * `+ - * /` are single correctly rounded IEEE f64 operations, never fused
//!   and never reordered, so round-to-nearest corner arithmetic bounds the
//!   evaluator's value without widening.
//! * `exp` and `tanh` are libm calls and are not correctly rounded. Each
//!   endpoint is widened by [`ULPS`] ulps, then clamped to the codomain of the
//!   true function. With a per-point error of at most `E` ulps, `libm(x)` can
//!   exceed `libm(y)` for `x < y` by `2E` ulps, and twice that across a binade
//!   boundary (an ulp of the upper binade is two ulps of the lower one). glibc
//!   documents `E = 1` for `exp` and `E = 2` for `tanh`, so `4E = 8` ulps cover
//!   the documented bounds; the condition is a libm error of at most `ULPS / 4`
//!   ulps per point, or a monotone libm.
//! * Every supported operator maps a NaN input to NaN, so a NaN row never
//!   becomes a number downstream. The bound covers the non-NaN values only.
//! * Any other operator gives no bound.

use crate::expr::{Node, Op};
use std::hint::black_box;

/// Pieces kept per node and per column.
pub const MAX_PIECES: usize = 6;
/// libm widening in ulps on each side of an `exp` or `tanh` endpoint: `2E`
/// for a monotonicity break of two points with error `E`, doubled across a
/// binade boundary, with the documented glibc maximum `E = 2` (`tanh`).
const ULPS: u32 = 8;

/// The documented glibc error bound the widening must cover, in ulps.
#[cfg(test)]
const DOCUMENTED_LIBM_ULPS: u32 = 2;
/// The protected division cuts the divisor at this magnitude.
const GUARD: f64 = 1e-12;
/// `exp` returns `+inf` above this argument.
const EXP_LIMIT: f64 = 700.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Iv {
    lo: f64,
    hi: f64,
}

const FULL: Iv = Iv { lo: f64::NEG_INFINITY, hi: f64::INFINITY };

/// Non-NaN values of a node lie in the union of the pieces (closed, extended
/// reals). Sorted and disjoint, at most [`MAX_PIECES`]. Empty: every row is NaN.
#[derive(Clone, Debug, PartialEq)]
pub struct Pieces(Vec<(f64, f64)>);

impl Pieces {
    pub fn ranges(&self) -> &[(f64, f64)] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Largest value any piece reaches; `-inf` when the set is empty.
    pub fn upper(&self) -> f64 {
        self.0.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max)
    }

    fn from_ivs(v: Vec<Iv>) -> Self {
        Pieces(merge(v).into_iter().map(|p| (p.lo, p.hi)).collect())
    }

    fn ivs(&self) -> Vec<Iv> {
        self.0.iter().map(|&(lo, hi)| Iv { lo, hi }).collect()
    }
}

/// Per column: `[min, largest negative]`, `[0, 0]` when a zero occurs and
/// `[smallest positive, max]`. NaN rows are left out.
#[derive(Clone, Debug)]
pub struct ColumnPieces {
    cols: Vec<Pieces>,
}

#[derive(Clone, Copy)]
struct ColumnScan {
    neg: Option<(f64, f64)>,
    zero: bool,
    pos: Option<(f64, f64)>,
}

impl ColumnScan {
    fn new() -> Self {
        ColumnScan { neg: None, zero: false, pos: None }
    }

    fn push(&mut self, v: f64) {
        if v.is_nan() {
        } else if v < 0.0 {
            self.neg = Some(match self.neg {
                None => (v, v),
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
            });
        } else if v > 0.0 {
            self.pos = Some(match self.pos {
                None => (v, v),
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
            });
        } else {
            self.zero = true;
        }
    }

    fn finish(self) -> Pieces {
        let mut out = Vec::new();
        if let Some((lo, hi)) = self.neg {
            out.push((lo, hi));
        }
        if self.zero {
            out.push((0.0, 0.0));
        }
        if let Some((lo, hi)) = self.pos {
            out.push((lo, hi));
        }
        Pieces(out)
    }
}

impl ColumnPieces {
    /// One pass over the SoA columns (`cols[var][row]`).
    pub fn from_columns(cols: &[Vec<f64>]) -> Self {
        ColumnPieces {
            cols: cols
                .iter()
                .map(|c| {
                    let mut s = ColumnScan::new();
                    c.iter().for_each(|&v| s.push(v));
                    s.finish()
                })
                .collect(),
        }
    }

    /// One pass over row-major rows (`x[row][var]`). A row shorter than the
    /// widest one reads as NaN in the missing columns, as the evaluator does.
    pub fn from_rows(x: &[Vec<f64>]) -> Self {
        let width = x.iter().map(Vec::len).max().unwrap_or(0);
        let mut scans = vec![ColumnScan::new(); width];
        for row in x {
            for (s, &v) in scans.iter_mut().zip(row) {
                s.push(v);
            }
        }
        ColumnPieces { cols: scans.into_iter().map(ColumnScan::finish).collect() }
    }

    pub fn len(&self) -> usize {
        self.cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }
}

/// Whether this thread rounds subnormals as IEEE does. A flush-to-zero or
/// denormals-are-zero mode could shrink an upper bound, so the rule is off then.
pub fn gradual_underflow() -> bool {
    let tiny = black_box(f64::MIN_POSITIVE) / black_box(2.0);
    tiny != 0.0 && tiny * black_box(1.0) != 0.0
}

fn empty(v: &[Iv]) -> bool {
    v.is_empty()
}

fn dn(mut v: f64) -> f64 {
    for _ in 0..ULPS {
        v = v.next_down();
    }
    v
}

fn up(mut v: f64) -> f64 {
    for _ in 0..ULPS {
        v = v.next_up();
    }
    v
}

fn neg(a: Iv) -> Iv {
    Iv { lo: -a.hi, hi: -a.lo }
}

fn corners(a: Iv, b: Iv, f: impl Fn(f64, f64) -> f64) -> Iv {
    let c = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
    if c.iter().any(|v| v.is_nan()) {
        return FULL;
    }
    Iv {
        lo: c.iter().copied().fold(f64::INFINITY, f64::min),
        hi: c.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn add(a: Iv, b: Iv) -> Iv {
    let (lo, hi) = (a.lo + b.lo, a.hi + b.hi);
    if lo.is_nan() || hi.is_nan() {
        FULL
    } else {
        Iv { lo, hi }
    }
}

fn sub(a: Iv, b: Iv) -> Iv {
    let (lo, hi) = (a.lo - b.hi, a.hi - b.lo);
    if lo.is_nan() || hi.is_nan() {
        FULL
    } else {
        Iv { lo, hi }
    }
}

fn mul(a: Iv, b: Iv) -> Iv {
    corners(a, b, |x, y| x * y)
}

/// The sign-definite parts of the divisor that the guard lets through.
fn div_parts(a: Iv, b: Iv) -> Vec<Iv> {
    let mut v = Vec::new();
    if b.lo <= -GUARD {
        v.push(corners(a, Iv { lo: b.lo, hi: b.hi.min(-GUARD) }, |x, y| x / y));
    }
    if b.hi >= GUARD {
        v.push(corners(a, Iv { lo: b.lo.max(GUARD), hi: b.hi }, |x, y| x / y));
    }
    v
}

fn square(a: Iv) -> Iv {
    if a.lo >= 0.0 {
        Iv { lo: a.lo * a.lo, hi: a.hi * a.hi }
    } else if a.hi <= 0.0 {
        Iv { lo: a.hi * a.hi, hi: a.lo * a.lo }
    } else {
        Iv { lo: 0.0, hi: (a.lo * a.lo).max(a.hi * a.hi) }
    }
}

fn abs(a: Iv) -> Iv {
    if a.lo >= 0.0 {
        a
    } else if a.hi <= 0.0 {
        neg(a)
    } else {
        Iv { lo: 0.0, hi: (-a.lo).max(a.hi) }
    }
}

/// `exp` without the evaluator guard (logistic and gauss call it unguarded).
fn exp_raw(a: Iv) -> Iv {
    Iv { lo: dn(a.lo.exp()).max(0.0), hi: up(a.hi.exp()) }
}

/// `exp` with the evaluator guard: an argument above 700 gives `+inf`.
fn exp_guarded(a: Iv) -> Iv {
    Iv {
        lo: if a.lo > EXP_LIMIT { f64::INFINITY } else { dn(a.lo.exp()).max(0.0) },
        hi: if a.hi > EXP_LIMIT { f64::INFINITY } else { up(a.hi.exp()) },
    }
}

fn tanh(a: Iv) -> Iv {
    Iv { lo: dn(a.lo.tanh()).max(-1.0), hi: up(a.hi.tanh()).min(1.0) }
}

fn logistic(a: Iv) -> Iv {
    let e = exp_raw(neg(a));
    Iv { lo: 1.0 / (1.0 + e.hi), hi: 1.0 / (1.0 + e.lo) }
}

fn gauss(a: Iv) -> Iv {
    let s = square(a);
    let e = exp_raw(Iv { lo: -s.hi, hi: -s.lo });
    Iv { lo: e.lo.max(0.0), hi: e.hi.min(1.0) }
}

/// Sort, join overlaps and, above [`MAX_PIECES`], join the closest pair. A NaN
/// endpoint gives the full line. Joining only widens a set.
fn merge(mut v: Vec<Iv>) -> Vec<Iv> {
    if v.iter().any(|p| p.lo.is_nan() || p.hi.is_nan()) {
        return vec![FULL];
    }
    v.sort_by(|a, b| a.lo.total_cmp(&b.lo));
    let mut out: Vec<Iv> = Vec::with_capacity(v.len());
    for p in v {
        match out.last_mut() {
            Some(l) if p.lo <= l.hi => l.hi = l.hi.max(p.hi),
            _ => out.push(p),
        }
    }
    while out.len() > MAX_PIECES {
        let mut best = 0;
        let mut gap = f64::INFINITY;
        for i in 0..out.len() - 1 {
            let g = out[i + 1].lo - out[i].hi;
            if g < gap {
                gap = g;
                best = i;
            }
        }
        let hi = out[best + 1].hi;
        out[best].hi = out[best].hi.max(hi);
        out.remove(best + 1);
    }
    out
}

fn map1(a: &[Iv], f: impl Fn(Iv) -> Iv) -> Vec<Iv> {
    merge(a.iter().map(|&p| f(p)).collect())
}

fn map2(a: &[Iv], b: &[Iv], f: impl Fn(Iv, Iv) -> Vec<Iv>) -> Vec<Iv> {
    let mut v = Vec::new();
    for &x in a {
        for &y in b {
            v.extend(f(x, y));
        }
    }
    merge(v)
}

/// Bitwise equal subtrees without custom blocks: they evaluate to the same
/// f64 on every row.
fn same(a: &Node, b: &Node) -> bool {
    a.op == b.op
        && a.val.to_bits() == b.val.to_bits()
        && a.block.is_none()
        && b.block.is_none()
        && a.kids.len() == b.kids.len()
        && a.kids.iter().zip(&b.kids).all(|(x, y)| same(x, y))
}

fn walk(n: &Node, cols: &ColumnPieces) -> Option<Vec<Iv>> {
    let kid = |k: usize| n.kids.get(k).and_then(|c| walk(c, cols));
    let pair = |f: fn(Iv, Iv) -> Iv| -> Option<Vec<Iv>> {
        let (a, b) = (kid(0)?, kid(1)?);
        Some(map2(&a, &b, |x, y| vec![f(x, y)]))
    };
    Some(match n.op {
        Op::Const => {
            if n.val.is_nan() {
                Vec::new()
            } else {
                vec![Iv { lo: n.val, hi: n.val }]
            }
        }
        Op::Input => cols.cols.get(n.val as usize)?.ivs(),
        Op::Neg => map1(&kid(0)?, neg),
        Op::Abs => map1(&kid(0)?, abs),
        Op::Square => map1(&kid(0)?, square),
        Op::Exp => map1(&kid(0)?, exp_guarded),
        Op::Tanh => map1(&kid(0)?, tanh),
        Op::Logistic => map1(&kid(0)?, logistic),
        Op::Gauss => map1(&kid(0)?, gauss),
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            if n.kids.len() != 2 {
                return None;
            }
            let twin = same(&n.kids[0], &n.kids[1]);
            match n.op {
                Op::Sub if twin => {
                    let a = kid(0)?;
                    if empty(&a) { a } else { vec![Iv { lo: 0.0, hi: 0.0 }] }
                }
                Op::Mul if twin => map1(&kid(0)?, square),
                Op::Div if twin => {
                    let a = kid(0)?;
                    if empty(&a) { a } else { vec![Iv { lo: 1.0, hi: 1.0 }] }
                }
                Op::Add => pair(add)?,
                Op::Sub => pair(sub)?,
                Op::Mul => pair(mul)?,
                _ => {
                    let (a, b) = (kid(0)?, kid(1)?);
                    map2(&a, &b, div_parts)
                }
            }
        }
        _ => return None,
    })
}

/// The pieces that hold every non-NaN value of `tree` over rows drawn from
/// `cols`, or `None` when the tree has an operator outside the supported set
/// or reads a column the pieces do not have.
pub fn bound(tree: &Node, cols: &ColumnPieces) -> Option<Pieces> {
    walk(tree, cols).map(Pieces::from_ivs)
}

/// Whether the bound may run: the calling thread and every pool worker round
/// subnormals as IEEE does. A flush-to-zero or denormals-are-zero mode on a
/// worker could make a tree that is flat in IEEE long there.
pub fn gate(calling_thread_gradual: bool, worker_gradual: &[bool]) -> bool {
    calling_thread_gradual && worker_gradual.iter().all(|&g| g)
}

/// True when no row can give a finite f64 prediction above `threshold`.
pub fn flat(tree: &Node, cols: &ColumnPieces, threshold: f64) -> bool {
    !threshold.is_nan()
        && bound(tree, cols).is_some_and(|p| p.upper() <= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    const THR: f64 = 1e-30;

    fn cols(columns: Vec<Vec<f64>>) -> ColumnPieces {
        ColumnPieces::from_columns(&columns)
    }

    fn un(op: Op, a: Node) -> Node {
        node(op, vec![a])
    }

    fn bin(op: Op, a: Node, b: Node) -> Node {
        node(op, vec![a, b])
    }

    fn ranges(tree: &Node, c: &ColumnPieces) -> Vec<(f64, f64)> {
        bound(tree, c).expect("supported").ranges().to_vec()
    }

    #[test]
    fn div_cuts_the_divisor_at_the_guard() {
        let tree = bin(Op::Div, constant(1.0), input(0));
        for edge in [1e-12, -1e-12] {
            let r = ranges(&tree, &cols(vec![vec![edge]]));
            assert_eq!(r, vec![(1.0 / edge, 1.0 / edge)], "|b| = 1e-12 divides");
        }
        for below in [9.9e-13, -9.9e-13, 0.0, -0.0] {
            let p = bound(&tree, &cols(vec![vec![below]])).unwrap();
            assert!(p.is_empty(), "{below:e} is under the guard and gives no piece");
        }
        let mixed = ranges(&tree, &cols(vec![vec![-2.0, 1e-13, 4.0]]));
        assert_eq!(mixed, vec![(-0.5, -0.5), (0.25, 1e12)]);
    }

    #[test]
    fn exp_guard_at_700() {
        let tree = un(Op::Exp, input(0));
        let at = ranges(&tree, &cols(vec![vec![700.0]]));
        assert_eq!(at.len(), 1);
        assert!(at[0].1.is_finite() && at[0].0 <= 700f64.exp() && 700f64.exp() <= at[0].1);
        let above = ranges(&tree, &cols(vec![vec![700.0000001]]));
        assert_eq!(above, vec![(f64::INFINITY, f64::INFINITY)]);
        let both = ranges(&tree, &cols(vec![vec![-1.0, 701.0]]));
        assert_eq!(both.last().unwrap().1, f64::INFINITY);
        assert!(both[0].0 >= 0.0);
        let low = ranges(&tree, &cols(vec![vec![-1e6]]));
        assert_eq!(low.len(), 1);
        assert!(low[0].0 == 0.0 && low[0].1 < 1e-300);
        assert!(flat(&un(Op::Exp, bin(Op::Sub, input(0), constant(1e6))), &cols(vec![vec![-5.0, 5.0]]), THR));
    }

    #[test]
    fn logistic_and_gauss_stay_in_unit_interval() {
        let c = cols(vec![vec![-1e300, -3.0, 0.0, 2.0, 1e300]]);
        for op in [Op::Logistic, Op::Gauss] {
            for p in bound(&un(op, input(0)), &c).unwrap().ranges() {
                assert!(p.0 >= 0.0 && p.1 <= 1.0, "{op:?} {p:?}");
            }
        }
        let flat_logistic = un(Op::Logistic, bin(Op::Sub, input(0), constant(3423.03)));
        assert!(flat(&flat_logistic, &cols(vec![vec![-10.0, 3000.0]]), THR));
        assert!(!flat(&flat_logistic, &cols(vec![vec![-10.0, 4000.0]]), THR));
    }

    #[test]
    fn identical_kids_rules() {
        let c = cols(vec![vec![-5.0, 0.5, 7.0], vec![f64::INFINITY, 1.0]]);
        let x = input(0);
        assert_eq!(ranges(&bin(Op::Sub, x.clone(), x.clone()), &c), vec![(0.0, 0.0)]);
        assert_eq!(ranges(&bin(Op::Div, x.clone(), x.clone()), &c), vec![(1.0, 1.0)]);
        assert_eq!(ranges(&bin(Op::Mul, x.clone(), x.clone()), &c), vec![(0.25, 49.0)]);
        let big = un(Op::Exp, x.clone());
        assert!(flat(&un(Op::Tanh, bin(Op::Sub, big.clone(), big.clone())), &c, THR));
        let tanh_of_zero = bound(&un(Op::Tanh, bin(Op::Sub, big.clone(), big)), &c).unwrap();
        assert!(tanh_of_zero.upper() < 1e-300);
        // x - y with equal values but different nodes is not an identical pair
        let diff = ranges(&bin(Op::Sub, x.clone(), constant(0.0)), &c);
        assert_eq!(diff, vec![(-5.0, -5.0), (0.5, 7.0)]);
        // a different constant bit pattern is not identical
        let signed = bin(Op::Sub, constant(0.0), constant(-0.0));
        assert_eq!(ranges(&signed, &c), vec![(0.0, 0.0)]);
        // an infinite row leaves NaN in x - x, and the rule keeps the zero
        let inf = bin(Op::Sub, input(1), input(1));
        assert_eq!(ranges(&inf, &c), vec![(0.0, 0.0)]);
    }

    #[test]
    fn unsupported_operator_gives_no_bound() {
        let c = cols(vec![vec![1.0, 2.0]]);
        for op in [Op::Ln, Op::Sqrt, Op::Step, Op::Sign, Op::Cube, Op::Floor, Op::Sin] {
            let t = un(op, input(0));
            assert!(bound(&t, &c).is_none(), "{op:?}");
            assert!(!flat(&un(Op::Neg, t), &c, f64::INFINITY));
        }
        for op in [Op::Min, Op::Max, Op::Pow] {
            assert!(bound(&bin(op, input(0), constant(1.0)), &c).is_none(), "{op:?}");
        }
        let cond = node(Op::If, vec![input(0), constant(1.0), constant(2.0)]);
        assert!(bound(&cond, &c).is_none());
        assert!(bound(&input(3), &c).is_none(), "a column the pieces lack");
        assert!(bound(&node(Op::Add, vec![input(0)]), &c).is_none(), "a missing child");
    }

    #[test]
    fn nan_column_is_empty() {
        let c = cols(vec![vec![f64::NAN, f64::NAN], vec![1.0, 2.0]]);
        assert!(bound(&input(0), &c).unwrap().is_empty());
        let t = bin(Op::Add, input(0), input(1));
        assert!(bound(&t, &c).unwrap().is_empty());
        assert!(flat(&t, &c, 0.0));
        assert!(bound(&constant(f64::NAN), &c).unwrap().is_empty());
        let rows = ColumnPieces::from_rows(&[vec![f64::NAN, 1.0], vec![f64::NAN]]);
        assert!(bound(&input(0), &rows).unwrap().is_empty());
        assert_eq!(rows.len(), 2);
        assert_eq!(ranges(&input(1), &rows), vec![(1.0, 1.0)], "a short row reads as NaN");
        assert!(!flat(&constant(-1.0), &c, f64::NAN), "a NaN threshold gives no bound");
    }

    #[test]
    fn zero_split_proves_gauss_of_scaled_input() {
        let tree = un(Op::Gauss, bin(Op::Mul, constant(1e6), input(0)));
        let away: Vec<f64> = vec![-3.0, -1.5, -1e-3, 1e-3, 2.0, 3.0];
        assert!(flat(&tree, &cols(vec![away.clone()]), THR));
        let mut with_zero = away;
        with_zero.push(0.0);
        assert!(!flat(&tree, &cols(vec![with_zero]), THR));
        let mut with_neg_zero = vec![-3.0, 3.0];
        with_neg_zero.push(-0.0);
        assert!(!flat(&tree, &cols(vec![with_neg_zero]), THR));
        let box_only = ColumnPieces { cols: vec![Pieces(vec![(-3.0, 3.0)])] };
        assert!(!flat(&tree, &box_only, THR), "one interval over the column cannot prove it");
    }

    #[test]
    fn column_pieces_split_at_zero() {
        let c = cols(vec![vec![5.0, -1.0, 0.0, -0.0, 2.0, -7.0, f64::NAN, 9.0]]);
        assert_eq!(ranges(&input(0), &c), vec![(-7.0, -1.0), (0.0, 0.0), (2.0, 9.0)]);
        let rows = ColumnPieces::from_rows(&[vec![1.0], vec![-1.0], vec![0.0]]);
        assert_eq!(ranges(&input(0), &rows), vec![(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn gradual_underflow_holds_by_default() {
        assert!(gradual_underflow());
    }

    #[test]
    fn libm_widening_covers_the_documented_bound() {
        assert!(ULPS >= 4 * DOCUMENTED_LIBM_ULPS);
        let down = |v: f64| (0..4 * DOCUMENTED_LIBM_ULPS).fold(v, |v, _| v.next_down());
        let up = |v: f64| (0..4 * DOCUMENTED_LIBM_ULPS).fold(v, |v, _| v.next_up());
        for x in [-30.0f64, -1.0, 0.3, 1.0, 17.0, 300.0] {
            let a = Iv { lo: x, hi: x };
            let e = exp_guarded(a);
            assert!(e.lo <= down(x.exp()) && up(x.exp()) <= e.hi, "exp {x}");
            let t = tanh(a);
            let (lo, hi) = (down(x.tanh()).max(-1.0), up(x.tanh()).min(1.0));
            assert!(t.lo <= lo && hi <= t.hi, "tanh {x}");
        }
    }

    #[test]
    fn gate_needs_every_thread() {
        assert!(gate(true, &[true, true]));
        assert!(gate(true, &[]));
        assert!(!gate(false, &[true, true]));
        assert!(!gate(true, &[true, false, true]));
    }

    #[test]
    fn libm_stays_inside_the_clamps() {
        for k in 0..200_000u32 {
            let x = -(k as f64) * 1e-3 - (k as f64).sqrt() * 1e-9;
            assert!(x.exp() <= 1.0, "exp({x}) = {}", x.exp());
            assert!(x.tanh() >= -1.0 && x.tanh() <= 1.0);
            assert!((-x).tanh() <= 1.0);
        }
        let tiny = -1e-300f64;
        assert!(tiny.exp() <= 1.0 && (-tiny).tanh() <= 1.0);
    }

    struct Rng(u64);

    impl Rng {
        fn u(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f64 {
            (self.u() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.u() % n as u64) as usize
        }
        fn special(&mut self) -> f64 {
            let s = if self.u() & 1 == 0 { 1.0 } else { -1.0 };
            match self.pick(10) {
                0 => 0.0,
                1 => -0.0,
                2 => s * 1e-12,
                3 => s * 1e6,
                4 => s * 700.0,
                5 => s * 709.78,
                6 => s * 10f64.powf(self.f() * 40.0 - 30.0),
                7 => s * 1e-30,
                8 => s * self.f() * 20.0,
                _ => s * 10f64.powf(self.f() * 6.0 - 3.0),
            }
        }
    }

    fn random_tree(r: &mut Rng, depth: u32, vars: usize, pool: &mut Vec<Node>) -> Node {
        if depth == 0 || r.pick(4) == 0 {
            return if r.pick(2) == 0 { constant(r.special()) } else { input(r.pick(vars)) };
        }
        if !pool.is_empty() && r.pick(6) == 0 {
            return pool[r.pick(pool.len())].clone();
        }
        let kid = |r: &mut Rng, pool: &mut Vec<Node>| random_tree(r, depth - 1, vars, pool);
        let t = match r.pick(13) {
            0 => un(Op::Neg, kid(r, pool)),
            1 => un(Op::Abs, kid(r, pool)),
            2 => un(Op::Square, kid(r, pool)),
            3 => un(Op::Exp, kid(r, pool)),
            4 => un(Op::Tanh, kid(r, pool)),
            5 => un(Op::Logistic, kid(r, pool)),
            6 => un(Op::Gauss, kid(r, pool)),
            7 => bin(Op::Add, kid(r, pool), kid(r, pool)),
            8 => bin(Op::Sub, kid(r, pool), kid(r, pool)),
            9 => bin(Op::Mul, kid(r, pool), kid(r, pool)),
            10 => bin(Op::Div, kid(r, pool), kid(r, pool)),
            11 => {
                let k = kid(r, pool);
                bin(Op::Sub, k.clone(), k)
            }
            _ => {
                let k = kid(r, pool);
                bin(if r.pick(2) == 0 { Op::Mul } else { Op::Div }, k.clone(), k)
            }
        };
        pool.push(t.clone());
        t
    }

    #[test]
    fn bound_holds_on_sampled_rows() {
        let (trees, rows, vars) = (20_000usize, 200usize, 3usize);
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut violations, mut non_nan, mut claims) = (0u64, 0u64, 0u64);
        for t in 0..trees {
            let mut pool = Vec::new();
            let tree = random_tree(&mut r, 4, vars, &mut pool);
            // each column: a handful of special values, so the pieces carry endpoints
            let columns: Vec<Vec<f64>> = (0..vars)
                .map(|_| {
                    let n = 1 + r.pick(6);
                    (0..n).map(|_| r.special()).collect()
                })
                .collect();
            let pieces = ColumnPieces::from_columns(&columns);
            let got = bound(&tree, &pieces).expect("only supported operators");
            let is_flat = flat(&tree, &pieces, THR);
            claims += is_flat as u64;
            for k in 0..rows {
                // endpoints and signed zeros come from the column values themselves
                let row: Vec<f64> = columns
                    .iter()
                    .map(|c| match (k + t) % 4 {
                        0 => c[k % c.len()],
                        _ => c[r.pick(c.len())],
                    })
                    .collect();
                let v = crate::expr::eval(&tree, &row);
                if v.is_nan() {
                    continue;
                }
                non_nan += 1;
                if !got.ranges().iter().any(|&(lo, hi)| lo <= v && v <= hi) {
                    violations += 1;
                    if violations <= 5 {
                        eprintln!("violation: {tree:?} row {row:?} value {v:e} bound {got:?}");
                    }
                }
                if is_flat {
                    assert!(!(v.is_finite() && v > THR), "flat claim met {v:e}: {tree:?} {row:?}");
                }
            }
        }
        assert_eq!(violations, 0, "{non_nan} non-NaN values, {claims} flat claims");
        assert!(non_nan > 1_000_000 && claims > 100, "{non_nan} {claims}");
    }
}
