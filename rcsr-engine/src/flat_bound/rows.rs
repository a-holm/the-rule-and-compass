//! #1051: the exact row decisions of the device book check, on the host.
//!
//! A per-row f64 interval with a NaN state bounds the value the CPU evaluator
//! gives for one row. A row whose interval lies on one side of the book
//! thresholds, with a finite value, has a decided position; a row that cannot
//! be finite is decided "non-finite"; every other row is undecided and the
//! host resolves it with [`crate::expr::eval`], the reference itself. The
//! bit planes then equal the CPU planes, so a book fed from them gives the
//! CPU totals bit for bit.
//!
//! This module is the host mirror of the device kernel: the same postfix
//! words ([`compile_check_program`]), the same interval rules and the same
//! integer-step widening ([`ulps_step`]). The CPU evaluator's `+ - * /` are
//! single IEEE operations, never fused, so round-to-nearest corner arithmetic
//! is exact on points; only `exp`, `tanh`, `logistic` and `gauss` widen.

use crate::expr::{Node, Op};

/// Widening of a libm endpoint in ulps (device and CPU libm differ).
pub const CHECK_ULPS: i64 = 64;
/// Operand stack of one check program.
pub const CHECK_STACK: usize = 16;
/// Undecided rows one slot may list before the host takes the full CPU pass.
pub const SLOT_UNDECIDED_CAP: usize = 4096;
/// Check slots one device group holds.
pub const CHECK_SLOT_CAP: usize = 64;

/// Opcodes of the check program. The values equal `OP_*` of `eval.cu`.
pub const CK_CONST: u32 = 0;
pub const CK_VAR: u32 = 1;
pub const CK_ADD: u32 = 2;
pub const CK_SUB: u32 = 3;
pub const CK_MUL: u32 = 4;
pub const CK_DIV: u32 = 5;
pub const CK_EXP: u32 = 9;
pub const CK_ABS: u32 = 12;
pub const CK_TANH: u32 = 13;
pub const CK_LOGISTIC: u32 = 14;
pub const CK_SQUARE: u32 = 15;
pub const CK_NEG: u32 = 17;
pub const CK_GAUSS: u32 = 18;
pub const CK_MIN: u32 = 30;
pub const CK_MAX: u32 = 31;
pub const CK_IF: u32 = 32;
pub const CK_GREATER: u32 = 37;
pub const CK_HALT: u32 = 255;

const GUARD: f64 = 1e-12;
const EXP_LIMIT: f64 = 700.0;

/// NaN state of a stack level.
pub const NAN_NO: u8 = 0;
pub const NAN_MAYBE: u8 = 1;
pub const NAN_CERTAIN: u8 = 2;

/// `[lo, hi]` bounds the CPU value when it is not NaN; `nan` says whether it
/// may be NaN. Endpoints are never NaN. A level with `nan == NAN_CERTAIN`
/// carries no value bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowIv {
    pub lo: f64,
    pub hi: f64,
    pub nan: u8,
}

const FULL: f64 = f64::INFINITY;

const NAN_ROW: RowIv = RowIv { lo: -FULL, hi: FULL, nan: NAN_CERTAIN };

fn point(v: f64) -> RowIv {
    if v.is_nan() {
        NAN_ROW
    } else {
        RowIv { lo: v, hi: v, nan: NAN_NO }
    }
}

fn is_point(a: RowIv) -> bool {
    a.lo == a.hi
}

/// `v` moved `k` ulps along the ordered bit pattern; infinities unchanged and
/// the result saturates at the infinities. The device uses the same integer
/// arithmetic.
pub fn ulps_step(v: f64, k: i64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    let mut i = v.to_bits() as i64;
    if i < 0 {
        i = i64::MIN.wrapping_sub(i);
    }
    i = i.saturating_add(k);
    const TOP: i64 = 0x7ff0_0000_0000_0000;
    i = i.clamp(-TOP, TOP);
    if i < 0 {
        i = i64::MIN.wrapping_sub(i);
    }
    f64::from_bits(i as u64)
}

fn dn(v: f64) -> f64 {
    ulps_step(v, -CHECK_ULPS)
}

fn up(v: f64) -> f64 {
    ulps_step(v, CHECK_ULPS)
}

fn join_nan(a: u8, b: u8) -> u8 {
    if a == NAN_CERTAIN || b == NAN_CERTAIN {
        NAN_CERTAIN
    } else {
        a.max(b)
    }
}

fn has_inf(a: RowIv) -> bool {
    a.lo == -FULL || a.hi == FULL
}

fn contains_zero(a: RowIv) -> bool {
    a.lo <= 0.0 && a.hi >= 0.0
}

fn clean(lo: f64, hi: f64) -> (f64, f64) {
    (if lo.is_nan() { -FULL } else { lo }, if hi.is_nan() { FULL } else { hi })
}

fn finish_nan(out: RowIv, a: RowIv, b: RowIv, possible: bool) -> RowIv {
    let inherited = join_nan(a.nan, b.nan);
    if inherited == NAN_CERTAIN {
        return NAN_ROW;
    }
    if possible {
        if is_point(a) && is_point(b) {
            return NAN_ROW;
        }
        return RowIv { lo: -FULL, hi: FULL, nan: NAN_MAYBE };
    }
    RowIv { nan: inherited, ..out }
}

fn add(a: RowIv, b: RowIv) -> RowIv {
    let (lo, hi) = clean(a.lo + b.lo, a.hi + b.hi);
    let possible = (a.hi == FULL && b.lo == -FULL) || (a.lo == -FULL && b.hi == FULL);
    finish_nan(RowIv { lo, hi, nan: NAN_NO }, a, b, possible)
}

fn sub(a: RowIv, b: RowIv) -> RowIv {
    let (lo, hi) = clean(a.lo - b.hi, a.hi - b.lo);
    let possible = (a.hi == FULL && b.hi == FULL) || (a.lo == -FULL && b.lo == -FULL);
    finish_nan(RowIv { lo, hi, nan: NAN_NO }, a, b, possible)
}

fn corners(a: RowIv, b: RowIv, f: impl Fn(f64, f64) -> f64) -> (f64, f64) {
    let c = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
    if c.iter().any(|v| v.is_nan()) {
        return (-FULL, FULL);
    }
    (
        c.iter().copied().fold(FULL, f64::min),
        c.iter().copied().fold(-FULL, f64::max),
    )
}

fn mul(a: RowIv, b: RowIv) -> RowIv {
    let possible = (contains_zero(a) && has_inf(b)) || (contains_zero(b) && has_inf(a));
    let (lo, hi) = corners(a, b, |x, y| x * y);
    finish_nan(RowIv { lo, hi, nan: NAN_NO }, a, b, possible)
}

fn div(a: RowIv, b: RowIv) -> RowIv {
    let inherited = join_nan(a.nan, b.nan);
    if inherited == NAN_CERTAIN {
        return NAN_ROW;
    }
    // the evaluator gives NaN for |b| < 1e-12
    if b.lo > -GUARD && b.hi < GUARD {
        return NAN_ROW;
    }
    let zone = b.lo < GUARD && b.hi > -GUARD;
    let (mut lo, mut hi) = (FULL, -FULL);
    let mut nan_possible = zone;
    if b.lo <= -GUARD {
        let part = RowIv { lo: b.lo, hi: b.hi.min(-GUARD), nan: NAN_NO };
        nan_possible |= has_inf(a) && has_inf(part);
        let (l, h) = corners(a, part, |x, y| x / y);
        lo = lo.min(l);
        hi = hi.max(h);
    }
    if b.hi >= GUARD {
        let part = RowIv { lo: b.lo.max(GUARD), hi: b.hi, nan: NAN_NO };
        nan_possible |= has_inf(a) && has_inf(part);
        let (l, h) = corners(a, part, |x, y| x / y);
        lo = lo.min(l);
        hi = hi.max(h);
    }
    let both_points = is_point(a) && is_point(b);
    if both_points && nan_possible && !zone {
        return NAN_ROW;
    }
    RowIv {
        lo,
        hi,
        nan: if nan_possible { NAN_MAYBE.max(inherited) } else { inherited },
    }
}

fn square_iv(lo: f64, hi: f64) -> (f64, f64) {
    if lo >= 0.0 {
        (lo * lo, hi * hi)
    } else if hi <= 0.0 {
        (hi * hi, lo * lo)
    } else {
        (0.0, (lo * lo).max(hi * hi))
    }
}

fn unary_nan(a: RowIv, f: impl Fn(RowIv) -> RowIv) -> RowIv {
    if a.nan == NAN_CERTAIN {
        return NAN_ROW;
    }
    RowIv { nan: a.nan, ..f(a) }
}

fn exp_guarded(a: RowIv) -> RowIv {
    RowIv {
        lo: if a.lo > EXP_LIMIT { FULL } else { dn(a.lo.exp()).max(0.0) },
        hi: if a.hi > EXP_LIMIT { FULL } else { up(a.hi.exp()) },
        nan: NAN_NO,
    }
}

fn exp_raw(lo: f64, hi: f64) -> (f64, f64) {
    (dn(lo.exp()).max(0.0), up(hi.exp()))
}

fn tanh(a: RowIv) -> RowIv {
    RowIv { lo: dn(a.lo.tanh()).max(-1.0), hi: up(a.hi.tanh()).min(1.0), nan: NAN_NO }
}

fn logistic(a: RowIv) -> RowIv {
    let (elo, ehi) = exp_raw(-a.hi, -a.lo);
    RowIv { lo: 1.0 / (1.0 + ehi), hi: 1.0 / (1.0 + elo), nan: NAN_NO }
}

fn gauss(a: RowIv) -> RowIv {
    let (slo, shi) = square_iv(a.lo, a.hi);
    let (elo, ehi) = exp_raw(-shi, -slo);
    RowIv { lo: elo.max(0.0), hi: ehi.min(1.0), nan: NAN_NO }
}

fn min_max(a: RowIv, b: RowIv, is_min: bool) -> RowIv {
    let inherited = join_nan(a.nan, b.nan);
    if inherited == NAN_CERTAIN {
        return NAN_ROW;
    }
    let (lo, hi) = if is_min { (a.lo.min(b.lo), a.hi.min(b.hi)) } else { (a.lo.max(b.lo), a.hi.max(b.hi)) };
    RowIv { lo, hi, nan: inherited }
}

fn greater(a: RowIv, b: RowIv) -> RowIv {
    let inherited = join_nan(a.nan, b.nan);
    if inherited == NAN_CERTAIN {
        return NAN_ROW;
    }
    let (lo, hi) = if a.lo > b.hi {
        (1.0, 1.0)
    } else if a.hi <= b.lo {
        (0.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    RowIv { lo, hi, nan: inherited }
}

fn select(c: RowIv, t: RowIv, e: RowIv) -> RowIv {
    // a NaN condition counts as "not > 0"
    if c.nan == NAN_CERTAIN || c.hi <= 0.0 {
        return e;
    }
    if c.nan == NAN_NO && c.lo > 0.0 {
        return t;
    }
    // undecided: the value comes from either branch
    if t.nan == NAN_CERTAIN && e.nan == NAN_CERTAIN {
        return NAN_ROW;
    }
    if t.nan == NAN_CERTAIN {
        return RowIv { nan: NAN_MAYBE, ..e };
    }
    if e.nan == NAN_CERTAIN {
        return RowIv { nan: NAN_MAYBE, ..t };
    }
    RowIv { lo: t.lo.min(e.lo), hi: t.hi.max(e.hi), nan: t.nan.max(e.nan) }
}

/// Postfix words of `tree`, or `None` when an operator is outside the 14, a
/// custom block is present, a column index is `>= ncols` (the evaluator reads
/// NaN there, a kernel must not read out of bounds), or the stack exceeds
/// [`CHECK_STACK`]. `Const` is followed by the f64 bits (low word, high word),
/// `Var` by the column index, and the program ends with [`CK_HALT`].
pub fn compile_check_program(tree: &Node, ncols: usize) -> Option<Vec<u32>> {
    fn emit(n: &Node, ncols: usize, out: &mut Vec<u32>, cur: &mut usize, peak: &mut usize) -> bool {
        if n.block.is_some() {
            return false;
        }
        let (code, arity) = match n.op {
            Op::Const => {
                let b = n.val.to_bits();
                out.extend_from_slice(&[CK_CONST, b as u32, (b >> 32) as u32]);
                *cur += 1;
                *peak = (*peak).max(*cur);
                return true;
            }
            Op::Input => {
                let i = n.val as usize;
                if n.val < 0.0 || n.val.is_nan() || i >= ncols {
                    return false;
                }
                out.extend_from_slice(&[CK_VAR, i as u32]);
                *cur += 1;
                *peak = (*peak).max(*cur);
                return true;
            }
            Op::Neg => (CK_NEG, 1),
            Op::Abs => (CK_ABS, 1),
            Op::Square => (CK_SQUARE, 1),
            Op::Exp => (CK_EXP, 1),
            Op::Tanh => (CK_TANH, 1),
            Op::Logistic => (CK_LOGISTIC, 1),
            Op::Gauss => (CK_GAUSS, 1),
            Op::Add => (CK_ADD, 2),
            Op::Sub => (CK_SUB, 2),
            Op::Mul => (CK_MUL, 2),
            Op::Div => (CK_DIV, 2),
            Op::Min => (CK_MIN, 2),
            Op::Max => (CK_MAX, 2),
            Op::Greater => (CK_GREATER, 2),
            Op::If => (CK_IF, 3),
            _ => return false,
        };
        if n.kids.len() != arity {
            return false;
        }
        for k in &n.kids {
            if !emit(k, ncols, out, cur, peak) {
                return false;
            }
        }
        *cur = *cur + 1 - arity;
        out.push(code);
        true
    }
    let mut out = Vec::new();
    let (mut cur, mut peak) = (0usize, 0usize);
    if !emit(tree, ncols, &mut out, &mut cur, &mut peak) || peak > CHECK_STACK {
        return None;
    }
    out.push(CK_HALT);
    Some(out)
}

/// The interval and NaN state of one row. `col(k)` is column `k` at that row.
pub fn row_iv(words: &[u32], col: &impl Fn(usize) -> f64) -> RowIv {
    let mut st = [NAN_ROW; CHECK_STACK + 1];
    let mut sp = 0usize;
    let mut i = 0usize;
    loop {
        let w = words[i];
        i += 1;
        match w {
            CK_HALT => break,
            CK_CONST => {
                let b = words[i] as u64 | (words[i + 1] as u64) << 32;
                i += 2;
                st[sp] = point(f64::from_bits(b));
                sp += 1;
            }
            CK_VAR => {
                st[sp] = point(col(words[i] as usize));
                i += 1;
                sp += 1;
            }
            CK_NEG => {
                let a = st[sp - 1];
                st[sp - 1] = unary_nan(a, |a| RowIv { lo: -a.hi, hi: -a.lo, nan: NAN_NO });
            }
            CK_ABS => {
                let a = st[sp - 1];
                st[sp - 1] = unary_nan(a, |a| {
                    if a.lo >= 0.0 {
                        a
                    } else if a.hi <= 0.0 {
                        RowIv { lo: -a.hi, hi: -a.lo, nan: NAN_NO }
                    } else {
                        RowIv { lo: 0.0, hi: (-a.lo).max(a.hi), nan: NAN_NO }
                    }
                });
            }
            CK_SQUARE => {
                let a = st[sp - 1];
                st[sp - 1] = unary_nan(a, |a| {
                    let (lo, hi) = square_iv(a.lo, a.hi);
                    RowIv { lo, hi, nan: NAN_NO }
                });
            }
            CK_EXP => st[sp - 1] = unary_nan(st[sp - 1], exp_guarded),
            CK_TANH => st[sp - 1] = unary_nan(st[sp - 1], tanh),
            CK_LOGISTIC => st[sp - 1] = unary_nan(st[sp - 1], logistic),
            CK_GAUSS => st[sp - 1] = unary_nan(st[sp - 1], gauss),
            CK_ADD | CK_SUB | CK_MUL | CK_DIV | CK_MIN | CK_MAX | CK_GREATER => {
                let (a, b) = (st[sp - 2], st[sp - 1]);
                sp -= 1;
                st[sp - 1] = match w {
                    CK_ADD => add(a, b),
                    CK_SUB => sub(a, b),
                    CK_MUL => mul(a, b),
                    CK_DIV => div(a, b),
                    CK_MIN => min_max(a, b, true),
                    CK_MAX => min_max(a, b, false),
                    _ => greater(a, b),
                };
            }
            CK_IF => {
                let (c, t, e) = (st[sp - 3], st[sp - 2], st[sp - 1]);
                sp -= 2;
                st[sp - 1] = select(c, t, e);
            }
            _ => unreachable!("check program word {w}"),
        }
    }
    st[0]
}

/// The decision a row gets against the book thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowClass {
    /// The CPU value is certainly NaN or infinite: the slot is `+inf`.
    Nonfinite,
    /// Finite and above the enter threshold: enter 1, stay 1.
    Enter,
    /// Finite, above the stay threshold, not above the enter threshold.
    Stay,
    /// Finite and not above the stay threshold: enter 0, stay 0.
    Neither,
    Undecided,
}

/// `enter = threshold + band` and `stay = threshold - band` as the automaton
/// of `BandParams::next` computes them. A plain book has `enter == stay`.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub enter: f64,
    pub stay: f64,
}

impl Thresholds {
    pub fn new(threshold: f64, band: f64) -> Option<Self> {
        let t = Thresholds { enter: threshold + band, stay: threshold - band };
        (t.enter.is_finite() && t.stay.is_finite() && band >= 0.0).then_some(t)
    }
}

pub fn classify(v: RowIv, t: Thresholds) -> RowClass {
    if v.nan == NAN_CERTAIN || v.lo == FULL || v.hi == -FULL {
        return RowClass::Nonfinite;
    }
    if v.nan != NAN_NO || v.lo == -FULL || v.hi == FULL {
        return RowClass::Undecided;
    }
    if v.lo > t.enter {
        RowClass::Enter
    } else if v.hi <= t.stay {
        RowClass::Neither
    } else if v.lo > t.stay && v.hi <= t.enter {
        RowClass::Stay
    } else {
        RowClass::Undecided
    }
}

/// The class the CPU value gives, from the prediction itself.
pub fn class_of_value(p: f64, t: Thresholds) -> RowClass {
    if !p.is_finite() {
        RowClass::Nonfinite
    } else if p > t.enter {
        RowClass::Enter
    } else if p > t.stay {
        RowClass::Stay
    } else {
        RowClass::Neither
    }
}

/// How a slot ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// A row is certainly non-finite: the slot scores `+inf`.
    Inf,
    /// Every row is decided or patched: the planes are exact.
    Settled,
    /// More than [`SLOT_UNDECIDED_CAP`] undecided rows: the full CPU pass.
    Unsettled,
}

/// The planes of one slot: word `w` holds rows `32 w ..= 32 w + 31`, bit
/// `row % 32`; bits past `n` are 0. `enter` implies `stay`.
#[derive(Clone, Debug)]
pub struct SlotPlanes {
    pub state: SlotState,
    pub enter: Vec<u32>,
    pub stay: Vec<u32>,
    /// Rows the intervals left undecided (before patching).
    pub undecided: Vec<usize>,
    /// Rows the CPU scalar evaluator resolved to non-finite while patching.
    pub patched_nonfinite: usize,
}

/// The device kernel followed by the host patch, on the host: decide every
/// row from the intervals, then resolve each undecided row with `eval_row`
/// (`expr::eval` on that row's inputs).
pub fn settle_slot(
    words: &[u32],
    cols: &[Vec<f64>],
    n: usize,
    t: Thresholds,
    mut eval_row: impl FnMut(usize) -> f64,
) -> SlotPlanes {
    let nw = n.div_ceil(32);
    let mut enter = vec![0u32; nw];
    let mut stay = vec![0u32; nw];
    let mut undecided = Vec::new();
    let mut bad = false;
    for row in 0..n {
        let v = row_iv(words, &|k| cols[k][row]);
        match classify(v, t) {
            RowClass::Nonfinite => {
                bad = true;
                break;
            }
            RowClass::Enter => {
                enter[row >> 5] |= 1 << (row & 31);
                stay[row >> 5] |= 1 << (row & 31);
            }
            RowClass::Stay => stay[row >> 5] |= 1 << (row & 31),
            RowClass::Neither => {}
            RowClass::Undecided => undecided.push(row),
        }
    }
    if bad {
        return SlotPlanes { state: SlotState::Inf, enter, stay, undecided, patched_nonfinite: 0 };
    }
    if undecided.len() > SLOT_UNDECIDED_CAP {
        return SlotPlanes { state: SlotState::Unsettled, enter, stay, undecided, patched_nonfinite: 0 };
    }
    let mut patched_nonfinite = 0;
    for &row in &undecided {
        match class_of_value(eval_row(row), t) {
            RowClass::Nonfinite => {
                patched_nonfinite += 1;
                return SlotPlanes { state: SlotState::Inf, enter, stay, undecided, patched_nonfinite };
            }
            RowClass::Enter => {
                enter[row >> 5] |= 1 << (row & 31);
                stay[row >> 5] |= 1 << (row & 31);
            }
            RowClass::Stay => stay[row >> 5] |= 1 << (row & 31),
            _ => {}
        }
    }
    SlotPlanes { state: SlotState::Settled, enter, stay, undecided, patched_nonfinite }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_step_matches_next_up_and_down() {
        for v in [0.0, -0.0, 1.0, -1.0, 1e-310, -1e-310, 1e300, -1e300, 0.1, f64::MAX, f64::MIN] {
            let mut up = v;
            let mut dn = v;
            for _ in 0..CHECK_ULPS {
                up = up.next_up();
                dn = dn.next_down();
            }
            let (u, d) = (ulps_step(v, CHECK_ULPS), ulps_step(v, -CHECK_ULPS));
            assert!(u == up || (u.is_infinite() && up.is_infinite()), "{v:e} up {u:e} {up:e}");
            assert!(d == dn || (d.is_infinite() && dn.is_infinite()), "{v:e} down {d:e} {dn:e}");
        }
        for v in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(ulps_step(v, 64), v);
        }
        assert!(ulps_step(f64::NAN, 64).is_nan());
        assert_eq!(ulps_step(f64::MAX, 64), f64::INFINITY);
    }

    #[test]
    fn nan_state_examples() {
        use crate::expr::{constant, input, node};
        let e = |x| node(Op::Exp, vec![x]);
        let tree = node(
            Op::Sub,
            vec![node(Op::Logistic, vec![node(Op::Sub, vec![e(input(0)), e(input(1))])]), constant(2.0)],
        );
        let w = compile_check_program(&tree, 2).unwrap();
        let v = row_iv(&w, &|_| 800.0);
        assert_eq!(v.nan, NAN_CERTAIN, "logistic(inf - inf) is NaN");
        let v = row_iv(&w, &|k| if k == 0 { 800.0 } else { 0.0 });
        assert_eq!(v.nan, NAN_NO);
    }
}
