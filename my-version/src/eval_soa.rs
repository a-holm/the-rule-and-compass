//! Structure-of-arrays (SoA) CPU evaluator — **stages 1 and 2 of #27**.
//!
//! The scalar walk in [`crate::expr::eval`] visits a pointer-based tree once per
//! data row. This module instead *compiles* a tree once into a flat postfix
//! instruction stream and evaluates that stream over whole columns: the data
//! layout is `cols[var][row]`, and every instruction produces one value per row.
//!
//! Stage 2 wires this evaluator into the search's fitness path
//! (`search::eval_error`). [`crate::search::Config::simd`] selects it per run;
//! `--no-simd` forces the historical scalar walk. A tree that
//! [`compile`] cannot handle — an aggregate, meta, constraint, block or
//! comparison operator — transparently falls back to [`crate::expr::eval`], so
//! the two paths always agree.
//!
//! [`columns`] transposes row-major inputs to the `cols[var][row]` layout.
//!
//! # API
//!
//! ```
//! use rcsr::eval_soa::{compile, eval_batch};
//! # use rcsr::expr::{constant, input, node, Op};
//! let tree = node(Op::Add, vec![input(0), constant(1.0)]);
//! let program = compile(&tree).expect("supported tree");
//! let cols = vec![vec![1.0, 2.0, 3.0]];
//! let mut out = vec![0.0; 3];
//! eval_batch(&program, &cols, 3, &mut out);
//! assert_eq!(out, [2.0, 3.0, 4.0]);
//! ```
//!
//! * [`compile`] returns `None` for an operator outside [`supports`], so the
//!   caller can fall back to the scalar walk instead of getting wrong numbers.
//! * [`eval_batch`] runs the column-tile kernel; the AVX build is chosen at
//!   runtime ([`avx_available`]) and the same loops are the portable build.
//! * [`eval_batch_scalar`] is the portable loop explicitly, for measurements.
//! * [`sse`] is a small convenience for benchmarks.
//!
//! # Supported operators
//!
//! Unary: `neg`, `exp`, `ln`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `sinh`,
//! `cosh`, `tanh`, `logistic`, `step`, `gauss`, `square`, `cube`, `floor`,
//! `ceil`, `round`, `sign`.
//!
//! Binary: `add`, `sub`, `mul`, `div`, `pow`, `min`, `max`.
//!
//! Ternary: `if`.
//!
//! Everything else — `mod`, `atan2`, `erf`, `erfc`, `asinh`, `acosh`, `atanh`,
//! the comparisons/logical operators, the constraints and the meta operators —
//! returns `None` from [`compile`]. The domain guards are copied from
//! [`crate::expr::eval`] verbatim (including `exp(x > 700) = +inf`,
//! `ln(x <= 0) = NaN`, `div(|b| < 1e-12) = NaN`, `sqrt(x < 0) = NaN`,
//! `pow(negative, non-integer) = NaN`), so supported trees agree numerically.
//!
//! # Column tiles
//!
//! [`compile`] folds the stream into steps with fixed destination slots, and an
//! `Input` or `Const` leaf feeding an operator becomes an operand read in
//! place. [`eval_batch`] runs each step over [`TILE`] rows in one loop. On
//! `x86_64` the loops are built under `#[target_feature(enable = "avx")]` and
//! selected at runtime with `is_x86_feature_detected!("avx")`; the same source
//! is the portable build. Nothing requires AVX2 or FMA. The transcendental
//! functions stay one scalar `f64` call per row, so every value comes from the
//! same operation as the scalar path.
//!
//! [`eval_batch_rows4`] is the previous four-row AVX kernel, kept as the
//! reference for tests and benchmarks.
//!
//! [`eval_batch_into`] takes a caller-owned [`Scratch`] for the tile stack.
//! Inputs that are missing or shorter than `rows` take the scalar loop.

use crate::expr::{Node, Op};

/// Number of rows processed at once by the four-row AVX kernel.
pub const SIMD_WIDTH: usize = 4;

/// A flat postfix instruction. Every instruction pushes one value, after
/// consuming [`Instr::arity`] operands, so the program is a stack machine.
#[derive(Clone, Copy, Debug)]
enum Instr {
    Const(f64),
    Input(usize),
    Neg,
    Exp,
    Ln,
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Logistic,
    Step,
    Gauss,
    Square,
    Cube,
    Floor,
    Ceil,
    Round,
    Sign,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    If,
}

impl Instr {
    /// Operands consumed, not counting the result pushed.
    fn arity(self) -> usize {
        use Instr::*;
        match self {
            Const(..) | Input(..) => 0,
            Neg | Exp | Ln | Abs | Sqrt | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic
            | Step | Gauss | Square | Cube | Floor | Ceil | Round | Sign => 1,
            Add | Sub | Mul | Div | Pow | Min | Max => 2,
            If => 3,
        }
    }
}

/// A tree compiled to a flat postfix program over SoA columns.
///
/// Construct one with [`compile`]; the fields are intentionally private so a
/// program cannot be hand-built in a way that breaks the stack invariants.
#[derive(Clone, Debug)]
pub struct Program {
    instrs: Vec<Instr>,
    /// Maximum operand-stack depth over all instructions.
    stack_depth: usize,
    /// One past the largest input index referenced by the program.
    n_inputs: usize,
    /// The tile program: leaves folded into operands, destination slots fixed.
    steps: Vec<Step>,
    /// Sorted distinct input indices the program reads.
    inputs: Vec<usize>,
}

impl Program {
    /// Number of instructions in the stream.
    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    /// A compiled tree always has at least one instruction.
    pub fn is_empty(&self) -> bool {
        self.instrs.is_empty()
    }

    /// Peak operand-stack depth; the evaluators allocate exactly this much.
    pub fn stack_depth(&self) -> usize {
        self.stack_depth
    }

    /// Number of input columns the program may read (`max(index) + 1`).
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }
}

/// Rows per tile of the column kernel.
pub const TILE: usize = 64;

/// Where a step reads an operand from.
#[derive(Clone, Copy, Debug)]
enum Src {
    /// The stack slot at this operand position. For the first operand this is
    /// always the destination slot itself.
    Slot(usize),
    /// An input column, read in place.
    Col(usize),
    /// A constant.
    Imm(f64),
}

/// One tile instruction. `a`, `b`, `c` are the operands in postfix order; the
/// result goes to slot `dst`.
#[derive(Clone, Copy, Debug)]
struct Step {
    ins: Instr,
    dst: usize,
    a: Src,
    b: Src,
    c: Src,
}

/// Fold the postfix stream into tile steps. Every instruction writes the slot
/// of its first operand, which is a compile-time fact, and `Input`/`Const`
/// leaves become operands instead of steps. A program that is a single leaf
/// keeps one copy step.
fn plan(instrs: &[Instr]) -> Vec<Step> {
    let mut stack: Vec<Src> = Vec::new();
    let mut steps = Vec::new();
    for &ins in instrs {
        match ins {
            Instr::Const(v) => stack.push(Src::Imm(v)),
            Instr::Input(v) => stack.push(Src::Col(v)),
            _ => {
                let at = stack.len() - ins.arity();
                let get = |k: usize| stack.get(at + k).copied().unwrap_or(Src::Imm(0.0));
                let step = Step {
                    ins,
                    dst: at,
                    a: get(0),
                    b: get(1),
                    c: get(2),
                };
                stack.truncate(at);
                steps.push(step);
                stack.push(Src::Slot(at));
            }
        }
    }
    if steps.is_empty() {
        if let (Some(&ins), Some(&a)) = (instrs.first(), stack.first()) {
            steps.push(Step {
                ins,
                dst: 0,
                a,
                b: Src::Imm(0.0),
                c: Src::Imm(0.0),
            });
        }
    }
    steps
}

/// Map an operator to its instruction, or `None` when this evaluator does not
/// implement it. Leaves (`Const`, `Input`) are handled by [`compile`] directly.
fn instr_for(op: Op) -> Option<Instr> {
    use Op::*;
    Some(match op {
        Neg => Instr::Neg,
        Exp => Instr::Exp,
        Ln => Instr::Ln,
        Abs => Instr::Abs,
        Sqrt => Instr::Sqrt,
        Sin => Instr::Sin,
        Cos => Instr::Cos,
        Tan => Instr::Tan,
        Sinh => Instr::Sinh,
        Cosh => Instr::Cosh,
        Tanh => Instr::Tanh,
        Logistic => Instr::Logistic,
        Step => Instr::Step,
        Gauss => Instr::Gauss,
        Square => Instr::Square,
        Cube => Instr::Cube,
        Floor => Instr::Floor,
        Ceil => Instr::Ceil,
        Round => Instr::Round,
        Sign => Instr::Sign,
        Add => Instr::Add,
        Sub => Instr::Sub,
        Mul => Instr::Mul,
        Div => Instr::Div,
        Pow => Instr::Pow,
        Min => Instr::Min,
        Max => Instr::Max,
        If => Instr::If,
        _ => return None,
    })
}

/// Whether [`compile`] supports `op`.
///
/// This is exactly the set listed in the module docs. Callers can use it to
/// decide up front whether a candidate is worth compiling, without walking the
/// tree twice.
pub fn supports(op: Op) -> bool {
    matches!(op, Op::Const | Op::Input) || instr_for(op).is_some()
}

/// Compile `tree` to a flat postfix program.
///
/// Returns `None` when any node uses an operator outside [`supports`] (or when a
/// node has fewer children than its operator requires). The caller must then use
/// the scalar tree walk; returning a partial program would be wrong.
pub fn compile(tree: &Node) -> Option<Program> {
    let mut instrs = Vec::new();
    let mut depth = 0usize;
    let mut max_depth = 0usize;
    let mut n_inputs = 0usize;
    emit(tree, &mut instrs, &mut depth, &mut max_depth, &mut n_inputs)?;
    debug_assert_eq!(depth, 1, "a complete expression leaves one value");
    let steps = plan(&instrs);
    let mut inputs: Vec<usize> = instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Input(v) => Some(*v),
            _ => None,
        })
        .collect();
    inputs.sort_unstable();
    inputs.dedup();
    Some(Program {
        instrs,
        stack_depth: max_depth,
        n_inputs,
        steps,
        inputs,
    })
}

/// Postfix emission. `depth` is the operand-stack depth *after* the subtree
/// emitted so far; `max_depth` is its peak, which is what the evaluators size
/// their stack to.
fn emit(
    n: &Node,
    out: &mut Vec<Instr>,
    depth: &mut usize,
    max_depth: &mut usize,
    n_inputs: &mut usize,
) -> Option<()> {
    match n.op {
        Op::Const => {
            out.push(Instr::Const(n.val));
            *depth += 1;
            *max_depth = (*max_depth).max(*depth);
        }
        Op::Input => {
            let i = n.val as usize;
            out.push(Instr::Input(i));
            *n_inputs = (*n_inputs).max(i.saturating_add(1));
            *depth += 1;
            *max_depth = (*max_depth).max(*depth);
        }
        op => {
            let ins = instr_for(op)?;
            let arity = ins.arity();
            if n.kids.len() < arity {
                return None;
            }
            for kid in n.kids.iter().take(arity) {
                emit(kid, out, depth, max_depth, n_inputs)?;
            }
            if *depth < arity {
                return None;
            }
            // arity operands become one result.
            *depth -= arity - 1;
            out.push(ins);
        }
    }
    Some(())
}

/// Scalar semantics of a unary instruction.
///
/// These match [`crate::expr::eval`] operation for operation, including every
/// domain guard; the AVX lane fallback calls exactly this function.
#[inline(always)]
fn apply_unary(ins: Instr, a: f64) -> f64 {
    use Instr::*;
    match ins {
        Neg => -a,
        Exp => {
            if a > 700.0 {
                f64::INFINITY
            } else {
                a.exp()
            }
        }
        Ln => {
            if a > 0.0 {
                a.ln()
            } else {
                f64::NAN
            }
        }
        Abs => a.abs(),
        Sqrt => {
            if a >= 0.0 {
                a.sqrt()
            } else {
                f64::NAN
            }
        }
        Sin => a.sin(),
        Cos => a.cos(),
        Tan => {
            let t = a.tan();
            if t.is_finite() {
                t
            } else {
                f64::NAN
            }
        }
        Sinh => a.sinh(),
        Cosh => a.cosh(),
        Tanh => a.tanh(),
        Logistic => 1.0 / (1.0 + (-a).exp()),
        Step => {
            if a > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Gauss => (-a * a).exp(),
        Square => a * a,
        Cube => a * a * a,
        Floor => a.floor(),
        Ceil => a.ceil(),
        Round => a.round(),
        Sign => {
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else if a == 0.0 {
                0.0
            } else {
                f64::NAN
            }
        }
        // The evaluator only asks for unary instructions here.
        _ => f64::NAN,
    }
}

/// Scalar semantics of a binary instruction, matching [`crate::expr::eval`].
#[inline(always)]
fn apply_binary(ins: Instr, a: f64, b: f64) -> f64 {
    use Instr::*;
    match ins {
        Add => a + b,
        Sub => a - b,
        Mul => a * b,
        Div => {
            if b.abs() < 1e-12 {
                f64::NAN
            } else {
                a / b
            }
        }
        Pow => {
            if a < 0.0 && (b - b.round()).abs() > 1e-9 {
                f64::NAN
            } else {
                a.powf(b)
            }
        }
        Min => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
        Max => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        // The evaluator only asks for binary instructions here.
        _ => f64::NAN,
    }
}

/// `cols[var][row]`, or `NaN` when the column or row is absent. This mirrors
/// `expr::eval`'s `Input` rule: an index outside the row is `NaN`.
#[inline(always)]
fn read_input(cols: &[Vec<f64>], var: usize, row: usize) -> f64 {
    cols.get(var)
        .and_then(|c| c.get(row))
        .copied()
        .unwrap_or(f64::NAN)
}

/// Portable scalar SoA loop over rows `start..end`.
///
/// The value stack is allocated once for the whole range, not per row.
fn eval_scalar_rows(p: &Program, cols: &[Vec<f64>], start: usize, end: usize, out: &mut [f64]) {
    if start >= end {
        return;
    }
    if p.instrs.is_empty() {
        for v in &mut out[start..end] {
            *v = f64::NAN;
        }
        return;
    }
    let mut stack = vec![f64::NAN; p.stack_depth.max(1)];
    for row in start..end {
        let mut sp = 0usize;
        for ins in &p.instrs {
            match *ins {
                Instr::Const(v) => {
                    stack[sp] = v;
                    sp += 1;
                }
                Instr::Input(var) => {
                    stack[sp] = read_input(cols, var, row);
                    sp += 1;
                }
                Instr::If => {
                    // Postfix order is cond, then-branch, else-branch.
                    let else_v = stack[sp - 1];
                    let then_v = stack[sp - 2];
                    let cond = stack[sp - 3];
                    stack[sp - 3] = if cond > 0.0 { then_v } else { else_v };
                    sp -= 2;
                }
                Instr::Add
                | Instr::Sub
                | Instr::Mul
                | Instr::Div
                | Instr::Pow
                | Instr::Min
                | Instr::Max => {
                    let b = stack[sp - 1];
                    let a = stack[sp - 2];
                    stack[sp - 2] = apply_binary(*ins, a, b);
                    sp -= 1;
                }
                unary => {
                    stack[sp - 1] = apply_unary(unary, stack[sp - 1]);
                }
            }
        }
        debug_assert_eq!(sp, 1);
        out[row] = stack[0];
    }
}

/// Transpose row-major inputs (`x[row][var]`) to the SoA layout
/// (`cols[var][row]`) this module evaluates over.
///
/// The number of columns is the longest input row, so a row that is shorter
/// than the rest reads as `NaN` exactly as [`crate::expr::eval`]'s `Input` rule
/// does. This is the one O(rows × vars) pass the caller pays per sample, not
/// per row of the tree walk.
pub fn columns(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let nvars = x.iter().map(Vec::len).max().unwrap_or(0);
    let mut cols = vec![vec![f64::NAN; x.len()]; nvars];
    for (row, values) in x.iter().enumerate() {
        for (var, &v) in values.iter().enumerate() {
            cols[var][row] = v;
        }
    }
    cols
}

/// Evaluate `program` over `rows` rows with the portable scalar SoA loop.
///
/// `cols[var][row]` holds the inputs; `out[row]` receives the result. At most
/// `out.len()` rows are written, so a short output buffer is safe (the excess is
/// ignored). Zero rows is a no-op.
pub fn eval_batch_scalar(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64]) {
    let rows = rows.min(out.len());
    eval_scalar_rows(p, cols, 0, rows, out);
}

/// The value stack of the tile kernel, owned by the caller so a worker reuses
/// it across candidates. It grows on demand and never shrinks.
#[derive(Debug, Default)]
pub struct Scratch {
    stack: Vec<f64>,
}

impl Scratch {
    /// An empty scratch; the first evaluation sizes it.
    pub fn new() -> Self {
        Scratch::default()
    }

    /// Current capacity in `f64` values, for tests that count regrowth.
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Whether the scratch has not been sized yet.
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

/// Evaluate `program` over `rows` rows, using AVX when the CPU supports it.
///
/// `cols[var][row]` holds the inputs; `out[row]` receives the result. At most
/// `out.len()` rows are written. Zero rows is a no-op. This allocates a fresh
/// [`Scratch`]; use [`eval_batch_into`] to reuse one.
pub fn eval_batch(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64]) {
    eval_batch_into(p, cols, rows, out, &mut Scratch::new());
}

/// [`eval_batch`] with a caller-owned [`Scratch`].
///
/// The column-tile kernel runs each instruction over [`TILE`] rows in one
/// loop. Every value comes from the same `f64` operation, in the same order,
/// as [`eval_batch_rows4`], [`eval_batch_scalar`] and [`crate::expr::eval`].
/// When an input the program reads is missing or shorter than `rows`, the
/// portable scalar loop supplies the `NaN` rule.
pub fn eval_batch_into(
    p: &Program,
    cols: &[Vec<f64>],
    rows: usize,
    out: &mut [f64],
    scratch: &mut Scratch,
) {
    let rows = rows.min(out.len());
    if rows == 0 {
        return;
    }
    if p.steps.is_empty() {
        for v in &mut out[..rows] {
            *v = f64::NAN;
        }
        return;
    }
    let covered = p
        .inputs
        .iter()
        .all(|&v| cols.get(v).is_some_and(|c| c.len() >= rows));
    if !covered {
        eval_scalar_rows(p, cols, 0, rows, out);
        return;
    }
    let need = p.stack_depth.max(1) * TILE;
    if scratch.stack.len() < need {
        scratch.stack.resize(need, 0.0);
    }
    let stack = &mut scratch.stack[..need];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: the feature is detected above and the function is
            // annotated `target_feature(enable = "avx")`.
            unsafe { tile::eval_avx(p, cols, rows, out, stack) };
            return;
        }
    }
    tile::eval(p, cols, rows, out, stack);
}

/// The four-row AVX kernel: the whole program once per group of four rows.
/// Kept as the reference [`eval_batch`] is measured and tested against.
pub fn eval_batch_rows4(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64]) {
    let rows = rows.min(out.len());
    if rows == 0 {
        return;
    }
    if p.instrs.is_empty() {
        for v in &mut out[..rows] {
            *v = f64::NAN;
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            // SAFETY: the feature is detected above and the function is
            // annotated `target_feature(enable = "avx")`.
            unsafe { avx::eval(p, cols, rows, out) };
            return;
        }
    }
    eval_batch_scalar(p, cols, rows, out);
}

/// Whether the AVX kernel will be used on this CPU.
pub fn avx_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Sum of squared residuals over the first `rows` rows.
///
/// This is the raw `Σ (pred - y)²`, not the engine's mean-based metric. It
/// returns `+inf` when any prediction is non-finite, matching the engine's
/// convention that a non-finite prediction is unusable. `rows` is clamped to
/// `y.len()`, and zero rows gives `0.0`.
pub fn sse(p: &Program, cols: &[Vec<f64>], rows: usize, y: &[f64]) -> f64 {
    let rows = rows.min(y.len());
    let mut pred = vec![0.0; rows];
    eval_batch(p, cols, rows, &mut pred);
    let mut acc = 0.0;
    for (&prediction, &target) in pred.iter().zip(y.iter()) {
        if !prediction.is_finite() {
            return f64::INFINITY;
        }
        let r = prediction - target;
        acc += r * r;
    }
    acc
}

/// The column-tile kernel. Selected by [`eval_batch_into`]; the AVX entry only
/// changes the code generation of the same source.
mod tile {
    use super::*;

    #[derive(Clone, Copy)]
    enum A<'a> {
        /// The destination itself, updated in place.
        D,
        S(&'a [f64]),
        K(f64),
    }

    #[derive(Clone, Copy)]
    enum B<'a> {
        S(&'a [f64]),
        K(f64),
    }

    #[inline(always)]
    fn at(o: B, i: usize) -> f64 {
        match o {
            B::S(s) => s[i],
            B::K(k) => k,
        }
    }

    #[inline(always)]
    fn un(d: &mut [f64], a: A, f: impl Fn(f64) -> f64) {
        match a {
            A::D => {
                for v in d.iter_mut() {
                    *v = f(*v);
                }
            }
            A::S(x) => {
                for (v, &xi) in d.iter_mut().zip(x) {
                    *v = f(xi);
                }
            }
            A::K(k) => d.fill(f(k)),
        }
    }

    #[inline(always)]
    fn bin(d: &mut [f64], a: A, b: B, f: impl Fn(f64, f64) -> f64) {
        match (a, b) {
            (A::D, B::S(y)) => {
                for (v, &yi) in d.iter_mut().zip(y) {
                    *v = f(*v, yi);
                }
            }
            (A::D, B::K(k)) => {
                for v in d.iter_mut() {
                    *v = f(*v, k);
                }
            }
            (A::S(x), B::S(y)) => {
                for ((v, &xi), &yi) in d.iter_mut().zip(x).zip(y) {
                    *v = f(xi, yi);
                }
            }
            (A::S(x), B::K(k)) => {
                for (v, &xi) in d.iter_mut().zip(x) {
                    *v = f(xi, k);
                }
            }
            (A::K(k), B::S(y)) => {
                for (v, &yi) in d.iter_mut().zip(y) {
                    *v = f(k, yi);
                }
            }
            (A::K(k), B::K(m)) => d.fill(f(k, m)),
        }
    }

    /// `if cond > 0 { then } else { else }` per row.
    #[inline(always)]
    fn select(d: &mut [f64], a: A, b: B, c: B) {
        match a {
            A::D => {
                for i in 0..d.len() {
                    d[i] = if d[i] > 0.0 { at(b, i) } else { at(c, i) };
                }
            }
            A::S(x) => {
                for i in 0..d.len() {
                    d[i] = if x[i] > 0.0 { at(b, i) } else { at(c, i) };
                }
            }
            A::K(k) => {
                for i in 0..d.len() {
                    d[i] = if k > 0.0 { at(b, i) } else { at(c, i) };
                }
            }
        }
    }

    macro_rules! unary_arms {
        ($ins:expr, $d:expr, $a:expr, $($op:ident),*) => {
            match $ins {
                $(Instr::$op => un($d, $a, |x| apply_unary(Instr::$op, x)),)*
                _ => {}
            }
        };
    }

    macro_rules! binary_arms {
        ($ins:expr, $d:expr, $a:expr, $b:expr, $($op:ident),*) => {
            match $ins {
                $(Instr::$op => bin($d, $a, $b, |x, y| apply_binary(Instr::$op, x, y)),)*
                _ => {}
            }
        };
    }

    #[inline(always)]
    fn run_tile(p: &Program, cols: &[Vec<f64>], base: usize, n: usize, s: &mut [f64]) {
        for st in &p.steps {
            let (lo, hi) = s.split_at_mut((st.dst + 1) * TILE);
            let d = &mut lo[st.dst * TILE..st.dst * TILE + n];
            let hi: &[f64] = hi;
            let a = match st.a {
                Src::Slot(_) => A::D,
                Src::Col(v) => A::S(&cols[v][base..base + n]),
                Src::Imm(k) => A::K(k),
            };
            let b_of = |src: Src| match src {
                Src::Slot(j) => B::S(&hi[(j - st.dst - 1) * TILE..(j - st.dst - 1) * TILE + n]),
                Src::Col(v) => B::S(&cols[v][base..base + n]),
                Src::Imm(k) => B::K(k),
            };
            match st.ins {
                Instr::Const(_) | Instr::Input(_) => un(d, a, |x| x),
                Instr::If => select(d, a, b_of(st.b), b_of(st.c)),
                Instr::Add
                | Instr::Sub
                | Instr::Mul
                | Instr::Div
                | Instr::Pow
                | Instr::Min
                | Instr::Max => {
                    let b = b_of(st.b);
                    binary_arms!(st.ins, d, a, b, Add, Sub, Mul, Div, Pow, Min, Max);
                }
                _ => unary_arms!(
                    st.ins, d, a, Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh,
                    Logistic, Step, Gauss, Square, Cube, Floor, Ceil, Round, Sign
                ),
            }
        }
    }

    #[inline(always)]
    fn run(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64], s: &mut [f64]) {
        let mut base = 0;
        while base < rows {
            let n = TILE.min(rows - base);
            run_tile(p, cols, base, n, s);
            out[base..base + n].copy_from_slice(&s[..n]);
            base += n;
        }
    }

    pub(super) fn eval(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64], s: &mut [f64]) {
        run(p, cols, rows, out, s);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn eval_avx(
        p: &Program,
        cols: &[Vec<f64>],
        rows: usize,
        out: &mut [f64],
        s: &mut [f64],
    ) {
        run(p, cols, rows, out, s);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    //! Four-lane AVX kernel. Selected at runtime by [`super::eval_batch`].

    use super::*;
    use std::arch::x86_64::*;

    /// Load four consecutive rows of column `var`, padding absent values with
    /// `NaN` exactly as [`read_input`] would.
    #[target_feature(enable = "avx")]
    unsafe fn load4(cols: &[Vec<f64>], var: usize, row: usize) -> __m256d {
        if let Some(c) = cols.get(var) {
            if row + 4 <= c.len() {
                return _mm256_loadu_pd(c.as_ptr().add(row));
            }
            let mut lanes = [f64::NAN; 4];
            let n = c.len().saturating_sub(row).min(4);
            for (j, lane) in lanes.iter_mut().take(n).enumerate() {
                *lane = c[row + j];
            }
            return _mm256_loadu_pd(lanes.as_ptr());
        }
        _mm256_set1_pd(f64::NAN)
    }

    /// Apply one scalar unary instruction to each of the four lanes.
    #[target_feature(enable = "avx")]
    unsafe fn map4(v: __m256d, ins: Instr) -> __m256d {
        let mut lanes = [0.0f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), v);
        for lane in &mut lanes {
            *lane = apply_unary(ins, *lane);
        }
        _mm256_loadu_pd(lanes.as_ptr())
    }

    /// `pow` per lane; it needs both the base and the exponent.
    #[target_feature(enable = "avx")]
    unsafe fn pow4(a: __m256d, b: __m256d) -> __m256d {
        let mut av = [0.0f64; 4];
        let mut bv = [0.0f64; 4];
        _mm256_storeu_pd(av.as_mut_ptr(), a);
        _mm256_storeu_pd(bv.as_mut_ptr(), b);
        for i in 0..4 {
            av[i] = apply_binary(Instr::Pow, av[i], bv[i]);
        }
        _mm256_loadu_pd(av.as_ptr())
    }

    /// The AVX kernel, four rows at a time, with a scalar tail.
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn eval(p: &Program, cols: &[Vec<f64>], rows: usize, out: &mut [f64]) {
        let zero = _mm256_setzero_pd();
        let sign = _mm256_set1_pd(-0.0);
        let nan = _mm256_set1_pd(f64::NAN);
        let eps = _mm256_set1_pd(1e-12);

        let mut stack = vec![zero; p.stack_depth.max(1)];
        let full = rows - rows % 4;
        let mut base = 0usize;
        while base < full {
            let mut sp = 0usize;
            for ins in &p.instrs {
                match *ins {
                    Instr::Const(v) => {
                        stack[sp] = _mm256_set1_pd(v);
                        sp += 1;
                    }
                    Instr::Input(var) => {
                        stack[sp] = load4(cols, var, base);
                        sp += 1;
                    }
                    Instr::If => {
                        let else_v = stack[sp - 1];
                        let then_v = stack[sp - 2];
                        let cond = stack[sp - 3];
                        let mask = _mm256_cmp_pd(cond, zero, _CMP_GT_OQ);
                        stack[sp - 3] = _mm256_blendv_pd(else_v, then_v, mask);
                        sp -= 2;
                    }
                    Instr::Add => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        stack[sp - 2] = _mm256_add_pd(a, b);
                        sp -= 1;
                    }
                    Instr::Sub => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        stack[sp - 2] = _mm256_sub_pd(a, b);
                        sp -= 1;
                    }
                    Instr::Mul => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        stack[sp - 2] = _mm256_mul_pd(a, b);
                        sp -= 1;
                    }
                    Instr::Div => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        let abs_b = _mm256_andnot_pd(sign, b);
                        let bad = _mm256_cmp_pd(abs_b, eps, _CMP_LT_OQ);
                        let q = _mm256_div_pd(a, b);
                        stack[sp - 2] = _mm256_blendv_pd(q, nan, bad);
                        sp -= 1;
                    }
                    Instr::Pow => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        stack[sp - 2] = pow4(a, b);
                        sp -= 1;
                    }
                    Instr::Min => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        let m = _mm256_min_pd(a, b);
                        let unordered = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
                        stack[sp - 2] = _mm256_blendv_pd(m, nan, unordered);
                        sp -= 1;
                    }
                    Instr::Max => {
                        let b = stack[sp - 1];
                        let a = stack[sp - 2];
                        let m = _mm256_max_pd(a, b);
                        let unordered = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
                        stack[sp - 2] = _mm256_blendv_pd(m, nan, unordered);
                        sp -= 1;
                    }
                    Instr::Neg => {
                        stack[sp - 1] = _mm256_xor_pd(stack[sp - 1], sign);
                    }
                    Instr::Abs => {
                        stack[sp - 1] = _mm256_andnot_pd(sign, stack[sp - 1]);
                    }
                    Instr::Sqrt => {
                        let a = stack[sp - 1];
                        let negative = _mm256_cmp_pd(a, zero, _CMP_LT_OQ);
                        let s = _mm256_sqrt_pd(a);
                        stack[sp - 1] = _mm256_blendv_pd(s, nan, negative);
                    }
                    Instr::Square => {
                        let a = stack[sp - 1];
                        stack[sp - 1] = _mm256_mul_pd(a, a);
                    }
                    Instr::Cube => {
                        let a = stack[sp - 1];
                        stack[sp - 1] = _mm256_mul_pd(_mm256_mul_pd(a, a), a);
                    }
                    // Everything else is a transcendental (or a comparison-like
                    // step) with no AVX instruction: one scalar `f64` call per
                    // lane, still without walking the tree.
                    unary => {
                        stack[sp - 1] = map4(stack[sp - 1], unary);
                    }
                }
            }
            debug_assert_eq!(sp, 1);
            _mm256_storeu_pd(out.as_mut_ptr().add(base), stack[0]);
            base += 4;
        }
        if base < rows {
            eval_scalar_rows(p, cols, base, rows, out);
        }
    }
}
