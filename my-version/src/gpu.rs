//! Optional GPU backend: batched fitness evaluation on an NVIDIA GPU.
//!
//! Enabled with `cargo build --features cuda`. The kernel lives in
//! `src/cuda/eval.cu`; this module compiles expression trees to its postfix
//! bytecode and wraps the C API.
//!
//! Why this shape (see `docs/optimisation.md` §4 and `docs/gpu-backend.md`):
//!
//! * the search's hot loop is "score P candidates against N rows", which is
//!   P × N independent evaluations — embarrassingly parallel;
//! * one thread block per expression, threads across rows, means every thread in
//!   a block runs the same instruction stream, so program structure causes no
//!   warp divergence;
//! * the representation is a compact postfix program, the same idea that makes
//!   the CPU side cache-friendly;
//! * FP32 throughout, because on both the RTX 3060 and the GB10 in the DGX Spark
//!   FP64 runs at 1/64 of FP32.

use crate::expr::{constant, node, Node, Op};

/* Architecture-name parsing and toolkit requirements. The same file is
 * `include!`d by `build.rs`; declaring it here gives it unit tests. */
#[path = "cuda/arch.rs"]
pub mod arch;

/* Opcodes, kept in sync with src/cuda/eval.cu. */
const OP_CONST: u32 = 0;
const OP_VAR: u32 = 1;
const OP_ADD: u32 = 2;
const OP_SUB: u32 = 3;
const OP_MUL: u32 = 4;
const OP_DIV: u32 = 5;
const OP_SIN: u32 = 6;
const OP_COS: u32 = 7;
const OP_TAN: u32 = 8;
const OP_EXP: u32 = 9;
const OP_LOG: u32 = 10;
const OP_SQRT: u32 = 11;
const OP_ABS: u32 = 12;
const OP_TANH: u32 = 13;
const OP_SIG: u32 = 14;
const OP_SQUARE: u32 = 15;
const OP_CUBE: u32 = 16;
const OP_NEG: u32 = 17;
const OP_GAUSS: u32 = 18;
const OP_STEP: u32 = 19;
const OP_POW: u32 = 20;
const OP_SINH: u32 = 21;
const OP_COSH: u32 = 22;
// #131: native hard operators.
const OP_FLOOR: u32 = 23;
const OP_CEIL: u32 = 24;
const OP_ROUND: u32 = 25;
const OP_MOD: u32 = 26;
const OP_ATAN2: u32 = 27;
const OP_ERF: u32 = 28;
const OP_ERFC: u32 = 29;
const OP_MIN: u32 = 30;
const OP_MAX: u32 = 31;
const OP_IF: u32 = 32;
const OP_SIGN: u32 = 33;
const OP_EQUAL: u32 = 34;
const OP_LESS: u32 = 35;
const OP_LESS_OR_EQUAL: u32 = 36;
const OP_GREATER: u32 = 37;
const OP_GREATER_OR_EQUAL: u32 = 38;
const OP_AND: u32 = 39;
const OP_OR: u32 = 40;
const OP_XOR: u32 = 41;
const OP_NOT: u32 = 42;
const OP_HALT: u32 = 255;

/// `(name, opcode)` for the enum-consistency test and for the per-generation
/// opcode report. Keep the order identical to `src/cuda/eval.cu`.
const OPCODE_TABLE: &[(&str, u32)] = &[
    ("OP_CONST", OP_CONST),
    ("OP_VAR", OP_VAR),
    ("OP_ADD", OP_ADD),
    ("OP_SUB", OP_SUB),
    ("OP_MUL", OP_MUL),
    ("OP_DIV", OP_DIV),
    ("OP_SIN", OP_SIN),
    ("OP_COS", OP_COS),
    ("OP_TAN", OP_TAN),
    ("OP_EXP", OP_EXP),
    ("OP_LOG", OP_LOG),
    ("OP_SQRT", OP_SQRT),
    ("OP_ABS", OP_ABS),
    ("OP_TANH", OP_TANH),
    ("OP_SIG", OP_SIG),
    ("OP_SQUARE", OP_SQUARE),
    ("OP_CUBE", OP_CUBE),
    ("OP_NEG", OP_NEG),
    ("OP_GAUSS", OP_GAUSS),
    ("OP_STEP", OP_STEP),
    ("OP_POW", OP_POW),
    ("OP_SINH", OP_SINH),
    ("OP_COSH", OP_COSH),
    ("OP_FLOOR", OP_FLOOR),
    ("OP_CEIL", OP_CEIL),
    ("OP_ROUND", OP_ROUND),
    ("OP_MOD", OP_MOD),
    ("OP_ATAN2", OP_ATAN2),
    ("OP_ERF", OP_ERF),
    ("OP_ERFC", OP_ERFC),
    ("OP_MIN", OP_MIN),
    ("OP_MAX", OP_MAX),
    ("OP_IF", OP_IF),
    ("OP_SIGN", OP_SIGN),
    ("OP_EQUAL", OP_EQUAL),
    ("OP_LESS", OP_LESS),
    ("OP_LESS_OR_EQUAL", OP_LESS_OR_EQUAL),
    ("OP_GREATER", OP_GREATER),
    ("OP_GREATER_OR_EQUAL", OP_GREATER_OR_EQUAL),
    ("OP_AND", OP_AND),
    ("OP_OR", OP_OR),
    ("OP_XOR", OP_XOR),
    ("OP_NOT", OP_NOT),
    ("OP_HALT", OP_HALT),
];

/// The kernel's operand stack (`RC_MAX_STACK` in `eval.cu`), less the headroom
/// the guard reserves. A postfix program deeper than this is rejected on the
/// host instead of silently overrunning the device stack.
const MAX_PROGRAM_STACK: usize = 56;

/// Why a tree cannot run on the kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The operator has no kernel opcode even after [`lower`]: `asinh`,
    /// `acosh`, `atanh`, constraints, aggregates and custom blocks.
    Unsupported(&'static str),
    /// The lowered postfix program would use more operand-stack slots than the
    /// kernel has (`RC_MAX_STACK`).
    StackOverflow { peak: usize, limit: usize },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Unsupported(op) => {
                write!(f, "operator `{}` has no GPU opcode", op)
            }
            CompileError::StackOverflow { peak, limit } => write!(
                f,
                "postfix program needs {peak} stack slots, the kernel allows {limit}"
            ),
        }
    }
}

/// One program that the kernel can run, plus the opcodes it actually emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledProgram {
    words: Vec<u32>,
    opcodes: Vec<(String, usize)>,
}

impl CompiledProgram {
    /// Postfix words, `HALT`-terminated.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// `(opcode name, occurrences)` in the **emitted** program, sorted by
    /// opcode number. Constants, variables and `HALT` are excluded.
    pub fn opcodes(&self) -> &[(String, usize)] {
        &self.opcodes
    }
}

/// Compile one expression through the one guarded entry point every caller must
/// use: [`lower`] first, then the operand-stack guard, then emission.
///
/// `compile_program` is what both the search's pool batch and
/// [`Population::add`] call, so the two paths cannot disagree about which
/// operators are native or how deep a program may be.
pub fn compile_program(tree: &Node) -> Result<CompiledProgram, CompileError> {
    let lowered = lower(tree);
    let peak = peak_stack(&lowered);
    if peak > MAX_PROGRAM_STACK {
        return Err(CompileError::StackOverflow {
            peak,
            limit: MAX_PROGRAM_STACK,
        });
    }
    let mut words = Vec::new();
    if !emit(&lowered, &mut words) {
        let op = first_unsupported(&lowered).unwrap_or("operator");
        return Err(CompileError::Unsupported(op));
    }
    words.push(OP_HALT);
    let opcodes = count_opcodes(&words);
    Ok(CompiledProgram { words, opcodes })
}

/// Low-level emitter. Call [`compile_program`] instead: it applies [`lower`]
/// and the stack guard. Kept private so no path can bypass the guard.
#[cfg(test)]
fn compile(n: &Node, out: &mut Vec<u32>) -> bool {
    if !emit(n, out) {
        return false;
    }
    out.push(OP_HALT);
    true
}

/// The first operator (in tree order) that [`emit`] does not handle.
fn first_unsupported(n: &Node) -> Option<&'static str> {
    use Op::*;
    let supported = matches!(
        n.op,
        Const
            | Input
            | Neg
            | Exp
            | Ln
            | Abs
            | Sqrt
            | Sin
            | Cos
            | Tan
            | Sinh
            | Cosh
            | Tanh
            | Logistic
            | Step
            | Gauss
            | Square
            | Cube
            | Floor
            | Ceil
            | Round
            | Sign
            | Erf
            | Erfc
            | Not
            | Add
            | Sub
            | Mul
            | Div
            | Pow
            | Mod
            | Atan2
            | Min
            | Max
            | Equal
            | Less
            | LessOrEqual
            | Greater
            | GreaterOrEqual
            | And
            | Or
            | Xor
            | If
    );
    if !supported {
        return Some(n.op.symbol());
    }
    n.kids.iter().find_map(first_unsupported)
}

/// Count emitted operator words, skipping the operand word after const/var.
fn count_opcodes(words: &[u32]) -> Vec<(String, usize)> {
    let mut counts = std::collections::BTreeMap::new();
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if w == OP_CONST || w == OP_VAR {
            i += 2;
        } else {
            if w != OP_HALT {
                *counts.entry(w).or_insert(0usize) += 1;
            }
            i += 1;
        }
    }
    counts
        .into_iter()
        .map(|(op, n)| (opcode_name(op).to_string(), n))
        .collect()
}

/// #121: rewrite operators the kernel lacks into exact equivalents over the
/// supported instruction set. The identities below are exact for finite values
/// (NaN semantics differ for a few, so search results are always verified by
/// the f64 CPU scorer before they are reported).
fn lower(n: &Node) -> Node {
    use Op::*;
    if matches!(n.op, Const | Input) {
        return n.clone();
    }
    let k: Vec<Node> = n.kids.iter().map(lower).collect();
    match n.op {
        Asinh => {
            // odd form: the naive ln(x + sqrt(x^2+1)) is wrong for negative x
            let x = k[0].clone();
            let sgn = node(Sub, vec![node(Step, vec![x.clone()]), node(Step, vec![node(Neg, vec![x.clone()])])]);
            let mag = node(Ln, vec![node(Add, vec![node(Abs, vec![x.clone()]), node(Sqrt, vec![node(Add, vec![node(Mul, vec![x.clone(), x]), constant(1.0)])])])]);
            node(Mul, vec![sgn, mag])
        }
        Acosh => {
            // factored (x-1)(x+1) survives x -> 1 in f32
            let x = k[0].clone();
            node(Ln, vec![node(Add, vec![x.clone(), node(Sqrt, vec![node(Mul, vec![node(Sub, vec![x.clone(), constant(1.0)]), node(Add, vec![x, constant(1.0)])])])])])
        }
        Atanh => {
            let x = k[0].clone();
            node(Mul, vec![constant(0.5), node(Ln, vec![node(Div, vec![node(Add, vec![constant(1.0), x.clone()]), node(Sub, vec![constant(1.0), x])])])])
        }
        _ => node(n.op, k),
    }
}

/// Emit the opcodes for a subtree. The `HALT` terminator is added only by
/// [`compile`], once, at the end of the whole program.
fn emit(n: &Node, out: &mut Vec<u32>) -> bool {
    use Op::*;
    match n.op {
        Const => {
            out.push(OP_CONST);
            out.push((n.val as f32).to_bits());
            true
        }
        Input => {
            out.push(OP_VAR);
            out.push(n.val as u32);
            true
        }
        Neg => emit(&n.kids[0], out) && {
            out.push(OP_NEG);
            true
        },
        Exp | Ln | Abs | Sqrt | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Step | Gauss
        | Square | Cube | Floor | Ceil | Round | Sign | Erf | Erfc | Not => {
            let op = match n.op {
                Exp => OP_EXP,
                Ln => OP_LOG,
                Abs => OP_ABS,
                Sqrt => OP_SQRT,
                Sin => OP_SIN,
                Cos => OP_COS,
                Tan => OP_TAN,
                Sinh => OP_SINH,
                Cosh => OP_COSH,
                Tanh => OP_TANH,
                Logistic => OP_SIG,
                Step => OP_STEP,
                Gauss => OP_GAUSS,
                Square => OP_SQUARE,
                Cube => OP_CUBE,
                Floor => OP_FLOOR,
                Ceil => OP_CEIL,
                Round => OP_ROUND,
                Sign => OP_SIGN,
                Erf => OP_ERF,
                Erfc => OP_ERFC,
                Not => OP_NOT,
                _ => unreachable!(),
            };
            emit(&n.kids[0], out) && {
                out.push(op);
                true
            }
        }
        Add | Sub | Mul | Div | Pow | Mod | Atan2 | Min | Max | Equal | Less | LessOrEqual
        | Greater | GreaterOrEqual | And | Or | Xor => {
            let op = match n.op {
                Add => OP_ADD,
                Sub => OP_SUB,
                Mul => OP_MUL,
                Div => OP_DIV,
                Pow => OP_POW,
                Mod => OP_MOD,
                Atan2 => OP_ATAN2,
                Min => OP_MIN,
                Max => OP_MAX,
                Equal => OP_EQUAL,
                Less => OP_LESS,
                LessOrEqual => OP_LESS_OR_EQUAL,
                Greater => OP_GREATER,
                GreaterOrEqual => OP_GREATER_OR_EQUAL,
                And => OP_AND,
                Or => OP_OR,
                Xor => OP_XOR,
                _ => unreachable!(),
            };
            emit(&n.kids[0], out) && emit(&n.kids[1], out) && {
                out.push(op);
                true
            }
        }
        If => {
            emit(&n.kids[0], out)
                && emit(&n.kids[1], out)
                && emit(&n.kids[2], out)
                && {
                    out.push(OP_IF);
                    true
                }
        }
        // Extended primitives have no kernel opcode yet; report them as
        // unsupported so the caller can fall back to the CPU evaluator.
        _ => false,
    }
}

#[repr(C)]
struct RawCtx {
    _private: [u8; 0],
}

extern "C" {
    fn rc_create(cols: *const f32, n: i32, v: i32, y: *const f32) -> *mut RawCtx;
    fn rc_destroy(ctx: *mut RawCtx);
    fn rc_set_programs(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
    ) -> i32;
    fn rc_eval(ctx: *mut RawCtx, out: *mut f32) -> f32;
    fn rc_predict(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        out: *mut f32,
    ) -> i32;
    fn rc_device_count() -> i32;
    /* #122: persistent, fixed-capacity, double-buffered program arena. */
    fn rc_pool_reserve(ctx: *mut RawCtx, nprog: i32, width: i32) -> i32;
    fn rc_pool_upload(ctx: *mut RawCtx, which: i32, prog: *const u32, nprog: i32) -> i32;
    fn rc_pool_eval(ctx: *mut RawCtx, which: i32, nprog: i32, width: i32, out: *mut f32) -> f32;
    fn rc_pool_hash(
        ctx: *mut RawCtx,
        which: i32,
        nprog: i32,
        width: i32,
        out: *mut u64,
    ) -> i32;
    fn rc_device_name() -> *const std::os::raw::c_char;
    fn rc_unified() -> i32;
}

/// Probe for a usable CUDA device once (issue #141).
///
/// This is the startup decision the CLI records in the resource summary and in
/// JSON: it asks the CUDA runtime for a device count and, when one answers,
/// reads the device attributes through [`GpuEvaluator::device_name`]. It does
/// not allocate or upload anything, so running it before the dataset is loaded
/// costs no device memory.
///
/// `RCSR_GPU=0` (or `false`/`off`/`no`) makes a CUDA build behave as if no
/// device were present; it is the supported way to exercise the fallback path
/// on a machine that has a GPU.
pub fn probe() -> crate::backend::Probe {
    if matches!(
        std::env::var("RCSR_GPU").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("no")
    ) {
        return crate::backend::Probe::cpu("RCSR_GPU=0 (device use disabled)");
    }
    let count = unsafe { rc_device_count() };
    if count < 0 {
        return crate::backend::Probe::cpu("CUDA device query failed (no driver or runtime?)");
    }
    if count == 0 {
        return crate::backend::Probe::cpu("no CUDA device found");
    }
    let name = GpuEvaluator::device_name();
    if name == "unknown" {
        return crate::backend::Probe::cpu("CUDA device attributes unavailable");
    }
    crate::backend::Probe::gpu(name)
}

/// A compiled population: the postfix programs of many expressions laid out
/// back to back, plus the offsets table that marks where each one starts.
///
/// This is the batch unit the kernel wants: one [`GpuEvaluator::eval_population`]
/// call transfers the whole population and launches one kernel with one block
/// per program, returning one SSE per program. It is deliberately *not* wired
/// into the search loop yet; the search owns its own scheduling.
/// #121: postfix peak stack depth. Leaves push; unary is net zero; an
/// n-ary operator pops n-1. Reject before emission when the kernel's 64-slot
/// stack could overflow.
fn peak_stack(n: &Node) -> usize {
    fn walk(n: &Node, cur: &mut usize, peak: &mut usize) {
        for k in &n.kids {
            walk(k, cur, peak);
        }
        match n.kids.len() {
            0 => {
                *cur += 1;
                if *cur > *peak { *peak = *cur; }
            }
            1 => {}
            k => { *cur -= k - 1; }
        }
    }
    let (mut cur, mut peak) = (0, 0);
    walk(n, &mut cur, &mut peak);
    peak
}

#[derive(Debug, Default, Clone)]
pub struct Population {
    prog: Vec<u32>,
    offsets: Vec<u32>,
    unsupported: usize,
    /// #131: operator occurrences, used for the per-generation `--json` report.
    ops: std::collections::BTreeMap<String, usize>,
}

impl Population {
    /// An empty population. The offsets table always starts with a zero.
    pub fn new() -> Self {
        Population {
            prog: Vec::new(),
            offsets: vec![0],
            unsupported: 0,
            ops: std::collections::BTreeMap::new(),
        }
    }

    /// Compile and append one expression. Returns `false` (and counts the tree
    /// in [`Population::unsupported`]) when the expression uses an operator the
    /// kernel does not implement; the population itself stays valid.
    pub fn add(&mut self, tree: &Node) -> bool {
        let compiled = match compile_program(tree) {
            Ok(c) => c,
            Err(_) => {
                self.unsupported += 1;
                return false;
            }
        };
        for (name, n) in compiled.opcodes {
            *self.ops.entry(name).or_default() += n;
        }
        self.prog.extend_from_slice(&compiled.words);
        self.offsets.push(self.prog.len() as u32);
        true
    }

    /// Build a population from a slice/iterator of trees, skipping unsupported
    /// ones.
    pub fn from_trees<'a, I>(trees: I) -> Self
    where
        I: IntoIterator<Item = &'a Node>,
    {
        let mut pop = Population::new();
        for tree in trees {
            pop.add(tree);
        }
        pop
    }

    /// Number of compiled programs.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Trees that could not be compiled because of an unsupported operator.
    pub fn unsupported(&self) -> usize {
        self.unsupported
    }

    /// The raw `(program bytes, offsets)` pair, e.g. for [`GpuEvaluator::eval_batch`].
    pub fn programs(&self) -> (&[u32], &[u32]) {
        (&self.prog, &self.offsets)
    }

    /// #131: `(opcode name, occurrences)` for this population, sorted by opcode.
    /// Const, var and halt are excluded; this is the "which opcodes were used"
    /// part of the per-generation GPU report.
    pub fn opcode_usage(&self) -> Vec<(String, usize)> {
        self.ops
            .iter()
            .map(|(name, &n)| (name.clone(), n))
            .collect()
    }
}

fn opcode_name(op: u32) -> &'static str {
    OPCODE_TABLE
        .iter()
        .find(|(_, value)| *value == op)
        .map(|(name, _)| *name)
        .unwrap_or("OP_UNKNOWN")
}

/// A resident GPU evaluator: the dataset is uploaded once, then many batches of
/// candidates can be scored without further host↔device traffic for the data.
pub struct GpuEvaluator {
    ctx: *mut RawCtx,
    /// Rows in the resident dataset, so a raw prediction buffer can be sized.
    n_rows: usize,
    /// #122: capacity and width of the persistent program arena, or `0` when
    /// it has not been reserved yet.
    pool_cap: usize,
    pool_width: usize,
    /// Which of the two arena buffers the last upload/evaluation used.
    pool_last: usize,
}

impl GpuEvaluator {
    /// Upload a dataset. `cols` is column-major: `cols[var * n_rows + row]`.
    pub fn new(cols: &[f32], n_rows: usize, n_vars: usize, y: &[f32]) -> Option<Self> {
        let ctx = unsafe { rc_create(cols.as_ptr(), n_rows as i32, n_vars as i32, y.as_ptr()) };
        if ctx.is_null() {
            None
        } else {
            Some(GpuEvaluator {
                ctx,
                n_rows,
                pool_cap: 0,
                pool_width: 0,
                pool_last: 0,
            })
        }
    }

    pub fn device_name() -> String {
        unsafe {
            let p = rc_device_name();
            if p.is_null() {
                return "unknown".to_string();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    /// Whether this build allocates its buffers with `cudaMallocManaged`
    /// (unified memory) rather than `cudaMalloc`.
    ///
    /// Unified memory is selected at build time: `RCSR_GPU_UNIFIED=1`, or by
    /// default for `RCSR_CUDA_ARCH=sm_121`.
    pub fn unified_memory() -> bool {
        unsafe { rc_unified() != 0 }
    }

    /// #131 fixtures: raw per-row predictions for a packed batch, `P × N`
    /// row-major. This is the test-only observation path: unlike
    /// [`Self::eval_batch`] it does not collapse a program to one SSE and it
    /// does not apply the hard-failure magnitude limit, so a fixture can see
    /// `NaN`, infinities and signed zeros exactly as the kernel produced them.
    ///
    /// Returns `None` on any CUDA failure.
    pub fn predictions(&mut self, prog: &[u32], offsets: &[u32]) -> Option<Vec<f32>> {
        let nprog = offsets.len().saturating_sub(1);
        if nprog == 0 {
            return Some(Vec::new());
        }
        let total = nprog.checked_mul(self.n_rows)?;
        let mut out = vec![0.0f32; total];
        let rc = unsafe {
            rc_predict(
                self.ctx,
                prog.as_ptr(),
                offsets.as_ptr(),
                nprog as i32,
                out.as_mut_ptr(),
            )
        };
        if rc == 0 {
            Some(out)
        } else {
            None
        }
    }

    /// Score a batch of compiled programs. Returns one sum-of-squared-errors per
    /// program, plus the kernel time in milliseconds.
    ///
    /// A negative time means the upload or the kernel failed; the output must
    /// then be ignored. The search uses this marker to fall back to the CPU.
    pub fn eval_batch(&mut self, prog: &[u32], offsets: &[u32]) -> (Vec<f32>, f32) {
        let nprog = offsets.len().saturating_sub(1);
        unsafe {
            if rc_set_programs(self.ctx, prog.as_ptr(), offsets.as_ptr(), nprog as i32) != 0 {
                return (Vec::new(), -1.0);
            }
            let mut out = vec![0.0f32; nprog];
            let ms = rc_eval(self.ctx, out.as_mut_ptr());
            (out, ms)
        }
    }

    /// #122: evaluate a fixed-capacity [`GenomePool`] through the persistent
    /// device arena. The arena is reserved once (and only grown if a later
    /// generation is larger); uploads alternate between its two buffers and no
    /// `cudaMalloc`/`cudaFree` happens inside the loop.
    pub fn eval_pool(&mut self, pool: &crate::genome::GenomePool) -> (Vec<f32>, f32) {
        let n = pool.len();
        if n == 0 {
            return (Vec::new(), 0.0);
        }
        let width = pool.width();
        unsafe {
            if self.pool_cap < pool.capacity() || self.pool_width != width {
                if rc_pool_reserve(self.ctx, pool.capacity() as i32, width as i32) != 0 {
                    return (Vec::new(), -1.0);
                }
                self.pool_cap = pool.capacity();
                self.pool_width = width;
            }
            let which = (self.pool_last ^ 1) as i32;
            if rc_pool_upload(self.ctx, which, pool.active().as_ptr(), n as i32) != 0 {
                return (Vec::new(), -1.0);
            }
            let mut out = vec![0.0f32; n];
            let ms = rc_pool_eval(self.ctx, which, n as i32, width as i32, out.as_mut_ptr());
            self.pool_last = which as usize;
            (out, ms)
        }
    }

    /// #122: canonical postfix hashes of the last [`Self::eval_pool`] batch,
    /// computed on the device with the same FNV-1a mix as
    /// [`crate::genome::postfix_hash`]. Used to check the two implementations
    /// agree and, in the plan, to compact a batch without a host hash pass.
    pub fn pool_hashes(&mut self, n: usize) -> Vec<u64> {
        if n == 0 || self.pool_width == 0 {
            return Vec::new();
        }
        unsafe {
            let mut out = vec![0u64; n];
            if rc_pool_hash(
                self.ctx,
                self.pool_last as i32,
                n as i32,
                self.pool_width as i32,
                out.as_mut_ptr(),
            ) != 0
            {
                return Vec::new();
            }
            out
        }
    }

    /// Score a whole [`Population`] in one call: one program upload and one
    /// kernel launch for every expression in the population, returning one SSE
    /// per program.
    ///
    /// This is the convenience the search will use when it batches a whole
    /// generation; it does not change [`Self::eval_batch`], which remains the
    /// primitive.
    pub fn eval_population(&mut self, population: &Population) -> (Vec<f32>, f32) {
        let (prog, offsets) = population.programs();
        self.eval_batch(prog, offsets)
    }
}

impl Drop for GpuEvaluator {
    fn drop(&mut self) {
        unsafe { rc_destroy(self.ctx) };
    }
}

/// Reference CPU scorer with the same semantics (sum of squared errors over the
/// rows), used to validate the GPU results.
pub fn cpu_sse(tree: &Node, cols: &[f32], n_rows: usize, n_vars: usize, y: &[f32]) -> f64 {
    let mut s = 0.0f64;
    let mut row = vec![0.0f64; n_vars];
    for i in 0..n_rows {
        for (v, r) in row.iter_mut().enumerate() {
            *r = cols[v * n_rows + i] as f64;
        }
        let pred = crate::expr::eval(tree, &row);
        if !pred.is_finite() || pred.abs() > 1e30 {
            // Match the kernel's shared magnitude limit for hard failures.
            s += 1e30;
            continue;
        }
        let r = pred - y[i] as f64;
        s += r * r;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    /// Small deterministic dataset plus a handful of benchmarkable expressions.
    fn fixture() -> (Vec<f32>, usize, usize, Vec<f32>, Vec<Node>) {
        let n = 128;
        let nv = 2;
        let mut cols = vec![0.0f32; nv * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let x0 = (i as f32 / n as f32) * 2.0 - 1.0;
            let x1 = ((i * 7 % 31) as f32 / 31.0) - 0.5;
            cols[i] = x0;
            cols[n + i] = x1;
            y[i] = x0.sin() + 0.25 * x1 * x1;
        }
        let trees = vec![
            node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![input(0), constant(2.0)]),
                    constant(1.0),
                ],
            ),
            node(Op::Sin, vec![input(0)]),
            node(
                Op::Add,
                vec![
                    node(Op::Square, vec![input(0)]),
                    node(Op::Mul, vec![input(1), constant(0.5)]),
                ],
            ),
            node(Op::Div, vec![constant(1.0), node(Op::Add, vec![input(0), constant(3.0)])]),
        ];
        (cols, n, nv, y, trees)
    }

    fn evaluator_or_skip(cols: &[f32], n: usize, nv: usize, y: &[f32]) -> Option<GpuEvaluator> {
        match GpuEvaluator::new(cols, n, nv, y) {
            Some(ev) => Some(ev),
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                None
            }
        }
    }

    #[test]
    fn population_skips_unsupported_trees() {
        let (_cols, _n, _nv, _y, trees) = fixture();
        let mut pop = Population::new();
        assert!(pop.is_empty());
        assert!(pop.add(&trees[0]));
        // #131: `if` is native; a constraint has no kernel opcode.
        assert!(pop.add(&node(
            Op::If,
            vec![input(0), constant(1.0), constant(2.0)]
        )));
        assert!(!pop.add(&node(Op::Require, vec![input(0)])));
        assert!(pop.add(&trees[1]));
        assert_eq!(pop.len(), 3);
        assert_eq!(pop.unsupported(), 1);
        let (prog, offsets) = pop.programs();
        assert_eq!(offsets.len(), pop.len() + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap() as usize, prog.len());
        assert!(offsets.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn fixed_slot_pool_matches_the_packed_batch_and_hashes() {
        use crate::genome::{postfix_hash, GenomeMeta, GenomePool};
        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };

        let width = 16;
        let mut pool = GenomePool::new(trees.len(), width);
        let mut expected_hash = Vec::new();
        for tree in &trees {
            let mut words = Vec::new();
            assert!(compile(tree, &mut words));
            expected_hash.push(postfix_hash(&words));
            assert!(pool.push(&words, GenomeMeta::default()));
        }
        let (slot_sse, ms) = ev.eval_pool(&pool);
        assert!(ms >= 0.0, "the fixed-slot kernel ran");
        assert_eq!(slot_sse.len(), trees.len());

        // The hashes are canonical: the device and host agree bit for bit.
        let device_hash = ev.pool_hashes(pool.len());
        assert_eq!(device_hash, expected_hash);

        // The fixed-slot kernel is the same evaluator: batch == one at a time.
        let pop = Population::from_trees(&trees);
        let (packed, _) = ev.eval_population(&pop);
        for i in 0..trees.len() {
            let tol = 1e-6 * (1.0 + packed[i].abs());
            assert!(
                (slot_sse[i] - packed[i]).abs() <= tol,
                "slot {}: {:.9} vs packed {:.9}",
                i,
                slot_sse[i],
                packed[i]
            );
        }
    }

    #[test]
    fn whole_population_matches_one_at_a_time() {
        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };

        let pop = Population::from_trees(&trees);
        assert_eq!(pop.len(), trees.len());
        let (batch, _) = ev.eval_population(&pop);
        assert_eq!(batch.len(), trees.len());

        for (i, tree) in trees.iter().enumerate() {
            let mut one = Vec::new();
            assert!(compile(tree, &mut one));
            let offsets = [0u32, one.len() as u32];
            let (single, _) = ev.eval_batch(&one, &offsets);
            assert_eq!(single.len(), 1);
            let tol = 1e-6 * (1.0 + single[0].abs());
            assert!(
                (batch[i] - single[0]).abs() <= tol,
                "program {}: batch SSE {} vs single SSE {}",
                i,
                batch[i],
                single[0]
            );
        }
    }

    #[test]
    fn gpu_batch_matches_cpu_reference() {
        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };

        let pop = Population::from_trees(&trees);
        let (gpu, _) = ev.eval_population(&pop);
        for (i, tree) in trees.iter().enumerate() {
            let cpu = cpu_sse(tree, &cols, n, nv, &y);
            let rel = (gpu[i] as f64 - cpu).abs() / (1.0 + cpu.abs());
            assert!(
                rel < 1e-4,
                "program {}: GPU SSE {} vs CPU SSE {} (relative {:.2e})",
                i,
                gpu[i],
                cpu,
                rel
            );
        }
    }

    /// #131: the Rust opcode constants must match the CUDA enum exactly.
    #[test]
    fn cuda_opcodes_match_rust_constants() {
        let src = include_str!("cuda/eval.cu");
        let mut found = std::collections::BTreeMap::new();
        for line in src.lines() {
            let line = line.trim();
            if !line.starts_with("OP_") {
                continue;
            }
            let Some((name, rest)) = line.split_once('=') else { continue };
            let name = name.trim();
            let value: u32 = rest
                .split(|c: char| !c.is_ascii_digit())
                .find(|s| !s.is_empty())
                .expect("opcode value")
                .parse()
                .expect("opcode number");
            if found.insert(name.to_string(), value).is_some() {
                panic!("duplicate {name} in eval.cu");
            }
        }
        let mut ours = std::collections::BTreeMap::new();
        for &(name, value) in OPCODE_TABLE {
            ours.insert(name.to_string(), value);
        }
        assert_eq!(found, ours, "gpu.rs and eval.cu opcode enums diverged");
    }

    /// #131: floor/ceil/round, sign, erf/erfc, min/max and if on special
    /// values (`NaN`, `±inf`, `±0`, ties, negatives).
    #[test]
    fn native_exact_ops_match_cpu_on_special_values() {
        let a = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            2.5,
            -2.5,
            3.5,
            -3.5,
            7.0,
            -7.0,
            1.0,
            -1.0,
            1e10,
            -1e10,
            0.5,
            -0.5,
            3.9196,
            9.19,
            10.05,
            8_388_608.0,
            -8_388_608.0,
        ];
        let b = [
            3.0,
            -3.0,
            3.0,
            3.0,
            -3.0,
            2.5,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.7,
            -0.7,
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            1.0,
            -1.0,
            2.0,
            3.0,
            -3.0,
        ];
        let n = a.len();
        let mut cols = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i] as f32;
            cols[n + i] = b[i] as f32;
            y[i] = 0.0;
        }
        let trees = vec![
            node(Op::Floor, vec![input(0)]),
            node(Op::Ceil, vec![input(0)]),
            node(Op::Round, vec![input(0)]),
            node(Op::Sign, vec![input(0)]),
            node(Op::Erf, vec![input(0)]),
            node(Op::Erfc, vec![input(0)]),
            node(Op::Min, vec![input(0), input(1)]),
            node(Op::Max, vec![input(0), input(1)]),
            node(Op::Min, vec![input(1), input(0)]),
            node(Op::Max, vec![input(1), input(0)]),
            node(
                Op::If,
                vec![input(0), input(1), node(Op::Neg, vec![input(1)])],
            ),
            node(Op::If, vec![input(0), constant(1.0), constant(2.0)]),
        ];
        assert_trees_match_cpu(&cols, n, 2, &y, &trees);
    }

    /// #131: comparisons and logical ops propagate NaN like `expr.rs`, and
    /// treat only strictly positive values as true.
    #[test]
    fn native_comparisons_and_logicals_match_cpu() {
        let a = [
            f64::NAN,
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.5,
            -2.5,
        ];
        let b = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            f64::NAN,
            -2.5,
        ];
        let n = a.len();
        let mut cols = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i] as f32;
            cols[n + i] = b[i] as f32;
            y[i] = 0.0;
        }
        let mut trees = Vec::new();
        for op in [
            Op::Equal,
            Op::Less,
            Op::LessOrEqual,
            Op::Greater,
            Op::GreaterOrEqual,
            Op::And,
            Op::Or,
            Op::Xor,
        ] {
            trees.push(node(op, vec![input(0), input(1)]));
        }
        trees.push(node(Op::Not, vec![input(0)]));
        assert_trees_match_cpu(&cols, n, 2, &y, &trees);
    }

    /// #131: `mod` is the floored host formula, including negative divisors.
    #[test]
    fn native_mod_matches_floored_semantics() {
        let a = [
            -8.0, -7.0, -1.0, 0.0, 1.0, 7.0, 8.0, 8.0, -8.0, 1e6, -1e6, f64::NAN,
            f64::INFINITY, 8_388_608.0,
        ];
        let b = [
            2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, -2.5, -2.5, 7.0, 7.0, 1.0, 1.0, 0.7,
        ];
        let n = a.len();
        let mut cols = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i] as f32;
            cols[n + i] = b[i] as f32;
            y[i] = 0.0;
        }
        let trees = vec![
            node(Op::Mod, vec![input(0), input(1)]),
            node(Op::Mod, vec![input(1), input(0)]),
        ];
        assert_trees_match_cpu(&cols, n, 2, &y, &trees);
    }

    /// #131: atan2 on signed zeros, the axes, infinities and NaN.
    #[test]
    fn native_atan2_matches_on_axes_and_signed_zero() {
        let yv = [
            0.0,
            -0.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0,
            f64::NAN,
            0.0,
            -0.0,
        ];
        let xv = [
            0.0,
            0.0,
            -0.0,
            -0.0,
            0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            1.0,
            -1.0,
            -1.0,
        ];
        let n = yv.len();
        let mut cols = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = yv[i] as f32;
            cols[n + i] = xv[i] as f32;
            y[i] = 0.0;
        }
        let trees = vec![node(Op::Atan2, vec![input(0), input(1)])];
        assert_trees_match_cpu(&cols, n, 2, &y, &trees);
    }

    /// #131: adding opcodes must not renumber existing programs.
    #[test]
    fn existing_program_bytes_are_unchanged() {
        let t = node(
            Op::Add,
            vec![node(Op::Sin, vec![input(0)]), constant(1.0)],
        );
        let mut buf = Vec::new();
        assert!(compile(&t, &mut buf));
        assert_eq!(
            buf,
            vec![
                OP_VAR,
                0,
                OP_SIN,
                OP_CONST,
                1.0f32.to_bits(),
                OP_ADD,
                OP_HALT
            ]
        );
    }

    /// #131: the per-generation report counts the opcodes that were emitted.
    #[test]
    fn opcode_usage_counts_native_ops() {
        let t = node(
            Op::Add,
            vec![
                node(Op::Floor, vec![input(0)]),
                node(Op::Min, vec![input(0), input(1)]),
            ],
        );
        let mut pop = Population::new();
        assert!(pop.add(&t));
        let usage = pop.opcode_usage();
        assert_eq!(
            usage,
            vec![
                ("OP_ADD".to_string(), 1),
                ("OP_FLOOR".to_string(), 1),
                ("OP_MIN".to_string(), 1),
            ]
        );
    }

    /// #131: `compile_program` is the guarded entry point; an unsupported
    /// operator is named, and a program deeper than the kernel stack is
    /// rejected on the host.
    #[test]
    fn guarded_compile_reports_unsupported_and_stack_overflow() {
        assert_eq!(
            compile_program(&node(Op::Require, vec![input(0)])).unwrap_err(),
            CompileError::Unsupported("require")
        );
        let mut deep = input(0);
        for _ in 0..60 {
            deep = node(Op::Add, vec![input(1), deep]);
        }
        assert!(matches!(
            compile_program(&deep),
            Err(CompileError::StackOverflow { .. })
        ));
        // `asinh` is lowered, not rejected, so both paths agree.
        let lowered = compile_program(&node(Op::Asinh, vec![input(0)])).unwrap();
        assert!(lowered.opcodes().iter().any(|(n, _)| n.as_str() == "OP_LOG"));
        assert!(lowered.opcodes().iter().all(|(n, _)| n.as_str() != "OP_ASINH"));
    }

    fn ulp_key(x: f32) -> i64 {
        let b = x.to_bits();
        if b & 0x8000_0000 != 0 {
            -((b & 0x7fff_ffff) as i64)
        } else {
            b as i64
        }
    }

    fn ulp_diff(a: f32, b: f32) -> i64 {
        (ulp_key(a) - ulp_key(b)).abs()
    }

    /// Raw per-row predictions for a batch, `P × N` row-major.
    fn raw_predictions(
        cols: &[f32],
        n: usize,
        nv: usize,
        y: &[f32],
        trees: &[Node],
    ) -> Option<Vec<f32>> {
        let mut ev = evaluator_or_skip(cols, n, nv, y)?;
        let pop = Population::from_trees(trees.iter());
        assert_eq!(
            pop.len(),
            trees.len(),
            "every fixture tree must compile: {} unsupported",
            pop.unsupported()
        );
        let (prog, offsets) = pop.programs();
        let preds = ev
            .predictions(prog, offsets)
            .expect("raw prediction readback failed");
        assert_eq!(preds.len(), trees.len() * n);
        Some(preds)
    }

    fn unary_rows() -> Vec<f32> {
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            0.5,
            -0.5,
            2.5,
            -2.5,
            3.5,
            -3.5,
            0.1,
            -0.1,
            1.0,
            -1.0,
            3.9196,
            9.19,
            10.05,
            123.456,
            -123.456,
            8_388_608.0,
            -8_388_608.0,
            1e10,
            -1e10,
            1e-30,
            -1e-30,
        ]
    }

    /// #131 acceptance: every opcode is checked against the f64 CPU reference
    /// row by row, with an explicit per-op upper bound in f32 ULP.
    ///
    /// * exact (`0` ULP): `floor`, `ceil`, `round`, `sign`, `min`, `max`, `if`,
    ///   comparisons, logicals, `not` — these are bit-exact on the f32 grid;
    /// * `mod` (1 ULP): the host formula `a - b*floor(a/b)` evaluated with an
    ///   f32 divide and fused multiply-add;
    /// * `atan2` (3 ULP), `erf` (2 ULP), `erfc` (4 ULP): libdevice bounds.
    #[test]
    fn per_opcode_ulp_bounds_against_the_f64_reference() {
        let a = unary_rows();
        let n = a.len();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| match i % 8 {
                0 => 3.0,
                1 => -3.0,
                2 => 2.5,
                3 => -2.5,
                4 => 0.7,
                5 => -0.7,
                6 => 1.0,
                _ => {
                    if v.is_nan() {
                        1.0
                    } else {
                        -1.0
                    }
                }
            })
            .collect();
        let mut cols = vec![0.0f32; 2 * n];
        let y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i];
            cols[n + i] = b[i];
        }
        let x = || input(0);
        let z = || input(1);
        let neg = |k: Node| node(Op::Neg, vec![k]);
        let cases: Vec<(Node, &str, i64)> = vec![
            (node(Op::Floor, vec![x()]), "floor", 0),
            (node(Op::Ceil, vec![x()]), "ceil", 0),
            (node(Op::Round, vec![x()]), "round", 0),
            (node(Op::Sign, vec![x()]), "sign", 0),
            (node(Op::Not, vec![x()]), "not", 0),
            (node(Op::Erf, vec![x()]), "erf", 2),
            (node(Op::Erfc, vec![x()]), "erfc", 4),
            (node(Op::Min, vec![x(), z()]), "min", 0),
            (node(Op::Max, vec![x(), z()]), "max", 0),
            (node(Op::Min, vec![z(), x()]), "min(rev)", 0),
            (node(Op::Max, vec![z(), x()]), "max(rev)", 0),
            (node(Op::Mod, vec![x(), z()]), "mod", 1),
            (node(Op::Mod, vec![z(), x()]), "mod(rev)", 1),
            (node(Op::Atan2, vec![x(), z()]), "atan2", 3),
            (node(Op::Atan2, vec![z(), x()]), "atan2(rev)", 3),
            (node(Op::Equal, vec![x(), z()]), "equal", 0),
            (node(Op::Less, vec![x(), z()]), "less", 0),
            (node(Op::LessOrEqual, vec![x(), z()]), "less_or_equal", 0),
            (node(Op::Greater, vec![x(), z()]), "greater", 0),
            (node(Op::GreaterOrEqual, vec![x(), z()]), "greater_or_equal", 0),
            (node(Op::And, vec![x(), z()]), "and", 0),
            (node(Op::Or, vec![x(), z()]), "or", 0),
            (node(Op::Xor, vec![x(), z()]), "xor", 0),
            (node(Op::If, vec![x(), z(), neg(z())]), "if", 0),
            (
                node(Op::If, vec![x(), constant(1.0), z()]),
                "if(else-arm)",
                0,
            ),
        ];
        let trees: Vec<Node> = cases.iter().map(|(t, _, _)| t.clone()).collect();
        let Some(preds) = raw_predictions(&cols, n, 2, &y, &trees) else {
            return;
        };
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![a[i] as f64, b[i] as f64])
            .collect();
        let mut worst = vec![0i64; cases.len()];
        for (k, (tree, name, bound)) in cases.iter().enumerate() {
            for i in 0..n {
                let got = preds[k * n + i];
                let want64 = crate::expr::eval(tree, &rows[i]);
                if want64.is_nan() {
                    assert!(got.is_nan(), "{name} row {i}: expected NaN, got {got}");
                    continue;
                }
                let want = want64 as f32;
                // Documented f32 limit for `mod`: above |a/b| = 2^25 the f32
                // quotient cannot carry the fractional part, so the remainder
                // is not the f64 remainder. It must still stay inside the
                // divisor's range (floored semantics) and is never used for a
                // reported number.
                if name.starts_with("mod") {
                    let q = (a[i] as f64 / b[i] as f64).abs();
                    if q > 33_554_432.0 {
                        let g = got as f64;
                        let bv = b[i] as f64;
                        assert!(!got.is_nan(), "{name} row {i}: lost the remainder");
                        if bv > 0.0 {
                            assert!(g >= 0.0 && g <= bv, "{name} row {i}: {g} outside [0,{bv}]");
                        } else {
                            assert!(g <= 0.0 && g >= bv, "{name} row {i}: {g} outside [{bv},0]");
                        }
                        continue;
                    }
                }
                let d = ulp_diff(got, want);
                worst[k] = worst[k].max(d);
                assert!(
                    d <= *bound,
                    "{name} row {i} (a={}, b={}): got {got:?} ({:#010x}), \
                     reference {want:?} ({:#010x}), {d} ULP > {bound}",
                    a[i],
                    b[i],
                    got.to_bits(),
                    want.to_bits()
                );
                if want64.is_infinite() {
                    assert_eq!(got, want, "{name} row {i}: infinity sign must match");
                }
            }
        }
    }

    /// #131 acceptance: the special-value cases named in the issue, asserted
    /// explicitly rather than through an aggregate error.
    #[test]
    fn native_special_values_are_exact() {
        let a = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            2.5,
            -2.5,
            3.5,
            -3.5,
            8_388_608.0,
            -8_388_608.0,
            3.9196,
            9.19,
            10.05,
        ];
        let n = a.len();
        let b: Vec<f32> = a
            .iter()
            .map(|&v| {
                if v.is_nan() || v == f32::INFINITY {
                    f32::NAN
                } else {
                    -3.0
                }
            })
            .collect();
        let mut cols = vec![0.0f32; 2 * n];
        let y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i];
            cols[n + i] = b[i];
        }
        let x = || input(0);
        let z = || input(1);
        let trees = vec![
            node(Op::Floor, vec![x()]),
            node(Op::Ceil, vec![x()]),
            node(Op::Round, vec![x()]),
            node(Op::Sign, vec![x()]),
            node(Op::Erf, vec![x()]),
            node(Op::Erfc, vec![x()]),
            node(Op::Min, vec![x(), z()]),
            node(Op::Max, vec![x(), z()]),
            node(Op::Atan2, vec![x(), z()]),
            node(Op::Mod, vec![x(), z()]),
        ];
        let Some(preds) = raw_predictions(&cols, n, 2, &y, &trees) else {
            return;
        };
        let row = |k: usize, i: usize| preds[k * n + i];
        // Signed zeros: floor/ceil/round preserve them; sign/erf normalise to
        // +0 like expr.rs.
        assert_eq!(row(0, 4).to_bits(), (-0.0f32).to_bits(), "floor(-0)");
        assert_eq!(row(1, 4).to_bits(), (-0.0f32).to_bits(), "ceil(-0)");
        assert_eq!(row(2, 4).to_bits(), (-0.0f32).to_bits(), "round(-0)");
        assert_eq!(row(3, 4).to_bits(), 0.0f32.to_bits(), "sign(-0)");
        assert_eq!(row(4, 4).to_bits(), 0.0f32.to_bits(), "erf(-0)");
        // Round is half-away-from-zero, not ties-to-even.
        assert_eq!(row(2, 5), 3.0, "round(2.5)");
        assert_eq!(row(2, 6), -3.0, "round(-2.5)");
        assert_eq!(row(2, 7), 4.0, "round(3.5)");
        assert_eq!(row(2, 8), -4.0, "round(-3.5)");
        // floor/ceil on infinities and NaN.
        assert_eq!(row(0, 1), f32::INFINITY);
        assert_eq!(row(0, 2), f32::NEG_INFINITY);
        assert!(row(0, 0).is_nan(), "floor(NaN)");
        assert_eq!(row(1, 1), f32::INFINITY);
        assert!(row(1, 0).is_nan(), "ceil(NaN)");
        // Large values are already integral: no range bound kicks in.
        assert_eq!(row(0, 9), 8_388_608.0);
        assert_eq!(row(0, 10), -8_388_608.0);
        // Min/max propagate NaN in either operand position.
        assert!(row(6, 0).is_nan(), "min(NaN, NaN)");
        assert!(row(6, 1).is_nan(), "min(inf, NaN)");
        assert!(row(7, 0).is_nan(), "max(NaN, NaN)");
        assert!(row(7, 1).is_nan(), "max(inf, NaN)");
        // erf saturates to 1 on the f32 grid; erfc's tail underflows.
        assert_eq!(row(4, 11), 1.0, "erf(3.9196)");
        assert!(
            row(5, 11) > 0.0 && row(5, 11) < 1e-6,
            "erfc(3.9196) is a small positive tail, got {}",
            row(5, 11)
        );
        // At x = 9.19 the f32 result is at the subnormal boundary: accept zero
        // or a tiny positive value (documented underflow, not a relative-ULP
        // claim), and require exact zero once the tail is below f32 range.
        assert!(
            row(5, 12) >= 0.0 && row(5, 12) <= 1e-36,
            "erfc(9.19) underflowed unexpectedly: {}",
            row(5, 12)
        );
        // At 10.05 the true tail is below the f32 range: libdevice returns the
        // smallest subnormal (1.4e-45) or zero. Both are "zero" for f32 here.
        assert!(
            row(5, 13) >= 0.0 && row(5, 13) <= f32::from_bits(2),
            "erfc(10.05) is below f32 range: {}",
            row(5, 13)
        );
        // atan2 on the negative axis: pi, not 0.
        assert!((row(8, 3) - std::f32::consts::PI).abs() < 1e-6);
        assert!((row(8, 4) + std::f32::consts::PI).abs() < 1e-6);
        // mod is floored. b = -3, so 2.5 -> 2.5 - (-3)*floor(-0.833) = -0.5.
        assert_eq!(row(9, 5), -0.5, "mod(2.5, -3)");
    }

    /// #131: the fixed-slot pool used by the search must run the same #131
    /// opcodes as the packed batch. Before this test the pool kernel silently
    /// returned 0 for every hard op.
    #[test]
    fn fixed_slot_pool_runs_the_native_hard_opcodes() {
        use crate::genome::{GenomeMeta, GenomePool};
        let a = [f64::NAN, 0.0, -0.0, 1.0, -1.0, 2.5, -2.5, 3.5, -3.5];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 2.5, -2.5, 1.0, -1.0];
        let n = a.len();
        let mut cols = vec![0.0f32; 2 * n];
        let y = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = a[i] as f32;
            cols[n + i] = b[i] as f32;
        }
        let mut ev = match evaluator_or_skip(&cols, n, 2, &y) {
            Some(ev) => ev,
            None => return,
        };
        let trees = vec![
            node(Op::Floor, vec![input(0)]),
            node(Op::Round, vec![input(0)]),
            node(Op::Erf, vec![input(0)]),
            node(Op::Atan2, vec![input(0), input(1)]),
            node(Op::Mod, vec![input(0), input(1)]),
            node(Op::Min, vec![input(0), input(1)]),
            node(Op::If, vec![input(0), input(1), input(0)]),
        ];
        let width = 16;
        let mut pool = GenomePool::new(trees.len(), width);
        for tree in &trees {
            let c = compile_program(tree).expect("hard op compiles");
            assert!(pool.push(c.words(), GenomeMeta::default()));
        }
        let (slot_sse, ms) = ev.eval_pool(&pool);
        assert!(ms >= 0.0, "the fixed-slot kernel ran");
        assert_eq!(slot_sse.len(), trees.len());

        let pop = Population::from_trees(trees.iter());
        let (packed, _) = ev.eval_population(&pop);
        for i in 0..trees.len() {
            let tol = 1e-6 * (1.0 + packed[i].abs());
            assert!(
                (slot_sse[i] - packed[i]).abs() <= tol,
                "{}: pool SSE {} vs packed {}",
                trees[i].op.symbol(),
                slot_sse[i],
                packed[i]
            );
            // ... and both agree with the f64 reference, so "0" is not accepted.
            let cpu = cpu_sse(&trees[i], &cols, n, 2, &y);
            let rel = (slot_sse[i] as f64 - cpu).abs() / (1.0 + cpu.abs());
            assert!(
                rel < 1e-4,
                "{}: pool SSE {} vs CPU {} (relative {:.2e})",
                trees[i].op.symbol(),
                slot_sse[i],
                cpu,
                rel
            );
        }
    }

    /// Score each tree on the GPU and compare the whole-population SSE with
    /// the f64 CPU reference. Skips on a machine with no usable device.
    fn assert_trees_match_cpu(cols: &[f32], n: usize, nv: usize, y: &[f32], trees: &[Node]) {
        let mut ev = match evaluator_or_skip(cols, n, nv, y) {
            Some(ev) => ev,
            None => return,
        };
        let pop = Population::from_trees(trees.iter());
        assert_eq!(
            pop.len(),
            trees.len(),
            "every #131 opcode must compile: {} unsupported",
            pop.unsupported()
        );
        let (gpu, ms) = ev.eval_population(&pop);
        assert!(ms >= 0.0, "CUDA batch failed");
        for (i, tree) in trees.iter().enumerate() {
            let cpu = cpu_sse(tree, cols, n, nv, y);
            let rel = (gpu[i] as f64 - cpu).abs() / (1.0 + cpu.abs());
            assert!(
                rel < 1e-4,
                "tree {} {:?}: GPU SSE {} vs CPU SSE {} (relative {:.2e})",
                i,
                tree.op,
                gpu[i],
                cpu,
                rel
            );
        }
    }
}


#[cfg(test)]
mod lower_tests {
    use super::*;
    use crate::expr::{eval, input};

    fn close(a: f64, b: f64) {
        if a.is_nan() && b.is_nan() { return; }
        assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())), "{a} vs {b}");
    }

    fn check(t: Node, rows: &[Vec<f64>]) {
        let l = lower(&t);
        let mut buf = Vec::new();
        assert!(compile(&l, &mut buf), "lowered tree must compile");
        for r in rows {
            close(eval(&t, r), eval(&l, r));
        }
    }

    #[test]
    fn lowered_operators_match_the_cpu() {
        let rows = [
            vec![-2.0, -0.5], vec![-0.5, 0.0], vec![0.0, 0.0],
            vec![0.25, 0.75], vec![1.5, 2.0], vec![3.0, -1.0],
        ];
        let x = || input(0);
        let y = || input(1);
        check(node(Op::Min, vec![x(), y()]), &rows);
        check(node(Op::Max, vec![x(), y()]), &rows);
        check(node(Op::If, vec![x(), y(), node(Op::Neg, vec![y()])]), &rows);
        check(node(Op::Sign, vec![x()]), &rows);
        check(node(Op::Less, vec![x(), y()]), &rows);
        check(node(Op::LessOrEqual, vec![x(), y()]), &rows);
        check(node(Op::Greater, vec![x(), y()]), &rows);
        check(node(Op::GreaterOrEqual, vec![x(), y()]), &rows);
        check(node(Op::Equal, vec![x(), y()]), &rows);
        check(node(Op::And, vec![x(), y()]), &rows);
        check(node(Op::Or, vec![x(), y()]), &rows);
        check(node(Op::Xor, vec![x(), y()]), &rows);
        check(node(Op::Not, vec![x()]), &rows);
        check(node(Op::Asinh, vec![x()]), &rows);
        check(node(Op::Atanh, vec![node(Op::Div, vec![x(), constant(4.0)])]), &rows);
    }

    #[test]
    fn acosh_matches_on_its_domain() {
        let t = node(Op::Acosh, vec![input(0)]);
        let l = lower(&t);
        for x in [1.0, 1.5, 2.0, 10.0] {
            close(eval(&t, &[x]), eval(&l, &[x]));
        }
    }
}

#[cfg(test)]
mod lower_extreme_tests {
    use super::*;
    use crate::expr::{constant, eval, input, node, Op};

    #[test]
    fn min_max_do_not_add_operands() {
        let cases = [(1e8, 1.0), (3e38, -3e38), (1.0, 1.0), (-0.0, 0.0)];
        for (a, b) in cases {
            let mn = node(Op::Min, vec![constant(a), constant(b)]);
            let mx = node(Op::Max, vec![constant(a), constant(b)]);
            assert_eq!(eval(&lower(&mn), &[]), a.min(b));
            assert_eq!(eval(&lower(&mx), &[]), a.max(b));
        }
    }

    #[test]
    fn if_is_exact_for_extreme_arms() {
        let t = node(Op::If, vec![input(0), constant(1.0), constant(1e8)]);
        let l = lower(&t);
        for (x, want) in [(1.0, 1.0), (-1.0, 1e8), (0.0, 1e8)] {
            assert_eq!(eval(&l, &[x]), want);
        }
    }

    #[test]
    fn asinh_negative_uses_the_odd_form() {
        let t = node(Op::Asinh, vec![input(0)]);
        let l = lower(&t);
        let a = eval(&t, &[-1e8]);
        let b = eval(&l, &[-1e8]);
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn peak_stack_guard_rejects_too_deep() {
        // right-deep add chain: postfix peak grows with depth
        let mut t = input(0);
        for _ in 0..60 {
            t = node(Op::Add, vec![input(1), t]);
        }
        assert!(peak_stack(&t) > 56);
        let mut pop = Population::new();
        assert!(!pop.add(&t), "a >56-deep program must be rejected before emission");
    }

    #[test]
    fn peak_stack_counts_ternary_pops() {
        // An `if` pushes 3 and pops 2 (net +1). Two ternaries under one add
        // peak at 4; counting a ternary as popping one would report 5.
        let leaf_if = |a: f64, b: f64| {
            node(Op::If, vec![input(0), constant(a), constant(b)])
        };
        let t = node(Op::Add, vec![leaf_if(1.0, 2.0), leaf_if(3.0, 4.0)]);
        assert_eq!(peak_stack(&t), 4);
    }
}
