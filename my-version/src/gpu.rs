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
use crate::gpu_stats::GpuStats;
use crate::metrics::{ScoreObjective, ScoreStatus};

/* Architecture-name parsing and toolkit requirements. The same file is
 * `include!`d by `build.rs`; declaring it here gives it unit tests. */
#[path = "cuda/arch.rs"]
pub mod arch;

/* Opcodes, kept in sync with src/cuda/eval.cu. */
pub const OP_CONST: u32 = 0;
pub const OP_VAR: u32 = 1;
pub const OP_ADD: u32 = 2;
pub const OP_SUB: u32 = 3;
pub const OP_MUL: u32 = 4;
pub const OP_DIV: u32 = 5;
pub const OP_SIN: u32 = 6;
pub const OP_COS: u32 = 7;
pub const OP_TAN: u32 = 8;
pub const OP_EXP: u32 = 9;
pub const OP_LOG: u32 = 10;
pub const OP_SQRT: u32 = 11;
pub const OP_ABS: u32 = 12;
pub const OP_TANH: u32 = 13;
pub const OP_SIG: u32 = 14;
pub const OP_SQUARE: u32 = 15;
pub const OP_CUBE: u32 = 16;
pub const OP_NEG: u32 = 17;
pub const OP_GAUSS: u32 = 18;
pub const OP_STEP: u32 = 19;
pub const OP_POW: u32 = 20;
pub const OP_SINH: u32 = 21;
pub const OP_COSH: u32 = 22;
// #131: native hard operators.
pub const OP_FLOOR: u32 = 23;
pub const OP_CEIL: u32 = 24;
pub const OP_ROUND: u32 = 25;
pub const OP_MOD: u32 = 26;
pub const OP_ATAN2: u32 = 27;
pub const OP_ERF: u32 = 28;
pub const OP_ERFC: u32 = 29;
pub const OP_MIN: u32 = 30;
pub const OP_MAX: u32 = 31;
pub const OP_IF: u32 = 32;
pub const OP_SIGN: u32 = 33;
pub const OP_EQUAL: u32 = 34;
pub const OP_LESS: u32 = 35;
pub const OP_LESS_OR_EQUAL: u32 = 36;
pub const OP_GREATER: u32 = 37;
pub const OP_GREATER_OR_EQUAL: u32 = 38;
pub const OP_AND: u32 = 39;
pub const OP_OR: u32 = 40;
pub const OP_XOR: u32 = 41;
pub const OP_NOT: u32 = 42;
pub const OP_HALT: u32 = 255;

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

/// The kernel's postfix stack size, matching `RC_MAX_STACK` in
/// `src/cuda/eval.cu`. The device guards every push against it; the host keeps
/// [`HOST_MAX_STACK`] slots of headroom so a rejected tree never reaches the
/// device through the normal path.
pub const RC_MAX_STACK: usize = 64;

/// #121: host gate for emitted programs. Lowering can expand a tree, so the
/// static peak stack is checked after [`lower`] and before a single word is
/// appended. The 8-slot margin leaves room for an opcode that pushes an
/// intermediate the pass did not model.
pub const HOST_MAX_STACK: usize = 56;

/// The kernel's operand stack (`RC_MAX_STACK` in `eval.cu`), less the headroom
/// the guard reserves. A postfix program deeper than this is rejected on the
/// host instead of silently overrunning the device stack.
const MAX_PROGRAM_STACK: usize = HOST_MAX_STACK;

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
    lowered: bool,
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

    /// Whether [`lower`] had to rewrite the tree before emission (#121).
    pub fn lowered(&self) -> bool {
        self.lowered
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
    Ok(CompiledProgram {
        words,
        opcodes,
        lowered: lowered != *tree,
    })
}

/// #121 compatibility wrapper over [`compile_program`]: emit into `out` and
/// report whether [`lower`] rewrote the tree. `None` means the tree is
/// unsupported or over-deep and must go to the CPU fallback.
pub fn compile_with_report(n: &Node, out: &mut Vec<u32>) -> Option<bool> {
    match compile_program(n) {
        Ok(c) => {
            out.extend_from_slice(c.words());
            Some(c.lowered)
        }
        Err(_) => None,
    }
}

/// Compile one expression to postfix bytecode. Returns `false` if the tree uses
/// an operator the kernel does not implement even after [`lower`].
pub fn compile(n: &Node, out: &mut Vec<u32>) -> bool {
    compile_with_report(n, out).is_some()
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

/// #122: the kernel opcode and arity of an operator that is emitted directly.
///
/// `Const`/`Input` are terminals, not operators; `Asinh`/`Acosh`/`Atanh` are
/// lowered to exact equivalents by [`compile`], and aggregates, meta nodes,
/// constraints and custom blocks have no kernel opcode. This function returns
/// `None` for all of those so a caller can reject a configuration instead of
/// silently producing a program the kernel cannot evaluate.
pub fn postfix_opcode(op: Op) -> Option<(u32, usize)> {
    use Op::*;
    let (code, arity) = match op {
        Neg => (OP_NEG, 1),
        Exp => (OP_EXP, 1),
        Ln => (OP_LOG, 1),
        Abs => (OP_ABS, 1),
        Sqrt => (OP_SQRT, 1),
        Sin => (OP_SIN, 1),
        Cos => (OP_COS, 1),
        Tan => (OP_TAN, 1),
        Sinh => (OP_SINH, 1),
        Cosh => (OP_COSH, 1),
        Tanh => (OP_TANH, 1),
        Logistic => (OP_SIG, 1),
        Step => (OP_STEP, 1),
        Gauss => (OP_GAUSS, 1),
        Square => (OP_SQUARE, 1),
        Cube => (OP_CUBE, 1),
        Floor => (OP_FLOOR, 1),
        Ceil => (OP_CEIL, 1),
        Round => (OP_ROUND, 1),
        Sign => (OP_SIGN, 1),
        Erf => (OP_ERF, 1),
        Erfc => (OP_ERFC, 1),
        Not => (OP_NOT, 1),
        Add => (OP_ADD, 2),
        Sub => (OP_SUB, 2),
        Mul => (OP_MUL, 2),
        Div => (OP_DIV, 2),
        Pow => (OP_POW, 2),
        Mod => (OP_MOD, 2),
        Atan2 => (OP_ATAN2, 2),
        Min => (OP_MIN, 2),
        Max => (OP_MAX, 2),
        Equal => (OP_EQUAL, 2),
        Less => (OP_LESS, 2),
        LessOrEqual => (OP_LESS_OR_EQUAL, 2),
        Greater => (OP_GREATER, 2),
        GreaterOrEqual => (OP_GREATER_OR_EQUAL, 2),
        And => (OP_AND, 2),
        Or => (OP_OR, 2),
        Xor => (OP_XOR, 2),
        If => (OP_IF, 3),
        _ => return None,
    };
    Some((code, arity))
}

/// #122: the inverse of [`postfix_opcode`], used to rebuild a tree from a
/// device-format postfix slot. Returns `None` for an unknown or terminal word.
pub fn opcode_arity(code: u32) -> Option<usize> {
    Some(match code {
        OP_NEG | OP_EXP | OP_LOG | OP_ABS | OP_SQRT | OP_SIN | OP_COS | OP_TAN | OP_SINH
        | OP_COSH | OP_TANH | OP_SIG | OP_STEP | OP_GAUSS | OP_SQUARE | OP_CUBE | OP_FLOOR
        | OP_CEIL | OP_ROUND | OP_SIGN | OP_ERF | OP_ERFC | OP_NOT => 1,
        OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_POW | OP_MOD | OP_ATAN2 | OP_MIN | OP_MAX
        | OP_EQUAL | OP_LESS | OP_LESS_OR_EQUAL | OP_GREATER | OP_GREATER_OR_EQUAL | OP_AND
        | OP_OR | OP_XOR => 2,
        OP_IF => 3,
        _ => return None,
    })
}

/// The [`Op`] behind a kernel opcode, for [`postfix::decode`] and for tests.
pub fn opcode_op(code: u32) -> Option<Op> {
    use Op::*;
    Some(match code {
        OP_NEG => Neg,
        OP_EXP => Exp,
        OP_LOG => Ln,
        OP_ABS => Abs,
        OP_SQRT => Sqrt,
        OP_SIN => Sin,
        OP_COS => Cos,
        OP_TAN => Tan,
        OP_SINH => Sinh,
        OP_COSH => Cosh,
        OP_TANH => Tanh,
        OP_SIG => Logistic,
        OP_STEP => Step,
        OP_GAUSS => Gauss,
        OP_SQUARE => Square,
        OP_CUBE => Cube,
        OP_FLOOR => Floor,
        OP_CEIL => Ceil,
        OP_ROUND => Round,
        OP_SIGN => Sign,
        OP_ERF => Erf,
        OP_ERFC => Erfc,
        OP_NOT => Not,
        OP_ADD => Add,
        OP_SUB => Sub,
        OP_MUL => Mul,
        OP_DIV => Div,
        OP_POW => Pow,
        OP_MOD => Mod,
        OP_ATAN2 => Atan2,
        OP_MIN => Min,
        OP_MAX => Max,
        OP_EQUAL => Equal,
        OP_LESS => Less,
        OP_LESS_OR_EQUAL => LessOrEqual,
        OP_GREATER => Greater,
        OP_GREATER_OR_EQUAL => GreaterOrEqual,
        OP_AND => And,
        OP_OR => Or,
        OP_XOR => Xor,
        OP_IF => If,
        _ => return None,
    })
}

#[repr(C)]
struct RawCtx {
    _private: [u8; 0],
}

/// #284: kernel span kinds read through `rc_last_span`.
pub(crate) const SPAN_EVAL: i32 = 0;
pub(crate) const SPAN_FIT: i32 = 1;

/// #118: the C-side ledger for one batch, in milliseconds. The host windows are
/// disjoint: allocation, H2D copy, launch, event sync and D2H copy. `kernel_ms`
/// is GPU-event time inside the wall and feeds the duty estimate, not the sum.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawStats {
    kernel_ms: f64,
    launch_ms: f64,
    sync_ms: f64,
    alloc_ms: f64,
    h2d_ms: f64,
    d2h_ms: f64,
    /// C-side envelope from `rc_stats_begin` to the end of the batch. Kept for
    /// auditing against the outer wall.
    #[allow(dead_code)]
    wall_ms: f64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    batch: i32,
    failed: i32,
}

/// #285: host-visible state of the persistent batch arena. Mirrors
/// `RcArenaReport` in `src/cuda/eval.cu` field for field.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RawArenaReport {
    async_copy: i32,
    reason: [std::os::raw::c_char; 96],
    pinned_bytes: u64,
    alloc_calls: u64,
    steady_alloc_calls: u64,
    overlapped_uploads: u64,
    batches: u64,
    fs_alloc_calls: u64,
    fs_growths: u64,
    fs_steady_alloc_calls: u64,
    fs_batches: u64,
    fs_dev_bytes: u64,
    fs_pinned_bytes: u64,
    tile_rows: u64,
    tiles: u64,
    tile_uploads: u64,
    tile_overlapped_uploads: u64,
    tile_pinned_bytes: u64,
    waves: u64,
}

/// `RcAllocReport` in `src/cuda/eval.cu` field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawAllocReport {
    requests: u64,
    max_request_bytes: u64,
    live_bytes: u64,
    peak_live_bytes: u64,
    refused: u64,
    max_request_tag: [u8; 32],
    refused_tag: [u8; 32],
}

/// #293: the device allocation ledger of one evaluator. Every device
/// allocation of the context is counted and tagged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocReport {
    pub requests: u64,
    pub max_request_bytes: u64,
    pub max_request_tag: String,
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub refused: u64,
    pub refused_tag: String,
}

fn tag_text(raw: &[u8]) -> String {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

impl AllocReport {
    /// The observed block of `gpu.memory_plan`.
    pub fn observed(&self) -> crate::memory::DeviceObserved {
        crate::memory::DeviceObserved {
            requests: self.requests,
            max_request_bytes: self.max_request_bytes,
            max_request_tag: self.max_request_tag.clone(),
            live_bytes: self.live_bytes,
            peak_live_bytes: self.peak_live_bytes,
            refused: self.refused,
            refused_tag: self.refused_tag.clone(),
        }
    }
}

/// #285: what the batch arena is doing, for a test, benchmark or report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaReport {
    /// `true` when H2D/D2H run asynchronously through the pinned ring.
    pub async_copy: bool,
    /// The recorded cause when the synchronous fallback is active.
    pub reason: String,
    /// Bytes currently pinned for host staging (hard cap 64 MiB).
    pub pinned_bytes: u64,
    /// Total device/host allocator calls made by the arena.
    pub alloc_calls: u64,
    /// Allocator calls after the first successful batch: the hot-path count.
    pub steady_alloc_calls: u64,
    /// Uploads that were enqueued while the previous kernel was still
    /// outstanding — the event-ordering proof of overlap.
    pub overlapped_uploads: u64,
    /// Batches collected through the arena.
    pub batches: u64,
    /// #334: allocator calls made by the persistent fit and score regions
    /// (device and pinned, allocate and free), cold and steady.
    pub fit_alloc_calls: u64,
    /// #334: how many times the fit/score regions grew to a new high-water mark.
    pub fit_growths: u64,
    /// #334: fit/score allocator calls made after the first successful batch of
    /// the same kind (fit, rank, order statistic or maximum). Zero while the
    /// batch shape stays inside the high-water mark.
    pub fit_steady_alloc_calls: u64,
    /// #334: fit and score batches completed through the persistent regions.
    pub fit_batches: u64,
    /// #334: current size of the persistent device region in bytes.
    pub fit_device_bytes: u64,
    /// #334: current size of the pinned staging region in bytes.
    pub fit_pinned_bytes: u64,
    /// #293: rows per tile, or `0` when the rows are resident.
    pub tile_rows: u64,
    /// #293: tile kernel launches, over all passes and waves.
    pub tiles: u64,
    /// #293: tile uploads issued.
    pub tile_uploads: u64,
    /// #293: uploads enqueued while the previous tile's kernel was still
    /// outstanding.
    pub tile_overlapped_uploads: u64,
    /// #293: bytes of pinned host staging for the tile slots.
    pub tile_pinned_bytes: u64,
    /// #293: candidate waves launched.
    pub waves: u64,
}

impl ArenaReport {
    fn from_raw(raw: &RawArenaReport) -> Self {
        let bytes: Vec<u8> = raw
            .reason
            .iter()
            .map(|&c| c as u8)
            .take_while(|&b| b != 0)
            .collect();
        ArenaReport {
            async_copy: raw.async_copy != 0,
            reason: String::from_utf8_lossy(&bytes).into_owned(),
            pinned_bytes: raw.pinned_bytes,
            alloc_calls: raw.alloc_calls,
            steady_alloc_calls: raw.steady_alloc_calls,
            overlapped_uploads: raw.overlapped_uploads,
            batches: raw.batches,
            fit_alloc_calls: raw.fs_alloc_calls,
            fit_growths: raw.fs_growths,
            fit_steady_alloc_calls: raw.fs_steady_alloc_calls,
            fit_batches: raw.fs_batches,
            fit_device_bytes: raw.fs_dev_bytes,
            fit_pinned_bytes: raw.fs_pinned_bytes,
            tile_rows: raw.tile_rows,
            tiles: raw.tiles,
            tile_uploads: raw.tile_uploads,
            tile_overlapped_uploads: raw.tile_overlapped_uploads,
            tile_pinned_bytes: raw.tile_pinned_bytes,
            waves: raw.waves,
        }
    }
}

extern "C" {
    fn rc_create(cols: *const f32, n: i32, v: i32, y: *const f32, budget: u64) -> *mut RawCtx;
    fn rc_create_tiled(
        cols: *const f32,
        n: i32,
        v: i32,
        y: *const f32,
        tile_rows: i32,
        budget: u64,
        wave_cap: u64,
    ) -> *mut RawCtx;
    fn rc_alloc_report(ctx: *mut RawCtx, out: *mut RawAllocReport) -> i32;
    fn rc_last_alloc_failure_tag() -> *const std::os::raw::c_char;
    fn rc_destroy(ctx: *mut RawCtx);
    fn rc_set_programs(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
    ) -> i32;
    fn rc_eval(ctx: *mut RawCtx, out: *mut f32, loss: i32) -> f32;
    /* #285: split submit/collect and arena introspection. `rc_eval` is the
       compatibility shim over these two. */
    fn rc_eval_submit(ctx: *mut RawCtx, loss: i32) -> i32;
    fn rc_eval_collect(ctx: *mut RawCtx, out: *mut f32) -> f32;
    fn rc_arena_report(ctx: *mut RawCtx, out: *mut RawArenaReport) -> i32;
    fn rc_arena_force_sync(ctx: *mut RawCtx, sync: i32) -> i32;
    /* #287: native score objectives over the resident dataset. `kind` is an
       RC_SCORE_* code, `pcount` holds one constant count per candidate (used
       only by the AIC family), and every output is f64. `status` is 0 for ok
       and 2 for a non-finite prediction; a negative return means the launch
       failed and all outputs must be discarded. */
    fn rc_score_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        kind: i32,
        pcount: *const i32,
        out: *mut f64,
        status: *mut i32,
    ) -> f32;
    #[cfg(test)]
    fn rc_score_rows(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        kind: i32,
        pcount: *const i32,
        bounds: *const i64,
        nbounds: i32,
        out: *mut f64,
        status: *mut i32,
    ) -> f32;
    fn rc_pool_score(
        ctx: *mut RawCtx,
        which: i32,
        nprog: i32,
        width: i32,
        kind: i32,
        pcount: *const i32,
        out: *mut f64,
        status: *mut i32,
    ) -> f32;
    /* #118: start and read one batch's host-time ledger. */
    fn rc_stats_begin(ctx: *mut RawCtx);
    fn rc_last_stats(ctx: *mut RawCtx, out: *mut RawStats) -> i32;
    /* #284: last kernel span by event timestamps (0 evaluator, 1 fit). */
    fn rc_last_span(ctx: *mut RawCtx, which: i32, start_ms: *mut f64, now_ms: *mut f64) -> i32;
    #[cfg(test)]
    fn rc_predict(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        out: *mut f32,
    ) -> i32;
    fn rc_device_count() -> i32;
    fn rc_device_memory(free_b: *mut usize, total_b: *mut usize, l2_bytes: *mut i32) -> i32;
    fn rc_last_alloc_failure() -> usize;
    /* #122: persistent, fixed-capacity, double-buffered program arena. */
    fn rc_pool_reserve(ctx: *mut RawCtx, nprog: i32, width: i32) -> i32;
    fn rc_pool_upload(ctx: *mut RawCtx, which: i32, prog: *const u32, nprog: i32) -> i32;
    fn rc_pool_eval(ctx: *mut RawCtx, which: i32, nprog: i32, width: i32, out: *mut f32, loss: i32) -> f32;
    fn rc_pool_hash(
        ctx: *mut RawCtx,
        which: i32,
        nprog: i32,
        width: i32,
        out: *mut u64,
    ) -> i32;
    fn rc_device_name() -> *const std::os::raw::c_char;
    fn rc_unified() -> i32;
    fn rc_copy_probe(ctx: *mut RawCtx, bytes: u64, reps: i32, h2d_ms: *mut f64, d2h_ms: *mut f64) -> i32;
    fn rc_set_fit_data(ctx: *mut RawCtx, cols: *const f64, y: *const f64, w: *const f64) -> i32;
    /* #234: the per-row label sign `t = 2l - 1` the log-loss fit arm reads. A
       null pointer clears it, and every log-loss launch is then refused.
       #289: the same resident vector is the classification scorer's label. */
    fn rc_set_fit_sign(ctx: *mut RawCtx, sign: *const f64) -> i32;
    /* #289: the per-row score weight vector, uploaded once per sample. A null
       pointer clears it (every row weighs 1). */
    fn rc_set_score_weights(ctx: *mut RawCtx, w: *const f64) -> i32;
    /* #169: batched LM constant fitting over the resident dataset.
       #229: `metric` selects the objective: 0 = squared, 1 = log-squashed,
       2 = correlation (1-|rho|), 3 = correlation_signed (1-rho).
       #224: `gauge` holds two constant slots per candidate (scale, offset);
       -1 means absent. The correlation path freezes and renormalises them. */
    fn rc_fit_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nprog: i32,
        kmax: i32,
        c0: *const f64,
        gauge: *const i32,
        iters: i32,
        metric: i32,
        outc: *mut f64,
        outsse: *mut f64,
        outstatus: *mut i32,
    ) -> f32;
    /* SM-1: the fit kernel's loss statistics and f64 J^T r at the start
       constants, `4 + kmax` doubles per candidate. Test support. */
    #[cfg(test)]
    fn rc_fit_probe(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nprog: i32,
        kmax: i32,
        c0: *const f64,
        metric: i32,
        phase: i32,
        out: *mut f64,
    ) -> i32;
    /* #230: exact rank sums of a batch of prediction vectors against the cached
       target ranks. `out` holds four u64 per segment `(num_hi, num_lo, aa_hi,
       aa_lo)`; `out_bb` receives the target's `Sigma B^2`. */
    fn rc_rank_score_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nsets: i32,
        ntrials: i32,
        kmax: i32,
        c0: *const f64,
        out: *mut u64,
        out_status: *mut i32,
        out_bb: *mut u64,
    ) -> f32;
    /* #230 test hook: the cached target ranks, per row. */
    #[cfg(test)]
    fn rc_rank_debug_b(ctx: *mut RawCtx, out: *mut i64) -> i32;
    /* #226/#232: an exact order statistic of the absolute errors of a batch of
       prediction vectors, one `f64` per segment (set x trial), with a labelled
       status (0 ok, 2 non-finite). `what` selects the statistic: 0 = median,
       1 = interquartile mean of ranks [n/4, 3n/4). The selection is a segmented
       sort of the |r| bit patterns. */
    fn rc_order_stat_score_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nsets: i32,
        ntrials: i32,
        kmax: i32,
        c0: *const f64,
        what: i32,
        out_value: *mut f64,
        out_status: *mut i32,
    ) -> f32;
    /* #288: the radix select path of `rc_order_stat_score_batch` for the
       median, the maximum and the exact interquartile mean. Same layout, codes
       and labels; `blocks_per_seg <= 0` is the production launch shape. */
    fn rc_order_stat_select_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nsets: i32,
        ntrials: i32,
        kmax: i32,
        c0: *const f64,
        what: i32,
        blocks_per_seg: i32,
        out_value: *mut f64,
        out_status: *mut i32,
    ) -> f32;
    /* #227: exact maximum-error scan of a batch of prediction vectors: per
       segment the exact `f64` `max |r|`, its lowest maximiser row, and — when
       `coords`/`levels` are given — the intersection `[lo, hi]` of the per-row
       feasibility intervals at that error level. Returns the stage time in ms,
       or a negative value when nothing may be trusted. */
    fn rc_max_scan_batch(
        ctx: *mut RawCtx,
        prog: *const u32,
        off: *const u32,
        pcount: *const i32,
        nsets: i32,
        ntrials: i32,
        kmax: i32,
        c0: *const f64,
        coords: *const i32,
        levels: *const f64,
        out: *mut RawMaxOut,
    ) -> f32;
    /* #169: block width, dynamic shared bytes and constant cap of the fit-kernel
       instantiation a batch with `kmax` constants would use. Diagnostics. */
    fn rc_fit_kernel_info(
        kmax: i32,
        threads: *mut i32,
        smem_bytes: *mut i32,
        max_constants: *mut i32,
    ) -> i32;
}

/// #169: one candidate's batched-GPU fit result.
#[derive(Clone, Debug)]
pub struct GpuFitOutput {
    /// Fitted constants (exact slots, length is the candidate's `p`).
    pub c: Vec<f64>,
    /// The kernel's final SSE (recomputed exactly on the CPU for finalists).
    pub sse: f64,
    /// #286: `0` usable; `4` rank-deficient but finite; `5` recovered last
    /// accepted iterate; `6` no improvement, start kept; `7` non-finite start
    /// (log-only, no fallback); `8` recovered with bad-gradient rows skipped.
    /// `1` is an arithmetic failure and `2` a non-finite row with no usable
    /// iterate; only those route to the exact CPU fallback.
    pub status: i32,
}


/// #227: one segment's result from the exact maximum-error scan. Mirrors
/// `RcMaxOut` in `src/cuda/eval.cu` field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawMaxOut {
    /// Exact `max |pred − y|`; `NaN` when `status != 0`.
    vmax: f64,
    /// Largest per-row lower bound on the coordinate step at the requested
    /// level (`+inf` when the level is infeasible or none was requested).
    lo: f64,
    /// Smallest per-row upper bound (`-inf` when infeasible).
    hi: f64,
    /// Lowest-index maximiser; `-1` when `status != 0`.
    row: i32,
    /// `0` ok; `2` a row was not finite.
    status: i32,
}

/// #293: a half-open row range `[start, start + len)` of the resident rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRange {
    pub start: usize,
    pub len: usize,
}

/// #287: one native scoring batch: `f64` values plus a per-candidate status.
///
/// Unlike the legacy `Vec<f32>` SSE evaluator, a non-finite prediction is not
/// folded into a finite sentinel: the value is `+inf` and the status says why.
/// A negative [`Self::ms`] means the launch failed and no value may be trusted.
#[derive(Clone, Debug)]
pub struct ScoreBatch {
    /// One score per candidate, exactly the configured [`ScoreObjective`].
    pub values: Vec<f64>,
    /// One status per candidate, in the same order.
    pub status: Vec<ScoreStatus>,
    /// Kernel time in milliseconds; negative on failure.
    pub ms: f32,
}

impl ScoreBatch {
    /// True when the launch succeeded and every candidate is usable.
    pub fn all_ok(&self) -> bool {
        self.ms >= 0.0 && !self.status.is_empty() && self.status.iter().all(|s| s.is_ok())
    }

    /// True when the vector shape is self-consistent.
    pub fn is_consistent(&self) -> bool {
        self.values.len() == self.status.len()
    }
}

/// #169: the fit-kernel instantiation that owns a batch whose largest constant
/// count is `p`: `(block threads, dynamic shared bytes, constant cap)`.
///
/// `None` means `p` is past every instantiation, so the batch cannot run on the
/// device at all. Diagnostics only: the dispatcher in `eval.cu` makes the same
/// choice, this only reports it so a measurement can state the shared footprint
/// it ran with.
#[cfg(feature = "cuda")]
pub fn fit_kernel_shape(p: usize) -> Option<(usize, usize, usize)> {
    let (mut threads, mut smem, mut maxc) = (0i32, 0i32, 0i32);
    let rc = unsafe { rc_fit_kernel_info(p as i32, &mut threads, &mut smem, &mut maxc) };
    if rc != 0 || threads <= 0 || smem <= 0 {
        return None;
    }
    Some((threads as usize, smem as usize, maxc as usize))
}

/// #169: CPU builds have no fit kernel, so there is no shape to report.
#[cfg(not(feature = "cuda"))]
pub fn fit_kernel_shape(_p: usize) -> Option<(usize, usize, usize)> {
    None
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

/// #293: free and total device memory and the L2 size. Call it after the
/// context exists, so the context's own overhead is already subtracted.
pub fn device_info() -> Option<crate::memory::DeviceInfo> {
    let (mut free, mut total, mut l2) = (0usize, 0usize, 0i32);
    let rc = unsafe { rc_device_memory(&mut free, &mut total, &mut l2) };
    if rc != 0 || total == 0 {
        return None;
    }
    Some(crate::memory::DeviceInfo {
        free: free as u64,
        total: total as u64,
        l2_bytes: l2.max(0) as u64,
    })
}

/// #293: bytes of the most recent allocation the device refused while a
/// context or its fit rows were being created.
pub fn last_alloc_failure() -> Option<u64> {
    match unsafe { rc_last_alloc_failure() } {
        0 => None,
        bytes => Some(bytes as u64),
    }
}

/// #293: the allocation tag of the most recent refused device allocation.
pub fn last_alloc_failure_tag() -> Option<String> {
    let raw = unsafe { rc_last_alloc_failure_tag() };
    if raw.is_null() {
        return None;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
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
    /// #121: programs that needed the exact lowering pass. The rest were
    /// accepted verbatim, so `native + lowered + unsupported == added`.
    lowered: usize,
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
            lowered: 0,
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
        if compiled.lowered {
            self.lowered += 1;
        }
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

    /// #121: compiled programs that needed the exact lowering pass.
    pub fn lowered(&self) -> usize {
        self.lowered
    }

    /// #121: compiled programs the kernel accepted verbatim.
    pub fn native(&self) -> usize {
        self.len() - self.lowered
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

pub use crate::metrics::ScoreStat;

/// #288: the binary class sizes of `y` under the metric's label rule, or
/// `None` when `y` is not two-valued.
///
/// The engine's classification metrics label a row positive when
/// `y > (min + max)/2` ([`crate::metrics::auc`]). For a two-valued target that
/// is exactly "the larger value"; the target's exact doubled centred ranks
/// then have only two values, `nneg` on the positives and `−npos` on the
/// negatives. This helper verifies that structure instead of assuming it, so a
/// multi-valued target can never be scored with a made-up label mask. A
/// single-valued target returns `Some((0, 0))` — chance, exactly like the CPU.
fn binary_class_sizes(y: &[f64]) -> Option<(usize, usize)> {
    let target = crate::metrics::RankTarget::new(y)?;
    let n = y.len();
    let mut npos = 0usize;
    let mut nneg = 0usize;
    for &b in target.ranks() {
        if b > 0 {
            npos += 1;
        } else if b < 0 {
            nneg += 1;
        }
    }
    if npos == 0 && nneg == 0 {
        return Some((0, 0));
    }
    if npos == 0 || nneg == 0 || npos + nneg != n {
        return None;
    }
    for &b in target.ranks() {
        let want = if b > 0 {
            nneg as i64
        } else if b < 0 {
            -(npos as i64)
        } else {
            return None;
        };
        if b != want {
            return None;
        }
    }
    Some((npos, nneg))
}

/// #288: the exact `1 − AUC` (the configured metric) from the integer rank
/// sums.
///
/// With `A_i = 2r_i − (n+1)` the device returns `Σ A_i B_i`, where a
/// two-valued target's `B` is `nneg` on the positives and `−npos` on the
/// negatives. Hence `Σ_pos A_i = Σ A_i B_i / n` exactly, and
///
/// ```text
/// U     = (Σ_pos A_i + npos·nneg) / 2
/// AUC   = U / (npos·nneg)
/// error = 1 − AUC
/// ```
///
/// Both numerator and denominator are exact integers, so this performs the
/// same single `f64` division and subtraction as [`crate::metrics::auc`] and
/// returns the same bits for the same prediction vector (the CPU's rank sum is
/// exact for any realistic `n < 2^53`). A single-class target is exactly `0.5`
/// (chance), like the CPU. `None` means "decline": the target's rank sums do
/// not have the two-class shape this derivation requires.
fn auc_error_from_rank_sums(
    sums: crate::metrics::RankSums,
    n: usize,
    classes: (usize, usize),
) -> Option<f64> {
    let (npos, nneg) = classes;
    if n == 0 || npos.checked_add(nneg)? > n {
        return None;
    }
    if npos == 0 || nneg == 0 {
        // A single-class target is chance, exactly like the CPU (0.5 error).
        return Some(0.5);
    }
    let pairs = (npos as i128) * (nneg as i128);
    // `Σ B² = npos·nneg·n` is a fingerprint of the two-run target B; if it
    // does not match, the cached ranks are not the label mask and AUC would be
    // wrong. Declining sends the run to the CPU, never to a fabricated value.
    if sums.bb != pairs * n as i128 {
        return None;
    }
    let n_i = n as i128;
    if sums.ab % n_i != 0 {
        return None;
    }
    let pos_a = sums.ab / n_i;
    // `U` need not be an integer: with average ranks a tied positive/negative
    // pair contributes exactly one half, so `2U` may be odd (a constant
    // prediction on an odd `npos·nneg` gives `U = npos·nneg/2`). Both the
    // integer and the half-integer are exact in f64, so the division below is
    // still the CPU's single correctly-rounded operation.
    let two_u = pos_a + pairs;
    let auc = (two_u as f64 / 2.0) / pairs as f64;
    // The same two operations as `metrics::score(.., Metric::Auc)`:
    // `1.0 - auc`. Keeping them here (rather than returning AUC) is what makes
    // the value bit-identical to the configured error.
    Some(1.0 - auc)
}
/// Device bytes a rank segment holds per row: the keys, the sorted copy, the
/// run bookkeeping and the radix sort's own alternate buffers.
pub(crate) const RANK_ROW_BYTES: u64 = 72;

/// Device bytes an order-statistic segment holds per row: the keys, the sorted
/// copy and the radix sort's alternate keys.
pub(crate) const ORDER_ROW_BYTES: u64 = 64;

/// #288: device bytes a radix-select segment holds per row for the median and
/// the maximum: the keys only. The histograms and prefixes are per segment and
/// fit the fixed 1 MiB charge.
pub(crate) const SELECT_ROW_BYTES: u64 = 8;

/// #288: device bytes a radix-select segment holds per row for the
/// interquartile mean: the keys, the interior keys of the half-size window,
/// their sorted copy and the sort's alternate buffer.
pub(crate) const SELECT_IQR_ROW_BYTES: u64 = 24;

/// #288: the order statistic a device call computes. The discriminants are the
/// `what` codes of `src/cuda/eval.cu` (`RC_OS_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderCode {
    /// `metrics::median_abs`, bit for bit.
    Median = 0,
    /// The #232 fit ladder's window mean: a fixed tree reduction that agrees
    /// with the CPU to `1e-12` relative. The ladder only compares trials.
    InterquartileTree = 1,
    /// `metrics::interquartile_mean_abs`, bit for bit: the rank window added
    /// in ascending order, like the CPU's left-to-right sum.
    InterquartileExact = 2,
    /// The largest `|r|`, bit for bit.
    Maximum = 3,
}

impl OrderCode {
    /// The fit ladder's statistic, read from its device code
    /// ([`crate::fit::OrderStat::device_code`]).
    fn for_fit(want: crate::fit::OrderStat) -> Self {
        if want.device_code() == OrderCode::Median as i32 {
            OrderCode::Median
        } else {
            OrderCode::InterquartileTree
        }
    }

    /// Rows from which the radix select path is faster than the segmented
    /// sort, measured by `bench_order_select_crossover` (`docs/gpu-backend.md`),
    /// or `None` when only the sort path computes the statistic (the #232 tree
    /// mean).
    pub(crate) fn select_min_rows(self) -> Option<usize> {
        match self {
            OrderCode::Median | OrderCode::Maximum => Some(ORDER_SELECT_MIN_ROWS),
            OrderCode::InterquartileExact => Some(IQR_SELECT_MIN_ROWS),
            OrderCode::InterquartileTree => None,
        }
    }
}

/// #288: the device path of an order statistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrderPath {
    /// CUB segmented radix sort of the keys, then a pick.
    Sort,
    /// Six prefix-filtered radix histogram passes (`src/cuda/radix_select.h`).
    Select,
}

/// #288: the measured median/maximum crossover of the select path.
pub(crate) const ORDER_SELECT_MIN_ROWS: usize = 4096;

/// #288: the measured interquartile crossover of the select path.
pub(crate) const IQR_SELECT_MIN_ROWS: usize = 65_536;

/// #288: the measured path for `code` on `n` rows. Below the crossover, and for
/// a statistic the select path does not compute, the sort path runs.
pub(crate) fn order_stat_path(code: OrderCode, n: usize) -> OrderPath {
    match code.select_min_rows() {
        Some(min) if n >= min => OrderPath::Select,
        _ => OrderPath::Sort,
    }
}


/// #293 F1: how a rank or order-statistic score call is split so that one
/// `n` rows and is charged `row_bytes` a row plus 1 MiB of fixed workspace.
/// `n` rows and is charged 64 bytes a row plus 1 MiB of fixed workspace.
/// Returns `(sets per wave, trials per call)`: whole trial lists while one list
/// fits, otherwise one set at a time with the trial list cut into sub-waves. A
/// single segment above the cap cannot be split further and still runs.
pub(crate) fn wave_shape(n: usize, ntrials: usize, cap: u64, row_bytes: u64) -> (usize, usize) {
    let per_seg = (n as u64).saturating_mul(row_bytes).saturating_add(1 << 20);
    let segs = (cap / per_seg.max(1)).max(1) as usize;
    if segs >= ntrials {
        ((segs / ntrials).max(1), ntrials)
    } else {
        (1, segs)
    }
}

/// A resident GPU evaluator: the dataset is uploaded once, then many batches of
/// candidates can be scored without further host↔device traffic for the data.
pub struct GpuEvaluator {
    ctx: *mut RawCtx,
    /// #122: capacity and width of the persistent program arena, or `0` when
    /// it has not been reserved yet.
    pool_cap: usize,
    pool_width: usize,
    /// Which of the two arena buffers the last upload/evaluation used.
    pool_last: usize,
    /// #118: rows in the resident dataset. Kept `u64` for the accounting
    /// totals; prediction buffers cast it to `usize`.
    n_rows: u64,
    /// #118: accumulated host-time ledger for every batch scored here.
    stats: GpuStats,
    /// #284: run clock for evaluator device intervals.
    run_start: std::time::Instant,
    /// #284: evaluator side of the run ledger. `GpuStats` keeps the additive C
    /// windows; this keeps the device intervals whose union is device-busy.
    ledger: crate::gpu_stats::RunLedger,
    /// #229: the legacy SSE objective. The search's native scoring path uses
    /// [`crate::metrics::ScoreObjective`] via `score_pool` (#287).
    loss: crate::fit::DeviceLoss,
    /// #288: the target's binary class sizes, cached from the exact `f64`
    /// target in [`Self::set_fit_data`]. `Some((0, 0))` is a single-class
    /// target (AUC is chance); `None` means the target is not two-valued, so a
    /// native AUC score is declined rather than computed from a wrong label
    /// mask. Unused by the order and rank-correlation statistics.
    target_classes: Option<(usize, usize)>,
    /// #293: the host copy of the feature columns a tiled context streams
    /// from. The device context holds a pointer into it.
    host_cols: Option<Vec<f32>>,
}

impl GpuEvaluator {
    /// Upload a dataset. `cols` is column-major: `cols[var * n_rows + row]`.
    pub fn new(cols: &[f32], n_rows: usize, n_vars: usize, y: &[f32]) -> Option<Self> {
        Self::with_budget(cols, n_rows, n_vars, y, None)
    }

    /// [`Self::new`] under a byte budget on the live device allocations of this
    /// evaluator (#293). A request that would pass it is refused with its tag.
    pub fn with_budget(
        cols: &[f32],
        n_rows: usize,
        n_vars: usize,
        y: &[f32],
        budget: Option<u64>,
    ) -> Option<Self> {
        let ctx = unsafe {
            rc_create(
                cols.as_ptr(),
                n_rows as i32,
                n_vars as i32,
                y.as_ptr(),
                budget.unwrap_or(0),
            )
        };
        if ctx.is_null() {
            None
        } else {
            Some(GpuEvaluator {
                ctx,
                n_rows: n_rows as u64,
                pool_cap: 0,
                pool_width: 0,
                pool_last: 0,
                stats: GpuStats::default(),
                run_start: std::time::Instant::now(),
                ledger: crate::gpu_stats::RunLedger::new(),
                loss: crate::fit::DeviceLoss::Squared,
                target_classes: None,
                host_cols: None,
            })
        }
    }

    /// #293: an evaluator whose feature columns stay on the host and stream
    /// through two device tiles of `tile_rows` rows. `cols` is column-major
    /// like [`Self::new`]. Scores are bit-identical to the resident evaluator
    /// for every tile size. `partials_cap` bounds the lane carry in bytes, so
    /// a population is scored in waves when it would pass it. The fit, the
    /// exact rank and order statistics and the f64 twin are refused.
    pub fn new_tiled(
        cols: Vec<f32>,
        n_rows: usize,
        n_vars: usize,
        y: &[f32],
        tile_rows: usize,
        budget: Option<u64>,
        partials_cap: Option<u64>,
    ) -> Option<Self> {
        if tile_rows == 0 || cols.len() != n_rows * n_vars || y.len() != n_rows {
            return None;
        }
        let ctx = unsafe {
            rc_create_tiled(
                cols.as_ptr(),
                n_rows as i32,
                n_vars as i32,
                y.as_ptr(),
                tile_rows as i32,
                budget.unwrap_or(0),
                partials_cap.unwrap_or(0),
            )
        };
        if ctx.is_null() {
            return None;
        }
        Some(GpuEvaluator {
            ctx,
            n_rows: n_rows as u64,
            pool_cap: 0,
            pool_width: 0,
            pool_last: 0,
            stats: GpuStats::default(),
            run_start: std::time::Instant::now(),
            ledger: crate::gpu_stats::RunLedger::new(),
            loss: crate::fit::DeviceLoss::Squared,
            target_classes: None,
            host_cols: Some(cols),
        })
    }

    /// #293: how the feature columns live on the device.
    pub fn residency(&self) -> crate::memory::Residency {
        if self.host_cols.is_some() {
            crate::memory::Residency::Tiled
        } else {
            crate::memory::Residency::Resident
        }
    }

    /// #229: the legacy SSE objective this evaluator accumulates.
    pub(crate) fn set_loss(&mut self, loss: crate::fit::DeviceLoss) {
        self.loss = loss;
    }

    /// #229: the legacy SSE objective. Test-only: the search's native scoring
    /// path selects a [`crate::metrics::ScoreObjective`] per batch.
    #[cfg(test)]
    pub(crate) fn loss(&self) -> crate::fit::DeviceLoss {
        self.loss
    }

    /// #169: upload exact f64 training columns for the batched fit's final
    /// polish. Column-major `cols[var * n_rows + row]`. Returns `false` when
    /// the device allocation fails (the CPU arm then remains authoritative).
    ///
    /// #229: `w` is the base `--weight` row vector; an empty slice means
    /// unweighted and uploads no weight array (bit-for-bit the old path).
    ///
    /// #228: public because it is part of the [`crate::fit::BatchedGpu`]
    /// contract: a caller that drives a whole generation through
    /// [`crate::fit::BatchedFitter`] must have uploaded the same rows (and the
    /// same weights) it will verify the proposals on.
    pub fn set_fit_data(&mut self, cols: &[f64], y: &[f64], w: &[f64]) -> bool {
        let wp = if w.is_empty() { std::ptr::null() } else { w.as_ptr() };
        let ok = unsafe { rc_set_fit_data(self.ctx, cols.as_ptr(), y.as_ptr(), wp) == 0 };
        if ok && y.len() as u64 == self.n_rows {
            // #288: cache the binary class sizes from the same exact `f64`
            // target the rank stage caches `B` from, so a native AUC uses one
            // label rule (the midpoint over the observed extremes) for the
            // whole run. A multi-valued target declines instead of guessing.
            self.target_classes = binary_class_sizes(y);
        } else if ok {
            self.target_classes = None;
        }
        ok
    }

    /// #288: the target class sizes a native AUC would use, or `None` when the
    /// exact target staged in [`Self::set_fit_data`] is not two-valued.
    /// Test-only: the search sets the target once and never reads it back.
    #[cfg(test)]
    pub(crate) fn auc_classes(&self) -> Option<(usize, usize)> {
        self.target_classes
    }

    /// #234/#289: upload the per-row label sign `t = 2·l − 1`, with `l` from
    /// the target midpoint ([`crate::fit::classification_signs`]).
    ///
    /// The same resident vector serves the log-loss **fit** and every native
    /// classification **score**: a one-block-per-candidate kernel cannot take
    /// the midpoint over the sample by itself, and the host already applies the
    /// single midpoint rule. Must follow the resident row count (which
    /// [`Self::new`] fixes) and precede any `DeviceLoss::Logistic` or
    /// classification batch; a launch without it is refused on the device, so
    /// the arm can never fall back to another objective. An empty slice clears
    /// the vector.
    pub fn set_fit_sign(&mut self, sign: &[f64]) -> bool {
        if sign.is_empty() {
            return unsafe { rc_set_fit_sign(self.ctx, std::ptr::null()) == 0 };
        }
        if sign.len() as u64 != self.n_rows {
            return false;
        }
        unsafe { rc_set_fit_sign(self.ctx, sign.as_ptr()) == 0 }
    }

    /// #289: upload the per-row weights the classification scorer reads.
    ///
    /// The classification kernel accumulates `Σ wᵢ·lossᵢ` and divides by
    /// `Σw` in f64, exactly `metrics::score_native_weighted`. A zero-weight row
    /// is skipped before its prediction is read. An empty slice clears the
    /// vector (every row weighs 1).
    pub fn set_score_weights(&mut self, w: &[f64]) -> bool {
        if w.is_empty() {
            return unsafe { rc_set_score_weights(self.ctx, std::ptr::null()) == 0 };
        }
        if w.len() as u64 != self.n_rows {
            return false;
        }
        unsafe { rc_set_score_weights(self.ctx, w.as_ptr()) == 0 }
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

    /// #290: wall time in milliseconds of `reps` pinned-host copies of `bytes`
    /// each way, one entry per rep. `None` when the device refuses the buffer
    /// or a copy fails.
    pub(crate) fn copy_probe(&mut self, bytes: usize, reps: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut h2d = vec![0.0f64; reps];
        let mut d2h = vec![0.0f64; reps];
        let rc = unsafe {
            rc_copy_probe(self.ctx, bytes as u64, reps as i32, h2d.as_mut_ptr(), d2h.as_mut_ptr())
        };
        if rc == 0 {
            Some((h2d, d2h))
        } else {
            None
        }
    }

    /// #131 fixtures: raw per-row predictions for a packed batch, `P × N`
    /// row-major. This is the test-only observation path: unlike
    /// [`Self::eval_batch`] it does not collapse a program to one SSE and it
    /// does not apply the hard-failure magnitude limit, so a fixture can see
    /// `NaN`, infinities and signed zeros exactly as the kernel produced them.
    ///
    /// Returns `None` on any CUDA failure.
    #[cfg(test)]
    pub(crate) fn predictions(&mut self, prog: &[u32], offsets: &[u32]) -> Option<Vec<f32>> {
        let nprog = offsets.len().saturating_sub(1);
        if nprog == 0 {
            return Some(Vec::new());
        }
        let total = nprog.checked_mul(usize::try_from(self.n_rows).ok()?)?;
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
    /// Either way the batch's host-time ledger is recorded, so a failed batch
    /// still shows the wall it cost.
    pub fn eval_batch(&mut self, prog: &[u32], offsets: &[u32]) -> (Vec<f32>, f32) {
        let nprog = offsets.len().saturating_sub(1);
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| {
            unsafe {
                rc_stats_begin(self.ctx);
                if rc_set_programs(self.ctx, prog.as_ptr(), offsets.as_ptr(), nprog as i32) != 0 {
                    ok = false;
                    return (Vec::new(), -1.0);
                }
                let mut out = vec![0.0f32; nprog];
                let ms = rc_eval(self.ctx, out.as_mut_ptr(), self.loss.code());
                if ms < 0.0 {
                    ok = false;
                }
                (out, ms)
            }
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        result
    }

    /// #293: the device allocation ledger of this evaluator.
    pub fn alloc_report(&self) -> AllocReport {
        let mut raw = RawAllocReport {
            requests: 0,
            max_request_bytes: 0,
            live_bytes: 0,
            peak_live_bytes: 0,
            refused: 0,
            max_request_tag: [0; 32],
            refused_tag: [0; 32],
        };
        unsafe { rc_alloc_report(self.ctx, &mut raw) };
        AllocReport {
            requests: raw.requests,
            max_request_bytes: raw.max_request_bytes,
            max_request_tag: tag_text(&raw.max_request_tag),
            live_bytes: raw.live_bytes,
            peak_live_bytes: raw.peak_live_bytes,
            refused: raw.refused,
            refused_tag: tag_text(&raw.refused_tag),
        }
    }

    /// #285: the persistent arena's current state.
    ///
    /// `steady_alloc_calls` is the number of allocator calls made *after* the
    /// first successful batch. With a stable batch shape it must not increase:
    /// the arena is allocated once and reused through both program slots.
    /// `reason` names the fallback when `async_copy` is false.
    pub fn arena_report(&self) -> ArenaReport {
        let mut raw = RawArenaReport {
            async_copy: 0,
            reason: [0; 96],
            pinned_bytes: 0,
            alloc_calls: 0,
            steady_alloc_calls: 0,
            overlapped_uploads: 0,
            batches: 0,
            fs_alloc_calls: 0,
            fs_growths: 0,
            fs_steady_alloc_calls: 0,
            fs_batches: 0,
            fs_dev_bytes: 0,
            fs_pinned_bytes: 0,
            tile_rows: 0,
            tiles: 0,
            tile_uploads: 0,
            tile_overlapped_uploads: 0,
            tile_pinned_bytes: 0,
            waves: 0,
        };
        if unsafe { rc_arena_report(self.ctx, &mut raw) } != 0 {
            return ArenaReport::from_raw(&raw);
        }
        ArenaReport::from_raw(&raw)
    }

    /// #285: force the blocking copy path (`sync = true`) or re-enable the
    /// pinned asynchronous path. Returns whether async copies were enabled
    /// before the call. Used by the parity test and as an operational escape
    /// hatch when a host cannot pin the staging ring.
    pub fn force_synchronous_copy(&mut self, sync: bool) -> bool {
        let previous = unsafe { rc_arena_force_sync(self.ctx, i32::from(sync)) };
        previous != 0
    }

    /// #285: score two packed batches with the second upload issued while the
    /// first kernel is still outstanding.
    ///
    /// This is the production double-buffer schedule made observable: batch A
    /// is uploaded and submitted on `S_eval`, batch B is then uploaded on
    /// `S_h2d` (the `overlapped_uploads` counter records that A's kernel was
    /// not yet complete when B's copy was enqueued), A is collected, and only
    /// then is B submitted. Arithmetic and launch shape are identical to two
    /// [`Self::eval_batch`] calls, so the errors are bit-for-bit equal.
    ///
    /// The ledger records the combined wall; this helper exists to exercise
    /// and document the event ordering rather than to change the search.
    #[allow(clippy::type_complexity)]
    pub fn eval_batch_pipelined(
        &mut self,
        a_prog: &[u32],
        a_off: &[u32],
        b_prog: &[u32],
        b_off: &[u32],
    ) -> ((Vec<f32>, f32), (Vec<f32>, f32)) {
        let na = a_off.len().saturating_sub(1);
        let nb = b_off.len().saturating_sub(1);
        if na == 0 || nb == 0 {
            return ((Vec::new(), 0.0), (Vec::new(), 0.0));
        }
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = unsafe {
            rc_stats_begin(self.ctx);
            let loss = self.loss.code();
            if rc_set_programs(self.ctx, a_prog.as_ptr(), a_off.as_ptr(), na as i32) != 0
                || rc_eval_submit(self.ctx, loss) != 0
            {
                ok = false;
                ((Vec::new(), -1.0), (Vec::new(), -1.0))
            } else {
                /* B's upload is enqueued while A's kernel may still run. */
                let b_ok =
                    rc_set_programs(self.ctx, b_prog.as_ptr(), b_off.as_ptr(), nb as i32) == 0;
                if !b_ok {
                    ok = false;
                }
                let mut out_a = vec![0.0f32; na];
                let ms_a = rc_eval_collect(self.ctx, out_a.as_mut_ptr());
                if ms_a < 0.0 {
                    ok = false;
                }
                if !b_ok || rc_eval_submit(self.ctx, loss) != 0 {
                    ok = false;
                    ((out_a, ms_a), (Vec::new(), -1.0))
                } else {
                    let mut out_b = vec![0.0f32; nb];
                    let ms_b = rc_eval_collect(self.ctx, out_b.as_mut_ptr());
                    if ms_b < 0.0 {
                        ok = false;
                    }
                    ((out_a, ms_a), (out_b, ms_b))
                }
            }
        };
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        result
    }

    /// #287: score a packed batch with a native [`ScoreObjective`].
    ///
    /// `pcount` supplies the fitted-constant count per candidate for the AIC
    /// family; it is ignored by every other objective, but its length must
    /// still equal `nprog` so a mis-sized batch is rejected on the host. The
    /// returned values are `f64` and must never be divided by the row count:
    /// the kernel already computes the configured mean/statistic, and a
    /// non-finite row is `+inf` with [`ScoreStatus::NonFinite`].
    pub fn score_batch(
        &mut self,
        prog: &[u32],
        offsets: &[u32],
        objective: ScoreObjective,
        pcount: &[i32],
    ) -> ScoreBatch {
        let nprog = offsets.len().saturating_sub(1);
        if nprog == 0 {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: 0.0,
            };
        }
        if pcount.len() != nprog {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: -1.0,
            };
        }
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| {
            let mut values = vec![0.0f64; nprog];
            let mut status = vec![0i32; nprog];
            unsafe {
                rc_stats_begin(self.ctx);
                // `rc_score_batch` uploads the packed programs itself, so the
                // batch ledger still covers one H2D window, not two.
                let ms = rc_score_batch(
                    self.ctx,
                    prog.as_ptr(),
                    offsets.as_ptr(),
                    nprog as i32,
                    objective.code(),
                    pcount.as_ptr(),
                    values.as_mut_ptr(),
                    status.as_mut_ptr(),
                );
                if ms < 0.0 {
                    ok = false;
                }
                let status: Vec<ScoreStatus> = status
                    .into_iter()
                    .map(|code| {
                        if ms < 0.0 {
                            ScoreStatus::DeviceError
                        } else {
                            ScoreStatus::from_code(code)
                        }
                    })
                    .collect();
                (values, status, ms)
            }
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        let (values, status, ms) = result;
        ScoreBatch { values, status, ms }
    }

    /// #293 seam: score an explicit row partition. `ranges` must tile
    /// `[0, n)` in order without gaps; the result is bit-identical to
    /// [`Self::score_batch`] for every partition. A malformed partition is a
    /// failed launch (negative `ms`).
    #[cfg(test)]
    pub(crate) fn score_rows(
        &mut self,
        prog: &[u32],
        offsets: &[u32],
        objective: ScoreObjective,
        pcount: &[i32],
        ranges: &[RowRange],
    ) -> ScoreBatch {
        let nprog = offsets.len().saturating_sub(1);
        let failed = || ScoreBatch {
            values: Vec::new(),
            status: Vec::new(),
            ms: -1.0,
        };
        if nprog == 0 || pcount.len() != nprog || ranges.is_empty() {
            return failed();
        }
        let mut bounds = Vec::with_capacity(ranges.len() + 1);
        let mut next = 0usize;
        bounds.push(0i64);
        for r in ranges {
            if r.start != next || r.len == 0 {
                return failed();
            }
            next += r.len;
            bounds.push(next as i64);
        }
        let mut values = vec![0.0f64; nprog];
        let mut status = vec![0i32; nprog];
        let ms = unsafe {
            rc_stats_begin(self.ctx);
            rc_score_rows(
                self.ctx,
                prog.as_ptr(),
                offsets.as_ptr(),
                nprog as i32,
                objective.code(),
                pcount.as_ptr(),
                bounds.as_ptr(),
                bounds.len() as i32,
                values.as_mut_ptr(),
                status.as_mut_ptr(),
            )
        };
        let status = status
            .into_iter()
            .map(|code| {
                if ms < 0.0 {
                    ScoreStatus::DeviceError
                } else {
                    ScoreStatus::from_code(code)
                }
            })
            .collect();
        ScoreBatch { values, status, ms }
    }

    /// #287: score a fixed-capacity [`GenomePool`] with a native objective.
    ///
    /// This is the search's production scoring entry point. It reuses the same
    /// persistent arena as [`Self::eval_pool`], so a generation still does no
    /// steady-state `cudaMalloc`/`cudaFree`.
    pub fn score_pool(
        &mut self,
        pool: &crate::genome::GenomePool,
        objective: ScoreObjective,
        pcount: &[i32],
    ) -> ScoreBatch {
        let n = pool.len();
        if n == 0 {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: 0.0,
            };
        }
        if pcount.len() != n {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: -1.0,
            };
        }
        let width = pool.width();
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| {
            let mut values = vec![0.0f64; n];
            let mut status = vec![0i32; n];
            unsafe {
                rc_stats_begin(self.ctx);
                if self.pool_cap < pool.capacity() || self.pool_width != width {
                    if rc_pool_reserve(self.ctx, pool.capacity() as i32, width as i32) != 0 {
                        ok = false;
                        return (Vec::new(), Vec::new(), -1.0);
                    }
                    self.pool_cap = pool.capacity();
                    self.pool_width = width;
                }
                let which = (self.pool_last ^ 1) as i32;
                if rc_pool_upload(self.ctx, which, pool.active().as_ptr(), n as i32) != 0 {
                    ok = false;
                    return (Vec::new(), Vec::new(), -1.0);
                }
                let ms = rc_pool_score(
                    self.ctx,
                    which,
                    n as i32,
                    width as i32,
                    objective.code(),
                    pcount.as_ptr(),
                    values.as_mut_ptr(),
                    status.as_mut_ptr(),
                );
                if ms < 0.0 {
                    ok = false;
                }
                self.pool_last = which as usize;
                let status: Vec<ScoreStatus> = status
                    .into_iter()
                    .map(|code| {
                        if ms < 0.0 {
                            ScoreStatus::DeviceError
                        } else {
                            ScoreStatus::from_code(code)
                        }
                    })
                    .collect();
                (values, status, ms)
            }
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        let (values, status, ms) = result;
        ScoreBatch { values, status, ms }
    }

    /// #287: score a [`Population`] by reusing [`Self::score_batch`].
    pub fn score_population(
        &mut self,
        population: &Population,
        objective: ScoreObjective,
        pcount: &[i32],
    ) -> ScoreBatch {
        let (prog, offsets) = population.programs();
        self.score_batch(prog, offsets, objective, pcount)
    }

    /// #287: score `n` fixed-width slots read directly from `arena`.
    ///
    /// The flat postfix path already keeps its programs slot-major and
    /// `HALT`-padded, so this is one upload and one launch. `pcount` is not
    /// carried by [`crate::postfix::Arena`]; the parameter-aware AIC family is
    /// kept off this path, and every other objective ignores the count.
    pub fn score_arena(
        &mut self,
        arena: &[u32],
        n: usize,
        width: usize,
        objective: ScoreObjective,
    ) -> ScoreBatch {
        if n == 0 {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: 0.0,
            };
        }
        if width < 2 || arena.len() < n * width {
            return ScoreBatch {
                values: Vec::new(),
                status: Vec::new(),
                ms: -1.0,
            };
        }
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| {
            let mut values = vec![0.0f64; n];
            let mut status = vec![0i32; n];
            unsafe {
                rc_stats_begin(self.ctx);
                if self.pool_cap < n || self.pool_width != width {
                    if rc_pool_reserve(self.ctx, n as i32, width as i32) != 0 {
                        ok = false;
                        return (Vec::new(), Vec::new(), -1.0);
                    }
                    self.pool_cap = n;
                    self.pool_width = width;
                }
                let which = (self.pool_last ^ 1) as i32;
                if rc_pool_upload(self.ctx, which, arena.as_ptr(), n as i32) != 0 {
                    ok = false;
                    return (Vec::new(), Vec::new(), -1.0);
                }
                let ms = rc_pool_score(
                    self.ctx,
                    which,
                    n as i32,
                    width as i32,
                    objective.code(),
                    std::ptr::null(),
                    values.as_mut_ptr(),
                    status.as_mut_ptr(),
                );
                if ms < 0.0 {
                    ok = false;
                }
                self.pool_last = which as usize;
                let status: Vec<ScoreStatus> = status
                    .into_iter()
                    .map(|code| {
                        if ms < 0.0 {
                            ScoreStatus::DeviceError
                        } else {
                            ScoreStatus::from_code(code)
                        }
                    })
                    .collect();
                (values, status, ms)
            }
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        let (values, status, ms) = result;
        ScoreBatch { values, status, ms }
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
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| {
            unsafe {
                rc_stats_begin(self.ctx);
                if self.pool_cap < pool.capacity() || self.pool_width != width {
                    if rc_pool_reserve(self.ctx, pool.capacity() as i32, width as i32) != 0 {
                        ok = false;
                        return (Vec::new(), -1.0);
                    }
                    self.pool_cap = pool.capacity();
                    self.pool_width = width;
                }
                let which = (self.pool_last ^ 1) as i32;
                if rc_pool_upload(self.ctx, which, pool.active().as_ptr(), n as i32) != 0 {
                    ok = false;
                    return (Vec::new(), -1.0);
                }
                let mut out = vec![0.0f32; n];
                let ms = rc_pool_eval(self.ctx, which, n as i32, width as i32, out.as_mut_ptr(), self.loss.code());
                if ms < 0.0 {
                    ok = false;
                }
                self.pool_last = which as usize;
                (out, ms)
            }
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        result
    }

    /// #118: read the C ledger for the batch that just ran and add it to the
    /// run totals. `wall_s` is measured around the whole call; the residue
    /// outside the C windows is recorded as `host_s`, so the phase sum closes
    /// to wall instead of hiding an `other` bucket.
    fn record_batch(&mut self, wall_s: f64, ok: bool) {
        let mut raw = RawStats::default();
        if unsafe { rc_last_stats(self.ctx, &mut raw) } == 0 {
            let failed = !ok || raw.failed != 0;
            // The outer wall is measured around the whole evaluator call, while
            // the C windows only cover reserve/upload/launch/sync/download. The
            // difference is real host glue: the output vector, the gaps between
            // those C calls and the Rust wrapper. Record it as `host_s` so the
            // phase sum closes instead of hiding an "other" bucket.
            let outer_ms = wall_s * 1e3;
            let c_windows_ms = raw.alloc_ms + raw.h2d_ms + raw.launch_ms + raw.sync_ms + raw.d2h_ms;
            let host_ms = (outer_ms - c_windows_ms).max(0.0);
            self.stats.add_batch(
                raw.kernel_ms,
                raw.launch_ms,
                raw.sync_ms,
                raw.alloc_ms,
                raw.h2d_ms,
                raw.d2h_ms,
                host_ms,
                outer_ms,
                raw.h2d_bytes,
                raw.d2h_bytes,
                raw.batch.max(0) as u64,
                self.n_rows,
                failed,
            );
            // #284: the evaluator's device interval is the raw kernel window,
            // not the host envelope. Place it on the run clock using the
            // measured outer wall; a failed batch keeps its interval too.
            let kernel_s = raw.kernel_ms.max(0.0) / 1e3;
            match self.take_span_start_s(SPAN_EVAL) {
                Some(start_s) => self.ledger.record_eval_kernel_event(0, start_s, kernel_s),
                None => {
                    let end_s = self.run_start.elapsed().as_secs_f64();
                    let start_s = (end_s - wall_s).max(0.0);
                    self.ledger.record_eval_kernel(0, start_s, kernel_s);
                }
            }
        }
    }

    /// #284: the start of the last kernel span of kind `which` on the run clock,
    /// from CUDA event timestamps. `None` when the events are unavailable, and
    /// the caller then keeps the host envelope.
    pub(crate) fn take_span_start_s(&mut self, which: i32) -> Option<f64> {
        let (mut start_ms, mut now_ms) = (0.0f64, 0.0f64);
        let ok = unsafe { rc_last_span(self.ctx, which, &mut start_ms, &mut now_ms) };
        if ok != 1 {
            return None;
        }
        let age_s = ((now_ms - start_ms) / 1e3).max(0.0);
        Some((self.run_start.elapsed().as_secs_f64() - age_s).max(0.0))
    }

    /// #284: place this evaluator's device intervals on the caller's run clock.
    pub(crate) fn set_run_start(&mut self, run_start: std::time::Instant) {
        self.run_start = run_start;
    }

    /// Accumulated accounting for every batch scored through this evaluator.
    pub fn stats(&self) -> GpuStats {
        self.stats
    }

    /// #284: device intervals recorded by this evaluator.
    pub fn ledger(&self) -> &crate::gpu_stats::RunLedger {
        &self.ledger
    }

    /// #122: evaluate `n` fixed-width slots read directly from `arena`.
    ///
    /// This is the primitive the postfix variation path uses: its arena is
    /// already slot-major and `HALT`-padded, so the evaluation is one upload
    /// and one launch with no per-program packing. The arena must hold at least
    /// `n * width` words and every slot must be a complete program.
    ///
    /// #118: the whole call is timed and added to the same full-path ledger as
    /// the packed batch, so a flat-variation run still reports alloc/H2D/
    /// launch/kernel/sync/D2H/host/wall.
    pub fn eval_arena(&mut self, arena: &[u32], n: usize, width: usize) -> (Vec<f32>, f32) {
        if n == 0 {
            return (Vec::new(), 0.0);
        }
        if width < 2 || arena.len() < n * width {
            return (Vec::new(), -1.0);
        }
        let started = std::time::Instant::now();
        let mut ok = true;
        let result = (|| unsafe {
            rc_stats_begin(self.ctx);
            if self.pool_cap < n || self.pool_width != width {
                if rc_pool_reserve(self.ctx, n as i32, width as i32) != 0 {
                    ok = false;
                    return (Vec::new(), -1.0);
                }
                self.pool_cap = n;
                self.pool_width = width;
            }
            let which = (self.pool_last ^ 1) as i32;
            if rc_pool_upload(self.ctx, which, arena.as_ptr(), n as i32) != 0 {
                ok = false;
                return (Vec::new(), -1.0);
            }
            let mut out = vec![0.0f32; n];
            let ms = rc_pool_eval(self.ctx, which, n as i32, width as i32, out.as_mut_ptr(), self.loss.code());
            if ms < 0.0 {
                ok = false;
            }
            self.pool_last = which as usize;
            (out, ms)
        })();
        self.record_batch(started.elapsed().as_secs_f64(), ok);
        result
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

    /// #169: fit the constants of a whole batch of candidates on the device.
    ///
    /// One block per candidate runs its own Levenberg–Marquardt loop over the
    /// resident rows; the host only uploads the small programs and start
    /// constants and downloads the results. A negative time means the launch
    /// failed and every result must be discarded (the caller falls back to the
    /// CPU for the whole batch).
    pub(crate) fn fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
        loss: crate::fit::DeviceLoss,
    ) -> (Vec<GpuFitOutput>, f32) {
        let nprog = plans.len();
        if nprog == 0 {
            return (Vec::new(), 0.0);
        }
        let kmax = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
        if kmax == 0 {
            return (Vec::new(), -1.0);
        }
        let mut words: Vec<u32> = Vec::new();
        let mut off: Vec<u32> = Vec::with_capacity(nprog + 1);
        let mut pcount: Vec<i32> = Vec::with_capacity(nprog);
        let mut c0 = vec![0.0f64; nprog * kmax];
        let mut gauge = vec![-1i32; nprog * 2];
        off.push(0);
        for (k, plan) in plans.iter().enumerate() {
            words.extend_from_slice(&plan.words);
            off.push(words.len() as u32);
            pcount.push(plan.nconst as i32);
            c0[k * kmax..k * kmax + plan.nconst].copy_from_slice(&plan.c0);
            gauge[k * 2..k * 2 + 2].copy_from_slice(&plan.gauge);
        }
        let mut outc = vec![0.0f64; nprog * kmax];
        let mut outsse = vec![0.0f64; nprog];
        let mut outstatus = vec![0i32; nprog];
        let ms = unsafe {
            rc_fit_batch(
                self.ctx,
                words.as_ptr(),
                off.as_ptr(),
                pcount.as_ptr(),
                nprog as i32,
                kmax as i32,
                c0.as_ptr(),
                gauge.as_ptr(),
                iters as i32,
                loss.code(),
                outc.as_mut_ptr(),
                outsse.as_mut_ptr(),
                outstatus.as_mut_ptr(),
            )
        };
        if ms < 0.0 {
            return (Vec::new(), ms);
        }
        let out = plans
            .iter()
            .enumerate()
            .map(|(k, plan)| GpuFitOutput {
                c: outc[k * kmax..k * kmax + plan.nconst].to_vec(),
                sse: outsse[k],
                status: outstatus[k],
            })
            .collect();
        (out, ms)
    }

    /// SM-1: the loss statistics `fit_kernel` sees at each plan's start
    /// constants, without a step: `[f32 assembly, f32 trial, f64 assembly,
    /// f64 trial]` and the f64 assembly's `Jᵀr`. `phase` selects the
    /// `hinge_ordinary` triple (0 logistic, 1 majorise-minimise). `None` when
    /// the device refused the batch.
    #[cfg(test)]
    pub(crate) fn fit_probe(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        loss: crate::fit::DeviceLoss,
        phase: i32,
    ) -> Option<Vec<([f64; 4], Vec<f64>)>> {
        let nprog = plans.len();
        let kmax = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
        if nprog == 0 || kmax == 0 {
            return None;
        }
        let mut words: Vec<u32> = Vec::new();
        let mut off: Vec<u32> = Vec::with_capacity(nprog + 1);
        let mut pcount: Vec<i32> = Vec::with_capacity(nprog);
        let mut c0 = vec![0.0f64; nprog * kmax];
        off.push(0);
        for (k, plan) in plans.iter().enumerate() {
            words.extend_from_slice(&plan.words);
            off.push(words.len() as u32);
            pcount.push(plan.nconst as i32);
            c0[k * kmax..k * kmax + plan.nconst].copy_from_slice(&plan.c0);
        }
        let stride = 4 + kmax;
        let mut out = vec![0.0f64; nprog * stride];
        let rc = unsafe {
            rc_fit_probe(
                self.ctx,
                words.as_ptr(),
                off.as_ptr(),
                pcount.as_ptr(),
                nprog as i32,
                kmax as i32,
                c0.as_ptr(),
                loss.code(),
                phase,
                out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return None;
        }
        Some(
            plans
                .iter()
                .enumerate()
                .map(|(k, plan)| {
                    let o = &out[k * stride..(k + 1) * stride];
                    ([o[0], o[1], o[2], o[3]], o[4..4 + plan.nconst].to_vec())
                })
                .collect(),
        )
    }

    /// #230: device memory the rank stage may use for one wave, in bytes.
    ///
    /// The evaluator keeps the resident dataset and the fit buffers alive, and
    /// a long search may share the device, so the wave that fits one call is
    /// sized conservatively rather than by "whatever cudaMalloc gives". A
    /// smaller wave changes only how many calls are made; the integer sums are
    /// exact, so the score does not depend on the split.
    const RANK_WAVE_BYTES: u64 = 512 * 1024 * 1024;

    /// #230: exact rank sums for `plans.len() * ntrials` prediction vectors.
    ///
    /// `trials[k][t]` is candidate `k`'s trial `t`, in constant-slot order.
    /// Returns `None` when the device declined (allocation, launch or a bad
    /// shape); a segment whose prediction was not finite is reported as
    /// `Some(None)`, which the exact scorer maps to `+inf` like the CPU.
    pub(crate) fn rank_score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<crate::metrics::RankSums>>>> {
        self.rank_score_trials_capped(plans, trials, Self::RANK_WAVE_BYTES)
    }

    /// [`Self::rank_score_trials`] with an explicit wave size in bytes. The
    /// sums are exact integers, so the split cannot change a score.
    pub(crate) fn rank_score_trials_capped(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        wave_bytes: u64,
    ) -> Option<Vec<Vec<Option<crate::metrics::RankSums>>>> {
        use crate::metrics::RankSums;
        let nsets = plans.len();
        if nsets == 0 || trials.len() != nsets {
            return None;
        }
        let ntrials = trials[0].len();
        if ntrials == 0 || trials.iter().any(|t| t.len() != ntrials) {
            return None;
        }
        let n = self.n_rows as usize;
        if n < 2 {
            return None;
        }
        let kmax = plans.iter().map(|p| p.nconst).max()?;
        let (wave_sets, chunk) = wave_shape(n, ntrials, wave_bytes, RANK_ROW_BYTES);
        let mut out: Vec<Vec<Option<RankSums>>> = vec![vec![None; ntrials]; nsets];
        let join = |hi: u64, lo: u64| -> i128 { (((hi as u128) << 64) | lo as u128) as i128 };
        let mut start = 0usize;
        while start < nsets {
            let end = (start + wave_sets).min(nsets);
            let wave = &plans[start..end];
            let mut words: Vec<u32> = Vec::new();
            let mut off: Vec<u32> = Vec::with_capacity(wave.len() + 1);
            let mut pcount: Vec<i32> = Vec::with_capacity(wave.len());
            off.push(0);
            for plan in wave {
                words.extend_from_slice(&plan.words);
                off.push(words.len() as u32);
                pcount.push(plan.nconst as i32);
            }
            let mut t0 = 0usize;
            while t0 < ntrials {
                let t1 = (t0 + chunk).min(ntrials);
                let width = t1 - t0;
                let segs = (end - start) * width;
                let mut c0 = vec![0.0f64; segs * kmax];
                for (k, list) in trials[start..end].iter().enumerate() {
                    for (t, c) in list[t0..t1].iter().enumerate() {
                        let base = (k * width + t) * kmax;
                        if c.len() != wave[k].nconst || c.len() > kmax {
                            return None;
                        }
                        c0[base..base + c.len()].copy_from_slice(c);
                    }
                }
                let mut raw = vec![0u64; segs * 4];
                let mut status = vec![0i32; segs];
                let mut bb = [0u64; 2];
                let ms = unsafe {
                    rc_rank_score_batch(
                        self.ctx,
                        words.as_ptr(),
                        off.as_ptr(),
                        pcount.as_ptr(),
                        (end - start) as i32,
                        width as i32,
                        kmax as i32,
                        c0.as_ptr(),
                        raw.as_mut_ptr(),
                        status.as_mut_ptr(),
                        bb.as_mut_ptr(),
                    )
                };
                if ms < 0.0 {
                    return None;
                }
                let sum_bb = join(bb[0], bb[1]);
                for seg in 0..segs {
                    let k = start + seg / width;
                    let t = t0 + seg % width;
                    // status != 0: a non-finite prediction in that segment, which
                    // the exact scorer maps to +inf exactly like the CPU.
                    out[k][t] = if status[seg] == 0 {
                        Some(RankSums {
                            ab: join(raw[seg * 4], raw[seg * 4 + 1]),
                            aa: join(raw[seg * 4 + 2], raw[seg * 4 + 3]),
                            bb: sum_bb,
                        })
                    } else {
                        None
                    };
                }
                t0 = t1;
            }
            start = end;
        }
        Some(out)
    }

    /// #230: the squared LM start followed by the exact rank polish, on device.
    ///
    /// This is the GPU counterpart of `fit_weighted_metric` for
    /// `rank_correlation` / `rank_correlation_signed`: the same squared start,
    /// the same ladder, the same first-improver rule and the same exact integer
    /// score. A candidate whose polish cannot be scored, or whose prediction is
    /// not finite, is reported with a non-zero status so the caller refits it
    /// with the exact CPU arm and labels the fallback.
    ///
    /// #231: `signed` selects `1 − ρ` instead of `1 − |ρ|`; it is the only
    /// difference between the two objectives on this stage (the integer sums
    /// are shared). The keep/replace decision is [`crate::fit::rank::choose_rank_fit`],
    /// so the orientation rule is the same in the search arm, the batched
    /// verifier and the unit tests. `iters == 0` is a true no-op: no LM step, no
    /// sweep, the incoming constants returned unchanged.
    pub(crate) fn rank_fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
        y: &[f64],
        signed: bool,
    ) -> (Vec<GpuFitOutput>, f32) {
        let incoming: Vec<Vec<f64>> = plans.iter().map(|p| p.c0.clone()).collect();
        if iters == 0 {
            // `--no-fit`: nothing may move, whatever the caller did.
            let out = incoming
                .iter()
                .map(|c| GpuFitOutput {
                    c: c.clone(),
                    sse: f64::NAN,
                    status: 0,
                })
                .collect();
            return (out, 0.0);
        }
        // The squared LM start is the same one `fit_weighted_metric` uses for
        // the rank family; the polished constants are judged against the
        // **incoming** constants, which is that CPU arm's monotone gate.
        let (mut out, lm_ms) = self.fit_batch(plans, iters, crate::fit::DeviceLoss::Squared);
        if out.is_empty() || out.len() != plans.len() {
            return (out, -1.0);
        }
        let target = match crate::metrics::RankTarget::new(y) {
            Some(t) if t.len() == self.n_rows as usize => t,
            _ => return (out, -1.0),
        };
        let idx: Vec<usize> = (0..out.len())
            .filter(|&k| crate::fit::fit_status_usable(out[k].status) && out[k].c.iter().all(|v| v.is_finite()))
            .collect();
        if idx.is_empty() {
            return (out, lm_ms);
        }
        let sub: Vec<&crate::fit::GpuFitPlan> = idx.iter().map(|&k| plans[k]).collect();
        let starts: Vec<Vec<f64>> = idx.iter().map(|&k| out[k].c.clone()).collect();
        let started = std::time::Instant::now();
        let Some(res) = crate::fit::polish_rank_batch(self, &sub, &starts, &target, signed, iters)
        else {
            for &k in &idx {
                out[k].status = 3; // the labelled `rank_declined` fallback
            }
            return (out, -1.0);
        };
        // The CPU arm keeps the incoming constants unless the polished squared
        // start is strictly better on the configured objective. Score the
        // incoming constants in one batch so the device arm obeys the same
        // monotone contract (a squared start can be worse on rho).
        let base_trials: Vec<Vec<Vec<f64>>> = idx
            .iter()
            .map(|&k| vec![incoming[k].clone()])
            .collect();
        let baseline = self.rank_score_trials(&sub, &base_trials);
        let polish_ms = started.elapsed().as_secs_f32() * 1000.0;
        for (j, &k) in idx.iter().enumerate() {
            let c = &res.constants[j];
            if c.len() != plans[k].nconst || !c.iter().all(|v| v.is_finite()) {
                out[k].status = 1;
                continue;
            }
            let start = match baseline.as_ref() {
                Some(b) => b.get(j).and_then(|t| t.first()).copied(),
                // The baseline call failed as a whole: `None`, a decline.
                None => None,
            };
            use crate::fit::RankFitChoice as Choice;
            match crate::fit::choose_rank_fit(res.scores[j], start, &target, signed) {
                Choice::TakePolished { score } => {
                    out[k].c = c.clone();
                    out[k].sse = score;
                }
                Choice::KeepStart { score } => {
                    // Keep the incoming constants (slot order) and its exact
                    // score: the CPU arm would have done the same.
                    out[k].c = incoming[k].clone();
                    out[k].sse = score;
                }
                // A prediction the exact rank stage cannot score is invalid,
                // not a better fit: label it `non_finite` and let the CPU arm
                // decide.
                Choice::NonFinite => out[k].status = 2,
                // No baseline, so no monotonicity to prove: never accept the
                // polished constants blind, label the decline and fall back.
                Choice::Declined => out[k].status = 3,
            }
        }
        (out, lm_ms + polish_ms)
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

    /// #226: device memory one median wave may use, in bytes.
    ///
    /// The keys and their sort output are one `u64` per row per segment and
    /// CUB's temporary storage is about the same again; 64 bytes a row is a
    /// conservative charge. A smaller wave changes only how many calls are
    /// made: the median of a segment is independent of the other segments, so
    /// the value does not depend on the split.
    const MEDIAN_WAVE_BYTES: u64 = 512 * 1024 * 1024;

    /// #226: exact median absolute error for `plans.len() * ntrials`
    /// prediction vectors (see [`Self::order_stat_score_trials`]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn median_score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<f64>>>> {
        self.order_stat_score_trials(plans, trials, crate::fit::OrderStat::Median)
    }

    /// #232: interquartile absolute error for `plans.len() * ntrials`
    /// prediction vectors (see [`Self::order_stat_score_trials`]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn interquartile_score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<f64>>>> {
        self.order_stat_score_trials(plans, trials, crate::fit::OrderStat::Interquartile)
    }

    /// #226/#232: an exact order statistic of the absolute errors for
    /// `plans.len() * ntrials` prediction vectors.
    ///
    /// `trials[k][t]` is candidate `k`'s trial `t`, in constant-slot order;
    /// `want` selects the statistic ([`crate::fit::OrderStat`]).
    /// Returns `None` when the device declined (allocation, launch or a bad
    /// shape); a segment whose prediction was not finite is reported as
    /// `Some(None)`, which the caller maps to an inactive trial and the exact
    /// CPU arm, never to a fabricated value.
    ///
    /// The device predicts in exact `f64` from the resident rows and keys each
    /// row by the `u64` bit pattern of `|pred − y|`, then
    /// * `median`: selects the middle key(s) (the radix select path from
    ///   [`ORDER_SELECT_MIN_ROWS`], the segmented sort below) and averages the
    ///   two middle values for even `n` — exactly what
    ///   [`crate::metrics::score_weighted`] reports, bit for bit;
    /// * `interquartile`: sorts the keys and averages ranks `[⌊n/4⌋, ⌊3n/4⌋)`
    ///   (all rows below `n = 4`) with a fixed 256-thread tree reduction, which
    ///   is the configured objective to `f64` rounding (`≤ 1e-12` relative),
    ///   not bit for bit. The ladder only compares trials; scoring uses
    ///   [`OrderCode::InterquartileExact`].
    pub(crate) fn order_stat_score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        want: crate::fit::OrderStat,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        self.order_stat_score_trials_capped(plans, trials, want, Self::MEDIAN_WAVE_BYTES)
    }

    /// [`Self::order_stat_score_trials`] with an explicit wave size in bytes.
    /// Every segment is computed on its own, so the split cannot change a value.
    pub(crate) fn order_stat_score_trials_capped(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        want: crate::fit::OrderStat,
        wave_bytes: u64,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        self.order_code_score(plans, trials, OrderCode::for_fit(want), wave_bytes)
    }

    /// #288: `code` on the measured path ([`order_stat_path`]), with the sort
    /// path as the fallback. Both paths return the same bits for every
    /// statistic the select path computes, so the choice is invisible to the
    /// caller.
    pub(crate) fn order_code_score(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        code: OrderCode,
        wave_bytes: u64,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        let path = order_stat_path(code, self.n_rows as usize);
        self.order_code_score_on(plans, trials, code, path, wave_bytes)
    }

    /// #288: [`Self::order_code_score`] on a given first path. A select call
    /// that fails as a whole (allocation, launch, an internal rank check, a
    /// statistic it does not compute) is re-run on the sort path with the sort
    /// path's own wave shape.
    pub(crate) fn order_code_score_on(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        code: OrderCode,
        path: OrderPath,
        wave_bytes: u64,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        if path == OrderPath::Select {
            if let Some(out) =
                self.order_code_trials(plans, trials, code, OrderPath::Select, wave_bytes, 0)
            {
                return Some(out);
            }
        }
        self.order_code_trials(plans, trials, code, OrderPath::Sort, wave_bytes, 0)
    }

    /// #226/#232/#288: one order statistic on one device path, split into
    /// waves of at most `wave_bytes`. `blocks_per_seg` is the select path's
    /// launch shape (`0` = production); the sort path ignores it.
    pub(crate) fn order_code_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        code: OrderCode,
        path: OrderPath,
        wave_bytes: u64,
        blocks_per_seg: i32,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        let nsets = plans.len();
        if nsets == 0 || trials.len() != nsets {
            return None;
        }
        let ntrials = trials[0].len();
        if ntrials == 0 || trials.iter().any(|t| t.len() != ntrials) {
            return None;
        }
        let n = self.n_rows as usize;
        if n == 0 {
            return None;
        }
        if (nsets * ntrials).checked_mul(n)? > i32::MAX as usize {
            return None;
        }
        if plans.iter().any(|p| p.nconst == 0) {
            return None;
        }
        let kmax = plans.iter().map(|p| p.nconst).max()?;
        let row_bytes = match (path, code) {
            (OrderPath::Sort, _) => ORDER_ROW_BYTES,
            (OrderPath::Select, OrderCode::InterquartileExact) => SELECT_IQR_ROW_BYTES,
            (OrderPath::Select, _) => SELECT_ROW_BYTES,
        };
        let (wave_sets, chunk) = wave_shape(n, ntrials, wave_bytes, row_bytes);
        let mut out: Vec<Vec<Option<f64>>> = vec![vec![None; ntrials]; nsets];
        let mut start = 0usize;
        while start < nsets {
            let end = (start + wave_sets).min(nsets);
            let wave = &plans[start..end];
            let mut words: Vec<u32> = Vec::new();
            let mut off: Vec<u32> = Vec::with_capacity(wave.len() + 1);
            let mut pcount: Vec<i32> = Vec::with_capacity(wave.len());
            off.push(0);
            for plan in wave {
                words.extend_from_slice(&plan.words);
                off.push(words.len() as u32);
                pcount.push(plan.nconst as i32);
            }
            let mut t0 = 0usize;
            while t0 < ntrials {
                let t1 = (t0 + chunk).min(ntrials);
                let width = t1 - t0;
                let nsegs = (end - start) * width;
                let mut c0 = vec![0.0f64; nsegs * kmax];
                for (k, list) in trials[start..end].iter().enumerate() {
                    for (t, c) in list[t0..t1].iter().enumerate() {
                        let base = (k * width + t) * kmax;
                        if c.len() != wave[k].nconst || c.len() > kmax {
                            return None;
                        }
                        c0[base..base + c.len()].copy_from_slice(c);
                    }
                }
                let mut values = vec![f64::NAN; nsegs];
                let mut status = vec![0i32; nsegs];
                let ms = unsafe {
                    match path {
                        OrderPath::Sort => rc_order_stat_score_batch(
                            self.ctx,
                            words.as_ptr(),
                            off.as_ptr(),
                            pcount.as_ptr(),
                            (end - start) as i32,
                            width as i32,
                            kmax as i32,
                            c0.as_ptr(),
                            code as i32,
                            values.as_mut_ptr(),
                            status.as_mut_ptr(),
                        ),
                        OrderPath::Select => rc_order_stat_select_batch(
                            self.ctx,
                            words.as_ptr(),
                            off.as_ptr(),
                            pcount.as_ptr(),
                            (end - start) as i32,
                            width as i32,
                            kmax as i32,
                            c0.as_ptr(),
                            code as i32,
                            blocks_per_seg,
                            values.as_mut_ptr(),
                            status.as_mut_ptr(),
                        ),
                    }
                };
                if ms < 0.0 {
                    return None;
                }
                for seg in 0..nsegs {
                    let k = start + seg / width;
                    let t = t0 + seg % width;
                    out[k][t] = if status[seg] == 0 && values[seg].is_finite() {
                        Some(values[seg])
                    } else {
                        None
                    };
                }
                t0 = t1;
            }
            start = end;
        }
        Some(out)
    }

    /// #226: the squared LM warm start followed by the exact median polish, on
    /// the device (see [`Self::order_stat_fit_batch`]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn median_fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
    ) -> (Vec<GpuFitOutput>, f32) {
        self.order_stat_fit_batch(plans, iters, crate::fit::OrderStat::Median)
    }

    /// #232: the squared LM warm start followed by the exact interquartile
    /// polish, on the device (see [`Self::order_stat_fit_batch`]).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn interquartile_fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
    ) -> (Vec<GpuFitOutput>, f32) {
        self.order_stat_fit_batch(plans, iters, crate::fit::OrderStat::Interquartile)
    }

    /// #226/#232: the squared LM warm start followed by the exact order-
    /// statistic polish, on the device.
    ///
    /// This is the GPU counterpart of `fit_weighted_metric(.., want.metric())`
    /// for the tail it can prove: the same squared start, the same ladder, the
    /// same first-improver rule and the same `f64` score (bit for bit for the
    /// median, to f64 rounding for the interquartile mean). The reported `sse`
    /// field carries that configured objective, never an SSE.
    ///
    /// A candidate whose polish cannot be scored, or whose prediction is not
    /// finite, is reported with a non-zero status so the caller refits it with
    /// the exact CPU arm and labels the fallback. `iters == 0` is a true no-op:
    /// the incoming constants are returned unchanged with status 0.
    pub(crate) fn order_stat_fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
        want: crate::fit::OrderStat,
    ) -> (Vec<GpuFitOutput>, f32) {
        let incoming: Vec<Vec<f64>> = plans.iter().map(|p| p.c0.clone()).collect();
        if iters == 0 {
            // `--no-fit`: no LM step and no polish. The statistic of the
            // incoming constants is still the exact objective, so the caller's
            // gate can compare it with itself and accept nothing.
            let out = incoming
                .iter()
                .map(|c| GpuFitOutput {
                    c: c.clone(),
                    sse: f64::NAN,
                    status: 0,
                })
                .collect();
            return (out, 0.0);
        }
        let (mut out, lm_ms) = self.fit_batch(plans, iters, crate::fit::DeviceLoss::Squared);
        if out.is_empty() || out.len() != plans.len() {
            return (out, -1.0);
        }
        let idx: Vec<usize> = (0..out.len())
            .filter(|&k| crate::fit::fit_status_usable(out[k].status) && out[k].c.iter().all(|v| v.is_finite()))
            .collect();
        if idx.is_empty() {
            return (out, lm_ms);
        }
        let sub: Vec<&crate::fit::GpuFitPlan> = idx.iter().map(|&k| plans[k]).collect();
        let starts: Vec<Vec<f64>> = idx.iter().map(|&k| out[k].c.clone()).collect();
        let started = std::time::Instant::now();
        let res = match want {
            crate::fit::OrderStat::Median => {
                crate::fit::polish_median_batch(self, &sub, &starts, iters.max(1))
            }
            crate::fit::OrderStat::Interquartile => {
                crate::fit::polish_interquartile_batch(self, &sub, &starts, iters.max(1))
            }
        };
        let Some(res) = res else {
            for &k in &idx {
                // The labelled `median_declined` / `interquartile_declined`
                // fallback; the caller names it from the status and the metric.
                out[k].status = 3;
            }
            return (out, -1.0);
        };
        // The CPU arm keeps the incoming constants unless the polished squared
        // start is strictly better on the configured objective. Score them in
        // one batch so the device arm obeys the same monotone contract.
        let base_trials: Vec<Vec<Vec<f64>>> = idx
            .iter()
            .map(|&k| vec![incoming[k].clone()])
            .collect();
        let baseline = self.order_stat_score_trials(&sub, &base_trials, want);
        let polish_ms = started.elapsed().as_secs_f32() * 1000.0;
        for (j, &k) in idx.iter().enumerate() {
            let c = &res.constants[j];
            if c.len() != plans[k].nconst || !c.iter().all(|v| v.is_finite()) {
                out[k].status = 1;
                continue;
            }
            if !res.scores[j].is_finite() {
                out[k].status = 2;
                continue;
            }
            let start_score = baseline
                .as_ref()
                .and_then(|b| b.get(j))
                .and_then(|t| t.first())
                .copied()
                .flatten()
                .unwrap_or(f64::INFINITY);
            if start_score.is_finite() && res.scores[j] >= start_score {
                out[k].c = incoming[k].clone();
                out[k].sse = start_score;
                continue;
            }
            out[k].c = c.clone();
            out[k].sse = res.scores[j];
        }
        (out, lm_ms + polish_ms)
    }

    /* -- #227: exact maximum-error (minimax) polish ----------------------- */

    /// #227: one exact maximum-error scan over `plans.len() × ntrials`
    /// segments (`seg = set * ntrials + trial`).
    ///
    /// `c0` holds the constants in slot order, `ntrials * kmax` per set.
    /// `coords`/`levels` are per segment, or `None` for "the maximum only".
    /// Returns one raw result per segment, or `None` when the device declined
    /// (bad shape, allocation or launch failure) — the caller then falls back
    /// to the exact CPU arm and labels it.
    fn max_scan(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        ntrials: usize,
        c0: &[f64],
        coords: Option<&[i32]>,
        levels: Option<&[f64]>,
    ) -> Option<Vec<RawMaxOut>> {
        let nsets = plans.len();
        if nsets == 0 || ntrials == 0 || plans.iter().any(|p| p.nconst == 0) {
            return None;
        }
        let kmax = plans.iter().map(|p| p.nconst).max()?;
        if self.n_rows == 0 {
            return None;
        }
        let nsegs = nsets.checked_mul(ntrials)?;
        if nsegs.checked_mul(self.n_rows as usize)? > i32::MAX as usize {
            return None;
        }
        if c0.len() != nsegs * kmax {
            return None;
        }
        if coords.is_some_and(|c| c.len() != nsegs) || levels.is_some_and(|l| l.len() != nsegs) {
            return None;
        }
        // Every buffer the arena needs is O(segments), never O(rows × segments),
        // so a wave only bounds a pathological batch (thousands of sets).
        const MAX_SCAN_WAVE_BYTES: u64 = 512 * 1024 * 1024;
        let per_trial = (kmax as u64) * 8 + 4 + 8 + 32 + 256;
        let mut out: Vec<RawMaxOut> = Vec::with_capacity(nsegs);
        let mut start = 0usize;
        while start < nsets {
            let words = plans[start..].iter().map(|p| p.words.len()).max().unwrap_or(0);
            let per_set = (words as u64) * 4 + (ntrials as u64) * per_trial;
            let wave = ((MAX_SCAN_WAVE_BYTES / per_set.max(1)).max(1) as usize).min(nsets - start);
            let end = start + wave;
            let wave_plans = &plans[start..end];
            let mut prog: Vec<u32> = Vec::new();
            let mut off: Vec<u32> = Vec::with_capacity(wave_plans.len() + 1);
            let mut pcount: Vec<i32> = Vec::with_capacity(wave_plans.len());
            off.push(0);
            for plan in wave_plans {
                prog.extend_from_slice(&plan.words);
                off.push(prog.len() as u32);
                pcount.push(plan.nconst as i32);
            }
            let wave_segs = wave_plans.len() * ntrials;
            let base = start * ntrials;
            let w_coords = coords.map(|c| c[base..base + wave_segs].to_vec());
            let w_levels = levels.map(|l| l[base..base + wave_segs].to_vec());
            let mut res = vec![
                RawMaxOut {
                    vmax: f64::NAN,
                    lo: f64::INFINITY,
                    hi: f64::NEG_INFINITY,
                    row: -1,
                    status: 2,
                };
                wave_segs
            ];
            let ms = unsafe {
                rc_max_scan_batch(
                    self.ctx,
                    prog.as_ptr(),
                    off.as_ptr(),
                    pcount.as_ptr(),
                    wave_plans.len() as i32,
                    ntrials as i32,
                    kmax as i32,
                    c0[base * kmax..].as_ptr(),
                    w_coords
                        .as_ref()
                        .map(|v| v.as_ptr())
                        .unwrap_or(std::ptr::null()),
                    w_levels
                        .as_ref()
                        .map(|v| v.as_ptr())
                        .unwrap_or(std::ptr::null()),
                    res.as_mut_ptr(),
                )
            };
            if ms < 0.0 {
                return None;
            }
            out.extend_from_slice(&res);
            start = end;
        }
        Some(out)
    }

    /// #227: the exact `max |r|` and its lowest maximiser for every
    /// `(candidate, trial)`.
    ///
    /// A maximum is order-free, so this statistic adds no rounding: the value
    /// is the exact `f64` maximum of `|pred − y|` over the resident rows, and a
    /// tie always reports the lowest row index. `Some(None)` marks a segment
    /// with a non-finite row; `None` means the device declined.
    pub(crate) fn max_score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<crate::fit::MaxStat>>>> {
        let nsets = plans.len();
        if nsets == 0 || trials.len() != nsets {
            return None;
        }
        let ntrials = trials[0].len();
        if ntrials == 0 || trials.iter().any(|t| t.len() != ntrials) {
            return None;
        }
        let kmax = plans.iter().map(|p| p.nconst).max()?;
        let mut c0 = vec![0.0f64; nsets * ntrials * kmax];
        for (k, list) in trials.iter().enumerate() {
            for (t, c) in list.iter().enumerate() {
                if c.len() != plans[k].nconst || c.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                let base = (k * ntrials + t) * kmax;
                c0[base..base + c.len()].copy_from_slice(c);
            }
        }
        let raw = self.max_scan(plans, ntrials, &c0, None, None)?;
        let mut out: Vec<Vec<Option<crate::fit::MaxStat>>> = vec![vec![None; ntrials]; nsets];
        for (seg, r) in raw.iter().enumerate() {
            let (k, t) = (seg / ntrials, seg % ntrials);
            out[k][t] = if r.status == 0 && r.vmax.is_finite() && r.row >= 0 {
                Some(crate::fit::MaxStat {
                    value: r.vmax,
                    row: r.row as usize,
                })
            } else {
                None
            };
        }
        Some(out)
    }

    /// #288: score a batch of **arbitrary candidates** with one exact
    /// order-statistic or rank statistic.
    ///
    /// `plans[k]` is candidate `k`'s flattened program and `constants[k]` its
    /// constant values in slot order. The device predicts every row in exact
    /// `f64`, then computes the configured statistic:
    ///
    /// * `Median`: the exact middle order statistic of `|r|`, on the radix
    ///   select path from [`ORDER_SELECT_MIN_ROWS`] and the sort path below;
    /// * `Interquartile`: the exact window mean of `|r|`, added in ascending
    ///   order like the CPU, so bit for bit; the select path selects the
    ///   window boundaries and sorts only the interior from
    ///   [`IQR_SELECT_MIN_ROWS`];
    /// * `Maximum`: the exact order-free `max |r|` scan (#227);
    /// * `RankCorrelation` / `RankCorrelationSigned`: the exact integer rank
    ///   sums of #230 mapped to `1 − |ρ|` / `1 − ρ`;
    /// * `Auc`: the same integer sums interpreted as the Mann–Whitney `U`
    ///   statistic and reported as `1 − AUC` ([`auc_error_from_rank_sums`]).
    ///
    /// Returns `None` when the device declines the call as a whole (bad
    /// shape, allocation or launch failure, or an AUC target that is not
    /// two-valued). A candidate whose prediction is not finite is returned as
    /// `Some(None)`, which the search maps to `+inf` — exactly what the CPU
    /// scorer does, never a finite sentinel. The split into waves is by
    /// candidate only; every value is independent of the batch composition.
    pub(crate) fn score_stat_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        constants: &[Vec<f64>],
        stat: ScoreStat,
    ) -> Option<Vec<Option<f64>>> {
        if plans.is_empty() || plans.len() != constants.len() {
            return None;
        }
        if self.n_rows < 2 {
            return None;
        }
        let n = self.n_rows as usize;
        let trials: Vec<Vec<Vec<f64>>> = constants.iter().map(|c| vec![c.clone()]).collect();
        let per_set = |v: Vec<Vec<Option<f64>>>| -> Option<Vec<Option<f64>>> {
            if v.len() != plans.len() {
                return None;
            }
            Some(v.into_iter().map(|mut t| t.drain(..).next().flatten()).collect())
        };
        match stat {
            ScoreStat::Median => per_set(self.order_code_score(
                plans,
                &trials,
                OrderCode::Median,
                Self::MEDIAN_WAVE_BYTES,
            )?),
            ScoreStat::Interquartile => per_set(self.order_code_score(
                plans,
                &trials,
                OrderCode::InterquartileExact,
                Self::MEDIAN_WAVE_BYTES,
            )?),
            ScoreStat::Maximum => {
                let raw = self.max_score_trials(plans, &trials)?;
                if raw.len() != plans.len() {
                    return None;
                }
                Some(
                    raw.into_iter()
                        .map(|mut t| t.drain(..).next().flatten().map(|s| s.value))
                        .collect(),
                )
            }
            ScoreStat::RankCorrelation | ScoreStat::RankCorrelationSigned => {
                let signed = stat == ScoreStat::RankCorrelationSigned;
                let sums = self.rank_score_trials(plans, &trials)?;
                Some(
                    sums.into_iter()
                        .map(|mut t| {
                            t.drain(..).next().flatten().map(|s| {
                                crate::metrics::score_rank_sums(Some(s), signed)
                            })
                        })
                        .collect(),
                )
            }
            ScoreStat::Auc => {
                let classes = self.target_classes?;
                let sums = self.rank_score_trials(plans, &trials)?;
                let mut out = Vec::with_capacity(sums.len());
                for mut t in sums {
                    match t.drain(..).next().flatten() {
                        // A non-finite prediction is unusable, like the CPU.
                        None => out.push(None),
                        // A target that is not the two-class shape this
                        // derivation proves: decline the whole call.
                        Some(s) => out.push(Some(auc_error_from_rank_sums(s, n, classes)?)),
                    }
                }
                Some(out)
            }
        }
    }

    /// #227: the exact level-interval test for one coordinate per candidate.
    ///
    /// `constants[k]` is candidate `k`'s current constants (slot order) and
    /// `coords[k][j]`/`levels[k][j]` the constant slot and error level to test.
    /// `Some(interval)` with `lo > hi` is the exact statement "this level is not
    /// attainable along the coordinate"; `None` marks a non-finite row.
    pub(crate) fn max_intervals(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        constants: &[Vec<f64>],
        coords: &[Vec<i32>],
        levels: &[Vec<f64>],
    ) -> Option<Vec<Vec<Option<crate::fit::LevelInterval>>>> {
        let nsets = plans.len();
        if nsets == 0 || constants.len() != nsets || coords.len() != nsets || levels.len() != nsets {
            return None;
        }
        let maxn = coords.iter().map(|c| c.len()).max().unwrap_or(0);
        if maxn == 0 {
            return Some(vec![Vec::new(); nsets]);
        }
        let kmax = plans.iter().map(|p| p.nconst).max()?;
        let mut c0 = vec![0.0f64; nsets * maxn * kmax];
        let mut cd = vec![-1i32; nsets * maxn];
        let mut lv = vec![f64::NAN; nsets * maxn];
        for k in 0..nsets {
            if constants[k].len() != plans[k].nconst || levels[k].len() != coords[k].len() {
                return None;
            }
            for j in 0..maxn {
                let seg = k * maxn + j;
                c0[seg * kmax..seg * kmax + constants[k].len()].copy_from_slice(&constants[k]);
                if j < coords[k].len() {
                    cd[seg] = coords[k][j];
                    lv[seg] = levels[k][j];
                }
            }
        }
        let raw = self.max_scan(plans, maxn, &c0, Some(&cd), Some(&lv))?;
        let mut out: Vec<Vec<Option<crate::fit::LevelInterval>>> = Vec::with_capacity(nsets);
        for k in 0..nsets {
            let mut per = Vec::with_capacity(coords[k].len());
            for j in 0..coords[k].len() {
                let r = raw[k * maxn + j];
                per.push(if r.status == 0 {
                    Some(crate::fit::LevelInterval { lo: r.lo, hi: r.hi })
                } else {
                    None
                });
            }
            out.push(per);
        }
        Some(out)
    }

    /// #227: the squared LM warm start followed by the exact minimax polish,
    /// on the device.
    ///
    /// This is the GPU counterpart of `fit_weighted_metric(.., Metric::Maximum)`
    /// for the tail it can prove: the same squared start, then the exact
    /// one-dimensional minimiser per constant by bisection on the error level,
    /// and the same exact objective comparison. The reported `sse` field carries
    /// the exact `max |r|` (the configured objective), never an SSE.
    ///
    /// A candidate whose polish cannot be scored, or whose prediction is not
    /// finite, is reported with a non-zero status so the caller refits it with
    /// the exact CPU arm and labels the fallback. `iters == 0` is a true no-op:
    /// the incoming constants are returned unchanged with status 0.
    pub(crate) fn max_fit_batch(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        iters: usize,
    ) -> (Vec<GpuFitOutput>, f32) {
        let incoming: Vec<Vec<f64>> = plans.iter().map(|p| p.c0.clone()).collect();
        if iters == 0 {
            // `--no-fit`: no LM step and no polish.
            let out = incoming
                .iter()
                .map(|c| GpuFitOutput {
                    c: c.clone(),
                    sse: f64::NAN,
                    status: 0,
                })
                .collect();
            return (out, 0.0);
        }
        let (mut out, lm_ms) = self.fit_batch(plans, iters, crate::fit::DeviceLoss::Squared);
        if out.is_empty() || out.len() != plans.len() {
            return (out, -1.0);
        }
        let idx: Vec<usize> = (0..out.len())
            .filter(|&k| crate::fit::fit_status_usable(out[k].status) && out[k].c.iter().all(|v| v.is_finite()))
            .collect();
        if idx.is_empty() {
            return (out, lm_ms);
        }
        let sub: Vec<&crate::fit::GpuFitPlan> = idx.iter().map(|&k| plans[k]).collect();
        let starts: Vec<Vec<f64>> = idx.iter().map(|&k| out[k].c.clone()).collect();
        let started = std::time::Instant::now();
        let Some(res) = crate::fit::polish_minimax_batch(self, &sub, &starts, iters.max(1))
        else {
            for &k in &idx {
                out[k].status = 3; // the labelled `maximum_declined` fallback
            }
            return (out, -1.0);
        };
        // The CPU arm keeps the incoming constants unless the polished squared
        // start is strictly better on the configured objective. Score them in
        // one batch so the device arm obeys the same monotone contract.
        let base_trials: Vec<Vec<Vec<f64>>> = idx
            .iter()
            .map(|&k| vec![incoming[k].clone()])
            .collect();
        let baseline = self.max_score_trials(&sub, &base_trials);
        let polish_ms = started.elapsed().as_secs_f32() * 1000.0;
        for (j, &k) in idx.iter().enumerate() {
            let c = &res.constants[j];
            if c.len() != plans[k].nconst || !c.iter().all(|v| v.is_finite()) {
                out[k].status = 1;
                continue;
            }
            if !res.scores[j].is_finite() {
                out[k].status = 2;
                continue;
            }
            let start_score = baseline
                .as_ref()
                .and_then(|b| b.get(j))
                .and_then(|t| t.first())
                .copied()
                .flatten()
                .map(|s| s.value)
                .unwrap_or(f64::INFINITY);
            if start_score.is_finite() && res.scores[j] >= start_score {
                out[k].c = incoming[k].clone();
                out[k].sse = start_score;
                continue;
            }
            out[k].c = c.clone();
            out[k].sse = res.scores[j];
        }
        (out, lm_ms + polish_ms)
    }
}

/// #227: the device is the `MinimaxTrialScorer` the batched exact minimax
/// polish drives. Every call returns an order-free statistic (the exact
/// maximum, its lowest maximiser, and the exact level interval), so the polish
/// cannot observe a reduction order or a scheduling artefact.
impl crate::fit::MinimaxTrialScorer for GpuEvaluator {
    fn score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<crate::fit::MaxStat>>>> {
        self.max_score_trials(plans, trials)
    }

    fn intervals(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        constants: &[Vec<f64>],
        coords: &[Vec<i32>],
        levels: &[Vec<f64>],
    ) -> Option<Vec<Vec<Option<crate::fit::LevelInterval>>>> {
        self.max_intervals(plans, constants, coords, levels)
    }
}

/// #226/#232: the device is the `OrderStatTrialScorer` the batched exact
/// median and interquartile polishes drive.
impl crate::fit::OrderStatTrialScorer for GpuEvaluator {
    fn score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
        want: crate::fit::OrderStat,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        self.order_stat_score_trials(plans, trials, want)
    }
}

/// #230: the device is the `RankTrialScorer` the batched exact-rank polish
/// drives. Every call returns the integer sums of the rank statistic, so the
/// polish cannot observe a floating-point reduction order.
impl crate::fit::RankTrialScorer for GpuEvaluator {
    fn score_trials(
        &mut self,
        plans: &[&crate::fit::GpuFitPlan],
        trials: &[Vec<Vec<f64>>],
    ) -> Option<Vec<Vec<Option<crate::metrics::RankSums>>>> {
        self.rank_score_trials(plans, trials)
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

    /// #121: a mixed population keeps its order and reports the
    /// native/lowered/unsupported split. The two GPU programs must score in
    /// insertion order, exactly where the caller will merge them.
    #[test]
    fn mixed_population_keeps_order_and_counts_lowered() {
        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let native = node(Op::If, vec![input(0), constant(1.0), constant(2.0)]);
        let lowered = node(Op::Asinh, vec![input(0)]);
        let unsupported = node(Op::Require, vec![trees[0].clone()]);
        let mut pop = Population::new();
        assert!(pop.add(&trees[0]));
        assert!(!pop.add(&unsupported));
        assert!(pop.add(&native));
        assert!(pop.add(&lowered));
        assert_eq!(pop.len(), 3);
        assert_eq!(pop.native(), 2, "the fixture tree and `if` are accepted verbatim");
        assert_eq!(pop.lowered(), 1, "asinh needs the lowering pass");
        assert_eq!(pop.unsupported(), 1);
        let (sse, ms) = ev.eval_population(&pop);
        assert!(ms >= 0.0);
        assert_eq!(sse.len(), 3);
        for (slot, tree) in [&trees[0], &native, &lowered].iter().enumerate() {
            let cpu = cpu_sse(tree, &cols, n, nv, &y);
            let rel = (sse[slot] as f64 - cpu).abs() / (1.0 + cpu.abs());
            assert!(
                rel < 1e-4,
                "slot {slot}: GPU {} vs CPU {} (relative {rel:.2e})",
                sse[slot],
                cpu
            );
        }
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
    fn pool_kernel_implements_every_emitted_opcode() {
        // #122: the fixed-slot kernel used to fall through to `r = 0` for
        // `min`/`max`/`mod`/comparisons/`if`. A program using them must score
        // exactly like the packed kernel.
        use crate::expr::{constant, input, node};
        let (cols, n, nv, y, _) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let x = || input(0);
        let trees = vec![
            node(Op::If, vec![
                node(Op::Greater, vec![x(), constant(0.0)]),
                node(Op::Min, vec![x(), constant(0.5)]),
                node(Op::Mod, vec![x(), constant(0.7)]),
            ]),
            node(Op::Max, vec![node(Op::Mul, vec![x(), x()]), constant(0.1)]),
            node(Op::Atan2, vec![x(), constant(2.0)]),
            node(Op::Erf, vec![node(Op::Tanh, vec![x()])]),
        ];
        let width = 32;
        let mut pool = crate::genome::GenomePool::new(trees.len(), width);
        for tree in &trees {
            let mut words = Vec::new();
            assert!(compile(tree, &mut words));
            assert!(pool.push(&words, crate::genome::GenomeMeta::default()));
        }
        let (slot_sse, ms) = ev.eval_pool(&pool);
        assert!(ms >= 0.0);
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

    /// #287: every native score objective matches the CPU reducer on the same
    /// prediction vector. The kernel sees f32 rows; the reference receives the
    /// identical f32 values widened to f64, so only reduction order can differ.
    #[test]
    fn native_score_objectives_match_the_cpu_reducer() {
        use crate::metrics::{score_native, Metric};

        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        let preds = ev.predictions(prog, offsets).expect("prediction readback");
        let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let objectives = [
            (Metric::Squared, ScoreObjective::Squared),
            (Metric::Absolute, ScoreObjective::Absolute),
            (Metric::LogSquashed, ScoreObjective::LogSquashed),
            (Metric::Signed, ScoreObjective::Signed),
            (Metric::Maximum, ScoreObjective::Maximum),
            (Metric::Correlation, ScoreObjective::Correlation),
            (Metric::CorrelationSigned, ScoreObjective::CorrelationSigned),
            (Metric::Aic, ScoreObjective::Aic),
            (Metric::Aicc, ScoreObjective::Aicc),
            (Metric::Bic, ScoreObjective::Bic),
        ];
        let pcount = vec![2i32; trees.len()];
        for (metric, objective) in objectives {
            let batch = ev.score_batch(prog, offsets, objective, &pcount);
            assert!(batch.ms >= 0.0, "{} launch", objective.name());
            assert!(batch.is_consistent());
            for k in 0..trees.len() {
                let want = score_native(
                    &preds[k * n..(k + 1) * n]
                        .iter()
                        .map(|&v| v as f64)
                        .collect::<Vec<_>>(),
                    &y64,
                    objective,
                    2,
                );
                let got = batch.values[k];
                assert_eq!(batch.status[k], ScoreStatus::Ok, "{} slot {k}", objective.name());
                if objective == ScoreObjective::Maximum {
                    assert_eq!(got.to_bits(), want.value.to_bits(), "{} slot {k}", metric.name());
                } else if want.value.is_infinite() {
                    assert_eq!(got, want.value, "{} slot {k}", metric.name());
                } else if objective.is_correlation() {
                    assert!(
                        (got - want.value).abs() <= 1e-9,
                        "{} slot {k}: GPU {got} vs CPU {}",
                        metric.name(),
                        want.value
                    );
                } else {
                    let tol = 1e-12 * (1.0 + want.value.abs());
                    assert!(
                        (got - want.value).abs() <= tol,
                        "{} slot {k}: GPU {got} vs CPU {} (tol {tol:e})",
                        metric.name(),
                        want.value
                    );
                }
            }
        }
    }

    /// The device `Absolute` scorer, chosen by the dispatch table for
    /// `implicit_residual`, equals the CPU `score_weighted(ImplicitResidual)` on an
    /// implicit circle sample (target 0) at the absolute metric's tier.
    #[test]
    fn implicit_residual_device_score_matches_the_cpu_metric() {
        use crate::metrics::{native_score_objective, score_weighted, Metric};

        let n = 256;
        let nv = 2;
        let mut cols = vec![0.0f32; nv * n];
        for i in 0..n {
            let a = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            let r = 1.0 + 0.05 * ((i * 7 % 13) as f32 / 13.0 - 0.5);
            cols[i] = r * a.cos();
            cols[n + i] = r * a.sin();
        }
        let y = vec![0.0f32; n];
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let circle = |c: f64| {
            node(
                Op::Sub,
                vec![
                    node(
                        Op::Add,
                        vec![
                            node(Op::Square, vec![input(0)]),
                            node(Op::Square, vec![input(1)]),
                        ],
                    ),
                    constant(c),
                ],
            )
        };
        let trees = vec![circle(1.0), circle(0.0), circle(2.5)];
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        let preds = ev.predictions(prog, offsets).expect("prediction readback");
        let objective = native_score_objective(Metric::ImplicitResidual)
            .expect("implicit_residual is natively scored");
        assert_eq!(objective, ScoreObjective::Absolute);
        let batch = ev.score_batch(prog, offsets, objective, &[1i32; 3]);
        assert!(batch.is_consistent());
        let zeros = vec![0.0f64; n];
        for k in 0..trees.len() {
            let pred: Vec<f64> = preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
            let want = score_weighted(&pred, &zeros, &[], Metric::ImplicitResidual, 1);
            assert_eq!(batch.status[k], ScoreStatus::Ok, "slot {k}");
            let tol = 1e-12 * (1.0 + want.abs());
            assert!(
                (batch.values[k] - want).abs() <= tol,
                "slot {k}: GPU {} vs CPU {want}",
                batch.values[k]
            );
        }
    }

    /// Weighted regression scoring: every native regression objective with a
    /// weighted form matches `metrics::score_weighted` on the same f32-rounded
    /// predictions, at the unweighted tiers (1e-12 relative, 1e-9 for the
    /// correlation family), for uniform, non-uniform and zero-heavy weights.
    /// All-zero weights are status 4, a weighted maximum launch is refused, and
    /// clearing the weights restores the unweighted scores.
    #[test]
    fn weighted_regression_scores_match_the_cpu_weighted_score() {
        use crate::metrics::{score_native_weighted, score_weighted, Metric};

        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        let preds = ev.predictions(prog, offsets).expect("prediction readback");
        let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let objectives = [
            (Metric::Squared, ScoreObjective::Squared),
            (Metric::Absolute, ScoreObjective::Absolute),
            (Metric::LogSquashed, ScoreObjective::LogSquashed),
            (Metric::Signed, ScoreObjective::Signed),
            (Metric::Correlation, ScoreObjective::Correlation),
            (Metric::CorrelationSigned, ScoreObjective::CorrelationSigned),
            (Metric::Aic, ScoreObjective::Aic),
            (Metric::Aicc, ScoreObjective::Aicc),
            (Metric::Bic, ScoreObjective::Bic),
        ];
        let uniform = vec![1.0f64; n];
        let ramp: Vec<f64> = (0..n).map(|i| 0.25 + (i % 9) as f64 * 0.5).collect();
        let zeros: Vec<f64> = (0..n)
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 + (i % 4) as f64 * 0.25 })
            .collect();
        let scaled: Vec<f64> = ramp.iter().map(|w| w * 1000.0).collect();
        let pcount = vec![2i32; trees.len()];
        for (label, w) in [
            ("uniform", &uniform),
            ("non-uniform", &ramp),
            ("zero weights", &zeros),
            ("scaled", &scaled),
        ] {
            assert!(ev.set_score_weights(w), "{label}: weights uploaded");
            for (metric, objective) in objectives {
                let batch = ev.score_batch(prog, offsets, objective, &pcount);
                assert!(batch.ms >= 0.0, "{label} {}: launch", objective.name());
                for k in 0..trees.len() {
                    let rows: Vec<f64> = preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
                    let want = score_weighted(&rows, &y64, w, metric, 2);
                    let native = score_native_weighted(&rows, &y64, w, objective, 2);
                    assert!(
                        (native.value - want).abs() <= 1e-14 * (1.0 + want.abs()),
                        "{label} {} slot {k}: reducers agree",
                        metric.name()
                    );
                    let got = batch.values[k];
                    assert_eq!(batch.status[k], ScoreStatus::Ok, "{label} {} slot {k}", metric.name());
                    let tol = if objective.is_correlation() {
                        1e-9
                    } else {
                        1e-12 * (1.0 + want.abs())
                    };
                    assert!(
                        (got - want).abs() <= tol,
                        "{label} {} slot {k}: GPU {got} vs CPU {want} (tol {tol:e})",
                        metric.name()
                    );
                }
            }
        }

        let none = vec![0.0f64; n];
        assert!(ev.set_score_weights(&none));
        for (metric, objective) in objectives {
            let batch = ev.score_batch(prog, offsets, objective, &pcount);
            assert!(batch.ms >= 0.0, "all-zero {}: launch", objective.name());
            for k in 0..trees.len() {
                assert_eq!(batch.status[k], ScoreStatus::AllZeroWeight, "all-zero {} slot {k}", metric.name());
                assert_eq!(batch.values[k], f64::INFINITY, "all-zero {} slot {k}", metric.name());
            }
        }

        assert!(ev.set_score_weights(&ramp));
        let refused = ev.score_batch(prog, offsets, ScoreObjective::Maximum, &pcount);
        assert!(refused.ms < 0.0, "a weighted maximum launch is refused");

        assert!(ev.set_score_weights(&[]));
        for (metric, objective) in objectives {
            let batch = ev.score_batch(prog, offsets, objective, &pcount);
            for k in 0..trees.len() {
                let rows: Vec<f64> = preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
                let want = score_native_weighted(&rows, &y64, &[], objective, 2).value;
                let tol = if objective.is_correlation() {
                    1e-9
                } else {
                    1e-12 * (1.0 + want.abs())
                };
                assert!(
                    (batch.values[k] - want).abs() <= tol,
                    "cleared {} slot {k}: GPU {} vs CPU {want}",
                    metric.name(),
                    batch.values[k]
                );
            }
        }
    }

    /// #287: a non-finite prediction is `+inf` with status 2 on the device, not
    /// the old finite `1e30/n` sentinel, for every native objective.
    #[test]
    fn native_score_nonfinite_is_inf_with_status() {
        let n = 16;
        let nv = 1;
        let cols: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let y = vec![0.0f32; n];
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let trees = vec![
            constant(f64::NAN),
            node(Op::Div, vec![constant(1.0), constant(0.0)]),
        ];
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        for objective in [
            ScoreObjective::Squared,
            ScoreObjective::Absolute,
            ScoreObjective::LogSquashed,
            ScoreObjective::Signed,
            ScoreObjective::Maximum,
            ScoreObjective::Correlation,
            ScoreObjective::CorrelationSigned,
            ScoreObjective::Aic,
            ScoreObjective::Aicc,
            ScoreObjective::Bic,
        ] {
            let pcount = vec![0i32; pop.len()];
            let batch = ev.score_batch(prog, offsets, objective, &pcount);
            assert!(batch.ms >= 0.0);
            for k in 0..pop.len() {
                assert_eq!(batch.status[k], ScoreStatus::NonFinite, "{}", objective.name());
                assert_eq!(batch.values[k], f64::INFINITY, "{}", objective.name());
                assert!(batch.values[k].is_infinite());
            }
        }
    }

    /// #422: a finite prediction of any f32 magnitude is scored like the CPU
    /// scores the same rows; only a non-finite prediction is `+inf`. The rows
    /// straddle 1e30 and reach the f32 maximum; one candidate overflows f32 to
    /// `inf` and is `+inf` with status 2 on both sides. Repeats are
    /// bit-identical.
    #[test]
    fn large_finite_predictions_score_like_the_cpu() {
        use crate::metrics::{score_weighted, Metric};

        let (cols, n, nv, y, _) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let scaled = |c: f64| node(Op::Mul, vec![input(0), constant(c)]);
        let trees = vec![
            scaled(1.5e30),
            scaled(1e30),
            scaled(4e30),
            scaled(1e35),
            scaled(3e38),
            node(Op::Add, vec![scaled(3e38), scaled(3e38)]),
        ];
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        let preds = ev.predictions(prog, offsets).expect("prediction readback");
        let y64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let objectives = [
            (Metric::Squared, ScoreObjective::Squared),
            (Metric::Absolute, ScoreObjective::Absolute),
            (Metric::LogSquashed, ScoreObjective::LogSquashed),
            (Metric::Signed, ScoreObjective::Signed),
            (Metric::Maximum, ScoreObjective::Maximum),
            (Metric::Correlation, ScoreObjective::Correlation),
            (Metric::CorrelationSigned, ScoreObjective::CorrelationSigned),
            (Metric::Aic, ScoreObjective::Aic),
            (Metric::Aicc, ScoreObjective::Aicc),
            (Metric::Bic, ScoreObjective::Bic),
        ];
        let pcount = vec![2i32; trees.len()];
        let uniform = vec![1.0f64; n];
        for weights in [&[][..], &uniform[..]] {
            assert!(ev.set_score_weights(weights));
            for (metric, objective) in objectives {
                if objective == ScoreObjective::Maximum && !weights.is_empty() {
                    continue;
                }
                let batch = ev.score_batch(prog, offsets, objective, &pcount);
                let again = ev.score_batch(prog, offsets, objective, &pcount);
                assert!(batch.ms >= 0.0, "{} launch", metric.name());
                for k in 0..trees.len() {
                    let rows: Vec<f64> =
                        preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
                    let want = score_weighted(&rows, &y64, weights, metric, 2);
                    let got = batch.values[k];
                    let what = format!("{} slot {k} weights {}", metric.name(), weights.len());
                    assert_eq!(got.to_bits(), again.values[k].to_bits(), "{what}: repeat");
                    if !want.is_finite() {
                        assert_eq!(batch.status[k], ScoreStatus::NonFinite, "{what}");
                        assert_eq!(got, f64::INFINITY, "{what}");
                        continue;
                    }
                    assert_eq!(batch.status[k], ScoreStatus::Ok, "{what}");
                    if objective == ScoreObjective::Maximum {
                        assert_eq!(got.to_bits(), want.to_bits(), "{what}");
                    } else if objective.is_correlation() {
                        assert!((got - want).abs() <= 1e-9, "{what}: GPU {got} vs CPU {want}");
                    } else {
                        let tol = 1e-9 * (1.0 + want.abs());
                        assert!((got - want).abs() <= tol, "{what}: GPU {got} vs CPU {want}");
                    }
                }
            }
        }
    }

    /// #287: the same candidate alone, in a batch of 7 and through the
    /// fixed-slot arena produces identical bits (two runs included).
    #[test]
    fn native_score_is_composition_free_and_repeatable() {
        use crate::genome::{GenomeMeta, GenomePool};

        let (cols, n, nv, y, trees) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let mut seven = Vec::new();
        for i in 0..7 {
            seven.push(trees[i % trees.len()].clone());
        }
        let pop = Population::from_trees(seven.iter());
        let (prog, offsets) = pop.programs();
        let pcount = vec![1i32; seven.len()];
        // A 447-candidate composition exercises the widest audited batch while
        // keeping the same reduction tree for the same n.
        let many_trees: Vec<Node> = (0..447).map(|i| trees[i % trees.len()].clone()).collect();
        let many_pop = Population::from_trees(many_trees.iter());
        let (many_prog, many_offsets) = many_pop.programs();
        let many_pcount = vec![1i32; many_trees.len()];
        for objective in [
            ScoreObjective::Squared,
            ScoreObjective::Absolute,
            ScoreObjective::LogSquashed,
            ScoreObjective::Signed,
            ScoreObjective::Maximum,
            ScoreObjective::Correlation,
            ScoreObjective::CorrelationSigned,
            ScoreObjective::Aic,
            ScoreObjective::Aicc,
            ScoreObjective::Bic,
        ] {
            let all = ev.score_batch(prog, offsets, objective, &pcount);
            let again = ev.score_batch(prog, offsets, objective, &pcount);
            assert!(all.all_ok(), "{}", objective.name());
            for k in 0..seven.len() {
                let one_start = offsets[k] as usize;
                let one_end = offsets[k + 1] as usize;
                let one = ev.score_batch(
                    &prog[one_start..one_end],
                    &[0, (one_end - one_start) as u32],
                    objective,
                    &[1],
                );
                assert_eq!(
                    all.values[k].to_bits(),
                    one.values[0].to_bits(),
                    "{} slot {k} alone vs batch",
                    objective.name()
                );
                assert_eq!(
                    all.values[k].to_bits(),
                    again.values[k].to_bits(),
                    "{} slot {k} repeat",
                    objective.name()
                );
            }
            // The fixed-slot pool must produce the same bits as the packed
            // batch (same n -> same reduction tree).
            let width = 32;
            let mut pool = GenomePool::new(seven.len(), width);
            let mut words = Vec::new();
            for tree in &seven {
                words.clear();
                assert!(compile(tree, &mut words));
                assert!(pool.push(&words, GenomeMeta::default()));
            }
            let slot = ev.score_pool(&pool, objective, &pcount);
            assert!(slot.all_ok(), "{} pool", objective.name());
            for k in 0..seven.len() {
                assert_eq!(
                    slot.values[k].to_bits(),
                    all.values[k].to_bits(),
                    "{} slot {k} pool vs packed",
                    objective.name()
                );
            }
            // One candidate in a batch of 447 must keep the bits it had alone
            // and in the batch of 7 (same n, same fixed reduction tree).
            let many = ev.score_batch(many_prog, many_offsets, objective, &many_pcount);
            assert!(many.all_ok(), "{} 447", objective.name());
            assert_eq!(
                many.values[0].to_bits(),
                all.values[0].to_bits(),
                "{} slot 0: batch of 447 vs batch of 7",
                objective.name()
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
    /// #121: the device backstop. A program that bypasses the host gate (here
    /// by writing straight into the genome pool) must be scored as a hard
    /// failure, not corrupt the local stack. Before the guard this wrote one
    /// float past `s` for every row.
    #[test]
    fn device_stack_guard_rejects_an_over_deep_program() {
        let (cols, n, nv, y, _) = fixture();
        let Some(mut ev) = evaluator_or_skip(&cols, n, nv, &y) else {
            return;
        };
        let mut pool = crate::genome::GenomePool::new(1, 2 * (RC_MAX_STACK + 1) + 1);
        pool.begin_generation();
        let mut words = Vec::new();
        // One push past RC_MAX_STACK: 65 constants with no consumer.
        for _ in 0..=RC_MAX_STACK {
            words.push(OP_CONST);
            words.push(1.0f32.to_bits());
        }
        words.push(OP_HALT);
        assert!(pool.push(&words, Default::default()));
        pool.commit();
        let (sse, ms) = ev.eval_pool(&pool);
        assert!(ms >= 0.0, "the guarded batch must not fail the launch");
        assert_eq!(sse.len(), 1);
        // Every row is a hard failure (1e30), so the SSE is n * 1e30.
        let want = n as f32 * 1e30f32;
        assert!(
            ((sse[0] - want) / want).abs() < 1e-5,
            "expected a hard-failure SSE of {want}, got {}",
            sse[0]
        );
    }

    /// #121: a deep tree that needs lowering is still rejected statically
    /// before any word is emitted (the lowering pass can add nodes).
    #[test]
    fn deep_lowered_tree_is_rejected_before_emission() {
        // asinh expands into ~10 nodes; wrapping it 8 times raises the peak
        // stack past the host gate.
        let mut t = input(0);
        for _ in 0..8 {
            t = node(Op::Asinh, vec![t]);
        }
        if peak_stack(&lower(&t)) <= HOST_MAX_STACK {
            // 8 nested expansions are not deep enough on this build; use a
            // right-deep add chain, which grows one stack slot per level.
            let mut deep = input(0);
            for _ in 0..HOST_MAX_STACK {
                deep = node(Op::Add, vec![input(1), deep]);
            }
            assert!(peak_stack(&deep) > HOST_MAX_STACK);
            let mut pop = Population::new();
            assert!(!pop.add(&deep));
            assert_eq!(pop.unsupported(), 1);
            return;
        }
        let mut pop = Population::new();
        assert!(!pop.add(&t), "an over-deep lowered tree must not be emitted");
        assert_eq!(pop.unsupported(), 1);
        assert_eq!(pop.lowered(), 0, "nothing may be counted as emitted");
    }

    /// #121: the host stack constant must match `RC_MAX_STACK` in the kernel.
    #[test]
    fn stack_limits_match_the_kernel() {
        let src = include_str!("cuda/eval.cu");
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with("#define RC_MAX_STACK"))
            .expect("the kernel defines RC_MAX_STACK");
        let value: usize = line
            .split_whitespace()
            .nth(2)
            .expect("a value")
            .parse()
            .expect("a number");
        assert_eq!(value, RC_MAX_STACK);
        assert!(HOST_MAX_STACK < RC_MAX_STACK);
    }

    /// #131: every evaluator kernel carries the same occupancy contract. On
    /// `sm_86` an SM holds 1536 threads (48 warps) and at most 16 blocks, so
    /// six blocks of `RC_MAX_THREADS` is full theoretical occupancy. The
    /// launch bounds make ptxas derive the matching ~40-register ceiling; the
    /// measured resource table lives in `docs/gpu-backend.md`.
    #[test]
    fn evaluator_kernels_carry_the_sm86_occupancy_contract() {
        let src = include_str!("cuda/eval.cu");
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with("#define RC_MAX_THREADS"))
            .expect("the kernel defines RC_MAX_THREADS");
        let max_threads: usize = line
            .split_whitespace()
            .nth(2)
            .expect("a value")
            .parse()
            .expect("a number");
        assert_eq!(max_threads, 256, "the contract was measured at 256 threads");
        assert_eq!(
            6 * max_threads,
            1536,
            "six blocks must fill all 48 warps of an sm_86 SM"
        );
        for kernel in ["eval_kernel", "eval_pool_kernel", "predict_kernel"] {
            let marker = format!("__launch_bounds__(RC_MAX_THREADS, 6) {kernel}");
            assert!(
                src.contains(&marker),
                "{kernel} lost its __launch_bounds__(RC_MAX_THREADS, 6) contract"
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

/// #169: correctness of the batched GPU Levenberg–Marquardt arm.
///
/// These tests compare the kernel against the historical CPU LM on the same
/// starting constants and rows. The reported SSE is recomputed on the CPU in
/// f64 at the fitted constants, so the comparison measures the constants rather
/// than the kernel's f32 row arithmetic.
#[cfg(test)]
mod fit_tests {
    use super::*;
    use crate::expr::{constant, eval, input, node, Node, Op};

    /// One basis function with no fitted constants of its own.
    fn basis(k: usize) -> Node {
        let x = input(0);
        match k {
            0 => constant(1.0),
            1 => x,
            2 => node(Op::Square, vec![x]),
            3 => node(Op::Cube, vec![x]),
            4 => node(Op::Sin, vec![x]),
            5 => node(Op::Cos, vec![x]),
            6 => node(Op::Tanh, vec![x]),
            7 => node(Op::Logistic, vec![x]),
            8 => node(Op::Gauss, vec![x]),
            9 => node(Op::Erf, vec![x]),
            10 => node(Op::Sinh, vec![x]),
            11 => node(Op::Cosh, vec![x]),
            12 => node(Op::Abs, vec![x]),
            13 => node(Op::Mul, vec![x.clone(), node(Op::Sin, vec![x])]),
            14 => node(Op::Mul, vec![x.clone(), node(Op::Cos, vec![x])]),
            _ => node(Op::Mul, vec![x.clone(), node(Op::Tanh, vec![x])]),
        }
    }

    /// `Σ coeffs[k] · basis(k)`, one constant per term.
    fn model(coeffs: &[f64]) -> Node {
        let mut terms = Vec::new();
        for (k, &c) in coeffs.iter().enumerate() {
            if k == 0 {
                // The intercept is a bare constant; multiplying it by a
                // constant-one basis would add a second (unidentified) slot.
                terms.push(constant(c));
            } else {
                terms.push(node(Op::Mul, vec![constant(c), basis(k)]));
            }
        }
        let mut tree = terms.pop().expect("at least one term");
        while let Some(t) = terms.pop() {
            tree = node(Op::Add, vec![t, tree]);
        }
        tree
    }

    fn dataset(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let a = (i as f64 / (n - 1) as f64) * 5.0 - 2.5;
            x.push(vec![a]);
        }
        let y = x.iter().map(|r| 0.0 + eval(&model(&[1.5]), r)).collect();
        (x, y)
    }

    fn cols_from(x: &[Vec<f64>], y: &[f64]) -> (Vec<f32>, Vec<f32>) {
        let n = x.len();
        let mut cols = vec![0.0f32; n];
        for i in 0..n {
            cols[i] = x[i][0] as f32;
        }
        (cols, y.iter().map(|&v| v as f32).collect())
    }

    fn constants(tree: &Node) -> Vec<f64> {
        fn walk(n: &Node, out: &mut Vec<f64>) {
            if n.op == Op::Const {
                out.push(n.val);
            }
            for k in &n.kids {
                walk(k, out);
            }
        }
        let mut out = Vec::new();
        walk(tree, &mut out);
        out
    }

    fn sse_at(tree: &mut Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
        crate::fit::fit_weighted(tree, x, y, &[], 0)
    }

    /// A deliberately well-conditioned design: one independent input column per
    /// constant (sinusoids at distinct frequencies over a full period, so the
    /// columns are orthogonal). The optimum is unique and the CPU/GPU constants
    /// are comparable at 1e-5.
    fn linear_fixture(
        p: usize,
        n: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>) {
        let target: Vec<f64> = (0..p).map(|k| df64(1.0 + 0.31 * k as f64)).collect();
        let start: Vec<f64> = (0..p).map(|k| df64(0.25 + 0.07 * k as f64)).collect();
        let mut x = vec![vec![0.0; p]; n];
        let mut cols = vec![0.0f32; n * p];
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
            let mut v = 0.0;
            for k in 0..p {
                let a = df64(((k + 1) as f64 * t).sin());
                x[i][k] = a;
                cols[k * n + i] = a as f32;
                v += target[k] * a;
            }
            let noise = (((i * 37) % 11) as f64 - 5.0) * 2e-4;
            y.push(v + noise);
        }
        let y32 = y.iter().map(|&v| v as f32).collect();
        (x, y, cols, y32, target, start)
    }

    fn df64(v: f64) -> f64 {
        v
    }

    fn linear_model(coeffs: &[f64]) -> Node {
        let mut terms = Vec::new();
        for (k, &c) in coeffs.iter().enumerate() {
            terms.push(node(Op::Mul, vec![constant(c), input(k)]));
        }
        let mut tree = terms.pop().expect("at least one term");
        while let Some(t) = terms.pop() {
            tree = node(Op::Add, vec![t, tree]);
        }
        tree
    }

    #[test]
    fn gpu_fit_matches_the_cpu_lm_for_p_1_to_20() {
        let n = 800;
        let mut errors: Vec<f64> = Vec::new();
        let mut worst_sse_rel = 0.0f64;
        for p in 1..=20usize {
            let (x, y, cols, y32, _target, start) = linear_fixture(p, n);
            let mut ev = match GpuEvaluator::new(&cols, n, p, &y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU test: no usable CUDA device");
                    return;
                }
            };

            let mut cols64 = vec![0.0f64; n * p];
            for i in 0..n {
                for k in 0..p {
                    cols64[k * n + i] = x[i][k];
                }
            }
            assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

            let mut cpu_tree = linear_model(&start);
            crate::fit::fit_weighted(&mut cpu_tree, &x, &y, &[], 8);
            let cpu_sse = crate::fit::fit_weighted(&mut cpu_tree, &x, &y, &[], 8);
            let cpu_c = constants(&cpu_tree);

            let mut gpu_tree = linear_model(&start);
            let plan = crate::fit::gpu_fit_plan(&gpu_tree).expect("well-conditioned plan");
            assert_eq!(plan.nconst, p, "p={}", p);
            let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Squared);
            assert!(ms >= 0.0, "fit kernel ran for p={}", p);
            assert_eq!(out.len(), 1);
            let res = &out[0];
            assert_eq!(res.status, 0, "p={} must be well conditioned", p);
            let plan = crate::fit::gpu_fit_plan(&gpu_tree).expect("plan");
            plan.apply(&mut gpu_tree, &res.c);
            let gpu_c = constants(&gpu_tree);
            assert_eq!(gpu_c.len(), p);

            let mut gpu_sse_tree = gpu_tree.clone();
            let gpu_sse = sse_at(&mut gpu_sse_tree, &x, &y);
            let rel = (gpu_sse - cpu_sse).abs() / cpu_sse.abs().max(1e-8);
            worst_sse_rel = worst_sse_rel.max(rel);
            for (a, b) in gpu_c.iter().zip(&cpu_c) {
                errors.push((a - b).abs() / (1.0 + b.abs()));
            }
            assert!(
                rel <= 1e-8,
                "p={}: SSE relative {:.3e} (gpu {:.12e}, cpu {:.12e})",
                p,
                rel,
                gpu_sse,
                cpu_sse
            );
            let worst_c = gpu_c
                .iter()
                .zip(&cpu_c)
                .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
                .fold(0.0f64, f64::max);
            eprintln!("p={:2} sse_rel={:.2e} max_const_err={:.2e}", p, rel, worst_c);
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = errors[errors.len() / 2];
        assert!(
            median <= 1e-5,
            "median constant error {:.3e} (worst SSE rel {:.3e})",
            median,
            worst_sse_rel
        );
    }

    #[test]
    fn gpu_fit_matches_a_nonlinear_cpu_fit() {
        let n = 800;
        let (x, _) = dataset(n);
        let target = model(&[2.0, -0.5, 0.8, 0.0, 1.0]);
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let noise = (((i * 37) % 11) as f64 - 5.0) * 2e-4;
                eval(&target, r) + noise
            })
            .collect();
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
        let mut cpu_tree = model(&[1.0, 0.2, 0.4, 0.1, 0.5]);
        let cpu_sse = crate::fit::fit_weighted(&mut cpu_tree, &x, &y, &[], 8);
        let cpu_c = constants(&cpu_tree);

        let mut gpu_tree = model(&[1.0, 0.2, 0.4, 0.1, 0.5]);
        let plan = crate::fit::gpu_fit_plan(&gpu_tree).expect("nonlinear plan");
        let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Squared);
        assert_eq!(out[0].status, 0, "the nonlinear fixture is well conditioned");
        plan.apply(&mut gpu_tree, &out[0].c);
        let gpu_c = constants(&gpu_tree);
        assert_eq!(cpu_c.len(), 5);
        assert_eq!(gpu_c.len(), 5);
        for (a, b) in gpu_c.iter().zip(&cpu_c) {
            let e = (a - b).abs() / (1.0 + b.abs());
            assert!(e <= 1e-5, "constant error {:.3e}: {} vs {}", e, a, b);
        }
        let mut gpu_sse_tree = gpu_tree.clone();
        let gpu_sse = sse_at(&mut gpu_sse_tree, &x, &y);
        let rel = (gpu_sse - cpu_sse).abs() / cpu_sse.abs().max(1e-8);
        assert!(rel <= 1e-8, "SSE relative {:.3e}", rel);
    }

    /// #169: the cap is 20 constants. Every `p` the kernel can take is offered
    /// to it (never silently dropped), and only `p > 20` is labelled
    /// `too_many_constants` for the CPU fallback.
    #[test]
    fn the_constant_cap_is_twenty_and_p_above_it_is_labelled() {
        use crate::fit::GpuFitReject;
        assert_eq!(crate::fit::GPU_FIT_MAX_CONSTANTS, 20);
        for p in 1..=20usize {
            let m = linear_model(&vec![0.1; p]);
            let plan = crate::fit::gpu_fit_plan_reason(&m)
                .unwrap_or_else(|e| panic!("p={} must be offered to the kernel: {:?}", p, e));
            assert_eq!(plan.nconst, p);
        }
        for p in 21..=24usize {
            let m = linear_model(&vec![0.1; p]);
            assert_eq!(
                crate::fit::gpu_fit_plan_reason(&m).err(),
                Some(GpuFitReject::TooManyConstants),
                "p={} must be labelled too_many_constants",
                p
            );
        }
    }

    /// #169: the two kernel instantiations are selected by the batch's largest
    /// constant count, and the p=20 instantiation stays under 48 KB of shared
    /// memory so two blocks still fit per SM.
    #[cfg(feature = "cuda")]
    #[test]
    fn fit_kernel_instantiations_report_their_shared_footprint() {
        let small = fit_kernel_shape(4).expect("p=4 has an instantiation");
        let large = fit_kernel_shape(20).expect("p=20 has an instantiation");
        assert_eq!(small.2, 16, "p<=16 keeps the 16-constant instantiation");
        assert_eq!(large.2, 20, "17..20 use the 20-constant instantiation");
        assert_eq!(small.0, 256);
        assert_eq!(large.0, 128);
        assert!(small.1 > 0 && small.1 <= 48 * 1024, "small smem {}", small.1);
        assert!(large.1 > 0 && large.1 <= 48 * 1024, "large smem {}", large.1);
        assert!(
            fit_kernel_shape(21).is_none(),
            "p=21 is past every instantiation, so it must stay on the CPU"
        );
    }

    /// #169: a mixed batch — `p = 4` beside `p = 17..=20`, i.e. a padded
    /// `kmax = 20` — gives every candidate the same constants as fitting it
    /// alone, bit for bit, even though the small candidate then runs in the
    /// 20-constant, 128-row-chunk instantiation instead of its own. All of
    /// them still match the CPU LM to the issue's tolerances.
    #[test]
    fn a_mixed_p_batch_matches_single_fits_and_the_cpu_lm() {
        let n = 600;
        let pmax = 20usize;
        let (x, y, cols, y32) = orthogonal_fixture(pmax, n);
        let mut ev = match GpuEvaluator::new(&cols, n, pmax, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n * pmax];
        for i in 0..n {
            for k in 0..pmax {
                cols64[k * n + i] = x[i][k];
            }
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let ps = [4usize, 17, 18, 19, 20];
        let mut starts = Vec::new();
        let mut cpu_sse = Vec::new();
        let mut singles = Vec::new();
        for &p in &ps {
            let start: Vec<f64> = (0..p).map(|k| 0.25 + 0.07 * k as f64).collect();
            let mut tree = linear_model(&start);
            let sse = crate::fit::fit_weighted(&mut tree, &x, &y, &[], 8);
            cpu_sse.push(sse);
            starts.push(start);
            let plan = crate::fit::gpu_fit_plan(&linear_model(
                &(0..p).map(|k| 0.25 + 0.07 * k as f64).collect::<Vec<_>>(),
            ))
            .expect("plan");
            let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Squared);
            assert_eq!(out[0].status, 0, "p={} must be well conditioned", p);
            singles.push(out[0].clone());
        }

        // One batch: kmax is 20, so every block uses the 20-constant
        // instantiation and the padded start constants of the small candidate.
        let plans: Vec<_> = starts
            .iter()
            .map(|s| crate::fit::gpu_fit_plan(&linear_model(s)).expect("plan"))
            .collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (mixed, _) = ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::Squared);
        assert_eq!(mixed.len(), ps.len());

        let mut errors = Vec::new();
        for (k, &p) in ps.iter().enumerate() {
            assert_eq!(mixed[k].status, 0, "mixed p={}", p);
            assert_eq!(
                mixed[k].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                singles[k].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "p={} changed in a padded batch",
                p
            );
            assert_eq!(mixed[k].sse.to_bits(), singles[k].sse.to_bits());
            let mut tree = linear_model(&starts[k]);
            plans[k].apply(&mut tree, &mixed[k].c);
            let gpu_c = constants(&tree);
            let mut cpu_tree = tree.clone();
            let gpu_sse = sse_at(&mut cpu_tree, &x, &y);
            let rel = (gpu_sse - cpu_sse[k]).abs() / cpu_sse[k].abs().max(1e-8);
            assert!(
                rel <= 1e-8,
                "p={}: SSE relative {:.3e} in a mixed batch",
                p,
                rel
            );
            // Compare with the CPU constants of the same candidate.
            let mut cpu_only = linear_model(&starts[k]);
            crate::fit::fit_weighted(&mut cpu_only, &x, &y, &[], 8);
            for (a, b) in gpu_c.iter().zip(constants(&cpu_only)) {
                errors.push((a - b).abs() / (1.0 + b.abs()));
            }
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = errors[errors.len() / 2];
        assert!(median <= 1e-5, "median constant error {:.3e}", median);
    }

    /// A 20-column design: column `k` is `sin((k+1)t)`, so the columns are
    /// orthogonal over the period and every constant is separately identified.
    fn orthogonal_fixture(pmax: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f32>, Vec<f32>) {
        let target: Vec<f64> = (0..pmax).map(|k| 1.0 + 0.31 * k as f64).collect();
        let mut x = vec![vec![0.0; pmax]; n];
        let mut cols = vec![0.0f32; n * pmax];
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
            let mut v = 0.0;
            for k in 0..pmax {
                let a = ((k + 1) as f64 * t).sin();
                x[i][k] = a;
                cols[k * n + i] = a as f32;
                v += target[k] * a;
            }
            let noise = (((i * 37) % 11) as f64 - 5.0) * 2e-4;
            y.push(v + noise);
        }
        let y32 = y.iter().map(|&v| v as f32).collect();
        (x, y, cols, y32)
    }

    #[test]
    fn rank_deficient_and_non_finite_fixtures_are_labelled() {
        let n = 400;
        let (x, _) = dataset(n);
        let (cols, y32) = cols_from(&x, &vec![0.0; n]);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };

        // Two identical columns: J^T J is exactly rank deficient.
        let x0 = input(0);
        let collinear = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(0.1), x0.clone()]),
                node(Op::Mul, vec![constant(0.2), x0.clone()]),
            ],
        );
        let tree = collinear;
        let plan = crate::fit::gpu_fit_plan(&tree).expect("collinear plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Squared);
        assert!(ms >= 0.0);
        assert_eq!(out[0].status, 4, "rank-deficient system must be labelled");

        // The same gate at the largest constant count: twenty collinear
        // columns leave a rank-1 J^T J, so the ownership path at p=20 must
        // still refuse to report constants.
        let mut wide = node(Op::Mul, vec![constant(0.1), input(0)]);
        for _ in 1..20 {
            wide = node(
                Op::Add,
                vec![wide, node(Op::Mul, vec![constant(0.2), input(0)])],
            );
        }
        let plan = crate::fit::gpu_fit_plan(&wide).expect("p=20 collinear plan");
        assert_eq!(plan.nconst, 20);
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Squared);
        assert!(ms >= 0.0);
        assert_eq!(out[0].status, 4, "p=20 rank deficiency must be labelled");

        // sqrt(c - x²) leaves the real domain for |x| > sqrt(c).
        let bad = node(
            Op::Sqrt,
            vec![node(
                Op::Sub,
                vec![constant(0.25), node(Op::Square, vec![input(0)])],
            )],
        );
        let y: Vec<f64> = x.iter().map(|r| eval(&bad, r)).collect();
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = GpuEvaluator::new(&cols, n, 1, &y32).expect("second evaluator");
        let plan = crate::fit::gpu_fit_plan(&bad).expect("sqrt plan");
        let (out, _) = ev.fit_batch(&[&plan], 4, crate::fit::DeviceLoss::Squared);
        assert_eq!(out[0].status, 7, "non-finite start must be labelled log-only");

        // Unsupported (piecewise-constant) operators never reach the kernel.
        let floor = node(Op::Floor, vec![node(Op::Mul, vec![constant(0.5), input(0)])]);
        assert!(crate::fit::gpu_fit_plan(&floor).is_none());
    }


    /// #169 measurement (ignored): fit-stage throughput of the batched kernel
    /// versus the CPU LM on `P` candidates over `N = 154k` rows. Run with
    /// `cargo test --release --features cuda --lib bench_batched_fit -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_batched_fit() {
        fn nl(c: &[f64]) -> Node {
            let x = input(0);
            let trig = |coef: f64, freq: f64, cos: bool| {
                let arg = node(Op::Mul, vec![constant(freq), x.clone()]);
                node(
                    Op::Mul,
                    vec![
                        constant(coef),
                        node(if cos { Op::Cos } else { Op::Sin }, vec![arg]),
                    ],
                )
            };
            match c.len() {
                4 => node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(c[3]), x.clone()]),
                        node(
                            Op::Add,
                            vec![
                                trig(c[1], c[2], false),
                                constant(c[0]),
                            ],
                        ),
                    ],
                ),
                8 => node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(c[7]), x.clone()]),
                        node(
                            Op::Add,
                            vec![
                                trig(c[1], c[2], false),
                                node(
                                    Op::Add,
                                    vec![
                                        trig(c[3], c[4], true),
                                        node(
                                            Op::Add,
                                            vec![
                                                node(
                                                    Op::Mul,
                                                    vec![
                                                        constant(c[5]),
                                                        node(Op::Tanh, vec![node(
                                                            Op::Mul,
                                                            vec![constant(c[6]), x.clone()],
                                                        )]),
                                                    ],
                                                ),
                                                constant(c[0]),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                _ => unreachable!(),
            }
        }

        let n = 154_000usize;
        let threads = 8usize;
        let (x, _) = dataset(n);
        for p in [4usize, 8] {
            // The issue's gate is P >= 512; measure a quarter of that for the
            // 8-thread CPU wall while keeping this benchmark under a minute.
            let candidates: usize = if p == 4 { 256 } else { 64 };
            let target: Vec<f64> = (0..p).map(|k| 0.5 + 0.23 * k as f64).collect();
            let start: Vec<f64> = target.iter().map(|&v| 0.9 * v).collect();
            let target_tree = nl(&target);
            let y: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let noise = (((i * 37) % 11) as f64 - 5.0) * 2e-4;
                    eval(&target_tree, r) + noise
                })
                .collect();
            let (cols, y32) = cols_from(&x, &y);

            let t0 = std::time::Instant::now();
            let per = candidates.div_ceil(threads);
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    let x = &x;
                    let y = &y;
                    let start = &start;
                    scope.spawn(move || {
                        for _ in 0..per {
                            let mut t = nl(start);
                            crate::fit::fit_weighted(&mut t, x, y, &[], 8);
                        }
                    });
                }
            });
            let cpu = t0.elapsed().as_secs_f64();

            let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU benchmark: no usable device");
                    return;
                }
            };
            let mut cols64 = vec![0.0f64; n];
            for i in 0..n {
                cols64[i] = x[i][0];
            }
            ev.set_fit_data(&cols64, &y, &[]);
            let plan = crate::fit::gpu_fit_plan(&nl(&start)).expect("plan");
            let plans: Vec<&crate::fit::GpuFitPlan> =
                (0..candidates).map(|_| &plan).collect();
            let t1 = std::time::Instant::now();
            let (out, ms) = ev.fit_batch(&plans, 8, crate::fit::DeviceLoss::Squared);
            let gpu = t1.elapsed().as_secs_f64();
            let ok = out.iter().filter(|r| r.status == 0).count();
            eprintln!(
                "p={} N={} P={} cpu={:.3}s gpu={:.3}s (kernel {:.1}ms, ok {}/{}) fit speedup={:.1}x",
                p,
                n,
                candidates,
                cpu,
                gpu,
                ms,
                ok,
                out.len(),
                cpu / gpu.max(1e-9)
            );
        }
    }

    /// #169 measurement (ignored): fit-stage throughput by constant count, for
    /// the p=17..20 box. `p=4` and `p=16` use the 256-thread instantiation,
    /// `p=20` the 128-thread one; the shared footprint each cell ran with is
    /// printed from `fit_kernel_shape`. `RCSR_BENCH_ARMS=gpu` repeats only the
    /// device side (for an instantiation A/B). Run with
    /// `cargo test --release --features cuda --lib bench_batched_fit_by_p -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_batched_fit_by_p() {
        let arms = std::env::var("RCSR_BENCH_ARMS").unwrap_or_default();
        let do_cpu = arms.is_empty() || arms.contains("cpu");
        let cpu_threads = 8usize;
        let pmax = 20usize;
        let cells = [
            (4usize, 100_000usize, 32usize),
            (16, 100_000, 32),
            (20, 100_000, 32),
            (20, 100_000, 256),
            (4, 1_000_000, 32),
            (16, 1_000_000, 32),
            (20, 1_000_000, 32),
            (20, 1_000_000, 256),
        ];
        let mut data: Option<(usize, usize, Vec<Vec<f64>>, Vec<f64>, Vec<f32>, Vec<f32>)> = None;
        for &(p, n, candidates) in &cells {
            // The CPU arm is the slow side; skip it for the big GPU-only cells
            // and when only the device is being compared.
            let cpu_arm = do_cpu && candidates <= 32;
            if data.as_ref().map(|d| (d.0, d.1)) != Some((n, p)) {
                let (x, y, cols, y32) = bench_fixture(pmax, p, n);
                data = Some((n, p, x, y, cols, y32));
            }
            let (_, _, x, y, cols, y32) = data.as_ref().expect("fixture");
            let start: Vec<f64> = (0..p).map(|k| 0.25 + 0.07 * k as f64).collect();

            let cpu = if cpu_arm {
                let t0 = std::time::Instant::now();
                let per = candidates.div_ceil(cpu_threads);
                std::thread::scope(|scope| {
                    for _ in 0..cpu_threads {
                        let (x, y, start) = (x, y, &start);
                        scope.spawn(move || {
                            for _ in 0..per {
                                let mut t = linear_model(start);
                                crate::fit::fit_weighted(&mut t, x, y, &[], 8);
                            }
                        });
                    }
                });
                t0.elapsed().as_secs_f64()
            } else {
                f64::NAN
            };

            let mut ev = match GpuEvaluator::new(cols, n, pmax, y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU benchmark: no usable device");
                    return;
                }
            };
            let mut cols64 = vec![0.0f64; n * pmax];
            for i in 0..n {
                for k in 0..pmax {
                    cols64[k * n + i] = x[i][k];
                }
            }
            assert!(ev.set_fit_data(&cols64, y, &[]), "f64 rows uploaded");
            let plan = crate::fit::gpu_fit_plan(&linear_model(&start)).expect("plan");
            let plans: Vec<&crate::fit::GpuFitPlan> = (0..candidates).map(|_| &plan).collect();
            let t1 = std::time::Instant::now();
            let (out, ms) = ev.fit_batch(&plans, 8, crate::fit::DeviceLoss::Squared);
            let gpu = t1.elapsed().as_secs_f64();
            let mut by_status = [0usize; 3];
            for r in &out {
                if (0..3).contains(&r.status) {
                    by_status[r.status as usize] += 1;
                }
            }
            let (threads0, smem, maxc) = fit_kernel_shape(p).expect("instantiation");
            eprintln!(
                "p={} N={} P={} threads={} smem={}B maxc={} cpu={:.3}s gpu={:.3}s (kernel {:.1}ms) \
                 cpu/cand={:.4}s gpu/cand={:.4}s speedup={:.1}x ok={}/{} status0={} status1={} status2={}",
                p, n, candidates, threads0, smem, maxc, cpu, gpu, ms,
                cpu / candidates as f64, gpu / candidates as f64,
                cpu / gpu.max(1e-9), by_status[0], out.len(),
                by_status[0], by_status[1], by_status[2]
            );
        }
    }

    /// `pmax` orthogonal columns over one period; the first `p` of them carry
    /// the target, so a `p`-constant model converges instead of chasing every
    /// column.
    fn bench_fixture(
        pmax: usize,
        p: usize,
        n: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f32>, Vec<f32>) {        let target: Vec<f64> = (0..p).map(|k| 1.0 + 0.31 * k as f64).collect();
        let mut x = vec![vec![0.0; pmax]; n];
        let mut cols = vec![0.0f32; n * pmax];
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
            let mut v = 0.0;
            for k in 0..pmax {
                let a = ((k + 1) as f64 * t).sin();
                x[i][k] = a;
                cols[k * n + i] = a as f32;
                if k < p {
                    v += target[k] * a;
                }
            }
            let noise = (((i * 37) % 11) as f64 - 5.0) * 2e-4;
            y.push(v + noise);
        }
        let y32 = y.iter().map(|&v| v as f32).collect();
        (x, y, cols, y32)
    }

    /// #229 measurement (ignored): fit-stage throughput and endpoint quality for
    /// the robust arm. The CPU side is the exact IRLS path
    /// (`fit::fit_weighted_metric(.., LogSquashed)`); the GPU side is the
    /// batched weighted-LM kernel. Both fit the same candidates on the same
    /// heavy-tailed rows, so the ratio is paired. Prints wall, speed-up, the
    /// mean configured loss of each endpoint and the fraction where the GPU
    /// endpoint is no worse. Run with
    /// `cargo test --release --features cuda --lib bench_batched_fit_log_squashed -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_batched_fit_log_squashed() {
        let n = std::env::var("RCSR_BENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000usize);
        let threads = 8usize;
        let pmax = 20usize;
        for (p, candidates) in [(4usize, 64usize), (8, 64)] {
            let (x, mut y, cols, _) = bench_fixture(pmax, p, n);
            // 10 % gross outliers: the robust and least-squares optima differ.
            for i in (3..n).step_by(10) {
                y[i] += 15.0;
            }
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let start: Vec<f64> = (0..p).map(|k| 0.25 + 0.07 * k as f64).collect();
            let loss_of = |c: &[f64]| -> f64 {
                let t = linear_model(c);
                let pred: Vec<f64> = x.iter().map(|r| eval(&t, r)).collect();
                metrics::score_weighted(&pred, &y, &[], Metric::LogSquashed, p)
            };

            let t0 = std::time::Instant::now();
            let per = candidates.div_ceil(threads);
            let cpu_losses: Vec<Vec<f64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let x = &x;
                        let y = &y;
                        let start = &start;
                        scope.spawn(move || {
                            let mut out = Vec::with_capacity(per);
                            for _ in 0..per {
                                let mut t = linear_model(start);
                                crate::fit::fit_weighted_metric(
                                    &mut t,
                                    x,
                                    y,
                                    &[],
                                    8,
                                    Metric::LogSquashed,
                                );
                                out.push(constants(&t));
                            }
                            out
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("cpu fit thread"))
                    .collect()
            });
            let cpu = t0.elapsed().as_secs_f64();

            let mut ev = match GpuEvaluator::new(&cols, n, pmax, &y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU benchmark: no usable device");
                    return;
                }
            };
            let mut cols64 = vec![0.0f64; n * pmax];
            for r in 0..n {
                for k in 0..pmax {
                    cols64[k * n + r] = x[r][k];
                }
            }
            ev.set_fit_data(&cols64, &y, &[]);
            let plan = crate::fit::gpu_fit_plan(&linear_model(&start)).expect("plan");
            let plans: Vec<&crate::fit::GpuFitPlan> = (0..candidates).map(|_| &plan).collect();
            let t1 = std::time::Instant::now();
            let (out, ms) = ev.fit_batch(&plans, 8, crate::fit::DeviceLoss::LogSquashed);
            let gpu = t1.elapsed().as_secs_f64();
            let ok = out.iter().filter(|r| r.status == 0).count();

            let mut no_worse = 0usize;
            let mut cpu_sum = 0.0;
            let mut gpu_sum = 0.0;
            for (c_res, c_cpu) in out.iter().zip(&cpu_losses) {
                let g = if c_res.status == 0 {
                    loss_of(&c_res.c)
                } else {
                    f64::INFINITY
                };
                let c = loss_of(c_cpu);
                cpu_sum += c;
                if g <= c * (1.0 + 1e-6) {
                    no_worse += 1;
                }
                if g.is_finite() {
                    gpu_sum += g;
                }
            }
            let npairs = cpu_losses.len().min(out.len()).max(1);
            eprintln!(
                "log_squashed p={} N={} P={} cpu={:.3}s gpu={:.3}s (kernel {:.1}ms, status0 {}/{}) fit speedup={:.1}x mean_loss cpu={:.6e} gpu={:.6e} no_worse={}/{}",
                p,
                n,
                candidates,
                cpu,
                gpu,
                ms,
                ok,
                out.len(),
                cpu / gpu.max(1e-9),
                cpu_sum / npairs as f64,
                gpu_sum / npairs as f64,
                no_worse,
                npairs
            );
        }
    }

    /// #169 diagnostic (ignored): does the batched kernel still match the CPU
    /// LM at the hourly cell's row count? Run with
    /// `cargo test --release --features cuda --lib diagnostic_gpu_fit_matches_cpu_at_one_million_rows -- --ignored --nocapture`.    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn diagnostic_gpu_fit_matches_cpu_at_one_million_rows() {
        use crate::expr::random_tree_with;
        use crate::metrics::Metric;
        use crate::rng::Mt;
        let n = 1_050_000usize;
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
            let s = ((i as u32).wrapping_mul(2_654_435_761) ^ 21) as f64 / u32::MAX as f64;
            x.push(vec![t, s * 2.0 - 1.0]);
            y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2);
        }
        let mut cols = vec![0.0f32; n * 2];
        for (r, row) in x.iter().enumerate() {
            cols[r] = row[0] as f32;
            cols[n + r] = row[1] as f32;
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let mut ev = match GpuEvaluator::new(&cols, n, 2, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n * 2];
        for (r, row) in x.iter().enumerate() {
            cols64[r] = row[0];
            cols64[n + r] = row[1];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
        let ops = [
            Op::Add, Op::Sub, Op::Mul, Op::Sin, Op::Cos, Op::Tanh, Op::Gauss, Op::Square,
            Op::Cube, Op::Neg,
        ];
        let mut rng = Mt::new(9);
        let mut trees = Vec::new();
        while trees.len() < 24 {
            let t = crate::expr::simplify(&random_tree_with(&mut rng, 2, &ops, 4, None));
            if crate::fit::gpu_fit_plan_reason(&t).is_ok() {
                trees.push(t);
            }
        }
        let plans: Vec<_> = trees
            .iter()
            .map(|t| crate::fit::gpu_fit_plan(t).expect("plan"))
            .collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (out, ms) = ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::Squared);
        eprintln!("kernel {:.1} ms for {} candidates", ms, out.len());
        let sample = crate::data::Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        };
        let mut worst = 0.0f64;
        for (k, ((tree, plan), res)) in trees.iter().zip(&plans).zip(&out).enumerate() {
            let mut cpu = tree.clone();
            let cpu_sse = crate::fit::fit_weighted_metric(
                &mut cpu, &sample.x, &sample.y, &[], 8, Metric::Squared,
            );
            let mut gpu = tree.clone();
            plan.apply(&mut gpu, &res.c);
            let gpu_sse = crate::search::eval_error(&gpu, &sample, Metric::Squared) * n as f64;
            let rel = (gpu_sse - cpu_sse).abs() / cpu_sse.max(1e-300);
            worst = worst.max(rel);
            eprintln!(
                "cand {} p={} status={} cpu={:.6e} gpu={:.6e} rel={:.2e}",
                k, plan.nconst, res.status, cpu_sse, gpu_sse, rel
            );
        }
        eprintln!("worst relative SSE gap {:.3e}", worst);
    }

    #[test]
    fn a_fitted_tree_is_independent_of_batch_composition() {
        let n = 400;
        let (x, _) = dataset(n);
        let target = model(&[2.0, -0.5, 1.0]);
        let y: Vec<f64> = x.iter().map(|r| eval(&target, r)).collect();
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let start = model(&[0.3, 0.2, 0.1]);
        let alone = crate::fit::gpu_fit_plan(&start).expect("plan");
        let (out_alone, _) = ev.fit_batch(&[&alone], 6, crate::fit::DeviceLoss::Squared);
        assert_eq!(out_alone[0].status, 0);

        // Same candidate repeated four times: every block owns one candidate,
        // so every result must be bit-identical to the lone fit.
        let plans: Vec<_> = (0..4)
            .map(|_| crate::fit::gpu_fit_plan(&start).expect("plan"))
            .collect();
        let plan_refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (out, _) = ev.fit_batch(&plan_refs, 6, crate::fit::DeviceLoss::Squared);
        for r in &out {
            assert_eq!(r.status, 0);
            assert_eq!(r.c, out_alone[0].c, "batch composition changed constants");
            assert_eq!(r.sse.to_bits(), out_alone[0].sse.to_bits());
        }
    }

    // ---- #229: LogSquashed on the batched kernel -------------------------

    use crate::metrics::{self, Metric};

    /// An affine model whose rows carry 12.5 % gross outliers. The robust
    /// optimum and the least-squares optimum are different points, which is
    /// what makes an SSE gate visible.
    fn robust_fixture(n: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            x.push(vec![t]);
            let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
            let outlier = i % 8 == 3;
            y.push(0.5 + 1.75 * t + noise + if outlier { 25.0 } else { 0.0 });
        }
        let (cols, y32) = cols_from(&x, &y);
        (x, y, cols, y32)
    }

    /// `mean ln(1 + r²)` for `tree` at its current constants — the configured
    /// objective, scored by the same CPU function the archive uses.
    fn log_squashed_at(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
        let pred: Vec<f64> = x.iter().map(|r| eval(tree, r)).collect();
        metrics::score_weighted(&pred, y, w, Metric::LogSquashed, crate::expr::count_constants(tree))
    }

    /// #229 parity: the batched weighted-LM/IRLS kernel must reach the CPU
    /// IRLS objective on a heavy-tailed fixture — not merely improve SSE.
    #[test]
    fn gpu_log_squashed_fit_matches_the_cpu_irls() {
        let n = 400;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let start = model(&[0.2, 1.0]);
        let mut cpu_tree = start.clone();
        let cpu = crate::fit::fit_weighted_metric(&mut cpu_tree, &x, &y, &[], 8, Metric::LogSquashed);
        assert!(cpu.is_finite() && cpu < log_squashed_at(&start, &x, &y, &[]));

        let plan = crate::fit::gpu_fit_plan(&start).expect("affine plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::LogSquashed);
        assert!(ms >= 0.0, "the log-squashed fit kernel ran");
        assert_eq!(out[0].status, 0, "the outlier fixture is well conditioned");
        let mut gpu_tree = start.clone();
        plan.apply(&mut gpu_tree, &out[0].c);
        let gpu = log_squashed_at(&gpu_tree, &x, &y, &[]);
        let rel = (gpu - cpu).abs() / cpu.abs().max(1e-12);
        assert!(
            gpu <= cpu * (1.0 + 1e-6),
            "the GPU fit is worse than the CPU IRLS on its own objective: gpu={:.12e} cpu={:.12e}",
            gpu,
            cpu
        );
        assert!(rel <= 1e-6, "GPU/CPU objective gap {:.3e} (gpu {:.12e}, cpu {:.12e})", rel, gpu, cpu);
        // The robust endpoint is measurably better than the least-squares one
        // on this fixture, so the comparison above is not vacuous.
        let mut l2 = start.clone();
        crate::fit::fit_weighted(&mut l2, &x, &y, &[], 8);
        assert!(
            log_squashed_at(&l2, &x, &y, &[]) > gpu + 1e-3,
            "the fixture must separate the robust and least-squares optima"
        );
    }

    /// #229: the same candidate produces bit-identical constants and the same
    /// objective whichever batch it travels in, twice in a row.
    #[test]
    fn gpu_log_squashed_fit_is_deterministic_and_composition_free() {
        let n = 400;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let start = model(&[0.2, 1.0]);
        let plan = crate::fit::gpu_fit_plan(&start).expect("plan");
        let (a, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::LogSquashed);
        let (b, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::LogSquashed);
        assert_eq!(a[0].c, b[0].c, "a repeated run must be bit-identical");
        assert_eq!(a[0].sse.to_bits(), b[0].sse.to_bits());

        let plans: Vec<_> = (0..4).map(|_| crate::fit::gpu_fit_plan(&start).expect("plan")).collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (batch, _) = ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::LogSquashed);
        for r in &batch {
            assert_eq!(r.c, a[0].c, "batch composition changed the robust constants");
            assert_eq!(r.sse.to_bits(), a[0].sse.to_bits());
        }
    }

    /// #229: the robust arm labels a rank-deficient system and a non-finite row
    /// instead of returning constants (those candidates go to the exact CPU
    /// arm with a cause).
    #[test]
    fn gpu_log_squashed_labels_ill_conditioning_and_non_finite_rows() {
        let n = 400;
        let (x, _y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let x0 = input(0);
        let collinear = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(0.1), x0.clone()]),
                node(Op::Mul, vec![constant(0.2), x0.clone()]),
            ],
        );
        let plan = crate::fit::gpu_fit_plan(&collinear).expect("collinear plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::LogSquashed);
        assert!(ms >= 0.0);
        assert_eq!(out[0].status, 4, "a rank-deficient robust system must be labelled");

        // sqrt(c - x²) leaves the real domain for |x| > sqrt(c).
        let bad = node(
            Op::Sqrt,
            vec![node(Op::Sub, vec![constant(0.25), node(Op::Square, vec![input(0)])])],
        );
        let y_bad: Vec<f64> = x.iter().map(|r| eval(&bad, r)).collect();
        let (cols, y32) = cols_from(&x, &y_bad);
        let mut ev = GpuEvaluator::new(&cols, n, 1, &y32).expect("second evaluator");
        let plan = crate::fit::gpu_fit_plan(&bad).expect("sqrt plan");
        let (out, _) = ev.fit_batch(&[&plan], 4, crate::fit::DeviceLoss::LogSquashed);
        assert_eq!(out[0].status, 7, "non-finite start must be labelled log-only");
    }

    /// #229: the base `--weight` vector scales the per-row IRLS weight, and a
    /// zero-weight row is skipped before any arithmetic. The reference is the
    /// exact CPU `log_squashed` arm on the same weights.
    ///
    /// The fixture separates two groups whose residuals are `O(1)` — a robust
    /// loss down-weights gross outliers by itself, so only *moderately* bad
    /// rows with a small base weight can show that `w` arrived at all. The
    /// low-weight half wants slope `-1`, the rest wants `+1.75`; a kernel that
    /// dropped `w` would land between them.
    #[test]
    fn gpu_log_squashed_honours_the_base_weight() {
        let n = 400;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut w = vec![1.0f64; n];
        for i in 0..n {
            let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            x.push(vec![t]);
            if i % 10 == 7 {
                // Masked: wild target, must never influence the fit.
                w[i] = 0.0;
                y.push(50.0 - 3.0 * t);
            } else if i % 2 == 0 {
                w[i] = 1.0;
                y.push(1.75 * t);
            } else {
                w[i] = 1e-2;
                y.push(-1.0 * t);
            }
        }
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &w), "f64 rows and weights uploaded");
        let start = model(&[0.2, 0.0]);
        let mut cpu_tree = start.clone();
        let cpu = crate::fit::fit_weighted_metric(&mut cpu_tree, &x, &y, &w, 8, Metric::LogSquashed);
        let plan = crate::fit::gpu_fit_plan(&start).expect("plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::LogSquashed);
        assert!(ms >= 0.0);
        assert_eq!(out[0].status, 0);
        let mut gpu_tree = start.clone();
        plan.apply(&mut gpu_tree, &out[0].c);
        let gpu = log_squashed_at(&gpu_tree, &x, &y, &w);
        assert!(
            gpu <= cpu * (1.0 + 1e-6),
            "weighted GPU fit worse than the CPU arm: gpu={:.12e} cpu={:.12e}",
            gpu,
            cpu
        );
        // A kernel that ignored `w` would fit both groups together.
        let mut unweighted = start.clone();
        crate::fit::fit_weighted_metric(&mut unweighted, &x, &y, &[], 8, Metric::LogSquashed);
        let wrong = log_squashed_at(&unweighted, &x, &y, &w);
        assert!(
            wrong > gpu + 1e-2,
            "the fixture must punish ignoring the base weight (got {:.6e} vs {:.6e})",
            wrong,
            gpu
        );
    }

    /// #229: the evaluator itself accumulates the configured objective. A
    /// `log_squashed` run must be ranked by `mean ln(1+r²)`, never by MSE.
    #[test]
    fn the_evaluator_scores_the_configured_loss_natively() {
        let n = 512;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let trees = vec![model(&[0.5, 1.75]), model(&[3.0, -0.5]), node(Op::Sin, vec![input(0)])];
        let mut pop = Population::new();
        for t in &trees {
            assert!(pop.add(t));
        }
        ev.set_loss(crate::fit::DeviceLoss::LogSquashed);
        assert_eq!(ev.loss(), crate::fit::DeviceLoss::LogSquashed);
        let (loss, ms) = ev.eval_population(&pop);
        assert!(ms >= 0.0);
        for (t, &got) in trees.iter().zip(&loss) {
            // `eval_population` returns the raw sum; the search divides by the
            // row count, which is the mean loss the archive would report.
            let mean = got as f64 / n as f64;
            let want = log_squashed_at(t, &x, &y, &[]);
            let rel = (mean - want).abs() / want.abs().max(1e-12);
            assert!(rel <= 1e-5, "device mean loss {:.9e} vs CPU {:.9e} (rel {:.2e})", mean, want, rel);
        }
        // The squared mode still accumulates SSE, so the two modes differ.
        ev.set_loss(crate::fit::DeviceLoss::Squared);
        let (sse, _) = ev.eval_population(&pop);
        let pred: Vec<f64> = x.iter().map(|r| eval(&trees[0], r)).collect();
        let want_sse = metrics::score(&pred, &y, Metric::Squared);
        let rel = (sse[0] as f64 / n as f64 - want_sse).abs() / want_sse.abs().max(1e-12);
        assert!(rel <= 1e-5, "device mean SSE {:.9e} vs CPU {:.9e}", sse[0] / n as f32, want_sse);
        assert_ne!(sse[0], loss[0], "the two native objectives must not coincide");
    }

    fn corr_fixture(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![-3.0 + 6.0 * i as f64 / (n - 1) as f64])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let noise = (((i * 29) % 13) as f64 - 6.0) * 1e-4;
                1.6 * (0.7 * r[0]).sin() + 0.35 * r[0] + 0.2 + noise
            })
            .collect();
        (x, y)
    }

    fn corr_model() -> Node {
        node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(0.8),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(1.1), input(0)])]),
                    ],
                ),
                constant(0.25),
            ],
        )
    }

    fn corr_at(tree: &Node, x: &[Vec<f64>], y: &[f64], metric: Metric) -> f64 {
        let pred: Vec<f64> = x.iter().map(|r| eval(tree, r)).collect();
        metrics::score_weighted(&pred, y, &[], metric, 0)
    }

    /// #224: the fit kernel minimises exactly `1 - |rho|`, freezes and
    /// renormalises the affine gauge, and is deterministic for a fixed device.
    /// The exact CPU f64 correlation arm is the reference.
    #[test]
    fn gpu_correlation_fit_matches_the_cpu_objective() {
        let n = 600usize;
        let (x, y) = corr_fixture(n);
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<_>>();
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let mut gpu_tree = corr_model();
        let plan = crate::fit::gpu_fit_plan(&gpu_tree).expect("correlation plan");
        assert_eq!(plan.gauge, [0, 2], "outer a*T+b pair");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
        assert!(ms >= 0.0, "correlation kernel ran");
        let res = &out[0];
        assert_eq!(res.status, 0, "well-conditioned correlation fixture");
        plan.apply(&mut gpu_tree, &res.c);
        assert!(res.c[0] == 1.0 || res.c[0] == -1.0, "scale={}", res.c[0]);
        assert_eq!(res.c[2], 0.0, "offset={}", res.c[2]);
        let gpu_score = corr_at(&gpu_tree, &x, &y, Metric::Correlation);

        let mut cpu_tree = corr_model();
        let cpu_score = crate::fit::fit_weighted_metric(
            &mut cpu_tree,
            &x,
            &y,
            &[],
            8,
            Metric::Correlation,
        );
        assert!(
            gpu_score <= cpu_score + 1e-6,
            "GPU {gpu_score} must match the CPU correlation fit {cpu_score}"
        );

        let (again, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
        assert_eq!(
            again[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            res.c.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(again[0].sse.to_bits(), res.sse.to_bits());
    }

    fn repeat_count(default: usize) -> usize {
        std::env::var("RCSR_REPEAT_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn fit_bits(out: &GpuFitOutput) -> Vec<u64> {
        let mut bits: Vec<u64> = out.c.iter().map(|v| v.to_bits()).collect();
        bits.push(out.sse.to_bits());
        bits
    }

    /// #359: one evaluator, `set_fit_data` interleaved with batches of different
    /// sizes. Every repeat must reproduce the first run bit for bit.
    #[test]
    fn correlation_fit_is_bit_stable_over_repeats() {
        let n = 600usize;
        let (x, y) = corr_fixture(n);
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<_>>();
        let tree = corr_model();
        let plan = crate::fit::gpu_fit_plan(&tree).expect("correlation plan");
        let mut reference: Option<Vec<u64>> = None;
        let mut failures = 0usize;
        let repeats = repeat_count(500);
        for i in 0..repeats {
            assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
            let width = 1 + i % 5;
            let plans: Vec<&crate::fit::GpuFitPlan> = (0..width).map(|_| &plan).collect();
            let (out, _) = ev.fit_batch(&plans, 8, crate::fit::DeviceLoss::Correlation);
            assert_eq!(out.len(), width);
            for o in &out {
                let bits = fit_bits(o);
                match &reference {
                    None => reference = Some(bits),
                    Some(r) => {
                        if *r != bits {
                            failures += 1;
                            eprintln!("repeat {i}: bits differ {bits:?} vs {r:?}");
                        }
                    }
                }
            }
        }
        assert_eq!(failures, 0, "{failures} of {repeats} repeats differed");
    }

    /// #359: the original two-fit test body, fresh evaluator each repeat.
    #[test]
    fn correlation_fit_fresh_evaluator_is_bit_stable() {
        let n = 600usize;
        let (x, y) = corr_fixture(n);
        let (cols, y32) = cols_from(&x, &y);
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<_>>();
        let plan = crate::fit::gpu_fit_plan(&corr_model()).expect("correlation plan");
        let mut failures = 0usize;
        let repeats = repeat_count(200);
        for i in 0..repeats {
            let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU test: no usable CUDA device");
                    return;
                }
            };
            assert!(ev.set_fit_data(&cols64, &y, &[]));
            let (a, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
            let (b, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
            if fit_bits(&a[0]) != fit_bits(&b[0]) {
                failures += 1;
                eprintln!("repeat {i}: {:?} vs {:?}", fit_bits(&a[0]), fit_bits(&b[0]));
            }
        }
        assert_eq!(failures, 0, "{failures} of {repeats} repeats differed");
    }

    #[test]
    fn gpu_correlation_reports_degenerate_and_non_finite() {
        let n = 200usize;
        let (x, mut y) = corr_fixture(n);
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<_>>();
        assert!(ev.set_fit_data(&cols64, &y, &[]));

        let flat = node(
            Op::Add,
            vec![
                node(Op::Sin, vec![node(Op::Mul, vec![constant(0.0), input(0)])]),
                constant(0.5),
            ],
        );
        let plan = crate::fit::gpu_fit_plan(&flat).expect("plan");
        let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
        assert_eq!(out[0].status, 0);
        assert_eq!(out[0].sse, 1.0);
        assert_eq!(out[0].c, plan.c0);

        y[7] = f64::NAN;
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let mut ev = GpuEvaluator::new(&cols, n, 1, &y32).expect("device");
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let plan = crate::fit::gpu_fit_plan(&corr_model()).expect("plan");
        let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Correlation);
        assert_eq!(out[0].status, 7, "non-finite start is labelled log-only");
    }

    /// signed correlation recovers an inverted start on the device by
    /// flipping the frozen gauge scale (a second orientation pass), never by
    /// reflecting the tree, and the result is bitwise deterministic.
    ///
    /// The fixture's shape `exp(x)` has no inner constant that could absorb the
    /// sign, and the gauge masks the scale slot, so the LM alone cannot change
    /// the orientation: only the second orientation pass can.
    #[test]
    fn gpu_signed_correlation_recovers_an_inverted_start() {
        let n = 600usize;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![-2.0 + 4.0 * i as f64 / (n - 1) as f64])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-4;
                -(1.5 * r[0].exp() + 0.3 + noise)
            })
            .collect();
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<f64>>();
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let model = || {
            node(
                Op::Add,
                vec![node(Op::Mul, vec![constant(0.5), node(Op::Exp, vec![input(0)])]), constant(0.1)],
            )
        };
        let mut tree = model();
        let start = corr_at(&tree, &x, &y, Metric::CorrelationSigned);
        assert!(start > 1.0, "the fixture must start inverted: {start}");
        let plan = crate::fit::gpu_fit_plan(&tree).expect("correlation plan");
        assert_eq!(plan.gauge, [0, 1], "outer a*T+b pair");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::CorrelationSigned);
        assert!(ms >= 0.0, "signed correlation kernel ran");
        assert_eq!(out[0].status, 0, "well-conditioned fixture");
        plan.apply(&mut tree, &out[0].c);
        assert_eq!(out[0].c[0], -1.0, "the probe must flip the frozen scale");
        assert_eq!(out[0].c[1], 0.0, "offset must be renormalised");
        let got = corr_at(&tree, &x, &y, Metric::CorrelationSigned);
        assert!(got < start, "signed device fit {start} -> {got}");
        assert!(got < 0.01, "the device did not recover the direction: {got}");
        // The exact objective agrees with the kernel's own score.
        assert!((out[0].sse - got).abs() < 1e-6, "kernel {} vs exact {got}", out[0].sse);

        let (again, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::CorrelationSigned);
        assert_eq!(
            again[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(again[0].sse.to_bits(), out[0].sse.to_bits());
    }

    /// A simplified tree (`Add(b, Mul(a, T))`, gauge `[1, 0]`) is flipped too:
    /// thread 0 negates the shared scale, not a private copy of the entry value.
    ///
    /// The fixture's shape `exp(x)` has no inner constant that could absorb the
    /// sign, and the gauge masks the scale slot, so the LM alone cannot change
    /// the orientation: only the second orientation pass can.
    #[test]
    fn gpu_signed_correlation_flips_a_simplified_gauge() {
        let n = 600usize;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![-2.0 + 4.0 * i as f64 / (n - 1) as f64])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-4;
                -(1.5 * r[0].exp() + 0.3 + noise)
            })
            .collect();
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<f64>>();
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let model = || {
            node(
                Op::Add,
                vec![node(Op::Mul, vec![constant(0.5), node(Op::Exp, vec![input(0)])]), constant(0.1)],
            )
        };
        let mut tree = crate::expr::simplify(&model());
        let start = corr_at(&tree, &x, &y, Metric::CorrelationSigned);
        assert!(start > 1.0, "the fixture must start inverted: {start}");
        let plan = crate::fit::gpu_fit_plan(&tree).expect("correlation plan");
        assert_eq!(plan.gauge, [1, 0], "simplify sorts the offset first");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::CorrelationSigned);
        assert!(ms >= 0.0, "signed correlation kernel ran");
        assert_eq!(out[0].status, 0, "well-conditioned fixture");
        plan.apply(&mut tree, &out[0].c);
        assert_eq!(out[0].c[1], -1.0, "the second pass must flip the frozen scale");
        assert_eq!(out[0].c[0], 0.0, "offset must be renormalised");
        let got = corr_at(&tree, &x, &y, Metric::CorrelationSigned);
        assert!(got < start, "signed device fit {start} -> {got}");
        assert!(got < 0.01, "the device did not recover the direction: {got}");
        // The exact objective agrees with the kernel's own score.
        assert!((out[0].sse - got).abs() < 1e-6, "kernel {} vs exact {got}", out[0].sse);

        let (again, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::CorrelationSigned);
        assert_eq!(
            again[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            out[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(again[0].sse.to_bits(), out[0].sse.to_bits());
    }

    /// paired measurement (ignored) of the signed correlation arms on
    /// recorded rows: historical CPU path against the batched kernel, at the
    /// row counts the cost model's crossover is fitted on.
    #[test]
    #[ignore]
    fn bench_correlation_fit_paired_signed() {
        for &n in &[500usize, 2_000, 10_000, 50_000] {
            for &candidates in &[1usize, 8, 64] {
                let (x, y0) = corr_fixture(n);
                let y: Vec<f64> = y0.iter().map(|v| -v).collect();
                let (cols, y32) = cols_from(&x, &y);
                let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
                    Some(ev) => ev,
                    None => {
                        eprintln!("skipping GPU benchmark: no usable device");
                        return;
                    }
                };
                let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<f64>>();
                assert!(ev.set_fit_data(&cols64, &y, &[]));
                let starts: Vec<f64> = (0..candidates)
                    .map(|k| 0.4 + 1.6 * k as f64 / candidates as f64)
                    .collect();
                let build = |b: f64| {
                    node(
                        Op::Add,
                        vec![
                            node(
                                Op::Mul,
                                vec![
                                    constant(0.8),
                                    node(Op::Sin, vec![node(Op::Mul, vec![constant(b), input(0)])]),
                                ],
                            ),
                            constant(0.25),
                        ],
                    )
                };
                let t0 = std::time::Instant::now();
                let mut cpu_scores = Vec::with_capacity(candidates);
                let mut start_scores = Vec::with_capacity(candidates);
                for &b in &starts {
                    let mut t = build(b);
                    start_scores.push(corr_at(&t, &x, &y, Metric::CorrelationSigned));
                    cpu_scores.push(crate::fit::fit_weighted_metric(
                        &mut t,
                        &x,
                        &y,
                        &[],
                        8,
                        Metric::CorrelationSigned,
                    ));
                }
                let cpu = t0.elapsed().as_secs_f64();
                let plans: Vec<crate::fit::GpuFitPlan> = starts
                    .iter()
                    .map(|&b| crate::fit::gpu_fit_plan(&build(b)).expect("plan"))
                    .collect();
                let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
                let t1 = std::time::Instant::now();
                let (out, kernel_ms) =
                    ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::CorrelationSigned);
                let gpu = t1.elapsed().as_secs_f64();
                let mut fallbacks = 0usize;
                let mut accepted = 0usize;
                let mut wins = 0usize;
                let mut deltas: Vec<f64> = Vec::new();
                for (k, res) in out.iter().enumerate() {
                    if res.status != 0 {
                        fallbacks += 1;
                        continue;
                    }
                    let mut t = build(starts[k]);
                    plans[k].apply(&mut t, &res.c);
                    let s = corr_at(&t, &x, &y, Metric::CorrelationSigned);
                    if s <= start_scores[k] + 1e-9 {
                        accepted += 1;
                    }
                    deltas.push(s - cpu_scores[k]);
                    if s <= cpu_scores[k] + 1e-6 {
                        wins += 1;
                    }
                }
                deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = deltas.get(deltas.len() / 2).copied().unwrap_or(f64::NAN);
                let worst = deltas.last().copied().unwrap_or(f64::NAN);
                eprintln!(
                    "signed correlation paired: N={n} P={candidates} p=3 cpu={cpu:.3}s gpu={gpu:.3}s kernel={kernel_ms:.1}ms speedup={:.1}x fallbacks={fallbacks} accepted={accepted}/{candidates} gpu<=cpu+1e-6={wins}/{candidates} median_delta={median:.3e} worst_delta={worst:.3e} row_candidates={}",
                    cpu / gpu.max(1e-9),
                    n * candidates
                );
            }
        }
    }

    /// #224 paired measurement (ignored): historical CPU correlation fitting
    /// against the batched correlation kernel on the same rows and starts.
    #[test]
    #[ignore]
    fn bench_correlation_fit_paired() {
        let n = 50_000usize;
        let (x, y) = corr_fixture(n);
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU benchmark: no usable device");
                return;
            }
        };
        let cols64 = cols.iter().map(|&v| v as f64).collect::<Vec<_>>();
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let candidates = 64usize;
        let starts: Vec<f64> = (0..candidates)
            .map(|k| 0.4 + 1.6 * k as f64 / candidates as f64)
            .collect();
        let build = |b: f64| {
            node(
                Op::Add,
                vec![
                    node(
                        Op::Mul,
                        vec![
                            constant(0.8),
                            node(Op::Sin, vec![node(Op::Mul, vec![constant(b), input(0)])]),
                        ],
                    ),
                    constant(0.25),
                ],
            )
        };
        let t0 = std::time::Instant::now();
        let mut cpu_scores = Vec::with_capacity(candidates);
        let mut start_scores = Vec::with_capacity(candidates);
        for &b in &starts {
            let mut t = build(b);
            start_scores.push(corr_at(&t, &x, &y, Metric::Correlation));
            cpu_scores.push(crate::fit::fit_weighted_metric(
                &mut t,
                &x,
                &y,
                &[],
                8,
                Metric::Correlation,
            ));
        }
        let cpu = t0.elapsed().as_secs_f64();
        let plans: Vec<crate::fit::GpuFitPlan> = starts
            .iter()
            .map(|&b| crate::fit::gpu_fit_plan(&build(b)).expect("plan"))
            .collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let t1 = std::time::Instant::now();
        let (out, kernel_ms) = ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::Correlation);
        let gpu = t1.elapsed().as_secs_f64();
        let mut fallbacks = 0usize;
        let mut accepted = 0usize;
        let mut wins = 0usize;
        let mut deltas: Vec<f64> = Vec::new();
        for (k, res) in out.iter().enumerate() {
            if res.status != 0 {
                fallbacks += 1;
                continue;
            }
            let mut t = build(starts[k]);
            plans[k].apply(&mut t, &res.c);
            let s = corr_at(&t, &x, &y, Metric::Correlation);
            if s <= start_scores[k] + 1e-9 {
                accepted += 1;
            }
            deltas.push(s - cpu_scores[k]);
            if s <= cpu_scores[k] + 1e-6 {
                wins += 1;
            }
        }
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = deltas.get(deltas.len() / 2).copied().unwrap_or(f64::NAN);
        let worst = deltas.last().copied().unwrap_or(f64::NAN);
        eprintln!(
            "correlation fit paired: N={n} P={candidates} p=3 cpu={cpu:.3}s gpu={gpu:.3}s kernel={kernel_ms:.1}ms speedup={:.1}x fallbacks={fallbacks} accepted={accepted}/{candidates} gpu<=cpu+1e-6={wins}/{candidates} median_delta={median:.3e} worst_delta={worst:.3e}",
            cpu / gpu.max(1e-9)
        );
    }

    // ---- #222: Absolute (MAE) on the batched kernel ----------------------

    /// `mean |r|` at the tree's current constants, scored by the CPU f64
    /// function the archive uses.
    fn absolute_at(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
        let pred: Vec<f64> = x.iter().map(|r| eval(tree, r)).collect();
        metrics::score_weighted(&pred, y, w, Metric::Absolute, crate::expr::count_constants(tree))
    }

    /// #222 parity: the weighted-LM/IRLS kernel must reach the CPU L1 IRLS
    /// objective on a heavy-tailed fixture, never an SSE endpoint.
    #[test]
    fn gpu_absolute_fit_matches_the_cpu_irls() {
        let n = 400;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");

        let start = model(&[0.2, 1.0]);
        let mut cpu_tree = start.clone();
        let cpu = crate::fit::fit_weighted_metric(&mut cpu_tree, &x, &y, &[], 8, Metric::Absolute);
        assert!(cpu.is_finite() && cpu < absolute_at(&start, &x, &y, &[]));

        let plan = crate::fit::gpu_fit_plan(&start).expect("affine plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        assert!(ms >= 0.0, "the absolute fit kernel ran");
        assert_eq!(out[0].status, 0, "the outlier fixture is well conditioned");
        let mut gpu_tree = start.clone();
        plan.apply(&mut gpu_tree, &out[0].c);
        let gpu = absolute_at(&gpu_tree, &x, &y, &[]);
        assert!(
            gpu <= cpu * (1.0 + 1e-5),
            "the GPU L1 fit is worse than the CPU IRLS: gpu={:.12e} cpu={:.12e}",
            gpu,
            cpu
        );
        let rel = (gpu - cpu).abs() / cpu.abs().max(1e-12);
        assert!(rel <= 1e-4, "GPU/CPU MAE gap {:.3e} (gpu {:.12e}, cpu {:.12e})", rel, gpu, cpu);
        // The fixture separates the L1 and L2 answers, so the comparison is
        // not vacuous and the kernel is not just running squared LM.
        let mut l2 = start.clone();
        crate::fit::fit_weighted(&mut l2, &x, &y, &[], 8);
        assert!(
            absolute_at(&l2, &x, &y, &[]) > gpu + 1e-3,
            "the fixture must separate the L1 and least-squares optima"
        );
    }

    /// #222: the kernel's objective statistic is the configured risk too:
    /// `Σ|r|` (up to the trial baseline), not `Σr²`.
    #[test]
    fn gpu_absolute_reports_the_configured_objective() {
        let n = 200;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let start = model(&[0.2, 1.0]);
        let plan = crate::fit::gpu_fit_plan(&start).expect("plan");
        let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        assert_eq!(out[0].status, 0);
        let mut fitted = start.clone();
        plan.apply(&mut fitted, &out[0].c);
        let pred: Vec<f64> = x.iter().map(|r| eval(&fitted, r)).collect();
        let want: f64 = pred.iter().zip(&y).map(|(p, t)| (p - t).abs()).sum();
        let rel = (out[0].sse - want).abs() / want.max(1e-12);
        assert!(
            rel <= 1e-4,
            "kernel objective {:.9e} vs exact Σ|r| {:.9e} (rel {:.2e})",
            out[0].sse,
            want,
            rel
        );
        let sse: f64 = pred.iter().zip(&y).map(|(p, t)| (p - t) * (p - t)).sum();
        assert_ne!(out[0].sse.to_bits(), sse.to_bits(), "the two objectives must not coincide");
    }

    /// #222: deterministic, and independent of batch composition.
    #[test]
    fn gpu_absolute_fit_is_deterministic_and_composition_free() {
        let n = 400;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let start = model(&[0.2, 1.0]);
        let plan = crate::fit::gpu_fit_plan(&start).expect("plan");
        let (a, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        let (b, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        assert_eq!(a[0].c, b[0].c, "a repeated run must be bit-identical");
        assert_eq!(a[0].sse.to_bits(), b[0].sse.to_bits());

        let plans: Vec<_> = (0..4).map(|_| crate::fit::gpu_fit_plan(&start).expect("plan")).collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (batch, _) = ev.fit_batch(&refs, 8, crate::fit::DeviceLoss::Absolute);
        for r in &batch {
            assert_eq!(r.c, a[0].c, "batch composition changed the L1 constants");
            assert_eq!(r.sse.to_bits(), a[0].sse.to_bits());
        }
    }

    /// #222: rank deficiency and non-finite rows are labelled, never returned
    /// as constants; the caller then uses the exact CPU arm with a cause.
    #[test]
    fn gpu_absolute_labels_ill_conditioning_and_non_finite_rows() {
        let n = 400;
        let (x, _y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let x0 = input(0);
        let collinear = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(0.1), x0.clone()]),
                node(Op::Mul, vec![constant(0.2), x0.clone()]),
            ],
        );
        let plan = crate::fit::gpu_fit_plan(&collinear).expect("collinear plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        assert!(ms >= 0.0);
        assert_eq!(out[0].status, 4, "a rank-deficient L1 system must be labelled");

        let bad = node(
            Op::Sqrt,
            vec![node(Op::Sub, vec![constant(0.25), node(Op::Square, vec![input(0)])])],
        );
        let y_bad: Vec<f64> = x.iter().map(|r| eval(&bad, r)).collect();
        let (cols, y32) = cols_from(&x, &y_bad);
        let mut ev = GpuEvaluator::new(&cols, n, 1, &y32).expect("second evaluator");
        let plan = crate::fit::gpu_fit_plan(&bad).expect("sqrt plan");
        let (out, _) = ev.fit_batch(&[&plan], 4, crate::fit::DeviceLoss::Absolute);
        assert_eq!(out[0].status, 7, "non-finite start must be labelled log-only");
    }

    /// #222: exact interpolation is the L1 arm's degeneracy. Every residual is
    /// zero at the optimum, so the weights hit the `1e-6` floor; the result
    /// must be either the exact fit or an explicit ill-conditioned status —
    /// never a silent SSE optimum or non-finite constants.
    #[test]
    fn gpu_absolute_zero_residuals_are_exact_or_labelled() {
        let n = 300;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0;
            x.push(vec![t]);
            y.push(0.5 + 1.75 * t);
        }
        let (cols, y32) = cols_from(&x, &y);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let plan = crate::fit::gpu_fit_plan(&model(&[0.2, 1.0])).expect("plan");
        let (out, _) = ev.fit_batch(&[&plan], 8, crate::fit::DeviceLoss::Absolute);
        assert!(
            crate::fit::fit_status_usable(out[0].status),
            "a zero-residual fit is exact or ill-conditioned, got {}",
            out[0].status
        );
        assert!(out[0].c.iter().all(|v| v.is_finite()));
        if out[0].status == 0 {
            let mut tree = model(&[0.2, 1.0]);
            plan.apply(&mut tree, &out[0].c);
            let mae = absolute_at(&tree, &x, &y, &[]);
            assert!(mae <= 1e-4, "an accepted zero-residual fit must be exact, MAE={mae:.3e}");
        }
    }

    /// #222: `--no-fit` at the kernel boundary leaves every constant at its
    /// seed value bit for bit.
    #[test]
    fn gpu_absolute_no_fit_does_not_move_constants() {
        let n = 300;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        let mut cols64 = vec![0.0f64; n];
        for i in 0..n {
            cols64[i] = x[i][0];
        }
        assert!(ev.set_fit_data(&cols64, &y, &[]));
        let plan = crate::fit::gpu_fit_plan(&model(&[0.2, 1.0])).expect("plan");
        let (out, _) = ev.fit_batch(&[&plan], 0, crate::fit::DeviceLoss::Absolute);
        assert_eq!(out[0].status, 0);
        assert_eq!(out[0].c, plan.c0, "--fit-iters 0 must not move a constant");
    }

    /// #222: the CUDA evaluator's native loss for `absolute` is `mean |r|`,
    /// matching the CPU scorer to f32 rounding, and it is not the SSE.
    #[test]
    fn the_evaluator_scores_absolute_natively() {
        let n = 300;
        let (x, y, cols, y32) = robust_fixture(n);
        let mut ev = match GpuEvaluator::new(&cols, n, 1, &y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return;
            }
        };
        ev.set_loss(crate::fit::DeviceLoss::Absolute);
        assert_eq!(ev.loss(), crate::fit::DeviceLoss::Absolute);
        let trees = vec![model(&[0.2, 1.0]), model(&[1.0, 2.0])];
        let mut pop = Population::new();
        for t in &trees {
            assert!(pop.add(t));
        }
        let (loss, ms) = ev.eval_population(&pop);
        assert!(ms >= 0.0);
        for (t, &got) in trees.iter().zip(&loss) {
            let pred: Vec<f64> = x.iter().map(|r| eval(t, r)).collect();
            let mean: f64 = pred.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum::<f64>() / n as f64;
            let got = got as f64 / n as f64;
            let rel = (got - mean).abs() / mean.max(1e-12);
            assert!(rel <= 1e-4, "device MAE {:.9e} vs CPU {:.9e} (rel {:.2e})", got, mean, rel);
        }
        ev.set_loss(crate::fit::DeviceLoss::Squared);
        let (sse, _) = ev.eval_population(&pop);
        assert_ne!(loss[0], sse[0], "the two native objectives must not coincide");
    }

    /// #222 measurement (ignored): fit-stage throughput and endpoint quality
    /// for the absolute arm. The CPU side is the exact L1 IRLS
    /// (`fit_weighted_metric(.., Absolute)`); the GPU side is the batched
    /// weighted-LM/IRLS kernel. Both fit the same candidates on the same rows,
    /// so the ratio is paired. Run with
    /// `cargo test --release --features cuda --lib bench_batched_fit_absolute -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_batched_fit_absolute() {
        let n = std::env::var("RCSR_BENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000usize);
        let threads = 8usize;
        let pmax = 20usize;
        for (p, candidates) in [(4usize, 64usize), (8, 64)] {
            let (x, mut y, cols, _) = bench_fixture(pmax, p, n);
            for i in (3..n).step_by(10) {
                y[i] += 15.0;
            }
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let start: Vec<f64> = (0..p).map(|k| 0.25 + 0.07 * k as f64).collect();
            let loss_of = |c: &[f64]| -> f64 {
                let t = linear_model(c);
                let pred: Vec<f64> = x.iter().map(|r| eval(&t, r)).collect();
                metrics::score_weighted(&pred, &y, &[], Metric::Absolute, p)
            };

            let t0 = std::time::Instant::now();
            let per = candidates.div_ceil(threads);
            let cpu_losses: Vec<Vec<f64>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let x = &x;
                        let y = &y;
                        let start = &start;
                        scope.spawn(move || {
                            let mut out = Vec::with_capacity(per);
                            for _ in 0..per {
                                let mut t = linear_model(start);
                                crate::fit::fit_weighted_metric(&mut t, x, y, &[], 8, Metric::Absolute);
                                out.push(constants(&t));
                            }
                            out
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("cpu fit thread"))
                    .collect()
            });
            let cpu = t0.elapsed().as_secs_f64();

            let mut ev = match GpuEvaluator::new(&cols, n, pmax, &y32) {
                Some(ev) => ev,
                None => {
                    eprintln!("skipping GPU benchmark: no usable device");
                    return;
                }
            };
            let mut cols64 = vec![0.0f64; n * pmax];
            for r in 0..n {
                for k in 0..pmax {
                    cols64[k * n + r] = x[r][k];
                }
            }
            ev.set_fit_data(&cols64, &y, &[]);
            let plan = crate::fit::gpu_fit_plan(&linear_model(&start)).expect("plan");
            let plans: Vec<&crate::fit::GpuFitPlan> = (0..candidates).map(|_| &plan).collect();
            let t1 = std::time::Instant::now();
            let (out, ms) = ev.fit_batch(&plans, 8, crate::fit::DeviceLoss::Absolute);
            let gpu = t1.elapsed().as_secs_f64();
            let ok = out.iter().filter(|r| r.status == 0).count();

            let mut no_worse = 0usize;
            let mut cpu_sum = 0.0;
            let mut gpu_sum = 0.0;
            for (c_res, c_cpu) in out.iter().zip(&cpu_losses) {
                let g = if c_res.status == 0 { loss_of(&c_res.c) } else { f64::INFINITY };
                let c = loss_of(c_cpu);
                cpu_sum += c;
                if g <= c * (1.0 + 1e-6) {
                    no_worse += 1;
                }
                if g.is_finite() {
                    gpu_sum += g;
                }
            }
            let npairs = cpu_losses.len().min(out.len()).max(1);
            eprintln!(
                "absolute p={} N={} P={} cpu={:.3}s gpu={:.3}s (kernel {:.1}ms, status0 {}/{}) fit speedup={:.1}x mean_loss cpu={:.6e} gpu={:.6e} no_worse={}/{}",
                p,
                n,
                candidates,
                cpu,
                gpu,
                ms,
                ok,
                out.len(),
                cpu / gpu.max(1e-9),
                cpu_sum / npairs as f64,
                gpu_sum / npairs as f64,
                no_worse,
                npairs
            );
        }
    }
}

/// #230: device-side exact rank stage.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod rank_tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::fit::{fit_weighted_metric, gpu_fit_plan_reason, polish_rank_batch};
    use crate::metrics::{Metric, RankSums, RankTarget};

    struct Fixture {
        cols32: Vec<f32>,
        cols64: Vec<f64>,
        n: usize,
        nv: usize,
        y32: Vec<f32>,
        y: Vec<f64>,
        x: Vec<Vec<f64>>,
    }

    /// A 2-column sample with duplicated rows in both `x` and `y`, so ties,
    /// duplicate predictions and `-0.0`/`+0.0` can all be exercised.
    fn fixture(n: usize) -> Fixture {
        let nv = 2;
        let mut cols32 = vec![0.0f32; nv * n];
        let mut cols64 = vec![0.0f64; nv * n];
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
            // Every third row repeats the previous x value (prediction ties),
            // and `sign` alternates so `abs(x0)` creates two tied groups.
            let t = if i % 3 == 2 && i > 0 { x[i - 1][0] } else { t };
            let s: f64 = if i % 2 == 0 { 0.0 } else { -0.0 };
            let v = 1.4 * (0.9 * t).sin() + 0.3 * t;
            let v = if i % 5 == 4 { 0.0 } else { v };
            x.push(vec![t, s]);
            y.push(v);
            cols32[i] = t as f32;
            cols32[n + i] = s as f32;
            cols64[i] = t;
            cols64[n + i] = s;
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        Fixture {
            cols32,
            cols64,
            n,
            nv,
            y32,
            y,
            x,
        }
    }

    fn evaluator(f: &Fixture) -> Option<GpuEvaluator> {
        let mut ev = match GpuEvaluator::new(&f.cols32, f.n, f.nv, &f.y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return None;
            }
        };
        assert!(ev.set_fit_data(&f.cols64, &f.y, &[]), "f64 fit data upload");
        Some(ev)
    }

    fn plan(t: &Node) -> crate::fit::GpuFitPlan {
        gpu_fit_plan_reason(t).expect("an eligible plan")
    }

    fn cpu_pred(f: &Fixture, t: &Node) -> Vec<f64> {
        f.x.iter().map(|r| crate::expr::eval(t, r)).collect()
    }

    fn score_of(sums: Option<RankSums>, target: &RankTarget, signed: bool) -> f64 {
        let sums = sums.map(|s| RankSums {
            bb: target.sum_bb(),
            ..s
        });
        crate::metrics::score_rank_sums(sums, signed)
    }

    fn linear() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.15)],
        )
    }

    fn tied() -> Node {
        node(Op::Abs, vec![node(Op::Add, vec![input(0), constant(-0.25)])])
    }

    fn flat() -> Node {
        node(Op::Mul, vec![constant(0.0), input(0)])
    }

    /// Every fixture: exact integer sums on the device equal the CPU's, bit for
    /// bit, including ties and duplicates.
    /// #230 measurement (ignored): paired CPU-exact versus device polish for
    /// one candidate, plus the per-call rank-stage cost. Run with
    /// `cargo test --release --features cuda --lib bench_rank_polish -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_rank_polish() {
        let n: usize = std::env::var("RCSR_BENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(154_000);
        let f = fixture(n);
        let Some(mut ev) = evaluator(&f) else { return };
        // Five constants: the pinned cell's p = 5.
        let t = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(0.8),
                        node(
                            Op::Sin,
                            vec![node(
                                Op::Add,
                                vec![
                                    node(Op::Mul, vec![constant(0.9), input(0)]),
                                    constant(0.1),
                                ],
                            )],
                        ),
                    ],
                ),
                node(Op::Mul, vec![constant(0.2), input(1)]),
            ],
        );
        let p = plan(&t);
        assert_eq!(p.nconst, 4);
        let target = RankTarget::new(&f.y).unwrap();
        // CPU: the exact ladder, target ranks cached once (as the engine does).
        let mut cpu = t.clone();
        let cpu_start = std::time::Instant::now();
        crate::fit::coordinate_polish(&mut cpu, &f.x, &f.y, &[], Metric::RankCorrelation, 8);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;
        // GPU: the same ladder, batched through the device rank stage.
        let gpu_start = std::time::Instant::now();
        let out = polish_rank_batch(&mut ev, &[&p], &[p.c0.clone()], &target, false, 8)
            .expect("device polish");
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0;
        // One batched call's cost, to expose the sort-dominated stage.
        let one_start = std::time::Instant::now();
        let mut trials = Vec::new();
        for k in 0..6 {
            let mut c = p.c0.clone();
            c[0] *= 1.0 + 0.01 * (k as f64 + 1.0);
            trials.push(c);
        }
        for _ in 0..3 {
            let _ = ev.rank_score_trials(&[&p], &[trials.clone()]).expect("rank call");
        }
        let call_ms = one_start.elapsed().as_secs_f64() * 1000.0 / 3.0;
        // One segment isolates the fixed per-call overhead from the sort work:
        // six segments should cost close to six times this if the stage is
        // sort-bound rather than latency-bound.
        let single_start = std::time::Instant::now();
        for _ in 0..3 {
            let _ = ev
                .rank_score_trials(&[&p], &[vec![p.c0.clone()]])
                .expect("rank call");
        }
        let single_seg_ms = single_start.elapsed().as_secs_f64() * 1000.0 / 3.0;
        // #231: cost model. Time one call carrying `s` segments for
        // `s = 1..=8`; a fit of `t(s) = a + b*s` separates the fixed
        // allocation/upload/launch cost from the per-segment (sort) cost. Both
        // orientations run the same integer stage, so one curve covers both.
        let mut cost = Vec::new();
        for s in 1..=8 {
            let mut per = Vec::new();
            for k in 0..s {
                let mut c = p.c0.clone();
                c[0] *= 1.0 + 0.001 * (k as f64 + 1.0);
                per.push(c);
            }
            let start = std::time::Instant::now();
            for _ in 0..3 {
                let _ = ev.rank_score_trials(&[&p], &[per.clone()]).expect("rank call");
            }
            cost.push((s, start.elapsed().as_secs_f64() * 1000.0 / 3.0));
        }
        // Least squares on (s, t).
        let m = cost.len() as f64;
        let sx: f64 = cost.iter().map(|(s, _)| *s as f64).sum();
        let sy: f64 = cost.iter().map(|(_, t)| *t).sum();
        let sxx: f64 = cost.iter().map(|(s, _)| (*s as f64) * (*s as f64)).sum();
        let sxy: f64 = cost.iter().map(|(s, t)| *s as f64 * *t).sum();
        let b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        let a = (sy - b * sx) / m;
        // Same constants? The device path must reproduce the CPU ladder.
        let cpu_c = crate::fit::constants_slots(&cpu);
        let same = cpu_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            == out.constants[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        eprintln!(
            "N={} p={} cpu_polish={:.1}ms gpu_polish={:.1}ms speedup={:.2}x call_6seg={:.2}ms call_1seg={:.2}ms ratio={:.2} same_constants={}",
            n,
            p.nconst,
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms.max(1e-9),
            call_ms,
            single_seg_ms,
            call_ms / single_seg_ms.max(1e-9),
            same
        );
        eprintln!(
            "  cost model: t(call) = {:.2} ms + {:.2} ms x segments  [{}]",
            a,
            b,
            cost.iter()
                .map(|(s, t)| format!("{s}segs={t:.2}ms"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        // #231: the signed arm costs exactly the unsigned one on the same
        // start (same sort work); record both.
        for signed in [false, true] {
            let start = std::time::Instant::now();
            let out = polish_rank_batch(&mut ev, &[&p], &[p.c0.clone()], &target, signed, 8)
                .expect("device polish");
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "  signed={signed} gpu_polish={:.1}ms score={:.6} constants={:?}",
                ms,
                out.scores[0],
                out.constants[0]
            );
        }
    }

    /// The cached target ranks are the CPU's `B`, row for row, on a target
    /// with duplicates and `-0.0`/`+0.0` pairs.
    #[test]
    fn target_ranks_match_the_cpu() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        let mut b = vec![0i64; f.n];
        let rc = unsafe { rc_rank_debug_b(ev.ctx, b.as_mut_ptr()) };
        assert_eq!(rc, 0, "the target stage ran");
        assert_eq!(b, target.ranks(), "device B differs from the CPU average ranks");
    }

    extern "C" {
        fn rc_rank_test_force_cub_fail(on: i32);
    }

    /// #410: a refused CUB size query fails the rank call before any buffer is
    /// sized, `rank_score_trials` reports `None` (the caller keeps the CPU
    /// arm), and the next call on the same context works.
    #[test]
    fn refused_cub_query_fails_the_rank_call_then_recovers() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let p = plan(&linear());
        let ok = ev.rank_score_trials(&[&p], &[vec![p.c0.clone()]]);
        assert!(ok.is_some(), "baseline rank call");
        unsafe { rc_rank_test_force_cub_fail(1) };
        let refused = ev.rank_score_trials(&[&p], &[vec![p.c0.clone()]]);
        unsafe { rc_rank_test_force_cub_fail(0) };
        assert!(refused.is_none(), "a refused CUB query must fail the rank call");
        let again = ev
            .rank_score_trials(&[&p], &[vec![p.c0.clone()]])
            .expect("no error may stay pending");
        assert_eq!(again[0][0].map(|s| (s.ab, s.aa)), ok.unwrap()[0][0].map(|s| (s.ab, s.aa)));
    }

    #[test]
    fn rank_sums_are_bit_identical_to_the_cpu() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        for t in [linear(), tied(), flat()] {
            let p = plan(&t);
            let pred = cpu_pred(&f, &t);
            let cpu = target.sums(&pred).expect("cpu sums");
            let gpu = ev
                .rank_score_trials(&[&p], &[vec![p.c0.clone()]])
                .expect("device rank call");
            let got = gpu[0][0].expect("a finite segment");
            assert_eq!(got.ab, cpu.ab, "Sigma A B differs");
            assert_eq!(got.aa, cpu.aa, "Sigma A A differs");
            assert_eq!(got.bb, cpu.bb, "Sigma B B differs");
            assert_eq!(got.bb, target.sum_bb());
        }
    }

    /// Ties, duplicated predictions and `-0.0`/`+0.0` produce the same score on
    /// both sides; a constant prediction is exactly 1.0 (never NaN).
    #[test]
    fn ties_and_constant_predictions_score_exactly() {
        let f = fixture(90);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        for (t, expect_constant) in [(tied(), false), (flat(), true)] {
            let p = plan(&t);
            let pred = cpu_pred(&f, &t);
            let cpu = target.score(&pred, false);
            let gpu_sums = ev
                .rank_score_trials(&[&p], &[vec![p.c0.clone()]])
                .expect("device rank call");
            let gpu = score_of(gpu_sums[0][0], &target, false);
            assert_eq!(gpu.to_bits(), cpu.to_bits(), "tie handling differs: {gpu} vs {cpu}");
            if expect_constant {
                assert_eq!(gpu, 1.0);
            } else {
                assert!(0.0 <= gpu && gpu <= 1.0);
            }
        }
        // Signed variant: 1 - rho, still exact.
        let p = plan(&tied());
        let pred = cpu_pred(&f, &tied());
        let cpu = target.score(&pred, true);
        let gpu_sums = ev
            .rank_score_trials(&[&p], &[vec![p.c0.clone()]])
            .expect("device rank call");
        assert_eq!(score_of(gpu_sums[0][0], &target, true).to_bits(), cpu.to_bits());
    }

    /// The batched device polish commits the same constants as the CPU ladder
    /// from the same start (the exactness claim of #230).
    #[test]
    fn device_polish_matches_the_cpu_ladder() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        for metric in [Metric::RankCorrelation, Metric::RankCorrelationSigned] {
            let signed = crate::fit::rank_metric_signed(metric).unwrap();
            let mut cpu = linear();
            crate::fit::coordinate_polish(&mut cpu, &f.x, &f.y, &[], metric, 8);
            let cpu_c = crate::fit::constants_slots(&cpu);
            let t = linear();
            let p = plan(&t);
            let out = polish_rank_batch(&mut ev, &[&p], &[p.c0.clone()], &target, signed, 8)
                .expect("device polish");
            assert_eq!(
                out.constants[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                cpu_c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{metric:?}: device polish must reproduce the CPU constants"
            );
            assert!(out.scores[0].is_finite());
        }
    }

    // -- #231: the signed orientation on the device --------------------------

    /// A strictly decreasing target: the incoming `linear()` slope is positive,
    /// so its signed score is above 1 and only the fitted orientation can go
    /// below it.
    fn decreasing_fixture(n: usize) -> Fixture {
        let nv = 1;
        let mut cols32 = vec![0.0f32; n];
        let mut cols64 = vec![0.0f64; n];
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / 10.0;
            // Duplicate targets every ninth row keep the average-rank path in
            // play on the device too.
            let v = if i % 9 == 4 && i > 0 { y[i - 1] } else { 5.0 - t };
            x.push(vec![t]);
            y.push(v);
            cols32[i] = t as f32;
            cols64[i] = t;
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        Fixture {
            cols32,
            cols64,
            n,
            nv,
            y32,
            y,
            x,
        }
    }

    fn reflected() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(-0.8), input(0)]), constant(-0.15)],
        )
    }

    /// #231: the device rank stage is orientation-free. Negating every constant
    /// negates the prediction exactly, so `Sigma A B` flips and `Sigma A A`,
    /// `Sigma B B` are untouched; the signed score of a reflection is `2 - the
    /// original` and the unsigned score is identical bit for bit.
    #[test]
    fn reflection_flips_the_cross_sum_exactly_on_the_device() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = linear();
        let p = plan(&t);
        let c = p.c0.clone();
        let negated: Vec<f64> = c.iter().map(|v| -v).collect();
        let out = ev
            .rank_score_trials(&[&p], &[vec![c, negated]])
            .expect("device rank call");
        let a = out[0][0].expect("the original is finite");
        let b = out[0][1].expect("the reflection is finite");
        assert_eq!(b.ab, -a.ab, "the device cross sum must flip exactly");
        assert_eq!(b.aa, a.aa);
        assert_eq!(b.bb, a.bb);
        let target = RankTarget::new(&f.y).unwrap();
        let up = score_of(Some(a), &target, true);
        let down = score_of(Some(b), &target, true);
        assert!(
            (up + down - 2.0).abs() <= 4.0 * f64::EPSILON,
            "signed(f) + signed(-f) must be 2 on the device: {up} + {down}"
        );
        assert_eq!(
            score_of(Some(a), &target, false).to_bits(),
            score_of(Some(b), &target, false).to_bits(),
            "the unsigned score cannot see a reflection"
        );
    }

    /// #231: the signed arm is exact, monotone from the incoming constants, and
    /// the number it reports is the CPU's exact signed score of the constants
    /// it returned — never an SSE and never an `|rho|`.
    #[test]
    fn signed_rank_fit_batch_is_exact_and_monotone() {
        let f = decreasing_fixture(120);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        for t in [linear(), reflected(), tied()] {
            let p = plan(&t);
            let incoming = target.score(&cpu_pred(&f, &t), true);
            let (out, ms) = ev.rank_fit_batch(&[&p], 6, &f.y, true);
            assert!(ms > 0.0, "the signed rank arm ran on the device");
            assert_eq!(out.len(), 1);
            if out[0].status != 0 {
                // A labelled fallback; the CPU arm owns that candidate.
                continue;
            }
            assert!(out[0].sse.is_finite());
            assert!(
                out[0].sse <= incoming,
                "the signed arm returned a worse score: {} > {}",
                out[0].sse,
                incoming
            );
            let mut trial = t.clone();
            p.apply(&mut trial, &out[0].c);
            let exact = target.score(&cpu_pred(&f, &trial), true);
            assert_eq!(
                out[0].sse.to_bits(),
                exact.to_bits(),
                "the reported number must be the exact CPU signed score"
            );
        }
        // Direction: `linear()` is increasing while the target decreases, so
        // its incoming signed score is above 1 and the fitted one must be
        // below it.
        let p = plan(&linear());
        let before = target.score(&cpu_pred(&f, &linear()), true);
        assert!(before > 1.0, "the fixture must start in the wrong direction: {before}");
        let (out, _) = ev.rank_fit_batch(&[&p], 6, &f.y, true);
        assert_eq!(out[0].status, 0);
        assert!(
            out[0].sse < 1.0,
            "the signed arm must find the positive direction: {}",
            out[0].sse
        );
    }

    /// #231: `--no-fit` (`iters == 0`) never moves a constant and never starts
    /// a device sweep, under either orientation.
    #[test]
    fn rank_fit_batch_with_zero_iters_is_a_noop() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        for signed in [false, true] {
            let t = linear();
            let p = plan(&t);
            let before = p.c0.clone();
            let (out, ms) = ev.rank_fit_batch(&[&p], 0, &f.y, signed);
            assert_eq!(ms, 0.0, "no device work for --no-fit");
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].status, 0);
            assert_eq!(
                out[0].c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                before.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "signed={signed}: --no-fit must not move a constant"
            );
        }
    }

    /// Splitting the same trials across two calls cannot change the sums: the
    /// integer reduction has no floating-point order to lose.
    #[test]
    fn sums_are_independent_of_the_batch_split() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = linear();
        let p = plan(&t);
        let mut trials: Vec<Vec<f64>> = Vec::new();
        for k in 0..6 {
            let mut c = p.c0.clone();
            c[0] += 0.01 * (k as f64 + 1.0);
            trials.push(c);
        }
        let all = ev
            .rank_score_trials(&[&p], &[trials.clone()])
            .expect("one call");
        let mut split = Vec::new();
        for t1 in &trials {
            let part = ev.rank_score_trials(&[&p], &[vec![t1.clone()]]).expect("split call");
            split.push(part[0][0]);
        }
        assert_eq!(all[0], split, "the split changed the integer sums");
        // And repeating the call is bit-identical.
        let again = ev.rank_score_trials(&[&p], &[trials]).expect("repeat");
        assert_eq!(all, again);
    }

    /// The end-to-end device arm returns constants at least as good as the
    /// squared start on the exact score, and never a non-finite score.
    #[test]
    fn rank_fit_batch_never_returns_a_worse_score() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![constant(1.2), node(Op::Sin, vec![node(Op::Mul, vec![constant(0.9), input(0)])])],
                ),
                constant(0.05),
            ],
        );
        let p = plan(&t);
        let start_tree = t.clone();
        let (out, ms) = ev.rank_fit_batch(&[&p], 6, &f.y, false);
        assert!(ms > 0.0, "the rank arm ran on the device");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].status, 0, "the fixture candidate is eligible");
        assert!(out[0].sse.is_finite());
        let target = RankTarget::new(&f.y).unwrap();
        let start_score = {
            let mut cpu_start = start_tree.clone();
            // The squared start is what the device LM began from.
            fit_weighted_metric(&mut cpu_start, &f.x, &f.y, &[], 6, Metric::RankCorrelation);
            let pred = cpu_pred(&f, &cpu_start);
            target.score(&pred, false)
        };
        // The device starts from its own LM constants, so this is a sanity
        // bound: the reported exact score is within the start's scale and no
        // worse than the (unpolished) plan start.
        let plan_start = target.score(&cpu_pred(&f, &start_tree), false);
        assert!(
            out[0].sse <= start_score.max(plan_start) + 1e-12,
            "device score {} worse than start {}",
            out[0].sse,
            start_score.max(plan_start)
        );
    }

    /// #288: one batch entry point scores every exact order/rank statistic,
    /// bit-identically to the CPU for the same `f64` prediction vector.
    #[test]
    fn score_batch_matches_the_cpu_for_every_statistic() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let target = RankTarget::new(&f.y).unwrap();
        for t in [linear(), tied(), flat()] {
            let p = plan(&t);
            let pred = cpu_pred(&f, &t);
            let constants = vec![p.c0.clone()];
            // The interquartile mean is scored by adding the window in
            // ascending order, the CPU's own left-to-right sum, so it is bit
            // for bit like the single order statistics.
            for stat in [ScoreStat::Median, ScoreStat::Interquartile, ScoreStat::Maximum] {
                let got = ev
                    .score_stat_batch(&[&p], &constants, stat)
                    .expect("device stat call")[0]
                    .expect("a finite segment");
                let want = crate::metrics::score(&pred, &f.y, stat.metric());
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{} differs: {got} vs {want}",
                    stat.name()
                );
            }
            for (stat, signed) in [
                (ScoreStat::RankCorrelation, false),
                (ScoreStat::RankCorrelationSigned, true),
            ] {
                let got = ev
                    .score_stat_batch(&[&p], &constants, stat)
                    .expect("device rank call")[0]
                    .expect("a finite segment");
                let want = target.score(&pred, signed);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{} differs: {got} vs {want}",
                    stat.name()
                );
            }
        }
    }

    /// #288: AUC is the exact integer rank (Mann–Whitney) statistic: it is
    /// bit-identical to the CPU for the same prediction vector, all-equal
    /// targets are chance, and a tied pair contributes exactly one half.
    #[test]
    fn score_batch_auc_is_exact_and_matches_the_cpu() {
        let mut f = fixture(90);
        // Two-valued target: every third row positive. The midpoint rule then
        // labels exactly these rows, and the rank stage's B is the label mask.
        for (i, y) in f.y.iter_mut().enumerate() {
            *y = if i % 3 == 0 { 1.0 } else { -1.0 };
        }
        f.y32 = f.y.iter().map(|&v| v as f32).collect();
        let Some(mut ev) = evaluator(&f) else { return };
        let (npos, nneg) = ev.auc_classes().expect("a two-valued target");
        assert_eq!(npos + nneg, f.n);
        assert_eq!(npos, f.n.div_ceil(3));
        for t in [linear(), tied(), flat()] {
            let p = plan(&t);
            let pred = cpu_pred(&f, &t);
            let got = ev
                .score_stat_batch(&[&p], &[p.c0.clone()], ScoreStat::Auc)
                .expect("device AUC call")[0]
                .expect("a finite segment");
            let want = crate::metrics::score(&pred, &f.y, Metric::Auc);
            assert_eq!(got.to_bits(), want.to_bits(), "AUC differs: {got} vs {want}");
        }
        // One positive and one negative with the same prediction: the pair is
        // a tie, so U = 1/2 and the error 1 − AUC = 1/2 exactly.
        let tie = node(Op::Mul, vec![constant(0.0), input(0)]);
        let p = plan(&tie);
        let got = ev.score_stat_batch(&[&p], &[p.c0.clone()], ScoreStat::Auc).unwrap()[0].unwrap();
        assert_eq!(got, 0.5, "a constant prediction must be exactly chance");
        // A single-class target has no positive/negative pair: chance, never
        // 0/0.
        let mut one = fixture(24);
        one.y.iter_mut().for_each(|y| *y = 2.0);
        one.y32 = one.y.iter().map(|&v| v as f32).collect();
        let Some(mut ev1) = evaluator(&one) else { return };
        let p = plan(&linear());
        assert_eq!(ev1.auc_classes(), Some((0, 0)));
        let got = ev1
            .score_stat_batch(&[&p], &[p.c0.clone()], ScoreStat::Auc)
            .expect("single-class AUC")[0]
            .expect("chance");
        assert_eq!(got, 0.5);
    }

    /// #288: ties make `U` a half-integer, not only an integer. A constant
/// prediction over an odd `npos·nneg` is the smallest case: every pair is a
/// tie and `1 − AUC` is exactly `0.5`, matching the CPU bit for bit.
#[test]
fn auc_handles_an_odd_number_of_positive_negative_pairs() {
    let y = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
    let pred = vec![0.0; y.len()];
    let target = RankTarget::new(&y).expect("finite target");
    let classes = binary_class_sizes(&y).expect("two classes");
    assert_eq!(classes, (3, 5));
    let got = auc_error_from_rank_sums(
        target.sums(&pred).expect("cpu sums"),
        y.len(),
        classes,
    )
    .expect("the odd pair count is valid");
    assert_eq!(got, 0.5);
    assert_eq!(
        got.to_bits(),
        crate::metrics::score(&pred, &y, Metric::Auc).to_bits()
    );
}

/// #288: the label rule is verified, not assumed. A target with three
    /// distinct values or a non-finite row has no two-class rank shape, so a
    /// native AUC declines instead of scoring a fabricated label mask.
    #[test]
    fn auc_declines_a_target_that_is_not_binary() {
        assert_eq!(binary_class_sizes(&[0.0, 1.0]), Some((1, 1)));
        assert_eq!(binary_class_sizes(&[1.0, 1.0]), Some((0, 0)));
        assert_eq!(binary_class_sizes(&[0.0, 1.0, 2.0]), None);
        assert_eq!(binary_class_sizes(&[0.0, f64::NAN]), None);
        let mut f = fixture(24);
        for (i, y) in f.y.iter_mut().enumerate() {
            *y = [0.0, 1.0, 2.0][i % 3];
        }
        f.y32 = f.y.iter().map(|&v| v as f32).collect();
        let Some(mut ev) = evaluator(&f) else { return };
        assert_eq!(ev.auc_classes(), None);
        let p = plan(&linear());
        assert_eq!(
            ev.score_stat_batch(&[&p], &[p.c0.clone()], ScoreStat::Auc),
            None,
            "a three-valued target must decline the native AUC"
        );
    }

    /// #288: batch composition and repetition cannot change a statistic, and a
    /// constant prediction is exactly `1.0` for the rank family (never NaN).
    #[test]
    fn score_batch_is_batch_composition_independent() {
        let f = fixture(80);
        let Some(mut ev) = evaluator(&f) else { return };
        let trees = [linear(), tied(), flat()];
        let plans: Vec<_> = trees.iter().map(plan).collect();
        let constants: Vec<_> = plans.iter().map(|p| p.c0.clone()).collect();
        for stat in [
            ScoreStat::Median,
            ScoreStat::Interquartile,
            ScoreStat::Maximum,
            ScoreStat::RankCorrelation,
            ScoreStat::RankCorrelationSigned,
        ] {
            let alone = ev
                .score_stat_batch(&[&plans[0]], &constants[..1], stat)
                .expect("single call");
            let batched = ev
                .score_stat_batch(&plans.iter().collect::<Vec<_>>(), &constants, stat)
                .expect("batched call");
            let again = ev
                .score_stat_batch(&plans.iter().collect::<Vec<_>>(), &constants, stat)
                .expect("repeat call");
            assert_eq!(
                alone[0].map(f64::to_bits),
                batched[0].map(f64::to_bits),
                "{} changed with batch composition",
                stat.name()
            );
            assert_eq!(
                batched.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>(),
                again.iter().map(|v| v.map(f64::to_bits)).collect::<Vec<_>>(),
                "{} is not deterministic",
                stat.name()
            );
            if stat.is_rank() {
                // flat() is a constant prediction: rho is 0 by the CPU rule.
                let flat = batched[2].expect("a finite segment");
                assert_eq!(flat, 1.0);
            }
        }
    }

    /// #288: a non-finite prediction is `None` (the caller's `+inf`) for every
    /// statistic, exactly like the CPU scorer; never a finite sentinel.
    #[test]
    fn score_batch_non_finite_predictions_are_unusable() {
        let f = fixture(24);
        let Some(mut ev) = evaluator(&f) else { return };
        let bad = node(
            Op::Ln,
            vec![node(
                Op::Mul,
                vec![constant(-1.0), node(Op::Add, vec![input(0), constant(1.0)])],
            )],
        );
        let p = plan(&bad);
        let pred = cpu_pred(&f, &bad);
        assert!(pred.iter().any(|v| !v.is_finite()), "the fixture is non-finite");
        for stat in [
            ScoreStat::Median,
            ScoreStat::Interquartile,
            ScoreStat::Maximum,
            ScoreStat::RankCorrelation,
            ScoreStat::RankCorrelationSigned,
        ] {
            let got = ev.score_stat_batch(&[&p], &[p.c0.clone()], stat).expect("a call");
            assert_eq!(got, vec![None], "{} fabricated a finite value", stat.name());
        }
    }

    /// A tree whose prediction is not finite is reported, not scored: the
    /// caller refits it on the CPU and labels the fallback.
    #[test]
    fn non_finite_predictions_are_refused_not_scored() {
        let f = fixture(32);
        let Some(mut ev) = evaluator(&f) else { return };
        // ln(-1 - x0) is NaN everywhere on this sample.
        let bad = node(
            Op::Ln,
            vec![node(Op::Mul, vec![constant(-1.0), node(Op::Add, vec![input(0), constant(1.0)])])],
        );
        let p = plan(&bad);
        let Some(res) = ev.rank_score_trials(&[&p], &[vec![p.c0.clone()]]) else {
            // A refusal of the whole call is also a correct answer.
            return;
        };
        assert!(res[0][0].is_none(), "a non-finite prediction must not be scored");
        let target = RankTarget::new(&f.y).unwrap();
        assert_eq!(score_of(res[0][0], &target, false), f64::INFINITY);
    }
}

/// #288: paired sort-based scoring measurement.
///
/// Run with
/// `cargo test --release --features cuda --lib bench_score_batch -- --ignored --nocapture`.
/// `RCSR_BENCH_N` (default 100 000) and `RCSR_BENCH_P` (default 8) override the
/// sample and candidate count. Prints ns/row for the exact CPU rank statistic,
/// the device rank/median/AUC scorer, the wave count and the device temp bytes
/// implied by `RANK_WAVE_BYTES`. The 2.456 M-row, `P = 447` cell of the issue is
/// reproducible with `RCSR_BENCH_N=2456306 RCSR_BENCH_P=447` on a device with
/// enough memory; it is kept out of the default suite because a wave allocates
/// `64 B/row`.
#[cfg(test)]
mod score_batch_bench {
    use super::*;
    use crate::fit::gpu_fit_plan_reason;
    use crate::metrics::RankTarget;

    fn fixture(n: usize) -> (Vec<f32>, Vec<f64>, usize, Vec<f32>, Vec<f64>, Vec<Vec<f64>>) {
        let nv = 2;
        let (mut c32, mut c64) = (vec![0.0f32; nv * n], vec![0.0f64; nv * n]);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = -3.0 + 6.0 * i as f64 / (n - 1) as f64;
            let v = 1.7 * (0.9 * t).sin() + 0.4 * t - 0.2;
            x.push(vec![t, 0.5 * t]);
            y.push(v);
            c32[i] = t as f32;
            c32[n + i] = (0.5 * t) as f32;
            c64[i] = t;
            c64[n + i] = 0.5 * t;
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        (c32, c64, nv, y32, y, x)
    }

    #[test]
    #[ignore]
    fn bench_score_batch() {
        let n: usize = std::env::var("RCSR_BENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100_000);
        let p: usize = std::env::var("RCSR_BENCH_P")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let (c32, c64, nv, y32, y, x) = fixture(n);
        let Some(mut ev) = GpuEvaluator::new(&c32, n, nv, &y32) else {
            eprintln!("skipping GPU measurement: no usable CUDA device");
            return;
        };
        assert!(ev.set_fit_data(&c64, &y, &[]), "f64 data upload");
        let tree = crate::expr::node(
            Op::Add,
            vec![
                crate::expr::node(
                    Op::Mul,
                    vec![crate::expr::constant(0.8), crate::expr::input(0)],
                ),
                crate::expr::constant(0.1),
            ],
        );
        let plan = gpu_fit_plan_reason(&tree).expect("an eligible plan");
        let constants: Vec<Vec<f64>> = (0..p)
            .map(|k| {
                let mut c = plan.c0.clone();
                c[0] *= 1.0 + 0.001 * (k as f64 + 1.0);
                c
            })
            .collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = (0..p).map(|_| &plan).collect();
        let pred: Vec<f64> = x.iter().map(|r| crate::expr::eval(&tree, r)).collect();
        let target = RankTarget::new(&y).expect("finite target");
        let cpu_start = std::time::Instant::now();
        for _ in 0..p {
            let _ = target.sums(&pred).expect("cpu sums");
        }
        let cpu_s = cpu_start.elapsed().as_secs_f64();
        let per_seg = (n as u64).saturating_mul(64).saturating_add(1 << 20);
        let wave_sets = (GpuEvaluator::RANK_WAVE_BYTES / per_seg.max(1)).max(1) as usize;
        for stat in [
            ScoreStat::Median,
            ScoreStat::RankCorrelation,
            ScoreStat::RankCorrelationSigned,
        ] {
            let start = std::time::Instant::now();
            let out = ev
                .score_stat_batch(&refs, &constants, stat)
                .expect("the device scored the batch");
            let s = start.elapsed().as_secs_f64();
            assert_eq!(out.len(), p);
            eprintln!(
                "{}: N={n} P={p} device={:.3} ms ({:.3e} s/row/cand) cpu_rank={:.3} ms ({:.3e} s/row/cand) waves={} wave_charge~{:.1} MB",
                stat.name(),
                s * 1000.0,
                s / (n as f64 * p as f64),
                cpu_s * 1000.0,
                cpu_s / (n as f64 * p as f64),
                p.div_ceil(wave_sets),
                (n as f64 * 64.0 * p.min(wave_sets) as f64) / (1024.0 * 1024.0)
            );
        }
    }
}

/// #227: the exact maximum-error arm on the device.
#[cfg(test)]
mod maximum_tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::fit::{gpu_fit_plan_reason, maximum_finalist, polish_minimax_batch};
    use crate::metrics::{score_weighted, Metric};

    struct Fixture {
        cols32: Vec<f32>,
        cols64: Vec<f64>,
        n: usize,
        nv: usize,
        y32: Vec<f32>,
        y: Vec<f64>,
        x: Vec<Vec<f64>>,
    }

    /// A one-column sample with a gross outlier (so the least-squares start is
    /// a poor minimax start), exact duplicate targets (so maximiser ties
    /// happen) and enough rows for the f64 sweeps to be exercised.
    fn fixture(n: usize) -> Fixture {
        let mut cols32 = vec![0.0f32; n];
        let mut cols64 = vec![0.0f64; n];
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = -2.0 + 4.0 * i as f64 / (n.max(2) - 1) as f64;
            let v = 1.4 * (0.9 * t).sin() + 0.3 * t;
            let v = if i % 5 == 4 && i > 0 { y[i - 1] } else { v };
            let v = if i == n / 3 { 40.0 } else { v };
            x.push(vec![t, 0.0]);
            y.push(v);
            cols32[i] = t as f32;
            cols64[i] = t;
        }
        Fixture {
            cols32,
            cols64,
            n,
            nv: 1,
            y32: y.iter().map(|&v| v as f32).collect(),
            y,
            x,
        }
    }

    fn evaluator(f: &Fixture) -> Option<GpuEvaluator> {
        let mut ev = match GpuEvaluator::new(&f.cols32, f.n, f.nv, &f.y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return None;
            }
        };
        assert!(ev.set_fit_data(&f.cols64, &f.y, &[]), "f64 fit data upload");
        Some(ev)
    }

    fn plan(t: &Node) -> crate::fit::GpuFitPlan {
        gpu_fit_plan_reason(t).expect("an eligible plan")
    }

    /// `a·x + b`: arithmetic only, so the device's f64 forward sweep and
    /// `expr::eval` agree to the last bit and the values compare exactly.
    fn affine() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.7), input(0)]), constant(0.1)],
        )
    }

    fn with_constants(t: &Node, c: &[f64]) -> Node {
        let p = plan(t);
        let mut out = t.clone();
        p.apply(&mut out, c);
        out
    }

    fn cpu_max(f: &Fixture, t: &Node) -> f64 {
        let pred: Vec<f64> = f.x.iter().map(|r| crate::expr::eval(t, r)).collect();
        score_weighted(&pred, &f.y, &[], Metric::Maximum, crate::expr::count_constants(t))
    }

    /// The exact maximum and the lowest maximiser row, on the CPU.
    fn cpu_maximiser(f: &Fixture, t: &Node) -> (f64, usize) {
        let mut best = (0.0f64, 0usize);
        for (row, r) in f.x.iter().enumerate() {
            let v = (crate::expr::eval(t, r) - f.y[row]).abs();
            if v > best.0 || (v == best.0 && row < best.1) {
                best = (v, row);
            }
        }
        best
    }

    /// The device maximum is exactly `Metric::Maximum` of the prediction, and
    /// its maximiser is the lowest row that attains it.
    #[test]
    fn the_device_maximum_is_the_configured_metric_and_its_lowest_maximiser() {
        for n in [5usize, 32, 97, 256] {
            let f = fixture(n);
            let Some(mut ev) = evaluator(&f) else { return };
            let t = affine();
            let p = plan(&t);
            for c in [p.c0.clone(), vec![0.0, 0.0], vec![1.0, 0.5], vec![-1.0, 2.0]] {
                let got = ev
                    .max_score_trials(&[&p], &[vec![c.clone()]])
                    .expect("the device answered")[0][0]
                    .expect("a finite prediction");
                let want = cpu_maximiser(&f, &with_constants(&t, &c));
                assert_eq!(got.value.to_bits(), want.0.to_bits(), "n={n}, c={c:?}");
                assert_eq!(got.row, want.1, "n={n}, c={c:?}");
                assert_eq!(
                    got.value.to_bits(),
                    cpu_max(&f, &with_constants(&t, &c)).to_bits(),
                    "n={n}, c={c:?}: the value is the configured maximum"
                );
            }
        }
    }

    /// A non-unique maximiser: two rows at the same `|r|` report the lowest of
    /// them, and the answer does not depend on how the batch is composed.
    #[test]
    fn a_tied_maximum_reports_the_lowest_row_on_the_device() {
        // y = ±2 on a constant-zero prediction: rows 0 and 1 are both 2 away.
        let n = 8;
        let f = Fixture {
            cols32: vec![0.0f32; n],
            cols64: vec![0.0f64; n],
            n,
            nv: 1,
            y32: (0..n).map(|i| if i % 2 == 0 { 2.0f32 } else { -2.0 }).collect(),
            y: (0..n).map(|i| if i % 2 == 0 { 2.0 } else { -2.0 }).collect(),
            x: vec![vec![0.0]; n],
        };
        let Some(mut ev) = evaluator(&f) else { return };
        let t = node(Op::Mul, vec![constant(0.0), input(0)]);
        let p = plan(&t);
        let one = ev
            .max_score_trials(&[&p], &[vec![p.c0.clone()]])
            .expect("device")[0][0]
            .expect("finite");
        assert_eq!(one.value, 2.0);
        assert_eq!(one.row, 0, "the lowest maximiser is reported");
        // Batched next to other candidates and trials: same answer.
        let three = ev
            .max_score_trials(
                &[&p],
                &[vec![vec![9.0], p.c0.clone(), vec![-9.0]]],
            )
            .expect("device")[0][1]
            .expect("finite");
        assert_eq!(three.value.to_bits(), one.value.to_bits());
        assert_eq!(three.row, one.row);
    }

    /// The level-interval test is exact for an affine coordinate: at the exact
    /// minimax level the interval contains the exact step, one ulp below it is
    /// infeasible, and a constant-only row cannot move the coordinate.
    #[test]
    fn the_level_interval_is_exact_for_an_affine_coordinate() {
        // One constant, targets 1, 2, 4: the minimax constant is 2.5 (level
        // 1.5). Rows 0 and 2 are the tight pair (1.5 ± 1.5 either side of 4
        // and 1), so they pin the step to the single point 2.5.
        let x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64]).collect();
        let y = [1.0, 2.0, 4.0];
        let n = 3;
        let f = Fixture {
            cols32: x.iter().map(|r| r[0] as f32).collect(),
            cols64: x.iter().map(|r| r[0]).collect(),
            n,
            nv: 1,
            y32: y.iter().map(|&v| v as f32).collect(),
            y: y.to_vec(),
            x: x.clone(),
        };
        let Some(mut ev) = evaluator(&f) else { return };
        let t = constant(0.0);
        let p = plan(&t);
        let c = vec![0.0];
        let at = |ev: &mut GpuEvaluator, level: f64| {
            ev.max_intervals(&[&p], &[c.clone()], &[vec![0]], &[vec![level]])
                .expect("device")[0][0]
                .expect("finite rows")
        };
        let exact = at(&mut ev, 1.5);
        assert!(exact.is_feasible(), "level 1.5 is attainable: {exact:?}");
        assert!(
            (exact.lo - 2.5).abs() <= 1e-12 && (exact.hi - 2.5).abs() <= 1e-12,
            "row 2 pins the step at 2.5: {exact:?}"
        );
        let below = at(&mut ev, 1.5 - 1e-9);
        assert!(!below.is_feasible(), "level 1.5 - 1e-9 is not: {below:?}");
        // A looser level admits a range of steps: `|c − y| <= 4` for 1 and 4
        // leaves `c ∈ [0, 5]`, and the middle target never binds.
        let loose = at(&mut ev, 4.0);
        assert!(loose.is_feasible(), "{loose:?}");
        assert_eq!((loose.lo, loose.hi), (0.0, 5.0), "{loose:?}");
    }

    /// The batched device polish reaches the exact 1-D minimiser, is monotone
    /// in the incoming constants, and reports the exact objective of the
    /// constants it returns.
    #[test]
    fn the_device_polish_reaches_the_exact_minimiser_and_is_monotone() {
        // A bare constant on 1, 2, 4: the minimax constant is the midrange 2.5.
        let n = 3;
        let y = [1.0, 2.0, 4.0];
        let f = Fixture {
            cols32: vec![0.0f32; n],
            cols64: vec![0.0f64; n],
            n,
            nv: 1,
            y32: y.iter().map(|&v| v as f32).collect(),
            y: y.to_vec(),
            x: vec![vec![0.0]; n],
        };
        let Some(mut ev) = evaluator(&f) else { return };
        let t = constant(7.0 / 3.0);
        let p = plan(&t);
        let out = polish_minimax_batch(&mut ev, &[&p], &[p.c0.clone()], 8)
            .expect("the device scorer answered");
        assert!((out.constants[0][0] - 2.5).abs() <= 1e-6, "{:?}", out.constants[0]);
        assert!(
            (out.scores[0] - 1.5).abs() <= 1e-6,
            "exact minimax 1.5, got {}",
            out.scores[0]
        );
        // |2.5 − 1| = |2.5 − 4| = 1.5: rows 0 and 2 tie, so the reported
        // maximiser is the lowest of them.
        assert_eq!(out.maximisers[0], Some(0), "the lowest maximiser is reported");

        // The full arm (squared start plus polish) never reports a worse
        // maximum than the constants it was given, and its number is the exact
        // objective of its constants.
        let g = fixture(97);
        let Some(mut ev) = evaluator(&g) else { return };
        let tree = affine();
        let plan = plan(&tree);
        let incoming = cpu_max(&g, &tree);
        let (out, ms) = ev.max_fit_batch(&[&plan], 8);
        assert!(ms > 0.0, "the maximum arm ran on the device");
        assert_eq!(out[0].status, 0, "the fixture candidate is eligible");
        assert!(
            out[0].sse <= incoming,
            "device maximum {} worse than the incoming constants {}",
            out[0].sse,
            incoming
        );
        let applied = with_constants(&tree, &out[0].c);
        assert_eq!(cpu_max(&g, &applied).to_bits(), out[0].sse.to_bits());
        // And the CPU polish on the device constants never beats the finalist.
        let mut tree_mut = applied.clone();
        let finalist = maximum_finalist(&mut tree_mut, &g.x, &g.y, &[], 8);
        assert!(finalist <= out[0].sse);
        assert_eq!(finalist.to_bits(), cpu_max(&g, &tree_mut).to_bits());
    }

    /// A batch whose candidates have different constant counts is fitted for
    /// every candidate: the device declines nothing (`status == 0`) and the
    /// reported objective is the exact maximum of the returned constants.
    #[test]
    fn a_mixed_batch_of_constant_counts_is_fitted_for_every_candidate() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let trees = vec![
            node(Op::Add, vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)]),
            node(
                Op::Add,
                vec![
                    node(
                        Op::Mul,
                        vec![
                            constant(0.8),
                            node(Op::Sin, vec![node(Op::Mul, vec![constant(1.0), input(0)])]),
                        ],
                    ),
                    constant(0.0),
                ],
            ),
            constant(0.25),
        ];
        let plans: Vec<crate::fit::GpuFitPlan> = trees.iter().map(plan).collect();
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
        let (out, ms) = ev.max_fit_batch(&refs, 8);
        assert!(ms > 0.0);
        assert_eq!(out.len(), trees.len());
        for (k, res) in out.iter().enumerate() {
            assert_eq!(res.status, 0, "candidate {k} (p = {}) was declined", plans[k].nconst);
            assert!(res.sse.is_finite());
            let applied = with_constants(&trees[k], &res.c);
            assert_eq!(cpu_max(&f, &applied).to_bits(), res.sse.to_bits());
        }
    }

    /// Results do not depend on how the batch is composed or on repetition.
    #[test]
    fn the_maximum_scan_is_deterministic_across_batch_composition() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let c = vec![0.8, -0.2];
        let alone = ev
            .max_score_trials(&[&p], &[vec![c.clone()]])
            .expect("device")[0][0]
            .expect("finite");
        let three = ev
            .max_score_trials(&[&p], &[vec![c.clone(), p.c0.clone(), vec![0.0, 0.0]]])
            .expect("device")[0][0]
            .expect("finite");
        assert_eq!(alone.value.to_bits(), three.value.to_bits());
        assert_eq!(alone.row, three.row);
        let mixed = ev
            .max_score_trials(&[&p, &p], &[vec![vec![2.0, 2.0]], vec![c.clone()]])
            .expect("device");
        assert_eq!(mixed[1][0].expect("finite").value.to_bits(), alone.value.to_bits());
        let again = ev
            .max_score_trials(&[&p], &[vec![c.clone()]])
            .expect("device")[0][0]
            .expect("finite");
        assert_eq!(again.value.to_bits(), alone.value.to_bits());
        assert_eq!(again.row, alone.row);
    }

    /// A non-finite prediction is refused, never scored as a huge finite one.
    #[test]
    fn a_non_finite_prediction_is_not_a_maximum() {
        let f = fixture(32);
        let Some(mut ev) = evaluator(&f) else { return };
        let bad = node(
            Op::Ln,
            vec![node(Op::Mul, vec![constant(-1.0), node(Op::Add, vec![input(0), constant(1.0)])])],
        );
        let p = plan(&bad);
        let Some(res) = ev.max_score_trials(&[&p], &[vec![p.c0.clone()]]) else {
            return; // a whole-call refusal is also correct
        };
        assert!(res[0][0].is_none(), "a non-finite prediction must not be scored");
    }

    /// A rank-deficient squared start is labelled and left to the CPU arm.
    #[test]
    fn an_ill_conditioned_squared_start_is_labelled() {
        let f = fixture(48);
        let Some(mut ev) = evaluator(&f) else { return };
        // Two constants that only ever appear as a sum: the squared design is
        // rank deficient, so the LM start cannot identify them.
        let t = node(Op::Add, vec![constant(2.0), constant(1.0)]);
        let p = plan(&t);
        let (out, ms) = ev.max_fit_batch(&[&p], 8);
        assert!(ms >= 0.0);
        assert_ne!(out[0].status, 0, "a singular start must be labelled, not polished");
    }

    /// `iters == 0` (`--no-fit`) is a true no-op: no LM step, no polish, the
    /// incoming constants unchanged.
    #[test]
    fn no_fit_does_not_enter_the_maximum_arm() {
        let f = fixture(32);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let (out, ms) = ev.max_fit_batch(&[&p], 0);
        assert_eq!(ms, 0.0);
        assert_eq!(out[0].status, 0);
        assert_eq!(out[0].c, p.c0);
    }
}

/// #226: the exact median arm on the device.
#[cfg(test)]
mod order_stat_tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::fit::{fit_weighted_metric, gpu_fit_plan_reason, refit_finalists_exact};
    use crate::metrics::{score_weighted, Metric};

    struct Fixture {
        cols32: Vec<f32>,
        cols64: Vec<f64>,
        n: usize,
        nv: usize,
        y32: Vec<f32>,
        y: Vec<f64>,
        x: Vec<Vec<f64>>,
    }

    /// A 2-column sample with exact duplicate targets (residual ties), a few
    /// gross outliers and one column that is constant zero, so ties, `-0.0`
    /// and a fully tied residual multiset can all be exercised.
    fn fixture(n: usize) -> Fixture {
        let nv = 2;
        let mut cols32 = vec![0.0f32; nv * n];
        let mut cols64 = vec![0.0f64; nv * n];
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = -2.0 + 4.0 * i as f64 / (n.max(2) - 1) as f64;
            // Every fifth row repeats the previous target exactly (ties).
            let v = 1.4 * (0.9 * t).sin() + 0.3 * t;
            let v = if i % 5 == 4 && i > 0 { y[i - 1] } else { v };
            let v = if i % 11 == 7 { 30.0 } else { v };
            x.push(vec![t, 0.0]);
            y.push(v);
            cols32[i] = t as f32;
            cols64[i] = t;
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        Fixture {
            cols32,
            cols64,
            n,
            nv,
            y32,
            y,
            x,
        }
    }

    fn evaluator(f: &Fixture) -> Option<GpuEvaluator> {
        let mut ev = match GpuEvaluator::new(&f.cols32, f.n, f.nv, &f.y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return None;
            }
        };
        assert!(ev.set_fit_data(&f.cols64, &f.y, &[]), "f64 fit data upload");
        Some(ev)
    }

    fn plan(t: &Node) -> crate::fit::GpuFitPlan {
        gpu_fit_plan_reason(t).expect("an eligible plan")
    }

    /// `a·x + b`: arithmetic-only, so the device's f64 forward sweep and
    /// `expr::eval` agree to the last bit and the order statistics can be
    /// compared exactly.
    fn affine() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.7), input(0)]), constant(0.1)],
        )
    }

    fn cpu_median(f: &Fixture, t: &Node) -> f64 {
        cpu_metric(f, t, Metric::Median)
    }

    /// #232: the configured interquartile window mean of the same rows.
    fn cpu_iqr(f: &Fixture, t: &Node) -> f64 {
        cpu_metric(f, t, Metric::Interquartile)
    }

    fn cpu_metric(f: &Fixture, t: &Node, metric: Metric) -> f64 {
        let pred: Vec<f64> = f.x.iter().map(|r| crate::expr::eval(t, r)).collect();
        score_weighted(&pred, &f.y, &[], metric, crate::expr::count_constants(t))
    }

    /// The same tree with the plan's constant slots set to `c`.
    fn with_constants(t: &Node, c: &[f64]) -> Node {
        let p = plan(t);
        let mut out = t.clone();
        p.apply(&mut out, c);
        out
    }

    fn constants_of(t: &Node) -> Vec<f64> {
        crate::fit::constants_slots(t)
    }

    /// The device median is the configured metric for odd and even row counts,
    /// with ties and with a constant prediction.
    extern "C" {
        fn rc_os_test_force_cub_fail(on: i32);
    }

    /// #410: a refused CUB size query fails the order-statistic sort call,
    /// `order_stat_score_trials` reports `None` (the caller keeps the CPU arm),
    /// and the next call on the same context works.
    #[test]
    fn refused_cub_query_fails_the_order_stat_call_then_recovers() {
        use crate::fit::OrderStat;
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let p = plan(&affine());
        let trials = [vec![p.c0.clone()]];
        let ok = ev.order_stat_score_trials(&[&p], &trials, OrderStat::Median);
        assert!(ok.is_some(), "baseline median call");
        unsafe { rc_os_test_force_cub_fail(1) };
        let refused = ev.order_stat_score_trials(&[&p], &trials, OrderStat::Median);
        unsafe { rc_os_test_force_cub_fail(0) };
        assert!(refused.is_none(), "a refused CUB query must fail the call");
        let again = ev
            .order_stat_score_trials(&[&p], &trials, OrderStat::Median)
            .expect("no error may stay pending");
        assert_eq!(again[0][0].map(f64::to_bits), ok.unwrap()[0][0].map(f64::to_bits));
    }

    #[test]
    fn the_device_median_is_the_configured_metric() {
        for n in [5usize, 6, 31, 32, 97, 128] {
            let f = fixture(n);
            let Some(mut ev) = evaluator(&f) else { return };
            let t = affine();
            let p = plan(&t);
            for c in [
                p.c0.clone(),
                vec![0.0, 0.0],
                vec![1.0, 0.5],
                vec![-1.0, 2.0],
            ] {
                let got = ev
                    .median_score_trials(&[&p], &[vec![c.clone()]])
                    .expect("the device answered")
                    [0][0]
                    .expect("a finite prediction");
                let want = cpu_median(&f, &with_constants(&t, &c));
                assert_eq!(got.to_bits(), want.to_bits(), "n={n}, c={c:?}");
            }
        }
    }

    /// A residual multiset that is entirely tied (a constant prediction on a
    /// constant target) still selects the exact value, and the even-count mean
    /// of two equal middles is exact.
    #[test]
    fn entirely_tied_residuals_select_the_exact_value() {
        for n in [2usize, 3, 4, 5] {
            let f = Fixture {
                cols32: vec![0.0f32; n],
                cols64: vec![0.0f64; n],
                n,
                nv: 1,
                y32: vec![1.0f32; n],
                y: vec![1.0f64; n],
                x: vec![vec![0.0]; n],
            };
            let Some(mut ev) = evaluator(&f) else { return };
            let t = node(Op::Mul, vec![constant(1.0), input(0)]);
            let p = plan(&t);
            let got = ev
                .median_score_trials(&[&p], &[vec![p.c0.clone()]])
                .expect("the device answered")[0][0]
                .expect("finite");
            // pred = c0 * 0 = 0, y = 1 -> every |r| = 1.
            assert_eq!(got, 1.0, "n={n}");
        }
    }

    /// Results do not depend on how the batch is composed: the same trial
    /// alone, repeated, or next to unrelated candidates and trials.
    #[test]
    fn selection_is_deterministic_across_batch_composition() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let c = vec![0.8, -0.2];
        let alone = ev
            .median_score_trials(&[&p], &[vec![c.clone()]])
            .expect("alone")
            [0][0]
            .expect("finite");
        let triples = ev
            .median_score_trials(&[&p], &[vec![c.clone(), p.c0.clone(), vec![0.0, 0.0]]])
            .expect("three trials")
            [0][0]
            .expect("finite");
        assert_eq!(alone.to_bits(), triples.to_bits());
        let batch = ev
            .median_score_trials(&[&p, &p], &[vec![vec![2.0, 2.0]], vec![c.clone()]])
            .expect("mixed batch");
        assert_eq!(batch[1][0].expect("finite").to_bits(), alone.to_bits());
        let again = ev
            .median_score_trials(&[&p], &[vec![c.clone()]])
            .expect("again")
            [0][0]
            .expect("finite");
        assert_eq!(again.to_bits(), alone.to_bits(), "repeated calls must agree");
    }

    /// A non-finite prediction is refused, never scored as a huge finite one.
    #[test]
    fn a_non_finite_prediction_is_not_a_median() {
        let f = fixture(32);
        let Some(mut ev) = evaluator(&f) else { return };
        let bad = node(
            Op::Ln,
            vec![node(Op::Mul, vec![constant(-1.0), node(Op::Add, vec![input(0), constant(1.0)])])],
        );
        let p = plan(&bad);
        let Some(res) = ev.median_score_trials(&[&p], &[vec![p.c0.clone()]]) else {
            return; // a whole-call refusal is also correct
        };
        assert!(res[0][0].is_none(), "a non-finite prediction must not be scored");
    }

    /// The batched device polish commits the same constants as the exact CPU
    /// ladder from the same start, and `median_fit_batch` never reports a
    /// worse median than the constants it was given.
    #[test]
    fn the_device_polish_matches_the_cpu_polish_from_the_same_start() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let start = affine();
        let p = plan(&start);
        for from in [p.c0.clone(), vec![1.0, 0.5], vec![-1.0, 2.0]] {
            let batched = crate::fit::polish_median_batch(
                &mut ev,
                &[&p],
                std::slice::from_ref(&from),
                8,
            )
            .expect("the device scorer answered");
            let mut cpu = with_constants(&start, &from);
            crate::fit::coordinate_polish(&mut cpu, &f.x, &f.y, &[], Metric::Median, 8);
            assert_eq!(
                batched.constants[0]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                constants_of(&cpu).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "from {from:?}: the device ladder must commit the CPU ladder"
            );
            assert_eq!(
                batched.scores[0].to_bits(),
                cpu_median(&f, &cpu).to_bits(),
                "from {from:?}"
            );
        }

        // The full arm (squared start plus polish) is monotone in the median of
        // the constants it was given.
        let (out, ms) = ev.median_fit_batch(&[&p], 8);
        assert!(ms > 0.0, "the median arm ran on the device");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].status, 0, "the fixture candidate is eligible");
        let incoming = cpu_median(&f, &start);
        assert!(
            out[0].sse <= incoming + 1e-12,
            "device median {} worse than the incoming constants {}",
            out[0].sse,
            incoming
        );
        // And the reported value is the exact median of the reported constants.
        assert_eq!(
            cpu_median(&f, &with_constants(&start, &out[0].c)).to_bits(),
            out[0].sse.to_bits()
        );
    }

    /// A rank-deficient design (two constants that only ever appear as a sum)
    /// is labelled ill-conditioned and handed to the CPU; the arm never
    /// polishes from a singular start.
    #[test]
    fn an_ill_conditioned_start_is_labelled_and_not_polished() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = node(Op::Add, vec![constant(2.0), constant(1.0)]);
        let p = plan(&t);
        let (out, _) = ev.median_fit_batch(&[&p], 8);
        assert_eq!(
            out[0].status, 4,
            "the rank-deficient normal equations must be labelled"
        );
    }

    /// `--no-fit` never moves a constant through the device arm.
    #[test]
    fn the_device_arm_is_a_no_op_with_zero_iterations() {
        let f = fixture(48);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let (out, _) = ev.median_fit_batch(&[&p], 0);
        assert_eq!(out[0].status, 0);
        assert_eq!(out[0].c, p.c0);
    }

    /// #226 measurement (ignored): the paired fit-stage timing the cost model
    /// is calibrated with. For each `n` it times the exact CPU arm, the CPU
    /// polish tail from the device's own warm start, the device arm, and one
    /// six-trial scorer call. Run with
    /// `cargo test --release --features cuda --lib median_tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement: run explicitly with --ignored --nocapture"]
    fn bench_median_fit_stage() {
        let ns: Vec<usize> = std::env::var("RCSR_BENCH_ROWS")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
            .unwrap_or_else(|| {
                vec![1_024, 2_048, 4_096, 10_000, 20_000, 50_000, 154_000]
            });
        let reps: usize = std::env::var("RCSR_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let tree = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(1.2),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(0.9), input(0)])]),
                    ],
                ),
                node(Op::Mul, vec![constant(0.3), input(0)]),
            ],
        );
        println!("n\tcpu_fit_ms\tcpu_polish_ms\tgpu_fit_ms\tgpu_polish_ms\tgpu_call_ms\tratio");
        for n in ns {
            let f = fixture(n);
            let Some(mut ev) = evaluator(&f) else { return };
            let p = plan(&tree);
            let mut cpu_fit = f64::INFINITY;
            let mut cpu_polish = f64::INFINITY;
            let mut gpu_fit = f64::INFINITY;
            let mut gpu_call = f64::INFINITY;
            let mut gpu_polish = f64::INFINITY;
            for _ in 0..reps.max(1) {
                let mut cpu = tree.clone();
                let t0 = std::time::Instant::now();
                fit_weighted_metric(&mut cpu, &f.x, &f.y, &[], 8, Metric::Median);
                cpu_fit = cpu_fit.min(t0.elapsed().as_secs_f64() * 1000.0);
                let (warm, _) = ev.fit_batch(&[&p], 8, crate::fit::DeviceLoss::Squared);
                let warm_tree = with_constants(&tree, &warm[0].c);
                let t0 = std::time::Instant::now();
                let mut cpu_tail = warm_tree.clone();
                crate::fit::coordinate_polish(&mut cpu_tail, &f.x, &f.y, &[], Metric::Median, 8);
                cpu_polish = cpu_polish.min(t0.elapsed().as_secs_f64() * 1000.0);
                let t0 = std::time::Instant::now();
                let (_out, _) = ev.median_fit_batch(&[&p], 8);
                gpu_fit = gpu_fit.min(t0.elapsed().as_secs_f64() * 1000.0);
                let t0 = std::time::Instant::now();
                let _ = crate::fit::polish_median_batch(&mut ev, &[&p], &[warm[0].c.clone()], 8);
                gpu_polish = gpu_polish.min(t0.elapsed().as_secs_f64() * 1000.0);
                let mut trials = Vec::new();
                for (mul, sign) in crate::fit::LADDER {
                    let mut c = warm[0].c.clone();
                    c[0] = c[0] + sign * mul * c[0].abs().max(1.0);
                    trials.push(c);
                }
                let t0 = std::time::Instant::now();
                let _ = ev.median_score_trials(&[&p], &[trials]);
                gpu_call = gpu_call.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "{n}\t{cpu_fit:.2}\t{cpu_polish:.2}\t{gpu_fit:.2}\t{gpu_polish:.2}\t{gpu_call:.2}\t{:.2}",
                cpu_polish / gpu_polish
            );
        }
    }

    /// A finalist refit on the median is exact and cannot be worse.
    #[test]
    fn the_device_arm_never_reports_a_worse_median_than_its_start() {
        let f = fixture(80);
        let Some(mut ev) = evaluator(&f) else { return };
        // Start deliberately away from the optimum.
        let t = with_constants(&affine(), &[4.0, -3.0]);
        let p = plan(&t);
        let before = cpu_median(&f, &t);
        let (out, _) = ev.median_fit_batch(&[&p], 8);
        assert_eq!(out[0].status, 0);
        assert!(out[0].sse <= before, "{} > {}", out[0].sse, before);
        let mut tree = with_constants(&affine(), &out[0].c);
        let after =
            refit_finalists_exact(std::slice::from_mut(&mut tree), &f.x, &f.y, &[], 8, Metric::Median);
        assert!(after[0] <= out[0].sse);
        let mut expect = with_constants(&affine(), &out[0].c);
        let via_cpu = fit_weighted_metric(&mut expect, &f.x, &f.y, &[], 8, Metric::Median);
        assert_eq!(via_cpu.to_bits(), after[0].to_bits());
        assert_eq!(constants_of(&tree), constants_of(&expect));
    }

    // -- #232: the interquartile arm ----------------------------------------

    /// Relative agreement the interquartile score must meet. The device sums
    /// the middle-50 % window with a fixed 256-thread tree reduction while the
    /// CPU sums the sorted slice left to right, so the two differ by rounding;
    /// `1e-12` relative is the tolerance `l1-quantile.md` §7 prescribes.
    const IQR_REL_TOL: f64 = 1e-12;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        let scale = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        (a - b).abs() <= rel * scale
    }

    /// The device interquartile mean is the configured metric for odd, even and
    /// sub-window row counts, including a negative start and a zero column.
    #[test]
    fn the_device_interquartile_is_the_configured_metric() {
        for n in [2usize, 3, 4, 5, 6, 7, 31, 32, 97, 128] {
            let f = fixture(n);
            let Some(mut ev) = evaluator(&f) else { return };
            let t = affine();
            let p = plan(&t);
            for c in [
                p.c0.clone(),
                vec![0.0, 0.0],
                vec![1.0, 0.5],
                vec![-1.0, 2.0],
                vec![12.0, -7.0],
            ] {
                let got = ev
                    .interquartile_score_trials(&[&p], &[vec![c.clone()]])
                    .expect("the device answered")[0][0]
                    .expect("a finite prediction");
                let want = cpu_iqr(&f, &with_constants(&t, &c));
                assert!(
                    close(got, want, IQR_REL_TOL),
                    "n={n}, c={c:?}: device {got} vs CPU {want}"
                );
            }
        }
    }

    /// A window boundary tie selects the same value as the CPU: ranks, not a
    /// value interval. Residuals `1, 2, 3, 3` have window mean `2.5`; the
    /// value-range shortcut would report `8/3`.
    #[test]
    fn a_top_boundary_tie_selects_the_rank_window_not_the_value_range() {
        let n = 4usize;
        let f = Fixture {
            cols32: vec![0.0f32; n],
            cols64: vec![0.0f64; n],
            n,
            nv: 1,
            y32: vec![1.0f32, 2.0, 3.0, 3.0],
            y: vec![1.0, 2.0, 3.0, 3.0],
            x: vec![vec![0.0]; n],
        };
        let Some(mut ev) = evaluator(&f) else { return };
        let t = node(Op::Mul, vec![constant(1.0), input(0)]);
        let p = plan(&t);
        let got = ev
            .interquartile_score_trials(&[&p], &[vec![p.c0.clone()]])
            .expect("the device answered")[0][0]
            .expect("finite");
        assert!(close(got, 2.5, IQR_REL_TOL), "got {got}");
        assert!(!close(got, 8.0 / 3.0, 1e-9), "the value-range rule must not be used");
        assert!(close(cpu_iqr(&f, &t), 2.5, 0.0), "the CPU reference is 2.5");
    }

    /// An entirely tied residual multiset (constant prediction, constant
    /// target) selects exactly that value at every size, including `n < 4`
    /// where the CPU arm averages all rows.
    #[test]
    fn entirely_tied_residuals_select_the_exact_window_mean() {
        for n in [2usize, 3, 4, 5, 6, 7, 8, 9] {
            let f = Fixture {
                cols32: vec![0.0f32; n],
                cols64: vec![0.0f64; n],
                n,
                nv: 1,
                y32: vec![1.0f32; n],
                y: vec![1.0f64; n],
                x: vec![vec![0.0]; n],
            };
            let Some(mut ev) = evaluator(&f) else { return };
            let t = node(Op::Mul, vec![constant(1.0), input(0)]);
            let p = plan(&t);
            let got = ev
                .interquartile_score_trials(&[&p], &[vec![p.c0.clone()]])
                .expect("the device answered")[0][0]
                .expect("finite");
            // pred = c0 * 0 = 0, y = 1 -> every |r| = 1.
            assert_eq!(got, 1.0, "n={n}");
        }
    }

    /// Results do not depend on how the batch is composed: alone, repeated, or
    /// next to unrelated candidates and trials — bit for bit, because the
    /// reduction order is fixed by the window and the thread count.
    #[test]
    fn interquartile_selection_is_deterministic_across_batch_composition() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let c = vec![0.8, -0.2];
        let alone = ev
            .interquartile_score_trials(&[&p], &[vec![c.clone()]])
            .expect("alone")[0][0]
            .expect("finite");
        let triples = ev
            .interquartile_score_trials(&[&p], &[vec![c.clone(), p.c0.clone(), vec![0.0, 0.0]]])
            .expect("three trials")[0][0]
            .expect("finite");
        assert_eq!(alone.to_bits(), triples.to_bits());
        let batch = ev
            .interquartile_score_trials(&[&p, &p], &[vec![vec![2.0, 2.0]], vec![c.clone()]])
            .expect("mixed batch");
        assert_eq!(batch[1][0].expect("finite").to_bits(), alone.to_bits());
        let again = ev
            .interquartile_score_trials(&[&p], &[vec![c.clone()]])
            .expect("again")[0][0]
            .expect("finite");
        assert_eq!(again.to_bits(), alone.to_bits(), "repeated calls must agree");
    }

    /// A non-finite prediction is refused, never scored as a huge finite one.
    #[test]
    fn a_non_finite_prediction_is_not_an_interquartile_mean() {
        let f = fixture(32);
        let Some(mut ev) = evaluator(&f) else { return };
        let bad = node(
            Op::Ln,
            vec![node(
                Op::Mul,
                vec![constant(-1.0), node(Op::Add, vec![input(0), constant(1.0)])],
            )],
        );
        let p = plan(&bad);
        let Some(res) = ev.interquartile_score_trials(&[&p], &[vec![p.c0.clone()]]) else {
            return; // a whole-call refusal is also correct
        };
        assert!(
            res[0][0].is_none(),
            "a non-finite prediction must not be scored"
        );
    }

    /// The batched device polish commits the same ladder rungs as the exact CPU
    /// `coordinate_polish` from the same start. The scores agree to the window
    /// reduction's rounding, and the rungs are 1 % apart, so agreement at
    /// `1e-9` relative identifies the same rung.
    #[test]
    fn the_device_interquartile_polish_matches_the_cpu_polish() {
        let f = fixture(96);
        let Some(mut ev) = evaluator(&f) else { return };
        let start = affine();
        let p = plan(&start);
        for from in [p.c0.clone(), vec![1.0, 0.5], vec![-1.0, 2.0]] {
            let batched = crate::fit::polish_interquartile_batch(
                &mut ev,
                &[&p],
                std::slice::from_ref(&from),
                8,
            )
            .expect("the device scorer answered");
            let mut cpu = with_constants(&start, &from);
            crate::fit::coordinate_polish(&mut cpu, &f.x, &f.y, &[], Metric::Interquartile, 8);
            let device_c = &batched.constants[0];
            let cpu_c = constants_of(&cpu);
            for (a, b) in device_c.iter().zip(&cpu_c) {
                assert!(
                    close(*a, *b, 1e-9),
                    "from {from:?}: device {device_c:?} vs CPU {cpu_c:?}"
                );
            }
            assert!(
                close(batched.scores[0], cpu_iqr(&f, &cpu), 1e-12),
                "from {from:?}: score {} vs {}",
                batched.scores[0],
                cpu_iqr(&f, &cpu)
            );
        }
    }

    /// The full arm (squared start plus interquartile polish) is monotone in
    /// the window mean of the constants it was given, and its reported value is
    /// the exact CPU interquartile mean of the reported constants.
    #[test]
    fn the_device_arm_never_reports_a_worse_interquartile_than_its_start() {
        let f = fixture(80);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = with_constants(&affine(), &[4.0, -3.0]);
        let p = plan(&t);
        let before = cpu_iqr(&f, &t);
        let (out, ms) = ev.interquartile_fit_batch(&[&p], 8);
        assert!(ms > 0.0, "the interquartile arm ran on the device");
        assert_eq!(out[0].status, 0, "the fixture candidate is eligible");
        assert!(out[0].sse <= before, "{} > {}", out[0].sse, before);
        let reported = cpu_iqr(&f, &with_constants(&affine(), &out[0].c));
        assert!(close(out[0].sse, reported, 1e-12), "{} vs {reported}", out[0].sse);
        // The exact CPU finalist refit is the same arm and never worse.
        let mut tree = with_constants(&affine(), &out[0].c);
        let after = refit_finalists_exact(
            std::slice::from_mut(&mut tree),
            &f.x,
            &f.y,
            &[],
            8,
            Metric::Interquartile,
        );
        assert!(after[0] <= out[0].sse);
        let mut expect = with_constants(&affine(), &out[0].c);
        let via_cpu = fit_weighted_metric(&mut expect, &f.x, &f.y, &[], 8, Metric::Interquartile);
        assert_eq!(via_cpu.to_bits(), after[0].to_bits());
        assert_eq!(constants_of(&tree), constants_of(&expect));
    }

    /// A rank-deficient design is labelled ill-conditioned and handed to the
    /// CPU; the interquartile arm never polishes from a singular start.
    #[test]
    fn an_ill_conditioned_interquartile_start_is_labelled() {
        let f = fixture(64);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = node(Op::Add, vec![constant(2.0), constant(1.0)]);
        let p = plan(&t);
        let (out, _) = ev.interquartile_fit_batch(&[&p], 8);
        assert_eq!(
            out[0].status, 4,
            "the rank-deficient normal equations must be labelled"
        );
    }

    /// `--no-fit` never moves a constant through the interquartile arm.
    #[test]
    fn the_interquartile_arm_is_a_no_op_with_zero_iterations() {
        let f = fixture(48);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let (out, _) = ev.interquartile_fit_batch(&[&p], 0);
        assert_eq!(out[0].status, 0);
        assert_eq!(out[0].c, p.c0);
    }

    /// #232 measurement (ignored): the paired fit-stage timing the interquartile
    /// cost model is calibrated with. For each `n` it times the exact CPU arm,
    /// the CPU interquartile polish tail from the device's own warm start, the
    /// device arm, and one six-trial scorer call. Run with
    /// `cargo test --release --features cuda --lib order_stat_tests::bench_interquartile
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement: run explicitly with --ignored --nocapture"]
    fn bench_interquartile_fit_stage() {
        let ns: Vec<usize> = std::env::var("RCSR_BENCH_ROWS")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
            .unwrap_or_else(|| vec![1_024, 2_048, 4_096, 10_000, 20_000, 50_000, 154_000]);
        let reps: usize = std::env::var("RCSR_BENCH_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let tree = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(1.2),
                        node(
                            Op::Sin,
                            vec![node(Op::Mul, vec![constant(0.9), input(0)])],
                        ),
                    ],
                ),
                node(Op::Mul, vec![constant(0.3), input(0)]),
            ],
        );
        println!(
            "n\tcpu_fit_ms\tcpu_polish_ms\tgpu_fit_ms\tgpu_polish_ms\tgpu_call_ms\tcpu_polish/gpu_polish"
        );
        for n in ns {
            let f = fixture(n);
            let Some(mut ev) = evaluator(&f) else { return };
            let p = plan(&tree);
            let mut cpu_fit = f64::INFINITY;
            let mut cpu_polish = f64::INFINITY;
            let mut gpu_fit = f64::INFINITY;
            let mut gpu_polish = f64::INFINITY;
            let mut gpu_call = f64::INFINITY;
            for _ in 0..reps.max(1) {
                let mut cpu = tree.clone();
                let t0 = std::time::Instant::now();
                fit_weighted_metric(&mut cpu, &f.x, &f.y, &[], 8, Metric::Interquartile);
                cpu_fit = cpu_fit.min(t0.elapsed().as_secs_f64() * 1000.0);
                let (warm, _) = ev.fit_batch(&[&p], 8, crate::fit::DeviceLoss::Squared);
                let warm_tree = with_constants(&tree, &warm[0].c);
                let t0 = std::time::Instant::now();
                let mut cpu_tail = warm_tree.clone();
                crate::fit::coordinate_polish(
                    &mut cpu_tail,
                    &f.x,
                    &f.y,
                    &[],
                    Metric::Interquartile,
                    8,
                );
                cpu_polish = cpu_polish.min(t0.elapsed().as_secs_f64() * 1000.0);
                let t0 = std::time::Instant::now();
                let (_out, _) = ev.interquartile_fit_batch(&[&p], 8);
                gpu_fit = gpu_fit.min(t0.elapsed().as_secs_f64() * 1000.0);
                let t0 = std::time::Instant::now();
                let _ = crate::fit::polish_interquartile_batch(
                    &mut ev,
                    &[&p],
                    &[warm[0].c.clone()],
                    8,
                );
                gpu_polish = gpu_polish.min(t0.elapsed().as_secs_f64() * 1000.0);
                let mut trials = Vec::new();
                for (mul, sign) in crate::fit::LADDER {
                    let mut c = warm[0].c.clone();
                    c[0] = c[0] + sign * mul * c[0].abs().max(1.0);
                    trials.push(c);
                }
                let t0 = std::time::Instant::now();
                let _ = ev.interquartile_score_trials(&[&p], &[trials]);
                gpu_call = gpu_call.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            println!(
                "{n}\t{cpu_fit:.2}\t{cpu_polish:.2}\t{gpu_fit:.2}\t{gpu_polish:.2}\t{gpu_call:.2}\t{:.2}",
                cpu_polish / gpu_polish
            );
        }
    }
}

/// #334: persistent fit and score buffers. The golden hashes were captured from
/// the per-call allocating path before the change; the persistent path must
/// reproduce them bit for bit.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod persistent_tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::fit::{gpu_fit_plan_reason, DeviceLoss};

    const N: usize = 1_536;

    struct Fixture {
        cols32: Vec<f32>,
        cols64: Vec<f64>,
        y32: Vec<f32>,
        y: Vec<f64>,
    }

    fn fixture(n: usize) -> Fixture {
        let nv = 2;
        let mut cols32 = vec![0.0f32; nv * n];
        let mut cols64 = vec![0.0f64; nv * n];
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
            let t = if i % 3 == 2 { -2.0 + 4.0 * (i - 1) as f64 / (n - 1) as f64 } else { t };
            let s = (((i * 37) % 101) as f64) / 50.0 - 1.0;
            let v = 1.4 * (0.9 * t).sin() + 0.3 * t + 0.1 * s;
            let v = if i % 5 == 4 { 0.0 } else { v };
            cols32[i] = t as f32;
            cols32[n + i] = s as f32;
            cols64[i] = t;
            cols64[n + i] = s;
            y.push(v);
        }
        let y32 = y.iter().map(|&v| v as f32).collect();
        Fixture { cols32, cols64, y32, y }
    }

    fn evaluator(f: &Fixture) -> Option<GpuEvaluator> {
        let mut ev = match GpuEvaluator::new(&f.cols32, N, 2, &f.y32) {
            Some(ev) => ev,
            None => {
                eprintln!("skipping GPU test: no usable CUDA device");
                return None;
            }
        };
        assert!(ev.set_fit_data(&f.cols64, &f.y, &[]), "f64 fit data upload");
        Some(ev)
    }

    fn trees() -> Vec<Node> {
        let lin = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.15)],
        );
        let sine = node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(1.1),
                        node(
                            Op::Sin,
                            vec![node(Op::Add, vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.1)])],
                        ),
                    ],
                ),
                node(Op::Mul, vec![constant(0.2), input(1)]),
            ],
        );
        let tied = node(Op::Abs, vec![node(Op::Add, vec![input(0), constant(-0.25)])]);
        vec![lin, sine, tied]
    }

    struct Fnv(u64);

    impl Fnv {
        fn new() -> Self {
            Fnv(0xcbf2_9ce4_8422_2325)
        }
        fn u64(&mut self, v: u64) {
            for b in v.to_le_bytes() {
                self.0 ^= b as u64;
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        fn f64(&mut self, v: f64) {
            self.u64(v.to_bits());
        }
    }

    fn trials_for(plans: &[&crate::fit::GpuFitPlan], ntrials: usize) -> Vec<Vec<Vec<f64>>> {
        plans
            .iter()
            .map(|p| {
                (0..ntrials)
                    .map(|t| {
                        let mut c = p.c0.clone();
                        for (i, v) in c.iter_mut().enumerate() {
                            *v *= 1.0 + 0.013 * (t as f64 + 1.0) + 0.002 * i as f64;
                        }
                        c
                    })
                    .collect()
            })
            .collect()
    }

    /// One fit batch per loss, one rank batch, both order statistics and the
    /// maximum scan, folded into one hash.
    fn workload(ev: &mut GpuEvaluator, plans: &[&crate::fit::GpuFitPlan]) -> u64 {
        let mut h = Fnv::new();
        for loss in [DeviceLoss::Squared, DeviceLoss::LogSquashed, DeviceLoss::Absolute] {
            let (out, ms) = ev.fit_batch(plans, 6, loss);
            assert!(ms >= 0.0, "{loss:?} fit batch ran");
            for o in &out {
                h.u64(o.status as u64);
                h.f64(o.sse);
                for &c in &o.c {
                    h.f64(c);
                }
            }
        }
        let trials = trials_for(plans, 4);
        let rank = ev.rank_score_trials(plans, &trials).expect("rank batch");
        for row in &rank {
            for s in row {
                match s {
                    Some(s) => {
                        h.u64(s.ab as u64);
                        h.u64((s.ab >> 64) as u64);
                        h.u64(s.aa as u64);
                        h.u64((s.aa >> 64) as u64);
                        h.u64(s.bb as u64);
                        h.u64((s.bb >> 64) as u64);
                    }
                    None => h.u64(u64::MAX),
                }
            }
        }
        for what in 0..2 {
            let r = if what == 0 {
                ev.median_score_trials(plans, &trials)
            } else {
                ev.interquartile_score_trials(plans, &trials)
            }
            .expect("order-statistic batch");
            for row in &r {
                for v in row {
                    h.f64(v.unwrap_or(f64::NAN));
                }
            }
        }
        let mx = ev.max_score_trials(plans, &trials).expect("max batch");
        for row in &mx {
            for v in row {
                match v {
                    Some(m) => {
                        h.f64(m.value);
                        h.u64(m.row as u64);
                    }
                    None => h.u64(u64::MAX),
                }
            }
        }
        h.0
    }

    fn plans_of(trees: &[Node]) -> Vec<crate::fit::GpuFitPlan> {
        trees.iter().map(|t| gpu_fit_plan_reason(t).expect("an eligible plan")).collect()
    }

    /// Value captured from the pre-change per-call allocating path.
    const GOLDEN_WORKLOAD: u64 = 0xdcc9_bb82_1db2_9ad8;

    #[test]
    fn persistent_buffers_reproduce_the_allocating_path_bit_for_bit() {
        let f = fixture(N);
        let Some(mut ev) = evaluator(&f) else { return };
        let owned = plans_of(&trees());
        let plans: Vec<&crate::fit::GpuFitPlan> = owned.iter().collect();
        let a = workload(&mut ev, &plans);
        eprintln!("PERSISTENT_WORKLOAD_HASH={a:#018x}");
        let b = workload(&mut ev, &plans);
        assert_eq!(a, b, "repeat on the same context differs");
        assert_eq!(a, GOLDEN_WORKLOAD, "differs from the pre-change path");
    }

    /// After one warm-up batch of each kind, 100 consecutive fit batches and
    /// 100 batches each of the rank, median and interquartile stages make no
    /// allocator call, grow nothing and leave both regions at their size.
    #[test]
    fn steady_state_fit_and_score_batches_make_no_allocator_calls() {
        let f = fixture(N);
        let Some(mut ev) = evaluator(&f) else { return };
        let owned = plans_of(&trees());
        let plans: Vec<&crate::fit::GpuFitPlan> = owned.iter().collect();
        let trials = trials_for(&plans, 4);
        let fit = |ev: &mut GpuEvaluator| {
            let (out, ms) = ev.fit_batch(&plans, 4, DeviceLoss::Squared);
            assert!(ms >= 0.0 && out.len() == plans.len(), "fit batch ran");
        };
        // Warm-up: the first batch of each kind sizes the regions.
        fit(&mut ev);
        ev.rank_score_trials(&plans, &trials).expect("rank warm-up");
        ev.median_score_trials(&plans, &trials).expect("median warm-up");
        ev.interquartile_score_trials(&plans, &trials).expect("iqr warm-up");
        ev.max_score_trials(&plans, &trials).expect("max warm-up");
        let warm = ev.arena_report();
        assert!(warm.fit_growths > 0 && warm.fit_alloc_calls > 0, "{warm:?}");
        assert!(warm.fit_device_bytes > 0 && warm.fit_pinned_bytes > 0, "{warm:?}");

        for _ in 0..100 {
            fit(&mut ev);
        }
        let after_fit = ev.arena_report();
        assert_eq!(after_fit.fit_steady_alloc_calls, 0, "{after_fit:?}");
        assert_eq!(after_fit.fit_alloc_calls, warm.fit_alloc_calls, "{after_fit:?}");
        assert_eq!(after_fit.fit_batches - warm.fit_batches, 100);

        for _ in 0..100 {
            ev.rank_score_trials(&plans, &trials).expect("rank batch");
        }
        let after_rank = ev.arena_report();
        assert_eq!(after_rank.fit_steady_alloc_calls, 0, "{after_rank:?}");
        assert_eq!(after_rank.fit_alloc_calls, warm.fit_alloc_calls, "{after_rank:?}");
        assert_eq!(after_rank.fit_batches - after_fit.fit_batches, 100);

        for what in 0..2 {
            for _ in 0..100 {
                if what == 0 {
                    ev.median_score_trials(&plans, &trials).expect("median batch");
                } else {
                    ev.interquartile_score_trials(&plans, &trials).expect("iqr batch");
                }
            }
        }
        for _ in 0..100 {
            ev.max_score_trials(&plans, &trials).expect("max batch");
        }
        let end = ev.arena_report();
        assert_eq!(end.fit_steady_alloc_calls, 0, "{end:?}");
        assert_eq!(end.fit_alloc_calls, warm.fit_alloc_calls, "{end:?}");
        assert_eq!(end.fit_growths, warm.fit_growths, "{end:?}");
        assert_eq!(end.fit_device_bytes, warm.fit_device_bytes, "{end:?}");
        assert_eq!(end.fit_pinned_bytes, warm.fit_pinned_bytes, "{end:?}");

        // A smaller batch stays inside the high-water mark.
        let small: Vec<&crate::fit::GpuFitPlan> = plans[..1].to_vec();
        let (out, ms) = ev.fit_batch(&small, 4, DeviceLoss::Squared);
        assert!(ms >= 0.0 && out.len() == 1);
        let small_trials = trials_for(&small, 2);
        ev.rank_score_trials(&small, &small_trials).expect("small rank batch");
        let shrunk = ev.arena_report();
        assert_eq!(shrunk.fit_alloc_calls, warm.fit_alloc_calls, "{shrunk:?}");
        assert_eq!(shrunk.fit_steady_alloc_calls, 0, "{shrunk:?}");
    }

    /// A batch past the high-water mark grows the regions exactly once, the
    /// growth is counted as a steady-state allocator call, and the next
    /// batch of that size is free again.
    #[test]
    fn a_larger_batch_grows_the_regions_once_and_is_counted() {
        let f = fixture(N);
        let Some(mut ev) = evaluator(&f) else { return };
        let owned = plans_of(&trees());
        let one: Vec<&crate::fit::GpuFitPlan> = owned[..1].iter().collect();
        let all: Vec<&crate::fit::GpuFitPlan> = owned.iter().collect();
        let (_, ms) = ev.fit_batch(&one, 4, DeviceLoss::Squared);
        assert!(ms >= 0.0);
        let before = ev.arena_report();
        assert_eq!(before.fit_steady_alloc_calls, 0, "{before:?}");
        let many: Vec<&crate::fit::GpuFitPlan> = (0..400).map(|k| all[k % all.len()]).collect();
        let (_, ms) = ev.fit_batch(&many, 4, DeviceLoss::Squared);
        assert!(ms >= 0.0);
        let grown = ev.arena_report();
        assert!(grown.fit_growths > before.fit_growths, "{grown:?}");
        assert!(grown.fit_steady_alloc_calls > 0, "{grown:?}");
        let (_, ms) = ev.fit_batch(&many, 4, DeviceLoss::Squared);
        assert!(ms >= 0.0);
        let again = ev.arena_report();
        assert_eq!(again.fit_alloc_calls, grown.fit_alloc_calls, "{again:?}");
        assert_eq!(again.fit_growths, grown.fit_growths, "{again:?}");
    }
}

/// #289 §6.2, moved from `tests/classification_gpu.rs` when `predictions`
/// became test-only (#293).
#[cfg(test)]
mod classification_parity_tests {
    use super::*;
    use crate::data::Sample;
    use crate::expr::{self, constant, input, node, Op};
    use crate::metrics::{self, Metric, ScoreObjective};

    fn binary_sample(n: usize, positives: f64, noise: f64, seed: u64) -> Sample {
        let mut x = Vec::with_capacity(n);
        let mut score = Vec::with_capacity(n);
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for i in 0..n {
            let t0 = i as f64 / (n.max(2) - 1) as f64 * 6.0 - 3.0;
            let t1 = next() * 2.0 - 1.0;
            let eps = next() * 2.0 - 1.0;
            x.push(vec![t0, t1]);
            score.push(0.9 * t0 + 0.5 * t1 + noise * 6.0 * eps);
        }
        let k = ((n as f64 * positives).round() as usize).clamp(1, n.max(1));
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut y = vec![0.0; n];
        for &i in order.iter().take(k) {
            y[i] = 1.0;
        }
        Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        }
    }

    fn affine(a: f64, b: f64) -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
        )
    }

    fn cpu_score(tree: &Node, s: &Sample, m: Metric) -> f64 {
        let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(tree, r)).collect();
        metrics::score_weighted(&pred, &s.y, &s.w, m, expr::count_constants(tree))
    }

    fn evaluator(s: &Sample) -> GpuEvaluator {
        let n = s.len();
        let v = s.x.first().map(|r| r.len()).unwrap_or(0);
        let mut cols = vec![0.0f32; n * v];
        for (r, row) in s.x.iter().enumerate() {
            for (k, &x) in row.iter().enumerate().take(v) {
                cols[k * n + r] = x as f32;
            }
        }
        let y32: Vec<f32> = s.y.iter().map(|&y| y as f32).collect();
        let mut ev = GpuEvaluator::new(&cols, n, v, &y32).expect("device evaluator");
        assert!(ev.set_fit_sign(&crate::fit::classification_signs_weighted(&s.y, &s.w)));
        if !s.w.is_empty() {
            assert!(ev.set_score_weights(&s.w));
        }
        ev
    }

    /// #289 §6.2: device vs CPU parity on identical f32 rows for every
    /// objective, with the search's weighted denominator.
    #[test]
    fn device_matches_cpu_for_every_classification_objective() {
        if !probe().available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(2_000, 0.35, 0.25, 21);
        let trees = [
            affine(0.7, -0.2),
            affine(-1.3, 0.4),
            affine(0.0, 0.0),
            node(Op::Tanh, vec![affine(1.1, 0.2)]),
        ];
        for objective in [
            ScoreObjective::LogLoss,
            ScoreObjective::Brier,
            ScoreObjective::HingeSigmoid,
            ScoreObjective::HingeOrdinary,
            ScoreObjective::HingeSquared,
        ] {
            let mut ev = evaluator(&s);
            let pop = Population::from_trees(trees.iter());
            assert_eq!(pop.len(), trees.len());
            let batch = ev.score_population(&pop, objective, &vec![0i32; pop.len()]);
            assert!(batch.ms >= 0.0, "kernel launch failed");
            assert!(batch.all_ok(), "{:?}", batch.status);
            let preds = ev
                .predictions(pop.programs().0, pop.programs().1)
                .expect("raw predictions");
            for (i, tree) in trees.iter().enumerate() {
                let got = batch.values[i];
                let rows: Vec<f64> = preds[i * s.len()..(i + 1) * s.len()]
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                let want = metrics::score_weighted(&rows, &s.y, &s.w, objective.metric(), 0);
                let err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    err <= 1e-6,
                    "{} model {i}: device {got} vs CPU {want} (rel {err:.2e})",
                    objective.name()
                );
                // f32 rows versus the f64 model: the wider bound.
                let f64want = cpu_score(tree, &s, objective.metric());
                let f32err = (got - f64want).abs() / f64want.abs().max(1.0);
                assert!(
                    f32err <= 1e-5,
                    "{} model {i}: f32-row error {f32err:.2e} exceeds 1e-5",
                    objective.name()
                );
            }
        }
    }

    /// #422: a finite prediction above 1e30 (up to the f32 maximum) is scored
    /// like the CPU scores the same rows, for every classification objective;
    /// an f32 overflow to `inf` is `+inf`. Repeats are bit-identical.
    #[test]
    fn large_finite_predictions_score_like_the_cpu() {
        if !probe().available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(512, 0.35, 0.25, 22);
        let trees = [
            affine(3e30, 0.0),
            affine(-3e30, 1e30),
            affine(1e35, 0.0),
            affine(3e38, 0.0),
            node(Op::Add, vec![affine(3e38, 0.0), affine(3e38, 0.0)]),
        ];
        for objective in [
            ScoreObjective::LogLoss,
            ScoreObjective::Brier,
            ScoreObjective::HingeSigmoid,
            ScoreObjective::HingeOrdinary,
            ScoreObjective::HingeSquared,
        ] {
            let mut ev = evaluator(&s);
            let pop = Population::from_trees(trees.iter());
            let pc = vec![0i32; pop.len()];
            let batch = ev.score_population(&pop, objective, &pc);
            let again = ev.score_population(&pop, objective, &pc);
            assert!(batch.ms >= 0.0, "kernel launch failed");
            let preds = ev
                .predictions(pop.programs().0, pop.programs().1)
                .expect("raw predictions");
            for k in 0..trees.len() {
                let rows: Vec<f64> = preds[k * s.len()..(k + 1) * s.len()]
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                let want = metrics::score_weighted(&rows, &s.y, &s.w, objective.metric(), 0);
                let got = batch.values[k];
                let what = format!("{} model {k}", objective.name());
                assert_eq!(got.to_bits(), again.values[k].to_bits(), "{what}: repeat");
                if want.is_finite() {
                    assert_eq!(batch.status[k], ScoreStatus::Ok, "{what}");
                    let err = (got - want).abs() / want.abs().max(1.0);
                    assert!(err <= 1e-6, "{what}: device {got} vs CPU {want}");
                } else {
                    assert_eq!(batch.status[k], ScoreStatus::NonFinite, "{what}");
                    assert_eq!(got, f64::INFINITY, "{what}");
                }
            }
        }
    }
}

/// #293: the P x N guard, the allocation ledger and the F1 wave split.
#[cfg(test)]
mod alloc_ledger_tests {
    use super::*;
    use crate::expr::{constant, input, node};
    use crate::fit::{gpu_fit_plan_reason, OrderStat};

    const MIB: u64 = 1024 * 1024;

    /// One variable with distinct values, no ties, plus an exact `f64` copy.
    fn data(n: usize) -> (Vec<f32>, Vec<f64>, Vec<f32>, Vec<f64>) {
        let mut x = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for i in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            x[i] = -2.0 + 4.0 * i as f64 / (n - 1) as f64 + 1e-3 * noise;
            y[i] = 0.7 * x[i] + 0.1 + 0.4 * noise;
        }
        let x32 = x.iter().map(|&v| v as f32).collect();
        let y32 = y.iter().map(|&v| v as f32).collect();
        (x32, x, y32, y)
    }

    fn evaluator(n: usize) -> Option<GpuEvaluator> {
        let (x32, x, y32, y) = data(n);
        let mut ev = GpuEvaluator::new(&x32, n, 1, &y32)?;
        assert!(ev.set_fit_data(&x, &y, &[]), "f64 fit data upload");
        Some(ev)
    }

    fn linear() -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.15)],
        )
    }

    fn trial_lists(nsets: usize, ntrials: usize) -> Vec<Vec<Vec<f64>>> {
        (0..nsets)
            .map(|k| {
                (0..ntrials)
                    .map(|t| vec![0.3 + 0.17 * t as f64 + 0.05 * k as f64, -0.2 + 0.11 * t as f64])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn wave_shape_keeps_every_call_inside_the_cap() {
        let cap = 512 * MIB;
        for row_bytes in [RANK_ROW_BYTES, ORDER_ROW_BYTES] {
        for n in [1_000usize, 100_000, 1_398_102, 1_500_000, 3_580_000, 4_775_757, 8_000_000] {
            for ntrials in [1usize, 2, 6, 7, 20] {
                let (sets, chunk) = wave_shape(n, ntrials, cap, row_bytes);
                assert!(sets >= 1 && chunk >= 1 && chunk <= ntrials);
                let per_seg = n as u64 * row_bytes + MIB;
                if per_seg <= cap {
                    assert!(
                        (sets * chunk) as u64 * per_seg <= cap,
                        "n {n} trials {ntrials}: {sets} sets x {chunk} trials exceed the cap"
                    );
                } else {
                    assert_eq!((sets, chunk), (1, 1), "a single segment cannot be split");
                }
                if chunk < ntrials {
                    assert_eq!(sets, 1, "a cut trial list is run one set at a time");
                }
            }
        }
        }
        // The minimum batch the issue names: 6 trials x 3.58M rows x 64 B.
        assert!(6 * 3_580_000u64 * 64 > cap);
        assert_eq!(wave_shape(3_580_000, 6, cap, ORDER_ROW_BYTES), (1, 2));
        assert_eq!(wave_shape(3_580_000, 6, cap, RANK_ROW_BYTES), (1, 2));
    }

    #[test]
    fn rank_trial_sub_waves_equal_the_unsplit_scores() {
        let Some(mut ev) = evaluator(3_000) else {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        };
        let t = linear();
        let p = gpu_fit_plan_reason(&t).expect("an eligible plan");
        let plans = [&p, &p];
        let trials = trial_lists(2, 6);
        let unsplit = ev
            .rank_score_trials_capped(&plans, &trials, u64::MAX)
            .expect("unsplit rank call");
        let per_seg = 3_000u64 * RANK_ROW_BYTES + MIB;
        for segs in [1u64, 2, 4, 5] {
            let cap = per_seg * segs;
            assert_eq!(wave_shape(3_000, 6, cap, RANK_ROW_BYTES).1, segs as usize);
            let split = ev
                .rank_score_trials_capped(&plans, &trials, cap)
                .expect("split rank call");
            assert_eq!(split, unsplit, "cap of {segs} segments changed a rank sum");
        }
    }

    #[test]
    fn median_trial_sub_waves_equal_the_unsplit_scores() {
        let Some(mut ev) = evaluator(3_001) else {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        };
        let t = linear();
        let p = gpu_fit_plan_reason(&t).expect("an eligible plan");
        let plans = [&p, &p];
        let trials = trial_lists(2, 6);
        let unsplit = ev
            .order_stat_score_trials_capped(&plans, &trials, OrderStat::Median, u64::MAX)
            .expect("unsplit median call");
        let per_seg = 3_001u64 * ORDER_ROW_BYTES + MIB;
        for segs in [1u64, 2, 4, 5] {
            assert_eq!(wave_shape(3_001, 6, per_seg * segs, ORDER_ROW_BYTES).1, segs as usize);
            let split = ev
                .order_stat_score_trials_capped(&plans, &trials, OrderStat::Median, per_seg * segs)
                .expect("split median call");
            for (a, b) in split.iter().flatten().zip(unsplit.iter().flatten()) {
                assert_eq!(
                    a.map(f64::to_bits),
                    b.map(f64::to_bits),
                    "cap of {segs} segments changed a median"
                );
            }
        }
    }

    /// Six trials of 1.5M rows need 6 x n x 64 B above the 512 MiB cap. The
    /// call must run in sub-waves and no single device request may pass it.
    #[test]
    fn rank_and_median_batches_above_the_cap_run_in_sub_waves() {
        let n = 1_500_000usize;
        assert!(6 * n as u64 * 64 > 512 * MIB);
        let Some(mut ev) = evaluator(n) else {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        };
        let t = linear();
        let p = gpu_fit_plan_reason(&t).expect("an eligible plan");
        let trials = trial_lists(1, 6);
        let before = ev.alloc_report();
        let sums = ev.rank_score_trials(&[&p], &trials).expect("rank call");
        assert_eq!(sums[0].len(), 6);
        assert!(sums[0].iter().all(|s| s.is_some()));
        let after_rank = ev.alloc_report();
        let med = ev
            .order_stat_score_trials(&[&p], &trials, OrderStat::Median)
            .expect("median call");
        assert!(med[0].iter().all(|v| v.is_some()));
        let after = ev.alloc_report();
        eprintln!(
            "max request {} B ({}), before {} B, after rank {} B",
            after.max_request_bytes, after.max_request_tag, before.max_request_bytes,
            after_rank.max_request_bytes
        );
        assert!(
            after_rank.max_request_bytes <= 512 * MIB,
            "rank request of {} B ({}) is above the cap",
            after_rank.max_request_bytes,
            after_rank.max_request_tag
        );
        assert!(
            after.max_request_bytes <= 512 * MIB,
            "median request of {} B ({}) is above the cap",
            after.max_request_bytes,
            after.max_request_tag
        );
        assert_eq!(after.refused, 0);
    }

    /// `rc_predict` refuses a P x N output above 64 MiB: C cannot see the
    /// `cfg(test)` gate on its caller.
    #[test]
    fn rc_predict_refuses_a_p_by_n_output_above_the_cap() {
        let n = 200_000usize;
        let (x32, _, y32, _) = data(n);
        let Some(mut ev) = GpuEvaluator::new(&x32, n, 1, &y32) else {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        };
        let trees: Vec<Node> = (0..100).map(|_| input(0)).collect();
        let pop = Population::from_trees(trees.iter());
        assert_eq!(pop.len(), 100);
        // 100 x 200,000 x 4 B = 80 MB.
        assert!(ev.predictions(pop.programs().0, pop.programs().1).is_none());
        let small: Vec<Node> = (0..2).map(|_| input(0)).collect();
        let pop = Population::from_trees(small.iter());
        let rows = ev
            .predictions(pop.programs().0, pop.programs().1)
            .expect("a small P x N output is allowed");
        assert_eq!(rows.len(), 2 * n);
    }

    /// `predictions` allocates P x N, so it exists only in test builds.
    #[test]
    fn predictions_is_compiled_only_for_tests() {
        let src = include_str!("gpu.rs");
        let at = src
            .find("pub(crate) fn predictions(")
            .expect("predictions is defined");
        let attr = src[..at].lines().rev().find(|l| !l.trim().is_empty()).unwrap();
        assert_eq!(attr.trim(), "#[cfg(test)]");
        let public = ["pub fn", " predictions("].concat();
        assert!(!src.contains(&public));
    }

    /// A budget of the resident bytes plus 1 MiB lets the context exist and
    /// refuses the f64 twin, naming the allocation.
    #[test]
    fn a_budget_refuses_the_f64_twin_with_its_tag() {
        let n = 200_000usize;
        let (x32, x, y32, y) = data(n);
        let resident = (n * 4 + n * 4) as u64;
        let Some(mut ev) = GpuEvaluator::with_budget(&x32, n, 1, &y32, Some(resident + MIB)) else {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        };
        assert!(!ev.set_fit_data(&x, &y, &[]));
        let report = ev.alloc_report();
        assert_eq!(report.refused_tag, "cols_f64");
        assert_eq!(report.refused, 1);
        assert_eq!(report.live_bytes, resident);
        assert_eq!(last_alloc_failure(), Some((n * 8) as u64));
        assert_eq!(last_alloc_failure_tag().as_deref(), Some("cols_f64"));
    }
}

/// #293: row-range scoring with the lane carry.
#[cfg(test)]
mod carry_tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    const KINDS: [ScoreObjective; 15] = [
        ScoreObjective::Squared,
        ScoreObjective::Absolute,
        ScoreObjective::LogSquashed,
        ScoreObjective::Signed,
        ScoreObjective::Maximum,
        ScoreObjective::Correlation,
        ScoreObjective::CorrelationSigned,
        ScoreObjective::Aic,
        ScoreObjective::Aicc,
        ScoreObjective::Bic,
        ScoreObjective::LogLoss,
        ScoreObjective::Brier,
        ScoreObjective::HingeSigmoid,
        ScoreObjective::HingeOrdinary,
        ScoreObjective::HingeSquared,
    ];

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn unit(&mut self) -> f64 {
            self.next() as f64 / (1u64 << 53) as f64
        }
    }

    struct Data {
        n: usize,
        cols: Vec<f32>,
        y: Vec<f32>,
        sign: Vec<f64>,
    }

    /// Rows 10, 12 and 266 (or the last row) carry `|y| = 7`: with the constant
    /// zero candidate the maximum ties inside one lane and across lanes.
    fn data(n: usize) -> Data {
        let mut rng = Lcg(0x5eed_0293 ^ n as u64);
        let mut cols = vec![0.0f32; 2 * n];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let a = rng.unit() * 4.0 - 2.0;
            let b = rng.unit() * 2.0 - 1.0;
            cols[i] = a as f32;
            cols[n + i] = b as f32;
            y[i] = (0.7 * a + 0.3 * (2.0 * b).sin() + 0.2 * (rng.unit() - 0.5)) as f32;
        }
        for (k, &row) in [10usize, 12, 266, n - 1].iter().enumerate() {
            if row < n {
                y[row] = if k % 2 == 0 { 7.0 } else { -7.0 };
            }
        }
        let sign = y.iter().map(|&v| if v > 0.1 { 1.0 } else { -1.0 }).collect();
        Data { n, cols, y, sign }
    }

    fn trees() -> Vec<Node> {
        let affine = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.15)],
        );
        let sine = node(
            Op::Add,
            vec![
                node(
                    Op::Sin,
                    vec![node(Op::Add, vec![node(Op::Mul, vec![constant(1.1), input(0)]), constant(0.1)])],
                ),
                node(Op::Mul, vec![constant(0.2), input(1)]),
            ],
        );
        vec![
            affine,
            sine,
            constant(0.0),
            constant(0.5),
            constant(f64::NAN),
            node(Op::Mul, vec![input(0), input(1)]),
            node(Op::Tanh, vec![node(Op::Mul, vec![constant(2.0), input(1)])]),
            node(Op::Sub, vec![node(Op::Square, vec![input(0)]), input(1)]),
        ]
    }

    fn evaluator(d: &Data, weights: &[f64]) -> Option<GpuEvaluator> {
        if !probe().available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return None;
        }
        let mut ev = GpuEvaluator::new(&d.cols, d.n, 2, &d.y)?;
        assert!(ev.set_fit_sign(&d.sign));
        if !weights.is_empty() {
            assert!(ev.set_score_weights(weights));
        }
        Some(ev)
    }

    fn uniform(n: usize, step: usize) -> Vec<RowRange> {
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let len = step.min(n - start);
            out.push(RowRange { start, len });
            start += len;
        }
        out
    }

    /// 30 seeded cuts, unaligned, with adjacent cuts so 1-row ranges occur.
    fn seeded(n: usize, seed: u64) -> Vec<RowRange> {
        let mut rng = Lcg(seed);
        let mut cuts = vec![1usize, 2, n - 1];
        while cuts.len() < 30 {
            let c = 1 + (rng.next() as usize) % (n - 1);
            cuts.push(c);
            if cuts.len() < 30 {
                cuts.push((c + 1).min(n - 1));
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        let mut out = Vec::new();
        let mut start = 0;
        for c in cuts {
            if c > start {
                out.push(RowRange { start, len: c - start });
                start = c;
            }
        }
        out.push(RowRange { start, len: n - start });
        out
    }

    fn partitions(n: usize) -> Vec<(String, Vec<RowRange>)> {
        let mut parts = vec![("whole".to_string(), vec![RowRange { start: 0, len: n }])];
        for step in [512usize, 1024, 8192] {
            parts.push((format!("uniform {step}"), uniform(n, step)));
        }
        for seed in [11u64, 29] {
            let p = seeded(n, seed);
            assert!(p.iter().any(|r| r.len == 1), "the seeded cuts include a 1-row range");
            parts.push((format!("seeded {seed}"), p));
        }
        parts
    }

    fn assert_same(label: &str, got: &ScoreBatch, want: &ScoreBatch) {
        assert!(got.ms >= 0.0, "{label}: launch failed");
        assert_eq!(got.values.len(), want.values.len(), "{label}");
        for k in 0..want.values.len() {
            assert_eq!(got.status[k], want.status[k], "{label} slot {k} status");
            assert_eq!(
                got.values[k].to_bits(),
                want.values[k].to_bits(),
                "{label} slot {k}: {} vs {}",
                got.values[k],
                want.values[k]
            );
        }
    }

    fn check_case(d: &Data, weights: &[f64], case: &str) {
        let Some(mut ev) = evaluator(d, weights) else { return };
        let trees = trees();
        let pop = Population::from_trees(trees.iter());
        assert_eq!(pop.len(), trees.len());
        let (prog, offsets) = pop.programs();
        let pcount = vec![2i32; pop.len()];
        let parts = partitions(d.n);
        for objective in KINDS {
            if objective == ScoreObjective::Maximum && !weights.is_empty() {
                let weighted = ev.score_batch(prog, offsets, objective, &pcount);
                assert!(weighted.ms < 0.0, "{case}: a weighted maximum launch is refused");
                continue;
            }
            let want = ev.score_batch(prog, offsets, objective, &pcount);
            assert!(want.ms >= 0.0, "{case} {}: resident launch", objective.name());
            assert_ne!(want.status[4], ScoreStatus::Ok, "{case} {}: the NaN candidate is unusable", objective.name());
            assert_eq!(want.values[4], f64::INFINITY);
            for (name, ranges) in &parts {
                let got = ev.score_rows(prog, offsets, objective, &pcount, ranges);
                assert_same(&format!("{case} n={} {} {name}", d.n, objective.name()), &got, &want);
            }
        }
    }

    #[test]
    fn row_ranges_match_the_resident_kernel_bitwise() {
        for n in [100usize, 70_001] {
            let d = data(n);
            check_case(&d, &[], "unweighted");
            let zero: Vec<f64> = (0..n)
                .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 + (i % 4) as f64 * 0.25 })
                .collect();
            check_case(&d, &zero, "zero weights");
            let mut negative = vec![1.0f64; n];
            negative[5] = -1.0;
            check_case(&d, &negative, "negative weight");
            check_case(&d, &vec![0.0f64; n], "all-zero weights");
        }
    }

    #[test]
    fn waves_do_not_change_scores() {
        let d = data(70_001);
        let Some(mut ev) = evaluator(&d, &[]) else { return };
        let base = trees();
        let all: Vec<Node> = (0..30).map(|i| base[i % base.len()].clone()).collect();
        let ranges = uniform(d.n, 8192);
        for objective in [ScoreObjective::Squared, ScoreObjective::Correlation, ScoreObjective::Maximum] {
            let score = |ev: &mut GpuEvaluator, trees: &[Node]| {
                let pop = Population::from_trees(trees.iter());
                let (prog, offsets) = pop.programs();
                let pcount = vec![2i32; pop.len()];
                ev.score_rows(prog, offsets, objective, &pcount, &ranges)
            };
            let whole = score(&mut ev, &all);
            let (a, b) = all.split_at(11);
            let mut parts = score(&mut ev, a);
            let tail = score(&mut ev, b);
            parts.values.extend(tail.values);
            parts.status.extend(tail.status);
            assert_same(objective.name(), &parts, &whole);
            let pop = Population::from_trees(all.iter());
            let (prog, offsets) = pop.programs();
            let resident = ev.score_batch(prog, offsets, objective, &vec![2i32; all.len()]);
            assert_same(objective.name(), &whole, &resident);
        }
    }

    #[test]
    fn carry_region_is_steady() {
        let d = data(70_001);
        let Some(mut ev) = evaluator(&d, &[]) else { return };
        let base = trees();
        let many: Vec<Node> = (0..512)
            .map(|i| {
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(0.001 * (i + 1) as f64), input(i % 2)]),
                        base[i % base.len()].clone(),
                    ],
                )
            })
            .collect();
        let pop = Population::from_trees(many.iter());
        let (prog, offsets) = pop.programs();
        let pcount = vec![2i32; pop.len()];
        let ranges = uniform(d.n, 8192);
        let first = ev.score_rows(prog, offsets, ScoreObjective::Squared, &pcount, &ranges);
        assert!(first.ms >= 0.0);
        let cold = ev.alloc_report();
        let cold_arena = ev.arena_report();
        assert_eq!(cold.max_request_tag, "carry");
        assert_eq!(cold.max_request_bytes, 512 * (16 * 256 + 16));
        for _ in 0..100 {
            let again = ev.score_rows(prog, offsets, ScoreObjective::Squared, &pcount, &ranges);
            assert_same("steady", &again, &first);
        }
        let steady = ev.alloc_report();
        let arena = ev.arena_report();
        assert_eq!(arena.steady_alloc_calls, 0, "{arena:?}");
        assert_eq!(arena.alloc_calls, cold_arena.alloc_calls, "{arena:?}");
        assert_eq!(steady.requests, cold.requests, "{steady:?}");
        assert_eq!(steady.live_bytes, cold.live_bytes);
        assert_eq!(steady.refused, 0);
    }
}

#[cfg(test)]
#[path = "gpu_order_select_tests.rs"]
mod order_select_tests;

#[cfg(test)]
#[path = "gpu_hinge_fit_tests.rs"]
mod hinge_fit_tests;
