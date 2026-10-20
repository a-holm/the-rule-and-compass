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

use crate::expr::{Node, Op};

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

/// Compile one expression to postfix bytecode. Returns `false` if the tree uses
/// an operator the kernel does not implement (for example `asinh`, `mod` on a
/// platform without the #131 opcodes, or constraints).
pub fn compile(n: &Node, out: &mut Vec<u32>) -> bool {
    if !emit(n, out) {
        return false;
    }
    out.push(OP_HALT);
    true
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
    fn rc_device_name() -> *const std::os::raw::c_char;
    fn rc_unified() -> i32;
}

/// A compiled population: the postfix programs of many expressions laid out
/// back to back, plus the offsets table that marks where each one starts.
///
/// This is the batch unit the kernel wants: one [`GpuEvaluator::eval_population`]
/// call transfers the whole population and launches one kernel with one block
/// per program, returning one SSE per program. It is deliberately *not* wired
/// into the search loop yet; the search owns its own scheduling.
#[derive(Debug, Default, Clone)]
pub struct Population {
    prog: Vec<u32>,
    offsets: Vec<u32>,
    unsupported: usize,
    /// #131: operator occurrences, used for the per-generation `--json` report.
    ops: std::collections::BTreeMap<u32, usize>,
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
        let mut buf = Vec::new();
        if !compile(tree, &mut buf) {
            self.unsupported += 1;
            return false;
        }
        // Count operator words, skipping the operand word after const/var.
        let mut i = 0;
        while i < buf.len() {
            let w = buf[i];
            if w == OP_CONST || w == OP_VAR {
                i += 2;
            } else {
                if w != OP_HALT {
                    *self.ops.entry(w).or_default() += 1;
                }
                i += 1;
            }
        }
        self.prog.extend_from_slice(&buf);
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
            .map(|(&op, &n)| (opcode_name(op).to_string(), n))
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
}

impl GpuEvaluator {
    /// Upload a dataset. `cols` is column-major: `cols[var * n_rows + row]`.
    pub fn new(cols: &[f32], n_rows: usize, n_vars: usize, y: &[f32]) -> Option<Self> {
        let ctx = unsafe { rc_create(cols.as_ptr(), n_rows as i32, n_vars as i32, y.as_ptr()) };
        if ctx.is_null() {
            None
        } else {
            Some(GpuEvaluator { ctx })
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
        // #131: `if` is native now, but `asinh` still has no kernel opcode.
        assert!(pop.add(&node(
            Op::If,
            vec![input(0), constant(1.0), constant(2.0)]
        )));
        assert!(!pop.add(&node(Op::Asinh, vec![input(0)])));
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
            f64::INFINITY, 3e38, -3e38,
        ];
        let b = [
            2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, -2.5, -2.5, 7.0, 7.0, 1.0, 1.0, 1.0,
            1.0,
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

