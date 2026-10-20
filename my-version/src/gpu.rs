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
const OP_HALT: u32 = 255;

/// Compile one expression to postfix bytecode. Returns `false` if the tree uses
/// an operator the kernel does not implement even after [`lower`] (for example
/// `floor`, `mod` or `atan2`).
pub fn compile(n: &Node, out: &mut Vec<u32>) -> bool {
    if !emit(n, out) {
        return false;
    }
    out.push(OP_HALT);
    true
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
    let abs_diff = |a: &Node, b: &Node| node(Abs, vec![node(Sub, vec![a.clone(), b.clone()])]);
    match n.op {
        Min | Max => {
            // #121: gated tie form. `(a+b-|a-b|)/2` overflows in f32
            // (min(1e8, 1) -> 0), so never add the operands.
            let a = k[0].clone(); let b = k[1].clone();
            let u = node(Step, vec![node(Sub, vec![b.clone(), a.clone()])]);
            let v = node(Step, vec![node(Sub, vec![a.clone(), b.clone()])]);
            let w = node(Sub, vec![node(Sub, vec![constant(1.0), u.clone()]), v.clone()]);
            if matches!(n.op, Min) {
                node(Add, vec![
                    node(Add, vec![node(Mul, vec![u, a.clone()]), node(Mul, vec![v, b])]),
                    node(Mul, vec![w, a]),
                ])
            } else {
                node(Add, vec![
                    node(Add, vec![node(Mul, vec![u, b]), node(Mul, vec![v, a.clone()])]),
                    node(Mul, vec![w, a]),
                ])
            }
        }
        If => {
            // #121: symmetric gate; the compact b + step(c)(a-b) is not exact
            // for extreme arms. Both arms are evaluated, so this is only used
            // when both are finite (the compiler's totality check is follow-up).
            let c = node(Step, vec![k[0].clone()]);
            let nc = node(Sub, vec![constant(1.0), c.clone()]);
            node(Add, vec![node(Mul, vec![c, k[1].clone()]), node(Mul, vec![nc, k[2].clone()])])
        }
        Sign => node(Sub, vec![node(Step, vec![k[0].clone()]), node(Step, vec![node(Neg, vec![k[0].clone()])])]),
        Less => node(Step, vec![node(Sub, vec![k[1].clone(), k[0].clone()])]),
        LessOrEqual => node(Sub, vec![constant(1.0), node(Step, vec![node(Sub, vec![k[0].clone(), k[1].clone()])])]),
        Greater => node(Step, vec![node(Sub, vec![k[0].clone(), k[1].clone()])]),
        GreaterOrEqual => node(Sub, vec![constant(1.0), node(Step, vec![node(Sub, vec![k[1].clone(), k[0].clone()])])]),
        Equal => {
            let d = abs_diff(&k[0], &k[1]);
            node(Sub, vec![
                node(Sub, vec![constant(1.0), node(Step, vec![d.clone()])]),
                node(Step, vec![node(Neg, vec![d])]),
            ])
        }
        Not => node(Sub, vec![constant(1.0), node(Step, vec![k[0].clone()])]),
        And => node(Mul, vec![node(Step, vec![k[0].clone()]), node(Step, vec![k[1].clone()])]),
        Or => node(Step, vec![node(Add, vec![k[0].clone(), k[1].clone()])]),
        Xor => node(Abs, vec![node(Sub, vec![node(Step, vec![k[0].clone()]), node(Step, vec![k[1].clone()])])]),
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
        | Square | Cube => {
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
                _ => unreachable!(),
            };
            emit(&n.kids[0], out) && {
                out.push(op);
                true
            }
        }
        Add | Sub | Mul | Div | Pow => {
            let op = match n.op {
                Add => OP_ADD,
                Sub => OP_SUB,
                Mul => OP_MUL,
                Div => OP_DIV,
                Pow => OP_POW,
                _ => unreachable!(),
            };
            emit(&n.kids[0], out) && emit(&n.kids[1], out) && {
                out.push(op);
                true
            }
        }
        If => false,
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
}

impl Population {
    /// An empty population. The offsets table always starts with a zero.
    pub fn new() -> Self {
        Population {
            prog: Vec::new(),
            offsets: vec![0],
            unsupported: 0,
        }
    }

    /// Compile and append one expression. Returns `false` (and counts the tree
    /// in [`Population::unsupported`]) when the expression uses an operator the
    /// kernel does not implement; the population itself stays valid.
    pub fn add(&mut self, tree: &Node) -> bool {
        let mut buf = Vec::new();
        let lowered = lower(tree);
        // #121: the kernel has no stack-bounds check (RC_MAX_STACK=64), so a
        // lowered program must be rejected before emission if it could overflow.
        if peak_stack(&lowered) > 56 {
            self.unsupported += 1;
            return false;
        }
        if !compile(&lowered, &mut buf) {
            self.unsupported += 1;
            return false;
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
        // `if` is lowered, but `floor` still has no kernel opcode.
        assert!(pop.add(&node(
            Op::If,
            vec![input(0), constant(1.0), constant(2.0)]
        )));
        assert!(!pop.add(&node(Op::Floor, vec![input(0)])));
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

    #[test]
    fn lowered_if_runs_on_the_gpu_and_matches_the_cpu() {
        let (cols, n, nv, y, _) = fixture();
        let mut ev = match evaluator_or_skip(&cols, n, nv, &y) {
            Some(ev) => ev,
            None => return,
        };
        let trees = vec![
            node(
                Op::If,
                vec![input(0), input(1), node(Op::Neg, vec![input(0)])],
            ),
            node(Op::Min, vec![input(0), input(1)]),
            node(Op::Max, vec![input(0), constant(0.25)]),
            node(Op::Sign, vec![input(1)]),
            node(Op::Greater, vec![input(0), input(1)]),
        ];
        let pop = Population::from_trees(&trees);
        assert_eq!(pop.len(), trees.len(), "every lowered tree must compile");
        assert_eq!(pop.unsupported(), 0);
        let (gpu, ms) = ev.eval_population(&pop);
        assert!(ms >= 0.0, "CUDA batch failed");
        for (i, tree) in trees.iter().enumerate() {
            let cpu = cpu_sse(tree, &cols, n, nv, &y);
            let rel = (gpu[i] as f64 - cpu).abs() / (1.0 + cpu.abs());
            assert!(
                rel < 1e-4,
                "lowered program {}: GPU SSE {} vs CPU SSE {} (relative {:.2e})",
                i,
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
