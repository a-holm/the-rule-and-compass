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
const OP_HALT: u32 = 255;

/// Compile one expression to postfix bytecode. Returns `false` if the tree uses
/// an operator the kernel does not implement (`if`, which is a branch and would
/// reintroduce divergence).
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
        if !compile(tree, &mut buf) {
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
        // `if` is a branch and has no kernel opcode.
        assert!(!pop.add(&node(
            Op::If,
            vec![input(0), constant(1.0), constant(2.0)]
        )));
        assert!(pop.add(&trees[1]));
        assert_eq!(pop.len(), 2);
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
}

