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
use crate::metrics::{self, Metric};

// #116: the solver portfolio's additional arms. Each child module uses the
// flattened tree, adjoint sweep and guarded QR from this module, so the arms
// share one arithmetic story and one fallback contract.
mod gpu;
mod stream;
mod varpro;

pub use gpu::{
    backend_description, fit_batch_verified, refit_finalists_exact, BatchStatus, BatchedFitter,
    VerifiedBatch,
};
pub use stream::{
    fit_stream, fit_stream_with, RowSource, SliceRows, StoreRows, StoreSpec, STREAM_BLOCK_ROWS,
};
pub use varpro::{cost_eligible as varpro_cost_eligible, VarProPartition};

/// `1/sqrt(pi)`, the constant in the error-function derivative.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// #116: which exact constant-fitting arm the search may use.
///
/// All arms minimise the same weighted squared error on the same rows in
/// `f64`; they differ only in *how* they reach the minimum. `Exact` is the
/// historical Levenberg–Marquardt path, bit for bit. `Auto` dispatches by
/// structure: the closed-form linear arm when the tree is affine in every
/// constant, variable projection when it is separable, and the historical LM
/// otherwise. `Linear`, `VarPro`, `Stream` and `Gpu` force one arm and fall
/// back to `Exact` whenever the arm cannot prove a safe result.
///
/// #169: `Gpu` is the opt-in batched GPU arm. During a CUDA-guided search the
/// offspring may be left unfitted for a whole-generation batch and fitted in
/// one block-per-candidate kernel; the integration point and exact verifier
/// live in the [`gpu`] module. Every unsupported, ill-conditioned or
/// non-finite candidate is refit on the CPU, and every finalist is refit
/// exactly in `f64` before reporting. When no batch context exists (a direct
/// [`fit_weighted_strategy`] call, a checkpoint resume, a non-CUDA build) the
/// arm behaves like `Auto`; it is never silently a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitStrategy {
    /// Today's CPU Levenberg–Marquardt.
    Exact,
    /// Structure-adaptive exact dispatch: linear, then VarPro, then LM.
    Auto,
    /// The linear arm with an `Exact` fallback when a tree is ineligible.
    Linear,
    /// Variable projection for separable models, `Exact` otherwise.
    VarPro,
    /// Out-of-core block LM with exact f64 sufficient-statistic merging.
    Stream,
    /// #169: batched GPU LM with a per-candidate CPU fallback.
    Gpu,
}

impl FitStrategy {
    /// Parse the CLI spelling. `cpu` is accepted as a synonym for `exact`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "exact" | "cpu" => Some(FitStrategy::Exact),
            "auto" => Some(FitStrategy::Auto),
            "linear" => Some(FitStrategy::Linear),
            "varpro" | "var_pro" => Some(FitStrategy::VarPro),
            "stream" | "out-of-core" | "outofcore" => Some(FitStrategy::Stream),
            "gpu" => Some(FitStrategy::Gpu),
            _ => None,
        }
    }

    /// The spelling used by the CLI, schema and checkpoints.
    pub fn name(self) -> &'static str {
        match self {
            FitStrategy::Exact => "exact",
            FitStrategy::Auto => "auto",
            FitStrategy::Linear => "linear",
            FitStrategy::VarPro => "varpro",
            FitStrategy::Stream => "stream",
            FitStrategy::Gpu => "gpu",
        }
    }

    /// #169: whether the search should leave offspring unfitted for the
    /// batched GPU arm instead of fitting each candidate on the CPU.
    pub fn defers_to_gpu(self) -> bool {
        matches!(self, FitStrategy::Gpu)
    }
}

/// #229: an objective a device kernel computes natively.
///
/// #236 closed the GPU evaluator to every non-squared metric because the
/// kernel only summed squared error: a `slope`/`absolute`/`rank` run was
/// silently ranked by MSE. A metric may use the device again only when this
/// enum names its objective *and* the kernel accumulates exactly that
/// objective. The discriminants are the `metric` argument of
/// `rc_eval`/`rc_pool_eval`/`rc_fit_batch` in `src/cuda/eval.cu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLoss {
    /// `Σ w·r²`.
    Squared,
    /// `Σ w·ln(1 + r²)`, minimised by IRLS with `2/(1 + r²)` row weights.
    LogSquashed,
}

impl DeviceLoss {
    /// The `metric` code the CUDA entry points take.
    pub fn code(self) -> i32 {
        match self {
            DeviceLoss::Squared => 0,
            DeviceLoss::LogSquashed => 1,
        }
    }

    /// The device objective for a metric, or `None` when the kernel cannot
    /// score it.
    pub fn for_metric(metric: Metric) -> Option<Self> {
        match metric {
            Metric::Squared => Some(DeviceLoss::Squared),
            Metric::LogSquashed => Some(DeviceLoss::LogSquashed),
            _ => None,
        }
    }
}

/// #229: the objective the **fit** kernel must minimise for a metric.
///
/// The information criteria fit by least squares and add their penalty when
/// scoring, so they share the squared objective; `log_squashed` uses the
/// robust arm. Everything else returns `None` and keeps the exact CPU fit.
pub fn fit_loss(metric: Metric) -> Option<DeviceLoss> {
    match metric {
        Metric::Aic | Metric::Aicc | Metric::Bic => Some(DeviceLoss::Squared),
        m => DeviceLoss::for_metric(m),
    }
}

/// #223 + #229: why an explicit `--fit-strategy gpu` cannot honour `metric`,
/// or `None` when it can.
///
/// The batched kernel (`src/cuda/eval.cu`, `fit_kernel`) minimises one
/// configured objective per launch: weighted squared error for the
/// least-squares family, and `log_squashed` as weighted LM/IRLS since #229.
/// Every metric outside [`fit_loss`] fits on the CPU through logistic IRLS or
/// the coordinate polish, so accepting the combination would silently
/// substitute a different objective for the one the user asked to optimise.
/// The CLI and the web front end turn this into a usage error before any data
/// is read; the search's own dispatch requires the same predicate, so the
/// refusal and the dispatch can never disagree.
pub fn gpu_fit_metric_error(metric: Metric) -> Option<String> {
    if fit_loss(metric).is_some() {
        return None;
    }
    Some(format!(
        "--fit-strategy gpu minimises squared error or log-squashed loss only, but \
         --metric {} has no native device objective, so the GPU LM cannot fit it. \
         Use --metric squared (aic, aicc and bic also fit by least squares) or \
         --metric log_squashed, or a CPU fit strategy: --fit-strategy exact, auto, \
         linear, varpro or stream.",
        metric.name()
    ))
}
}

/// #116: the largest design matrix (`rows × constants`) the linear arm will
/// materialise. Above it the exact LM path runs, so a large dataset never pays
/// an unbounded allocation. 8M cells is 64 MiB of `f64` at the cap.
const LINEAR_MAX_CELLS: usize = 8_000_000;

/// #116: largest number of constants the linear arm will solve for.
const LINEAR_MAX_CONSTANTS: usize = 64;

/// #116: pre-registered rank tolerance for the column-scaled QR. Every column
/// is normalised to unit 2-norm before factorisation, so a trailing pivot norm
/// at or below this is treated as rank-deficient and the tree falls back to the
/// exact LM path.
const LINEAR_RANK_TOL: f64 = 1e-12;

/// #116: process-wide counters for `--fit-report`. They are diagnostics only;
/// no search decision reads them, so they cannot change a trajectory.
static LINEAR_FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VARPRO_FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STREAM_FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EXACT_FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GPU_FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GPU_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GPU_REFITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// #116: all arm counts since process start. Kept as a struct so new arms do
/// not change the meaning of older report fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FitArmCounts {
    pub linear: u64,
    pub varpro: u64,
    pub stream: u64,
    pub exact: u64,
    pub gpu: u64,
    pub gpu_fallbacks: u64,
    pub gpu_refits: u64,
}

impl FitArmCounts {
    /// Difference from an earlier reading (for one run in a reused process).
    pub fn since(self, before: FitArmCounts) -> FitArmCounts {
        FitArmCounts {
            linear: self.linear.saturating_sub(before.linear),
            varpro: self.varpro.saturating_sub(before.varpro),
            stream: self.stream.saturating_sub(before.stream),
            exact: self.exact.saturating_sub(before.exact),
            gpu: self.gpu.saturating_sub(before.gpu),
            gpu_fallbacks: self.gpu_fallbacks.saturating_sub(before.gpu_fallbacks),
            gpu_refits: self.gpu_refits.saturating_sub(before.gpu_refits),
        }
    }

    pub fn total(self) -> u64 {
        self.linear + self.varpro + self.stream + self.exact
    }
}

/// #116: `(linear, exact-LM)` fit-arm calls since process start.
///
/// Retained for callers written before the portfolio grew. Use
/// [`fit_arm_counts_full`] to see VarPro, stream and GPU counts as well; the
/// second value here counts only calls that ended in the historical LM arm.
pub fn fit_arm_counts() -> (u64, u64) {
    let counts = fit_arm_counts_full();
    (counts.linear, counts.exact)
}

/// #116: all arm counts since process start.
pub fn fit_arm_counts_full() -> FitArmCounts {
    use std::sync::atomic::Ordering;
    FitArmCounts {
        linear: LINEAR_FITS.load(Ordering::Relaxed),
        varpro: VARPRO_FITS.load(Ordering::Relaxed),
        stream: STREAM_FITS.load(Ordering::Relaxed),
        exact: EXACT_FITS.load(Ordering::Relaxed),
        gpu: GPU_FITS.load(Ordering::Relaxed),
        gpu_fallbacks: GPU_FALLBACKS.load(Ordering::Relaxed),
        gpu_refits: GPU_REFITS.load(Ordering::Relaxed),
    }
}

/// #169: record one batched-GPU outcome (`true` = usable, `false` = fallback).
pub(crate) fn note_gpu_fit(usable: bool) {
    use std::sync::atomic::Ordering;
    if usable {
        GPU_FITS.fetch_add(1, Ordering::Relaxed);
    } else {
        GPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// #169: record one exact CPU `f64` finalist refit.
pub(crate) fn note_gpu_refit() {
    GPU_REFITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

// #169: batched-GPU fit accounting. `GPU_FITS` counts candidates the kernel
// completed with a usable result; `GPU_FALLBACKS` counts candidates the search
// then refit on the CPU (unsupported tree, non-finite row, ill-conditioned
// normal equations, or no device). `GPU_REFITS` counts the exact finalist
// refits performed before reporting. Diagnostics only; no decision reads them.
/// #169: `(gpu, fallback, exact_refit)` counts since process start.
pub fn gpu_fit_counts() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering;
    (
        GPU_FITS.load(Ordering::Relaxed),
        GPU_FALLBACKS.load(Ordering::Relaxed),
        GPU_REFITS.load(Ordering::Relaxed),
    )
}

/// #169: largest number of constants the batched GPU kernel fits.
///
/// The kernel is instantiated for 16 constants with a 256-row chunk (about
/// 39.7 KB of shared memory) and for 20 constants with a 128-row chunk (about
/// 38.1 KB). Both fit two blocks per SM on `sm_86`; the larger alternative,
/// 20 constants with a 256-row chunk, needs about 49.6 KB and only one block
/// per SM. Any candidate with more than this many constants is labelled
/// `too_many_constants` and refit by the exact CPU arm.
pub const GPU_FIT_MAX_CONSTANTS: usize = 20;

/// #169: largest flattened program (nodes) the batched GPU kernel fits.
pub const GPU_FIT_MAX_NODES: usize = 96;

/// #169: fit-plan opcodes, kept numerically identical to the evaluator
/// opcodes in `src/gpu.rs` / `src/cuda/eval.cu` so the two kernels share the
/// arithmetic cases.
pub(crate) mod gpu_op {
    pub const CONST: u32 = 0;
    pub const VAR: u32 = 1;
    pub const ADD: u32 = 2;
    pub const SUB: u32 = 3;
    pub const MUL: u32 = 4;
    pub const DIV: u32 = 5;
    pub const SIN: u32 = 6;
    pub const COS: u32 = 7;
    pub const TAN: u32 = 8;
    pub const EXP: u32 = 9;
    pub const LOG: u32 = 10;
    pub const SQRT: u32 = 11;
    pub const ABS: u32 = 12;
    pub const TANH: u32 = 13;
    pub const SIG: u32 = 14;
    pub const SQUARE: u32 = 15;
    pub const CUBE: u32 = 16;
    pub const NEG: u32 = 17;
    pub const GAUSS: u32 = 18;
    pub const POW: u32 = 20;
    pub const SINH: u32 = 21;
    pub const COSH: u32 = 22;
    pub const ERF: u32 = 28;
    pub const ERFC: u32 = 29;
}

/// #169: map an operator to the fit kernel's opcode. Only operators with a
/// well-defined analytic derivative in both the value and adjoint sweeps are
/// eligible; everything else (piecewise-constant, logical, constraint, block,
/// aggregate and meta operators) makes the candidate fall back to the CPU.
fn gpu_fit_opcode(op: Op) -> Option<u32> {
    use gpu_op as g;
    use Op::*;
    Some(match op {
        Const => g::CONST,
        Input => g::VAR,
        Add => g::ADD,
        Sub => g::SUB,
        Mul => g::MUL,
        Div => g::DIV,
        Sin => g::SIN,
        Cos => g::COS,
        Tan => g::TAN,
        Exp => g::EXP,
        Ln => g::LOG,
        Sqrt => g::SQRT,
        Abs => g::ABS,
        Tanh => g::TANH,
        Logistic => g::SIG,
        Square => g::SQUARE,
        Cube => g::CUBE,
        Neg => g::NEG,
        Gauss => g::GAUSS,
        Pow => g::POW,
        Sinh => g::SINH,
        Cosh => g::COSH,
        Erf => g::ERF,
        Erfc => g::ERFC,
        _ => return None,
    })
}

/// #169: a candidate flattened for the batched GPU fit kernel.
///
/// The program is a postorder array of `4`-word records `[op, kid0, kid1,
/// slot]`; constants are kept in `c0` so the kernel can move them, and `flat`
/// is retained so the fitted values can be written back into the tree exactly
/// like the CPU LM path does.
pub(crate) struct GpuFitPlan {
    pub(crate) words: Vec<u32>,
    pub(crate) nconst: usize,
    pub(crate) c0: Vec<f64>,
    flat: Flat,
}

impl GpuFitPlan {
    /// Write `c` back into `tree`'s constant slots.
    pub(crate) fn apply(&self, tree: &mut Node, c: &[f64]) {
        write_constants(tree, &self.flat, c);
    }
}

/// #169: why a candidate cannot be offered to the batched GPU fit kernel.
///
/// These are the structural rejection reasons the admission policy labels
/// separately from the kernel's runtime statuses (ill-conditioned/non-finite)
/// and from the search-level "too few rows" guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFitReject {
    /// Too many constants for the kernel's instantiations (`p > 20`).
    TooManyConstants,
    /// The flattened program exceeds [`GPU_FIT_MAX_NODES`].
    TooManyNodes,
    /// The tree has no constants to fit.
    NoConstants,
    /// Block bodies share constants; the CPU path owns those.
    SharedConstants,
    /// An operator without an analytic value/adjoint pair.
    UnsupportedOp,
}

impl GpuFitReject {
    /// Stable label used by the JSON ledger and the docs.
    pub fn label(self) -> &'static str {
        match self {
            GpuFitReject::TooManyConstants => "too_many_constants",
            GpuFitReject::TooManyNodes => "too_many_nodes",
            GpuFitReject::NoConstants => "no_constants",
            GpuFitReject::SharedConstants => "shared_constants",
            GpuFitReject::UnsupportedOp => "unsupported_op",
        }
    }
}

/// #169: build the GPU fit plan for `tree`, or say why it is ineligible.
pub(crate) fn gpu_fit_plan_reason(tree: &Node) -> Result<GpuFitPlan, GpuFitReject> {
    let f = flatten(tree);
    if f.nconst == 0 {
        return Err(GpuFitReject::NoConstants);
    }
    if f.nconst > GPU_FIT_MAX_CONSTANTS {
        return Err(GpuFitReject::TooManyConstants);
    }
    if f.op.len() > GPU_FIT_MAX_NODES {
        return Err(GpuFitReject::TooManyNodes);
    }
    if !f.shared.is_empty() {
        return Err(GpuFitReject::SharedConstants); // the CPU path owns block bodies
    }
    let mut words = Vec::with_capacity(f.op.len() * 4);
    for i in 0..f.op.len() {
        let op = match gpu_fit_opcode(f.op[i]) {
            Some(op) => op,
            None => return Err(GpuFitReject::UnsupportedOp),
        };
        let kids = &f.kids[i];
        let k0 = kids.first().copied().unwrap_or(0) as u32;
        let k1 = kids.get(1).copied().unwrap_or(0) as u32;
        let slot = if f.cslot[i] >= 0 {
            f.cslot[i] as u32
        } else if f.op[i] == Op::Input {
            f.val[i] as u32
        } else {
            u32::MAX
        };
        words.extend_from_slice(&[op, k0, k1, slot]);
    }
    let mut c0 = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c0[f.cslot[i] as usize] = f.val[i];
        }
    }
    Ok(GpuFitPlan {
        words,
        nconst: f.nconst,
        c0,
        flat: f,
    })
}

/// #169: build the GPU fit plan for `tree`, or `None` when the candidate must
/// stay on the CPU (non-affine block bodies, too many constants or nodes, or an
/// operator without an analytic GPU derivative).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gpu_fit_plan(tree: &Node) -> Option<GpuFitPlan> {
    gpu_fit_plan_reason(tree).ok()
}

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
        // #87: `Exp` and `Div` are the two operators that can leave a
        // *finite* fitter value where `expr::eval` returns NaN or infinity.
        // The other operators' Rust implementations already produce a
        // non-finite value in the same cases (Ln(0) -> -inf, Sqrt(<0) -> NaN,
        // Tan(inf) -> NaN, Pow -> NaN, Atan2 -> NaN), so they cannot create a
        // reusable finite SSE. Guarding them too changed the multi-output
        // search trajectory without fixing a finite-vs-infinite divergence.
        Exp => {
            let x = a(0);
            if x > 700.0 {
                f64::INFINITY
            } else {
                x.exp()
            }
        }
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
        // #87: `expr::eval` rejects a near-zero denominator with NaN.
        Div => {
            let b = a(1);
            if b.abs() < 1e-12 {
                f64::NAN
            } else {
                a(0) / b
            }
        }
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
fn solve(a: &[Vec<f64>], b: &mut [f64], lambda: f64, k: usize) -> Option<Vec<f64>> {
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
    // #87: the scorer rejects a prediction/target length mismatch; the fitter
    // must not silently fit only the shorter of the two.
    if x.len() != y.len() {
        return f64::INFINITY;
    }
    if !w.is_empty() && w.len() != x.len() {
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

/// The weighted normal equations at one set of constants: `JᵀWJ` and `JᵀWr`.
/// Neither depends on the damping `λ`, so one build serves every retry at the
/// same `c` (MINPACK's `lmdif` does the same).
struct NormalEquations {
    jtj: Vec<Vec<f64>>,
    jtr: Vec<f64>,
}

/// Build `JᵀWJ` and `JᵀWr` at `c`, one row sweep with the adjoint pass.
#[allow(clippy::too_many_arguments)]
fn normal_equations(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    k: usize,
    v: &mut [f64],
    adj: &mut [f64],
    grad: &mut [f64],
) -> Option<NormalEquations> {
    // #87: mirror the scorer's length contract.
    if x.len() != y.len() {
        return None;
    }
    if !w.is_empty() && w.len() != x.len() {
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
        // Both the Jacobian and the residual carry sqrt(w), so the accumulated
        // matrices are `JᵀWJ` and `JᵀWr` (not a half-weighted system).
        let jw = sw.sqrt();
        for i in 0..k {
            let gi = jw * grad[i];
            if !gi.is_finite() {
                return None;
            }
            jtr[i] += gi * r;
            for j in 0..=i {
                jtj[i][j] += gi * (jw * grad[j]);
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            jtj[j][i] = jtj[i][j];
        }
    }
    Some(NormalEquations { jtj, jtr })
}

#[cfg(test)]
thread_local! {
    /// #135: number of LM/polish trial steps taken on this test thread.
    static FIT_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Solve the cached normal equations for one damping value and score the trial
/// constants. This is the old `gn_step` without the row sweep.
#[allow(clippy::too_many_arguments)]
fn gn_trial(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    lambda: f64,
    k: usize,
    n: &NormalEquations,
    v: &mut [f64],
) -> Option<(Vec<f64>, f64)> {
    #[cfg(test)]
    FIT_STEPS.with(|n| n.set(n.get() + 1));
    let mut rhs: Vec<f64> = n.jtr.iter().map(|v| -v).collect();
    let step = solve(&n.jtj, &mut rhs, lambda, k)?;
    let trial: Vec<f64> = (0..k)
        .map(|i| (c[i] + step[i]).clamp(-1e6, 1e6))
        .collect();
    let s = sse(f, x, y, w, &trial, v);
    Some((trial, s))
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
    let n = normal_equations(f, x, y, w, c, k, v, adj, grad)?;
    gn_trial(f, x, y, w, c, lambda, k, &n, v)
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
/// Piecewise-constant operators whose adjoint is zero almost everywhere.
///
/// A constant fitted through `sign` or `step` has no useful gradient: the
/// switch position stays wherever the search put it, so the printed threshold
/// is arbitrary. This function lists the operators found in a tree so the CLI
/// can warn when a fitted model contains one. `floor`, `ceil`, `round` and the
/// comparisons have the same zero derivative but are usually deliberate
/// rounding/structure, so they are not reported here.
pub fn zero_adjoint_ops(tree: &Node) -> Vec<&'static str> {
    fn walk(n: &Node, out: &mut Vec<&'static str>) {
        match n.op {
            Op::Sign => {
                if !out.contains(&"sign") {
                    out.push("sign");
                }
            }
            Op::Step => {
                if !out.contains(&"step") {
                    out.push("step");
                }
            }
            _ => {}
        }
        for k in &n.kids {
            walk(k, out);
        }
        if let Some(block) = &n.block {
            walk(&block.body, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

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
    fit_weighted_strategy(tree, x, y, w, iters, FitStrategy::Exact)
}

/// #116: weighted least squares through a selectable exact arm.
///
/// `FitStrategy::Exact` is exactly [`fit_weighted`] (the historical LM path),
/// so every existing caller, fixture and checkpoint trajectory is unchanged.
/// `Auto` tries the closed-form linear arm, then variable projection, and
/// falls back to LM for any tree those arms cannot solve exactly. The explicit
/// arms (`Linear`, `VarPro`, `Stream`) use only their own arm before the same
/// fallback.
pub fn fit_weighted_strategy(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    strategy: FitStrategy,
) -> f64 {
    use std::sync::atomic::Ordering;
    // #87: the scorer rejects a prediction/target length mismatch; the fitter
    // must not silently fit only the shorter of the two.
    if x.len() != y.len() {
        return f64::INFINITY;
    }
    if !w.is_empty() && w.len() != x.len() {
        return f64::INFINITY;
    }
    // #135: zero iterations means `--no-fit` on *every* strategy. In
    // particular the closed-form linear arm is still a least-squares solve and
    // must not run; `fit_lm` returns the generated constants' SSE without
    // taking a step. This holds for `Gpu` too: no batch kernel is dispatched.
    if iters == 0 {
        return fit_lm(tree, x, y, w, 0);
    }
    // #169: `Gpu` only changes the search's *scheduling* (offspring are left
    // for the batched kernel). Any direct call lacks that batch context, so it
    // uses the same exact CPU arms as `Auto`: the linear minimiser when it
    // applies, otherwise the historical LM. It is never a no-op.
    if strategy != FitStrategy::Exact {
        // #169: a direct call has no batch context. Record the fallback and use
        // the same exact CPU arms as `Auto`, never a no-op.
        if strategy == FitStrategy::Gpu {
            GPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
        let try_linear = matches!(strategy, FitStrategy::Auto | FitStrategy::Linear | FitStrategy::Gpu);
        let try_varpro = matches!(strategy, FitStrategy::Auto | FitStrategy::VarPro | FitStrategy::Gpu);
        if try_linear {
            if let Some(s) = fit_linear(tree, x, y, w) {
                LINEAR_FITS.fetch_add(1, Ordering::Relaxed);
                return s;
            }
        }
        // #116: structural cost eligibility. `auto` and the direct-GPU fallback
        // only select VarPro when the partition is measured to amortise the
        // outer Jacobian (`s >= 2` or `q >= 4`); otherwise the exact LM arm is
        // both cheaper and the reference. The explicit `varpro` arm is a user
        // override and keeps the old behaviour.
        let auto_like = matches!(strategy, FitStrategy::Auto | FitStrategy::Gpu);
        let varpro_eligible = !auto_like
            || separable_partition(tree)
                .map(|part| varpro::cost_eligible(&part))
                .unwrap_or(false);
        if try_varpro && varpro_eligible {
            // `auto` promises never to be worse than the reference LM arm.
            // VarPro is not a global solver for a general separable problem,
            // so run the exact LM path from the same starting tree and keep
            // whichever objective is lower. The explicit `varpro` arm does not
            // pay this verification cost.
            let original = (strategy == FitStrategy::Auto).then(|| tree.clone());
            if let Some(s) = varpro::fit_varpro(tree, x, y, w, iters) {
                if let Some(mut lm_tree) = original {
                    let lm_sse = fit_lm(&mut lm_tree, x, y, w, iters);
                    if lm_sse < s {
                        *tree = lm_tree;
                        EXACT_FITS.fetch_add(1, Ordering::Relaxed);
                        return lm_sse;
                    }
                }
                VARPRO_FITS.fetch_add(1, Ordering::Relaxed);
                return s;
            }
        }
        if strategy == FitStrategy::Stream {
            if let Some(s) = stream::fit_stream(tree, x, y, w, iters) {
                STREAM_FITS.fetch_add(1, Ordering::Relaxed);
                return s;
            }
        }
    }
    EXACT_FITS.fetch_add(1, Ordering::Relaxed);
    fit_lm(tree, x, y, w, iters)
}

/// #116: run the streaming arm over an explicit row source and record the arm
/// count. This is the entry point the search uses when a disk-backed
/// [`StoreRows`] was resolved for the training rows. `None` means the caller
/// must fall back to the exact resident arm.
pub fn fit_stream_source<S: RowSource>(tree: &mut Node, source: &S, iters: usize) -> Option<f64> {
    let sse = stream::fit_stream_with(tree, source, iters)?;
    STREAM_FITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(sse)
}

/// #116: the structural partition VarPro would use, for tests and `--fit-report`.
/// `None` means the tree is affine in every constant (the linear arm owns it)
/// or has no constant block that can be eliminated exactly.
pub fn separable_partition(tree: &Node) -> Option<VarProPartition> {
    varpro::partition(&flatten(tree))
}

/// The historical Levenberg–Marquardt fitter (see [`fit_weighted`]).
fn fit_lm(tree: &mut Node, x: &[Vec<f64>], y: &[f64], w: &[f64], iters: usize) -> f64 {
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
    // #135: `--no-fit` / `--fit-iters 0` must not fit at all. The LM loop
    // below runs `iters.max(1)` and a four-step polish, so without this early
    // return a "no fit" search still performs five LM iterations per candidate.
    if iters == 0 {
        return sse(&f, x, y, w, &c, &mut v);
    }
    // #87: this standalone SSE is intentionally kept. The first `gn_step`
    // forms the same residuals, but using them as the acceptance baseline
    // changed the LM trajectory on guard-free trees; exactness beats the one
    // saved sweep. Only the *final* SSE is reused (and only for `squared`).
    let mut best_sse = sse(&f, x, y, w, &c, &mut v);
    let mut lambda = 1e-3;
    // #95: `JᵀJ`/`Jᵀr` depend on `c`, not on `λ`. Build them once per outer
    // iteration and reuse the cached matrices for every damping retry; only a
    // change of `c` (an accepted step) invalidates them.
    let mut normal: Option<NormalEquations> = None;

    for _ in 0..iters.max(1) {
        if normal.is_none() {
            normal =
                normal_equations(&f, x, y, w, &c, k, &mut v, &mut adj, &mut grad);
        }
        let Some(n) = normal.as_ref() else { break };
        match gn_trial(&f, x, y, w, &c, lambda, k, n, &mut v) {
            Some((trial, tsse)) => {
                if tsse < best_sse {
                    c = trial;
                    best_sse = tsse;
                    best_c = c.clone();
                    lambda = (lambda / 8.0).max(1e-12);
                    normal = None; // `c` moved: rebuild on the next iteration
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

    write_constants(tree, &f, &best_c);
    best_sse
}

/// Write fitted constant values back into the tree, including constants shared
/// inside block bodies.
fn write_constants(tree: &mut Node, f: &Flat, best_c: &[f64]) {
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
    apply_shared_constants(tree, &shared_slots, best_c);
}

/// #116: closed-form weighted least squares for a tree that is affine in every
/// constant.
///
/// For such a tree `f(x; c) = f(x; c₀) + Σⱼ (cⱼ − c₀ⱼ)·gⱼ(x)`, and the adjoint
/// sweep at `c₀` gives the coefficients `gⱼ` exactly and independently of `c₀`.
/// The arm builds the weighted design matrix `√w·J`, solves it with a
/// column-scaled, column-pivoted Householder QR (never the normal equations, so
/// the condition number is not squared), and returns the weighted SSE. It
/// returns `None` — leaving the tree untouched — when the tree is not affine,
/// when the design matrix is too large, when the system is rank-deficient, when
/// a value is non-finite, or when the closed-form solution does not improve on
/// the starting constants; the caller then runs the exact LM path.
fn fit_linear(tree: &mut Node, x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<f64> {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) || x.is_empty() {
        return None;
    }
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || k > LINEAR_MAX_CONSTANTS || x.len().saturating_mul(k) > LINEAR_MAX_CELLS {
        return None;
    }
    if !affine_in_constants(&f) {
        return None;
    }

    let n = f.op.len();
    let mut c0 = vec![0.0; k];
    for i in 0..n {
        if f.cslot[i] >= 0 {
            c0[f.cslot[i] as usize] = f.val[i];
        }
    }
    let mut v = vec![0.0; n];
    let mut adj = vec![0.0; n];
    let mut grad = vec![0.0; k];

    // One row sweep: weighted Jacobian (row-major) and the negative weighted
    // residual at c0. Zero-weight rows contribute nothing, exactly as in `sse`.
    let mut jac = vec![0.0; x.len() * k];
    let mut rhs = vec![0.0; x.len()];
    let mut m = 0usize;
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if sw == 0.0 {
            continue;
        }
        if !(sw.is_finite() && sw > 0.0) {
            return None;
        }
        eval_row(&f, row, &c0, &mut v);
        let pred = v[f.root];
        if !pred.is_finite() {
            return None;
        }
        backward(&f, &v, &mut adj, &mut grad);
        let root_w = sw.sqrt();
        let base = m * k;
        for (j, &g) in grad.iter().enumerate() {
            if !g.is_finite() {
                return None;
            }
            jac[base + j] = root_w * g;
        }
        rhs[m] = -root_w * (pred - t);
        if !rhs[m].is_finite() {
            return None;
        }
        m += 1;
    }
    if m < k {
        return None;
    }

    // Column scaling makes the rank test independent of constant magnitude.
    let mut scale = vec![0.0; k];
    for j in 0..k {
        let mut s = 0.0;
        for i in 0..m {
            let a = jac[i * k + j];
            s += a * a;
        }
        let norm = s.sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        scale[j] = norm;
        for i in 0..m {
            jac[i * k + j] /= norm;
        }
    }

    let scaled = qr_least_squares(&mut jac, m, k, &mut rhs[..m])?;

    let mut best_c = c0.clone();
    for j in 0..k {
        best_c[j] += scaled[j] / scale[j];
    }
    let trial_sse = sse(&f, x, y, w, &best_c, &mut v);
    if !trial_sse.is_finite() {
        return None;
    }
    // Never let a numerical edge case replace a better starting point; the
    // exact LM path gets the tree unchanged when this fails.
    let start_sse = sse(&f, x, y, w, &c0, &mut v);
    if trial_sse > start_sse {
        return None;
    }
    if !best_c.iter().all(|c| c.is_finite()) {
        return None;
    }

    write_constants(tree, &f, &best_c);
    Some(trial_sse)
}

/// #116: is the flattened tree affine in every constant?
///
/// `has_const[i]` is true when the subtree at `i` reads a constant. `affine[i]`
/// is true when that subtree's value is an affine function of all constants.
/// The rules are structural, not numerical: a product of two constant-dependent
/// factors is bilinear (rejected); every nonlinear operator is accepted only
/// when its whole subtree is constant-free. `if` may choose between affine
/// branches only when its condition is constant-free.
fn affine_in_constants(f: &Flat) -> bool {
    let n = f.op.len();
    let mut has_const = vec![false; n];
    let mut affine = vec![true; n];
    for i in 0..n {
        let ks = &f.kids[i];
        if f.cslot[i] >= 0 {
            has_const[i] = true;
            affine[i] = true;
            continue;
        }
        if f.op[i] == Op::Input {
            continue;
        }
        has_const[i] = ks.iter().any(|&j| has_const[j]);
        affine[i] = match f.op[i] {
            Op::Neg => affine[ks[0]],
            Op::Add | Op::Sub => affine[ks[0]] && affine[ks[1]],
            // Affine only when at most one factor depends on a constant.
            Op::Mul => {
                affine[ks[0]] && affine[ks[1]] && !(has_const[ks[0]] && has_const[ks[1]])
            }
            Op::Div => affine[ks[0]] && !has_const[ks[1]],
            Op::If => !has_const[ks[0]] && affine[ks[1]] && affine[ks[2]],
            // Every other operator is affine only when it ignores the
            // constants entirely: a function, product, comparison or
            // constraint applied to a constant-dependent argument is not.
            _ => !has_const[i],
        };
    }
    affine[f.root]
}

/// #116: solve `min ||A x − b||₂` by Householder QR with column pivoting.
///
/// `a` is row-major `m × n` and already column-scaled; `b` is overwritten. The
/// returned vector is in the scaled coordinate system (the caller divides by
/// the column scales). `None` means rank-deficient or non-finite.
fn qr_least_squares(a: &mut [f64], m: usize, n: usize, b: &mut [f64]) -> Option<Vec<f64>> {
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Column pivoting on the trailing block. Columns start at unit norm, so
        // an absolute tolerance is a condition-number bound.
        let mut best = k;
        let mut best_norm = 0.0;
        for j in k..n {
            let mut s = 0.0;
            for i in k..m {
                let v = a[i * n + j];
                s += v * v;
            }
            let norm = s.sqrt();
            if norm > best_norm {
                best_norm = norm;
                best = j;
            }
        }
        if !(best_norm.is_finite() && best_norm > LINEAR_RANK_TOL) {
            return None;
        }
        if best != k {
            for i in 0..m {
                a.swap(i * n + k, i * n + best);
            }
            piv.swap(k, best);
        }

        // Householder reflector for column k below the diagonal.
        let mut nx = 0.0;
        for i in k..m {
            let v = a[i * n + k];
            nx += v * v;
        }
        let nx = nx.sqrt();
        if !(nx.is_finite() && nx > 0.0) {
            return None;
        }
        let alpha = if a[k * n + k] >= 0.0 { -nx } else { nx };
        let v0 = a[k * n + k] - alpha;
        let mut vtv = v0 * v0;
        for i in (k + 1)..m {
            let v = a[i * n + k];
            vtv += v * v;
        }
        if !(vtv.is_finite() && vtv > 0.0) {
            return None;
        }
        // Apply H = I − 2vvᵀ/vtv to the trailing columns and to b.
        for j in (k + 1)..n {
            let mut dot = v0 * a[k * n + j];
            for i in (k + 1)..m {
                dot += a[i * n + k] * a[i * n + j];
            }
            let f = 2.0 * dot / vtv;
            a[k * n + j] -= f * v0;
            for i in (k + 1)..m {
                a[i * n + j] -= f * a[i * n + k];
            }
        }
        let mut dot = v0 * b[k];
        for i in (k + 1)..m {
            dot += a[i * n + k] * b[i];
        }
        let f = 2.0 * dot / vtv;
        b[k] -= f * v0;
        for i in (k + 1)..m {
            b[i] -= f * a[i * n + k];
        }
        a[k * n + k] = alpha;
        for i in (k + 1)..m {
            a[i * n + k] = 0.0;
        }
    }

    // R z = Qᵀb, then undo the column permutation.
    let mut z = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * z[j];
        }
        let d = a[i * n + i];
        if !(d.is_finite() && d.abs() > LINEAR_RANK_TOL) {
            return None;
        }
        z[i] = s / d;
    }
    let mut out = vec![0.0; n];
    for k in 0..n {
        if !z[k].is_finite() {
            return None;
        }
        out[piv[k]] = z[k];
    }
    Some(out)
}

/// Fit constants against the **configured** metric (issue #117).
///
/// `fit_weighted` is the squared-error solver and remains the exact default:
/// `squared`, `aic`, `aicc` and `bic` delegate to it without a single changed
/// arithmetic operation. The other metrics use the same LM/QR machinery with a
/// deterministic outer reweighting:
///
/// * `absolute` — IRLS with weights `1 / max(|r|, ε)` (the L1 majoriser).
/// * `log_squashed` — IRLS with weights `1 / (1 + r²)`, the derivative of
///   `ln(1 + r²)` split over the residual.
/// * `auc`, `log_loss`, `hinge` — logistic IRLS on the binary label, with the
///   model output treated as a logit (the convention all three metrics use).
///   `auc` and `hinge` share the logistic surrogate because their own
///   objectives are piecewise constant or non-smooth; the surrogate is convex
///   and the scored metric decides whether a trial is accepted.
/// * `correlation`, `rank_correlation`, `signed`, `median`, `maximum`,
///   `interquartile` and `slope` — a squared fit
///   followed by a deterministic coordinate polish on the configured metric.
///   These losses have no least-squares normal equations (rank and order
///   statistics are not differentiable); the polish is the documented fallback.
///
/// A trial is written back only when it improves the configured loss, so the
/// result is never worse than the starting constants. The returned value is the
/// fitted **sum of squared errors** for the squared family (preserving #87's
/// exact reuse) and the configured loss otherwise.
pub fn fit_weighted_metric(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> f64 {
    use Metric::*;
    // #135: `--no-fit` / `--fit-iters 0` must be a true no-op for every metric,
    // not only the squared family. IRLS, logistic IRLS and the coordinate
    // polish all force a minimum of one sweep, so return the current score
    // before reaching any of them. For the squared family this keeps the
    // historical weighted-SSE return value rather than the mean-squared score.
    if iters == 0 {
        if metric.is_least_squares() {
            return fit_weighted(tree, x, y, w, 0);
        }
        return metric_score(tree, x, y, w, metric);
    }
    // The exact historical path. Information criteria fit by least squares and
    // add their parameter penalty when scoring; they do not change the
    // constants' objective. #223 keeps the family in one predicate so the GPU
    // dispatch and the CLI refusal can never disagree with this fit.
    if metric.is_least_squares() {
        return fit_weighted(tree, x, y, w, iters);
    }
    match metric {
        // `Metric::is_least_squares` returned above for these.
        Squared | Aic | Aicc | Bic => {
            unreachable!("least-squares metrics are fitted by fit_weighted")
        }
        Absolute => irls_fit(tree, x, y, w, iters, metric, |r, _| {
            1.0 / r.abs().max(1e-6)
        }),
        LogSquashed => irls_fit(tree, x, y, w, iters, metric, |r, _| {
            1.0 / (1.0 + r * r)
        }),
        Auc | LogLoss | Hinge => logistic_irls(tree, x, y, w, iters, metric),
        // Smooth but not reducible to a weighted least-squares problem, and the
        // non-smooth order/rank/absolute-bias losses: start from the squared
        // fit, then polish the actual objective.
        Correlation
        | CorrelationSigned
        | RankCorrelation
        | RankCorrelationSigned
        | Signed
        | Median
        | Maximum
        | Interquartile
        | Slope
        | ImplicitResidual
        | ImplicitDerivative => {
            fit_weighted(tree, x, y, w, iters);
            coordinate_polish(tree, x, y, w, metric, iters.max(1));
            metric_score(tree, x, y, w, metric)
        }
    }
}

/// #116 + #117: strategy and metric together.
///
/// The closed-form linear arm minimises squared error only, so it is used for
/// the squared/AIC family. Every other metric keeps the IRLS/polish path from
/// #117, which is the exact objective for that metric.
pub fn fit_weighted_strategy_metric(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    strategy: FitStrategy,
    metric: Metric,
) -> f64 {
    if matches!(metric, Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic) {
        fit_weighted_strategy(tree, x, y, w, iters, strategy)
    } else {
        fit_weighted_metric(tree, x, y, w, iters, metric)
    }
}

/// Unweighted entry point for [`fit_weighted_metric`].
pub fn fit_metric(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    iters: usize,
    metric: Metric,
) -> f64 {
    fit_weighted_metric(tree, x, y, &[], iters, metric)
}

/// Evaluate `tree` on every row.
fn predictions(tree: &Node, x: &[Vec<f64>]) -> Vec<f64> {
    x.iter().map(|row| expr::eval(tree, row)).collect()
}

/// The configured metric at the tree's current constants.
fn metric_score(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64], metric: Metric) -> f64 {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    let pred = predictions(tree, x);
    metrics::score_weighted(&pred, y, w, metric, expr::count_constants(tree))
}

/// One IRLS round: fit weighted least squares to `target` with per-row
/// `weights`, then keep the trial only if the configured metric improves.
///
/// `target` and `weights` have one entry per row; the base `w` vector (if any)
/// is folded into `weights` before the call.
fn irls_fit<F>(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
    weight_of: F,
) -> f64
where
    F: Fn(f64, usize) -> f64,
{
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    let mut best = metric_score(tree, x, y, w, metric);
    if !best.is_finite() {
        return best;
    }
    // Keep the total number of LM sweeps close to `--fit-iters`: four weight
    // updates with `iters/4` inner sweeps each for the default of 8.
    let rounds = iters.clamp(1, 4);
    let inner = (iters / rounds).max(1);
    for _ in 0..rounds {
        let pred = predictions(tree, x);
        let mut target = Vec::with_capacity(x.len());
        let mut weights = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let r = pred[i] - y[i];
            if !r.is_finite() {
                return best;
            }
            let base = if w.is_empty() { 1.0 } else { w[i] };
            // A zero-weight row must stay invisible to the fit, exactly as in
            // `fit_weighted`.
            weights.push(if base > 0.0 {
                base * weight_of(r, i).max(0.0)
            } else {
                0.0
            });
            target.push(y[i]);
        }
        let mut trial = tree.clone();
        fit_weighted(&mut trial, x, &target, &weights, inner);
        let score = metric_score(&trial, x, y, w, metric);
        if score.is_finite() && score < best {
            best = score;
            *tree = trial;
        } else {
            break; // the majoriser no longer moves the objective
        }
    }
    // A final weighted solve at the converged weights removes the LM damping
    // that the short inner solves deliberately leave behind. It is only kept
    // when the configured loss improves, so it cannot make the fit worse.
    let pred = predictions(tree, x);
    if pred.iter().all(|p| p.is_finite()) {
        let mut target = Vec::with_capacity(x.len());
        let mut weights = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let r = pred[i] - y[i];
            let base = if w.is_empty() { 1.0 } else { w[i] };
            weights.push(if base > 0.0 {
                base * weight_of(r, i).max(0.0)
            } else {
                0.0
            });
            target.push(y[i]);
        }
        let mut trial = tree.clone();
        fit_weighted(&mut trial, x, &target, &weights, iters.max(4));
        let score = metric_score(&trial, x, y, w, metric);
        if score.is_finite() && score < best {
            best = score;
            *tree = trial;
        }
    }
    best
}

/// Binary labels at the target midpoint, matching [`Metric::is_classification`].
fn classification_labels(y: &[f64]) -> Vec<f64> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in y {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let mid = 0.5 * (lo + hi);
    y.iter().map(|&v| if v > mid { 1.0 } else { 0.0 }).collect()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Logistic IRLS for the classification metrics.
///
/// The model output is the logit `p`; the label is `t ∈ {-1, +1}`. Log loss is
/// `ln(1 + exp(-t·p))`, whose Newton step is the weighted least-squares problem
/// `min Σ q(1−q)·(p − z)²` with `q = sigmoid(t·p)` and `z = p + t/q`. The
/// pseudo-response `z` is bounded because `q` is clamped; every trial is judged
/// by the configured metric, so `auc` and `hinge` may accept or reject a step
/// on their own (non-smooth) criterion.
fn logistic_irls(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> f64 {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    let mut best = metric_score(tree, x, y, w, metric);
    if !best.is_finite() {
        return best;
    }
    let labels = classification_labels(y);
    let rounds = iters.max(1).min(16);
    for _ in 0..rounds {
        let pred = predictions(tree, x);
        if pred.iter().any(|p| !p.is_finite()) {
            break;
        }
        let mut target = Vec::with_capacity(x.len());
        let mut weights = Vec::with_capacity(x.len());
        for (i, &p) in pred.iter().enumerate() {
            let t = 2.0 * labels[i] - 1.0;
            let q = sigmoid(t * p).clamp(1e-9, 1.0 - 1e-9);
            let base = if w.is_empty() { 1.0 } else { w[i] };
            weights.push(if base > 0.0 { base * q * (1.0 - q) } else { 0.0 });
            target.push((p + t / q).clamp(-1e6, 1e6));
        }
        let mut trial = tree.clone();
        fit_weighted(&mut trial, x, &target, &weights, 2);
        let score = metric_score(&trial, x, y, w, metric);
        if score.is_finite() && score < best {
            best = score;
            *tree = trial;
        } else {
            break;
        }
    }
    best
}

/// Deterministic coordinate polish on a metric without a least-squares form.
///
/// The squared fit supplies the starting point. Each constant is then moved by
/// one of six geometric steps (`±10%, ±3%, ±1%` of `max(|c|, 1)`) and the move
/// is kept only if the configured metric strictly improves. Sweeps stop early
/// when a full pass makes no improvement. No randomness is involved.
fn coordinate_polish(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
    sweeps: usize,
) {
    let mut best = metric_score(tree, x, y, w, metric);
    if !best.is_finite() {
        return;
    }
    let mut positions = Vec::new();
    let mut counter = 0;
    const_positions(tree, &mut positions, &mut counter);
    if positions.is_empty() {
        return;
    }
    for _ in 0..sweeps.min(12) {
        let mut improved = false;
        for &pos in &positions {
            let Some(node) = expr::nth(tree, pos) else {
                continue;
            };
            let original = node.val;
            let scale = original.abs().max(1.0);
            let mut accepted = false;
            for &(mul, sign) in &[
                (0.1, 1.0),
                (0.1, -1.0),
                (0.03, 1.0),
                (0.03, -1.0),
                (0.01, 1.0),
                (0.01, -1.0),
            ] {
                let trial_value = original + sign * mul * scale;
                if let Some(node) = expr::nth_mut(tree, pos) {
                    node.val = trial_value;
                }
                let score = metric_score(tree, x, y, w, metric);
                if score.is_finite() && score < best {
                    best = score;
                    improved = true;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                // No candidate improved: put the original value back.
                if let Some(node) = expr::nth_mut(tree, pos) {
                    node.val = original;
                }
            }
        }
        if !improved {
            break;
        }
    }
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

    /// #169: the GPU fit plan's constant cap is 20. `p <= 20` is offered to the
    /// kernel (no silent fallback) and `p = 21` is the first labelled
    /// `too_many_constants` rejection. CPU-only: the plan builder is shared by
    /// every build.
    #[test]
    fn the_gpu_fit_plan_accepts_twenty_constants_not_twenty_one() {
        fn many_constants(p: usize) -> Node {
            let mut t = node(Op::Mul, vec![constant(0.1), input(0)]);
            for k in 1..p {
                t = node(
                    Op::Add,
                    vec![
                        t,
                        node(Op::Mul, vec![constant(0.2 + 0.01 * k as f64), input(k)]),
                    ],
                );
            }
            t
        }
        assert_eq!(GPU_FIT_MAX_CONSTANTS, 20);
        for p in 17..=20usize {
            let plan = gpu_fit_plan_reason(&many_constants(p))
                .unwrap_or_else(|e| panic!("p={} must be accepted: {:?}", p, e));
            assert_eq!(plan.nconst, p);
        }
        assert_eq!(
            gpu_fit_plan_reason(&many_constants(21)).err(),
            Some(GpuFitReject::TooManyConstants)
        );
    }

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
    fn zero_adjoint_warning_lists_only_sign_and_step() {
        let step = node(Op::Step, vec![node(Op::Mul, vec![constant(1.3), input(0)])]);
        assert_eq!(zero_adjoint_ops(&step), vec!["step"]);
        let sign = node(Op::Sign, vec![step.clone()]);
        assert_eq!(zero_adjoint_ops(&sign), vec!["sign", "step"]);
        // A smooth tree raises no warning, and rounding is not reported.
        let smooth = node(Op::Sin, vec![node(Op::Mul, vec![constant(1.3), input(0)])]);
        assert!(zero_adjoint_ops(&smooth).is_empty());
        let rounded = node(Op::Round, vec![node(Op::Mul, vec![constant(1.3), input(0)])]);
        assert!(zero_adjoint_ops(&rounded).is_empty());
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

    /// Evaluate a tree through the fitter's flat representation, including the
    /// constants stored on its nodes.
    fn flat_value(tree: &Node, x: &[f64]) -> f64 {
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
        v[f.root]
    }

    fn flat_sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let f = flatten(tree);
        let n = f.op.len();
        let mut c = vec![0.0; f.nconst];
        for i in 0..n {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        let mut v = vec![0.0; n];
        sse(&f, x, y, &[], &c, &mut v)
    }

    /// #87: the fitter's guarded operators must produce the same value as
    /// `expr::eval`, bit for bit. A finite fitter value where the scorer says
    /// NaN/infinity would make the reused SSE wrong.
    #[test]
    fn value_of_matches_expr_eval_on_guarded_operators() {
        let div = || {
            node(
                Op::Div,
                vec![
                    constant(1.0),
                    node(Op::Sub, vec![input(0), constant(0.5)]),
                ],
            )
        };
        let cases = vec![
            // Div: denominator exactly zero and just inside the guard.
            (div(), vec![0.5]),
            (div(), vec![0.5 + 1e-13]),
            // Exp: > 700 is +inf in `expr::eval`; 700 is still finite.
            (
                node(Op::Exp, vec![node(Op::Mul, vec![constant(701.0), input(0)])]),
                vec![1.0],
            ),
            (
                node(Op::Exp, vec![node(Op::Mul, vec![constant(700.0), input(0)])]),
                vec![1.0],
            ),
        ];
        for (tree, x) in cases {
            let want = expr::eval(&tree, &x);
            let got = flat_value(&tree, &x);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "flat value {got} vs expr::eval {want} for {tree:?}"
            );
        }
    }

    /// #87: a guard that fires makes the fitter's SSE infinite, exactly as the
    /// scorer rejects the candidate, so no finite SSE can be reused.
    #[test]
    fn sse_is_infinite_when_a_guard_fires() {
        let x = vec![vec![0.5]];
        let y = vec![0.0];
        let div = node(
            Op::Div,
            vec![
                constant(1.0),
                node(Op::Sub, vec![input(0), constant(0.5)]),
            ],
        );
        assert!(flat_sse(&div, &x, &y).is_infinite());

        let x = vec![vec![1.0]];
        let exp = node(Op::Exp, vec![node(Op::Mul, vec![constant(701.0), input(0)])]);
        assert!(flat_sse(&exp, &x, &y).is_infinite());
    }

    /// #87: the fitter must reject a prediction/target length mismatch instead
    /// of silently fitting the shorter of the two.
    fn line(a: f64, b: f64) -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
        )
    }

    fn values(tree: &Node) -> Vec<f64> {
        let mut out = Vec::new();
        let mut counter = 0;
        let mut positions = Vec::new();
        const_positions(tree, &mut positions, &mut counter);
        for pos in positions {
            out.push(expr::nth(tree, pos).unwrap().val);
        }
        out
    }

    #[test]
    fn squared_metric_fitting_is_bit_identical_to_the_old_path() {
        let x: Vec<Vec<f64>> = (0..31)
            .map(|i| vec![-1.0 + 2.0 * i as f64 / 30.0])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 2.0 * r[0].sin() + 0.25 * r[0] * r[0])
            .collect();
        let mut old = line(0.5, 0.0);
        let mut new = old.clone();
        let old_sse = fit_weighted(&mut old, &x, &y, &[], 8);
        let new_sse = fit_weighted_metric(&mut new, &x, &y, &[], 8, Metric::Squared);
        assert_eq!(old_sse.to_bits(), new_sse.to_bits());
        let (a, b) = (values(&old), values(&new));
        assert_eq!(a.len(), b.len());
        for (u, v) in a.iter().zip(&b) {
            assert_eq!(u.to_bits(), v.to_bits(), "squared changed a constant");
        }
    }

    /// An outlier the squared loss chases and the L1 loss ignores.
    fn outlier_fixture() -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (i, &v) in xs.iter().enumerate() {
            x.push(vec![v]);
            y.push(if i == xs.len() - 1 { 20.0 } else { 2.0 * v + 1.0 });
        }
        (x, y)
    }

    #[test]
    fn l1_fitting_beats_squared_fitting_on_the_l1_objective() {
        let (x, y) = outlier_fixture();
        let mut l1 = line(0.5, 0.0);
        let mut l2 = line(0.5, 0.0);
        let l1_loss = fit_weighted_metric(&mut l1, &x, &y, &[], 16, Metric::Absolute);
        fit_weighted_metric(&mut l2, &x, &y, &[], 16, Metric::Squared);
        let l1_at_l1 = metric_score(&l1, &x, &y, &[], Metric::Absolute);
        let l1_at_l2 = metric_score(&l2, &x, &y, &[], Metric::Absolute);
        assert!(l1_at_l1 <= l1_at_l2, "{l1_at_l1} vs {l1_at_l2}");
        assert!((l1_loss - l1_at_l1).abs() < 1e-12);
        // The two objectives really do have different optima.
        assert!(l1_at_l2 > 1e-3, "the fixture must be asymmetric");
        let sse_l1 = metric_score(&l1, &x, &y, &[], Metric::Squared);
        let sse_l2 = metric_score(&l2, &x, &y, &[], Metric::Squared);
        assert!(sse_l2 < sse_l1, "squared should still win on its own loss");
    }

    #[test]
    fn log_squashed_fitting_beats_squared_fitting_on_its_objective() {
        let (x, y) = outlier_fixture();
        let mut robust = line(0.5, 0.0);
        let mut l2 = line(0.5, 0.0);
        fit_weighted_metric(&mut robust, &x, &y, &[], 16, Metric::LogSquashed);
        fit_weighted_metric(&mut l2, &x, &y, &[], 16, Metric::Squared);
        let a = metric_score(&robust, &x, &y, &[], Metric::LogSquashed);
        let b = metric_score(&l2, &x, &y, &[], Metric::LogSquashed);
        assert!(a < b, "log_squashed fit {a} vs squared fit {b}");
    }

    #[test]
    fn logistic_fitting_beats_squared_fitting_on_log_loss() {
        let x: Vec<Vec<f64>> = (0..21)
            .map(|i| vec![-1.0 + 2.0 * i as f64 / 20.0])
            .collect();
        // A deliberately soft, separable label boundary.
        let y: Vec<f64> = x.iter().map(|r| if r[0] > 0.1 { 1.0 } else { 0.0 }).collect();
        let mut logistic = line(0.1, 0.0);
        let mut l2 = line(0.1, 0.0);
        fit_weighted_metric(&mut logistic, &x, &y, &[], 16, Metric::LogLoss);
        fit_weighted_metric(&mut l2, &x, &y, &[], 16, Metric::Squared);
        let a = metric_score(&logistic, &x, &y, &[], Metric::LogLoss);
        let b = metric_score(&l2, &x, &y, &[], Metric::LogLoss);
        assert!(a < b, "logistic fit {a} vs squared fit {b}");
        // The same fitted logit is also a better hinge classifier.
        let hinge_a = metric_score(&logistic, &x, &y, &[], Metric::Hinge);
        let hinge_b = metric_score(&l2, &x, &y, &[], Metric::Hinge);
        assert!(hinge_a <= hinge_b + 1e-12);
    }

    #[test]
    fn correlation_fitting_beats_squared_fitting_on_correlation() {
        let (x, y) = outlier_fixture();
        let mut corr = line(0.5, 0.0);
        let mut l2 = line(0.5, 0.0);
        fit_weighted_metric(&mut corr, &x, &y, &[], 12, Metric::Correlation);
        fit_weighted_metric(&mut l2, &x, &y, &[], 12, Metric::Squared);
        let a = metric_score(&corr, &x, &y, &[], Metric::Correlation);
        let b = metric_score(&l2, &x, &y, &[], Metric::Correlation);
        assert!(a < b, "correlation fit {a} vs squared fit {b}");
    }

    #[test]
    fn l1_fit_recovers_a_clean_proportional_law() {
        let x: Vec<Vec<f64>> = (0..21)
            .map(|i| vec![-1.0 + 2.0 * i as f64 / 20.0])
            .collect();
        let y: Vec<f64> = x.iter().map(|r| r[0]).collect();
        let mut tree = node(Op::Mul, vec![constant(3.0), input(0)]);
        let loss = fit_weighted_metric(&mut tree, &x, &y, &[], 8, Metric::Absolute);
        assert!(loss < 1e-12, "loss={loss}, values={:?}", values(&tree));
    }

    #[test]
    fn fit_rejects_a_length_mismatch() {
        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![0.0];
        let mut tree = node(Op::Mul, vec![constant(2.0), input(0)]);
        assert!(fit_weighted(&mut tree, &x, &y, &[], 4).is_infinite());
        assert!(flat_sse(&tree, &x, &y).is_infinite());
    }

    /// Fitted constants in slot order, read back through the flat form so the
    /// result does not depend on the tree walk used for write-back.
    fn fitted_constants(tree: &Node) -> Vec<f64> {
        let f = flatten(tree);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        c
    }

    /// #95: caching `JᵀJ`/`Jᵀr` across damping retries must be bit-identical to
    /// rebuilding them for every λ. These are the constants and SSE recorded
    /// from the per-retry implementation. The `exp` case starts badly scaled
    /// and rejects several steps before finding the basin.
    #[test]
    fn lm_cached_fit_matches_recorded_baseline() {
        let lin_x: Vec<Vec<f64>> =
            (0..41).map(|i| vec![-4.0 + 0.2 * i as f64]).collect();
        let lin_y: Vec<f64> = lin_x
            .iter()
            .map(|r| 2.0 * r[0] + 1.0 + 0.05 * (3.0 * r[0]).sin())
            .collect();

        let quad_x: Vec<Vec<f64>> =
            (0..31).map(|i| vec![-3.0 + 0.2 * i as f64]).collect();
        let quad_y: Vec<f64> = quad_x
            .iter()
            .map(|r| 0.5 * r[0] * r[0] - 2.0 * r[0] + 1.0 + 0.02 * (5.0 * r[0]).cos())
            .collect();

        let exp_x: Vec<Vec<f64>> =
            (0..41).map(|i| vec![-1.0 + 0.05 * i as f64]).collect();
        let exp_y: Vec<f64> = exp_x
            .iter()
            .map(|r| (1.5 * r[0]).exp() + 0.001 * (10.0 * r[0]).sin())
            .collect();

        let rat_x: Vec<Vec<f64>> =
            (0..19).map(|i| vec![0.5 + 0.25 * i as f64]).collect();
        let rat_y: Vec<f64> = rat_x.iter().map(|r| 2.0 / (r[0] - 0.2) + 0.5).collect();

        let sin_x: Vec<Vec<f64>> =
            (0..63).map(|i| vec![-3.1 + 0.1 * i as f64]).collect();
        let sin_y: Vec<f64> = sin_x
            .iter()
            .map(|r| 2.5 * (0.7 * r[0]).sin() + 0.3)
            .collect();

        let weighted_w: Vec<f64> = (0..41)
            .map(|i| match i % 5 {
                0 => 0.0,
                1 | 2 => 0.25,
                _ => 2.0,
            })
            .collect();

        let cases: Vec<(&str, Node, Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> = vec![
            (
                "linear",
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(0.0), input(0)]),
                        constant(0.0),
                    ],
                ),
                lin_x.clone(),
                lin_y.clone(),
                vec![],
            ),
            (
                "quadratic",
                node(
                    Op::Add,
                    vec![
                        node(
                            Op::Mul,
                            vec![
                                constant(0.0),
                                node(Op::Mul, vec![input(0), input(0)]),
                            ],
                        ),
                        node(
                            Op::Add,
                            vec![
                                node(Op::Mul, vec![constant(0.0), input(0)]),
                                constant(0.0),
                            ],
                        ),
                    ],
                ),
                quad_x,
                quad_y,
                vec![],
            ),
            (
                // Badly scaled start: exp(3x) fitted to exp(1.5x). Damping
                // rejects step after step here, which is the path #95 caches.
                "exp_bad_scale",
                node(
                    Op::Add,
                    vec![
                        node(Op::Exp, vec![node(Op::Mul, vec![constant(3.0), input(0)])]),
                        constant(0.5),
                    ],
                ),
                exp_x,
                exp_y,
                vec![],
            ),
            (
                "rational",
                node(
                    Op::Add,
                    vec![
                        node(
                            Op::Div,
                            vec![
                                constant(1.0),
                                node(Op::Sub, vec![input(0), constant(0.0)]),
                            ],
                        ),
                        constant(0.0),
                    ],
                ),
                rat_x,
                rat_y,
                vec![],
            ),
            (
                "damped_sine",
                node(
                    Op::Add,
                    vec![
                        node(
                            Op::Mul,
                            vec![
                                constant(1.0),
                                node(Op::Sin, vec![node(Op::Mul, vec![
                                    constant(1.0),
                                    input(0),
                                ])]),
                            ],
                        ),
                        constant(0.0),
                    ],
                ),
                sin_x,
                sin_y,
                vec![],
            ),
            (
                "weighted_linear",
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(0.0), input(0)]),
                        constant(0.0),
                    ],
                ),
                lin_x,
                lin_y,
                weighted_w,
            ),
        ];

        let expected: Vec<(Vec<u64>, u64)> = vec![
            (vec![0x3ffff3cde812a095, 0x3fefffffffffffff], 0x3fa9c6058c8f6e1a),
            (
                vec![0x3fe000441da12855, 0xc000000000000000, 0x3ff000b65a9771ca],
                0x3f78fdaf8bfef98f,
            ),
            (vec![0x3ff8004d303015c9, 0xbf097785fa307cfa], 0x3ef4074508c02711),
            (
                vec![0x3ffffffffffffffe, 0x3fc999999999999d, 0x3fe0000000000001],
                0x39b8400000000000,
            ),
            (
                vec![0x4004000000000000, 0x3fe6666666666666, 0x3fd3333333333333],
                0x0000000000000000,
            ),
            // #117: the weighted normal equations now scale both Jacobian and
            // residual by sqrt(w), so this is a genuinely weighted solve.
            (vec![0x3ffff2ce37c38d91, 0x3feff88cc309f8c7], 0x3fab8577d7aab41d),
        ];
        let mut got = Vec::new();
        for (name, mut tree, x, y, w) in cases {
            let sse = fit_weighted(&mut tree, &x, &y, &w, 8);
            let c: Vec<u64> = fitted_constants(&tree).iter().map(|v| v.to_bits()).collect();
            println!("CASE {name} consts={c:x?} sse={:016x}", sse.to_bits());
            got.push((c, sse.to_bits()));
        }
        assert_eq!(got, expected, "constants/SSE changed");
    }

    /// #135: `--fit-iters 0` / `--no-fit` must leave the constants exactly as
    /// generated and perform no LM or polish step. The step counter is bumped
    /// by every trial, so zero proves `gn_trial`/`gn_step` were never reached.
    #[test]
    fn zero_iters_does_not_fit_or_take_a_step() {
        FIT_STEPS.with(|n| n.set(0));
        let x: Vec<Vec<f64>> = (0..21).map(|i| vec![i as f64 / 5.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 1.0).collect();
        let mut tree = node(Op::Mul, vec![constant(3.0), input(0)]);
        let before = fitted_constants(&tree);
        let initial_sse = flat_sse(&tree, &x, &y);

        let sse = fit_weighted(&mut tree, &x, &y, &[], 0);

        assert_eq!(fitted_constants(&tree), before, "constants must not move");
        assert_eq!(sse.to_bits(), initial_sse.to_bits(), "SSE must be the generated one");
        assert_eq!(FIT_STEPS.with(|n| n.get()), 0, "no LM/polish step may run");

        // The same tree with the default budget must change the constants and
        // take steps; otherwise the counter assertion above would be vacuous.
        let fitted = fit_weighted(&mut tree, &x, &y, &[], 8);
        assert!(FIT_STEPS.with(|n| n.get()) > 0, "positive iters must take steps");
        assert_ne!(fitted_constants(&tree), before, "positive iters must fit");
        assert!(fitted < initial_sse, "positive iters must improve the SSE");
    }

    /// #135: zero iterations is a true no-op for *every* metric, including the
    /// IRLS/logistic paths that previously forced `iters.max(1)`, and for the
    /// closed-form linear arm that previously ran before the zero check.
    #[test]
    fn zero_iters_is_a_no_op_for_every_metric() {
        let x: Vec<Vec<f64>> = (0..41).map(|i| vec![-2.0 + 4.0 * i as f64 / 40.0]).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 1.7 * r[0] * r[0] - 0.4 * r[0] + 0.8)
            .collect();
        let metrics = [
            Metric::Squared,
            Metric::Absolute,
            Metric::LogSquashed,
            Metric::Auc,
            Metric::LogLoss,
            Metric::Hinge,
            Metric::Correlation,
            Metric::CorrelationSigned,
            Metric::RankCorrelation,
            Metric::RankCorrelationSigned,
            Metric::Signed,
            Metric::Median,
            Metric::Maximum,
            Metric::Interquartile,
            Metric::Slope,
            Metric::Aic,
            Metric::Aicc,
            Metric::Bic,
        ];
        for metric in metrics {
            let mut tree = node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![constant(3.0), input(0)]),
                    constant(0.5),
                ],
            );
            let before = fitted_constants(&tree);
            FIT_STEPS.with(|n| n.set(0));
            let score = fit_weighted_metric(&mut tree, &x, &y, &[], 0, metric);
            assert_eq!(
                fitted_constants(&tree),
                before,
                "{}: constants must stay as generated",
                metric.name()
            );
            assert!(
                score.is_finite(),
                "{}: the generated tree must still be scored, got {score}",
                metric.name()
            );
            assert_eq!(
                FIT_STEPS.with(|n| n.get()),
                0,
                "{}: no LM/polish step may run",
                metric.name()
            );

            // Positive iterations must still take steps, so the counter
            // assertion above cannot pass vacuously.
            FIT_STEPS.with(|n| n.set(0));
            fit_weighted_metric(&mut tree, &x, &y, &[], 8, metric);
            assert!(
                FIT_STEPS.with(|n| n.get()) > 0,
                "{}: positive iterations must take steps",
                metric.name()
            );
        }
    }

    /// #135: `auto`/`linear` must not take the closed-form least-squares arm
    /// when the budget is zero; on an exactly affine tree that arm would move
    /// the constants even though `--no-fit` was requested.
    #[test]
    fn zero_iters_skips_the_linear_arm_for_auto_and_linear() {
        let x: Vec<Vec<f64>> = (0..41).map(|i| vec![-2.0 + 4.0 * i as f64 / 40.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] + 2.0).collect();
        for strategy in [FitStrategy::Auto, FitStrategy::Linear] {
            let mut tree = node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![constant(1.0), input(0)]),
                    constant(0.0),
                ],
            );
            let before = fitted_constants(&tree);
            let initial_sse = flat_sse(&tree, &x, &y);
            FIT_STEPS.with(|n| n.set(0));
            let sse = fit_weighted_strategy(&mut tree, &x, &y, &[], 0, strategy);
            assert_eq!(
                fitted_constants(&tree),
                before,
                "{strategy:?}: constants must stay as generated"
            );
            assert_eq!(
                sse.to_bits(),
                initial_sse.to_bits(),
                "{strategy:?}: SSE must be the generated one"
            );
            assert_eq!(
                FIT_STEPS.with(|n| n.get()),
                0,
                "{strategy:?}: no least-squares or LM step may run"
            );
        }
    }

    /// #116: `a*x + b` is affine in both constants and the closed-form arm
    /// recovers the generating law to machine precision in one solve.
    #[test]
    fn linear_arm_recovers_an_exact_affine_law() {
        let x: Vec<Vec<f64>> = (0..64).map(|i| vec![-2.0 + 4.0 * i as f64 / 63.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] + 2.0).collect();
        let mut tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)],
        );
        assert!(affine_in_constants(&flatten(&tree)));

        let sse = fit_weighted_strategy(&mut tree, &x, &y, &[], 8, FitStrategy::Linear);
        let c = fitted_constants(&tree);
        assert!(sse < 1e-24, "an exact affine law should fit to rounding: {sse}");
        assert!((c[0] - 3.0).abs() < 1e-12, "slope {c:?}");
        assert!((c[1] - 2.0).abs() < 1e-12, "intercept {c:?}");
    }

    /// #116: on affine trees the closed-form objective is never worse than the
    /// exact LM objective, and the constants agree to a pre-registered 1e-8
    /// relative tolerance. The tolerance is deliberately loose for LM: its
    /// undamped polish stops at a relative SSE reduction rather than solving
    /// the linear system directly.
    #[test]
    fn linear_arm_agrees_with_exact_lm_on_affine_trees() {
        let x: Vec<Vec<f64>> = (0..101)
            .map(|i| {
                let t = -2.0 + 4.0 * i as f64 / 100.0;
                vec![t, (3.0 * t).sin()]
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 0.7 * r[0] * r[0] - 1.3 * r[1] + 2.5 + 0.01 * (7.0 * r[0]).cos())
            .collect();
        let build = || {
            node(
                Op::Add,
                vec![
                    node(
                        Op::Add,
                        vec![
                            node(
                                Op::Mul,
                                vec![constant(0.3), node(Op::Square, vec![input(0)])],
                            ),
                            node(Op::Mul, vec![constant(0.4), input(1)]),
                        ],
                    ),
                    constant(0.5),
                ],
            )
        };

        let mut lm = build();
        let lm_sse = fit_weighted(&mut lm, &x, &y, &[], 40);
        let mut qr = build();
        let qr_sse = fit_weighted_strategy(&mut qr, &x, &y, &[], 40, FitStrategy::Auto);
        assert!(
            qr_sse <= lm_sse * (1.0 + 1e-12),
            "QR {qr_sse} must not be worse than LM {lm_sse}"
        );
        let (a, b) = (fitted_constants(&lm), fitted_constants(&qr));
        for (i, (&u, &v)) in a.iter().zip(&b).enumerate() {
            let rel = (u - v).abs() / (1.0 + u.abs());
            assert!(rel <= 1e-8, "constant {i}: LM {u} vs QR {v} (relative {rel:e})");
        }
    }

    /// #116: a weighted affine fit ignores zero-weight rows exactly as the
    /// exact path does, and matches the closed-form weighted solution.
    #[test]
    fn linear_arm_is_weighted_least_squares() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        // Row 19 is a gross outlier with weight zero.
        let mut y = y;
        y[19] = 1.0e6;
        let mut w = vec![1.0; 20];
        w[19] = 0.0;
        // Row 0 and row 1 get larger weights, so the solution shifts slightly.
        w[0] = 4.0;
        w[1] = 9.0;
        let mut tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.0)],
        );
        let sse = fit_weighted_strategy(&mut tree, &x, &y, &w, 8, FitStrategy::Linear);
        assert!(sse.is_finite());
        // Compute the same normal equations directly for the two-parameter model.
        let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..20 {
            let (xi, yi, wi) = (x[i][0], y[i], w[i]);
            sw += wi;
            swx += wi * xi;
            swxx += wi * xi * xi;
            swy += wi * yi;
            swxy += wi * xi * yi;
        }
        let det = sw * swxx - swx * swx;
        let b = (sw * swxy - swx * swy) / det;
        let a = (swy - b * swx) / sw;
        let c = fitted_constants(&tree);
        assert!((c[0] - b).abs() <= 1e-10 * (1.0 + b.abs()), "slope {} vs {b}", c[0]);
        assert!((c[1] - a).abs() <= 1e-10 * (1.0 + a.abs()), "intercept {} vs {a}", c[1]);
    }

    /// #116: nonlinear, bilinear and otherwise non-affine trees are not
    /// classified as affine and take the exact LM path unchanged, bit for bit.
    #[test]
    fn non_affine_trees_fall_back_to_exact_lm_bit_for_bit() {
        let x: Vec<Vec<f64>> = (0..41).map(|i| vec![-2.0 + 0.1 * i as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 1.7 * (0.9 * r[0]).sin() + 0.4).collect();
        let cases = [
            // a*sin(b*x) + c: b is nonlinear.
            node(
                Op::Add,
                vec![
                    node(
                        Op::Mul,
                        vec![constant(0.5), node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])])],
                    ),
                    constant(0.0),
                ],
            ),
            // a*(b + x): bilinear in a and b.
            node(
                Op::Mul,
                vec![constant(1.0), node(Op::Add, vec![constant(1.0), input(0)])],
            ),
            // exp(a*x): nonlinear in a.
            node(Op::Exp, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
        ];
        for (i, tree) in cases.iter().enumerate() {
            assert!(!affine_in_constants(&flatten(tree)), "case {i} must not be affine");
            let mut lm = tree.clone();
            let mut qr = tree.clone();
            let a = fit_weighted(&mut lm, &x, &y, &[], 8);
            let b = fit_weighted_strategy(&mut qr, &x, &y, &[], 8, FitStrategy::Linear);
            assert_eq!(a.to_bits(), b.to_bits(), "case {i}: SSE changed");
            let ca: Vec<u64> = fitted_constants(&lm).iter().map(|v| v.to_bits()).collect();
            let cb: Vec<u64> = fitted_constants(&qr).iter().map(|v| v.to_bits()).collect();
            assert_eq!(ca, cb, "case {i}: constants changed");
        }
    }

    /// #116: a rank-deficient affine system (two indistinguishable constants)
    /// cannot be solved by the linear arm; it falls back to exact LM without
    /// touching the tree, and the result is bit-identical.
    /// #116: every strategy has a stable CLI/schema/checkpoint spelling and
    /// `cpu` remains an alias for the historical arm.
    #[test]
    fn strategy_names_round_trip() {
        for (name, strategy) in [
            ("exact", FitStrategy::Exact),
            ("cpu", FitStrategy::Exact),
            ("auto", FitStrategy::Auto),
            ("linear", FitStrategy::Linear),
            ("varpro", FitStrategy::VarPro),
            ("stream", FitStrategy::Stream),
            ("gpu", FitStrategy::Gpu),
        ] {
            assert_eq!(FitStrategy::parse(name), Some(strategy), "{name}");
        }
        for s in [
            FitStrategy::Exact,
            FitStrategy::Auto,
            FitStrategy::Linear,
            FitStrategy::VarPro,
            FitStrategy::Stream,
            FitStrategy::Gpu,
        ] {
            assert_eq!(FitStrategy::parse(s.name()), Some(s));
        }
        assert!(FitStrategy::parse("convex").is_none());
        assert!(FitStrategy::Gpu.defers_to_gpu());
        assert!(!FitStrategy::Auto.defers_to_gpu());
    }

    /// #223: an explicit GPU request is refused outside the least-squares
    /// family, and the refusal says why and what to use instead. The
    /// least-squares family is accepted, including `squared` and the three
    /// information criteria.
    #[test]
    fn explicit_gpu_names_the_gpu_lm_and_a_way_out_for_other_metrics() {
        for m in Metric::all() {
            let why = gpu_fit_metric_error(*m);
            if m.is_least_squares() {
                assert!(why.is_none(), "{} must stay accepted", m.name());
                continue;
            }
            let why = why.unwrap_or_else(|| panic!("{} must be refused", m.name()));
            assert!(
                why.contains("minimises squared error only"),
                "the refusal must name the GPU LM's objective: {why}"
            );
            assert!(
                why.contains(&format!("--metric {}", m.name())),
                "the refusal must name the metric: {why}"
            );
            assert!(
                why.contains("--metric squared"),
                "the refusal must point at --metric squared: {why}"
            );
            for strategy in ["exact", "auto", "linear", "varpro", "stream"] {
                assert!(
                    why.contains(strategy),
                    "the refusal must offer the CPU strategy {strategy}: {why}"
                );
            }
        }
    }

    #[test]
    fn rank_deficient_affine_tree_falls_back_to_exact_lm() {
        let x: Vec<Vec<f64>> = (0..9).map(|i| vec![i as f64]).collect();
        let y = vec![3.0, 2.8, 3.1, 2.9, 3.2, 3.0, 3.1, 2.95, 3.05];
        let tree = node(Op::Add, vec![constant(2.0), constant(1.0)]);
        let mut lm = tree.clone();
        let mut linear = tree.clone();
        let a = fit_weighted(&mut lm, &x, &y, &[], 8);
        let b = fit_weighted_strategy(&mut linear, &x, &y, &[], 8, FitStrategy::Linear);
        assert_eq!(a.to_bits(), b.to_bits());
        let ca: Vec<u64> = fitted_constants(&lm).iter().map(|v| v.to_bits()).collect();
        let cb: Vec<u64> = fitted_constants(&linear).iter().map(|v| v.to_bits()).collect();
        assert_eq!(ca, cb);
    }
}
