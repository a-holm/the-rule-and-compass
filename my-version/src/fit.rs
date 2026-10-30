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
mod correlation;
mod gpu;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
mod interquartile;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
mod median;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
mod minimax;
mod rank;
mod stream;
mod varpro;

pub use gpu::{
    backend_description, exact_metric_score, fit_batch_verified, fit_status_usable, gate_proposal,
    is_squared_family, maximum_finalist, refit_finalists_exact, signed_finalist, BatchStatus,
    BatchedFitter, ProposalGate, VerifiedBatch,
};
#[cfg(feature = "cuda")]
pub use gpu::BatchedGpu;
pub use stream::{
    fit_stream, fit_stream_with, RowSource, SliceRows, StoreRows, StoreSpec, STREAM_BLOCK_ROWS,
};
// #226: the median polish tail is only wired into the CUDA arm; a CPU-only
// build keeps the module (its unit tests) without re-exporting anything.
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use median::{polish_median_batch, OrderStatTrialScorer};
// #232: the interquartile arm is the same ladder on the middle-50 % window
// mean; it shares the driver above and differs only in the statistic.
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use interquartile::polish_interquartile_batch;
// #227: the maximum-error (minimax) polish tail, likewise CUDA-only in
// production and unit-tested from a CPU reference scorer in a plain build.
#[cfg(any(feature = "cuda", test))]
pub(crate) use minimax::{
    polish_minimax_batch, LevelInterval, MaxStat, MinimaxTrialScorer,
};
// The exact-rank stage is only wired into the CUDA arm; a CPU-only build keeps
// the module (its unit tests) without re-exporting anything.
#[cfg(any(feature = "cuda", test))]
pub(crate) use rank::polish_rank_batch;
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use rank::{exact_rank_score, RankTrialScorer};
// #231: the exact, orientation-aware rank gate is shared by the CPU verifier,
// the batched verifier and the device arm, so it exists in every build.
pub(crate) use rank::{choose_rank_fit, score_constants, RankFitChoice};
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
pub enum FitObjective {
    /// `Σ w·r²`.
    Squared,
    /// `Σ w·ln(1 + r²)`, minimised by IRLS with `2/(1 + r²)` row weights.
    LogSquashed,
    /// #224: `1 − |ρ|`. A **fit-only** objective: scoring is the separate
    /// [`crate::metrics::ScoreObjective::Correlation`] table (#287), so
    /// [`DeviceLoss::for_metric`] and [`fit_loss`] do not map it. Reachable
    /// through the fit kernel's direct batch API, not through the search fit
    /// dispatch.
    Correlation,
    /// #224: `1 − ρ`. Fit-only for the same reason as [`DeviceLoss::Correlation`];
    /// native scoring is [`crate::metrics::ScoreObjective::CorrelationSigned`].
    CorrelationSigned,
    /// #222: `Σ w·|r|`, minimised by per-row IRLS weights `1/max(|r|, 1e-6)`
    /// with true-`|r|` acceptance. Both the evaluator and the batched fit use
    /// this code, so a run can neither rank nor fit MAE with a squared-error
    /// surrogate.
    Absolute,
    /// #234: `Σ w·ln(1 + exp(-t·p))` on the binary label sign `t`, the model
    /// output read as a logit, minimised by Fisher scoring with `q(1-q)` row
    /// weights. A **fit-only** objective: [`DeviceLoss::for_metric`] does not
    /// map it, because the evaluator accumulates `ρ(pred − y)` and has no label
    /// or logit semantics. Reachable through [`fit_loss`] (fit dispatch) and
    /// the fit kernel's direct batch API only.
    Logistic,
}

/// Backwards-compatible name for the fit objective. New code should say
/// [`FitObjective`]: the scoring decision lives in
/// [`crate::metrics::ScoreObjective`] and is independent of this enum (#287).
pub type DeviceLoss = FitObjective;

impl FitObjective {
    /// The `metric` code the CUDA entry points take.
    pub fn code(self) -> i32 {
        match self {
            DeviceLoss::Squared => 0,
            DeviceLoss::LogSquashed => 1,
            DeviceLoss::Correlation => 2,
            DeviceLoss::CorrelationSigned => 3,
            DeviceLoss::Absolute => 4,
            DeviceLoss::Logistic => 5,
        }
    }

    /// Whether the legacy f32 evaluator entry points (`rc_eval`, `rc_pool_eval`)
    /// can accumulate this objective. Scoring itself is now driven by
    /// [`crate::metrics::ScoreObjective`] (#287); this flag only labels the
    /// fit-only arms.
    pub fn scores_on_device(self) -> bool {
        match self {
            DeviceLoss::Squared | DeviceLoss::LogSquashed | DeviceLoss::Absolute => true,
            DeviceLoss::Correlation | DeviceLoss::CorrelationSigned | DeviceLoss::Logistic => false,
        }
    }

    /// The fit objective for a metric that shares the squared/log/absolute
    /// family, or `None` otherwise.
    ///
    /// #234/#287: `Logistic` is deliberately absent — a `log_loss` run scores
    /// every candidate on the exact CPU metric and only its constants come from
    /// the device. Scoring is decided separately by
    /// [`crate::metrics::score_dispatch`].
    pub fn for_metric(metric: Metric) -> Option<Self> {
        match metric {
            Metric::Squared => Some(DeviceLoss::Squared),
            Metric::LogSquashed => Some(DeviceLoss::LogSquashed),
            Metric::Absolute => Some(DeviceLoss::Absolute),
            _ => None,
        }
    }
}

/// #229: the objective the **fit** kernel must minimise for a metric.
///
/// The information criteria fit by least squares and add their penalty when
/// scoring, so they share the squared objective; `log_squashed` uses the
/// robust arm; #222's `absolute` uses the L1/IRLS arm with true-`|r|`
/// acceptance. Everything else returns `None` and keeps the exact CPU fit.
pub fn fit_loss(metric: Metric) -> Option<DeviceLoss> {
    match metric {
        Metric::Aic | Metric::Aicc | Metric::Bic => Some(DeviceLoss::Squared),
        // #234: a native fit objective with no native fitness. The search
        // admits it as a fit-only arm; every score is the exact CPU log-loss.
        Metric::LogLoss => Some(DeviceLoss::Logistic),
        m => DeviceLoss::for_metric(m),
    }
}

/// #234: `Some(loss)` when the device fits `metric` but the evaluator cannot
/// score it, so every fitness value and every finalist refit stays exact CPU
/// `f64`; `None` when the metric is scored natively on the device or has no
/// device fit objective at all.
///
/// `log_loss` was the first metric in this class whose fit is the batched LM
/// kernel's own objective. #289 gives it (and Brier/hinge) a native scorer, so
/// the predicate now returns `None` for it; the split still names itself once
/// and the search's gate, scoring path and fit dispatch all read it.
pub fn fit_only_loss(metric: Metric) -> Option<DeviceLoss> {
    let loss = fit_loss(metric)?;
    // #289: a metric is fit-only only when the evaluator has no native
    // **score** objective either. `log_loss` used to be the example; it now
    // has a native per-row scorer (`ScoreObjective::LogLoss`), so it is no
    // longer fit-only and the split stays a single predicate.
    if loss.scores_on_device() || crate::metrics::ScoreObjective::for_metric(metric).is_some() {
        None
    } else {
        Some(loss)
    }
}

/// #223 + #229: why an explicit `--fit-strategy gpu` cannot honour `metric`,
/// or `None` when it can.
///
/// The batched kernel (`src/cuda/eval.cu`, `fit_kernel`) minimises one
/// configured objective per launch: weighted squared error for the
/// least-squares family, `log_squashed` as weighted LM/IRLS since #229,
/// `absolute` as weighted LM/IRLS with true-`|r|` acceptance since #222, and
/// the logistic cross-entropy on the binary label since #234.
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
    // #230: the rank family is honoured by the exact integer rank polish
    // (`rank_fit_only`), which is a **fit-only** device arm: the evaluator
    // still has no native rank scorer, so the fitness stays on the CPU.
    if rank_metric_signed(metric).is_some() {
        return None;
    }
    // #226: `median` is honoured by the batched exact median polish
    // (`median_fit_only`), also fit-only: the device sorts the residual bits,
    // the CPU scores every candidate with the configured median. #232: the
    // `interquartile` arm is the same stage on the middle-50 % window mean.
    if order_stat_fit_only(metric) {
        return None;
    }
    // #227: `maximum` is honoured by the batched exact maximum-error scan
    // (`maximum_fit_only`), also fit-only: the device reduces the exact
    // `max |r|` and drives the bisection on the error level, the CPU scores
    // every candidate with the configured maximum.
    if maximum_fit_only(metric) {
        return None;
    }
    Some(format!(
        "--fit-strategy gpu minimises squared error, log-squashed loss, mean \
         absolute error, the logistic log-loss or the exact rank, median, \
         interquartile and maximum objectives, but --metric {} has no device \
         fit objective, so the GPU arms cannot fit it. Use --metric squared \
         (aic, aicc and bic also fit by least squares), --metric log_squashed, \
         --metric absolute, --metric log_loss, --metric rank_correlation or \
         rank_correlation_signed, --metric median, --metric interquartile, \
         --metric maximum, or a CPU fit strategy: --fit-strategy exact, auto, \
         linear, varpro or stream.",
        metric.name()
    ))
}

/// #230: `Some(signed)` when `--fit-strategy gpu` can fit `metric` on the
/// device through the batched exact rank polish, `None` otherwise.
///
/// This is deliberately **not** [`fit_loss`]: the batched LM kernel does not
/// minimise Spearman's rho, the rank stage does. It is also not
/// [`DeviceLoss::for_metric`]: the fit dispatch has no rho LM objective, so a run
/// using this arm keeps every fitness value on the CPU. Callers use this
/// predicate for the fit dispatch; the *scoring* gate remains
/// [`crate::metrics::ScoreObjective`] instead.
pub fn rank_fit_only(metric: Metric) -> Option<bool> {
    rank_metric_signed(metric)
}

/// #226/#232: which order statistic of `|r|` an arm scores.
///
/// The two members share the ladder, the batched driver and the device sort;
/// they differ only in the pick. `metric()` is the CPU objective that is the
/// authority for the arm, so a device score can never be compared with a
/// different loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub(crate) enum OrderStat {
    /// `metrics::median_abs`: the middle value, or the mean of the two middles.
    Median,
    /// `metrics::interquartile_mean_abs`: the mean of ranks `[⌊n/4⌋, ⌊3n/4⌋)`.
    Interquartile,
}

impl OrderStat {
    /// The configured CPU metric this statistic is the definition of.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn metric(self) -> Metric {
        match self {
            OrderStat::Median => Metric::Median,
            OrderStat::Interquartile => Metric::Interquartile,
        }
    }

    /// The `what` code `rc_order_stat_score_batch` takes. Keep in sync with
    /// `src/cuda/eval.cu` (`k_order_stat_pick`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn device_code(self) -> i32 {
        match self {
            OrderStat::Median => 0,
            OrderStat::Interquartile => 1,
        }
    }

    /// The statistic for a metric, or `None` for every other metric.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    pub(crate) fn for_metric(metric: Metric) -> Option<Self> {
        match metric {
            Metric::Median => Some(OrderStat::Median),
            Metric::Interquartile => Some(OrderStat::Interquartile),
            _ => None,
        }
    }

    /// Stable label for the ledger, the docs and the fallback names.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    pub(crate) fn label(self) -> &'static str {
        match self {
            OrderStat::Median => "median",
            OrderStat::Interquartile => "interquartile",
        }
    }
}

/// #226: `true` when `--fit-strategy gpu` can fit `median` on the device
/// through the batched exact median polish, `false` for every other metric.
///
/// Like [`rank_fit_only`] this is a **fit-only** arm, not a [`fit_loss`]: the
/// device selects the exact median of `|pred − y|` (a segmented sort of the
/// residual bits) and never accumulates an SSE. The evaluator cannot score a
/// median, so this predicate selects the fit stage only; the *scoring* gate
/// remains [`crate::metrics::score_dispatch`] and every fitness value stays CPU `f64`.
pub fn median_fit_only(metric: Metric) -> bool {
    metric == Metric::Median
}

/// #232: `true` when `--fit-strategy gpu` can fit `interquartile` on the
/// device through the batched middle-50 % window polish, `false` otherwise.
///
/// Like [`median_fit_only`] this is a **fit-only** arm, not a [`fit_loss`]: the
/// device sorts the `|r|` bit patterns, averages ranks `[⌊n/4⌋, ⌊3n/4⌋)` in
/// `f64` and never accumulates an SSE. The evaluator cannot score the
/// interquartile mean, so this predicate selects the fit stage only.
pub fn interquartile_fit_only(metric: Metric) -> bool {
    metric == Metric::Interquartile
}

/// #226/#232: `true` when `--fit-strategy gpu` fits `metric` through the
/// batched **order-statistic** polish (median or interquartile), the arm those
/// two metrics share.
///
/// Both are fit-only: the device moves constants, the CPU scores and refits.
/// Every dispatch site reads this one predicate, so the arm and the admission
/// gate cannot disagree about which metrics it covers.
pub fn order_stat_fit_only(metric: Metric) -> bool {
    median_fit_only(metric) || interquartile_fit_only(metric)
}

/// #290: route an arm-specific measured cost pair through the common dispatch
/// model. The old helpers answered `n >= floor && gpu_s < cpu_s`; that boolean
/// is now a zero-hysteresis [`crate::dispatch::Dispatcher`] decision, so the
/// arm gates and the generic model cannot drift apart. Zero launch and zero
/// transfer preserve the arm model exactly; the floor is expressed as a named
/// `too_few_rows` refusal instead of a bare comparison.
fn modelled_gpu_worthwhile(
    n: usize,
    p: usize,
    sweeps: usize,
    cpu_s: f64,
    gpu_s: f64,
    min_rows: usize,
) -> bool {
    let mut calibration = crate::dispatch::Calibration::default();
    calibration.hysteresis = 0.0;
    calibration.launch_s = 0.0;
    let mut dispatcher = crate::dispatch::Dispatcher::new(calibration);
    let workload = crate::dispatch::Workload::fit(1, n, p, sweeps)
        .with_costs(cpu_s, gpu_s)
        .with_min_rows(min_rows)
        .with_bytes(0);
    dispatcher.plan(crate::dispatch::Stage::Fit, &workload).route
        == crate::dispatch::Route::Device
}

/// #226: rows below which the device median polish is not worth its per-call
/// latency and the CPU arm stays.
///
/// Derived from [`median_fit_cost`] (measured on the RTX 3060, see
/// `docs/gpu-backend.md` §"Median absolute error fitting"): one call costs a
/// fixed launch/sort setup plus a per-row term, so a small batch of small
/// samples loses to the CPU's single sort. At and above it the modelled device
/// polish is cheaper for every constant count the kernel accepts (`p ≤ 20`). The
/// decision is a row count, never a timing measurement, so it is reproducible.
pub const MEDIAN_GPU_MIN_ROWS: usize = 4_096;

/// #226: modelled wall time of **one** candidate's median fit tail, in
/// seconds, for `sweeps` ladder sweeps and `p` constants.
///
/// Both sides run the same ladder: `sweeps · p` calls of six trials. The CPU
/// side pays one prediction sweep plus one `n log n` sort of `f64` residuals
/// per trial; the device pays one prediction sweep, one segmented radix sort of
/// `u64` keys and one pick per trial, plus a fixed per-call cost. The
/// coefficients are the median-arm measurement recorded in
/// `docs/gpu-backend.md`, not guesses:
///
/// ```text
/// cpu_s = sweeps·p·6·CPU_ROW_S·n
/// gpu_s = sweeps·p·(GPU_CALL_S + 6·GPU_ROW_S·n) + GPU_CALL_S
/// ```
///
/// The CPU term is linear in `n` at the measured sizes: the `n log n` sort is
/// partly paid for by the sequential pass that feeds it (the fit is a linear
/// upper bound from 5 000 to 154 000 rows). `gpu_s` charges the baseline call
/// once, hence the trailing `GPU_CALL_S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MedianFitCost {
    /// Modelled CPU polish time, seconds.
    pub cpu_s: f64,
    /// Modelled device polish time, seconds.
    pub gpu_s: f64,
}

impl MedianFitCost {
    /// `cpu_s / gpu_s`, the modelled speed-up of the device tail.
    pub fn speedup(&self) -> f64 {
        if self.gpu_s > 0.0 {
            self.cpu_s / self.gpu_s
        } else {
            f64::INFINITY
        }
    }

    /// Whether the model prefers the device for this shape.
    pub fn prefers_gpu(&self) -> bool {
        self.gpu_s < self.cpu_s
    }
}

/// #226: per-row cost of one CPU trial (prediction sweep plus the `f64` sort
/// of the residuals), seconds. Measured mean over 5 000–154 000 rows.
const MEDIAN_CPU_ROW_S: f64 = 3.3e-8;
/// #226: fixed cost of one device call (uploads, CUB size query, buffer
/// allocation, launches and sync), seconds.
const MEDIAN_GPU_CALL_S: f64 = 3.5e-4;
/// #226: per-row cost of one device trial (f64 predict + segmented radix sort
/// of the keys + pick), seconds.
const MEDIAN_GPU_ROW_S: f64 = 3.95e-9;

/// #226: modelled CPU-vs-device cost of the median polish tail.
pub fn median_fit_cost(n: usize, p: usize, sweeps: usize) -> MedianFitCost {
    let nf = n.max(1) as f64;
    let trials = (sweeps.max(1) * p.max(1) * 6) as f64;
    let calls = (sweeps.max(1) * p.max(1)) as f64;
    MedianFitCost {
        cpu_s: trials * MEDIAN_CPU_ROW_S * nf,
        gpu_s: calls * (MEDIAN_GPU_CALL_S + 6.0 * MEDIAN_GPU_ROW_S * nf)
            + MEDIAN_GPU_CALL_S,
    }
}

/// #226: whether the device median polish is predicted to be faster than the
/// CPU polish for a sample of `n` rows and `p` constants.
pub fn median_gpu_fit_worthwhile(n: usize, p: usize, sweeps: usize) -> bool {
    let cost = median_fit_cost(n, p, sweeps);
    modelled_gpu_worthwhile(n, p, sweeps, cost.cpu_s, cost.gpu_s, MEDIAN_GPU_MIN_ROWS)
}

// --- #232: interquartile arm -------------------------------------------------

/// #232: rows below which the device interquartile polish is not worth its
/// per-call latency and the exact CPU arm stays.
///
/// Derived from [`interquartile_fit_cost`] (measured on the RTX 3060, see
/// `docs/gpu-backend.md` §"Interquartile absolute error fitting"): like the
/// median arm the device pays a fixed launch/sort cost per call and wins on
/// rows, so a small sample keeps the CPU's single sort. The decision is a row
/// count, never a timing measurement, so it is reproducible.
pub const INTERQUARTILE_GPU_MIN_ROWS: usize = 4_096;

/// #232: fixed cost of one device interquartile call (uploads, CUB size query,
/// buffer allocation, launches and sync), seconds. Measured: a six-trial call
/// costs 0.18 ms at 2 000 rows and 3.75 ms at 154 000, which is an intercept of
/// ~0.16 ms and a 6·3.9 ns/row slope.
const IQR_GPU_CALL_S: f64 = 1.6e-4;
/// #232: per-row cost of one device interquartile trial (f64 predict, segmented
/// radix sort of the keys, 256-thread window reduction), seconds. Measured
/// slope of the six-trial call: 3.9 ns/row — the parallel window sum is inside
/// the sort's noise, which is why it is below the median arm's 3.95 ns.
const IQR_GPU_ROW_S: f64 = 3.9e-9;

/// #232: modelled CPU-vs-device cost of the interquartile polish tail.
///
/// Both sides run `sweeps · p` calls of six trials. The CPU side is exactly the
/// median arm's per-trial work (one prediction sweep plus one `n log n` sort of
/// `f64` residuals, hence the shared [`MEDIAN_CPU_ROW_S`]); the device side is
/// the same sort as the median arm plus a **parallel** 256-thread window sum
/// over ~`n/2` keys, whose measured cost is inside the sort's noise (the
/// per-row coefficient is 3.9 ns, below the median's 3.95 ns). Both arms keep
/// the median arm's cost shape:
///
/// ```text
/// cpu_s = sweeps·p·6·MEDIAN_CPU_ROW_S·n
/// gpu_s = sweeps·p·(IQR_GPU_CALL_S + 6·IQR_GPU_ROW_S·n) + IQR_GPU_CALL_S
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterquartileFitCost {
    /// Modelled CPU polish time, seconds.
    pub cpu_s: f64,
    /// Modelled device polish time, seconds.
    pub gpu_s: f64,
}

impl InterquartileFitCost {
    /// `cpu_s / gpu_s`, the modelled speed-up of the device tail.
    pub fn speedup(&self) -> f64 {
        if self.gpu_s > 0.0 {
            self.cpu_s / self.gpu_s
        } else {
            f64::INFINITY
        }
    }

    /// Whether the model prefers the device for this shape.
    pub fn prefers_gpu(&self) -> bool {
        self.gpu_s < self.cpu_s
    }
}

/// #232: modelled CPU-vs-device cost of the interquartile polish tail, in
/// seconds.
///
/// The CPU side deliberately uses the median arm's per-trial coefficient: the
/// two metrics sort the same residual multiset, so the *shape* is identical.
/// The measured IQR runs (RTX 3060, see `docs/gpu-backend.md`) take longer per
/// trial because the IQR ladder more often improves on every sweep and so runs
/// all of them; charging the median's coefficient therefore under-counts the
/// CPU side and makes the admission model conservative.
pub fn interquartile_fit_cost(n: usize, p: usize, sweeps: usize) -> InterquartileFitCost {
    let nf = n.max(1) as f64;
    let trials = (sweeps.max(1) * p.max(1) * 6) as f64;
    let calls = (sweeps.max(1) * p.max(1)) as f64;
    InterquartileFitCost {
        cpu_s: trials * MEDIAN_CPU_ROW_S * nf,
        gpu_s: calls * (IQR_GPU_CALL_S + 6.0 * IQR_GPU_ROW_S * nf) + IQR_GPU_CALL_S,
    }
}

/// #232: whether the device interquartile polish is predicted to be faster
/// than the exact CPU arm for a sample of `n` rows and `p` constants.
pub fn interquartile_gpu_fit_worthwhile(n: usize, p: usize, sweeps: usize) -> bool {
    let cost = interquartile_fit_cost(n, p, sweeps);
    modelled_gpu_worthwhile(
        n,
        p,
        sweeps,
        cost.cpu_s,
        cost.gpu_s,
        INTERQUARTILE_GPU_MIN_ROWS,
    )
}

/// #227: `true` when `--fit-strategy gpu` can fit `maximum` on the device
/// through the batched exact maximum-error polish, `false` for every other
/// metric.
///
/// Like [`rank_fit_only`] and [`median_fit_only`] this is a **fit-only** arm,
/// not a [`fit_loss`]: the device reduces the exact `max |r|` (an order-free
/// integer max over the `|r|` bit patterns) and drives the bisection on the
/// error level, and the CPU scores every candidate with the configured
/// maximum. The *scoring* gate is [`crate::metrics::score_dispatch`] (#287), so a
/// `--metric maximum` run reports the `cpu` backend with that reason.
///
/// A **weighted** maximum is refused: `weights_supported` excludes `Maximum`
/// (`metrics::score_weighted` returns `+inf` for it), so there is no weighted
/// objective to fit and the device must not invent `max √wᵢ|rᵢ|`.
pub fn maximum_fit_only(metric: Metric) -> bool {
    metric == Metric::Maximum
}

/// #227: rows below which the device maximum-error polish is not worth its
/// per-call latency and the exact CPU arm stays.
///
/// Derived from [`maximum_fit_cost`] (measured on the RTX 3060, see
/// `docs/gpu-backend.md` §"Maximum absolute error fitting"): the bisection on
/// the error level calls the scan once per round, so a small sample pays more
/// in per-call latency than it saves in row work. The paired measurement with
/// the search's minimum admitted batch (32 candidates) shows the device ahead
/// even at 256 rows (2.6×), so 512 rows is the conservative floor: at or above
/// it the model predicts a speed-up for **every** constant count the kernel
/// accepts (`p = 1..=20`), not just the measured `p = 2`. The decision is a row
/// count, never a timing measurement, so it is reproducible.
pub const MAXIMUM_GPU_MIN_ROWS: usize = 512;

/// #227: modelled wall time of **one** candidate's maximum-error fit tail, in
/// seconds, for `sweeps` sweeps and `p` constants.
///
/// ```text
/// cpu_s = START_S + sweeps·p·6·CPU_EVAL_ROW_S·n
/// gpu_s = START_S + calls·GPU_CALL_S + sweeps·ROUNDS·p·GPU_SCAN_ROW_S·n
/// calls = 2 + sweeps·(ROUNDS + 1)
/// ```
///
/// The CPU side is `coordinate_polish`: `sweeps·p` constants × six fixed
/// geometric probes, one full `f64` prediction sweep each. The device side is
/// the baseline scan plus, per sweep, `ROUNDS` level scans (each a forward plus
/// an adjoint sweep over the rows) and one batched proposal score; every
/// coordinate of a candidate rides in **one** call, which is why the device
/// term scales with `p` per row but not per call. `GPU_CALL_S` is the
/// per-candidate share of one call at the search's minimum admitted batch
/// (32 candidates).
///
/// The coefficients are the paired measurement in
/// `docs/gpu-backend.md` §"Maximum absolute error fitting" (RTX 3060, p = 2,
/// 32 candidates, `--fit-iters 8`): CPU 23.4 / 78.3 / 623.8 ms and device
/// 9.1 / 10.4 / 74.2 ms at 256 / 1 024 / 8 192 rows. The CPU term is linear in
/// `n` over that range; the fitted intercept is within 20 % of `START_S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaximumFitCost {
    /// Modelled CPU polish time, seconds.
    pub cpu_s: f64,
    /// Modelled device polish time, seconds.
    pub gpu_s: f64,
}

impl MaximumFitCost {
    /// `cpu_s / gpu_s`, the modelled speed-up of the device tail.
    pub fn speedup(&self) -> f64 {
        if self.gpu_s > 0.0 {
            self.cpu_s / self.gpu_s
        } else {
            f64::INFINITY
        }
    }

    /// Whether the model prefers the device for this shape.
    pub fn prefers_gpu(&self) -> bool {
        self.gpu_s < self.cpu_s
    }
}

/// #227: fixed cost of the squared LM warm start (both sides run one), seconds.
const MAXIMUM_START_S: f64 = 1.5e-4;
/// #227: per-row cost of one CPU probe (`expr::eval` in `f64` plus the max
/// fold), seconds. Measured 24.6 ns for the search's minimum batch.
const MAXIMUM_CPU_EVAL_ROW_S: f64 = 2.46e-8;
/// #227: per-candidate share of one device scan call (one packed upload, one
/// launch, one event synchronisation and one download), seconds.
const MAXIMUM_GPU_CALL_S: f64 = 8.2e-7;
/// #227: per-row, per-coordinate cost of one device level scan (the `f64`
/// forward plus the adjoint sweep), seconds.
const MAXIMUM_GPU_SCAN_ROW_S: f64 = 5.0e-10;

/// #227: modelled CPU-vs-device cost of the maximum-error polish tail.
pub fn maximum_fit_cost(n: usize, p: usize, sweeps: usize) -> MaximumFitCost {
    let nf = n.max(1) as f64;
    let sweeps = sweeps.max(1) as f64;
    let rounds = minimax::MINIMAX_BISECT_ROUNDS as f64;
    let p = p.max(1) as f64;
    let probes = sweeps * p * 6.0;
    let calls = 2.0 + sweeps * (rounds + 1.0);
    let scans = sweeps * rounds * p;
    MaximumFitCost {
        cpu_s: MAXIMUM_START_S + probes * MAXIMUM_CPU_EVAL_ROW_S * nf,
        gpu_s: MAXIMUM_START_S + calls * MAXIMUM_GPU_CALL_S + scans * MAXIMUM_GPU_SCAN_ROW_S * nf,
    }
}

/// #227: whether the device maximum-error polish is predicted to be faster
/// than the CPU polish for a sample of `n` rows and `p` constants.
pub fn maximum_gpu_fit_worthwhile(n: usize, p: usize, sweeps: usize) -> bool {
    let cost = maximum_fit_cost(n, p, sweeps);
    modelled_gpu_worthwhile(n, p, sweeps, cost.cpu_s, cost.gpu_s, MAXIMUM_GPU_MIN_ROWS)
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
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub(crate) struct GpuFitPlan {
    pub(crate) words: Vec<u32>,
    pub(crate) nconst: usize,
    pub(crate) c0: Vec<f64>,
    /// #230: slots in the order `const_positions` visits the constant nodes,
    /// which is the order the CPU coordinate polish moves them in. The CUDA
    /// rank polish walks this list so a batch commits the same first improver.
    pub(crate) order: Vec<usize>,
    /// #224: the free affine pair (`scale`, `offset`) as constant slots, `-1`
    /// when absent. The correlation kernel freezes these slots and
    /// renormalises them; other metrics ignore the field.
    pub(crate) gauge: [i32; 2],
    flat: Flat,
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
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
    // #230: map pre-order tree positions onto constant slots. A block body
    // aliases one slot from several tree positions, so the two orderings stop
    // being a permutation; such a tree is rejected by `SharedConstants` above.
    let mut positions = Vec::with_capacity(f.nconst);
    let mut counter = 0usize;
    const_positions(tree, &mut positions, &mut counter);
    if positions.len() != f.nconst {
        return Err(GpuFitReject::SharedConstants);
    }
    let mut slot_of_pos = vec![usize::MAX; counter.max(1)];
    for (slot, &pos) in f.slot_pos.iter().enumerate() {
        if pos >= slot_of_pos.len() || slot_of_pos[pos] != usize::MAX {
            return Err(GpuFitReject::SharedConstants);
        }
        slot_of_pos[pos] = slot;
    }
    let mut order = Vec::with_capacity(f.nconst);
    for &pos in &positions {
        let slot = slot_of_pos.get(pos).copied().unwrap_or(usize::MAX);
        if slot == usize::MAX {
            return Err(GpuFitReject::SharedConstants);
        }
        order.push(slot);
    }
    let gauge = correlation::gauge_of(&f).slots();
    Ok(GpuFitPlan {
        words,
        nconst: f.nconst,
        c0,
        order,
        gauge,
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
    /// #230: pre-order node index (`expr::nth` numbering) of each constant
    /// slot, indexed by slot. `slot_pos[s]` is the tree position that holds
    /// slot `s`, which is what maps the CUDA slot order onto the CPU polish's
    /// `const_positions` order.
    slot_pos: Vec<usize>,
    /// Pre-order counter used to fill `slot_pos`.
    next_pos: usize,
}

fn push_flat(f: &mut Flat, op: Op, val: f64, kids: Vec<usize>, cslot: i32) -> usize {
    f.op.push(op);
    f.val.push(val);
    f.kids.push(kids);
    f.cslot.push(cslot);
    f.op.len() - 1
}

fn build(n: &Node, f: &mut Flat) -> usize {
    let my_pos = f.next_pos;
    f.next_pos += 1;
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
        f.slot_pos.push(my_pos);
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
        slot_pos: Vec::new(),
        next_pos: 0,
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

// ---------------------------------------------------------------------------
// #228: the signed-bias arm — a constrained least-squares projection.
// ---------------------------------------------------------------------------

/// #228: the exact signed-bias statistics at one set of constants.
///
/// With `rᵢ = f(xᵢ; c) − yᵢ`, `gᵢ = ∇_c rᵢ` and the raw per-row weights `wᵢ`:
///
/// ```text
/// sum_r  = Σ wᵢ rᵢ          (the objective numerator: |sum_r| / weight)
/// weight = Σ wᵢ
/// grad   = Σ wᵢ gᵢ          (the gradient of sum_r, i.e. the constraint row)
/// jtj    = Σ wᵢ gᵢ gᵢᵀ      (weighted normal matrix, sqrt(w)-folded as in
/// jtr    = Σ wᵢ gᵢ rᵢ       `normal_equations`, so the two share one row sweep)
/// abs_sum= Σ |wᵢ rᵢ|        (the rounding scale of sum_r)
/// ```
///
/// `sum_r` is accumulated in the same row order and with the same operations as
/// `metrics::score_weighted`, so `|sum_r| / weight` is bit-identical to the
/// reported `signed` objective on the same rows.
struct SignedSystem {
    jtj: Vec<Vec<f64>>,
    jtr: Vec<f64>,
    grad: Vec<f64>,
    sum_r: f64,
    weight: f64,
    abs_sum: f64,
    /// `Σ wᵢ(|predᵢ| + |yᵢ|)`: the magnitudes whose rounded difference *is*
    /// every residual. See [`Self::noise`].
    mag_sum: f64,
}

impl SignedSystem {
    fn objective(&self) -> f64 {
        (self.sum_r / self.weight).abs()
    }

    /// #228: the rounding scale of the objective.
    ///
    /// Two independent `f64` effects set it, and neither is about the data
    /// scale alone:
    ///
    /// 1. every residual is a rounded difference, `|δrᵢ| ≲ u(|predᵢ| + |yᵢ|)`,
    ///    so the numerator inherits `u·Σ wᵢ(|predᵢ| + |yᵢ|)`;
    /// 2. the step itself is a `f64` solve of the normal matrix, so the
    ///    constants — and with them the reachable bias — carry a `κ(M)·u`
    ///    factor; the caller passes the measured conditioning.
    ///
    /// The numerator is accumulated with Neumaier compensation
    /// ([`Compensated`]), so the `n`-term summation bound is not part of the
    /// floor. An objective at or below `noise` cannot be distinguished from
    /// exact zero by a `f64` evaluation on these rows, which is why the arm
    /// certifies zero there instead of chasing the last bits — and why "exact
    /// zero bias" never means "bit-identical constants".
    fn noise(&self, kappa: f64) -> f64 {
        4.0 * kappa.clamp(1.0, SIGNED_NOISE_COND_CAP)
            * f64::EPSILON
            * (self.mag_sum + self.abs_sum)
            / self.weight
    }
}

/// Neumaier compensated summation. The signed objective *is* a sum, so its
/// numerator must not inherit the `n·u·Σ|r|` bound of a naive `f64` sum: with
/// `n = 10⁵` that bound would make "zero" unprovable on data where the true
/// sum is `1e-16`. The residual rounding (`u·(|pred| + |y|)` per row) remains
/// and is what [`SignedSystem::noise`] certifies against.
#[derive(Clone, Copy)]
struct Compensated {
    sum: f64,
    comp: f64,
}

impl Compensated {
    fn new() -> Self {
        Compensated { sum: 0.0, comp: 0.0 }
    }

    fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.comp += (self.sum - t) + v;
        } else {
            self.comp += (v - t) + self.sum;
        }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// Build the [#228] statistics at `c` in one row sweep. `None` means the rows
/// and targets do not match, a weight is negative, or a prediction, residual or
/// derivative was not finite.
#[allow(clippy::too_many_arguments)]
fn signed_system(
    f: &Flat,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    c: &[f64],
    k: usize,
    v: &mut [f64],
    adj: &mut [f64],
    grad: &mut [f64],
) -> Option<SignedSystem> {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return None;
    }
    let mut jtj = vec![vec![0.0; k]; k];
    let mut jtr = vec![0.0; k];
    let mut cgrad = vec![0.0; k];
    let (mut sum_r, mut weight, mut abs_sum) = (Compensated::new(), Compensated::new(), Compensated::new());
    let mut mag_sum = Compensated::new();
    for (i, (row, &t)) in x.iter().zip(y).enumerate() {
        let sw = w.get(i).copied().unwrap_or(1.0);
        if !sw.is_finite() || sw < 0.0 {
            return None;
        }
        if sw == 0.0 {
            continue; // a zero-weight row never influences the fit or the bias
        }
        eval_row(f, row, c, v);
        let pred = v[f.root];
        let r = pred - t;
        if !pred.is_finite() || !r.is_finite() {
            return None;
        }
        backward(f, v, adj, grad);
        let jw = sw.sqrt();
        sum_r.add(sw * r);
        abs_sum.add((sw * r).abs());
        weight.add(sw);
        mag_sum.add(sw * (pred.abs() + t.abs()));
        for a in 0..k {
            let gi = grad[a];
            if !gi.is_finite() {
                return None;
            }
            cgrad[a] += sw * gi;
            jtr[a] += jw * gi * (jw * r);
            for j in 0..=a {
                jtj[a][j] += jw * gi * (jw * grad[j]);
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            jtj[j][i] = jtj[i][j];
        }
    }
    let (sum_r, weight, abs_sum, mag_sum) =
        (sum_r.value(), weight.value(), abs_sum.value(), mag_sum.value());
    if !(weight > 0.0) || !sum_r.is_finite() || !abs_sum.is_finite() || !mag_sum.is_finite() {
        return None;
    }
    Some(SignedSystem {
        jtj,
        jtr,
        grad: cgrad,
        sum_r,
        weight,
        abs_sum,
        mag_sum,
    })
}

/// [SignedProjection::IllConditioned] threshold: the smallest Cholesky pivot
/// relative to its own diagonal, `min Lᵢᵢ² / (JᵀWJ)ᵢᵢ`. It is scale-free (a
/// column rescaling cannot change it) and it is the same pre-registered rank
/// tolerance the closed-form linear arm uses (`LINEAR_RANK_TOL`). A design whose
/// trailing pivot falls below it is rank-deficient, so the constrained step
/// would be amplified rounding and is declined. The GPU fit kernel gates the
/// *ratio* of the Cholesky diagonal (`1e4`) instead, because it runs in `f32`
/// rows and needs a wider margin.
const SIGNED_RANK_TOL: f64 = 1e-12;

/// [SignedProjection::Projected] stops scaling its rounding floor with the
/// design conditioning beyond this factor. A design whose conditioning is worse
/// is still fitted (the step is exact in exact arithmetic), but the arm does not
/// pretend that a bias of `1e-6` is "exact zero": the outcome is then
/// `not_converged`.
const SIGNED_NOISE_COND_CAP: f64 = 1e6;

/// #228: why a signed-bias projection did or did not move the constants.
///
/// The labels are stable: they are counted by [`signed_fit_counts`], printed by
/// `--fit-report` and documented in `docs/optimisation.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignedProjection {
    /// The constants are on the zero-bias manifold, or were moved onto it: the
    /// exact objective is at or below its `f64` rounding scale.
    Projected,
    /// The model cannot represent the weighted mean residual (`Σ w g = 0` or
    /// `gradᵀ M⁻¹ grad = 0`): no constant moves the bias, so no projection
    /// exists and the starting constants are kept unchanged.
    Unidentifiable,
    /// The rounds ran but the exact objective did not reach its rounding scale
    /// (a nonlinear tree, or a clamp); any improvement that was proved is kept.
    NotConverged,
    /// The weighted normal matrix failed the conditioning gate.
    IllConditioned,
    /// A non-finite weight, prediction, residual or derivative, or a
    /// length/shape mismatch between rows, targets and weights.
    NonFinite,
    /// The tree has no constants to fit.
    NoConstants,
}

impl SignedProjection {
    /// Stable label for `--fit-report` and the docs.
    pub fn label(self) -> &'static str {
        match self {
            SignedProjection::Projected => "projected",
            SignedProjection::Unidentifiable => "unidentifiable",
            SignedProjection::NotConverged => "not_converged",
            SignedProjection::IllConditioned => "ill_conditioned",
            SignedProjection::NonFinite => "non_finite",
            SignedProjection::NoConstants => "no_constants",
        }
    }

    /// Every outcome, in reporting order.
    pub const ALL: [SignedProjection; 6] = [
        SignedProjection::Projected,
        SignedProjection::Unidentifiable,
        SignedProjection::NotConverged,
        SignedProjection::IllConditioned,
        SignedProjection::NonFinite,
        SignedProjection::NoConstants,
    ];

    /// Whether the constants were proved to sit on the zero-bias manifold.
    pub fn is_projected(self) -> bool {
        self == SignedProjection::Projected
    }
}

/// #228: process-wide counts of signed-bias projection outcomes, since process
/// start. Diagnostics only; no search decision reads them.
pub fn signed_fit_counts() -> [(&'static str, u64); 6] {
    use std::sync::atomic::Ordering;
    let mut out = [("", 0); 6];
    for (slot, outcome) in out.iter_mut().zip(SignedProjection::ALL) {
        *slot = (outcome.label(), SIGNED_COUNTS[outcome as usize].load(Ordering::Relaxed));
    }
    out
}

// #228: one counter per `SignedProjection` variant, indexed by its discriminant
// order (`Projected` = 0 … `NoConstants` = 5). Diagnostics only.
static SIGNED_COUNTS: [std::sync::atomic::AtomicU64; 6] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn note_signed_projection(outcome: SignedProjection) {
    SIGNED_COUNTS[outcome as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Undamped Cholesky factorisation of a symmetric positive-definite `a`
/// (`k × k` row-major). `None` on a non-positive or non-finite pivot.
fn chol(a: &[Vec<f64>], k: usize) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i][j];
            for p in 0..j {
                s -= l[i][p] * l[j][p];
            }
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L Lᵀ x = b` in place for a Cholesky factor `l`.
fn chol_solve(l: &[Vec<f64>], b: &mut [f64], k: usize) {
    for i in 0..k {
        let mut s = b[i];
        for p in 0..i {
            s -= l[i][p] * b[p];
        }
        b[i] = s / l[i][i];
    }
    for i in (0..k).rev() {
        let mut s = b[i];
        for p in (i + 1)..k {
            s -= l[p][i] * b[p];
        }
        b[i] = s / l[i][i];
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// #228: move the constants onto the zero-mean-residual manifold by a
/// **constrained least-squares** step, and report why if they cannot be.
///
/// For the affine residual model `r(c + δ) = r(c) + J δ` the problem is
///
/// ```text
/// min_δ  || W^{1/2} (r + J δ) ||²      subject to   Σ wᵢ (rᵢ + gᵢᵀδ) = 0
/// ```
///
/// whose KKT system is the augmented saddle-point problem
///
/// ```text
/// [ JᵀWJ   cgrad ] [ δ ]   [ −JᵀWr ]
/// [ cgradᵀ   0   ] [ μ ] = [ −sum_r ]
/// ```
///
/// This solves it exactly as written, with two right-hand sides on the *same*
/// SPD normal matrix (the standard constrained-LS reduction, Scott & Tůma):
/// `δ = δ₀ − μ·M⁻¹cgrad`, `δ₀ = −M⁻¹JᵀWr`, `μ = (sum_r + cgradᵀδ₀) / cgradᵀM⁻¹cgrad`.
/// That is one assembly pass plus one factorisation and two substitutions —
/// two passes, no generic optimiser and no step-size search.
///
/// The geometry is why this, and not a descent rule, is the arm: `|mean
/// residual|` is a norm of an affine functional of `r` whose optimal set is the
/// codimension-1 manifold `Σ wᵢ rᵢ = 0`, all of it at value 0. A rule that
/// accepts only strict improvement halts at the first manifold point and says
/// nothing about where on the manifold to stop. The constrained LS says it:
/// among zero-bias constants it returns the one closest to the squared fit in
/// the `JᵀWJ` metric, i.e. the minimum weighted SSE point of the manifold
/// (exact for trees affine in their constants, since the squared fit is the
/// unconstrained minimiser; local otherwise, as the CPU polish is).
///
/// A trial is written back only when the exact objective `|Σ w r| / Σ w` does
/// not increase, so the arm can never make the configured loss worse. It stops
/// as soon as the objective is at or below its `f64` rounding scale (the
/// reported objective is then exactly what `metrics::score_weighted` returns).
/// `rounds` is the number of Newton rounds on the linearised constraint: one
/// is exact for affine trees, and at most two are used.
pub fn signed_projection(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rounds: usize,
) -> SignedProjection {
    let outcome = signed_projection_inner(tree, x, y, w, rounds);
    note_signed_projection(outcome);
    outcome
}

fn signed_projection_inner(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    rounds: usize,
) -> SignedProjection {
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || x.is_empty() {
        return SignedProjection::NoConstants;
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
    let Some(start) = signed_system(&f, x, y, w, &c, k, &mut v, &mut adj, &mut grad) else {
        return SignedProjection::NonFinite;
    };
    // Already on the manifold relative to the unconditional floor.
    let mut best = start.objective();
    if best <= start.noise(1.0) {
        // Already on the manifold: the squared fit is the minimum-SSE point of
        // it, so there is nothing to project and the constants stay untouched.
        return SignedProjection::Projected;
    }
    let mut system = Some(start);
    let mut moved = false;
    for _ in 0..rounds.max(1) {
        let sys = match system.take() {
            Some(sys) => sys,
            None => match signed_system(&f, x, y, w, &c, k, &mut v, &mut adj, &mut grad) {
                Some(sys) => sys,
                None => return SignedProjection::NonFinite,
            },
        };
        // The constraint row must be able to move the mean at all.
        if sys.grad.iter().all(|&g| g == 0.0) {
            return SignedProjection::Unidentifiable;
        }
        let Some(l) = chol(&sys.jtj, k) else {
            return SignedProjection::IllConditioned;
        };
        // Scale-free rank test and conditioning estimate: the ratio of the
        // largest to the smallest scaled Cholesky pivot is a lower bound on the
        // condition number of the normal matrix.
        let (mut lo_scaled, mut hi_scaled) = (f64::INFINITY, 0.0f64);
        for i in 0..k {
            let pivot = l[i][i] * l[i][i];
            let diag = sys.jtj[i][i];
            if !(pivot.is_finite()) || !(diag > 0.0) {
                return SignedProjection::IllConditioned;
            }
            let scaled = pivot / diag;
            lo_scaled = lo_scaled.min(scaled);
            hi_scaled = hi_scaled.max(scaled);
        }
        if lo_scaled < SIGNED_RANK_TOL {
            return SignedProjection::IllConditioned;
        }
        let kappa = (hi_scaled / lo_scaled).max(1.0);
        let mut step = sys.jtr.iter().map(|v| -v).collect::<Vec<f64>>();
        chol_solve(&l, &mut step, k);
        let mut dir = sys.grad.clone();
        chol_solve(&l, &mut dir, k);
        // `cgradᵀ M⁻¹ cgrad` is a squared norm, so it is non-negative and zero
        // exactly when the constraint row is outside the range of `M`.
        let cn = dot(&sys.grad, &dir);
        if !(cn > 0.0) || !cn.is_finite() {
            return SignedProjection::Unidentifiable;
        }
        let mu = (sys.sum_r + dot(&sys.grad, &step)) / cn;
        if !mu.is_finite() {
            return SignedProjection::NonFinite;
        }
        let trial: Vec<f64> = (0..k)
            .map(|i| (c[i] + step[i] - mu * dir[i]).clamp(-1e6, 1e6))
            .collect();
        let Some(trial_system) =
            signed_system(&f, x, y, w, &trial, k, &mut v, &mut adj, &mut grad)
        else {
            return SignedProjection::NonFinite;
        };
        let trial_objective = trial_system.objective();
        if !(trial_objective <= best) {
            // The linearisation is no longer a descent direction (a clamp, or a
            // nonlinear tree): keep the best point proved so far.
            break;
        }
        let improved = trial_objective < best;
        c = trial;
        best = trial_objective;
        moved |= improved;
        if best <= trial_system.noise(kappa) {
            if improved {
                write_constants(tree, &f, &c);
            }
            return SignedProjection::Projected;
        }
        if !improved {
            break;
        }
        system = Some(trial_system);
    }
    // Not on the manifold after the round budget. Any proved improvement is
    // kept (it never makes the configured objective worse) and the outcome says
    // the exact zero was not certified.
    if moved {
        write_constants(tree, &f, &c);
    }
    SignedProjection::NotConverged
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
/// * `correlation`, `correlation_signed` — the exact analytic gradient and
///   Gauss–Newton curvature of the Pearson correlation, with a frozen affine
///   gauge (see [`correlation`]). No squared pre-fit is involved.
/// * `rank_correlation`, `signed`, `median`, `maximum`, `interquartile` and
///   `slope` — a squared fit followed by a deterministic coordinate polish on
///   the configured metric. These losses have no least-squares normal equations
///   (rank and order statistics are not differentiable); the polish is the
///   documented fallback, and it now runs on a clone so it can never replace a
///   better starting point.
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
        // #289: Brier and the SVM hinge forms have exact **scores** but no
        // exact least-squares/IRLS weight on the CPU. The logistic IRLS step is
        // a warm start and the configured metric decides whether it is kept, so
        // the arm never claims to minimise a different objective: the returned
        // value is always `metric_score` of the configured metric.
        Auc
        | LogLoss
        | Brier
        | Hinge
        | HingeOrdinary
        | HingeSquared => logistic_irls(tree, x, y, w, iters, metric),
        // #224: correlation has an exact objective, an analytic gradient and a
        // structural gauge handler; it never falls back to a squared fit.
        Correlation | CorrelationSigned => {
            correlation::fit(tree, x, y, w, iters, metric)
        }
        // The remaining smooth-but-not-least-squares and non-smooth order/rank
        // losses: start from the squared fit, then polish the actual objective.
        // The squared pre-fit runs on a clone, so a polish that finds no
        // improvement keeps the incoming constants (metric-monotone refit).
        RankCorrelation
        | RankCorrelationSigned
        | Median
        | Maximum
        | Interquartile
        | Slope
        | ImplicitResidual
        | ImplicitDerivative => {
            let start = metric_score(tree, x, y, w, metric);
            let mut trial = tree.clone();
            fit_weighted(&mut trial, x, y, w, iters);
            coordinate_polish(&mut trial, x, y, w, metric, iters.max(1));
            let score = metric_score(&trial, x, y, w, metric);
            if score.is_finite() && score < start {
                *tree = trial;
                score
            } else {
                start
            }
        }
        // #228: signed bias is the one non-squared family with an exact
        // closed-form arm. The squared fit is the warm start; the arm is the
        // constrained-LS projection onto the zero-mean-residual manifold
        // (`signed_projection`), which is two passes and never a generic
        // optimiser. When the projection declines without moving the constants
        // (the model cannot represent the mean, a rank-deficient design, or a
        // non-finite row) the historical coordinate polish runs, exactly as
        // before, so no candidate loses the old behaviour. When it *did* move
        // the constants the arm stops there: the polish's fixed geometric
        // probes cost `48p` full row sweeps and its strict-improvement rule is
        // exactly the rule that cannot select a point on the zero-bias
        // manifold.
        Signed => {
            fit_weighted(tree, x, y, w, iters);
            let before = metric_score(tree, x, y, w, metric);
            let outcome = signed_projection(tree, x, y, w, 2);
            let after = metric_score(tree, x, y, w, metric);
            let moved = after.is_finite() && before.is_finite() && after < before;
            if !outcome.is_projected() && !moved {
                coordinate_polish(tree, x, y, w, metric, iters.max(1));
            }
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

/// #234: the label sign `t = 2·[y > (min y + max y)/2] − 1` the log-loss fit
/// optimises against.
///
/// This is exactly the rule [`logistic_irls`] uses, computed from the full
/// sample and uploaded once to the device (`GpuEvaluator::set_fit_sign`), so
/// the device row weights `q(1−q)` are built from the same margin the CPU arm
/// uses. It is subset-independent for a two-valued target, and a single-valued
/// target degenerates to all `−1` — the caller must refuse that case (see
/// [`class_counts`]) instead of fitting a sign no data supports.
pub fn classification_signs(y: &[f64]) -> Vec<f64> {
    classification_labels(y)
        .into_iter()
        .map(|l| 2.0 * l - 1.0)
        .collect()
}

/// #289: the label sign under the **weighted** midpoint rule.
///
/// The midpoint is taken over rows with `w > 0` only, matching
/// [`crate::metrics::score_weighted`]'s weighted classification branch: a
/// zero-weight row is invisible, so a wild or masked target in it cannot move
/// the label of a row that counts. Masked rows get `-1`, which no scorer or
/// fitter reads because it skips `w = 0` before the row.
pub fn classification_signs_weighted(y: &[f64], w: &[f64]) -> Vec<f64> {
    if w.is_empty() {
        return classification_signs(y);
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&v, &wi) in y.iter().zip(w) {
        if wi > 0.0 {
            if !v.is_finite() {
                // A positive-weight non-finite target is invalid, not a label.
                // NaN signs make the device reject the row instead of silently
                // assigning it to a class the CPU scorer refuses.
                return vec![f64::NAN; y.len()];
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let mid = 0.5 * (lo + hi);
    y.iter()
        .zip(w)
        .map(|(&v, &wi)| if wi > 0.0 && v > mid { 1.0 } else { -1.0 })
        .collect()
}

/// #289: `(positives, negatives)` among rows with `w > 0`.
pub fn class_counts_weighted(y: &[f64], w: &[f64]) -> (usize, usize) {
    if !w.is_empty()
        && y.iter()
            .zip(w)
            .any(|(&v, &wi)| wi > 0.0 && !v.is_finite())
    {
        // No valid labels: the caller labels this sample `single_class` and
        // keeps the exact CPU arm rather than fitting a NaN sign.
        return (0, 0);
    }
    let signs = classification_signs_weighted(y, w);
    let pos = if w.is_empty() {
        signs.iter().filter(|&&t| t > 0.0).count()
    } else {
        signs
            .iter()
            .zip(w)
            .filter(|(&t, &wi)| wi > 0.0 && t > 0.0)
            .count()
    };
    let total = if w.is_empty() {
        y.len()
    } else {
        w.iter().filter(|&&v| v > 0.0).count()
    };
    (pos, total - pos)
}

/// #234: `(positives, negatives)` under the [`classification_signs`] rule.
///
/// A log-loss fit needs both classes: with one class the unconstrained
/// likelihood has no finite minimiser and the fit only drives the logit to the
/// clamp, which is not a calibration. The arm reports the counts so the search
/// can label a single-class candidate set instead of scoring it.
pub fn class_counts(y: &[f64]) -> (usize, usize) {
    let signs = classification_signs(y);
    let pos = signs.iter().filter(|&&t| t > 0.0).count();
    (pos, signs.len() - pos)
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
/// is kept only if the configured metric strictly improves. The **first**
/// improver in that fixed ladder order is committed, never a later or larger
/// one, so the trajectory is reproducible. Sweeps stop early when a full pass
/// makes no improvement. No randomness is involved.
///
/// #230: for `rank_correlation` and `rank_correlation_signed` the trial score is
/// the **exact integer** form from [`metrics::RankTarget`] instead of the f64
/// Pearson-on-ranks value. The two agree to a few ulps; the integer form is
/// order-independent, which is what lets the CUDA rank stage reproduce this
/// ladder constant for constant (see [`rank::polish_rank_batch`]). The target
/// ranks are computed once per call instead of once per trial.
pub(crate) fn coordinate_polish(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    metric: Metric,
    sweeps: usize,
) {
    let signed = rank_metric_signed(metric);
    // The exact path is used only when it is defined (finite target, no
    // weights); otherwise the f64 metric stays the authority, exactly as before.
    let target = match signed {
        Some(_) if w.is_empty() && x.len() == y.len() => metrics::RankTarget::new(y),
        _ => None,
    };
    let score_of = |tree: &Node| match (&target, signed) {
        (Some(t), Some(signed)) => {
            let pred = predictions(tree, x);
            t.score(&pred, signed)
        }
        _ => metric_score(tree, x, y, w, metric),
    };
    let mut best = score_of(tree);
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
            for (mul, sign) in LADDER {
                let trial_value = original + sign * mul * scale;
                if let Some(node) = expr::nth_mut(tree, pos) {
                    node.val = trial_value;
                }
                let score = score_of(tree);
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

/// #230: the tree's constants in `GpuFitPlan` slot order (test/debug helper).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn constants_slots(tree: &Node) -> Vec<f64> {
    let f = flatten(tree);
    let mut c = vec![0.0; f.nconst];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    c
}

/// #230: `Some(signed)` when `metric` is one of the exact-rank objectives.
///
/// `signed` distinguishes `1 − ρ` (`RankCorrelationSigned`) from `1 − |ρ|`
/// (`RankCorrelation`); every other metric returns `None`.
pub(crate) fn rank_metric_signed(metric: Metric) -> Option<bool> {
    match metric {
        Metric::RankCorrelation => Some(false),
        Metric::RankCorrelationSigned => Some(true),
        _ => None,
    }
}

/// #230: the fixed coordinate-polish ladder, in acceptance order.
///
/// `(step fraction, sign)`: `+10%, −10%, +3%, −3%, +1%, −1%` of
/// `max(|c|, 1)`. The CPU polish, the CUDA rank stage and the tests all read
/// this one table, so "the first improver" cannot drift between them.
pub(crate) const LADDER: [(f64, f64); 6] = [
    (0.1, 1.0),
    (0.1, -1.0),
    (0.03, 1.0),
    (0.03, -1.0),
    (0.01, 1.0),
    (0.01, -1.0),
];

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
    /// `JᵀJ` is (near-)singular, or the Jacobian has lost rank: the constants
    /// are not separately determined.
    pub ill_conditioned: bool,
    /// #257: rank of the correlation-scaled Gram matrix of the Jacobian
    /// (eigenvalues above `IDENTIFICATION_TAU` times the largest).
    pub rank: usize,
    /// #257: `sqrt(lambda_max / lambda_min)` of that matrix; infinite when
    /// `lambda_min <= 0`.
    pub condition_index: f64,
    /// #257: constants that carry a rank-loss direction (Belsley variance
    /// decomposition proportion at least `VDP_INVOLVED`); their errors are
    /// infinite.
    pub involved: Vec<usize>,
    /// #257: Driscoll-Kraay sandwich standard errors, when requested.
    pub std_errors_robust: Option<Vec<f64>>,
    pub relative_robust: Option<Vec<f64>>,
}

impl ConstantErrors {
    /// Every constant is separately determined: the Jacobian has full rank.
    pub fn identified(&self) -> bool {
        self.rank == self.values.len()
    }

    /// Practical identification: every robust relative error is at most 1.
    /// `None` without robust errors.
    pub fn practically_identified(&self) -> Option<bool> {
        self.relative_robust
            .as_ref()
            .map(|r| r.iter().all(|v| *v <= 1.0))
    }
}

/// A Gram eigenvalue at or below this fraction of the largest is a lost rank
/// (condition index above 1e4).
pub const IDENTIFICATION_TAU: f64 = 1e-8;
/// Belsley variance-decomposition proportion that attributes a rank-loss
/// direction to a constant.
pub const VDP_INVOLVED: f64 = 0.5;

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi sweeps over
/// `p < q` in row-major order (at most 64 sweeps; stops when the off-diagonal
/// sum of squares is below `1e-30 * trace^2`). Returns the eigenvalues in
/// descending order and the eigenvectors as `v[row][column]`, column `l`
/// belonging to eigenvalue `l`. Deterministic.
pub fn jacobi_eigen_sym(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let k = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0; k]; k];
    for i in 0..k {
        v[i][i] = 1.0;
    }
    let trace: f64 = (0..k).map(|i| m[i][i]).sum();
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += m[p][q] * m[p][q];
            }
        }
        if off <= 1e-30 * trace * trace {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = m[p][q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * apq);
                let t = if theta.abs() > 1e150 {
                    0.5 / theta
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let (arp, arq) = (m[r][p], m[r][q]);
                    m[r][p] = c * arp - s * arq;
                    m[r][q] = s * arp + c * arq;
                }
                for r in 0..k {
                    let (apr, aqr) = (m[p][r], m[q][r]);
                    m[p][r] = c * apr - s * aqr;
                    m[q][r] = s * apr + c * aqr;
                }
                for r in 0..k {
                    let (vrp, vrq) = (v[r][p], v[r][q]);
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| m[y][y].total_cmp(&m[x][x]).then(x.cmp(&y)));
    let vals = order.iter().map(|&i| m[i][i]).collect();
    let vecs = (0..k)
        .map(|r| order.iter().map(|&l| v[r][l]).collect())
        .collect();
    (vals, vecs)
}

struct Ident {
    rank: usize,
    condition_index: f64,
    involved: Vec<usize>,
    /// Pseudo-inverse of the scaled Gram over the kept directions: the finite
    /// errors of the uninvolved constants of a rank-deficient model.
    pinv: Vec<Vec<f64>>,
}

fn identification(corr: &[Vec<f64>]) -> Ident {
    let k = corr.len();
    let (vals, vecs) = jacobi_eigen_sym(corr);
    let lmax = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lmin = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let lost = |l: f64| !(l > IDENTIFICATION_TAU * lmax);
    let rank = vals.iter().filter(|&&l| !lost(l)).count();
    let condition_index = if lmin > 0.0 && lmax > 0.0 {
        (lmax / lmin).sqrt()
    } else {
        f64::INFINITY
    };
    let floor = f64::EPSILON * lmax;
    let mut involved = Vec::new();
    for j in 0..k {
        let phi: Vec<f64> = (0..k).map(|l| vecs[j][l] * vecs[j][l] / vals[l].max(floor)).collect();
        let total: f64 = phi.iter().sum();
        if total > 0.0 && (0..k).any(|l| lost(vals[l]) && phi[l] / total >= VDP_INVOLVED) {
            involved.push(j);
        }
    }
    let mut pinv = vec![vec![0.0; k]; k];
    for l in 0..k {
        if lost(vals[l]) {
            continue;
        }
        for a in 0..k {
            for b in 0..k {
                pinv[a][b] += vecs[a][l] * vecs[b][l] / vals[l];
            }
        }
    }
    Ident { rank, condition_index, involved, pinv }
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
    constant_errors_core(tree, x, y, w, None)
}

/// As [`constant_errors_weighted`], plus Driscoll-Kraay sandwich errors
/// `(JᵀJ)⁻¹ Ω (JᵀJ)⁻¹`, with `Ω` built from per-timestamp sums of
/// `wᵢ Jᵢ rᵢ` (`keys` has one time key per row) and Bartlett lag `lag`.
pub fn constant_errors_robust(
    tree: &Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    keys: &[f64],
    lag: usize,
) -> Option<ConstantErrors> {
    if keys.len() != x.len().min(y.len()) {
        return None;
    }
    constant_errors_core(tree, x, y, w, Some((keys, lag)))
}

fn constant_errors_core(
    tree: &Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    robust: Option<(&[f64], usize)>,
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
    let key_index = robust.map(|(keys, _)| crate::noise::KeyIndex::new(keys));
    let mut group_sums = key_index
        .as_ref()
        .map(|ix| vec![0.0; ix.groups * k])
        .unwrap_or_default();
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
        if let Some(ix) = &key_index {
            let g = ix.group[row_i];
            for j in 0..k {
                group_sums[g * k + j] += sw * grad[j] * r;
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
    let mut flat: Vec<usize> = Vec::new();
    for i in 0..k {
        if jtj[i][i] > 0.0 && jtj[i][i].is_finite() {
            scale[i] = jtj[i][i].sqrt();
        } else {
            flat.push(i);
        }
    }
    if !flat.is_empty() {
        return Some(unidentified(c, dof, flat, None));
    }
    let mut corr = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            corr[i][j] = jtj[i][j] / (scale[i] * scale[j]);
        }
    }
    let ident = identification(&corr);
    let (inv, pivot_ratio) = match invert(&corr, k) {
        Some(r) => r,
        // An exact singularity: a rank-deficient model still reports the
        // pseudo-inverse errors of its uninvolved constants.
        None if ident.rank < k => (ident.pinv.clone(), 0.0),
        None => return Some(unidentified(c, dof, (0..k).collect(), Some(ident))),
    };
    // Below this pivot ratio the columns of `J` are nearly collinear: the
    // constants can trade off and the standard errors are not meaningful.
    const ILL_CONDITIONED: f64 = 1e-12;
    let identified = ident.rank == k;
    // A rank-deficient model has no usable inverse; its uninvolved constants
    // take their errors from the pseudo-inverse over the kept directions.
    let inv = if identified { inv } else { ident.pinv.clone() };
    let ill_conditioned = pivot_ratio < ILL_CONDITIONED || !identified;
    let sigma2 = rss / dof as f64;
    let mut std_errors = vec![0.0; k];
    let mut relative = vec![0.0; k];
    for i in 0..k {
        let cov_ii = sigma2 * inv[i][i] / (scale[i] * scale[i]);
        let se = if ident.involved.contains(&i) {
            f64::INFINITY
        } else if cov_ii >= 0.0 {
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
    let (std_errors_robust, relative_robust) = match (&key_index, robust) {
        (Some(ix), Some((_, lag))) => {
            let omega = crate::noise::dk_cov_from_group_sums(&group_sums, ix.groups, k, lag);
            // cov = D^-1 inv D^-1 Omega D^-1 inv D^-1 in scaled coordinates.
            let os: Vec<Vec<f64>> = (0..k)
                .map(|a| (0..k).map(|b| omega[a][b] / (scale[a] * scale[b])).collect())
                .collect();
            let mut se = vec![0.0; k];
            let mut rel = vec![0.0; k];
            for i in 0..k {
                let mut acc = 0.0;
                for a in 0..k {
                    for b in 0..k {
                        acc += inv[i][a] * os[a][b] * inv[b][i];
                    }
                }
                let cov_ii = acc / (scale[i] * scale[i]);
                se[i] = if ident.involved.contains(&i) || !(cov_ii >= 0.0) {
                    f64::INFINITY
                } else {
                    cov_ii.sqrt()
                };
                rel[i] = if c[i] == 0.0 { f64::INFINITY } else { se[i] / c[i].abs() };
            }
            (Some(se), Some(rel))
        }
        _ => (None, None),
    };
    Some(ConstantErrors {
        values: c,
        std_errors,
        relative,
        dof,
        ill_conditioned,
        rank: ident.rank,
        condition_index: ident.condition_index,
        involved: ident.involved,
        std_errors_robust,
        relative_robust,
    })
}

/// Fallback for an unidentified system: values kept, every error infinite.
fn unidentified(values: Vec<f64>, dof: usize, involved: Vec<usize>, ident: Option<Ident>) -> ConstantErrors {
    let k = values.len();
    let (rank, condition_index) = match ident {
        Some(i) => (i.rank, i.condition_index),
        None => (k - involved.len(), f64::INFINITY),
    };
    ConstantErrors {
        values,
        std_errors: vec![f64::INFINITY; k],
        relative: vec![f64::INFINITY; k],
        dof,
        ill_conditioned: true,
        rank,
        condition_index,
        involved,
        std_errors_robust: None,
        relative_robust: None,
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

    /// #222: for a scale-only model `a·x` on positive `x`, the L1 optimum is
    /// the (base-weight × x)-weighted median of `y_i / x_i` — an exact,
    /// independent oracle for the IRLS arm, not a comparison with another
    /// iteration of the same code.
    #[test]
    fn l1_fit_matches_the_weighted_median_on_a_scale_model() {
        let ratios = [
            2.0, 2.1, 1.9, 2.05, 1.95, 30.0, 2.02, 1.98, -12.0, 2.03, 1.97,
        ];
        let xs = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5];
        let base = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.75, 1.0, 1.0];
        let x: Vec<Vec<f64>> = xs.iter().map(|&v| vec![v]).collect();
        let y: Vec<f64> = xs.iter().zip(&ratios).map(|(x, r)| x * r).collect();
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap());
        let total: f64 = order.iter().map(|&i| base[i] * xs[i]).sum();
        let mut acc = 0.0;
        let mut median = f64::NAN;
        for &i in &order {
            acc += base[i] * xs[i];
            if acc >= 0.5 * total {
                median = ratios[i];
                break;
            }
        }
        let mut tree = node(Op::Mul, vec![constant(1.0), input(0)]);
        let fitted_loss = fit_weighted_metric(&mut tree, &x, &y, &base, 64, Metric::Absolute);
        let a = fitted_constants(&tree)[0];
        assert!(
            (a - median).abs() <= 1e-2 * median.abs().max(1.0),
            "IRLS {a} vs weighted median {median}"
        );
        // The parity claim is on the objective: the IRLS endpoint must not be
        // worse than the exact weighted median of this scale model.
        let pred_at = |c: f64| -> Vec<f64> { x.iter().map(|r| c * r[0]).collect() };
        // Lawson IRLS approaches the kink asymptotically (one weight goes to
        // infinity at the median), so the honest quality gate is that the
        // endpoint is within 0.1 % of the exact oracle, not bit-equal to it.
        let at_median = metrics::score_weighted(&pred_at(median), &y, &base, Metric::Absolute, 1);
        assert!(
            fitted_loss <= at_median * (1.0 + 1e-3),
            "IRLS loss {fitted_loss} worse than the weighted-median oracle {at_median}"
        );
    }

    /// #222: the arm is deterministic (bit for bit) and a non-finite row is
    /// reported as an infinite loss without corrupting the constants.
    #[test]
    fn l1_fit_is_deterministic_and_rejects_non_finite_rows() {
        let (x, y) = outlier_fixture();
        let mut a = line(0.5, 0.0);
        let mut b = line(0.5, 0.0);
        let la = fit_weighted_metric(&mut a, &x, &y, &[], 8, Metric::Absolute);
        let lb = fit_weighted_metric(&mut b, &x, &y, &[], 8, Metric::Absolute);
        assert_eq!(la.to_bits(), lb.to_bits());
        assert_eq!(
            fitted_constants(&a).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fitted_constants(&b).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut bad_y = y.clone();
        bad_y[2] = f64::NAN;
        let mut t = line(0.5, 0.0);
        let loss = fit_weighted_metric(&mut t, &x, &bad_y, &[], 8, Metric::Absolute);
        assert!(!loss.is_finite(), "a non-finite metric score must not be reported as a fit: {loss}");
        assert!(fitted_constants(&t).iter().all(|v| v.is_finite()));
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

    /// #223 + #229 + #230 + #226 + #222 + #227 + #232: an explicit GPU request
    /// is refused for every metric without a device fit objective, and the
    /// refusal says why and what to use instead. `squared`/`aic`/`aicc`/`bic`
    /// fit by least squares, `log_squashed` by weighted IRLS (#229), `absolute`
    /// by L1 IRLS (#222), the rank family by the exact integer polish (#230),
    /// `median`/`interquartile` by the exact order-statistic polish
    /// (#226/#232) and `maximum` by the exact maximum-error polish (#227), so
    /// every metric with a device fit objective is accepted.
    #[test]
    fn explicit_gpu_names_the_gpu_lm_and_a_way_out_for_other_metrics() {
        for m in Metric::all() {
            let why = gpu_fit_metric_error(*m);
            if fit_loss(*m).is_some()
                || rank_fit_only(*m).is_some()
                || order_stat_fit_only(*m)
                || maximum_fit_only(*m)
            {
                assert!(why.is_none(), "{} must stay accepted", m.name());
                continue;
            }
            let why = why.unwrap_or_else(|| panic!("{} must be refused", m.name()));
            assert!(
                why.contains("minimises squared error, log-squashed loss, mean"),
                "the refusal must name the device objectives: {why}"
            );
            assert!(
                why.contains(&format!("--metric {}", m.name())),
                "the refusal must name the metric: {why}"
            );
            assert!(
                why.contains("--metric squared"),
                "the refusal must point at --metric squared: {why}"
            );
            assert!(
                why.contains("--metric log_squashed"),
                "the refusal must point at the robust arm: {why}"
            );
            assert!(
                why.contains("--metric absolute"),
                "the refusal must point at the L1 arm: {why}"
            );
            for strategy in ["exact", "auto", "linear", "varpro", "stream"] {
                assert!(
                    why.contains(strategy),
                    "the refusal must offer the CPU strategy {strategy}: {why}"
                );
            }
        }
        // The robust arm is native since #229 and the rank arm fits since
        // #230: neither may be refused.
        assert!(gpu_fit_metric_error(Metric::LogSquashed).is_none());
        assert!(gpu_fit_metric_error(Metric::Squared).is_none());
        assert!(gpu_fit_metric_error(Metric::RankCorrelation).is_none());
        assert!(gpu_fit_metric_error(Metric::RankCorrelationSigned).is_none());
        assert!(gpu_fit_metric_error(Metric::Absolute).is_none());
        assert_eq!(rank_fit_only(Metric::RankCorrelation), Some(false));
        assert_eq!(rank_fit_only(Metric::RankCorrelationSigned), Some(true));
        assert_eq!(rank_fit_only(Metric::Squared), None);
        // #226: `median` has an exact device polish (fit-only), so an explicit
        // `--fit-strategy gpu` is no longer a usage error for it.
        assert!(gpu_fit_metric_error(Metric::Median).is_none());
        assert!(median_fit_only(Metric::Median));
        assert!(!median_fit_only(Metric::Absolute));
        assert!(!median_fit_only(Metric::RankCorrelation));
        // #232: `interquartile` is the same polish on the middle-50 % window
        // mean, so it is accepted and labelled as its own arm.
        assert!(gpu_fit_metric_error(Metric::Interquartile).is_none());
        assert!(interquartile_fit_only(Metric::Interquartile));
        assert!(!interquartile_fit_only(Metric::Median));
        assert!(order_stat_fit_only(Metric::Median));
        assert!(order_stat_fit_only(Metric::Interquartile));
        assert!(!order_stat_fit_only(Metric::Maximum));
        assert!(!order_stat_fit_only(Metric::Slope));
        assert!(!order_stat_fit_only(Metric::Absolute));
        // #227: `maximum` has an exact device polish (fit-only) as well.
        assert!(gpu_fit_metric_error(Metric::Maximum).is_none());
        assert!(maximum_fit_only(Metric::Maximum));
        assert!(!maximum_fit_only(Metric::Median));
        assert!(!maximum_fit_only(Metric::Absolute));
        assert!(!maximum_fit_only(Metric::Squared));
    }

    /// #227: the maximum cost model, as a decision. Below the row floor the
    /// CPU polish stays; at and above it the device wins for every constant
    /// count the kernel accepts, and the model is monotone in `n`.
    #[test]
    fn the_maximum_cost_model_prefers_the_device_only_above_its_row_floor() {
        assert!(!maximum_gpu_fit_worthwhile(MAXIMUM_GPU_MIN_ROWS - 1, 4, 8));
        for p in 1..=20 {
            assert!(
                maximum_gpu_fit_worthwhile(MAXIMUM_GPU_MIN_ROWS, p, 8),
                "p={p} must be admitted at the floor"
            );
            assert!(
                maximum_gpu_fit_worthwhile(154_000, p, 8),
                "p={p} must be admitted at 154k rows"
            );
        }
        let small = maximum_fit_cost(5_000, 4, 8);
        let large = maximum_fit_cost(154_000, 4, 8);
        assert!(large.speedup() > small.speedup());
        assert!(large.speedup() > 1.0, "the model must show a speed-up at 154k");
        let floor = maximum_fit_cost(MAXIMUM_GPU_MIN_ROWS, 4, 8);
        assert!(floor.prefers_gpu(), "floor cost {floor:?}");
        assert!(floor.cpu_s > 0.0 && floor.gpu_s > 0.0);
        // More constants make the device relatively better, because every
        // coordinate of a candidate rides in the same batched call.
        for n in [MAXIMUM_GPU_MIN_ROWS, 5_000, 154_000] {
            let mut last = 0.0;
            for p in 1..=20 {
                let s = maximum_fit_cost(n, p, 8).speedup();
                assert!(s > last, "n={n}: the speed-up must grow with p ({s} vs {last})");
                last = s;
            }
        }
        // `--no-fit`/one iteration still has a well-defined model, and the
        // device never wins on zero information.
        assert!(maximum_fit_cost(0, 1, 8).cpu_s > 0.0);
    }

    /// #226: the measured cost model, as a decision. Below the row floor the
    /// CPU polish stays; above it the device wins for every constant count the
    /// kernel accepts, and the model is monotone in `n` and in `p`.
    #[test]
    fn the_median_cost_model_prefers_the_device_only_above_its_row_floor() {
        assert!(!median_gpu_fit_worthwhile(MEDIAN_GPU_MIN_ROWS - 1, 4, 8));
        for p in 1..=20 {
            assert!(
                median_gpu_fit_worthwhile(MEDIAN_GPU_MIN_ROWS, p, 8),
                "p={p} must be admitted at the floor"
            );
            assert!(
                median_gpu_fit_worthwhile(154_000, p, 8),
                "p={p} must be admitted at 154k rows"
            );
            assert!(
                median_fit_cost(154_000, p, 8).speedup() > 1.0,
                "p={p}: the model must show a speed-up at 154k rows"
            );
        }
        // Monotone in the sample size for a fixed shape.
        let small = median_fit_cost(5_000, 4, 8);
        let large = median_fit_cost(154_000, 4, 8);
        assert!(large.speedup() > small.speedup());
        // At 4k rows and p=4 the modelled device tail must still beat the CPU,
        // which is what makes the floor the *smallest* admitted count.
        let floor = median_fit_cost(MEDIAN_GPU_MIN_ROWS, 4, 8);
        assert!(floor.prefers_gpu(), "floor cost {floor:?}");
        let below = median_fit_cost(MEDIAN_GPU_MIN_ROWS - 1, 4, 8);
        assert!(!median_gpu_fit_worthwhile(MEDIAN_GPU_MIN_ROWS - 1, 4, 8));
        assert!(below.cpu_s > 0.0 && below.gpu_s > 0.0);
    }

    /// #232: the interquartile cost model is a decision, not a guess. Below the
    /// measured row floor the exact CPU arm stays; at and above it the device
    /// wins for every constant count the kernel accepts, the model is monotone
    /// in `n` and in `p`, and its CPU side is the median arm's per-trial cost
    /// (predict + sort) by construction.
    #[test]
    fn the_interquartile_cost_model_prefers_the_device_only_above_its_row_floor() {
        assert!(!interquartile_gpu_fit_worthwhile(
            INTERQUARTILE_GPU_MIN_ROWS - 1,
            4,
            8
        ));
        for p in 1..=20 {
            assert!(
                interquartile_gpu_fit_worthwhile(INTERQUARTILE_GPU_MIN_ROWS, p, 8),
                "p={p} must be admitted at the floor"
            );
            assert!(
                interquartile_gpu_fit_worthwhile(154_000, p, 8),
                "p={p} must be admitted at 154k rows"
            );
            assert!(
                interquartile_fit_cost(154_000, p, 8).speedup() > 1.0,
                "p={p}: the model must show a speed-up at 154k rows"
            );
        }
        // Monotone in the sample size and in the constant count.
        let small = interquartile_fit_cost(5_000, 4, 8);
        let large = interquartile_fit_cost(154_000, 4, 8);
        assert!(large.speedup() > small.speedup());
        assert!(
            interquartile_fit_cost(154_000, 8, 8).speedup()
                > interquartile_fit_cost(154_000, 1, 8).speedup()
        );
        // The floor itself is admitted and the row below it is not.
        let floor = interquartile_fit_cost(INTERQUARTILE_GPU_MIN_ROWS, 4, 8);
        assert!(floor.prefers_gpu(), "floor cost {floor:?}");
        assert!(floor.cpu_s > 0.0 && floor.gpu_s > 0.0);
        // The CPU side is the median arm's per-trial cost: the two metrics sort
        // the same residual multiset.
        let iqr = interquartile_fit_cost(154_000, 4, 8);
        let med = median_fit_cost(154_000, 4, 8);
        assert_eq!(iqr.cpu_s.to_bits(), med.cpu_s.to_bits());
        // Label and device code are distinct, so the C entry point cannot mix
        // the two statistics up.
        assert_eq!(OrderStat::for_metric(Metric::Interquartile), Some(OrderStat::Interquartile));
        assert_eq!(OrderStat::for_metric(Metric::Median), Some(OrderStat::Median));
        assert_eq!(OrderStat::for_metric(Metric::Absolute), None);
        assert_ne!(
            OrderStat::Interquartile.device_code(),
            OrderStat::Median.device_code()
        );
        assert_eq!(OrderStat::Interquartile.label(), "interquartile");
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

    // ---- #257: structural identification and robust errors ----

    fn wiggle(i: usize, salt: f64) -> f64 {
        ((i as f64 + 1.0) * 12.9898 + salt).sin() * 0.5
    }

    fn spread_rows(n: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..cols).map(|c| (i as f64) * 0.1 + wiggle(i, c as f64 * 3.0) + c as f64).collect())
            .collect()
    }

    #[test]
    fn exact_alias_of_two_offsets_is_rank_deficient() {
        // c0 + c1*(c2 + x): c0 and c1*c2 trade off exactly.
        let x = spread_rows(40, 1);
        let y: Vec<f64> = x.iter().enumerate().map(|(i, r)| 0.3 + 2.0 * (0.5 + r[0]) + 0.05 * wiggle(i, 1.0)).collect();
        let tree = node(
            Op::Add,
            vec![constant(0.3), node(Op::Mul, vec![constant(2.0), node(Op::Add, vec![constant(0.5), input(0)])])],
        );
        let e = constant_errors(&tree, &x, &y).expect("errors");
        assert_eq!(e.rank, 2);
        assert_eq!(e.involved, vec![0, 2]);
        assert!(e.std_errors[0].is_infinite() && e.std_errors[2].is_infinite());
        assert!(e.std_errors[1].is_finite());
        assert!(e.ill_conditioned);
        assert!(!e.identified());
    }

    #[test]
    fn rank_six_shape_is_flagged() {
        // c0 + c1/(c2 + exp(x1)) + c3*(c4 - x2): c0 and c4 alias.
        let x = spread_rows(60, 2);
        let tree = node(
            Op::Add,
            vec![
                node(
                    Op::Add,
                    vec![
                        constant(0.2),
                        node(Op::Div, vec![constant(1.5), node(Op::Add, vec![constant(2.0), node(Op::Exp, vec![input(0)])])]),
                    ],
                ),
                node(Op::Mul, vec![constant(0.7), node(Op::Sub, vec![constant(0.4), input(1)])]),
            ],
        );
        let y: Vec<f64> = x.iter().enumerate().map(|(i, r)| 0.1 * r[1] + 0.02 * wiggle(i, 2.0)).collect();
        let e = constant_errors(&tree, &x, &y).expect("errors");
        assert_eq!(e.rank, 4, "{e:?}");
        assert_eq!(e.involved, vec![0, 4]);
        assert!(e.ill_conditioned);
        assert!(e.std_errors[0].is_infinite() && e.std_errors[4].is_infinite());
    }

    #[test]
    fn near_collinear_but_identified_reports_condition_index() {
        let rho: f64 = 1.0 - 1e-6;
        let s = (1.0 - rho * rho).sqrt();
        let x = vec![vec![1.0, rho], vec![0.0, s], vec![0.0, 0.0]];
        let y = vec![1.0, 2.0, 3.0];
        let tree = node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(1.0), input(0)]), node(Op::Mul, vec![constant(1.0), input(1)])],
        );
        let e = constant_errors(&tree, &x, &y).expect("errors");
        assert!(e.identified() && e.involved.is_empty());
        let dot: f64 = x.iter().map(|r| r[0] * r[1]).sum();
        let n0: f64 = x.iter().map(|r| r[0] * r[0]).sum();
        let n1: f64 = x.iter().map(|r| r[1] * r[1]).sum();
        let r = dot / (n0 * n1).sqrt();
        let want = ((1.0 + r) / (1.0 - r)).sqrt();
        assert!(((e.condition_index - want) / want).abs() < 1e-9, "{} {}", e.condition_index, want);
        assert!((e.condition_index - 1414.2132).abs() < 1e-3, "{}", e.condition_index);
    }

    #[test]
    fn jacobi_eigen_matches_closed_form_2x2_and_is_deterministic() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = jacobi_eigen_sym(&a);
        assert!((vals[0] - 3.0).abs() < 1e-14 && (vals[1] - 1.0).abs() < 1e-14);
        let norm: f64 = vecs[0][0] * vecs[0][0] + vecs[1][0] * vecs[1][0];
        assert!((norm - 1.0).abs() < 1e-14);
        let b = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 1.0]];
        let (v1, e1) = jacobi_eigen_sym(&b);
        let (v2, e2) = jacobi_eigen_sym(&b);
        for (p, q) in v1.iter().zip(&v2) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
        for (r1, r2) in e1.iter().zip(&e2) {
            for (p, q) in r1.iter().zip(r2) {
                assert_eq!(p.to_bits(), q.to_bits());
            }
        }
        let trace: f64 = v1.iter().sum();
        assert!((trace - 8.0).abs() < 1e-13);
    }

    #[test]
    fn full_rank_std_errors_are_bit_identical() {
        // Reference: the Gauss-Jordan route the function has always used.
        let x = rows(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let noise = [0.01, -0.015, 0.008, -0.012, 0.02, -0.005, 0.006, -0.02, 0.011, -0.003];
        let y: Vec<f64> = x.iter().zip(noise).map(|(r, e)| 2.0 * r[0] + 1.0 + e).collect();
        let (a, b) = (1.9, 1.1);
        let tree = node(Op::Add, vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)]);
        let e = constant_errors(&tree, &x, &y).expect("errors");
        let mut jtj = vec![vec![0.0; 2]; 2];
        let mut rss = 0.0;
        for (r, t) in x.iter().zip(&y) {
            let g = [r[0], 1.0];
            for i in 0..2 {
                for j in 0..=i {
                    jtj[i][j] += g[i] * g[j];
                }
            }
            let res = a * r[0] + b - t;
            rss += res * res;
        }
        jtj[0][1] = jtj[1][0];
        let sc = [jtj[0][0].sqrt(), jtj[1][1].sqrt()];
        let corr: Vec<Vec<f64>> = (0..2).map(|i| (0..2).map(|j| jtj[i][j] / (sc[i] * sc[j])).collect()).collect();
        let (inv, _) = invert(&corr, 2).unwrap();
        let sigma2 = rss / (y.len() - 2) as f64;
        for i in 0..2 {
            let want = (sigma2 * inv[i][i] / (sc[i] * sc[i])).sqrt();
            assert_eq!(e.std_errors[i].to_bits(), want.to_bits());
        }
        assert!(e.identified() && !e.ill_conditioned);
    }

    #[test]
    fn duplicating_rows_within_a_timestamp_keeps_robust_se() {
        let n = 30;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.3 + wiggle(i, 0.0)]).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, r)| 2.0 * r[0] + 1.0 + 0.3 * wiggle(i, 5.0)).collect();
        let keys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let tree = node(Op::Add, vec![node(Op::Mul, vec![constant(2.0), input(0)]), constant(1.0)]);
        let single = constant_errors_robust(&tree, &x, &y, &[], &keys, 2).unwrap();
        let x2: Vec<Vec<f64>> = x.iter().flat_map(|r| [r.clone(), r.clone()]).collect();
        let y2: Vec<f64> = y.iter().flat_map(|v| [*v, *v]).collect();
        let k2: Vec<f64> = keys.iter().flat_map(|v| [*v, *v]).collect();
        let double = constant_errors_robust(&tree, &x2, &y2, &[], &k2, 2).unwrap();
        let (r1, r2) = (single.std_errors_robust.unwrap(), double.std_errors_robust.unwrap());
        for i in 0..2 {
            assert!(((r1[i] - r2[i]) / r1[i]).abs() < 1e-12, "{} {}", r1[i], r2[i]);
            // iid: sigma^2 uses n - k, so the exact factor is sqrt((n-k)/(2n-k)).
            let factor = (((n - 2) as f64) / ((2 * n - 2) as f64)).sqrt();
            let want = single.std_errors[i] * factor;
            assert!(((double.std_errors[i] - want) / want).abs() < 1e-12);
        }
    }
}
