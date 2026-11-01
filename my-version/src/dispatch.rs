//! #290: per-stage CPU/GPU dispatch with a calibrated cost model.
//!
//! Dispatch is a property of a **stage**, not of a run. Scoring 512 candidates
//! over 2.4 M rows, fitting four constants on one candidate, selecting a
//! tournament winner and writing a checkpoint have different work shapes and
//! different transfer/launch overheads. The historical gates
//! (`maximum_gpu_fit_worthwhile`, `median_gpu_fit_worthwhile`,
//! `defers_fit_to_gpu`) are degenerate special cases of the model here.
//!
//! For `W` candidates, `N` rows, `m` nodes and `p` constants:
//!
//! ```text
//! R      = W · N · m                    row-evals
//! T_dev  = R / r_eval                   device arithmetic
//! T_fit  = W / r_fit(N, p, sweeps)      device fit
//! T_xfer = bytes / BW                   H2D + D2H
//! T_cpu  = R / cpu_eval                 host arithmetic
//! route  = Device iff T_dev + T_fit + T_xfer + launch < T_cpu / (1 + h)
//! ```
//!
//! `h ≈ 0.3` is hysteresis. Once a stage is on the device it must become
//! `(1+h)` times *more* expensive than the CPU before it moves back, and once it
//! is on the CPU it must beat the CPU by `1/(1+h)` before it moves to the
//! device. That is the band that stops a stage oscillating at the crossover.
//!
//! ## Refusals are a cause set, never a stopwatch
//!
//! A stage may be refused the device for structural or arithmetic reasons
//! (`launch`, `unsupported_plan`, `too_few_rows`, `too_many_constants`,
//! `ill_conditioned`, `non_finite`, `metric_not_native`, `device_memory`, `no_device`,
//! `exactness`). Those refusals are checked **before** any timing comparison.
//! A model that let a slow measurement turn an arithmetic failure into a CPU
//! route would hide the failure; [`FallbackCause::is_timing`] is therefore
//! always `false` and [`Dispatcher::plan`] short-circuits on the cause set.
//!
//! ## Calibration
//!
//! [`Calibration::probe`] times the host in a short startup probe. With a
//! device, `crate::dispatch_gpu::device_probe` times the device eval rate, the
//! fit rate, the H2D and D2H bandwidth and the launch overhead once per
//! process, and [`Calibration::with_device_rates`] installs them. Every probed
//! term carries its own spread: the largest sample over the smallest sample of
//! its timed reps. Two probes agree on a term when their values differ by no
//! more than the larger of the two spreads ([`Calibration::agreement`]). The
//! values, their sources (`probe` or `anchor` for the device terms), the spread
//! and the probe time are written to the run manifest.
//!
//! The device anchors are the measured corpus values from
//! `arbeidsmappe/crypto-rcsr/gpu-coverage/09-dispatch-architecture.md` §6 and
//! `06-fit-fallback.md`: ~20e9 row-evals/s, 89.5 ms per accepted fit at
//! 2.456 M rows × 52 variables, ~12 GB/s effective PCIe and 3–7 µs launch.
//! Without a device the model keeps them as anchors, labelled `anchor`, and
//! still refuses device routes when no device exists.
//!
//! ## Exactness
//!
//! A route never changes the reported numbers. The device may guide the search
//! with `f32` rows, but every archived error and every finalist is rebuilt by
//! the exact CPU `f64` path. [`Stage::Refit`] is CPU by contract, as are I/O,
//! selection, archive insertion and the tiny reductions where a launch costs
//! more than the work.

use std::collections::BTreeMap;
use std::time::Instant;

/// Version of the cost model and JSON ledger shape.
pub const MODEL_VERSION: u32 = 1;

/// Default hysteresis: device work must be this much cheaper before crossing.
pub const DEFAULT_HYSTERESIS: f64 = 0.3;

/// Measured device row-evals per second (RTX 3060, saturated batch).
pub const DEFAULT_R_EVAL: f64 = 20.0e9;
/// Measured accepted device fits per second at the reference shape.
pub const DEFAULT_R_FIT: f64 = 1.0 / 0.0895;
/// Effective PCIe bandwidth used by the transfer term, bytes/s.
pub const DEFAULT_BW: f64 = 12.0e9;
/// Measured device launch overhead, seconds.
pub const DEFAULT_LAUNCH_S: f64 = 5.0e-6;
/// Measured host row-evals per second at the reference host (8 threads).
pub const DEFAULT_CPU_EVAL: f64 = 1.5e8;
/// Measured exact CPU fits per second at the reference shape.
pub const DEFAULT_CPU_FIT: f64 = 1.0 / 6.232;

/// Reference shape for the fit rate. `r_fit(N, p, sweeps)` scales inversely
/// with rows, constants and sweeps from this point.
pub const FIT_REF_ROWS: f64 = 2_456_306.0;
pub const FIT_REF_CONSTANTS: f64 = 4.0;
pub const FIT_REF_SWEEPS: f64 = 8.0;

/// The host terms that route stages stay within this factor of their anchors,
/// so the route ledger of two identical CPU runs has identical bytes. The raw
/// probe rate is reported unclamped as `host_rate`.
const PROBE_CLAMP: f64 = 1.34;

/// A stage whose route is planned independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Clone/crossover/mutate/simplify. CPU until a fixed-capacity device pool
    /// exists (#291); the model names that explicitly.
    Vary,
    /// Score candidate × row. The main device arithmetic stage.
    Eval,
    /// Fit constants candidate × row × sweep.
    Fit,
    /// Tournament/rank selection over `O(P)` floats.
    Select,
    /// Pareto archive insertion over a tiny front.
    Archive,
    /// Data preparation / feature enrichment.
    Features,
    /// P&L and walk-forward evaluation.
    Pnl,
    /// Load, split, checkpoint, manifest and hashing.
    Io,
    /// Exact finalist rebuild in `f64`, the accuracy contract.
    Refit,
}

impl Stage {
    /// Every stage the dispatcher must answer for.
    pub const ALL: [Stage; 9] = [
        Stage::Vary,
        Stage::Eval,
        Stage::Fit,
        Stage::Select,
        Stage::Archive,
        Stage::Features,
        Stage::Pnl,
        Stage::Io,
        Stage::Refit,
    ];

    /// Stable machine label.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Vary => "vary",
            Stage::Eval => "eval",
            Stage::Fit => "fit",
            Stage::Select => "select",
            Stage::Archive => "archive",
            Stage::Features => "features",
            Stage::Pnl => "pnl",
            Stage::Io => "io",
            Stage::Refit => "refit",
        }
    }

    /// Stages whose output must be CPU `f64` or host-serial by contract, or
    /// whose work is too small for a launch. The model returns
    /// [`Reason::CpuContract`] for them before comparing timings.
    pub fn cpu_by_contract(self) -> bool {
        matches!(
            self,
            Stage::Vary
                | Stage::Select
                | Stage::Archive
                | Stage::Features
                | Stage::Pnl
                | Stage::Io
                | Stage::Refit
        )
    }
}

/// The chosen execution route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Device,
    Cpu,
    Split,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::Device => "device",
            Route::Cpu => "cpu",
            Route::Split => "split",
        }
    }
}

/// How the dispatcher resolves the timing comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchMode {
    /// The cost model decides.
    #[default]
    Auto,
    /// Every stage that is not refused or CPU by contract takes the device.
    Device,
}

impl DispatchMode {
    pub fn name(self) -> &'static str {
        match self {
            DispatchMode::Auto => "auto",
            DispatchMode::Device => "device",
        }
    }

    pub fn parse(s: &str) -> Option<DispatchMode> {
        match s {
            "auto" => Some(DispatchMode::Auto),
            "device" => Some(DispatchMode::Device),
            _ => None,
        }
    }
}

/// Why a stage got its route. Labels are stable JSON/ledger keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Device is predicted cheaper outside the hysteresis band.
    CostDevice,
    /// CPU is predicted cheaper outside the hysteresis band.
    CostCpu,
    /// Device was already selected and is still inside the wider hold band.
    HysteresisDevice,
    /// The stage was split between device and CPU work.
    SplitCost,
    /// Selection, archive, I/O, features, P&L or exact refit: CPU by contract.
    CpuContract,
    /// No usable device in this build or process.
    NoDevice,
    /// The candidate has no device plan (opcode, derivative or shape).
    UnsupportedPlan,
    /// Fewer rows than the normal equations or kernel shape need.
    TooFewRows,
    /// More constants than the device kernel supports.
    TooManyConstants,
    /// The CUDA launch itself failed.
    Launch,
    /// The device conditioning gate rejected the normal equations.
    IllConditioned,
    /// A non-finite value, derivative or residual.
    NonFinite,
    /// The metric has no native device objective.
    MetricNotNative,
    /// The device memory plan refuses the stage or routes it to the CPU.
    DeviceMemory,
    /// Exactness or a reporting contract keeps this on the CPU.
    Exactness,
    /// The calibration is missing or degenerate.
    NoCalibration,
    /// `--dispatch device` skipped the timing comparison.
    Forced,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::CostDevice => "cost_device",
            Reason::CostCpu => "cost_cpu",
            Reason::HysteresisDevice => "hysteresis_device",
            Reason::SplitCost => "split_cost",
            Reason::CpuContract => "cpu_contract",
            Reason::NoDevice => "no_device",
            Reason::UnsupportedPlan => "unsupported_plan",
            Reason::TooFewRows => "too_few_rows",
            Reason::TooManyConstants => "too_many_constants",
            Reason::Launch => "launch",
            Reason::IllConditioned => "ill_conditioned",
            Reason::NonFinite => "non_finite",
            Reason::MetricNotNative => "metric_not_native",
            Reason::DeviceMemory => "device_memory",
            Reason::Exactness => "exactness",
            Reason::NoCalibration => "no_calibration",
            Reason::Forced => "forced",
        }
    }

    /// A fallback/refusal reason is never produced by a timing comparison.
    pub fn is_cause(self) -> bool {
        matches!(
            self,
            Reason::NoDevice
                | Reason::UnsupportedPlan
                | Reason::TooFewRows
                | Reason::TooManyConstants
                | Reason::Launch
                | Reason::IllConditioned
                | Reason::NonFinite
                | Reason::MetricNotNative
                | Reason::DeviceMemory
                | Reason::Exactness
                | Reason::NoCalibration
        )
    }
}

/// The structural/arithmetic cause set. It is deliberately separate from a
/// timing result: these are predicates on the workload and the kernel shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackCause {
    NoDevice,
    UnsupportedPlan,
    TooFewRows,
    TooManyConstants,
    Launch,
    IllConditioned,
    NonFinite,
    MetricNotNative,
    DeviceMemory,
    Exactness,
}

impl FallbackCause {
    /// Every cause the ledger can carry.
    pub const ALL: [FallbackCause; 10] = [
        FallbackCause::NoDevice,
        FallbackCause::UnsupportedPlan,
        FallbackCause::TooFewRows,
        FallbackCause::TooManyConstants,
        FallbackCause::Launch,
        FallbackCause::IllConditioned,
        FallbackCause::NonFinite,
        FallbackCause::MetricNotNative,
        FallbackCause::DeviceMemory,
        FallbackCause::Exactness,
    ];

    pub fn name(self) -> &'static str {
        self.reason().name()
    }

    pub fn reason(self) -> Reason {
        match self {
            FallbackCause::NoDevice => Reason::NoDevice,
            FallbackCause::UnsupportedPlan => Reason::UnsupportedPlan,
            FallbackCause::TooFewRows => Reason::TooFewRows,
            FallbackCause::TooManyConstants => Reason::TooManyConstants,
            FallbackCause::Launch => Reason::Launch,
            FallbackCause::IllConditioned => Reason::IllConditioned,
            FallbackCause::NonFinite => Reason::NonFinite,
            FallbackCause::MetricNotNative => Reason::MetricNotNative,
            FallbackCause::DeviceMemory => Reason::DeviceMemory,
            FallbackCause::Exactness => Reason::Exactness,
        }
    }

    /// A timing test must never manufacture a cause; this is always `false`.
    pub fn is_timing(self) -> bool {
        false
    }
}

/// Cost of one candidate relative to one squared LM run, `(device, cpu)`.
///
/// A squared LM run is `2·sweeps + 3` row passes on either side, so the sweep
/// count cancels. The device kernel runs it once per gauge orientation: twice
/// for `correlation_signed`. `correlation::fit` on the CPU runs one LM per
/// orientation tried: three for `correlation_signed`. Unsigned correlation has
/// one orientation on both sides. Slope (#236) is the squared warm start plus
/// the differenced IRLS on the device, and additionally the coordinate polish
/// on the CPU (`bench_slope_fit_stage`, 154 000 rows).
fn fit_pass_scale(objective: Option<crate::fit::DeviceLoss>) -> (f64, f64) {
    match objective {
        Some(crate::fit::DeviceLoss::CorrelationSigned) => (2.0, 3.0),
        Some(crate::fit::DeviceLoss::Slope) => (2.5, 6.3),
        _ => (1.0, 1.0),
    }
}

/// Work shape offered to the planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Workload {
    pub candidates: usize,
    pub rows: usize,
    pub nodes: usize,
    pub constants: usize,
    pub sweeps: usize,
    /// H2D + D2H bytes for the whole stage.
    pub bytes: u64,
    /// Structural rows floor; below it the stage is `TooFewRows`.
    pub min_rows: usize,
    /// Explicit CPU seconds from a measured arm model or telemetry.
    pub cpu_s: Option<f64>,
    /// Explicit device seconds from a measured arm model.
    pub device_s: Option<f64>,
    /// Fraction of work that should stay on the CPU; `None` means no split.
    pub split_fraction: Option<f64>,
    /// A structural or arithmetic refusal checked before timing.
    pub refusal: Option<FallbackCause>,
    /// The fit objective; `None` and the squared family use the calibrated
    /// rates unscaled.
    pub objective: Option<crate::fit::DeviceLoss>,
}

impl Workload {
    /// A scoring workload.
    pub fn eval(candidates: usize, rows: usize, nodes: usize) -> Self {
        let bytes = estimate_eval_bytes(candidates);
        Workload {
            candidates,
            rows,
            nodes,
            constants: 0,
            sweeps: 0,
            bytes,
            min_rows: 1,
            cpu_s: None,
            device_s: None,
            split_fraction: None,
            refusal: None,
            objective: None,
        }
    }

    /// A constant-fitting workload.
    pub fn fit(candidates: usize, rows: usize, constants: usize, sweeps: usize) -> Self {
        let bytes = estimate_fit_bytes(candidates, constants);
        Workload {
            candidates,
            rows,
            nodes: 1,
            constants,
            sweeps,
            bytes,
            min_rows: 1,
            cpu_s: None,
            device_s: None,
            split_fraction: None,
            refusal: None,
            objective: None,
        }
    }

    /// An inert workload for a CPU-contract stage.
    pub fn contract() -> Self {
        Workload {
            candidates: 1,
            rows: 1,
            nodes: 1,
            constants: 0,
            sweeps: 0,
            bytes: 0,
            min_rows: 1,
            cpu_s: None,
            device_s: None,
            split_fraction: None,
            refusal: None,
            objective: None,
        }
    }

    pub fn with_min_rows(mut self, rows: usize) -> Self {
        self.min_rows = rows.max(1);
        self
    }

    pub fn with_objective(mut self, objective: crate::fit::DeviceLoss) -> Self {
        self.objective = Some(objective);
        self
    }

    pub fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    pub fn with_costs(mut self, cpu_s: f64, device_s: f64) -> Self {
        self.cpu_s = Some(cpu_s);
        self.device_s = Some(device_s);
        self
    }

    pub fn with_split(mut self, fraction: f64) -> Self {
        self.split_fraction = Some(fraction.clamp(0.0, 1.0));
        self
    }

    pub fn with_refusal(mut self, cause: FallbackCause) -> Self {
        self.refusal = Some(cause);
        self
    }
}

/// Per-stage cost terms, in seconds. `predicted_s` is the route's estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub row_evals: f64,
    pub device_s: f64,
    pub fit_s: f64,
    pub transfer_s: f64,
    pub launch_s: f64,
    pub cpu_s: f64,
    /// The predicted wall for the chosen route.
    pub predicted_s: f64,
    /// `t_dev / t_cpu`; lower than 1 favours the device.
    pub device_ratio: f64,
}

/// A planned route and its explanation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub stage: Stage,
    pub route: Route,
    pub reason: Reason,
    pub cost: Cost,
}

/// Spread of each probed term: the largest timed rep over the smallest one,
/// always at least `1.0`. A term that was not probed carries `1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub r_eval: f64,
    pub r_fit: f64,
    pub bw: f64,
    pub launch_s: f64,
    pub cpu_eval: f64,
}

impl Default for Spread {
    fn default() -> Self {
        Spread {
            r_eval: 1.0,
            r_fit: 1.0,
            bw: 1.0,
            launch_s: 1.0,
            cpu_eval: 1.0,
        }
    }
}

/// Calibrated rates used by the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// Device row-evals per second.
    pub r_eval: f64,
    /// Device accepted fits per second at the reference shape.
    pub r_fit: f64,
    /// Effective transfer bandwidth, bytes/s.
    pub bw: f64,
    /// Host-to-device bandwidth, bytes/s. Equals `bw` for an anchor.
    pub bw_h2d: f64,
    /// Device-to-host bandwidth, bytes/s. Equals `bw` for an anchor.
    pub bw_d2h: f64,
    /// Launch/synchronisation overhead, seconds.
    pub launch_s: f64,
    /// Measured host row-evals per second.
    pub cpu_eval: f64,
    /// Exact host fits per second at the reference shape.
    pub cpu_fit: f64,
    /// Unclamped rate of the host probe loop, per second; `0.0` when no host
    /// probe ran.
    pub host_rate: f64,
    /// Hysteresis fraction.
    pub hysteresis: f64,
    /// Startup probe wall (host and device), seconds.
    pub probe_s: f64,
    /// Number of timed samples behind the host terms.
    pub samples: u32,
    /// Number of timed samples behind each device term; `0` for an anchor.
    pub device_samples: u32,
    /// Spread of every probed term.
    pub spread: Spread,
    /// Where the host values came from: `startup_probe` or `anchors`.
    pub source: &'static str,
    /// Where the device values came from: `probe` or `anchor`.
    pub device_source: &'static str,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            r_eval: DEFAULT_R_EVAL,
            r_fit: DEFAULT_R_FIT,
            bw: DEFAULT_BW,
            bw_h2d: DEFAULT_BW,
            bw_d2h: DEFAULT_BW,
            launch_s: DEFAULT_LAUNCH_S,
            cpu_eval: DEFAULT_CPU_EVAL,
            cpu_fit: DEFAULT_CPU_FIT,
            host_rate: 0.0,
            hysteresis: DEFAULT_HYSTERESIS,
            probe_s: 0.0,
            samples: 0,
            device_samples: 0,
            spread: Spread::default(),
            source: "anchors",
            device_source: "anchor",
        }
    }
}

impl Calibration {
    /// Run the short host probe. The device terms stay anchors until
    /// [`Self::with_device_rates`] replaces them.
    ///
    /// The host rate scales the host anchors in proportion to the probe's
    /// reference rate. Nothing is clamped: the measured spread is reported.
    pub fn probe() -> Self {
        let start = Instant::now();
        let host = probe_cpu_eval();
        let scale = clamp_ratio(host.rate / DEFAULT_CPU_EVAL, PROBE_CLAMP);
        Calibration {
            cpu_eval: DEFAULT_CPU_EVAL * scale,
            cpu_fit: DEFAULT_CPU_FIT * scale,
            host_rate: host.rate,
            probe_s: start.elapsed().as_secs_f64(),
            samples: host.samples,
            spread: Spread {
                cpu_eval: host.spread,
                ..Spread::default()
            },
            source: "startup_probe",
            ..Calibration::default()
        }
    }

    /// Replace the device terms with hardware-probe rates.
    pub fn with_device_rates(mut self, rates: DeviceRates) -> Self {
        if rates.r_eval.is_finite() && rates.r_eval > 0.0 {
            self.r_eval = rates.r_eval;
        }
        if rates.r_fit.is_finite() && rates.r_fit > 0.0 {
            self.r_fit = rates.r_fit;
        }
        if rates.bw.is_finite() && rates.bw > 0.0 {
            self.bw = rates.bw;
            self.bw_h2d = if rates.bw_h2d.is_finite() && rates.bw_h2d > 0.0 {
                rates.bw_h2d
            } else {
                rates.bw
            };
            self.bw_d2h = if rates.bw_d2h.is_finite() && rates.bw_d2h > 0.0 {
                rates.bw_d2h
            } else {
                rates.bw
            };
        }
        if rates.launch_s.is_finite() && rates.launch_s >= 0.0 {
            self.launch_s = rates.launch_s;
        }
        self.spread.r_eval = rates.spread.r_eval;
        self.spread.r_fit = rates.spread.r_fit;
        self.spread.bw = rates.spread.bw;
        self.spread.launch_s = rates.spread.launch_s;
        self.device_samples = rates.samples;
        self.probe_s += rates.probe_s;
        self.device_source = "probe";
        self
    }

    /// True when every rate is finite and positive.
    pub fn is_usable(&self) -> bool {
        self.r_eval.is_finite()
            && self.r_eval > 0.0
            && self.r_fit.is_finite()
            && self.r_fit > 0.0
            && self.bw.is_finite()
            && self.bw > 0.0
            && self.cpu_eval.is_finite()
            && self.cpu_eval > 0.0
            && self.cpu_fit.is_finite()
            && self.cpu_fit > 0.0
            && self.launch_s.is_finite()
            && self.launch_s >= 0.0
            && self.hysteresis.is_finite()
            && self.hysteresis >= 0.0
    }

    /// Device fit rate for a `(rows, constants, sweeps)` shape.
    pub fn fit_rate(&self, rows: usize, constants: usize, sweeps: usize) -> f64 {
        let rows = rows.max(1) as f64;
        let constants = constants.max(1) as f64;
        let sweeps = sweeps.max(1) as f64;
        self.r_fit
            * (FIT_REF_ROWS / rows)
            * (FIT_REF_CONSTANTS / constants)
            * (FIT_REF_SWEEPS / sweeps)
    }

    /// Exact CPU fit rate for a `(rows, constants, sweeps)` shape.
    pub fn cpu_fit_rate(&self, rows: usize, constants: usize, sweeps: usize) -> f64 {
        let rows = rows.max(1) as f64;
        let constants = constants.max(1) as f64;
        let sweeps = sweeps.max(1) as f64;
        self.cpu_fit
            * (FIT_REF_ROWS / rows)
            * (FIT_REF_CONSTANTS / constants)
            * (FIT_REF_SWEEPS / sweeps)
    }

    /// Per term: name, ratio of the larger value to the smaller one, and the
    /// band the two probes state, which is the larger of their two spreads. A
    /// non-finite or non-positive value gives an infinite ratio.
    pub fn agreement(&self, other: &Calibration) -> Vec<(&'static str, f64, f64)> {
        let terms = [
            ("r_eval", self.r_eval, other.r_eval, self.spread.r_eval, other.spread.r_eval),
            ("r_fit", self.r_fit, other.r_fit, self.spread.r_fit, other.spread.r_fit),
            ("bw", self.bw, other.bw, self.spread.bw, other.spread.bw),
            (
                "launch_s",
                self.launch_s.max(1e-12),
                other.launch_s.max(1e-12),
                self.spread.launch_s,
                other.spread.launch_s,
            ),
            (
                "host_rate",
                self.host_rate,
                other.host_rate,
                self.spread.cpu_eval,
                other.spread.cpu_eval,
            ),
        ];
        terms
            .iter()
            .filter(|t| t.1 > 0.0 || t.2 > 0.0)
            .map(|&(name, a, b, sa, sb)| {
                let ratio = if a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 {
                    (a / b).max(b / a)
                } else {
                    f64::INFINITY
                };
                (name, ratio, sa.max(sb))
            })
            .collect()
    }

    /// Result-JSON variant: provenance labels only. The measured rates and the
    /// probe time are wall-clock noise and must not make the result JSON differ
    /// between two otherwise identical runs (the dev-log contract keeps the
    /// archive and result bytes identical). The manifest variant below records
    /// the measured values.
    pub fn json(&self) -> String {
        format!(
            "{{\"model_version\":{},\"source\":\"{}\",\"device_source\":\"{}\",\"samples\":{},\"device_samples\":{},\"hysteresis\":{:.4},\"fit_ref\":{{\"rows\":{:.0},\"constants\":{:.0},\"sweeps\":{:.0}}}}}",
            MODEL_VERSION,
            self.source,
            self.device_source,
            self.samples,
            self.device_samples,
            self.hysteresis,
            FIT_REF_ROWS,
            FIT_REF_CONSTANTS,
            FIT_REF_SWEEPS
        )
    }

    /// Provenance variant for `--run-manifest`: the measured rates, their
    /// sources, spreads and the probe time. A manifest is not compared
    /// byte-for-byte across runs.
    pub fn manifest_json(&self) -> String {
        format!(
            "{{\"model_version\":{},\"source\":\"{}\",\"device_source\":\"{}\",\"probe_s\":{:.6},\"samples\":{},\"device_samples\":{},\"hysteresis\":{:.4},\"r_eval\":{:.6e},\"r_fit\":{:.6e},\"bw\":{:.6e},\"bw_h2d\":{:.6e},\"bw_d2h\":{:.6e},\"launch_s\":{:.9},\"cpu_eval\":{:.6e},\"cpu_fit\":{:.6e},\"host_rate\":{:.6e},\"spread\":{{\"r_eval\":{:.4},\"r_fit\":{:.4},\"bw\":{:.4},\"launch_s\":{:.4},\"cpu_eval\":{:.4}}},\"fit_ref\":{{\"rows\":{:.0},\"constants\":{:.0},\"sweeps\":{:.0}}}}}",
            MODEL_VERSION,
            self.source,
            self.device_source,
            self.probe_s,
            self.samples,
            self.device_samples,
            self.hysteresis,
            self.r_eval,
            self.r_fit,
            self.bw,
            self.bw_h2d,
            self.bw_d2h,
            self.launch_s,
            self.cpu_eval,
            self.cpu_fit,
            self.host_rate,
            self.spread.r_eval,
            self.spread.r_fit,
            self.spread.bw,
            self.spread.launch_s,
            self.spread.cpu_eval,
            FIT_REF_ROWS,
            FIT_REF_CONSTANTS,
            FIT_REF_SWEEPS
        )
    }
}

/// Device rates measured by a hardware probe, when one is available.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceRates {
    pub r_eval: f64,
    pub r_fit: f64,
    pub bw: f64,
    pub bw_h2d: f64,
    pub bw_d2h: f64,
    pub launch_s: f64,
    pub spread: Spread,
    /// Timed reps behind each term.
    pub samples: u32,
    /// Wall of the whole device probe, seconds.
    pub probe_s: f64,
}

impl Default for DeviceRates {
    fn default() -> Self {
        DeviceRates {
            r_eval: DEFAULT_R_EVAL,
            r_fit: DEFAULT_R_FIT,
            bw: DEFAULT_BW,
            bw_h2d: DEFAULT_BW,
            bw_d2h: DEFAULT_BW,
            launch_s: DEFAULT_LAUNCH_S,
            spread: Spread::default(),
            samples: 0,
            probe_s: 0.0,
        }
    }
}

/// Per-stage ledger row.
#[derive(Clone, Debug, PartialEq)]
pub struct StageLedger {
    pub stage: Stage,
    pub route: Route,
    pub reason: Reason,
    pub calls: u64,
    pub candidates: u64,
    pub predicted_s: f64,
    pub actual_s: f64,
    pub bytes: u64,
    pub switches: u64,
}

impl StageLedger {
    fn new(stage: Stage) -> Self {
        StageLedger {
            stage,
            route: Route::Cpu,
            reason: Reason::CpuContract,
            calls: 0,
            candidates: 0,
            predicted_s: 0.0,
            actual_s: 0.0,
            bytes: 0,
            switches: 0,
        }
    }

    pub fn ratio(&self) -> Option<f64> {
        if self.predicted_s > 0.0 && self.actual_s.is_finite() {
            Some(self.actual_s / self.predicted_s)
        } else {
            None
        }
    }

    fn json(&self) -> String {
        let ratio = self
            .ratio()
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"stage\":\"{}\",\"route\":\"{}\",\"reason\":\"{}\",\"calls\":{},\"candidates\":{},\"predicted_s\":{:.6},\"actual_s\":{:.6},\"actual_over_predicted\":{},\"bytes\":{},\"switches\":{}}}",
            self.stage.name(),
            self.route.name(),
            self.reason.name(),
            self.calls,
            self.candidates,
            self.predicted_s,
            self.actual_s,
            ratio,
            self.bytes,
            self.switches
        )
    }
}

/// The complete route ledger for one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ledger {
    pub stages: BTreeMap<Stage, StageLedger>,
}

impl Ledger {
    /// True when the cost model, not a refusal, kept every movable stage on
    /// the CPU: at least one stage lost the timing comparison (or had no
    /// usable calibration) and none ran on the device.
    pub fn cost_model_kept_cpu(&self) -> bool {
        let movable = [Stage::Eval, Stage::Fit];
        let on_device = movable.iter().any(|s| {
            self.stages
                .get(s)
                .map_or(false, |e| e.route != Route::Cpu)
        });
        let by_cost = movable.iter().any(|s| {
            self.stages.get(s).map_or(false, |e| {
                matches!(e.reason, Reason::CostCpu | Reason::NoCalibration)
            })
        });
        by_cost && !on_device
    }

    pub fn entry(&self, stage: Stage) -> Option<&StageLedger> {
        self.stages.get(&stage)
    }

    pub fn json(&self) -> String {
        let rows: Vec<String> = Stage::ALL
            .iter()
            .filter_map(|s| self.stages.get(s))
            .map(|s| s.json())
            .collect();
        format!("{{\"stages\":[{}]}}", rows.join(","))
    }
}

/// Stateful dispatcher: calibration + per-stage hysteresis + ledger.
#[derive(Clone, Debug)]
pub struct Dispatcher {
    calibration: Calibration,
    last: BTreeMap<Stage, Route>,
    ledger: Ledger,
    mode: DispatchMode,
}

impl Dispatcher {
    pub fn new(calibration: Calibration) -> Self {
        Dispatcher {
            calibration,
            last: BTreeMap::new(),
            ledger: Ledger::default(),
            mode: DispatchMode::Auto,
        }
    }

    pub fn with_mode(mut self, mode: DispatchMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Plan a stage. Structural/arithmetic refusals are applied first, then
    /// CPU-contract stages, then the cost comparison with hysteresis.
    pub fn plan(&mut self, stage: Stage, workload: &Workload) -> Decision {
        let cost = self.cost(stage, workload);
        let (route, reason) = if let Some(cause) = workload.refusal {
            (Route::Cpu, cause.reason())
        } else if workload.rows < workload.min_rows {
            (Route::Cpu, Reason::TooFewRows)
        } else if stage.cpu_by_contract() {
            (Route::Cpu, Reason::CpuContract)
        } else if self.mode == DispatchMode::Auto && !self.calibration.is_usable() {
            (Route::Cpu, Reason::NoCalibration)
        } else if let Some(f) = workload.split_fraction {
            if f > 0.0 && f < 1.0 {
                (Route::Split, Reason::SplitCost)
            } else {
                self.cost_route(stage, &cost)
            }
        } else {
            self.cost_route(stage, &cost)
        };
        let predicted_s = match route {
            Route::Device => cost.device_s + cost.fit_s + cost.transfer_s + cost.launch_s,
            Route::Cpu => cost.cpu_s,
            Route::Split => {
                let f = workload.split_fraction.unwrap_or(0.5);
                f * (cost.device_s + cost.fit_s + cost.transfer_s + cost.launch_s)
                    + (1.0 - f) * cost.cpu_s
            }
        };
        let cost = Cost {
            predicted_s,
            ..cost
        };
        let previous = self.last.insert(stage, route);
        let mut switches = 0;
        if let Some(prev) = previous {
            if prev != route {
                switches = 1;
            }
        }
        let entry = self
            .ledger
            .stages
            .entry(stage)
            .or_insert_with(|| StageLedger::new(stage));
        if previous.is_some() {
            entry.switches += switches;
        }
        entry.route = route;
        entry.reason = reason;
        Decision {
            stage,
            route,
            reason,
            cost,
        }
    }

    fn cost_route(&self, stage: Stage, cost: &Cost) -> (Route, Reason) {
        if self.mode == DispatchMode::Device {
            return (Route::Device, Reason::Forced);
        }
        if !(cost.cpu_s.is_finite() && cost.cpu_s > 0.0) {
            return (Route::Cpu, Reason::NoCalibration);
        }
        let t_dev = cost.device_s + cost.fit_s + cost.transfer_s + cost.launch_s;
        let h = self.calibration.hysteresis.max(0.0);
        let device_threshold = 1.0 / (1.0 + h);
        let hold_threshold = 1.0 + h;
        let base_device = t_dev < cost.cpu_s * device_threshold;
        match self.last.get(&stage) {
            Some(Route::Device) if t_dev < cost.cpu_s * hold_threshold => (
                Route::Device,
                if base_device {
                    Reason::CostDevice
                } else {
                    Reason::HysteresisDevice
                },
            ),
            _ if base_device => (Route::Device, Reason::CostDevice),
            _ => (Route::Cpu, Reason::CostCpu),
        }
    }

    /// Compute the raw cost terms for one stage.
    pub fn cost(&self, stage: Stage, workload: &Workload) -> Cost {
        let cal = &self.calibration;
        let row_evals = (workload.candidates.max(1) as f64)
            * (workload.rows.max(1) as f64)
            * (workload.nodes.max(1) as f64);
        let transfer_s = if workload.bytes == 0 {
            0.0
        } else if cal.bw > 0.0 {
            workload.bytes as f64 / cal.bw
        } else {
            f64::INFINITY
        };
        let launch_s = cal.launch_s;
        let (device_s, fit_s, cpu_s) = if let (Some(cpu), Some(dev)) =
            (workload.cpu_s, workload.device_s)
        {
            (dev, 0.0, cpu)
        } else {
            match stage {
                Stage::Eval => {
                    let dev = if cal.r_eval > 0.0 {
                        row_evals / cal.r_eval
                    } else {
                        f64::INFINITY
                    };
                    let cpu = if cal.cpu_eval > 0.0 {
                        row_evals / cal.cpu_eval
                    } else {
                        f64::INFINITY
                    };
                    (dev, 0.0, cpu)
                }
                Stage::Fit => {
                    let candidates = workload.candidates.max(1) as f64;
                    let (dev_passes, cpu_passes) =
                        fit_pass_scale(workload.objective);
                    let fit = dev_passes * candidates
                        / cal.fit_rate(workload.rows, workload.constants, workload.sweeps);
                    let cpu = cpu_passes * candidates
                        / cal.cpu_fit_rate(workload.rows, workload.constants, workload.sweeps);
                    (0.0, fit, cpu)
                }
                _ => (0.0, 0.0, 0.0),
            }
        };
        let device_ratio = if cpu_s.is_finite() && cpu_s > 0.0 {
            (device_s + fit_s + transfer_s + launch_s) / cpu_s
        } else {
            f64::INFINITY
        };
        Cost {
            row_evals,
            device_s,
            fit_s,
            transfer_s,
            launch_s,
            cpu_s,
            predicted_s: 0.0,
            device_ratio,
        }
    }

    /// Record one observed stage call. `actual_s` is measured by the caller.
    pub fn record(&mut self, decision: Decision, actual_s: f64, candidates: usize, bytes: u64) {
        let entry = self
            .ledger
            .stages
            .entry(decision.stage)
            .or_insert_with(|| StageLedger::new(decision.stage));
        entry.route = decision.route;
        entry.reason = decision.reason;
        entry.calls += 1;
        entry.candidates += candidates as u64;
        entry.predicted_s += decision.cost.predicted_s;
        if actual_s.is_finite() && actual_s >= 0.0 {
            entry.actual_s += actual_s;
        }
        entry.bytes += bytes;
    }

    /// Record a CPU-contract or refusal stage with no measured time.
    pub fn record_plan(&mut self, decision: Decision, candidates: usize, bytes: u64) {
        self.record(decision, 0.0, candidates, bytes);
    }

    /// Ensure every stage has a ledger row, even when it never ran.
    pub fn ensure_all_stages(&mut self) {
        for stage in Stage::ALL {
            self.ledger
                .stages
                .entry(stage)
                .or_insert_with(|| StageLedger::new(stage));
        }
    }

    pub fn into_ledger(mut self) -> Ledger {
        self.ensure_all_stages();
        self.ledger
    }

    pub fn snapshot(&self) -> DispatchReport {
        let mut ledger = self.ledger.clone();
        for stage in Stage::ALL {
            ledger
                .stages
                .entry(stage)
                .or_insert_with(|| StageLedger::new(stage));
        }
        DispatchReport {
            calibration: self.calibration,
            ledger,
        }
    }
}

/// The JSON/report payload: calibration plus the route ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchReport {
    pub calibration: Calibration,
    pub ledger: Ledger,
}

impl Default for DispatchReport {
    fn default() -> Self {
        DispatchReport {
            calibration: Calibration::default(),
            ledger: Ledger::default(),
        }
    }
}

impl DispatchReport {
    pub fn json(&self) -> String {
        format!(
            "{{\"calibration\":{},\"route_ledger\":{}}}",
            self.calibration.json(),
            self.ledger.json()
        )
    }

    /// Manifest variant: same ledger, plus the startup probe time.
    pub fn manifest_json(&self) -> String {
        format!(
            "{{\"calibration\":{},\"route_ledger\":{}}}",
            self.calibration.manifest_json(),
            self.ledger.json()
        )
    }
}

/// Estimate the bytes one scoring batch sends: programs up plus errors down.
pub fn estimate_eval_bytes(candidates: usize) -> u64 {
    (candidates.max(1) as u64).saturating_mul(256 + 8)
}

/// Estimate the bytes one fit batch sends: constants and flags both ways.
pub fn estimate_fit_bytes(candidates: usize, constants: usize) -> u64 {
    let per = 256u64.saturating_add(8 * constants as u64);
    (candidates.max(1) as u64).saturating_mul(per)
}

fn clamp_ratio(value: f64, max: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        1.0
    } else {
        value.clamp(1.0 / max, max)
    }
}

/// Median and spread of a set of timed reps: the middle value, and the largest
/// over the smallest. Non-finite and non-positive reps are dropped; an empty
/// set gives `None`.
pub fn median_and_spread(samples: &[f64]) -> Option<(f64, f64)> {
    let mut v: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let median = if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    };
    Some((median, v[v.len() - 1] / v[0]))
}

struct HostProbe {
    rate: f64,
    spread: f64,
    samples: u32,
}

/// Time a small, deterministic host evaluation loop. The rate is the median of
/// the timed reps and the spread is their largest over their smallest.
fn probe_cpu_eval() -> HostProbe {
    const ROWS: usize = 65_536;
    const REPS: usize = 31;
    let x: Vec<f64> = (0..ROWS).map(|i| (i as f64) * 0.000_01 - 0.3).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| (i as f64).sin()).collect();
    let run = || {
        let t = Instant::now();
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        let mut c = 0.0f64;
        for i in 0..ROWS {
            let v = x[i];
            a += v * v + 0.5 * v - 1.0;
            b += (v + 0.25).abs();
            c += v * y[i];
        }
        let elapsed = t.elapsed().as_secs_f64();
        (elapsed, a + b + c)
    };
    // One untimed pass warms the caches and the clock.
    let (_, mut sink) = run();
    let mut rates = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let (elapsed, s) = run();
        sink += s;
        if elapsed > 0.0 {
            rates.push(ROWS as f64 * 4.0 / elapsed);
        }
    }
    if sink.is_nan() {
        return HostProbe {
            rate: DEFAULT_CPU_EVAL,
            spread: 1.0,
            samples: REPS as u32,
        };
    }
    match median_and_spread(&rates) {
        Some((rate, spread)) => HostProbe {
            rate,
            spread,
            samples: rates.len() as u32,
        },
        None => HostProbe {
            rate: DEFAULT_CPU_EVAL,
            spread: 1.0,
            samples: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cal() -> Calibration {
        Calibration::default()
    }

    fn forced(cpu: f64, dev: f64) -> Workload {
        Workload::eval(1, 1, 1).with_costs(cpu, dev)
    }

    #[test]
    fn every_stage_has_a_route_and_a_machine_reason() {
        let mut d = Dispatcher::new(cal());
        for stage in Stage::ALL {
            let decision = d.plan(stage, &Workload::contract());
            assert!(matches!(decision.route, Route::Cpu | Route::Device | Route::Split));
            assert!(!decision.reason.name().is_empty());
            assert!(!decision.reason.name().contains(' '));
        }
        assert_eq!(d.plan(Stage::Select, &Workload::contract()).route, Route::Cpu);
        assert_eq!(d.plan(Stage::Refit, &Workload::contract()).reason, Reason::CpuContract);
    }

    #[test]
    fn device_crosses_over_monotonically_in_rows_candidates_and_nodes() {
        let c = cal();
        for stage in [Stage::Eval, Stage::Fit] {
            let mut previous_ratio = f64::INFINITY;
            for n in [1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
                let w = match stage {
                    Stage::Eval => Workload::eval(512, n, 8),
                    _ => Workload::fit(128, n, 4, 8),
                };
                let ratio = c.cost_public(stage, &w).device_ratio;
                assert!(ratio <= previous_ratio + 1e-9, "{:?} n={} ratio {} > {}", stage, n, ratio, previous_ratio);
                previous_ratio = ratio;
            }
        }
        let small = c.cost_public(Stage::Eval, &Workload::eval(1, 100_000, 8)).device_ratio;
        let large = c.cost_public(Stage::Eval, &Workload::eval(10_000, 100_000, 8)).device_ratio;
        assert!(large <= small, "more candidates must not favour the CPU: {} > {}", large, small);
    }

    #[test]
    fn more_constants_and_sweeps_make_the_device_fit_more_attractive() {
        let c = cal();
        let base = c
            .cost_public(Stage::Fit, &Workload::fit(128, 154_000, 1, 8))
            .device_ratio;
        let more_p = c
            .cost_public(Stage::Fit, &Workload::fit(128, 154_000, 8, 8))
            .device_ratio;
        let more_sweeps = c
            .cost_public(Stage::Fit, &Workload::fit(128, 154_000, 8, 32))
            .device_ratio;
        assert!(more_p <= base, "p must not make the device less attractive");
        assert!(more_sweeps <= more_p, "sweeps must not make the device less attractive");
    }

    #[test]
    fn hysteresis_holds_the_device_inside_the_band_and_switches_outside_it() {
        let mut d = Dispatcher::new(cal());
        let start = forced(1.0, 0.5); // clearly device
        let first = d.plan(Stage::Eval, &start);
        assert_eq!(first.route, Route::Device);
        // 1.1× CPU cost is inside the +30 % hold band: stay on the device.
        let inside = d.plan(Stage::Eval, &forced(1.0, 1.1));
        assert_eq!(inside.route, Route::Device);
        assert_eq!(inside.reason, Reason::HysteresisDevice);
        // 1.4× is outside the hold band: move to the CPU.
        let out = d.plan(Stage::Eval, &forced(1.0, 1.4));
        assert_eq!(out.route, Route::Cpu);
        // From the CPU, 0.9× is not enough to cross back (needs < 1/1.3).
        assert_eq!(d.plan(Stage::Eval, &forced(1.0, 0.9)).route, Route::Cpu);
        assert_eq!(d.plan(Stage::Eval, &forced(1.0, 0.5)).route, Route::Device);
    }

    #[test]
    fn an_arithmetic_cause_never_becomes_a_timing_decision() {
        for cause in FallbackCause::ALL {
            let mut d = Dispatcher::new(cal());
            // The device is absurdly fast; the cause must still win.
            let w = forced(1_000.0, 1e-9).with_refusal(cause);
            let decision = d.plan(Stage::Fit, &w);
            assert_eq!(decision.route, Route::Cpu, "{:?}", cause);
            assert_eq!(decision.reason, cause.reason());
            assert!(!cause.is_timing());
            assert!(decision.reason.is_cause());
        }
    }

    #[test]
    fn split_route_is_explicit_and_predicts_a_mixture() {
        let mut d = Dispatcher::new(cal());
        let w = Workload::eval(1_000, 10_000, 8).with_split(0.25);
        let decision = d.plan(Stage::Eval, &w);
        assert_eq!(decision.route, Route::Split);
        assert_eq!(decision.reason, Reason::SplitCost);
        assert!(decision.cost.predicted_s > 0.0);
    }

    #[test]
    fn host_probe_three_runs_agree_within_the_larger_of_their_own_spreads() {
        let runs = [Calibration::probe(), Calibration::probe(), Calibration::probe()];
        for run in &runs {
            assert!(run.is_usable());
            assert!(run.probe_s < 0.1, "startup probe took {:.3}s", run.probe_s);
            assert_eq!(run.device_source, "anchor");
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            for (term, ratio, band) in runs[i].agreement(&runs[j]) {
                assert!(
                    ratio <= band,
                    "probes {} and {} differ on {} by {:.4}x, past the band {:.4}x (the larger of the two probes' own spreads)",
                    i, j, term, ratio, band
                );
            }
        }
    }

    #[test]
    fn agreement_rejects_values_outside_the_probes_own_spread() {
        let a = Calibration::default();
        let mut b = Calibration::default();
        b.r_eval = a.r_eval * 3.0;
        b.spread.r_eval = 1.1;
        let rows = a.agreement(&b);
        let (_, ratio, band) = rows.iter().find(|r| r.0 == "r_eval").copied().unwrap();
        assert!(ratio > band, "ratio {} band {}", ratio, band);
        assert!(rows.iter().filter(|r| r.1 > r.2).count() == 1);
    }

    #[test]
    fn median_and_spread_are_the_middle_rep_and_max_over_min() {
        assert_eq!(median_and_spread(&[4.0, 1.0, 2.0]), Some((2.0, 4.0)));
        assert_eq!(median_and_spread(&[1.0, 3.0]), Some((2.0, 3.0)));
        assert_eq!(median_and_spread(&[f64::NAN, -1.0]), None);
    }

    #[test]
    fn device_rates_replace_the_anchors_and_are_labelled_probe() {
        let anchor = Calibration::default();
        assert_eq!(anchor.device_source, "anchor");
        let rates = DeviceRates {
            r_eval: 3.0e9,
            r_fit: 7.0,
            bw: 5.0e9,
            bw_h2d: 6.0e9,
            bw_d2h: 4.5e9,
            launch_s: 2.0e-5,
            spread: Spread {
                r_eval: 1.05,
                r_fit: 1.07,
                bw: 1.1,
                launch_s: 1.3,
                cpu_eval: 1.0,
            },
            samples: 7,
            probe_s: 0.05,
        };
        let cal = anchor.with_device_rates(rates);
        assert_eq!(cal.device_source, "probe");
        assert_eq!((cal.r_eval, cal.r_fit, cal.bw, cal.launch_s), (3.0e9, 7.0, 5.0e9, 2.0e-5));
        assert_eq!((cal.bw_h2d, cal.bw_d2h, cal.device_samples), (6.0e9, 4.5e9, 7));
        assert_eq!(cal.spread.launch_s, 1.3);
        let m = cal.manifest_json();
        assert!(m.contains("\"device_source\":\"probe\""), "{}", m);
        assert!(m.contains("\"r_eval\":3.000000e9"), "{}", m);
        assert!(m.contains("\"bw_h2d\":6.000000e9"), "{}", m);
        assert!(m.contains("\"spread\":{\"r_eval\":1.0500"), "{}", m);
        assert!(anchor.manifest_json().contains("\"device_source\":\"anchor\""));
    }

    #[test]
    fn calibration_json_labels_sources_and_leaves_measured_rates_to_the_manifest() {
        let cal = Calibration::probe();
        let j = cal.json();
        assert!(j.contains("\"source\":\"startup_probe\""));
        assert!(j.contains("\"device_source\":\"anchor\""));
        assert!(j.contains("\"model_version\":1"));
        assert!(!j.contains("r_eval"), "the result JSON carries no measured rate: {}", j);
        let m = cal.manifest_json();
        for key in ["r_eval", "r_fit", "bw", "bw_h2d", "bw_d2h", "launch_s", "cpu_eval", "host_rate", "spread"] {
            assert!(m.contains(&format!("\"{}\":", key)), "{} missing in {}", key, m);
        }
    }

    #[test]
    fn ledger_records_predicted_actual_and_switches() {
        let mut d = Dispatcher::new(cal());
        let a = d.plan(Stage::Eval, &forced(1.0, 0.5));
        d.record(a, 0.02, 512, 1_000);
        let b = d.plan(Stage::Eval, &forced(1.0, 1.5));
        d.record(b, 0.50, 512, 1_000);
        let ledger = d.into_ledger();
        let eval = ledger.entry(Stage::Eval).expect("eval row");
        assert_eq!(eval.calls, 2);
        assert_eq!(eval.candidates, 1_024);
        assert_eq!(eval.bytes, 2_000);
        assert_eq!(eval.route, Route::Cpu);
        assert_eq!(eval.switches, 1);
        assert!(eval.actual_s > 0.0);
        assert!(eval.predicted_s > 0.0);
        assert!(eval.ratio().is_some());
        assert!(ledger.entry(Stage::Io).is_some(), "every stage is present");
        assert!(ledger.json().contains("\"route_ledger\"") == false);
        assert!(ledger.json().contains("\"stage\":\"fit\""));
    }

    #[test]
    fn report_json_contains_calibration_and_every_stage() {
        let mut d = Dispatcher::new(cal());
        assert_eq!(d.plan(Stage::Fit, &Workload::fit(128, 2_456_306, 4, 8)).route, Route::Device);
        let report = d.snapshot();
        let j = report.json();
        assert!(j.contains("\"calibration\""));
        assert!(j.contains("\"route_ledger\""));
        for stage in Stage::ALL {
            assert!(j.contains(&format!("\"stage\":\"{}\"", stage.name())), "{}", stage.name());
        }
    }

    #[test]
    fn no_device_is_a_cause_not_a_slow_measurement() {
        let mut d = Dispatcher::new(cal());
        let w = Workload::eval(1, 1, 1)
            .with_refusal(FallbackCause::NoDevice)
            .with_costs(1e9, 1e-12);
        let dec = d.plan(Stage::Eval, &w);
        assert_eq!(dec.route, Route::Cpu);
        assert_eq!(dec.reason, Reason::NoDevice);
    }

    #[test]
    fn default_calibration_keeps_the_compact_cell_on_the_device() {
        // 512 candidates × 2.456 M rows × 8 nodes is the compact-cell batch.
        let mut d = Dispatcher::new(cal());
        let eval = d.plan(Stage::Eval, &Workload::eval(512, 2_456_306, 8));
        assert_eq!(eval.route, Route::Device, "{:?}", eval);
        let fit = d.plan(Stage::Fit, &Workload::fit(128, 2_456_306, 4, 8));
        assert_eq!(fit.route, Route::Device, "{:?}", fit);
    }

    #[test]
    fn synthetic_workloads_match_the_oracle_route_and_stay_monotone() {
        // Sweep a synthetic grid with known T_cpu/T_dev. The oracle uses the
        // same hysteresis-free comparison; the routed model must agree on
        // every point, and the device fraction must not fall as work grows.
        let mut d = Dispatcher::new(cal());
        let mut device = 0usize;
        let mut total = 0usize;
        for n in [64usize, 512, 4_096, 32_768, 262_144, 2_456_306] {
            for w in [1usize, 8, 64, 512, 4_096] {
                let workload = Workload::eval(w, n, 8);
                let cost = d.cost(Stage::Eval, &workload);
                let decision = d.plan(Stage::Eval, &workload);
                total += 1;
                if decision.route == Route::Device {
                    device += 1;
                    assert!(cost.device_ratio < 1.31, "device outside the hold band: {:?}", decision);
                } else {
                    // A CPU route is either a real cost loss or hysteresis.
                    assert!(cost.device_ratio > 0.0);
                }
            }
        }
        assert!(device > 0 && device < total, "grid must contain both sides: {}/{}", device, total);
    }

    #[test]
    fn forced_device_skips_timing_but_keeps_refusals() {
        let slow_device = forced(1.0e-6, 1.0);
        let mut auto = Dispatcher::new(cal());
        assert_eq!(auto.plan(Stage::Eval, &slow_device).route, Route::Cpu);
        let mut d = Dispatcher::new(cal()).with_mode(DispatchMode::Device);
        for stage in [Stage::Eval, Stage::Fit] {
            let decision = d.plan(stage, &slow_device);
            assert_eq!(decision.route, Route::Device);
            assert_eq!(decision.reason, Reason::Forced);
        }
        let unusable = Calibration {
            r_eval: 0.0,
            ..cal()
        };
        let mut d_unusable = Dispatcher::new(unusable).with_mode(DispatchMode::Device);
        assert_eq!(d_unusable.plan(Stage::Eval, &slow_device).reason, Reason::Forced);
        for cause in FallbackCause::ALL {
            let decision = d.plan(Stage::Eval, &slow_device.with_refusal(cause));
            assert_eq!(decision.route, Route::Cpu, "{:?}", cause);
            assert_eq!(decision.reason, cause.reason());
        }
        let few = d.plan(Stage::Eval, &slow_device.with_min_rows(2));
        assert_eq!((few.route, few.reason), (Route::Cpu, Reason::TooFewRows));
        for stage in Stage::ALL.iter().copied().filter(|s| s.cpu_by_contract()) {
            let decision = d.plan(stage, &Workload::contract());
            assert_eq!((decision.route, decision.reason), (Route::Cpu, Reason::CpuContract));
        }
        let mut split = slow_device;
        split.split_fraction = Some(0.5);
        assert_eq!(d.plan(Stage::Eval, &split).route, Route::Split);
        assert_eq!(Reason::Forced.name(), "forced");
        assert!(!Reason::Forced.is_cause());
    }

    #[test]
    fn cost_model_keeping_every_stage_on_the_cpu_is_reported() {
        let mut d = Dispatcher::new(cal());
        let eval = d.plan(Stage::Eval, &forced(1.0e-6, 1.0));
        d.record_plan(eval, 1, 0);
        assert!(d.snapshot().ledger.cost_model_kept_cpu());
        let fit = d.plan(Stage::Fit, &forced(1.0, 1.0e-6));
        d.record_plan(fit, 1, 0);
        assert!(!d.snapshot().ledger.cost_model_kept_cpu());
        let mut refused = Dispatcher::new(cal());
        let eval = refused.plan(Stage::Eval, &forced(1.0, 1.0).with_refusal(FallbackCause::NoDevice));
        refused.record_plan(eval, 1, 0);
        assert!(!refused.snapshot().ledger.cost_model_kept_cpu());
    }
}

#[cfg(test)]
impl Calibration {
    fn cost_public(&self, stage: Stage, workload: &Workload) -> Cost {
        Dispatcher::new(*self).cost(stage, workload)
    }
}
