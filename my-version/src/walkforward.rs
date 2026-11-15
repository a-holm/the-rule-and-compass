//! Walk-forward evaluation over the column store (issue #124).
//!
//! This is the Rust replacement for the pilot's Python backtest. It reads a
//! `.rcsb` store group by group (or a CSV as a fallback), evaluates a supplied
//! expression with the engine's own [`crate::expr::eval`], applies a long/flat
//! position rule and charges costs on position changes. Nothing here shells out
//! to Python or evaluates a code string.
//!
//! # Semantics (deliberately small and checkable)
//!
//! * Time is the only split axis. Every window has a half-open training range
//!   `[train_start, train_end)` and test range `[test_start, test_end)`.
//! * The gap `test_start - train_end` must be at least the label horizon
//!   (`--horizon`) and the requested embargo. Rows in the gap are purged.
//! * The threshold is an exact type-7 quantile (numpy's default) of the
//!   expression score over the training rows, or over an optional trailing
//!   validation slice of those rows. The test block is never consulted. #305
//!   selects the two ranks around `h = (n - 1) q` by order-preserving radix
//!   count and interpolates in `f64`; a pre-registered `--threshold-fixed`
//!   value is reported with `threshold_source = "fixed"`.
//! * A row enters the P&L denominator only when both score and target are
//!   finite, through the shared `trading_gpu::ValidMask`.
//! * `position = 1` when the score is strictly greater than the threshold and
//!   the score is finite, otherwise `0`. There is no short leg and no leverage.
//! * Costs are `cost_bps` per side on `|position - previous position|` inside
//!   each group; the first position starts from flat (0). A round trip therefore
//!   costs `2 * cost_bps`.
//! * Rows sharing a timestamp are equal-weighted, exactly as in the Python
//!   reference. The period return is the mean net return across the groups
//!   present at that timestamp.
//!
//! # Memory
//!
//! A `.rcsb` store is scanned one row group at a time. The resident state is
//! one group plus the test rows selected by the windows and one score per
//! training row per window. A CSV is read as a resident [`Table`]; the same
//! accumulation rule applies afterwards.

use crate::data::{self, Table};
use crate::expr::{self, Node};
use crate::store::{self, FileStore};
use crate::trading_gpu::{self, Threshold, ThresholdSource, ValidMask};
use crate::trading::MissingScore;
use crate::inference;
use crate::wf_protocol::NullBook;
use std::collections::HashMap;

/// Default per-side cost in basis points (10 bp, as in the pilot).
pub const DEFAULT_COST_BPS: f64 = 10.0;
/// Default score quantile used to set the long threshold (q70).
pub const DEFAULT_QUANTILE: f64 = 0.70;
/// Seconds in a (crypto) year, used to turn a time step into periods per year.
pub const SECONDS_PER_YEAR: f64 = 365.0 * 86_400.0;

/// Gate-state rule `apply_mask` applies; `procedure_spec` names it in the candidate id.
pub const MASK_STATE_RULE: &str = "window-restart/1";

/// Options of the long-only evaluation protocol. The defaults leave every
/// existing report value unchanged: the new report sections only add fields,
/// and the bootstrap and null sections stay off until a draw count is set.
#[derive(Clone, Debug)]
pub struct ProtocolOptions {
    /// Seconds from the row stamp to the label's entry. Part of the required purge.
    pub entry_offset: f64,
    /// Seconds of execution delay. Part of the required purge.
    pub delay: f64,
    /// Validation start per window label, overriding `validation_frac`.
    pub validation_starts: Vec<(String, f64)>,
    /// Circular block bootstrap draws; 0 disables the bootstrap section.
    pub bootstrap_draws: usize,
    /// Block length in periods; `None` uses `max(horizon, hold, n^(1/3))`.
    pub block_len: Option<usize>,
    /// Draws of the block-permutation and matched-exposure nulls; 0 disables them.
    pub null_draws: usize,
    /// Total trials behind the candidates, for DSR; `None` uses the candidate count.
    pub trials: Option<usize>,
    /// Where `trials` comes from, named in the report.
    pub trials_source: &'static str,
    /// Seed of every protocol draw.
    pub seed: u32,
    /// Even block count of the CSCV PBO.
    pub pbo_blocks: usize,
    /// Benjamini-Hochberg false discovery rate.
    pub fdr_q: f64,
    /// #309: score every candidate on every window in one batch (device kernel
    /// when available) and report it beside the exact finalists.
    pub batch_pnl: bool,
    /// #1189: windows from a splits table: the pooled path starts flat at every window
    /// that does not touch the end of the previous one, and charges no seam there.
    pub flat_gaps: bool,
    /// #487: where the cost and the return units came from. `None` is the
    /// historical reading: the flag or default cost and simple returns.
    pub cost: Option<crate::target::CostDecl>,
    /// #482: the run.json the candidates came from.
    pub front: Option<FrontSource>,
    /// #807: which string of the run the candidates are; `None` for `--expr`.
    pub expression_source: Option<crate::run_json::ExpressionSource>,
    /// #482: labels of windows that overlap the run's training or validation
    /// rows and were allowed anyway.
    pub in_sample_windows: Vec<String>,
    /// #523: the final locked read reports the selected candidate only: no
    /// other candidate is evaluated on the test block, the inference has no
    /// candidate columns and `batch_pnl` is refused.
    pub final_read: bool,
    /// Half-open time ranges `[S, E)` that stay sealed in this run. A window that
    /// reaches `[S - P, E + max(L, P))` is refused and rows inside it are dropped at
    /// read time, with `P` the required purge and `L` `locked_embargo`. The sealed
    /// read lists the other declared ranges only.
    pub locked_ranges: Vec<(i64, i64)>,
    /// Seconds after the end of every locked range whose rows read it.
    pub locked_embargo: i64,
    /// The locked test of the run when a flag or the store declared it. It replaces
    /// the legacy 2026H2 box in the window guard; the declared ranges are enforced
    /// through `locked_ranges`.
    pub declared_lock: Option<crate::lockbox::Declared>,
    /// #925: how `--band` is read under `--policy band`. With
    /// [`crate::trading::BandUnits::Sd`] each window resolves the half-width of
    /// every candidate from its finite training scores only.
    pub band_units: crate::trading::BandUnits,
    /// #946: the band each candidate was priced with by the search that wrote
    /// `--from-run` or `--front`, in candidate order. A window never resolves
    /// another band from its rows.
    pub run_bands: Option<Vec<crate::trading::BandSpec>>,
    /// #930: quorum of `--policy vote`, in (0, 1].
    pub quorum: Option<f64>,
    /// `split.fit_label`: the label report facts of the target (see
    /// [`crate::labels::fit_label_json`]).
    pub fit_label: Option<String>,
    /// The horizon comes from a target spec, not from `--horizon` alone; the
    /// report names it as the annualisation source.
    pub horizon_from_spec: bool,
    /// `stride_seconds` of the target spec: the period width is the larger of the
    /// horizon and this.
    pub stride_seconds: Option<f64>,
    /// #1094: `--threshold-fit fitness`. Each window's cut is fitted on its
    /// validation slice (training rows without one); `threshold_fixed` is the
    /// reference cut and the fallback.
    pub threshold_fit: Option<CutFit>,
    /// #1180: the regime gate the run priced its book under (`--pnl-mask`),
    /// carried like the band. Rows with the gate off drop out of the test book
    /// and every on-run is its own book: it starts flat and pays its exit.
    pub mask: Option<crate::regime_gate::GateSpec>,
    /// #1175: a regime calendar that stratifies the pooled periods. A reporting input,
    /// never read by a search.
    pub regimes: Option<crate::regimes::Calendar>,
    /// #1175: causal regime expressions; each splits the pooled periods into on and off.
    pub regime_exprs: Vec<crate::regimes::ExprSpec>,
    /// #1175: emit exposure, entries, gross and buy-and-hold per period with the strata.
    pub period_series: bool,
    /// #1199: the periods of the pooled series (`--period-grid`).
    pub period_grid: PeriodGrid,
    /// #1203: what the book does on a row with a finite target and no finite
    /// score (`--missing-score`).
    pub missing_score: MissingScore,
    /// #1176: `--emit-series`: the pooled block adds the cost, entry-stamp and
    /// buy-and-hold net series and `entry_times` to the #1175 series.
    pub emit_series: bool,
    /// #1176: `--all-members`: every candidate of `--front` is also priced alone,
    /// with its own carried threshold and band, into `members`. Reporting only.
    pub all_members: bool,
}

/// #1199: which periods the pooled series of a report has. The grid never
/// changes the book: positions, costs and exits are the same under both rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PeriodGrid {
    /// A period where some row has a finite score and a finite target.
    #[default]
    Score,
    /// A period where some test row of the window has a finite target before
    /// the gate; a period where the book prices no row books net 0.
    Target,
}

impl PeriodGrid {
    pub fn parse(s: &str) -> Result<PeriodGrid, String> {
        match s {
            "score" => Ok(PeriodGrid::Score),
            "target" => Ok(PeriodGrid::Target),
            other => Err(format!("--period-grid needs score or target, got '{}'", other)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PeriodGrid::Score => "score",
            PeriodGrid::Target => "target",
        }
    }
}

/// #1199: the grid of the pooled series under `--period-grid target`.
#[derive(Clone, Debug, PartialEq)]
pub struct GridSummary {
    pub rule: PeriodGrid,
    /// Periods with no priced row (no finite score, or the gate off): net 0.
    pub empty_periods: usize,
    /// Priced rows of every period of the pooled series; 0 marks an empty period.
    pub period_rows: Vec<usize>,
}

/// #1094: the rule of `--threshold-fit fitness`: the cut with the lowest block
/// fitness among the cuts with at least `min_entry_times` distinct entry
/// timestamps on the source rows of a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutFit {
    /// `--cut-min-entries`.
    pub min_entry_times: u32,
    /// `--cut-blocks`: time blocks of the fitness.
    pub blocks: usize,
    /// `--cut-lambda`: weight of the block regret sd.
    pub lambda: f64,
}

impl Default for CutFit {
    fn default() -> Self {
        CutFit { min_entry_times: 30, blocks: 4, lambda: 1.0 }
    }
}

/// #1094: one cut of a window's sweep, priced on the source rows.
#[derive(Clone, Debug, PartialEq)]
pub struct CutPoint {
    /// Source rows at or below the cut.
    pub index: usize,
    pub value: f64,
    pub fitness: f64,
    pub net: f64,
    pub flips: u32,
    pub exits: u32,
    pub entry_times: u32,
    /// Long rows over source rows.
    pub exposure: f64,
}

/// #1094: the widest run of valid cuts around the fitted one whose fitness
/// stays within `0.1 * (1 - best)` of the best.
#[derive(Clone, Debug, PartialEq)]
pub struct CutPlateau {
    pub lo: f64,
    pub hi: f64,
    pub rows_lo: usize,
    pub rows_hi: usize,
}

/// #1094: the cut of one window under `--threshold-fit fitness`.
#[derive(Clone, Debug)]
pub struct CutReport {
    /// `ValidationFitness` or `TrainFitness`; `Fixed` when the window fell back.
    pub source: ThresholdSource,
    /// The threshold of the window.
    pub value: f64,
    /// Source rows with a finite score and target.
    pub rows: usize,
    /// The cut the window uses, when a sweep ran.
    pub cut: Option<CutPoint>,
    pub reference_value: f64,
    pub reference: Option<CutPoint>,
    pub plateau: Option<CutPlateau>,
    pub fit: CutFit,
    /// `no_rows`, `book_error`, `no_valid_cut` or `floor`.
    pub fallback: Option<&'static str>,
    pub fallback_detail: Option<String>,
    /// `(long-row share asked for, the cut)`, 32 rows.
    pub profile: Vec<(f64, CutPoint)>,
}

/// #482: where a candidate list came from: a run's Pareto front.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontSource {
    pub run: String,
    pub run_sha256: String,
    /// Archive rank label of every candidate, in candidate order.
    pub labels: Vec<String>,
    /// #1176: archive rank of every candidate, in candidate order.
    pub ranks: Vec<usize>,
}

impl Default for ProtocolOptions {
    fn default() -> Self {
        ProtocolOptions {
            entry_offset: 0.0,
            delay: 0.0,
            validation_starts: Vec::new(),
            bootstrap_draws: 0,
            block_len: None,
            null_draws: 0,
            trials: None,
            trials_source: "candidate_count",
            seed: 20_260_101,
            pbo_blocks: 16,
            fdr_q: 0.10,
            batch_pnl: false,
            flat_gaps: false,
            cost: None,
            front: None,
            expression_source: None,
            in_sample_windows: Vec::new(),
            final_read: false,
            locked_ranges: Vec::new(),
            locked_embargo: 0,
            declared_lock: None,
            band_units: crate::trading::BandUnits::Score,
            run_bands: None,
            quorum: None,
            fit_label: None,
            horizon_from_spec: false,
            stride_seconds: None,
            threshold_fit: None,
            mask: None,
            regimes: None,
            regime_exprs: Vec::new(),
            period_series: false,
            period_grid: PeriodGrid::Score,
            missing_score: MissingScore::Flat,
            emit_series: false,
            all_members: false,
        }
    }
}

impl ProtocolOptions {
    /// The cost block the report carries.
    pub fn cost_decl(&self, per_side_bps: f64) -> crate::target::CostDecl {
        self.cost.clone().unwrap_or(crate::target::CostDecl {
            per_side_bps,
            source: crate::target::CostSource::Default,
            return_units: "simple".to_string(),
            spec_per_side_bps: None,
        })
    }

    fn log_returns(&self) -> bool {
        self.cost.as_ref().is_some_and(|c| c.log_returns())
    }
}

impl ProtocolOptions {
    /// `entry_offset + horizon + delay`: the minimum distance from a training
    /// row to a test row so that no training label reaches the test decision.
    pub fn required_purge_seconds(&self, horizon: f64) -> f64 {
        self.entry_offset + horizon + self.delay
    }

    /// Whole seconds of purge in front of a sealed range: the label purge, or
    /// the spec embargo when that is larger.
    pub fn locked_purge_seconds(&self, spec: &EvalSpec) -> i64 {
        self.required_purge_seconds(spec.horizon).max(spec.embargo_seconds()).ceil() as i64
    }

    /// The seconds after the end of a sealed range that no row may read:
    /// `max(L, P)`.
    pub fn locked_embargo_seconds(&self, spec: &EvalSpec) -> i64 {
        self.locked_embargo.max(0).max(self.locked_purge_seconds(spec))
    }

    /// `(S, E, first, E + max(L, P))` of every sealed range with `P` the purge a
    /// training label needs. A row is read when `t < first` or `t >= E + max(L, P)`:
    /// the half-open predicate of `Lockbox::reaches`, so the row at `S - P` is
    /// kept. Stamps are whole seconds.
    pub fn locked_reach(&self, spec: &EvalSpec) -> Vec<(f64, f64, f64, f64)> {
        let purge = self.locked_purge_seconds(spec);
        let embargo = self.locked_embargo_seconds(spec);
        self.locked_ranges
            .iter()
            .map(|&(s, e)| {
                let range = crate::lockbox::Lockbox { start: s, end: e };
                (s as f64, e as f64, range.first_reaching(purge) as f64, (e + embargo) as f64)
            })
            .collect()
    }
}

/// One walk-forward window. Bounds are epoch seconds (or whatever numeric time
/// unit the time column uses); all ranges are half-open.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub label: String,
    pub train_start: f64,
    pub train_end: f64,
    pub test_start: f64,
    pub test_end: f64,
}

impl Window {
    /// Parse `LABEL,TRAIN_START,TRAIN_END,TEST_START,TEST_END`.
    ///
    /// A four-field form is accepted and gets an automatic label. Timestamps
    /// may be numbers or `YYYY-MM-DD` / `YYYY-MM-DDTHH:MM:SS`.
    pub fn parse(src: &str) -> Result<Self, String> {
        let parts: Vec<&str> = src.split(',').map(|p| p.trim()).collect();
        let (label, rest) = match parts.len() {
            5 => (parts[0].to_string(), &parts[1..]),
            4 => (format!("window-{}", src), &parts[..]),
            _ => {
                return Err(format!(
                    "window '{}' must be LABEL,TRAIN_START,TRAIN_END,TEST_START,TEST_END \
                     or the same four bounds without a label",
                    src
                ))
            }
        };
        let mut bounds = [0.0f64; 4];
        for (i, p) in rest.iter().enumerate() {
            bounds[i] = data::parse_time(p)
                .ok_or_else(|| format!("window '{}': cannot parse time '{}'", src, p))?;
        }
        Ok(Window {
            label,
            train_start: bounds[0],
            train_end: bounds[1],
            test_start: bounds[2],
            test_end: bounds[3],
        })
    }
}

/// The position policy. `Dp` keeps the threshold strategy as the reported book
/// and adds the exact hindsight DP as a diagnostic bound, never as a strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Threshold,
    Band,
    Dp,
    /// Per timestamp, the top `floor(n / 3)` finite scores are long (#493).
    TopTercile,
    /// #930: long when at least `ceil(quorum x K)` of the K expressions are long.
    Vote,
}

impl Policy {
    pub fn parse(src: &str) -> Result<Policy, String> {
        match src {
            "threshold" => Ok(Policy::Threshold),
            "band" => Ok(Policy::Band),
            "dp" | "hindsight" => Ok(Policy::Dp),
            "top-tercile" | "top_tercile" => Ok(Policy::TopTercile),
            "vote" => Ok(Policy::Vote),
            other => Err(format!(
                "unknown policy '{}'; expected threshold | band | dp | top-tercile | vote",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Threshold => "threshold",
            Policy::Band => "band",
            Policy::Dp => "dp",
            Policy::TopTercile => "top_tercile",
            Policy::Vote => "vote",
        }
    }
}

/// Everything the evaluator needs. Construct this from the CLI or directly
/// from Rust; [`run_path`] dispatches `.rcsb` vs CSV.
#[derive(Clone, Debug)]
pub struct EvalSpec {
    pub expression: String,
    /// Feature columns. Empty means every numeric column except the target,
    /// time and group columns.
    pub vars: Vec<String>,
    pub target: String,
    pub time: String,
    pub group: Option<String>,
    pub windows: Vec<Window>,
    /// Forward-label horizon in seconds. Also the minimum purge/embargo.
    pub horizon: f64,
    /// Extra embargo between the training end and the test start.
    pub embargo: Option<f64>,
    /// Cost per side in basis points.
    pub cost_bps: f64,
    /// Quantile of the training/validation score used as the long threshold.
    pub threshold_quantile: f64,
    /// Pre-registered fixed threshold. When set, it overrides the quantile and
    /// the report says `threshold_source = "fixed"`.
    pub threshold_fixed: Option<f64>,
    /// Optional trailing fraction of the training time range kept as validation.
    pub validation_frac: f64,
    /// Override the inferred annualisation (e.g. 365 for calendar days,
    /// 252 for trading days, 8760 for hourly bars).
    pub periods_per_year: Option<f64>,
    /// Extra candidate expressions (issue #124). When non-empty and larger
    /// than one, every window selects one candidate by its validation error;
    /// the test block is only scored after that choice is fixed. Requires
    /// `validation_frac > 0`.
    pub candidates: Vec<String>,
    /// #308: select among candidates with the band protocol instead of
    /// validation MSE. The screening score is the fused long-only reduction on
    /// validation rows; the admitted set is rescored on the full validation rows
    /// in exact CPU `f64` and the winner is chosen by
    /// `(net, turnover, complexity, index)`. Requires `validation_frac > 0` and
    /// more than one `--expr`.
    pub band_select: bool,
    /// #304: recompute each test block's long-only P&L through the fused
    /// canonical reduction (the device kernel when available, the identical
    /// CPU canonical path otherwise) and report it beside the scalar evaluator.
    /// The scalar evaluator remains the reported authority.
    pub device_pnl: bool,
    /// #310: seed of the matched-exposure random-long control. Recorded in the
    /// report; the same seed reproduces the same control.
    pub control_seed: u64,
    /// #306: threshold | band | dp. `dp` keeps the threshold book and adds the
    /// hindsight bound; it is never reported as the strategy's P&L.
    pub policy: Policy,
    /// #306: half-width `beta` of the no-trade band, in score units. Only used
    /// by [`Policy::Band`]; `0` reproduces the plain threshold rule.
    pub band_width: f64,
    /// #306: minimum consecutive long rows before a band exit is allowed
    /// (>= 1; `1` is the plain band).
    pub min_hold: usize,
    /// #306: run the band/cooldown and hindsight DP scans on the device. The
    /// device result is bit-identical to the CPU scan; when the device refuses,
    /// the CPU scan runs and the report names the reason.
    pub gpu: bool,
    /// Reset every pool seam to flat (the historical behaviour). The default
    /// carries the ending position of one test block into the next, so pooled
    /// total variation includes `|p_end - q_start|`.
    pub pool_reset_flat: bool,
    /// Permit a test window that overlaps the 2026H2 lockbox.
    /// `false` is the safe default: every runner refuses it.
    pub allow_lockbox: bool,
}

impl EvalSpec {
    /// The effective embargo: the larger of the horizon and `embargo`.
    pub fn embargo_seconds(&self) -> f64 {
        self.embargo.unwrap_or(self.horizon).max(self.horizon)
    }
}

/// A metric block, shared by per-window and pooled output.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    /// Annualised Sharpe ratio, using the sample standard deviation (ddof=1).
    pub sharpe: f64,
    /// Compound annual growth rate, from the period returns.
    pub cagr: f64,
    /// Most negative peak-to-trough equity drawdown (a negative number).
    pub max_drawdown: f64,
    /// Annualised Sortino ratio (#307). `NaN` when undefined.
    pub sortino: f64,
    /// Per-period log-growth from `ln_1p` (#307); `-inf` on ruin.
    pub log_growth: f64,
    /// Minimum acceptable return used by the Sortino denominator.
    pub sortino_mar: f64,
    /// Position changes per year (sum of |delta position| / years).
    pub turnover: f64,
    /// Fraction of test rows with a long position.
    pub exposure: f64,
    /// Fraction of long rows whose forward return is strictly positive.
    pub hit_rate: f64,
    /// Hit rate at which the average long trade breaks even after realised cost.
    pub break_even_hit_rate: f64,
    /// Number of position changes (an entry and its exit count as two).
    pub trades: usize,
    /// Number of flat-to-long transitions.
    pub entries: usize,
    /// Test rows in the window (after dropping non-finite score/target rows).
    pub rows: usize,
    /// Test rows with a long position.
    pub active_rows: usize,
    /// Distinct test timestamps used in the return series.
    pub periods: usize,
    /// Periods per year used for Sharpe and CAGR.
    pub periods_per_year: f64,
    /// Total compounded return over the window.
    pub total_return: f64,
    /// Mean net return per period (after costs).
    pub mean_return: f64,
    /// Mean realised cost per long row, in basis points.
    pub avg_cost_bps: f64,
    /// Sharpe of the gross (pre-cost) return series.
    pub gross_sharpe: f64,
    /// Total cost charged over the window, as a fraction of notional.
    pub total_cost: f64,
    /// Total traded notional, `sum |delta position|`.
    pub total_abs_delta: f64,
}

/// One evaluated window.
#[derive(Clone, Debug)]
pub struct WindowResult {
    pub window: Window,
    /// The expression chosen for this window (the only one when a single
    /// `--expr` was supplied).
    pub expression: String,
    /// Zero-based candidate index in the report's candidate list.
    pub candidate: usize,
    /// Validation mean squared error of the selected expression against the
    /// target, when a validation slice exists. This is the selection statistic;
    /// the test block never enters it.
    pub validation_error: Option<f64>,
    pub threshold: f64,
    /// #930: the threshold of each member under `--policy vote`; empty otherwise.
    pub member_thresholds: Vec<f64>,
    /// `train` or `validation`.
    pub threshold_source: String,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub dropped_rows: usize,
    /// #482: the window overlaps rows the run trained or selected on.
    pub in_sample: bool,
    /// Time step inferred from the test timestamps, when it was inferable.
    pub inferred_period_seconds: Option<f64>,
    pub metrics: Metrics,
    /// #304: fused canonical/device long-only P&L for this test block.
    pub device_pnl: Option<DevicePnlCheck>,
    /// #310: cash, buy-and-hold, constant and matched-exposure random long for
    /// this test block, priced on the same rows with the same costs.
    pub controls: Option<crate::baselines::ControlSet>,
    /// #306: band/cooldown carry check when the band policy was selected.
    pub band: Option<BandCheck>,
    /// #306: exact hindsight DP bound when the dp policy was selected.
    pub hindsight_bound: Option<HindsightBound>,
    /// #308: banded finalist provenance when `--select-band` was used.
    pub band_selection: Option<crate::selection::BandSelection>,
    /// Latest stamp of any row used for training or validation. Its label ends
    /// at most `required_purge_seconds` later, which stays below `test_start`.
    pub train_last_time: Option<f64>,
    /// #1094: the fitted cut under `--threshold-fit fitness`.
    pub cut: Option<CutReport>,
    /// #1236: under `--emit-series`, the distinct entry stamps of the window's pooled
    /// book; the windows of a report sum to `pooled.entry_times`.
    pub entry_times: Option<usize>,
}

/// #306: the band/cooldown book and its associative carry.
#[derive(Clone, Debug)]
pub struct BandCheck {
    pub band_width: f64,
    pub min_hold: usize,
    /// `carry2` for the 2-state band, `cooldown-K` for the K+1-state variant.
    pub carry: String,
    pub groups: usize,
    /// Additive net from the composed carry (including every final exit), plus
    /// the return of the held rows under `--missing-score hold` (#1203).
    pub net: f64,
    /// Scalar evaluator's sum of per-row net returns.
    pub scalar_net: f64,
    /// `cuda` when the device scan produced the carries, `cpu` otherwise.
    pub backend: String,
    /// Why `--gpu` fell back to the CPU scan; `None` without a fallback.
    pub fallback_reason: Option<String>,
    /// #925: the band of this window's candidate; `band_width` is its
    /// resolved half-width.
    pub spec: Option<crate::trading::BandSpec>,
}

/// #306: the exact hindsight DP, labelled as a bound and not a strategy.
#[derive(Clone, Debug)]
pub struct HindsightBound {
    /// Always `hindsight_bound`; kept explicit in the JSON.
    pub label: String,
    /// Exact `O(2n)` max-plus DP value.
    pub net: f64,
    /// The reported threshold book's net for the same rows.
    pub rule_net: f64,
    /// `net - rule_net`; always non-negative up to floating-point rounding.
    pub gap: f64,
    pub groups: usize,
    /// `rule_net <= net` (with a 1e-12 absolute tolerance for summation order).
    pub rule_le_bound: bool,
    /// `cuda` when the device scan produced the carries, `cpu` otherwise.
    pub backend: String,
    /// Why `--gpu` fell back to the CPU scan; `None` without a fallback.
    pub fallback_reason: Option<String>,
}

/// Run a #306 scan on the device when asked, else (or on refusal) on the CPU.
/// Returns the scan, the backend label and the fallback reason.
fn scan_with_backend<T>(
    gpu: bool,
    what: &str,
    device: impl FnOnce() -> Result<T, String>,
    cpu: impl FnOnce() -> Result<T, String>,
) -> (T, String, Option<String>) {
    let fallback_reason = if gpu {
        match device() {
            Ok(scan) => return (scan, "cuda".to_string(), None),
            Err(reason) => Some(reason),
        }
    } else {
        None
    };
    let scan = cpu().unwrap_or_else(|e| panic!("{} failed: {}", what, e));
    (scan, "cpu".to_string(), fallback_reason)
}

/// #304: one test block's fused long-only reduction, plus the two benchmarks.
#[derive(Clone, Debug)]
pub struct DevicePnlCheck {
    /// Host-side wall of the reduction call and its start; a host envelope, not CUDA events.
    pub host_s: f64,
    pub started: std::time::Instant,
    /// `cuda` when the device kernel ran, `cpu-canonical` when the identical
    /// CPU segment reduction ran (non-CUDA build or no device).
    pub backend: String,
    pub segment_rows: usize,
    pub tile_candidates: usize,
    pub strategy: crate::trading_gpu::PnlTotals,
    pub cash: crate::trading_gpu::PnlTotals,
    pub buy_and_hold: crate::trading_gpu::PnlTotals,
    /// Sum of the scalar evaluator's per-row net returns.
    pub cpu_net: f64,
    /// `strategy.net - cpu_net`; a screening check, not a reported P&L.
    pub net_delta: f64,
    /// Device totals are bit-identical to the canonical CPU reduction on the
    /// same inputs, counts included: a kernel fault check.
    pub bit_equal: bool,
    /// Device integer counts equal the scalar book's.
    pub counts_equal: bool,
    /// Rows whose position changes when the score is narrowed to f32.
    pub f32_flip_rows: u64,
}

/// #304: report-level summary of the fused reduction.
#[derive(Clone, Debug)]
pub struct DevicePnlSummary {
    pub backend: String,
    pub segment_rows: usize,
    pub tile_candidates: usize,
    pub windows: usize,
    pub rows: u64,
    pub max_abs_net_delta: f64,
    /// Windows whose device block ran on the CUDA kernel.
    pub device_windows: usize,
    pub bit_mismatch_windows: usize,
    pub count_mismatch_windows: usize,
    pub f32_flip_rows: u64,
}

/// The complete report for all windows plus the pooled series.
#[derive(Clone, Debug)]
pub struct Report {
    pub expression: String,
    pub target: String,
    pub variables: Vec<String>,
    pub time_column: String,
    pub group_by: Option<String>,
    pub windows: Vec<WindowResult>,
    pub pooled: Metrics,
    /// #926: pooled out-of-fold net return of every period and the period
    /// stamps, in the order of the pooled series.
    pub period_net: Vec<f64>,
    pub period_time: Vec<f64>,
    /// #1199: `Some` under `--period-grid target`; `None` is the score grid.
    pub period_grid: Option<GridSummary>,
    /// #1175: the strata of the pooled periods; `None` without `--regimes`, `--regime-expr` or `--period-series`.
    pub regimes: Option<crate::regimes::RegimeReport>,
    /// #1175: per-period exposure, entries, gross and buy-and-hold beside `period_net`.
    pub period_extra: Option<PeriodExtra>,
    /// #1176: print the `--emit-series` fields of `period_extra`.
    pub emit_series: bool,
    /// #1176: every front member priced alone (`--all-members`).
    pub members: Option<Vec<Member>>,
    /// #487: cost source and return units of the book.
    pub cost: crate::target::CostDecl,
    pub cost_bps: f64,
    pub threshold_quantile: f64,
    /// The pre-registered fixed threshold, when one was supplied.
    pub threshold_fixed: Option<f64>,
    /// #1094: `--threshold-fit fitness` and its rule.
    pub threshold_fit: Option<CutFit>,
    /// #306: position policy and its band parameters.
    pub policy: Policy,
    /// #930: quorum and votes needed of the members (`candidates`) under `--policy vote`.
    pub quorum: Option<f64>,
    pub votes_required: usize,
    pub band_width: f64,
    /// #925: how `band_width` is read.
    pub band_units: crate::trading::BandUnits,
    /// #946: `run` when the band is the one a run priced, else `flag`.
    pub band_source: &'static str,
    pub min_hold: usize,
    /// #306: `--gpu` was set for the band/DP scans.
    pub gpu: bool,
    pub validation_frac: f64,
    pub horizon: f64,
    pub embargo: f64,
    pub periods_per_year: f64,
    /// How the pooled annualisation was chosen.
    pub annualization: String,
    /// `target_spec`, `horizon_flag`, `explicit` or `inferred_labelled_rows`.
    pub annualization_source: &'static str,
    /// The period width in seconds; `None` for an explicit override without a declared horizon.
    pub period_seconds: Option<f64>,
    /// Candidate expressions in report order.
    pub candidates: Vec<String>,
    /// Which fibre selected the candidate(s). Always validation when more than
    /// one candidate is present; `test` is never a selection input.
    pub selected_on: String,
    /// #482: the run whose Pareto front supplied the candidates.
    pub front: Option<FrontSource>,
    /// #807: `exact` or `rounded` for candidates read from a run.json.
    pub expression_source: Option<crate::run_json::ExpressionSource>,
    /// #308: whether the band protocol selected the candidate.
    pub band_select: bool,
    /// True when no test row influenced candidate or threshold selection.
    pub test_used_for_selection: bool,
    /// Pooled seam policy and the extra cost the carried boundary charged.
    pub pool_reset_flat: bool,
    pub seam_cost: f64,
    pub seam_trades: usize,
    /// #1203: the missing-score rule of the book, the pooled exits it paid at
    /// the last priced row before a missing score, and the rows with a finite
    /// score and no finite target inside a run that no earlier return covers.
    pub missing_score: MissingScore,
    pub missing_score_exits: usize,
    pub unpriced_target_rows: usize,
    /// Fold sign test: computed only when folds are independent (no training
    /// interval overlaps another fold's test block). Expanding folds overlap by
    /// construction, so their sign test is reported as not applicable.
    pub sign_test: SignTest,
    /// `store-stream` or `csv-table`.
    pub source: String,
    pub memory_bound: String,
    /// #145: was the lockbox override explicitly set for this run?
    pub allow_lockbox: bool,
    /// The box the window guard covered: the declared lock, else the legacy 2026H2 box.
    pub guarded_lockbox: crate::lockbox::Lockbox,
    /// #304: fused long-only reduction summary when `--device-pnl` was set.
    pub device_pnl: Option<DevicePnlSummary>,
    /// #309: batch pass over all candidates and windows (`--batch-pnl`).
    pub batch_pnl: Option<crate::wf_batch::BatchSummary>,
    /// #284: run-scoped phase ledger (`io`, `pnl`, `host_other`, device-busy intervals). Not part of `json()`, which is the reproduction hash.
    pub ledger: crate::gpu_stats::RunLedger,
    /// Columns and bytes the store scan read; `None` on the table path. Not part of `json()`.
    pub scan: Option<ScanStats>,
    /// `entry_offset) + horizon + delay`.
    pub required_purge_seconds: f64,
    /// Smallest positive step between consecutive test rows of one group.
    pub min_group_step: Option<f64>,
    /// Smallest `test_start - train_end` over the windows.
    pub min_purge_seconds: f64,
    /// `split.fit_label` of the printed report; the reproduction hash leaves it out.
    pub fit_label: Option<String>,
    pub economics: crate::wf_protocol::Economics,
    pub inference: Option<crate::wf_protocol::Inference>,
    dropped_rows: usize,
}

/// #1176: one front member priced alone under `--all-members`: the report of
/// `--expr <member>` with the member's carried threshold and band. Nothing in
/// it feeds the selection of the parent report.
#[derive(Clone, Debug)]
pub struct Member {
    pub archive_rank: usize,
    pub label: String,
    pub complexity: usize,
    /// The expression has no input: the book takes one position on every row.
    pub constant: bool,
    pub report: Report,
}

/// A one-sided binomial sign test over fold expectancies.
#[derive(Clone, Debug, Default)]
pub struct SignTest {
    /// Folds whose mean net return is strictly positive.
    pub positive: usize,
    /// Folds with a finite mean net return.
    pub total: usize,
    /// `P(X >= positive)` under `Bin(total, 0.5)`, or `None` when the test is
    /// not applicable.
    pub p_value: Option<f64>,
    /// Why the test was or was not computed.
    pub method: String,
    pub applicable: bool,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Evaluate `spec` against a `.rcsb` store or a CSV table.
/// A target-column value in simple-return units: a `log` column is converted
/// with `exp_m1`, the rule of the search objective.
fn ret_in(opts: &ProtocolOptions, y: f64) -> f64 {
    if opts.log_returns() {
        y.exp_m1()
    } else {
        y
    }
}

pub fn run_path(path: &str, spec: &EvalSpec) -> Result<Report, String> {
    run_path_with(path, spec, &ProtocolOptions::default())
}

/// [`run_path`] with the protocol options.
pub fn run_path_with(path: &str, spec: &EvalSpec, opts: &ProtocolOptions) -> Result<Report, String> {
    let clock = std::time::Instant::now();
    if store::is_store_path(path) {
        let mut report = run_store_clocked(path, spec, opts, clock)?;
        close_ledger(&mut report.ledger, clock);
        Ok(report)
    } else {
        let t_read = std::time::Instant::now();
        let table = data::read_table(path)?;
        let read_s = t_read.elapsed().as_secs_f64();
        let mut report = run_table_clocked(&table, spec, "csv-table", opts, clock)?;
        report.ledger.record_stage(crate::gpu_stats::Stage::Io, read_s);
        close_ledger(&mut report.ledger, clock);
        Ok(report)
    }
}

/// #284: the ledger clock covers the whole call. Stage windows are measured;
/// `host_other` is the rest of the wall (layout, parsing, assembly).
fn close_ledger(ledger: &mut crate::gpu_stats::RunLedger, clock: std::time::Instant) {
    close_ledger_at(ledger, clock.elapsed().as_secs_f64());
}

fn close_ledger_at(ledger: &mut crate::gpu_stats::RunLedger, wall: f64) {
    let rest = wall - ledger.stage_sum_s();
    ledger.record_stage(crate::gpu_stats::Stage::HostOther, rest);
    // `s + (wall - s)` can land one ULP off `wall`; the booked sum is the wall then.
    ledger.set_wall(if rest > 0.0 { ledger.stage_sum_s() } else { wall });
}

/// Evaluate a resident table. The CLI's CSV path and tests use this.
pub fn run_table(table: &Table, spec: &EvalSpec, source: &str) -> Result<Report, String> {
    run_table_with(table, spec, source, &ProtocolOptions::default())
}

/// [`run_table`] with the protocol options.
pub fn run_table_with(
    table: &Table,
    spec: &EvalSpec,
    source: &str,
    opts: &ProtocolOptions,
) -> Result<Report, String> {
    let clock = std::time::Instant::now();
    let mut report = run_table_clocked(table, spec, source, opts, clock)?;
    close_ledger(&mut report.ledger, clock);
    Ok(report)
}

fn run_table_clocked(
    table: &Table,
    spec: &EvalSpec,
    source: &str,
    opts: &ProtocolOptions,
    clock: std::time::Instant,
) -> Result<Report, String> {
    let mut layout = resolve_layout(spec, &table.names, &table.numeric)?;
    add_mask_columns(&mut layout, &table.names, &table.numeric, opts)?;
    add_regime_columns(&mut layout, &table.names, &table.numeric, opts)?;
    let trees = parse_expressions(spec, &layout)?;
    validate_windows(spec, opts)?;

    let times = data::time_keys(table, layout.time);
    let group_ids = layout.group.map(|g| table_group_ids(table, g));
    let t_pnl = std::time::Instant::now();
    let mut acc = Accum::new(spec, opts, trees.len())?;
    acc.gate_base = layout.vars.len();
    acc.set_regimes(&layout);
    if opts.final_read {
        acc.pass = Pass::TrainOnly;
    }
    let mut features = Vec::with_capacity(layout.vars.len());
    for pass in 0..(if opts.final_read { 2 } else { 1 }) {
        if pass == 1 {
            acc.begin_test_pass(spec, &trees)?;
        }
    for r in 0..table.len() {
        let t = times.value(r);
        if !acc.wants(t) {
            continue;
        }
        features.clear();
        for &c in layout.vars.iter().chain(&layout.mask_cols).chain(&layout.regime_cols) {
            features.push(table.value(r, c));
        }
        let g = group_ids.as_ref().map_or(0, |ids| ids[r]);
        acc.add(t, g, &features, ret_in(opts, table.value(r, layout.target)), &trees);
    }
    }
    let mut report = acc.finish(
        clock,
        spec,
        opts,
        &layout,
        source,
        "O(test rows + one score per training row per window); CSV is resident",
        &trees,
    )?;
    report.ledger.record_stage(crate::gpu_stats::Stage::Pnl, t_pnl.elapsed().as_secs_f64());
    Ok(report)
}

/// Evaluate a `.rcsb` store directly, scanning it one row group at a time.
pub fn run_store(path: &str, spec: &EvalSpec) -> Result<Report, String> {
    run_store_with(path, spec, &ProtocolOptions::default())
}

/// [`run_store`] with the protocol options.
pub fn run_store_with(path: &str, spec: &EvalSpec, opts: &ProtocolOptions) -> Result<Report, String> {
    let clock = std::time::Instant::now();
    let mut report = run_store_clocked(path, spec, opts, clock)?;
    close_ledger(&mut report.ledger, clock);
    Ok(report)
}

/// What the store scan read (issue #1034): value columns of the projection and the bytes decoded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScanStats {
    pub store_columns: usize,
    pub named_variables: usize,
    pub projected_columns: usize,
    pub groups_read: usize,
    pub groups_skipped: usize,
    pub value_bytes: u64,
}

impl ScanStats {
    fn json(&self) -> String {
        format!(
            "{{\"source\":\"store\",\"store_columns\":{},\"named_variables\":{},\"projected_columns\":{},\"groups_read\":{},\"groups_skipped\":{},\"value_bytes\":{}}}",
            self.store_columns, self.named_variables, self.projected_columns, self.groups_read, self.groups_skipped, self.value_bytes
        )
    }
}

/// Positions in `layout.vars` that `n` names. Block bodies are not walked: their inputs are argument positions.
fn named_inputs(n: &Node, out: &mut Vec<usize>) {
    if n.op == expr::Op::Input && !out.contains(&(n.val as usize)) {
        out.push(n.val as usize);
    }
    for k in &n.kids {
        named_inputs(k, out);
    }
}

fn run_store_clocked(
    path: &str,
    spec: &EvalSpec,
    opts: &ProtocolOptions,
    clock: std::time::Instant,
) -> Result<Report, String> {
    let fs = FileStore::open(path)?;
    let names = fs.header().names();
    let numeric = fs.header().numeric();
    let mut layout = resolve_layout(spec, &names, &numeric)?;
    layout.time_i64 = fs.header().dtypes()[layout.time] == store::DType::I64;
    add_mask_columns(&mut layout, &names, &numeric, opts)?;
    add_regime_columns(&mut layout, &names, &numeric, opts)?;
    let trees = parse_expressions(spec, &layout)?;
    validate_windows(spec, opts)?;

    let mut acc = Accum::new(spec, opts, trees.len())?;
    acc.gate_base = layout.vars.len();
    acc.set_regimes(&layout);
    if opts.final_read {
        acc.pass = Pass::TrainOnly;
    }
    let header = fs.header();
    let t_pnl = std::time::Instant::now();
    let mut io_s = 0.0f64;
    let group_ids: Vec<u64> = match layout.group {
        Some(c) if !numeric[c] => header.levels(c).map_or_else(Vec::new, |l| l.iter().map(|s| label_group_id(s)).collect()),
        _ => Vec::new(),
    };
    let mut named: Vec<usize> = Vec::new();
    for tree in &trees {
        named_inputs(tree, &mut named);
    }
    named.retain(|&p| p < layout.vars.len());
    let time_f64 = layout.time_numeric && !layout.time_i64;
    let group_f64 = layout.group.filter(|&g| numeric[g]);
    let mut needed: Vec<usize> = named.iter().map(|&p| layout.vars[p]).collect();
    needed.extend(layout.mask_cols.iter().copied());
    needed.extend(layout.regime_cols.iter().copied());
    needed.push(layout.target);
    if time_f64 {
        needed.push(layout.time);
    }
    needed.extend(group_f64);
    let projection = store::Projection::from_indices(&fs, &needed)?;
    let at = |index: usize| -> usize {
        projection.columns().iter().position(|c| c.index == index).expect("projected column")
    };
    let mut var_at: Vec<(usize, usize)> = named.iter().map(|&p| (p, at(layout.vars[p]))).collect();
    for (k, &c) in layout.mask_cols.iter().enumerate() {
        var_at.push((layout.vars.len() + k, at(c)));
    }
    for (k, &c) in layout.regime_cols.iter().enumerate() {
        var_at.push((layout.vars.len() + layout.mask_cols.len() + k, at(c)));
    }
    let target_at = at(layout.target);
    let time_at = time_f64.then(|| at(layout.time));
    let group_at = group_f64.map(at);
    let mut scan = ScanStats {
        store_columns: header.n_cols,
        named_variables: named.len(),
        projected_columns: projection.len(),
        ..ScanStats::default()
    };
    let mut pg = store::ProjectedGroup::new();
    // Positions no candidate names stay 0.0: `expr::eval` reads a row only at `Input` nodes.
    let mut features = vec![0.0f64; layout.vars.len() + layout.mask_cols.len() + layout.regime_cols.len()];
    let time_levels = header.levels(layout.time);
    let time_of_code = |code: u32| -> f64 {
        time_levels
            .and_then(|l| l.get(code as usize))
            .and_then(|s| data::parse_time(s))
            .unwrap_or(f64::NAN)
    };
    for pass in 0..(if opts.final_read { 2 } else { 1 }) {
        if pass == 1 {
            acc.begin_test_pass(spec, &trees)?;
        }
    for group in 0..fs.n_groups() {
        let t_io = std::time::Instant::now();
        let len = header.group_len(group);
        let time_i64 = if layout.time_i64 {
            Some(fs.read_i64(group, layout.time)?)
        } else {
            None
        };
        let time_codes = if layout.time_numeric || layout.time_i64 {
            None
        } else {
            Some(fs.read_codes(group, layout.time)?)
        };
        // An exact i64 or dictionary time is known before the value read; an f64 time is part of it.
        let known_unwanted = match (&time_i64, &time_codes) {
            (Some(ints), _) => !ints.iter().any(|&i| acc.wants(i as f64)),
            (None, Some(codes)) => !codes.iter().any(|&c| acc.wants(time_of_code(c))),
            (None, None) => false,
        };
        if known_unwanted {
            io_s += t_io.elapsed().as_secs_f64();
            scan.groups_skipped += 1;
            continue;
        }
        let stats = fs.read_group_projected(group, &projection, &mut pg)?;
        scan.groups_read += stats.groups;
        scan.value_bytes += stats.value_bytes;
        let group_codes = match layout.group {
            Some(c) if !numeric[c] => Some(fs.read_codes(group, c)?),
            _ => None,
        };
        io_s += t_io.elapsed().as_secs_f64();
        let var_cols: Vec<(usize, &[f64])> = var_at.iter().map(|&(p, c)| (p, pg.col(c))).collect();
        let target_col = pg.col(target_at);
        for r in 0..len {
            let t = match (&time_i64, &time_codes) {
                (Some(ints), _) => ints.get(r).copied().unwrap_or(i64::MIN) as f64,
                (None, None) => pg.col(time_at.expect("f64 time"))[r],
                (None, Some(codes)) => time_of_code(codes.get(r).copied().unwrap_or(0)),
            };
            if !acc.wants(t) {
                continue;
            }
            for &(p, col) in &var_cols {
                features[p] = col[r];
            }
            let g = match (layout.group, &group_codes) {
                (None, _) => 0,
                (Some(_), Some(codes)) => {
                    let code = codes.get(r).copied().unwrap_or(0) as usize;
                    group_ids.get(code).copied().unwrap_or(label_group_id(""))
                }
                (Some(_), None) => group_key(pg.col(group_at.expect("numeric group"))[r]),
            };
            let y = ret_in(opts, target_col[r]);
            acc.add(t, g, &features, y, &trees);
        }
    }
    }
    let bound = format!(
        "O(test rows + one score per training row per window); the .rcsb store is scanned \
         group by group, one {} -row group resident",
        fs.group_rows()
    );
    let mut report = acc.finish(clock, spec, opts, &layout, "store-stream", &bound, &trees)?;
    report.scan = Some(scan);
    report.ledger.record_stage(crate::gpu_stats::Stage::Io, io_s);
    report.ledger.record_stage(crate::gpu_stats::Stage::Pnl, t_pnl.elapsed().as_secs_f64() - io_s);
    Ok(report)
}

// ---------------------------------------------------------------------------
// Layout and expression validation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Layout {
    target: usize,
    time: usize,
    time_numeric: bool,
    group: Option<usize>,
    vars: Vec<usize>,
    var_names: Vec<String>,
    /// The store keeps the time column as `i64`; read it exactly rather than
    /// through the `f64` value block (which is absent).
    time_i64: bool,
    /// #1180: columns of the regime gate, read after the variables in every
    /// feature vector.
    mask_cols: Vec<usize>,
    /// #1175: the union of the columns of the regime expressions, read after the mask
    /// columns, and for every expression the position of each of its columns in it.
    regime_cols: Vec<usize>,
    regime_slots: Vec<Vec<usize>>,
}

/// #1175: the table columns of the regime expressions of `opts.regime_exprs`.
fn add_regime_columns(layout: &mut Layout, names: &[String], numeric: &[bool], opts: &ProtocolOptions) -> Result<(), String> {
    for e in &opts.regime_exprs {
        let mut slots = Vec::new();
        for name in &e.gate.columns {
            let j = names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| format!("the regime expression {} reads column '{}', which is not in the data", e.name, name))?;
            if !numeric.get(j).copied().unwrap_or(false) {
                return Err(format!("the regime expression {} column '{}' must be numeric", e.name, name));
            }
            slots.push(match layout.regime_cols.iter().position(|&c| c == j) {
                Some(s) => s,
                None => {
                    layout.regime_cols.push(j);
                    layout.regime_cols.len() - 1
                }
            });
        }
        layout.regime_slots.push(slots);
    }
    Ok(())
}

/// #1180: the table columns of the gate of `opts.mask`.
fn add_mask_columns(layout: &mut Layout, names: &[String], numeric: &[bool], opts: &ProtocolOptions) -> Result<(), String> {
    let Some(gate) = &opts.mask else { return Ok(()) };
    for name in &gate.columns {
        let j = names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("the run's regime mask reads column '{}', which is not in the data (pass --unmasked to price the raw formula)", name))?;
        if !numeric.get(j).copied().unwrap_or(false) {
            return Err(format!("the run's regime mask column '{}' must be numeric", name));
        }
        layout.mask_cols.push(j);
    }
    Ok(())
}

fn resolve_layout(spec: &EvalSpec, names: &[String], numeric: &[bool]) -> Result<Layout, String> {
    let find = |name: &str| -> Result<usize, String> {
        names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("column '{}' not found; available: {:?}", name, names))
    };
    let target = find(&spec.target)?;
    if !numeric.get(target).copied().unwrap_or(false) {
        return Err(format!("target '{}' must be numeric", spec.target));
    }
    let time = find(&spec.time)?;
    let group = match &spec.group {
        Some(name) => Some(find(name)?),
        None => None,
    };
    let vars: Vec<usize> = if spec.vars.is_empty() {
        (0..names.len())
            .filter(|&j| {
                numeric.get(j).copied().unwrap_or(false)
                    && j != target
                    && j != time
                    && Some(j) != group
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(spec.vars.len());
        for name in &spec.vars {
            out.push(find(name)?);
        }
        out
    };
    if vars.is_empty() {
        return Err("no feature columns: pass --vars or add numeric columns".to_string());
    }
    for &v in &vars {
        if !numeric.get(v).copied().unwrap_or(false) {
            return Err(format!(
                "feature '{}' must be numeric (walk-forward expressions are f64)",
                names[v]
            ));
        }
    }
    let var_names = vars.iter().map(|&j| names[j].clone()).collect();
    Ok(Layout {
        target,
        time,
        time_numeric: numeric.get(time).copied().unwrap_or(false),
        group,
        vars,
        var_names,
        time_i64: false,
        mask_cols: Vec::new(),
        regime_cols: Vec::new(),
        regime_slots: Vec::new(),
    })
}

fn parse_expressions(spec: &EvalSpec, layout: &Layout) -> Result<Vec<Node>, String> {
    let mut out = Vec::new();
    for src in candidate_expressions(spec) {
        let tree = expr::parse(&src, &layout.var_names)
            .map_err(|e| format!("cannot parse expression '{}': {}", src, e))?;
        if has_dataset_op(&tree) {
            return Err(
                "walk-forward expressions cannot use dataset aggregates or meta operators; \
                 the threshold must depend on the window, not on a whole-column statistic"
                    .to_string(),
            );
        }
        out.push(tree);
    }
    Ok(out)
}

fn has_dataset_op(n: &Node) -> bool {
    n.op.is_agg() || n.op.is_meta() || n.kids.iter().any(has_dataset_op)
}

/// The refusal message when `--policy top-tercile` is combined with a flag it has no
/// meaning for. The CLI reports it as a usage error (#832).
pub fn top_tercile_refusal(spec: &EvalSpec, opts: &ProtocolOptions) -> Option<String> {
    let refused = [
        (spec.threshold_fixed.is_some(), "--threshold-fixed"),
        (spec.band_width != 0.0, "--band"),
        (spec.min_hold != 1, "--min-hold"),
        (spec.band_select, "--select-band"),
        (spec.device_pnl, "--device-pnl"),
        (opts.batch_pnl, "--batch-pnl"),
    ];
    refused.iter().find(|(set, _)| *set).map(|(_, flag)| {
        format!(
            "{} is not supported with --policy top-tercile: the book has no threshold, band or device arm",
            flag
        )
    })
}

/// #1094: the refusal message when `--threshold-fit fitness` meets a policy or
/// flag whose book is not the plain CPU threshold book. The CLI reports it as a
/// usage error.
pub fn threshold_fit_conflict(policy: Policy, band_select: bool, device_pnl: bool, batch_pnl: bool) -> Option<String> {
    if policy != Policy::Threshold {
        return Some(format!(
            "--threshold-fit fitness prices the plain threshold book; --policy {} is refused",
            policy.as_str()
        ));
    }
    let refused = [(band_select, "--select-band"), (device_pnl, "--device-pnl"), (batch_pnl, "--batch-pnl")];
    refused.iter().find(|(set, _)| *set).map(|(_, flag)| {
        format!("--threshold-fit fitness is not supported with {}: the fitted cut prices the CPU threshold book only", flag)
    })
}

/// #1094: why `--threshold-fit fitness` cannot run with this spec, or `None`.
pub fn threshold_fit_refusal(spec: &EvalSpec, opts: &ProtocolOptions) -> Option<String> {
    let fit = opts.threshold_fit.as_ref()?;
    if fit.min_entry_times == 0 {
        return Some("--threshold-fit fitness: --cut-min-entries must be at least 1".to_string());
    }
    if fit.blocks == 0 || fit.blocks > crate::trading::FITNESS_MAX_BLOCKS {
        return Some(format!(
            "--threshold-fit fitness: --cut-blocks must be 1 to {}, got {}",
            crate::trading::FITNESS_MAX_BLOCKS,
            fit.blocks
        ));
    }
    if !fit.lambda.is_finite() || fit.lambda < 0.0 {
        return Some(format!("--threshold-fit fitness: --cut-lambda must be finite and >= 0, got {}", fit.lambda));
    }
    if let Some(msg) = threshold_fit_conflict(spec.policy, spec.band_select, spec.device_pnl, opts.batch_pnl) {
        return Some(msg);
    }
    if spec.threshold_fixed.is_none() {
        return Some("--threshold-fit fitness needs a reference cut: pass --threshold-fixed or --from-run".to_string());
    }
    None
}

/// #1176: the refusal of `--all-members` for these inputs, or `None`. The CLI
/// reports it as a usage error before any data work.
pub fn all_members_conflict(front: bool, band_select: bool, threshold_fit: bool, vote: bool, locked_read: bool) -> Option<String> {
    if !front {
        return Some("--all-members needs --front: it prices every member of a run's Pareto front".to_string());
    }
    if locked_read {
        return Some("--all-members refuses --locked-test: a locked read reports the sealed candidate only".to_string());
    }
    let refused = [(band_select, "--select-band"), (threshold_fit, "--threshold-fit"), (vote, "--policy vote")];
    refused.iter().find(|(set, _)| *set).map(|(_, flag)| {
        format!("--all-members refuses {}: it prices every front member with its carried threshold and band and selects none", flag)
    })
}

/// #1176: why `--all-members` cannot run with this spec, or `None`.
pub fn all_members_refusal(spec: &EvalSpec, opts: &ProtocolOptions) -> Option<String> {
    if !opts.all_members {
        return None;
    }
    all_members_conflict(opts.front.is_some(), spec.band_select, opts.threshold_fit.is_some(), spec.policy == Policy::Vote, opts.final_read)
}

/// Votes needed for a long row: `ceil(quorum x K)`, at least 1.
pub fn votes_required(quorum: f64, members: usize) -> usize {
    ((quorum * members as f64 - 1e-9).ceil().max(1.0) as usize).min(members.max(1))
}

/// #930: the refusal message when `--policy vote` is combined with a flag or
/// value it cannot honour. The CLI reports it as a usage error.
pub fn vote_refusal(spec: &EvalSpec, opts: &ProtocolOptions) -> Option<String> {
    let refused = [
        (spec.device_pnl, "--device-pnl", "the device arm prices the threshold book"),
        (opts.batch_pnl, "--batch-pnl", "the batch arm scores one expression per column"),
        (spec.band_select, "--select-band", "the members are all used, none is selected"),
        (opts.run_bands.is_some(), "--from-run/--front bands", "the band counts votes"),
        (opts.band_units != crate::trading::BandUnits::Score, "--band-units", "the band counts votes"),
    ];
    if let Some((_, flag, why)) = refused.iter().find(|(set, _, _)| *set) {
        return Some(format!("{} is not supported with --policy vote: {}", flag, why));
    }
    match opts.quorum {
        Some(q) if q > 0.0 && q <= 1.0 => {}
        _ => return Some("--policy vote needs --quorum in (0, 1]".to_string()),
    }
    if spec.band_width.fract() != 0.0 {
        return Some("--band counts votes under --policy vote: pass a whole number".to_string());
    }
    None
}

/// #930: replace the member score columns of the test block by one vote count
/// per row (NaN when any member is non-finite, so the shared valid-row mask
/// drops the row) and return the threshold that reads it. The thresholds come
/// from the training or validation rows through `select_candidate`; no test
/// value enters them. A long book stays long while `votes >= need - band_votes`.
fn apply_vote(d: &mut WindowData, thresholds: &[f64], need: usize, band_votes: f64) -> f64 {
    let n = d.test_rows.len();
    let votes: Vec<f64> = (0..n)
        .map(|i| {
            let mut count = 0usize;
            for (scores, &theta) in d.test_scores.iter().zip(thresholds) {
                let s = scores[i];
                if !s.is_finite() {
                    return f64::NAN;
                }
                count += usize::from(s > theta);
            }
            count as f64
        })
        .collect();
    d.test_scores = vec![votes];
    need as f64 - 0.5 - band_votes / 2.0
}

fn validate_windows(spec: &EvalSpec, opts: &ProtocolOptions) -> Result<(), String> {
    check_explicit_ppy(spec, opts)?;
    if spec.windows.is_empty() {
        return Err("at least one --window is required".to_string());
    }
    if !(0.0..=1.0).contains(&spec.threshold_quantile) {
        return Err("--threshold-quantile must be in [0, 1]".to_string());
    }
    if let Some(theta) = spec.threshold_fixed {
        if !theta.is_finite() {
            return Err("--threshold-fixed must be finite".to_string());
        }
    }
    if !(0.0..1.0).contains(&spec.validation_frac) {
        return Err("--validation-frac must be in [0, 1)".to_string());
    }
    if !spec.cost_bps.is_finite() || spec.cost_bps < 0.0 {
        return Err("--cost-bps must be finite and non-negative".to_string());
    }
    if !spec.band_width.is_finite() || spec.band_width < 0.0 {
        return Err("--band must be finite and non-negative".to_string());
    }
    if spec.min_hold == 0 {
        return Err("--min-hold must be at least 1".to_string());
    }
    if let Some(bands) = &opts.run_bands {
        let n = spec.candidates.len().max(1);
        if bands.len() != n {
            return Err(format!("the run carries {} bands for {} candidates", bands.len(), n));
        }
    }
    if let Some(msg) = all_members_refusal(spec, opts) {
        return Err(msg);
    }
    if spec.policy == Policy::TopTercile {
        if let Some(msg) = top_tercile_refusal(spec, opts) {
            return Err(msg);
        }
    }
    if spec.policy == Policy::Band && spec.device_pnl {
        return Err("--device-pnl is not supported with --policy band: the device arm prices the threshold book".to_string());
    }
    if spec.policy == Policy::Vote {
        if let Some(msg) = vote_refusal(spec, opts) {
            return Err(msg);
        }
    }
    if let Some(msg) = threshold_fit_refusal(spec, opts) {
        return Err(msg);
    }
    let table_validation = !opts.validation_starts.is_empty()
        && spec.windows.iter().all(|w| opts.validation_starts.iter().any(|(l, _)| *l == w.label));
    if spec.candidates.len() > 1 && spec.policy != Policy::Vote && spec.validation_frac <= 0.0 && !table_validation && !opts.all_members {
        return Err(
            "multiple --expr candidates need a validation slice (--validation-frac > 0 or a windows \
             table with validation_start): select on a validation slice, never on the test block"
                .to_string(),
        );
    }
    if spec.candidates.iter().any(|e| e.trim().is_empty()) {
        return Err("candidate expressions must be non-empty".to_string());
    }
    let required = opts.required_purge_seconds(spec.horizon);
    if !(opts.entry_offset.is_finite() && opts.entry_offset >= 0.0 && opts.delay.is_finite() && opts.delay >= 0.0) {
        return Err("--entry-offset and --delay must be finite and non-negative".to_string());
    }
    if opts.bootstrap_draws > 0 && opts.bootstrap_draws < 100 {
        return Err("--bootstrap needs at least 100 draws".to_string());
    }
    if opts.final_read && opts.batch_pnl {
        return Err(
            "--batch-pnl scores every candidate on the locked test; a final read reports the selected \
             candidate only"
                .to_string(),
        );
    }
    if opts.pbo_blocks < 2 || opts.pbo_blocks % 2 != 0 || opts.pbo_blocks > 24 {
        return Err("--pbo-blocks must be even and in 2..=24".to_string());
    }
    if !(opts.fdr_q > 0.0 && opts.fdr_q < 1.0) {
        return Err("--fdr-q must be in (0, 1)".to_string());
    }
    if !opts.validation_starts.is_empty() && spec.validation_frac > 0.0 {
        return Err("explicit validation starts and --validation-frac are mutually exclusive".to_string());
    }
    let embargo = spec.embargo_seconds().max(required);
    for w in &spec.windows {
        if !(w.train_start < w.train_end) {
            return Err(format!(
                "window '{}': train_start must be before train_end",
                w.label
            ));
        }
        if !(w.test_start < w.test_end) {
            return Err(format!(
                "window '{}': test_start must be before test_end",
                w.label
            ));
        }
        // #145: a training range inside 2026H2 with a 2027 test would train on
        // the holdout. Check both sides through the one central guard.
        let guard = crate::lockbox::Guard::for_run(opts.declared_lock.as_ref(), spec.allow_lockbox);
        guard.check(
            &format!("window '{}' training range", w.label),
            w.train_start.floor() as i64,
            w.train_end.ceil() as i64,
        )?;
        guard.check(
            &format!("window '{}' test range", w.label),
            w.test_start.floor() as i64,
            w.test_end.ceil() as i64,
        )?;
        for (ls, le, lo, hi) in opts.locked_reach(spec) {
            let side = if w.train_start < hi && w.train_end > lo {
                Some("training")
            } else if w.test_start < hi && w.test_end > lo {
                Some("test")
            } else {
                None
            };
            if let Some(side) = side {
                return Err(format!(
                    "window '{}': the {} range reaches the declared locked test {}..{} (label end = stamp + {} s, \
                     lookback embargo {} s after the end); end it at or before {} or start it at or after {}, \
                     or read the locked test only with --locked-test and a sealed list",
                    w.label,
                    side,
                    crate::protocol::iso(ls),
                    crate::protocol::iso(le),
                    opts.locked_purge_seconds(spec),
                    hi - le,
                    crate::protocol::iso(lo),
                    crate::protocol::iso(hi)
                ));
            }
        }
        let gap = w.test_start - w.train_end;
        if gap + 1e-9 < embargo {
            return Err(format!(
                "window '{}': purge gap is {} seconds but entry offset + horizon + delay / embargo \
                 needs {}; move test_start later or lower --horizon/--embargo/--entry-offset/--delay",
                w.label, gap, embargo
            ));
        }
    }
    let mut order: Vec<&Window> = spec.windows.iter().collect();
    order.sort_by(|a, b| a.test_start.partial_cmp(&b.test_start).unwrap_or(std::cmp::Ordering::Equal));
    for pair in order.windows(2) {
        if pair[1].test_start + 1e-9 < pair[0].test_end {
            return Err(format!(
                "windows '{}' and '{}' overlap in the test region; walk-forward test blocks \
                 must be disjoint and chronological",
                pair[0].label, pair[1].label
            ));
        }
    }
    Ok(())
}

fn group_key(v: f64) -> u64 {
    if v.is_nan() {
        u64::MAX
    } else {
        (v + 0.0).to_bits()
    }
}

/// Group id of a text label: it depends on the label alone, never on the other
/// rows of the file, so a label that exists only elsewhere cannot move it.
fn label_group_id(text: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    0x8000_0000_0000_0000 | h
}

fn table_group_ids(table: &Table, col: usize) -> Vec<u64> {
    // A numeric key is its value on both readers; the dictionary only keeps its source text.
    if table.numeric[col] {
        return table.col(col).iter().map(|&v| group_key(v)).collect();
    }
    if let Some(dict) = table.dict(col) {
        let ids: Vec<u64> = dict.levels.iter().map(|l| label_group_id(l)).collect();
        let missing = label_group_id("");
        dict.codes.iter().map(|&code| ids.get(code as usize).copied().unwrap_or(missing)).collect()
    } else {
        table.col(col).iter().map(|&v| group_key(v)).collect()
    }
}

// ---------------------------------------------------------------------------
// Accumulation
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RowBase {
    time: f64,
    group: u64,
    target: f64,
}

struct WindowData {
    window: Window,
    validation_start: Option<f64>,
    /// One score vector per candidate, aligned with the validation/test rows.
    train_scores: Vec<Vec<f64>>,
    /// #925: latest time of a row in `train_scores`, per candidate.
    train_score_max_time: Vec<f64>,
    validation_rows: Vec<RowBase>,
    validation_scores: Vec<Vec<f64>>,
    test_rows: Vec<RowBase>,
    /// #1180: the gate value of every test row (NaN without a mask).
    test_gate: Vec<f64>,
    /// #1175: the value of every regime expression on every test row, one vector per expression.
    test_regime: Vec<Vec<f64>>,
    test_scores: Vec<Vec<f64>>,
    /// Sum and count of the finite training-interval targets (#310 constant
    /// control).
    train_target_sum: f64,
    train_target_n: u64,
    /// Latest stamp of any row that entered training or validation.
    train_max_time: f64,
    /// #1176: training rows outside the validation slice, scored or not.
    train_seen: usize,
    dropped: usize,
    /// #1094: the training rows and every candidate's score on them (NaN kept),
    /// the source of a fitted cut in a window without a validation slice.
    keep_train_rows: bool,
    train_rows: Vec<RowBase>,
    train_row_scores: Vec<Vec<f64>>,
}

/// Which rows an [`Accum`] scores. A final read scores in two passes: the
/// training and validation rows of every candidate first, then, once the
/// candidate of each window is fixed, the test rows of that candidate only.
#[derive(Clone, Debug, PartialEq)]
enum Pass {
    All,
    TrainOnly,
    TestOnly,
}

struct Accum {
    /// Sealed ranges with the purge before and the embargo after: rows that reach
    /// one are never read.
    locked: Vec<crate::lockbox::Lockbox>,
    locked_purge: i64,
    locked_embargo: i64,
    windows: Vec<WindowData>,
    candidates: usize,
    scores: Vec<f64>,
    pass: Pass,
    /// Candidate of each window, fixed from training and validation rows
    /// before any test row is scored (final read only).
    forced: Option<Vec<usize>>,
    /// Candidates scored on the test pass (final read only).
    test_candidates: Vec<usize>,
    /// `--policy vote`: the final read scores every member on the test rows.
    vote: bool,
    threshold_fit: Option<CutFit>,
    /// #1180: the regime gate and the feature index of its first column.
    gate: Option<crate::regime_gate::GateSpec>,
    gate_base: usize,
    /// #1175: the regime expressions, the position of their columns among the regime
    /// columns and the feature index of the first one.
    regime_exprs: Vec<crate::regimes::ExprSpec>,
    regime_slots: Vec<Vec<usize>>,
    regime_base: usize,
}

impl Accum {
    fn new(spec: &EvalSpec, opts: &ProtocolOptions, candidates: usize) -> Result<Self, String> {
        // Report and pool in test-time order even if the windows were written
        // out of order on the command line. Validation already proved that the
        // test blocks do not overlap.
        let mut ordered: Vec<&Window> = spec.windows.iter().collect();
        ordered.sort_by(|a, b| {
            a.test_start
                .partial_cmp(&b.test_start)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.label.cmp(&b.label))
        });
        let mut windows = Vec::with_capacity(ordered.len());
        for w in ordered {
            let validation_start = if let Some((_, s)) = opts.validation_starts.iter().find(|(l, _)| *l == w.label) {
                if !(*s > w.train_start && *s < w.train_end) {
                    return Err(format!(
                        "window '{}': validation start must lie inside the training range",
                        w.label
                    ));
                }
                Some(*s)
            } else if spec.validation_frac > 0.0 {
                let span = w.train_end - w.train_start;
                let start = w.train_end - span * spec.validation_frac;
                if !(start > w.train_start) {
                    return Err(format!(
                        "window '{}': --validation-frac leaves no training rows",
                        w.label
                    ));
                }
                Some(start)
            } else {
                None
            };
            windows.push(WindowData {
                window: w.clone(),
                validation_start,
                train_scores: vec![Vec::new(); candidates],
                train_score_max_time: vec![f64::NEG_INFINITY; candidates],
                validation_rows: Vec::new(),
                validation_scores: vec![Vec::new(); candidates],
                test_rows: Vec::new(),
                test_gate: Vec::new(),
                test_regime: vec![Vec::new(); opts.regime_exprs.len()],
                test_scores: vec![Vec::new(); candidates],
                train_target_sum: 0.0,
                train_target_n: 0,
                train_max_time: f64::NEG_INFINITY,
                train_seen: 0,
                dropped: 0,
                keep_train_rows: opts.threshold_fit.is_some() && validation_start.is_none(),
                train_rows: Vec::new(),
                train_row_scores: vec![Vec::new(); candidates],
            });
        }
        Ok(Accum {
            locked: opts.locked_ranges.iter().map(|&(start, end)| crate::lockbox::Lockbox { start, end }).collect(),
            locked_purge: opts.locked_purge_seconds(spec),
            locked_embargo: opts.locked_embargo_seconds(spec),
            windows,
            candidates,
            scores: vec![0.0; candidates],
            pass: Pass::All,
            forced: None,
            test_candidates: Vec::new(),
            vote: spec.policy == Policy::Vote,
            threshold_fit: opts.threshold_fit,
            gate: opts.mask.clone(),
            gate_base: 0,
            regime_exprs: opts.regime_exprs.clone(),
            regime_slots: Vec::new(),
            regime_base: 0,
        })
    }

    fn set_regimes(&mut self, layout: &Layout) {
        self.regime_slots = layout.regime_slots.clone();
        self.regime_base = layout.vars.len() + layout.mask_cols.len();
    }

    fn wants(&self, t: f64) -> bool {
        if !t.is_finite() {
            return false;
        }
        let stamp = t.floor().clamp(-1e15, 1e15) as i64;
        if self.locked.iter().any(|r| r.reaches(stamp, self.locked_purge, self.locked_embargo)) {
            return false;
        }
        self.windows.iter().any(|d| {
            let train = t >= d.window.train_start && t < d.window.train_end;
            let test = t >= d.window.test_start && t < d.window.test_end;
            match self.pass {
                Pass::All => train || test,
                Pass::TrainOnly => train,
                Pass::TestOnly => test,
            }
        })
    }

    fn add(&mut self, t: f64, group: u64, features: &[f64], target: f64, trees: &[Node]) {
        debug_assert_eq!(trees.len(), self.candidates);
        if !self.regime_exprs.is_empty() && self.pass != Pass::TrainOnly {
            let values: Vec<f64> = self
                .regime_exprs
                .iter()
                .zip(&self.regime_slots)
                .map(|(e, slots)| {
                    let row: Vec<f64> = slots.iter().map(|&s| features[self.regime_base + s]).collect();
                    expr::eval(&e.gate.tree, &row)
                })
                .collect();
            for d in &mut self.windows {
                if t >= d.window.test_start && t < d.window.test_end {
                    for (dst, &v) in d.test_regime.iter_mut().zip(&values) {
                        dst.push(v);
                    }
                }
            }
        }
        let gate = match &self.gate {
            Some(g) => {
                let row: Vec<f64> = (0..g.columns.len()).map(|k| features[self.gate_base + k]).collect();
                expr::eval(&g.tree, &row)
            }
            None => f64::NAN,
        };
        match self.pass {
            Pass::All | Pass::TrainOnly => {
                for (s, tree) in self.scores.iter_mut().zip(trees) {
                    *s = expr::eval(tree, features);
                }
                push_scores(&mut self.windows, t, group, (target, gate), &self.scores, self.pass == Pass::All);
            }
            Pass::TestOnly => {
                if self.vote {
                    for (s, tree) in self.scores.iter_mut().zip(trees) {
                        *s = expr::eval(tree, features);
                    }
                    push_all_test(&mut self.windows, t, group, (target, gate), &self.scores);
                    return;
                }
                for &c in &self.test_candidates {
                    self.scores[c] = expr::eval(&trees[c], features);
                }
                let forced = self.forced.as_deref().expect("the test pass follows the selection");
                push_selected_test(&mut self.windows, forced, t, group, (target, gate), &self.scores);
            }
        }
    }

    /// Final read: end the training pass. Fix the candidate of every window
    /// from its training and validation scores, then score the test rows of
    /// those candidates only. No test row has been scored before this call.
    fn begin_test_pass(&mut self, spec: &EvalSpec, trees: &[Node]) -> Result<(), String> {
        debug_assert!(self.pass == Pass::TrainOnly);
        let mut chosen = Vec::with_capacity(self.windows.len());
        for d in &self.windows {
            let has_validation = d.validation_start.is_some();
            if has_validation && trees.len() > 1 && d.validation_rows.is_empty() {
                return Err(format!(
                    "window '{}': no validation rows for candidate selection",
                    d.window.label
                ));
            }
            chosen.push(select_candidate(d, spec, trees, has_validation, self.threshold_fit.as_ref())?.candidate);
        }
        if self.vote {
            self.test_candidates = (0..trees.len()).collect();
            self.forced = Some(vec![0; self.windows.len()]);
            self.pass = Pass::TestOnly;
            return Ok(());
        }
        let mut distinct = chosen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        self.test_candidates = distinct;
        self.forced = Some(chosen);
        self.pass = Pass::TestOnly;
        Ok(())
    }

    /// #1176: the accumulation a scan of candidate `k` alone would have made:
    /// the same rows, the scores of `k` and its own dropped-row count.
    fn member(&self, k: usize) -> Accum {
        let windows = self
            .windows
            .iter()
            .map(|d| {
                let test_bad = d.test_rows.iter().zip(&d.test_scores[k]).filter(|(r, s)| !r.target.is_finite() || !s.is_finite()).count();
                WindowData {
                    window: d.window.clone(),
                    validation_start: d.validation_start,
                    train_scores: vec![d.train_scores[k].clone()],
                    train_score_max_time: vec![d.train_score_max_time[k]],
                    validation_rows: d.validation_rows.clone(),
                    validation_scores: vec![d.validation_scores[k].clone()],
                    test_rows: d.test_rows.clone(),
                    test_gate: d.test_gate.clone(),
                    test_regime: Vec::new(),
                    test_scores: vec![d.test_scores[k].clone()],
                    train_target_sum: d.train_target_sum,
                    train_target_n: d.train_target_n,
                    train_max_time: d.train_max_time,
                    train_seen: d.train_seen,
                    dropped: d.train_seen - d.train_scores[k].len() + test_bad,
                    keep_train_rows: d.keep_train_rows,
                    train_rows: d.train_rows.clone(),
                    train_row_scores: vec![d.train_row_scores[k].clone()],
                }
            })
            .collect();
        Accum {
            locked: self.locked.clone(),
            locked_purge: self.locked_purge,
            locked_embargo: self.locked_embargo,
            windows,
            candidates: 1,
            scores: vec![0.0],
            pass: Pass::All,
            forced: None,
            test_candidates: Vec::new(),
            vote: false,
            threshold_fit: None,
            gate: self.gate.clone(),
            gate_base: self.gate_base,
            regime_exprs: Vec::new(),
            regime_slots: Vec::new(),
            regime_base: self.regime_base,
        }
    }

    /// #1176: every front member priced alone, as `--expr <member>` with the
    /// member's carried band. The members never enter the selection of the
    /// parent report; they carry no bootstrap, null or batch section.
    #[allow(clippy::too_many_arguments)]
    fn price_members(
        &self,
        clock: std::time::Instant,
        spec: &EvalSpec,
        opts: &ProtocolOptions,
        layout: &Layout,
        source: &str,
        memory_bound: &str,
        trees: &[Node],
    ) -> Result<Vec<Member>, String> {
        let front = opts.front.as_ref().ok_or_else(|| "--all-members needs --front".to_string())?;
        let expressions = candidate_expressions(spec);
        let mut out = Vec::with_capacity(self.candidates);
        for k in 0..self.candidates {
            let one = EvalSpec {
                expression: expressions[k].clone(),
                candidates: Vec::new(),
                band_select: false,
                device_pnl: false,
                ..spec.clone()
            };
            let one_opts = ProtocolOptions {
                bootstrap_draws: 0,
                null_draws: 0,
                batch_pnl: false,
                run_bands: opts.run_bands.as_ref().map(|b| vec![b[k].clone()]),
                front: Some(FrontSource { labels: vec![front.labels[k].clone()], ranks: vec![front.ranks[k]], ..front.clone() }),
                regimes: None,
                regime_exprs: Vec::new(),
                period_series: true,
                emit_series: true,
                all_members: false,
                ..opts.clone()
            };
            let report = self
                .member(k)
                .finish(clock, &one, &one_opts, layout, source, memory_bound, &trees[k..=k])
                .map_err(|e| format!("{}: {}", front.labels[k], e))?;
            out.push(Member {
                archive_rank: front.ranks[k],
                label: front.labels[k].clone(),
                complexity: expr::complexity(&trees[k]),
                constant: crate::trading::is_constant(&trees[k]),
                report,
            });
        }
        Ok(out)
    }



    fn finish(
        self,
        clock: std::time::Instant,
        spec: &EvalSpec,
        opts: &ProtocolOptions,
        layout: &Layout,
        source: &str,
        memory_bound: &str,
        trees: &[Node],
    ) -> Result<Report, String> {
        let member_reports = if opts.all_members {
            Some(self.price_members(clock, spec, opts, layout, source, memory_bound, trees)?)
        } else {
            None
        };
        let emit_series = opts.emit_series || opts.all_members;
        let cost_side = spec.cost_bps / 10_000.0;
        let mut windows = Vec::with_capacity(self.windows.len());
        let mut pooled_returns: Vec<f64> = Vec::new();
        let mut pooled_gross: Vec<f64> = Vec::new();
        let mut pooled_stats = Stats::default();
        let mut pooled_ppy: Vec<f64> = Vec::new();
        let mut dropped = 0usize;
        let mut seam_cost = 0.0f64;
        let mut seam_trades = 0usize;
        let mut previous_end: Option<HashMap<u64, f64>> = None;
        let mut pooled_previous: HashMap<u64, f64> = HashMap::new();
        let mut previous_test_end: Option<f64> = None;
        let expressions = candidate_expressions(spec);
        let want_inference = opts.bootstrap_draws > 0 || opts.null_draws > 0;
        let vote = spec.policy == Policy::Vote;
        let members = trees.len();
        let need = if vote { votes_required(opts.quorum.unwrap_or(1.0), members) } else { 0 };
        // A vote book is one synthetic candidate; its band counts votes.
        let eval_policy = if vote {
            if spec.band_width > 0.0 || spec.min_hold > 1 { Policy::Band } else { Policy::Threshold }
        } else {
            spec.policy
        };
        let n_candidates = if vote { 1 } else { members };
        let mut min_group_step: Option<f64> = None;
        let mut period_source: Option<PeriodSource> = None;
        let mut pooled_times: Vec<f64> = Vec::new();
        let mut fresh_all: Vec<f64> = Vec::new();
        let mut market_all: Vec<f64> = Vec::new();
        let mut bh_all: Vec<f64> = Vec::new();
        let mut const_all: Vec<f64> = Vec::new();
        let (mut bh_rows, mut bh_long, mut const_rows, mut const_long) = (0u64, 0u64, 0u64, 0u64);
        let mut cand_cols: Vec<Vec<f64>> = vec![Vec::new(); n_candidates];
        let want_columns = opts.bootstrap_draws > 0 && !opts.final_read;
        let (mut column_bh, mut column_const): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let mut books: Vec<crate::wf_protocol::NullBook> = Vec::new();
        let mut batch_in: Vec<crate::wf_batch::WindowScores> = Vec::new();
        let forced = self.forced.as_deref();
        let target_grid = opts.period_grid == PeriodGrid::Target;
        let mut period_rows: Vec<usize> = Vec::new();
        let mut empty_periods = 0usize;
        let mut unpriced_target_rows = 0usize;
        let want_strata = opts.regimes.is_some() || !opts.regime_exprs.is_empty() || opts.period_series || emit_series;
        let mut strata_in = StrataIn::new(opts.regime_exprs.len());
        let mut entry_times = 0usize;
        let validated = self.windows.iter().all(|d| d.validation_start.is_some());
        for (window_index, mut d) in self.windows.into_iter().enumerate() {
            // #1199: the target grid, its period width and the rows that define it come from the rows before the gate.
            let grid_source = target_grid.then(|| {
                let rows: Vec<usize> = (0..d.test_rows.len()).filter(|&i| d.test_rows[i].target.is_finite()).collect();
                (target_times(&d.test_rows), window_period(spec, opts, &d.test_rows), rows)
            });
            let regime_states: Vec<Vec<bool>> = opts
                .regime_exprs
                .iter()
                .zip(&d.test_regime)
                .map(|(e, values)| {
                    let groups: Vec<u64> = d.test_rows.iter().map(|r| r.group).collect();
                    crate::regime_gate::hysteresis_states(values, &groups, e.gate.band, e.gate.dwell)
                })
                .collect();
            // #1203: the in-report candidate columns sit on the target grid of the rows before the gate.
            let column_times = (want_columns && !target_grid).then(|| target_times(&d.test_rows));
            if let Some(gate) = &opts.mask {
                apply_mask(&mut d, gate, window_index);
            }
            {
                let labelled = d.test_rows.iter().filter(|r| r.target.is_finite());
                let book = crate::trading::TradingRows {
                    time: labelled.clone().map(|r| r.time).collect(),
                    group: labelled.map(|r| r.group).collect(),
                    funding: Vec::new(),
                };
                let step = crate::trading::check_return_spacing(&book, spec.horizon)
                    .map_err(|e| format!("window '{}': {}", d.window.label, e))?;
                min_group_step = match (min_group_step, step) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
            }
            let has_validation = d.validation_start.is_some();
            if has_validation && expressions.len() > 1
                && d.validation_rows.is_empty()
            {
                return Err(format!(
                    "window '{}': no validation rows for candidate selection",
                    d.window.label
                ));
            }
            if d.test_rows.is_empty() {
                return Err(format!(
                    "window '{}': no finite test rows in [{}, {})",
                    d.window.label, d.window.test_start, d.window.test_end
                ));
            }
            let Selected { mut candidate, mut validation_error, mut threshold, band_selection, mut thresholds, mut cuts } =
                select_candidate(&d, spec, trees, has_validation, opts.threshold_fit.as_ref())?;
            let cut = if vote { None } else { cuts[candidate].take() };
            let member_thresholds = if vote { thresholds.clone() } else { Vec::new() };
            if vote {
                threshold = apply_vote(&mut d, &thresholds, need, spec.band_width);
                thresholds = vec![threshold];
                candidate = 0;
                validation_error = None;
            }
            // #925: each candidate's band, resolved on its own training scores.
            let bands: Vec<crate::trading::BandSpec> = if vote {
                vec![crate::trading::BandSpec::score(spec.band_width / 2.0)]
            } else {
                (0..d.train_scores.len()).map(|c| window_band(&d, c, spec.band_width, opts)).collect()
            };
            let band_width = bands[candidate].resolved;
            if let Some(fixed) = forced {
                if fixed[window_index] != candidate {
                    return Err(format!(
                        "window '{}': the candidate chosen before the locked scoring differs from the one chosen after it",
                        d.window.label
                    ));
                }
            }
            let source_name = if spec.policy == Policy::TopTercile {
                ThresholdSource::None
            } else if let Some(c) = &cut {
                c.source
            } else if spec.threshold_fixed.is_some() {
                ThresholdSource::Fixed
            } else if has_validation {
                ThresholdSource::ValidationType7
            } else {
                ThresholdSource::TrainType7
            };
            let (ppy, inferred_dt, bucket, source) = match &grid_source {
                Some((_, period, _)) => *period,
                None => window_period(spec, opts, &d.test_rows),
            };
            period_source = Some(source);
            let mut fresh_previous: HashMap<u64, f64> = HashMap::new();
            let ev = evaluate_window(
                &d.test_rows,
                &d.test_scores[candidate],
                threshold,
                cost_side,
                ppy,
                bucket,
                &mut fresh_previous,
                eval_policy,
                band_width,
                spec.min_hold,
                spec.gpu,
                spec.device_pnl,
                Some(ControlInput {
                    seed: spec.control_seed,
                    train_mean: if d.train_target_n == 0 {
                        f64::NAN
                    } else {
                        d.train_target_sum / d.train_target_n as f64
                    },
                    check_device: spec.device_pnl,
                }),
                opts.null_draws > 0,
                opts.missing_score,
            );
            // Pooled path: carry each group's position across the seam unless
            // the caller asked for the historical flat reset.
            let flat_here = spec.pool_reset_flat || (opts.flat_gaps && previous_test_end != Some(d.window.test_start));
            previous_test_end = Some(d.window.test_end);
            let pev = if spec.pool_reset_flat {
                evaluate_window(
                    &d.test_rows,
                    &d.test_scores[candidate],
                    threshold,
                    cost_side,
                    ppy,
                    bucket,
                    &mut HashMap::new(),
                    eval_policy,
                    band_width,
                    spec.min_hold,
                    spec.gpu,
                    false,
                    None,
                    false,
                    opts.missing_score,
                )
            } else {
                if flat_here {
                    pooled_previous.clear();
                }
                evaluate_window(
                    &d.test_rows,
                    &d.test_scores[candidate],
                    threshold,
                    cost_side,
                    ppy,
                    bucket,
                    &mut pooled_previous,
                    eval_policy,
                    band_width,
                    spec.min_hold,
                    spec.gpu,
                    false,
                    None,
                    false,
                    opts.missing_score,
                )
            };
            if !spec.pool_reset_flat && opts.mask.is_none() {
                if let Some(prev_end) = previous_end.as_ref().filter(|_| !flat_here) {
                    for (group, &q_start) in &ev.first_positions {
                        let p_end = prev_end.get(group).copied().unwrap_or(0.0);
                        let delta = (q_start - p_end).abs();
                        seam_cost += cost_side * delta;
                        if delta != 0.0 {
                            seam_trades += 1;
                        }
                    }
                }
                previous_end = Some(ev.end_positions.clone());
            }
            unpriced_target_rows += unpriced_rows(&d.test_rows, &d.test_scores[candidate], spec.horizon, opts.missing_score);
            let train_rows = d.train_scores[candidate].len()
                + d.validation_scores[candidate].len();
            dropped += d.dropped;
            let grid = match &grid_source {
                Some((times, _, _)) => {
                    Some(Grid::new(times, bucket, &pev.times).map_err(|e| format!("window '{}': {}", d.window.label, e))?)
                }
                None => None,
            };
            let column_grid = match &column_times {
                Some(times) => Some(Grid::new(times, bucket, &ev.times).map_err(|e| format!("window '{}': {}", d.window.label, e))?),
                None => None,
            };
            let on_grid = |v: &[f64]| -> Vec<f64> {
                match &grid {
                    Some(g) => g.pad(v),
                    None => v.to_vec(),
                }
            };
            if want_strata {
                let valid: Vec<usize> = match &grid_source {
                    Some((_, _, rows)) => rows.clone(),
                    None => (0..d.test_rows.len())
                        .filter(|&i| {
                            let scored = opts.missing_score == MissingScore::Hold || d.test_scores[candidate].get(i).is_some_and(|s| s.is_finite());
                            scored && d.test_rows[i].target.is_finite()
                        })
                        .collect(),
                };
                strata_in.add_window(&pev, bucket, opts, &d, &valid, &regime_states, grid.as_ref());
            }
            pooled_returns.extend(on_grid(&pev.returns));
            pooled_gross.extend(on_grid(&pev.gross_returns));
            pooled_stats.add_assign(&pev.stats);
            entry_times += pev.entry_times;
            pooled_ppy.push(ppy);
            match &grid {
                Some(g) => {
                    pooled_times.extend_from_slice(&g.times);
                    period_rows.extend(g.pad_counts(&pev.counts));
                    empty_periods += g.times.len() - g.slots.len();
                }
                None => pooled_times.extend_from_slice(&pev.times),
            }
            fresh_all.extend(on_grid(&ev.returns));
            market_all.extend(on_grid(&ev.market));
            if let Some(c) = &ev.controls {
                bh_all.extend(on_grid(&c.series[1]));
                const_all.extend(on_grid(&c.series[2]));
                let bh = &c.get(crate::baselines::ControlKind::BuyAndHold).totals;
                let cn = &c.get(crate::baselines::ControlKind::Constant).totals;
                bh_rows += bh.rows;
                bh_long += bh.long_rows;
                const_rows += cn.rows;
                const_long += cn.long_rows;
            }
            if let Some(cg) = grid.as_ref().or(column_grid.as_ref()).filter(|_| want_columns) {
                // #1203: every candidate column, the selected one included, sits on the
                // window's target grid with 0 where the candidate prices no row, and so
                // do the SPA baselines of the columns.
                for c in 0..n_candidates {
                    if c == candidate {
                        cand_cols[c].extend(cg.pad(&ev.returns));
                        continue;
                    }
                    let other = evaluate_window(
                        &d.test_rows,
                        &d.test_scores[c],
                        thresholds[c],
                        cost_side,
                        ppy,
                        bucket,
                        &mut HashMap::new(),
                        eval_policy,
                        bands[c].resolved,
                        spec.min_hold,
                        false,
                        false,
                        None,
                        false,
                        opts.missing_score,
                    );
                    let slots = cg.slots_of(&other.times).map_err(|e| format!("window '{}': {}", d.window.label, e))?;
                    cand_cols[c].extend(pad_at(&slots, cg.times.len(), &other.returns));
                }
                if let Some(c) = &ev.controls {
                    column_bh.extend(cg.pad(&c.series[1]));
                    column_const.extend(cg.pad(&c.series[2]));
                }
            }
            if let Some(b) = ev.book {
                books.push(match &grid {
                    Some(g) => g.book(b),
                    None => b,
                });
            }
            if opts.batch_pnl {
                batch_in.push(crate::wf_batch::WindowScores {
                    rows: crate::wf_batch::BatchWindow {
                        label: d.window.label.clone(),
                        time: d.test_rows.iter().map(|r| r.time).collect(),
                        group: d.test_rows.iter().map(|r| r.group).collect(),
                        target: d.test_rows.iter().map(|r| r.target).collect(),
                        train_mean: if d.train_target_n == 0 {
                            f64::NAN
                        } else {
                            d.train_target_sum / d.train_target_n as f64
                        },
                    },
                    scores: std::mem::take(&mut d.test_scores),
                    thresholds: thresholds.clone(),
                    selected: candidate,
                    exact_net: ev.stats.net_sum,
                });
            }
            let train_last_time = if d.train_max_time.is_finite() { Some(d.train_max_time) } else { None };
            let in_sample = opts.in_sample_windows.contains(&d.window.label);
            windows.push(WindowResult {
                window: d.window,
                expression: if vote {
                    format!("vote {}/{}: {}", need, members, expressions.join(" ; "))
                } else {
                    expr::to_string(&trees[candidate], &layout.var_names)
                },
                candidate,
                validation_error,
                threshold,
                member_thresholds,
                threshold_source: source_name.as_str().to_string(),
                train_rows,
                validation_rows: d.validation_scores[candidate].len(),
                test_rows: ev.stats.rows,
                dropped_rows: d.dropped,
                in_sample,
                train_last_time,
                inferred_period_seconds: inferred_dt,
                metrics: ev.metrics,
                device_pnl: ev.device,
                controls: ev.controls,
                band: ev.band.map(|b| BandCheck { spec: Some(bands[candidate].clone()), ..b }),
                hindsight_bound: ev.hindsight_bound,
                band_selection,
                cut,
                entry_times: emit_series.then_some(pev.entry_times),
            });
        }

        let pooled_ppy = match spec.periods_per_year {
            Some(v) => v,
            None => median(&mut pooled_ppy).unwrap_or(365.0),
        };
        let pooled = summarize(&pooled_returns, &pooled_gross, &pooled_stats, pooled_ppy);
        if opts.horizon_from_spec {
            crate::target::check_period_width(spec.horizon, opts.stride_seconds, min_group_step)?;
        }
        let required_purge = opts.required_purge_seconds(spec.horizon);
        let min_gap = windows
            .iter()
            .map(|w| w.window.test_start - w.window.train_end)
            .fold(f64::INFINITY, f64::min);
        let period_seconds = {
            let mut v: Vec<f64> = windows.iter().filter_map(|w| w.inferred_period_seconds).collect();
            median(&mut v)
        };
        let dt = period_seconds.unwrap_or(1.0);
        let horizon_periods = if spec.horizon > 0.0 && dt > 0.0 {
            (spec.horizon / dt).ceil() as usize
        } else {
            0
        };
        let hold_periods = if pooled.entries > 0 {
            (pooled.active_rows as f64 / pooled.entries as f64).ceil() as usize
        } else {
            1
        };
        let floor_block = horizon_periods.max(hold_periods).max(1);
        if let Some(l) = opts.block_len {
            if l < floor_block {
                return Err(format!(
                    "--block-len {} is shorter than max(horizon, hold) = {} periods; \
                     blocks must cover the label overlap and the holding period",
                    l, floor_block
                ));
            }
        }
        let block_len = opts
            .block_len
            .unwrap_or_else(|| inference::auto_block_len(pooled_returns.len(), horizon_periods, hold_periods));
        let economics = crate::wf_protocol::economics(&crate::wf_protocol::EconomicsInput {
            net: &pooled_returns,
            gross: &pooled_gross,
            times: &pooled_times,
            cost_bps: spec.cost_bps,
            period_seconds,
            periods_per_year: pooled_ppy,
            sortino: pooled.sortino,
            cagr: pooled.cagr,
            max_drawdown: pooled.max_drawdown,
            exposure: pooled.exposure,
            fresh: &fresh_all,
            market: &market_all,
            buy_and_hold: &bh_all,
            constant: &const_all,
            bh_exposure: if bh_rows == 0 { f64::NAN } else { bh_long as f64 / bh_rows as f64 },
            constant_exposure: if const_rows == 0 { f64::NAN } else { const_long as f64 / const_rows as f64 },
        });
        let inference_report = if want_inference {
            let candidates: Vec<Vec<f64>> = if want_columns { cand_cols } else { Vec::new() };
            // #1203: SPA compares the candidate columns with a baseline on their grid.
            let (bh, constant) = if candidates.is_empty() { (&bh_all, &const_all) } else { (&column_bh, &column_const) };
            let zeros = vec![0.0; candidates.first().map_or(fresh_all.len(), Vec::len)];
            let baseline: &[f64] = match economics.best_baseline.as_str() {
                "buy_and_hold" => bh,
                "constant_long_flat" => constant,
                _ => &zeros,
            };
            let mut inferred = crate::wf_protocol::infer(&crate::wf_protocol::InferenceInput {
                draws: opts.bootstrap_draws,
                null_draws: opts.null_draws,
                block_len,
                seed: opts.seed,
                control_seed: spec.control_seed,
                periods_per_year: pooled_ppy,
                pooled: &pooled_returns,
                fresh: &fresh_all,
                candidates: &candidates,
                trials: opts.trials.unwrap_or(0),
                trials_source: if opts.trials.is_some() && opts.trials_source == "candidate_count" { "flag" } else { opts.trials_source },
                pbo_blocks: opts.pbo_blocks,
                fdr_q: opts.fdr_q,
                best_baseline: &economics.best_baseline,
                baseline_series: baseline,
                buy_and_hold: &bh_all,
                books: &books,
            })?;
            if opts.final_read {
                if let Some(m) = inferred.multiple.as_mut() {
                    m.pbo = None;
                    m.pbo_error = Some("final read reports the selected candidate".to_string());
                }
            }
            Some(inferred)
        } else {
            None
        };
        let annualization_source = period_source.unwrap_or(PeriodSource::InferredLabelledRows);
        let annualization = match annualization_source {
            PeriodSource::TargetSpec | PeriodSource::HorizonFlag => {
                "365-day year over the label horizon (one period per horizon)".to_string()
            }
            PeriodSource::Explicit => "explicit --periods-per-year".to_string(),
            PeriodSource::InferredLabelledRows => {
                "inferred from the median step of the labelled test times (365-day year)".to_string()
            }
        };
        let sign_test = fold_sign_test(&windows);
        let batch_started = clock.elapsed().as_secs_f64();
        let batch_pnl = if opts.batch_pnl {
            Some(crate::wf_batch::summarize(
                &batch_in,
                cost_side,
                !spec.pool_reset_flat,
                spec.control_seed,
                opts.missing_score,
            )?)
        } else {
            None
        };
        let mut ledger = crate::gpu_stats::RunLedger::new();
        if let Some(b) = &batch_pnl {
            let mut at = batch_started;
            for r in std::iter::once(&b.per_window).chain(b.pooled.as_ref()) {
                let s = &r.stats;
                ledger.record_interval(crate::gpu_stats::Stage::Pnl, 0, at, at + s.device_s.min(s.wall_s));
                at += s.wall_s;
            }
        }
        let device_pnl = if spec.device_pnl {
            let mut backend = String::new();
            let mut rows = 0u64;
            let mut count = 0usize;
            let mut max_abs_net_delta = 0.0f64;
            let mut device_windows = 0usize;
            let mut bit_mismatch_windows = 0usize;
            let mut count_mismatch_windows = 0usize;
            let mut f32_flip_rows = 0u64;
            for w in &windows {
                if let Some(d) = &w.device_pnl {
                    if d.backend == "cuda" {
                        device_windows += 1;
                        let at = d.started.saturating_duration_since(clock).as_secs_f64();
                        ledger.record_interval(crate::gpu_stats::Stage::Pnl, 0, at, at + d.host_s);
                    }
                    if backend.is_empty() {
                        backend = d.backend.clone();
                    }
                    rows += d.strategy.rows;
                    count += 1;
                    max_abs_net_delta = max_abs_net_delta.max(d.net_delta.abs());
                    bit_mismatch_windows += usize::from(!d.bit_equal);
                    count_mismatch_windows += usize::from(!d.counts_equal);
                    f32_flip_rows += d.f32_flip_rows;
                }
            }
            Some(DevicePnlSummary {
                backend,
                segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
                tile_candidates: crate::trading_gpu::MAX_TILE_CANDIDATES,
                windows: count,
                rows,
                max_abs_net_delta,
                device_windows,
                bit_mismatch_windows,
                count_mismatch_windows,
                f32_flip_rows,
            })
        } else {
            None
        };
        let selected_on = if vote {
            "none (all members vote; thresholds from validation or train rows)".to_string()
        } else if expressions.len() > 1 && !validated {
            "none (--all-members without a validation slice: the report book of every window is the first front member)".to_string()
        } else if expressions.len() > 1 {
            "validation".to_string()
        } else if spec.validation_frac > 0.0 {
            "validation (threshold only)".to_string()
        } else {
            "train (threshold only)".to_string()
        };
        let (regimes, period_extra) = if want_strata {
            let r = (opts.regimes.is_some() || !opts.regime_exprs.is_empty()).then(|| strata_in.report(&pooled_returns, opts, pooled_ppy));
            let mut extra = strata_in.extra;
            if emit_series {
                extra.cost = pooled_gross.iter().zip(&pooled_returns).map(|(g, n)| g - n).collect();
                extra.entry_stamp = extra.entries.iter().map(|&e| if e > 0.0 { 1.0 } else { 0.0 }).collect();
                extra.bh_net = bh_all.clone();
                extra.entry_times = entry_times;
            }
            (r, Some(extra))
        } else {
            (None, None)
        };
        Ok(Report {
            expression: windows
                .first()
                .map(|w| w.expression.clone())
                .unwrap_or_else(|| expressions.first().cloned().unwrap_or_default()),
            target: spec.target.clone(),
            variables: layout.var_names.clone(),
            time_column: spec.time.clone(),
            group_by: spec.group.clone(),
            windows,
            pooled,
            period_net: pooled_returns,
            period_time: pooled_times,
            period_grid: target_grid.then(|| GridSummary { rule: PeriodGrid::Target, empty_periods, period_rows }),
            regimes,
            period_extra,
            emit_series,
            members: member_reports,
            cost: opts.cost_decl(spec.cost_bps),
            cost_bps: spec.cost_bps,
            threshold_quantile: spec.threshold_quantile,
            threshold_fixed: spec.threshold_fixed,
            threshold_fit: opts.threshold_fit,
            policy: spec.policy,
            quorum: if vote { opts.quorum } else { None },
            votes_required: need,
            band_width: spec.band_width,
            band_units: opts.band_units,
            band_source: if opts.run_bands.is_some() { "run" } else { "flag" },
            min_hold: spec.min_hold,
            gpu: spec.gpu,
            validation_frac: spec.validation_frac,
            horizon: spec.horizon,
            embargo: spec.embargo_seconds(),
            periods_per_year: pooled_ppy,
            annualization,
            annualization_source: annualization_source.as_str(),
            period_seconds,
            candidates: expressions,
            selected_on,
            front: opts.front.clone(),
            expression_source: opts.expression_source,
            band_select: spec.band_select,
            test_used_for_selection: false,
            pool_reset_flat: spec.pool_reset_flat,
            seam_cost,
            seam_trades,
            missing_score: opts.missing_score,
            missing_score_exits: pooled_stats.missing_score_exits,
            unpriced_target_rows,
            sign_test,
            source: source.to_string(),
            memory_bound: memory_bound.to_string(),
            allow_lockbox: spec.allow_lockbox,
            guarded_lockbox: crate::lockbox::Guard::for_run(opts.declared_lock.as_ref(), spec.allow_lockbox).lockbox,
            device_pnl,
            batch_pnl,
            ledger,
            scan: None,
            required_purge_seconds: required_purge,
            min_group_step,
            min_purge_seconds: min_gap,
            fit_label: opts.fit_label.clone(),
            economics,
            inference: inference_report,
            dropped_rows: dropped,
        })
    }
}

/// #1180: price the test rows of a window under the regime gate. The gate
/// state is read over the labelled rows in file order, per group from the
/// group's first row of the window, exactly as the search read its test slice.
/// Every on-run becomes its own group (a flat start and a paid exit, unique
/// across windows, so no position carries over a seam) and the target of every
/// off row becomes NaN, which drops the row from the book.
fn apply_mask(d: &mut WindowData, gate: &crate::regime_gate::GateSpec, window_index: usize) {
    let n = d.test_rows.len();
    let labelled: Vec<usize> = (0..n).filter(|&i| d.test_rows[i].target.is_finite()).collect();
    let values: Vec<f64> = labelled.iter().map(|&i| d.test_gate[i]).collect();
    let groups: Vec<u64> = labelled.iter().map(|&i| d.test_rows[i].group).collect();
    let on_labelled = crate::regime_gate::hysteresis_states(&values, &groups, gate.band, gate.dwell);
    let mut on = vec![false; n];
    for (k, &i) in labelled.iter().enumerate() {
        on[i] = on_labelled[k];
    }
    let all_groups: Vec<u64> = d.test_rows.iter().map(|r| r.group).collect();
    let ids = crate::regime_gate::run_ids(&all_groups, &on);
    let base = (window_index as u64 + 1) << 40;
    for i in 0..n {
        d.test_rows[i].group = base + ids[i];
        if !on[i] {
            d.test_rows[i].target = f64::NAN;
        }
    }
}

/// #925: the band of candidate `c` in window `d`: `multiple` in score units, or
/// `multiple` times the sd of the candidate's finite training scores (rows
/// before the validation start, never validation or test rows).
fn window_band(d: &WindowData, c: usize, multiple: f64, opts: &ProtocolOptions) -> crate::trading::BandSpec {
    if let Some(bands) = &opts.run_bands {
        return bands[c].clone();
    }
    let units = opts.band_units;
    if units == crate::trading::BandUnits::Score {
        return crate::trading::BandSpec::score(multiple);
    }
    let t = d.train_score_max_time[c];
    crate::trading::BandSpec::from_sd(units, multiple, crate::trading::score_sd(&d.train_scores[c]), t.is_finite().then_some(t))
}

fn push_scores(
    windows: &mut [WindowData],
    t: f64,
    group: u64,
    (target, gate): (f64, f64),
    scores: &[f64],
    with_test: bool,
) {
    for d in windows {
        let in_train = t >= d.window.train_start && t < d.window.train_end;
        let in_test = t >= d.window.test_start && t < d.window.test_end;
        if in_train {
            if t > d.train_max_time {
                d.train_max_time = t;
            }
            if target.is_finite() {
                d.train_target_sum += target;
                d.train_target_n += 1;
            }
            if d.validation_start.map_or(false, |s| t >= s) {
                d.validation_rows.push(RowBase {
                    time: t,
                    group,
                    target,
                });
                for (dst, &score) in d.validation_scores.iter_mut().zip(scores) {
                    dst.push(if score.is_finite() { score } else { f64::NAN });
                }
            } else {
                d.train_seen += 1;
                if d.keep_train_rows {
                    d.train_rows.push(RowBase { time: t, group, target });
                    for (dst, &score) in d.train_row_scores.iter_mut().zip(scores) {
                        dst.push(score);
                    }
                }
                for (c, (dst, &score)) in d.train_scores.iter_mut().zip(scores).enumerate() {
                    if score.is_finite() {
                        dst.push(score);
                        if t > d.train_score_max_time[c] {
                            d.train_score_max_time[c] = t;
                        }
                    } else {
                        d.dropped += 1;
                    }
                }
            }
        }
        if in_test && with_test {
            d.test_rows.push(RowBase {
                time: t,
                group,
                target,
            });
            d.test_gate.push(gate);
            let bad = !target.is_finite() || scores.iter().any(|s| !s.is_finite());
            for (dst, &score) in d.test_scores.iter_mut().zip(scores) {
                dst.push(score);
            }
            if bad {
                d.dropped += 1;
            }
        }
    }
}

/// Final read of a vote book: a test row with the score of every member.
fn push_all_test(windows: &mut [WindowData], t: f64, group: u64, (target, gate): (f64, f64), scores: &[f64]) {
    for d in windows {
        if t >= d.window.test_start && t < d.window.test_end {
            d.test_rows.push(RowBase { time: t, group, target });
            d.test_gate.push(gate);
            for (dst, &score) in d.test_scores.iter_mut().zip(scores) {
                dst.push(score);
            }
            if !target.is_finite() || scores.iter().any(|s| !s.is_finite()) {
                d.dropped += 1;
            }
        }
    }
}

/// Final read: record a test row for every window that contains it, with the
/// score of that window's fixed candidate only. The other candidates' test
/// columns stay empty and are never evaluated.
fn push_selected_test(
    windows: &mut [WindowData],
    chosen: &[usize],
    t: f64,
    group: u64,
    (target, gate): (f64, f64),
    scores: &[f64],
) {
    for (d, &c) in windows.iter_mut().zip(chosen) {
        if t >= d.window.test_start && t < d.window.test_end {
            d.test_rows.push(RowBase { time: t, group, target });
            d.test_gate.push(gate);
            d.test_scores[c].push(scores[c]);
            if !target.is_finite() || !scores[c].is_finite() {
                d.dropped += 1;
            }
        }
    }
}

fn candidate_expressions(spec: &EvalSpec) -> Vec<String> {
    if spec.candidates.is_empty() {
        vec![spec.expression.clone()]
    } else {
        spec.candidates.clone()
    }
}

fn validation_mse(rows: &[RowBase], scores: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (row, &score) in rows.iter().zip(scores) {
        if score.is_finite() && row.target.is_finite() {
            let e = score - row.target;
            sum += e * e;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

/// One-sided binomial sign test over fold expectancies.
///
/// The test assumes independent folds. Expanding walk-forward violates that:
/// `test_k` sits inside `train_{k+1}`. When any training interval overlaps
/// another fold's test block the test is reported as not applicable rather
/// than computed under a product law the folds do not obey.
fn fold_sign_test(windows: &[WindowResult]) -> SignTest {
    let mut positive = 0usize;
    let mut total = 0usize;
    for w in windows {
        if w.metrics.mean_return.is_finite() {
            total += 1;
            if w.metrics.mean_return > 0.0 {
                positive += 1;
            }
        }
    }
    let mut ordered: Vec<&WindowResult> = windows.iter().collect();
    ordered.sort_by(|a, b| {
        a.window
            .test_start
            .partial_cmp(&b.window.test_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut overlap = false;
    for (i, a) in ordered.iter().enumerate() {
        for b in ordered.iter().skip(i + 1) {
            let train_overlaps_test = a.window.train_start < b.window.test_end
                && b.window.test_start < a.window.train_end;
            let test_overlaps_train = b.window.train_start < a.window.test_end
                && a.window.test_start < b.window.train_end;
            if train_overlaps_test || test_overlaps_train {
                overlap = true;
                break;
            }
        }
        if overlap {
            break;
        }
    }
    if total < 2 {
        return SignTest {
            positive,
            total,
            p_value: None,
            method: "not applicable: fewer than two folds with a finite mean".to_string(),
            applicable: false,
        };
    }
    if overlap {
        return SignTest {
            positive,
            total,
            p_value: None,
            method: "not computed: expanding/overlapping folds are dependent; use                      disjoint rolling folds or a block bootstrap over the return series"
                .to_string(),
            applicable: false,
        };
    }
    let p = binomial_tail(positive, total);
    SignTest {
        positive,
        total,
        p_value: Some(p),
        method: "one-sided binomial P(X >= positive) under Bin(total, 0.5); independent folds only"
            .to_string(),
        applicable: true,
    }
}

/// `P(X >= k)` for `X ~ Bin(n, 0.5)`, computed in log space.
pub fn binomial_tail(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let k = k.min(n);
    let ln_half = std::f64::consts::LN_2;
    let mut sum = 0.0f64;
    for j in k..=n {
        let ln_c = ln_factorial(n) - ln_factorial(j) - ln_factorial(n - j);
        sum += (ln_c - n as f64 * ln_half).exp();
    }
    sum.min(1.0)
}

fn ln_factorial(n: usize) -> f64 {
    // n is a fold count; a direct sum is exact enough and keeps this crate
    // std-only with no dependency on a gamma approximation.
    let mut s = 0.0;
    for i in 2..=n {
        s += (i as f64).ln();
    }
    s
}

// ---------------------------------------------------------------------------
// Portfolio simulation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub(crate) rows: usize,
    pub(crate) active: usize,
    pub(crate) hits: usize,
    pub(crate) wins: usize,
    pub(crate) losses: usize,
    pub(crate) win_sum: f64,
    pub(crate) loss_sum: f64,
    pub(crate) total_cost: f64,
    pub(crate) total_abs_delta: f64,
    pub(crate) trades: usize,
    pub(crate) entries: usize,
    pub(crate) exits: usize,
    /// #309: rule exits (position falls), excluding the final exit.
    pub(crate) rule_exits: usize,
    /// #1203: final exits of runs that a row without a score ends (`Flat`).
    pub(crate) missing_score_exits: usize,
    /// #309: row-order sums of `p * r` and of the net row terms.
    pub(crate) gross_sum: f64,
    pub(crate) net_sum: f64,
}

impl Stats {
    fn add_assign(&mut self, other: &Stats) {
        self.rows += other.rows;
        self.active += other.active;
        self.hits += other.hits;
        self.wins += other.wins;
        self.losses += other.losses;
        self.win_sum += other.win_sum;
        self.loss_sum += other.loss_sum;
        self.total_cost += other.total_cost;
        self.total_abs_delta += other.total_abs_delta;
        self.trades += other.trades;
        self.entries += other.entries;
        self.exits += other.exits;
        self.rule_exits += other.rule_exits;
        self.missing_score_exits += other.missing_score_exits;
        self.gross_sum += other.gross_sum;
        self.net_sum += other.net_sum;
    }
}

struct WindowEval {
    metrics: Metrics,
    returns: Vec<f64>,
    gross_returns: Vec<f64>,
    /// Period stamps of `returns`, ascending.
    times: Vec<f64>,
    /// Equal-weight mean target per period: the always-long universe, no cost.
    market: Vec<f64>,
    /// #1175: mean position and entry count of every period.
    exposure: Vec<f64>,
    entries: Vec<f64>,
    /// #1176: distinct row timestamps with an entry.
    entry_times: usize,
    book: Option<NullBook>,
    stats: Stats,
    first_positions: HashMap<u64, f64>,
    end_positions: HashMap<u64, f64>,
    device: Option<DevicePnlCheck>,
    band: Option<BandCheck>,
    hindsight_bound: Option<HindsightBound>,
    controls: Option<crate::baselines::ControlSet>,
    /// Priced rows of every period of `returns`.
    counts: Vec<usize>,
}

/// #1199: the periods of one window under `--period-grid target` and the grid
/// period of every period the book priced.
struct Grid {
    cross: CrossSection,
    times: Vec<f64>,
    slots: Vec<usize>,
}

impl Grid {
    /// `target_times`: the ascending stamps of the rows with a finite target
    /// before the gate; `priced`: the period stamps of the book.
    fn new(target_times: &[f64], bucket: f64, priced: &[f64]) -> Result<Grid, String> {
        let mut cross = CrossSection::with_bucket(bucket);
        for &t in target_times {
            cross.add(t, 0.0, 0.0);
        }
        let times = cross.periods.iter().map(|p| p.time).collect();
        let mut grid = Grid { cross, times, slots: Vec::new() };
        grid.slots = grid.slots_of(priced)?;
        Ok(grid)
    }

    /// The grid period of every stamp of `times` (ascending).
    fn slots_of(&self, times: &[f64]) -> Result<Vec<usize>, String> {
        let mut out = Vec::with_capacity(times.len());
        let mut j = 0usize;
        for &t in times {
            let key = self.cross.key(t);
            while j < self.times.len() && self.cross.key(self.times[j]) < key {
                j += 1;
            }
            if j == self.times.len() || self.cross.key(self.times[j]) != key {
                return Err(format!("the priced period at {} has no row with a finite target", jnum(t)));
            }
            out.push(j);
        }
        Ok(out)
    }

    fn pad(&self, v: &[f64]) -> Vec<f64> {
        pad_at(&self.slots, self.times.len(), v)
    }

    fn pad_counts(&self, v: &[usize]) -> Vec<usize> {
        let mut out = vec![0usize; self.times.len()];
        for (k, &s) in self.slots.iter().enumerate() {
            out[s] = v[k];
        }
        out
    }

    /// The null book on the grid: an empty period has no row and no cost.
    fn book(&self, b: NullBook) -> NullBook {
        let mut period_count = vec![0u32; self.times.len()];
        let mut period_cost = vec![0.0f64; self.times.len()];
        for (k, &s) in self.slots.iter().enumerate() {
            period_count[s] = b.period_count[k];
            period_cost[s] = b.period_cost[k];
        }
        NullBook { period: b.period.iter().map(|&p| self.slots[p as usize] as u32).collect(), period_count, period_cost, ..b }
    }
}

fn pad_at(slots: &[usize], len: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; len];
    for (k, &s) in slots.iter().enumerate() {
        out[s] = v[k];
    }
    out
}

fn target_times(rows: &[RowBase]) -> Vec<f64> {
    let mut t: Vec<f64> = rows.iter().filter(|r| r.target.is_finite()).map(|r| r.time).collect();
    t.sort_by(f64::total_cmp);
    t
}

/// #1203: rows with a finite score and no finite target that lie inside a run
/// of the book (between two priced rows of one run) where the return of the
/// run's previous priced row does not cover them (`t >= t_prev + horizon`;
/// every such row when `horizon` is not positive).
fn unpriced_rows(rows: &[RowBase], scores: &[f64], horizon: f64, missing: MissingScore) -> usize {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| rows[a].group.cmp(&rows[b].group).then_with(|| rows[a].time.total_cmp(&rows[b].time)).then_with(|| a.cmp(&b)));
    let (mut count, mut pending) = (0usize, 0usize);
    let mut last: Option<(u64, f64)> = None;
    for &i in &idx {
        let (row, scored) = (&rows[i], scores[i].is_finite());
        if last.is_some_and(|(g, _)| g != row.group) {
            last = None;
            pending = 0;
        }
        if row.target.is_finite() {
            if scored || missing == MissingScore::Hold {
                if last.is_some() {
                    count += pending;
                }
                last = Some((row.group, row.time));
            } else {
                last = None;
            }
            pending = 0;
        } else if let (true, Some((_, t0))) = (scored, last) {
            if !(horizon > 0.0) || row.time >= t0 + horizon {
                pending += 1;
            }
        }
    }
    count
}

/// #1175: the per-period series of the pooled book beside `period_net`.
#[derive(Clone, Debug, Default)]
pub struct PeriodExtra {
    pub gross: Vec<f64>,
    pub exposure: Vec<f64>,
    pub entries: Vec<f64>,
    /// Equal-weight mean target per period: buy-and-hold of the same rows, no cost.
    pub bh: Vec<f64>,
    /// Bucket start of every period: the instant a calendar row is tested against.
    pub bucket_start: Vec<f64>,
    /// #1176 (`--emit-series`): `gross - net` of every period, so the net at `m` times
    /// the cost is `gross - m * cost`, the repricing of the cost ladder.
    pub cost: Vec<f64>,
    /// #1176: 1 for a period where at least one row enters a long run, else 0.
    pub entry_stamp: Vec<f64>,
    /// #1176: net of the buy-and-hold control on the same rows at the report cost.
    pub bh_net: Vec<f64>,
    /// #1176: distinct timestamps at which at least one row enters a long run.
    pub entry_times: usize,
}

/// #1175: the pooled per-period series and the regime expression states, window by window.
struct StrataIn {
    extra: PeriodExtra,
    states: Vec<Vec<bool>>,
    counts: Vec<crate::regimes::RowCounts>,
}

impl StrataIn {
    fn new(exprs: usize) -> Self {
        StrataIn { extra: PeriodExtra::default(), states: vec![Vec::new(); exprs], counts: vec![Default::default(); exprs] }
    }

    /// `valid` are the rows behind the period states: the priced rows, or under the #1199 target grid
    /// the rows with a finite target before the gate. On the grid every series has the grid's length
    /// and an empty period has no gross, exposure, entry or buy-and-hold return.
    #[allow(clippy::too_many_arguments)]
    fn add_window(&mut self, pev: &WindowEval, bucket: f64, opts: &ProtocolOptions, d: &WindowData, valid: &[usize], regime_states: &[Vec<bool>], grid: Option<&Grid>) {
        let key = |t: f64| if bucket > 0.0 { (t / bucket).floor() } else { t };
        let pad = |v: &[f64]| -> Vec<f64> {
            match grid {
                Some(g) => g.pad(v),
                None => v.to_vec(),
            }
        };
        let times: &[f64] = grid.map_or(&pev.times, |g| &g.times);
        self.extra.gross.extend(pad(&pev.gross_returns));
        self.extra.exposure.extend(pad(&pev.exposure));
        self.extra.entries.extend(pad(&pev.entries));
        self.extra.bh.extend(pad(&pev.market));
        self.extra.bucket_start.extend(times.iter().map(|&t| if bucket > 0.0 { key(t) * bucket } else { t }));
        let period_keys: Vec<f64> = times.iter().map(|&t| key(t)).collect();
        let row_keys: Vec<f64> = valid.iter().map(|&i| key(d.test_rows[i].time)).collect();
        for k in 0..opts.regime_exprs.len() {
            let on: Vec<bool> = valid.iter().map(|&i| regime_states[k][i]).collect();
            let finite: Vec<bool> = valid.iter().map(|&i| d.test_regime[k][i].is_finite()).collect();
            let (st, c) = crate::regimes::period_states(&row_keys, &on, &finite, &period_keys);
            self.states[k].extend(st);
            self.counts[k].add(c);
        }
    }

    fn report(&self, net: &[f64], opts: &ProtocolOptions, ppy: f64) -> crate::regimes::RegimeReport {
        use crate::regimes::{expr_block, strata, RegimeReport, Series, StatOpts};
        let o = StatOpts { periods_per_year: ppy, draws: opts.bootstrap_draws, block_len: opts.block_len, seed: opts.seed };
        let s = Series {
            net,
            gross: Some(&self.extra.gross),
            exposure: Some(&self.extra.exposure),
            entries: Some(&self.extra.entries),
            bh: Some(&self.extra.bh),
        };
        let mut r = RegimeReport::default();
        if let Some(cal) = &opts.regimes {
            let (names, assign) = cal.assign(&self.extra.bucket_start);
            r.calendar_sha256 = Some(cal.sha256.clone());
            r.calendar_rows = cal.rows.len();
            r.calendar = strata(&names, "calendar", &assign, &s, &o);
        }
        for (k, e) in opts.regime_exprs.iter().enumerate() {
            r.exprs.push(expr_block(e, &self.states[k], self.counts[k], &s, &o));
        }
        r
    }
}

/// #310: what the control ladder needs beyond the test rows.
struct ControlInput {
    seed: u64,
    /// Mean finite training target; `NaN` when there is none.
    train_mean: f64,
    check_device: bool,
}

/// One period's `(sum, count)` cross-section.
///
/// A period is one timestamp, or with a bucket width `S` all timestamps with
/// the same `floor(t / S)`; its stamp is the first timestamp seen. The
/// equal-weight return is `sum / count` over the rows in the period. Carrying
/// the pair (rather than a pre-divided mean) is what makes a symbol-parallel
/// decomposition merge to exactly the sequential result: denominators differ
/// when a symbol is missing at a timestamp.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrossSection {
    pub periods: Vec<PeriodSum>,
    /// Bucket width in seconds; `0` keeps one period per timestamp.
    bucket: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodSum {
    pub time: f64,
    pub net_sum: f64,
    pub gross_sum: f64,
    pub count: usize,
}

impl CrossSection {
    /// Periods of `seconds` width: rows with equal `floor(t / seconds)` share one
    /// period. A non-positive width is one period per timestamp.
    pub fn with_bucket(seconds: f64) -> CrossSection {
        CrossSection { periods: Vec::new(), bucket: if seconds > 0.0 { seconds } else { 0.0 } }
    }

    fn key(&self, time: f64) -> f64 {
        if self.bucket > 0.0 { (time / self.bucket).floor() } else { time }
    }

    /// Add one row's net and gross return. Calls must arrive in nondecreasing
    /// time order; equal timestamps accumulate into one period.
    pub fn add(&mut self, time: f64, net: f64, gross: f64) {
        let key = self.key(time);
        let bucket = self.bucket;
        if let Some(last) = self.periods.last_mut() {
            let last_key = if bucket > 0.0 { (last.time / bucket).floor() } else { last.time };
            if last_key == key {
                last.net_sum += net;
                last.gross_sum += gross;
                last.count += 1;
                return;
            }
        }
        self.periods.push(PeriodSum {
            time,
            net_sum: net,
            gross_sum: gross,
            count: 1,
        });
    }

    /// Merge an independently accumulated shard. Both sides must be ordered by
    /// time; equal timestamps add their sums and counts.
    pub fn merge(&mut self, other: &CrossSection) {
        let mut out = Vec::with_capacity(self.periods.len() + other.periods.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.periods.len() || j < other.periods.len() {
            let take_other = if i == self.periods.len() {
                true
            } else if j == other.periods.len() {
                false
            } else {
                self.key(other.periods[j].time)
                    .partial_cmp(&self.key(self.periods[i].time))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .is_lt()
            };
            if take_other {
                out.push(other.periods[j]);
                j += 1;
            } else if j < other.periods.len() && self.key(other.periods[j].time) == self.key(self.periods[i].time) {
                let a = self.periods[i];
                let b = other.periods[j];
                out.push(PeriodSum {
                    time: a.time.min(b.time),
                    net_sum: a.net_sum + b.net_sum,
                    gross_sum: a.gross_sum + b.gross_sum,
                    count: a.count + b.count,
                });
                i += 1;
                j += 1;
            } else {
                out.push(self.periods[i]);
                i += 1;
            }
        }
        self.periods = out;
    }

    /// Equal-weight net and gross series, in timestamp order.
    pub fn series(&self) -> (Vec<f64>, Vec<f64>) {
        let mut net = Vec::with_capacity(self.periods.len());
        let mut gross = Vec::with_capacity(self.periods.len());
        for p in &self.periods {
            net.push(p.net_sum / p.count as f64);
            gross.push(p.gross_sum / p.count as f64);
        }
        (net, gross)
    }
}

/// The position rule, exposed so a test can pin the "never short" property.
///
/// It is the score-only rule; reported P&L applies the stricter shared mask
/// (score **and** target finite) first. The comparison is strict, so
/// `score == threshold` is flat.
pub fn position_for(score: f64, threshold: f64) -> f64 {
    trading_gpu::long_position(score, threshold) as f64
}

/// The scalar long-only evaluator on explicit rows: the reference that
/// `trading::evaluate` must reproduce bit for bit under the plain threshold
/// rule and the same missing-score rule (#1203).
pub fn reference_metrics(
    time: &[f64],
    group: &[u64],
    target: &[f64],
    scores: &[f64],
    threshold: f64,
    cost_side: f64,
    periods_per_year: f64,
    missing: MissingScore,
) -> Metrics {
    let rows: Vec<RowBase> = (0..time.len())
        .map(|i| RowBase {
            time: time[i],
            group: group[i],
            target: target[i],
        })
        .collect();
    let mut previous: HashMap<u64, f64> = HashMap::new();
    evaluate_window(
        &rows,
        scores,
        threshold,
        cost_side,
        periods_per_year,
        0.0,
        &mut previous,
        Policy::Threshold,
        0.0,
        1,
        false,
        false,
        None,
        false,
        missing,
    )
    .metrics
}

/// #309: the scalar evaluator's row-level totals for one test block, the exact
/// CPU reference of one `(candidate, window)` batch cell. `previous` carries
/// each group's last position: pass a fresh map for a window that starts flat,
/// the pooled map to carry the seam.
pub fn reference_cell(
    time: &[f64],
    group: &[u64],
    target: &[f64],
    scores: &[f64],
    threshold: f64,
    cost_side: f64,
    previous: &mut HashMap<u64, f64>,
    missing: MissingScore,
) -> trading_gpu::PnlTotals {
    let rows: Vec<RowBase> = (0..time.len())
        .map(|i| RowBase {
            time: time[i],
            group: group[i],
            target: target[i],
        })
        .collect();
    let ev = evaluate_window(
        &rows,
        scores,
        threshold,
        cost_side,
        1.0,
        0.0,
        previous,
        Policy::Threshold,
        0.0,
        1,
        false,
        false,
        None,
        false,
        missing,
    );
    let s = &ev.stats;
    trading_gpu::PnlTotals {
        rows: s.rows as u64,
        long_rows: s.active as u64,
        entries: s.entries as u64,
        exits: (s.rule_exits + s.exits) as u64,
        trades: s.trades as u64,
        hits: s.hits as u64,
        final_exits: s.exits as u64,
        gross: s.gross_sum,
        cost: s.total_cost,
        turnover: s.total_abs_delta,
        net: s.net_sum,
    }
}

/// #1203: the book key of every row in book order. Run 0 of a group keeps the
/// group id, so the carried seam applies to it; every later run gets a key
/// that no group of the window, no other run and no carried book uses.
fn run_keys(groups: &[u64], runs: &[u32], rows: &[RowBase], previous: &HashMap<u64, f64>) -> Vec<u64> {
    if runs.iter().all(|&r| r == 0) {
        return groups.to_vec();
    }
    let mix = |mut x: u64| {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    let mut used: std::collections::HashSet<u64> = rows.iter().map(|r| r.group).chain(previous.keys().copied()).collect();
    let mut out = Vec::with_capacity(groups.len());
    let mut last: Option<(u64, u32, u64)> = None;
    for (&g, &r) in groups.iter().zip(runs) {
        if r == 0 {
            out.push(g);
            continue;
        }
        let key = match last {
            Some((lg, lr, key)) if lg == g && lr == r => key,
            _ => {
                let mut key = mix(g ^ mix(r as u64));
                while !used.insert(key) {
                    key = mix(key);
                }
                last = Some((g, r, key));
                key
            }
        };
        out.push(key);
    }
    out
}

/// `previous` carries each group's last position. Pass a fresh map for a
/// window that starts flat; pass the pooled map to carry the seam.
fn evaluate_window(
    rows: &[RowBase],
    scores: &[f64],
    threshold: f64,
    cost_side: f64,
    ppy: f64,
    bucket: f64,
    previous: &mut HashMap<u64, f64>,
    policy: Policy,
    band_width: f64,
    min_hold: usize,
    gpu: bool,
    device_pnl: bool,
    control_input: Option<ControlInput>,
    keep_book: bool,
    missing: MissingScore,
) -> WindowEval {
    assert_eq!(rows.len(), scores.len());
    // #305: one shared mask over the rows with a finite target. `position_for`
    // alone would keep a finite score with a NaN target; that is the six-row
    // difference documented in docs/walkforward.md. #1203: a row with a finite
    // target and no finite score follows `missing` (`trading::score_runs`).
    let mut labelled: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].target.is_finite()).collect();
    labelled.sort_by(|&a, &b| {
        rows[a]
            .group
            .cmp(&rows[b].group)
            .then_with(|| rows[a].time.partial_cmp(&rows[b].time).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.cmp(&b))
    });
    let lab_groups: Vec<u64> = labelled.iter().map(|&i| rows[i].group).collect();
    let lab_scored: Vec<bool> = labelled.iter().map(|&i| scores[i].is_finite()).collect();
    let runs = crate::trading::score_runs(&lab_groups, &lab_scored, missing);
    let mut mask = ValidMask::new(rows.len());
    let mut row_run = vec![0u32; rows.len()];
    let mut row_ends = vec![false; rows.len()];
    for (k, &i) in labelled.iter().enumerate() {
        if let Some(run) = runs.run[k] {
            mask.set(i, true);
            row_run[i] = run;
            row_ends[i] = runs.ends_at_missing[k];
        }
    }
    let valid: Vec<usize> = mask.indices().collect();
    let n = valid.len();
    let mut positions = vec![0.0f64; n];
    let mut costs = vec![0.0f64; n];
    let mut net = vec![0.0f64; n];
    let mut entered = vec![0.0f64; n];
    let mut stats = Stats {
        rows: n,
        ..Default::default()
    };
    let mut first_positions: HashMap<u64, f64> = HashMap::new();

    // Position changes are per group, in time order. `previous` may carry the
    // ending position of an earlier test block.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ra = &rows[valid[a]];
        let rb = &rows[valid[b]];
        ra.group
            .cmp(&rb.group)
            .then_with(|| ra.time.partial_cmp(&rb.time).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.cmp(&b))
    });

    // Rows in the group-contiguous order used by the carry scans and by the
    // #304 reduction. The scalar book below stays in this same order.
    let mut ord_scores = Vec::with_capacity(n);
    let mut ord_returns = Vec::with_capacity(n);
    let mut ord_groups = Vec::with_capacity(n);
    for &i in &order {
        let row = &rows[valid[i]];
        ord_scores.push(scores[valid[i]]);
        ord_returns.push(row.target);
        ord_groups.push(row.group);
    }
    // #1203: the book of every row. Every scan, control, null and bound below
    // reads these keys, so a run under `Flat` is its own book; `previous`,
    // `first_positions` and `end_positions` stay keyed by group.
    let ord_runs: Vec<u32> = order.iter().map(|&i| row_run[valid[i]]).collect();
    let ord_keys = run_keys(&ord_groups, &ord_runs, rows, previous);
    let ord_scored: Vec<bool> = ord_scores.iter().map(|s| s.is_finite()).collect();
    let any_held = ord_scored.iter().any(|&s| !s);
    // Under `Hold` the band and cooldown scans read the scored rows only, so
    // their state carries across a held row.
    let scan_rows: Vec<usize> = (0..n).filter(|&k| ord_scored[k]).collect();
    let (scan_scores, scan_returns, scan_keys): (Vec<f64>, Vec<f64>, Vec<u64>) = if any_held {
        (
            scan_rows.iter().map(|&k| ord_scores[k]).collect(),
            scan_rows.iter().map(|&k| ord_returns[k]).collect(),
            scan_rows.iter().map(|&k| ord_keys[k]).collect(),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let (scan_scores, scan_returns, scan_keys): (&[f64], &[f64], &[u64]) =
        if any_held { (&scan_scores, &scan_returns, &scan_keys) } else { (&ord_scores, &ord_returns, &ord_keys) };

    // #306: the band/cooldown book. `positions` are aligned with `order`;
    // `net` includes the final exit of every group. `dp` deliberately keeps
    // the threshold book: the DP is a bound, not a strategy.
    // One device launch over all rows: a chunk only partitions launches, and a
    // launch only runs the chains of the groups it touches.
    let device_chunk = scan_returns.len().max(1);
    type BandBook = (Vec<f64>, f64, usize, String, String, Option<String>);
    let band_scan: Option<BandBook> = if policy == Policy::Band {
        let params = crate::trading_gpu::BandParams {
            threshold,
            band: band_width,
            min_hold,
        };
        if min_hold > 1 {
            let (scan, backend, reason) = scan_with_backend(
                gpu,
                "cooldown scan",
                || {
                    crate::trading_gpu::cooldown_scan_device(
                        scan_scores,
                        scan_returns,
                        scan_keys,
                        &params,
                        cost_side,
                        crate::trading_gpu::SCAN_THREADS,
                        device_chunk,
                    )
                },
                || {
                    crate::trading_gpu::cooldown_scan(
                        scan_scores,
                        scan_returns,
                        scan_keys,
                        &params,
                        cost_side,
                        65_536,
                    )
                },
            );
            let net = scan.net();
            Some((
                scan.positions,
                net,
                scan.carries.len(),
                format!("cooldown-{}", min_hold),
                backend,
                reason,
            ))
        } else {
            let (scan, backend, reason) = scan_with_backend(
                gpu,
                "band scan",
                || {
                    crate::trading_gpu::band_scan_device(
                        scan_scores,
                        scan_returns,
                        scan_keys,
                        &params,
                        cost_side,
                        crate::trading_gpu::SCAN_THREADS,
                        device_chunk,
                    )
                },
                || {
                    crate::trading_gpu::band_scan(
                        scan_scores,
                        scan_returns,
                        scan_keys,
                        &params,
                        cost_side,
                        65_536,
                    )
                },
            );
            let net = scan.net();
            Some((
                scan.positions,
                net,
                scan.carries.len(),
                "carry2".to_string(),
                backend,
                reason,
            ))
        }
    } else {
        None
    };

    // #493: the cross-section is every row of the timestamp with a finite
    // score, before the target mask.
    let tercile: Option<Vec<f64>> = (policy == Policy::TopTercile).then(|| {
        let times: Vec<f64> = rows.iter().map(|r| r.time).collect();
        crate::trading::top_tercile_positions(scores, &times)
    });
    let mut ord_pos = vec![0.0f64; n];
    match (&band_scan, &tercile) {
        (Some((scanned, ..)), _) => {
            for (j, &k) in scan_rows.iter().enumerate() {
                ord_pos[k] = scanned[j];
            }
        }
        (None, Some(top)) => {
            for (k, &i) in order.iter().enumerate() {
                ord_pos[k] = top[valid[i]];
            }
        }
        // #305: strict `>` and a non-finite score is flat.
        (None, None) => {
            for k in 0..n {
                ord_pos[k] = trading_gpu::long_position(ord_scores[k], threshold) as f64;
            }
        }
    }
    if any_held {
        crate::trading::hold_positions(&ord_keys, &ord_scored, &mut ord_pos, |g| previous.get(g).copied().unwrap_or(0.0));
    }
    // #1203: a group whose first labelled row has no score has no seam
    // position: no score decides that row. Under `Flat` the carried position
    // closed at the earlier block's end; under `Hold` the row keeps it.
    let mut leading_gap: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for k in 0..labelled.len() {
        if !lab_scored[k] && (k == 0 || lab_groups[k] != lab_groups[k - 1]) {
            leading_gap.insert(lab_groups[k]);
        }
    }
    for (k, &i) in order.iter().enumerate() {
        let row = &rows[valid[i]];
        let pos = ord_pos[k];
        let key = ord_keys[k];
        let prev = previous.get(&key).copied().unwrap_or(0.0);
        let delta = pos - prev;
        let cost = cost_side * delta.abs();
        positions[i] = pos;
        costs[i] = cost;
        net[i] = pos * row.target - cost;
        previous.insert(key, pos);
        if !leading_gap.contains(&row.group) {
            first_positions.entry(row.group).or_insert(pos);
        }

        stats.total_abs_delta += delta.abs();
        stats.gross_sum += pos * row.target;
        stats.net_sum += net[i];
        if delta < 0.0 {
            stats.rule_exits += 1;
        }
        stats.total_cost += cost;
        if delta != 0.0 {
            stats.trades += 1;
        }
        if prev == 0.0 && pos > 0.0 {
            entered[i] = 1.0;
            stats.entries += 1;
        }
        if pos > 0.0 {
            stats.active += 1;
            if row.target > 0.0 {
                stats.hits += 1;
                stats.wins += 1;
                stats.win_sum += row.target;
            } else {
                stats.losses += 1;
                stats.loss_sum += row.target;
            }
        }
    }

    // #145: charge the final exit of every book (#1203: every run under
    // `Flat`) at its last row. A flat strategy is free; a held position pays
    // the same per-side cost as its entry.
    let mut last_of_key: HashMap<u64, usize> = HashMap::new();
    for (k, &key) in ord_keys.iter().enumerate() {
        last_of_key.insert(key, k);
    }
    for (key, k) in last_of_key {
        let p = previous.get(&key).copied().unwrap_or(0.0);
        if p != 0.0 {
            let i = order[k];
            let c = cost_side * p.abs();
            net[i] -= c;
            stats.total_cost += c;
            stats.net_sum -= c;
            stats.exits += 1;
            if row_ends[valid[i]] {
                stats.missing_score_exits += 1;
            }
        }
    }
    // #1203: under `Flat` a group leaves the window with the position of its
    // last run, flat when a row without a score ends its labelled rows; the
    // keys of the later runs close here.
    if lab_scored.iter().any(|&s| !s) && missing == MissingScore::Flat {
        let mut row_key = vec![0u64; rows.len()];
        for (k, &i) in order.iter().enumerate() {
            row_key[valid[i]] = ord_keys[k];
        }
        let mut ends: Vec<(u64, f64)> = Vec::new();
        for k in 0..labelled.len() {
            if k + 1 == labelled.len() || lab_groups[k + 1] != lab_groups[k] {
                let end = if lab_scored[k] { previous.get(&row_key[labelled[k]]).copied().unwrap_or(0.0) } else { 0.0 };
                ends.push((lab_groups[k], end));
            }
        }
        for (k, &key) in ord_keys.iter().enumerate() {
            if ord_runs[k] > 0 {
                previous.remove(&key);
            }
        }
        for (g, end) in ends {
            previous.insert(g, end);
        }
    }
    let end_positions = previous.clone();

    // #304: recompute the same long-only book through the fused canonical
    // reduction. Rows are already in `(group, time)` order, which is the
    // group-contiguous order the reducer requires. The scalar evaluator above
    // stays the reported authority; this block is a device/canonical check.
    let device = if device_pnl {
        let started = std::time::Instant::now();
        // #1203: a held row repeats the score of the row before it in its book, so
        // the threshold reduction holds the same position; a leading held row is flat.
        let mut dev_scores = ord_scores.clone();
        for k in 0..n {
            if !ord_scored[k] {
                dev_scores[k] = if k > 0 && ord_keys[k] == ord_keys[k - 1] { dev_scores[k - 1] } else { f64::NAN };
            }
        }
        let params = crate::trading_gpu::PnlParams {
            threshold,
            per_side_bps: cost_side * 10_000.0,
            segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
            cash_return: 0.0,
        };
        #[cfg(feature = "cuda")]
        let report = crate::trading_gpu::reduce_device(
            &dev_scores,
            &ord_returns,
            &ord_keys,
            &params,
            256,
            65_536,
        );
        #[cfg(feature = "cuda")]
        let backend = if report.is_some() {
            "cuda".to_string()
        } else {
            "cpu-canonical".to_string()
        };
        #[cfg(not(feature = "cuda"))]
        let report: Option<crate::trading_gpu::PnlReport> = None;
        #[cfg(not(feature = "cuda"))]
        let backend = "cpu-canonical".to_string();
        let host_s = started.elapsed().as_secs_f64();
        let canonical = crate::trading_gpu::reduce(&dev_scores, &ord_returns, &ord_keys, &params)
            .unwrap_or_else(|e| panic!("canonical P&L reduction failed: {}", e));
        let report = report.unwrap_or_else(|| canonical.clone());
        let bit_equal = report.strategy.identical(&canonical.strategy);
        let counts_equal = {
            let s = &report.strategy;
            s.rows == stats.rows as u64
                && s.long_rows == stats.active as u64
                && s.entries == stats.entries as u64
                && s.trades == stats.trades as u64
                && s.hits == stats.hits as u64
                && s.final_exits == stats.exits as u64
                && s.exits == (stats.rule_exits + stats.exits) as u64
        };
        let cpu_net = net.iter().sum::<f64>();
        Some(DevicePnlCheck {
            host_s,
            started,
            backend,
            segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
            tile_candidates: crate::trading_gpu::MAX_TILE_CANDIDATES,
            strategy: report.strategy,
            cash: report.cash,
            buy_and_hold: report.buy_and_hold,
            cpu_net,
            net_delta: report.strategy.net - cpu_net,
            bit_equal,
            counts_equal,
            f32_flip_rows: crate::trading_gpu::f32_flip_rows(&dev_scores, threshold),
        })
    } else {
        None
    };

    let scalar_net = net.iter().sum::<f64>();
    // #1203: the scan prices the scored rows; the book adds the held rows' return.
    let held_gross: f64 = (0..n).filter(|&k| !ord_scored[k]).map(|k| ord_pos[k] * ord_returns[k]).sum();
    let band = band_scan.map(
        |(_, band_net, groups, carry, backend, fallback_reason)| BandCheck {
            spec: None,
            band_width,
            min_hold,
            carry,
            groups,
            net: if any_held { band_net + held_gross } else { band_net },
            scalar_net,
            backend,
            fallback_reason,
        },
    );

    // #306: exact hindsight DP for the same rows and costs. It is reported as
    // a bound only; the threshold (or band) book above remains the strategy.
    let hindsight_bound = if policy == Policy::Dp {
        let (scan, backend, fallback_reason) = scan_with_backend(
            gpu,
            "hindsight DP scan",
            || {
                crate::trading_gpu::dp_scan_device(
                    &ord_returns,
                    &ord_keys,
                    cost_side,
                    crate::trading_gpu::SCAN_THREADS,
                    device_chunk,
                )
            },
            || crate::trading_gpu::dp_scan(&ord_returns, &ord_keys, cost_side, 65_536),
        );
        let bound = scan.bound(cost_side);
        Some(HindsightBound {
            label: "hindsight_bound".to_string(),
            net: bound,
            rule_net: scalar_net,
            gap: bound - scalar_net,
            groups: scan.carries.len(),
            rule_le_bound: scalar_net <= bound + 1e-12,
            backend,
            fallback_reason,
        })
    } else {
        None
    };


    // Equal-weight the groups present at each timestamp. Times are sorted so the
    // return series has the same order as the Python reference.
    let ord_positions: Vec<f64> = order.iter().map(|&i| positions[i]).collect();
    let group_order = order.clone();
    let mut time_order = order;
    time_order.sort_by(|&a, &b| {
        let ra = &rows[valid[a]];
        let rb = &rows[valid[b]];
        ra.time
            .partial_cmp(&rb.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ra.group.cmp(&rb.group))
    });
    let mut cross = CrossSection::with_bucket(bucket);
    for &i in &time_order {
        let row = &rows[valid[i]];
        cross.add(row.time, net[i], positions[i] * row.target);
    }
    let (returns, gross_returns) = cross.series();
    let mut book_cross = CrossSection::with_bucket(bucket);
    for &i in &time_order {
        book_cross.add(rows[valid[i]].time, positions[i], entered[i]);
    }
    let exposure: Vec<f64> = book_cross.periods.iter().map(|p| p.net_sum / p.count as f64).collect();
    let entries: Vec<f64> = book_cross.periods.iter().map(|p| p.gross_sum).collect();
    let mut entry_stamps: Vec<f64> = (0..n).filter(|&i| entered[i] > 0.0).map(|i| rows[valid[i]].time).collect();
    entry_stamps.sort_by(f64::total_cmp);
    entry_stamps.dedup();
    let entry_times = entry_stamps.len();

    // #310: the control ladder on the same rows, the same cost and the same
    // equal-weight period aggregation. Every control starts flat, like the
    // #304 reduction; the strategy's carried seam is not applied to controls.
    let controls = control_input.map(|c| {
        let period_series = |ord_net: &[f64]| -> Vec<f64> {
            let mut by_index = vec![0.0f64; n];
            for (k, &i) in group_order.iter().enumerate() {
                by_index[i] = ord_net[k];
            }
            let mut cs = CrossSection::with_bucket(bucket);
            for &i in &time_order {
                cs.add(rows[valid[i]].time, by_index[i], 0.0);
            }
            cs.series().0
        };
        crate::baselines::evaluate_controls(
            &ord_positions,
            &ord_returns,
            &ord_keys,
            cost_side,
            c.train_mean,
            c.seed,
            ppy,
            c.check_device,
            &period_series,
        )
        .unwrap_or_else(|e| panic!("control ladder failed: {}", e))
    });

    let times: Vec<f64> = cross.periods.iter().map(|p| p.time).collect();
    let counts: Vec<usize> = cross.periods.iter().map(|p| p.count).collect();
    let mut market_cross = CrossSection::with_bucket(bucket);
    for &i in &time_order {
        let row = &rows[valid[i]];
        market_cross.add(row.time, row.target, row.target);
    }
    let market = market_cross.series().0;
    let book = if keep_book {
        let mut period_of = vec![0u32; n];
        let mut current: i64 = -1;
        let mut last_time = f64::NAN;
        let mut period_count: Vec<u32> = Vec::new();
        for &i in &time_order {
            let t = if bucket > 0.0 { (rows[valid[i]].time / bucket).floor() } else { rows[valid[i]].time };
            if current < 0 || t != last_time {
                current += 1;
                last_time = t;
                period_count.push(0);
            }
            period_of[i] = current as u32;
            period_count[current as usize] += 1;
        }
        Some(NullBook {
            positions: ord_positions.clone(),
            returns: ord_returns.clone(),
            groups: ord_keys.clone(),
            period: group_order.iter().map(|&i| period_of[i]).collect(),
            period_count,
            period_cost: gross_returns.iter().zip(&returns).map(|(g, n)| g - n).collect(),
            side: cost_side,
        })
    } else {
        None
    };

    let metrics = summarize(&returns, &gross_returns, &stats, ppy);
    WindowEval {
        metrics,
        returns,
        gross_returns,
        times,
        counts,
        market,
        exposure,
        entries,
        entry_times,
        book,
        stats,
        first_positions,
        end_positions,
        device,
        band,
        hindsight_bound,
        controls,
    }
}

pub(crate) fn summarize(returns: &[f64], gross: &[f64], stats: &Stats, ppy: f64) -> Metrics {
    let periods = returns.len();
    let mean_return = if periods == 0 {
        f64::NAN
    } else {
        returns.iter().sum::<f64>() / periods as f64
    };
    let sharpe_ratio = sharpe(returns, ppy);
    let gross_sharpe = sharpe(gross, ppy);
    let (total_return, cagr, max_drawdown) = equity_metrics(returns, ppy);
    // #307: the same period series through the chunk-mergeable risk monoids.
    // `max_drawdown` stays on the serial reference; the risk-module parity test
    // pins the monoid to that reference, and Sortino/log-growth are new report
    // fields from the monoid.
    let risk = crate::risk::risk_stats(returns, ppy, crate::risk::DEFAULT_SORTINO_MAR);
    let exposure = ratio(stats.active, stats.rows);
    let hit_rate = ratio(stats.hits, stats.active);
    let avg_cost = if stats.active == 0 {
        f64::NAN
    } else {
        stats.total_cost / stats.active as f64
    };
    let break_even_hit_rate = if stats.wins > 0 && stats.losses > 0 && avg_cost.is_finite() {
        let avg_win = stats.win_sum / stats.wins as f64;
        let avg_loss = (stats.loss_sum / stats.losses as f64).abs();
        (avg_loss + avg_cost) / (avg_win + avg_loss)
    } else if stats.losses == 0 && stats.wins > 0 {
        0.0
    } else {
        f64::NAN
    };
    let years = if periods == 0 || ppy <= 0.0 {
        f64::NAN
    } else {
        periods as f64 / ppy
    };
    let turnover = if years.is_finite() && years > 0.0 {
        stats.total_abs_delta / years
    } else {
        f64::NAN
    };
    Metrics {
        sharpe: sharpe_ratio,
        cagr,
        max_drawdown,
        sortino: risk.sortino,
        log_growth: risk.log_growth_per_period,
        sortino_mar: risk.sortino_mar,
        turnover,
        exposure,
        hit_rate,
        break_even_hit_rate,
        trades: stats.trades,
        entries: stats.entries,
        rows: stats.rows,
        active_rows: stats.active,
        periods,
        periods_per_year: ppy,
        total_return,
        mean_return,
        avg_cost_bps: avg_cost * 10_000.0,
        gross_sharpe,
        total_cost: stats.total_cost,
        total_abs_delta: stats.total_abs_delta,
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        f64::NAN
    } else {
        num as f64 / den as f64
    }
}

/// Annualised Sharpe with the sample standard deviation (ddof = 1).
pub fn sharpe(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.len() < 2 || periods_per_year <= 0.0 {
        return f64::NAN;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<f64>()
        / (n - 1.0);
    let sd = var.sqrt();
    if sd > 0.0 {
        mean / sd * periods_per_year.sqrt()
    } else {
        f64::NAN
    }
}

pub(crate) fn equity_metrics(returns: &[f64], ppy: f64) -> (f64, f64, f64) {
    if returns.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut equity = 1.0f64;
    let mut peak = 1.0f64;
    let mut max_dd = 0.0f64;
    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = if peak > 0.0 { equity / peak - 1.0 } else { 0.0 };
        if dd < max_dd {
            max_dd = dd;
        }
    }
    let total_return = equity - 1.0;
    let cagr = if equity > 0.0 && ppy > 0.0 {
        equity.powf(ppy / returns.len() as f64) - 1.0
    } else {
        -1.0
    };
    (total_return, cagr, max_dd)
}

// ---------------------------------------------------------------------------
// Quantiles and frequency
// ---------------------------------------------------------------------------

/// Linear-interpolation quantile (numpy's default, Hyndman-Fan type 7).
///
/// #305: this delegates to the exact order-preserving radix selection in
/// [`crate::trading_gpu`]. The result is bit-identical to the former sort
/// implementation (pinned by `tests/walkforward.rs` and `tests/pnl_gpu.rs`),
/// but selection is `O(4n)` counts and cannot depend on a comparison-sort
/// tie order. Non-finite values are filtered exactly as before.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    trading_gpu::type7_quantile_f64(values, q).theta
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) * 0.5)
    }
}

/// Where the period width `S` of the annualisation came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodSource {
    TargetSpec,
    HorizonFlag,
    Explicit,
    InferredLabelledRows,
}

impl PeriodSource {
    pub fn as_str(self) -> &'static str {
        match self {
            PeriodSource::TargetSpec => "target_spec",
            PeriodSource::HorizonFlag => "horizon_flag",
            PeriodSource::Explicit => "explicit",
            PeriodSource::InferredLabelledRows => "inferred_labelled_rows",
        }
    }
}

/// The period width declared by the run: the target spec's horizon or, when
/// wider, its label stride (#1053), else `--horizon`. One independent return
/// per group spans exactly this width.
fn declared_period(spec: &EvalSpec, opts: &ProtocolOptions) -> Option<(f64, PeriodSource)> {
    if spec.horizon.is_finite() && spec.horizon >= 1.0 && spec.horizon <= SECONDS_PER_YEAR {
        let source = if opts.horizon_from_spec { PeriodSource::TargetSpec } else { PeriodSource::HorizonFlag };
        let width = match (source, opts.stride_seconds) {
            (PeriodSource::TargetSpec, Some(stride)) => spec.horizon.max(stride),
            _ => spec.horizon,
        };
        if width > SECONDS_PER_YEAR {
            return None;
        }
        Some((width, source))
    } else {
        None
    }
}

/// Refuse an explicit `--periods-per-year` that contradicts the declared period.
fn check_explicit_ppy(spec: &EvalSpec, opts: &ProtocolOptions) -> Result<(), String> {
    if let (Some(given), Some((s, PeriodSource::TargetSpec))) = (spec.periods_per_year, declared_period(spec, opts)) {
        let derived = SECONDS_PER_YEAR / s;
        if ((given - derived) / derived).abs() > 1e-9 {
            return Err(format!(
                "--periods-per-year {} conflicts with {} derived from the target spec period width ({} s, the larger of horizon_seconds and stride_seconds); remove the flag or edit the spec",
                given, derived, s
            ));
        }
    }
    Ok(())
}

/// Annualisation of one window: `(periods_per_year, period_seconds, bucket
/// width, source)`. The bucket width is `0` (one period per timestamp) unless
/// the run declares the period.
fn window_period(spec: &EvalSpec, opts: &ProtocolOptions, rows: &[RowBase]) -> (f64, Option<f64>, f64, PeriodSource) {
    if let Some((s, source)) = declared_period(spec, opts) {
        return match spec.periods_per_year {
            Some(v) if source == PeriodSource::HorizonFlag => (v, Some(s), s, PeriodSource::Explicit),
            _ => (SECONDS_PER_YEAR / s, Some(s), s, source),
        };
    }
    match spec.periods_per_year {
        Some(v) => (v, None, 0.0, PeriodSource::Explicit),
        None => {
            let (ppy, dt) = infer_frequency(rows);
            (ppy, dt, 0.0, PeriodSource::InferredLabelledRows)
        }
    }
}

/// Median step of the distinct times of rows that carry a finite target.
fn infer_frequency(rows: &[RowBase]) -> (f64, Option<f64>) {
    let mut times: Vec<f64> = rows.iter().filter(|r| r.target.is_finite()).map(|r| r.time).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| *a == *b);
    let mut diffs: Vec<f64> = times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 1e-9)
        .collect();
    match median(&mut diffs) {
        Some(dt) if dt >= 1.0 && dt <= SECONDS_PER_YEAR => (SECONDS_PER_YEAR / dt, Some(dt)),
        _ => (365.0, None),
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

pub(crate) fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub(crate) fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

/// #304: one fused-reduction total as JSON.
pub(crate) fn pnl_totals_json(t: &crate::trading_gpu::PnlTotals) -> String {
    format!(
        "{{\"rows\":{},\"long_rows\":{},\"entries\":{},\"exits\":{},\
         \"trades\":{},\"hits\":{},\"final_exits\":{},\"gross\":{},\
         \"cost\":{},\"turnover\":{},\"net\":{},\"exposure\":{},\"hit_rate\":{}}}",
        t.rows,
        t.long_rows,
        t.entries,
        t.exits,
        t.trades,
        t.hits,
        t.final_exits,
        jnum(t.gross),
        jnum(t.cost),
        jnum(t.turnover),
        jnum(t.net),
        jnum(t.exposure()),
        jnum(t.hit_rate())
    )
}

fn device_pnl_json(d: &DevicePnlCheck) -> String {
    format!(
        "{{\"backend\":{},\"segment_rows\":{},\"tile_candidates\":{},\
         \"strategy\":{},\"cash\":{},\"buy_and_hold\":{},\
         \"cpu_net\":{},\"net_delta\":{},\"bit_equal\":{},\"counts_equal\":{},\
         \"f32_flip_rows\":{}}}",
        jstr(&d.backend),
        d.segment_rows,
        d.tile_candidates,
        pnl_totals_json(&d.strategy),
        pnl_totals_json(&d.cash),
        pnl_totals_json(&d.buy_and_hold),
        jnum(d.cpu_net),
        jnum(d.net_delta),
        d.bit_equal,
        d.counts_equal,
        d.f32_flip_rows
    )
}

/// #310: the control ladder as JSON.
fn controls_json(c: &crate::baselines::ControlSet) -> String {
    let rows: Vec<String> = c
        .rows
        .iter()
        .map(|r| {
            format!(
                "{{\"control\":{},\"totals\":{},\"exposure\":{},\"turnover\":{},\
                 \"beta\":{},\"excess_over_cash\":{},\"sharpe\":{},\
                 \"mean_return\":{},\"periods\":{}}}",
                jstr(r.kind.as_str()),
                pnl_totals_json(&r.totals),
                jnum(r.exposure),
                jnum(r.turnover),
                jnum(r.beta),
                jnum(r.excess_over_cash),
                jnum(r.sharpe),
                jnum(r.mean_return),
                r.periods
            )
        })
        .collect();
    let device = c.device.as_ref().map_or("null".to_string(), |d| {
        format!(
            "{{\"backend\":{},\"max_abs_delta\":{},\"counts_equal\":{}}}",
            jstr(&d.backend),
            jnum(d.max_abs_delta),
            d.counts_equal
        )
    });
    format!(
        "{{\"seed\":{},\"constant_side\":{},\"train_mean\":{},\
         \"beta_benchmark\":\"buy_and_hold\",\"controls\":[{}],\"device\":{}}}",
        c.seed,
        jstr(if c.constant_long { "long" } else { "flat" }),
        jnum(c.train_mean),
        rows.join(","),
        device
    )
}

fn band_check_json(b: &BandCheck) -> String {
    format!(
        "{{\"band_width\":{},\"min_hold\":{},\"carry\":{},\"groups\":{},\
         \"net\":{},\"scalar_net\":{},\"net_delta\":{},\"backend\":{},\
         \"fallback_reason\":{}{}}}",
        jnum(b.band_width),
        b.min_hold,
        jstr(&b.carry),
        b.groups,
        jnum(b.net),
        jnum(b.scalar_net),
        jnum(b.net - b.scalar_net),
        jstr(&b.backend),
        b.fallback_reason.as_deref().map_or("null".to_string(), jstr),
        b.spec.as_ref().map_or_else(String::new, |s| s.json_fields())
    )
}

fn hindsight_bound_json(h: &HindsightBound) -> String {
    format!(
        "{{\"label\":{},\"is_strategy\":false,\"net\":{},\"rule_net\":{},\
         \"gap\":{},\"groups\":{},\"rule_le_bound\":{},\"backend\":{},\
         \"fallback_reason\":{}}}",
        jstr(&h.label),
        jnum(h.net),
        jnum(h.rule_net),
        jnum(h.gap),
        h.groups,
        h.rule_le_bound,
        jstr(&h.backend),
        h.fallback_reason.as_deref().map_or("null".to_string(), jstr)
    )
}

fn md_num(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "n/a".to_string()
    }
}

fn md_bps(v: f64) -> String {
    if v.is_finite() {
        format!("{:.4}", v)
    } else {
        "n/a".to_string()
    }
}

impl Metrics {
    fn json(&self) -> String {
        format!(
            "{{\"sharpe\":{},\"cagr\":{},\"max_drawdown\":{},\"sortino\":{},\
             \"sortino_mar\":{},\"log_growth\":{},\"turnover\":{},\
             \"exposure\":{},\"hit_rate\":{},\"break_even_hit_rate\":{},\
             \"trades\":{},\"entries\":{},\"rows\":{},\"active_rows\":{},\
             \"periods\":{},\"periods_per_year\":{},\"total_return\":{},\
             \"mean_return\":{},\"avg_cost_bps\":{},\"gross_sharpe\":{},\
             \"total_cost\":{},\"total_abs_delta\":{}}}",
            jnum(self.sharpe),
            jnum(self.cagr),
            jnum(self.max_drawdown),
            jnum(self.sortino),
            jnum(self.sortino_mar),
            jnum(self.log_growth),
            jnum(self.turnover),
            jnum(self.exposure),
            jnum(self.hit_rate),
            jnum(self.break_even_hit_rate),
            self.trades,
            self.entries,
            self.rows,
            self.active_rows,
            self.periods,
            jnum(self.periods_per_year),
            jnum(self.total_return),
            jnum(self.mean_return),
            jnum(self.avg_cost_bps),
            jnum(self.gross_sharpe),
            jnum(self.total_cost),
            jnum(self.total_abs_delta)
        )
    }
}

/// #1094: the `cut` object of a window.
fn cut_json(c: &CutReport) -> String {
    let num = |v: Option<f64>| v.map_or("null".to_string(), jnum);
    let int = |v: Option<u64>| v.map_or("null".to_string(), |x| x.to_string());
    let (p, r) = (c.cut.as_ref(), c.reference.as_ref());
    let profile: Vec<String> = c
        .profile
        .iter()
        .map(|(share, q)| {
            format!(
                "{{\"long_share\":{},\"index\":{},\"value\":{},\"fitness\":{},\"net\":{},\"entry_times\":{}}}",
                jnum(*share),
                q.index,
                jnum(q.value),
                jnum(q.fitness),
                jnum(q.net),
                q.entry_times
            )
        })
        .collect();
    format!(
        "{{\"source\":{},\"value\":{},\"index\":{},\"rows\":{},\"fitness\":{},\"net\":{},\"flips\":{},\
         \"exits\":{},\"entry_times\":{},\"exposure\":{},\"reference\":{{\"value\":{},\"index\":{},\
         \"fitness\":{},\"net\":{},\"entry_times\":{},\"exposure\":{}}},\"plateau\":{},\
         \"floor\":{{\"min_entry_times\":{}}},\"blocks\":{},\"lambda\":{},\"fallback\":{},\
         \"fallback_detail\":{},\"profile\":[{}]}}",
        jstr(c.source.as_str()),
        jnum(c.value),
        int(p.map(|q| q.index as u64)),
        c.rows,
        num(p.map(|q| q.fitness)),
        num(p.map(|q| q.net)),
        int(p.map(|q| q.flips as u64)),
        int(p.map(|q| q.exits as u64)),
        int(p.map(|q| q.entry_times as u64)),
        num(p.map(|q| q.exposure)),
        jnum(c.reference_value),
        int(r.map(|q| q.index as u64)),
        num(r.map(|q| q.fitness)),
        num(r.map(|q| q.net)),
        int(r.map(|q| q.entry_times as u64)),
        num(r.map(|q| q.exposure)),
        c.plateau.as_ref().map_or("null".to_string(), |q| format!(
            "{{\"lo\":{},\"hi\":{},\"rows_lo\":{},\"rows_hi\":{}}}",
            jnum(q.lo),
            jnum(q.hi),
            q.rows_lo,
            q.rows_hi
        )),
        c.fit.min_entry_times,
        c.fit.blocks,
        jnum(c.fit.lambda),
        c.fallback.map_or("null".to_string(), jstr),
        c.fallback_detail.as_deref().map_or("null".to_string(), jstr),
        profile.join(",")
    )
}

impl WindowResult {
    fn json(&self) -> String {
        let mut s = format!(
            "{{\"label\":{},\"train_start\":{},\"train_end\":{},\"test_start\":{},\
             \"test_end\":{},\"purge_seconds\":{},\"expression\":{},\
             \"candidate\":{},\"validation_error\":{},\"threshold\":{},\
             \"threshold_source\":{},\"train_rows\":{},\"validation_rows\":{},\
             \"test_rows\":{},\"dropped_rows\":{},\"inferred_period_seconds\":{},\
             \"metrics\":{},\"device_pnl\":{},\"controls\":{},\"band\":{},\"hindsight_bound\":{},\
             \"band_selection\":{},\"train_last_time\":{}}}",
            jstr(&self.window.label),
            jnum(self.window.train_start),
            jnum(self.window.train_end),
            jnum(self.window.test_start),
            jnum(self.window.test_end),
            jnum(self.window.test_start - self.window.train_end),
            jstr(&self.expression),
            self.candidate,
            self.validation_error.map_or("null".to_string(), jnum),
            jnum(self.threshold),
            jstr(&self.threshold_source),
            self.train_rows,
            self.validation_rows,
            self.test_rows,
            self.dropped_rows,
            self.inferred_period_seconds.map_or("null".to_string(), jnum),
            self.metrics.json(),
            self.device_pnl
                .as_ref()
                .map_or("null".to_string(), device_pnl_json),
            self.controls
                .as_ref()
                .map_or("null".to_string(), controls_json),
            self.band
                .as_ref()
                .map_or("null".to_string(), band_check_json),
            self.hindsight_bound
                .as_ref()
                .map_or("null".to_string(), hindsight_bound_json),
            self.band_selection
                .as_ref()
                .map_or("null".to_string(), |s| s.json()),
            self.train_last_time.map_or("null".to_string(), jnum)
        );
        if !self.member_thresholds.is_empty() {
            s.pop();
            s.push_str(&format!(",\"member_thresholds\":[{}]}}", self.member_thresholds.iter().map(|t| jnum(*t)).collect::<Vec<_>>().join(",")));
        }
        if let Some(c) = &self.cut {
            s.pop();
            s.push_str(&format!(",\"cut\":{}}}", cut_json(c)));
        }
        if self.in_sample {
            s.pop();
            s.push_str(",\"in_sample\":true}");
        }
        if let Some(n) = self.entry_times {
            s.pop();
            s.push_str(&format!(",\"entry_times\":{}}}", n));
        }
        s
    }
}

impl Member {
    /// #1176: one row of `members`: per-window metrics, pooled metrics with the
    /// cost ladder, and the per-period series on the report's period grid.
    fn json(&self) -> String {
        let r = &self.report;
        let series = |v: &[f64]| v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(",");
        let windows: Vec<String> = r
            .windows
            .iter()
            .map(|w| {
                format!(
                    "{{\"label\":{},\"threshold\":{},\"threshold_source\":{},\"train_rows\":{},\"validation_rows\":{},\
                     \"test_rows\":{},\"dropped_rows\":{},\"in_sample\":{},\"metrics\":{}}}",
                    jstr(&w.window.label),
                    jnum(w.threshold),
                    jstr(&w.threshold_source),
                    w.train_rows,
                    w.validation_rows,
                    w.test_rows,
                    w.dropped_rows,
                    w.in_sample,
                    w.metrics.json()
                )
            })
            .collect();
        let empty = PeriodExtra::default();
        let x = r.period_extra.as_ref().unwrap_or(&empty);
        let e = &r.economics;
        let mut pooled = r.pooled.json();
        pooled.pop();
        pooled.push_str(&format!(
            ",\"net_mean\":{},\"entry_times\":{},\"seam_cost\":{},\"missing_score_exits\":{},\"unpriced_target_rows\":{},\
             \"break_even_bps_per_side\":{},\"break_even_status\":{},\"cost_ladder\":{}}}",
            jnum(r.pooled.mean_return),
            x.entry_times,
            jnum(r.seam_cost),
            r.missing_score_exits,
            r.unpriced_target_rows,
            jnum(e.break_even_bps_per_side),
            jstr(e.break_even_status),
            e.ladder_json()
        ));
        let rows = r.period_grid.as_ref().map_or(String::new(), |g| {
            format!(",\"period_rows\":[{}]", g.period_rows.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","))
        });
        format!(
            "{{\"archive_rank\":{},\"label\":{},\"complexity\":{},\"expression\":{},\"constant\":{},\"dropped_rows\":{},\
             \"windows\":[{}],\"pooled\":{},\"series\":{{\"period_time\":[{}],\"period_net\":[{}],\"period_gross\":[{}],\
             \"period_cost\":[{}],\"period_exposure\":[{}],\"period_entries\":[{}],\"period_entry_stamp\":[{}],\
             \"period_bh\":[{}],\"period_bh_net\":[{}]{}}}}}",
            self.archive_rank,
            jstr(&self.label),
            self.complexity,
            jstr(r.candidates.first().map_or("", String::as_str)),
            self.constant,
            r.dropped_rows,
            windows.join(","),
            pooled,
            series(&r.period_time),
            series(&r.period_net),
            series(&x.gross),
            series(&x.cost),
            series(&x.exposure),
            series(&x.entries),
            series(&x.entry_stamp),
            series(&x.bh),
            series(&x.bh_net),
            rows
        )
    }
}

impl Report {
    /// #482: `source`, `run`, `run_sha256` and the candidate labels, with a
    /// leading comma. Empty when the candidates came from `--expr`.
    fn front_json(&self, paths: bool) -> String {
        match &self.front {
            Some(f) => format!(
                ",\"source\":\"{}\"{},\"run_sha256\":{},\"candidate_labels\":[{}]",
                if self.members.is_some() { "front_all_members" } else { "front" },
                if paths { format!(",\"run\":{}", jstr(&f.run)) } else { String::new() },
                jstr(&f.run_sha256),
                f.labels.iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",")
            ),
            None => String::new(),
        }
    }

    /// Machine-readable JSON. The deterministic report, with the `--front` path.
    pub fn json(&self) -> String {
        self.json_with(true)
    }

    /// [`Report::json`] without file paths: its SHA-256 is the reproduction hash.
    pub fn hashed_json(&self) -> String {
        self.json_with(false)
    }

    fn json_with(&self, paths: bool) -> String {
        let mut s = self.base_json(paths);
        s.pop();
        s.push_str(&format!(
            ",\"split\":{{\"required_purge_seconds\":{},\"purge_seconds\":{},\"horizon_seconds\":{},\
             \"embargo_seconds\":{},\"min_group_step_seconds\":{}{}}},\"economics\":{},\"inference\":{}}}",
            jnum(self.required_purge_seconds),
            jnum(self.min_purge_seconds),
            jnum(self.horizon),
            jnum(self.embargo),
            self.min_group_step.map_or("null".to_string(), jnum),
            match (&self.fit_label, paths) {
                (Some(f), true) => format!(",\"fit_label\":{}", f),
                _ => String::new(),
            },
            self.economics.json(),
            self.inference.as_ref().map_or("null".to_string(), |i| i.json())
        ));
        if let Some(r) = &self.regimes {
            s.pop();
            s.push_str(&format!(",\"regimes\":{}}}", r.json()));
        }
        if let Some(m) = &self.members {
            s.pop();
            let items: Vec<String> = m.iter().map(Member::json).collect();
            s.push_str(&format!(",\"diagnostic\":true,\"members\":[{}]}}", items.join(",")));
        }
        s
    }

    /// [`Report::json`] with a `protocol` object (lockbox and ledger facts).
    pub fn json_with_protocol(&self, protocol: &str) -> String {
        let mut s = self.json();
        s.pop();
        s.push_str(&format!(",\"protocol\":{}}}", protocol));
        s
    }

    /// The printed report: [`Report::json_with_protocol`] (or [`Report::json`])
    /// plus the #284 phase ledger and the `--batch-pnl` timings. Both hold
    /// wall-clock seconds, so they stay outside the reproduction hash.
    pub fn json_with_ledger(&self, protocol: Option<&str>) -> String {
        use crate::gpu_stats::Stage;
        let mut s = match protocol {
            Some(p) => self.json_with_protocol(p),
            None => self.json(),
        };
        s.pop();
        if let Some(b) = &self.batch_pnl {
            s.push_str(&format!(",\"batch_timings\":{}", b.timings_json()));
        }
        if let Some(sc) = &self.scan {
            s.push_str(&format!(",\"scan\":{}", sc.json()));
        }
        s.push_str(&format!(
            ",\"ledger\":{}}}",
            self.ledger.phase_json(
                "walkforward",
                &[Stage::Vary, Stage::Eval, Stage::Fit, Stage::Select, Stage::Archive, Stage::Features]
            )
        ));
        s
    }

    /// Window definitions, cost model, position rule and metrics, as required
    /// by issue #124.
    fn base_json(&self, paths: bool) -> String {
        let windows: Vec<String> = self.windows.iter().map(WindowResult::json).collect();
        let lb = self.guarded_lockbox;
        let overlap: Vec<String> = self
            .windows
            .iter()
            .map(|w| {
                let train_overlap = w.window.train_start < lb.end as f64
                    && w.window.train_end > lb.start as f64;
                let test_overlap = w.window.test_start < lb.end as f64
                    && w.window.test_end > lb.start as f64;
                format!(
                    "{{\"label\":{},\"train_overlap\":{},\"test_overlap\":{}}}",
                    jstr(&w.window.label),
                    train_overlap,
                    test_overlap
                )
            })
            .collect();
        let lockbox_json = format!(
            "{{\"start\":{},\"end\":{},\"permitted\":{},\"evaluated\":{},\"windows\":[{}]}}",
            lb.start,
            lb.end,
            self.allow_lockbox,
            self.allow_lockbox,
            overlap.join(",")
        );
        let position_rule = match self.policy {
            Policy::Threshold => {
                "long/flat: 1 if score > threshold else 0; never short; no leverage"
            }
            Policy::Band => {
                "long/flat no-trade band: enter when score > threshold+band, exit when \
                 score <= threshold-band after the minimum hold; never short; no leverage"
            }
            Policy::Dp => {
                "reported book is the threshold rule; the DP value is a hindsight_bound, \
                 not a strategy; never short; no leverage"
            }
            Policy::TopTercile => {
                "top-tercile long-only: per timestamp 1 for the top floor(n/3) finite scores (ties by row order), else 0; no threshold; never short; no leverage"
            }
            Policy::Vote => {
                "long/flat vote book: long when at least votes_required of the member expressions have score > their own threshold, flat otherwise (a member with a non-finite score drops the row); with band B the long book stays long while votes >= votes_required - B; never short; no leverage"
            }
        };
        format!(
            "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"walkforward\",\
             \"expression\":{},\"target\":{},\"variables\":[{}],\"time\":{},\
             \"group_by\":{},\
             \"window_definition\":{},\
             \"purge\":{{\"horizon_seconds\":{},\"embargo_seconds\":{}}},\
             \"cost_model\":{{\"per_side_bps\":{},\"round_trip_bps\":{},\
             \"charged_on\":\"|position change| per group; per-window metrics start flat, pooled carries the boundary\"}},\
             \"cost\":{},\
             \"position_rule\":{},\
             \"policy\":{{\"name\":{},\"hindsight_bound\":{},\"band_width\":{},\
             \"band_units\":{},\"band_source\":{},\"min_hold\":{},\"short\":false,\"gpu\":{}{}}},\
             \"threshold\":{{\"quantile\":{},\"fixed\":{},\
             \"method\":\"exact order-preserving radix select + linear interpolation (numpy type 7) in f64\",\
             \"sources\":[\"train_type7\",\"validation_type7\",\"fixed\"],\
             \"screening_only\":false,\"source\":{}{}}},\
             \"annualization\":{{\"periods_per_year\":{},\"method\":{},\"source\":{},\"period_seconds\":{}}},\
             \"risk_model\":{{\"mode\":\"per-period\",\"periods_per_year\":{},\
             \"sharpe\":{{\"scale\":\"sqrt(periods_per_year)\",\"ddof\":1,\
             \"undefined\":\"n<2, sd=0 (cash) or periods_per_year<=0\"}},\
             \"sortino\":{{\"mar\":{},\
             \"denominator\":\"population RMS of min(R-MAR,0) over all periods (ddof=0)\",\
             \"scale\":\"sqrt(periods_per_year)\"}},\
             \"log_growth\":{{\"transform\":\"log1p\",\"unit\":\"per-period\",\
             \"ruin\":\"R<=-1 => -inf\"}},\
             \"drawdown\":{{\"equity\":\"log\",\"unit\":\"fraction of peak\",\
             \"sign\":\"negative or zero\",\"ruin\":\"-1\"}},\
             \"beta_alpha\":{{\"beta\":\"dimensionless\",\"alpha\":\"per-period\",\
             \"annualized_alpha_scale\":\"periods_per_year\"}}}},\
             \"candidates\":[{}],\"selection\":{{\"selected_on\":{},\
             \"band_select\":{},\"test_used_for_selection\":{},\
             \"candidate_count\":{}{}{}}},\
             \"pooled_seam\":{{\"reset_flat\":{},\"seam_cost\":{},\"seam_trades\":{}}},\
             \"book\":{{\"missing_score\":{},\"missing_score_exits\":{},\"unpriced_target_rows\":{}}},\
             \"sign_test\":{{\"positive\":{},\"total\":{},\"applicable\":{},\
             \"p_value\":{},\"method\":{}}},\
             \"lockbox\":{},\
             \"source\":{},\"memory_bound\":{},\"dropped_rows\":{},\
             \"device_pnl\":{},\"batch_pnl\":{},\"windows\":[{}],\"pooled\":{}}}",
            jstr(env!("CARGO_PKG_VERSION")),
            jstr(&self.expression),
            jstr(&self.target),
            self.variables
                .iter()
                .map(|v| jstr(v))
                .collect::<Vec<_>>()
                .join(","),
            jstr(&self.time_column),
            self.group_by.as_ref().map_or("null".to_string(), |g| jstr(g)),
            self.windows_json(),
            jnum(self.horizon),
            jnum(self.embargo),
            jnum(self.cost_bps),
            jnum(self.cost_bps * 2.0),
            self.cost.json(),
            jstr(position_rule),
            jstr(self.policy.as_str()),
            self.policy == Policy::Dp,
            jnum(self.band_width),
            jstr(self.band_units.name()),
            jstr(self.band_source),
            self.min_hold,
            self.gpu,
            self.vote_json(),
            jnum(self.threshold_quantile),
            self.threshold_fixed.map_or("null".to_string(), jnum),
            jstr(self
                .windows
                .first()
                .map(|w| w.threshold_source.as_str())
                .unwrap_or("none")),
            self.threshold_fit_json(),
            jnum(self.periods_per_year),
            jstr(&self.annualization),
            jstr(self.annualization_source),
            self.period_seconds.map_or("null".to_string(), jnum),
            jnum(self.periods_per_year),
            jnum(crate::risk::DEFAULT_SORTINO_MAR),
            self.candidates
                .iter()
                .map(|v| jstr(v))
                .collect::<Vec<_>>()
                .join(","),
            jstr(&self.selected_on),
            self.band_select,
            self.test_used_for_selection,
            self.candidates.len(),
            self.front_json(paths),
            self.expression_source
                .map(|s| format!(",\"expression_source\":\"{}\"", s.as_str()))
                .unwrap_or_default(),
            self.pool_reset_flat,
            jnum(self.seam_cost),
            self.seam_trades,
            jstr(self.missing_score.name()),
            self.missing_score_exits,
            self.unpriced_target_rows,
            self.sign_test.positive,
            self.sign_test.total,
            self.sign_test.applicable,
            self.sign_test.p_value.map_or("null".to_string(), jnum),
            jstr(&self.sign_test.method),
            lockbox_json,
            jstr(&self.source),
            jstr(&self.memory_bound),
            self.dropped_rows,
            self.device_pnl.as_ref().map_or("null".to_string(), |d| {
                format!(
                    "{{\"backend\":{},\"segment_rows\":{},\"tile_candidates\":{},\
                     \"windows\":{},\"rows\":{},\"max_abs_net_delta\":{},\
                     \"device_windows\":{},\"bit_mismatch_windows\":{},\
                     \"count_mismatch_windows\":{},\"f32_flip_rows\":{}}}",
                    jstr(&d.backend),
                    d.segment_rows,
                    d.tile_candidates,
                    d.windows,
                    d.rows,
                    jnum(d.max_abs_net_delta),
                    d.device_windows,
                    d.bit_mismatch_windows,
                    d.count_mismatch_windows,
                    d.f32_flip_rows
                )
            }),
            self.batch_pnl
                .as_ref()
                .map_or("null".to_string(), |b| b.json()),
            windows.join(","),
            self.pooled_json()
        )
    }

    /// #1094: the fields `--threshold-fit` adds to the `threshold` object, with
    /// a leading comma; empty without the flag.
    fn threshold_fit_json(&self) -> String {
        match &self.threshold_fit {
            Some(f) => format!(
                ",\"fit\":\"fitness\",\"fit_method\":\"per window, the cut of lowest block fitness on the validation rows \
                 (training rows without a validation slice) among cuts with at least cut_min_entries distinct entry \
                 timestamps, one sort and one sweep in f64; the fixed threshold when no cut passes\",\
                 \"fit_sources\":[\"train_fitness\",\"validation_fitness\",\"fixed\"],\
                 \"cut_min_entries\":{},\"cut_blocks\":{},\"cut_lambda\":{}",
                f.min_entry_times,
                f.blocks,
                jnum(f.lambda)
            ),
            None => String::new(),
        }
    }

    fn vote_json(&self) -> String {
        match self.quorum {
            Some(q) if self.policy == Policy::Vote => format!(
                ",\"vote\":{{\"quorum\":{},\"votes_required\":{},\"members\":[{}]}}",
                jnum(q),
                self.votes_required,
                self.candidates.iter().map(|m| jstr(m)).collect::<Vec<_>>().join(",")
            ),
            _ => String::new(),
        }
    }

    /// #926: the pooled metrics with `period_net` and `period_time`.
    fn pooled_json(&self) -> String {
        let series = |v: &[f64]| v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(",");
        let mut s = self.pooled.json();
        s.pop();
        let grid = self.period_grid.as_ref().map_or(String::new(), |g| {
            format!(
                ",\"grid\":{{\"rule\":{},\"periods\":{},\"empty_periods\":{}}},\"period_rows\":[{}]",
                jstr(g.rule.as_str()),
                g.period_rows.len(),
                g.empty_periods,
                g.period_rows.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",")
            )
        });
        s.push_str(&format!(
            ",\"period_net\":[{}],\"period_time\":[{}]{}}}",
            series(&self.period_net),
            series(&self.period_time),
            grid
        ));
        if let Some(x) = &self.period_extra {
            s.pop();
            s.push_str(&format!(
                ",\"period_gross\":[{}],\"period_exposure\":[{}],\"period_entries\":[{}],\"period_bh\":[{}]}}",
                series(&x.gross),
                series(&x.exposure),
                series(&x.entries),
                series(&x.bh)
            ));
            if self.emit_series {
                s.pop();
                s.push_str(&format!(
                    ",\"period_cost\":[{}],\"period_entry_stamp\":[{}],\"period_bh_net\":[{}],\"entry_times\":{}}}",
                    series(&x.cost),
                    series(&x.entry_stamp),
                    series(&x.bh_net),
                    x.entry_times
                ));
            }
        }
        s
    }

    fn windows_json(&self) -> String {
        let items: Vec<String> = self
            .windows
            .iter()
            .map(|w| {
                format!(
                    "{{\"label\":{},\"train_start\":{},\"train_end\":{},\
                     \"test_start\":{},\"test_end\":{}}}",
                    jstr(&w.window.label),
                    jnum(w.window.train_start),
                    jnum(w.window.train_end),
                    jnum(w.window.test_start),
                    jnum(w.window.test_end)
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// Markdown report with the same numbers as [`Report::json`].
    pub fn markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Walk-forward evaluation\n\n");
        s.push_str(&format!(
            "Expression: `{}`\n\nTarget: `{}`  \nFeatures: `{}`  \nTime: `{}`  \nGroup: `{}`\n\n",
            self.expression,
            self.target,
            self.variables.join(", "),
            self.time_column,
            self.group_by.as_deref().unwrap_or("(none: one series)")
        ));
        s.push_str("## Rules\n\n");
        let policy_line = match self.policy {
            Policy::Threshold => {
                "- Long/flat only: `position = 1` when `score > threshold`, else `0`; never short, no leverage.\n".to_string()
            }
            Policy::Band => format!(
                "- Long/flat no-trade band (`beta = {}`, minimum hold {}): enter when `score > threshold + beta`, exit when `score <= threshold - beta` after the hold; never short, no leverage.\n",
                md_num(self.band_width),
                self.min_hold
            ),
            Policy::Dp => {
                "- Reported book: the plain threshold rule. The DP value is a **hindsight bound**, not a strategy; never short, no leverage.\n".to_string()
            }
            Policy::TopTercile => {
                "- Top-tercile long-only: per timestamp, `position = 1` for the top `floor(n/3)` finite scores (ties by row order), else `0`; fewer than 3 finite scores hold nothing; never short, no leverage.\n".to_string()
            }
            Policy::Vote => format!(
                "- Long/flat vote book: `position = 1` when at least {} of the {} members have `score > own threshold`, else `0`; vote band {}, minimum hold {}; never short, no leverage.\n",
                self.votes_required,
                self.candidates.len(),
                md_num(self.band_width),
                self.min_hold
            ),
        };
        let threshold_rule = if self.policy == Policy::TopTercile {
            "none: the policy ranks the scores of each timestamp (source `none`)".to_string()
        } else if let (Some(f), Some(theta)) = (self.threshold_fit, self.threshold_fixed) {
            format!(
                "fitted per window on its validation rows (training rows without a validation slice): the cut of \
                 lowest block fitness ({} blocks, lambda {}) among cuts with at least {} distinct entry timestamps, \
                 frozen on the test block; the reference theta = {} where no cut passes (sources `train_fitness`, \
                 `validation_fitness`, `fixed`)",
                f.blocks,
                f.lambda,
                f.min_entry_times,
                theta
            )
        } else if let Some(theta) = self.threshold_fixed {
            format!(
                "pre-registered fixed theta = {} (no data fitting; source `fixed`)",
                theta
            )
        } else {
            format!(
                "q{:.4} of the {} scores, exact type-7 order-statistic selection \
                 (order-preserving radix select, f64 interpolation; source `{}`)",
                self.threshold_quantile,
                if self.validation_frac > 0.0 {
                    "training/validation split"
                } else {
                    "training"
                },
                self.windows
                    .first()
                    .map(|w| w.threshold_source.as_str())
                    .unwrap_or("none")
            )
        };
        s.push_str(&format!(
            "{}- Threshold: {}.\n\
             - Cost: {:.4} bp per side on `|position change|`, so {:.4} bp per round trip.\n\
             - Per-window metrics start every group flat; the pooled path {} and its total\n\
               variation includes the seam term `|p_end - q_start|`.\n\
             - Selection: {}; the test block is never a selection input.\n\
             - Purge: horizon {:.1}s, embargo {:.1}s.\n\
             - Annualisation: {} ({:.6} periods/year).\n\
             - Source: {}.\n\n",
            policy_line,
            threshold_rule,
            self.cost_bps,
            self.cost_bps * 2.0,
            if self.pool_reset_flat {
                "resets every group to flat at a window boundary (historical)"
            } else {
                "carries each group's ending position across the next boundary"
            },
            self.selected_on,
            self.horizon,
            self.embargo,
            self.annualization,
            self.periods_per_year,
            self.source
        ));
        if self.candidates.len() > 1 {
            s.push_str("## Selection (validation only)\n\n");
            s.push_str(
                "| window | candidate | validation MSE | expression |\n\
                 |---|---:|---:|---|\n",
            );
            for w in &self.windows {
                s.push_str(&format!(
                    "| {} | {} | {} | `{}` |\n",
                    w.window.label,
                    w.candidate + 1,
                    md_num(w.validation_error.unwrap_or(f64::NAN)),
                    w.expression
                ));
            }
            s.push('\n');
        }
        s.push_str("## Per window\n\n");
        s.push_str(
            "| window | train | val | test | threshold | Sharpe | CAGR | max DD | turnover | exposure | hit | break-even | trades |\n\
             |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        for w in &self.windows {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                w.window.label,
                w.train_rows,
                w.validation_rows,
                w.test_rows,
                md_num(w.threshold),
                md_num(w.metrics.sharpe),
                md_num(w.metrics.cagr),
                md_num(w.metrics.max_drawdown),
                md_num(w.metrics.turnover),
                md_num(w.metrics.exposure),
                md_num(w.metrics.hit_rate),
                md_num(w.metrics.break_even_hit_rate),
                w.metrics.trades
            ));
        }
        if self.threshold_fit.is_some() {
            s.push_str(
                "\n## Fitted cut (source rows)\n\n\
                 Fitness, net, entry timestamps and exposure are measured on the source rows, where the cut was chosen; \
                 they are not out-of-sample results.\n\n\
                 | window | source | cut | reference | rows | fitness | net | entry times | exposure | plateau | fallback |\n\
                 |---|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n",
            );
            for w in &self.windows {
                if let Some(c) = &w.cut {
                    let p = c.cut.as_ref();
                    s.push_str(&format!(
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                        w.window.label,
                        c.source.as_str(),
                        md_num(c.value),
                        md_num(c.reference_value),
                        c.rows,
                        md_num(p.map_or(f64::NAN, |q| q.fitness)),
                        md_num(p.map_or(f64::NAN, |q| q.net)),
                        p.map_or("n/a".to_string(), |q| q.entry_times.to_string()),
                        md_num(p.map_or(f64::NAN, |q| q.exposure)),
                        c.plateau.as_ref().map_or("n/a".to_string(), |q| format!(
                            "{} to {} ({} rows)",
                            md_num(q.lo),
                            md_num(q.hi),
                            q.rows_hi - q.rows_lo
                        )),
                        c.fallback.unwrap_or("none")
                    ));
                }
            }
        }
        s.push_str("\n## Controls (long-only, per window)\n\n");
        s.push_str(
            "Each control is priced on the window's valid test rows with the same cost, starts flat and pays its final exit. Beta is on the buy-and-hold period series. Cash Sharpe is n/a (zero variance).\n\n\
             | window | control | net | cost | exposure | turnover | beta | excess over cash | Sharpe | entries |\n\
             |---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        for w in &self.windows {
            if let Some(c) = &w.controls {
                for r in &c.rows {
                    s.push_str(&format!(
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                        w.window.label,
                        r.kind.as_str(),
                        md_num(r.totals.net),
                        md_num(r.totals.cost),
                        md_num(r.exposure),
                        md_num(r.turnover),
                        md_num(r.beta),
                        md_num(r.excess_over_cash),
                        md_num(r.sharpe),
                        r.totals.entries
                    ));
                }
            }
        }
        if let Some(c) = self.windows.first().and_then(|w| w.controls.as_ref()) {
            s.push_str(&format!(
                "\nMatched-exposure seed {}; the constant control is long iff the training mean target is positive.\n",
                c.seed
            ));
        }
        s.push_str("\n## Pooled\n\n");
        s.push_str(
            "| Sharpe | CAGR | max DD | turnover | exposure | hit | break-even | trades | entries | rows | periods |\n\
             |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        let p = &self.pooled;
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            md_num(p.sharpe),
            md_num(p.cagr),
            md_num(p.max_drawdown),
            md_num(p.turnover),
            md_num(p.exposure),
            md_num(p.hit_rate),
            md_num(p.break_even_hit_rate),
            p.trades,
            p.entries,
            p.rows,
            p.periods
        ));
        s.push_str(&format!(
            "\nMean net return per period: {} ({} bp); mean cost per long row: {} bp; \
             gross Sharpe: {}.\n",
            md_num(p.mean_return),
            md_bps(p.mean_return * 10_000.0),
            md_bps(p.avg_cost_bps),
            md_num(p.gross_sharpe)
        ));
        s.push_str(&format!(
            "Sortino (MAR {}): {}; log-growth per period: {}.\n",
            md_num(p.sortino_mar),
            md_num(p.sortino),
            md_num(p.log_growth)
        ));
        s.push_str(&format!(
            "\nPooled seam: {}; extra seam cost {} ({} trade(s)); pooled total variation {}.\n",
            if self.pool_reset_flat {
                "flat reset at every boundary"
            } else {
                "boundary position carried"
            },
            md_bps(self.seam_cost * 10_000.0),
            self.seam_trades,
            md_num(p.total_abs_delta)
        ));
        if let Some(g) = &self.period_grid {
            s.push_str(&format!(
                "\nPeriod grid: {} (every period with a finite target before the gate); {} of {} periods have no priced row and book net 0.\n",
                g.rule.as_str(),
                g.empty_periods,
                g.period_rows.len()
            ));
        }
        if self.sign_test.applicable {
            s.push_str(&format!(
                "\nFold sign test: {}/{} positive, one-sided p = {}. {}\n",
                self.sign_test.positive,
                self.sign_test.total,
                md_num(self.sign_test.p_value.unwrap_or(f64::NAN)),
                self.sign_test.method
            ));
        } else {
            s.push_str(&format!(
                "\nFold sign test: not applicable ({}/{} positive). {}\n",
                self.sign_test.positive, self.sign_test.total, self.sign_test.method
            ));
        }
        s.push_str(&format!("\nMemory: {}\n", self.memory_bound));
        s.push('\n');
        s.push_str(&self.economics.markdown());
        if let Some(i) = &self.inference {
            s.push('\n');
            s.push_str(&i.markdown());
        }
        if let Some(r) = &self.regimes {
            s.push('\n');
            s.push_str(&r.markdown());
        }
        s
    }

    /// Compact terminal summary.
    pub fn text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "walkforward: {} = {}  (long/flat, {:.4} bp/side, q{:.2}, ppy {:.4})\n",
            self.target, self.expression, self.cost_bps, self.threshold_quantile, self.periods_per_year
        ));
        s.push_str("window                 train    val    test  threshold    Sharpe      CAGR      maxDD   turnover   exposure      hit  break-even  trades\n");
        for w in &self.windows {
            s.push_str(&format!(
                "{:<20} {:>6} {:>6} {:>7} {:>10.6} {:>9.3} {:>9.2}% {:>9.2}% {:>9.3} {:>9.3}% {:>9.3} {:>10.3} {:>7}\n",
                w.window.label,
                w.train_rows,
                w.validation_rows,
                w.test_rows,
                w.threshold,
                w.metrics.sharpe,
                w.metrics.cagr * 100.0,
                w.metrics.max_drawdown * 100.0,
                w.metrics.turnover,
                w.metrics.exposure * 100.0,
                w.metrics.hit_rate,
                w.metrics.break_even_hit_rate,
                w.metrics.trades
            ));
        }
        for w in &self.windows {
            if let Some(c) = &w.controls {
                let cell = |k: crate::baselines::ControlKind| {
                    let r = c.get(k);
                    format!("{}={:.6}", k.as_str(), r.totals.net)
                };
                s.push_str(&format!(
                    "controls {:<20} net: {}  {}  {}  {}  (seed {})\n",
                    w.window.label,
                    cell(crate::baselines::ControlKind::Cash),
                    cell(crate::baselines::ControlKind::BuyAndHold),
                    cell(crate::baselines::ControlKind::Constant),
                    cell(crate::baselines::ControlKind::MatchedRandomLong),
                    c.seed
                ));
            }
        }
        let p = &self.pooled;
        s.push_str(&format!(
            "pooled: Sharpe {:.4}  CAGR {:.2}%  maxDD {:.2}%  turnover {:.3}/yr  exposure {:.3}%  \
             hit {:.3}  break-even {:.3}  trades {}  rows {}\n",
            p.sharpe,
            p.cagr * 100.0,
            p.max_drawdown * 100.0,
            p.turnover,
            p.exposure * 100.0,
            p.hit_rate,
            p.break_even_hit_rate,
            p.trades,
            p.rows
        ));
        s.push_str(&format!(
            "risk: Sortino(MAR {:.4}) {:.4}  log-growth/period {:.6}\n",
            p.sortino_mar, p.sortino, p.log_growth
        ));
        s.push_str(&format!(
            "seam: {}  seam cost {:.4} bp  seam trades {}  pooled |delta| {:.6}",
            if self.pool_reset_flat { "reset" } else { "carried" },
            self.seam_cost * 10_000.0,
            self.seam_trades,
            p.total_abs_delta
        ));
        if self.sign_test.applicable {
            s.push_str(&format!(
                "  sign {}/{} p={:.6}",
                self.sign_test.positive,
                self.sign_test.total,
                self.sign_test.p_value.unwrap_or(f64::NAN)
            ));
        } else {
            s.push_str(&format!(
                "  sign n/a ({}/{})",
                self.sign_test.positive, self.sign_test.total
            ));
        }
        s.push('\n');
        s
    }
}

struct Selected {
    candidate: usize,
    validation_error: Option<f64>,
    threshold: f64,
    band_selection: Option<crate::selection::BandSelection>,
    thresholds: Vec<f64>,
    /// #1094: the fitted cut of every candidate under `--threshold-fit`.
    cuts: Vec<Option<CutReport>>,
}

/// #1094: long-row shares of the cut profile, ascending: `0.5 * 0.7^j` and
/// `1 - 0.5 * 0.7^j` for `j = 0..15`.
fn profile_shares() -> Vec<f64> {
    let mut tail = Vec::with_capacity(16);
    let mut p = 0.5f64;
    for _ in 0..16 {
        tail.push(p);
        p *= 0.7;
    }
    let mut shares: Vec<f64> = tail.iter().rev().copied().collect();
    shares.extend(tail.iter().map(|p| 1.0 - p));
    shares
}

/// #1094: the cut of one candidate in one window from its source rows (the
/// validation slice, else the training rows), with `reference` as the
/// fallback. The test rows are not an input.
fn fit_cut(rows: &[RowBase], scores: &[f64], reference: f64, fit: &CutFit, cost_bps: f64, source: ThresholdSource) -> CutReport {
    let mut idx: Vec<usize> = (0..rows.len()).filter(|&i| scores[i].is_finite() && rows[i].target.is_finite()).collect();
    idx.sort_by(|&a, &b| {
        rows[a].group.cmp(&rows[b].group).then_with(|| rows[a].time.total_cmp(&rows[b].time)).then_with(|| a.cmp(&b))
    });
    let mut report = CutReport {
        source: ThresholdSource::Fixed,
        value: reference,
        rows: idx.len(),
        cut: None,
        reference_value: reference,
        reference: None,
        plateau: None,
        fit: *fit,
        fallback: None,
        fallback_detail: None,
        profile: Vec::new(),
    };
    if idx.len() < 2 {
        report.fallback = Some("no_rows");
        return report;
    }
    let pred: Vec<f64> = idx.iter().map(|&i| scores[i]).collect();
    let time: Vec<f64> = idx.iter().map(|&i| rows[i].time).collect();
    let slice = data::TradingSlice {
        rows: crate::trading::TradingRows {
            time: time.clone(),
            group: idx.iter().map(|&i| rows[i].group).collect(),
            funding: Vec::new(),
        },
        ret: idx.iter().map(|&i| rows[i].target).collect(),
    };
    let book_spec = crate::trading::TradingSpec { cost_bps, min_trades: 1, threshold: reference, ..crate::trading::TradingSpec::default() };
    let sweep = match crate::trading::FitnessBook::with_blocks(&slice, &book_spec, 0.0, fit.blocks, fit.lambda)
        .and_then(|book| book.cut_sweep(&pred, &time))
    {
        Ok(s) => s,
        Err(e) => {
            report.fallback = Some("book_error");
            report.fallback_detail = Some(e);
            return report;
        }
    };
    let n = sweep.rows();
    let point = |t: usize| CutPoint {
        index: t,
        value: sweep.value(&pred, t),
        fitness: sweep.fitness(t),
        net: sweep.net(t),
        flips: sweep.flips(t),
        exits: sweep.exits(t),
        entry_times: sweep.entry_times(t),
        exposure: (n - t) as f64 / n as f64,
    };
    let reference_index = sweep.index_of(&pred, reference);
    let reference_point = CutPoint { value: reference, ..point(reference_index) };
    report.profile = profile_shares()
        .into_iter()
        .map(|p| (p, point(sweep.snap(n - (p * n as f64).round() as usize))))
        .collect();
    match sweep.best(&crate::trading::CutFloor { min_entry_times: fit.min_entry_times }, reference_index) {
        Some(t) => {
            let chosen = point(t);
            let (lo, hi) = sweep.plateau(t);
            report.plateau = Some(CutPlateau { lo: sweep.value(&pred, lo), hi: sweep.value(&pred, hi), rows_lo: lo, rows_hi: hi });
            report.source = source;
            report.value = chosen.value;
            report.cut = Some(chosen);
        }
        None => {
            let inner = (1..n).any(|t| sweep.is_valid(t) && sweep.fitness(t).is_finite());
            report.fallback = Some(if inner { "floor" } else { "no_valid_cut" });
            report.cut = Some(reference_point.clone());
        }
    }
    report.reference = Some(reference_point);
    report
}

/// The candidate of one window. Reads the training and validation rows only,
/// never `test_rows` or `test_scores`, so the choice cannot depend on a test
/// value and can be made before the test block is scored.
fn select_candidate(
    d: &WindowData,
    spec: &EvalSpec,
    trees: &[Node],
    has_validation: bool,
    fit: Option<&CutFit>,
) -> Result<Selected, String> {
            let mut best: Option<(usize, f64, f64)> = None;
            let mut thresholds = vec![f64::NAN; d.train_scores.len()];
            let mut cuts: Vec<Option<CutReport>> = vec![None; d.train_scores.len()];
            for (c, scores) in d.train_scores.iter().enumerate() {
                let source_scores = if has_validation {
                    &d.validation_scores[c]
                } else {
                    scores
                };
                if source_scores.is_empty() && spec.threshold_fixed.is_none() && spec.policy != Policy::TopTercile {
                    return Err(format!(
                        "window '{}': no finite {} scores for candidate {} to set the threshold",
                        d.window.label,
                        if has_validation { "validation" } else { "training" },
                        c + 1
                    ));
                }
                let err = if has_validation {
                    validation_mse(&d.validation_rows, &d.validation_scores[c])
                } else {
                    0.0
                };
                let threshold = if spec.policy == Policy::TopTercile {
                    f64::NAN
                } else if let (Some(fit), Some(theta)) = (fit, spec.threshold_fixed) {
                    let report = if has_validation {
                        fit_cut(&d.validation_rows, &d.validation_scores[c], theta, fit, spec.cost_bps, ThresholdSource::ValidationFitness)
                    } else {
                        fit_cut(&d.train_rows, &d.train_row_scores[c], theta, fit, spec.cost_bps, ThresholdSource::TrainFitness)
                    };
                    let value = report.value;
                    cuts[c] = Some(report);
                    value
                } else if let Some(theta) = spec.threshold_fixed {
                    Threshold::fixed(theta)?.value()
                } else if has_validation {
                    Threshold::validation_type7(source_scores, spec.threshold_quantile).value()
                } else {
                    Threshold::train_type7(source_scores, spec.threshold_quantile).value()
                };
                thresholds[c] = threshold;
                match best {
                    Some((_, best_err, _)) if best_err <= err => {}
                    _ => best = Some((c, err, threshold)),
                }
            }
            let mut band_selection = None;
            let (candidate, validation_error, threshold) = if spec.band_select {
                if !has_validation {
                    return Err(format!(
                        "window '{}': --select-band needs a validation slice (--validation-frac > 0 or a windows \
                         table with validation_start), because the test block may not choose the candidate",
                        d.window.label
                    ));
                }
                let targets: Vec<f64> = d.validation_rows.iter().map(|r| r.target).collect();
                let groups: Vec<u64> = d.validation_rows.iter().map(|r| r.group).collect();
                let times: Vec<f64> = d.validation_rows.iter().map(|r| r.time).collect();
                let rows = crate::selection::BandRows {
                    targets: &targets,
                    groups: &groups,
                    times: &times,
                };
                let options = crate::selection::BandOptions {
                    cost_bps: spec.cost_bps,
                    ..crate::selection::BandOptions::default()
                };
                let mut candidates: Vec<crate::selection::BandCandidate<'_>> =
                    Vec::with_capacity(d.validation_scores.len());
                for c in 0..d.validation_scores.len() {
                    let surrogate = crate::selection::screening_net(
                        &rows,
                        &d.validation_scores[c],
                        thresholds[c],
                        &options,
                    )?;
                    candidates.push(crate::selection::BandCandidate {
                        index: c,
                        surrogate,
                        complexity: expr::complexity(&trees[c]),
                        scores: &d.validation_scores[c],
                        threshold: thresholds[c],
                    });
                }
                let selection = crate::selection::select_band(&rows, &candidates, &options)
                    .map_err(|e| format!("window '{}': {}", d.window.label, e))?;
                let chosen = selection.selected_index;
                let err = validation_mse(&d.validation_rows, &d.validation_scores[chosen]);
                band_selection = Some(selection);
                (chosen, Some(err), thresholds[chosen])
            } else {
                let (c, err, theta) = best.expect("at least one candidate");
                let err = if has_validation { Some(err) } else { None };
                (c, err, theta)
            };
    Ok(Selected { candidate, validation_error, threshold, band_selection, thresholds, cuts })
}

#[cfg(test)]
mod locked_reach_tests {
    use super::*;

    fn spec(horizon: f64) -> EvalSpec {
        EvalSpec {
            expression: "x".to_string(),
            vars: vec!["x".to_string()],
            target: "target".to_string(),
            time: "ts".to_string(),
            group: None,
            windows: vec![Window { label: "w".to_string(), train_start: 0.0, train_end: 20.0, test_start: 20.0, test_end: 40.0 }],
            horizon,
            embargo: None,
            cost_bps: 10.0,
            threshold_quantile: 0.5,
            threshold_fixed: None,
            validation_frac: 0.0,
            periods_per_year: Some(4.0),
            candidates: Vec::new(),
            band_select: false,
            pool_reset_flat: false,
            allow_lockbox: false,
            device_pnl: false,
            control_seed: 0,
            policy: Policy::Threshold,
            band_width: 0.0,
            min_hold: 1,
            gpu: false,
        }
    }


    fn protocol(entry_offset: f64, embargo: i64, ranges: &[(i64, i64)]) -> ProtocolOptions {
        ProtocolOptions { entry_offset, locked_ranges: ranges.to_vec(), locked_embargo: embargo, ..ProtocolOptions::default() }
    }

    fn wide(mut spec: EvalSpec) -> EvalSpec {
        spec.windows[0] = Window { label: "w".to_string(), train_start: 0.0, train_end: 4e9, test_start: 4e9, test_end: 5e9 };
        spec
    }

    #[test]
    fn the_read_time_drop_spans_the_purge_before_and_the_embargo_after_a_sealed_range() {
        let mut spec = spec(3.0);
        spec.windows[0].test_end = 60.0;
        let opts = ProtocolOptions { locked_ranges: vec![(30, 36)], locked_embargo: 5, ..ProtocolOptions::default() };
        let acc = Accum::new(&spec, &opts, 1).unwrap();
        assert_eq!(opts.locked_reach(&spec), vec![(30.0, 36.0, 28.0, 41.0)]);
        for t in 0..60 {
            assert_eq!(acc.wants(t as f64), t < 28 || t >= 41, "row {t}");
        }
    }

    #[test]
    fn fold_reach_agrees_with_label_report_and_lockbox_check() {
        use crate::labels::{LabelAccumulator, LongLabelSpec};
        use crate::lockbox::{Declared, Embargo, EmbargoSource, Lockbox};
        const HOUR: i64 = 3600;
        let (start, end) = (1_600_200_000, 1_600_900_000);
        let range = Lockbox { start, end };
        for h in [1usize, 4, 24, 72, 168, 336] {
            let mut labels = LongLabelSpec::new("a", HOUR, h);
            labels.sigma_window = 5;
            let purge = labels.purge_seconds();
            let mut spec = wide(spec(0.0));
            spec.horizon = (h as i64 * HOUR) as f64;
            let opts = protocol(HOUR as f64, 0, &[(start, end)]);
            let acc = Accum::new(&spec, &opts, 1).unwrap();
            let report = LabelAccumulator::with_range(&labels, Some(range)).with_embargo(purge);
            let declared = Declared {
                lockbox: range,
                source: "flag",
                also: None,
                embargo: Embargo { seconds: purge, source: EmbargoSource::LabelPurge },
            };
            let edges = (start - purge - 2..=start + 2).chain(end + purge - 2..=end + purge + 2);
            for t in edges {
                let fold = acc.wants(t as f64);
                let kept = report.keeps(t);
                let accepted = declared.check_times("training split", "ts", &[t], purge).is_ok();
                assert_eq!((fold, kept), (accepted, accepted), "h{h} t {t} (S {start}, E {end}, purge {purge})");
                let one_row = |a: i64| Window { label: "w".to_string(), train_start: a as f64, train_end: (a + 1) as f64, test_start: 4e9, test_end: 5e9 };
                let mut one = spec.clone();
                one.windows[0] = one_row(t);
                assert_eq!(validate_windows(&one, &opts).is_ok(), accepted, "h{h} guard t {t}");
            }
        }
    }

    #[test]
    fn locked_reach_edges_follow_the_shared_predicate() {
        // purge 0: the row at S is the first one read.
        let mut zero = wide(spec(0.0));
        zero.horizon = 0.0;
        let opts = protocol(0.0, 0, &[(30, 36)]);
        let acc = Accum::new(&zero, &opts, 1).unwrap();
        assert_eq!(opts.locked_reach(&zero), vec![(30.0, 36.0, 30.0, 36.0)]);
        assert!([29.0, 36.0].iter().all(|&t| acc.wants(t)) && [30.0, 35.0].iter().all(|&t| !acc.wants(t)));

        // embargo above the purge extends the upper side only.
        let long = wide(spec(3.0));
        let opts = protocol(0.0, 10, &[(30, 36)]);
        let acc = Accum::new(&long, &opts, 1).unwrap();
        assert_eq!(opts.locked_reach(&long), vec![(30.0, 36.0, 28.0, 46.0)]);
        assert!(acc.wants(27.0) && !acc.wants(28.0) && !acc.wants(45.0) && acc.wants(46.0));

        // embargo below the purge: the purge sets the upper side.
        let opts = protocol(0.0, 1, &[(30, 36)]);
        assert_eq!(opts.locked_reach(&long), vec![(30.0, 36.0, 28.0, 39.0)]);

        // two ranges, a purge that is not a multiple of a bar.
        let odd = wide(spec(7.0));
        let opts = protocol(0.0, 0, &[(100, 110), (200, 210)]);
        let acc = Accum::new(&odd, &opts, 1).unwrap();
        let drop = |t: i64| (94..117).contains(&t) || (194..217).contains(&t);
        for t in 80..230 {
            assert_eq!(acc.wants(t as f64), !drop(t), "row {t}");
        }

        // a window that starts at the first reaching row is refused, one second later is not.
        let mut guard = wide(spec(3.0));
        let opts = protocol(0.0, 0, &[(30, 36)]);
        guard.windows[0] = Window { label: "w".to_string(), train_start: 0.0, train_end: 28.0, test_start: 40.0, test_end: 50.0 };
        assert!(validate_windows(&guard, &opts).is_ok());
        guard.windows[0].train_end = 29.0;
        assert!(validate_windows(&guard, &opts).unwrap_err().contains("reaches the declared locked test"));
    }

    #[test]
    fn infer_frequency_ignores_unlabelled_rows() {
        let row = |i: usize, labelled: bool| RowBase {
            time: i as f64 * 3600.0,
            group: 0,
            target: if labelled { 0.01 } else { f64::NAN },
        };
        let h1: Vec<RowBase> = (0..120).map(|i| row(i, i % 5 != 2)).collect();
        let (ppy, dt) = infer_frequency(&h1);
        assert_eq!(dt, Some(3600.0));
        assert_eq!(ppy, 8760.0);

        let daily: Vec<RowBase> = (0..24 * 20).map(|i| row(i, i % 24 == 0)).collect();
        let (ppy, dt) = infer_frequency(&daily);
        assert_eq!(dt, Some(86_400.0));
        assert_eq!(ppy, 365.0);
    }

    #[test]
    fn bucket_series_equals_the_timestamp_series_on_a_shared_grid() {
        let s = 3600.0;
        let mut plain = CrossSection::default();
        let mut bucket = CrossSection::with_bucket(s);
        for k in 0..10 {
            for g in 0..3 {
                let r = 0.001 * (k * 3 + g) as f64;
                plain.add(1000.0 * s + k as f64 * s, r, r);
                bucket.add(1000.0 * s + k as f64 * s, r, r);
            }
        }
        assert_eq!(plain.periods, bucket.periods);
        assert_eq!(plain.series(), bucket.series());
    }

    #[test]
    fn staggered_rows_share_a_bucket_and_merge_by_bucket() {
        let day = 86_400.0;
        let mut cs = CrossSection::with_bucket(day);
        for (t, r) in [(0.0, 0.01), (7.0 * 3600.0, 0.02), (15.0 * 3600.0, 0.03), (day, 0.04)] {
            cs.add(t, r, r);
        }
        assert_eq!(cs.periods.len(), 2);
        assert_eq!(cs.periods[0].count, 3);
        assert_eq!(cs.periods[0].time, 0.0);
        let (net, _) = cs.series();
        assert!((net[0] - 0.02).abs() < 1e-15);
        let mut a = CrossSection::with_bucket(day);
        let mut b = CrossSection::with_bucket(day);
        a.add(0.0, 0.01, 0.01);
        b.add(7.0 * 3600.0, 0.03, 0.03);
        b.add(day, 0.04, 0.04);
        a.merge(&b);
        assert_eq!(a.periods.len(), 2);
        assert_eq!(a.periods[0].count, 2);
        assert_eq!(a.periods[0].time, 0.0);
    }
}

#[cfg(test)]
mod close_ledger_tests {
    use super::*;
    use crate::gpu_stats::{RunLedger, Stage};

    fn next(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let unit = (*state >> 11) as f64 / (1u64 << 53) as f64;
        1e-6 * (1e7f64).powf(unit)
    }

    #[test]
    fn the_ledger_closes_exactly_for_any_stage_seconds() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut inexact_sum = 0usize;
        for set in 0..1000 {
            let mut ledger = RunLedger::new();
            let wall = next(&mut state);
            for stage in [Stage::Io, Stage::Pnl, Stage::Fit] {
                ledger.record_stage(stage, wall * (next(&mut state) / 10.0) / 3.0);
            }
            let sum = ledger.stage_sum_s();
            if sum + (wall - sum) != wall {
                inexact_sum += 1;
            }
            let booked: Vec<u64> = [Stage::Io, Stage::Pnl, Stage::Fit].iter().map(|s| ledger.stage_s(*s).to_bits()).collect();
            close_ledger_at(&mut ledger, wall);
            let after: Vec<u64> = [Stage::Io, Stage::Pnl, Stage::Fit].iter().map(|s| ledger.stage_s(*s).to_bits()).collect();
            assert_eq!(booked, after, "set {set}: measured stages are untouched");
            assert_eq!(ledger.closure_residual().value, 0.0, "set {set}");
            assert_eq!(ledger.stage_sum_s(), ledger.wall_s(), "set {set}");
            assert!((ledger.wall_s() - wall).abs() <= wall * 2.3e-16, "set {set}: wall moved more than one ULP");
        }
        assert!(inexact_sum > 0, "the generated sets never hit s + (wall - s) != wall");
    }

    #[test]
    fn an_overbooked_ledger_keeps_its_wall() {
        let mut ledger = RunLedger::new();
        ledger.record_stage(Stage::Pnl, 2.0);
        close_ledger_at(&mut ledger, 1.0);
        assert_eq!(ledger.wall_s(), 1.0);
        assert_eq!(ledger.closure_residual().value, 0.0);
    }
}

#[cfg(test)]
mod group_key_tests {
    use super::*;

    #[test]
    fn signed_zeros_and_nans_share_one_id_each() {
        assert_eq!(group_key(-0.0), group_key(0.0));
        assert_eq!(group_key(f64::NAN), u64::MAX);
        assert_ne!(group_key(1.0), group_key(-1.0));
    }
}
