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

use crate::column::GroupBuf;
use crate::data::{self, Table};
use crate::expr::{self, Node};
use crate::store::{self, FileStore};
use crate::trading_gpu::{self, Threshold, ThresholdSource, ValidMask};
use std::collections::HashMap;

/// Default per-side cost in basis points (10 bp, as in the pilot).
pub const DEFAULT_COST_BPS: f64 = 10.0;
/// Default score quantile used to set the long threshold (q70).
pub const DEFAULT_QUANTILE: f64 = 0.70;
/// Seconds in a (crypto) year, used to turn a time step into periods per year.
pub const SECONDS_PER_YEAR: f64 = 365.0 * 86_400.0;

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
}

impl Policy {
    pub fn parse(src: &str) -> Result<Policy, String> {
        match src {
            "threshold" => Ok(Policy::Threshold),
            "band" => Ok(Policy::Band),
            "dp" | "hindsight" => Ok(Policy::Dp),
            other => Err(format!(
                "unknown policy '{}'; expected threshold | band | dp",
                other
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Threshold => "threshold",
            Policy::Band => "band",
            Policy::Dp => "dp",
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
    /// `train` or `validation`.
    pub threshold_source: String,
    pub train_rows: usize,
    pub validation_rows: usize,
    pub test_rows: usize,
    pub dropped_rows: usize,
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
}

/// #306: the band/cooldown book and its associative carry.
#[derive(Clone, Debug)]
pub struct BandCheck {
    pub band_width: f64,
    pub min_hold: usize,
    /// `carry2` for the 2-state band, `cooldown-K` for the K+1-state variant.
    pub carry: String,
    pub groups: usize,
    /// Additive net from the composed carry (including every final exit).
    pub net: f64,
    /// Scalar evaluator's sum of per-row net returns.
    pub scalar_net: f64,
    /// `cuda` when the device scan produced the carries, `cpu` otherwise.
    pub backend: String,
    /// Why `--gpu` fell back to the CPU scan; `None` without a fallback.
    pub fallback_reason: Option<String>,
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
    pub cost_bps: f64,
    pub threshold_quantile: f64,
    /// The pre-registered fixed threshold, when one was supplied.
    pub threshold_fixed: Option<f64>,
    /// #306: position policy and its band parameters.
    pub policy: Policy,
    pub band_width: f64,
    pub min_hold: usize,
    /// #306: `--gpu` was set for the band/DP scans.
    pub gpu: bool,
    pub validation_frac: f64,
    pub horizon: f64,
    pub embargo: f64,
    pub periods_per_year: f64,
    /// How the pooled annualisation was chosen.
    pub annualization: String,
    /// Candidate expressions in report order.
    pub candidates: Vec<String>,
    /// Which fibre selected the candidate(s). Always validation when more than
    /// one candidate is present; `test` is never a selection input.
    pub selected_on: String,
    /// #308: whether the band protocol selected the candidate.
    pub band_select: bool,
    /// True when no test row influenced candidate or threshold selection.
    pub test_used_for_selection: bool,
    /// Pooled seam policy and the extra cost the carried boundary charged.
    pub pool_reset_flat: bool,
    pub seam_cost: f64,
    pub seam_trades: usize,
    /// Fold sign test: computed only when folds are independent (no training
    /// interval overlaps another fold's test block). Expanding folds overlap by
    /// construction, so their sign test is reported as not applicable.
    pub sign_test: SignTest,
    /// `store-stream` or `csv-table`.
    pub source: String,
    pub memory_bound: String,
    /// #145: was the lockbox override explicitly set for this run?
    pub allow_lockbox: bool,
    /// #304: fused long-only reduction summary when `--device-pnl` was set.
    pub device_pnl: Option<DevicePnlSummary>,
    dropped_rows: usize,
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
pub fn run_path(path: &str, spec: &EvalSpec) -> Result<Report, String> {
    if store::is_store_path(path) {
        run_store(path, spec)
    } else {
        let table = data::read_table(path)?;
        run_table(&table, spec, "csv-table")
    }
}

/// Evaluate a resident table. The CLI's CSV path and tests use this.
pub fn run_table(table: &Table, spec: &EvalSpec, source: &str) -> Result<Report, String> {
    let layout = resolve_layout(spec, &table.names, &table.numeric)?;
    let trees = parse_expressions(spec, &layout)?;
    validate_windows(spec)?;

    let times = data::time_keys(table, layout.time);
    let group_ids = layout.group.map(|g| table_group_ids(table, g));
    let mut acc = Accum::new(spec, trees.len())?;
    let mut features = Vec::with_capacity(layout.vars.len());
    for r in 0..table.len() {
        let t = times.value(r);
        if !acc.wants(t) {
            continue;
        }
        features.clear();
        for &c in &layout.vars {
            features.push(table.value(r, c));
        }
        let g = group_ids.as_ref().map_or(0, |ids| ids[r]);
        acc.add(t, g, &features, table.value(r, layout.target), &trees);
    }
    acc.finish(
        spec,
        &layout,
        source,
        "O(test rows + one score per training row per window); CSV is resident",
        &trees,
    )
}

/// Evaluate a `.rcsb` store directly, scanning it one row group at a time.
pub fn run_store(path: &str, spec: &EvalSpec) -> Result<Report, String> {
    let fs = FileStore::open(path)?;
    let names = fs.header().names();
    let numeric = fs.header().numeric();
    let mut layout = resolve_layout(spec, &names, &numeric)?;
    layout.time_i64 = fs.header().dtypes()[layout.time] == store::DType::I64;
    let trees = parse_expressions(spec, &layout)?;
    validate_windows(spec)?;

    let mut acc = Accum::new(spec, trees.len())?;
    let header = fs.header();
    let mut buf = GroupBuf::new();
    let mut scratch = Vec::new();
    let mut features = Vec::with_capacity(layout.vars.len());
    for group in 0..fs.n_groups() {
        let view = fs.load_group_with(group, &mut buf, &mut scratch)?;
        let len = view.len();
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
        let group_codes = match layout.group {
            Some(c) if !numeric[c] => Some(fs.read_codes(group, c)?),
            _ => None,
        };
        let time_levels = header.levels(layout.time);
        for r in 0..len {
            let t = match (&time_i64, &time_codes) {
                (Some(ints), _) => ints.get(r).copied().unwrap_or(i64::MIN) as f64,
                (None, None) => view.col(layout.time)[r],
                (None, Some(codes)) => {
                    let code = codes.get(r).copied().unwrap_or(0) as usize;
                    time_levels
                        .and_then(|l| l.get(code))
                        .and_then(|s| data::parse_time(s))
                        .unwrap_or(f64::NAN)
                }
            };
            if !acc.wants(t) {
                continue;
            }
            features.clear();
            for &c in &layout.vars {
                features.push(view.col(c)[r]);
            }
            let g = match (layout.group, &group_codes) {
                (None, _) => 0,
                (Some(_), Some(codes)) => {
                    0x8000_0000_0000_0000 | codes.get(r).copied().unwrap_or(0) as u64
                }
                (Some(c), None) => group_key(view.col(c)[r]),
            };
            let y = view.col(layout.target)[r];
            acc.add(t, g, &features, y, &trees);
        }
    }
    let bound = format!(
        "O(test rows + one score per training row per window); the .rcsb store is scanned \
         group by group, one {} -row group resident",
        fs.group_rows()
    );
    acc.finish(spec, &layout, "store-stream", &bound, &trees)
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

fn validate_windows(spec: &EvalSpec) -> Result<(), String> {
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
    if spec.candidates.len() > 1 && spec.validation_frac <= 0.0 {
        return Err(
            "multiple --expr candidates need --validation-frac > 0: select on a validation \
             slice, never on the test block"
                .to_string(),
        );
    }
    if spec.candidates.iter().any(|e| e.trim().is_empty()) {
        return Err("candidate expressions must be non-empty".to_string());
    }
    let embargo = spec.embargo_seconds();
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
        let guard = crate::lockbox::Guard::new(spec.allow_lockbox, None);
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
        let gap = w.test_start - w.train_end;
        if gap + 1e-9 < embargo {
            return Err(format!(
                "window '{}': purge gap is {} seconds but horizon/embargo needs {}; \
                 move test_start later or lower --horizon/--embargo",
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
        v.to_bits()
    }
}

fn table_group_ids(table: &Table, col: usize) -> Vec<u64> {
    if let Some(dict) = table.dict(col) {
        let mut ids: HashMap<&str, u64> = HashMap::new();
        let mut next = 0u64;
        dict.codes
            .iter()
            .map(|&code| {
                let text = dict.levels.get(code as usize).map(|s| s.as_str()).unwrap_or("");
                *ids.entry(text).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect()
    } else {
        table.col(col).iter().map(|&v| group_key(v)).collect()
    }
}

// ---------------------------------------------------------------------------
// Accumulation
// ---------------------------------------------------------------------------

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
    validation_rows: Vec<RowBase>,
    validation_scores: Vec<Vec<f64>>,
    test_rows: Vec<RowBase>,
    test_scores: Vec<Vec<f64>>,
    /// Sum and count of the finite training-interval targets (#310 constant
    /// control).
    train_target_sum: f64,
    train_target_n: u64,
    dropped: usize,
}

struct Accum {
    windows: Vec<WindowData>,
    candidates: usize,
    scores: Vec<f64>,
}

impl Accum {
    fn new(spec: &EvalSpec, candidates: usize) -> Result<Self, String> {
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
            let validation_start = if spec.validation_frac > 0.0 {
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
                validation_rows: Vec::new(),
                validation_scores: vec![Vec::new(); candidates],
                test_rows: Vec::new(),
                test_scores: vec![Vec::new(); candidates],
                train_target_sum: 0.0,
                train_target_n: 0,
                dropped: 0,
            });
        }
        Ok(Accum {
            windows,
            candidates,
            scores: vec![0.0; candidates],
        })
    }

    fn wants(&self, t: f64) -> bool {
        if !t.is_finite() {
            return false;
        }
        self.windows.iter().any(|d| {
            (t >= d.window.train_start && t < d.window.train_end)
                || (t >= d.window.test_start && t < d.window.test_end)
        })
    }

    fn add(&mut self, t: f64, group: u64, features: &[f64], target: f64, trees: &[Node]) {
        debug_assert_eq!(trees.len(), self.candidates);
        for (s, tree) in self.scores.iter_mut().zip(trees) {
            *s = expr::eval(tree, features);
        }
        push_scores(&mut self.windows, t, group, target, &self.scores);
    }


    fn finish(
        self,
        spec: &EvalSpec,
        layout: &Layout,
        source: &str,
        memory_bound: &str,
        trees: &[Node],
    ) -> Result<Report, String> {
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
        let expressions = candidate_expressions(spec);
        for d in self.windows {
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
            // Score every candidate on the validation fibre, then freeze the
            // choice before touching the test block.
            let mut best: Option<(usize, f64, f64)> = None;
            let mut thresholds = vec![f64::NAN; d.train_scores.len()];
            for (c, scores) in d.train_scores.iter().enumerate() {
                let source_scores = if has_validation {
                    &d.validation_scores[c]
                } else {
                    scores
                };
                if source_scores.is_empty() && spec.threshold_fixed.is_none() {
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
                let threshold = if let Some(theta) = spec.threshold_fixed {
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
                        "window '{}': --select-band needs --validation-frac > 0, \
                         because the test block may not choose the candidate",
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
            let source_name = if spec.threshold_fixed.is_some() {
                ThresholdSource::Fixed
            } else if has_validation {
                ThresholdSource::ValidationType7
            } else {
                ThresholdSource::TrainType7
            };
            let (ppy, inferred_dt) = match spec.periods_per_year {
                Some(v) => (v, None),
                None => infer_frequency(&d.test_rows),
            };
            let mut fresh_previous: HashMap<u64, f64> = HashMap::new();
            let ev = evaluate_window(
                &d.test_rows,
                &d.test_scores[candidate],
                threshold,
                cost_side,
                ppy,
                &mut fresh_previous,
                spec.policy,
                spec.band_width,
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
            );
            // Pooled path: carry each group's position across the seam unless
            // the caller asked for the historical flat reset.
            let pev = if spec.pool_reset_flat {
                evaluate_window(
                    &d.test_rows,
                    &d.test_scores[candidate],
                    threshold,
                    cost_side,
                    ppy,
                    &mut HashMap::new(),
                    spec.policy,
                    spec.band_width,
                    spec.min_hold,
                    spec.gpu,
                    false,
                    None,
                )
            } else {
                evaluate_window(
                    &d.test_rows,
                    &d.test_scores[candidate],
                    threshold,
                    cost_side,
                    ppy,
                    &mut pooled_previous,
                    spec.policy,
                    spec.band_width,
                    spec.min_hold,
                    spec.gpu,
                    false,
                    None,
                )
            };
            if !spec.pool_reset_flat {
                if let Some(prev_end) = &previous_end {
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
            let train_rows = d.train_scores[candidate].len()
                + d.validation_scores[candidate].len();
            dropped += d.dropped;
            pooled_returns.extend_from_slice(&pev.returns);
            pooled_gross.extend_from_slice(&pev.gross_returns);
            pooled_stats.add_assign(&pev.stats);
            pooled_ppy.push(ppy);
            windows.push(WindowResult {
                window: d.window,
                expression: expr::to_string(&trees[candidate], &layout.var_names),
                candidate,
                validation_error,
                threshold,
                threshold_source: source_name.as_str().to_string(),
                train_rows,
                validation_rows: d.validation_scores[candidate].len(),
                test_rows: ev.stats.rows,
                dropped_rows: d.dropped,
                inferred_period_seconds: inferred_dt,
                metrics: ev.metrics,
                device_pnl: ev.device,
                controls: ev.controls,
                band: ev.band,
                hindsight_bound: ev.hindsight_bound,
                band_selection,
            });
        }

        let pooled_ppy = match spec.periods_per_year {
            Some(v) => v,
            None => median(&mut pooled_ppy).unwrap_or(365.0),
        };
        let pooled = summarize(&pooled_returns, &pooled_gross, &pooled_stats, pooled_ppy);
        let annualization = if spec.periods_per_year.is_some() {
            "explicit --periods-per-year".to_string()
        } else {
            "inferred from the median test time step (365-day year)".to_string()
        };
        let sign_test = fold_sign_test(&windows);
        let device_pnl = if spec.device_pnl {
            let mut backend = String::new();
            let mut rows = 0u64;
            let mut count = 0usize;
            let mut max_abs_net_delta = 0.0f64;
            for w in &windows {
                if let Some(d) = &w.device_pnl {
                    if backend.is_empty() {
                        backend = d.backend.clone();
                    }
                    rows += d.strategy.rows;
                    count += 1;
                    max_abs_net_delta = max_abs_net_delta.max(d.net_delta.abs());
                }
            }
            Some(DevicePnlSummary {
                backend,
                segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
                tile_candidates: crate::trading_gpu::MAX_TILE_CANDIDATES,
                windows: count,
                rows,
                max_abs_net_delta,
            })
        } else {
            None
        };
        let selected_on = if expressions.len() > 1 {
            "validation".to_string()
        } else if spec.validation_frac > 0.0 {
            "validation (threshold only)".to_string()
        } else {
            "train (threshold only)".to_string()
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
            cost_bps: spec.cost_bps,
            threshold_quantile: spec.threshold_quantile,
            threshold_fixed: spec.threshold_fixed,
            policy: spec.policy,
            band_width: spec.band_width,
            min_hold: spec.min_hold,
            gpu: spec.gpu,
            validation_frac: spec.validation_frac,
            horizon: spec.horizon,
            embargo: spec.embargo_seconds(),
            periods_per_year: pooled_ppy,
            annualization,
            candidates: expressions,
            selected_on,
            band_select: spec.band_select,
            test_used_for_selection: false,
            pool_reset_flat: spec.pool_reset_flat,
            seam_cost,
            seam_trades,
            sign_test,
            source: source.to_string(),
            memory_bound: memory_bound.to_string(),
            allow_lockbox: spec.allow_lockbox,
            device_pnl,
            dropped_rows: dropped,
        })
    }
}

fn push_scores(
    windows: &mut [WindowData],
    t: f64,
    group: u64,
    target: f64,
    scores: &[f64],
) {
    for d in windows {
        let in_train = t >= d.window.train_start && t < d.window.train_end;
        let in_test = t >= d.window.test_start && t < d.window.test_end;
        if in_train {
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
                for (dst, &score) in d.train_scores.iter_mut().zip(scores) {
                    if score.is_finite() {
                        dst.push(score);
                    } else {
                        d.dropped += 1;
                    }
                }
            }
        }
        if in_test {
            d.test_rows.push(RowBase {
                time: t,
                group,
                target,
            });
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
struct Stats {
    rows: usize,
    active: usize,
    hits: usize,
    wins: usize,
    losses: usize,
    win_sum: f64,
    loss_sum: f64,
    total_cost: f64,
    total_abs_delta: f64,
    trades: usize,
    entries: usize,
    exits: usize,
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
    }
}

struct WindowEval {
    metrics: Metrics,
    returns: Vec<f64>,
    gross_returns: Vec<f64>,
    stats: Stats,
    first_positions: HashMap<u64, f64>,
    end_positions: HashMap<u64, f64>,
    device: Option<DevicePnlCheck>,
    band: Option<BandCheck>,
    hindsight_bound: Option<HindsightBound>,
    controls: Option<crate::baselines::ControlSet>,
}

/// #310: what the control ladder needs beyond the test rows.
struct ControlInput {
    seed: u64,
    /// Mean finite training target; `NaN` when there is none.
    train_mean: f64,
    check_device: bool,
}

/// One timestamp's `(sum, count)` cross-section.
///
/// The equal-weight return is `sum / count` over the groups present at the
/// timestamp. Carrying the pair (rather than a pre-divided mean) is what makes
/// a symbol-parallel decomposition merge to exactly the sequential result:
/// denominators differ when a symbol is missing at a timestamp.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrossSection {
    pub periods: Vec<PeriodSum>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodSum {
    pub time: f64,
    pub net_sum: f64,
    pub gross_sum: f64,
    pub count: usize,
}

impl CrossSection {
    /// Add one row's net and gross return. Calls must arrive in nondecreasing
    /// time order; equal timestamps accumulate into one period.
    pub fn add(&mut self, time: f64, net: f64, gross: f64) {
        if let Some(last) = self.periods.last_mut() {
            if last.time == time {
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
                other.periods[j]
                    .time
                    .partial_cmp(&self.periods[i].time)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .is_lt()
            };
            if take_other {
                out.push(other.periods[j]);
                j += 1;
            } else if j < other.periods.len() && other.periods[j].time == self.periods[i].time {
                let a = self.periods[i];
                let b = other.periods[j];
                out.push(PeriodSum {
                    time: a.time,
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

/// `previous` carries each group's last position. Pass a fresh map for a
/// window that starts flat; pass the pooled map to carry the seam.
fn evaluate_window(
    rows: &[RowBase],
    scores: &[f64],
    threshold: f64,
    cost_side: f64,
    ppy: f64,
    previous: &mut HashMap<u64, f64>,
    policy: Policy,
    band_width: f64,
    min_hold: usize,
    gpu: bool,
    device_pnl: bool,
    control_input: Option<ControlInput>,
) -> WindowEval {
    assert_eq!(rows.len(), scores.len());
    // #305: one shared mask, score and target both finite. `position_for`
    // alone would keep a finite score with a NaN target; that is the six-row
    // difference documented in docs/walkforward.md.
    let mut mask = ValidMask::new(rows.len());
    for i in 0..rows.len() {
        mask.set(i, scores[i].is_finite() && rows[i].target.is_finite());
    }
    let valid: Vec<usize> = mask.indices().collect();
    let n = valid.len();
    let mut positions = vec![0.0f64; n];
    let mut costs = vec![0.0f64; n];
    let mut net = vec![0.0f64; n];
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

    // #306: the band/cooldown book. `positions` are aligned with `order`;
    // `net` includes the final exit of every group. `dp` deliberately keeps
    // the threshold book: the DP is a bound, not a strategy.
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
                        &ord_scores,
                        &ord_returns,
                        &ord_groups,
                        &params,
                        cost_side,
                        crate::trading_gpu::SCAN_THREADS,
                        65_536,
                    )
                },
                || {
                    crate::trading_gpu::cooldown_scan(
                        &ord_scores,
                        &ord_returns,
                        &ord_groups,
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
                        &ord_scores,
                        &ord_returns,
                        &ord_groups,
                        &params,
                        cost_side,
                        crate::trading_gpu::SCAN_THREADS,
                        65_536,
                    )
                },
                || {
                    crate::trading_gpu::band_scan(
                        &ord_scores,
                        &ord_returns,
                        &ord_groups,
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

    for (k, &i) in order.iter().enumerate() {
        let row = &rows[valid[i]];
        let pos = match &band_scan {
            Some((positions, ..)) => positions[k],
            // #305: strict `>` and a non-finite score is flat.
            None => trading_gpu::long_position(scores[valid[i]], threshold) as f64,
        };
        let prev = previous.get(&row.group).copied().unwrap_or(0.0);
        let delta = pos - prev;
        let cost = cost_side * delta.abs();
        positions[i] = pos;
        costs[i] = cost;
        net[i] = pos * row.target - cost;
        previous.insert(row.group, pos);
        first_positions.entry(row.group).or_insert(pos);

        stats.total_abs_delta += delta.abs();
        stats.total_cost += cost;
        if delta != 0.0 {
            stats.trades += 1;
        }
        if prev == 0.0 && pos > 0.0 {
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
    let end_positions = previous.clone();

    // #145: charge the final exit. A flat strategy is free; a held position
    // pays the same per-side cost as its entry.
    let mut last_of_group: HashMap<u64, usize> = HashMap::new();
    for (&g, _) in previous.iter() {
        if let Some((i, _)) = order
            .iter()
            .filter(|&&i| rows[i].group == g)
            .map(|&i| (i, rows[i].time))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            last_of_group.insert(g, i);
        }
    }
    for (g, i) in last_of_group {
        let p = previous.get(&g).copied().unwrap_or(0.0);
        if p != 0.0 {
            let c = cost_side * p.abs();
            net[i] -= c;
            stats.total_cost += c;
            stats.exits += 1;
        }
    }

    // #304: recompute the same long-only book through the fused canonical
    // reduction. Rows are already in `(group, time)` order, which is the
    // group-contiguous order the reducer requires. The scalar evaluator above
    // stays the reported authority; this block is a device/canonical check.
    let device = if device_pnl {
        let params = crate::trading_gpu::PnlParams {
            threshold,
            per_side_bps: cost_side * 10_000.0,
            segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
            cash_return: 0.0,
        };
        #[cfg(feature = "cuda")]
        let report = crate::trading_gpu::reduce_device(
            &ord_scores,
            &ord_returns,
            &ord_groups,
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
        let report = match report {
            Some(r) => r,
            None => crate::trading_gpu::reduce(&ord_scores, &ord_returns, &ord_groups, &params)
                .unwrap_or_else(|e| panic!("canonical P&L reduction failed: {}", e)),
        };
        let cpu_net = net.iter().sum::<f64>();
        Some(DevicePnlCheck {
            backend,
            segment_rows: crate::trading_gpu::DEFAULT_SEGMENT_ROWS,
            tile_candidates: crate::trading_gpu::MAX_TILE_CANDIDATES,
            strategy: report.strategy,
            cash: report.cash,
            buy_and_hold: report.buy_and_hold,
            cpu_net,
            net_delta: report.strategy.net - cpu_net,
        })
    } else {
        None
    };

    let scalar_net = net.iter().sum::<f64>();
    let band = band_scan.map(
        |(_, band_net, groups, carry, backend, fallback_reason)| BandCheck {
            band_width,
            min_hold,
            carry,
            groups,
            net: band_net,
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
                    &ord_groups,
                    cost_side,
                    crate::trading_gpu::SCAN_THREADS,
                    65_536,
                )
            },
            || crate::trading_gpu::dp_scan(&ord_returns, &ord_groups, cost_side, 65_536),
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
    let mut cross = CrossSection::default();
    for &i in &time_order {
        let row = &rows[valid[i]];
        cross.add(row.time, net[i], positions[i] * row.target);
    }
    let (returns, gross_returns) = cross.series();

    // #310: the control ladder on the same rows, the same cost and the same
    // equal-weight period aggregation. Every control starts flat, like the
    // #304 reduction; the strategy's carried seam is not applied to controls.
    let controls = control_input.map(|c| {
        let period_series = |ord_net: &[f64]| -> Vec<f64> {
            let mut by_index = vec![0.0f64; n];
            for (k, &i) in group_order.iter().enumerate() {
                by_index[i] = ord_net[k];
            }
            let mut cs = CrossSection::default();
            for &i in &time_order {
                cs.add(rows[valid[i]].time, by_index[i], 0.0);
            }
            cs.series().0
        };
        crate::baselines::evaluate_controls(
            &ord_positions,
            &ord_returns,
            &ord_groups,
            cost_side,
            c.train_mean,
            c.seed,
            ppy,
            c.check_device,
            &period_series,
        )
        .unwrap_or_else(|e| panic!("control ladder failed: {}", e))
    });

    let metrics = summarize(&returns, &gross_returns, &stats, ppy);
    WindowEval {
        metrics,
        returns,
        gross_returns,
        stats,
        first_positions,
        end_positions,
        device,
        band,
        hindsight_bound,
        controls,
    }
}

fn summarize(returns: &[f64], gross: &[f64], stats: &Stats, ppy: f64) -> Metrics {
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

fn infer_frequency(rows: &[RowBase]) -> (f64, Option<f64>) {
    let mut times: Vec<f64> = rows.iter().map(|r| r.time).collect();
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

fn jstr(s: &str) -> String {
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

fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

/// #304: one fused-reduction total as JSON.
fn pnl_totals_json(t: &crate::trading_gpu::PnlTotals) -> String {
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
         \"cpu_net\":{},\"net_delta\":{}}}",
        jstr(&d.backend),
        d.segment_rows,
        d.tile_candidates,
        pnl_totals_json(&d.strategy),
        pnl_totals_json(&d.cash),
        pnl_totals_json(&d.buy_and_hold),
        jnum(d.cpu_net),
        jnum(d.net_delta)
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
         \"fallback_reason\":{}}}",
        jnum(b.band_width),
        b.min_hold,
        jstr(&b.carry),
        b.groups,
        jnum(b.net),
        jnum(b.scalar_net),
        jnum(b.net - b.scalar_net),
        jstr(&b.backend),
        b.fallback_reason.as_deref().map_or("null".to_string(), jstr)
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

impl WindowResult {
    fn json(&self) -> String {
        format!(
            "{{\"label\":{},\"train_start\":{},\"train_end\":{},\"test_start\":{},\
             \"test_end\":{},\"purge_seconds\":{},\"expression\":{},\
             \"candidate\":{},\"validation_error\":{},\"threshold\":{},\
             \"threshold_source\":{},\"train_rows\":{},\"validation_rows\":{},\
             \"test_rows\":{},\"dropped_rows\":{},\"inferred_period_seconds\":{},\
             \"metrics\":{},\"device_pnl\":{},\"controls\":{},\"band\":{},\"hindsight_bound\":{},\
             \"band_selection\":{}}}",
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
                .map_or("null".to_string(), |s| s.json())
        )
    }
}

impl Report {
    /// Machine-readable JSON: window definitions, cost model, position rule and
    /// metrics, as required by issue #124.
    pub fn json(&self) -> String {
        let windows: Vec<String> = self.windows.iter().map(WindowResult::json).collect();
        let lb = crate::lockbox::Lockbox::crypto_2026h2();
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
        };
        format!(
            "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"walkforward\",\
             \"expression\":{},\"target\":{},\"variables\":[{}],\"time\":{},\
             \"group_by\":{},\
             \"window_definition\":{},\
             \"purge\":{{\"horizon_seconds\":{},\"embargo_seconds\":{}}},\
             \"cost_model\":{{\"per_side_bps\":{},\"round_trip_bps\":{},\
             \"charged_on\":\"|position change| per group; per-window metrics start flat, pooled carries the boundary\"}},\
             \"position_rule\":{},\
             \"policy\":{{\"name\":{},\"hindsight_bound\":{},\"band_width\":{},\
             \"min_hold\":{},\"short\":false,\"gpu\":{}}},\
             \"threshold\":{{\"quantile\":{},\"fixed\":{},\
             \"method\":\"exact order-preserving radix select + linear interpolation (numpy type 7) in f64\",\
             \"sources\":[\"train_type7\",\"validation_type7\",\"fixed\"],\
             \"screening_only\":false,\"source\":{}}},\
             \"annualization\":{{\"periods_per_year\":{},\"method\":{}}},\
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
             \"candidate_count\":{}}},\
             \"pooled_seam\":{{\"reset_flat\":{},\"seam_cost\":{},\"seam_trades\":{}}},\
             \"sign_test\":{{\"positive\":{},\"total\":{},\"applicable\":{},\
             \"p_value\":{},\"method\":{}}},\
             \"lockbox\":{},\
             \"source\":{},\"memory_bound\":{},\"dropped_rows\":{},\
             \"device_pnl\":{},\"windows\":[{}],\"pooled\":{}}}",
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
            jstr(position_rule),
            jstr(self.policy.as_str()),
            self.policy == Policy::Dp,
            jnum(self.band_width),
            self.min_hold,
            self.gpu,
            jnum(self.threshold_quantile),
            self.threshold_fixed.map_or("null".to_string(), jnum),
            jstr(self
                .windows
                .first()
                .map(|w| w.threshold_source.as_str())
                .unwrap_or("none")),
            jnum(self.periods_per_year),
            jstr(&self.annualization),
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
            self.pool_reset_flat,
            jnum(self.seam_cost),
            self.seam_trades,
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
                     \"windows\":{},\"rows\":{},\"max_abs_net_delta\":{}}}",
                    jstr(&d.backend),
                    d.segment_rows,
                    d.tile_candidates,
                    d.windows,
                    d.rows,
                    jnum(d.max_abs_net_delta)
                )
            }),
            windows.join(","),
            self.pooled.json()
        )
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
        };
        let threshold_rule = if let Some(theta) = self.threshold_fixed {
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
