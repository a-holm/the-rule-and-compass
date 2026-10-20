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
//! * The threshold is a quantile of the expression score over the training
//!   rows, or over an optional trailing validation slice of those rows. The
//!   test block is never consulted. The quantile uses the linear interpolation
//!   rule (numpy's default, type 7).
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
    /// Optional trailing fraction of the training time range kept as validation.
    pub validation_frac: f64,
    /// Override the inferred annualisation (e.g. 365 for calendar days,
    /// 252 for trading days, 8760 for hourly bars).
    pub periods_per_year: Option<f64>,
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
}

/// One evaluated window.
#[derive(Clone, Debug)]
pub struct WindowResult {
    pub window: Window,
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
    pub validation_frac: f64,
    pub horizon: f64,
    pub embargo: f64,
    pub periods_per_year: f64,
    /// How the pooled annualisation was chosen.
    pub annualization: String,
    /// `store-stream` or `csv-table`.
    pub source: String,
    pub memory_bound: String,
    dropped_rows: usize,
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
    let tree = parse_expression(spec, &layout)?;
    validate_windows(spec)?;

    let times = data::time_keys(table, layout.time);
    let group_ids = layout.group.map(|g| table_group_ids(table, g));
    let mut acc = Accum::new(spec)?;
    let mut features = Vec::with_capacity(layout.vars.len());
    for r in 0..table.len() {
        let t = times[r];
        if !acc.wants(t) {
            continue;
        }
        features.clear();
        for &c in &layout.vars {
            features.push(table.value(r, c));
        }
        let g = group_ids.as_ref().map_or(0, |ids| ids[r]);
        acc.add(t, g, &features, table.value(r, layout.target), &tree);
    }
    acc.finish(
        spec,
        &layout,
        source,
        "O(test rows + one score per training row per window); CSV is resident",
    )
}

/// Evaluate a `.rcsb` store directly, scanning it one row group at a time.
pub fn run_store(path: &str, spec: &EvalSpec) -> Result<Report, String> {
    let fs = FileStore::open(path)?;
    let names = fs.header().names();
    let numeric = fs.header().numeric();
    let layout = resolve_layout(spec, &names, &numeric)?;
    let tree = parse_expression(spec, &layout)?;
    validate_windows(spec)?;

    let mut acc = Accum::new(spec)?;
    let header = fs.header();
    let mut buf = GroupBuf::new();
    let mut scratch = Vec::new();
    let mut features = Vec::with_capacity(layout.vars.len());
    for group in 0..fs.n_groups() {
        let view = fs.load_group_with(group, &mut buf, &mut scratch)?;
        let len = view.len();
        let time_codes = if layout.time_numeric {
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
            let t = match &time_codes {
                None => view.col(layout.time)[r],
                Some(codes) => {
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
            acc.add(t, g, &features, y, &tree);
        }
    }
    let bound = format!(
        "O(test rows + one score per training row per window); the .rcsb store is scanned \
         group by group, one {} -row group resident",
        fs.group_rows()
    );
    acc.finish(spec, &layout, "store-stream", &bound)
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
    })
}

fn parse_expression(spec: &EvalSpec, layout: &Layout) -> Result<Node, String> {
    let tree = expr::parse(&spec.expression, &layout.var_names)
        .map_err(|e| format!("cannot parse expression '{}': {}", spec.expression, e))?;
    if has_dataset_op(&tree) {
        return Err(
            "walk-forward expressions cannot use dataset aggregates or meta operators; \
             the threshold must depend on the window, not on a whole-column statistic"
                .to_string(),
        );
    }
    Ok(tree)
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
    if !(0.0..1.0).contains(&spec.validation_frac) {
        return Err("--validation-frac must be in [0, 1)".to_string());
    }
    if !spec.cost_bps.is_finite() || spec.cost_bps < 0.0 {
        return Err("--cost-bps must be finite and non-negative".to_string());
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

struct WindowData {
    window: Window,
    validation_start: Option<f64>,
    train_scores: Vec<f64>,
    validation_scores: Vec<f64>,
    test_rows: Vec<TestRow>,
    dropped: usize,
}

struct TestRow {
    time: f64,
    group: u64,
    score: f64,
    target: f64,
}

struct Accum {
    windows: Vec<WindowData>,
}

impl Accum {
    fn new(spec: &EvalSpec) -> Result<Self, String> {
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
                train_scores: Vec::new(),
                validation_scores: Vec::new(),
                test_rows: Vec::new(),
                dropped: 0,
            });
        }
        Ok(Accum { windows })
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

    fn add(&mut self, t: f64, group: u64, features: &[f64], target: f64, tree: &Node) {
        let score = expr::eval(tree, features);
        for d in &mut self.windows {
            if t >= d.window.train_start && t < d.window.train_end {
                if !score.is_finite() {
                    d.dropped += 1;
                } else if d.validation_start.map_or(false, |s| t >= s) {
                    d.validation_scores.push(score);
                } else {
                    d.train_scores.push(score);
                }
            }
            if t >= d.window.test_start && t < d.window.test_end {
                if score.is_finite() && target.is_finite() {
                    d.test_rows.push(TestRow {
                        time: t,
                        group,
                        score,
                        target,
                    });
                } else {
                    d.dropped += 1;
                }
            }
        }
    }

    fn finish(
        self,
        spec: &EvalSpec,
        layout: &Layout,
        source: &str,
        memory_bound: &str,
    ) -> Result<Report, String> {
        let mut windows = Vec::with_capacity(self.windows.len());
        let mut pooled_returns: Vec<f64> = Vec::new();
        let mut pooled_gross: Vec<f64> = Vec::new();
        let mut pooled_stats = Stats::default();
        let mut pooled_ppy: Vec<f64> = Vec::new();
        let mut dropped = 0usize;
        for d in self.windows {
            let source_scores = if d.validation_start.is_some() {
                &d.validation_scores
            } else {
                &d.train_scores
            };
            if source_scores.is_empty() {
                return Err(format!(
                    "window '{}': no finite {} scores to set the threshold",
                    d.window.label,
                    if d.validation_start.is_some() {
                        "validation"
                    } else {
                        "training"
                    }
                ));
            }
            if d.test_rows.is_empty() {
                return Err(format!(
                    "window '{}': no finite test rows in [{}, {})",
                    d.window.label, d.window.test_start, d.window.test_end
                ));
            }
            let threshold = quantile(source_scores, spec.threshold_quantile);
            let source_name = if d.validation_start.is_some() {
                "validation"
            } else {
                "train"
            };
            let (ppy, inferred_dt) = match spec.periods_per_year {
                Some(v) => (v, None),
                None => infer_frequency(&d.test_rows),
            };
            let ev = evaluate_window(&d.test_rows, threshold, spec.cost_bps / 10_000.0, ppy);
            let train_rows = d.train_scores.len() + d.validation_scores.len();
            dropped += d.dropped;
            pooled_returns.extend_from_slice(&ev.returns);
            pooled_gross.extend_from_slice(&ev.gross_returns);
            pooled_stats.add_assign(&ev.stats);
            pooled_ppy.push(ppy);
            windows.push(WindowResult {
                window: d.window,
                threshold,
                threshold_source: source_name.to_string(),
                train_rows,
                validation_rows: d.validation_scores.len(),
                test_rows: d.test_rows.len(),
                dropped_rows: d.dropped,
                inferred_period_seconds: inferred_dt,
                metrics: ev.metrics,
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
        Ok(Report {
            expression: expr::to_string(&parse_expression(spec, layout)?, &layout.var_names),
            target: spec.target.clone(),
            variables: layout.var_names.clone(),
            time_column: spec.time.clone(),
            group_by: spec.group.clone(),
            windows,
            pooled,
            cost_bps: spec.cost_bps,
            threshold_quantile: spec.threshold_quantile,
            validation_frac: spec.validation_frac,
            horizon: spec.horizon,
            embargo: spec.embargo_seconds(),
            periods_per_year: pooled_ppy,
            annualization,
            source: source.to_string(),
            memory_bound: memory_bound.to_string(),
            dropped_rows: dropped,
        })
    }
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
    }
}

struct WindowEval {
    metrics: Metrics,
    returns: Vec<f64>,
    gross_returns: Vec<f64>,
    stats: Stats,
}

/// The position rule, exposed so a test can pin the "never short" property.
pub fn position_for(score: f64, threshold: f64) -> f64 {
    if score.is_finite() && score > threshold {
        1.0
    } else {
        0.0
    }
}

fn evaluate_window(rows: &[TestRow], threshold: f64, cost_side: f64, ppy: f64) -> WindowEval {
    let n = rows.len();
    let mut positions = vec![0.0f64; n];
    let mut costs = vec![0.0f64; n];
    let mut net = vec![0.0f64; n];
    let mut stats = Stats {
        rows: n,
        ..Default::default()
    };

    // Position changes are per group, in time order; every group starts flat.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        rows[a]
            .group
            .cmp(&rows[b].group)
            .then_with(|| rows[a].time.partial_cmp(&rows[b].time).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.cmp(&b))
    });
    let mut previous: HashMap<u64, f64> = HashMap::new();
    for &i in &order {
        let pos = position_for(rows[i].score, threshold);
        let prev = previous.get(&rows[i].group).copied().unwrap_or(0.0);
        let delta = pos - prev;
        let cost = cost_side * delta.abs();
        positions[i] = pos;
        costs[i] = cost;
        net[i] = pos * rows[i].target - cost;
        previous.insert(rows[i].group, pos);

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
            if rows[i].target > 0.0 {
                stats.hits += 1;
                stats.wins += 1;
                stats.win_sum += rows[i].target;
            } else {
                stats.losses += 1;
                stats.loss_sum += rows[i].target;
            }
        }
    }

    // Equal-weight the groups present at each timestamp. Times are sorted so the
    // return series has the same order as the Python reference.
    let mut time_order = order;
    time_order.sort_by(|&a, &b| {
        rows[a]
            .time
            .partial_cmp(&rows[b].time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| rows[a].group.cmp(&rows[b].group))
    });
    let mut returns = Vec::new();
    let mut gross_returns = Vec::new();
    let mut i = 0usize;
    while i < time_order.len() {
        let t = rows[time_order[i]].time;
        let mut sum = 0.0;
        let mut gross = 0.0;
        let mut count = 0usize;
        while i < time_order.len() && rows[time_order[i]].time == t {
            let idx = time_order[i];
            sum += net[idx];
            gross += positions[idx] * rows[idx].target;
            count += 1;
            i += 1;
        }
        returns.push(sum / count as f64);
        gross_returns.push(gross / count as f64);
    }

    let metrics = summarize(&returns, &gross_returns, &stats, ppy);
    WindowEval {
        metrics,
        returns,
        gross_returns,
        stats,
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

fn equity_metrics(returns: &[f64], ppy: f64) -> (f64, f64, f64) {
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
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let q = q.clamp(0.0, 1.0);
    let h = (v.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (h - lo as f64) * (v[hi] - v[lo])
    }
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

fn infer_frequency(rows: &[TestRow]) -> (f64, Option<f64>) {
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
            "{{\"sharpe\":{},\"cagr\":{},\"max_drawdown\":{},\"turnover\":{},\
             \"exposure\":{},\"hit_rate\":{},\"break_even_hit_rate\":{},\
             \"trades\":{},\"entries\":{},\"rows\":{},\"active_rows\":{},\
             \"periods\":{},\"periods_per_year\":{},\"total_return\":{},\
             \"mean_return\":{},\"avg_cost_bps\":{},\"gross_sharpe\":{}}}",
            jnum(self.sharpe),
            jnum(self.cagr),
            jnum(self.max_drawdown),
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
            jnum(self.gross_sharpe)
        )
    }
}

impl WindowResult {
    fn json(&self) -> String {
        format!(
            "{{\"label\":{},\"train_start\":{},\"train_end\":{},\"test_start\":{},\
             \"test_end\":{},\"purge_seconds\":{},\"threshold\":{},\
             \"threshold_source\":{},\"train_rows\":{},\"validation_rows\":{},\
             \"test_rows\":{},\"dropped_rows\":{},\"inferred_period_seconds\":{},\
             \"metrics\":{}}}",
            jstr(&self.window.label),
            jnum(self.window.train_start),
            jnum(self.window.train_end),
            jnum(self.window.test_start),
            jnum(self.window.test_end),
            jnum(self.window.test_start - self.window.train_end),
            jnum(self.threshold),
            jstr(&self.threshold_source),
            self.train_rows,
            self.validation_rows,
            self.test_rows,
            self.dropped_rows,
            self.inferred_period_seconds.map_or("null".to_string(), jnum),
            self.metrics.json()
        )
    }
}

impl Report {
    /// Machine-readable JSON: window definitions, cost model, position rule and
    /// metrics, as required by issue #124.
    pub fn json(&self) -> String {
        let windows: Vec<String> = self.windows.iter().map(WindowResult::json).collect();
        format!(
            "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"walkforward\",\
             \"expression\":{},\"target\":{},\"variables\":[{}],\"time\":{},\
             \"group_by\":{},\
             \"window_definition\":{},\
             \"purge\":{{\"horizon_seconds\":{},\"embargo_seconds\":{}}},\
             \"cost_model\":{{\"per_side_bps\":{},\"round_trip_bps\":{},\
             \"charged_on\":\"|position change| per group, starting from flat\"}},\
             \"position_rule\":\"long/flat: 1 if score > threshold else 0; never short; no leverage\",\
             \"threshold\":{{\"quantile\":{},\"method\":\"linear interpolation (numpy type 7)\",\
             \"source\":\"train or validation only\"}},\
             \"annualization\":{{\"periods_per_year\":{},\"method\":{}}},\
             \"source\":{},\"memory_bound\":{},\"dropped_rows\":{},\
             \"windows\":[{}],\"pooled\":{}}}",
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
            jnum(self.threshold_quantile),
            jnum(self.periods_per_year),
            jstr(&self.annualization),
            jstr(&self.source),
            jstr(&self.memory_bound),
            self.dropped_rows,
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
        s.push_str(&format!(
            "- Long/flat only: `position = 1` when `score > threshold`, else `0`; never short, no leverage.\n\
             - Threshold: q{:.4} of the {} scores, linear interpolation (numpy type 7).\n\
             - Cost: {:.4} bp per side on `|position change|`, so {:.4} bp per round trip; each group starts flat.\n\
             - Purge: horizon {:.1}s, embargo {:.1}s.\n\
             - Annualisation: {} ({:.6} periods/year).\n\
             - Source: {}.\n\n",
            self.threshold_quantile,
            if self.validation_frac > 0.0 {
                "training/validation split"
            } else {
                "training"
            },
            self.cost_bps,
            self.cost_bps * 2.0,
            self.horizon,
            self.embargo,
            self.annualization,
            self.periods_per_year,
            self.source
        ));
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
        s
    }
}
