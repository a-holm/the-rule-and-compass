//! Exact long/flat P&L, Sharpe and log-growth on a prediction vector.
//!
//! The book is long-only. A position is `p_t` in `[0, max_position]`:
//! `1[pred > theta]` by default, or `clip((pred - theta) / tau, 0, 1)` with a
//! sizing scale. A negative position is a hard error. Per row of one group,
//! in time order:
//!
//! ```text
//! gross_t = p_t * r_t + (1 - p_t) * cash_return - p_t * funding_t
//! cost_t  = c_side * |p_t - p_{t-1}|      (the first row starts from flat)
//! net_t   = gross_t - cost_t              (the last row of a group also pays
//!                                          c_side * p_end for the exit)
//! ```
//!
//! Rows are equal-weighted per timestamp (the walk-forward cross-section). The
//! Sharpe is daily: sub-daily periods are compounded per calendar day and
//! annualised with `sqrt(365)`, with the sample standard deviation (ddof = 1).
//! Under the plain threshold rule with no cash return and no funding the
//! period metrics reproduce [`crate::walkforward`] bit for bit.
//!
//! Everything is exact CPU `f64`. There is no short leg.

use crate::risk::{self, RiskStats};
use crate::trading_gpu;
use crate::walkforward::{self, CrossSection, Metrics, Stats, SECONDS_PER_YEAR};
use std::collections::BTreeMap;

pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// Per-side costs, in basis points, at which the cost ladder re-prices the book.
pub const LADDER_BPS: [f64; 5] = [0.0, 5.0, 10.0, 20.0, 50.0];

const BP: f64 = 1e-4;

/// What finalist selection minimises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// The configured error metric (the historical behaviour).
    Metric,
    Pnl,
    Sharpe,
    LogGrowth,
}

impl Objective {
    pub fn parse(s: &str) -> Result<Objective, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mse" | "metric" => Ok(Objective::Metric),
            "pnl" => Ok(Objective::Pnl),
            "sharpe" => Ok(Objective::Sharpe),
            "log_growth" | "log-growth" => Ok(Objective::LogGrowth),
            other => Err(format!(
                "unknown --objective '{}' (expected mse, pnl, sharpe or log_growth)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Objective::Metric => "mse",
            Objective::Pnl => "pnl",
            Objective::Sharpe => "sharpe",
            Objective::LogGrowth => "log_growth",
        }
    }

    pub fn is_trading(self) -> bool {
        self != Objective::Metric
    }

    /// Lower is better. A book with fewer than `min_trades` trades, or a
    /// non-finite value (a `NaN` Sharpe is unusable, never zero), is
    /// `+inf`.
    pub fn score(self, stats: &PathStats, min_trades: usize) -> f64 {
        if stats.trades < min_trades {
            return f64::INFINITY;
        }
        let v = match self {
            Objective::Metric => return f64::INFINITY,
            Objective::Pnl => -stats.net_pnl,
            Objective::Sharpe => -stats.sharpe,
            Objective::LogGrowth => -stats.log_growth,
        };
        if v.is_finite() {
            v
        } else {
            f64::INFINITY
        }
    }
}

/// The book's assumptions. Costs are per side, in basis points of notional.
#[derive(Clone, Debug, PartialEq)]
pub struct TradingSpec {
    pub cost_bps: f64,
    pub fee_bps: f64,
    pub half_spread_bps: f64,
    pub slippage_bps: f64,
    /// Name of the per-period funding column, for reporting; the values travel
    /// in [`TradingRows::funding`].
    pub funding_col: Option<String>,
    /// Return per period earned while flat.
    pub cash_return: f64,
    pub min_trades: usize,
    /// Minimum consecutive long rows, entry row included, before an exit
    /// (indicator rule only).
    pub min_holding: usize,
    pub threshold: f64,
    /// Sizing scale; `None` is the `{0, 1}` indicator.
    pub tau: Option<f64>,
    pub max_position: f64,
    /// Override of the annualisation. `None` compounds sub-daily periods per
    /// day and uses 365.
    pub periods_per_year: Option<f64>,
}

impl Default for TradingSpec {
    fn default() -> Self {
        TradingSpec {
            cost_bps: 0.0,
            fee_bps: 0.0,
            half_spread_bps: 0.0,
            slippage_bps: 0.0,
            funding_col: None,
            cash_return: 0.0,
            min_trades: 1,
            min_holding: 1,
            threshold: 0.0,
            tau: None,
            max_position: 1.0,
            periods_per_year: None,
        }
    }
}

impl TradingSpec {
    /// Total charge per side, in basis points.
    pub fn per_side_bps(&self) -> f64 {
        self.cost_bps + self.fee_bps + self.half_spread_bps + self.slippage_bps
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("cost_bps", self.cost_bps),
            ("fee_bps", self.fee_bps),
            ("half_spread_bps", self.half_spread_bps),
            ("slippage_bps", self.slippage_bps),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("{} must be finite and non-negative, got {}", name, v));
            }
        }
        if !self.cash_return.is_finite() {
            return Err("cash_return must be finite".to_string());
        }
        if !self.threshold.is_finite() {
            return Err("threshold must be finite".to_string());
        }
        if !(self.max_position > 0.0 && self.max_position <= 1.0) {
            return Err(format!(
                "max_position must be in (0, 1], got {}: the book is long-only and unlevered",
                self.max_position
            ));
        }
        if let Some(tau) = self.tau {
            if !(tau.is_finite() && tau > 0.0) {
                return Err(format!("tau must be positive and finite, got {}", tau));
            }
            if self.min_holding > 1 {
                return Err("min_holding needs the indicator position (no tau)".to_string());
            }
        }
        if self.min_holding == 0 {
            return Err("min_holding must be at least 1".to_string());
        }
        if let Some(p) = self.periods_per_year {
            if !(p.is_finite() && p > 0.0) {
                return Err(format!("periods_per_year must be positive, got {}", p));
            }
        }
        Ok(())
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"position\":\"long_only\",\"threshold\":{},\"tau\":{},\"max_position\":{},\
             \"cost_bps\":{},\"fee_bps\":{},\"half_spread_bps\":{},\"slippage_bps\":{},\
             \"per_side_bps\":{},\"funding_col\":{},\"cash_return\":{},\"min_trades\":{},\
             \"min_holding\":{}}}",
            num(self.threshold),
            self.tau.map(num).unwrap_or_else(|| "null".to_string()),
            num(self.max_position),
            num(self.cost_bps),
            num(self.fee_bps),
            num(self.half_spread_bps),
            num(self.slippage_bps),
            num(self.per_side_bps()),
            self.funding_col
                .as_deref()
                .map(crate::devlog::text)
                .unwrap_or_else(|| "null".to_string()),
            num(self.cash_return),
            self.min_trades,
            self.min_holding
        )
    }
}

/// Time, group and funding of the rows, aligned with the prediction vector.
#[derive(Clone, Debug, Default)]
pub struct TradingRows {
    pub time: Vec<f64>,
    pub group: Vec<u64>,
    /// Per-period funding rate paid while long; empty means none.
    pub funding: Vec<f64>,
}

impl TradingRows {
    /// First row of its group.
    pub const FIRST: u8 = 1;
    /// Last row of its group.
    pub const LAST: u8 = 2;

    /// Each group is one contiguous run and time is non-decreasing inside a run.
    pub fn is_canonical(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        for i in 0..self.group.len() {
            if i > 0 && self.group[i] == self.group[i - 1] {
                if self.time[i] < self.time[i - 1] {
                    return false;
                }
            } else if !seen.insert(self.group[i]) {
                return false;
            }
        }
        true
    }

    /// Bit `FIRST` on the first row of each group, bit `LAST` on the last.
    /// Valid only on canonical rows.
    pub fn seam_flags(&self) -> Vec<u8> {
        let n = self.group.len();
        (0..n)
            .map(|i| {
                let first = i == 0 || self.group[i] != self.group[i - 1];
                let last = i + 1 == n || self.group[i] != self.group[i + 1];
                (if first { Self::FIRST } else { 0 }) | (if last { Self::LAST } else { 0 })
            })
            .collect()
    }
}

/// Net P&L, growth and drawdown of one comparison book.
#[derive(Clone, Copy, Debug)]
pub struct Headline {
    pub net_pnl: f64,
    pub total_return: f64,
    pub log_growth: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
}

impl Headline {
    fn json(&self) -> String {
        format!(
            "{{\"net_pnl\":{},\"total_return\":{},\"log_growth\":{},\"sharpe\":{},\"max_drawdown\":{}}}",
            num(self.net_pnl),
            num(self.total_return),
            num(self.log_growth),
            num(self.sharpe),
            num(self.max_drawdown)
        )
    }
}

/// The book re-priced at one per-side cost.
#[derive(Clone, Copy, Debug)]
pub struct LadderRow {
    pub cost_bps: f64,
    pub net_pnl: f64,
    pub sharpe: f64,
}

#[derive(Clone, Debug)]
pub struct PathStats {
    pub rows: usize,
    pub groups: usize,
    pub periods: usize,
    pub periods_per_year: f64,
    /// `daily_compounded` or `per_period`.
    pub annualisation: &'static str,
    /// Sum over rows of `gross_t`.
    pub gross_pnl: f64,
    /// Sum over rows of `net_t`, including every exit.
    pub net_pnl: f64,
    pub total_cost: f64,
    pub funding_paid: f64,
    pub total_return: f64,
    /// Sum of `ln(1 + R)` over the annualisation series; `-inf` on ruin.
    pub log_growth: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown: f64,
    pub calmar: f64,
    /// Sum of `|delta p|` per year, exits excluded (the walk-forward unit).
    pub turnover: f64,
    pub trades: usize,
    pub entries: usize,
    pub exits: usize,
    pub trades_per_year: f64,
    /// Fraction of rows with a position.
    pub exposure: f64,
    pub hit_rate: f64,
    /// Per-side cost at which `net_pnl` is zero.
    pub break_even_bps: f64,
    pub cash: Headline,
    pub buy_and_hold: Headline,
    pub ladder: Vec<LadderRow>,
    /// The walk-forward metric block on the per-timestamp series.
    pub period: Metrics,
}

impl PathStats {
    pub fn json(&self) -> String {
        let ladder: Vec<String> = self
            .ladder
            .iter()
            .map(|l| {
                format!(
                    "{{\"cost_bps\":{},\"net_pnl\":{},\"sharpe\":{}}}",
                    num(l.cost_bps),
                    num(l.net_pnl),
                    num(l.sharpe)
                )
            })
            .collect();
        format!(
            "{{\"rows\":{},\"groups\":{},\"periods\":{},\"periods_per_year\":{},\"annualisation\":\"{}\",\
             \"gross_pnl\":{},\"net_pnl\":{},\"total_cost\":{},\"funding_paid\":{},\"total_return\":{},\
             \"log_growth\":{},\"sharpe\":{},\"sortino\":{},\"max_drawdown\":{},\"calmar\":{},\
             \"turnover\":{},\"trades\":{},\"entries\":{},\"exits\":{},\"trades_per_year\":{},\
             \"exposure\":{},\"hit_rate\":{},\"break_even_bps_per_side\":{},\
             \"cash\":{},\"buy_and_hold\":{},\"cost_ladder\":[{}],\"period_sharpe\":{}}}",
            self.rows,
            self.groups,
            self.periods,
            num(self.periods_per_year),
            self.annualisation,
            num(self.gross_pnl),
            num(self.net_pnl),
            num(self.total_cost),
            num(self.funding_paid),
            num(self.total_return),
            num(self.log_growth),
            num(self.sharpe),
            num(self.sortino),
            num(self.max_drawdown),
            num(self.calmar),
            num(self.turnover),
            self.trades,
            self.entries,
            self.exits,
            num(self.trades_per_year),
            num(self.exposure),
            num(self.hit_rate),
            num(self.break_even_bps),
            self.cash.json(),
            self.buy_and_hold.json(),
            ladder.join(","),
            num(self.period.sharpe)
        )
    }
}

fn num(v: f64) -> String {
    crate::devlog::num(v)
}

/// Usable rows in `(group, time, input index)` order.
struct Ctx {
    time: Vec<f64>,
    group: Vec<u64>,
    ret: Vec<f64>,
    funding: Option<Vec<f64>>,
    cash: f64,
    /// Row indices in `(group, time)` order.
    order: Vec<usize>,
    /// Row indices in `(time, group)` order.
    time_order: Vec<usize>,
    groups: usize,
}

impl Ctx {
    fn new(
        keep: &[usize],
        r: &[f64],
        rows: &TradingRows,
        cash: f64,
    ) -> Ctx {
        let time: Vec<f64> = keep.iter().map(|&i| rows.time[i]).collect();
        let group: Vec<u64> = keep.iter().map(|&i| rows.group[i]).collect();
        let ret: Vec<f64> = keep.iter().map(|&i| r[i]).collect();
        let funding = if rows.funding.is_empty() {
            None
        } else {
            Some(keep.iter().map(|&i| rows.funding[i]).collect())
        };
        let n = keep.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            group[a]
                .cmp(&group[b])
                .then_with(|| time[a].partial_cmp(&time[b]).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.cmp(&b))
        });
        let mut time_order = order.clone();
        time_order.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| group[a].cmp(&group[b]))
        });
        let mut groups = 0usize;
        let mut prev: Option<u64> = None;
        for &i in &order {
            if prev != Some(group[i]) {
                groups += 1;
                prev = Some(group[i]);
            }
        }
        Ctx {
            time,
            group,
            ret,
            funding,
            cash,
            order,
            time_order,
            groups,
        }
    }
}

struct Book {
    gross: Vec<f64>,
    net: Vec<f64>,
    stats: Stats,
    cross: CrossSection,
    funding_paid: f64,
    abs_delta_with_exit: f64,
}

fn run_book(ctx: &Ctx, positions: &[f64], cost_side: f64) -> Book {
    let n = ctx.ret.len();
    let mut gross = vec![0.0f64; n];
    let mut net = vec![0.0f64; n];
    let mut stats = Stats {
        rows: n,
        ..Default::default()
    };
    let mut previous: BTreeMap<u64, f64> = BTreeMap::new();
    let mut funding_paid = 0.0f64;
    let mut abs_delta_with_exit = 0.0f64;
    let mut last_of_group: BTreeMap<u64, usize> = BTreeMap::new();
    for &i in &ctx.order {
        let g = ctx.group[i];
        let pos = positions[i];
        let r = ctx.ret[i];
        let prev = previous.get(&g).copied().unwrap_or(0.0);
        let delta = pos - prev;
        let cost = cost_side * delta.abs();
        let mut gr = pos * r;
        if ctx.cash != 0.0 {
            gr += (1.0 - pos) * ctx.cash;
        }
        if let Some(f) = &ctx.funding {
            let paid = f[i] * pos;
            gr -= paid;
            funding_paid += paid;
        }
        gross[i] = gr;
        net[i] = gr - cost;
        previous.insert(g, pos);
        last_of_group.insert(g, i);

        stats.total_abs_delta += delta.abs();
        abs_delta_with_exit += delta.abs();
        stats.total_cost += cost;
        if delta != 0.0 {
            stats.trades += 1;
        }
        if prev == 0.0 && pos > 0.0 {
            stats.entries += 1;
        }
        if pos > 0.0 {
            stats.active += 1;
            if r > 0.0 {
                stats.hits += 1;
                stats.wins += 1;
                stats.win_sum += r;
            } else {
                stats.losses += 1;
                stats.loss_sum += r;
            }
        }
    }
    for (g, i) in last_of_group {
        let p = previous.get(&g).copied().unwrap_or(0.0);
        if p != 0.0 {
            let c = cost_side * p.abs();
            net[i] -= c;
            stats.total_cost += c;
            stats.exits += 1;
            abs_delta_with_exit += p.abs();
        }
    }
    let mut cross = CrossSection::default();
    for &i in &ctx.time_order {
        cross.add(ctx.time[i], net[i], gross[i]);
    }
    Book {
        gross,
        net,
        stats,
        cross,
        funding_paid,
        abs_delta_with_exit,
    }
}

struct Series {
    risk: RiskStats,
    annualisation: &'static str,
    /// Annualisation of the per-timestamp block (the walk-forward rule).
    period_ppy: f64,
    returns: Vec<f64>,
    gross_returns: Vec<f64>,
}

fn median_step(times: &[f64]) -> Option<f64> {
    let mut diffs: Vec<f64> = times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d > 1e-9)
        .collect();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = diffs.len() / 2;
    Some(if diffs.len() % 2 == 1 {
        diffs[mid]
    } else {
        (diffs[mid - 1] + diffs[mid]) * 0.5
    })
}

fn series_of(cross: &CrossSection, spec: &TradingSpec) -> Series {
    let times: Vec<f64> = cross.periods.iter().map(|p| p.time).collect();
    let (returns, gross_returns) = cross.series();
    let step = median_step(&times);
    let period_ppy = match spec.periods_per_year {
        Some(v) => v,
        None => match step {
            Some(dt) if dt >= 1.0 && dt <= SECONDS_PER_YEAR => SECONDS_PER_YEAR / dt,
            _ => 365.0,
        },
    };
    let sub_daily = spec.periods_per_year.is_none() && step.is_some_and(|dt| dt < SECONDS_PER_DAY);
    let (series, ppy, annualisation) = if sub_daily {
        let mut days: Vec<f64> = Vec::new();
        let mut last_day = i64::MIN;
        for (t, r) in times.iter().zip(&returns) {
            let day = (t / SECONDS_PER_DAY).floor() as i64;
            if day != last_day {
                days.push(1.0);
                last_day = day;
            }
            if let Some(eq) = days.last_mut() {
                *eq *= 1.0 + r;
            }
        }
        (days.iter().map(|eq| eq - 1.0).collect(), 365.0, "daily_compounded")
    } else {
        (returns.clone(), period_ppy, "per_period")
    };
    Series {
        risk: risk::risk_stats(&series, ppy, risk::DEFAULT_SORTINO_MAR),
        annualisation,
        period_ppy,
        returns,
        gross_returns,
    }
}

fn headline(book: &Book, series: &Series) -> Headline {
    Headline {
        net_pnl: book.net.iter().sum::<f64>(),
        total_return: series.risk.total_return,
        log_growth: series.risk.total_log_growth,
        sharpe: series.risk.sharpe,
        max_drawdown: series.risk.max_drawdown,
    }
}

fn finish(ctx: &Ctx, positions: &[f64], spec: &TradingSpec) -> Result<PathStats, String> {
    let n = ctx.ret.len();
    if n == 0 {
        return Err("no usable rows (every prediction or return is non-finite)".to_string());
    }
    for (k, &p) in positions.iter().enumerate() {
        if !p.is_finite() {
            return Err(format!("position {} is not finite", k));
        }
        if p < 0.0 {
            return Err(format!(
                "negative position {} at usable row {}: the book is long-only, there is no short leg",
                p, k
            ));
        }
        if p > spec.max_position {
            return Err(format!(
                "position {} at usable row {} exceeds max_position {}",
                p, k, spec.max_position
            ));
        }
    }
    let cost_side = spec.per_side_bps() * BP;
    let book = run_book(ctx, positions, cost_side);
    let series = series_of(&book.cross, spec);
    let period = walkforward::summarize(
        &series.returns,
        &series.gross_returns,
        &book.stats,
        series.period_ppy,
    );
    let main = headline(&book, &series);

    let flat = run_book(ctx, &vec![0.0; n], cost_side);
    let flat_series = series_of(&flat.cross, spec);
    let held = run_book(ctx, &vec![1.0; n], cost_side);
    let held_series = series_of(&held.cross, spec);

    let ladder: Vec<LadderRow> = LADDER_BPS
        .iter()
        .map(|&bps| {
            let b = run_book(ctx, positions, bps * BP);
            let s = series_of(&b.cross, spec);
            LadderRow {
                cost_bps: bps,
                net_pnl: b.net.iter().sum::<f64>(),
                sharpe: s.risk.sharpe,
            }
        })
        .collect();

    let gross_pnl = book.gross.iter().sum::<f64>();
    let break_even_bps = if book.abs_delta_with_exit > 0.0 {
        gross_pnl / book.abs_delta_with_exit / BP
    } else {
        f64::NAN
    };
    let years = if period.periods > 0 && series.period_ppy > 0.0 {
        period.periods as f64 / series.period_ppy
    } else {
        f64::NAN
    };
    let calmar = if series.risk.max_drawdown < 0.0 {
        series.risk.cagr / series.risk.max_drawdown.abs()
    } else {
        f64::NAN
    };
    Ok(PathStats {
        rows: n,
        groups: ctx.groups,
        periods: period.periods,
        periods_per_year: series.risk.periods_per_year,
        annualisation: series.annualisation,
        gross_pnl,
        net_pnl: main.net_pnl,
        total_cost: book.stats.total_cost,
        funding_paid: book.funding_paid,
        total_return: main.total_return,
        log_growth: main.log_growth,
        sharpe: main.sharpe,
        sortino: series.risk.sortino,
        max_drawdown: main.max_drawdown,
        calmar,
        turnover: period.turnover,
        trades: book.stats.trades,
        entries: book.stats.entries,
        exits: book.stats.exits,
        trades_per_year: book.stats.trades as f64 / years,
        exposure: period.exposure,
        hit_rate: period.hit_rate,
        break_even_bps,
        cash: headline(&flat, &flat_series),
        buy_and_hold: headline(&held, &held_series),
        ladder,
        period,
    })
}

/// Smallest positive step between consecutive rows of one group, in seconds,
/// and the rows that realise the first violation of `horizon_seconds`.
/// Non-finite times and equal timestamps are skipped.
fn group_steps(time: &[f64], group: &[u64], horizon_seconds: f64) -> (Option<f64>, Option<(u64, usize, usize, f64)>) {
    let mut order: Vec<usize> = (0..time.len().min(group.len())).filter(|&i| time[i].is_finite()).collect();
    order.sort_by(|&a, &b| {
        group[a]
            .cmp(&group[b])
            .then_with(|| time[a].partial_cmp(&time[b]).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.cmp(&b))
    });
    let limit = horizon_seconds * (1.0 - 1e-9);
    let mut min_step: Option<f64> = None;
    let mut bad: Option<(u64, usize, usize, f64)> = None;
    for w in order.windows(2) {
        let (a, b) = (w[0], w[1]);
        if group[a] != group[b] {
            continue;
        }
        let step = time[b] - time[a];
        if !(step > 0.0) {
            continue;
        }
        if min_step.map_or(true, |m| step < m) {
            min_step = Some(step);
        }
        if bad.is_none() && step < limit {
            bad = Some((group[a], a, b, step));
        }
    }
    (min_step, bad)
}

/// Smallest positive step between consecutive rows of one group, in seconds,
/// in (group, time) order. `None` when no group has two rows.
pub fn min_group_step(rows: &TradingRows) -> Option<f64> {
    group_steps(&rows.time, &rows.group, 0.0).0
}

/// Refuse a book whose per-row returns overlap: some group has two rows
/// closer than `horizon_seconds`. Returns the measured minimum step. A
/// non-positive horizon skips the check.
pub fn check_return_spacing(rows: &TradingRows, horizon_seconds: f64) -> Result<Option<f64>, String> {
    let (min_step, bad) = group_steps(&rows.time, &rows.group, horizon_seconds);
    if horizon_seconds > 0.0 {
        if let Some((g, a, b, step)) = bad {
            return Err(format!(
                "overlapping returns: group {} has rows {} and {} {} seconds apart, below the {}-second \
                 label horizon; use a label stride of at least the horizon (long:<h>,stride=<h>) or a \
                 one-bar label",
                g, a, b, step, horizon_seconds
            ));
        }
    }
    Ok(min_step)
}

fn check_lengths(n: usize, r: &[f64], rows: &TradingRows) -> Result<(), String> {
    if r.len() != n || rows.time.len() != n || rows.group.len() != n {
        return Err(format!(
            "length mismatch: {} predictions, {} returns, {} times, {} groups",
            n,
            r.len(),
            rows.time.len(),
            rows.group.len()
        ));
    }
    if !rows.funding.is_empty() && rows.funding.len() != n {
        return Err(format!("{} funding values for {} rows", rows.funding.len(), n));
    }
    Ok(())
}

fn usable(i: usize, r: &[f64], rows: &TradingRows) -> bool {
    r[i].is_finite()
        && rows.time[i].is_finite()
        && (rows.funding.is_empty() || rows.funding[i].is_finite())
}

/// Position of one prediction under the spec's map. `NaN` for a non-finite
/// prediction, which the caller drops.
pub fn position_of(pred: f64, spec: &TradingSpec) -> f64 {
    if !pred.is_finite() {
        return f64::NAN;
    }
    match spec.tau {
        None => trading_gpu::long_position(pred, spec.threshold) as f64 * spec.max_position,
        Some(tau) => ((pred - spec.threshold) / tau).clamp(0.0, 1.0) * spec.max_position,
    }
}

/// Score a prediction vector against the row returns `r`.
///
/// Rows whose prediction, return or time is non-finite are dropped, the same
/// shared mask as the walk-forward evaluator.
pub fn evaluate(
    pred: &[f64],
    r: &[f64],
    rows: &TradingRows,
    spec: &TradingSpec,
) -> Result<PathStats, String> {
    spec.validate()?;
    check_lengths(pred.len(), r, rows)?;
    let keep: Vec<usize> = (0..pred.len())
        .filter(|&i| pred[i].is_finite() && usable(i, r, rows))
        .collect();
    let ctx = Ctx::new(&keep, r, rows, spec.cash_return);
    let scores: Vec<f64> = keep.iter().map(|&i| pred[i]).collect();
    let positions: Vec<f64> = if spec.min_holding > 1 {
        let mut positions = vec![0.0f64; keep.len()];
        let mut current: Option<u64> = None;
        let mut held = 0usize;
        for &i in &ctx.order {
            if current != Some(ctx.group[i]) {
                current = Some(ctx.group[i]);
                held = 0;
            }
            let want = position_of(scores[i], spec);
            let long = if held > 0 && want == 0.0 && held < spec.min_holding {
                spec.max_position
            } else {
                want
            };
            held = if long > 0.0 { held + 1 } else { 0 };
            positions[i] = long;
        }
        positions
    } else {
        scores.iter().map(|&s| position_of(s, spec)).collect()
    };
    finish(&ctx, &positions, spec)
}

/// Score explicit positions, one per row. A negative position is an error.
pub fn evaluate_positions(
    positions: &[f64],
    r: &[f64],
    rows: &TradingRows,
    spec: &TradingSpec,
) -> Result<PathStats, String> {
    spec.validate()?;
    check_lengths(positions.len(), r, rows)?;
    let keep: Vec<usize> = (0..positions.len()).filter(|&i| usable(i, r, rows)).collect();
    let ctx = Ctx::new(&keep, r, rows, spec.cash_return);
    let kept: Vec<f64> = keep.iter().map(|&i| positions[i]).collect();
    finish(&ctx, &kept, spec)
}

/// True when the expression reads no variable.
pub fn is_constant(tree: &crate::expr::Node) -> bool {
    tree.op != crate::expr::Op::Input && tree.kids.iter().all(is_constant)
}

/// Rows per canonical summation segment of the fitness gross sum.
pub const FITNESS_SEGMENT_ROWS: usize = trading_gpu::DEFAULT_SEGMENT_ROWS;
/// Lanes inside one segment.
pub const FITNESS_LANES: usize = trading_gpu::PNL_LANES;
/// Warp sums that the segments are spread over.
pub const FITNESS_WARPS: usize = 8;

/// The long/flat book of one prediction vector, reduced to what the search
/// ranks by (#480).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitnessTotals {
    pub gross: f64,
    pub flips: u64,
    pub exits: u64,
    pub long_rows: u64,
    /// Some prediction was not finite.
    pub nonfinite: bool,
}

/// Exact CPU `f64` long/flat net regret on the canonical training rows.
///
/// `regret = (B - net) / B`, with `net` the excess net over cash and `B` the
/// hindsight DP bound on the same rows. Lower is better; all-flat is exactly 1.
#[derive(Clone, Debug)]
pub struct FitnessBook {
    excess: Vec<f64>,
    flags: Vec<u8>,
    groups: usize,
    side: f64,
    threshold: f64,
    bound: f64,
    min_trades: usize,
    hash: u64,
}

fn fnv_u64(h: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

impl FitnessBook {
    /// `horizon_seconds` is the label horizon of the target; rows of one group
    /// closer than that are refused (#468). A non-positive horizon skips it.
    pub fn new(slice: &crate::data::TradingSlice, spec: &TradingSpec, horizon_seconds: f64) -> Result<Self, String> {
        spec.validate()?;
        if spec.tau.is_some() || spec.min_holding > 1 || spec.max_position != 1.0 {
            return Err(
                "the fitness book is the plain {0, 1} indicator: it needs min_holding 1, max_position 1 and no tau"
                    .to_string(),
            );
        }
        let n = slice.ret.len();
        if n == 0 {
            return Err("no training rows".to_string());
        }
        let rows = &slice.rows;
        if rows.time.len() != n || rows.group.len() != n || (!rows.funding.is_empty() && rows.funding.len() != n) {
            return Err(format!(
                "length mismatch: {} returns, {} times, {} groups, {} funding values",
                n,
                rows.time.len(),
                rows.group.len(),
                rows.funding.len()
            ));
        }
        if !rows.is_canonical() {
            return Err(
                "training rows are not in canonical (group, time) order; set canonical_train_order".to_string(),
            );
        }
        check_return_spacing(rows, horizon_seconds)?;
        let mut excess = Vec::with_capacity(n);
        for i in 0..n {
            let f = if rows.funding.is_empty() { 0.0 } else { rows.funding[i] };
            let e = slice.ret[i] - f - spec.cash_return;
            if !e.is_finite() || !rows.time[i].is_finite() {
                return Err(format!("training row {} has a non-finite return, funding or time", i));
            }
            excess.push(e);
        }
        let flags = rows.seam_flags();
        let groups = flags.iter().filter(|&&f| f & TradingRows::FIRST != 0).count();
        let side = spec.per_side_bps() * BP;
        let bound = trading_gpu::dp_bound(&excess, &rows.group, side)?;
        if !(bound > 0.0) {
            return Err(format!(
                "no profitable long/flat path exists on the training rows at {} bp per side (hindsight bound {})",
                spec.per_side_bps(),
                bound
            ));
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for e in &excess {
            fnv_u64(&mut hash, e.to_bits());
        }
        for f in &flags {
            fnv_u64(&mut hash, *f as u64);
        }
        fnv_u64(&mut hash, side.to_bits());
        fnv_u64(&mut hash, spec.threshold.to_bits());
        fnv_u64(&mut hash, spec.min_trades as u64);
        Ok(FitnessBook {
            excess,
            flags,
            groups,
            side,
            threshold: spec.threshold,
            bound,
            min_trades: spec.min_trades,
            hash,
        })
    }

    pub fn len(&self) -> usize {
        self.excess.len()
    }

    pub fn is_empty(&self) -> bool {
        self.excess.is_empty()
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn side(&self) -> f64 {
        self.side
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn bound(&self) -> f64 {
        self.bound
    }

    pub fn min_trades(&self) -> usize {
        self.min_trades
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Reduce one prediction vector. `gross` uses the fixed segment, lane and
    /// warp tree documented on [`FITNESS_SEGMENT_ROWS`]; integer counts are
    /// order-free.
    pub fn totals(&self, pred: &[f64]) -> FitnessTotals {
        debug_assert_eq!(pred.len(), self.excess.len());
        let n = self.excess.len().min(pred.len());
        let mut warps = [0.0f64; FITNESS_WARPS];
        let mut flips = 0u64;
        let mut exits = 0u64;
        let mut long_rows = 0u64;
        let mut nonfinite = false;
        let mut prev = false;
        let mut seg_start = 0usize;
        let mut seg = 0usize;
        while seg_start < n {
            let seg_end = (seg_start + FITNESS_SEGMENT_ROWS).min(n);
            let mut lanes = [0.0f64; FITNESS_LANES];
            for k in seg_start..seg_end {
                let p = pred[k];
                let long = if p.is_finite() {
                    p > self.threshold
                } else {
                    nonfinite = true;
                    false
                };
                let flag = self.flags[k];
                if flag & TradingRows::FIRST != 0 {
                    prev = false;
                }
                if long != prev {
                    flips += 1;
                }
                if long {
                    long_rows += 1;
                    lanes[(k - seg_start) % FITNESS_LANES] += self.excess[k];
                    if flag & TradingRows::LAST != 0 {
                        exits += 1;
                    }
                }
                prev = long;
            }
            let mut half = FITNESS_LANES / 2;
            while half >= 1 {
                for i in 0..half {
                    lanes[i] += lanes[i + half];
                }
                half /= 2;
            }
            warps[seg % FITNESS_WARPS] += lanes[0];
            seg += 1;
            seg_start = seg_end;
        }
        let mut half = FITNESS_WARPS / 2;
        while half >= 1 {
            for i in 0..half {
                warps[i] += warps[i + half];
            }
            half /= 2;
        }
        FitnessTotals {
            gross: warps[0],
            flips,
            exits,
            long_rows,
            nonfinite,
        }
    }

    /// Excess net over cash: `gross - side * (flips + exits)`.
    pub fn net_of(&self, t: &FitnessTotals) -> f64 {
        t.gross - self.side * (t.flips + t.exits) as f64
    }

    /// `(B - net) / B`; `+inf` for a non-finite prediction or fewer than
    /// `min_trades` flips.
    pub fn regret_of(&self, t: &FitnessTotals) -> f64 {
        if t.nonfinite || (t.flips as usize) < self.min_trades {
            return f64::INFINITY;
        }
        let r = (self.bound - self.net_of(t)) / self.bound;
        if r.is_finite() {
            r
        } else {
            f64::INFINITY
        }
    }

    pub fn regret(&self, pred: &[f64]) -> f64 {
        self.regret_of(&self.totals(pred))
    }
}

/// The finalist chosen by a trading objective, with both books.
#[derive(Clone, Debug)]
pub struct FinalistReport {
    pub objective: Objective,
    pub spec: TradingSpec,
    /// Smallest positive step between consecutive rows of one group over the
    /// validation and test slices.
    pub min_group_step: Option<f64>,
    pub horizon_seconds: f64,
    pub cost: crate::target::CostDecl,
    /// The finalist expression has no variable: every quantile or fixed
    /// threshold makes it a flat or always-long book.
    pub constant: bool,
    pub candidates: usize,
    pub usable: usize,
    pub expression: String,
    pub archive_rank: usize,
    pub fit_metric: String,
    pub fit_error: f64,
    pub validation_error: f64,
    pub test_error: f64,
    /// The objective on the validation slice (lower is better).
    pub validation_value: f64,
    pub validation: Result<PathStats, String>,
    /// #480: the value the search ranked candidates by, when it was not the metric.
    pub search_fitness: Option<&'static str>,
    pub test: Result<PathStats, String>,
}

impl FinalistReport {
    fn spec_json(&self) -> String {
        let mut s = self.spec.json();
        s.pop();
        s.push_str(&format!(
            ",\"min_group_step_seconds\":{},\"horizon_seconds\":{}}}",
            self.min_group_step.map_or_else(|| "null".to_string(), num),
            num(self.horizon_seconds)
        ));
        s
    }

    pub fn json(&self, backend: &str, backend_reason: Option<&str>) -> String {
        let block = |r: &Result<PathStats, String>| match r {
            Ok(s) => s.json(),
            Err(e) => format!("{{\"unusable\":{}}}", crate::devlog::text(e)),
        };
        format!(
            "{{\"name\":\"{}\",\"position\":\"long_only\",\"selection\":\"validation_objective\",\
             \"cost\":{},\"candidates\":{},\"usable_candidates\":{},\"exact\":\"cpu_f64\",\
             \"backend\":{},\"backend_reason\":{},\"spec\":{},\
             \"finalist\":{{\"expression\":{},\"archive_rank\":{},\"constant\":{},\"fit_metric\":{},\"fit_error\":{},\
             \"validation_error\":{},\"test_error\":{},\"validation_objective\":{}{}}},\
             \"validation\":{},\"test\":{}}}",
            self.objective.name(),
            self.cost.json(),
            self.candidates,
            self.usable,
            crate::devlog::text(backend),
            backend_reason
                .map(crate::devlog::text)
                .unwrap_or_else(|| "null".to_string()),
            self.spec_json(),
            crate::devlog::text(&self.expression),
            self.archive_rank,
            self.constant,
            crate::devlog::text(&self.fit_metric),
            num(self.fit_error),
            num(self.validation_error),
            num(self.test_error),
            num(self.validation_value),
            self.search_fitness
                .map(|k| format!(",\"search_fitness\":\"{}\"", k))
                .unwrap_or_default(),
            block(&self.validation),
            block(&self.test)
        )
    }

    /// One summary line for the text report.
    pub fn text(&self) -> String {
        let line = |name: &str, r: &Result<PathStats, String>| match r {
            Ok(s) => format!(
                "{}: net={:.6} sharpe={:.3} log_growth={:.6} exposure={:.3} trades={} break_even_bps={:.2} (cash net={:.6}, buy-and-hold net={:.6})",
                name,
                s.net_pnl,
                s.sharpe,
                s.log_growth,
                s.exposure,
                s.trades,
                s.break_even_bps,
                s.cash.net_pnl,
                s.buy_and_hold.net_pnl
            ),
            Err(e) => format!("{}: unusable ({})", name, e),
        };
        format!(
            "# objective {}: {} of {} candidates usable; {}; {}",
            self.objective.name(),
            self.usable,
            self.candidates,
            line("validation", &self.validation),
            line("test", &self.test)
        )
    }
}

#[cfg(test)]
mod spacing_tests {
    use super::*;

    fn rows(pairs: &[(u64, f64)]) -> TradingRows {
        TradingRows {
            time: pairs.iter().map(|p| p.1).collect(),
            group: pairs.iter().map(|p| p.0).collect(),
            funding: Vec::new(),
        }
    }

    #[test]
    fn step_equal_to_horizon_passes() {
        let r = rows(&[(1, 0.0), (1, 3600.0), (1, 7200.0)]);
        assert_eq!(check_return_spacing(&r, 3600.0), Ok(Some(3600.0)));
    }

    #[test]
    fn one_second_below_horizon_fails() {
        let r = rows(&[(1, 0.0), (1, 3600.0), (1, 7199.0)]);
        let e = check_return_spacing(&r, 3600.0).unwrap_err();
        assert!(e.starts_with("overlapping returns: group 1 has rows 1 and 2 3599 seconds apart"), "{e}");
    }

    #[test]
    fn single_row_groups_pass() {
        let r = rows(&[(1, 0.0), (2, 10.0), (3, 20.0)]);
        assert_eq!(check_return_spacing(&r, 3600.0), Ok(None));
        assert_eq!(min_group_step(&r), None);
    }

    #[test]
    fn groups_are_checked_separately() {
        let r = rows(&[(1, 0.0), (2, 10.0), (1, 3600.0), (2, 3610.0)]);
        assert_eq!(check_return_spacing(&r, 3600.0), Ok(Some(3600.0)));
        assert_eq!(min_group_step(&r), Some(3600.0));
    }

    #[test]
    fn equal_timestamps_and_nan_are_skipped() {
        let r = rows(&[(1, 0.0), (1, 0.0), (1, f64::NAN), (1, 3600.0)]);
        assert_eq!(check_return_spacing(&r, 3600.0), Ok(Some(3600.0)));
    }

    #[test]
    fn nonpositive_horizon_skips() {
        let r = rows(&[(1, 0.0), (1, 1.0)]);
        assert_eq!(check_return_spacing(&r, 0.0), Ok(Some(1.0)));
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    fn rows(group: &[u64], time: &[f64]) -> TradingRows {
        TradingRows { time: time.to_vec(), group: group.to_vec(), funding: Vec::new() }
    }

    #[test]
    fn is_canonical_needs_contiguous_groups_and_ordered_time() {
        assert!(rows(&[0, 0, 1, 1, 1, 2], &[0.0, 1.0, 0.0, 0.0, 5.0, 3.0]).is_canonical());
        assert!(!rows(&[0, 1, 0, 1], &[0.0, 0.0, 1.0, 1.0]).is_canonical());
        assert!(!rows(&[0, 0, 1], &[1.0, 0.0, 0.0]).is_canonical());
        assert!(rows(&[], &[]).is_canonical());
    }

    #[test]
    fn seam_flags_mark_one_first_and_one_last_per_group() {
        let r = rows(&[0, 0, 0, 1, 2, 2], &[0.0; 6]);
        let f = r.seam_flags();
        assert_eq!(f, vec![1, 0, 2, 3, 1, 2]);
        assert_eq!(f.iter().filter(|&&x| x & TradingRows::FIRST != 0).count(), 3);
        assert_eq!(f.iter().filter(|&&x| x & TradingRows::LAST != 0).count(), 3);
    }
}
