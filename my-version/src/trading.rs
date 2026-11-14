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
//! `PositionMap::TopTercile` replaces the threshold: per timestamp the top
//! `floor(n / 3)` finite predictions are long (`p = 1`), every other row is flat.
//! The rank uses only the predictions of that timestamp.
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

/// How a prediction becomes a position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PositionMap {
    /// `1[pred > threshold]`, or the `tau` ramp.
    #[default]
    Threshold,
    /// Per timestamp, the top `floor(n / 3)` finite predictions are long.
    TopTercile,
}

impl PositionMap {
    pub fn parse(s: &str) -> Result<PositionMap, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "long_only" | "threshold" => Ok(PositionMap::Threshold),
            "top_tercile" | "top-tercile" => Ok(PositionMap::TopTercile),
            other => Err(format!(
                "unknown --position '{}' (expected long_only or top_tercile; short is refused: the book is long-only)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PositionMap::Threshold => "long_only",
            PositionMap::TopTercile => "top_tercile",
        }
    }
}

/// #925: how `--band B` is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BandUnits {
    /// `B` is the half-width in score units.
    #[default]
    Score,
    /// `B` is a multiple of the candidate's training-score standard deviation.
    Sd,
}

impl BandUnits {
    pub fn parse(s: &str) -> Result<BandUnits, String> {
        match s.trim() {
            "score" => Ok(BandUnits::Score),
            "sd" => Ok(BandUnits::Sd),
            other => Err(format!("--band-units must be score or sd, got '{}'", other)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BandUnits::Score => "score",
            BandUnits::Sd => "sd",
        }
    }
}

/// #925: sample standard deviation (ddof 1) of the finite scores, and how many
/// entered it. `sd` is `None` for fewer than 2 finite scores, equal finite
/// scores (a constant score), or a zero or non-finite value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreSd {
    pub sd: Option<f64>,
    pub rows: usize,
}

/// #945: count, mean and centred sum of squares of a set of finite scores.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// The pairwise merge of Chan, Golub and LeVeque with the count-weighted
    /// mean; an empty side returns the other unchanged.
    fn merge(self, b: Moments) -> Moments {
        if b.n == 0 {
            return self;
        }
        if self.n == 0 {
            return b;
        }
        let (na, nb, nn) = (self.n as f64, b.n as f64, (self.n + b.n) as f64);
        let d = b.mean - self.mean;
        Moments { n: self.n + b.n, mean: (na * self.mean + nb * b.mean) / nn, m2: self.m2 + b.m2 + d * d * (na * nb / nn) }
    }
}

/// #945: one lane's finite scores of a segment as the sums of `x - k` and
/// `(x - k)^2`, with `k` its first score.
#[derive(Clone, Copy, Debug, Default)]
struct LaneSums {
    n: u64,
    k: f64,
    s: f64,
    q: f64,
}

impl LaneSums {
    fn push(&mut self, x: f64) {
        if self.n == 0 {
            self.k = x;
        }
        self.n += 1;
        let d = x - self.k;
        self.s += d;
        self.q += d * d;
    }

    /// The lane's [`Moments`]: mean `k + s / n`, centred sum of squares
    /// `q - s * (s / n)`, at least 0.
    fn moments(self) -> Moments {
        if self.n == 0 {
            return Moments::default();
        }
        let t = self.s / self.n as f64;
        let m2 = self.q - self.s * t;
        Moments { n: self.n, mean: self.k + t, m2: if m2 < 0.0 { 0.0 } else { m2 } }
    }
}

/// [`Moments`] of the finite `x` in the device tree: lane `(k - segment
/// start) % 32` of each [`FITNESS_SEGMENT_ROWS`]-row segment adds its rows in
/// ascending order ([`LaneSums`]), the lanes merge 16, 8, 4, 2, 1, segment `s`
/// merges into warp `s % 8` in ascending `s` and the warps merge 4, 2, 1: the
/// device order operation for operation.
fn tree_moments(x: &[f64]) -> Moments {
    let mut warps = [Moments::default(); FITNESS_WARPS];
    let mut seg_start = 0usize;
    let mut seg = 0usize;
    while seg_start < x.len() {
        let seg_end = (seg_start + FITNESS_SEGMENT_ROWS).min(x.len());
        let mut sums = [LaneSums::default(); FITNESS_LANES];
        for k in seg_start..seg_end {
            if x[k].is_finite() {
                sums[(k - seg_start) % FITNESS_LANES].push(x[k]);
            }
        }
        let mut lanes = sums.map(LaneSums::moments);
        let mut half = FITNESS_LANES / 2;
        while half >= 1 {
            for i in 0..half {
                lanes[i] = lanes[i].merge(lanes[i + half]);
            }
            half /= 2;
        }
        warps[seg % FITNESS_WARPS] = warps[seg % FITNESS_WARPS].merge(lanes[0]);
        seg += 1;
        seg_start = seg_end;
    }
    let mut half = FITNESS_WARPS / 2;
    while half >= 1 {
        for i in 0..half {
            warps[i] = warps[i].merge(warps[i + half]);
        }
        half /= 2;
    }
    warps[0]
}

/// #925, #945: the training-score standard deviation behind `--band-units
/// sd`, in one pass over the scores: [`tree_moments`] in the fixed device
/// order, so the device reduction equals it bit for bit on the same `f32`
/// predictions.
pub fn score_sd(scores: &[f64]) -> ScoreSd {
    let mut rows = 0usize;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in scores.iter().filter(|v| v.is_finite()) {
        rows += 1;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    // A constant score is degenerate even when rounding leaves a tiny centred
    // sum of squares.
    if rows < 2 || lo == hi {
        return ScoreSd { sd: None, rows };
    }
    let m = tree_moments(scores);
    let sd = (m.m2 / (rows - 1) as f64).sqrt();
    ScoreSd { sd: (sd.is_finite() && sd > 0.0).then_some(sd), rows }
}

/// #925: the no-trade band of the book. `resolved` is the half-width in score
/// units that the book prices; with [`BandUnits::Sd`] it is `multiple` times
/// the standard deviation of the candidate's scores on its training rows, and
/// a book on other rows takes it as given.
#[derive(Clone, Debug, PartialEq)]
pub struct BandSpec {
    pub units: BandUnits,
    /// `--band B`.
    pub multiple: f64,
    /// Training-score sd; `None` in score units or when degenerate.
    pub sd: Option<f64>,
    /// Finite training scores in the sd (sd units only).
    pub sd_rows: Option<usize>,
    /// Latest time of a row in the sd (sd units only).
    pub sd_max_time: Option<f64>,
    pub resolved: f64,
}

impl Default for BandSpec {
    fn default() -> Self {
        BandSpec::score(0.0)
    }
}

impl BandSpec {
    /// A band in score units: the multiple is the half-width.
    pub fn score(multiple: f64) -> BandSpec {
        BandSpec { units: BandUnits::Score, multiple, sd: None, sd_rows: None, sd_max_time: None, resolved: multiple }
    }

    /// Resolve the band from one candidate's scores on its training rows
    /// (`time` aligned with `scores`). A degenerate sd resolves to 0.
    pub fn from_training(units: BandUnits, multiple: f64, scores: &[f64], time: &[f64]) -> BandSpec {
        if units == BandUnits::Score {
            return BandSpec::score(multiple);
        }
        let max_time = scores
            .iter()
            .zip(time)
            .filter(|(s, t)| s.is_finite() && t.is_finite())
            .map(|(_, &t)| t)
            .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))));
        BandSpec::from_sd(units, multiple, score_sd(scores), max_time)
    }

    /// The band from a computed training-score sd and the latest time of a
    /// row in it.
    pub fn from_sd(units: BandUnits, multiple: f64, sd: ScoreSd, max_time: Option<f64>) -> BandSpec {
        if units == BandUnits::Score {
            return BandSpec::score(multiple);
        }
        BandSpec {
            units,
            multiple,
            sd: sd.sd,
            sd_rows: Some(sd.rows),
            sd_max_time: max_time,
            resolved: sd.sd.map_or(0.0, |s| multiple * s),
        }
    }

    /// The JSON members, each with a leading comma.
    pub fn json_fields(&self) -> String {
        let sd_units = self.units == BandUnits::Sd;
        format!(
            ",\"band\":{},\"band_units\":\"{}\",\"band_sd\":{},\"band_resolved\":{},\"band_sd_rows\":{},\
             \"band_sd_max_time\":{},\"band_sd_degenerate\":{}",
            num(self.multiple),
            self.units.name(),
            self.sd.map_or_else(|| "null".to_string(), num),
            num(self.resolved),
            self.sd_rows.map_or_else(|| "null".to_string(), |r| r.to_string()),
            self.sd_max_time.map_or_else(|| "null".to_string(), num),
            if sd_units { if self.sd.is_none() { "true" } else { "false" } } else { "null" }
        )
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
    /// #925: no-trade band of the indicator rule: enter above
    /// `threshold + band.resolved`, exit at or below `threshold - band.resolved`.
    pub band: BandSpec,
    pub threshold: f64,
    /// Sizing scale; `None` is the `{0, 1}` indicator.
    pub tau: Option<f64>,
    pub max_position: f64,
    pub position: PositionMap,
    /// Override of the annualisation. `None` derives it from `period_seconds`,
    /// else from the median step of the period stamps.
    pub periods_per_year: Option<f64>,
    /// Period width: the label horizon, equal to the label stride. Rows with the
    /// same `floor(t / period_seconds)` form one equal-weight period.
    pub period_seconds: Option<f64>,
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
            band: BandSpec::default(),
            threshold: 0.0,
            tau: None,
            max_position: 1.0,
            position: PositionMap::Threshold,
            periods_per_year: None,
            period_seconds: None,
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
        if self.position == PositionMap::TopTercile && (self.tau.is_some() || self.min_holding > 1 || self.max_position != 1.0) {
            return Err("the top_tercile book is the unit {0, 1} position: it needs no tau, min_holding 1 and max_position 1".to_string());
        }
        for (name, v) in [("band", self.band.multiple), ("resolved band", self.band.resolved)] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("the {} must be finite and non-negative, got {}", name, v));
            }
        }
        if self.band.multiple > 0.0 {
            if self.position == PositionMap::TopTercile {
                return Err("--band is not supported with --position top_tercile: the tercile book has no threshold".to_string());
            }
            if self.tau.is_some() {
                return Err("the band needs the indicator position (no tau)".to_string());
            }
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
            "{{\"position\":\"{}\",\"threshold\":{},\"tau\":{},\"max_position\":{},\
             \"cost_bps\":{},\"fee_bps\":{},\"half_spread_bps\":{},\"slippage_bps\":{},\
             \"per_side_bps\":{},\"funding_col\":{},\"cash_return\":{},\"min_trades\":{},\
             \"min_holding\":{}{}}}",
            self.position.name(),
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
            self.min_holding,
            self.band.json_fields()
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
    /// Always `per_period`: one equal-weight return per period of `period_seconds`.
    pub annualisation: &'static str,
    /// Sum over rows of `gross_t`.
    pub gross_pnl: f64,
    /// Sum over rows of `net_t`, including every exit.
    pub net_pnl: f64,
    /// `net_pnl / rows * periods_per_year`: the mean net return per row, annualised.
    pub net_per_row_annualised: f64,
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
    /// Per-side cost at which `net_pnl` is zero. NaN when `gross_pnl <= 0`.
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
             \"gross_pnl\":{},\"net_pnl\":{},\"net_per_row_annualised\":{},\"total_cost\":{},\"funding_paid\":{},\"total_return\":{},\
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
            num(self.net_per_row_annualised),
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
    /// Period bucket width in seconds; `0` is one period per timestamp.
    bucket: f64,
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
        bucket: f64,
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
            bucket,
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
    let mut cross = CrossSection::with_bucket(ctx.bucket);
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
    let (returns, gross_returns) = cross.series();
    let period_ppy = match (spec.periods_per_year, spec.period_seconds) {
        (Some(v), _) => v,
        (None, Some(s)) if s >= 1.0 && s <= SECONDS_PER_YEAR => SECONDS_PER_YEAR / s,
        _ => {
            let times: Vec<f64> = cross.periods.iter().map(|p| p.time).collect();
            match median_step(&times) {
                Some(dt) if dt >= 1.0 && dt <= SECONDS_PER_YEAR => SECONDS_PER_YEAR / dt,
                _ => 365.0,
            }
        }
    };
    Series {
        risk: risk::risk_stats(&returns, period_ppy, risk::DEFAULT_SORTINO_MAR),
        annualisation: "per_period",
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
    let break_even_bps = if book.abs_delta_with_exit > 0.0 && gross_pnl > 0.0 {
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
        net_per_row_annualised: main.net_pnl / n as f64 * series.period_ppy,
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

/// Top-tercile positions: per timestamp, the highest `floor(n / 3)` of the `n`
/// finite predictions get `1.0`, ties by row order; every other row gets `0.0`.
/// A timestamp with fewer than 3 finite predictions holds nothing. Only the
/// predictions of one timestamp enter its ranks.
pub fn top_tercile_positions(pred: &[f64], time: &[f64]) -> Vec<f64> {
    let mut pos = vec![0.0f64; pred.len()];
    let mut order: Vec<usize> = (0..pred.len().min(time.len()))
        .filter(|&i| pred[i].is_finite() && time[i].is_finite())
        .collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| pred[b].total_cmp(&pred[a])).then_with(|| a.cmp(&b)));
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && time[order[end]] == time[order[start]] {
            end += 1;
        }
        for &k in order[start..end].iter().take((end - start) / 3) {
            pos[k] = 1.0;
        }
        start = end;
    }
    pos
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
    let ctx = Ctx::new(&keep, r, rows, spec.cash_return, spec.period_seconds.unwrap_or(0.0));
    let positions = positions_of(pred, rows, &keep, &ctx, spec);
    finish(&ctx, &positions, spec)
}

/// Position of every usable row, in `ctx` row order.
fn positions_of(pred: &[f64], rows: &TradingRows, keep: &[usize], ctx: &Ctx, spec: &TradingSpec) -> Vec<f64> {
    let scores: Vec<f64> = keep.iter().map(|&i| pred[i]).collect();
    if spec.position == PositionMap::TopTercile {
        let all = top_tercile_positions(pred, &rows.time);
        keep.iter().map(|&i| all[i]).collect()
    } else if spec.min_holding > 1 || spec.band.resolved > 0.0 {
        // #925: the band automaton of `walkforward --policy band`; with a zero
        // band it is the minimum-hold rule.
        let params = trading_gpu::BandParams {
            threshold: spec.threshold,
            band: spec.band.resolved,
            min_hold: spec.min_holding,
        };
        let mut positions = vec![0.0f64; keep.len()];
        let mut current: Option<u64> = None;
        let mut state = trading_gpu::BandState::default();
        for &i in &ctx.order {
            if current != Some(ctx.group[i]) {
                current = Some(ctx.group[i]);
                state = trading_gpu::BandState::default();
            }
            state = params.next(state, scores[i]);
            positions[i] = if state.long { spec.max_position } else { 0.0 };
        }
        positions
    } else {
        scores.iter().map(|&s| position_of(s, spec)).collect()
    }
}

/// Per-block alpha of one prediction vector against exposure-matched
/// buy-and-hold (#928).
#[derive(Clone, Debug, PartialEq)]
pub struct BlockedAlpha {
    /// `net_k - exposure_k * bh_k` per block, in net excess per period.
    pub block_alpha: Vec<f64>,
    /// Mean of `series`.
    pub pooled_alpha: f64,
    /// Alpha of every period bucket (every distinct timestamp without `period_seconds`), in time order.
    pub series: Vec<f64>,
    /// Position changes of the book.
    pub trades: usize,
    /// Rows held long.
    pub long_rows: usize,
    /// Entries into a long position.
    pub entries: usize,
}

/// Cut the usable rows into `blocks` runs of distinct timestamps and price the
/// book against buy-and-hold of the same rows.
///
/// Per period the row means are taken, so every period weighs the same. A
/// period is one timestamp, or one `floor(t / period_seconds)` bucket when the
/// spec sets `period_seconds`; a bucket never straddles two blocks. In
/// block `k`, `net_k` is the mean book net excess over the flat book,
/// `exposure_k` the mean position and `bh_k` the mean buy-and-hold net excess
/// (entry and exit costs included). A block without exposure has alpha 0.
pub fn blocked_alpha(
    pred: &[f64],
    r: &[f64],
    rows: &TradingRows,
    spec: &TradingSpec,
    blocks: usize,
) -> Result<BlockedAlpha, String> {
    spec.validate()?;
    check_lengths(pred.len(), r, rows)?;
    let keep: Vec<usize> = (0..pred.len())
        .filter(|&i| pred[i].is_finite() && usable(i, r, rows))
        .collect();
    if keep.is_empty() {
        return Err("no usable rows (every prediction or return is non-finite)".to_string());
    }
    let ctx = Ctx::new(&keep, r, rows, spec.cash_return, spec.period_seconds.unwrap_or(0.0));
    let positions = positions_of(pred, rows, &keep, &ctx, spec);
    let n = ctx.ret.len();
    let cost_side = spec.per_side_bps() * BP;
    let book = run_book(&ctx, &positions, cost_side);
    let flat = run_book(&ctx, &vec![0.0; n], cost_side);
    let held = run_book(&ctx, &vec![1.0; n], cost_side);
    let keys: Vec<f64> = if ctx.bucket > 0.0 { ctx.time.iter().map(|t| (t / ctx.bucket).floor()).collect() } else { ctx.time.clone() };
    let ids = block_ids(&keys, blocks)?;
    let mut series_net: Vec<f64> = Vec::new();
    let mut series_pos: Vec<f64> = Vec::new();
    let mut series_bh: Vec<f64> = Vec::new();
    let mut series_block: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let t = keys[ctx.time_order[i]];
        let (mut net, mut pos, mut bh, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        let block = ids[ctx.time_order[i]] as usize;
        while i < n && keys[ctx.time_order[i]] == t {
            let row = ctx.time_order[i];
            net += book.net[row] - flat.net[row];
            pos += positions[row];
            bh += held.net[row] - flat.net[row];
            count += 1;
            i += 1;
        }
        let c = count as f64;
        series_net.push(net / c);
        series_pos.push(pos / c);
        series_bh.push(bh / c);
        series_block.push(block);
    }
    let mut block_alpha = vec![0.0f64; blocks];
    let mut exposure = vec![0.0f64; blocks];
    for b in 0..blocks {
        let idx: Vec<usize> = (0..series_block.len()).filter(|&k| series_block[k] == b).collect();
        let m = idx.len() as f64;
        let e = idx.iter().map(|&k| series_pos[k]).sum::<f64>() / m;
        exposure[b] = e;
        if e > 0.0 {
            let net = idx.iter().map(|&k| series_net[k]).sum::<f64>() / m;
            let bh = idx.iter().map(|&k| series_bh[k]).sum::<f64>() / m;
            block_alpha[b] = net - e * bh;
        }
    }
    let series: Vec<f64> = (0..series_block.len())
        .map(|k| {
            let e = exposure[series_block[k]];
            if e > 0.0 {
                series_net[k] - e * series_bh[k]
            } else {
                0.0
            }
        })
        .collect();
    let pooled_alpha = series.iter().sum::<f64>() / series.len() as f64;
    Ok(BlockedAlpha {
        block_alpha,
        pooled_alpha,
        series,
        trades: book.stats.trades,
        long_rows: book.stats.active,
        entries: book.stats.entries,
    })
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
    let ctx = Ctx::new(&keep, r, rows, spec.cash_return, spec.period_seconds.unwrap_or(0.0));
    let kept: Vec<f64> = keep.iter().map(|&i| positions[i]).collect();
    finish(&ctx, &kept, spec)
}

/// True when the expression reads no variable.
pub fn is_constant(tree: &crate::expr::Node) -> bool {
    tree.op != crate::expr::Op::Input && tree.kids.iter().all(is_constant)
}

/// #849: the one position the tree takes on every input under the book's rule
/// (`Some(true)` long, `Some(false)` flat), or `None` when the inputs decide.
/// Only the indicator rule `1[pred > threshold]` has a fixed position: a tau
/// ramp sizes by the score and the tercile rule ranks it.
pub fn fixed_position(tree: &crate::expr::Node, spec: &TradingSpec) -> Option<bool> {
    if spec.position != PositionMap::Threshold || spec.tau.is_some() {
        return None;
    }
    crate::expr::fixed_position(tree, spec.threshold)
}

/// #849: the position the validation book realised on every evaluated row
/// (`Some(true)` long on all rows, `Some(false)` on none), at any threshold. It
/// reports the rows, where [`fixed_position`] reports the formula: a range that
/// crosses a tiny threshold hides an always-long tree. Only the indicator rule
/// has one position per row.
pub fn realised_position(exposure: f64, spec: &TradingSpec) -> Option<bool> {
    if spec.position != PositionMap::Threshold || spec.tau.is_some() {
        return None;
    }
    if exposure == 1.0 {
        Some(true)
    } else if exposure == 0.0 {
        Some(false)
    } else {
        None
    }
}

/// The report text of [`fixed_position`]: `"long"`, `"flat"` or `null`.
pub fn fixed_position_json(fixed: Option<bool>) -> &'static str {
    match fixed {
        Some(true) => "\"long\"",
        Some(false) => "\"flat\"",
        None => "null",
    }
}

/// Rows per canonical summation segment of the fitness gross sum.
pub const FITNESS_SEGMENT_ROWS: usize = trading_gpu::DEFAULT_SEGMENT_ROWS;
/// Lanes inside one segment.
pub const FITNESS_LANES: usize = trading_gpu::PNL_LANES;
/// Warp sums that the segments are spread over.
pub const FITNESS_WARPS: usize = 8;
/// #945: the last word of an sd-unit book's hash, naming the one-pass
/// training-score sd of [`score_sd`]. The two-pass sd of #925 added no word,
/// so its checkpoints refuse to resume.
pub const SCORE_SD_ALGORITHM: u64 = 2;
/// Most time blocks of the consistency term (`--pnl-blocks`).
pub const FITNESS_MAX_BLOCKS: usize = 8;
/// #925: most minimum-hold rows the device band scorer carries (`K + 1`
/// automaton states); a longer hold scores on the CPU.
pub const FITNESS_DEVICE_MAX_HOLD: usize = 32;
/// The block id of a row sits in bits 2 to 4 of its [`FitnessBook::flags`] byte.
pub const FITNESS_BLOCK_SHIFT: u8 = 2;

/// Time block of every row: the sorted distinct timestamps are cut into
/// `blocks` runs of equal count, the last run takes the remainder. Rows with
/// equal timestamps share a block.
pub fn block_ids(time: &[f64], blocks: usize) -> Result<Vec<u8>, String> {
    let mut distinct: Vec<f64> = time.to_vec();
    distinct.sort_by(|a, b| a.total_cmp(b));
    distinct.dedup_by(|a, b| a == b);
    if blocks == 0 || blocks > FITNESS_MAX_BLOCKS {
        return Err(format!("the block count must be 1 to {}, got {}", FITNESS_MAX_BLOCKS, blocks));
    }
    if distinct.len() < blocks {
        return Err(format!(
            "{} time blocks need at least {} distinct training timestamps, found {}",
            blocks,
            blocks,
            distinct.len()
        ));
    }
    let per = distinct.len() / blocks;
    Ok(time
        .iter()
        .map(|t| {
            let at = distinct.partition_point(|d| d.total_cmp(t).is_lt());
            (at / per).min(blocks - 1) as u8
        })
        .collect())
}

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

/// #1084: the counts of one long/flat book that the book descriptor of the
/// quality-diversity archive reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BookShape {
    pub rows: u64,
    pub long_rows: u64,
    /// Round trips: every run of the book starts flat, so each entry is closed
    /// by one flip or by the exit at the run's last row, and
    /// `flips + exits = 2 * entries`.
    pub entries: u64,
}

impl BookShape {
    /// The shape of a book of `rows` rows from its totals (one per time block).
    pub fn of(rows: usize, totals: &[FitnessTotals]) -> BookShape {
        let long_rows = totals.iter().map(|t| t.long_rows).sum();
        let turns: u64 = totals.iter().map(|t| t.flips + t.exits).sum();
        BookShape { rows: rows as u64, long_rows, entries: turns / 2 }
    }

    /// Long rows over rows; 0 for an empty book.
    pub fn exposure(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.long_rows as f64 / self.rows as f64
        }
    }

    /// Long rows per entry: `long_rows / max(1, entries)`; 0 for a flat book.
    pub fn mean_holding(&self) -> f64 {
        self.long_rows as f64 / self.entries.max(1) as f64
    }
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
    lambda: f64,
    block_bounds: Vec<f64>,
    /// #925: `--band B` and its units, and the minimum hold.
    band: f64,
    band_units: BandUnits,
    min_hold: usize,
    /// #1023: the sub-books of the lexicase cases, built on first use.
    cases: std::sync::OnceLock<BookCases>,
}

/// #1023: the `(group, time block)` sub-books of a [`FitnessBook`]: one per run
/// of its seam flags, in row order. Canonical rows keep each group's rows in
/// time order, so one group's rows inside one block are one run.
#[derive(Clone, Debug, PartialEq)]
pub struct BookCases {
    /// Row range `[start, end)` of each run.
    pub ranges: Vec<(usize, usize)>,
    /// Hindsight bound of each run's rows.
    pub bounds: Vec<f64>,
    /// The runs with a positive bound: the lexicase cases, in run order.
    pub kept: Vec<usize>,
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
        Self::with_blocks(slice, spec, horizon_seconds, 1, 1.0)
    }

    /// The book with the time-block consistency term (#484): the distinct
    /// training timestamps are cut into `blocks` runs of equal count (the last
    /// takes the remainder), every `(group, block)` run is its own book, and
    /// the fitness is the mean block regret plus `lambda` times their sample
    /// standard deviation. `blocks == 1` is [`FitnessBook::new`] exactly.
    pub fn with_blocks(
        slice: &crate::data::TradingSlice,
        spec: &TradingSpec,
        horizon_seconds: f64,
        blocks: usize,
        lambda: f64,
    ) -> Result<Self, String> {
        if blocks == 0 || blocks > FITNESS_MAX_BLOCKS {
            return Err(format!("the block count must be 1 to {}, got {}", FITNESS_MAX_BLOCKS, blocks));
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!("the block lambda must be finite and >= 0, got {}", lambda));
        }
        spec.validate()?;
        if spec.position != PositionMap::Threshold {
            return Err("the fitness book prices the threshold rule only, not the top_tercile map".to_string());
        }
        if spec.tau.is_some() || spec.max_position != 1.0 {
            return Err(
                "the fitness book is the {0, 1} indicator: it needs max_position 1 and no tau".to_string(),
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
        let mut flags = rows.seam_flags();
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
        let mut block_bounds = vec![bound];
        if blocks > 1 {
            let ids = block_ids(&rows.time, blocks)?;
            block_bounds.clear();
            for b in 0..blocks {
                let mut e = Vec::new();
                let mut g = Vec::new();
                for i in 0..n {
                    if ids[i] as usize == b {
                        e.push(excess[i]);
                        g.push(rows.group[i]);
                    }
                }
                let bb = trading_gpu::dp_bound(&e, &g, side)?;
                if !(bb > 0.0) {
                    return Err(format!(
                        "no profitable long/flat path exists in time block {} of {} at {} bp per side (hindsight bound {})",
                        b + 1,
                        blocks,
                        spec.per_side_bps(),
                        bb
                    ));
                }
                block_bounds.push(bb);
            }
            for i in 0..n {
                let first = flags[i] & TradingRows::FIRST != 0 || (i > 0 && ids[i] != ids[i - 1]);
                let last = flags[i] & TradingRows::LAST != 0 || (i + 1 < n && ids[i] != ids[i + 1]);
                flags[i] = (if first { TradingRows::FIRST } else { 0 })
                    | (if last { TradingRows::LAST } else { 0 })
                    | (ids[i] << FITNESS_BLOCK_SHIFT);
            }
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
        if blocks > 1 {
            fnv_u64(&mut hash, blocks as u64);
            fnv_u64(&mut hash, lambda.to_bits());
            for b in &block_bounds {
                fnv_u64(&mut hash, b.to_bits());
            }
        }
        let band = spec.band.multiple;
        if band != 0.0 || spec.min_holding != 1 || spec.band.units != BandUnits::Score {
            fnv_u64(&mut hash, band.to_bits());
            fnv_u64(&mut hash, if spec.band.units == BandUnits::Sd { 1 } else { 0 });
            fnv_u64(&mut hash, spec.min_holding as u64);
            if spec.band.units == BandUnits::Sd {
                fnv_u64(&mut hash, SCORE_SD_ALGORITHM);
            }
        }
        Ok(FitnessBook {
            excess,
            flags,
            groups,
            side,
            threshold: spec.threshold,
            bound,
            min_trades: spec.min_trades,
            hash,
            lambda,
            block_bounds,
            band,
            band_units: spec.band.units,
            min_hold: spec.min_holding,
            cases: std::sync::OnceLock::new(),
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

    /// Excess return per row (return minus funding minus cash), the gross term.
    pub fn excess(&self) -> &[f64] {
        &self.excess
    }

    /// Seam flags per row: [`TradingRows::FIRST`] and [`TradingRows::LAST`]. With
    /// several blocks they mark the seams of every `(group, block)` run, and
    /// bits 2 to 4 carry the block id ([`FITNESS_BLOCK_SHIFT`]).
    pub fn flags(&self) -> &[u8] {
        &self.flags
    }

    fn check_len(&self, pred: &[f64]) {
        assert_eq!(pred.len(), self.excess.len(), "fitness book: {} predictions for {} rows", pred.len(), self.excess.len());
    }

    /// Reduce one prediction vector as a single book (a multi-block book uses
    /// [`FitnessBook::block_totals`]). `gross` uses the fixed segment, lane and
    /// warp tree documented on [`FITNESS_SEGMENT_ROWS`]; integer counts are
    /// order-free.
    pub fn totals(&self, pred: &[f64]) -> FitnessTotals {
        self.check_len(pred);
        if !self.is_plain() {
            return self.band_pass(pred, self.resolve_band(pred), false)[0];
        }
        let n = self.excess.len();
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
        self.regret_and_shape(pred).0
    }

    /// #1084: [`FitnessBook::regret`] and the [`BookShape`] of the same
    /// reduction, so the shape follows the position rule the regret prices.
    pub fn regret_and_shape(&self, pred: &[f64]) -> (f64, BookShape) {
        if self.blocks() == 1 {
            let t = self.totals(pred);
            (self.regret_of(&t), BookShape::of(self.len(), std::slice::from_ref(&t)))
        } else {
            let t = self.block_totals(pred);
            (self.fitness_of(&t), BookShape::of(self.len(), &t))
        }
    }

    /// Time blocks of the consistency term; 1 is the plain regret.
    pub fn blocks(&self) -> usize {
        self.block_bounds.len()
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Hindsight bound of each block (one entry, [`FitnessBook::bound`], for a
    /// single block).
    pub fn block_bounds(&self) -> &[f64] {
        &self.block_bounds
    }

    /// Reduce one prediction vector to one [`FitnessTotals`] per time block.
    /// Each block keeps its own segment, lane and warp tree: block `b` adds
    /// the long rows of block `b` into lane `(row - segment start) % 32`, folds
    /// the lanes of every segment, and spreads the segment sums over the warps
    /// exactly as [`FitnessBook::totals`] does for the whole book.
    pub fn block_totals(&self, pred: &[f64]) -> Vec<FitnessTotals> {
        self.check_len(pred);
        if !self.is_plain() {
            return self.band_pass(pred, self.resolve_band(pred), true);
        }
        let nb = self.blocks();
        let n = self.excess.len();
        let mut warps = vec![[0.0f64; FITNESS_WARPS]; nb];
        let mut out = vec![
            FitnessTotals { gross: 0.0, flips: 0, exits: 0, long_rows: 0, nonfinite: false };
            nb
        ];
        let mut prev = false;
        let mut seg_start = 0usize;
        let mut seg = 0usize;
        while seg_start < n {
            let seg_end = (seg_start + FITNESS_SEGMENT_ROWS).min(n);
            let mut lanes = vec![[0.0f64; FITNESS_LANES]; nb];
            for k in seg_start..seg_end {
                let p = pred[k];
                let flag = self.flags[k];
                let b = (flag >> FITNESS_BLOCK_SHIFT) as usize & (FITNESS_MAX_BLOCKS - 1);
                let long = if p.is_finite() {
                    p > self.threshold
                } else {
                    out[b].nonfinite = true;
                    false
                };
                if flag & TradingRows::FIRST != 0 {
                    prev = false;
                }
                if long != prev {
                    out[b].flips += 1;
                }
                if long {
                    out[b].long_rows += 1;
                    lanes[b][(k - seg_start) % FITNESS_LANES] += self.excess[k];
                    if flag & TradingRows::LAST != 0 {
                        out[b].exits += 1;
                    }
                }
                prev = long;
            }
            for b in 0..nb {
                let l = &mut lanes[b];
                let mut half = FITNESS_LANES / 2;
                while half >= 1 {
                    for i in 0..half {
                        l[i] += l[i + half];
                    }
                    half /= 2;
                }
                warps[b][seg % FITNESS_WARPS] += l[0];
            }
            seg += 1;
            seg_start = seg_end;
        }
        for b in 0..nb {
            let w = &mut warps[b];
            let mut half = FITNESS_WARPS / 2;
            while half >= 1 {
                for i in 0..half {
                    w[i] += w[i + half];
                }
                half /= 2;
            }
            out[b].gross = w[0];
        }
        out
    }

    /// Regret of block `b` from its totals; `+inf` when it is not finite.
    pub fn block_regret_of(&self, b: usize, t: &FitnessTotals) -> f64 {
        let bound = self.block_bounds[b];
        let r = (bound - self.net_of(t)) / bound;
        if r.is_finite() {
            r
        } else {
            f64::INFINITY
        }
    }

    /// Regret of every block.
    pub fn block_regrets(&self, totals: &[FitnessTotals]) -> Vec<f64> {
        totals.iter().enumerate().map(|(b, t)| self.block_regret_of(b, t)).collect()
    }

    /// The search fitness from per-block totals: `mean + lambda * sd` of the
    /// block regrets (sample standard deviation, ddof 1); with one block the
    /// plain regret. `+inf` for a non-finite prediction, fewer than
    /// `min_trades` flips over all blocks, or a non-finite block regret.
    pub fn fitness_of(&self, totals: &[FitnessTotals]) -> f64 {
        let nb = self.blocks();
        debug_assert_eq!(totals.len(), nb);
        if nb == 1 {
            return self.regret_of(&totals[0]);
        }
        let flips: u64 = totals.iter().map(|t| t.flips).sum();
        if totals.iter().any(|t| t.nonfinite) || (flips as usize) < self.min_trades {
            return f64::INFINITY;
        }
        let r = self.block_regrets(totals);
        if r.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        let mut sum = 0.0f64;
        for v in &r {
            sum += *v;
        }
        let mean = sum / nb as f64;
        let mut ss = 0.0f64;
        for v in &r {
            let d = *v - mean;
            ss += d * d;
        }
        let sd = (ss / (nb - 1) as f64).sqrt();
        let f = mean + self.lambda * sd;
        if f.is_finite() {
            f
        } else {
            f64::INFINITY
        }
    }

    /// #925: `--band B` (score units, or a multiple of the training-score sd).
    pub fn band(&self) -> f64 {
        self.band
    }

    pub fn band_units(&self) -> BandUnits {
        self.band_units
    }

    /// #925: minimum consecutive long rows, entry row included.
    pub fn min_hold(&self) -> usize {
        self.min_hold
    }

    /// The plain threshold rule: no band and a minimum hold of 1.
    pub fn is_plain(&self) -> bool {
        self.band == 0.0 && self.min_hold == 1
    }

    /// #925: the half-width in score units this prediction vector is priced
    /// with: `B`, or `B` times the sd of its finite predictions on these
    /// training rows ([`score_sd`]; 0 when degenerate).
    pub fn resolve_band(&self, pred: &[f64]) -> f64 {
        match self.band_units {
            BandUnits::Score => self.band,
            BandUnits::Sd => score_sd(pred).sd.map_or(0.0, |s| self.band * s),
        }
    }

    /// #925: the band automaton ([`trading_gpu::BandParams::next`]) over the
    /// canonical rows at half-width `band`, the state reset to flat at every
    /// run start of [`FitnessBook::flags`]. Per block when `per_block`, else
    /// one book; each keeps the segment, lane and warp tree of
    /// [`FitnessBook::totals`] for `gross`.
    pub fn band_pass(&self, pred: &[f64], band: f64, per_block: bool) -> Vec<FitnessTotals> {
        self.check_len(pred);
        let nb = if per_block { self.blocks() } else { 1 };
        let n = self.excess.len();
        let params = trading_gpu::BandParams { threshold: self.threshold, band, min_hold: self.min_hold };
        let mut warps = vec![[0.0f64; FITNESS_WARPS]; nb];
        let mut out = vec![FitnessTotals { gross: 0.0, flips: 0, exits: 0, long_rows: 0, nonfinite: false }; nb];
        let mut state = trading_gpu::BandState::default();
        let mut prev = false;
        let mut seg_start = 0usize;
        let mut seg = 0usize;
        while seg_start < n {
            let seg_end = (seg_start + FITNESS_SEGMENT_ROWS).min(n);
            let mut lanes = vec![[0.0f64; FITNESS_LANES]; nb];
            for k in seg_start..seg_end {
                let p = pred[k];
                let flag = self.flags[k];
                let b = if per_block { (flag >> FITNESS_BLOCK_SHIFT) as usize & (FITNESS_MAX_BLOCKS - 1) } else { 0 };
                if !p.is_finite() {
                    out[b].nonfinite = true;
                }
                if flag & TradingRows::FIRST != 0 {
                    state = trading_gpu::BandState::default();
                    prev = false;
                }
                state = params.next(state, p);
                let long = state.long;
                if long != prev {
                    out[b].flips += 1;
                }
                if long {
                    out[b].long_rows += 1;
                    lanes[b][(k - seg_start) % FITNESS_LANES] += self.excess[k];
                    if flag & TradingRows::LAST != 0 {
                        out[b].exits += 1;
                    }
                }
                prev = long;
            }
            for b in 0..nb {
                let l = &mut lanes[b];
                let mut half = FITNESS_LANES / 2;
                while half >= 1 {
                    for i in 0..half {
                        l[i] += l[i + half];
                    }
                    half /= 2;
                }
                warps[b][seg % FITNESS_WARPS] += l[0];
            }
            seg += 1;
            seg_start = seg_end;
        }
        for b in 0..nb {
            let w = &mut warps[b];
            let mut half = FITNESS_WARPS / 2;
            while half >= 1 {
                for i in 0..half {
                    w[i] += w[i + half];
                }
                half /= 2;
            }
            out[b].gross = w[0];
        }
        out
    }

    /// #1023: the sub-books of the lexicase cases.
    pub fn cases(&self) -> &BookCases {
        self.cases.get_or_init(|| {
            let n = self.excess.len();
            let mut ranges = Vec::new();
            let mut start = 0usize;
            for i in 1..n {
                if self.flags[i] & TradingRows::FIRST != 0 {
                    ranges.push((start, i));
                    start = i;
                }
            }
            if n > 0 {
                ranges.push((start, n));
            }
            let bounds: Vec<f64> = ranges
                .iter()
                .map(|&(s, e)| trading_gpu::dp_bound(&self.excess[s..e], &vec![0; e - s], self.side).unwrap_or(f64::NAN))
                .collect();
            let kept = (0..ranges.len()).filter(|&c| bounds[c] > 0.0).collect();
            BookCases { ranges, bounds, kept }
        })
    }

    /// #1023: the totals of every run of [`FitnessBook::cases`] under the
    /// position rule of the whole book (its band, resolved on all of `pred`, and
    /// its minimum hold). Each run keeps the segment, lane and warp tree of
    /// [`FitnessBook::totals`] from its own first row, so a run of a plain book
    /// or of a score-unit band totals as a separate book of its rows.
    pub fn case_totals(&self, pred: &[f64]) -> Vec<FitnessTotals> {
        self.check_len(pred);
        let band = (!self.is_plain()).then(|| self.resolve_band(pred));
        self.cases().ranges.iter().map(|&(s, e)| self.run_totals(pred, s, e, band)).collect()
    }

    /// #1023: the regret `(B_c - net_c) / B_c` of every kept case, from the same
    /// reduction as [`FitnessBook::case_totals`]; `+inf` for a non-finite
    /// prediction in the case or a non-finite regret. A case has no trade
    /// floor: a flat case scores exactly 1.
    pub fn case_errors(&self, pred: &[f64]) -> Vec<f32> {
        self.check_len(pred);
        let band = (!self.is_plain()).then(|| self.resolve_band(pred));
        let cases = self.cases();
        cases
            .kept
            .iter()
            .map(|&c| {
                let (s, e) = cases.ranges[c];
                let t = self.run_totals(pred, s, e, band);
                let r = (cases.bounds[c] - self.net_of(&t)) / cases.bounds[c];
                if t.nonfinite || !r.is_finite() {
                    f32::INFINITY
                } else {
                    r as f32
                }
            })
            .collect()
    }

    /// #1023: one run `[start, end)` as one book; `band` is `None` for the
    /// plain rule, else the band automaton's half-width.
    fn run_totals(&self, pred: &[f64], start: usize, end: usize, band: Option<f64>) -> FitnessTotals {
        let params = band.map(|band| trading_gpu::BandParams { threshold: self.threshold, band, min_hold: self.min_hold });
        let mut state = trading_gpu::BandState::default();
        let mut t = FitnessTotals { gross: 0.0, flips: 0, exits: 0, long_rows: 0, nonfinite: false };
        let mut warps = [0.0f64; FITNESS_WARPS];
        let mut prev = false;
        let mut seg_start = start;
        let mut seg = 0usize;
        while seg_start < end {
            let seg_end = ((seg_start / FITNESS_SEGMENT_ROWS + 1) * FITNESS_SEGMENT_ROWS).min(end); // WIP red
            let mut lanes = [0.0f64; FITNESS_LANES];
            for k in seg_start..seg_end {
                let p = pred[k];
                let flag = self.flags[k];
                if !p.is_finite() {
                    t.nonfinite = true;
                }
                if flag & TradingRows::FIRST != 0 {
                    state = trading_gpu::BandState::default();
                    prev = false;
                }
                let long = match params {
                    Some(params) => {
                        state = params.next(state, p);
                        state.long
                    }
                    None => p.is_finite() && p > self.threshold,
                };
                if long != prev {
                    t.flips += 1;
                }
                if long {
                    t.long_rows += 1;
                    lanes[k % FITNESS_LANES] += self.excess[k]; // WIP red
                    if flag & TradingRows::LAST != 0 {
                        t.exits += 1;
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
        t.gross = warps[0];
        t
    }

    /// #1094: every root cut of `pred` in one sort and one sweep. `time` is the
    /// row timestamp, aligned with `pred`; it counts distinct entry timestamps.
    /// Errors: a non-finite prediction or time, a length mismatch, a book that
    /// is not plain ([`FitnessBook::is_plain`]).
    pub fn cut_sweep(&self, pred: &[f64], time: &[f64]) -> Result<CutSweep, String> {
        let n = self.excess.len();
        if pred.len() != n || time.len() != n {
            return Err(format!(
                "cut sweep: length mismatch: {} predictions and {} times for {} rows",
                pred.len(),
                time.len(),
                n
            ));
        }
        if !self.is_plain() {
            return Err("cut sweep: the book is not plain (it has a band or a minimum hold)".to_string());
        }
        if n >= u32::MAX as usize {
            return Err(format!("cut sweep: {} rows do not fit a u32 row index", n));
        }
        let mut sorted: Vec<(u64, u32)> = Vec::with_capacity(n);
        let mut stamps: Vec<u64> = Vec::with_capacity(n);
        for i in 0..n {
            let key = trading_gpu::order_key_f64(pred[i]).ok_or_else(|| format!("cut sweep: prediction {} is not finite", i))?;
            let stamp = trading_gpu::order_key_f64(time[i]).ok_or_else(|| format!("cut sweep: time {} is not finite", i))?;
            sorted.push((key, i as u32));
            stamps.push(stamp);
        }
        sorted.sort_unstable();
        let order: Vec<u32> = sorted.iter().map(|&(_, i)| i).collect();
        let mut rank = vec![0u32; n];
        for (r, &i) in order.iter().enumerate() {
            rank[i as usize] = r as u32;
        }
        let nb = self.blocks();
        let block = |i: usize| (self.flags[i] >> FITNESS_BLOCK_SHIFT) as usize & (FITNESS_MAX_BLOCKS - 1);
        // `diff[b * w + t]`: change of a count of block `b` where the cut reaches `t`.
        let w = n + 2;
        let mut flip_diff = vec![0i32; nb * w];
        let mut exit_diff = vec![0i32; nb * w];
        let mut entries: Vec<(u64, u32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            let (b, ri, flag) = (block(i), rank[i] as usize, self.flags[i]);
            if flag & TradingRows::FIRST != 0 {
                // A run starts flat: an entry while this row is long, `t <= ri`.
                flip_diff[b * w] += 1;
                flip_diff[b * w + ri + 1] -= 1;
                entries.push((stamps[i], 0, ri as u32));
            } else {
                let rp = rank[i - 1] as usize;
                let (lo, hi) = if rp < ri { (rp, ri) } else { (ri, rp) };
                // Exactly one row of the pair is long: `lo < t <= hi`.
                flip_diff[b * w + lo + 1] += 1;
                flip_diff[b * w + hi + 1] -= 1;
                if rp < ri {
                    entries.push((stamps[i], rp as u32 + 1, ri as u32));
                }
            }
            if flag & TradingRows::LAST != 0 {
                exit_diff[b * w] += 1;
                exit_diff[b * w + ri + 1] -= 1;
            }
        }
        // A timestamp counts once on the union of its rows' entry intervals.
        entries.sort_unstable();
        let mut entry_diff = vec![0i32; w];
        let mut k = 0;
        while k < entries.len() {
            let (stamp, mut lo, mut hi) = entries[k];
            k += 1;
            while k < entries.len() && entries[k].0 == stamp {
                let (_, a, c) = entries[k];
                if a <= hi + 1 {
                    hi = hi.max(c);
                } else {
                    entry_diff[lo as usize] += 1;
                    entry_diff[hi as usize + 1] -= 1;
                    lo = a;
                    hi = c;
                }
                k += 1;
            }
            entry_diff[lo as usize] += 1;
            entry_diff[hi as usize + 1] -= 1;
        }
        drop(entries);
        let mut entry_times = vec![0u32; n + 1];
        let mut running = 0i64;
        for t in 0..=n {
            running += entry_diff[t] as i64;
            entry_times[t] = running as u32;
        }
        // Descending `t` adds the row of rank `t` to the long set. Every interval
        // adds +1 and -1, so the count at `t` is minus the differences above `t`.
        let mut fitness = vec![f64::INFINITY; n + 1];
        let mut net = vec![f64::NAN; n + 1];
        let mut flips = vec![0u32; n + 1];
        let mut exits = vec![0u32; n + 1];
        let mut totals = vec![FitnessTotals { gross: 0.0, flips: 0, exits: 0, long_rows: 0, nonfinite: false }; nb];
        let mut flips_above = vec![0i64; nb];
        let mut exits_above = vec![0i64; nb];
        for t in (0..=n).rev() {
            if t < n {
                let i = order[t] as usize;
                let b = block(i);
                totals[b].gross += self.excess[i];
                totals[b].long_rows += 1;
            }
            let (mut f, mut e) = (0u64, 0u64);
            for b in 0..nb {
                totals[b].flips = (-flips_above[b]) as u64;
                totals[b].exits = (-exits_above[b]) as u64;
                f += totals[b].flips;
                e += totals[b].exits;
            }
            flips[t] = f as u32;
            exits[t] = e as u32;
            if t == 0 || t == n || sorted[t - 1].0 < sorted[t].0 {
                fitness[t] = self.fitness_of(&totals);
                net[t] = totals.iter().map(|x| self.net_of(x)).sum();
            }
            for b in 0..nb {
                flips_above[b] += flip_diff[b * w + t] as i64;
                exits_above[b] += exit_diff[b * w + t] as i64;
            }
        }
        Ok(CutSweep { order, fitness, net, flips, exits, entry_times })
    }
}

/// #1094: the floor of [`CutSweep::best`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutFloor {
    /// Fewest distinct timestamps at which at least one row enters.
    pub min_entry_times: u32,
}

/// #1094: the books of every root cut of one prediction vector on a plain
/// [`FitnessBook`]. Cut `t` in `0..=n` is long on the `n - t` rows of highest
/// prediction. `t` is valid when it does not split equal predictions (`-0.0`
/// equals `+0.0`); `t = 0` and `t = n` are always valid.
#[derive(Clone, Debug)]
pub struct CutSweep {
    /// Row indices by ascending prediction, ties by row index.
    order: Vec<u32>,
    /// `n + 1` values; `+inf` where `t` splits a tie.
    fitness: Vec<f64>,
    /// Sum over blocks of `gross - side * (flips + exits)`; NaN where invalid.
    net: Vec<f64>,
    flips: Vec<u32>,
    exits: Vec<u32>,
    /// Distinct timestamps at which at least one row enters.
    entry_times: Vec<u32>,
}

impl CutSweep {
    /// Rows of the book (`n`); the cuts are `0..=n`.
    pub fn rows(&self) -> usize {
        self.order.len()
    }

    pub fn is_valid(&self, t: usize) -> bool {
        !self.net[t].is_nan()
    }

    /// [`FitnessBook::fitness_of`] of cut `t`; `+inf` when `t` is not valid.
    pub fn fitness(&self, t: usize) -> f64 {
        self.fitness[t]
    }

    pub fn net(&self, t: usize) -> f64 {
        self.net[t]
    }

    pub fn flips(&self, t: usize) -> u32 {
        self.flips[t]
    }

    pub fn exits(&self, t: usize) -> u32 {
        self.exits[t]
    }

    pub fn entry_times(&self, t: usize) -> u32 {
        self.entry_times[t]
    }

    /// A threshold that realises cut `t` under `pred > threshold`: the midpoint
    /// of the two neighbouring predictions, or the lower one when the midpoint
    /// is not strictly below the upper one. `-inf` for `t = 0`, `+inf` for `n`.
    pub fn value(&self, pred: &[f64], t: usize) -> f64 {
        let n = self.order.len();
        if t == 0 {
            return f64::NEG_INFINITY;
        }
        if t >= n {
            return f64::INFINITY;
        }
        let lo = pred[self.order[t - 1] as usize];
        let hi = pred[self.order[t] as usize];
        let mid = lo + (hi - lo) * 0.5;
        if mid < hi {
            mid
        } else {
            lo
        }
    }

    /// The cut of threshold `theta`: the number of rows with `pred <= theta`.
    pub fn index_of(&self, pred: &[f64], theta: f64) -> usize {
        self.order.partition_point(|&i| pred[i as usize] <= theta)
    }

    /// The smallest valid cut at or above `t` (`n` at the latest).
    pub fn snap(&self, t: usize) -> usize {
        let n = self.order.len();
        (t.min(n)..n).find(|&k| self.is_valid(k)).unwrap_or(n)
    }

    /// The fitted cut: among valid `t` with `0 < t < n`, finite fitness and at
    /// least `floor.min_entry_times` entry timestamps, the lowest fitness; ties
    /// go to the smallest `|t - reference|`, then to the smaller `t`. `None`
    /// when no cut passes.
    pub fn best(&self, floor: &CutFloor, reference: usize) -> Option<usize> {
        let mut best: Option<usize> = None;
        for t in 1..self.order.len() {
            let f = self.fitness[t];
            if !self.is_valid(t) || !f.is_finite() || self.entry_times[t] < floor.min_entry_times {
                continue;
            }
            best = match best {
                Some(b) if !(f < self.fitness[b] || (f == self.fitness[b] && t.abs_diff(reference) < b.abs_diff(reference))) => Some(b),
                _ => Some(t),
            };
        }
        best
    }

    /// The widest run of valid inner cuts around `best` whose fitness stays
    /// within `0.1 * (1 - f)` of the best fitness `f`; `(best, best)` when `f`
    /// is 1 or more. Returns the lowest and the highest cut of the run.
    pub fn plateau(&self, best: usize) -> (usize, usize) {
        let n = self.order.len();
        let f = self.fitness[best];
        if !(f < 1.0) {
            return (best, best);
        }
        let limit = f + 0.1 * (1.0 - f);
        let inside = |k: usize| self.fitness[k] <= limit;
        let mut lo = best;
        for k in (1..best).rev() {
            if !self.is_valid(k) {
                continue;
            }
            if !inside(k) {
                break;
            }
            lo = k;
        }
        let mut hi = best;
        for k in best + 1..n {
            if !self.is_valid(k) {
                continue;
            }
            if !inside(k) {
                break;
            }
            hi = k;
        }
        (lo, hi)
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
    /// #849: the position the finalist holds on every input, when it has one.
    pub fixed_position: Option<bool>,
    /// #849: the position the validation book realised on every row, when it had one.
    pub fixed_position_realised: Option<bool>,
    pub candidates: usize,
    pub usable: usize,
    /// #1026: members gated by `--select-require-edge`; `None` without the flag.
    pub edge_rejected: Option<usize>,
    pub expression: String,
    /// #807: the finalist with every constant at full precision.
    pub expression_exact: String,
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
    /// #928: the rule that chose the finalist.
    pub selection: &'static str,
    /// #928: members of `objective.selection_rule`, without braces.
    pub selection_rule: String,
    /// #928: object naming the front a re-selection read.
    pub reselected_from: Option<String>,
    /// #929: the label whose return column prices the book, when it is not the fit label.
    pub book_target: Option<String>,
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
            "{{\"name\":\"{}\",{}\"position\":\"{}\",\"selection\":\"{}\",\
             \"selection_rule\":{{{}{}}},\"cost\":{},\"candidates\":{},\"usable_candidates\":{}{},\"exact\":\"cpu_f64\",\
             \"backend\":{},\"backend_reason\":{},\"spec\":{},\
             \"finalist\":{{\"expression\":{},\"expression_exact\":{},\"archive_rank\":{},\"constant\":{},\"fixed_position\":{},\"fixed_position_realised\":{},\"fit_metric\":{},\"fit_error\":{},\
             \"validation_error\":{},\"test_error\":{},\"validation_objective\":{}{}}},\
             \"validation\":{},\"test\":{}}}",
            self.objective.name(),
            self.book_target.as_deref().map(|b| format!("\"book_target\":{},", crate::devlog::text(b))).unwrap_or_default(),
            self.spec.position.name(),
            self.selection,
            self.selection_rule,
            self.reselected_from
                .as_deref()
                .map(|r| format!(",\"reselected_from\":{}", r))
                .unwrap_or_default(),
            self.cost.json(),
            self.candidates,
            self.usable,
            self.edge_rejected.map(|n| format!(",\"edge_rejected\":{}", n)).unwrap_or_default(),
            crate::devlog::text(backend),
            backend_reason
                .map(crate::devlog::text)
                .unwrap_or_else(|| "null".to_string()),
            self.spec_json(),
            crate::devlog::text(&self.expression),
            crate::devlog::text(&self.expression_exact),
            self.archive_rank,
            self.constant,
            fixed_position_json(self.fixed_position),
            fixed_position_json(self.fixed_position_realised),
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

#[cfg(test)]
mod block_tests {
    use super::*;

    #[test]
    fn block_ids_cut_distinct_timestamps_into_equal_runs() {
        let t = [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // 9 distinct stamps, 4 blocks of 2, the last takes 3.
        assert_eq!(block_ids(&t, 4).unwrap(), vec![0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3]);
        assert_eq!(block_ids(&t, 1).unwrap(), vec![0; 11]);
        assert!(block_ids(&t[..3], 4).unwrap_err().contains("distinct training timestamps"));
        assert!(block_ids(&t, 9).is_err());
    }

    fn panel(sizes: &[usize]) -> crate::data::TradingSlice {
        let mut rows = TradingRows::default();
        let mut ret = Vec::new();
        let mut k = 0usize;
        for (g, &n) in sizes.iter().enumerate() {
            for i in 0..n {
                rows.group.push(g as u64);
                rows.time.push(900.0 * i as f64);
                let x = k as f64 * 0.618_033_988_749_894_9;
                ret.push(0.004 * (x - x.floor() - 0.45));
                k += 1;
            }
        }
        crate::data::TradingSlice { rows, ret }
    }

    fn spec() -> TradingSpec {
        TradingSpec { cost_bps: 5.0, slippage_bps: 1.0, min_trades: 0, ..Default::default() }
    }

    fn pred(n: usize) -> Vec<f64> {
        (0..n).map(|i| if (i * 7 + i / 5) % 3 == 0 { 1.0 } else { -1.0 }).collect()
    }

    fn short_book() -> (FitnessBook, Vec<f64>) {
        let s = panel(&[60, 70]);
        let book = FitnessBook::new(&s, &spec(), 900.0).unwrap();
        let short = pred(s.ret.len() - 1);
        (book, short)
    }

    #[test]
    #[should_panic(expected = "fitness book")]
    fn fitness_book_refuses_a_short_prediction_vector_in_totals() {
        let (b, p) = short_book();
        b.totals(&p);
    }

    #[test]
    #[should_panic(expected = "fitness book")]
    fn fitness_book_refuses_a_short_prediction_vector_in_block_totals() {
        let (b, p) = short_book();
        b.block_totals(&p);
    }

    #[test]
    #[should_panic(expected = "fitness book")]
    fn fitness_book_refuses_a_short_prediction_vector_in_band_pass() {
        let (b, p) = short_book();
        b.band_pass(&p, 0.0, false);
    }

    #[test]
    fn one_block_is_the_plain_book() {
        let s = panel(&[60, 70, 1, 50]);
        let plain = FitnessBook::new(&s, &spec(), 900.0).unwrap();
        let one = FitnessBook::with_blocks(&s, &spec(), 900.0, 1, 3.0).unwrap();
        assert_eq!(plain.hash(), one.hash());
        assert_eq!(plain.flags(), one.flags());
        let p = pred(s.ret.len());
        assert_eq!(plain.regret(&p).to_bits(), one.regret(&p).to_bits());
        let t = plain.totals(&p);
        let b = one.block_totals(&p);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].gross.to_bits(), t.gross.to_bits());
        assert_eq!((b[0].flips, b[0].exits, b[0].long_rows), (t.flips, t.exits, t.long_rows));
    }

    #[test]
    fn block_seams_split_runs_inside_a_group() {
        let mut s = panel(&[8, 8]);
        s.ret.iter_mut().for_each(|r| *r = 0.01);
        let b = FitnessBook::with_blocks(&s, &spec(), 900.0, 2, 1.0).unwrap();
        let f = b.flags();
        let id = |i: usize| f[i] >> FITNESS_BLOCK_SHIFT;
        let seam = |i: usize| f[i] & 3;
        assert_eq!((0..16).map(id).collect::<Vec<_>>(), vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(seam(0), 1);
        assert_eq!(seam(3), 2);
        assert_eq!(seam(4), 1);
        assert_eq!(seam(7), 2);
        assert_eq!(seam(8), 1);
        assert_eq!(b.groups(), 2);
    }

    /// The block rows as their own training slice.
    fn block_slice(s: &crate::data::TradingSlice, ids: &[u8], b: u8) -> crate::data::TradingSlice {
        let mut rows = TradingRows::default();
        let mut ret = Vec::new();
        for i in 0..ids.len() {
            if ids[i] == b {
                rows.group.push(s.rows.group[i]);
                rows.time.push(s.rows.time[i]);
                ret.push(s.ret[i]);
            }
        }
        crate::data::TradingSlice { rows, ret }
    }

    #[test]
    fn block_regret_equals_separate_books_and_fitness_is_mean_plus_lambda_sd() {
        let s = panel(&[9000, 4000, 1, 7000, 300]);
        let k = 4usize;
        let lambda = 1.5;
        let book = FitnessBook::with_blocks(&s, &spec(), 900.0, k, lambda).unwrap();
        let ids = block_ids(&s.rows.time, k).unwrap();
        let p = pred(s.ret.len());
        let totals = book.block_totals(&p);
        let regrets = book.block_regrets(&totals);
        let mut want = Vec::new();
        for b in 0..k {
            let sl = block_slice(&s, &ids, b as u8);
            let bp: Vec<f64> = (0..ids.len()).filter(|&i| ids[i] as usize == b).map(|i| p[i]).collect();
            let plain = FitnessBook::new(&sl, &spec(), 900.0).unwrap();
            assert_eq!(plain.bound().to_bits(), book.block_bounds()[b].to_bits());
            let r = plain.regret(&bp);
            assert!((r - regrets[b]).abs() <= 1e-12, "block {b}: {r} vs {}", regrets[b]);
            want.push(r);
        }
        let mean = want.iter().sum::<f64>() / k as f64;
        let sd = (want.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (k - 1) as f64).sqrt();
        let fit = book.regret(&p);
        assert!((fit - (mean + lambda * sd)).abs() <= 1e-12, "{fit} vs {}", mean + lambda * sd);
        let flat = FitnessBook::with_blocks(&s, &spec(), 900.0, k, 0.0).unwrap();
        assert!((flat.regret(&p) - mean).abs() <= 1e-12);
    }

    /// #1023: every `(group, block)` run totals as a separate book of its own
    /// rows, bit for bit, for the plain rule and the minimum-hold automaton;
    /// the case regret is that book's regret.
    #[test]
    fn case_totals_equal_separate_books() {
        // 20 000 distinct stamps cut into 2 blocks of 10 000: each of the 3
        // groups has rows in both blocks, and a 10 000-row run spans two
        // 8 192-row segments.
        let s = panel(&[20_000, 18_000, 16_000]);
        let ids = block_ids(&s.rows.time, 2).unwrap();
        for sp in [spec(), TradingSpec { min_holding: 3, ..spec() }] {
            let book = FitnessBook::with_blocks(&s, &sp, 900.0, 2, 1.0).unwrap();
            let cases = book.cases();
            assert_eq!(cases.ranges.len(), 6, "3 groups x 2 blocks");
            assert_eq!(cases.kept, (0..6).collect::<Vec<_>>());
            let p = pred(s.ret.len());
            let totals = book.case_totals(&p);
            let errors = book.case_errors(&p);
            for (c, &(start, end)) in cases.ranges.iter().enumerate() {
                assert!((start..end).all(|i| s.rows.group[i] == s.rows.group[start] && ids[i] == ids[start]));
                let mut rows = TradingRows::default();
                rows.group = s.rows.group[start..end].to_vec();
                rows.time = s.rows.time[start..end].to_vec();
                let one = crate::data::TradingSlice { rows, ret: s.ret[start..end].to_vec() };
                let separate = FitnessBook::new(&one, &sp, 900.0).unwrap();
                let want = separate.totals(&p[start..end]);
                let got = totals[c];
                assert_eq!(got.gross.to_bits(), want.gross.to_bits(), "case {c}: gross");
                assert_eq!((got.flips, got.exits, got.long_rows), (want.flips, want.exits, want.long_rows), "case {c}");
                assert_eq!(cases.bounds[c].to_bits(), separate.bound().to_bits(), "case {c}: bound");
                assert_eq!(errors[c].to_bits(), (separate.regret_of(&want) as f32).to_bits(), "case {c}: regret");
            }
            // The runs of one block hold the block's counts.
            let blocks = book.block_totals(&p);
            for (b, block) in blocks.iter().enumerate() {
                let of_block: Vec<&FitnessTotals> =
                    cases.ranges.iter().zip(&totals).filter(|((s0, _), _)| ids[*s0] as usize == b).map(|(_, t)| t).collect();
                assert_eq!(of_block.iter().map(|t| t.flips + t.exits).sum::<u64>(), block.flips + block.exits);
                assert_eq!(of_block.iter().map(|t| t.long_rows).sum::<u64>(), block.long_rows);
            }
        }
        // A non-finite prediction makes its case `+inf` and leaves the others.
        let book = FitnessBook::with_blocks(&s, &spec(), 900.0, 2, 1.0).unwrap();
        let mut p = pred(s.ret.len());
        p[book.cases().ranges[3].0 + 5] = f64::NAN;
        let errors = book.case_errors(&p);
        assert_eq!(errors[3], f32::INFINITY);
        assert!(errors.iter().enumerate().all(|(c, e)| c == 3 || e.is_finite()));
        // A flat case scores exactly 1.
        let flat = book.case_errors(&vec![-1.0; s.ret.len()]);
        assert!(flat.iter().all(|&e| e == 1.0), "{flat:?}");
    }

    /// `F` flat, `L` long; one character per row.
    fn positions(rows: &str) -> Vec<f64> {
        rows.chars().map(|c| if c == 'L' { 1.0 } else { -1.0 }).collect()
    }

    /// #1084: the shape counts long rows and round trips of the book the
    /// regret prices, from the same reduction, for the plain rule, block seams
    /// and the minimum-hold automaton.
    #[test]
    fn regret_and_shape_reads_the_book_the_regret_prices() {
        let mut s = panel(&[10, 10]);
        s.ret.iter_mut().for_each(|r| *r = 0.01);
        // Group 0: two round trips closed by flips; group 1: one closed by a
        // flip, one by the exit at the group's last row.
        let p = positions("FLLLFFLLFFLLFFFFFLLL");
        let plain = FitnessBook::new(&s, &spec(), 900.0).unwrap();
        let (r, shape) = plain.regret_and_shape(&p);
        assert_eq!(r.to_bits(), plain.regret_of(&plain.totals(&p)).to_bits());
        assert_eq!(shape, BookShape { rows: 20, long_rows: 10, entries: 4 });
        assert_eq!((shape.exposure(), shape.mean_holding()), (0.5, 2.5));
        // A position held across the block seam is two round trips, as the
        // blocked regret charges it.
        let seam = positions("FFFLLLLFFFFFFFFFFFFF");
        assert_eq!(plain.regret_and_shape(&seam).1, BookShape { rows: 20, long_rows: 4, entries: 1 });
        let blocked = FitnessBook::with_blocks(&s, &spec(), 900.0, 2, 1.0).unwrap();
        let (r, shape) = blocked.regret_and_shape(&seam);
        assert_eq!(r.to_bits(), blocked.fitness_of(&blocked.block_totals(&seam)).to_bits());
        assert_eq!(shape, BookShape { rows: 20, long_rows: 4, entries: 2 });
        // A minimum hold of 3 keeps a one-row signal long for three rows; the
        // last group ends long after two rows.
        let short = positions("FLFFFFFFFFFFFFFFFFLF");
        assert_eq!(plain.regret_and_shape(&short).1, BookShape { rows: 20, long_rows: 2, entries: 2 });
        let held = FitnessBook::new(&s, &TradingSpec { min_holding: 3, ..spec() }, 900.0).unwrap();
        let (r, shape) = held.regret_and_shape(&short);
        assert_eq!(r.to_bits(), held.regret_of(&held.totals(&short)).to_bits());
        assert_eq!(shape, BookShape { rows: 20, long_rows: 5, entries: 2 });
        // A flat book has no entry and no holding.
        let flat = plain.regret_and_shape(&positions(&"F".repeat(20))).1;
        assert_eq!((flat.long_rows, flat.entries, flat.mean_holding()), (0, 0, 0.0));
    }

    #[test]
    fn non_finite_and_min_trades_score_infinity() {
        let s = panel(&[40, 40]);
        let sp = TradingSpec { min_trades: 3, ..spec() };
        let b = FitnessBook::with_blocks(&s, &sp, 900.0, 2, 1.0).unwrap();
        assert_eq!(b.regret(&vec![-1.0; 80]), f64::INFINITY);
        let mut p = pred(80);
        assert!(b.regret(&p).is_finite());
        p[50] = f64::NAN;
        assert_eq!(b.regret(&p), f64::INFINITY);
    }

    #[test]
    fn refusals_name_the_block_problem() {
        let s = panel(&[40, 40]);
        assert!(FitnessBook::with_blocks(&s, &spec(), 900.0, 0, 1.0).is_err());
        assert!(FitnessBook::with_blocks(&s, &spec(), 900.0, 9, 1.0).is_err());
        assert!(FitnessBook::with_blocks(&s, &spec(), 900.0, 2, -1.0).is_err());
        assert!(FitnessBook::with_blocks(&s, &spec(), 900.0, 2, f64::NAN).is_err());
        assert!(FitnessBook::with_blocks(&s, &spec(), 900.0, 41, 1.0).is_err());
    }
}

#[cfg(test)]
mod fixed_position_tests {
    use super::*;
    use crate::expr::{input, node, Op};

    /// #849: the report names the one position of `gauss(x)` and `-gauss(x)`,
    /// and leaves a tree the inputs decide at `null`.
    #[test]
    fn fixed_position_follows_the_book_rule() {
        let spec = TradingSpec::default();
        let g = node(Op::Gauss, vec![input(0)]);
        let neg = node(Op::Neg, vec![g.clone()]);
        assert_eq!(fixed_position(&g, &spec), Some(true));
        assert_eq!(fixed_position(&neg, &spec), Some(false));
        assert_eq!(fixed_position(&input(0), &spec), None);
        assert_eq!(fixed_position_json(Some(true)), "\"long\"");
        assert_eq!(fixed_position_json(Some(false)), "\"flat\"");
        assert_eq!(fixed_position_json(None), "null");
    }

    /// #849: the threshold is the book's, and only the indicator rule has a
    /// fixed position.
    #[test]
    fn fixed_position_reads_the_spec_threshold_and_rule() {
        let g = node(Op::Gauss, vec![input(0)]);
        let shifted = TradingSpec { threshold: 1e-30, ..TradingSpec::default() };
        assert_eq!(fixed_position(&g, &shifted), None);
        let high = TradingSpec { threshold: 1.0, ..TradingSpec::default() };
        assert_eq!(fixed_position(&g, &high), Some(false));
        let ramp = TradingSpec { tau: Some(0.5), ..TradingSpec::default() };
        assert_eq!(fixed_position(&g, &ramp), None);
        let tercile = TradingSpec { position: PositionMap::TopTercile, ..TradingSpec::default() };
        assert_eq!(fixed_position(&g, &tercile), None);
    }
}

#[cfg(test)]
mod realised_position_tests {
    use super::*;

    /// #849: the realised position reads the book rows, at any threshold.
    #[test]
    fn realised_position_reads_the_exposure_of_the_book() {
        let spec = TradingSpec { threshold: 1e-30, ..TradingSpec::default() };
        assert_eq!(realised_position(1.0, &spec), Some(true));
        assert_eq!(realised_position(0.0, &spec), Some(false));
        assert_eq!(realised_position(0.5, &spec), None);
        assert_eq!(realised_position(1.0, &TradingSpec { tau: Some(0.5), ..spec.clone() }), None);
        assert_eq!(
            realised_position(1.0, &TradingSpec { position: PositionMap::TopTercile, ..spec }),
            None
        );
    }
}

#[cfg(test)]
mod band_tests {
    use super::*;

    fn fold_sum(v: &mut [f64]) {
        let mut half = v.len() / 2;
        while half >= 1 {
            for i in 0..half {
                v[i] += v[i + half];
            }
            half /= 2;
        }
    }

    /// The #925 training-score sd: the mean, then the centred sum of squares,
    /// each summed per lane, lane tree and warp tree in the device order.
    fn two_pass_tree_sd(x: &[f64]) -> Option<f64> {
        let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
        let (lo, hi) = finite.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        if finite.len() < 2 || lo == hi {
            return None;
        }
        let tree = |term: &dyn Fn(f64) -> f64| {
            let mut warps = [0.0f64; FITNESS_WARPS];
            for (seg, chunk) in x.chunks(FITNESS_SEGMENT_ROWS).enumerate() {
                let mut lanes = [0.0f64; FITNESS_LANES];
                for (k, &v) in chunk.iter().enumerate() {
                    if v.is_finite() {
                        lanes[k % FITNESS_LANES] += term(v);
                    }
                }
                fold_sum(&mut lanes);
                warps[seg % FITNESS_WARPS] += lanes[0];
            }
            fold_sum(&mut warps);
            warps[0]
        };
        let mean = tree(&|v| v) / finite.len() as f64;
        let ss = tree(&|v| (v - mean) * (v - mean));
        let sd = (ss / (finite.len() - 1) as f64).sqrt();
        (sd.is_finite() && sd > 0.0).then_some(sd)
    }

    /// The one-pass sd written out from its definition: each lane sums its
    /// finite scores shifted by its first one, `(n, mean, M2)` follow, and the
    /// lanes, segments and warps merge pairwise (Chan, Golub and LeVeque).
    fn one_pass_reference(x: &[f64]) -> Option<f64> {
        type M = (u64, f64, f64);
        fn merge(a: M, b: M) -> M {
            match (a.0, b.0) {
                (_, 0) => a,
                (0, _) => b,
                _ => {
                    let (na, nb, nn) = (a.0 as f64, b.0 as f64, (a.0 + b.0) as f64);
                    let d = b.1 - a.1;
                    (a.0 + b.0, (na * a.1 + nb * b.1) / nn, a.2 + b.2 + d * d * (na * nb / nn))
                }
            }
        }
        fn halve(v: &mut [M]) {
            let mut half = v.len() / 2;
            while half >= 1 {
                for i in 0..half {
                    v[i] = merge(v[i], v[i + half]);
                }
                half /= 2;
            }
        }
        let mut warps = [(0u64, 0.0, 0.0); FITNESS_WARPS];
        for (seg, chunk) in x.chunks(FITNESS_SEGMENT_ROWS).enumerate() {
            let mut lanes: Vec<M> = (0..FITNESS_LANES)
                .map(|l| {
                    let v: Vec<f64> = chunk.iter().skip(l).step_by(FITNESS_LANES).copied().filter(|v| v.is_finite()).collect();
                    let Some(&k) = v.first() else {
                        return (0, 0.0, 0.0);
                    };
                    let (s, q) = v.iter().fold((0.0, 0.0), |(s, q), &x| (s + (x - k), q + (x - k) * (x - k)));
                    let t = s / v.len() as f64;
                    let m2 = q - s * t;
                    (v.len() as u64, k + t, if m2 < 0.0 { 0.0 } else { m2 })
                })
                .collect();
            halve(&mut lanes);
            warps[seg % FITNESS_WARPS] = merge(warps[seg % FITNESS_WARPS], lanes[0]);
        }
        halve(&mut warps);
        let (n, _, m2) = warps[0];
        let sd = (m2 / (n - 1) as f64).sqrt();
        (sd.is_finite() && sd > 0.0).then_some(sd)
    }

    /// #945: `score_sd` is the one-pass sd of `one_pass_reference` bit for
    /// bit, and within 1e-12 relative of the #925 two-pass sd and of a plain
    /// two-pass sd, also for scores far from 0; constant, single, non-finite
    /// and overflowing inputs are degenerate.
    #[test]
    fn score_sd_is_the_one_pass_tree_within_1e12_of_the_two_pass_sd() {
        let golden = |i: usize| (i as f64 * 0.618_033_988_749_894_9).fract();
        let mut v: Vec<f64> = (0..20_000).map(|i| (golden(i) - 0.3) * 4.0).collect();
        v[17] = f64::NAN;
        v[9_000] = f64::INFINITY;
        let offset: Vec<f64> = (0..100_003).map(|i| 1e3 + 1e-3 * (golden(i) - 0.5) + if i % 4_099 == 0 { f64::NAN } else { 0.0 }).collect();
        let tailed: Vec<f64> = (0..70_001).map(|i| (golden(i) - 0.5).powi(3) * 1e6 + golden(7 * i)).collect();
        let mut worst = 0.0f64;
        for x in [&v, &offset, &tailed] {
            let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
            let got = score_sd(x);
            assert_eq!(got.rows, finite.len());
            let sd = got.sd.expect("a spread score has an sd");
            assert_eq!(sd.to_bits(), one_pass_reference(x).unwrap().to_bits(), "{sd}");
            let two = two_pass_tree_sd(x).unwrap();
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let naive = (finite.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (finite.len() - 1) as f64).sqrt();
            for reference in [two, naive] {
                let rel = (sd - reference).abs() / reference;
                assert!(rel <= 1e-12, "{sd} vs {reference}: {rel:e}");
                worst = worst.max(rel);
            }
        }
        println!("one-pass against two-pass sd: largest relative difference {worst:e}");
        assert_eq!(score_sd(&[1.0]), ScoreSd { sd: None, rows: 1 });
        assert_eq!(score_sd(&[]), ScoreSd { sd: None, rows: 0 });
        assert_eq!(score_sd(&[f64::NAN, f64::INFINITY]), ScoreSd { sd: None, rows: 0 });
        assert_eq!(score_sd(&[f64::NAN, 3.0]), ScoreSd { sd: None, rows: 1 });
        assert_eq!(score_sd(&[2.0, 2.0, 2.0]), ScoreSd { sd: None, rows: 3 });
        assert_eq!(score_sd(&[1.0, f64::MAX, -f64::MAX]), ScoreSd { sd: None, rows: 3 });
        assert_eq!(score_sd(&[f64::MAX, f64::MAX / 2.0]), ScoreSd { sd: None, rows: 2 });
    }

    /// #925: the objective's band positions are the walk-forward automaton's.
    #[test]
    fn evaluate_prices_the_walkforward_band_positions() {
        let n = 400;
        let rows = TradingRows {
            time: (0..n).map(|i| (i % 200) as f64 * 900.0).collect(),
            group: (0..n).map(|i| (i / 200) as u64).collect(),
            funding: Vec::new(),
        };
        let mut v = 0.0f64;
        let pred: Vec<f64> = (0..n)
            .map(|i| {
                v = 0.8 * v + ((i as f64 * 0.754_877_666_246_692_7).fract() - 0.5);
                v
            })
            .collect();
        let r: Vec<f64> = (0..n).map(|i| 0.01 * ((i as f64 * 0.324_717_957_244_746).fract() - 0.5)).collect();
        for (band, hold) in [(0.0, 1usize), (0.25, 1), (0.0, 5), (0.4, 7)] {
            let spec = TradingSpec { band: BandSpec::score(band), min_holding: hold, cost_bps: 5.0, ..TradingSpec::default() };
            let book = evaluate(&pred, &r, &rows, &spec).unwrap();
            let params = trading_gpu::BandParams::new(0.0, band, hold).unwrap();
            let pos = trading_gpu::band_positions(&pred, &rows.group, &params).unwrap();
            let explicit = evaluate_positions(&pos, &r, &rows, &spec).unwrap();
            assert_eq!(book.net_pnl.to_bits(), explicit.net_pnl.to_bits(), "band {band} hold {hold}");
            assert_eq!(book.trades, explicit.trades);
        }
    }
}

#[cfg(test)]
mod cut_sweep_tests {
    use super::*;
    use std::collections::BTreeSet;

    struct Mix(u64);

    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uni(&mut self) -> f64 {
            ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn gauss(&mut self) -> f64 {
            let (u, v) = (self.uni(), self.uni());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// `groups` groups of `steps` hourly rows in canonical order. The market
    /// feature `z` of each stamp is shared by the groups.
    fn panel(rng: &mut Mix, groups: usize, steps: usize) -> (crate::data::TradingSlice, Vec<f64>) {
        let z: Vec<f64> = (0..steps).map(|_| rng.gauss()).collect();
        let (mut time, mut group, mut ret) = (Vec::new(), Vec::new(), Vec::new());
        for g in 0..groups {
            for (k, zk) in z.iter().enumerate() {
                time.push(k as f64 * 3600.0);
                group.push(g as u64);
                ret.push(0.004 * rng.gauss() + 0.002 * zk);
            }
        }
        let rows = TradingRows { time, group, funding: Vec::new() };
        let zs = (0..groups * steps).map(|i| z[i % steps]).collect();
        (crate::data::TradingSlice { rows, ret }, zs)
    }

    fn book(slice: &crate::data::TradingSlice, blocks: usize) -> FitnessBook {
        let spec = TradingSpec { cost_bps: 5.0, min_trades: 1, ..TradingSpec::default() };
        FitnessBook::with_blocks(slice, &spec, 0.0, blocks, 1.0).unwrap()
    }

    fn entry_count(book: &FitnessBook, pred: &[f64], time: &[f64], theta: f64) -> u32 {
        let mut seen = BTreeSet::new();
        let mut prev = false;
        for i in 0..pred.len() {
            if book.flags()[i] & TradingRows::FIRST != 0 {
                prev = false;
            }
            let long = pred[i] > theta;
            if long && !prev {
                seen.insert(trading_gpu::order_key_f64(time[i]).unwrap());
            }
            prev = long;
        }
        seen.len() as u32
    }

    /// Every cut of the sweep against a book whose threshold is that cut's value.
    fn check(book: &FitnessBook, pred: &[f64], time: &[f64]) -> usize {
        let sw = book.cut_sweep(pred, time).unwrap();
        let n = pred.len();
        assert_eq!(sw.rows(), n);
        let mut sorted = pred.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut checked = 0;
        for t in 0..=n {
            let valid = t == 0 || t == n || sorted[t - 1] < sorted[t];
            assert_eq!(sw.is_valid(t), valid, "validity of cut {t}");
            if !valid {
                assert_eq!(sw.fitness(t), f64::INFINITY, "cut {t} inside a tie");
                continue;
            }
            let theta = sw.value(pred, t);
            assert_eq!(pred.iter().filter(|&&p| p > theta).count(), n - t, "value of cut {t}");
            assert_eq!(sw.index_of(pred, theta), t);
            let mut at = book.clone();
            at.threshold = theta;
            let want = at.regret(pred);
            let got = sw.fitness(t);
            if want.is_finite() || got.is_finite() {
                assert!((want - got).abs() <= 1e-9 * want.abs().max(1.0), "cut {t}: book {want} sweep {got}");
            }
            let totals = at.block_totals(pred);
            let flips: u64 = totals.iter().map(|x| x.flips).sum();
            let exits: u64 = totals.iter().map(|x| x.exits).sum();
            assert_eq!(sw.flips(t) as u64, flips, "flips of cut {t}");
            assert_eq!(sw.exits(t) as u64, exits, "exits of cut {t}");
            let net: f64 = totals.iter().map(|x| at.net_of(x)).sum();
            assert!((net - sw.net(t)).abs() <= 1e-9 * net.abs().max(1.0), "net of cut {t}: {net} vs {}", sw.net(t));
            assert_eq!(sw.entry_times(t), entry_count(&at, pred, time, theta), "entry timestamps of cut {t}");
            checked += 1;
        }
        checked
    }

    #[test]
    fn cut_sweep_equals_the_book_at_every_cut() {
        let mut rng = Mix(1094);
        let mut cuts = 0;
        let mut ties = 0;
        for blocks in [1usize, 2, 4] {
            let (slice, z) = panel(&mut rng, 3, 667);
            let b = book(&slice, blocks);
            let pred: Vec<f64> = z.iter().map(|zk| zk + 0.5 * rng.gauss()).collect();
            cuts += check(&b, &pred, &slice.rows.time);
            let rounded: Vec<f64> = pred.iter().map(|p| (p * 4.0).round() / 4.0).collect();
            assert!(rounded.iter().any(|p| *p == 0.0 && p.is_sign_negative()), "the rounded panel has -0.0");
            assert!(rounded.iter().any(|p| *p == 0.0 && p.is_sign_positive()), "the rounded panel has +0.0");
            let valid = check(&b, &rounded, &slice.rows.time);
            ties += rounded.len() + 1 - valid;
            cuts += valid;
        }
        assert!(cuts > 6_000, "{cuts} cuts checked");
        assert!(ties > 4_500, "{ties} cuts inside ties");
    }

    #[test]
    fn cut_sweep_refuses_bad_input() {
        let mut rng = Mix(7);
        let (slice, z) = panel(&mut rng, 2, 50);
        let b = book(&slice, 1);
        let time = &slice.rows.time;
        let mut bad = z.clone();
        bad[3] = f64::NAN;
        assert!(b.cut_sweep(&bad, time).unwrap_err().contains("not finite"));
        assert!(b.cut_sweep(&z[1..], time).unwrap_err().contains("length"));
        let spec = TradingSpec { cost_bps: 5.0, min_holding: 2, ..TradingSpec::default() };
        let held = FitnessBook::with_blocks(&slice, &spec, 0.0, 1, 1.0).unwrap();
        assert!(held.cut_sweep(&z, time).unwrap_err().contains("plain"));
    }

    fn hand_sweep(fitness: &[f64], entry_times: &[u32]) -> CutSweep {
        let n = fitness.len() - 1;
        CutSweep {
            order: (0..n as u32).collect(),
            fitness: fitness.to_vec(),
            net: vec![0.0; n + 1],
            flips: vec![0; n + 1],
            exits: vec![0; n + 1],
            entry_times: entry_times.to_vec(),
        }
    }

    #[test]
    fn best_cut_applies_the_floor_and_the_tie_rule() {
        let floor = |e| CutFloor { min_entry_times: e };
        // t = 0 and t = n are never chosen, however low their fitness.
        let s = hand_sweep(&[0.1, 0.9, 0.5, 0.7, 0.6, 0.95, 0.2], &[99, 40, 5, 40, 40, 40, 0]);
        assert_eq!(s.best(&floor(30), 1), Some(4));
        assert_eq!(s.best(&floor(1), 1), Some(2));
        assert_eq!(s.best(&floor(41), 1), None);
        let tie = hand_sweep(&[1.0, 0.5, 0.8, 0.8, 0.5, 1.0], &[40; 6]);
        assert_eq!(tie.best(&floor(30), 4), Some(4));
        assert_eq!(tie.best(&floor(30), 2), Some(1));
        let even = hand_sweep(&[1.0, 0.5, 0.8, 0.5, 0.9, 1.0], &[40; 6]);
        assert_eq!(even.best(&floor(30), 2), Some(1));
        let mut invalid = hand_sweep(&[1.0, 0.7, f64::INFINITY, 0.9, 1.0], &[40; 5]);
        invalid.net[2] = f64::NAN;
        assert_eq!(invalid.best(&floor(30), 2), Some(1));
        let none = hand_sweep(&[0.0, f64::INFINITY, f64::INFINITY, 0.0], &[40; 4]);
        assert_eq!(none.best(&floor(1), 1), None);
    }

    /// The return pays only when the market feature is below its 3 % quantile.
    #[test]
    fn fitted_cut_finds_a_planted_gate() {
        let (groups, steps) = (4usize, 4000usize);
        let mut rng = Mix(268);
        let z: Vec<f64> = (0..steps).map(|_| rng.gauss()).collect();
        let mut sorted = z.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q03 = sorted[steps * 3 / 100];
        let mu = 0.01;
        let market: Vec<f64> = z.iter().map(|&zk| 0.003 * rng.gauss() + if zk < q03 { mu } else { 0.0 } - mu * 0.03).collect();
        let (mut time, mut group, mut ret, mut pred) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for g in 0..groups {
            for k in 0..steps {
                time.push(k as f64 * 3600.0);
                group.push(g as u64);
                ret.push(market[k] + 0.002 * rng.gauss());
                pred.push(-z[k]);
            }
        }
        let slice = crate::data::TradingSlice { rows: TradingRows { time: time.clone(), group, funding: Vec::new() }, ret };
        let spec = TradingSpec { cost_bps: 10.0, min_trades: 1, ..TradingSpec::default() };
        let b = FitnessBook::with_blocks(&slice, &spec, 0.0, 4, 1.0).unwrap();
        let sw = b.cut_sweep(&pred, &time).unwrap();
        let n = pred.len();
        let reference = sw.index_of(&pred, 0.0);
        let t = sw.best(&CutFloor { min_entry_times: 30 }, reference).expect("a cut passes the floor");
        let share = (n - t) as f64 / n as f64;
        let reference_share = (n - reference) as f64 / n as f64;
        assert!((0.02..=0.05).contains(&share), "fitted long share {share}");
        assert!(reference_share > 0.4, "reference long share {reference_share}");
        assert!(sw.fitness(t) < sw.fitness(reference));
    }

    /// Seconds of one sweep at the campaign training shape: 306 groups x 7,403
    /// hours, 4 blocks. A measurement, run by hand on an idle core.
    #[test]
    #[ignore]
    fn cut_sweep_cost_at_the_campaign_shape() {
        let mut rng = Mix(11);
        let (slice, z) = panel(&mut rng, 306, 7403);
        let b = book(&slice, 4);
        let pred: Vec<f64> = z.iter().map(|zk| -zk + 0.5 * rng.gauss()).collect();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let sw = b.cut_sweep(&pred, &slice.rows.time).unwrap();
            best = best.min(start.elapsed().as_secs_f64());
            assert_eq!(sw.rows(), 306 * 7403);
        }
        println!("cut_sweep: {} rows, 306 groups, 4 blocks: best of 3 {:.3} s", 306 * 7403, best);
    }
}
