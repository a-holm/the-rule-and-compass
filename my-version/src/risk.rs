//! Exact, deterministic risk monoids (issue #307).
//!
//! Every statistic here is computed from the **period return series** `R_t`
//! produced by the walk-forward evaluator (`walkforward::CrossSection::series`).
//! The CPU `f64` series is the reference; this module only reorganises the
//! arithmetic into associative carries so a caller may split a series into
//! canonical chunks (or per-thread shards) and merge the partials.
//!
//! # Units and annualisation
//!
//! * The input `R_t` is a **per-period net return** (a fraction, `0.01` = 1 %).
//! * `Sharpe`, `Sortino` and `alpha` are **per-period** means/scales. Their
//!   annualised form multiplies by `sqrt(periods_per_year)` for the two ratios
//!   and by `periods_per_year` for alpha. [`RiskStats`] records the
//!   `periods_per_year` it used and exposes both per-period and annualised
//!   values where that is meaningful.
//! * A *daily* Sharpe from hourly bars is a different statistic: first compound
//!   each day's hourly returns with [`compound_blocks`], then annualise the
//!   daily series with `periods_per_year = 365`. Never pass hourly returns with
//!   `sqrt(365)`; pass the compounded daily series.
//! * Log-growth uses [`f64::ln_1p`] on each period and sums. A period with
//!   `R_t <= -1` is explicit **ruin**: the total log-growth is `-inf` and the
//!   drawdown is `-100 %`. There is no silent clamp.
//! * Beta and alpha are dimensionless/per-period respectively; `alpha` is
//!   annualised by multiplying by `periods_per_year`.
//! * Cash (`R_t = 0` for every period) has zero variance, therefore `NaN`
//!   Sharpe and `NaN` Sortino — never `0`.
//!
//! # Determinism
//!
//! Floating-point addition is not associative. The contract is therefore a
//! **fixed canonical merge order**, not order-independent bits. Input is split
//! into [`CANONICAL_SEGMENT`]-sized segments from the start of the accumulator;
//! segments are reduced in that order and merged left-to-right. [`RiskAccumulator`]
//! buffers a trailing partial segment so that calling [`RiskAccumulator::push_chunk`]
//! with any sequence of chunk sizes gives the same bits. A parallel caller must
//! start each shard on a canonical segment boundary and merge the shard
//! accumulators in ascending order; concrete shard lengths that are multiples of
//! [`CANONICAL_SEGMENT`] satisfy this by construction.
//!
//! # Monoids
//!
//! | statistic | carry | merge |
//! |---|---|---|
//! | mean / variance | `(n, mean, M2)` | Chan et al. |
//! | Sortino | `(n, Σ min(R-MAR,0)²)` | add counts and squares |
//! | log-growth | `(n, Σ ln(1+R), ruined)` | add, ruin dominates |
//! | drawdown | `(n, total, min L, max L, dd, ruined)` | `dd = max(a.dd, b.dd, a.max - (a.total+b.min), -(a.total+b.min))` |
//! | beta / alpha | `(n, Σx, Σy, Σxy, Σx²)` | add moments |
//!
//! # Sortino definition
//!
//! The downside deviation is the **population** root-mean-square of the
//! negative excess returns over *all* periods:
//!
//! ```text
//! DD = sqrt( Σ min(R_t - MAR, 0)² / n )        (ddof = 0)
//! Sortino = (mean(R) - MAR) / DD * sqrt(periods_per_year)
//! ```
//!
//! It is `NaN` when `n < 2`, when `DD = 0` (no period below MAR), or when the
//! annualisation is non-positive. The default MAR is zero.

/// Canonical segment length. A shard must start on a multiple of this many
/// input periods for cross-shard bit identity.
pub const CANONICAL_SEGMENT: usize = 4_096;

/// Default minimum acceptable return for Sortino: zero excess over cash.
pub const DEFAULT_SORTINO_MAR: f64 = 0.0;

/// How a per-period series is turned into an annual figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnualisationMode {
    /// The input series is already the reporting period (hourly, daily, ...).
    PerPeriod,
    /// The input was compounded into days before the risk pass (see
    /// [`compound_blocks`]); the daily series is annualised with 365.
    DailyCompounded,
}

impl AnnualisationMode {
    /// Stable name recorded in `--json` and the schema.
    pub fn as_str(self) -> &'static str {
        match self {
            AnnualisationMode::PerPeriod => "per-period",
            AnnualisationMode::DailyCompounded => "daily-compounded",
        }
    }

    /// Periods per year for the declared mode.
    pub fn periods_per_year(self, periods_per_day: f64) -> f64 {
        match self {
            AnnualisationMode::PerPeriod => periods_per_day,
            AnnualisationMode::DailyCompounded => 365.0,
        }
    }
}

/// Compound consecutive blocks of period returns into one return per block.
///
/// This is the explicit "daily from hourly" step: `compound_blocks(hourly, 24)`
/// returns daily returns, which are then passed with `periods_per_year = 365`.
/// A final short block is compounded as-is. Returns are `NaN`/`inf` if the
/// product is not finite; callers should filter the input first.
pub fn compound_blocks(returns: &[f64], block: usize) -> Vec<f64> {
    if block <= 1 {
        return returns.to_vec();
    }
    returns
        .chunks(block)
        .map(|chunk| chunk.iter().fold(1.0f64, |eq, &r| eq * (1.0 + r)) - 1.0)
        .collect()
}

// ---------------------------------------------------------------------------
// Moments: Welford / Chan
// ---------------------------------------------------------------------------

/// Running count, mean and centred second moment (`Σ(x-mean)²`).
///
/// Sequential updates use Welford; merging two accumulators uses Chan's
/// parallel update. This avoids the catastrophic cancellation of
/// `Σx² - (Σx)²/n`, which is about `(mean/sd)²` worse (400× at a per-period
/// Sharpe of 0.05).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
}

impl Default for Welford {
    fn default() -> Self {
        Self::identity()
    }
}

impl Welford {
    pub const fn identity() -> Self {
        Welford {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add one finite return. Non-finite input is ignored (the walk-forward
    /// path already drops non-finite score/target rows).
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (x - self.mean);
    }

    /// Chan's merge. Counts add; the mean moves by the count-weighted
    /// difference; `M2` adds both centred sums and the between-group term.
    pub fn merge(a: &Welford, b: &Welford) -> Welford {
        if a.n == 0 {
            return *b;
        }
        if b.n == 0 {
            return *a;
        }
        let n = a.n + b.n;
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * (b.n as f64 / n as f64);
        let m2 = a.m2 + b.m2 + delta * delta * (a.n as f64 * b.n as f64 / n as f64);
        Welford { n, mean, m2 }
    }

    pub fn sample_variance(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        (self.m2 / (self.n - 1) as f64).max(0.0)
    }

    pub fn sample_sd(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Annualised Sharpe with `ddof = 1`; `NaN` for `n < 2` or zero variance.
    pub fn sharpe(&self, periods_per_year: f64) -> f64 {
        let sd = self.sample_sd();
        if self.n < 2 || !(periods_per_year > 0.0) || !(sd > 0.0) {
            return f64::NAN;
        }
        self.mean / sd * periods_per_year.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Sortino
// ---------------------------------------------------------------------------

/// `(n, Σ min(R-MAR,0)²)` with a fixed MAR.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sortino {
    pub n: u64,
    pub mar: f64,
    pub downside_m2: f64,
}

impl Sortino {
    pub const fn identity(mar: f64) -> Self {
        Sortino {
            n: 0,
            mar,
            downside_m2: 0.0,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let excess = x - self.mar;
        if excess < 0.0 {
            self.downside_m2 += excess * excess;
        }
    }

    pub fn merge(a: &Sortino, b: &Sortino) -> Sortino {
        debug_assert_eq!(
            a.mar.to_bits(),
            b.mar.to_bits(),
            "cannot merge Sortino accumulators with different MARs"
        );
        Sortino {
            n: a.n + b.n,
            mar: a.mar,
            downside_m2: a.downside_m2 + b.downside_m2,
        }
    }

    /// Population RMS downside deviation over all `n` periods (`ddof = 0`).
    pub fn downside_deviation(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        (self.downside_m2 / self.n as f64).sqrt()
    }

    /// Annualised Sortino. `mean` is the per-period mean of the same series.
    pub fn ratio(&self, mean: f64, periods_per_year: f64) -> f64 {
        let dd = self.downside_deviation();
        if self.n < 2 || !(periods_per_year > 0.0) || !(dd > 0.0) {
            return f64::NAN;
        }
        (mean - self.mar) / dd * periods_per_year.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Log-growth and ruin
// ---------------------------------------------------------------------------

/// `(n, Σ ln(1+R), ruined)` on the compounded equity path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogGrowth {
    pub n: u64,
    pub sum_log1p: f64,
    pub ruined: bool,
}

impl Default for LogGrowth {
    fn default() -> Self {
        Self::identity()
    }
}

impl LogGrowth {
    pub const fn identity() -> Self {
        LogGrowth {
            n: 0,
            sum_log1p: 0.0,
            ruined: false,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        if x <= -1.0 {
            self.ruined = true;
            self.sum_log1p = f64::NEG_INFINITY;
            return;
        }
        if !self.ruined {
            self.sum_log1p += x.ln_1p();
        }
    }

    pub fn merge(a: &LogGrowth, b: &LogGrowth) -> LogGrowth {
        let n = a.n + b.n;
        if a.ruined || b.ruined {
            LogGrowth {
                n,
                sum_log1p: f64::NEG_INFINITY,
                ruined: true,
            }
        } else {
            LogGrowth {
                n,
                sum_log1p: a.sum_log1p + b.sum_log1p,
                ruined: false,
            }
        }
    }

    /// Total log-growth; `-inf` on ruin and `NaN` for an empty series.
    pub fn total(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else if self.ruined {
            f64::NEG_INFINITY
        } else {
            self.sum_log1p
        }
    }

    /// Per-period log-growth (`total / n`); `-inf` on ruin.
    pub fn per_period(&self) -> f64 {
        let total = self.total();
        if self.n == 0 {
            f64::NAN
        } else {
            total / self.n as f64
        }
    }

    /// Compounded total return over the retained periods.
    pub fn total_return(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else if self.ruined {
            -1.0
        } else {
            self.sum_log1p.exp_m1()
        }
    }
}

// ---------------------------------------------------------------------------
// Drawdown on log equity
// ---------------------------------------------------------------------------

/// Drawdown carry over a segment of log-equity levels.
///
/// `total` is the segment's log return sum; `min`/`max` are the smallest and
/// largest levels **relative to the segment start**, including the virtual
/// level 0 before the first period; and `dd` is the largest log drawdown inside
/// the segment (including a drop from the segment start). The triple is
/// associative; [`Drawdown::max_drawdown`] converts the log drop to the
/// negative fraction reported by `walkforward::equity_metrics`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drawdown {
    pub n: u64,
    pub total: f64,
    pub min: f64,
    pub max: f64,
    pub dd: f64,
    pub ruined: bool,
}

impl Default for Drawdown {
    fn default() -> Self {
        Self::identity()
    }
}

impl Drawdown {
    /// Empty-segment identity `(+inf, -inf, 0)` for `min`, `max`, `dd`.
    pub const fn identity() -> Self {
        Drawdown {
            n: 0,
            total: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            dd: 0.0,
            ruined: false,
        }
    }

    fn ruined(n: u64) -> Self {
        Drawdown {
            n,
            total: f64::NEG_INFINITY,
            min: f64::NEG_INFINITY,
            max: f64::NEG_INFINITY,
            dd: f64::INFINITY,
            ruined: true,
        }
    }

    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        if x <= -1.0 {
            *self = Drawdown::ruined(self.n);
            return;
        }
        if self.ruined {
            return;
        }
        let level = x.ln_1p();
        // The virtual level before the first period is 0 (equity 1), which is
        // also the first peak. For later periods `self.max` is that peak.
        let peak_before = if self.n == 1 { 0.0 } else { self.max };
        self.total += level;
        if self.n == 1 {
            // The virtual level before the first period is 0 (equity 1); it is
            // both a possible peak and a possible trough.
            self.min = self.total.min(0.0);
            self.max = self.total.max(0.0);
        } else {
            self.min = self.min.min(self.total);
            self.max = self.max.max(self.total);
        }
        self.dd = self.dd.max(peak_before - self.total).max(0.0);
    }

    pub fn merge(a: &Drawdown, b: &Drawdown) -> Drawdown {
        if a.ruined || b.ruined {
            return Drawdown::ruined(a.n + b.n);
        }
        if a.n == 0 {
            return *b;
        }
        if b.n == 0 {
            return *a;
        }
        let total = a.total + b.total;
        let min = a.min.min(a.total + b.min);
        let max = a.max.max(a.total + b.max);
        // a.max is a peak inside a; the global start (level 0) is also a
        // candidate peak, so include the full start-to-b-trough drop.
        let cross = a.max - (a.total + b.min);
        let start_drop = -(a.total + b.min);
        let dd = a.dd.max(b.dd).max(cross).max(start_drop).max(0.0);
        Drawdown {
            n: a.n + b.n,
            total,
            min,
            max,
            dd,
            ruined: false,
        }
    }

    /// Most negative peak-to-trough return (`0` for monotone-up, `-1` for ruin).
    pub fn max_drawdown(&self) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        if self.ruined {
            return -1.0;
        }
        if self.dd == 0.0 {
            0.0
        } else {
            (-self.dd).exp_m1()
        }
    }

    /// Compounded total return over the retained periods.
    pub fn total_return(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else if self.ruined {
            -1.0
        } else {
            self.total.exp_m1()
        }
    }
}

// ---------------------------------------------------------------------------
// Beta / alpha
// ---------------------------------------------------------------------------

/// `(n, Σx, Σy, Σxy, Σx²)` for a strategy series `y` against a benchmark `x`.
///
/// `beta = cov(x,y) / var(x)` with sample (`n-1`) covariance and variance;
/// `alpha = mean(y) - beta * mean(x)` per period. `NaN` when `n < 2` or the
/// benchmark has zero variance. The sums are the carry named in report 03 §3;
/// both series must be aligned on the same periods.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaAlpha {
    pub n: u64,
    pub sum_x: f64,
    pub sum_y: f64,
    pub sum_xy: f64,
    pub sum_x2: f64,
}

impl Default for BetaAlpha {
    fn default() -> Self {
        Self::identity()
    }
}

impl BetaAlpha {
    pub const fn identity() -> Self {
        BetaAlpha {
            n: 0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xy: 0.0,
            sum_x2: 0.0,
        }
    }

    pub fn push(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xy += x * y;
        self.sum_x2 += x * x;
    }

    pub fn merge(a: &BetaAlpha, b: &BetaAlpha) -> BetaAlpha {
        BetaAlpha {
            n: a.n + b.n,
            sum_x: a.sum_x + b.sum_x,
            sum_y: a.sum_y + b.sum_y,
            sum_xy: a.sum_xy + b.sum_xy,
            sum_x2: a.sum_x2 + b.sum_x2,
        }
    }

    pub fn mean_x(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sum_x / self.n as f64
        }
    }

    pub fn mean_y(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sum_y / self.n as f64
        }
    }

    /// Sample covariance (`ddof = 1`).
    pub fn covariance(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        (self.sum_xy - self.sum_x * self.sum_y / n) / (n - 1.0)
    }

    /// Sample benchmark variance (`ddof = 1`).
    pub fn benchmark_variance(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        ((self.sum_x2 - self.sum_x * self.sum_x / n) / (n - 1.0)).max(0.0)
    }

    pub fn beta(&self) -> f64 {
        let var = self.benchmark_variance();
        if self.n < 2 || !(var > 0.0) {
            return f64::NAN;
        }
        self.covariance() / var
    }

    /// Per-period alpha.
    pub fn alpha(&self) -> f64 {
        let beta = self.beta();
        if !beta.is_finite() {
            return f64::NAN;
        }
        self.mean_y() - beta * self.mean_x()
    }

    /// Annualised alpha (`alpha * periods_per_year`); `NaN` if undefined.
    pub fn annualised_alpha(&self, periods_per_year: f64) -> f64 {
        let alpha = self.alpha();
        if !alpha.is_finite() || !(periods_per_year > 0.0) {
            f64::NAN
        } else {
            alpha * periods_per_year
        }
    }
}

// ---------------------------------------------------------------------------
// Combined segment and accumulator
// ---------------------------------------------------------------------------

/// One canonical segment's carry for all univariate risk statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskPart {
    pub moments: Welford,
    pub sortino: Sortino,
    pub growth: LogGrowth,
    pub drawdown: Drawdown,
}

impl RiskPart {
    pub fn from_returns(returns: &[f64], sortino_mar: f64) -> Self {
        let mut moments = Welford::identity();
        let mut sortino = Sortino::identity(sortino_mar);
        let mut growth = LogGrowth::identity();
        let mut drawdown = Drawdown::identity();
        for &r in returns {
            moments.push(r);
            sortino.push(r);
            growth.push(r);
            drawdown.push(r);
        }
        RiskPart {
            moments,
            sortino,
            growth,
            drawdown,
        }
    }

    pub fn merge(a: &RiskPart, b: &RiskPart) -> Self {
        RiskPart {
            moments: Welford::merge(&a.moments, &b.moments),
            sortino: Sortino::merge(&a.sortino, &b.sortino),
            growth: LogGrowth::merge(&a.growth, &b.growth),
            drawdown: Drawdown::merge(&a.drawdown, &b.drawdown),
        }
    }
}

/// Final risk numbers for one return series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiskStats {
    pub n: u64,
    pub mean: f64,
    pub sample_sd: f64,
    /// Annualised Sharpe (`mean / sd * sqrt(periods_per_year)`).
    pub sharpe: f64,
    /// Per-period Sharpe (`mean / sd`), the unannualised unit.
    pub sharpe_per_period: f64,
    pub sortino_mar: f64,
    /// Annualised Sortino.
    pub sortino: f64,
    pub sortino_per_period: f64,
    pub periods_per_year: f64,
    /// Σ ln(1+R); `-inf` on ruin.
    pub total_log_growth: f64,
    /// Σ ln(1+R) / n; `-inf` on ruin.
    pub log_growth_per_period: f64,
    pub total_return: f64,
    pub cagr: f64,
    /// Negative peak-to-trough fraction (`0` up, `-1` ruin).
    pub max_drawdown: f64,
    pub ruined: bool,
}

impl RiskStats {
    /// Compute from a completed [`RiskPart`].
    pub fn from_part(part: &RiskPart, periods_per_year: f64) -> Self {
        let m = part.moments;
        let sd = m.sample_sd();
        let sharpe_per_period = if m.n >= 2 && sd > 0.0 {
            m.mean / sd
        } else {
            f64::NAN
        };
        let sharpe = if sharpe_per_period.is_finite() && periods_per_year > 0.0 {
            sharpe_per_period * periods_per_year.sqrt()
        } else {
            f64::NAN
        };
        let dd = part.sortino.downside_deviation();
        let sortino_per_period = if part.sortino.n >= 2 && dd > 0.0 {
            (m.mean - part.sortino.mar) / dd
        } else {
            f64::NAN
        };
        let sortino = if sortino_per_period.is_finite() && periods_per_year > 0.0 {
            sortino_per_period * periods_per_year.sqrt()
        } else {
            f64::NAN
        };
        let total_log = part.growth.total();
        let log_growth_per_period = part.growth.per_period();
        let total_return = part.growth.total_return();
        let cagr = if m.n == 0 {
            f64::NAN
        } else if part.growth.ruined {
            -1.0
        } else if periods_per_year > 0.0 && total_log.is_finite() {
            (total_log * periods_per_year / m.n as f64).exp_m1()
        } else {
            f64::NAN
        };
        RiskStats {
            n: m.n,
            mean: m.mean,
            sample_sd: sd,
            sharpe,
            sharpe_per_period,
            sortino_mar: part.sortino.mar,
            sortino,
            sortino_per_period,
            periods_per_year,
            total_log_growth: total_log,
            log_growth_per_period,
            total_return,
            cagr,
            max_drawdown: part.drawdown.max_drawdown(),
            ruined: part.growth.ruined,
        }
    }

    /// `true` when Sharpe is undefined because the series is cash (zero
    /// variance), not merely short.
    pub fn is_cash(&self) -> bool {
        self.n >= 2 && self.sample_sd == 0.0
    }
}

/// Chunk-merge accumulator for all univariate risk statistics.
///
/// Feed returns with [`Self::push_chunk`] in any chunk sizes (the trailing
/// partial segment is buffered), or build one accumulator per canonical shard
/// and [`Self::merge`] them in ascending order.
#[derive(Clone, Debug)]
pub struct RiskAccumulator {
    sortino_mar: f64,
    parts: Vec<RiskPart>,
    pending: Vec<f64>,
}

impl RiskAccumulator {
    pub fn new(sortino_mar: f64) -> Self {
        RiskAccumulator {
            sortino_mar,
            parts: Vec::new(),
            pending: Vec::with_capacity(CANONICAL_SEGMENT),
        }
    }

    pub fn from_returns(returns: &[f64], sortino_mar: f64) -> Self {
        let mut acc = Self::new(sortino_mar);
        acc.push_chunk(returns);
        acc
    }

    /// Append a chunk of returns. Chunk boundaries are invisible: only
    /// canonical segment boundaries reset the per-segment Welford/drawdown
    /// carries.
    pub fn push_chunk(&mut self, returns: &[f64]) {
        for &r in returns {
            if !r.is_finite() {
                continue;
            }
            self.pending.push(r);
            if self.pending.len() == CANONICAL_SEGMENT {
                self.flush_pending();
            }
        }
    }

    fn flush_pending(&mut self) {
        if !self.pending.is_empty() {
            self.parts.push(RiskPart::from_returns(&self.pending, self.sortino_mar));
            self.pending.clear();
        }
    }

    /// Merge `other` after `self`. Both sides must start on a canonical segment
    /// boundary (all [`Self::push_chunk`] input before the merge is aligned).
    /// A trailing partial on either side is flushed as its own segment, so the
    /// result stays deterministic; exact bit identity across a parallel split
    /// requires shard lengths that are multiples of [`CANONICAL_SEGMENT`].
    pub fn merge(&mut self, other: &RiskAccumulator) {
        debug_assert_eq!(
            self.sortino_mar.to_bits(),
            other.sortino_mar.to_bits(),
            "cannot merge risk accumulators with different MARs"
        );
        self.flush_pending();
        let mut right = other.clone();
        right.flush_pending();
        self.parts.extend(right.parts);
    }

    pub fn finish(&self, periods_per_year: f64) -> RiskStats {
        let mut merged = RiskPart::from_returns(&[], self.sortino_mar);
        for part in &self.parts {
            merged = RiskPart::merge(&merged, part);
        }
        if !self.pending.is_empty() {
            let tail = RiskPart::from_returns(&self.pending, self.sortino_mar);
            merged = RiskPart::merge(&merged, &tail);
        }
        RiskStats::from_part(&merged, periods_per_year)
    }

    /// Number of canonical segments already reduced (excluding any buffered
    /// tail). Useful as a deterministic debug/telemetry counter.
    pub fn segments(&self) -> usize {
        self.parts.len()
    }
}

/// Convenience: all univariate risk statistics for a complete return series.
pub fn risk_stats(returns: &[f64], periods_per_year: f64, sortino_mar: f64) -> RiskStats {
    RiskAccumulator::from_returns(returns, sortino_mar).finish(periods_per_year)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walkforward;

    /// Deterministic uniform-ish returns in roughly [-2 %, +2 %].
    fn returns(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            out.push((u - 0.5) * 0.04);
        }
        out
    }

    #[test]
    fn sharpe_matches_cpu_reference_on_100k() {
        let r = returns(100_000, 0x5eed);
        let stats = risk_stats(&r, 8760.0, DEFAULT_SORTINO_MAR);
        let cpu = walkforward::sharpe(&r, 8760.0);
        assert!(
            (stats.sharpe - cpu).abs() < 1e-12,
            "monoid {} vs cpu {}",
            stats.sharpe,
            cpu
        );
        // The raw-sum shortcut really does cancel here: per-period Sharpe is
        // about 0.01 and (mean/sd)^2 would amplify the error by ~10^4.
        assert!(stats.sharpe.is_finite());
    }

    #[test]
    fn drawdown_matches_cpu_reference_on_100k() {
        let r = returns(100_000, 0xdd);
        let stats = risk_stats(&r, 365.0, DEFAULT_SORTINO_MAR);
        let (_, _, cpu_dd) = walkforward::equity_metrics(&r, 365.0);
        assert!(
            (stats.max_drawdown - cpu_dd).abs() < 1e-12,
            "monoid {} vs cpu {}",
            stats.max_drawdown,
            cpu_dd
        );
    }

    #[test]
    fn drawdown_monotone_up_is_zero_and_full_loss_is_minus_one() {
        let up = [0.01, 0.02, 0.03, 0.001];
        assert_eq!(
            risk_stats(&up, 365.0, DEFAULT_SORTINO_MAR).max_drawdown,
            0.0
        );
        let ruined = [0.01, -1.0, 0.5];
        let stats = risk_stats(&ruined, 365.0, DEFAULT_SORTINO_MAR);
        assert_eq!(stats.max_drawdown, -1.0);
        assert!(stats.ruined);
        assert_eq!(stats.total_log_growth, f64::NEG_INFINITY);
        assert_eq!(stats.total_return, -1.0);
    }

    #[test]
    fn sortino_mar_and_denominator_are_pinned() {
        // MAR 0, n = 4. Downside squares: 0^2 + 0.01^2 + 0^2 + 0.02^2.
        let r = [0.02, -0.01, 0.03, -0.02];
        let stats = risk_stats(&r, 252.0, 0.0);
        let mean = (0.02 - 0.01 + 0.03 - 0.02) / 4.0;
        let dd = ((0.01f64.powi(2) + 0.02f64.powi(2)) / 4.0).sqrt();
        let expected = mean / dd * 252.0f64.sqrt();
        assert!(
            (stats.sortino - expected).abs() < 1e-12,
            "sortino {} vs expected {} (mean {}, dd {})",
            stats.sortino,
            expected,
            stats.mean,
            dd
        );
        assert!((stats.mean - mean).abs() < 1e-15);
        // A positive-only series has DD = 0, so Sortino is undefined (not inf).
        assert!(risk_stats(&[0.01, 0.02], 252.0, 0.0).sortino.is_nan());
    }

    #[test]
    fn log_growth_uses_log1p_and_marks_ruin() {
        let r = [0.01, -0.02, 0.03];
        let stats = risk_stats(&r, 365.0, DEFAULT_SORTINO_MAR);
        let expected = (1.01f64.ln() + 0.98f64.ln() + 1.03f64.ln()) / 3.0;
        assert!((stats.log_growth_per_period - expected).abs() < 1e-15);
        assert!((stats.total_return - (1.01 * 0.98 * 1.03 - 1.0)).abs() < 1e-15);
        let ruin = risk_stats(&[0.01, -1.0], 365.0, DEFAULT_SORTINO_MAR);
        assert!(ruin.ruined);
        assert_eq!(ruin.log_growth_per_period, f64::NEG_INFINITY);
    }

    #[test]
    fn beta_alpha_matches_manual_two_series_fixture() {
        let x = [0.01, 0.02, -0.01, 0.03];
        let y = [0.02, 0.01, 0.00, 0.04];
        let mut ba = BetaAlpha::identity();
        for (&a, &b) in x.iter().zip(&y) {
            ba.push(a, b);
        }
        let mx = x.iter().sum::<f64>() / 4.0;
        let my = y.iter().sum::<f64>() / 4.0;
        let cov = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| (a - mx) * (b - my))
            .sum::<f64>()
            / 3.0;
        let var = x.iter().map(|&a| (a - mx) * (a - mx)).sum::<f64>() / 3.0;
        let beta = cov / var;
        let alpha = my - beta * mx;
        assert!((ba.beta() - beta).abs() < 1e-12, "{} vs {}", ba.beta(), beta);
        assert!(
            (ba.alpha() - alpha).abs() < 1e-12,
            "{} vs {}",
            ba.alpha(),
            alpha
        );
        assert!((ba.annualised_alpha(365.0) - alpha * 365.0).abs() < 1e-12);
    }

    #[test]
    fn cash_has_nan_sharpe_not_zero() {
        let cash = [0.0; 100];
        let stats = risk_stats(&cash, 8760.0, DEFAULT_SORTINO_MAR);
        assert!(stats.is_cash());
        assert!(stats.sharpe.is_nan());
        assert!(stats.sortino.is_nan());
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.max_drawdown, 0.0);
    }

    #[test]
    fn short_series_has_nan_sharpe() {
        assert!(risk_stats(&[], 365.0, 0.0).sharpe.is_nan());
        assert!(risk_stats(&[0.01], 365.0, 0.0).sharpe.is_nan());
    }

    #[test]
    fn sequential_chunk_sizes_are_bit_identical() {
        let r = returns(100_000, 0xabc);
        let one = risk_stats(&r, 8760.0, 0.0);
        for sizes in [&[1usize][..], &[4_096][..], &[65_536][..], &[7_777, 3, 20_000][..]] {
            let mut acc = RiskAccumulator::new(0.0);
            let mut at = 0usize;
            for &n in sizes.iter().cycle() {
                if at >= r.len() {
                    break;
                }
                let end = (at + n).min(r.len());
                acc.push_chunk(&r[at..end]);
                at = end;
            }
            assert_eq!(acc.finish(8760.0), one, "chunk pattern {sizes:?}");
        }
    }

    #[test]
    fn canonical_shards_merge_bit_identically() {
        let r = returns(100_000, 0xdef);
        let one = risk_stats(&r, 365.0, 0.0);
        for shard in [CANONICAL_SEGMENT, CANONICAL_SEGMENT * 2, CANONICAL_SEGMENT * 16] {
            let mut out = RiskAccumulator::new(0.0);
            for chunk in r.chunks(shard) {
                let part = RiskAccumulator::from_returns(chunk, 0.0);
                out.merge(&part);
            }
            assert_eq!(out.finish(365.0), one, "shard size {shard}");
        }
    }

    #[test]
    fn daily_compounding_is_explicit() {
        let hourly = [0.01, -0.005, 0.002, 0.0, 0.004, -0.001];
        let daily = compound_blocks(&hourly, 2);
        assert_eq!(daily.len(), 3);
        assert!((daily[0] - (1.01 * 0.995 - 1.0)).abs() < 1e-15);
        let stats = risk_stats(&daily, AnnualisationMode::DailyCompounded.periods_per_year(24.0), 0.0);
        assert_eq!(stats.periods_per_year, 365.0);
        assert_eq!(AnnualisationMode::DailyCompounded.as_str(), "daily-compounded");
    }
}
