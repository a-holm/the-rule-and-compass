//! #125: a first-class OHLCV market-data layer.
//!
//! First increment: canonical [`Bar`] and [`Timeframe`] types, exact streaming
//! resampling with causal alignment, and a multi-timeframe join that can only
//! see a higher-timeframe bar **after its bucket has closed**. Calendars,
//! corporate actions and the feature/label libraries are follow-up work; the
//! types here are the seam those will use.
//!
//! Every aggregation is closed-form and deterministic:
//!
//! * `open` = first, `high` = max, `low` = min, `close` = last;
//! * `volume`, `quote_volume`, `trades`, taker-buy base/quote = sums;
//! * empty buckets are **not** invented, so gaps stay visible;
//! * a zero-volume bucket is kept and flagged `traded = false` rather than
//!   dropped (dropping biases the sample);
//! * a bucket that has not closed is flagged `complete = false` by
//!   [`resample_as_of`] and is never visible to a causal join.

/// One OHLCV bar. `time` is the bucket start in seconds since the Unix epoch
/// (UTC); all prices/volumes are in the source units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades: f64,
    pub taker_buy_base: f64,
    pub taker_buy_quote: f64,
    /// The bucket has closed (all its base bars were seen).
    pub complete: bool,
    /// At least one trade/positive volume occurred in the bucket.
    pub traded: bool,
}

impl Bar {
    /// A bar with the optional fields zeroed and `complete = true`.
    pub fn new(time: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Bar {
            time,
            open,
            high,
            low,
            close,
            volume,
            quote_volume: 0.0,
            trades: 0.0,
            taker_buy_base: 0.0,
            taker_buy_quote: 0.0,
            complete: true,
            traded: volume > 0.0,
        }
    }

    pub fn with_flow(mut self, quote_volume: f64, trades: f64) -> Self {
        self.quote_volume = quote_volume;
        self.trades = trades;
        self.traded = self.volume > 0.0 || trades > 0.0;
        self
    }

    /// One-past-the-end instant of the bucket for `tf`.
    pub fn bucket_end(&self, tf: Timeframe) -> i64 {
        self.time + tf.seconds()
    }

    /// VWAP from quote/base volume when base volume is positive, else `None`.
    pub fn vwap(&self) -> Option<f64> {
        if self.volume > 0.0 {
            Some(self.quote_volume / self.volume)
        } else {
            None
        }
    }
}

/// A fixed-duration timeframe. Calendars (sessions, half-days, DST) are a
/// separate layer; these are epoch-aligned UTC buckets, correct for 24/7
/// markets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Timeframe {
    M1,
    M3,
    M5,
    M15,
    M30,
    H1,
    H2,
    H4,
    D1,
    W1,
}

impl Timeframe {
    pub fn seconds(self) -> i64 {
        match self {
            Timeframe::M1 => 60,
            Timeframe::M3 => 3 * 60,
            Timeframe::M5 => 5 * 60,
            Timeframe::M15 => 15 * 60,
            Timeframe::M30 => 30 * 60,
            Timeframe::H1 => 60 * 60,
            Timeframe::H2 => 2 * 60 * 60,
            Timeframe::H4 => 4 * 60 * 60,
            Timeframe::D1 => 24 * 60 * 60,
            Timeframe::W1 => 7 * 24 * 60 * 60,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "1m" => Ok(Timeframe::M1),
            "3m" => Ok(Timeframe::M3),
            "5m" => Ok(Timeframe::M5),
            "15m" => Ok(Timeframe::M15),
            "30m" => Ok(Timeframe::M30),
            "1h" => Ok(Timeframe::H1),
            "2h" => Ok(Timeframe::H2),
            "4h" => Ok(Timeframe::H4),
            "1d" => Ok(Timeframe::D1),
            "1w" => Ok(Timeframe::W1),
            other => Err(format!(
                "unknown timeframe '{}' (expected 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d or 1w)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M3 => "3m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1h",
            Timeframe::H2 => "2h",
            Timeframe::H4 => "4h",
            Timeframe::D1 => "1d",
            Timeframe::W1 => "1w",
        }
    }

    /// Epoch-aligned bucket start containing `time` (floored for negatives).
    pub fn bucket_start(self, time: i64) -> i64 {
        let s = self.seconds();
        time.div_euclid(s) * s
    }
}

/// Exact resampling of a sorted base series into `tf` buckets.
///
/// The input must be strictly increasing in `time`; a duplicate or a step
/// backwards is refused so a silent reorder cannot invent a bar. Empty buckets
/// keep the gap (no forward fill).
pub fn resample(bars: &[Bar], tf: Timeframe) -> Result<Vec<Bar>, String> {
    resample_as_of(bars, tf, i64::MAX)
}

/// [`resample`] with an explicit "now": the bucket containing `as_of` is kept
/// but flagged `complete = false`, and later buckets are not emitted. This is
/// how a live series avoids presenting a partial bar as history.
pub fn resample_as_of(bars: &[Bar], tf: Timeframe, as_of: i64) -> Result<Vec<Bar>, String> {
    if bars.is_empty() {
        return Ok(Vec::new());
    }
    for w in bars.windows(2) {
        if w[1].time <= w[0].time {
            return Err(format!(
                "bars are not strictly increasing at {} -> {}; sort and de-duplicate first",
                w[0].time, w[1].time
            ));
        }
    }
    let mut out: Vec<Bar> = Vec::new();
    let mut i = 0usize;
    while i < bars.len() {
        let start = tf.bucket_start(bars[i].time);
        if start > as_of {
            break;
        }
        let mut j = i + 1;
        while j < bars.len() && tf.bucket_start(bars[j].time) == start {
            j += 1;
        }
        let group = &bars[i..j];
        let mut bar = Bar {
            time: start,
            open: group[0].open,
            high: group[0].high,
            low: group[0].low,
            close: group[group.len() - 1].close,
            volume: 0.0,
            quote_volume: 0.0,
            trades: 0.0,
            taker_buy_base: 0.0,
            taker_buy_quote: 0.0,
            complete: start + tf.seconds() <= as_of,
            traded: false,
        };
        for b in group {
            bar.high = bar.high.max(b.high);
            bar.low = bar.low.min(b.low);
            bar.volume += b.volume;
            bar.quote_volume += b.quote_volume;
            bar.trades += b.trades;
            bar.taker_buy_base += b.taker_buy_base;
            bar.taker_buy_quote += b.taker_buy_quote;
        }
        bar.traded = bar.volume > 0.0 || bar.trades > 0.0;
        out.push(bar);
        i = j;
    }
    Ok(out)
}

/// Why a join cannot see a higher-timeframe bar yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    /// The bucket has closed at or before the decision instant.
    Visible,
    /// The bucket is still open (or is in the future): look-ahead.
    Open,
}

/// The last closed `higher` bar as of `decision_time`, if any.
///
/// A bar is visible only when `bar.time + higher_tf <= decision_time`. That is
/// the whole no-look-ahead rule: a 15m decision at 10:00 cannot see the 10:00
/// hourly bar (it closes at 11:00) nor the current daily bar.
pub fn last_closed_as_of<'a>(
    higher: &'a [Bar],
    higher_tf: Timeframe,
    decision_time: i64,
) -> Option<&'a Bar> {
    // Bars are sorted by start. Find the last start whose bucket has closed.
    let mut lo = 0usize;
    let mut hi = higher.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if higher[mid].time + higher_tf.seconds() <= decision_time {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.checked_sub(1).map(|i| &higher[i])
}

/// Causal multi-timeframe join: one higher-timeframe snapshot per base bar,
/// taken from the last **closed** higher bucket.
///
/// `base` and `higher` must both be sorted and strictly increasing. The
/// function refuses a higher series whose bars are not aligned to
/// `higher_tf`, because an unaligned bar has no well-defined close instant.
pub fn join_causal(
    base: &[Bar],
    higher: &[Bar],
    higher_tf: Timeframe,
) -> Result<Vec<Option<Bar>>, String> {
    for w in higher.windows(2) {
        if w[1].time <= w[0].time {
            return Err("higher-timeframe bars are not strictly increasing".to_string());
        }
    }
    for b in higher {
        if higher_tf.bucket_start(b.time) != b.time {
            return Err(format!(
                "higher-timeframe bar at {} is not aligned to {}",
                b.time,
                higher_tf.name()
            ));
        }
    }
    Ok(base
        .iter()
        .map(|b| last_closed_as_of(higher, higher_tf, b.time).copied())
        .collect())
}

/// Report every base row whose joined higher bar would not have closed yet.
///
/// This is the regression test for a deliberately shifted (future) column: a
/// correctly joined column returns an empty vector, a future bar is reported.
pub fn lookahead_violations(
    base: &[Bar],
    joined: &[Option<Bar>],
    higher_tf: Timeframe,
) -> Vec<usize> {
    let mut bad = Vec::new();
    for (i, slot) in joined.iter().enumerate() {
        if let Some(bar) = slot {
            if bar.time + higher_tf.seconds() > base[i].time {
                bad.push(i);
            }
        }
    }
    bad
}

/// Classify a candidate higher bar at a decision instant (used by tests and by
/// callers that build a column themselves).
pub fn visibility(bar: &Bar, higher_tf: Timeframe, decision_time: i64) -> Visibility {
    if bar.time + higher_tf.seconds() <= decision_time {
        Visibility::Visible
    } else {
        Visibility::Open
    }
}
