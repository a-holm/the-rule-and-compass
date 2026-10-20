//! #125: a first-class OHLCV market-data layer.
//!
//! The layer owns the things that are easy to get subtly wrong in a bespoke
//! builder:
//!
//! * a canonical [`Bar`] and [`Timeframe`], with exact, streaming resampling
//!   (`open` = first, `high` = max, `low` = min, `close` = last, flows = sum);
//! * [`Calendar`]s: crypto 24/7 and the NYSE/LSE session calendars with
//!   holidays, half-days and DST;
//! * corporate actions (splits and cash dividends) with raw and adjusted
//!   prices both recorded;
//! * symbol migrations/splices with a measured boundary ratio and a hard
//!   failure when the declared factor is wrong;
//! * causal multi-timeframe joins: a higher bar is visible only after its
//!   bucket has closed;
//! * gap detection and masks, so a feature never bridges an outage.
//!
//! Features and labels live in [`crate::features`]; this module supplies the
//! bar algebra they build on. Everything is pure `std` and deterministic.

use std::collections::BTreeMap;

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

    /// `true` when the bar has a usable OHLC and a positive price.
    pub fn valid(&self) -> bool {
        self.open.is_finite()
            && self.high.is_finite()
            && self.low.is_finite()
            && self.close.is_finite()
            && self.open > 0.0
            && self.high >= self.low
            && self.close > 0.0
    }
}

/// A fixed-duration timeframe. Intraday buckets are epoch-aligned UTC buckets;
/// `1d`/`1w` are calendar-aware when resampled with a [`Calendar`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    /// Number of `self`-sized buckets in `other` when `other` is a multiple,
    /// else `None`.
    pub fn per(self, other: Timeframe) -> Option<i64> {
        let a = self.seconds();
        let b = other.seconds();
        if b % a == 0 {
            Some(b / a)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Civil-time helpers (pure UTC, no external timezone database)
// ---------------------------------------------------------------------------

/// Days since 1970-01-01 for a proleptic Gregorian date (Hinnant's algorithm).
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let m = m as i64;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`].
pub fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (mp + if mp < 10 { 3 } else { -9 }) as u32;
    ((y + if m <= 2 { 1 } else { 0 }) as i32, m, d)
}

/// Day of week, 0 = Sunday .. 6 = Saturday.
pub fn weekday(days: i64) -> u32 {
    ((days + 4).rem_euclid(7)) as u32
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    }
}

/// The `n`th weekday `wd` (0 = Sunday) of a month.
fn nth_weekday(y: i32, m: u32, wd: u32, n: u32) -> i64 {
    let first = days_from_civil(y, m, 1);
    let first_wd = weekday(first);
    let delta = (wd + 7 - first_wd) % 7;
    first + delta as i64 + 7 * (n as i64 - 1)
}

fn last_weekday(y: i32, m: u32, wd: u32) -> i64 {
    let last = days_from_civil(y, m, days_in_month(y, m));
    let last_wd = weekday(last);
    last - ((last_wd + 7 - wd) % 7) as i64
}

/// Gregorian Easter Sunday (anonymous Gregorian computus).
fn easter(y: i32) -> i64 {
    let a = y % 19;
    let b = y / 100;
    let c = y % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    days_from_civil(y, month as u32, day as u32)
}

fn observed(days: i64) -> i64 {
    // Saturday -> Friday, Sunday -> Monday.
    match weekday(days) {
        0 => days + 1,
        6 => days - 1,
        _ => days,
    }
}

// ---------------------------------------------------------------------------
// Calendars
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Calendar {
    /// Crypto spot: every UTC day is a session; no holidays, no DST.
    Crypto247,
    /// NYSE/Nasdaq regular trading hours (09:30-16:00 America/New_York).
    Nyse,
    /// LSE regular hours (08:00-16:30 Europe/London).
    Lse,
}

/// One trading session, in UTC seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Session {
    pub open: i64,
    pub close: i64,
    /// Early close (13:00 New York, 12:30 London).
    pub half_day: bool,
}

impl Session {
    pub fn contains(&self, ts: i64) -> bool {
        self.open <= ts && ts < self.close
    }
}

impl Calendar {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "24x7" | "24/7" | "crypto" | "always" => Some(Calendar::Crypto247),
            "nyse" | "nasdaq" | "us" | "rth" => Some(Calendar::Nyse),
            "lse" | "uk" => Some(Calendar::Lse),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Calendar::Crypto247 => "24x7",
            Calendar::Nyse => "nyse",
            Calendar::Lse => "lse",
        }
    }

    pub fn timezone(self) -> &'static str {
        match self {
            Calendar::Crypto247 => "UTC",
            Calendar::Nyse => "America/New_York",
            Calendar::Lse => "Europe/London",
        }
    }

    /// UTC offset in seconds at a UTC instant.
    pub fn utc_offset(self, ts: i64) -> i64 {
        match self {
            Calendar::Crypto247 => 0,
            Calendar::Nyse => {
                let (y, _, _) = civil_from_days(ts.div_euclid(86_400));
                let start = nth_weekday(y, 3, 0, 2) * 86_400 + 7 * 3_600; // 02:00 EST
                let end = nth_weekday(y, 11, 0, 1) * 86_400 + 6 * 3_600; // 02:00 EDT
                if ts >= start && ts < end {
                    -4 * 3_600
                } else {
                    -5 * 3_600
                }
            }
            Calendar::Lse => {
                let (y, _, _) = civil_from_days(ts.div_euclid(86_400));
                let start = last_weekday(y, 3, 0) * 86_400 + 3_600;
                let end = last_weekday(y, 10, 0) * 86_400 + 3_600;
                if ts >= start && ts < end {
                    3_600
                } else {
                    0
                }
            }
        }
    }

    /// Convert a local civil instant to UTC seconds. The offset is resolved at
    /// the *local* date, which is what a session boundary needs.
    fn local_to_utc(self, y: i32, m: u32, d: u32, hour: i64, minute: i64) -> i64 {
        let naive = days_from_civil(y, m, d) * 86_400 + hour * 3_600 + minute * 60;
        // First guess with the standard offset, then re-resolve in case the
        // instant itself falls in the other DST regime.
        let first = naive - self.utc_offset(naive);
        naive - self.utc_offset(first)
    }

    /// Is the exchange closed all day (weekend/holiday)?
    pub fn is_trading_day(self, y: i32, m: u32, d: u32) -> bool {
        let days = days_from_civil(y, m, d);
        match weekday(days) {
            0 | 6 => return false,
            _ => {}
        }
        match self {
            Calendar::Crypto247 => true,
            Calendar::Nyse => {
                let holidays = [
                    observed(days_from_civil(y, 1, 1)),
                    nth_weekday(y, 1, 1, 3),  // MLK
                    nth_weekday(y, 2, 1, 3),  // Presidents
                    easter(y) - 2,            // Good Friday
                    last_weekday(y, 5, 1),    // Memorial
                    observed(days_from_civil(y, 6, 19)), // Juneteenth
                    observed(days_from_civil(y, 7, 4)),  // Independence
                    nth_weekday(y, 9, 1, 1),  // Labor
                    nth_weekday(y, 11, 4, 4), // Thanksgiving
                    observed(days_from_civil(y, 12, 25)), // Christmas
                ];
                !holidays.contains(&days)
            }
            Calendar::Lse => {
                let holidays = [
                    observed(days_from_civil(y, 1, 1)),
                    easter(y) - 2, // Good Friday
                    easter(y) + 1, // Easter Monday
                    nth_weekday(y, 5, 1, 1),  // Early May
                    last_weekday(y, 5, 1),    // Spring
                    last_weekday(y, 8, 1),    // Summer
                    observed(days_from_civil(y, 12, 25)),
                    observed(days_from_civil(y, 12, 26)),
                ];
                !holidays.contains(&days)
            }
        }
    }

    /// Is `days` (a local civil date) an early close?
    pub fn is_half_day(self, y: i32, m: u32, d: u32) -> bool {
        let days = days_from_civil(y, m, d);
        if weekday(days) == 0 || weekday(days) == 6 {
            return false;
        }
        match self {
            Calendar::Crypto247 => false,
            Calendar::Nyse => {
                let thanksgiving = nth_weekday(y, 11, 4, 4);
                days == thanksgiving + 1
                    || (m == 7 && d == 3)
                    || (m == 12 && d == 24)
            }
            Calendar::Lse => m == 12 && d == 24,
        }
    }

    /// The session on a local civil date, if the exchange trades.
    pub fn session_on(self, y: i32, m: u32, d: u32) -> Option<Session> {
        let days = days_from_civil(y, m, d);
        match self {
            Calendar::Crypto247 => Some(Session {
                open: days * 86_400,
                close: (days + 1) * 86_400,
                half_day: false,
            }),
            Calendar::Nyse => {
                if !self.is_trading_day(y, m, d) {
                    return None;
                }
                let half = self.is_half_day(y, m, d);
                Some(Session {
                    open: self.local_to_utc(y, m, d, 9, 30),
                    close: self.local_to_utc(y, m, d, if half { 13 } else { 16 }, 0),
                    half_day: half,
                })
            }
            Calendar::Lse => {
                if !self.is_trading_day(y, m, d) {
                    return None;
                }
                let half = self.is_half_day(y, m, d);
                Some(Session {
                    open: self.local_to_utc(y, m, d, 8, 0),
                    close: self.local_to_utc(y, m, d, if half { 12 } else { 16 }, 30),
                    half_day: half,
                })
            }
        }
    }

    /// The session containing `ts`, if any.
    pub fn session_at(self, ts: i64) -> Option<Session> {
        match self {
            Calendar::Crypto247 => {
                let day = ts.div_euclid(86_400);
                Some(Session {
                    open: day * 86_400,
                    close: (day + 1) * 86_400,
                    half_day: false,
                })
            }
            _ => {
                // Sessions are local-date bounded; check the local date and its
                // neighbours so a UTC instant near midnight is not missed.
                let local = ts + self.utc_offset(ts);
                let day = local.div_euclid(86_400);
                for delta in [-1i64, 0, 1] {
                    let (y, m, d) = civil_from_days(day + delta);
                    if let Some(s) = self.session_on(y, m, d) {
                        if s.contains(ts) {
                            return Some(s);
                        }
                    }
                }
                None
            }
        }
    }

    pub fn is_open(self, ts: i64) -> bool {
        self.session_at(ts).is_some()
    }

    /// Next session at or after `ts`.
    pub fn next_session(self, ts: i64) -> Option<Session> {
        if let Some(s) = self.session_at(ts) {
            return Some(s);
        }
        let local = ts + self.utc_offset(ts);
        let mut day = local.div_euclid(86_400);
        for _ in 0..400 {
            day += 1;
            let (y, m, d) = civil_from_days(day);
            if let Some(s) = self.session_on(y, m, d) {
                if s.close > ts {
                    return Some(s);
                }
            }
        }
        None
    }

    /// Session-aware bucket start: `1d` buckets open at the session open and
    /// `1w` buckets at the first session of the ISO week; intraday buckets are
    /// epoch-aligned. A bar outside a session gets its epoch bucket.
    pub fn bucket_start(self, ts: i64, tf: Timeframe) -> i64 {
        if self == Calendar::Crypto247 {
            return tf.bucket_start(ts);
        }
        match tf {
            Timeframe::D1 => self
                .session_at(ts)
                .map(|s| s.open)
                .unwrap_or_else(|| tf.bucket_start(ts)),
            Timeframe::W1 => {
                let day = ts.div_euclid(86_400);
                let wd = weekday(day) as i64;
                let monday = if wd == 0 { day - 6 } else { day - wd + 1 };
                let (y, m, d) = civil_from_days(monday);
                self.session_on(y, m, d)
                    .map(|s| s.open)
                    .unwrap_or(monday * 86_400)
            }
            _ => tf.bucket_start(ts),
        }
    }

    /// Session close for a bar opened at `bucket_start` (or `None` when the
    /// bucket is outside a session).
    pub fn bucket_close(self, bucket_start: i64, tf: Timeframe) -> Option<i64> {
        match self {
            Calendar::Crypto247 => Some(bucket_start + tf.seconds()),
            _ if tf == Timeframe::D1 => self.session_at(bucket_start).map(|s| s.close),
            _ if tf == Timeframe::W1 => Some(bucket_start + 7 * 86_400),
            _ => Some(bucket_start + tf.seconds()),
        }
    }
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

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
    resample_calendar_as_of(bars, tf, Calendar::Crypto247, as_of)
}

/// Calendar-aware resampling. Intraday buckets are epoch-aligned and a bar may
/// not cross a session boundary; daily/weekly buckets are keyed by the session
/// open. A base bar that crosses a session boundary is refused, never split
/// silently.
pub fn resample_calendar(
    bars: &[Bar],
    tf: Timeframe,
    cal: Calendar,
) -> Result<Vec<Bar>, String> {
    resample_calendar_as_of(bars, tf, cal, i64::MAX)
}

pub fn resample_calendar_as_of(
    bars: &[Bar],
    tf: Timeframe,
    cal: Calendar,
    as_of: i64,
) -> Result<Vec<Bar>, String> {
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
    if cal != Calendar::Crypto247 && matches!(tf, Timeframe::D1 | Timeframe::W1) {
        for b in bars {
            if cal.session_at(b.time).is_none() {
                return Err(format!(
                    "bar at {} is outside a {} session; split it with the calendar",
                    b.time,
                    cal.name()
                ));
            }
        }
    }
    let mut out: Vec<Bar> = Vec::new();
    let mut i = 0usize;
    while i < bars.len() {
        let start = cal.bucket_start(bars[i].time, tf);
        if start > as_of {
            break;
        }
        let mut j = i + 1;
        while j < bars.len() && cal.bucket_start(bars[j].time, tf) == start {
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
            complete: group[group.len() - 1].complete,
            traded: false,
        };
        let close_instant = cal.bucket_close(start, tf).unwrap_or(start + tf.seconds());
        bar.complete &= close_instant <= as_of;
        for b in group {
            bar.high = bar.high.max(b.high);
            bar.low = bar.low.min(b.low);
            bar.volume += b.volume;
            bar.quote_volume += b.quote_volume;
            bar.trades += b.trades;
            bar.taker_buy_base += b.taker_buy_base;
            bar.taker_buy_quote += b.taker_buy_quote;
            bar.complete &= b.complete;
        }
        bar.traded = bar.volume > 0.0 || bar.trades > 0.0;
        out.push(bar);
        i = j;
    }
    Ok(out)
}

/// A streaming resampler: feed base bars in order, receive a sealed bar when a
/// bucket changes. Memory is one bucket, not the whole series.
#[derive(Clone, Debug)]
pub struct Resampler {
    tf: Timeframe,
    current: Option<Bar>,
}

impl Resampler {
    pub fn new(tf: Timeframe) -> Self {
        Resampler { tf, current: None }
    }

    /// Feed one base bar; returns the previous bucket when this bar starts a
    /// new one.
    pub fn push(&mut self, bar: Bar) -> Option<Bar> {
        let start = self.tf.bucket_start(bar.time);
        match self.current.as_mut() {
            Some(cur) if cur.time == start => {
                cur.high = cur.high.max(bar.high);
                cur.low = cur.low.min(bar.low);
                cur.close = bar.close;
                cur.volume += bar.volume;
                cur.quote_volume += bar.quote_volume;
                cur.trades += bar.trades;
                cur.taker_buy_base += bar.taker_buy_base;
                cur.taker_buy_quote += bar.taker_buy_quote;
                cur.complete &= bar.complete;
                cur.traded |= bar.volume > 0.0 || bar.trades > 0.0;
                None
            }
            _ => {
                let sealed = self.current.replace(Bar {
                    time: start,
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                    volume: bar.volume,
                    quote_volume: bar.quote_volume,
                    trades: bar.trades,
                    taker_buy_base: bar.taker_buy_base,
                    taker_buy_quote: bar.taker_buy_quote,
                    complete: bar.complete,
                    traded: bar.volume > 0.0 || bar.trades > 0.0,
                });
                sealed
            }
        }
    }

    /// Flush the final (possibly partial) bucket.
    pub fn finish(&mut self) -> Option<Bar> {
        self.current.take()
    }
}

// ---------------------------------------------------------------------------
// Causal multi-timeframe joins
// ---------------------------------------------------------------------------

/// Why a join cannot see a higher-timeframe bar yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Visible,
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

/// Causal calendar-aware variant: a higher bucket is visible when its
/// calendar close is at or before the decision instant.
pub fn last_closed_calendar_as_of<'a>(
    higher: &'a [Bar],
    higher_tf: Timeframe,
    cal: Calendar,
    decision_time: i64,
) -> Option<&'a Bar> {
    higher
        .iter()
        .take_while(|b| cal.bucket_close(b.time, higher_tf).unwrap_or(b.time + higher_tf.seconds()) <= decision_time)
        .last()
}

/// Causal multi-timeframe join: one higher-timeframe snapshot per base bar,
/// taken from the last **closed** higher bucket.
pub fn join_causal(
    base: &[Bar],
    higher: &[Bar],
    higher_tf: Timeframe,
) -> Result<Vec<Option<Bar>>, String> {
    join_causal_calendar(base, higher, higher_tf, Calendar::Crypto247)
}

pub fn join_causal_calendar(
    base: &[Bar],
    higher: &[Bar],
    higher_tf: Timeframe,
    cal: Calendar,
) -> Result<Vec<Option<Bar>>, String> {
    for w in higher.windows(2) {
        if w[1].time <= w[0].time {
            return Err("higher-timeframe bars are not strictly increasing".to_string());
        }
    }
    for b in higher {
        if cal == Calendar::Crypto247 && higher_tf.bucket_start(b.time) != b.time {
            return Err(format!(
                "higher-timeframe bar at {} is not aligned to {}",
                b.time,
                higher_tf.name()
            ));
        }
    }
    Ok(base
        .iter()
        .map(|b| {
            if cal == Calendar::Crypto247 {
                last_closed_as_of(higher, higher_tf, b.time).copied()
            } else {
                last_closed_calendar_as_of(higher, higher_tf, cal, b.time).copied()
            }
        })
        .collect())
}

/// Join a scalar higher-timeframe column (e.g. a feature) onto base rows. The
/// returned value is `None` until the corresponding higher bar has closed.
pub fn join_column_causal(
    base: &[Bar],
    higher: &[Bar],
    higher_tf: Timeframe,
    higher_values: &[f64],
) -> Result<Vec<Option<f64>>, String> {
    if higher.len() != higher_values.len() {
        return Err(format!(
            "higher bars ({}) and values ({}) differ in length",
            higher.len(),
            higher_values.len()
        ));
    }
    let joined = join_causal(base, higher, higher_tf)?;
    let mut out = Vec::with_capacity(base.len());
    for slot in joined {
        out.push(slot.map(|b| {
            higher
                .iter()
                .position(|x| x.time == b.time)
                .map(|i| higher_values[i])
                .unwrap_or(f64::NAN)
        }));
    }
    Ok(out)
}

/// Report every base row whose joined higher bar would not have closed yet.
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

pub fn visibility(bar: &Bar, higher_tf: Timeframe, decision_time: i64) -> Visibility {
    if bar.time + higher_tf.seconds() <= decision_time {
        Visibility::Visible
    } else {
        Visibility::Open
    }
}

// ---------------------------------------------------------------------------
// Streaming bars out of a `.rcsb` column store
// ---------------------------------------------------------------------------

/// Stream OHLCV bars out of a column store without materialising the table.
///
/// The store is read one row group at a time; `f` sees bars in store order.
/// Column names are matched case-insensitively through the usual aliases
/// (`ts`/`timestamp`/`time`, `o`/`open`, ...); optional columns default to 0.
pub fn for_each_bar_from_store<F>(path: &str, mut f: F) -> Result<(), String>
where
    F: FnMut(Bar) -> Result<(), String>,
{
    use crate::column::GroupBuf;
    use crate::store::FileStore;
    let store = FileStore::open(path)?;
    let names: Vec<String> = store.header().names();
    let find = |aliases: &[&str]| -> Option<usize> {
        names.iter().position(|n| {
            let l = n.to_ascii_lowercase();
            aliases.iter().any(|a| l == *a)
        })
    };
    let time_col = find(&["ts", "timestamp", "time", "date", "datetime"])
        .ok_or_else(|| format!("{}: no time column", path))?;
    let open_col = find(&["open", "o"]).ok_or_else(|| format!("{}: no open column", path))?;
    let high_col = find(&["high", "h"]).ok_or_else(|| format!("{}: no high column", path))?;
    let low_col = find(&["low", "l"]).ok_or_else(|| format!("{}: no low column", path))?;
    let close_col = find(&["close", "c"]).ok_or_else(|| format!("{}: no close column", path))?;
    let vol_col = find(&["volume", "vol", "v"]);
    let quote_col = find(&["quote_volume", "quotevolume", "qv"]);
    let trades_col = find(&["trades", "n_trades"]);
    let taker_base_col = find(&["taker_buy_base", "takerbuybase"]);
    let taker_quote_col = find(&["taker_buy_quote", "takerbuyquote"]);
    let mut buf = GroupBuf::new();
    let mut scratch = Vec::new();
    for g in 0..store.n_groups() {
        let view = store.load_group_with(g, &mut buf, &mut scratch)?;
        let times = if store.header().dtypes()[time_col] == crate::store::DType::I64 {
            store.read_i64(g, time_col)?
        } else {
            (0..view.len())
                .map(|r| view.col(time_col)[r] as i64)
                .collect()
        };
        let get = |c: Option<usize>, r: usize| -> f64 {
            c.map(|c| view.col(c)[r]).unwrap_or(0.0)
        };
        for r in 0..view.len() {
            let volume = get(vol_col, r);
            let mut bar = Bar {
                time: times[r],
                open: view.col(open_col)[r],
                high: view.col(high_col)[r],
                low: view.col(low_col)[r],
                close: view.col(close_col)[r],
                volume,
                quote_volume: get(quote_col, r),
                trades: get(trades_col, r),
                taker_buy_base: get(taker_base_col, r),
                taker_buy_quote: get(taker_quote_col, r),
                complete: true,
                traded: volume > 0.0,
            };
            if bar.trades > 0.0 {
                bar.traded = true;
            }
            f(bar)?;
        }
    }
    Ok(())
}

/// Collect all bars from a store. For a huge store prefer
/// [`for_each_bar_from_store`], which keeps one group resident.
pub fn bars_from_store(path: &str) -> Result<Vec<Bar>, String> {
    let mut out = Vec::new();
    for_each_bar_from_store(path, |b| {
        out.push(b);
        Ok(())
    })?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Corporate actions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionKind {
    /// `ratio` new shares for one old share (2.0 = two-for-one split).
    Split { ratio: f64 },
    /// Cash per share, in the price currency.
    Dividend { cash: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorporateAction {
    /// Ex-date as a UTC day number (00:00 UTC of that day).
    pub ex_date: i64,
    pub kind: ActionKind,
}

/// Raw bars plus the derived total-return series and the per-bar factors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdjustedSeries {
    pub raw: Vec<Bar>,
    /// Multiplicative price factor applied to each raw bar.
    pub price_factor: Vec<f64>,
    /// Multiplicative base-volume factor (splits divide the historical price
    /// and multiply the historical base volume).
    pub volume_factor: Vec<f64>,
    /// Adjusted OHLCV, same length and order as `raw`.
    pub adjusted: Vec<Bar>,
}

impl AdjustedSeries {
    /// Apply a factor to one bar's prices.
    fn scale(bar: &Bar, pf: f64, vf: f64) -> Bar {
        Bar {
            time: bar.time,
            open: bar.open * pf,
            high: bar.high * pf,
            low: bar.low * pf,
            close: bar.close * pf,
            volume: bar.volume * vf,
            quote_volume: bar.quote_volume,
            trades: bar.trades,
            taker_buy_base: bar.taker_buy_base * vf,
            taker_buy_quote: bar.taker_buy_quote,
            complete: bar.complete,
            traded: bar.traded,
        }
    }
}

/// Apply split and dividend actions. Both raw and adjusted bars are returned;
/// an action on/after a bar's instant adjusts that bar, so the series is
/// continuous at the open following the ex-date.
pub fn adjust(actions: &[CorporateAction], bars: &[Bar]) -> Result<AdjustedSeries, String> {
    if bars.windows(2).any(|w| w[1].time <= w[0].time) {
        return Err("bars must be strictly increasing before adjustment".to_string());
    }
    let mut sorted = actions.to_vec();
    sorted.sort_by_key(|a| a.ex_date);
    for a in &sorted {
        match a.kind {
            ActionKind::Split { ratio } if !(ratio.is_finite() && ratio > 0.0) => {
                return Err(format!("split ratio {} is not positive", ratio))
            }
            ActionKind::Dividend { cash } if !cash.is_finite() => {
                return Err("dividend cash is not finite".to_string())
            }
            _ => {}
        }
    }
    let mut price_factor = vec![1.0f64; bars.len()];
    let mut volume_factor = vec![1.0f64; bars.len()];
    for a in &sorted {
        // Previous close for a cash dividend: the last bar before the ex-date.
        let prev_close = bars
            .iter()
            .rev()
            .find(|b| b.time < a.ex_date * 86_400)
            .map(|b| b.close);
        let (pf, vf) = match a.kind {
            ActionKind::Split { ratio } => (1.0 / ratio, ratio),
            ActionKind::Dividend { cash } => {
                let prev = prev_close.ok_or_else(|| {
                    format!("dividend on {} has no prior close", a.ex_date)
                })?;
                if prev <= 0.0 || cash >= prev {
                    return Err(format!(
                        "dividend {} on {} is not below the prior close {}",
                        cash, a.ex_date, prev
                    ));
                }
                ((prev - cash) / prev, 1.0)
            }
        };
        for (i, b) in bars.iter().enumerate() {
            if b.time < a.ex_date * 86_400 {
                price_factor[i] *= pf;
                volume_factor[i] *= vf;
            }
        }
    }
    let adjusted: Vec<Bar> = bars
        .iter()
        .enumerate()
        .map(|(i, b)| AdjustedSeries::scale(b, price_factor[i], volume_factor[i]))
        .collect();
    Ok(AdjustedSeries {
        raw: bars.to_vec(),
        price_factor,
        volume_factor,
        adjusted,
    })
}

// ---------------------------------------------------------------------------
// Symbol migrations / splices
// ---------------------------------------------------------------------------

/// A symbol migration: `old` trades as `new` with `k` old units per new unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Migration {
    pub old: String,
    pub new: String,
    pub k: f64,
    /// Allowed absolute log error of the measured boundary ratio versus `k`.
    pub tolerance: f64,
}

impl Migration {
    pub fn new(old: &str, new: &str, k: f64) -> Self {
        Migration {
            old: old.to_string(),
            new: new.to_string(),
            k,
            tolerance: 0.005,
        }
    }
}

/// Measured old/new boundary ratio: `old_close / new_open`. A correct `k` makes
/// this close to `k`.
pub fn splice_ratio(old_last: &Bar, new_first: &Bar) -> f64 {
    old_last.close / new_first.open
}

/// Verify a splice boundary and scale the old leg.
///
/// Prices are divided by `k`, base volume and taker-buy base are multiplied by
/// `k`, and old bars at or after the new leg's first instant are removed (the
/// overlap belongs to the successor). A measured ratio outside
/// `|ln(ratio/k)| <= tolerance` is a hard error naming both values.
pub fn splice(
    old: &[Bar],
    new: &[Bar],
    k: f64,
    tolerance: f64,
) -> Result<Vec<Bar>, String> {
    if !(k.is_finite() && k > 0.0) {
        return Err(format!("splice factor k={} is not positive", k));
    }
    let old_last = old.last().ok_or("splice: old leg is empty")?;
    let new_first = new.first().ok_or("splice: new leg is empty")?;
    let ratio = splice_ratio(old_last, new_first);
    let err = (ratio / k).ln().abs();
    if err > tolerance {
        return Err(format!(
            "splice {}->{}: measured ratio {:.6} differs from k={:.6} by {:.4}% (tolerance {:.4}%)",
            "legacy",
            "successor",
            ratio,
            k,
            err * 100.0,
            tolerance * 100.0
        ));
    }
    let cutoff = new_first.time;
    let mut out: Vec<Bar> = old
        .iter()
        .filter(|b| b.time < cutoff)
        .map(|b| Bar {
            time: b.time,
            open: b.open / k,
            high: b.high / k,
            low: b.low / k,
            close: b.close / k,
            volume: b.volume * k,
            quote_volume: b.quote_volume,
            trades: b.trades,
            taker_buy_base: b.taker_buy_base * k,
            taker_buy_quote: b.taker_buy_quote,
            complete: b.complete,
            traded: b.traded,
        })
        .collect();
    out.extend_from_slice(new);
    out.sort_by_key(|b| b.time);
    if out.windows(2).any(|w| w[1].time == w[0].time) {
        return Err("splice: overlapping timestamps remain after the cut".to_string());
    }
    Ok(out)
}

/// Apply a migration table to a set of per-symbol series.
///
/// Each migration is verified; a cycle, a missing leg or a bad factor is an
/// error. The successor keeps its name; the old series is spliced into it.
pub fn apply_migrations(
    series: &BTreeMap<String, Vec<Bar>>,
    migrations: &[Migration],
) -> Result<BTreeMap<String, Vec<Bar>>, String> {
    let mut out = series.clone();
    // Process in a deterministic order; reject chains by applying once.
    let mut sorted = migrations.to_vec();
    sorted.sort_by(|a, b| a.old.cmp(&b.old).then(a.new.cmp(&b.new)));
    let mut consumed = std::collections::BTreeSet::new();
    for m in &sorted {
        if !consumed.insert(m.old.clone()) {
            return Err(format!("migration chain touches '{}' twice", m.old));
        }
        let old = out
            .remove(&m.old)
            .ok_or_else(|| format!("migration source '{}' is not in the panel", m.old))?;
        let new = out
            .remove(&m.new)
            .ok_or_else(|| format!("migration target '{}' is not in the panel", m.new))?;
        let spliced = splice(&old, &new, m.k, m.tolerance).map_err(|e| {
            format!("{}: {}", m.old, e)
        })?;
        out.insert(m.new.clone(), spliced);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gaps and masks
// ---------------------------------------------------------------------------

/// Number of missing `tf` buckets between the first and last bar.
pub fn gap_buckets(bars: &[Bar], tf: Timeframe) -> usize {
    if bars.len() < 2 {
        return 0;
    }
    let first = tf.bucket_start(bars[0].time);
    let last = tf.bucket_start(bars[bars.len() - 1].time);
    let expected = (last - first) / tf.seconds() + 1;
    let mut present = 1i64;
    for w in bars.windows(2) {
        if tf.bucket_start(w[1].time) != tf.bucket_start(w[0].time) {
            present += 1;
        }
    }
    (expected - present).max(0) as usize
}

/// `true` when the step from `times[i-1]` to `times[i]` is longer than one
/// bucket (first row is not a gap).
pub fn is_gap_step(times: &[i64], i: usize, tf: Timeframe) -> bool {
    i > 0 && times[i] - times[i - 1] > tf.seconds()
}

/// A mask over rows: `true` when a lookback window ending at that row spans a
/// gap, so the feature must be `NaN` instead of bridging the outage.
pub fn window_gap_mask(times: &[i64], tf: Timeframe, lookback: usize) -> Vec<bool> {
    let mut mask = vec![false; times.len()];
    for i in 0..times.len() {
        let start = i.saturating_sub(lookback.saturating_sub(1));
        mask[i] = (start + 1..=i).any(|j| is_gap_step(times, j, tf));
    }
    mask
}

/// A mask over rows: `true` when the forward label window `[i, i+horizon]`
/// spans a gap. The label is then missing, not a return across the outage.
pub fn label_gap_mask(times: &[i64], tf: Timeframe, horizon: usize) -> Vec<bool> {
    let mut mask = vec![false; times.len()];
    for i in 0..times.len() {
        let end = (i + horizon).min(times.len().saturating_sub(1));
        mask[i] = (i + 1..=end).any(|j| is_gap_step(times, j, tf));
    }
    mask
}

/// Longest gap in buckets between consecutive bars, with its start index.
pub fn longest_gap(bars: &[Bar], tf: Timeframe) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for i in 1..bars.len() {
        let missing = ((bars[i].time - bars[i - 1].time) / tf.seconds() - 1).max(0) as usize;
        if missing > 0 && best.map(|(m, _)| missing > m).unwrap_or(true) {
            best = Some((missing, i - 1));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minute(t: i64, c: f64) -> Bar {
        Bar::new(t * 60, c, c + 1.0, c - 1.0, c, 1.0).with_flow(c, 1.0)
    }

    #[test]
    fn civil_roundtrip_and_weekday() {
        for days in [-10_000i64, 0, 19_723, 30_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        // 1970-01-01 was a Thursday (4).
        assert_eq!(weekday(0), 4);
        // 2024-03-10 was a Sunday.
        assert_eq!(weekday(days_from_civil(2024, 3, 10)), 0);
    }

    #[test]
    fn resampler_fold_is_associative() {
        let base: Vec<Bar> = (0..60).map(|i| minute(i, 100.0 + i as f64)).collect();
        let hourly = resample(&base, Timeframe::H1).unwrap();
        assert_eq!(hourly.len(), 1);
        assert_eq!(hourly[0].open, 100.0);
        assert_eq!(hourly[0].close, 159.0);
        assert_eq!(hourly[0].high, 160.0);
        assert_eq!(hourly[0].low, 99.0);
        assert_eq!(hourly[0].volume, 60.0);

        let mut rs = Resampler::new(Timeframe::H1);
        let mut streamed = Vec::new();
        for b in base {
            if let Some(s) = rs.push(b) {
                streamed.push(s);
            }
        }
        if let Some(s) = rs.finish() {
            streamed.push(s);
        }
        assert_eq!(streamed, hourly);
    }
}
