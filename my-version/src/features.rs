//! #125: causal feature and label libraries for the OHLCV layer.
//!
//! Every feature is a function of bars strictly at or before its own row. A
//! feature whose lookback window spans a gap is `NaN`, never a value bridged
//! across the outage. Warm-up rows are `NaN` and the maximum warm-up is
//! reported so a builder can drop them and print the first usable timestamp.
//!
//! Feature groups:
//!
//! * `ta:core` — returns, moving-average ratios, Bollinger z, ATR, RSI,
//!   volume ratio;
//! * `ta:full` — the core plus NATR, ADX/DI, CMF, MFI and VWAP deviation;
//! * `exotic:cheap` — realised moments and range/close-position;
//! * `candle:geometry` — body and wick geometry, opening gap;
//! * `cross:btc` — relative return and beta versus a supplied benchmark;
//! * `calendar` — UTC day-of-week and hour cycles.
//!
//! Labels enter at the **next open**, exit at the close `horizon` bars later,
//! subtract a symmetric round-trip cost, optionally divide by trailing
//! volatility and can be strided. A label that spans a gap is missing.

use crate::ohlcv::{label_gap_mask, window_gap_mask, Bar, Timeframe};

/// True when the `bars` rows ending at `i` form one contiguous run at the
/// timeframe spacing. A window that spans a missing bucket is never valid.
fn contiguous(times: &[i64], i: usize, bars: usize, tf: Timeframe) -> bool {
    if bars == 0 {
        return true;
    }
    if i + 1 < bars {
        return false;
    }
    times[i + 1 - bars] == times[i] - (bars as i64 - 1) * tf.seconds()
}

/// Mask every value whose lookback window is not contiguous.
fn mask_window(v: &mut [f64], times: &[i64], tf: Timeframe, bars: usize) {
    for (i, x) in v.iter_mut().enumerate() {
        if !contiguous(times, i, bars, tf) {
            *x = f64::NAN;
        }
    }
}

/// One computed feature column.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureColumn {
    pub name: String,
    pub values: Vec<f64>,
    /// Number of leading rows that cannot be computed.
    pub warmup: usize,
}

/// One computed label column.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelColumn {
    pub name: String,
    pub values: Vec<f64>,
    pub warmup: usize,
    /// Rows dropped because the label window spans a gap.
    pub gap_rows: usize,
    /// Rows dropped because of `stride`.
    pub stride_rows: usize,
}

/// Expand a comma-separated feature specification. Unknown names and groups
/// are refused rather than silently ignored.
pub fn expand(spec: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        let names: Vec<String> = match item {
            "ta:core" => core_names(),
            "ta:full" => full_names(),
            "exotic:cheap" => exotic_names(),
            "candle:geometry" => candle_names(),
            "cross:btc" => cross_names(),
            "calendar" => calendar_names(),
            other => {
                if known(other) {
                    vec![other.to_string()]
                } else {
                    return Err(format!(
                        "unknown feature '{}' (groups: ta:core, ta:full, exotic:cheap, \
                         candle:geometry, cross:btc, calendar)",
                        other
                    ));
                }
            }
        };
        for n in names {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    if out.is_empty() {
        return Err("empty feature specification".to_string());
    }
    Ok(out)
}

fn core_names() -> Vec<String> {
    ["ret1", "ret5", "ma20_ratio", "ma60_ratio", "bb20_z", "atr14", "rsi14", "vol20_ratio"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn full_names() -> Vec<String> {
    let mut v = core_names();
    for n in ["natr14", "adx14", "di_plus", "di_minus", "cmf20", "mfi14", "vwap20_dev"] {
        v.push(n.to_string());
    }
    v
}

fn exotic_names() -> Vec<String> {
    ["rmean5", "rstd5", "skew20", "kurt20", "range1", "close_pos20"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn candle_names() -> Vec<String> {
    ["body1", "upper_wick1", "lower_wick1", "open_gap1"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn cross_names() -> Vec<String> {
    ["cross_ret1", "cross_beta20"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn calendar_names() -> Vec<String> {
    ["dow_sin", "dow_cos", "hour_sin", "hour_cos"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn known(name: &str) -> bool {
    core_names().contains(&name.to_string())
        || full_names().contains(&name.to_string())
        || exotic_names().contains(&name.to_string())
        || candle_names().contains(&name.to_string())
        || cross_names().contains(&name.to_string())
        || calendar_names().contains(&name.to_string())
}

fn returns(bars: &[Bar], tf: Timeframe) -> Vec<f64> {
    let n = bars.len();
    let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
    let mut r = vec![f64::NAN; n];
    for i in 1..n {
        if contiguous(&times, i, 2, tf) && bars[i - 1].close > 0.0 && bars[i].close > 0.0 {
            r[i] = (bars[i].close / bars[i - 1].close).ln();
        }
    }
    r
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

fn std(x: &[f64], sample: bool) -> f64 {
    let n = x.len();
    let m = mean(x);
    if n < 2 || !m.is_finite() {
        return f64::NAN;
    }
    let denom = if sample { n - 1 } else { n };
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / denom as f64).sqrt()
}

fn skew(x: &[f64]) -> f64 {
    let n = x.len();
    let m = mean(x);
    let s = std(x, false);
    if n < 3 || !s.is_finite() || s == 0.0 {
        return f64::NAN;
    }
    x.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>() / n as f64
}

fn kurt(x: &[f64]) -> f64 {
    let n = x.len();
    let m = mean(x);
    let s = std(x, false);
    if n < 4 || !s.is_finite() || s == 0.0 {
        return f64::NAN;
    }
    x.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>() / n as f64 - 3.0
}

fn rolling_mean(bars: &[Bar], i: usize, w: usize, f: impl Fn(&Bar) -> f64) -> f64 {
    if i + 1 < w {
        return f64::NAN;
    }
    mean(&bars[i + 1 - w..=i].iter().map(|b| f(b)).collect::<Vec<_>>())
}

fn rolling_std(bars: &[Bar], i: usize, w: usize, f: impl Fn(&Bar) -> f64) -> f64 {
    if i + 1 < w {
        return f64::NAN;
    }
    std(&bars[i + 1 - w..=i].iter().map(|b| f(b)).collect::<Vec<_>>(), false)
}

/// Wilder smoothing seeded with a simple mean, restarted after a gap.
fn wilder(x: &[f64], period: usize, gap: &[bool]) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    if x.len() < period {
        return out;
    }
    // Skip a leading NaN run (for example DI, whose first row is undefined)
    // and seed on the first complete window.
    let mut start = 0usize;
    while start + period <= x.len() && !x[start..start + period].iter().all(|v| v.is_finite()) {
        start += 1;
    }
    if start + period > x.len() {
        return out;
    }
    let mut seed = 0.0;
    for i in start..start + period {
        seed += x[i];
    }
    let mut prev = seed / period as f64;
    out[start + period - 1] = prev;
    for i in start + period..x.len() {
        if gap[i] || !x[i].is_finite() {
            // Restart: the next full window seeds again.
            if i + period <= x.len() && x[i..i + period].iter().all(|v| v.is_finite()) {
                let s: f64 = x[i..i + period].iter().sum();
                prev = s / period as f64;
                out[i + period - 1] = prev;
            }
            continue;
        }
        prev = (prev * (period as f64 - 1.0) + x[i]) / period as f64;
        out[i] = prev;
    }
    out
}

fn true_range(bars: &[Bar], i: usize) -> f64 {
    if i == 0 {
        return bars[0].high - bars[0].low;
    }
    let prev = bars[i - 1].close;
    (bars[i].high - bars[i].low)
        .max((bars[i].high - prev).abs())
        .max((bars[i].low - prev).abs())
}

fn atr(bars: &[Bar], period: usize, gap: &[bool]) -> Vec<f64> {
    let tr: Vec<f64> = (0..bars.len()).map(|i| true_range(bars, i)).collect();
    wilder(&tr, period, gap)
}

fn rsi(bars: &[Bar], period: usize, gap: &[bool]) -> Vec<f64> {
    let mut gain = vec![0.0f64; bars.len()];
    let mut loss = vec![0.0f64; bars.len()];
    for i in 1..bars.len() {
        let d = bars[i].close - bars[i - 1].close;
        gain[i] = d.max(0.0);
        loss[i] = (-d).max(0.0);
    }
    let ag = wilder(&gain, period, gap);
    let al = wilder(&loss, period, gap);
    ag.iter()
        .zip(al.iter())
        .map(|(g, l)| {
            if !g.is_finite() || !l.is_finite() {
                f64::NAN
            } else if *l == 0.0 {
                100.0
            } else {
                let rs = g / l;
                100.0 - 100.0 / (1.0 + rs)
            }
        })
        .collect()
}

fn cmf(bars: &[Bar], i: usize, w: usize) -> f64 {
    if i + 1 < w {
        return f64::NAN;
    }
    let mut mfv = 0.0;
    let mut vol = 0.0;
    for b in &bars[i + 1 - w..=i] {
        let range = b.high - b.low;
        let mult = if range > 0.0 {
            ((b.close - b.low) - (b.high - b.close)) / range
        } else {
            0.0
        };
        mfv += mult * b.volume;
        vol += b.volume;
    }
    if vol > 0.0 {
        mfv / vol
    } else {
        f64::NAN
    }
}

fn mfi(bars: &[Bar], i: usize, w: usize) -> f64 {
    if i + 1 < w + 1 || i == 0 {
        return f64::NAN;
    }
    let mut pos = 0.0;
    let mut neg = 0.0;
    for j in i + 1 - w..=i {
        let tp = (bars[j].high + bars[j].low + bars[j].close) / 3.0;
        let prev = (bars[j - 1].high + bars[j - 1].low + bars[j - 1].close) / 3.0;
        let flow = tp * bars[j].volume;
        if tp > prev {
            pos += flow;
        } else if tp < prev {
            neg += flow;
        }
    }
    if neg == 0.0 {
        100.0
    } else {
        let mr = pos / neg;
        100.0 - 100.0 / (1.0 + mr)
    }
}

/// Compute the requested feature columns. `cross` is required for
/// `cross:btc`-style features and is aligned by timestamp.
pub fn compute(
    bars: &[Bar],
    tf: Timeframe,
    specs: &[String],
) -> Result<Vec<FeatureColumn>, String> {
    compute_with_cross(bars, tf, specs, None)
}

pub fn compute_with_cross(
    bars: &[Bar],
    tf: Timeframe,
    specs: &[String],
    cross: Option<&[Bar]>,
) -> Result<Vec<FeatureColumn>, String> {
    let n = bars.len();
    let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
    let rets = returns(bars, tf);
    let mut out = Vec::new();
    for name in specs {
        let (values, warmup) = match name.as_str() {
            "ret1" => (rets.clone(), 1),
            "ret5" => {
                let mut v = vec![f64::NAN; n];
                for i in 5..n {
                    if contiguous(&times, i, 6, tf)
                        && bars[i - 5].close > 0.0
                        && bars[i].close > 0.0
                    {
                        v[i] = (bars[i].close / bars[i - 5].close).ln();
                    }
                }
                (v, 6)
            }
            "ma20_ratio" | "ma60_ratio" => {
                let w = if name == "ma20_ratio" { 20 } else { 60 };
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    let m = rolling_mean(bars, i, w, |b| b.close);
                    if m.is_finite() && m > 0.0 {
                        v[i] = bars[i].close / m - 1.0;
                    }
                }
                mask_window(&mut v, &times, tf, w);
                (v, w)
            }
            "bb20_z" => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    let m = rolling_mean(bars, i, 20, |b| b.close);
                    let s = rolling_std(bars, i, 20, |b| b.close);
                    if m.is_finite() && s.is_finite() && s > 0.0 {
                        v[i] = (bars[i].close - m) / s;
                    }
                }
                mask_window(&mut v, &times, tf, 20);
                (v, 20)
            }
            "atr14" | "natr14" => {
                let gap = window_gap_mask(&times, tf, 15);
                let a = atr(bars, 14, &gap);
                let mut v = a.clone();
                if name == "natr14" {
                    for i in 0..n {
                        v[i] = if a[i].is_finite() && bars[i].close > 0.0 {
                            a[i] / bars[i].close * 100.0
                        } else {
                            f64::NAN
                        };
                    }
                }
                for (i, g) in gap.iter().enumerate() {
                    if *g {
                        v[i] = f64::NAN;
                    }
                }
                (v, 15)
            }
            "rsi14" => {
                let gap = window_gap_mask(&times, tf, 15);
                (rsi(bars, 14, &gap), 15)
            }
            "adx14" | "di_plus" | "di_minus" => {
                let gap = window_gap_mask(&times, tf, 28);
                let a = atr(bars, 14, &gap);
                let mut plus = vec![f64::NAN; n];
                let mut minus = vec![f64::NAN; n];
                for i in 1..n {
                    if !a[i].is_finite() || a[i] == 0.0 {
                        continue;
                    }
                    let up = bars[i].high - bars[i - 1].high;
                    let down = bars[i - 1].low - bars[i].low;
                    let pdm = if up > down && up > 0.0 { up } else { 0.0 };
                    let mdm = if down > up && down > 0.0 { down } else { 0.0 };
                    plus[i] = 100.0 * pdm / a[i];
                    minus[i] = 100.0 * mdm / a[i];
                }
                // Smooth the DI series with Wilder's average.
                let sp = wilder(&plus, 14, &gap);
                let sm = wilder(&minus, 14, &gap);
                let mut dx = vec![f64::NAN; n];
                for i in 0..n {
                    if sp[i].is_finite() && sm[i].is_finite() && sp[i] + sm[i] > 0.0 {
                        dx[i] = 100.0 * (sp[i] - sm[i]).abs() / (sp[i] + sm[i]);
                    }
                }
                let adx = wilder(&dx, 14, &gap);
                let v = match name.as_str() {
                    "adx14" => adx,
                    "di_plus" => sp,
                    _ => sm,
                };
                let mut v = v;
                for (i, g) in gap.iter().enumerate() {
                    if *g {
                        v[i] = f64::NAN;
                    }
                }
                (v, 28)
            }
            "vol20_ratio" => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    let m = rolling_mean(bars, i, 20, |b| b.volume);
                    if m.is_finite() && m > 0.0 {
                        v[i] = bars[i].volume / m;
                    }
                }
                mask_window(&mut v, &times, tf, 20);
                (v, 20)
            }
            "cmf20" => {
                let mut v: Vec<f64> = (0..n).map(|i| cmf(bars, i, 20)).collect();
                mask_window(&mut v, &times, tf, 20);
                (v, 20)
            }
            "mfi14" => {
                let mut v: Vec<f64> = (0..n).map(|i| mfi(bars, i, 14)).collect();
                mask_window(&mut v, &times, tf, 15);
                (v, 15)
            }
            "vwap20_dev" => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    if i + 1 < 20 {
                        continue;
                    }
                    let q: f64 = bars[i + 1 - 20..=i].iter().map(|b| b.quote_volume).sum();
                    let vol: f64 = bars[i + 1 - 20..=i].iter().map(|b| b.volume).sum();
                    if vol > 0.0 {
                        let vwap = q / vol;
                        if vwap > 0.0 {
                            v[i] = bars[i].close / vwap - 1.0;
                        }
                    }
                }
                mask_window(&mut v, &times, tf, 20);
                (v, 20)
            }
            "rmean5" | "rstd5" => {
                let mut v = vec![f64::NAN; n];
                for i in 5..n {
                    let w = &rets[i + 1 - 5..=i];
                    if w.iter().all(|x| x.is_finite()) {
                        v[i] = if name == "rmean5" { mean(w) } else { std(w, false) };
                    }
                }
                (v, 6)
            }
            "skew20" | "kurt20" => {
                let mut v = vec![f64::NAN; n];
                for i in 20..n {
                    let w = &rets[i + 1 - 20..=i];
                    if w.iter().all(|x| x.is_finite()) {
                        v[i] = if name == "skew20" { skew(w) } else { kurt(w) };
                    }
                }
                (v, 21)
            }
            "range1" => (
                (0..n)
                    .map(|i| {
                        if bars[i].close > 0.0 {
                            (bars[i].high - bars[i].low) / bars[i].close
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
                0,
            ),
            "close_pos20" => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    if i + 1 < 20 {
                        continue;
                    }
                    let hi = bars[i + 1 - 20..=i]
                        .iter()
                        .map(|b| b.high)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let lo = bars[i + 1 - 20..=i]
                        .iter()
                        .map(|b| b.low)
                        .fold(f64::INFINITY, f64::min);
                    if hi > lo {
                        v[i] = (bars[i].close - lo) / (hi - lo);
                    }
                }
                mask_window(&mut v, &times, tf, 20);
                (v, 20)
            }
            "body1" | "upper_wick1" | "lower_wick1" | "open_gap1" => {
                let mut v = vec![f64::NAN; n];
                for i in 0..n {
                    let b = &bars[i];
                    let range = b.high - b.low;
                    if range <= 0.0 {
                        continue;
                    }
                    v[i] = match name.as_str() {
                        "body1" => (b.close - b.open) / range,
                        "upper_wick1" => (b.high - b.open.max(b.close)) / range,
                        "lower_wick1" => (b.open.min(b.close) - b.low) / range,
                        _ => {
                            if i == 0 || bars[i - 1].close <= 0.0 || !contiguous(&times, i, 2, tf) {
                                f64::NAN
                            } else {
                                b.open / bars[i - 1].close - 1.0
                            }
                        }
                    };
                }
                (v, 0)
            }
            "cross_ret1" | "cross_beta20" => {
                let other = cross.ok_or_else(|| {
                    format!("feature '{}' needs a cross-asset series", name)
                })?;
                if other.len() != n {
                    return Err(format!(
                        "cross series has {} bars, base series {}",
                        other.len(),
                        n
                    ));
                }
                let mut by_time = std::collections::BTreeMap::new();
                for b in other {
                    by_time.insert(b.time, b.close);
                }
                let mut v = vec![f64::NAN; n];
                let mut ri = vec![f64::NAN; n];
                for i in 0..n {
                    let (c, p) = (
                        by_time.get(&bars[i].time).copied(),
                        if i > 0 {
                            by_time.get(&bars[i - 1].time).copied()
                        } else {
                            None
                        },
                    );
                    if let (Some(c), Some(p)) = (c, p) {
                        if c > 0.0 && p > 0.0 && contiguous(&times, i, 2, tf) {
                            ri[i] = (c / p).ln();
                        }
                    }
                }
                if name == "cross_ret1" {
                    v = ri;
                } else {
                    for i in 20..n {
                        let a = &rets[i + 1 - 20..=i];
                        let b = &ri[i + 1 - 20..=i];
                        if a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()) {
                            let ma = mean(a);
                            let mb = mean(b);
                            let cov: f64 = a
                                .iter()
                                .zip(b)
                                .map(|(x, y)| (x - ma) * (y - mb))
                                .sum::<f64>()
                                / a.len() as f64;
                            let var = b.iter().map(|y| (y - mb).powi(2)).sum::<f64>()
                                / b.len() as f64;
                            if var > 0.0 {
                                v[i] = cov / var;
                            }
                        }
                    }
                    mask_window(&mut v, &times, tf, 21);
                }
                (v, 21)
            }
            "dow_sin" | "dow_cos" | "hour_sin" | "hour_cos" => {
                let mut v = vec![f64::NAN; n];
                for (i, b) in bars.iter().enumerate() {
                    let secs = b.time.rem_euclid(86_400);
                    let frac = secs as f64 / 86_400.0;
                    if name.starts_with("hour") {
                        let a = 2.0 * std::f64::consts::PI * frac;
                        v[i] = if name == "hour_sin" { a.sin() } else { a.cos() };
                    } else {
                        let days = b.time.div_euclid(86_400);
                        let dow = crate::ohlcv::weekday(days) as f64;
                        let a = 2.0 * std::f64::consts::PI * dow / 7.0;
                        v[i] = if name == "dow_sin" { a.sin() } else { a.cos() };
                    }
                }
                (v, 0)
            }
            other => return Err(format!("unknown feature '{}'", other)),
        };
        out.push(FeatureColumn {
            name: name.clone(),
            values,
            warmup,
        });
    }
    Ok(out)
}

/// Label specification: next-open entry, close exit, symmetric cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelSpec {
    /// Forward horizon in bars.
    pub horizon: usize,
    /// Cost per side in basis points; the label subtracts `2 * cost_bps`.
    pub cost_bps: f64,
    /// Trailing volatility window; 0 disables normalisation.
    pub vol_window: usize,
    /// Keep every `stride`-th label only; 1 keeps all.
    pub stride: usize,
}

impl Default for LabelSpec {
    fn default() -> Self {
        LabelSpec {
            horizon: 1,
            cost_bps: 10.0,
            vol_window: 0,
            stride: 1,
        }
    }
}

impl LabelSpec {
    /// Parse `fwd:1h,cost=10bp,volnorm=96,stride=1`. The horizon may be a
    /// timeframe name (resolved against `tf`) or an integer bar count.
    pub fn parse(src: &str, tf: Timeframe) -> Result<Self, String> {
        let mut spec = LabelSpec::default();
        let mut horizon_set = false;
        for part in src.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(h) = part.strip_prefix("fwd:") {
                let h = h.trim();
                if let Ok(n) = h.parse::<usize>() {
                    spec.horizon = n;
                } else {
                    let htf = Timeframe::parse(h)?;
                    let n = htf.seconds() / tf.seconds();
                    if n <= 0 {
                        return Err(format!(
                            "label horizon {} is shorter than the base timeframe {}",
                            h,
                            tf.name()
                        ));
                    }
                    spec.horizon = n as usize;
                }
                horizon_set = true;
            } else if let Some(v) = part.strip_prefix("cost=") {
                let v = v.trim();
                let (num, bps) = if let Some(x) = v.strip_suffix("bp") {
                    (x, true)
                } else if let Some(x) = v.strip_suffix('%') {
                    (x, false)
                } else {
                    (v, true)
                };
                let n: f64 = num
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad cost '{}'", part))?;
                spec.cost_bps = if bps { n } else { n * 100.0 };
            } else if let Some(v) = part.strip_prefix("volnorm=") {
                spec.vol_window = v
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad volnorm '{}'", part))?;
            } else if let Some(v) = part.strip_prefix("side=") {
                crate::labels::check_side(v.trim())?;
            } else if let Some(v) = part.strip_prefix("stride=") {
                spec.stride = v
                    .trim()
                    .parse()
                    .map_err(|_| format!("bad stride '{}'", part))?;
                if spec.stride == 0 {
                    return Err("stride must be >= 1".to_string());
                }
            } else {
                return Err(format!("unknown label option '{}'", part));
            }
        }
        if !horizon_set {
            return Err("label spec needs 'fwd:<horizon>'".to_string());
        }
        if spec.cost_bps < 0.0 || !spec.cost_bps.is_finite() {
            return Err("cost must be finite and non-negative".to_string());
        }
        Ok(spec)
    }

    /// A stride below the horizon overlaps labels; from four hours on that
    /// needs `--allow-overlap`.
    pub fn check_overlap(&self, tf: Timeframe, allow_overlap: bool) -> Result<(), String> {
        let seconds = self.horizon as i64 * tf.seconds();
        if self.stride < self.horizon && seconds >= crate::labels::OVERLAP_GUARD_SECONDS && !allow_overlap {
            return Err(format!(
                "label fwd:{}: stride {} is below the {}-bar horizon, so labels overlap; use stride>={} or pass --allow-overlap",
                self.horizon, self.stride, self.horizon, self.horizon
            ));
        }
        Ok(())
    }
}

/// Causal forward label: `log(close[i+h] / open[i+1]) - 2 * cost`, divided by
/// trailing volatility when `vol_window > 0`. Entering at the next open is why
/// index `i` never uses its own close as the entry.
pub fn forward_label(bars: &[Bar], tf: Timeframe, spec: &LabelSpec) -> LabelColumn {
    let n = bars.len();
    let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
    let gap = label_gap_mask(&times, tf, spec.horizon);
    let rets = returns(bars, tf);
    let cost = 2.0 * spec.cost_bps / 10_000.0;
    let mut values = vec![f64::NAN; n];
    let mut gap_rows = 0usize;
    let mut stride_rows = 0usize;
    for i in 0..n {
        if i + spec.horizon >= n {
            continue;
        }
        if gap[i] {
            gap_rows += 1;
            continue;
        }
        if spec.stride > 1 && i % spec.stride != 0 {
            stride_rows += 1;
            continue;
        }
        let entry = bars[i + 1].open;
        let exit = bars[i + spec.horizon].close;
        if !(entry > 0.0 && exit > 0.0) {
            continue;
        }
        let mut y = (exit / entry).ln() - cost;
        if spec.vol_window > 1 {
            let start = (i + 1).saturating_sub(spec.vol_window);
            let w = &rets[start..=i];
            if w.iter().all(|v| v.is_finite()) {
                let s = std(w, false);
                if s > 0.0 {
                    y /= s;
                } else {
                    y = f64::NAN;
                }
            } else {
                y = f64::NAN;
            }
        }
        values[i] = y;
    }
    let warmup = 1 + spec
        .vol_window
        .saturating_sub(1)
        .max(if spec.horizon > 1 { 1 } else { 0 });
    LabelColumn {
        name: format!("fwd{}", spec.horizon),
        values,
        warmup,
        gap_rows,
        stride_rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let p = 100.0 + i as f64;
                Bar::new(i as i64 * 60, p, p + 1.0, p - 1.0, p, 1.0).with_flow(p, 1.0)
            })
            .collect()
    }

    #[test]
    fn expand_refuses_an_unknown_name() {
        assert!(expand("ta:core").unwrap().contains(&"ret1".to_string()));
        assert!(expand("nope").is_err());
    }

    #[test]
    fn features_are_causal_and_report_warmup() {
        let b = bars(100);
        let cols = compute(&b, Timeframe::M1, &expand("ta:core").unwrap()).unwrap();
        for c in &cols {
            for v in c.values.iter().take(c.warmup.min(5)) {
                assert!(v.is_nan(), "{} has a value in warm-up", c.name);
            }
            assert!(c.values.iter().any(|v| v.is_finite()));
        }
    }

    #[test]
    fn label_enters_next_open_and_charges_both_sides() {
        let b = bars(10);
        let spec = LabelSpec {
            horizon: 1,
            cost_bps: 10.0,
            vol_window: 0,
            stride: 1,
        };
        let l = forward_label(&b, Timeframe::M1, &spec);
        // entry = open[1] = 101, exit = close[1] = 101, gross 0, cost 20bp.
        assert!((l.values[0] - (-0.002)).abs() < 1e-12);
    }

    #[test]
    fn a_gap_masks_the_label_instead_of_bridging_it() {
        let mut b = bars(10);
        b[5].time += 60 * 10; // ten-minute hole
        let spec = LabelSpec {
            horizon: 2,
            ..Default::default()
        };
        let l = forward_label(&b, Timeframe::M1, &spec);
        assert!(l.values[3].is_nan(), "label spans the gap");
        assert!(l.values[7].is_finite(), "label after the gap is valid");
        assert!(l.gap_rows > 0);
    }

    #[test]
    fn cross_feature_needs_a_matching_series() {
        let b = bars(50);
        assert!(compute(&b, Timeframe::M1, &expand("cross:btc").unwrap()).is_err());
        let c = compute_with_cross(
            &b,
            Timeframe::M1,
            &expand("cross:btc").unwrap(),
            Some(&b),
        )
        .unwrap();
        assert_eq!(c.len(), 2);
    }
}
