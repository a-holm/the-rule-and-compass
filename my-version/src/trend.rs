//! Long-term trend family (#932): daily moving-average ratios and slopes, the
//! 52-week Donchian position and distances, and 4 to 52 week momentum, each
//! ratio, slope and momentum also scaled by the trailing 20-day volatility.
//!
//! Every leaf is a function of closed UTC daily bars. A base row at time `t`
//! sees the last daily bucket with `day + 86400 <= t`, and only when that
//! bucket is the day before the day of `t`; the running day is never used.
//! Daily bars follow the `mtf_1d_*` contract: every UTC day with at least one
//! base bar is a bar built from the bars it has, a missing day is a gap, and
//! every window that spans a gap is `NaN`; no value is forward-filled.
//!
//! The device path (`mtf_gpu.rs`, `cuda/mtf.cu`) reproduces this module bit
//! for bit: the moving averages and the volatility use the compensated block
//! sums of [`crate::window_stats`], extrema are sequential window scans and the
//! ratios use plain IEEE division and subtraction.

use crate::enrich::RegistryEntry;
use crate::ohlcv::{self, Bar, Timeframe};
use crate::window_stats::WinStats;
use std::collections::BTreeMap;

pub const DAY: i64 = 86_400;
pub const MA_DAYS: [usize; 4] = [20, 50, 100, 200];
pub const MOM_WEEKS: [usize; 4] = [4, 12, 26, 52];
pub const DONCHIAN_DAYS: usize = 364;
pub const VOL_DAYS: usize = 20;

/// The 27 columns in device leaf order.
pub const TREND_NAMES: [&str; 27] = [
    "trend_ma20_ratio",
    "trend_ma50_ratio",
    "trend_ma100_ratio",
    "trend_ma200_ratio",
    "trend_ma20_slope",
    "trend_ma50_slope",
    "trend_ma100_slope",
    "trend_ma200_slope",
    "trend_don364_pos",
    "trend_hi364_dist",
    "trend_lo364_dist",
    "trend_mom4w",
    "trend_mom12w",
    "trend_mom26w",
    "trend_mom52w",
    "trend_ma20_ratio_z",
    "trend_ma50_ratio_z",
    "trend_ma100_ratio_z",
    "trend_ma200_ratio_z",
    "trend_ma20_slope_z",
    "trend_ma50_slope_z",
    "trend_ma100_slope_z",
    "trend_ma200_slope_z",
    "trend_mom4w_z",
    "trend_mom12w_z",
    "trend_mom26w_z",
    "trend_mom52w_z",
];

/// Prefix of the refusal for a base bar the daily resample cannot use; the
/// command line maps it to exit 2.
pub const BAR_REFUSAL: &str = "trend features need a base bar that divides one day";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf {
    Ratio(usize),
    Slope(usize),
    DonPos,
    HiDist,
    LoDist,
    Mom(usize),
    RatioZ(usize),
    SlopeZ(usize),
    MomZ(usize),
}

pub fn trend_names() -> Vec<String> {
    TREND_NAMES.iter().map(|s| s.to_string()).collect()
}

pub fn is_trend(name: &str) -> bool {
    leaf_index(name).is_some()
}

/// Position of `name` in [`TREND_NAMES`], the device leaf index.
pub fn leaf_index(name: &str) -> Option<usize> {
    TREND_NAMES.iter().position(|n| *n == name)
}

pub fn leaf(name: &str) -> Option<Leaf> {
    let i = leaf_index(name)?;
    Some(match i {
        0..=3 => Leaf::Ratio(MA_DAYS[i]),
        4..=7 => Leaf::Slope(MA_DAYS[i - 4]),
        8 => Leaf::DonPos,
        9 => Leaf::HiDist,
        10 => Leaf::LoDist,
        11..=14 => Leaf::Mom(MOM_WEEKS[i - 11]),
        15..=18 => Leaf::RatioZ(MA_DAYS[i - 15]),
        19..=22 => Leaf::SlopeZ(MA_DAYS[i - 19]),
        _ => Leaf::MomZ(MOM_WEEKS[i - 23]),
    })
}

/// Slope lag `L = N / 5` of the `N`-day average.
pub fn slope_lag(n: usize) -> usize {
    n / 5
}

fn raw_first_index(l: Leaf) -> usize {
    match l {
        Leaf::Ratio(n) | Leaf::RatioZ(n) => n - 1,
        Leaf::Slope(n) | Leaf::SlopeZ(n) => n - 1 + slope_lag(n),
        Leaf::DonPos | Leaf::HiDist | Leaf::LoDist => DONCHIAN_DAYS - 1,
        Leaf::Mom(k) | Leaf::MomZ(k) => 7 * k,
    }
}

/// First daily index with a finite value on a series without missing days. A scaled
/// leaf also needs the 20 daily returns of its volatility (21 closes).
pub fn first_finite_day(name: &str) -> Option<usize> {
    let l = leaf(name)?;
    let raw = raw_first_index(l);
    Some(match l {
        Leaf::RatioZ(_) | Leaf::SlopeZ(_) | Leaf::MomZ(_) => raw.max(VOL_DAYS),
        _ => raw,
    })
}

/// Span `T` in days of the scaled quantity: `N` for a ratio or slope, `7k`
/// for momentum.
pub fn z_span_days(l: Leaf) -> Option<usize> {
    match l {
        Leaf::RatioZ(n) | Leaf::SlopeZ(n) => Some(n),
        Leaf::MomZ(k) => Some(7 * k),
        _ => None,
    }
}

/// Lookback rule `rcsr-lookback/1` for a trend column: the first finite daily
/// index plus the closed day and the join lag, in seconds.
pub fn lookback_seconds(name: &str) -> Option<i64> {
    first_finite_day(name).map(|d| (d as i64 + 2) * DAY)
}

pub fn formula(name: &str) -> Option<String> {
    let l = leaf(name)?;
    let z = |raw: String, t: usize| format!("({}) / (s20_j * sqrt({})); s20 = population sd of 20 one-day simple returns; s20 = 0 is NaN", raw, t);
    Some(match l {
        Leaf::Ratio(n) => format!("D[j].c / SMA{}(j) - 1", n),
        Leaf::Slope(n) => format!("SMA{}(j) / SMA{}(j-{}) - 1", n, n, slope_lag(n)),
        Leaf::DonPos => "(c - min low 364) / (max high 364 - min low 364) - 1/2".to_string(),
        Leaf::HiDist => "c / max high 364 - 1".to_string(),
        Leaf::LoDist => "c / min low 364 - 1".to_string(),
        Leaf::Mom(k) => format!("c[j] / c[j-{}] - 1", 7 * k),
        Leaf::RatioZ(n) => z(format!("D[j].c / SMA{}(j) - 1", n), n),
        Leaf::SlopeZ(n) => z(format!("SMA{}(j) / SMA{}(j-{}) - 1", n, n, slope_lag(n)), n),
        Leaf::MomZ(k) => z(format!("c[j] / c[j-{}] - 1", 7 * k), 7 * k),
    })
}

pub const PIT_RULE: &str = "closed UTC daily bars only: day + 86400 <= t and the day before the day of t; \
                            the running day is never used; a day with any base bar is a daily bar (as mtf_1d_*); \
                            a missing day makes every window containing it NaN; no forward fill";

pub fn parity_tier(name: &str) -> &'static str {
    if name.ends_with("_z") {
        "1e-12"
    } else {
        "bitwise"
    }
}

/// Registry entries of the 27 columns. They are not part of the default
/// financial feature set; `registry::lookup` resolves them next to it.
pub fn registry() -> Vec<RegistryEntry> {
    TREND_NAMES
        .iter()
        .map(|name| RegistryEntry {
            name: name.to_string(),
            family: "trend",
            timeframe: "1d",
            warmup: first_finite_day(name).unwrap_or(0),
            available: true,
            derived: false,
            parity_tier: parity_tier(name),
            formula: format!("resample to UTC days; {}; causal close join", formula(name).unwrap_or_default()),
            source: "base OHLC bars resampled to UTC days",
            pit: PIT_RULE,
            reason: None,
        })
        .collect()
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Base rows before the first finite value of a series of hourly bars that
/// starts at 00:00 UTC.
pub fn first_finite_hourly_row(name: &str) -> Option<usize> {
    first_finite_day(name).map(|d| (d + 1) * 24)
}

/// The `columns` array of `rcsr schema` `gpu_feature_parity.trend`.
pub fn schema_columns_json() -> String {
    let rows: Vec<String> = registry()
        .iter()
        .map(|e| {
            format!(
                "{{\"name\":{},\"family\":{},\"timeframe\":{},\"warmup\":{},\"lookback_seconds\":{},\
                 \"parity_tier\":{},\"formula\":{},\"pit\":{}}}",
                jstr(&e.name),
                jstr(e.family),
                jstr(e.timeframe),
                e.warmup,
                lookback_seconds(&e.name).unwrap_or(0),
                jstr(e.parity_tier),
                jstr(&e.formula),
                jstr(e.pit)
            )
        })
        .collect();
    format!("[{}]", rows.join(","))
}

/// The checked-in `docs/trend-inventory.json`: one row per column.
pub fn inventory_json() -> String {
    let rows: Vec<String> = registry()
        .iter()
        .map(|e| {
            let tests: Vec<&str> = crate::registry::test_ids(e);
            format!(
                "    {{\"name\":{},\"family\":{},\"timeframe\":{},\"warmup_days\":{},\"first_finite_row_1h\":{},\
                 \"lookback_seconds\":{},\"pit_rule\":{},\"dtype\":\"f64\",\"unit\":\"dimensionless\",\
                 \"implementation\":{},\"parity_tier\":{},\"source\":{},\"test_id\":{}}}",
                jstr(&e.name),
                jstr(e.family),
                jstr(e.timeframe),
                e.warmup,
                first_finite_hourly_row(&e.name).unwrap_or(0),
                lookback_seconds(&e.name).unwrap_or(0),
                jstr(e.pit),
                jstr(&e.formula),
                jstr(e.parity_tier),
                jstr(e.source),
                jstr(&tests.join("; "))
            )
        })
        .collect();
    format!(
        "{{\n  \"family\": \"trend\",\n  \"count\": {},\n  \"default_set\": false,\n  \"columns\": [\n{}\n  ]\n}}\n",
        rows.len(),
        rows.join(",\n")
    )
}

pub fn unknown_name(name: &str) -> String {
    format!("unknown trend feature '{}' (the trend family is: {})", name, TREND_NAMES.join(", "))
}

/// Refuse a `trend_*` item of a `--features` list that is not one of the 27.
pub fn check_spec(spec: &str) -> Result<(), String> {
    for item in spec.split(',').map(str::trim) {
        if item.starts_with("trend_") && !is_trend(item) {
            return Err(unknown_name(item));
        }
    }
    Ok(())
}

/// The base bar check of the daily resample: the bar must divide one day.
pub fn check_bar(bar_seconds: i64) -> Result<usize, String> {
    if bar_seconds <= 0 || bar_seconds > DAY || DAY % bar_seconds != 0 {
        return Err(format!(
            "{} (86400 s); the base bar is {} s. Resample the store to a bar of at most one day that divides 86400 s",
            BAR_REFUSAL, bar_seconds
        ));
    }
    Ok((DAY / bar_seconds) as usize)
}

/// Daily leaves of one symbol, `[leaf][day]` in [`TREND_NAMES`] order.
///
/// `times` are the day starts of the daily bars (a missing day is a gap); a
/// non-finite value is missing. The windows are the `mtf_1d_*` ones: a moving
/// average or extremum needs every bar of its contiguous window, a momentum
/// its two contiguous ends.
pub fn daily_leaves(times: &[i64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Vec<f64>> {
    let n = times.len();
    let mask = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| if v.is_finite() { *v } else { f64::NAN }).collect() };
    let c = mask(close);
    let hi = mask(high);
    let lo = mask(low);
    let contig = |j: usize, w: usize| j + 1 >= w && times[j + 1 - w] == times[j] - (w as i64 - 1) * DAY;
    let ws = WinStats::new(&c, times, DAY);
    let sma = |j: usize, w: usize| if contig(j, w) { ws.mean_strict(j, w) } else { None };
    let mut ret1 = vec![f64::NAN; n];
    for j in 1..n {
        if times[j] - times[j - 1] == DAY && c[j].is_finite() && c[j - 1].is_finite() && c[j - 1] > 0.0 {
            ret1[j] = c[j] / c[j - 1] - 1.0;
        }
    }
    let rs = WinStats::new(&ret1, times, DAY);
    let s20: Vec<f64> = (0..n)
        .map(|j| if contig(j, VOL_DAYS) { rs.std_pop(j, VOL_DAYS).unwrap_or(f64::NAN) } else { f64::NAN })
        .collect();
    let mut out = vec![vec![f64::NAN; n]; TREND_NAMES.len()];
    for j in 0..n {
        for (k, &w) in MA_DAYS.iter().enumerate() {
            if let Some(m) = sma(j, w) {
                if m > 0.0 {
                    out[k][j] = c[j] / m - 1.0;
                }
            }
            let l = slope_lag(w);
            if contig(j, w + l) {
                if let (Some(now), Some(prev)) = (sma(j, w), sma(j - l, w)) {
                    if prev > 0.0 {
                        out[4 + k][j] = now / prev - 1.0;
                    }
                }
            }
        }
        if contig(j, DONCHIAN_DAYS) && c[j].is_finite() {
            let mut mx = f64::NEG_INFINITY;
            let mut mn = f64::INFINITY;
            let mut finite = true;
            for q in j + 1 - DONCHIAN_DAYS..=j {
                if !hi[q].is_finite() || !lo[q].is_finite() {
                    finite = false;
                    break;
                }
                mx = mx.max(hi[q]);
                mn = mn.min(lo[q]);
            }
            if finite {
                if mx > mn {
                    out[8][j] = (c[j] - mn) / (mx - mn) - 0.5;
                }
                if mx > 0.0 {
                    out[9][j] = c[j] / mx - 1.0;
                }
                if mn > 0.0 {
                    out[10][j] = c[j] / mn - 1.0;
                }
            }
        }
        for (k, &weeks) in MOM_WEEKS.iter().enumerate() {
            let span = 7 * weeks;
            if contig(j, span + 1) && c[j].is_finite() && c[j - span].is_finite() && c[j - span] > 0.0 {
                out[11 + k][j] = c[j] / c[j - span] - 1.0;
            }
        }
        let s = s20[j];
        if s.is_finite() && s > 0.0 {
            for k in 0..4 {
                let t = MA_DAYS[k] as f64;
                out[15 + k][j] = out[k][j] / (s * t.sqrt());
                out[19 + k][j] = out[4 + k][j] / (s * t.sqrt());
                let tm = (7 * MOM_WEEKS[k]) as f64;
                out[23 + k][j] = out[11 + k][j] / (s * tm.sqrt());
            }
        }
    }
    out
}

/// Every requested trend column of one symbol, joined onto its base rows.
///
/// A base row takes the daily value of the last bucket that closed at or
/// before it, and only when that bucket is the day before its own day; any
/// other row is `NaN`.
pub fn compute(base: &[Bar], bar_seconds: i64, requested: &[String]) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let wanted: Vec<(&str, usize)> = requested
        .iter()
        .filter_map(|n| leaf_index(n).map(|i| (n.as_str(), i)))
        .collect();
    let mut out = BTreeMap::new();
    if wanted.is_empty() {
        return Ok(out);
    }
    check_bar(bar_seconds)?;
    let tf = Timeframe::D1;
    let days = ohlcv::resample(base, tf)?;
    let times: Vec<i64> = days.iter().map(|d| d.time).collect();
    let high: Vec<f64> = days.iter().map(|d| d.high).collect();
    let low: Vec<f64> = days.iter().map(|d| d.low).collect();
    let close: Vec<f64> = days.iter().map(|d| d.close).collect();
    let leaves = daily_leaves(&times, &high, &low, &close);
    let joined = ohlcv::join_causal(base, &days, tf)?;
    let index: BTreeMap<i64, usize> = times.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    let slot: Vec<Option<usize>> = joined
        .iter()
        .zip(base)
        .map(|(j, b)| match j {
            Some(d) if d.time == tf.bucket_start(b.time) - DAY => index.get(&d.time).copied(),
            _ => None,
        })
        .collect();
    for (name, i) in wanted {
        let values = slot.iter().map(|s| s.map_or(f64::NAN, |j| leaves[i][j])).collect();
        out.insert(name.to_string(), values);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_family_has_27_unique_names_in_leaf_order() {
        let mut names = trend_names();
        assert_eq!(names.len(), 27);
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 27);
        for (i, n) in TREND_NAMES.iter().enumerate() {
            assert_eq!(leaf_index(n), Some(i));
            assert!(is_trend(n));
        }
        assert!(!is_trend("trend_ma30_ratio"));
        assert!(!is_trend("trend96"));
    }

    #[test]
    fn warm_up_table_matches_the_issue() {
        let want = [
            ("trend_ma20_ratio", 19),
            ("trend_ma50_ratio", 49),
            ("trend_ma100_ratio", 99),
            ("trend_ma200_ratio", 199),
            ("trend_ma20_slope", 23),
            ("trend_ma50_slope", 59),
            ("trend_ma100_slope", 119),
            ("trend_ma200_slope", 239),
            ("trend_don364_pos", 363),
            ("trend_hi364_dist", 363),
            ("trend_lo364_dist", 363),
            ("trend_mom4w", 28),
            ("trend_mom12w", 84),
            ("trend_mom26w", 182),
            ("trend_mom52w", 364),
            ("trend_ma20_ratio_z", 20),
            ("trend_ma200_ratio_z", 199),
            ("trend_ma20_slope_z", 23),
            ("trend_mom4w_z", 28),
            ("trend_mom52w_z", 364),
        ];
        for (name, day) in want {
            assert_eq!(first_finite_day(name), Some(day), "{name}");
        }
        assert_eq!(first_finite_hourly_row("trend_ma200_ratio"), Some(4_800));
        assert_eq!(first_finite_hourly_row("trend_ma200_slope"), Some(5_760));
        assert_eq!(first_finite_hourly_row("trend_don364_pos"), Some(8_736));
        assert_eq!(first_finite_hourly_row("trend_mom52w"), Some(8_760));
        assert_eq!(first_finite_hourly_row("trend_mom4w"), Some(696));
        assert_eq!(first_finite_hourly_row("trend_ma20_ratio_z"), Some(504));
    }

    #[test]
    fn lookback_is_first_finite_day_plus_two_days() {
        assert_eq!(lookback_seconds("trend_mom52w"), Some(31_622_400));
        assert_eq!(lookback_seconds("trend_ma20_ratio"), Some(21 * DAY));
        assert_eq!(lookback_seconds("trend_ma200_slope"), Some(241 * DAY));
        let max = TREND_NAMES.iter().filter_map(|n| lookback_seconds(n)).max();
        assert_eq!(max, Some(366 * DAY));
        assert_eq!(lookback_seconds("mtf_1d_vol"), None);
    }

    #[test]
    fn leaf_formulas_on_a_short_flat_then_rising_series() {
        // 400 days, close 100 for the first 380 days, then +1 per day.
        let n = 400usize;
        let times: Vec<i64> = (0..n as i64).map(|d| d * DAY).collect();
        let close: Vec<f64> = (0..n).map(|d| if d < 380 { 100.0 } else { 100.0 + (d - 379) as f64 }).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let v = daily_leaves(&times, &high, &low, &close);
        // Flat window: every ratio, slope and momentum is exactly zero; the
        // volatility is zero, so the scaled leaves are NaN.
        let j = 379;
        for k in 0..15 {
            if (8..=10).contains(&k) {
                continue;
            }
            assert_eq!(v[k][j], 0.0, "{} at {j}", TREND_NAMES[k]);
        }
        for k in 15..27 {
            assert!(v[k][j].is_nan(), "{} must be NaN when s20 is 0", TREND_NAMES[k]);
        }
        assert_eq!(v[8][j], 0.0);
        assert_eq!(v[9][j], 100.0 / 102.0 - 1.0);
        assert_eq!(v[10][j], 100.0 / 98.0 - 1.0);
        // After the rise: momentum over 4 weeks reads 28 days back.
        let j = 399;
        assert_eq!(v[11][j], close[j] / close[j - 28] - 1.0);
        assert!(v[23][j].is_finite());
    }

    #[test]
    fn a_missing_day_masks_every_window_that_spans_it() {
        // Day 100 is missing: index 100 holds day 101.
        let times: Vec<i64> = (0..300i64).filter(|d| *d != 100).map(|d| d * DAY).collect();
        let n = times.len();
        let close: Vec<f64> = (0..n).map(|d| 100.0 + (d as f64 * 0.3).sin()).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let v = daily_leaves(&times, &high, &low, &close);
        // ma20_ratio: windows ending on days 101..=119 (indices 100..=118) span the gap.
        for j in 100..119 {
            assert!(v[0][j].is_nan(), "ma20_ratio at {j}");
        }
        assert!(v[0][99].is_finite() && v[0][119].is_finite());
        // mom4w: the end 28 days back is missing or not contiguous up to day 128.
        for j in 100..128 {
            assert!(v[11][j].is_nan(), "mom4w at {j}");
        }
        assert!(v[11][128].is_finite());
    }

    #[test]
    fn a_bar_that_does_not_divide_a_day_is_refused() {
        assert_eq!(check_bar(3600), Ok(24));
        assert_eq!(check_bar(86_400), Ok(1));
        assert!(check_bar(172_800).unwrap_err().starts_with(BAR_REFUSAL));
        assert!(check_bar(7_000).unwrap_err().starts_with(BAR_REFUSAL));
        assert!(check_bar(0).is_err());
    }
}
