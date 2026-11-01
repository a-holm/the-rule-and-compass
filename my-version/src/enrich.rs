//! `rcsr enrich` — append pre-computed OHLCV features to an existing store
//! (issue #200).
//!
//! The store is read **column by column, one symbol at a time**. Every feature
//! column is written to its own temporary `f64` file in row order, and the
//! output store is produced by copying the existing value/code blocks verbatim
//! and appending the new column blocks ([`crate::store::write_store_appending`]).
//! The whole panel is never rebuilt as a row-major table, and no existing
//! column is decoded and re-encoded.
//!
//! # Causality and gaps
//!
//! A feature at row `i` reads only bars at or before `i`. A window is valid
//! only when every bucket between its first and last bar is present at the
//! inferred or requested bar spacing; a window that spans a listing,
//! delisting or migration hole is `NaN`. Wilder/EMA state restarts after a
//! gap rather than carrying across it. Warm-up rows are `NaN` and the report
//! records the nominal warm-up, the observed `NaN` count and the first finite
//! row per column.
//!
//! # Determinism
//!
//! Feature values are computed in a fixed order and hashed while they are
//! written. Changing `--chunk-rows` changes only read granularity, never the
//! arithmetic, so the per-column hashes and the output store checksum are
//! stable. Cross-sectional ranks break ties by symbol name, then row order.
//!
//! # Backends
//!
//! The CPU path is always available. A cost model
//! ([`plan_groups`]) compares an estimated CPU time with an estimated GPU time
//! (resident upload + transfer + kernel) for every feature family and records
//! the decision in the report. CUDA builds register exact f64 kernels for the
//! elementwise, rolling, extrema, Wilder/EMA, RSI, label and rank families; a
//! family is dispatched only when its estimated GPU time is lower (or when
//! `--gpu on` forces it). Small stores, unsupported #125 names and any family
//! that does not repay the upload stay on the CPU with the reason recorded.
//! CPU-only builds behave exactly as before. See `docs/ohlcv.md`.

use crate::features::{self, LabelSpec};
use crate::manifest::{self, Sha256Hasher};
use crate::ohlcv::{
    civil_from_days, join_causal, resample, weekday, Bar, Timeframe,
};
use crate::store::{self, AppendColumn, DType, FileStore};
use crate::window_stats::{self, WinStats};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const EXIT_USAGE: i32 = 2;
const EXIT_INPUT: i32 = 3;

// ---------------------------------------------------------------------------
// Feature catalogue
// ---------------------------------------------------------------------------

/// Group names accepted by `--features` in addition to individual names.
pub const GROUPS: &[&str] = &[
    "returns",
    "ma",
    "vol",
    "atr",
    "rsi",
    "bollinger",
    "vz",
    "range",
    "candle",
    "donchian",
    "calendar",
    "lags",
    "xs",
    "btc",
    "cross",
    "py15m",
    "mtf",
    "compact",
    "flow",
    "price_action",
    "all",
];

/// The 52 compact-panel feature names, in the exact compact order.
///
/// These are computed at the **15 m base** by [`compact_all`]. The hourly compact
/// panel is the projection `ts % 3600 == 0` of this output; windows are never
/// recomputed on hourly bars.
pub const COMPACT_NAMES: &[&str] = &[
    "r16", "r96", "r672", "ma96", "ma672", "ma_slope672", "macd_hist",
    "natr14", "vol16", "vol96", "vol_ts", "vol_ratio", "vol_z2880",
    "parkinson96", "rv_ratio", "rsi14", "bb96", "bb_width96", "adx14",
    "di_diff14", "cmf20", "mfi14", "vwap_dev96", "vol_z96", "taker_ratio96",
    "taker_imb16", "trade_intensity96", "clv1", "clv4", "body1", "batr",
    "wasym1", "lower_wick", "rng1", "volrat", "gapatr", "ou_hl96", "acf1",
    "vr4", "rskew96", "jump_share", "semivar_asym", "cusum_abs_z", "btc_r96",
    "btc_vol96", "rel_btc96", "beta384", "tod_sin", "tod_cos", "dow_sin",
    "dow_cos", "don96",
];

pub fn compact_names() -> Vec<String> {
    COMPACT_NAMES.iter().map(|s| s.to_string()).collect()
}

/// The complete OHLCV/flow family: the derived VWAP plus the nine derived flow
/// columns of issue #270. The raw ten OHLCV/flow fields are input columns and
/// are registered separately; `vwap` may be either an input column (built by
/// `rcsr prep`, #259) or materialised by this group.
///
/// Every window is trailing and inclusive of `t`, in 15 m bars. A window that
/// spans a gap is `NaN`; `OBV`/`CVD` cumulative state restarts after a gap so a
/// listing/delisting/migration boundary is never bridged. Zero volume or zero
/// trades gives `NaN` for the affected ratios and illiquidity (never `0`).
pub const FLOW_NAMES: &[&str] = &[
    "vwap",
    "taker_ratio96",
    "taker_imb16",
    "trade_intensity96",
    "avg_trade_size96",
    "amihud96",
    "zero_ret_share96",
    "obv_z96",
    "cvd_proxy",
    "vwap_dev96",
];

pub fn flow_names() -> Vec<String> {
    FLOW_NAMES.iter().map(|s| s.to_string()).collect()
}

/// Higher timeframes materialised by the multi-timeframe family.
pub const MTF_TIMEFRAMES: &[&str] = &["1h", "4h", "1d"];

/// The twelve feature names per higher timeframe.
///
/// Their windows count **higher-timeframe bars**, not base bars. Every window
/// requires contiguous higher buckets; a gap makes the window `NaN`.
pub const MTF_LEAVES: &[&str] = &[
    "ret1",
    "ret4",
    "ret24",
    "ma_ratio",
    "vol",
    "atr_pct",
    "rsi",
    "close_pos",
    "donchian_pos",
    "range_pct",
    "trend_slope",
    "bar_age",
];

/// Multi-timeframe context columns from the default financial feature set.
///
/// The names are `mtf_<timeframe>_<leaf>` (`mtf_1h_ret1`, ..., `mtf_1d_bar_age`),
/// twelve leaves for each of [`MTF_TIMEFRAMES`] — 36 columns in total. The
/// registry and `rcsr schema` use this function as their single source of
/// truth; a name that is not returned here is not an MTF feature.
pub fn mtf_names() -> Vec<String> {
    let mut out = Vec::new();
    for tf in MTF_TIMEFRAMES {
        for leaf in MTF_LEAVES {
            out.push(format!("mtf_{}_{}", tf, leaf));
        }
    }
    out
}

/// True when `name` belongs to the multi-timeframe family.
pub fn is_mtf(name: &str) -> bool {
    mtf_parts(name).is_some()
}

fn mtf_parts(name: &str) -> Option<(&str, &str)> {
    let (tf, leaf) = name.strip_prefix("mtf_")?.split_once('_')?;
    if MTF_TIMEFRAMES.contains(&tf) && MTF_LEAVES.contains(&leaf) {
        Some((tf, leaf))
    } else {
        None
    }
}

/// Nominal warm-up, in higher-timeframe bars, for one MTF leaf.
pub fn mtf_warmup(leaf: &str) -> usize {
    match leaf {
        "ret1" => 1,
        "ret4" => 4,
        "ret24" => 24,
        "ma_ratio" => 19,
        "vol" => 20,
        "atr_pct" | "rsi" => 14,
        "close_pos" | "range_pct" | "bar_age" => 0,
        "donchian_pos" => 19,
        "trend_slope" => 39,
        _ => 0,
    }
}

/// Formula text for one MTF leaf, in higher-timeframe bars.
fn mtf_formula(leaf: &str) -> &'static str {
    match leaf {
        "ret1" => "C_t/C_{t-1} - 1",
        "ret4" => "C_t/C_{t-4} - 1",
        "ret24" => "C_t/C_{t-24} - 1",
        "ma_ratio" => "C_t/MA20_t - 1",
        "vol" => "population sd of one-bar simple returns over 20 bars",
        "atr_pct" => "ATR14_t/C_t (Wilder, seeded on 14 contiguous true ranges)",
        "rsi" => "RSI14_t/50 - 1",
        "close_pos" => "(C_t-L_t)/(H_t-L_t) - 0.5",
        "donchian_pos" => "(C_t-min L20)/(max H20-min L20) - 0.5",
        "range_pct" => "(H_t-L_t)/C_t",
        "trend_slope" => "MA20_t/MA20_{t-20} - 1",
        "bar_age" => "base_time - (higher.time + timeframe_seconds)",
        _ => "higher-timeframe feature",
    }
}

/// The complete default candle/price-action block required by issue #261.
///
/// `batr` is materialised by the compact profile (and shares that one
/// implementation); the other eleven are produced by [`price_action_all`] at
/// the 15 m base. The compact candle subset is the parity/regression half:
/// `clv1 clv4 body1 batr wasym1 lower_wick rng1 volrat gapatr don96`.
pub const PRICE_ACTION_NAMES: &[&str] = &[
    "clv16",
    "body16",
    "rng_atr",
    "prev_range_pos",
    "inside_s",
    "outside_s",
    "engulf_s",
    "harami_s",
    "run8",
    "nr4",
    "trend96",
];

/// Benchmark BTC context columns produced from the same timestamp-keyed series
/// as the compact and `py15m` joins.
pub const CROSS_BTC_NAMES: &[&str] = &[
    "btc_r1",
    "btc_r16",
    "btc_r20",
    "btc_r96",
    "btc_r672",
    "btc_vol20",
    "btc_vol96",
    "rel_btc_r1",
    "rel_btc_r16",
    "rel_btc_r20",
    "rel_btc_r96",
    "beta_btc_60",
    "beta_btc_384",
];

/// Benchmark ETH context columns. They mirror the BTC rows and use
/// `--eth-symbol` / `--eth-data`; a requested ETH column without that series is
/// a hard error, never a zero column.
pub const CROSS_ETH_NAMES: &[&str] = &[
    "eth_r1",
    "eth_r16",
    "eth_r20",
    "eth_r96",
    "eth_r672",
    "eth_vol20",
    "eth_vol96",
    "rel_eth_r1",
    "rel_eth_r16",
    "rel_eth_r20",
    "rel_eth_r96",
    "beta_eth_60",
    "beta_eth_384",
];

/// Equal-weight market state, breadth, dispersion and ex-self relative
/// columns. They require at least [`MIN_CROSS_MEMBERS`] valid point-in-time
/// members at the timestamp, otherwise every value at that timestamp is `NaN`.
pub const CROSS_MARKET_NAMES: &[&str] = &[
    "mkt_r16",
    "mkt_r96",
    "mkt_vol96",
    "breadth",
    "dispersion",
    "avg_corr",
    "rel_mkt16",
    "rel_mkt96",
];

/// Cross-sectional ranks. The scale is `[-1, 1]`, point-in-time members only.
pub const CROSS_RANK_NAMES: &[&str] = &[
    "xs_rank_r1",
    "xs_rank_r16",
    "xs_rank_r20",
    "xs_rank_vol20",
    "xs_rank_amihud96",
];

/// Older market-state names registered by #260/#269. They are aliases of the
/// canonical #273 producers, not a second implementation.

/// The twelve default continuous/structure columns (issue #261). `batr` is the
/// only name shared with the compact candle subset.
pub const DEFAULT_CANDLE_NAMES: &[&str] = &[
    "clv16",
    "body16",
    "batr",
    "rng_atr",
    "prev_range_pos",
    "inside_s",
    "outside_s",
    "engulf_s",
    "harami_s",
    "run8",
    "nr4",
    "trend96",
];

pub fn price_action_names() -> Vec<String> {
    PRICE_ACTION_NAMES.iter().map(|s| s.to_string()).collect()
}

fn price_action_formula(name: &str) -> &'static str {
    match name {
        "clv16" => "mean((C-L)/(H-L) - 1/2, 16); NaN on R=0 or a gap in the window",
        "body16" => "mean((C-O)/(H-L), 16); NaN on R=0 or a gap in the window",
        "rng_atr" => "(H-L)/ATR14; NaN when ATR14<=0 or the bar is invalid",
        "prev_range_pos" => "(C_t-L_{t-1})/R_{t-1} - 1/2; NaN when t-1 is missing or R_{t-1}=0",
        "inside_s" => "relu(1-relu(H_t-H_{t-1})/A)*relu(1-relu(L_{t-1}-L_t)/A)",
        "outside_s" => "relu(1-relu(H_{t-1}-H_t)/A)*relu(1-relu(L_t-L_{t-1})/A)",
        "engulf_s" => "signed bullish-minus-bearish continuous engulfing with body-ratio and ATR penetration terms",
        "harami_s" => "signed opposite-body continuous harami with body shrink and containment terms",
        "run8" => "clip(consecutive same-sign close changes,-8,8)/8; resets at a gap",
        "nr4" => "R_t/min(R_{t-3..t}); NaN unless four contiguous bars have R>0",
        "trend96" => "C/SMA96 - 1; NaN unless the 96-bar window is contiguous",
        _ => "see docs/ohlcv.md",
    }
}

fn price_action_warmup(name: &str) -> usize {
    match name {
        "clv16" | "body16" => 15,
        "rng_atr" => 14,
        "prev_range_pos" => 1,
        "inside_s" | "outside_s" | "engulf_s" | "harami_s" => 1,
        "run8" => 1,
        "nr4" => 3,
        "trend96" => 95,
        _ => 0,
    }
}

/// Cross-sectional market-state columns (whole-panel pass).

pub const MARKET_STATE_NAMES: &[&str] = &[
    "mkt_ew_ret1",
    "mkt_ew_ret16",
    "mkt_breadth",
    "mkt_dispersion",
    "btc_volume_share",
    "cs_rank_r60",
];

/// The complete #273 cross-asset inventory in stable emission order.
pub fn cross_names() -> Vec<String> {
    CROSS_BTC_NAMES
        .iter()
        .chain(CROSS_ETH_NAMES)
        .chain(CROSS_MARKET_NAMES)
        .chain(CROSS_RANK_NAMES)
        .chain(MARKET_STATE_NAMES)
        .map(|s| s.to_string())
        .collect()
}

/// Minimum point-in-time members for a cross-sectional statistic.
pub const MIN_CROSS_MEMBERS: usize = 20;

/// Source column a cross-family member is computed from.
pub fn cross_source(name: &str) -> Option<&'static str> {
    Some(match name {
        "xs_rank_r1" | "mkt_ew_ret1" | "avg_corr" => "r1",
        "xs_rank_r16" | "mkt_r16" | "mkt_ew_ret16" | "breadth" | "dispersion"
        | "mkt_breadth" | "mkt_dispersion" | "rel_mkt16" => "r16",
        "xs_rank_r20" => "r20",
        "xs_rank_vol20" => "vol20",
        "xs_rank_amihud96" => "amihud96",
        "mkt_r96" | "rel_mkt96" => "r96",
        "mkt_vol96" => "vol96",
        "cs_rank_r60" => "r60",
        "btc_volume_share" => "volume",
        _ => return None,
    })
}

/// Registry formula text for a cross-family column.
fn cross_formula(name: &str) -> &'static str {
    match name {
        "btc_r1" | "btc_r16" | "btc_r20" | "btc_r96" | "btc_r672" => {
            "ln(C_BTC,t/C_BTC,t-k) at the same timestamp"
        }
        "eth_r1" | "eth_r16" | "eth_r20" | "eth_r96" | "eth_r672" => {
            "ln(C_ETH,t/C_ETH,t-k) at the same timestamp"
        }
        "btc_vol20" | "btc_vol96" | "eth_vol20" | "eth_vol96" => {
            "population sd of benchmark one-bar log returns over w"
        }
        "rel_btc_r1" | "rel_btc_r16" | "rel_btc_r20" | "rel_btc_r96" => {
            "asset return r_k minus btc_r_k at the same timestamp"
        }
        "rel_eth_r1" | "rel_eth_r16" | "rel_eth_r20" | "rel_eth_r96" => {
            "asset return r_k minus eth_r_k at the same timestamp"
        }
        "beta_btc_60" | "beta_btc_384" => "clip(cov(r_i,r_BTC,w)/var(r_BTC,w),-5,5)",
        "beta_eth_60" | "beta_eth_384" => "clip(cov(r_i,r_ETH,w)/var(r_ETH,w),-5,5)",
        "mkt_r16" | "mkt_ew_ret16" => "equal-weight mean of PIT member r16 at t",
        "mkt_r96" => "equal-weight mean of PIT member r96 at t",
        "mkt_vol96" => "equal-weight mean of PIT member vol96 at t",
        "breadth" | "mkt_breadth" => "share of PIT members with r16 > 0",
        "dispersion" | "mkt_dispersion" => "population sd of PIT member r16 at t",
        "avg_corr" => {
            "mean_i corr(r_i,mkt_r1,96) over PIT members; proxy for average pairwise correlation"
        }
        "rel_mkt16" => "r16_i - ex-self equal-weight mean of member r16",
        "rel_mkt96" => "r96_i - ex-self equal-weight mean of member r96",
        "mkt_ew_ret1" => "equal-weight mean of PIT member r1 at t",
        "btc_volume_share" => "BTC volume / total PIT member volume at t",
        "xs_rank_r1" | "xs_rank_r16" | "xs_rank_r20" | "xs_rank_vol20"
        | "xs_rank_amihud96" | "cs_rank_r60" => {
            "2*(rank-1)/(n-1)-1 per timestamp; PIT finite, N>=20, ties symbol then row, -0==+0; one device (time,symbol,row) permutation reused"
        }
        _ => "see docs/ohlcv.md",
    }
}

/// Nominal warm-up for a cross-family column, in 15 m base bars.
fn cross_warmup(name: &str) -> usize {
    match name {
        "btc_r1" | "eth_r1" | "mkt_ew_ret1" => 1,
        "btc_r16" | "eth_r16" | "rel_btc_r16" | "rel_eth_r16" | "mkt_r16"
        | "mkt_ew_ret16" | "breadth" | "dispersion" | "mkt_breadth"
        | "mkt_dispersion" | "rel_mkt16" | "xs_rank_r16" => 16,
        "btc_r20" | "eth_r20" | "rel_btc_r20" | "rel_eth_r20" | "btc_vol20"
        | "eth_vol20" | "xs_rank_r20" | "xs_rank_vol20" => 20,
        "btc_r96" | "eth_r96" | "rel_btc_r96" | "rel_eth_r96" | "btc_vol96"
        | "eth_vol96" | "mkt_r96" | "mkt_vol96" | "rel_mkt96" | "avg_corr"
        | "xs_rank_amihud96" => 96,
        "btc_r672" | "eth_r672" => 672,
        "beta_btc_60" | "beta_eth_60" => 59,
        "beta_btc_384" | "beta_eth_384" => 383,
        "xs_rank_r1" => 1,
        "cs_rank_r60" => 60,
        "btc_volume_share" => 0,
        _ => 0,
    }
}

/// Reserved derivative columns; see `crate::derivatives`.
pub const DERIVATIVE_NAMES: &[&str] = crate::derivatives::NAMES;

/// Reason a registered name cannot be computed yet, if it is unavailable.
pub fn unavailable_reason(name: &str) -> Option<String> {
    crate::derivatives::refusal(name)
}

fn compact_formula(name: &str) -> &'static str {
    match name {
        "r16" | "r96" | "r672" => "ln(C_t / C_{t-w})",
        "ma96" | "ma672" => "C_t / SMA_w(t) - 1",
        "ma_slope672" => "SMA672(t) / SMA672(t-672) - 1",
        "macd_hist" => "EMA12/EMA26 - 1 minus its EMA9",
        "natr14" => "100 * ATR14 / C",
        "vol16" | "vol96" => "population sd of 15m log returns over w",
        "vol_ts" => "vol16 / vol96",
        "vol_ratio" => "vol96 / vol672",
        "vol_z2880" => "(vol96 - mean2880(vol96)) / sd2880(vol96)",
        "parkinson96" => "sqrt(mean(log(H/L)^2,96)/(4 ln 2))",
        "rv_ratio" => "ln(rv96/rv672), rv_w=mean(lr^2,w)",
        "rsi14" => "RSI14/50 - 1",
        "bb96" => "clip((C-SMA96)/(2 sd96),-1,1)",
        "bb_width96" => "4 sd96 / SMA96",
        "adx14" => "ADX14/100",
        "di_diff14" => "(PDI-MDI)/100",
        "cmf20" => "mean(mfm*volume,20)/mean(volume,20)",
        "mfi14" => "MFI14/50 - 1",
        "vwap_dev96" => "(C-VWAP96)/VWAP96",
        "vol_z96" => "(log1p(V)-mean96)/sd96",
        "taker_ratio96" => "mean(taker_buy_base,96)/mean(volume,96)",
        "taker_imb16" => "(2 mean(tb,16)-mean(V,16))/mean(V,16)",
        "trade_intensity96" => "ln(1+mean(trades,96)/mean(volume,96))",
        "clv1" => "(C-L)/(H-L) - 1/2",
        "clv4" => "mean(clv1,4)",
        "body1" => "(C-O)/(H-L)",
        "batr" => "(C-O)/ATR14",
        "wasym1" => "(U-D)/(H-L); equals body1 - 2*clv1 for H>L",
        "lower_wick" => "(min(O,C)-L)/(H-L)",
        "rng1" => "(H-L)/C",
        "volrat" => "ATR16/ATR96",
        "gapatr" => "(O_t-C_{t-1})/ATR14",
        "ou_hl96" => "ln(1+clip(-ln2/ln(phi),0,1000)), phi clipped AR(1) of log(C/SMA96)",
        "acf1" => "lag-1 autocorrelation of 15m log returns over 384",
        "vr4" => "rstd(r4,384)^2/(4 rstd(lr,384)^2)",
        "rskew96" => "m3/m2^1.5 of 15m log returns over 96",
        "jump_share" => "max(rv96-bv,0)/rv96",
        "semivar_asym" => "(downside semivar - upside semivar)/rv96",
        "cusum_abs_z" => "|cs_t-cs_{t-96}|/sqrt(96)",
        "btc_r96" => "BTC 96-bar return at the same timestamp",
        "btc_vol96" => "BTC vol96 at the same timestamp",
        "rel_btc96" => "r96 - btc_r96",
        "beta384" => "clip(clip(cov(r_i,r_BTC,384),-0.05,0.05)/var(r_BTC,384),-5,5)",
        "tod_sin" | "tod_cos" => "sin/cos(2*pi*bar_index_in_day/bars_per_day)",
        "dow_sin" | "dow_cos" => "sin/cos(2*pi*weekday/7), Monday=0",
        "don96" => "(C-min L96)/(max H96-min L96) - 1/2",
        _ => "see docs/ohlcv.md",
    }
}

fn flow_formula(name: &str) -> &'static str {
    match name {
        "vwap" => "quote_volume / volume",
        "taker_ratio96" => "mean(taker_buy_base,96)/mean(volume,96)",
        "taker_imb16" => "(2 mean(tb,16)-mean(V,16))/mean(V,16)",
        "trade_intensity96" => "ln(1+mean(trades,96)/mean(volume,96))",
        "avg_trade_size96" => "ln(1+mean(quote_volume/max(trades,1),96))",
        "amihud96" => "1e6 mean(|ln(C_t/C_{t-1})|/quote_volume,96)",
        "zero_ret_share96" => "mean(1[ln(C_t/C_{t-1})==0],96)",
        "obv_z96" => "z-score of OBV over 96; OBV_t=OBV_{t-1}+sign(dC) volume",
        "cvd_proxy" => "cumulative(2 taker_buy_base-volume)/std(cumulative,96)",
        "vwap_dev96" => "(C-mean(quote_volume,96)/mean(volume,96))/(mean(quote_volume,96)/mean(volume,96))",
        _ => "see docs/ohlcv.md",
    }
}

/// Raw columns required by one flow feature. `close`/`open`/`high`/`low` are
/// mandatory for every enrich run; these are the optional raw fields.
fn flow_requires(name: &str) -> &'static [&'static str] {
    match name {
        "vwap" | "vwap_dev96" => &["volume", "quote_volume"],
        "taker_ratio96" | "taker_imb16" => &["volume", "taker_buy_base"],
        "trade_intensity96" => &["volume", "trades"],
        "avg_trade_size96" => &["quote_volume", "trades"],
        "amihud96" => &["quote_volume"],
        "zero_ret_share96" => &[],
        "obv_z96" => &["volume"],
        "cvd_proxy" => &["volume", "taker_buy_base"],
        "btc_volume_share" => &["volume"],
        _ => &[],
    }
}

fn flow_warmup(name: &str) -> usize {
    match name {
        "vwap" => 0,
        "taker_imb16" => 15,
        "amihud96" | "zero_ret_share96" => 96,
        _ => 95,
    }
}

/// One registered column in the default financial feature set.
#[derive(Clone, Debug)]
pub struct RegistryEntry {
    pub name: String,
    pub family: &'static str,
    pub timeframe: &'static str,
    pub warmup: usize,
    pub available: bool,
    pub derived: bool,
    /// Declared CPU/GPU parity tier: `bitwise`, `1e-12`, `unavailable`, or
    /// `reference` when this issue does not claim a device tier.
    pub parity_tier: &'static str,
    pub formula: String,
    pub source: &'static str,
    pub pit: &'static str,
    pub reason: Option<String>,
}

/// Declared parity tier for a registry entry.
///
/// Only the fused flow / price-action columns and the reserved derivatives make
/// a device claim in #297; every other name is the CPU reference and is marked
/// `reference` rather than silently implying byte parity.
pub fn declared_parity_tier(name: &str, family: &str) -> &'static str {
    if family == "derivatives" {
        return "unavailable";
    }
    if let Some(tier) = crate::features_gpu::device_tier(name) {
        return tier;
    }
    if let Some(tier) = crate::bench_gpu::device_tier(name) {
        return tier;
    }
    if device_kernel_exists(name) {
        return engine_device_tier(name);
    }
    "reference"
}

/// Declared tier of an engine column with a device kernel (#303).
///
/// `1e-12`: anything that evaluates `ln`, `sin` or `cos` (CUDA and the host
/// libm differ in the last bit), anything derived from the 1-bar log return,
/// and the MTF `vol` leaf. `bitwise`: elementwise geometry, min/max, lags of
/// exact values, ranks, rolling means and standard deviations (the compensated
/// window sums of `window_stats` run the same operations on both sides), and
/// recurrences evaluated in the CPU order.
fn engine_device_tier(name: &str) -> &'static str {
    let numeric = |s: Option<&str>| s.map_or(false, |r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()));
    let log_based = numeric(name.strip_prefix('r'))
        || numeric(name.strip_prefix("vol"))
        || name.starts_with("lag_r1_")
        || name.starts_with("dow_")
        || name.starts_with("hour_")
        || name.starts_with("month_")
        || name.starts_with("dom_")
        || name.starts_with("tod_")
        || (is_mtf(name) && name.ends_with("_vol"))
        || matches!(
            name,
            "ret1" | "ret5" | "vz96" | "vol_z96" | "acf1" | "vr4" | "rskew96" | "ou_hl96"
                | "jump_share" | "semivar_asym" | "cusum_abs_z" | "rmean5" | "rstd5"
        );
    if log_based {
        "1e-12"
    } else {
        "bitwise"
    }
}

/// Registry entry of a stored long-only label column (`lbl_<label>_<kind>`).
/// The family is `label`: the column is a target or a training weight, never a
/// model input. Label names are chosen per run, so entries are derived from
/// the column name instead of being listed in [`default_registry`].
pub fn label_registry_entry(name: &str) -> Option<RegistryEntry> {
    let (_, kind) = crate::labels::parse_column(name)?;
    let formula = match kind {
        "fwd" => "ln(close[t+h+d] / open[t+1+d]), long gross log return",
        "y" => "clip(fwd / (max(s_W, floor) * sqrt(h)), -clip, +clip)",
        "long_net" => "fwd - long round-trip cost (- funding when supplied)",
        "action" => "1 if fwd > cost else 0, never -1",
        "t1" => "end of the exit bar, epoch seconds",
        "exit_ret" => "ln(exit price / open[t+1+d]) of the long-side triple barrier",
        "m_net" => "1[exit_ret > cost]",
        "m_barrier" => "1[profit barrier touched first]",
        _ => "average-uniqueness weight, mean 1 over the valid rows",
    };
    Some(RegistryEntry {
        name: name.to_string(),
        family: "label",
        timeframe: "bar",
        warmup: 0,
        available: true,
        derived: true,
        parity_tier: "reference",
        formula: formula.to_string(),
        source: "labels",
        pit: "label: reads bars t+1 through t+h+d, so it is a target only; vol-window kinds are NaN until s_W has W returns",
        reason: None,
    })
}

/// The default financial feature registry.
///
/// It enumerates (a) the compact 52 regression subset with exact formulas,
/// (b) every engine column already accepted by `rcsr enrich`, and (c) every
/// default-set column that is registered but not yet materialised, with the
/// missing dependency. Unknown names remain hard errors.
pub fn default_registry() -> Vec<RegistryEntry> {
    fn push_unique(out: &mut Vec<RegistryEntry>, e: RegistryEntry) {
        if !out.iter().any(|x| x.name == e.name) {
            out.push(e);
        }
    }
    let mut out: Vec<RegistryEntry> = Vec::new();
    // Raw OHLCV/flow inputs.
    for (name, family) in [
        ("ts", "identity"),
        ("symbol", "identity"),
        ("listing_id", "identity"),
        ("member_asof", "identity"),
        ("gap", "identity"),
        ("flags", "identity"),
        ("open", "price"),
        ("high", "price"),
        ("low", "price"),
        ("close", "price"),
        ("volume", "flow"),
        ("quote_volume", "flow"),
        ("trades", "flow"),
        ("taker_buy_base", "flow"),
        ("taker_buy_quote", "flow"),
    ] {
        push_unique(&mut out, RegistryEntry {
            name: name.to_string(),
            family,
            timeframe: "15m",
            warmup: 0,
            available: true,
            derived: false,
            parity_tier: declared_parity_tier(name, family),
            formula: "input column".to_string(),
            source: "input",
            pit: "bar t",
            reason: None,
        });
    }
    // OHLCV/flow family (#270): vwap plus the nine derived flow columns.
    // Registered before the compact loop so the canonical flow formula and
    // family win for the four names the compact subset shares.
    for name in FLOW_NAMES {
        push_unique(&mut out, RegistryEntry {
            name: name.to_string(),
            family: "flow",
            timeframe: "15m",
            warmup: flow_warmup(name),
            available: true,
            derived: false,
            parity_tier: declared_parity_tier(name, "flow"),
            formula: flow_formula(name).to_string(),
            source: "raw OHLCV/flow columns",
            pit: "close of bar t; trailing windows inclusive; gaps are NaN; \
                  zero volume/trades is NaN for affected ratios",
            reason: None,
        });
    }
    // Complete cross-asset, breadth, dispersion and rank family (#273).
    // Registered before the compact/py15m loops so the canonical market and
    // rank definitions win for the names those legacy groups share.
    for name in cross_names() {
        let warmup = cross_warmup(&name);
        let formula = cross_formula(&name).to_string();
        push_unique(&mut out, RegistryEntry {
            name: name.clone(),
            family: "cross",
            timeframe: "15m",
            warmup,
            available: true,
            derived: false,
            parity_tier: declared_parity_tier(&name, "cross"),
            formula,
            source: match cross_source(&name) {
                Some("volume") => "raw volume + all PIT members",
                Some("r1") | Some("r16") | Some("r20") | Some("r96") | Some("vol20")
                | Some("vol96") | Some("r60") | Some("amihud96") => {
                    "all PIT members at the same timestamp"
                }
                _ => "benchmark series by exact timestamp",
            },
            pit: if CROSS_BTC_NAMES.contains(&name.as_str()) {
                "same timestamp only; a missing benchmark row is NaN, never forward-filled"
            } else if CROSS_ETH_NAMES.contains(&name.as_str()) {
                "same timestamp only; a missing ETH row is NaN, never forward-filled"
            } else {
                "rows present at t with member_asof <= t; N>=20; no forward fill; gaps are NaN"
            },
            reason: None,
        });
    }
    // Compact subset with exact formulas.
    for name in COMPACT_NAMES {
        let family = compact_family(name);
        let source = if family == "candle" || *name == "batr" {
            "ohlcv::Bar; not a target construction column"
        } else {
            "ohlcv::Bar"
        };
        push_unique(&mut out, RegistryEntry {
            name: name.to_string(),
            family,
            timeframe: "15m",
            warmup: nominal_warmup(name),
            available: true,
            derived: *name == "wasym1",
            parity_tier: declared_parity_tier(name, family),
            formula: compact_formula(name).to_string(),
            source,
            pit: "close of bar t; trailing windows inclusive; gaps are NaN",
            reason: None,
        });
    }
    // The complete engine feature-bucket inventory, including the compact
    // engine names that no catalogue group emits (`r672`, `ma96`, ...).
    for name in engine_names() {
        if out.iter().any(|x| x.name == name) {
            continue;
        }
        push_unique(&mut out, RegistryEntry {
            family: family_of(&name),
            timeframe: "15m",
            warmup: nominal_warmup(&name),
            available: true,
            derived: false,
            parity_tier: declared_parity_tier(&name, family_of(&name)),
            formula: engine_formula(&name).to_string(),
            source: "ohlcv::Bar",
            pit: "close of bar t; trailing windows inclusive; gaps are NaN; no forward fill",
            reason: None,
            name,
        });
    }
    // Multi-timeframe family: 12 leaves x {1h,4h,1d}. The join is causal and
    // the partial current higher bucket is never visible.
    for tf in MTF_TIMEFRAMES {
        for leaf in MTF_LEAVES {
            let name = format!("mtf_{}_{}", tf, leaf);
            let parity_tier = declared_parity_tier(&name, "mtf");
            push_unique(&mut out, RegistryEntry {
                name,
                family: "mtf",
                timeframe: tf,
                warmup: mtf_warmup(leaf),
                available: true,
                derived: false,
                parity_tier,
                formula: format!("resample to {}; {}; causal close join", tf, mtf_formula(leaf)),
                source: "base OHLCV bars",
                pit: "higher.time + timeframe_seconds() <= t; last closed higher bar only; \
                      partial/current higher bar never used; gaps are NaN",
                reason: None,
            });
        }
    }
    for name in PRICE_ACTION_NAMES {
        push_unique(&mut out, RegistryEntry {
            name: name.to_string(),
            family: "price_action",
            timeframe: "15m",
            warmup: price_action_warmup(name),
            available: true,
            derived: false,
            parity_tier: declared_parity_tier(name, "price_action"),
            formula: price_action_formula(name).to_string(),
            source: "15m OHLCV bars; not a target construction column",
            pit: "close of bar t; trailing windows inclusive; any missing bucket \
                  or zero-range/zero-volume bar in the window is NaN",
            reason: None,
        });
    }
    for name in DERIVATIVE_NAMES {
        push_unique(&mut out, RegistryEntry {
            name: name.to_string(),
            family: "derivatives",
            timeframe: "15m",
            warmup: 0,
            available: false,
            derived: false,
            parity_tier: "unavailable",
            formula: "reserved derivative column".to_string(),
            source: "none",
            pit: "not available",
            reason: unavailable_reason(name),
        });
    }
    out
}

/// JSON array for `rcsr schema` and the enrich report.
pub fn registry_json() -> String {
    let entries: Vec<String> = default_registry()
        .iter()
        .map(|e| {
            format!(
                "{{\"name\":{},\"family\":{},\"timeframe\":{},\"warmup\":{},\
                 \"available\":{},\"derived\":{},\"parity_tier\":{},\"formula\":{},\"formula_sha256\":{},\
                 \"source\":{},\"pit\":{},\"reason\":{}{}}}",
                jstr(&e.name),
                jstr(e.family),
                jstr(e.timeframe),
                e.warmup,
                e.available,
                e.derived,
                jstr(e.parity_tier),
                jstr(&e.formula),
                jstr(&manifest::sha256_hex(e.formula.as_bytes())),
                if e.family == "derivatives" { "null".to_string() } else { jstr(e.source) },
                jstr(e.pit),
                e.reason.as_deref().map(jstr).unwrap_or_else(|| "null".to_string()),
                crate::derivatives::registry_extra_json(&e.name)
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Names that appear in more than one engine group, mapped to the single
/// implementation that wins regardless of the requesting group. This is the
/// explicit alias/collision audit; a name never means two formulas.
pub const CANONICAL_ENGINE_NAMES: &[(&str, &str)] = &[
    ("rsi14", "compact"),
    ("mfi14", "compact"),
    ("cmf20", "compact"),
    ("adx14", "compact"),
    ("clv1", "compact"),
    ("don96", "compact"),
    ("macd_hist", "compact"),
    ("vwap_dev96", "flow"),
    ("obv_z96", "flow"),
    ("taker_ratio96", "flow"),
    ("taker_imb16", "flow"),
    ("trade_intensity96", "flow"),
    ("bb20_bw", "py15m/standalone (same formula)"),
    ("natr14", "features/compact (same formula)"),
];

/// Machine-readable engine inventory for `rcsr schema`.
pub fn engine_catalogue_json() -> String {
    let names = engine_names();
    let engine: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let reg = default_registry();
    let entries: Vec<String> = engine
        .iter()
        .filter_map(|name| reg.iter().find(|e| e.name == *name))
        .map(|e| {
            format!(
                "{{\"name\":{},\"family\":{},\"warmup\":{},\"available\":{},\"parity_tier\":{},\"formula\":{}}}",
                jstr(&e.name),
                jstr(e.family),
                e.warmup,
                e.available,
                jstr(e.parity_tier),
                jstr(&e.formula)
            )
        })
        .collect();
    let aliases: Vec<String> = CANONICAL_ENGINE_NAMES
        .iter()
        .map(|(n, source)| format!("{{\"name\":{},\"canonical\":{}}}", jstr(n), jstr(source)))
        .collect();
    format!(
        "{{\"count\":{},\"catalogue_count\":{},\"names\":[{}],\"shared_names\":[{}]}}",
        engine.len(),
        catalogue_names().len(),
        entries.join(","),
        aliases.join(",")
    )
}

pub fn is_compact(name: &str) -> bool {
    COMPACT_NAMES.contains(&name)
}

/// True when a compact name has no exact implementation elsewhere.
///
/// `r16`, `r96`, `macd_hist`, `natr14`, `adx14`, `cmf20`, `vwap_dev96`,
/// `vol_z96` and the two weekday columns already have identical standalone
/// implementations; every other compact name (including the four collision
/// names `clv1`, `rsi14`, `mfi14`, `don96`) is routed to the compact reference.
pub fn needs_compact(name: &str) -> bool {
    is_compact(name)
        && !matches!(
            name,
            "r16"
                | "r96"
                | "macd_hist"
                | "natr14"
                | "adx14"
                | "cmf20"
                | "vwap_dev96"
                | "vol_z96"
                | "dow_sin"
                | "dow_cos"
        )
}

fn return_names() -> Vec<String> {
    (1..=120).map(|k| format!("r{}", k)).collect()
}

fn ma_names() -> Vec<String> {
    [5usize, 10, 20, 50, 100, 200]
        .iter()
        .map(|w| format!("ma{}", w))
        .collect()
}

fn vol_names() -> Vec<String> {
    [5usize, 20, 60, 120].iter().map(|w| format!("vol{}", w)).collect()
}

fn atr_names() -> Vec<String> {
    vec!["atr14".to_string(), "natr14".to_string()]
}

fn rsi_names() -> Vec<String> {
    vec!["rsi14".to_string()]
}

fn bollinger_names() -> Vec<String> {
    vec!["bb20_z".to_string(), "bb20_bw".to_string()]
}

fn vz_names() -> Vec<String> {
    vec!["vz96".to_string()]
}

fn range_names() -> Vec<String> {
    vec!["range1".to_string()]
}

fn candle_names() -> Vec<String> {
    ["clv1", "body1", "upper_wick1", "lower_wick1", "open_gap1"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn donchian_names() -> Vec<String> {
    vec!["don20".to_string(), "don96".to_string()]
}

fn calendar_names() -> Vec<String> {
    [
        "dow_sin",
        "dow_cos",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn lag_names() -> Vec<String> {
    (1..=10).map(|k| format!("lag_r1_{}", k)).collect()
}

fn xs_names() -> Vec<String> {
    vec![
        "xs_rank_r1".to_string(),
        "xs_rank_r20".to_string(),
        "xs_rank_vol20".to_string(),
    ]
}

fn btc_names() -> Vec<String> {
    [
        "btc_r1",
        "btc_r20",
        "btc_vol20",
        "rel_btc_r1",
        "rel_btc_r20",
        "beta_btc_60",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The 35 columns added by the Python `enrich_panel_15m.py` step.
pub fn py15m_names() -> Vec<String> {
    [
        "ema8",
        "ema20",
        "ema50",
        "ema100",
        "ema200",
        "macd",
        "macd_sig",
        "macd_hist",
        "rsi6",
        "rsi28",
        "bb50",
        "bb20_bw",
        "bb50_bw",
        "atr50",
        "stoch_k",
        "stoch_d",
        "adx14",
        "di_plus",
        "di_minus",
        "obv_z96",
        "vwap_dev96",
        "park16",
        "gk16",
        "skew96",
        "kurt96",
        "r192",
        "month_sin",
        "month_cos",
        "dom_sin",
        "dom_cos",
        "mkt_r16",
        "mkt_r96",
        "mkt_vol96",
        "rel_r16",
        "rel_r96",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The engine catalogue emitted by
/// `--features all,py15m,ta:full,exotic:cheap,cross:btc`, in fixed group order,
/// with duplicates kept once. This is the ~216-name engine bucket set.
pub fn catalogue_names() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for group in ["all", "py15m", "ta:full", "exotic:cheap", "cross:btc"] {
        if let Ok(names) = expand(group) {
            for n in names {
                if !out.contains(&n) {
                    out.push(n);
                }
            }
        }
    }
    out
}

/// The complete engine feature-bucket inventory: [`catalogue_names`] plus the
/// compact engine columns that have no catalogue group (`r672`, `ma96`, ...).
/// Each name here must have exactly one registry entry and one canonical
/// producer.
pub fn engine_names() -> Vec<String> {
    let mut out = catalogue_names();
    for n in compact_names() {
        if !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// Exact formula for an engine name, independent of which group requested it.
/// Compact names keep [`compact_formula`]; this table covers the catalogue names
/// that are not in the compact subset.
fn engine_formula(name: &str) -> &'static str {
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        return match k {
            1 => "ln(C_t / C_{t-1})",
            5 => "ln(C_t / C_{t-5})",
            _ => "ln(C_t / C_{t-k}) over exactly k contiguous bars",
        };
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        let _ = w;
        return "C_t / SMA_w(t) - 1, SMA_w trailing and contiguous";
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        let _ = w;
        return "population sd of 1-bar log returns over w contiguous bars";
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        let _ = k;
        return "r1 at t-k; NaN unless the two bars are exactly k steps apart";
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        let _ = w;
        return "(C_t - min L_w)/(max H_w - min L_w); don96 is centred by -1/2";
    }
    if let Some(rest) = name.strip_prefix("xs_rank_") {
        return match rest {
            "r1" => "per-timestamp rank of r1: (rank-1)/(n_finite-1)",
            "r20" => "per-timestamp rank of r20: (rank-1)/(n_finite-1)",
            "vol20" => "per-timestamp rank of vol20: (rank-1)/(n_finite-1)",
            _ => "per-timestamp cross-sectional rank: (rank-1)/(n_finite-1)",
        };
    }
    match name {
        "atr14" => "Wilder ATR(14) in price units, seeded on the first 14 true ranges",
        "natr14" => "100 * ATR14 / C_t",
        "atr50" => "ATR50 / C_t (fractional, Python py15m definition)",
        "bb20_z" => "(C_t - SMA20) / sd20, population sd",
        "bb20_bw" => "4 * sd20 / SMA20",
        "bb50" => "clip((C_t - SMA50)/(2 sd50), -1, 1)",
        "bb50_bw" => "4 * sd50 / SMA50",
        "vz96" => "(ln(1+V_t) - mean96(ln(1+V))) / sd96(ln(1+V))",
        "range1" => "(H_t - L_t) / C_t",
        "upper_wick1" => "(H_t - max(O_t,C_t)) / (H_t - L_t)",
        "lower_wick1" => "(min(O_t,C_t) - L_t) / (H_t - L_t)",
        "open_gap1" => "O_t / C_{t-1} - 1; NaN across a gap",
        "dow_sin" | "dow_cos" => "sin/cos(2*pi*weekday/7), Monday=0, UTC",
        "hour_sin" | "hour_cos" => "sin/cos(2*pi*seconds_into_day/86400), UTC",
        "month_sin" | "month_cos" => "sin/cos(2*pi*(month-1)/12), UTC",
        "dom_sin" | "dom_cos" => "sin/cos(2*pi*(day_of_month-1)/days_in_month), UTC",
        "ema8" | "ema20" | "ema50" | "ema100" | "ema200" => {
            "C_t / EMA_span(t) - 1, EMA seeded and restarted after a gap"
        }
        "macd" => "EMA12/EMA26 - 1",
        "macd_sig" => "EMA9 of macd",
        "macd_hist" => "macd - EMA9(macd)",
        "rsi6" => "RSI6/50 - 1",
        "rsi28" => "RSI28/50 - 1",
        "stoch_k" => "clip((C_t-min L14)/(max H14-min L14)-1/2, -1/2, 1/2)",
        "stoch_d" => "mean3(stoch_k)",
        "adx14" => "ADX14/100",
        "di_plus" => "PDI14/100",
        "di_minus" => "MDI14/100",
        "obv_z96" => "z-score of OBV over 96; OBV_t=OBV_{t-1}+sign(dC)V_t",
        "vwap_dev96" => "(C_t - VWAP96)/VWAP96, VWAP96=mean(quote_volume,96)/mean(volume,96)",
        "park16" => "sqrt(mean(log(H/L)^2,16)/(4 ln 2))",
        "gk16" => "sqrt(max(0.5 mean(log(H/L)^2,16) - (2 ln2 -1) mean(log(C/O)^2,16)))",
        "skew96" | "kurt96" => "third/fourth population moment of 1-bar log returns over 96",
        "r192" => "ln(C_t / C_{t-192}) over exactly 192 contiguous bars",
        "mkt_r16" => "benchmark r16 at the same timestamp",
        "mkt_r96" => "benchmark r96 at the same timestamp",
        "mkt_vol96" => "benchmark vol96 at the same timestamp",
        "rel_r16" => "r16 - benchmark r16 at the same timestamp",
        "rel_r96" => "r96 - benchmark r96 at the same timestamp",
        "btc_r1" => "benchmark 1-bar log return at the same timestamp",
        "btc_r20" => "benchmark 20-bar log return at the same timestamp",
        "btc_vol20" => "benchmark population sd of 1-bar log returns over 20",
        "rel_btc_r1" => "r1 - benchmark r1",
        "rel_btc_r20" => "r20 - benchmark r20",
        "beta_btc_60" => "cov(r_self,r_benchmark,60)/var(r_benchmark,60)",
        "cross_ret1" => "benchmark 1-bar log return at the same timestamp",
        "cross_beta20" => "cov(r_self,r_benchmark,20)/var(r_benchmark,20)",
        "ret1" => "ln(C_t / C_{t-1}); NaN across a gap",
        "ret5" => "ln(C_t / C_{t-5}) over exactly 5 contiguous bars",
        "ma20_ratio" | "ma60_ratio" => "C_t / SMA_w(t) - 1, w=20 or 60",
        "vol20_ratio" => "V_t / mean20(V)",
        "vwap20_dev" => "(C_t - VWAP20)/VWAP20, VWAP20=sum(quote_volume,20)/sum(volume,20)",
        "rmean5" => "mean of the last 5 one-bar log returns",
        "rstd5" => "population sd of the last 5 one-bar log returns",
        "skew20" | "kurt20" => "third/fourth population moment of 1-bar log returns over 20",
        "close_pos20" => "(C_t - min L20)/(max H20 - min L20)",
        "di_diff14" => "(PDI14 - MDI14)/100",
        "batr" => "(C_t - O_t)/ATR14",
        "gapatr" => "(O_t - C_{t-1})/ATR14",
        "volrat" => "ATR16/ATR96",
        "bb96" => "clip((C_t - SMA96)/(2 sd96), -1, 1)",
        "bb_width96" => "4 sd96 / SMA96",
        _ => "see docs/ohlcv.md",
    }
}

/// Nominal warm-up for an engine catalogue name. Compact names use
/// [`compact_warmup`] so the reference profile stays authoritative.
fn engine_warmup(name: &str) -> usize {
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        return k;
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        return w.saturating_sub(1);
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        return w;
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        return k;
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        return w.saturating_sub(1);
    }
    match name {
        "ret1" => 1,
        "ret5" => 5,
        "ma20_ratio" => 19,
        "ma60_ratio" => 59,
        "vol20_ratio" => 19,
        "bb20_z" | "bb20_bw" => 19,
        "bb50" | "bb50_bw" => 49,
        "vz96" => 95,
        "range1" | "upper_wick1" | "lower_wick1" | "open_gap1" => 0,
        "atr14" | "natr14" | "di_plus" | "di_minus" | "adx14" => 14,
        "atr50" => 50,
        "rsi6" => 6,
        "rsi28" => 28,
        "stoch_k" | "stoch_d" => 13,
        "ema8" | "ema20" | "ema50" | "ema100" | "ema200" | "macd" | "macd_sig"
        | "macd_hist" => 0,
        "obv_z96" | "vwap_dev96" => 95,
        "park16" | "gk16" => 15,
        "skew96" | "kurt96" => 96,
        "r192" => 192,
        "mkt_r16" | "rel_r16" => 16,
        "mkt_r96" | "rel_r96" | "mkt_vol96" => 96,
        "btc_r1" | "cross_ret1" => 1,
        "btc_r20" | "btc_vol20" | "rel_btc_r20" | "cross_beta20" => 20,
        "rel_btc_r1" => 1,
        "beta_btc_60" => 59,
        "xs_rank_r1" => 1,
        "xs_rank_r20" | "xs_rank_vol20" => 20,
        "vwap20_dev" | "close_pos20" => 19,
        "rmean5" | "rstd5" => 5,
        "skew20" | "kurt20" => 20,
        "dow_sin" | "dow_cos" | "hour_sin" | "hour_cos" | "month_sin" | "month_cos"
        | "dom_sin" | "dom_cos" => 0,
        _ => 0,
    }
}

/// Expand a comma-separated catalogue specification.
///
/// Groups expand in a fixed order and duplicates are kept once. Names known to
/// [`crate::features`] are accepted too, so `ta:core` and `ta:full` keep their
/// #125 meaning; those columns are computed by that module rather than copied
/// here.
pub fn expand(spec: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    let mut push = |names: Vec<String>| {
        for n in names {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    };
    for raw in spec.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        let names = match item {
            "returns" => return_names(),
            "ma" => ma_names(),
            "vol" => vol_names(),
            "atr" => atr_names(),
            "rsi" => rsi_names(),
            "bollinger" => bollinger_names(),
            "vz" => vz_names(),
            "range" => range_names(),
            "candle" => candle_names(),
            "donchian" => donchian_names(),
            "calendar" => calendar_names(),
            "lags" => lag_names(),
            "xs" => xs_names(),
            "btc" => btc_names(),
            "cross" => cross_names(),
            "py15m" => py15m_names(),
            "mtf" => mtf_names(),
            "compact" => compact_names(),
            "flow" => flow_names(),
            "price_action" => price_action_names(),
            "derivatives" => return Err(crate::derivatives::group_refusal()),
            "all" => {
                let mut v = Vec::new();
                for g in [
                    "returns", "ma", "vol", "atr", "rsi", "bollinger", "vz", "range",
                    "candle", "donchian", "calendar", "lags", "xs", "btc",
                ] {
                    v.extend(expand(g)?);
                }
                v
            }
            other => {
                if is_own(other) {
                    vec![other.to_string()]
                } else if features::expand(other).is_ok() {
                    // #125 group or name.
                    features::expand(other)?
                } else if let Some(reason) = unavailable_reason(other) {
                    return Err(format!(
                        "registered feature '{}' is unavailable: {}",
                        other, reason
                    ));
                } else {
                    return Err(format!(
                        "unknown feature '{}' (groups: {}; individual names are listed in \
                         `rcsr enrich --help`)",
                        other,
                        GROUPS.join(", ")
                    ));
                }
            }
        };
        push(names);
    }
    if out.is_empty() {
        return Err("empty feature specification".to_string());
    }
    Ok(out)
}

/// Names implemented by this module (as opposed to delegated to #125).
pub fn is_own(name: &str) -> bool {
    if is_compact(name)
        || FLOW_NAMES.contains(&name)
        || is_mtf(name)
        || cross_names().iter().any(|n| n == name)
        || PRICE_ACTION_NAMES.contains(&name)
    {
        return true;
    }
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        if (1..=120).contains(&k) {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        if [5, 10, 20, 50, 100, 200].contains(&w) {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        if [5, 20, 60, 120].contains(&w) {
            return true;
        }
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        if (1..=10).contains(&k) {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        if [20, 96].contains(&w) {
            return true;
        }
    }
    matches!(
        name,
        "bb20_bw"
            | "vz96"
            | "clv1"
            | "xs_rank_r1"
            | "xs_rank_r20"
            | "xs_rank_vol20"
            | "btc_r1"
            | "btc_r20"
            | "btc_vol20"
            | "rel_btc_r1"
            | "rel_btc_r20"
            | "beta_btc_60"
            | "cross_ret1"
            | "cross_beta20"
            | "month_sin"
            | "month_cos"
            | "dom_sin"
            | "dom_cos"
    ) || py15m_names().iter().any(|n| n == name)
}

/// Nominal warm-up in bars for one catalogue name. The report also records the
/// observed first finite row, which is the authoritative number for a series
/// that starts after a gap.
/// Nominal warm-up in 15 m bars for a compact feature.
fn compact_warmup(name: &str) -> Option<usize> {
    Some(match name {
        "r16" => 16,
        "r96" => 96,
        "r672" => 672,
        "ma96" => 95,
        "ma672" => 671,
        "ma_slope672" => 671 + 672,
        "macd_hist" => 0,
        "natr14" => 14,
        "vol16" => 16,
        "vol96" => 96,
        "vol_ts" => 96,
        "vol_ratio" => 672,
        "vol_z2880" => 96 + 2880 - 1,
        "parkinson96" => 95,
        "rv_ratio" => 672,
        "rsi14" => 14,
        "bb96" | "bb_width96" => 95,
        "adx14" | "di_diff14" => 14,
        "cmf20" => 19,
        "mfi14" => 14,
        "vwap_dev96" | "vol_z96" | "taker_ratio96" | "trade_intensity96" => 95,
        "taker_imb16" => 15,
        "clv1" | "body1" | "wasym1" | "lower_wick" | "rng1" => 0,
        "clv4" => 3,
        "batr" => 14,
        "volrat" => 96,
        "gapatr" => 1,
        "ou_hl96" => 191,
        "acf1" => 385,
        "vr4" => 387,
        "rskew96" | "jump_share" | "semivar_asym" => 96,
        "cusum_abs_z" => 96,
        "btc_r96" | "btc_vol96" | "rel_btc96" => 96,
        "beta384" => 385,
        "tod_sin" | "tod_cos" | "dow_sin" | "dow_cos" => 0,
        "don96" => 95,
        _ => return None,
    })
}

pub fn nominal_warmup(name: &str) -> usize {
    if FLOW_NAMES.contains(&name) {
        return flow_warmup(name);
    }
    if cross_names().iter().any(|n| n == name) {
        return cross_warmup(name);
    }
    if PRICE_ACTION_NAMES.contains(&name) {
        return price_action_warmup(name);
    }
    if let Some((_, leaf)) = mtf_parts(name) {
        return mtf_warmup(leaf);
    }
    if let Some(w) = compact_warmup(name) {
        return w;
    }
    engine_warmup(name)
}

// ---------------------------------------------------------------------------
// Options and report
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMode {
    Auto,
    On,
    Off,
}

impl GpuMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(GpuMode::Auto),
            "on" | "force" | "gpu" => Ok(GpuMode::On),
            "off" | "no" | "cpu" => Ok(GpuMode::Off),
            other => Err(format!("unknown --gpu mode '{}' (auto, on, off)", other)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub data: String,
    pub out: Option<String>,
    pub features: Vec<String>,
    pub labels: Vec<String>,
    pub replace: bool,
    pub chunk_rows: usize,
    pub group_rows: usize,
    pub bar_seconds: Option<i64>,
    pub time_col: Option<String>,
    pub symbol_col: Option<String>,
    pub cross_symbol: String,
    pub cross_data: Option<String>,
    pub eth_symbol: String,
    pub eth_data: Option<String>,
    pub gpu: GpuMode,
    /// Measured sequential write bandwidth used by cost model v2 (#300).
    pub write_bw: Option<f64>,
    pub allow_lockbox: bool,
    pub allow_overlap: bool,
    pub dry_run: bool,
    pub json: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            data: String::new(),
            out: None,
            features: Vec::new(),
            labels: Vec::new(),
            replace: false,
            chunk_rows: store::DEFAULT_GROUP_ROWS,
            group_rows: 0,
            bar_seconds: None,
            time_col: None,
            symbol_col: None,
            cross_symbol: "BTCUSDT".to_string(),
            cross_data: None,
            eth_symbol: "ETHUSDT".to_string(),
            eth_data: None,
            gpu: GpuMode::Auto,
            write_bw: None,
            allow_lockbox: false,
            allow_overlap: false,
            dry_run: false,
            json: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FeatureReport {
    pub name: String,
    pub warmup: usize,
    pub nan_rows: usize,
    pub first_finite_row: Option<usize>,
    /// Epoch seconds of the first finite row, or `None`.
    pub first_finite_time: Option<i64>,
    /// Finite rows divided by all rows (`NaN` rows are the complement).
    pub prevalence: f64,
    pub hash: String,
    pub label: bool,
}

/// One long-only label family written to the store.
#[derive(Clone, Debug)]
pub struct LongLabelOutput {
    pub spec: crate::labels::LongLabelSpec,
    /// Stored column names in [`crate::labels::COLUMN_KINDS`] order.
    pub columns: Vec<String>,
    /// The regression target column (`lbl_<name>_y`).
    pub target: String,
    pub report: crate::labels::LabelReport,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub input: String,
    pub output: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub output_bytes: u64,
    pub rows: usize,
    pub columns_in: usize,
    pub columns_out: usize,
    pub groups: usize,
    pub symbols: usize,
    pub group_rows: usize,
    pub chunk_rows: usize,
    pub bar_seconds: i64,
    pub features: Vec<FeatureReport>,
    pub plans: Vec<GroupPlan>,
    /// Long-only label families written by this run (#262).
    pub long_labels: Vec<LongLabelOutput>,
    pub backend: String,
    pub gpu_requested: String,
    pub gpu_available: bool,
    pub gpu_device: Option<String>,
    /// Output feature/label columns computed by a CUDA kernel.
    pub gpu_columns: usize,
    /// Output feature/label columns computed on the CPU (including fallbacks).
    pub cpu_columns: usize,
    /// `gpu_columns / (gpu_columns + cpu_columns)`.
    pub gpu_share: f64,
    /// Columns planned for the GPU that fell back to the CPU; always `gpu_fallback_columns.len()`.
    pub gpu_fallback: usize,
    /// Name and reason (`<path>: <detail>`) of every fallback column.
    pub gpu_fallback_columns: Vec<(String, String)>,
    /// Measured device windows for the enrichment phase (#118 ledger).
    pub gpu_kernel_seconds: f64,
    /// Measured device→host time (#294/#300), separate from kernel time.
    pub gpu_d2h_seconds: f64,
    /// Measured host staging writes of GPU output blocks (#300).
    pub gpu_write_seconds: f64,
    /// Write bandwidth charged by cost model v2 (#300), bytes/second.
    pub write_bw: f64,
    pub gpu_transfer_bytes: u64,
    /// Measured host→device and device→host windows (#294 benchmark ledger).
    pub gpu_h2d_seconds: f64,
    /// #294: FNV-1a 64 of the resolved device read projection.
    pub projection_hash: Option<String>,
    /// #294: exact header columns the projected transport read.
    pub projection_columns: Vec<String>,
    pub read_seconds: f64,
    pub feature_seconds: f64,
    /// Host-thread partition of `feature_seconds` (#300): `feature_read_seconds` + `feature_context_seconds` + `feature_gpu_seconds` + `gpu_write_seconds` + `feature_cpu_seconds` + `feature_other_seconds`. The device times above are busy measures and may overlap this timeline.
    pub feature_read_seconds: f64,
    pub feature_context_seconds: f64,
    pub feature_gpu_seconds: f64,
    pub feature_cpu_seconds: f64,
    pub feature_other_seconds: f64,
    pub xs_seconds: f64,
    pub write_seconds: f64,
    pub total_seconds: f64,
    pub peak_rss_bytes: u64,
}

impl Report {
    pub fn text_lines(&self) -> Vec<String> {
        let mut out = vec![
            format!(
                "# enrich: rows={} cols_in={} cols_out={} groups={} symbols={} bar_seconds={}",
                self.rows,
                self.columns_in,
                self.columns_out,
                self.groups,
                self.symbols,
                self.bar_seconds
            ),
            format!(
                "# enrich: features={} labels={} group_rows={} chunk_rows={}",
                self.features.iter().filter(|f| !f.label).count(),
                self.features.iter().filter(|f| f.label).count(),
                self.group_rows,
                self.chunk_rows
            ),
            format!(
                "# enrich: backend={} gpu_requested={} gpu_available={} device={}",
                self.backend,
                self.gpu_requested,
                self.gpu_available,
                self.gpu_device.as_deref().unwrap_or("none")
            ),
            format!(
                "# enrich: gpu_columns={} cpu_columns={} gpu_share={:.3} fallback={} kernel={:.3}s d2h={:.3}s write={:.3}s transfer_bytes={} write_bw={:.0}",
                self.gpu_columns,
                self.cpu_columns,
                self.gpu_share,
                self.gpu_fallback,
                self.gpu_kernel_seconds,
                self.gpu_d2h_seconds,
                self.gpu_write_seconds,
                self.gpu_transfer_bytes,
                self.write_bw
            ),
            format!(
                "# enrich: read={:.3}s feature={:.3}s xs={:.3}s write={:.3}s total={:.3}s peak_rss={:.1} MiB",
                self.read_seconds,
                self.feature_seconds,
                self.xs_seconds,
                self.write_seconds,
                self.total_seconds,
                self.peak_rss_bytes as f64 / (1024.0 * 1024.0)
            ),
            format!(
                "# enrich: feature_parts read={:.3}s context={:.3}s gpu={:.3}s write={:.3}s cpu={:.3}s other={:.3}s",
                self.feature_read_seconds,
                self.feature_context_seconds,
                self.feature_gpu_seconds,
                self.gpu_write_seconds,
                self.feature_cpu_seconds,
                self.feature_other_seconds
            ),
        ];
        if let Some(hash) = &self.projection_hash {
            out.push(format!(
                "# enrich: projection_hash={} columns={}",
                hash,
                self.projection_columns.join(",")
            ));
        }
        for f in &self.features {
            out.push(format!(
                "# enrich-feature: name={} warmup={} nan_rows={} prevalence={:.6} first_finite_row={} first_finite_time={} first_finite_iso={} label={}",
                f.name,
                f.warmup,
                f.nan_rows,
                f.prevalence,
                f.first_finite_row
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                f.first_finite_time
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                f.first_finite_time
                    .map(format_utc_seconds)
                    .unwrap_or_else(|| "none".to_string()),
                f.label
            ));
        }
        for l in &self.long_labels {
            let r = &l.report;
            out.push(format!(
                "# enrich-label: name={} target={} long_only=true rows={} valid_rows={} barrier_rows={} clipped_share={:.6} cost_to_move={:.6} effective_n={:.3} purge_seconds={} label_hash={}",
                l.spec.name, l.target, r.rows, r.valid_rows, r.barrier_rows, r.clipped_share,
                r.cost_to_move, r.effective_n, r.purge_seconds, r.label_hash
            ));
            for m in &r.top_moves {
                out.push(format!(
                    "# enrich-label-move: name={} symbol={} time={} iso={} fwd={:.6}",
                    l.spec.name,
                    m.symbol.as_deref().unwrap_or("none"),
                    m.time,
                    format_utc_seconds(m.time),
                    m.fwd
                ));
            }
        }
        for (name, reason) in &self.gpu_fallback_columns {
            out.push(format!("# enrich-fallback: name={name} reason={reason}"));
        }
        for p in &self.plans {
            out.push(format!(
                "# enrich-plan: group={} rows={} cols={} gpu_cols={} cpu_cols={} op={} cpu_ms={:.1} gpu_ms={:.1} backend={} reason={}",
                p.group,
                p.rows,
                p.cols,
                p.gpu_cols,
                p.cpu_cols,
                p.op,
                p.cpu_ms,
                p.gpu_ms,
                p.backend,
                p.reason
            ));
        }
        out
    }

    pub fn json(&self) -> String {
        let features: Vec<String> = self
            .features
            .iter()
            .map(|f| {
                format!(
                    "{{\"name\":{},\"warmup\":{},\"nan_rows\":{},\"prevalence\":{:.12},\
                     \"first_finite_row\":{},\"first_finite_time\":{},\"first_finite_iso\":{},\
                     \"label\":{},\"sha256\":{}}}",
                    jstr(&f.name),
                    f.warmup,
                    f.nan_rows,
                    f.prevalence,
                    f.first_finite_row
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    f.first_finite_time
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    f.first_finite_time
                        .map(|v| jstr(&format_utc_seconds(v)))
                        .unwrap_or_else(|| "null".to_string()),
                    f.label,
                    jstr(&f.hash)
                )
            })
            .collect();
        let long_labels: Vec<String> = self
            .long_labels
            .iter()
            .map(|l| {
                format!(
                    "{{\"name\":{},\"target\":{},\"columns\":[{}],\"report\":{}}}",
                    jstr(&l.spec.name),
                    jstr(&l.target),
                    l.columns.iter().map(|c| jstr(c)).collect::<Vec<_>>().join(","),
                    l.report.json()
                )
            })
            .collect();
        let plans: Vec<String> = self
            .plans
            .iter()
            .map(|p| {
                format!(
                    "{{\"group\":{},\"rows\":{},\"columns\":{},\"gpu_columns\":{},\"cpu_columns\":{},\"op\":{},\"cpu_ms\":{:.3},\"gpu_ms\":{:.3},\"backend\":{},\"reason\":{}}}",
                    jstr(&p.group),
                    p.rows,
                    p.cols,
                    p.gpu_cols,
                    p.cpu_cols,
                    jstr(&p.op),
                    p.cpu_ms,
                    p.gpu_ms,
                    jstr(p.backend),
                    jstr(&p.reason)
                )
            })
            .collect();
        format!(
            "{{\n  \"tool\":\"rcsr enrich\",\n  \"input\":{},\"output\":{},\n  \
             \"input_sha256\":{},\"output_sha256\":{},\"output_bytes\":{},\n  \
             \"rows\":{},\"columns_in\":{},\"columns_out\":{},\"groups\":{},\"symbols\":{},\n  \
             \"group_rows\":{},\"chunk_rows\":{},\"bar_seconds\":{},\n  \
             \"backend\":{},\"gpu_requested\":{},\"gpu_available\":{},\"gpu_device\":{},\n  \
             \"gpu\":{{\"columns\":{},\"cpu_columns\":{},\"share\":{:.6},\"fallback\":{},\"kernel_s\":{:.6},\"h2d_s\":{:.6},\"d2h_s\":{:.6},\"write_s\":{:.6},\"transfer_bytes\":{},\"write_bw\":{:.0},\"fallback_columns\":[{}]}},\n  \
             \"projection\":{{\"hash\":{},\"columns\":[{}]}},\n  \
             \"timing\":{{\"read_s\":{:.6},\"feature_s\":{:.6},\"feature_read_s\":{:.6},\"feature_context_s\":{:.6},\"feature_gpu_s\":{:.6},\"feature_cpu_s\":{:.6},\"feature_other_s\":{:.6},\"xs_s\":{:.6},\"write_s\":{:.6},\"total_s\":{:.6}}},\n  \
             \"peak_rss_bytes\":{},\n  \"features\":[{}],\n  \"long_labels\":[{}],\n  \"plan\":[{}]\n}}\n",
            jstr(&self.input),
            jstr(&self.output),
            jstr(&self.input_sha256),
            jstr(&self.output_sha256),
            self.output_bytes,
            self.rows,
            self.columns_in,
            self.columns_out,
            self.groups,
            self.symbols,
            self.group_rows,
            self.chunk_rows,
            self.bar_seconds,
            jstr(&self.backend),
            jstr(&self.gpu_requested),
            self.gpu_available,
            self.gpu_device
                .as_deref()
                .map(jstr)
                .unwrap_or_else(|| "null".to_string()),
            self.gpu_columns,
            self.cpu_columns,
            self.gpu_share,
            self.gpu_fallback,
            self.gpu_kernel_seconds,
            self.gpu_h2d_seconds,
            self.gpu_d2h_seconds,
            self.gpu_write_seconds,
            self.gpu_transfer_bytes,
            self.write_bw,
            self
                .gpu_fallback_columns
                .iter()
                .map(|(n, r)| format!("{{\"name\":{},\"reason\":{}}}", jstr(n), jstr(r)))
                .collect::<Vec<_>>()
                .join(","),
            self.projection_hash
                .as_deref()
                .map(jstr)
                .unwrap_or_else(|| "null".to_string()),
            self.projection_columns
                .iter()
                .map(|c| jstr(c))
                .collect::<Vec<_>>()
                .join(","),
            self.read_seconds,
            self.feature_seconds,
            self.feature_read_seconds,
            self.feature_context_seconds,
            self.feature_gpu_seconds,
            self.feature_cpu_seconds,
            self.feature_other_seconds,
            self.xs_seconds,
            self.write_seconds,
            self.total_seconds,
            self.peak_rss_bytes,
            features.join(","),
            long_labels.join(","),
            plans.join(",")
        )
    }
}

/// Columns planned for the device that the CPU computed instead, with the
/// path that failed. Each group prints one warning.
#[derive(Default)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct FallbackLog {
    pub(crate) columns: Vec<(String, String)>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl FallbackLog {
    pub(crate) fn record<'a>(
        &mut self,
        path: &str,
        detail: &str,
        names: impl IntoIterator<Item = &'a str>,
    ) {
        let fresh: Vec<&str> = names
            .into_iter()
            .filter(|n| !self.columns.iter().any(|(c, _)| c == n))
            .collect();
        if fresh.is_empty() {
            return;
        }
        eprintln!(
            "warning: device {path} failed ({detail}); computing {} columns on the CPU: {}",
            fresh.len(),
            fresh.join(",")
        );
        for n in fresh {
            self.columns.push((n.to_string(), format!("{path}: {detail}")));
        }
    }
}

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
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Peak resident set size from `/proc/self/status` (`VmHWM`), or 0 elsewhere.
pub fn peak_rss_bytes() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok()) {
                return kb.saturating_mul(1024);
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// CPU/GPU cost model
// ---------------------------------------------------------------------------

/// Operation class a feature family reduces to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpClass {
    Elementwise,
    Calendar,
    RollingSum,
    RollingStd,
    RollingExtrema,
    WilderScan,
    Rank,
    Join,
}

impl OpClass {
    pub fn name(self) -> &'static str {
        match self {
            OpClass::Elementwise => "elementwise",
            OpClass::Calendar => "calendar-trig",
            OpClass::RollingSum => "rolling-sum",
            OpClass::RollingStd => "rolling-std",
            OpClass::RollingExtrema => "rolling-extrema",
            OpClass::WilderScan => "wilder-scan",
            OpClass::Rank => "segmented-rank",
            OpClass::Join => "join",
        }
    }
    fn cpu_cells_per_second(self) -> f64 {
        // Calibrated on the 3.875M-row store (2026-10-21), including the
        // per-value temp write and hash that both arms pay. These are measured
        // throughputs, not peak-core claims.
        match self {
            OpClass::Elementwise => 15_000_000.0,
            OpClass::Calendar => 10_000_000.0,
            OpClass::RollingSum => 12_000_000.0,
            OpClass::RollingStd => 10_000_000.0,
            OpClass::RollingExtrema => 15_000_000.0,
            OpClass::WilderScan => 20_000_000.0,
            OpClass::Rank => 4_000_000.0,
            // #299: per-row BTreeMap lookups; 15 lookup columns on the
            // 16.1M-row panel took 451.1 s (0.54 M cells/s), and the 28-column
            // family with 7 rolling-beta columns 1250.6 s (0.36 M cells/s).
            OpClass::Join => 360_000.0,
        }
    }
    fn gpu_cells_per_second(self) -> f64 {
        // Conservative f64 figures for the RTX 3060; f64 is 1/64 of f32 on
        // consumer Ampere, so these are deliberately far below a TFLOPS claim.
        // Sequential one-thread-per-segment kernels are slower per cell than
        // the CPU, which is why their rows below sit under the CPU rate.
        match self {
            OpClass::Elementwise => 350_000_000.0,
            OpClass::Calendar => 300_000_000.0,
            OpClass::RollingSum => 100_000_000.0,
            OpClass::RollingStd => 60_000_000.0,
            // #298 blocked extrema: 4.58 ms for 1M rows at w=96 on the RTX
            // 3060 (218M rows/s). 180M is the conservative planner figure.
            OpClass::RollingExtrema => 180_000_000.0,
            OpClass::WilderScan => 8_000_000.0,
            OpClass::Rank => 40_000_000.0,
            // #299: kernel + D2H of the 28-column family, 451 M cells in 3.78 s.
            OpClass::Join => 100_000_000.0,
        }
    }
}

/// One family's dispatch decision and its estimated cost.
#[derive(Clone, Debug)]
pub struct GroupPlan {
    pub group: String,
    pub rows: usize,
    pub cols: usize,
    /// Columns in this family with an exact CUDA kernel.
    pub gpu_cols: usize,
    /// Columns that stay on the CPU (unsupported names or a CPU dispatch).
    pub cpu_cols: usize,
    pub op: String,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub backend: &'static str,
    pub reason: String,
}

/// Families in the catalogue, with the operation class they are costed as.
/// Family of a compact feature, used by the cost model and the registry.
pub fn compact_family(name: &str) -> &'static str {
    match name {
        "r16" | "r96" | "r672" => "returns",
        "ma96" | "ma672" | "ma_slope672" | "macd_hist" => "ma",
        "vol16" | "vol96" | "vol_ts" | "vol_ratio" | "vol_z2880" | "parkinson96"
        | "rv_ratio" | "bb96" | "bb_width96" | "vol_z96" | "volrat" => "vol",
        "natr14" | "adx14" | "di_diff14" | "batr" | "gapatr" => "atr",
        "rsi14" | "mfi14" => "rsi",
        "cmf20" | "taker_ratio96" | "taker_imb16" | "trade_intensity96" => "flow",
        "vwap_dev96" => "flow",
        "clv1" | "clv4" | "body1" | "wasym1" | "lower_wick" | "rng1" => "candle",
        "ou_hl96" | "acf1" | "vr4" | "rskew96" | "jump_share" | "semivar_asym"
        | "cusum_abs_z" => "exotic",
        "btc_r96" | "btc_vol96" | "rel_btc96" | "beta384" => "btc",
        "tod_sin" | "tod_cos" | "dow_sin" | "dow_cos" => "calendar",
        "don96" => "donchian",
        _ => "other",
    }
}

pub fn family_of(name: &str) -> &'static str {
    if FLOW_NAMES.contains(&name) {
        return "flow";
    }
    if PRICE_ACTION_NAMES.contains(&name) {
        return "price_action";
    }
    if is_compact(name) {
        return compact_family(name);
    }
    if is_mtf(name) {
        return "mtf";
    }
    match name {
        "ret1" | "ret5" => return "returns",
        "rmean5" | "rstd5" | "close_pos20" => return "exotic",
        "vwap20_dev" => return "flow",
        "rel_r16" | "rel_r96" => return "btc",
        _ => {}
    }
    if name.starts_with("r") && name[1..].parse::<usize>().is_ok() {
        "returns"
    } else if name.starts_with("ma") || name.starts_with("ema") {
        "ma"
    } else if name.starts_with("vol")
        || name == "vz96"
        || name.starts_with("skew")
        || name.starts_with("kurt")
        || name.starts_with("park")
        || name.starts_with("gk")
    {
        "vol"
    } else if name.starts_with("don") {
        "donchian"
    } else if name.starts_with("atr")
        || name.starts_with("natr")
        || name.starts_with("adx")
        || name.starts_with("di_")
        || name.starts_with("stoch")
    {
        "atr"
    } else if name.starts_with("rsi") {
        "rsi"
    } else if name.starts_with("bb") {
        "bollinger"
    } else if name.starts_with("lag_") {
        "lags"
    } else if name.starts_with("xs_") {
        "xs"
    } else if name.starts_with("btc")
        || name.starts_with("rel_btc")
        || name.starts_with("beta_btc")
        || name.starts_with("cross_")
        || name.starts_with("mkt_")
        || name.starts_with("eth_")
        || name.starts_with("rel_eth")
        || name.starts_with("beta_eth")
    {
        "btc"
    } else if name.starts_with("dow_")
        || name.starts_with("hour_")
        || name.starts_with("month_")
        || name.starts_with("dom_")
        || name.starts_with("tod_")
    {
        "calendar"
    } else if matches!(
        name,
        "range1"
            | "clv1"
            | "clv4"
            | "body1"
            | "upper_wick1"
            | "lower_wick1"
            | "open_gap1"
            | "wasym1"
            | "rng1"
    ) {
        "candle"
    } else if py15m_names().iter().any(|n| n == name) {
        "py15m"
    } else {
        "other"
    }
}

/// True when this build has an exact enrichment kernel for `name`.
///
/// The CPU path is the reference. A CUDA kernel is registered only for the
/// families whose parity test (`tests/enrich.rs::gpu_columns_match_cpu`) passes;
/// every other name falls back to the CPU without changing the output.
pub fn gpu_supports(name: &str) -> bool {
    cfg!(feature = "cuda") && device_kernel_exists(name)
}

/// True when the column has a device kernel in a `--features cuda` build,
/// whatever this build is. Parity tiers are declared from this so the registry
/// and `schema` do not depend on the build.
pub fn device_kernel_exists(name: &str) -> bool {
    // #297: the fused flow and price-action producers materialise their
    // columns with a declared parity tier; they must be reachable even when a
    // shared name (cmf20) is also in the compact subset.
    if crate::features_gpu::is_device_flow(name)
        || crate::features_gpu::is_device_price_action(name)
    {
        return true;
    }
    // #299: benchmark joins, relative returns and rolling beta, including the
    // compact names that share the benchmark join.
    if crate::bench_gpu::is_device_bench(name) {
        return true;
    }
    if is_compact(name)
        && !matches!(
            name,
            // #298: exact tail kernels. The compact pass remains the CPU
            // reference for every name not listed here.
            "clv1"
                | "dow_sin"
                | "dow_cos"
                | "macd_hist"
                | "r16"
                | "r96"
                | "r672"
                | "ma96"
                | "ma672"
                | "ma_slope672"
                | "natr14"
                | "adx14"
                | "di_diff14"
                | "bb96"
                | "bb_width96"
                | "batr"
                | "gapatr"
                | "volrat"
                | "don96"
                | "ou_hl96"
                | "acf1"
                | "vr4"
                | "rskew96"
                | "jump_share"
                | "semivar_asym"
                | "cusum_abs_z"
                | "tod_sin"
                | "tod_cos"
                | "vol16"
                | "vol96"
                | "vol_z96"
                | "rsi14"
        )
    {
        // The compact profile is computed by one CPU reference pass. Only the
        // compact names whose arithmetic has an exact registered kernel may
        // dispatch; the rest stay on the reference path.
        return false;
    }
    if name.starts_with("xs_rank_") {
        // Cross-sectional ranks use the device segmented sort in the xs pass.
        return true;
    }
    if is_mtf(name) {
        // #296: the complete MTF family has a device resample + causal join.
        return true;
    }
    if name == "ret1" || name == "ret5" {
        return true;
    }
    if name == "ma_slope672" {
        return true;
    }
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        if k > 0 {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        if w > 0 {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        if w > 0 {
            return true;
        }
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        if (1..=10).contains(&k) {
            return true;
        }
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        if w == 20 || w == 96 {
            return true;
        }
    }
    matches!(
        name,
        "bb20_bw"
            | "bb50_bw"
            | "bb_width96"
            | "bb20_z"
            | "bb50"
            | "bb96"
            | "vz96"
            | "vol_z96"
            | "range1"
            | "clv1"
            | "dow_sin"
            | "dow_cos"
            | "hour_sin"
            | "hour_cos"
            | "month_sin"
            | "month_cos"
            | "dom_sin"
            | "dom_cos"
            | "tod_sin"
            | "tod_cos"
            | "ema8"
            | "ema20"
            | "ema50"
            | "ema100"
            | "ema200"
            | "macd"
            | "macd_sig"
            | "macd_hist"
            | "rsi6"
            | "rsi14"
            | "rsi28"
            | "atr14"
            | "natr14"
            | "atr50"
            | "batr"
            | "gapatr"
            | "volrat"
            | "stoch_k"
            | "stoch_d"
            | "adx14"
            | "di_plus"
            | "di_minus"
            | "di_diff14"
            | "rmean5"
            | "rstd5"
            | "close_pos20"
            | "ou_hl96"
            | "acf1"
            | "vr4"
            | "rskew96"
            | "jump_share"
            | "semivar_asym"
            | "cusum_abs_z"
    )
}

/// True when any CUDA enrichment kernel is compiled into this build.
pub fn gpu_kernels_compiled() -> bool {
    cfg!(feature = "cuda")
}

/// Compare estimated CPU and GPU time for each family.
///
/// GPU is chosen only when at least one column has an exact kernel,
/// `gpu_available` is true, the family has at least [`GPU_MIN_ROWS`] rows and
/// the estimated GPU time for the supported columns (transfer + kernel) is
/// strictly lower than the CPU estimate for the same columns. Small stores
/// stay on the CPU.
pub const GPU_MIN_ROWS: usize = 1_000_000;

static GPU_FLOOR_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Test hook (#303): lower the row floor below which `--gpu on` still plans
/// the CPU, so parity fixtures and prefix cuts below one million rows run on
/// the device. Process-wide; never called by the CLI.
pub fn set_gpu_floor_for_tests(rows: usize) {
    GPU_FLOOR_OVERRIDE.store(rows, std::sync::atomic::Ordering::Relaxed);
}

fn gpu_row_floor() -> usize {
    match GPU_FLOOR_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        usize::MAX => GPU_MIN_ROWS,
        n => n,
    }
}
pub const GPU_TRANSFER_BYTES_PER_SECOND: f64 = 12_000_000_000.0;
/// Measured sequential write bandwidth of the shared ext4 store on
/// `/mnt/lager` (2026-10-23, 512 MiB `dd conv=fdatasync`: **222 MB/s**). This is
/// the temp-file write term the pre-#300 model omitted. It is deliberately a
/// conservative filesystem figure: the same disk measured 64 MB/s while the
/// 16M-row full-catalogue run was appending, so a planner on a busier disk
/// should pass a smaller value through [`plan_groups_with_write_bw`] (the CLI
/// exposes `--write-bw`).
pub const GPU_WRITE_BYTES_PER_SECOND: f64 = 222_000_000.0;
/// Device-to-host transfer bandwidth used by the cost model. Kept separate from
/// the resident upload rate so the write term cannot hide behind the bus.
pub const GPU_D2H_BYTES_PER_SECOND: f64 = GPU_TRANSFER_BYTES_PER_SECOND;
/// Hard cap for the per-family output arena (#300). A block whose columns do
/// not fit falls back to per-column staging.
pub const GPU_OUTPUT_ARENA_BYTES: u64 = 64 * 1024 * 1024;

/// Measure sequential write bandwidth (bytes/second) in `dir`.
///
/// Writes `bytes` of zeroes with a 1 MiB buffer and `sync_all` on the file, so
/// the result is a durable-write figure rather than page-cache speed. Used by
/// the ignored `measure_write_bandwidth` test and by operators calibrating
/// `--write-bw` for a different filesystem.
pub fn measure_write_bandwidth(dir: &Path, bytes: u64) -> Result<f64, String> {
    use std::io::Write;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let path = dir.join(format!("rcsr-bw-{}.bin", std::process::id()));
    let mut file = std::fs::File::create(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let buf = vec![0u8; 1 << 20];
    let started = std::time::Instant::now();
    let mut left = bytes;
    while left > 0 {
        let n = left.min(buf.len() as u64) as usize;
        file.write_all(&buf[..n])
            .map_err(|e| format!("{}: {}", path.display(), e))?;
        left -= n as u64;
    }
    file.sync_all()
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    let seconds = started.elapsed().as_secs_f64();
    drop(file);
    let _ = std::fs::remove_file(&path);
    if seconds <= 0.0 {
        return Err("write bandwidth clock produced zero seconds".to_string());
    }
    Ok(bytes as f64 / seconds)
}

/// Fixed per-run device setup (context, probe, launch glue).
pub const GPU_FIXED_OVERHEAD_MS: f64 = 50.0;
/// One-time host read of the source columns plus the resident upload, per
/// million rows. Measured at roughly 0.65 s/Mrow on the 3.875M-row store; it is
/// why a ten-column family does not pay for a GPU pass even though its kernel
/// is milliseconds long.
pub const GPU_CONTEXT_MS_PER_MROW: f64 = 650.0;

pub fn plan_groups(
    counts: &BTreeMap<&'static str, (usize, OpClass)>,
    gpu_cols: &BTreeMap<&'static str, usize>,
    n_rows: usize,
    gpu_requested: GpuMode,
    gpu_available: bool,
) -> Vec<GroupPlan> {
    plan_groups_full(
        counts,
        gpu_cols,
        n_rows,
        usize::MAX,
        gpu_requested,
        gpu_available,
        GPU_WRITE_BYTES_PER_SECOND,
    )
}

/// Cost model v2 (#300): include the measured output write.
///
/// ```text
/// gpu_s = fixed + context(n) + h2d/BW_h2d + cells/rate_kernel
///       + d2h/BW_d2h + write_bytes/BW_write
/// ```
///
/// `write_bps` is the calibrated filesystem write bandwidth. A slower disk can
/// flip a family to the CPU, which is exactly the decision the old model could
/// not express.
pub fn plan_groups_with_write_bw(
    counts: &BTreeMap<&'static str, (usize, OpClass)>,
    gpu_cols: &BTreeMap<&'static str, usize>,
    n_rows: usize,
    gpu_requested: GpuMode,
    gpu_available: bool,
    write_bps: f64,
) -> Vec<GroupPlan> {
    plan_groups_full(
        counts,
        gpu_cols,
        n_rows,
        usize::MAX,
        gpu_requested,
        gpu_available,
        write_bps,
    )
}

/// Minimum symbol segments for an automatic fused-family dispatch. The fused
/// consumers parallelise over segments: with two symbols the 109-block device
/// grid is idle and the CPU reference is faster. `--gpu on` still forces it.
pub const FUSED_MIN_SEGMENTS: usize = 32;

/// [`plan_groups`] with the store's symbol count, so fused families can decline
/// when their segment-parallel consumers would not fill the device.
pub fn plan_groups_with_segments(
    counts: &BTreeMap<&'static str, (usize, OpClass)>,
    gpu_cols: &BTreeMap<&'static str, usize>,
    n_rows: usize,
    n_segments: usize,
    gpu_requested: GpuMode,
    gpu_available: bool,
) -> Vec<GroupPlan> {
    plan_groups_full(
        counts,
        gpu_cols,
        n_rows,
        n_segments,
        gpu_requested,
        gpu_available,
        GPU_WRITE_BYTES_PER_SECOND,
    )
}

/// The combined planner: symbol count (fused-family floor) and calibrated
/// write bandwidth (#300).
#[allow(clippy::too_many_arguments)]
pub fn plan_groups_full(
    counts: &BTreeMap<&'static str, (usize, OpClass)>,
    gpu_cols: &BTreeMap<&'static str, usize>,
    n_rows: usize,
    n_segments: usize,
    gpu_requested: GpuMode,
    gpu_available: bool,
    write_bps: f64,
) -> Vec<GroupPlan> {
    let mut out = Vec::new();
    let write_bps = if write_bps.is_finite() && write_bps > 0.0 {
        write_bps
    } else {
        GPU_WRITE_BYTES_PER_SECOND
    };
    for (group, (cols, op)) in counts {
        let op = *op;
        let cells = n_rows.saturating_mul(*cols).max(1) as f64;
        let cpu_s = match crate::features_gpu::fused_cost(group) {
            Some(cost) => {
                cost.cpu_ms_per_mrow_column * (n_rows as f64 / 1_000_000.0)
                    * (*cols as f64)
                    / 1000.0
            }
            None => cells / op.cpu_cells_per_second(),
        };
        let supported = (*gpu_cols.get(group).unwrap_or(&0)).min(*cols);
        let gpu_cells_n = n_rows.saturating_mul(supported).max(1);
        let gpu_cells = gpu_cells_n as f64;
        let transfer_bytes = gpu_cells_n as f64 * 8.0;
        let context_ms =
            GPU_FIXED_OVERHEAD_MS + GPU_CONTEXT_MS_PER_MROW * n_rows as f64 / 1_000_000.0;
        let gpu_s = if let Some(fused_ms) =
            crate::features_gpu::estimate_fused_ms(group, n_rows, supported)
        {
            // Fused base + consumers: the base is paid once, not per column.
            // The write/D2H terms are #300's calibrated output-path cost.
            context_ms / 1000.0
                + transfer_bytes / GPU_TRANSFER_BYTES_PER_SECOND
                + fused_ms / 1000.0
                + transfer_bytes / GPU_D2H_BYTES_PER_SECOND
                + transfer_bytes / write_bps
        } else {
            context_ms / 1000.0
                + transfer_bytes / GPU_TRANSFER_BYTES_PER_SECOND
                + gpu_cells / op.gpu_cells_per_second()
                + transfer_bytes / GPU_D2H_BYTES_PER_SECOND
                + transfer_bytes / write_bps
        };
        let cpu_supported_s = gpu_cells / op.cpu_cells_per_second();
        let (backend, reason) = if gpu_requested == GpuMode::Off {
            ("cpu", "GPU disabled by --no-gpu/--cpu".to_string())
        } else if n_rows < gpu_row_floor() {
            (
                "cpu",
                format!("{} rows is below the {} row GPU floor", n_rows, gpu_row_floor()),
            )
        } else if supported == 0 {
            ("cpu", "no exact GPU kernel registered for this family".to_string())
        } else if !cfg!(feature = "cuda") {
            ("cpu", "built without --features cuda".to_string())
        } else if !gpu_available {
            ("cpu", "no usable CUDA device".to_string())
        } else if gpu_requested == GpuMode::On {
            ("gpu", "forced by --gpu on".to_string())
        } else if crate::features_gpu::fused_cost(group).is_some()
            && n_segments < FUSED_MIN_SEGMENTS
        {
            (
                "cpu",
                format!(
                    "{} symbol segments is below the {} segment floor for the fused {} family",
                    n_segments, FUSED_MIN_SEGMENTS, group
                ),
            )
        } else if gpu_s < cpu_supported_s {
            ("gpu", "estimated GPU time is below CPU".to_string())
        } else {
            ("cpu", "estimated GPU time is not below CPU".to_string())
        };
        let dispatched = backend == "gpu";
        out.push(GroupPlan {
            group: (*group).to_string(),
            rows: n_rows,
            cols: *cols,
            gpu_cols: if dispatched { supported } else { 0 },
            cpu_cols: if dispatched { *cols - supported } else { *cols },
            op: op.name().to_string(),
            cpu_ms: cpu_s * 1000.0,
            gpu_ms: gpu_s * 1000.0,
            backend,
            reason,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Series loading and segmentation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Segment {
    symbol: String,
    key: u32,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug)]
struct Series {
    symbol: String,
    times: Vec<i64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    quote_volume: Vec<f64>,
    trades: Vec<f64>,
    taker_buy_base: Vec<f64>,
    step: i64,
}

impl Series {
    fn len(&self) -> usize {
        self.times.len()
    }
    fn bars(&self) -> Vec<Bar> {
        (0..self.len())
            .map(|i| {
                let b = Bar {
                    time: self.times[i],
                    open: self.open[i],
                    high: self.high[i],
                    low: self.low[i],
                    close: self.close[i],
                    volume: self.volume[i],
                    quote_volume: self.quote_volume[i],
                    trades: self.trades[i],
                    taker_buy_base: self.taker_buy_base[i],
                    taker_buy_quote: 0.0,
                    complete: true,
                    traded: self.volume[i] > 0.0,
                };
                b
            })
            .collect()
    }
}

fn find_col(names: &[String], aliases: &[&str]) -> Option<usize> {
    names.iter().position(|n| {
        let l = n.to_ascii_lowercase();
        aliases.iter().any(|a| l == *a)
    })
}

fn find_symbol_col(names: &[String], explicit: Option<&str>) -> Option<usize> {
    if let Some(e) = explicit {
        return names.iter().position(|n| n == e);
    }
    find_col(names, &["symbol", "ticker", "asset", "pair"])
}

fn find_time_col(names: &[String], explicit: Option<&str>) -> Option<usize> {
    if let Some(e) = explicit {
        return names.iter().position(|n| n == e);
    }
    find_col(names, &["ts", "time", "timestamp", "date", "datetime", "open_time", "unix"])
}

/// Build the symbol segments and infer the global bar spacing.
///
/// Only the time and symbol columns are read in this pass. Every symbol must
/// occupy one contiguous run of rows; a symbol that reappears after another is
/// refused, because the per-symbol feature state cannot be carried back.
fn segments(store: &FileStore, time_col: usize, symbol_col: Option<usize>) -> Result<(Vec<Segment>, i64), String> {
    let mut segs: Vec<Segment> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut step_hist: BTreeMap<i64, usize> = BTreeMap::new();
    let mut prev_time: Option<i64> = None;
    let mut prev_sym: Option<usize> = None;
    let mut current: Option<(String, u32)> = None;
    for g in 0..store.n_groups() {
        let len = store.header().group_len(g);
        let times = match store.header().dtypes()[time_col] {
            DType::I64 => store.read_i64(g, time_col)?,
            _ => store
                .read_f64_group(g, time_col)?
                .into_iter()
                .map(|v| v as i64)
                .collect(),
        };
        let codes: Vec<u32> = match symbol_col {
            None => vec![0u32; len],
            Some(c) => {
                let meta = store.header().column(c).unwrap();
                if meta.has_codes {
                    store.read_codes(g, c)?
                } else {
                    store
                        .read_f64_group(g, c)?
                        .into_iter()
                        .map(|v| if v.is_finite() && v >= 0.0 { v as u32 } else { 0 })
                        .collect()
                }
            }
        };
        let levels: Vec<String> = match symbol_col {
            None => vec!["_all".to_string()],
            Some(c) => store
                .header()
                .levels(c)
                .map(|l| l.to_vec())
                .unwrap_or_else(|| vec!["_all".to_string()]),
        };
        for r in 0..len {
            let code = codes.get(r).copied().unwrap_or(0) as usize;
            let sym = levels
                .get(code)
                .cloned()
                .unwrap_or_else(|| format!("symbol{}", code));
            if current.as_ref().map(|(s, _)| s != &sym).unwrap_or(true) {
                if seen.contains(&sym) {
                    return Err(format!(
                        "symbol '{}' is not contiguous: the store is time-major (symbol rows \
                         interleave) or unsorted. `rcsr enrich` currently needs one contiguous \
                         run per symbol, as written by `rcsr prep`; time-major `rcsr panel` \
                         stores are a follow-up",
                        sym
                    ));
                }
                seen.insert(sym.clone());
                let start = segs.iter().map(|s| s.len).sum::<usize>();
                segs.push(Segment {
                    symbol: sym.clone(),
                    key: 0,
                    start,
                    len: 0,
                });
                current = Some((sym, 0));
            }
            if let Some(seg) = segs.last_mut() {
                seg.len += 1;
            }
            let t = times[r];
            if let (Some(pt), Some(ps)) = (prev_time, prev_sym) {
                if ps == code {
                    let d = t - pt;
                    if d > 0 {
                        *step_hist.entry(d).or_insert(0) += 1;
                    }
                }
            }
            prev_time = Some(t);
            prev_sym = Some(code);
        }
    }
    let step = step_hist
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(d, _)| d)
        .unwrap_or(0);
    // Ties in a cross-section break by symbol name, then row order, so assign
    // each symbol its lexical rank rather than its first-appearance order.
    let mut names: Vec<String> = segs.iter().map(|s| s.symbol.clone()).collect();
    names.sort();
    for seg in &mut segs {
        seg.key = names
            .binary_search(&seg.symbol)
            .map(|k| k as u32)
            .unwrap_or(0);
    }
    Ok((segs, step))
}

fn segment_column(store: &FileStore, seg: &Segment, col: usize) -> Result<Vec<f64>, String> {
    let group_rows = store.group_rows();
    let mut out = Vec::with_capacity(seg.len);
    let first = seg.start / group_rows;
    let last = (seg.start + seg.len.saturating_sub(1)) / group_rows;
    for g in first..=last {
        let vals = store.read_f64_group(g, col)?;
        let base = g * group_rows;
        let lo = seg.start.max(base);
        let hi = (seg.start + seg.len).min(base + vals.len());
        if hi > lo {
            out.extend_from_slice(&vals[lo - base..hi - base]);
        }
    }
    Ok(out)
}

fn segment_i64(store: &FileStore, seg: &Segment, col: usize, dtype: DType) -> Result<Vec<i64>, String> {
    if dtype != DType::I64 {
        return Ok(segment_column(store, seg, col)?
            .into_iter()
            .map(|v| v as i64)
            .collect());
    }
    let group_rows = store.group_rows();
    let mut out = Vec::with_capacity(seg.len);
    let first = seg.start / group_rows;
    let last = (seg.start + seg.len.saturating_sub(1)) / group_rows;
    for g in first..=last {
        let vals = store.read_i64(g, col)?;
        let base = g * group_rows;
        let lo = seg.start.max(base);
        let hi = (seg.start + seg.len).min(base + vals.len());
        if hi > lo {
            out.extend_from_slice(&vals[lo - base..hi - base]);
        }
    }
    Ok(out)
}

/// Read one row of the time column by global row index. Used only after the
/// feature pass, to turn `first_finite_row` into a timestamp for the report.
fn time_at_row(
    store: &FileStore,
    col: usize,
    dtype: DType,
    row: usize,
) -> Result<i64, String> {
    let group_rows = store.group_rows();
    let g = row / group_rows;
    let within = row % group_rows;
    if dtype == DType::I64 {
        let vals = store.read_i64(g, col)?;
        vals.get(within)
            .copied()
            .ok_or_else(|| format!("time row {} is outside store group {}", row, g))
    } else {
        let vals = store.read_f64_group(g, col)?;
        Ok(vals
            .get(within)
            .copied()
            .ok_or_else(|| format!("time row {} is outside store group {}", row, g))?
            as i64)
    }
}

pub(crate) fn format_utc_seconds(ts: i64) -> String {
    let days = ts.div_euclid(86_400);
    let rem = ts.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(feature = "cuda")]
fn column_all(store: &FileStore, col: usize) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(store.n_rows());
    for g in 0..store.n_groups() {
        out.extend(store.read_f64_group(g, col)?);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn i64_column_all(store: &FileStore, col: usize, dtype: DType) -> Result<Vec<i64>, String> {
    if dtype != DType::I64 {
        return Ok(column_all(store, col)?
            .into_iter()
            .map(|v| v as i64)
            .collect());
    }
    let mut out = Vec::with_capacity(store.n_rows());
    for g in 0..store.n_groups() {
        out.extend(store.read_i64(g, col)?);
    }
    Ok(out)
}

fn load_series(
    store: &FileStore,
    seg: &Segment,
    cols: &Cols,
    step: i64,
) -> Result<Series, String> {
    let times = segment_i64(store, seg, cols.time, cols.time_dtype)?;
    if times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!(
            "symbol '{}' has non-increasing timestamps; sort by (symbol, time) first",
            seg.symbol
        ));
    }
    let get = |c: usize| segment_column(store, seg, c);
    let close = get(cols.close)?;
    let open = get(cols.open)?;
    let high = get(cols.high)?;
    let low = get(cols.low)?;
    let volume = match cols.volume {
        Some(c) => get(c)?,
        None => vec![f64::NAN; seg.len],
    };
    let quote_volume = match cols.quote_volume {
        Some(c) => get(c)?,
        None => vec![f64::NAN; seg.len],
    };
    let trades = match cols.trades {
        Some(c) => get(c)?,
        None => vec![f64::NAN; seg.len],
    };
    let taker_buy_base = match cols.taker_buy_base {
        Some(c) => get(c)?,
        None => vec![f64::NAN; seg.len],
    };
    Ok(Series {
        symbol: seg.symbol.clone(),
        times,
        open,
        high,
        low,
        close,
        volume,
        quote_volume,
        trades,
        taker_buy_base,
        step,
    })
}

struct Cols {
    time: usize,
    time_dtype: DType,
    open: usize,
    high: usize,
    low: usize,
    close: usize,
    volume: Option<usize>,
    quote_volume: Option<usize>,
    trades: Option<usize>,
    taker_buy_base: Option<usize>,
    /// Only read by the CUDA context (#297); the CPU producers do not need it.
    #[allow(dead_code)]
    taker_buy_quote: Option<usize>,
    symbol: Option<usize>,
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

fn contig(times: &[i64], i: usize, w: usize, step: i64) -> bool {
    if w == 0 {
        return true;
    }
    if i + 1 < w || step <= 0 {
        return false;
    }
    times[i + 1 - w] == times[i] - (w as i64 - 1) * step
}

/// Causal window statistics for a rolling mean/std. Non-finite cells are
/// skipped and counted as missing, so a window with any missing cell is
/// invalid for the strict accessors. Sums are block-local compensated
/// (`window_stats`), so a window depends only on its own rows to rounding.
struct Prefix<'a>(WinStats<'a>);

impl<'a> Prefix<'a> {
    fn new(x: &'a [f64], times: &[i64], step: i64) -> Self {
        Prefix(WinStats::new(x, times, step))
    }
    fn mean(&self, i: usize, w: usize) -> Option<f64> {
        self.0.mean_strict(i, w)
    }
    fn std(&self, i: usize, w: usize) -> Option<f64> {
        self.0.std_pop(i, w)
    }
}

fn ret_k(s: &Series, k: usize) -> Vec<f64> {
    let n = s.len();
    let mut out = vec![f64::NAN; n];
    for i in k..n {
        if s.times[i] - s.times[i - k] == k as i64 * s.step
            && s.close[i] > 0.0
            && s.close[i - k] > 0.0
        {
            out[i] = (s.close[i] / s.close[i - k]).ln();
        }
    }
    out
}

fn true_range(s: &Series) -> Vec<f64> {
    let n = s.len();
    let mut tr = vec![f64::NAN; n];
    for i in 0..n {
        if !(s.high[i].is_finite() && s.low[i].is_finite()) {
            continue;
        }
        tr[i] = if i == 0
            || !s.close[i - 1].is_finite()
            || s.times[i] - s.times[i - 1] != s.step
        {
            s.high[i] - s.low[i]
        } else {
            let pc = s.close[i - 1];
            (s.high[i] - s.low[i])
                .max((s.high[i] - pc).abs())
                .max((s.low[i] - pc).abs())
        };
    }
    tr
}

fn gap_mask(s: &Series) -> Vec<bool> {
    let mut g = vec![false; s.len()];
    for i in 1..s.len() {
        g[i] = s.times[i] - s.times[i - 1] != s.step;
    }
    g
}

/// The Python `wilder()` recurrence used by `enrich_panel_15m.py`: seed on the
/// first finite value and recurse, with no period-sized seed window.
fn wilder_py(x: &[f64], period: usize, gap: &[bool]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if period == 0 {
        return out;
    }
    let a = 1.0 / period as f64;
    let mut prev = f64::NAN;
    for i in 0..n {
        if gap.get(i).copied().unwrap_or(false) {
            prev = f64::NAN;
        }
        let v = x[i];
        if !v.is_finite() {
            continue;
        }
        prev = if !prev.is_finite() { v } else { prev + a * (v - prev) };
        out[i] = prev;
    }
    out
}

fn ema(x: &[f64], span: usize, gap: &[bool]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if span == 0 || n == 0 {
        return out;
    }
    let a = 2.0 / (span as f64 + 1.0);
    let mut prev = f64::NAN;
    for i in 0..n {
        if gap.get(i).copied().unwrap_or(false) {
            prev = f64::NAN;
        }
        if !x[i].is_finite() {
            prev = f64::NAN;
            continue;
        }
        prev = if prev.is_finite() {
            a * x[i] + (1.0 - a) * prev
        } else {
            x[i]
        };
        out[i] = prev;
    }
    out
}

/// Python-style rolling mean over a window: sums finite cells and divides by
/// the finite count, then masks a window that spans a gap.
fn rmean_py(x: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    let p = WinStats::new(x, times, step);
    for i in w - 1..n {
        if !contig(times, i, w, step) {
            continue;
        }
        if let Some(m) = p.mean_finite(i, w) {
            out[i] = m;
        }
    }
    out
}

fn rstd_py(x: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    let p = WinStats::new(x, times, step);
    for i in w - 1..n {
        if !contig(times, i, w, step) {
            continue;
        }
        if let Some(v) = p.std_pop(i, w) {
            out[i] = v;
        }
    }
    out
}

fn rsi_py(close: &[f64], period: usize, gap: &[bool]) -> Vec<f64> {
    let n = close.len();
    let mut d = vec![0.0f64; n];
    for i in 1..n {
        if gap.get(i).copied().unwrap_or(false) {
            // The close-to-close move spans the hole; it is missing, not zero.
            d[i] = f64::NAN;
        } else if close[i].is_finite() && close[i - 1].is_finite() {
            d[i] = close[i] - close[i - 1];
        }
    }
    let up: Vec<f64> = d.iter().map(|v| v.max(0.0)).collect();
    let dn: Vec<f64> = d.iter().map(|v| (-v).max(0.0)).collect();
    let a = wilder_py(&up, period, gap);
    let b = wilder_py(&dn, period, gap);
    (0..n)
        .map(|i| {
            if !(a[i].is_finite() && b[i].is_finite()) || b[i] <= 0.0 {
                f64::NAN
            } else {
                (100.0 - 100.0 / (1.0 + a[i] / b[i])) / 50.0 - 1.0
            }
        })
        .collect()
}

fn calendar_frac(t: i64) -> f64 {
    (t.rem_euclid(86_400) as f64) / 86_400.0
}

fn cyclic(frac: f64) -> (f64, f64) {
    let a = 2.0 * std::f64::consts::PI * frac;
    (a.sin(), a.cos())
}

// ---------------------------------------------------------------------------
// Base arrays and features
// ---------------------------------------------------------------------------

struct Base {
    ret1: Vec<f64>,
    tr: Vec<f64>,
    gap: Vec<bool>,
}

impl Base {
    fn new(s: &Series) -> Self {
        let ret1 = ret_k(s, 1);
        let tr = true_range(s);
        Base {
            ret1,
            tr,
            gap: gap_mask(s),
        }
    }
}

fn ma_ratio(s: &Series, w: usize) -> Vec<f64> {
    let p = Prefix::new(&s.close, &s.times, s.step);
    let mut out = vec![f64::NAN; s.len()];
    for i in 0..s.len() {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        if let Some(m) = p.mean(i, w) {
            if m > 0.0 {
                out[i] = s.close[i] / m - 1.0;
            }
        }
    }
    out
}

fn vol_w(s: &Series, b: &Base, w: usize) -> Vec<f64> {
    let p = Prefix::new(&b.ret1, &s.times, s.step);
    let mut out = vec![f64::NAN; s.len()];
    for i in w..s.len() {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        if let Some(v) = p.std(i, w) {
            out[i] = v;
        }
    }
    out
}

fn bb_width(s: &Series, w: usize) -> Vec<f64> {
    let p = Prefix::new(&s.close, &s.times, s.step);
    let mut out = vec![f64::NAN; s.len()];
    for i in 0..s.len() {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        if let (Some(m), Some(sd)) = (p.mean(i, w), p.std(i, w)) {
            if m > 0.0 {
                out[i] = 4.0 * sd / m;
            }
        }
    }
    out
}

fn vz(s: &Series, w: usize) -> Vec<f64> {
    let x: Vec<f64> = s
        .volume
        .iter()
        .map(|v| if *v >= 0.0 { (1.0 + v).ln() } else { f64::NAN })
        .collect();
    let p = Prefix::new(&x, &s.times, s.step);
    let mut out = vec![f64::NAN; s.len()];
    for i in 0..s.len() {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        if let (Some(m), Some(sd)) = (p.mean(i, w), p.std(i, w)) {
            if sd > 0.0 {
                out[i] = (x[i] - m) / sd;
            }
        }
    }
    out
}

fn range1(s: &Series) -> Vec<f64> {
    (0..s.len())
        .map(|i| {
            if s.close[i] > 0.0 {
                (s.high[i] - s.low[i]) / s.close[i]
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn clv1(s: &Series) -> Vec<f64> {
    // Compact definition: (C-L)/(H-L) - 1/2, with the compact builder's
    // degenerate H==L convention (-1/2).
    (0..s.len())
        .map(|i| {
            if !(s.high[i].is_finite() && s.low[i].is_finite() && s.close[i].is_finite()) {
                return f64::NAN;
            }
            let r = s.high[i] - s.low[i];
            if r > 0.0 {
                (s.close[i] - s.low[i]) / r - 0.5
            } else {
                -0.5
            }
        })
        .collect()
}

fn donchian(s: &Series, w: usize) -> Vec<f64> {
    // Same strict window policy as the device extrema: a non-finite high or
    // low anywhere in the window makes the channel missing. This is bit-exact
    // with the blocked prefix-suffix kernel.
    let hi = rolling_max(&s.high, w, &s.times, s.step);
    let lo = rolling_min(&s.low, w, &s.times, s.step);
    (0..s.len())
        .map(|i| {
            if hi[i] > lo[i] && lo[i].is_finite() {
                (s.close[i] - lo[i]) / (hi[i] - lo[i])
            } else {
                f64::NAN
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-timeframe family
// ---------------------------------------------------------------------------

fn series_from_bars(bars: &[Bar], step: i64) -> Series {
    Series {
        symbol: String::new(),
        times: bars.iter().map(|b| b.time).collect(),
        open: bars.iter().map(|b| b.open).collect(),
        high: bars.iter().map(|b| b.high).collect(),
        low: bars.iter().map(|b| b.low).collect(),
        close: bars.iter().map(|b| b.close).collect(),
        volume: bars.iter().map(|b| b.volume).collect(),
        quote_volume: bars.iter().map(|b| b.quote_volume).collect(),
        trades: bars.iter().map(|b| b.trades).collect(),
        taker_buy_base: bars.iter().map(|b| b.taker_buy_base).collect(),
        step,
    }
}

/// Simple close-to-close return over `k` contiguous higher bars.
fn mtf_ret(h: &Series, k: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; h.len()];
    for i in k..h.len() {
        if h.times[i] - h.times[i - k] == k as i64 * h.step
            && h.close[i].is_finite()
            && h.close[i - k].is_finite()
            && h.close[i - k] > 0.0
        {
            out[i] = h.close[i] / h.close[i - k] - 1.0;
        }
    }
    out
}

/// Population standard deviation of one-bar simple returns over 20 bars.
fn mtf_vol(h: &Series, w: usize) -> Vec<f64> {
    let returns = mtf_ret(h, 1);
    rstd_py(&returns, w, &h.times, h.step)
}

/// Wilder ATR seeded on `period` contiguous true ranges. A gap resets the
/// seed, so ATR never carries state across an outage.
fn mtf_atr(h: &Series, period: usize) -> Vec<f64> {
    let tr = true_range(h);
    let mut out = vec![f64::NAN; h.len()];
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut prev = f64::NAN;
    for i in 0..h.len() {
        if i > 0 && h.times[i] - h.times[i - 1] != h.step {
            count = 0;
            sum = 0.0;
            prev = f64::NAN;
        }
        if !tr[i].is_finite() {
            count = 0;
            sum = 0.0;
            prev = f64::NAN;
            continue;
        }
        if count < period {
            sum += tr[i];
            count += 1;
            if count == period {
                prev = sum / period as f64;
                out[i] = prev;
            }
        } else {
            prev += (tr[i] - prev) / period as f64;
            out[i] = prev;
        }
    }
    out
}

fn rsi_norm(gain: f64, loss: f64) -> f64 {
    let rsi = if loss > 0.0 {
        100.0 - 100.0 / (1.0 + gain / loss)
    } else if gain > 0.0 {
        100.0
    } else {
        50.0
    };
    rsi / 50.0 - 1.0
}

/// Wilder RSI(14), normalised as `RSI/50 - 1`. Seeded on 14 contiguous
/// up/down deltas and reset after a gap.
fn mtf_rsi(h: &Series, period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; h.len()];
    let mut gain = 0.0f64;
    let mut loss = 0.0f64;
    let mut count = 0usize;
    let mut avg_gain = f64::NAN;
    let mut avg_loss = f64::NAN;
    for i in 1..h.len() {
        if h.times[i] - h.times[i - 1] != h.step
            || !h.close[i].is_finite()
            || !h.close[i - 1].is_finite()
        {
            gain = 0.0;
            loss = 0.0;
            count = 0;
            avg_gain = f64::NAN;
            avg_loss = f64::NAN;
            continue;
        }
        let d = h.close[i] - h.close[i - 1];
        let up = d.max(0.0);
        let dn = (-d).max(0.0);
        if !avg_gain.is_finite() {
            gain += up;
            loss += dn;
            count += 1;
            if count == period {
                avg_gain = gain / period as f64;
                avg_loss = loss / period as f64;
                out[i] = rsi_norm(avg_gain, avg_loss);
            }
        } else {
            avg_gain = (avg_gain * (period as f64 - 1.0) + up) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + dn) / period as f64;
            out[i] = rsi_norm(avg_gain, avg_loss);
        }
    }
    out
}

fn mtf_close_pos(h: &Series) -> Vec<f64> {
    (0..h.len())
        .map(|i| {
            let r = h.high[i] - h.low[i];
            if r > 0.0 && h.close[i].is_finite() && h.high[i].is_finite() && h.low[i].is_finite() {
                (h.close[i] - h.low[i]) / r - 0.5
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn mtf_donchian_pos(h: &Series, w: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; h.len()];
    for i in 0..h.len() {
        if !contig(&h.times, i, w, h.step) {
            continue;
        }
        let lo = i + 1 - w;
        let mut hi = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        let mut finite = true;
        for j in lo..=i {
            if !h.high[j].is_finite() || !h.low[j].is_finite() {
                finite = false;
                break;
            }
            hi = hi.max(h.high[j]);
            low = low.min(h.low[j]);
        }
        if finite && hi > low && h.close[i].is_finite() {
            out[i] = (h.close[i] - low) / (hi - low) - 0.5;
        }
    }
    out
}

fn mtf_range_pct(h: &Series) -> Vec<f64> {
    (0..h.len())
        .map(|i| {
            if h.close[i].is_finite()
                && h.close[i] > 0.0
                && h.high[i].is_finite()
                && h.low[i].is_finite()
            {
                (h.high[i] - h.low[i]) / h.close[i]
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn mtf_trend_slope(h: &Series) -> Vec<f64> {
    let p = Prefix::new(&h.close, &h.times, h.step);
    let mut out = vec![f64::NAN; h.len()];
    for i in 39..h.len() {
        if !contig(&h.times, i, 40, h.step) {
            continue;
        }
        if let (Some(now), Some(prev)) = (p.mean(i, 20), p.mean(i - 20, 20)) {
            if prev > 0.0 {
                out[i] = now / prev - 1.0;
            }
        }
    }
    out
}

fn mtf_leaf_values(h: &Series, leaf: &str) -> Vec<f64> {
    match leaf {
        "ret1" => mtf_ret(h, 1),
        "ret4" => mtf_ret(h, 4),
        "ret24" => mtf_ret(h, 24),
        "ma_ratio" => ma_ratio(h, 20),
        "vol" => mtf_vol(h, 20),
        "atr_pct" => {
            let atr = mtf_atr(h, 14);
            (0..h.len())
                .map(|i| {
                    if atr[i].is_finite() && h.close[i].is_finite() && h.close[i] > 0.0 {
                        atr[i] / h.close[i]
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        }
        "rsi" => mtf_rsi(h, 14),
        "close_pos" => mtf_close_pos(h),
        "donchian_pos" => mtf_donchian_pos(h, 20),
        "range_pct" => mtf_range_pct(h),
        "trend_slope" => mtf_trend_slope(h),
        _ => vec![f64::NAN; h.len()],
    }
}

/// Compute every requested MTF column for one symbol.
///
/// The base series is resampled to each requested higher timeframe, the leaf is
/// evaluated on the higher bars, and the scalar column is joined at the last
/// higher bucket whose **close** is at or before the base decision time:
///
/// ```text
/// visible <=> higher.time + higher_tf.seconds() <= base.time
/// ```
///
/// A partial or current higher bucket is therefore never used. If the expected
/// last-closed higher bucket is missing (a gap or outage), every value leaf is
/// `NaN`; `bar_age` stays finite and reports the exact staleness in seconds.
fn mtf_all(s: &Series, requested: &[String]) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let base = s.bars();
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for tf_name in MTF_TIMEFRAMES {
        let leaves: Vec<(&str, &str)> = requested
            .iter()
            .filter_map(|name| {
                let (tf, leaf) = mtf_parts(name)?;
                if tf == *tf_name {
                    Some((name.as_str(), leaf))
                } else {
                    None
                }
            })
            .collect();
        if leaves.is_empty() {
            continue;
        }
        let tf = Timeframe::parse(tf_name)?;
        let higher = resample(&base, tf)?;
        let h = series_from_bars(&higher, tf.seconds());
        let joined = join_causal(&base, &higher, tf)?;
        let index: BTreeMap<i64, usize> = higher
            .iter()
            .enumerate()
            .map(|(i, b)| (b.time, i))
            .collect();
        let mut cache: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        for (name, leaf) in leaves {
            let values = if leaf == "bar_age" {
                joined
                    .iter()
                    .enumerate()
                    .map(|(i, slot)| {
                        slot.map(|b| (base[i].time - (b.time + tf.seconds())) as f64)
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            } else {
                let raw = cache
                    .entry(leaf)
                    .or_insert_with(|| mtf_leaf_values(&h, leaf));
                let mut vals = Vec::with_capacity(base.len());
                for (i, slot) in joined.iter().enumerate() {
                    match slot {
                        Some(b) => {
                            // Reject a stale snapshot: the last closed bucket
                            // must be the one immediately preceding this base
                            // decision bucket. Otherwise the higher series has
                            // a gap and the value would bridge it.
                            let expected = tf.bucket_start(base[i].time) - tf.seconds();
                            if b.time == expected {
                                vals.push(index.get(&b.time).map(|j| raw[*j]).unwrap_or(f64::NAN));
                            } else {
                                vals.push(f64::NAN);
                            }
                        }
                        None => vals.push(f64::NAN),
                    }
                }
                vals
            };
            out.insert(name.to_string(), values);
        }
    }
    Ok(out)
}

fn lag(s: &Series, b: &Base, k: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; s.len()];
    for i in k..s.len() {
        if s.times[i] - s.times[i - k] == k as i64 * s.step {
            out[i] = b.ret1[i - k];
        }
    }
    out
}

fn calendar_column(s: &Series, which: &str) -> Vec<f64> {
    (0..s.len())
        .map(|i| {
            let t = s.times[i];
            match which {
                "hour_sin" | "hour_cos" => {
                    let (sn, cs) = cyclic(calendar_frac(t));
                    if which == "hour_sin" { sn } else { cs }
                }
                "dow_sin" | "dow_cos" => {
                    let dow = weekday(t.div_euclid(86_400)) as f64;
                    let a = 2.0 * std::f64::consts::PI * dow / 7.0;
                    if which == "dow_sin" { a.sin() } else { a.cos() }
                }
                "month_sin" | "month_cos" => {
                    let (_, m, _) = civil_from_days(t.div_euclid(86_400));
                    let a = 2.0 * std::f64::consts::PI * m as f64 / 12.0;
                    if which == "month_sin" { a.sin() } else { a.cos() }
                }
                "dom_sin" | "dom_cos" => {
                    let (_, _, d) = civil_from_days(t.div_euclid(86_400));
                    let a = 2.0 * std::f64::consts::PI * d as f64 / 31.0;
                    if which == "dom_sin" { a.sin() } else { a.cos() }
                }
                _ => f64::NAN,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cross-market context
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Cross {
    symbol: String,
    r1: BTreeMap<i64, f64>,
    r16: BTreeMap<i64, f64>,
    r20: BTreeMap<i64, f64>,
    r96: BTreeMap<i64, f64>,
    r672: BTreeMap<i64, f64>,
    vol20: BTreeMap<i64, f64>,
    vol96: BTreeMap<i64, f64>,
    volume: BTreeMap<i64, f64>,
}

impl Cross {
    fn from_series(s: &Series) -> Self {
        let b = Base::new(s);
        let r1 = ret_k(s, 1);
        let r16 = ret_k(s, 16);
        let r20 = ret_k(s, 20);
        let r96 = ret_k(s, 96);
        let r672 = ret_k(s, 672);
        let vol20 = vol_w(s, &b, 20);
        let vol96 = vol_w(s, &b, 96);
        let map = |v: &[f64]| {
            let mut m = BTreeMap::new();
            for (i, x) in v.iter().enumerate() {
                m.insert(s.times[i], *x);
            }
            m
        };
        Cross {
            symbol: s.symbol.clone(),
            r1: map(&r1),
            r16: map(&r16),
            r20: map(&r20),
            r96: map(&r96),
            r672: map(&r672),
            vol20: map(&vol20),
            vol96: map(&vol96),
            volume: map(&s.volume),
        }
    }
    fn at(map: &BTreeMap<i64, f64>, t: i64) -> f64 {
        map.get(&t).copied().unwrap_or(f64::NAN)
    }
    /// The benchmark as sorted unique `(time, planes)` arrays for the device
    /// join (#299). Empty when the benchmark symbol was not loaded.
    fn bench_series(&self) -> crate::bench_gpu::BenchSeries {
        use crate::bench_gpu::*;
        let plane = |m: &BTreeMap<i64, f64>| m.values().copied().collect::<Vec<f64>>();
        let mut b = BenchSeries::default();
        b.times = self.r1.keys().copied().collect();
        b.planes[PLANE_R1] = plane(&self.r1);
        b.planes[PLANE_R16] = plane(&self.r16);
        b.planes[PLANE_R20] = plane(&self.r20);
        b.planes[PLANE_R96] = plane(&self.r96);
        b.planes[PLANE_R672] = plane(&self.r672);
        b.planes[PLANE_VOL20] = plane(&self.vol20);
        b.planes[PLANE_VOL96] = plane(&self.vol96);
        b.planes[PLANE_VOLUME] = plane(&self.volume);
        b
    }
}

fn beta_window(
    s: &Series,
    x: &[f64],
    y: &BTreeMap<i64, f64>,
    w: usize,
) -> Vec<f64> {
    let n = s.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        let lo = i + 1 - w;
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        let mut ok = true;
        for j in lo..=i {
            let yv = Cross::at(y, s.times[j]);
            if !x[j].is_finite() || !yv.is_finite() {
                ok = false;
                break;
            }
            sx += x[j];
            sy += yv;
            syy += yv * yv;
            sxy += x[j] * yv;
        }
        if !ok {
            continue;
        }
        let nf = w as f64;
        let cov = sxy / nf - (sx / nf) * (sy / nf);
        let var = syy / nf - (sy / nf) * (sy / nf);
        if var > 0.0 {
            out[i] = (cov / var).max(-5.0).min(5.0);
        }
    }
    out
}

/// Load one benchmark symbol from the base store or an optional external store.
///
/// The benchmark is keyed by exact timestamps; there is no forward fill. A
/// missing row is simply absent from the map and therefore `NaN` at the join.
fn load_benchmark(
    store: &FileStore,
    segs: &[Segment],
    cols: &Cols,
    step: i64,
    symbol: &str,
    external: Option<&str>,
) -> Result<Cross, String> {
    if let Some(seg) = segs.iter().find(|s| s.symbol == symbol) {
        return Ok(Cross::from_series(&load_series(store, seg, cols, step)?));
    }
    let Some(path) = external else {
        return Ok(Cross::default());
    };
    let cs = FileStore::open(path)?;
    let cnames = cs.header().names();
    let time = find_time_col(&cnames, None)
        .ok_or_else(|| format!("{}: benchmark store has no time column", path))?;
    let ccols = Cols {
        time,
        time_dtype: cs.header().dtypes()[time],
        open: find_col(&cnames, &["open", "o"])
            .ok_or_else(|| format!("{}: benchmark store has no open", path))?,
        high: find_col(&cnames, &["high", "h"])
            .ok_or_else(|| format!("{}: benchmark store has no high", path))?,
        low: find_col(&cnames, &["low", "l"])
            .ok_or_else(|| format!("{}: benchmark store has no low", path))?,
        close: find_col(&cnames, &["close", "c"])
            .ok_or_else(|| format!("{}: benchmark store has no close", path))?,
        volume: find_col(&cnames, &["volume", "vol", "v"]),
        quote_volume: find_col(&cnames, &["quote_volume", "quotevolume", "qv"]),
        trades: find_col(&cnames, &["trades", "trade_count", "trades_count", "n"]),
        taker_buy_base: find_col(
            &cnames,
            &["taker_buy_base", "taker_buy", "takerbuybase", "taker_buy_volume", "tb"],
        ),
        taker_buy_quote: find_col(
            &cnames,
            &["taker_buy_quote", "takerbuyquote", "tbq"],
        ),
        symbol: find_symbol_col(&cnames, None),
    };
    let (csegs, _) = segments(&cs, ccols.time, ccols.symbol)?;
    let cseg = csegs
        .iter()
        .find(|s| s.symbol == symbol)
        .or_else(|| csegs.first())
        .ok_or_else(|| format!("{}: benchmark store has no rows", path))?;
    Ok(Cross::from_series(&load_series(&cs, cseg, &ccols, step)?))
}

// ---------------------------------------------------------------------------
// py15m parity family
// ---------------------------------------------------------------------------

fn rolling_min(x: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !contig(times, i, w, step) {
            continue;
        }
        let lo = i + 1 - w;
        let mut m = f64::INFINITY;
        let mut ok = true;
        for &v in &x[lo..=i] {
            if !v.is_finite() {
                ok = false;
                break;
            }
            m = m.min(v);
        }
        if ok {
            out[i] = m;
        }
    }
    out
}

fn rolling_max(x: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if !contig(times, i, w, step) {
            continue;
        }
        let lo = i + 1 - w;
        let mut m = f64::NEG_INFINITY;
        let mut ok = true;
        for &v in &x[lo..=i] {
            if !v.is_finite() {
                ok = false;
                break;
            }
            m = m.max(v);
        }
        if ok {
            out[i] = m;
        }
    }
    out
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    if !v.is_finite() {
        f64::NAN
    } else {
        v.max(lo).min(hi)
    }
}

/// All 35 columns of the Python `enrich_panel_15m.py` step.
///
/// The formulas are the Python ones (including the `rmean`/`rstd` finite-count
/// convention and the `wilder()` recurrence), with two deliberate fixes: a
/// window that spans a gap is `NaN` instead of bridged, and the first bar's
/// true range uses `high - low` instead of NumPy's `np.roll` wrap-around that
/// reads the *last* close. On a contiguous fixture the two agree to rounding.
fn py15m_all(s: &Series, b: &Base, cross: &Cross) -> BTreeMap<String, Vec<f64>> {
    let n = s.len();
    let c = &s.close;
    let o = &s.open;
    let h = &s.high;
    let l = &s.low;
    let v = &s.volume;
    let gap = &b.gap;
    let times = &s.times;
    let step = s.step;
    let mut out = BTreeMap::new();

    let e8 = ema(c, 8, gap);
    let e20 = ema(c, 20, gap);
    let e50 = ema(c, 50, gap);
    let e100 = ema(c, 100, gap);
    let e200 = ema(c, 200, gap);
    let ratio = |e: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| if e[i].is_finite() && c[i] > 0.0 { c[i] / e[i] - 1.0 } else { f64::NAN })
            .collect()
    };
    out.insert("ema8".to_string(), ratio(&e8));
    out.insert("ema20".to_string(), ratio(&e20));
    out.insert("ema50".to_string(), ratio(&e50));
    out.insert("ema100".to_string(), ratio(&e100));
    out.insert("ema200".to_string(), ratio(&e200));

    let e12 = ema(c, 12, gap);
    let e26 = ema(c, 26, gap);
    let macd: Vec<f64> = (0..n)
        .map(|i| {
            if e12[i].is_finite() && e26[i].is_finite() && e26[i] > 0.0 {
                e12[i] / e26[i] - 1.0
            } else {
                f64::NAN
            }
        })
        .collect();
    let macd_sig = ema(&macd, 9, gap);
    let macd_hist: Vec<f64> = (0..n)
        .map(|i| if macd[i].is_finite() && macd_sig[i].is_finite() { macd[i] - macd_sig[i] } else { f64::NAN })
        .collect();
    out.insert("macd".to_string(), macd);
    out.insert("macd_sig".to_string(), macd_sig);
    out.insert("macd_hist".to_string(), macd_hist);

    out.insert("rsi6".to_string(), rsi_py(c, 6, gap));
    out.insert("rsi28".to_string(), rsi_py(c, 28, gap));

    let mid50 = rmean_py(c, 50, times, step);
    let sd50 = rstd_py(c, 50, times, step);
    let bb50: Vec<f64> = (0..n)
        .map(|i| {
            if mid50[i].is_finite() && sd50[i].is_finite() && sd50[i] > 0.0 {
                clip((c[i] - mid50[i]) / (2.0 * sd50[i]), -1.0, 1.0)
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert("bb50".to_string(), bb50);
    let m20 = rmean_py(c, 20, times, step);
    let s20 = rstd_py(c, 20, times, step);
    let bw20: Vec<f64> = (0..n)
        .map(|i| if m20[i] > 0.0 && s20[i].is_finite() { 4.0 * s20[i] / m20[i] } else { f64::NAN })
        .collect();
    let bw50: Vec<f64> = (0..n)
        .map(|i| if mid50[i] > 0.0 && sd50[i].is_finite() { 4.0 * sd50[i] / mid50[i] } else { f64::NAN })
        .collect();
    out.insert("bb20_bw".to_string(), bw20);
    out.insert("bb50_bw".to_string(), bw50);

    let tr = &b.tr;
    let atr50 = wilder_py(tr, 50, gap);
    let atr50: Vec<f64> = (0..n)
        .map(|i| if atr50[i].is_finite() && c[i] > 0.0 { atr50[i] / c[i] } else { f64::NAN })
        .collect();
    out.insert("atr50".to_string(), atr50);

    let ll = rolling_min(l, 14, times, step);
    let hh = rolling_max(h, 14, times, step);
    let k: Vec<f64> = (0..n)
        .map(|i| {
            if ll[i].is_finite() && hh[i].is_finite() && hh[i] > ll[i] {
                clip((c[i] - ll[i]) / (hh[i] - ll[i]) - 0.5, -0.5, 0.5)
            } else {
                f64::NAN
            }
        })
        .collect();
    let d = rmean_py(&k, 3, times, step);
    out.insert("stoch_k".to_string(), k);
    out.insert("stoch_d".to_string(), d);

    // ADX/DI exactly as the Python script: PDM/MDM (with a causal first bar
    // of zero) smoothed by the Python Wilder recurrence, divided by ATR.
    let mut pdm = vec![0.0f64; n];
    let mut mdm = vec![0.0f64; n];
    for i in 1..n {
        if times[i] - times[i - 1] != step {
            // Directional movement across a hole is missing, not zero.
            pdm[i] = f64::NAN;
            mdm[i] = f64::NAN;
            continue;
        }
        let up = h[i] - h[i - 1];
        let dn = l[i - 1] - l[i];
        if up > dn && up > 0.0 {
            pdm[i] = up;
        }
        if dn > up && dn > 0.0 {
            mdm[i] = dn;
        }
    }
    let atr14 = wilder_py(tr, 14, gap);
    let pdi_r = wilder_py(&pdm, 14, gap);
    let mdi_r = wilder_py(&mdm, 14, gap);
    let mut pdi = vec![f64::NAN; n];
    let mut mdi = vec![f64::NAN; n];
    for i in 0..n {
        if atr14[i] > 0.0 {
            pdi[i] = 100.0 * pdi_r[i] / atr14[i];
            mdi[i] = 100.0 * mdi_r[i] / atr14[i];
        }
    }
    let dx: Vec<f64> = (0..n)
        .map(|i| {
            if pdi[i].is_finite() && mdi[i].is_finite() && pdi[i] + mdi[i] > 0.0 {
                100.0 * (pdi[i] - mdi[i]).abs() / (pdi[i] + mdi[i])
            } else {
                f64::NAN
            }
        })
        .collect();
    let adx = wilder_py(&dx, 14, gap);
    let adx: Vec<f64> = adx.iter().map(|v| v / 100.0).collect();
    let di_plus: Vec<f64> = pdi.iter().map(|v| v / 100.0).collect();
    let di_minus: Vec<f64> = mdi.iter().map(|v| v / 100.0).collect();
    out.insert("adx14".to_string(), adx);
    out.insert("di_plus".to_string(), di_plus);
    out.insert("di_minus".to_string(), di_minus);

    let mut obv = vec![0.0f64; n];
    for i in 1..n {
        let sign = if c[i] > c[i - 1] { 1.0 } else if c[i] < c[i - 1] { -1.0 } else { 0.0 };
        obv[i] = obv[i - 1] + sign * v[i];
    }
    let obv_m = rmean_py(&obv, 96, times, step);
    let obv_s = rstd_py(&obv, 96, times, step);
    let obv_z: Vec<f64> = (0..n)
        .map(|i| {
            if obv_s[i] > 0.0 && obv_m[i].is_finite() {
                (obv[i] - obv_m[i]) / obv_s[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert("obv_z96".to_string(), obv_z);

    let tp: Vec<f64> = (0..n).map(|i| (h[i] + l[i] + c[i]) / 3.0).collect();
    let pv: Vec<f64> = (0..n).map(|i| tp[i] * v[i]).collect();
    let pvm = rmean_py(&pv, 96, times, step);
    let vm = rmean_py(v, 96, times, step);
    let vwap: Vec<f64> = (0..n)
        .map(|i| if vm[i] > 0.0 && pvm[i].is_finite() { pvm[i] / vm[i] } else { f64::NAN })
        .collect();
    let vd: Vec<f64> = (0..n)
        .map(|i| if vwap[i] > 0.0 { (c[i] - vwap[i]) / vwap[i] } else { f64::NAN })
        .collect();
    out.insert("vwap_dev96".to_string(), vd);

    let hl: Vec<f64> = (0..n)
        .map(|i| {
            if l[i] > 0.0 && h[i] > 0.0 {
                (h[i] / l[i].max(1e-12)).ln()
            } else {
                f64::NAN
            }
        })
        .collect();
    let hl2: Vec<f64> = hl.iter().map(|v| v * v).collect();
    let hl2m = rmean_py(&hl2, 16, times, step);
    let park: Vec<f64> = hl2m
        .iter()
        .map(|m| if m.is_finite() { (m / (4.0 * std::f64::consts::LN_2)).sqrt() } else { f64::NAN })
        .collect();
    out.insert("park16".to_string(), park);

    let co: Vec<f64> = (0..n)
        .map(|i| {
            if o[i] > 0.0 && c[i] > 0.0 {
                (c[i] / o[i].max(1e-12)).ln()
            } else {
                f64::NAN
            }
        })
        .collect();
    let co2: Vec<f64> = co.iter().map(|v| v * v).collect();
    let co2m = rmean_py(&co2, 16, times, step);
    let gk: Vec<f64> = (0..n)
        .map(|i| {
            if hl2m[i].is_finite() && co2m[i].is_finite() {
                let v = 0.5 * hl2m[i] - (2.0 * std::f64::consts::LN_2 - 1.0) * co2m[i];
                v.max(0.0).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert("gk16".to_string(), gk);

    // 1-bar log returns with a zero at the first bar, as the Python script's
    // `lr` (the base panel's `r1` is NaN in warm-up; the Python series is 0).
    let mut lr1 = ret_k(s, 1);
    if !lr1.is_empty() {
        lr1[0] = 0.0;
    }
    let rmean96 = rmean_py(&lr1, 96, times, step);
    let rstd96 = rstd_py(&lr1, 96, times, step);
    // The Python script centers the *whole* return series by its per-row
    // rolling mean and then takes a rolling mean of the centred powers. That is
    // its convention (not a textbook central moment); reproduce it exactly so
    // the panel is byte-comparable.
    let c3: Vec<f64> = (0..n)
        .map(|j| {
            if lr1[j].is_finite() && rmean96[j].is_finite() {
                let d = lr1[j] - rmean96[j];
                d * d * d
            } else {
                f64::NAN
            }
        })
        .collect();
    let c4: Vec<f64> = (0..n)
        .map(|j| {
            if lr1[j].is_finite() && rmean96[j].is_finite() {
                let d = lr1[j] - rmean96[j];
                d * d * d * d
            } else {
                f64::NAN
            }
        })
        .collect();
    let s3 = rmean_py(&c3, 96, times, step);
    let s4 = rmean_py(&c4, 96, times, step);
    let mut sk = vec![f64::NAN; n];
    let mut ku = vec![f64::NAN; n];
    for i in 0..n {
        if !contig(times, i, 96, step) {
            continue;
        }
        let s2 = rstd96[i] * rstd96[i];
        if !(s2 > 0.0) || !s3[i].is_finite() || !s4[i].is_finite() {
            continue;
        }
        sk[i] = s3[i] / s2.max(1e-30).powf(1.5);
        ku[i] = s4[i] / s2.max(1e-30).powi(2) - 3.0;
    }
    out.insert("skew96".to_string(), sk);
    out.insert("kurt96".to_string(), ku);

    let mut r192 = vec![f64::NAN; n];
    for i in 192..n {
        if times[i] - times[i - 192] == 192 * step && c[i] > 0.0 && c[i - 192] > 0.0 {
            r192[i] = (c[i] / c[i - 192]).ln();
        }
    }
    out.insert("r192".to_string(), r192);

    out.insert("month_sin".to_string(), calendar_column(s, "month_sin"));
    out.insert("month_cos".to_string(), calendar_column(s, "month_cos"));
    out.insert("dom_sin".to_string(), calendar_column(s, "dom_sin"));
    out.insert("dom_cos".to_string(), calendar_column(s, "dom_cos"));

    let r16 = ret_k(s, 16);
    let r96 = ret_k(s, 96);
    let mkt = |map: &BTreeMap<i64, f64>| -> Vec<f64> {
        (0..n).map(|i| Cross::at(map, times[i])).collect()
    };
    let mr16 = mkt(&cross.r16);
    let mr96 = mkt(&cross.r96);
    let mv96 = mkt(&cross.vol96);
    let rel16: Vec<f64> = (0..n)
        .map(|i| if r16[i].is_finite() && mr16[i].is_finite() { r16[i] - mr16[i] } else { f64::NAN })
        .collect();
    let rel96: Vec<f64> = (0..n)
        .map(|i| if r96[i].is_finite() && mr96[i].is_finite() { r96[i] - mr96[i] } else { f64::NAN })
        .collect();
    out.insert("mkt_r16".to_string(), mr16);
    out.insert("mkt_r96".to_string(), mr96);
    out.insert("mkt_vol96".to_string(), mv96);
    out.insert("rel_r16".to_string(), rel16);
    out.insert("rel_r96".to_string(), rel96);

    // py15m shares `obv_z96` and `vwap_dev96` with the flow family; use the one
    // canonical implementation. With a contiguous fixture and `quote_volume =
    // typical_price * volume`, this is numerically identical to the legacy
    // Python formula.
    let flow = flow_all(s);
    for name in ["obv_z96", "vwap_dev96"] {
        if let Some(vals) = flow.get(name) {
            out.insert(name.to_string(), vals.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Compact 52-feature profile (15 m base, exact compact-builder formulas)
// ---------------------------------------------------------------------------

/// A lagged copy of `x` that is valid only across one contiguous bar.
fn rlag(x: &[f64], lag: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for i in lag..n {
        if times[i] - times[i - lag] == lag as i64 * step {
            out[i] = x[i - lag];
        }
    }
    out
}

fn div(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && b != 0.0 {
        a / b
    } else {
        f64::NAN
    }
}

fn div_pos(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && b > 0.0 {
        a / b
    } else {
        f64::NAN
    }
}

/// Compute the complete OHLCV/flow family for one symbol (#270).
///
/// All windows are trailing and inclusive of `t`; a window whose timestamps are
/// not one contiguous run at `s.step` is `NaN`. `OBV` and `CVD` cumulative
/// state restarts after a gap, so a listing/delisting/migration boundary is
/// never bridged. A zero-volume bar makes `vwap`, the taker ratios, trade
/// intensity, Amihud and `vwap_dev96` `NaN`; zero trades makes trade intensity
/// `NaN` and `avg_trade_size96` uses the specified `max(trades, 1)`.
fn flow_all(s: &Series) -> BTreeMap<String, Vec<f64>> {
    let n = s.len();
    let c = &s.close;
    let v = &s.volume;
    let qv = &s.quote_volume;
    let trades = &s.trades;
    let tb = &s.taker_buy_base;
    let times = &s.times;
    let step = s.step;
    let mut out = BTreeMap::new();

    // vwap is the raw quote-volume / base-volume ratio in price units.
    out.insert(
        "vwap".to_string(),
        (0..n).map(|i| div_pos(qv[i], v[i])).collect(),
    );

    let v96 = rmean_py(v, 96, times, step);
    let qv96 = rmean_py(qv, 96, times, step);
    let tb96 = rmean_py(tb, 96, times, step);
    let trades96 = rmean_py(trades, 96, times, step);
    let v16 = rmean_py(v, 16, times, step);
    let tb16 = rmean_py(tb, 16, times, step);

    out.insert(
        "taker_ratio96".to_string(),
        (0..n).map(|i| div_pos(tb96[i], v96[i])).collect(),
    );
    out.insert(
        "taker_imb16".to_string(),
        (0..n)
            .map(|i| div_pos(2.0 * tb16[i] - v16[i], v16[i]))
            .collect(),
    );
    out.insert(
        "trade_intensity96".to_string(),
        (0..n)
            .map(|i| {
                if trades96[i].is_finite() && v96[i] > 0.0 && trades96[i] > 0.0 {
                    (1.0 + trades96[i] / v96[i]).ln()
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    // Per-bar quote volume per trade, then the trailing log mean. `max(trades,1)`
    // is explicit in the issue definition, so a zero-trade bar is not a gap.
    let per_trade: Vec<f64> = (0..n)
        .map(|i| {
            if qv[i].is_finite() && qv[i] >= 0.0 && trades[i].is_finite() {
                qv[i] / trades[i].max(1.0)
            } else {
                f64::NAN
            }
        })
        .collect();
    let per_trade96 = rmean_py(&per_trade, 96, times, step);
    out.insert(
        "avg_trade_size96".to_string(),
        (0..n)
            .map(|i| {
                if per_trade96[i].is_finite() && per_trade96[i] >= 0.0 {
                    (1.0 + per_trade96[i]).ln()
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    let lr = ret_k(s, 1);
    let amihud_bar: Vec<f64> = (0..n)
        .map(|i| {
            if lr[i].is_finite() && qv[i].is_finite() && qv[i] > 0.0 {
                lr[i].abs() / qv[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert(
        "amihud96".to_string(),
        rmean_py(&amihud_bar, 96, times, step)
            .into_iter()
            .map(|x| x * 1e6)
            .collect(),
    );

    let zero_bar: Vec<f64> = (0..n)
        .map(|i| {
            if lr[i].is_finite() {
                if lr[i] == 0.0 { 1.0 } else { 0.0 }
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert(
        "zero_ret_share96".to_string(),
        rmean_py(&zero_bar, 96, times, step),
    );

    // OBV restarts at a gap; within a contiguous run it follows the issue's
    // recurrence with sign(0) = 0.
    let mut obv = vec![f64::NAN; n];
    for i in 0..n {
        if i == 0 {
            obv[i] = if v[i].is_finite() { 0.0 } else { f64::NAN };
            continue;
        }
        if times[i] - times[i - 1] != step {
            obv[i] = if v[i].is_finite() { 0.0 } else { f64::NAN };
            continue;
        }
        if !obv[i - 1].is_finite() || !c[i].is_finite() || !c[i - 1].is_finite() || !v[i].is_finite() {
            obv[i] = f64::NAN;
            continue;
        }
        let sign = if c[i] > c[i - 1] {
            1.0
        } else if c[i] < c[i - 1] {
            -1.0
        } else {
            0.0
        };
        obv[i] = obv[i - 1] + sign * v[i];
    }
    let obv_m = rmean_py(&obv, 96, times, step);
    let obv_s = rstd_py(&obv, 96, times, step);
    out.insert(
        "obv_z96".to_string(),
        (0..n)
            .map(|i| {
                if obv_s[i] > 0.0 && obv[i].is_finite() && obv_m[i].is_finite() {
                    (obv[i] - obv_m[i]) / obv_s[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    // CVD restarts at a gap; the proxy is the cumulative taker imbalance
    // normalised by its own trailing 96-bar population standard deviation.
    let mut cvd = vec![f64::NAN; n];
    for i in 0..n {
        if i > 0 && times[i] - times[i - 1] != step {
            cvd[i] = f64::NAN;
            continue;
        }
        if !v[i].is_finite() || !tb[i].is_finite() {
            cvd[i] = f64::NAN;
            continue;
        }
        let net = 2.0 * tb[i] - v[i];
        let prev = if i == 0 { 0.0 } else { cvd[i - 1] };
        cvd[i] = if prev.is_finite() { prev + net } else { net };
    }
    let cvd_s = rstd_py(&cvd, 96, times, step);
    out.insert(
        "cvd_proxy".to_string(),
        (0..n)
            .map(|i| {
                if cvd[i].is_finite() && cvd_s[i] > 0.0 {
                    cvd[i] / cvd_s[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    // VWAP deviation uses the issue's quote-volume VWAP, not the typical-price
    // proxy used by the legacy compact/py15m builders.
    let vwap96: Vec<f64> = (0..n).map(|i| div_pos(qv96[i], v96[i])).collect();
    out.insert(
        "vwap_dev96".to_string(),
        (0..n)
            .map(|i| {
                if c[i].is_finite() && vwap96[i] > 0.0 {
                    (c[i] - vwap96[i]) / vwap96[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    out
}

/// A rolling mean that requires every cell of the window to be finite. This is
/// the strict geometry rule used by `clv16`/`body16`: one zero-range or invalid
/// bar makes the whole window `NaN` instead of being silently skipped.
fn rmean_strict(x: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    let p = WinStats::new(x, times, step);
    for i in w - 1..n {
        if !contig(times, i, w, step) {
            continue;
        }
        if let Some(m) = p.mean_strict(i, w) {
            out[i] = m;
        }
    }
    out
}

fn relu(x: f64) -> f64 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        f64::NAN
    }
}

fn clamp01(x: f64) -> f64 {
    if x.is_finite() {
        x.max(0.0).min(1.0)
    } else {
        f64::NAN
    }
}

/// `relu(x)/(|x|+eps)`: 1 for positive, 0 for negative, 0 at zero.
fn pos_part(x: f64) -> f64 {
    if x.is_finite() {
        relu(x) / (x.abs() + 1e-12)
    } else {
        f64::NAN
    }
}

fn neg_part(x: f64) -> f64 {
    if x.is_finite() {
        relu(-x) / (x.abs() + 1e-12)
    } else {
        f64::NAN
    }
}

/// Compute the complete default candle/price-action block (#261) for one
/// symbol, at the 15 m base (or whatever spacing the store uses).
///
/// `clv`, `body`, `wasym`, `rng_atr`, `inside_s`, `outside_s`, `engulf_s` and
/// `harami_s` use **strict** geometry: a bar with `H == L`, a zero-volume bar
/// (when the store carries volume) or a missing bucket is `NaN`, never `0`.
/// The compact `clv1`/`body1`/`batr` names keep their documented compact
/// convention instead, because there is exactly one implementation per name.
///
/// Every window is trailing and inclusive of `t`; a window that spans a missing
/// bucket is `NaN`, and no value bridges a listing, delisting, migration or
/// outage hole.
fn price_action_all(s: &Series, b: &Base) -> BTreeMap<String, Vec<f64>> {
    let n = s.len();
    let c = &s.close;
    let o = &s.open;
    let h = &s.high;
    let l = &s.low;
    let v = &s.volume;
    let times = &s.times;
    let step = s.step;
    let atr = wilder_py(&b.tr, 14, &b.gap);
    let mut out = BTreeMap::new();

    // Strict single-bar geometry. Volume is only a gate when the store carries
    // the column; an absent volume column is all-NaN here (never a zero-volume
    // bar), so it must not poison the geometry.
    let mut clv = vec![f64::NAN; n];
    let mut body = vec![f64::NAN; n];
    let mut rng = vec![f64::NAN; n];
    let mut valid = vec![false; n];
    for i in 0..n {
        let volume_ok = !v[i].is_finite() || v[i] > 0.0;
        let finite = o[i].is_finite()
            && h[i].is_finite()
            && l[i].is_finite()
            && c[i].is_finite();
        let r = h[i] - l[i];
        if !finite || !volume_ok || r <= 0.0 {
            continue;
        }
        valid[i] = true;
        clv[i] = (c[i] - l[i]) / r - 0.5;
        body[i] = (c[i] - o[i]) / r;
        rng[i] = r;
    }

    out.insert("clv16".to_string(), rmean_strict(&clv, 16, times, step));
    out.insert("body16".to_string(), rmean_strict(&body, 16, times, step));
    out.insert(
        "rng_atr".to_string(),
        (0..n)
            .map(|i| {
                if valid[i] && atr[i] > 0.0 {
                    rng[i] / atr[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    // Close inside the previous bar's range, as a centred [0,1] position.
    let mut prev_range_pos = vec![f64::NAN; n];
    for i in 1..n {
        if !valid[i] || !valid[i - 1] || times[i] - times[i - 1] != step {
            continue;
        }
        let r0 = h[i - 1] - l[i - 1];
        if r0 > 0.0 {
            prev_range_pos[i] = (c[i] - l[i - 1]) / r0 - 0.5;
        }
    }
    out.insert("prev_range_pos".to_string(), prev_range_pos);

    // Continuous two-bar strengths. Relu penetrations replace hard flags; a
    // missing or degenerate previous bar is NaN.
    let mut inside = vec![f64::NAN; n];
    let mut outside = vec![f64::NAN; n];
    let mut engulf = vec![f64::NAN; n];
    let mut harami = vec![f64::NAN; n];
    for i in 1..n {
        if !valid[i] || !valid[i - 1] || times[i] - times[i - 1] != step {
            continue;
        }
        let a = atr[i];
        if !(a.is_finite() && a > 0.0) {
            continue;
        }
        inside[i] = relu(1.0 - relu(h[i] - h[i - 1]) / a)
            * relu(1.0 - relu(l[i - 1] - l[i]) / a);
        outside[i] = relu(1.0 - relu(h[i - 1] - h[i]) / a)
            * relu(1.0 - relu(l[i] - l[i - 1]) / a);

        let b0 = body[i - 1];
        let b1 = body[i];
        let size = relu((b1.abs() - b0.abs()) / (b1.abs() + b0.abs() + 1e-12));
        let bull = pos_part(b1)
            * neg_part(b0)
            * size
            * clamp01(relu(c[i] - o[i - 1]) / a);
        let bear = neg_part(b1)
            * pos_part(b0)
            * size
            * clamp01(relu(o[i] - c[i - 1]) / a);
        engulf[i] = bull - bear;

        // Harami: opposite body colours, today's body at most half of the
        // previous body, and today's body contained in the previous body.
        let shrink = if b0.abs() > 1e-12 {
            relu(1.0 - b1.abs() / (0.5 * b0.abs()))
        } else {
            0.0
        };
        let r0 = h[i - 1] - l[i - 1];
        let (hi0, lo0) = (o[i - 1].max(c[i - 1]), o[i - 1].min(c[i - 1]));
        let (hi1, lo1) = (o[i].max(c[i]), o[i].min(c[i]));
        let contain = if r0 > 0.0 {
            relu(1.0 - relu(hi1 - hi0) / r0) * relu(1.0 - relu(lo0 - lo1) / r0)
        } else {
            0.0
        };
        let opp = pos_part(b0) * neg_part(b1) + neg_part(b0) * pos_part(b1);
        let dir = pos_part(b1) - neg_part(b1);
        harami[i] = dir * opp * shrink * contain;
    }
    out.insert("inside_s".to_string(), inside);
    out.insert("outside_s".to_string(), outside);
    out.insert("engulf_s".to_string(), engulf);
    out.insert("harami_s".to_string(), harami);

    // Signed close-to-close streak, clipped to +/-8 and divided by 8. A gap or
    // a non-finite close resets the streak; the reset bar is NaN because its
    // predecessor is not comparable.
    let mut run8 = vec![f64::NAN; n];
    let mut streak: i32 = 0;
    for i in 0..n {
        if i == 0 || b.gap[i] || !c[i].is_finite() || !c[i - 1].is_finite() {
            streak = 0;
            continue;
        }
        if c[i] > c[i - 1] {
            streak = if streak > 0 { streak + 1 } else { 1 };
        } else if c[i] < c[i - 1] {
            streak = if streak < 0 { streak - 1 } else { -1 };
        } else {
            streak = 0;
        }
        let clipped = streak.max(-8).min(8) as f64;
        run8[i] = clipped / 8.0;
    }
    out.insert("run8".to_string(), run8);

    // Four-bar compression ratio R_t / min(R_{t-3..t}); requires four
    // contiguous, strictly positive ranges.
    let mut nr4 = vec![f64::NAN; n];
    for i in 3..n {
        if !contig(times, i, 4, step) {
            continue;
        }
        let mut ok = true;
        let mut min_r = f64::INFINITY;
        for j in i - 3..=i {
            if !valid[j] {
                ok = false;
                break;
            }
            min_r = min_r.min(h[j] - l[j]);
        }
        if ok && min_r > 0.0 {
            nr4[i] = (h[i] - l[i]) / min_r;
        }
    }
    out.insert("nr4".to_string(), nr4);

    // Trend context: close relative to the trailing 96-bar simple mean. The
    // mean is taken over strict-valid closes, so a zero-range or zero-volume
    // bar anywhere in the window makes the feature NaN instead of being
    // silently skipped.
    let close_strict: Vec<f64> = (0..n)
        .map(|i| if valid[i] { c[i] } else { f64::NAN })
        .collect();
    let ma96 = rmean_strict(&close_strict, 96, times, step);
    out.insert(
        "trend96".to_string(),
        (0..n)
            .map(|i| {
                if valid[i] && ma96[i] > 0.0 && c[i] > 0.0 {
                    c[i] / ma96[i] - 1.0
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    out
}

/// Compute the 52 compact features for one symbol.
///
/// The formulas are those of `pilot/build_panel_15m_100.py` at the 15 m base,
/// with the issue's gap policy: a window whose timestamps are not contiguous is
/// `NaN`, and Wilder/EMA state restarts after a gap. The compact builder used
/// positional NumPy windows and therefore bridged gaps; that difference is
/// deliberate and is recorded per column in the parity report. On contiguous
/// rows (the parity fixture) the two definitions agree to rounding.
///
/// Degenerate bars (`high == low`) use the compact convention:
/// `clv1 = -0.5`, `body1 = 0`, `lower_wick = NaN`, `wasym1 = NaN`.
fn compact_all(s: &Series, b: &Base, cross: &Cross) -> BTreeMap<String, Vec<f64>> {
    let n = s.len();
    let c = &s.close;
    let o = &s.open;
    let h = &s.high;
    let l = &s.low;
    let v = &s.volume;
    let tb = &s.taker_buy_base;
    let trades = &s.trades;
    let times = &s.times;
    let step = s.step;
    let gap = &b.gap;
    let mut out = BTreeMap::new();

    // Returns and moving averages.
    out.insert("r16".to_string(), ret_k(s, 16));
    out.insert("r96".to_string(), ret_k(s, 96));
    out.insert("r672".to_string(), ret_k(s, 672));
    let ma96 = rmean_py(c, 96, times, step);
    let ma672 = rmean_py(c, 672, times, step);
    let ratio = |m: &[f64]| -> Vec<f64> {
        (0..n).map(|i| if c[i] > 0.0 { div(c[i], m[i]) - 1.0 } else { f64::NAN }).collect()
    };
    out.insert("ma96".to_string(), ratio(&ma96));
    out.insert("ma672".to_string(), ratio(&ma672));
    let lag672 = rlag(&ma672, 672, times, step);
    out.insert(
        "ma_slope672".to_string(),
        (0..n)
            .map(|i| if ma672[i].is_finite() && lag672[i].is_finite() && lag672[i] != 0.0 {
                ma672[i] / lag672[i] - 1.0
            } else {
                f64::NAN
            })
            .collect(),
    );

    let e12 = ema(c, 12, gap);
    let e26 = ema(c, 26, gap);
    let macd: Vec<f64> = (0..n)
        .map(|i| if e12[i].is_finite() && e26[i].is_finite() && e26[i] > 0.0 {
            e12[i] / e26[i] - 1.0
        } else {
            f64::NAN
        })
        .collect();
    let macd_sig = ema(&macd, 9, gap);
    out.insert(
        "macd_hist".to_string(),
        (0..n)
            .map(|i| if macd[i].is_finite() && macd_sig[i].is_finite() {
                macd[i] - macd_sig[i]
            } else {
                f64::NAN
            })
            .collect(),
    );

    // ATR family.
    let tr = &b.tr;
    let atr14 = wilder_py(tr, 14, gap);
    let atr16 = wilder_py(tr, 16, gap);
    let atr96 = wilder_py(tr, 96, gap);
    out.insert(
        "natr14".to_string(),
        (0..n).map(|i| if atr14[i].is_finite() { div_pos(100.0 * atr14[i], c[i]) } else { f64::NAN }).collect(),
    );

    // Volatility.
    let lr = ret_k(s, 1);
    let vol16 = rstd_py(&lr, 16, times, step);
    let vol96 = rstd_py(&lr, 96, times, step);
    let vol672 = rstd_py(&lr, 672, times, step);
    out.insert("vol16".to_string(), vol16.clone());
    out.insert("vol96".to_string(), vol96.clone());
    out.insert(
        "vol_ts".to_string(),
        (0..n).map(|i| div_pos(vol16[i], vol96[i])).collect(),
    );
    out.insert(
        "vol_ratio".to_string(),
        (0..n).map(|i| div_pos(vol96[i], vol672[i])).collect(),
    );
    let vz2880_mean = rmean_py(&vol96, 2880, times, step);
    let vz2880_std = rstd_py(&vol96, 2880, times, step);
    out.insert(
        "vol_z2880".to_string(),
        (0..n)
            .map(|i| if vz2880_std[i] > 0.0 { div(vol96[i] - vz2880_mean[i], vz2880_std[i]) } else { f64::NAN })
            .collect(),
    );
    let hl = (0..n)
        .map(|i| if h[i] > 0.0 && l[i] > 0.0 { (h[i] / l[i].max(1e-12)).ln() } else { f64::NAN })
        .collect::<Vec<f64>>();
    let hl2: Vec<f64> = hl.iter().map(|x| x * x).collect();
    let hl2m = rmean_py(&hl2, 96, times, step);
    out.insert(
        "parkinson96".to_string(),
        hl2m.iter()
            .map(|m| if m.is_finite() { (m / (4.0 * std::f64::consts::LN_2)).sqrt() } else { f64::NAN })
            .collect(),
    );
    let rv = |w: usize| {
        let x: Vec<f64> = lr.iter().map(|r| if r.is_finite() { r * r } else { f64::NAN }).collect();
        rmean_py(&x, w, times, step)
    };
    let rv96 = rv(96);
    let rv672 = rv(672);
    out.insert(
        "rv_ratio".to_string(),
        (0..n)
            .map(|i| if rv96[i] > 0.0 && rv672[i] > 0.0 { (rv96[i] / rv672[i]).ln() } else { f64::NAN })
            .collect(),
    );

    // RSI / Bollinger / DI.
    out.insert("rsi14".to_string(), rsi_py(c, 14, gap));
    let sd96 = rstd_py(c, 96, times, step);
    out.insert(
        "bb96".to_string(),
        (0..n)
            .map(|i| if sd96[i] > 0.0 {
                clip(div(c[i] - ma96[i], 2.0 * sd96[i]), -1.0, 1.0)
            } else {
                f64::NAN
            })
            .collect(),
    );
    out.insert(
        "bb_width96".to_string(),
        (0..n).map(|i| if sd96[i].is_finite() { div_pos(4.0 * sd96[i], ma96[i]) } else { f64::NAN }).collect(),
    );

    let mut pdm = vec![0.0f64; n];
    let mut mdm = vec![0.0f64; n];
    for i in 1..n {
        if times[i] - times[i - 1] != step {
            pdm[i] = f64::NAN;
            mdm[i] = f64::NAN;
            continue;
        }
        let up = h[i] - h[i - 1];
        let dn = l[i - 1] - l[i];
        if up > dn && up > 0.0 {
            pdm[i] = up;
        }
        if dn > up && dn > 0.0 {
            mdm[i] = dn;
        }
    }
    let pdi_r = wilder_py(&pdm, 14, gap);
    let mdi_r = wilder_py(&mdm, 14, gap);
    let mut pdi = vec![f64::NAN; n];
    let mut mdi = vec![f64::NAN; n];
    for i in 0..n {
        if atr14[i] > 0.0 {
            pdi[i] = 100.0 * pdi_r[i] / atr14[i];
            mdi[i] = 100.0 * mdi_r[i] / atr14[i];
        }
    }
    let dx: Vec<f64> = (0..n)
        .map(|i| {
            if pdi[i].is_finite() && mdi[i].is_finite() && pdi[i] + mdi[i] > 0.0 {
                100.0 * (pdi[i] - mdi[i]).abs() / (pdi[i] + mdi[i])
            } else {
                f64::NAN
            }
        })
        .collect();
    let adx = wilder_py(&dx, 14, gap);
    out.insert("adx14".to_string(), adx.iter().map(|x| x / 100.0).collect());
    out.insert(
        "di_diff14".to_string(),
        (0..n)
            .map(|i| if pdi[i].is_finite() && mdi[i].is_finite() { (pdi[i] - mdi[i]) / 100.0 } else { f64::NAN })
            .collect(),
    );

    // Flow / volume-price.
    let mfm: Vec<f64> = (0..n)
        .map(|i| {
            if h[i] > l[i] {
                ((c[i] - l[i]) - (h[i] - c[i])) / (h[i] - l[i])
            } else {
                f64::NAN
            }
        })
        .collect();
    let mfv: Vec<f64> = mfm.iter().zip(v).map(|(m, x)| m * x).collect();
    let mfv20 = rmean_py(&mfv, 20, times, step);
    let v20 = rmean_py(v, 20, times, step);
    out.insert("cmf20".to_string(), (0..n).map(|i| div_pos(mfv20[i], v20[i])).collect());

    let tp: Vec<f64> = (0..n).map(|i| (h[i] + l[i] + c[i]) / 3.0).collect();
    let mf: Vec<f64> = tp.iter().zip(v).map(|(p, x)| p * x).collect();
    let mut pos = vec![0.0f64; n];
    let mut neg = vec![0.0f64; n];
    for i in 1..n {
        if times[i] - times[i - 1] != step {
            pos[i] = f64::NAN;
            neg[i] = f64::NAN;
            continue;
        }
        if tp[i] > tp[i - 1] {
            pos[i] = mf[i];
        } else if tp[i] < tp[i - 1] {
            neg[i] = mf[i];
        }
    }
    let wpos = wilder_py(&pos, 14, gap);
    let wneg = wilder_py(&neg, 14, gap);
    out.insert(
        "mfi14".to_string(),
        (0..n)
            .map(|i| {
                if wpos[i].is_finite() && wneg[i] > 0.0 {
                    (100.0 - 100.0 / (1.0 + wpos[i] / wneg[i])) / 50.0 - 1.0
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
    let pv: Vec<f64> = tp.iter().zip(v).map(|(p, x)| p * x).collect();
    let pvm = rmean_py(&pv, 96, times, step);
    let v96 = rmean_py(v, 96, times, step);
    let vwap: Vec<f64> = (0..n).map(|i| div_pos(pvm[i], v96[i])).collect();
    out.insert(
        "vwap_dev96".to_string(),
        (0..n).map(|i| if vwap[i] > 0.0 { div(c[i] - vwap[i], vwap[i]) } else { f64::NAN }).collect(),
    );
    let logv: Vec<f64> = v.iter().map(|x| if *x >= 0.0 { (1.0 + x).ln() } else { f64::NAN }).collect();
    let lv_m = rmean_py(&logv, 96, times, step);
    let lv_s = rstd_py(&logv, 96, times, step);
    out.insert(
        "vol_z96".to_string(),
        (0..n).map(|i| if lv_s[i] > 0.0 { div(logv[i] - lv_m[i], lv_s[i]) } else { f64::NAN }).collect(),
    );
    let tb96 = rmean_py(tb, 96, times, step);
    out.insert("taker_ratio96".to_string(), (0..n).map(|i| div_pos(tb96[i], v96[i])).collect());
    let tb16 = rmean_py(tb, 16, times, step);
    let v16m = rmean_py(v, 16, times, step);
    out.insert(
        "taker_imb16".to_string(),
        (0..n).map(|i| div_pos(2.0 * tb16[i] - v16m[i], v16m[i])).collect(),
    );
    let trades96 = rmean_py(trades, 96, times, step);
    out.insert(
        "trade_intensity96".to_string(),
        (0..n).map(|i| if trades96[i].is_finite() && v96[i] > 0.0 {
            (1.0 + trades96[i] / v96[i]).ln()
        } else {
            f64::NAN
        }).collect(),
    );

    // Candle geometry (compact conventions).
    let clv: Vec<f64> = (0..n)
        .map(|i| {
            if h[i].is_finite() && l[i].is_finite() && c[i].is_finite() {
                if h[i] > l[i] { (c[i] - l[i]) / (h[i] - l[i]) - 0.5 } else { -0.5 }
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert("clv1".to_string(), clv.clone());
    out.insert("clv4".to_string(), rmean_py(&clv, 4, times, step));
    let body: Vec<f64> = (0..n)
        .map(|i| {
            if h[i].is_finite() && l[i].is_finite() && c[i].is_finite() && o[i].is_finite() {
                if h[i] > l[i] { (c[i] - o[i]) / (h[i] - l[i]) } else { 0.0 }
            } else {
                f64::NAN
            }
        })
        .collect();
    out.insert("body1".to_string(), body.clone());
    out.insert(
        "batr".to_string(),
        (0..n).map(|i| if atr14[i] > 0.0 { div(c[i] - o[i], atr14[i]) } else { f64::NAN }).collect(),
    );
    let upper: Vec<f64> = (0..n)
        .map(|i| if h[i] > l[i] { (h[i] - o[i].max(c[i])) / (h[i] - l[i]) } else { f64::NAN })
        .collect();
    let lower: Vec<f64> = (0..n)
        .map(|i| if h[i] > l[i] { (o[i].min(c[i]) - l[i]) / (h[i] - l[i]) } else { f64::NAN })
        .collect();
    out.insert(
        "wasym1".to_string(),
        (0..n)
            .map(|i| if upper[i].is_finite() && lower[i].is_finite() { upper[i] - lower[i] } else { f64::NAN })
            .collect(),
    );
    out.insert("lower_wick".to_string(), lower);
    out.insert("rng1".to_string(), (0..n).map(|i| div_pos(h[i] - l[i], c[i])).collect());
    out.insert("volrat".to_string(), (0..n).map(|i| div_pos(atr16[i], atr96[i])).collect());
    let prev_close = rlag(c, 1, times, step);
    out.insert(
        "gapatr".to_string(),
        (0..n)
            .map(|i| if atr14[i] > 0.0 && prev_close[i].is_finite() {
                div(o[i] - prev_close[i], atr14[i])
            } else {
                f64::NAN
            })
            .collect(),
    );

    // Exotic statistical family.
    let x: Vec<f64> = (0..n)
        .map(|i| if c[i] > 0.0 && ma96[i] > 0.0 { window_stats::log1p_det((c[i] - ma96[i]) / ma96[i]) } else { f64::NAN })
        .collect();
    let xl = rlag(&x, 1, times, step);
    let ou_ratio = window_stats::comoment_ratio(&x, &xl, 96, times, step);
    out.insert(
        "ou_hl96".to_string(),
        ou_ratio
            .iter()
            .map(|r| {
                if r.is_finite() {
                    let phi = r.max(1e-6).min(0.999999);
                    let hl = -std::f64::consts::LN_2 / phi.ln();
                    (1.0 + hl.max(0.0).min(1000.0)).ln()
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
    let lrl = rlag(&lr, 1, times, step);
    let lr_lrl: Vec<f64> = lr.iter().zip(&lrl).map(|(a, b)| a * b).collect();
    let num96 = rmean_py(&lr_lrl, 384, times, step);
    let lr384 = rmean_py(&lr, 384, times, step);
    let lrl384 = rmean_py(&lrl, 384, times, step);
    let lr384_sd = rstd_py(&lr, 384, times, step);
    let lrl384_sd = rstd_py(&lrl, 384, times, step);
    out.insert(
        "acf1".to_string(),
        (0..n)
            .map(|i| {
                let den = lr384_sd[i] * lrl384_sd[i];
                if den > 0.0 && num96[i].is_finite() {
                    (num96[i] - lr384[i] * lrl384[i]) / den
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
    let r4 = ret_k(s, 4);
    let r4_sd = rstd_py(&r4, 384, times, step);
    out.insert(
        "vr4".to_string(),
        (0..n)
            .map(|i| {
                let den = 4.0 * lr384_sd[i] * lr384_sd[i];
                if den > 0.0 { div(r4_sd[i] * r4_sd[i], den) } else { f64::NAN }
            })
            .collect(),
    );
    let lr_m = rmean_py(&lr, 96, times, step);
    let d2: Vec<f64> = lr.iter().zip(&lr_m).map(|(r, m)| (r - m) * (r - m)).collect();
    let d3: Vec<f64> = lr.iter().zip(&lr_m).map(|(r, m)| (r - m) * (r - m) * (r - m)).collect();
    let m2 = rmean_py(&d2, 96, times, step);
    let m3 = rmean_py(&d3, 96, times, step);
    out.insert(
        "rskew96".to_string(),
        (0..n).map(|i| if m2[i] > 0.0 { div(m3[i], m2[i].powf(1.5)) } else { f64::NAN }).collect(),
    );
    let abs_lr: Vec<f64> = lr.iter().map(|r| r.abs()).collect();
    let abs_lrl: Vec<f64> = lrl.iter().map(|r| r.abs()).collect();
    let abs_prod: Vec<f64> = abs_lr.iter().zip(&abs_lrl).map(|(a, b)| a * b).collect();
    let bv = rmean_py(&abs_prod, 96, times, step);
    out.insert(
        "jump_share".to_string(),
        (0..n)
            .map(|i| if rv96[i] > 0.0 {
                ((rv96[i] - (std::f64::consts::PI / 2.0) * bv[i]).max(0.0)) / rv96[i]
            } else {
                f64::NAN
            })
            .collect(),
    );
    let dn2: Vec<f64> = lr
        .iter()
        .map(|r| if !r.is_finite() { f64::NAN } else if *r < 0.0 { r * r } else { 0.0 })
        .collect();
    let up2: Vec<f64> = lr
        .iter()
        .map(|r| if !r.is_finite() { f64::NAN } else if *r > 0.0 { r * r } else { 0.0 })
        .collect();
    let rsm = rmean_py(&dn2, 96, times, step);
    let rsp = rmean_py(&up2, 96, times, step);
    out.insert(
        "semivar_asym".to_string(),
        (0..n).map(|i| if rv96[i] > 0.0 { div(rsm[i] - rsp[i], rv96[i]) } else { f64::NAN }).collect(),
    );
    let lr_sd96 = rstd_py(&lr, 96, times, step);
    let zr: Vec<f64> = (0..n)
        .map(|i| if lr[i].is_finite() && lr_sd96[i] > 0.0 { (lr[i] - lr_m[i]) / lr_sd96[i] } else { f64::NAN })
        .collect();
    let mut cs = vec![0.0f64; n];
    let mut acc = 0.0;
    for i in 0..n {
        if i > 0 && gap[i] {
            acc = 0.0;
        }
        if zr[i].is_finite() {
            acc += zr[i];
        }
        cs[i] = acc;
    }
    let mut cusum = vec![f64::NAN; n];
    for i in 96..n {
        if contig(times, i, 96, step) {
            let base = if gap[i - 95] { 0.0 } else { cs[i - 96] };
            cusum[i] = (cs[i] - base).abs() / (96.0f64).sqrt();
        }
    }
    out.insert("cusum_abs_z".to_string(), cusum);

    // Cross-asset BTC features at the same timestamp.
    let btc_r96: Vec<f64> = (0..n).map(|i| Cross::at(&cross.r96, times[i])).collect();
    let btc_vol96: Vec<f64> = (0..n).map(|i| Cross::at(&cross.vol96, times[i])).collect();
    out.insert("btc_r96".to_string(), btc_r96.clone());
    out.insert("btc_vol96".to_string(), btc_vol96);
    let r96_self = ret_k(s, 96);
    out.insert(
        "rel_btc96".to_string(),
        (0..n)
            .map(|i| {
                if r96_self[i].is_finite() && btc_r96[i].is_finite() {
                    r96_self[i] - btc_r96[i]
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );
    let mut beta = vec![f64::NAN; n];
    for i in 0..n {
        if !contig(times, i, 384, step) {
            continue;
        }
        let lo = i + 1 - 384;
        let (mut sx, mut sy, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        let mut ok = true;
        for j in lo..=i {
            let br = Cross::at(&cross.r1, times[j]);
            if !lr[j].is_finite() || !br.is_finite() {
                ok = false;
                break;
            }
            sx += lr[j];
            sy += br;
            syy += br * br;
            sxy += lr[j] * br;
        }
        if ok {
            let nf = 384.0;
            let cov = (sxy / nf - (sx / nf) * (sy / nf)).max(-0.05).min(0.05);
            let vb = syy / nf - (sy / nf) * (sy / nf);
            if vb > 0.0 {
                beta[i] = (cov / vb).max(-5.0).min(5.0);
            }
        }
    }
    out.insert("beta384".to_string(), beta);

    // Clock and Donchian.
    let bars_per_day = if step > 0 { 86_400 / step } else { 1 };
    let tod = |i: usize, want_sin: bool| {
        let idx = if step > 0 { times[i].rem_euclid(86_400) / step } else { 0 };
        let a = 2.0 * std::f64::consts::PI * idx as f64 / bars_per_day as f64;
        if want_sin { a.sin() } else { a.cos() }
    };
    out.insert("tod_sin".to_string(), (0..n).map(|i| tod(i, true)).collect());
    out.insert("tod_cos".to_string(), (0..n).map(|i| tod(i, false)).collect());
    let dow = |i: usize, want_sin: bool| {
        let ww = weekday(times[i].div_euclid(86_400)) as f64;
        let a = 2.0 * std::f64::consts::PI * ww / 7.0;
        if want_sin { a.sin() } else { a.cos() }
    };
    out.insert("dow_sin".to_string(), (0..n).map(|i| dow(i, true)).collect());
    out.insert("dow_cos".to_string(), (0..n).map(|i| dow(i, false)).collect());
    let lo96 = rolling_min(l, 96, times, step);
    let hi96 = rolling_max(h, 96, times, step);
    out.insert(
        "don96".to_string(),
        (0..n)
            .map(|i| {
                if lo96[i].is_finite() && hi96[i].is_finite() {
                    if hi96[i] > lo96[i] {
                        (c[i] - lo96[i]) / (hi96[i] - lo96[i]) - 0.5
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            })
            .collect(),
    );

    // The four flow names shared with the compact subset use the canonical
    // #270 definition, including quote-volume VWAP. There is exactly one
    // implementation per name.
    let flow = flow_all(s);
    for name in [
        "vwap_dev96",
        "taker_ratio96",
        "taker_imb16",
        "trade_intensity96",
    ] {
        if let Some(vals) = flow.get(name) {
            out.insert(name.to_string(), vals.clone());
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Temp column writer
// ---------------------------------------------------------------------------

struct TempColumn {
    name: String,
    path: PathBuf,
    writer: Option<BufWriter<std::fs::File>>,
    /// Reused byte staging so a column segment reaches the writer and the hash
    /// as one block instead of one 8-byte call per value.
    buf: Vec<u8>,
    hash: Sha256Hasher,
    warmup: usize,
    nan_rows: usize,
    first_finite: Option<usize>,
    label: bool,
    row_offset: usize,
}

/// Staging size for [`TempColumn::write_values`]; a multiple of 8 and of the
/// 64-byte SHA-256 block.
const WRITE_CHUNK_BYTES: usize = 1 << 16;

impl TempColumn {
    fn create(dir: &Path, name: &str, warmup: usize, label: bool) -> Result<Self, String> {
        let safe: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let path = dir.join(format!("col-{}.f64", safe));
        let file = std::fs::File::create(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Ok(TempColumn {
            name: name.to_string(),
            path,
            writer: Some(BufWriter::with_capacity(1 << 20, file)),
            buf: Vec::with_capacity(WRITE_CHUNK_BYTES),
            hash: Sha256Hasher::new(),
            warmup,
            nan_rows: 0,
            first_finite: None,
            label,
            row_offset: 0,
        })
    }
    /// Discard the current contents and start the column again. Used only when
    /// the canonical whole-panel pass replaces a legacy per-symbol preview
    /// (the five py15m market names).
    fn reset(&mut self) -> Result<(), String> {
        self.flush()?;
        self.writer.take();
        let file = std::fs::File::create(&self.path)
            .map_err(|e| format!("{}: {}", self.path.display(), e))?;
        self.writer = Some(BufWriter::with_capacity(1 << 20, file));
        self.buf.clear();
        self.hash = Sha256Hasher::new();
        self.nan_rows = 0;
        self.first_finite = None;
        self.row_offset = 0;
        Ok(())
    }

    /// Flush the staging block into the writer and the hash (same bytes, same
    /// order, so the digest is unchanged).
    fn drain_buf(&mut self) -> Result<(), String> {
        if self.buf.is_empty() {
            return Ok(());
        }
        if let Some(w) = self.writer.as_mut() {
            w.write_all(&self.buf)
                .map_err(|e| format!("{}: {}", self.path.display(), e))?;
            self.hash.update(&self.buf);
        }
        self.buf.clear();
        Ok(())
    }
    fn write_values(&mut self, values: &[f64]) -> Result<(), String> {
        if self.writer.is_none() {
            return Err("temp column already closed".to_string());
        }
        for &v in values {
            if !v.is_finite() {
                self.nan_rows += 1;
            } else if self.first_finite.is_none() {
                self.first_finite = Some(self.row_offset);
            }
            self.buf.extend_from_slice(&v.to_le_bytes());
            if self.buf.len() >= WRITE_CHUNK_BYTES {
                self.drain_buf()?;
            }
            self.row_offset += 1;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        self.drain_buf()?;
        if let Some(w) = self.writer.as_mut() {
            w.flush()
                .map_err(|e| format!("{}: {}", self.path.display(), e))?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<FeatureReport, String> {
        self.flush()?;
        if let Some(mut w) = self.writer.take() {
            w.flush().map_err(|e| format!("{}: {}", self.path.display(), e))?;
        }
        // `finish` consumes the hash: move it out with a fresh empty state.
        let hash = std::mem::replace(&mut self.hash, Sha256Hasher::new()).finalize();
        Ok(FeatureReport {
            name: self.name.clone(),
            warmup: self.warmup,
            nan_rows: self.nan_rows,
            first_finite_row: self.first_finite,
            first_finite_time: None,
            prevalence: 0.0,
            hash,
            label: self.label,
        })
    }
}

// ---------------------------------------------------------------------------
// Computation driver
// ---------------------------------------------------------------------------

struct Context<'a> {
    legacy: Vec<String>,
    own: Vec<String>,
    py15m_requested: Vec<String>,
    compact: Vec<String>,
    flow: Vec<String>,
    price_action: Vec<String>,
    mtf: Vec<String>,
    labels: Vec<(String, LabelSpec)>,
    tf: Option<Timeframe>,
    long_labels: Vec<crate::labels::LongLabelSpec>,
    cross: &'a Cross,
    eth: &'a Cross,
}

fn compute_series(
    s: &Series,
    ctx: &Context,
    cols: &mut BTreeMap<String, TempColumn>,
    long_acc: &mut [crate::labels::LabelAccumulator],
) -> Result<(), String> {
    let base = Base::new(s);
    let bars = if !ctx.legacy.is_empty() || !ctx.labels.is_empty() || !ctx.long_labels.is_empty() {
        Some(s.bars())
    } else {
        None
    };
    if !ctx.legacy.is_empty() {
        let computed = features::compute_with_cross(
            bars.as_ref().unwrap(),
            ctx.tf.unwrap_or(Timeframe::H1),
            &ctx.legacy,
            None,
        )?;
        for c in computed {
            if let Some(t) = cols.get_mut(&c.name) {
                // #125 reports the exact warm-up for its own names; use it
                // rather than this module's nominal table.
                t.warmup = c.warmup;
                t.write_values(&c.values)?;
            }
        }
    }
    if !ctx.py15m_requested.is_empty() {
        let all = py15m_all(s, &base, ctx.cross);
        for name in &ctx.py15m_requested {
            let values = all
                .get(name)
                .ok_or_else(|| format!("py15m column '{}' was not produced", name))?;
            if let Some(t) = cols.get_mut(name) {
                t.write_values(values)?;
            }
        }
    }
    if !ctx.compact.is_empty() {
        let all = compact_all(s, &base, ctx.cross);
        for name in &ctx.compact {
            let values = all
                .get(name)
                .ok_or_else(|| format!("compact column '{}' was not produced", name))?;
            if let Some(t) = cols.get_mut(name) {
                t.write_values(values)?;
            }
        }
    }
    if !ctx.flow.is_empty() {
        let all = flow_all(s);
        for name in &ctx.flow {
            let values = all
                .get(name)
                .ok_or_else(|| format!("flow column '{}' was not produced", name))?;
            if let Some(t) = cols.get_mut(name) {
                t.write_values(values)?;
            }
        }
    }
    if !ctx.price_action.is_empty() {
        let all = price_action_all(s, &base);
        for name in &ctx.price_action {
            let values = all
                .get(name)
                .ok_or_else(|| format!("price_action column '{}' was not produced", name))?;
            if let Some(t) = cols.get_mut(name) {
                t.write_values(values)?;
            }
        }
    }
    if !ctx.mtf.is_empty() {
        let all = mtf_all(s, &ctx.mtf)?;
        for name in &ctx.mtf {
            let values = all
                .get(name)
                .ok_or_else(|| format!("multi-timeframe column '{}' was not produced", name))?;
            if let Some(t) = cols.get_mut(name) {
                t.write_values(values)?;
            }
        }
    }
    for name in &ctx.own {
        let (values, warmup) = compute_own(name, s, &base, ctx)?;
        if let Some(t) = cols.get_mut(name) {
            debug_assert_eq!(t.warmup, warmup);
            t.write_values(&values)?;
        }
    }
    if let Some(bars) = bars {
        for (lname, spec) in &ctx.labels {
            let tf = ctx
                .tf
                .ok_or_else(|| "labels need a recognised timeframe (use --bar-seconds)".to_string())?;
            let col = features::forward_label(&bars, tf, spec);
            if let Some(t) = cols.get_mut(lname) {
                t.write_values(&col.values)?;
            }
        }
        for (spec, acc) in ctx.long_labels.iter().zip(long_acc.iter_mut()) {
            let built = crate::labels::build(&bars, spec, None)?;
            for (kind, _) in crate::labels::COLUMN_KINDS {
                if let (Some(t), Some(v)) = (cols.get_mut(&spec.column(kind)), built.column(kind)) {
                    t.write_values(v)?;
                }
            }
            acc.add(&s.symbol, &built);
        }
    }
    Ok(())
}

fn benchmark_return(s: &Series, b: &Base, cross: &Cross, k: usize, relative: bool) -> Vec<f64> {
    let own = if k == 1 { b.ret1.clone() } else { ret_k(s, k) };
    let map = match k {
        1 => &cross.r1,
        16 => &cross.r16,
        20 => &cross.r20,
        96 => &cross.r96,
        672 => &cross.r672,
        _ => return vec![f64::NAN; s.len()],
    };
    (0..s.len())
        .map(|i| {
            let bench = Cross::at(map, s.times[i]);
            if relative {
                if own[i].is_finite() && bench.is_finite() {
                    own[i] - bench
                } else {
                    f64::NAN
                }
            } else {
                bench
            }
        })
        .collect()
}

fn benchmark_vol(s: &Series, cross: &Cross, w: usize) -> Vec<f64> {
    let map = match w {
        20 => &cross.vol20,
        96 => &cross.vol96,
        _ => return vec![f64::NAN; s.len()],
    };
    (0..s.len())
        .map(|i| Cross::at(map, s.times[i]))
        .collect()
}

fn benchmark_beta(s: &Series, b: &Base, cross: &Cross, w: usize) -> Vec<f64> {
    if s.symbol == cross.symbol {
        (0..s.len())
            .map(|i| {
                if i + 1 >= w && contig(&s.times, i, w, s.step) {
                    1.0
                } else {
                    f64::NAN
                }
            })
            .collect()
    } else {
        beta_window(s, &b.ret1, &cross.r1, w)
    }
}

fn compute_own(
    name: &str,
    s: &Series,
    b: &Base,
    ctx: &Context,
) -> Result<(Vec<f64>, usize), String> {
    let n = s.len();
    let warm = nominal_warmup(name);
    let v = if let Some(k) = name.strip_prefix('r').and_then(|x| x.parse::<usize>().ok()) {
        ret_k(s, k)
    } else if let Some(w) = name.strip_prefix("ma").and_then(|x| x.parse::<usize>().ok()) {
        ma_ratio(s, w)
    } else if let Some(w) = name.strip_prefix("vol").and_then(|x| x.parse::<usize>().ok()) {
        vol_w(s, b, w)
    } else if let Some(k) = name.strip_prefix("lag_r1_").and_then(|x| x.parse::<usize>().ok()) {
        lag(s, b, k)
    } else if let Some(w) = name.strip_prefix("don").and_then(|x| x.parse::<usize>().ok()) {
        donchian(s, w)
    } else {
        match name {
            "bb20_bw" => bb_width(s, 20),
            "vz96" | "vol_z96" => vz(s, 96),
            "natr14" => {
                let atr = wilder_py(&b.tr, 14, &b.gap);
                (0..n)
                    .map(|i| if atr[i].is_finite() { div_pos(100.0 * atr[i], s.close[i]) } else { f64::NAN })
                    .collect()
            }
            "cmf20" => {
                let mfm: Vec<f64> = (0..n)
                    .map(|i| {
                        if s.high[i] > s.low[i] {
                            ((s.close[i] - s.low[i]) - (s.high[i] - s.close[i]))
                                / (s.high[i] - s.low[i])
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
                let mfv: Vec<f64> = mfm.iter().zip(&s.volume).map(|(m, v)| m * v).collect();
                let mfv20 = rmean_py(&mfv, 20, &s.times, s.step);
                let v20 = rmean_py(&s.volume, 20, &s.times, s.step);
                (0..n).map(|i| div_pos(mfv20[i], v20[i])).collect()
            }
            "clv1" => clv1(s),
            "range1" => range1(s),
            "dow_sin" | "dow_cos" | "hour_sin" | "hour_cos" | "month_sin" | "month_cos"
            | "dom_sin" | "dom_cos" => calendar_column(s, name),
            "btc_r1" | "rel_btc_r1" | "cross_ret1" => {
                benchmark_return(s, b, ctx.cross, 1, name == "rel_btc_r1")
            }
            "btc_r16" | "rel_btc_r16" => {
                benchmark_return(s, b, ctx.cross, 16, name == "rel_btc_r16")
            }
            "btc_r20" | "rel_btc_r20" => {
                benchmark_return(s, b, ctx.cross, 20, name == "rel_btc_r20")
            }
            "btc_r96" | "rel_btc_r96" => {
                benchmark_return(s, b, ctx.cross, 96, name == "rel_btc_r96")
            }
            "btc_r672" => benchmark_return(s, b, ctx.cross, 672, false),
            "btc_vol20" => benchmark_vol(s, ctx.cross, 20),
            "btc_vol96" => benchmark_vol(s, ctx.cross, 96),
            "beta_btc_60" | "cross_beta20" => {
                let w = if name == "cross_beta20" { 20 } else { 60 };
                benchmark_beta(s, b, ctx.cross, w)
            }
            "beta_btc_384" => benchmark_beta(s, b, ctx.cross, 384),
            "eth_r1" | "rel_eth_r1" => benchmark_return(s, b, ctx.eth, 1, name == "rel_eth_r1"),
            "eth_r16" | "rel_eth_r16" => {
                benchmark_return(s, b, ctx.eth, 16, name == "rel_eth_r16")
            }
            "eth_r20" | "rel_eth_r20" => {
                benchmark_return(s, b, ctx.eth, 20, name == "rel_eth_r20")
            }
            "eth_r96" | "rel_eth_r96" => {
                benchmark_return(s, b, ctx.eth, 96, name == "rel_eth_r96")
            }
            "eth_r672" => benchmark_return(s, b, ctx.eth, 672, false),
            "eth_vol20" => benchmark_vol(s, ctx.eth, 20),
            "eth_vol96" => benchmark_vol(s, ctx.eth, 96),
            "beta_eth_60" => benchmark_beta(s, b, ctx.eth, 60),
            "beta_eth_384" => benchmark_beta(s, b, ctx.eth, 384),
            other => {
                if FLOW_NAMES.contains(&other) {
                    let mut all = flow_all(s);
                    return Ok((
                        all.remove(other)
                            .ok_or_else(|| format!("flow column '{}' was not produced", other))?,
                        warm,
                    ));
                }
                if is_compact(other) {
                    let mut all = compact_all(s, b, ctx.cross);
                    return Ok((
                        all.remove(other)
                            .ok_or_else(|| format!("compact column '{}' was not produced", other))?,
                        warm,
                    ));
                }
                if PRICE_ACTION_NAMES.contains(&other) {
                    let mut all = price_action_all(s, b);
                    return Ok((
                        all.remove(other)
                            .ok_or_else(|| format!("price_action column '{}' was not produced", other))?,
                        warm,
                    ));
                }
                return Err(format!("unknown feature '{}'", other));
            }
        }
    };
    Ok((v, warm))
}

// ---------------------------------------------------------------------------
// Cross-sectional ranks
// ---------------------------------------------------------------------------

fn read_f64_file(path: &Path, expect_rows: usize) -> Result<Vec<f64>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let len = f.metadata().map_err(|e| format!("{}: {}", path.display(), e))?.len();
    if len != (expect_rows as u64) * 8 {
        return Err(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            len,
            expect_rows as u64 * 8
        ));
    }
    let mut bytes = vec![0u8; len as usize];
    f.read_exact(&mut bytes)
        .map_err(|e| format!("{}: {}", path.display(), e))?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            f64::from_le_bytes(b)
        })
        .collect())
}

/// Rank `src` across symbols at each timestamp.
///
/// The result is `2*(rank - 1)/(n_finite - 1) - 1` in `[-1, 1]`, with ties
/// broken by symbol name and then row order. A timestamp with fewer than
/// [`MIN_CROSS_MEMBERS`] finite values, or a non-finite source value, is `NaN`.
/// Ranks are exact rational values rounded once to f64, so repeats are
/// bit-stable.
pub fn xs_rank(
    src: &[f64],
    times: &[i64],
    sym_key: &[u32],
    rows: usize,
) -> Result<Vec<f64>, String> {
    rank_with_membership(src, times, sym_key, None, rows)
}

/// Point-in-time form of [`xs_rank`]. Rows with `member[r] == false` are
/// excluded from both the rank and the member count, and their output is NaN.
fn rank_with_membership(
    src: &[f64],
    times: &[i64],
    sym_key: &[u32],
    member: Option<&[bool]>,
    rows: usize,
) -> Result<Vec<f64>, String> {
    if src.len() != rows || times.len() != rows || sym_key.len() != rows {
        return Err("cross-sectional inputs have inconsistent lengths".to_string());
    }
    if let Some(m) = member {
        if m.len() != rows {
            return Err("cross-sectional membership has inconsistent length".to_string());
        }
    }
    // Guard the one-column-at-a-time working set (source + ts + key + order +
    // output) against a hard memory ceiling.
    let bytes = (rows as u64).saturating_mul(32);
    if let Some(limit) = store::detected_hard_budget() {
        if bytes > limit {
            return Err(format!(
                "cross-sectional ranks need about {} MiB for {} rows, above the hard limit of {} MiB; \
                 use a smaller row range or --features without xs",
                bytes / (1024 * 1024),
                rows,
                limit / (1024 * 1024)
            ));
        }
    }
    let mut order: Vec<u32> = (0..rows as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        times[a]
            .cmp(&times[b])
            .then(sym_key[a].cmp(&sym_key[b]))
            .then(a.cmp(&b))
    });
    Ok(rank_in_order(src, times, &order, sym_key, member, rows))
}

/// Rank against an order already sorted by `(time, symbol, row)`.
fn rank_in_order(
    src: &[f64],
    times: &[i64],
    order: &[u32],
    sym_key: &[u32],
    member: Option<&[bool]>,
    rows: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; rows];
    let mut i = 0usize;
    while i < order.len() {
        let t = times[order[i] as usize];
        let mut j = i;
        while j < order.len() && times[order[j] as usize] == t {
            j += 1;
        }
        let mut group: Vec<usize> = (i..j).map(|k| order[k] as usize).collect();
        group.retain(|&r| src[r].is_finite() && member.map(|m| m[r]).unwrap_or(true));
        group.sort_unstable_by(|&a, &b| {
            src[a]
                .partial_cmp(&src[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(sym_key[a].cmp(&sym_key[b]))
                .then(a.cmp(&b))
        });
        let m = group.len();
        if m >= MIN_CROSS_MEMBERS {
            for (rank, &r) in group.iter().enumerate() {
                out[r] = 2.0 * rank as f64 / (m - 1) as f64 - 1.0;
            }
        }
        i = j;
    }
    out
}

// ---------------------------------------------------------------------------
// Whole-panel PIT cross-section pass
// ---------------------------------------------------------------------------

/// Row context shared by every point-in-time cross-sectional statistic.
struct CrossPanel {
    times: Vec<i64>,
    sym_key: Vec<u32>,
    /// `member_asof <= t` when the input carries a membership column; every
    /// row otherwise. Pre-listing/post-delisting rows never enter a statistic.
    member: Vec<bool>,
    /// Row order `(time, symbol, row)` and the start index of each timestamp
    /// group (`starts.len() == groups+1`). Built on first CPU use only; the
    /// device path keeps its own permutation.
    grouped: std::sync::OnceLock<(Vec<u32>, Vec<usize>)>,
}

impl CrossPanel {
    fn new(
        store: &FileStore,
        segs: &[Segment],
        time_col: usize,
        time_dtype: DType,
        member_col: Option<(usize, DType)>,
    ) -> Result<Self, String> {
        let rows = store.n_rows();
        let mut times = Vec::with_capacity(rows);
        for seg in segs {
            times.extend(segment_i64(store, seg, time_col, time_dtype)?);
        }
        let mut sym_key = vec![0u32; rows];
        for seg in segs {
            for r in seg.start..seg.start + seg.len {
                sym_key[r] = seg.key;
            }
        }
        let mut member = vec![true; rows];
        if let Some((col, dtype)) = member_col {
            let mut vals = Vec::with_capacity(rows);
            for seg in segs {
                if dtype == DType::I64 {
                    vals.extend(segment_i64(store, seg, col, dtype)?.into_iter().map(|v| v as f64));
                } else {
                    vals.extend(segment_column(store, seg, col)?);
                }
            }
            for r in 0..rows {
                member[r] = vals[r].is_finite() && vals[r] <= times[r] as f64;
            }
        }
        Ok(CrossPanel {
            times,
            sym_key,
            member,
            grouped: std::sync::OnceLock::new(),
        })
    }

    fn groups(&self) -> impl Iterator<Item = &[u32]> {
        let (order, starts) = self.grouped.get_or_init(|| {
            let (times, sym_key) = (&self.times, &self.sym_key);
            let mut order: Vec<u32> = (0..times.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                let (a, b) = (a as usize, b as usize);
                times[a]
                    .cmp(&times[b])
                    .then(sym_key[a].cmp(&sym_key[b]))
                    .then(a.cmp(&b))
            });
            let mut starts = Vec::new();
            let mut i = 0usize;
            while i < order.len() {
                starts.push(i);
                let t = times[order[i] as usize];
                while i < order.len() && times[order[i] as usize] == t {
                    i += 1;
                }
            }
            starts.push(order.len());
            (order, starts)
        });
        starts.windows(2).map(move |w| &order[w[0]..w[1]])
    }
}

fn write_cross_output(
    temp: &mut BTreeMap<String, TempColumn>,
    requested: &[String],
    names: &[&str],
    values: &[f64],
) -> Result<(), String> {
    for name in names {
        if !requested.iter().any(|n| n == name) {
            continue;
        }
        let t = temp
            .get_mut(*name)
            .ok_or_else(|| format!("missing temporary column '{}'", name))?;
        t.reset()?;
        t.write_values(values)?;
    }
    Ok(())
}

fn source_values(
    temp: &BTreeMap<String, TempColumn>,
    name: &str,
    rows: usize,
) -> Result<Vec<f64>, String> {
    let col = temp
        .get(name)
        .ok_or_else(|| format!("cross-sectional source '{}' was not produced", name))?;
    read_f64_file(&col.path, rows)
}

/// True when the market-state pass owns any requested column.
fn wants_market_state(requested: &[String]) -> bool {
    requested.iter().any(|n| {
        CROSS_MARKET_NAMES.contains(&n.as_str()) || MARKET_STATE_NAMES.contains(&n.as_str())
    })
}

/// Whether the per-timestamp market means run on the device (#299).
#[cfg(feature = "cuda")]
fn market_device_enabled(mode: GpuMode, rows: usize) -> bool {
    match mode {
        GpuMode::Off => false,
        GpuMode::On => true,
        GpuMode::Auto => rows >= GPU_MIN_ROWS,
    }
}

/// Compute equal-weight market state, breadth, dispersion, relative-to-market
/// and the average-correlation proxy from the point-in-time cross-sections.
///
/// With a device (#299) the per-timestamp reductions run there in the same
/// `(time, symbol, row)` addition order; any device failure falls back to the
/// CPU loop for that block. Returns the columns the device produced.
#[allow(clippy::too_many_arguments)]
fn market_state_pass(
    panel: &CrossPanel,
    temp: &mut BTreeMap<String, TempColumn>,
    requested: &[String],
    segs: &[Segment],
    store: &FileStore,
    cols: &Cols,
    cross: &Cross,
    step: i64,
    mut dev: Option<&mut dyn crate::bench_gpu::MarketDevice>,
) -> Result<BTreeSet<String>, String> {
    use crate::bench_gpu::{WANT_BREADTH, WANT_DISPERSION, WANT_MEAN, WANT_REL};
    let rows = panel.times.len();
    let wants = |n: &str| requested.iter().any(|x| x == n);
    let mut done: BTreeSet<String> = BTreeSet::new();
    let wants_r16 = wants("mkt_r16")
        || wants("mkt_ew_ret16")
        || wants("breadth")
        || wants("mkt_breadth")
        || wants("dispersion")
        || wants("mkt_dispersion")
        || wants("rel_mkt16");
    let wants_r96 = wants("mkt_r96") || wants("rel_mkt96");
    let wants_vol96 = wants("mkt_vol96");
    let wants_corr = wants("avg_corr");
    let wants_r1 = wants("mkt_ew_ret1") || wants_corr;

    if wants_r16 {
        let src = source_values(temp, "r16", rows)?;
        let mut mask = WANT_MEAN;
        if wants("breadth") || wants("mkt_breadth") {
            mask |= WANT_BREADTH;
        }
        if wants("dispersion") || wants("mkt_dispersion") {
            mask |= WANT_DISPERSION;
        }
        if wants("rel_mkt16") {
            mask |= WANT_REL;
        }
        let device = dev.as_mut().and_then(|d| d.group_stats(&src, mask));
        let on_device = device.is_some();
        let (mkt, breadth, dispersion, rel) = if let Some(st) = device {
            (st.mean, st.breadth, st.dispersion, st.rel)
        } else {
        let mut mkt = vec![f64::NAN; rows];
        let mut breadth = vec![f64::NAN; rows];
        let mut dispersion = vec![f64::NAN; rows];
        let mut rel = vec![f64::NAN; rows];
        for group in panel.groups() {
            let mut vals = Vec::new();
            let mut sum = 0.0;
            for &r in group {
                let r = r as usize;
                if panel.member[r] && src[r].is_finite() {
                    vals.push(r);
                    sum += src[r];
                }
            }
            let n = vals.len();
            if n < MIN_CROSS_MEMBERS {
                continue;
            }
            let nf = n as f64;
            let mean = sum / nf;
            let mut ss = 0.0;
            let mut positive = 0usize;
            for &r in &vals {
                let d = src[r] - mean;
                ss += d * d;
                if src[r] > 0.0 {
                    positive += 1;
                }
            }
            let sd = (ss / nf).max(0.0).sqrt();
            let br = positive as f64 / nf;
            for &r in group {
                let r = r as usize;
                if !panel.member[r] {
                    continue;
                }
                mkt[r] = mean;
                breadth[r] = br;
                dispersion[r] = sd;
                if src[r].is_finite() && n > 1 {
                    rel[r] = src[r] - (sum - src[r]) / (n - 1) as f64;
                }
            }
        }
        (mkt, breadth, dispersion, rel)
        };
        if on_device {
            for n in [
                "mkt_r16", "mkt_ew_ret16", "breadth", "mkt_breadth", "dispersion",
                "mkt_dispersion", "rel_mkt16",
            ] {
                if wants(n) {
                    done.insert(n.to_string());
                }
            }
        }
        write_cross_output(temp, requested, &["mkt_r16", "mkt_ew_ret16"], &mkt)?;
        write_cross_output(temp, requested, &["breadth", "mkt_breadth"], &breadth)?;
        write_cross_output(temp, requested, &["dispersion", "mkt_dispersion"], &dispersion)?;
        write_cross_output(temp, requested, &["rel_mkt16"], &rel)?;
    }

    if wants_r96 {
        let src = source_values(temp, "r96", rows)?;
        let mask = WANT_MEAN | if wants("rel_mkt96") { WANT_REL } else { 0 };
        let device = dev.as_mut().and_then(|d| d.group_stats(&src, mask));
        let on_device = device.is_some();
        let (mkt, rel) = if let Some(st) = device {
            (st.mean, st.rel)
        } else {
        let mut mkt = vec![f64::NAN; rows];
        let mut rel = vec![f64::NAN; rows];
        for group in panel.groups() {
            let mut vals = Vec::new();
            let mut sum = 0.0;
            for &r in group {
                let r = r as usize;
                if panel.member[r] && src[r].is_finite() {
                    vals.push(r);
                    sum += src[r];
                }
            }
            let n = vals.len();
            if n < MIN_CROSS_MEMBERS {
                continue;
            }
            let mean = sum / n as f64;
            for &r in group {
                let r = r as usize;
                if !panel.member[r] {
                    continue;
                }
                mkt[r] = mean;
                if src[r].is_finite() && n > 1 {
                    rel[r] = src[r] - (sum - src[r]) / (n - 1) as f64;
                }
            }
        }
        (mkt, rel)
        };
        if on_device {
            for n in ["mkt_r96", "rel_mkt96"] {
                if wants(n) {
                    done.insert(n.to_string());
                }
            }
        }
        write_cross_output(temp, requested, &["mkt_r96"], &mkt)?;
        write_cross_output(temp, requested, &["rel_mkt96"], &rel)?;
    }

    if wants_vol96 {
        let src = source_values(temp, "vol96", rows)?;
        let device = dev.as_mut().and_then(|d| d.group_stats(&src, WANT_MEAN));
        let on_device = device.is_some();
        let mkt = if let Some(st) = device {
            st.mean
        } else {
        let mut mkt = vec![f64::NAN; rows];
        for group in panel.groups() {
            let mut vals = Vec::new();
            let mut sum = 0.0;
            for &r in group {
                let r = r as usize;
                if panel.member[r] && src[r].is_finite() {
                    vals.push(r);
                    sum += src[r];
                }
            }
            if vals.len() < MIN_CROSS_MEMBERS {
                continue;
            }
            let mean = sum / vals.len() as f64;
            for &r in group {
                let r = r as usize;
                if panel.member[r] {
                    mkt[r] = mean;
                }
            }
        }
        mkt
        };
        if on_device {
            done.insert("mkt_vol96".to_string());
        }
        write_cross_output(temp, requested, &["mkt_vol96"], &mkt)?;
    }

    if wants_r1 {
        let r1 = source_values(temp, "r1", rows)?;
        // The CPU equal-weight mean per timestamp, also the `by_time` lookup the
        // CPU correlation proxy needs.
        let cpu_means = || {
            let mut mkt = vec![f64::NAN; rows];
            let mut by_time = BTreeMap::new();
            for group in panel.groups() {
                let mut vals = Vec::new();
                let mut sum = 0.0;
                for &r in group {
                    let r = r as usize;
                    if panel.member[r] && r1[r].is_finite() {
                        vals.push(r);
                        sum += r1[r];
                    }
                }
                if vals.len() < MIN_CROSS_MEMBERS {
                    continue;
                }
                let mean = sum / vals.len() as f64;
                let t = panel.times[group[0] as usize];
                by_time.insert(t, mean);
                for &r in group {
                    let r = r as usize;
                    if panel.member[r] {
                        mkt[r] = mean;
                    }
                }
            }
            (mkt, by_time)
        };
        let mut cpu_by_time = None;
        if wants("mkt_ew_ret1") {
            let device = dev.as_mut().and_then(|d| d.group_stats(&r1, WANT_MEAN));
            let mkt = match device {
                Some(st) => {
                    done.insert("mkt_ew_ret1".to_string());
                    st.mean
                }
                None => {
                    let (mkt, by_time) = cpu_means();
                    cpu_by_time = Some(by_time);
                    mkt
                }
            };
            write_cross_output(temp, requested, &["mkt_ew_ret1"], &mkt)?;
        }

        if wants_corr {
            if let Some(avg) = dev.as_mut().and_then(|d| d.avg_corr(&r1)) {
                done.insert("avg_corr".to_string());
                write_cross_output(temp, requested, &["avg_corr"], &avg)?;
            } else {
            let by_time = match cpu_by_time.take() {
                Some(b) => b,
                None => cpu_means().1,
            };
            // Average correlation proxy: mean over members of the trailing
            // 96-bar correlation between the member's one-bar return and the
            // equal-weight market return. O(rows) via per-symbol prefixes.
            let mut member_corr = vec![f64::NAN; rows];
            for seg in segs {
                let nt = seg.len;
                if nt < 96 {
                    continue;
                }
                let mut px = vec![0.0f64; nt + 1];
                let mut py = vec![0.0f64; nt + 1];
                let mut pxx = vec![0.0f64; nt + 1];
                let mut pyy = vec![0.0f64; nt + 1];
                let mut pxy = vec![0.0f64; nt + 1];
                let mut pc = vec![0usize; nt + 1];
                for j in 0..nt {
                    let g = seg.start + j;
                    let x = r1[g];
                    let y = if panel.member[g] {
                        Cross::at(&by_time, panel.times[g])
                    } else {
                        f64::NAN
                    };
                    let ok = x.is_finite() && y.is_finite();
                    px[j + 1] = px[j] + if ok { x } else { 0.0 };
                    py[j + 1] = py[j] + if ok { y } else { 0.0 };
                    pxx[j + 1] = pxx[j] + if ok { x * x } else { 0.0 };
                    pyy[j + 1] = pyy[j] + if ok { y * y } else { 0.0 };
                    pxy[j + 1] = pxy[j] + if ok { x * y } else { 0.0 };
                    pc[j + 1] = pc[j] + if ok { 1 } else { 0 };
                }
                let seg_times = &panel.times[seg.start..seg.start + nt];
                for i in 0..nt {
                    if !contig(seg_times, i, 96, step) {
                        continue;
                    }
                    let lo = i + 1 - 96;
                    if pc[i + 1] - pc[lo] != 96 {
                        continue;
                    }
                    let w = 96.0;
                    let sx = px[i + 1] - px[lo];
                    let sy = py[i + 1] - py[lo];
                    let sxx = pxx[i + 1] - pxx[lo];
                    let syy = pyy[i + 1] - pyy[lo];
                    let sxy = pxy[i + 1] - pxy[lo];
                    let vx = sxx - sx * sx / w;
                    let vy = syy - sy * sy / w;
                    if vx > 0.0 && vy > 0.0 {
                        member_corr[seg.start + i] =
                            ((sxy - sx * sy / w) / (vx * vy).sqrt()).max(-1.0).min(1.0);
                    }
                }
            }
            let mut avg = vec![f64::NAN; rows];
            for group in panel.groups() {
                let mut vals = Vec::new();
                let mut sum = 0.0;
                for &r in group {
                    let r = r as usize;
                    if panel.member[r] && member_corr[r].is_finite() {
                        vals.push(r);
                        sum += member_corr[r];
                    }
                }
                if vals.len() < MIN_CROSS_MEMBERS {
                    continue;
                }
                let mean = sum / vals.len() as f64;
                for &r in group {
                    let r = r as usize;
                    if panel.member[r] {
                        avg[r] = mean;
                    }
                }
            }
            write_cross_output(temp, requested, &["avg_corr"], &avg)?;
            }
        }
    }

    if wants("btc_volume_share") {
        let vol = match cols.volume {
            Some(col) => {
                let mut vals = Vec::with_capacity(rows);
                for seg in segs {
                    vals.extend(segment_column(store, seg, col)?);
                }
                vals
            }
            None => {
                return Err(format!(
                    "feature 'btc_volume_share' needs the raw 'volume' column"
                ))
            }
        };
        let bench = cross.bench_series();
        let device = dev.as_mut().and_then(|d| d.volume_share(&vol, &bench));
        let on_device = device.is_some();
        let share = if let Some(share) = device {
            share
        } else {
        let mut share = vec![f64::NAN; rows];
        for group in panel.groups() {
            let mut total = 0.0;
            let mut count = 0usize;
            for &r in group {
                let r = r as usize;
                if panel.member[r] && vol[r].is_finite() {
                    total += vol[r];
                    count += 1;
                }
            }
            if count < MIN_CROSS_MEMBERS || !(total > 0.0) {
                continue;
            }
            let t = panel.times[group[0] as usize];
            let btc = Cross::at(&cross.volume, t);
            if !btc.is_finite() {
                continue;
            }
            let v = btc / total;
            for &r in group {
                let r = r as usize;
                if panel.member[r] {
                    share[r] = v;
                }
            }
        }
        share
        };
        if on_device {
            done.insert("btc_volume_share".to_string());
        }
        write_cross_output(temp, requested, &["btc_volume_share"], &share)?;
    }

    Ok(done)
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Dispatch one registered catalogue name to its CUDA kernel.
#[cfg(feature = "cuda")]
fn gpu_feature(name: &str, gpu: &mut crate::enrich_gpu::EnrichGpu) -> Option<Vec<f64>> {
    if name == "ret1" {
        return gpu.ret(1);
    }
    if name == "ret5" {
        return gpu.ret(5);
    }
    if name == "ma_slope672" {
        return gpu.ma_slope(672, 672);
    }
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        return gpu.ret(k);
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        return gpu.ma(w, ma_window_policy(name));
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        return gpu.vol(w);
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        return gpu.lag(k);
    }
    if name == "don96" {
        return gpu.donchian_centred(96);
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        return gpu.donchian(w);
    }
    match name {
        "bb20_bw" => gpu.bb_width(20),
        "bb50_bw" => gpu.bb_width(50),
        "bb_width96" => gpu.bb_width(96),
        "bb20_z" => gpu.bb_z(20, 0),
        "bb50" => gpu.bb_z(50, 1),
        "bb96" => gpu.bb_z(96, 1),
        "vz96" | "vol_z96" => gpu.vz(96),
        "range1" => gpu.range(),
        "clv1" => gpu.clv(),
        "dow_sin" => gpu.calendar(0),
        "dow_cos" => gpu.calendar(1),
        "hour_sin" => gpu.calendar(2),
        "hour_cos" => gpu.calendar(3),
        "month_sin" => gpu.calendar(4),
        "month_cos" => gpu.calendar(5),
        "dom_sin" => gpu.calendar(6),
        "dom_cos" => gpu.calendar(7),
        "tod_sin" => gpu.tod(true),
        "tod_cos" => gpu.tod(false),
        "ema8" => gpu.ema(8),
        "ema20" => gpu.ema(20),
        "ema50" => gpu.ema(50),
        "ema100" => gpu.ema(100),
        "ema200" => gpu.ema(200),
        "rsi6" => gpu.rsi_py(6),
        "rsi28" => gpu.rsi_py(28),
        "rsi14" => gpu.rsi_py(14),
        "atr14" => gpu.atr(14, 3),
        "natr14" => gpu.atr(14, 1),
        "atr50" => gpu.atr(50, 2),
        "batr" => gpu.batr(),
        "gapatr" => gpu.gapatr(),
        "volrat" => gpu.volrat(),
        "stoch_k" => gpu.stoch(0),
        "stoch_d" => gpu.stoch(1),
        "adx14" => gpu.adx(0),
        "di_plus" => gpu.adx(1),
        "di_minus" => gpu.adx(2),
        "di_diff14" => gpu.adx(3),
        "rmean5" => gpu.exotic(0),
        "rstd5" => gpu.exotic(1),
        "close_pos20" => gpu.exotic(2),
        "ou_hl96" => gpu.exotic(3),
        "acf1" => gpu.exotic(4),
        "vr4" => gpu.exotic(5),
        "rskew96" => gpu.exotic(6),
        "jump_share" => gpu.exotic(7),
        "semivar_asym" => gpu.exotic(8),
        "cusum_abs_z" => gpu.exotic(9),
        _ => None,
    }
}

/// NaN policy of the device `maN` mean: `ma96` and `ma672` average the finite
/// closes of the window like the compact reference; the catalogue `maN`
/// columns need a full window.
#[cfg(feature = "cuda")]
fn ma_window_policy(name: &str) -> i32 {
    if matches!(name, "ma96" | "ma672") {
        window_stats::MEAN_FINITE
    } else {
        window_stats::MEAN_STRICT
    }
}

/// #300: the same dispatch as [`gpu_feature`], as a descriptor for the batched
/// output arena. `None` means the name has no single-slot batch kernel.
#[cfg(feature = "cuda")]
fn gpu_batch_op(name: &str) -> Option<crate::enrich_gpu::BatchOp> {
    use crate::enrich_gpu::BatchOp;
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        return Some(BatchOp::ret(k));
    }
    if let Some(w) = name.strip_prefix("ma").and_then(|s| s.parse::<usize>().ok()) {
        return Some(BatchOp::ma(w, ma_window_policy(name)));
    }
    if let Some(w) = name.strip_prefix("vol").and_then(|s| s.parse::<usize>().ok()) {
        return Some(BatchOp::vol(w));
    }
    if let Some(k) = name.strip_prefix("lag_r1_").and_then(|s| s.parse::<usize>().ok()) {
        return Some(BatchOp::lag(k));
    }
    if name == "don96" {
        // The centred compact form is not the generic donchian batch op.
        return None;
    }
    if let Some(w) = name.strip_prefix("don").and_then(|s| s.parse::<usize>().ok()) {
        return Some(BatchOp::donchian(w));
    }
    Some(match name {
        "bb20_bw" => BatchOp::bb_width(20),
        "bb50_bw" => BatchOp::bb_width(50),
        "vz96" => BatchOp::vz(96),
        "range1" => BatchOp::range(),
        "clv1" => BatchOp::clv(),
        "dow_sin" => BatchOp::calendar(0),
        "dow_cos" => BatchOp::calendar(1),
        "hour_sin" => BatchOp::calendar(2),
        "hour_cos" => BatchOp::calendar(3),
        "month_sin" => BatchOp::calendar(4),
        "month_cos" => BatchOp::calendar(5),
        "dom_sin" => BatchOp::calendar(6),
        "dom_cos" => BatchOp::calendar(7),
        "ema8" => BatchOp::ema(8),
        "ema20" => BatchOp::ema(20),
        "ema50" => BatchOp::ema(50),
        "ema100" => BatchOp::ema(100),
        "ema200" => BatchOp::ema(200),
        "rsi6" => BatchOp::rsi(6),
        "rsi28" => BatchOp::rsi(28),
        "atr50" => BatchOp::atr(50),
        _ => return None,
    })
}

/// Run an enrichment into a temporary output file.
pub fn build(opts: &Options) -> Result<Report, String> {
    enrich(opts)
}

/// Run an enrichment into a temporary output file.
fn enrich(opts: &Options) -> Result<Report, String> {
    let started = Instant::now();
    let data_is_store = Path::new(&opts.data)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("rcsb"))
        .unwrap_or(false)
        || store::is_store_path(&opts.data);
    let out = opts
        .out
        .clone()
        .unwrap_or_else(|| format!("{}.enriched.rcsb", opts.data));
    let work_dir = temp_dir_for(&out);
    std::fs::create_dir_all(&work_dir)
        .map_err(|e| format!("{}: {}", work_dir.display(), e))?;

    let result = enrich_inner(opts, &out, &work_dir, data_is_store, started);
    let _ = std::fs::remove_dir_all(&work_dir);
    result
}

fn temp_dir_for(out: &str) -> PathBuf {
    let pid = std::process::id();
    PathBuf::from(format!("{}.enrich-tmp-{}", out, pid))
}

fn enrich_inner(
    opts: &Options,
    out: &str,
    work_dir: &Path,
    data_is_store: bool,
    started: Instant,
) -> Result<Report, String> {
    let t_read = Instant::now();
    // A CSV is loaded resident once, then handed to the same column-store path.
    // The store path stays out-of-core; this is the documented CSV exception.
    let csv_store;
    let input_path: String = if data_is_store {
        opts.data.clone()
    } else {
        let table = crate::data::read_table_with_time(&opts.data, false, opts.time_col.as_deref())?;
        let time_idx = opts
            .time_col
            .as_deref()
            .and_then(|n| table.names.iter().position(|x| x == n))
            .or_else(|| {
                table
                    .names
                    .iter()
                    .position(|n| ["ts", "time", "timestamp", "date", "datetime"].contains(&n.to_ascii_lowercase().as_str()))
            });
        csv_store = work_dir.join("input.rcsb").to_string_lossy().into_owned();
        let group_rows = if opts.group_rows > 0 {
            opts.group_rows
        } else {
            store::DEFAULT_GROUP_ROWS
        };
        store::write_store_with_time(&csv_store, &table, group_rows, time_idx)?;
        csv_store.clone()
    };
    let store = FileStore::open(&input_path)?;
    let names = store.header().names();
    let time_col = find_time_col(&names, opts.time_col.as_deref())
        .ok_or_else(|| format!("{}: no time column", input_path))?;
    let symbol_col = find_symbol_col(&names, opts.symbol_col.as_deref());
    if let Some(s) = opts.symbol_col.as_deref() {
        if symbol_col.is_none() {
            return Err(format!("--symbol column '{}' is not in the store", s));
        }
    }
    let cols = Cols {
        time: time_col,
        time_dtype: store.header().dtypes()[time_col],
        open: find_col(&names, &["open", "o"]).ok_or_else(|| format!("{}: no open column", input_path))?,
        high: find_col(&names, &["high", "h"]).ok_or_else(|| format!("{}: no high column", input_path))?,
        low: find_col(&names, &["low", "l"]).ok_or_else(|| format!("{}: no low column", input_path))?,
        close: find_col(&names, &["close", "c"]).ok_or_else(|| format!("{}: no close column", input_path))?,
        volume: find_col(&names, &["volume", "vol", "v"]),
        quote_volume: find_col(&names, &["quote_volume", "quotevolume", "qv"]),
        trades: find_col(&names, &["trades", "trade_count", "trades_count", "n"]),
        taker_buy_base: find_col(
            &names,
            &["taker_buy_base", "taker_buy", "takerbuybase", "taker_buy_volume", "tb"],
        ),
        taker_buy_quote: find_col(
            &names,
            &["taker_buy_quote", "takerbuyquote", "tbq"],
        ),
        symbol: symbol_col,
    };
    let (segs, inferred_step) = segments(&store, time_col, symbol_col)?;
    let step = opts.bar_seconds.unwrap_or(inferred_step);
    if step <= 0 {
        return Err("could not infer the bar spacing; pass --bar-seconds".to_string());
    }
    let tf = tf_for_step(step);
    let read_seconds = t_read.elapsed().as_secs_f64();

    // Expand the catalogue and decide which names this module computes and
    // which #125 computes.
    let mut requested: Vec<String> = Vec::new();
    for f in &opts.features {
        for n in expand(f)? {
            if !requested.contains(&n) {
                requested.push(n);
            }
        }
    }
    if requested.is_empty() && opts.labels.is_empty() {
        return Err("enrich needs --features and/or --label".to_string());
    }
    // The target spec names `vol<W>` as the scale column, so the store carries it.
    for src in opts.labels.iter().filter(|l| l.trim_start().starts_with("long:")) {
        let scale = format!("vol{}", crate::labels::LongLabelSpec::declared_sigma_window(src));
        if !names.contains(&scale) && !requested.contains(&scale) {
            if expand(&scale).is_err() {
                return Err(format!("label scale column '{}' is not an enrich feature; use vol=5, 20, 60 or 120, or add the column to the store", scale));
            }
            requested.push(scale);
        }
    }
    let compact_profile = opts
        .features
        .iter()
        .any(|f| f.split(',').any(|x| x.trim() == "compact"));
    let mut legacy = Vec::new();
    let mut own = Vec::new();
    let mut py15m_requested = Vec::new();
    let mut compact_requested = Vec::new();
    let mut flow_requested = Vec::new();
    let mut price_action_requested = Vec::new();
    let mut mtf_requested = Vec::new();
    for n in &requested {
        if FLOW_NAMES.contains(&n.as_str()) {
            // One canonical implementation for the flow names shared with the
            // compact and py15m groups.
            flow_requested.push(n.clone());
        } else if PRICE_ACTION_NAMES.contains(&n.as_str()) {
            price_action_requested.push(n.clone());
        } else if is_compact(n) && (compact_profile || needs_compact(n)) {
            compact_requested.push(n.clone());
        } else if matches!(
            n.as_str(),
            "bb20_bw" | "month_sin" | "month_cos" | "dom_sin" | "dom_cos"
        ) {
            // These names are shared by a standalone group (bollinger/calendar)
            // and the py15m family. Compute the cheap standalone form unless
            // the user asked for the whole py15m family.
            if !requested.iter().any(|x| x == "py15m") {
                own.push(n.clone());
            } else if py15m_names().contains(n) {
                py15m_requested.push(n.clone());
            }
        } else if py15m_names().contains(n) {
            py15m_requested.push(n.clone());
        } else if n.starts_with("xs_rank_") || n == "cs_rank_r60" {
            // Computed in the cross-sectional rank pass, which needs every
            // symbol and point-in-time membership.
        } else if (CROSS_MARKET_NAMES.contains(&n.as_str())
            || MARKET_STATE_NAMES.contains(&n.as_str()))
            && !py15m_names().contains(n)
        {
            // Computed in the whole-panel market-state pass. The five py15m
            // names are still produced by `py15m_all` and overwritten by the
            // canonical #273 pass below, so they are tracked there.
        } else if is_mtf(n) {
            // Computed by the per-symbol resample + causal join pass.
            mtf_requested.push(n.clone());
        } else if is_own(n) {
            own.push(n.clone());
        } else {
            legacy.push(n.clone());
        }
    }
    // `cross:btc` legacy names are implemented here (they need the same BTC
    // context as the new `btc` family).
    for n in ["cross_ret1", "cross_beta20"] {
        if legacy.contains(&n.to_string()) {
            legacy.retain(|x| x != n);
            own.push(n.to_string());
        }
    }

    // The flow family and the compact flow names read raw exchange fields. A
    // missing raw column is a hard input error naming both the feature and the
    // column, never a zero column.
    let col_present = |col: &str| match col {
        "volume" => cols.volume.is_some(),
        "quote_volume" => cols.quote_volume.is_some(),
        "trades" => cols.trades.is_some(),
        "taker_buy_base" => cols.taker_buy_base.is_some(),
        _ => true,
    };
    for name in &requested {
        for req in flow_requires(name) {
            if !col_present(req) {
                return Err(format!(
                    "{}: feature '{}' needs the raw '{}' column",
                    input_path, name, req
                ));
            }
        }
        // A cross-column source may itself need a raw field (amihud96 needs
        // quote_volume). Fail closed with the consumer and the field named.
        if let Some(src) = cross_source(name) {
            for req in flow_requires(src) {
                if !col_present(req) {
                    return Err(format!(
                        "{}: feature '{}' needs the raw '{}' column (source '{}')",
                        input_path, name, req, src
                    ));
                }
            }
        }
    }
    let compact_flow_names = ["cmf20", "mfi14", "vol_z96"];
    let wants_flow = compact_requested
        .iter()
        .any(|n| compact_flow_names.contains(&n.as_str()));
    if wants_flow && cols.volume.is_none() {
        return Err(format!(
            "{}: compact features {:?} need the raw 'volume' column",
            input_path,
            compact_requested
                .iter()
                .filter(|n| compact_flow_names.contains(&n.as_str()))
                .collect::<Vec<_>>()
        ));
    }

    // Labels.
    let mut long_specs: Vec<crate::labels::LongLabelSpec> = Vec::new();
    let mut labels: Vec<(String, LabelSpec)> = Vec::new();
    for spec_src in &opts.labels {
        if spec_src.trim_start().starts_with("long:") {
            let spec = crate::labels::LongLabelSpec::parse(spec_src, step)?;
            spec.check_overlap(opts.allow_overlap)?;
            if long_specs.iter().any(|x| x.name == spec.name) {
                return Err(format!("label name '{}' is used twice; set name=<id>", spec.name));
            }
            long_specs.push(spec);
            continue;
        }
        let tf = tf.ok_or_else(|| {
            format!(
                "label '{}' needs a recognised bar spacing (got {}s); pass --bar-seconds",
                spec_src, step
            )
        })?;
        let spec = LabelSpec::parse(spec_src, tf)?;
        spec.check_overlap(tf, opts.allow_overlap)?;
        let base = format!("fwd{}", spec.horizon);
        let mut name = base.clone();
        let mut k = 2;
        while labels.iter().any(|(n, _)| n == &name) || requested.contains(&name) {
            name = format!("{}_{}", base, k);
            k += 1;
        }
        labels.push((name, spec));
    }

    // Cross-market context: the benchmark symbols' own series, keyed by exact
    // timestamp. ETH is loaded only when an ETH column is requested.
    let cross = load_benchmark(
        &store,
        &segs,
        &cols,
        step,
        &opts.cross_symbol,
        opts.cross_data.as_deref(),
    )?;
    let needs_eth = requested
        .iter()
        .any(|n| n.starts_with("eth_") || n.starts_with("rel_eth") || n.starts_with("beta_eth"));
    let eth = if needs_eth {
        load_benchmark(
            &store,
            &segs,
            &cols,
            step,
            &opts.eth_symbol,
            opts.eth_data.as_deref(),
        )?
    } else {
        Cross::default()
    };

    let needs_cross = requested.iter().any(|n| {
        n.starts_with("btc")
            || n.starts_with("rel_btc")
            || n.starts_with("beta_btc")
            || n.starts_with("cross_")
            || n == "rel_r16"
            || n == "rel_r96"
            || n == "beta384"
            || n == "btc_volume_share"
    });
    if needs_cross && cross.symbol.is_empty() {
        return Err(format!(
            "btc/cross features need the benchmark series '{}' in --data or a --cross-data store",
            opts.cross_symbol
        ));
    }
    if needs_eth && eth.symbol.is_empty() {
        return Err(format!(
            "eth/cross features need the ETH series '{}' in --data or an --eth-data store",
            opts.eth_symbol
        ));
    }

    // GPU plan.
    let gpu = crate::backend::probe();
    let (gpu_available, gpu_device) = (gpu.available(), gpu.device.clone());
    let mut fam_counts: BTreeMap<&'static str, (usize, OpClass)> = BTreeMap::new();
    let classify = |name: &str| -> OpClass {
        if name.starts_with("xs_") {
            OpClass::Rank
        } else if crate::features_gpu::is_device_flow(name) {
            OpClass::RollingStd
        } else if crate::features_gpu::is_device_price_action(name) {
            OpClass::WilderScan
        } else if FLOW_NAMES.contains(&name) {
            OpClass::RollingStd
        } else if name.starts_with("vol")
            || name.starts_with("bb")
            || name == "vz96"
            || name.starts_with("macd")
            || name.starts_with("skew")
            || name.starts_with("kurt")
            || name.starts_with("park")
            || name.starts_with("gk")
        {
            OpClass::RollingStd
        } else if name.starts_with("dow_")
            || name.starts_with("hour_")
            || name.starts_with("month_")
            || name.starts_with("dom_")
            || name.starts_with("tod_")
        {
            OpClass::Calendar
        } else if name.starts_with("don") {
            OpClass::RollingExtrema
        } else if name.starts_with("ma") || name.starts_with("ema") {
            OpClass::RollingSum
        } else if name.starts_with("rsi")
            || name.starts_with("atr")
            || name.starts_with("natr")
            || name.starts_with("adx")
            || name.starts_with("di_")
            || name == "stoch_k"
            || name == "stoch_d"
        {
            OpClass::WilderScan
        } else if name.starts_with("btc")
            || name.starts_with("rel_btc")
            || name.starts_with("beta_btc")
            || name.starts_with("eth_")
            || name.starts_with("rel_eth")
            || name.starts_with("beta_eth")
            || name.starts_with("mkt_")
            || name.starts_with("cross_")
        {
            OpClass::Join
        } else if is_mtf(name) {
            // Higher-timeframe resample + monotone causal join.
            OpClass::Join
        } else {
            OpClass::Elementwise
        }
    };
    let mut gpu_cols: BTreeMap<&'static str, usize> = BTreeMap::new();
    for n in requested.iter() {
        let fam = family_of(n);
        let e = fam_counts.entry(fam).or_insert((0, classify(n)));
        e.0 += 1;
        if gpu_supports(n) {
            *gpu_cols.entry(fam).or_insert(0) += 1;
        }
    }
    if !labels.is_empty() {
        let e = fam_counts.entry("labels").or_insert((0, OpClass::Elementwise));
        e.0 += labels.len();
        if gpu_kernels_compiled() {
            *gpu_cols.entry("labels").or_insert(0) += labels.len();
        }
    }
    let write_bw = opts.write_bw.unwrap_or(GPU_WRITE_BYTES_PER_SECOND);
    let mut plans = plan_groups_full(
        &fam_counts,
        &gpu_cols,
        store.n_rows(),
        segs.len(),
        opts.gpu,
        gpu_available,
        write_bw,
    );
    // #296: the 1.05M paired fixture measured the device MTF phase slower than
    // the CPU reference (5.08 s vs 6.45 s on two symbol segments). Until a
    // 16M paired run establishes the crossover, keep the new device path
    // opt-in with `--gpu on`; `auto` remains the CPU reference.
    if opts.gpu != GpuMode::On {
        if let Some(plan) = plans.iter_mut().find(|p| p.group == "mtf" && p.backend == "gpu") {
            plan.backend = "cpu";
            plan.gpu_cols = 0;
            plan.cpu_cols = plan.cols;
            plan.reason =
                "MTF device crossover not measured at 16M rows; use --gpu on".to_string();
        }
    }

    // Input checksum (the same file is copied, not rebuilt).
    let input_sha = manifest::sha256_file(&opts.data)
        .or_else(|_| manifest::sha256_file(&input_path))
        .unwrap_or_else(|_| "unknown".to_string());

    // Lockbox guard for labelled builds: a label is an evaluation target.
    if !labels.is_empty() || !long_specs.is_empty() {
        for seg in &segs {
            let times = segment_i64(&store, seg, time_col, cols.time_dtype)?;
            crate::prep::check_labelled_rows(&times, opts.allow_lockbox)?;
        }
    }

    // Create one temp column per requested feature and label.
    let mut temp: BTreeMap<String, TempColumn> = BTreeMap::new();
    for n in &requested {
        temp.insert(
            n.clone(),
            TempColumn::create(work_dir, n, nominal_warmup(n), false)?,
        );
    }
    for (n, _) in &labels {
        temp.insert(n.clone(), TempColumn::create(work_dir, n, 0, true)?);
    }
    let long_names: Vec<String> = long_specs
        .iter()
        .flat_map(|sp| crate::labels::COLUMN_KINDS.iter().map(move |(k, _)| sp.column(k)))
        .collect();
    for sp in &long_specs {
        for (kind, _) in crate::labels::COLUMN_KINDS {
            let n = sp.column(kind);
            temp.insert(n.clone(), TempColumn::create(work_dir, &n, sp.column_warmup(kind), true)?);
        }
    }
    let mut long_acc: Vec<crate::labels::LabelAccumulator> =
        long_specs.iter().map(crate::labels::LabelAccumulator::new).collect();

    // Cross-sectional source columns are computed even when the user did not
    // ask for them, then removed if they are not part of the output. The same
    // set feeds the market-state pass and the rank pass.
    let member_col = find_col(&names, &["member_asof"]);
    let member_dtype = member_col.map(|c| store.header().dtypes()[c]);
    let mut cross_sources: BTreeSet<String> = BTreeSet::new();
    for n in &requested {
        if let Some(src) = cross_source(n) {
            cross_sources.insert(src.to_string());
            if src != "volume" && !requested.contains(&src.to_string()) {
                temp.entry(src.to_string())
                    .or_insert(TempColumn::create(work_dir, src, nominal_warmup(src), false)?);
            }
        }
    }
    for src in &cross_sources {
        if src != "volume" && !requested.contains(src) && !own.contains(src) {
            own.push(src.clone());
        }
    }

    // The feature phase includes both the GPU kernel pass and the CPU fallback
    // pass; the cross-sectional pass is timed separately below.
    let t_feat = Instant::now();
    // Host-thread partition of the feature phase (#300): disjoint intervals on this thread.
    #[allow(unused_mut, unused_variables)]
    let mut feature_read_seconds = 0.0f64;
    #[allow(unused_mut, unused_variables)]
    let mut feature_context_seconds = 0.0f64;
    #[allow(unused_mut, unused_variables)]
    let mut feature_gpu_seconds = 0.0f64;

    // Which output columns the cost model sent to a registered kernel.
    let mut gpu_names: BTreeSet<String> = BTreeSet::new();
    if gpu_available && opts.gpu != GpuMode::Off {
        for n in &requested {
            if n.starts_with("xs_rank_") || !gpu_supports(n) {
                continue;
            }
            let fam = family_of(n);
            if plans.iter().any(|p| p.group == fam && p.backend == "gpu") {
                gpu_names.insert(n.clone());
            }
        }
    }
    #[allow(unused_variables)]
    let gpu_labels = !labels.is_empty()
        && gpu_available
        && opts.gpu != GpuMode::Off
        && plans
            .iter()
            .any(|p| p.group == "labels" && p.backend == "gpu");

    let mut gpu_done: BTreeSet<String> = BTreeSet::new();
    let mut fallback_log = FallbackLog::default();
    #[allow(unused_mut)]
    let mut gpu_kernel_seconds = 0.0;
    #[allow(unused_mut)]
    let mut gpu_d2h_seconds = 0.0;
    #[allow(unused_mut)]
    let mut gpu_write_seconds = 0.0;
    #[allow(unused_mut)]
    let mut gpu_transfer_bytes = 0u64;
    #[allow(unused_mut)]
    let mut gpu_h2d_seconds = 0.0;
    // #294: the resolved device read projection, recorded in the report so the
    // audited column list cannot drift from the bytes actually read.
    #[allow(unused_mut)]
    let mut device_projection: Option<(String, Vec<String>)> = None;
    #[allow(unused_mut, unused_variables)]
    let mut side_kernel_seconds = 0.0;
    #[allow(unused_mut, unused_variables)]
    let mut side_transfer_bytes = 0u64;
    #[allow(unused_mut, unused_variables)]
    let mut side_h2d_seconds = 0.0;
    #[allow(unused_mut, unused_variables)]
    let mut side_d2h_seconds = 0.0;
    #[cfg(feature = "cuda")]
    let mut gpu_ctx: Option<crate::enrich_gpu::EnrichGpu> = None;

    #[cfg(feature = "cuda")]
    {
        // #296: the MTF family owns a separate context because it needs the
        // higher-timeframe arena, not the base-length enrichment scratch. It
        // reads only the base columns the projection also needs (time + OHLC).
        let mtf_wanted: Vec<String> = gpu_names
            .iter()
            .filter(|n| is_mtf(n))
            .cloned()
            .collect();
        if !mtf_wanted.is_empty() {
            let tr = Instant::now();
            let times = i64_column_all(&store, time_col, cols.time_dtype)?;
            let open = column_all(&store, cols.open)?;
            let high = column_all(&store, cols.high)?;
            let low = column_all(&store, cols.low)?;
            let close = column_all(&store, cols.close)?;
            feature_read_seconds += tr.elapsed().as_secs_f64();
            let seg_start: Vec<u32> = segs.iter().map(|s| s.start as u32).collect();
            let seg_len: Vec<u32> = segs.iter().map(|s| s.len as u32).collect();
            let tg = Instant::now();
            let mtf_result = crate::mtf_gpu::compute(
                &times,
                &open,
                &high,
                &low,
                &close,
                &seg_start,
                &seg_len,
                step,
                &mtf_wanted,
            );
            feature_gpu_seconds += tg.elapsed().as_secs_f64();
            match mtf_result {
                Ok((columns, stats)) => {
                    for (name, values) in columns {
                        if let Some(t) = temp.get_mut(&name) {
                            let tw = Instant::now();
                            t.write_values(&values)?;
                            gpu_write_seconds += tw.elapsed().as_secs_f64();
                            gpu_done.insert(name);
                        }
                    }
                    side_kernel_seconds += stats.kernel_s;
                    side_transfer_bytes += stats.h2d_bytes + stats.d2h_bytes;
                    side_h2d_seconds += stats.h2d_s;
                    side_d2h_seconds += stats.d2h_s;
                }
                Err(e) => {
                    eprintln!(
                        "warning: device MTF failed ({e}); computing {} columns on the CPU",
                        mtf_wanted.len()
                    );
                    fallback_log.record("mtf", &e.to_string(), mtf_wanted.iter().map(String::as_str));
                }
            }
            for name in &mtf_wanted {
                gpu_names.remove(name);
            }
        }
        // #299: benchmark joins, relative returns and rolling beta run in their
        // own small context: base time + close, the segment table and the
        // benchmark series. A failure keeps the CPU reference for all of them.
        let bench_wanted: Vec<String> = gpu_names
            .iter()
            .filter(|n| crate::bench_gpu::is_device_bench(n))
            .cloned()
            .collect();
        if !bench_wanted.is_empty() {
            let tr = Instant::now();
            let times = i64_column_all(&store, time_col, cols.time_dtype)?;
            let close = column_all(&store, cols.close)?;
            feature_read_seconds += tr.elapsed().as_secs_f64();
            let seg_start: Vec<u32> = segs.iter().map(|s| s.start as u32).collect();
            let seg_len: Vec<u32> = segs.iter().map(|s| s.len as u32).collect();
            let btc = cross.bench_series();
            let eth_series = eth.bench_series();
            let self_flags: Vec<u8> = segs
                .iter()
                .map(|s| (!cross.symbol.is_empty() && s.symbol == cross.symbol) as u8)
                .chain(
                    segs.iter()
                        .map(|s| (!eth.symbol.is_empty() && s.symbol == eth.symbol) as u8),
                )
                .collect();
            let tg = Instant::now();
            let bench_result = crate::bench_gpu::compute(
                &times,
                &close,
                &seg_start,
                &seg_len,
                step,
                Some(&btc),
                Some(&eth_series),
                &self_flags,
                &bench_wanted,
            );
            feature_gpu_seconds += tg.elapsed().as_secs_f64();
            match bench_result {
                Some((columns, stats)) => {
                    for (name, values) in columns {
                        if let Some(t) = temp.get_mut(&name) {
                            let tw = Instant::now();
                            t.write_values(&values)?;
                            gpu_write_seconds += tw.elapsed().as_secs_f64();
                            gpu_done.insert(name);
                        }
                    }
                    side_kernel_seconds += stats.kernel_s;
                    side_transfer_bytes += stats.h2d_bytes + stats.d2h_bytes;
                    side_h2d_seconds += stats.h2d_s;
                    side_d2h_seconds += stats.d2h_s;
                }
                None => fallback_log.record("bench", "no result", bench_wanted.iter().map(String::as_str)),
            }
            for name in &bench_wanted {
                gpu_names.remove(name);
            }
        }
    }
    #[cfg(feature = "cuda")]
    if !gpu_names.is_empty() || gpu_labels {
        // Projected transport: only the raw columns the device needs, read
        // group by group into a bounded pinned ring. Open/high/low/close are
        // required; volume/quote_volume/trades/taker_buy_* are optional and
        // start as NaN.
        let mut wanted: Vec<(usize, crate::enrich_gpu::DeviceColumn)> = vec![
            (time_col, crate::enrich_gpu::DeviceColumn::Time),
            (cols.open, crate::enrich_gpu::DeviceColumn::Value(0)),
            (cols.high, crate::enrich_gpu::DeviceColumn::Value(1)),
            (cols.low, crate::enrich_gpu::DeviceColumn::Value(2)),
            (cols.close, crate::enrich_gpu::DeviceColumn::Value(3)),
        ];
        if let Some(c) = cols.volume {
            wanted.push((c, crate::enrich_gpu::DeviceColumn::Value(4)));
        }
        if let Some(c) = cols.quote_volume {
            wanted.push((c, crate::enrich_gpu::DeviceColumn::Value(5)));
        }
        if let Some(c) = cols.trades {
            wanted.push((c, crate::enrich_gpu::DeviceColumn::Value(6)));
        }
        if let Some(c) = cols.taker_buy_base {
            wanted.push((c, crate::enrich_gpu::DeviceColumn::Value(7)));
        }
        if let Some(c) = cols.taker_buy_quote {
            wanted.push((c, crate::enrich_gpu::DeviceColumn::Value(8)));
        }
        wanted.sort_by_key(|(i, _)| *i);
        wanted.dedup_by_key(|(i, _)| *i);
        let proj_names: Vec<String> = wanted.iter().map(|(i, _)| names[*i].clone()).collect();
        let seg_start: Vec<u32> = segs.iter().map(|s| s.start as u32).collect();
        let seg_len: Vec<u32> = segs.iter().map(|s| s.len as u32).collect();
        let precompute: Vec<String> = gpu_names
            .iter()
            .filter(|n| crate::enrich_gpu::tile_feature(n).is_some())
            .cloned()
            .collect();
        if let Ok(projection) = crate::store::Projection::resolve_owned(&store, &proj_names) {
            let sources: Vec<crate::enrich_gpu::ProjectedSource> = projection
                .columns()
                .iter()
                .filter_map(|c| {
                    wanted
                        .iter()
                        .find(|(i, _)| *i == c.index)
                        .map(|(i, d)| crate::enrich_gpu::ProjectedSource {
                            header_index: *i,
                            destination: *d,
                        })
                })
                .collect();
            device_projection = Some((projection.hash_hex(), projection.names()));
            // A/B escape hatch: force the historical whole-column upload.
            let disabled = std::env::var("RCSR_NO_PROJECTED_UPLOAD")
                .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false);
            if disabled {
                device_projection = None;
            } else {
                let tc = Instant::now();
                gpu_ctx = crate::enrich_gpu::EnrichGpu::from_projected(
                    &store,
                    projection,
                    &sources,
                    &seg_start,
                    &seg_len,
                    step,
                    &precompute,
                    3,
                );
                feature_context_seconds += tc.elapsed().as_secs_f64();
            }
        }
        if gpu_ctx.is_none() {
            // Fallback: the historical whole-column upload. It is the narrow
            // path for an exotic store layout, not the default.
            let tr = Instant::now();
            let times = i64_column_all(&store, time_col, cols.time_dtype)?;
            let nan = vec![f64::NAN; store.n_rows()];
            let volume = match cols.volume {
                Some(c) => column_all(&store, c)?,
                None => nan.clone(),
            };
            let quote_volume = match cols.quote_volume {
                Some(c) => column_all(&store, c)?,
                None => nan.clone(),
            };
            let open = column_all(&store, cols.open)?;
            let high = column_all(&store, cols.high)?;
            let low = column_all(&store, cols.low)?;
            let close = column_all(&store, cols.close)?;
            // #297: optional flow columns are present only when the store
            // carries them. An absent column is uploaded as NaN and the fused
            // base pass emits NaN, never 0.
            let trades = match cols.trades {
                Some(c) => column_all(&store, c)?,
                None => nan.clone(),
            };
            let taker_buy_base = match cols.taker_buy_base {
                Some(c) => column_all(&store, c)?,
                None => nan.clone(),
            };
            let taker_buy_quote = match cols.taker_buy_quote {
                Some(c) => column_all(&store, c)?,
                None => nan,
            };
            feature_read_seconds += tr.elapsed().as_secs_f64();
            let mut present = 0u32;
            if cols.volume.is_some() {
                present |= crate::features_gpu::PRESENT_VOLUME;
            }
            if cols.quote_volume.is_some() {
                present |= crate::features_gpu::PRESENT_QUOTE_VOLUME;
            }
            if cols.trades.is_some() {
                present |= crate::features_gpu::PRESENT_TRADES;
            }
            if cols.taker_buy_base.is_some() {
                present |= crate::features_gpu::PRESENT_TAKER_BUY;
            }
            if cols.taker_buy_quote.is_some() {
                present |= crate::features_gpu::PRESENT_TAKER_BUY_QV;
            }
            let tc = Instant::now();
            gpu_ctx = crate::enrich_gpu::EnrichGpu::new_with_flow(
                &times,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &quote_volume,
                &trades,
                &taker_buy_base,
                &taker_buy_quote,
                present,
                &seg_start,
                &seg_len,
                step,
            );
            feature_context_seconds += tc.elapsed().as_secs_f64();
        }
        if gpu_ctx.is_none() {
            // No device context: the whole planned GPU share falls back.
            fallback_log.record(
                "no-device",
                "no device context",
                gpu_names
                    .iter()
                    .map(String::as_str)
                    .chain(if gpu_labels { labels.iter().map(|(n, _)| n.as_str()).collect() } else { Vec::new() }),
            );
            gpu_names.clear();
        }
    }

    #[cfg(feature = "cuda")]
    if let Some(gpu) = gpu_ctx.as_mut() {
        let t_gpu_block = Instant::now();
        let write_before_block = gpu_write_seconds;
        // Features computed per uploaded tile stay resident until downloaded.
        // Drain them first: they would otherwise hold the device memory the
        // fused flow and price-action planes need.
        for name in gpu.precomputed_names() {
            let Some(vals) = gpu.take_precomputed(&name) else {
                continue;
            };
            let tw = Instant::now();
            let ok = match temp.get_mut(&name) {
                Some(t) => t.write_values(&vals).is_ok(),
                None => false,
            };
            gpu_write_seconds += tw.elapsed().as_secs_f64();
            if ok {
                gpu_done.insert(name.clone());
            } else {
                fallback_log.record("feature", "temp write failed", [name.as_str()]);
            }
            gpu_names.remove(&name);
        }
        // MACD is three output columns from one scan; compute it once.
        let macd_wanted: Vec<String> = gpu_names
            .iter()
            .filter(|n| matches!(n.as_str(), "macd" | "macd_sig" | "macd_hist"))
            .cloned()
            .collect();
        if !macd_wanted.is_empty() {
            if let Some((m, s, h)) = gpu.macd() {
                for name in &macd_wanted {
                    let vals = match name.as_str() {
                        "macd" => &m,
                        "macd_sig" => &s,
                        _ => &h,
                    };
                    if let Some(t) = temp.get_mut(name) {
                        let tw = Instant::now();
                        t.write_values(vals)?;
                        gpu_write_seconds += tw.elapsed().as_secs_f64();
                        gpu_done.insert(name.clone());
                    }
                }
            } else {
                fallback_log.record("macd", "no result", macd_wanted.iter().map(String::as_str));
            }
            for n in &macd_wanted {
                gpu_names.remove(n);
            }
        }
        // #297: the flow and price-action families are fused. Run the base
        // pass once per family, then download each requested column. A family
        // failure falls back to the CPU reference for every column in it.
        let flow_wanted: Vec<String> = gpu_names
            .iter()
            .filter(|n| crate::features_gpu::is_device_flow(n))
            .cloned()
            .collect();
        if !flow_wanted.is_empty() {
            let refs: Vec<&str> = flow_wanted.iter().map(String::as_str).collect();
            match gpu.flow_all(&refs) {
                Some(cols) => {
                    for (name, vals) in cols {
                        if let Some(t) = temp.get_mut(&name) {
                            t.write_values(&vals)?;
                            gpu_done.insert(name);
                        }
                    }
                }
                None => fallback_log.record("flow", "no result", flow_wanted.iter().map(String::as_str)),
            }
            for n in &flow_wanted {
                gpu_names.remove(n);
            }
        }
        let pa_wanted: Vec<String> = gpu_names
            .iter()
            .filter(|n| crate::features_gpu::is_device_price_action(n))
            .cloned()
            .collect();
        if !pa_wanted.is_empty() {
            let refs: Vec<&str> = pa_wanted.iter().map(String::as_str).collect();
            match gpu.price_action_all(&refs) {
                Some(cols) => {
                    for (name, vals) in cols {
                        if let Some(t) = temp.get_mut(&name) {
                            t.write_values(&vals)?;
                            gpu_done.insert(name);
                        }
                    }
                }
                None => fallback_log.record("price_action", "no result", pa_wanted.iter().map(String::as_str)),
            }
            for n in &pa_wanted {
                gpu_names.remove(n);
            }
        }
        let names: Vec<String> = gpu_names
            .iter()
            .filter(|n| !n.starts_with("macd"))
            .cloned()
            .collect();
        // #300: batch family/column blocks through one output arena and one
        // pinned chunked D2H. Falls back to the per-column staging path when
        // the arena cannot hold the block (a 16M-row column is 128 MiB, above
        // the 64 MiB cap) or when a kernel fails.
        let mut fallback: Vec<String> = Vec::new();
        // `RCSR_ENRICH_GPU_BATCH=0` keeps the per-column staging path for a
        // paired A/B measurement of the batched output arena.
        let cap = if std::env::var("RCSR_ENRICH_GPU_BATCH").ok().as_deref() == Some("0") {
            0
        } else {
            gpu.batch_capacity()
        };
        let mut jobs: Vec<(String, crate::enrich_gpu::BatchOp)> = Vec::new();
        for name in &names {
            match gpu_batch_op(name) {
                Some(op) => jobs.push((name.clone(), op)),
                None => fallback.push(name.clone()),
            }
        }
        let mut i = 0usize;
        while i < jobs.len() && cap > 0 {
            let start = i;
            let mut slots = 0usize;
            while i < jobs.len() && slots + jobs[i].1.slots() <= cap {
                slots += jobs[i].1.slots();
                i += 1;
            }
            if i == start {
                fallback.push(jobs[i].0.clone());
                i += 1;
                continue;
            }
            let ops: Vec<crate::enrich_gpu::BatchOp> =
                jobs[start..i].iter().map(|(_, op)| *op).collect();
            let mut written = false;
            if let Some(block) = gpu.batch(&ops) {
                written = true;
                for (slot, (name, _)) in jobs[start..i].iter().enumerate() {
                    let tw = Instant::now();
                    let ok = match temp.get_mut(name) {
                        Some(t) => t.write_values(block.column(slot)).is_ok(),
                        None => false,
                    };
                    gpu_write_seconds += tw.elapsed().as_secs_f64();
                    if ok {
                        gpu_done.insert(name.clone());
                    } else {
                        written = false;
                    }
                }
            }
            if !written {
                for (name, _) in &jobs[start..i] {
                    fallback.push(name.clone());
                }
            }
        }
        let _ = fallback;
        let mut no_result: Vec<String> = Vec::new();
        let mut write_failed: Vec<String> = Vec::new();
        for name in names {
            if gpu_done.contains(&name) {
                gpu_names.remove(&name);
                continue;
            }
            let vals = gpu_feature(&name, gpu);
            let ok = match vals {
                Some(vals) => match temp.get_mut(&name) {
                    Some(t) => {
                        let tw = Instant::now();
                        let ok = t.write_values(&vals).is_ok();
                        gpu_write_seconds += tw.elapsed().as_secs_f64();
                        if !ok {
                            write_failed.push(name.clone());
                        }
                        ok
                    }
                    None => {
                        write_failed.push(name.clone());
                        false
                    }
                },
                None => {
                    no_result.push(name.clone());
                    false
                }
            };
            if ok {
                gpu_done.insert(name.clone());
            }
            gpu_names.remove(&name);
        }
        fallback_log.record("feature", "no result", no_result.iter().map(String::as_str));
        fallback_log.record("feature", "temp write failed", write_failed.iter().map(String::as_str));
        if gpu_labels {
            for (name, spec) in &labels {
                let vals = gpu.label(spec.horizon, spec.cost_bps, spec.vol_window, spec.stride);
                match vals {
                    Some(v) => {
                        if let Some(t) = temp.get_mut(name) {
                            let tw = Instant::now();
                            t.write_values(&v)?;
                            gpu_write_seconds += tw.elapsed().as_secs_f64();
                            gpu_done.insert(name.clone());
                        }
                    }
                    None => fallback_log.record("label", "no result", [name.as_str()]),
                }
            }
        }
        feature_gpu_seconds +=
            t_gpu_block.elapsed().as_secs_f64() - (gpu_write_seconds - write_before_block);
        let stats = gpu.stats();
        gpu_kernel_seconds = stats.kernel_s + side_kernel_seconds;
        gpu_transfer_bytes = stats.h2d_bytes + stats.d2h_bytes + side_transfer_bytes;
        gpu_h2d_seconds = stats.h2d_s + side_h2d_seconds;
        gpu_d2h_seconds = stats.d2h_s + side_d2h_seconds;
    }

    // If only the MTF context ran, its ledger is still the GPU share.
    #[cfg(feature = "cuda")]
    if gpu_ctx.is_none() {
        gpu_kernel_seconds = side_kernel_seconds;
        gpu_transfer_bytes = side_transfer_bytes;
        gpu_h2d_seconds = side_h2d_seconds;
        gpu_d2h_seconds = side_d2h_seconds;
    }

    // Everything planned for the CPU (including failed GPU calls).
    let t_cpu_pass = Instant::now();
    let cpu_own: Vec<String> = own
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_legacy: Vec<String> = legacy
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_py15m: Vec<String> = py15m_requested
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_compact: Vec<String> = compact_requested
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_flow: Vec<String> = flow_requested
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_price_action: Vec<String> = price_action_requested
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_mtf: Vec<String> = mtf_requested
        .iter()
        .filter(|n| !gpu_done.contains(*n))
        .cloned()
        .collect();
    let cpu_labels: Vec<(String, LabelSpec)> = labels
        .iter()
        .filter(|(n, _)| !gpu_done.contains(n))
        .cloned()
        .collect();
    let ctx = Context {
        legacy: cpu_legacy,
        own: cpu_own,
        py15m_requested: cpu_py15m,
        compact: cpu_compact,
        flow: cpu_flow,
        price_action: cpu_price_action,
        mtf: cpu_mtf,
        labels: cpu_labels,
        long_labels: long_specs.clone(),
        tf,
        cross: &cross,
        eth: &eth,
    };

    // Feature pass, one symbol at a time in row order. `--chunk-rows` bounds
    // the read/compute batch; it never changes arithmetic or output order.
    let has_cpu_work = !ctx.legacy.is_empty()
        || !ctx.own.is_empty()
        || !ctx.py15m_requested.is_empty()
        || !ctx.compact.is_empty()
        || !ctx.flow.is_empty()
        || !ctx.price_action.is_empty()
        || !ctx.mtf.is_empty()
        || !ctx.labels.is_empty()
        || !ctx.long_labels.is_empty();
    if has_cpu_work {
        let mut i = 0usize;
        while i < segs.len() {
            let start = i;
            let mut rows = 0usize;
            while i < segs.len() && (rows == 0 || rows + segs[i].len <= opts.chunk_rows) {
                rows += segs[i].len;
                i += 1;
            }
            for seg in &segs[start..i] {
                let s = load_series(&store, seg, &cols, step)?;
                compute_series(&s, &ctx, &mut temp, &mut long_acc)?;
            }
        }
    }
    let feature_cpu_seconds = t_cpu_pass.elapsed().as_secs_f64();
    let feature_seconds = t_feat.elapsed().as_secs_f64();
    let feature_other_seconds = feature_seconds
        - (feature_read_seconds
            + feature_context_seconds
            + feature_gpu_seconds
            + gpu_write_seconds
            + feature_cpu_seconds);

    // The cross-sectional pass reads the source columns back, so flush the
    // buffered writers first (the files are row-complete at this point).
    for t in temp.values_mut() {
        t.flush()?;
    }

    // Point-in-time cross-section pass: market state, then ranks. The two
    // passes share one `(time, symbol, row)` order and one membership mask.
    let t_xs = Instant::now();
    if !cross_sources.is_empty() {
        let panel = CrossPanel::new(
            &store,
            &segs,
            time_col,
            cols.time_dtype,
            member_col.map(|c| (c, member_dtype.unwrap())),
        )?;

        #[cfg(feature = "cuda")]
        let xs_rank_gpu = requested
            .iter()
            .any(|n| n.starts_with("xs_rank_"))
            && gpu_available
            && opts.gpu != GpuMode::Off
            && plans
                .iter()
                .any(|p| p.group == "xs" && p.backend == "gpu");
        // #299: the per-timestamp market means run on the device through the
        // same resident (time, symbol, row) permutation as the ranks.
        #[cfg(feature = "cuda")]
        let market_gpu = wants_market_state(&requested)
            && gpu_available
            && market_device_enabled(opts.gpu, store.n_rows());
        #[cfg(feature = "cuda")]
        if (xs_rank_gpu || market_gpu) && gpu_ctx.is_none() {
            let times_all = i64_column_all(&store, time_col, cols.time_dtype)?;
            let nan = vec![f64::NAN; store.n_rows()];
            let volume = match cols.volume {
                Some(c) => column_all(&store, c)?,
                None => nan.clone(),
            };
            let quote_volume = match cols.quote_volume {
                Some(c) => column_all(&store, c)?,
                None => nan,
            };
            let nan_col = vec![f64::NAN; store.n_rows()];
            let trades = match cols.trades {
                Some(c) => column_all(&store, c)?,
                None => nan_col.clone(),
            };
            let taker_buy_base = match cols.taker_buy_base {
                Some(c) => column_all(&store, c)?,
                None => nan_col.clone(),
            };
            let taker_buy_quote = match cols.taker_buy_quote {
                Some(c) => column_all(&store, c)?,
                None => nan_col,
            };
            let mut present = 0u32;
            if cols.volume.is_some() {
                present |= crate::features_gpu::PRESENT_VOLUME;
            }
            if cols.quote_volume.is_some() {
                present |= crate::features_gpu::PRESENT_QUOTE_VOLUME;
            }
            if cols.trades.is_some() {
                present |= crate::features_gpu::PRESENT_TRADES;
            }
            if cols.taker_buy_base.is_some() {
                present |= crate::features_gpu::PRESENT_TAKER_BUY;
            }
            if cols.taker_buy_quote.is_some() {
                present |= crate::features_gpu::PRESENT_TAKER_BUY_QV;
            }
            let seg_start: Vec<u32> = segs.iter().map(|s| s.start as u32).collect();
            let seg_len: Vec<u32> = segs.iter().map(|s| s.len as u32).collect();
            gpu_ctx = crate::enrich_gpu::EnrichGpu::new_with_flow(
                &times_all,
                &column_all(&store, cols.open)?,
                &column_all(&store, cols.high)?,
                &column_all(&store, cols.low)?,
                &column_all(&store, cols.close)?,
                &volume,
                &quote_volume,
                &trades,
                &taker_buy_base,
                &taker_buy_quote,
                present,
                &seg_start,
                &seg_len,
                step,
            );
        }

        #[cfg(feature = "cuda")]
        let mut market_prepared = false;
        #[cfg(feature = "cuda")]
        if market_gpu {
            if let Some(gpu) = gpu_ctx.as_mut() {
                market_prepared = gpu.prepare_cross_section(&panel.sym_key, Some(&panel.member));
            }
        }

        if wants_market_state(&requested) {
            #[cfg(feature = "cuda")]
            let device: Option<&mut dyn crate::bench_gpu::MarketDevice> = if market_prepared {
                gpu_ctx
                    .as_mut()
                    .map(|g| g as &mut dyn crate::bench_gpu::MarketDevice)
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let device: Option<&mut dyn crate::bench_gpu::MarketDevice> = None;
            let market_done = market_state_pass(
                &panel,
                &mut temp,
                &requested,
                &segs,
                &store,
                &cols,
                &cross,
                step,
                device,
            )?;
            #[cfg(feature = "cuda")]
            if market_gpu {
                fallback_log.record(
                    "market",
                    "no result",
                    requested
                        .iter()
                        .filter(|n| {
                            crate::bench_gpu::is_device_market(n) && !market_done.contains(*n)
                        })
                        .map(String::as_str),
                );
                if let Some(gpu) = gpu_ctx.as_ref() {
                    let stats = gpu.stats();
                    gpu_kernel_seconds = stats.kernel_s + side_kernel_seconds;
                    gpu_transfer_bytes =
                        stats.h2d_bytes + stats.d2h_bytes + side_transfer_bytes;
                    gpu_h2d_seconds = stats.h2d_s + side_h2d_seconds;
                    gpu_d2h_seconds = stats.d2h_s + side_d2h_seconds;
                }
            }
            for n in market_done {
                gpu_done.insert(n);
            }
        }

        let rank_names: Vec<String> = requested
            .iter()
            .filter(|n| n.starts_with("xs_rank_") || n.as_str() == "cs_rank_r60")
            .cloned()
            .collect();
        if !rank_names.is_empty() {
            #[cfg(feature = "cuda")]
            let mut xs_prepared = false;
            for rank_name in &rank_names {
                let src = cross_source(rank_name)
                    .ok_or_else(|| format!("no source registered for '{}'", rank_name))?;
                let mut values = source_values(&temp, src, store.n_rows())?;
                #[allow(unused_mut)]
                let mut ranks = None;
                let gpu_eligible = rank_name.starts_with("xs_rank_")
                    && gpu_available
                    && opts.gpu != GpuMode::Off
                    && plans
                        .iter()
                        .any(|p| p.group == "xs" && p.backend == "gpu");
                #[cfg(feature = "cuda")]
                if gpu_eligible {
                    if let Some(gpu) = gpu_ctx.as_mut() {
                        // One device (time, symbol, row) permutation and one
                        // membership mask serve every rank column.
                        if !xs_prepared
                            && gpu.prepare_cross_section(
                                &panel.sym_key,
                                Some(&panel.member),
                            )
                        {
                            xs_prepared = true;
                        }
                        if xs_prepared {
                            ranks = gpu.rank(&values);
                            let stats = gpu.stats();
                            gpu_kernel_seconds = stats.kernel_s + side_kernel_seconds;
                            gpu_d2h_seconds = stats.d2h_s + side_d2h_seconds;
                            gpu_h2d_seconds = stats.h2d_s + side_h2d_seconds;
                            gpu_transfer_bytes =
                                stats.h2d_bytes + stats.d2h_bytes + side_transfer_bytes;
                        }
                    }
                }
                let (ranks, from_gpu) = match ranks {
                    Some(r) => (r, true),
                    None => {
                        if gpu_eligible {
                            fallback_log.record("rank", "no result", [rank_name.as_str()]);
                        }
                        (
                            rank_with_membership(
                                &values,
                                &panel.times,
                                &panel.sym_key,
                                Some(&panel.member),
                                store.n_rows(),
                            )?,
                            false,
                        )
                    }
                };
                if let Some(t) = temp.get_mut(rank_name) {
                    t.reset()?;
                    t.write_values(&ranks)?;
                }
                if from_gpu {
                    gpu_done.insert(rank_name.clone());
                }
                values.clear();
            }
        }

        // Remove sources that are not requested output columns.
        for src in &cross_sources {
            if src != "volume" && !requested.contains(src) {
                if let Some(t) = temp.remove(src) {
                    let _ = std::fs::remove_file(t.path);
                }
            }
        }
    }
    let xs_seconds = t_xs.elapsed().as_secs_f64();

    let long_outputs: Vec<LongLabelOutput> = long_specs
        .iter()
        .zip(long_acc)
        .map(|(sp, acc)| LongLabelOutput {
            spec: sp.clone(),
            columns: crate::labels::COLUMN_KINDS.iter().map(|(k, _)| sp.column(k)).collect(),
            target: sp.column("y"),
            report: acc.finish(),
        })
        .collect();

    let total_columns = requested.len() + labels.len() + long_names.len();
    let gpu_columns = gpu_done.len().min(total_columns);
    let cpu_columns = total_columns.saturating_sub(gpu_columns);
    let gpu_share = if total_columns == 0 {
        0.0
    } else {
        gpu_columns as f64 / total_columns as f64
    };

    // Finish temp columns and build the append list in the requested order.
    let mut reports: Vec<FeatureReport> = Vec::new();
    let mut extras: Vec<AppendColumn> = Vec::new();
    let order: Vec<String> = requested
        .iter()
        .chain(labels.iter().map(|(n, _)| n))
        .chain(long_names.iter())
        .cloned()
        .collect();
    for name in &order {
        let mut t = temp
            .remove(name)
            .ok_or_else(|| format!("missing temporary column '{}'", name))?;
        if t.row_offset != store.n_rows() {
            return Err(format!(
                "column '{}' has {} rows, expected {}",
                name, t.row_offset, store.n_rows()
            ));
        }
        let r = t.finish()?;
        extras.push(AppendColumn {
            name: name.clone(),
            file: t.path.clone(),
            replace: opts.replace,
        });
        reports.push(r);
    }
    for (_, t) in temp {
        // Any remaining temp file is an unreferenced xs source; remove it.
        let _ = std::fs::remove_file(t.path);
    }
    let order: Vec<String> = requested
        .iter()
        .chain(labels.iter().map(|(n, _)| n))
        .chain(long_names.iter())
        .cloned()
        .collect();
    reports.sort_by_key(|r| order.iter().position(|n| n == &r.name).unwrap_or(usize::MAX));

    // Per-column prevalence and first finite timestamp. The timestamp is read
    // back by global row index, so the report names a date rather than only an
    // offset into the store.
    let rows_total = store.n_rows();
    for r in &mut reports {
        r.prevalence = if rows_total == 0 {
            0.0
        } else {
            (rows_total.saturating_sub(r.nan_rows)) as f64 / rows_total as f64
        };
        r.first_finite_time = match r.first_finite_row {
            Some(row) => Some(time_at_row(&store, time_col, cols.time_dtype, row)?),
            None => None,
        };
    }

    if opts.dry_run {
        for e in &extras {
            let _ = std::fs::remove_file(&e.file);
        }
        let report = Report {
            input: opts.data.clone(),
            output: out.to_string(),
            input_sha256: input_sha,
            output_sha256: String::new(),
            output_bytes: 0,
            rows: store.n_rows(),
            columns_in: store.n_cols(),
            columns_out: expected_columns_out(&store, &requested, &labels, &long_names, opts.replace),
            groups: store.n_groups(),
            symbols: segs.len(),
            group_rows: store.group_rows(),
            chunk_rows: opts.chunk_rows,
            bar_seconds: step,
            features: reports,
            plans,
            long_labels: long_outputs,
            backend: if gpu_columns > 0 { "gpu" } else { "cpu" }.to_string(),
            gpu_requested: format!("{:?}", opts.gpu).to_lowercase(),
            gpu_available,
            gpu_device,
            gpu_columns,
            cpu_columns,
            gpu_share,
            gpu_fallback: fallback_log.columns.len(),
            gpu_fallback_columns: fallback_log.columns.clone(),
            gpu_kernel_seconds,
            gpu_d2h_seconds,
            gpu_write_seconds,
            write_bw,
            gpu_transfer_bytes,
            gpu_h2d_seconds,
            projection_hash: device_projection.as_ref().map(|(h, _)| h.clone()),
            projection_columns: device_projection
                .as_ref()
                .map(|(_, c)| c.clone())
                .unwrap_or_default(),
            read_seconds,
            feature_seconds,
            feature_read_seconds,
            feature_context_seconds,
            feature_gpu_seconds,
            feature_cpu_seconds,
            feature_other_seconds,
            xs_seconds,
            write_seconds: 0.0,
            total_seconds: started.elapsed().as_secs_f64(),
            peak_rss_bytes: peak_rss_bytes(),
        };
        return Ok(report);
    }

    // Append pass: copy existing blocks and write the new columns. The writer
    // hashes the bytes as they stream out, so the report gets the output
    // checksum without a second multi-gigabyte read.
    let t_write = Instant::now();
    let tmp_out = work_dir.join("out.rcsb").to_string_lossy().into_owned();
    let output_sha = store::write_store_appending_hashed(&tmp_out, &store, &extras)?;
    std::fs::rename(&tmp_out, out).map_err(|e| format!("{}: {}", out, e))?;
    let write_seconds = t_write.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);

    let columns_out = FileStore::open(out)?.n_cols();
    write_label_sidecars(&opts.data, out, &long_outputs)?;
    let backend = if gpu_columns > 0 { "gpu" } else { "cpu" };
    Ok(Report {
        input: opts.data.clone(),
        output: out.to_string(),
        input_sha256: input_sha,
        output_sha256: output_sha,
        output_bytes: bytes,
        rows: store.n_rows(),
        columns_in: store.n_cols(),
        columns_out,
        groups: store.n_groups(),
        symbols: segs.len(),
        group_rows: store.group_rows(),
        chunk_rows: opts.chunk_rows,
        bar_seconds: step,
        features: reports,
        plans,
        long_labels: long_outputs,
        backend: backend.to_string(),
        gpu_requested: format!("{:?}", opts.gpu).to_lowercase(),
        gpu_available,
        gpu_device,
        gpu_columns,
        cpu_columns,
        gpu_share,
        gpu_fallback: fallback_log.columns.len(),
        gpu_fallback_columns: fallback_log.columns,
        gpu_kernel_seconds,
        gpu_d2h_seconds,
        gpu_write_seconds,
        write_bw,
        gpu_transfer_bytes,
        gpu_h2d_seconds,
        projection_hash: device_projection.as_ref().map(|(h, _)| h.clone()),
        projection_columns: device_projection
            .as_ref()
            .map(|(_, c)| c.clone())
            .unwrap_or_default(),
        read_seconds,
        feature_seconds,
        feature_read_seconds,
        feature_context_seconds,
        feature_gpu_seconds,
        feature_cpu_seconds,
        feature_other_seconds,
        xs_seconds,
        write_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
        peak_rss_bytes: peak_rss_bytes(),
    })
}

/// Entry lines (`{"target":...}`, no trailing comma) of a label sidecar.
fn label_sidecar_entries(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| l.starts_with("{\"target\":"))
                .map(|l| l.trim_end().trim_end_matches(',').to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Write `<out>.labels.json` (the label reports) and add the `y` target specs
/// to `<out>.targets.json`. Entries of an input store's sidecars that this run
/// does not rewrite carry over, so chained enrich runs keep earlier labels.
fn write_label_sidecars(data: &str, out: &str, outputs: &[LongLabelOutput]) -> Result<(), String> {
    if outputs.is_empty() {
        return Ok(());
    }
    let labels_path = crate::target::label_sidecar_path(out).ok_or("label sidecar needs a file output")?;
    let from = if Path::new(&labels_path).is_file() {
        Some(labels_path.clone())
    } else {
        crate::target::label_sidecar_path(data).filter(|p| Path::new(p).is_file())
    };
    let mut lines: Vec<String> = from.map(|p| label_sidecar_entries(&p)).unwrap_or_default();
    for o in outputs {
        let head = format!("{{\"target\":{},", jstr(&o.target));
        lines.retain(|l| !l.starts_with(&head));
        lines.push(format!("{}\"report\":{}}}", head, o.report.json()));
    }
    let body = format!("{{\"schema\":\"rcsr-label-report/1\",\"labels\":[\n{}\n]}}\n", lines.join(",\n"));
    crate::atomic::write_text_atomic(&labels_path, &body)?;

    let targets_path = crate::target::sidecar_path(out).ok_or("target sidecar needs a file output")?;
    let from = if Path::new(&targets_path).is_file() {
        Some(targets_path.clone())
    } else {
        crate::target::sidecar_path(data).filter(|p| Path::new(p).is_file())
    };
    let mut specs: Vec<crate::target::TargetSpec> = from
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| crate::target::parse_file(&t).ok())
        .unwrap_or_default();
    for o in outputs {
        let ts = o.spec.target_spec(&o.target, &o.spec.column("fwd"));
        specs.retain(|s| s.column != ts.column);
        specs.push(ts);
    }
    let body = format!(
        "{{\"schema\":{},\"targets\":[{}]}}\n",
        jstr(crate::target::SCHEMA),
        specs.iter().map(|s| format!("{{{}}}", s.file_json())).collect::<Vec<_>>().join(",")
    );
    crate::atomic::write_text_atomic(&targets_path, &body)
}

fn tf_for_step(step: i64) -> Option<Timeframe> {
    match step {
        60 => Some(Timeframe::M1),
        180 => Some(Timeframe::M3),
        300 => Some(Timeframe::M5),
        900 => Some(Timeframe::M15),
        1800 => Some(Timeframe::M30),
        3600 => Some(Timeframe::H1),
        7200 => Some(Timeframe::H2),
        14400 => Some(Timeframe::H4),
        86400 => Some(Timeframe::D1),
        604800 => Some(Timeframe::W1),
        _ => None,
    }
}

fn expected_columns_out(
    store: &FileStore,
    requested: &[String],
    labels: &[(String, LabelSpec)],
    long_names: &[String],
    replace: bool,
) -> usize {
    let mut names: Vec<String> = store.header().names();
    for n in requested.iter().chain(labels.iter().map(|(n, _)| n)).chain(long_names.iter()) {
        if names.contains(n) {
            if !replace {
                // The writer will refuse this; the dry run still reports the
                // intended schema so the user sees the conflict immediately.
            }
        } else {
            names.push(n.clone());
        }
    }
    names.len()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

pub fn run(args: &[String]) -> ! {
    let mut opts = Options::default();
    let mut i = 0usize;
    macro_rules! val {
        () => {{
            let v = args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: {} requires a value", args[i]);
                std::process::exit(EXIT_USAGE);
            });
            i += 1;
            v
        }};
    }
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--data" | "-d" => opts.data = val!(),
            "--out" | "-o" => opts.out = Some(val!()),
            "--features" => opts.features.push(val!()),
            "--label" => {
                let v = val!();
                if v.trim_start().starts_with("long:") && !crate::labels::LongLabelSpec::declares_fee(&v) {
                    eprintln!("error: {}", crate::labels::missing_fee_message(&v));
                    std::process::exit(EXIT_USAGE);
                }
                opts.labels.push(v);
            }
            "--allow-overlap" => opts.allow_overlap = true,
            "--allow-short" => fail(&crate::labels::refuse_allow_short()),
            "--side" => {
                let v = val!();
                if let Err(e) = crate::labels::check_side(&v) {
                    fail(&e);
                }
            }
            "--replace" => opts.replace = true,
            "--chunk-rows" => {
                let v = val!();
                opts.chunk_rows = v.parse().unwrap_or(0);
                if opts.chunk_rows == 0 {
                    fail("--chunk-rows needs a positive integer");
                }
            }
            "--group-rows" => {
                let v = val!();
                opts.group_rows = v.parse().unwrap_or(0);
                if opts.group_rows == 0 {
                    fail("--group-rows needs a positive integer");
                }
            }
            "--bar-seconds" => {
                let v = val!();
                opts.bar_seconds = Some(v.parse().unwrap_or(0));
                if opts.bar_seconds.unwrap_or(0) <= 0 {
                    fail("--bar-seconds needs a positive integer");
                }
            }
            "--time" => opts.time_col = Some(val!()),
            "--symbol" => opts.symbol_col = Some(val!()),
            "--cross-symbol" => opts.cross_symbol = val!(),
            "--cross-data" => opts.cross_data = Some(val!()),
            "--eth-symbol" => opts.eth_symbol = val!(),
            "--eth-data" => opts.eth_data = Some(val!()),
            "--gpu" => {
                if let Some(next) = args.get(i + 1) {
                    if matches!(next.as_str(), "auto" | "on" | "off" | "cpu" | "force") {
                        i += 1;
                        match GpuMode::parse(next) {
                            Ok(m) => opts.gpu = m,
                            Err(e) => fail(&e),
                        }
                        i += 1;
                        continue;
                    }
                }
                opts.gpu = GpuMode::On;
            }
            "--no-gpu" | "--cpu" => opts.gpu = GpuMode::Off,
            "--write-bw" => {
                let v = val!();
                let mbps: f64 = v.parse().unwrap_or(0.0);
                if !(mbps > 0.0) {
                    fail("--write-bw needs a positive MB/s value");
                }
                opts.write_bw = Some(mbps * 1_000_000.0);
            }
            "--allow-lockbox" => opts.allow_lockbox = true,
            "--dry-run" => opts.dry_run = true,
            "--json" => opts.json = true,
            other => fail(&format!("unknown enrich option '{}'\n\n{}", other, usage())),
        }
        i += 1;
    }
    if opts.data.is_empty() {
        fail("enrich needs --data <store.rcsb|CSV>");
    }
    if opts.out.is_none() && !opts.dry_run {
        fail("enrich needs --out <store.rcsb> (or --dry-run)");
    }
    if opts.features.is_empty() && opts.labels.is_empty() {
        fail("enrich needs --features and/or --label");
    }
    let report = match enrich(&opts) {
        Ok(r) => r,
        Err(e) => fail(&e),
    };
    if !opts.dry_run {
        if let Some(out) = &opts.out {
            let sidecar = format!("{}.enrich.json", out);
            if let Err(e) = crate::atomic::write_text_atomic(&sidecar, &report.json()) {
                eprintln!("warning: could not write {}: {}", sidecar, e);
            }
        }
    }
    if opts.json {
        print!("{}", report.json());
    } else {
        for line in report.text_lines() {
            println!("{}", line);
        }
        eprintln!(
            "enrich: {} rows x {} -> {} cols, {} ({} bytes){}",
            report.rows,
            report.columns_in,
            report.columns_out,
            report.output,
            report.output_bytes,
            if opts.dry_run { " (dry run, nothing written)" } else { "" }
        );
    }
    std::process::exit(0);
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(EXIT_INPUT);
}

pub fn usage() -> String {
    "rcsr enrich — append causal, gap-aware OHLCV features to an existing store

USAGE:
  rcsr enrich --data <store.rcsb|CSV> --out <store.rcsb> --features <list> [options]

INPUT / OUTPUT:
  --data, -d <path>       existing .rcsb column store (or a CSV, loaded once)
  --out, -o <path>        output store; the input is never modified unless the
                          path is the same, in which case a temp file is renamed
  --replace               overwrite existing feature columns in place; without
                          it a duplicate column name is an error and every
                          existing column is preserved byte for byte
  --chunk-rows <n>        read/compute batch size (default 65536); arithmetic
                          and output hashes are identical for every size
  --group-rows <n>        group size for a CSV input store (store input keeps
                          its own layout)
  --bar-seconds <n>       expected bar spacing; inferred from the modal spacing
  --time <col>            time column (default ts/time/timestamp/date)
  --symbol <col>          panel symbol column (default symbol/ticker/asset)
  --cross-symbol <name>   benchmark symbol for btc/cross features (default BTCUSDT)
  --cross-data <store>    benchmark store when --data does not contain it
  --eth-symbol <name>     ETH benchmark symbol (default ETHUSDT)
  --eth-data <store>      ETH benchmark store when --data does not contain it
  --label <spec>          fwd:<tf|bars>,cost=<bp>,volnorm=<bars>,stride=<n>
                          (repeatable; labels are targets, not features)
  --label long:<tf|bars>[,name=<id>][,delay=<bars>],fee=<bp>[,spread=<bp>]
                          [,slip=<bp>][,vol=<bars>][,floor=<bp>][,clip=<c>][,up=<u>]
                          [,down=<d>][,stride=<n>]
                          long-only label family (CPU f64): lbl_<name>_{fwd,y,
                          long_net,action,t1,exit_ret,m_net,m_barrier,weight};
                          writes <out>.labels.json (reports, label hash, top
                          10 absolute moves) and a y target spec (cost_bps, cost_per_side_bps)
                          in <out>.targets.json; fee= is required (fee=0 for none, exit 2
                          otherwise); never a --vars input
  --allow-overlap         permit a label stride below the horizon for horizons of
                          4h and longer (labels overlap; use uniqueness weights)
  --side long             the only side; --side short and --allow-short are refused
  --allow-lockbox         permit labelled rows in 2026H2 (final test only)
  --gpu [auto|on|off]     auto (default) dispatches a registered exact kernel
                          only when the cost model estimates it faster; on
                          forces every registered kernel above the 1M-row floor;
                          off runs the CPU reference. The MTF family uses one
                          device resample + causal join per timeframe and keeps
                          the CPU NaN/gap/bar_age contract; it is opt-in with
                          --gpu on until the 16M crossover is measured. The
                          benchmark family (btc_*, eth_*, rel_*, beta_*,
                          cross_ret1, cross_beta20, beta384) joins the BTC/ETH
                          series by exact timestamp on the device (a missing
                          benchmark bar is NaN) and the per-timestamp market
                          means (mkt_*, breadth, dispersion, rel_mkt*, avg_corr,
                          btc_volume_share) reuse the cross-section permutation.
  --no-gpu, --cpu         force CPU
  --write-bw <MB/s>       measured sequential write bandwidth for cost model v2
                          (default 222 MB/s on the shared ext4 store)
  --dry-run               print the report and write nothing
  --json                  print the JSON report on stdout; gpu.fallback_columns lists
                          each column the device left to the CPU, with the failing path

FEATURE GROUPS (the default financial feature set; see `rcsr schema`):
  returns    r1..r120, log close-to-close over exactly k contiguous bars
  ma         ma5 ma10 ma20 ma50 ma100 ma200, close / trailing mean - 1
  vol        vol5 vol20 vol60 vol120, population std of log returns
  atr        atr14 (price units), natr14 (percent)
  rsi        rsi14 as RSI/50 - 1 (compact definition; breaking from 0..100)
  bollinger  bb20_z, bb20_bw
  vz         vz96, z-score of log1p(volume)
  range      range1, (high - low) / close
  candle     clv1 body1 upper_wick1 lower_wick1 open_gap1
             (clv1 = (C-L)/(H-L) - 1/2, the compact definition)
  donchian   don20, don96; don96 is the compact centred channel position
  calendar   dow_sin dow_cos hour_sin hour_cos month_sin month_cos dom_sin dom_cos
  lags       lag_r1_1 .. lag_r1_10
  xs         xs_rank_r1 xs_rank_r20 xs_rank_vol20 (per-timestamp cross-section;
             finite PIT members only, ties by symbol then row, N>=20, and the
             CUDA path reuses one device (time,symbol,row) permutation)
  btc        btc_r1 btc_r20 btc_vol20 rel_btc_r1 rel_btc_r20 beta_btc_60
  cross      complete #273 family: btc/eth returns, vol and beta; equal-weight
             market mkt_r16 mkt_r96 mkt_vol96; breadth dispersion avg_corr;
             rel_mkt16 rel_mkt96; xs_rank_r1/r16/r20/vol20/amihud96 and the
             legacy mkt_ew_* / cs_rank_r60 aliases. Ranks are [-1,1] and need
             >=20 point-in-time members; a smaller cross-section is NaN.
  py15m      the 35 columns added by the Python enrich_panel_15m.py step
  mtf        the 36 multi-timeframe columns: mtf_<1h|4h|1d>_<leaf> for the
             twelve leaves ret1 ret4 ret24 ma_ratio vol atr_pct rsi close_pos
             donchian_pos range_pct trend_slope bar_age. Each higher bar is
             resampled from the base series and joined only after it closes:
             higher.time + timeframe_seconds <= base.time. Windows count
             higher-timeframe bars; a gap is NaN, and bar_age reports the
             exact seconds since the last closed higher bar. With --features
             cuda and --gpu on, all 36 columns dispatch to the device resample
             + causal join (src/cuda/mtf.cu); auto stays on the CPU until the
             16M crossover is measured. Non-transcendental leaves are bitwise
             and vol is within 1e-12.
  compact    the exact 52-column compact regression subset, 15m base, compact
             names and order (r16..don96). It is a parity/smoke subset, not the
             production feature set.
  flow       vwap plus the nine derived flow columns: taker_ratio96,
             taker_imb16, trade_intensity96, avg_trade_size96, amihud96,
             zero_ret_share96, obv_z96, cvd_proxy, vwap_dev96. Trailing 15m
             windows; a gap-spanning window is NaN; zero volume/trades is NaN
             for the affected ratios and illiquidity. In a --features cuda build
             the twelve derived flow columns (these ten plus cmf20 and
             vwap20_dev) come from one fused base pass; the registry family has
             17 columns including the five raw flow inputs.
  price_action  clv16 body16 rng_atr prev_range_pos inside_s outside_s
             engulf_s harami_s run8 nr4 trend96 (#261). Continuous geometry and
             multi-bar structure; two-bar strengths replace hard flags. A
             zero-range/zero-volume bar, a missing prior bucket or a window that
             spans a gap is NaN. `batr` is materialised by the compact profile
             and completes the twelve-name default block. The eleven columns
             have a fused CUDA producer; `rcsr schema` declares their
             bitwise/1e-12 parity tier.
  all        every legacy group above except cross, py15m, mtf, compact, flow and price_action

  #125 names and groups (ta:core, ta:full, exotic:cheap, candle:geometry,
  cross:btc, calendar) are also accepted and are computed by `rcsr::features`.

  The default financial feature set is the union of the groups above, the
  compact subset, the raw OHLCV/flow fields and the registered multi-timeframe,
  cross-sectional market-state and reserved derivative columns. The engine
  inventory has 254 names: 216 in `all,py15m,ta:full,exotic:cheap,cross:btc`
  plus 38 compact-only engine names. `rcsr schema` prints `engine_catalogue`
  (name, family, warm-up, availability, exact formula and the canonical
  shared-name map) and `feature_registry`; a registered-but-unavailable name
  fails closed with the missing data dependency instead of being silently
  dropped. All 36 multi-timeframe columns are `available:true`, and the complete
  cross-asset/breadth/rank family is `available:true`; `cross` expands it.
  The ten derivative columns (`funding_rate`, `funding_carry_24h`, `basis_bps`,
  `oi`, `oi_chg_1h`, `oi_chg_24h`, `ls_ratio`, `taker_buy_sell_ratio`,
  `liq_long`, `liq_short`) are `available:false` with source null. A request
  for one, or for the group `derivatives`, is refused and names the missing raw
  field; it never yields zeros, an empty column or a forward fill.


  Flow and compact flow features read the raw `volume`, `quote_volume`,
  `trades` and `taker_buy_base` (alias `taker_buy`) fields. A missing raw field
  is a hard error naming both the feature and the column. `vwap_dev96` and the
  other shared names have one canonical quote-volume definition, not a
  group-dependent one.

GPU (CUDA builds): registered f64 kernels cover returns/lags, calendar, moving
sums and variances, Donchian, Bollinger/VZ, EMA/Wilder scans, RSI, labels and
cross-sectional ranks. The source columns are uploaded once per run. The report
records gpu_columns, cpu_columns, gpu_share, fallback, kernel seconds and
transfer bytes; auto keeps a family on the CPU when the kernel does not repay
the transfer. CPU-only builds behave exactly as before.

The output store checksum and every feature column hash are written to
<out>.enrich.json. See docs/ohlcv.md for the catalogue, gap and warm-up rules.
"
    .to_string()
}


#[cfg(test)]
mod fallback_tests {
    use super::*;

    #[test]
    fn every_fallback_records_name_and_path_reason_once() {
        let mut log = FallbackLog::default();
        log.record("flow", "no result", ["flow_a", "flow_b"]);
        log.record("feature", "no result", ["flow_b", "ema8"]);
        log.record("rank", "no result", std::iter::empty());
        let got: Vec<(&str, &str)> = log
            .columns
            .iter()
            .map(|(n, r)| (n.as_str(), r.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("flow_a", "flow: no result"),
                ("flow_b", "flow: no result"),
                ("ema8", "feature: no result"),
            ]
        );
    }

    #[test]
    fn report_lists_fallback_columns_in_text_and_json() {
        let mut log = FallbackLog::default();
        log.record("market", "no result", ["breadth"]);
        let report = Report {
            gpu_fallback: log.columns.len(),
            gpu_fallback_columns: log.columns,
            ..Default::default()
        };
        assert!(report
            .text_lines()
            .contains(&"# enrich-fallback: name=breadth reason=market: no result".to_string()));
        assert!(report
            .json()
            .contains("\"fallback_columns\":[{\"name\":\"breadth\",\"reason\":\"market: no result\"}]"));
    }
}
