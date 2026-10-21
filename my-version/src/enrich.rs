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
//! (I/O + transfer + kernel) for every feature family and records the decision
//! in the report. GPU execution is only selected for a family whose exact
//! kernel is registered *and* whose estimated GPU time is lower; this build
//! registers no enrichment kernels, so every family falls back to CPU with the
//! reason recorded. A CUDA build therefore behaves exactly like a CPU-only
//! build for `enrich` today (see `docs/ohlcv.md`).

use crate::features::{self, LabelSpec};
use crate::manifest::{self, Sha256Hasher};
use crate::ohlcv::{civil_from_days, weekday, Bar, Timeframe};
use crate::store::{self, AppendColumn, DType, FileStore};
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
    "py15m",
    "all",
];

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
            "py15m" => py15m_names(),
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
pub fn nominal_warmup(name: &str) -> usize {
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
        "atr14" | "natr14" => 14,
        "rsi14" => 14,
        "bb20_z" | "bb20_bw" => 19,
        "vz96" => 95,
        "range1" | "clv1" | "body1" | "upper_wick1" | "lower_wick1" | "open_gap1" => 0,
        "dow_sin" | "dow_cos" | "hour_sin" | "hour_cos" | "month_sin" | "month_cos"
        | "dom_sin" | "dom_cos" => 0,
        "beta_btc_60" => 59,
        "btc_r20" | "rel_btc_r20" | "btc_vol20" => 20,
        "ema8" | "macd" | "macd_sig" | "macd_hist" => 0,
        "ema20" => 0,
        "ema50" => 0,
        "ema100" => 0,
        "ema200" => 0,
        "rsi6" => 6,
        "rsi28" => 28,
        "bb50" | "bb50_bw" => 49,
        "atr50" => 50,
        "stoch_k" | "stoch_d" => 13,
        "adx14" | "di_plus" | "di_minus" => 14,
        "obv_z96" | "vwap_dev96" => 95,
        "park16" | "gk16" => 15,
        "skew96" | "kurt96" => 96,
        "r192" => 192,
        _ => 0,
    }
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
    fn parse(s: &str) -> Result<Self, String> {
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
    pub gpu: GpuMode,
    pub allow_lockbox: bool,
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
            gpu: GpuMode::Auto,
            allow_lockbox: false,
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
    pub hash: String,
    pub label: bool,
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
    pub backend: String,
    pub gpu_requested: String,
    pub gpu_available: bool,
    pub gpu_device: Option<String>,
    pub read_seconds: f64,
    pub feature_seconds: f64,
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
                "# enrich: read={:.3}s feature={:.3}s xs={:.3}s write={:.3}s total={:.3}s peak_rss={:.1} MiB",
                self.read_seconds,
                self.feature_seconds,
                self.xs_seconds,
                self.write_seconds,
                self.total_seconds,
                self.peak_rss_bytes as f64 / (1024.0 * 1024.0)
            ),
        ];
        for p in &self.plans {
            out.push(format!(
                "# enrich-plan: group={} rows={} cols={} op={} cpu_ms={:.1} gpu_ms={:.1} backend={} reason={}",
                p.group, p.rows, p.cols, p.op, p.cpu_ms, p.gpu_ms, p.backend, p.reason
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
                    "{{\"name\":{},\"warmup\":{},\"nan_rows\":{},\"first_finite_row\":{},\"label\":{},\"sha256\":{}}}",
                    jstr(&f.name),
                    f.warmup,
                    f.nan_rows,
                    f.first_finite_row
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                    f.label,
                    jstr(&f.hash)
                )
            })
            .collect();
        let plans: Vec<String> = self
            .plans
            .iter()
            .map(|p| {
                format!(
                    "{{\"group\":{},\"rows\":{},\"columns\":{},\"op\":{},\"cpu_ms\":{:.3},\"gpu_ms\":{:.3},\"backend\":{},\"reason\":{}}}",
                    jstr(&p.group),
                    p.rows,
                    p.cols,
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
             \"timing\":{{\"read_s\":{:.6},\"feature_s\":{:.6},\"xs_s\":{:.6},\"write_s\":{:.6},\"total_s\":{:.6}}},\n  \
             \"peak_rss_bytes\":{},\n  \"features\":[{}],\n  \"plan\":[{}]\n}}\n",
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
            self.read_seconds,
            self.feature_seconds,
            self.xs_seconds,
            self.write_seconds,
            self.total_seconds,
            self.peak_rss_bytes,
            features.join(","),
            plans.join(",")
        )
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
    RollingSum,
    RollingStd,
    WilderScan,
    Rank,
    Join,
}

impl OpClass {
    pub fn name(self) -> &'static str {
        match self {
            OpClass::Elementwise => "elementwise",
            OpClass::RollingSum => "rolling-sum",
            OpClass::RollingStd => "rolling-std",
            OpClass::WilderScan => "wilder-scan",
            OpClass::Rank => "segmented-rank",
            OpClass::Join => "join",
        }
    }
    fn cpu_cells_per_second(self) -> f64 {
        match self {
            OpClass::Elementwise => 80_000_000.0,
            OpClass::RollingSum => 70_000_000.0,
            OpClass::RollingStd => 55_000_000.0,
            OpClass::WilderScan => 25_000_000.0,
            OpClass::Rank => 4_000_000.0,
            OpClass::Join => 40_000_000.0,
        }
    }
    fn gpu_cells_per_second(self) -> f64 {
        // Conservative f64 figures for the RTX 3060; f64 is 1/64 of f32 on
        // consumer Ampere, so these are deliberately far below a TFLOPS claim.
        match self {
            OpClass::Elementwise => 350_000_000.0,
            OpClass::RollingSum => 180_000_000.0,
            OpClass::RollingStd => 150_000_000.0,
            OpClass::WilderScan => 12_000_000.0,
            OpClass::Rank => 60_000_000.0,
            OpClass::Join => 250_000_000.0,
        }
    }
}

/// One family's dispatch decision and its estimated cost.
#[derive(Clone, Debug)]
pub struct GroupPlan {
    pub group: String,
    pub rows: usize,
    pub cols: usize,
    pub op: String,
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub backend: &'static str,
    pub reason: String,
}

/// Families in the catalogue, with the operation class they are costed as.
pub fn family_of(name: &str) -> &'static str {
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
    } else if name.starts_with("atr")
        || name.starts_with("natr")
        || name.starts_with("don")
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

/// True when this build has an exact enrichment kernel for the class.
///
/// The CPU path is the reference. A CUDA feature kernel would be registered
/// here only after it produced identical columns on the parity fixture; none
/// is registered in this increment, which is why the report always says CPU.
pub fn gpu_kernel_registered(_op: OpClass) -> bool {
    false
}

/// Compare estimated CPU and GPU time for each family.
///
/// GPU is chosen only when a kernel is registered, `gpu_available` is true,
/// the family has at least [`GPU_MIN_ROWS`] rows and the estimated GPU time
/// (transfer + kernel) is strictly lower than the CPU estimate.
pub const GPU_MIN_ROWS: usize = 1_000_000;
pub const GPU_TRANSFER_BYTES_PER_SECOND: f64 = 12_000_000_000.0;
pub const GPU_FIXED_OVERHEAD_MS: f64 = 1.5;

pub fn plan_groups(
    counts: &BTreeMap<&'static str, (usize, OpClass)>,
    n_rows: usize,
    gpu_requested: GpuMode,
    gpu_available: bool,
) -> Vec<GroupPlan> {
    let mut out = Vec::new();
    for (group, (cols, op)) in counts {
        let op = *op;
        let cells = n_rows.saturating_mul(*cols).max(1) as f64;
        let cpu_s = cells / op.cpu_cells_per_second();
        let bytes = n_rows.saturating_mul(*cols).max(1) as f64 * 8.0 * 2.0;
        let gpu_s = GPU_FIXED_OVERHEAD_MS / 1000.0
            + bytes / GPU_TRANSFER_BYTES_PER_SECOND
            + cells / op.gpu_cells_per_second();
        let registered = gpu_kernel_registered(op);
        let (backend, reason) = if gpu_requested == GpuMode::Off {
            ("cpu", "GPU disabled by --no-gpu/--cpu".to_string())
        } else if !gpu_available {
            ("cpu", "no usable CUDA device".to_string())
        } else if n_rows < GPU_MIN_ROWS {
            (
                "cpu",
                format!("{} rows is below the {} row GPU floor", n_rows, GPU_MIN_ROWS),
            )
        } else if !registered {
            ("cpu", "no exact GPU kernel registered for this family".to_string())
        } else if gpu_s < cpu_s {
            ("gpu", "estimated GPU time is below CPU".to_string())
        } else {
            ("cpu", "estimated GPU time is not below CPU".to_string())
        };
        out.push(GroupPlan {
            group: (*group).to_string(),
            rows: n_rows,
            cols: *cols,
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
                    trades: 0.0,
                    taker_buy_base: 0.0,
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
                        "symbol '{}' is not contiguous: it reappears after another symbol; \
                         sort the store by (symbol, time) or run `rcsr panel` first",
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
    Ok(Series {
        symbol: seg.symbol.clone(),
        times,
        open,
        high,
        low,
        close,
        volume,
        quote_volume,
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

/// Causal prefix sums for a rolling mean/std. Non-finite cells are skipped and
/// counted as missing, so a window with any missing cell is invalid.
struct Prefix {
    sum: Vec<f64>,
    sum2: Vec<f64>,
    count: Vec<usize>,
}

impl Prefix {
    fn new(x: &[f64]) -> Self {
        let mut sum = Vec::with_capacity(x.len() + 1);
        let mut sum2 = Vec::with_capacity(x.len() + 1);
        let mut count = Vec::with_capacity(x.len() + 1);
        sum.push(0.0);
        sum2.push(0.0);
        count.push(0);
        let (mut s, mut s2, mut c) = (0.0, 0.0, 0usize);
        for &v in x {
            if v.is_finite() {
                s += v;
                s2 += v * v;
                c += 1;
            }
            sum.push(s);
            sum2.push(s2);
            count.push(c);
        }
        Prefix { sum, sum2, count }
    }
    fn mean(&self, i: usize, w: usize) -> Option<f64> {
        if i + 1 < w {
            return None;
        }
        let lo = i + 1 - w;
        if self.count[i + 1] - self.count[lo] < w {
            return None;
        }
        Some((self.sum[i + 1] - self.sum[lo]) / w as f64)
    }
    fn std(&self, i: usize, w: usize) -> Option<f64> {
        let m = self.mean(i, w)?;
        let lo = i + 1 - w;
        let s2 = (self.sum2[i + 1] - self.sum2[lo]) / w as f64;
        Some((s2 - m * m).max(0.0).sqrt())
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
        tr[i] = if i == 0 || !s.close[i - 1].is_finite() {
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
    let p = Prefix::new(x);
    for i in w - 1..n {
        if !contig(times, i, w, step) {
            continue;
        }
        let lo = i + 1 - w;
        let cnt = p.count[i + 1] - p.count[lo];
        if cnt > 0 {
            out[i] = (p.sum[i + 1] - p.sum[lo]) / cnt as f64;
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
    let p = Prefix::new(x);
    for i in w - 1..n {
        if !contig(times, i, w, step) {
            continue;
        }
        let lo = i + 1 - w;
        let cnt = p.count[i + 1] - p.count[lo];
        if cnt < w {
            continue;
        }
        let m = (p.sum[i + 1] - p.sum[lo]) / w as f64;
        let s2 = (p.sum2[i + 1] - p.sum2[lo]) / w as f64;
        out[i] = (s2 - m * m).max(0.0).sqrt();
    }
    out
}

fn rsi_py(close: &[f64], period: usize, gap: &[bool]) -> Vec<f64> {
    let n = close.len();
    let mut d = vec![0.0f64; n];
    for i in 1..n {
        if close[i].is_finite() && close[i - 1].is_finite() {
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
    let p = Prefix::new(&s.close);
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
    let p = Prefix::new(&b.ret1);
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
    let p = Prefix::new(&s.close);
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
    let p = Prefix::new(&x);
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
    (0..s.len())
        .map(|i| {
            let r = s.high[i] - s.low[i];
            if r > 0.0 {
                ((s.close[i] - s.low[i]) - (s.high[i] - s.close[i])) / r
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn donchian(s: &Series, w: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; s.len()];
    for i in 0..s.len() {
        if !contig(&s.times, i, w, s.step) {
            continue;
        }
        let lo = i + 1 - w;
        let mut hi = f64::NEG_INFINITY;
        let mut low = f64::INFINITY;
        for j in lo..=i {
            hi = hi.max(s.high[j]);
            low = low.min(s.low[j]);
        }
        if hi > low && low.is_finite() {
            out[i] = (s.close[i] - low) / (hi - low);
        }
    }
    out
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
    r20: BTreeMap<i64, f64>,
    vol20: BTreeMap<i64, f64>,
    r16: BTreeMap<i64, f64>,
    r96: BTreeMap<i64, f64>,
    vol96: BTreeMap<i64, f64>,
}

impl Cross {
    fn from_series(s: &Series) -> Self {
        let b = Base::new(s);
        let r1 = ret_k(s, 1);
        let r20 = ret_k(s, 20);
        let vol20 = vol_w(s, &b, 20);
        let r16 = ret_k(s, 16);
        let r96 = ret_k(s, 96);
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
            r20: map(&r20),
            vol20: map(&vol20),
            r16: map(&r16),
            r96: map(&r96),
            vol96: map(&vol96),
        }
    }
    fn at(map: &BTreeMap<i64, f64>, t: i64) -> f64 {
        map.get(&t).copied().unwrap_or(f64::NAN)
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
            out[i] = cov / var;
        }
    }
    out
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
    out
}

// ---------------------------------------------------------------------------
// Temp column writer
// ---------------------------------------------------------------------------

struct TempColumn {
    name: String,
    path: PathBuf,
    writer: Option<BufWriter<std::fs::File>>,
    hash: Sha256Hasher,
    warmup: usize,
    nan_rows: usize,
    first_finite: Option<usize>,
    label: bool,
    row_offset: usize,
}

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
            hash: Sha256Hasher::new(),
            warmup,
            nan_rows: 0,
            first_finite: None,
            label,
            row_offset: 0,
        })
    }
    fn write_values(&mut self, values: &[f64]) -> Result<(), String> {
        let w = self.writer.as_mut().ok_or("temp column already closed")?;
        let mut buf = [0u8; 8];
        for &v in values {
            if !v.is_finite() {
                self.nan_rows += 1;
            } else if self.first_finite.is_none() {
                self.first_finite = Some(self.row_offset);
            }
            buf.copy_from_slice(&v.to_le_bytes());
            w.write_all(&buf)
                .map_err(|e| format!("{}: {}", self.path.display(), e))?;
            self.hash.update(&buf);
            self.row_offset += 1;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()
                .map_err(|e| format!("{}: {}", self.path.display(), e))?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<FeatureReport, String> {
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
    labels: Vec<(String, LabelSpec)>,
    tf: Option<Timeframe>,
    cross: &'a Cross,
}

fn compute_series(
    s: &Series,
    ctx: &Context,
    cols: &mut BTreeMap<String, TempColumn>,
) -> Result<(), String> {
    let base = Base::new(s);
    let bars = if !ctx.legacy.is_empty() || !ctx.labels.is_empty() {
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
    let needs_btc = ctx
        .own
        .iter()
        .any(|n| n.starts_with("btc") || n.starts_with("rel_btc") || n.starts_with("beta_btc") || n.starts_with("cross_"));
    let btc_r1 = if needs_btc {
        Some(
            (0..s.len())
                .map(|i| Cross::at(&ctx.cross.r1, s.times[i]))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    for name in &ctx.own {
        let (values, warmup) = compute_own(name, s, &base, ctx, btc_r1.as_deref())?;
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
    }
    Ok(())
}

fn compute_own(
    name: &str,
    s: &Series,
    b: &Base,
    ctx: &Context,
    btc_r1: Option<&[f64]>,
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
            "vz96" => vz(s, 96),
            "clv1" => clv1(s),
            "range1" => range1(s),
            "dow_sin" | "dow_cos" | "hour_sin" | "hour_cos" | "month_sin" | "month_cos"
            | "dom_sin" | "dom_cos" => calendar_column(s, name),
            "btc_r1" | "rel_btc_r1" | "cross_ret1" => {
                let br = btc_r1.ok_or("BTC features need a cross series")?;
                (0..n)
                    .map(|i| {
                        if name == "rel_btc_r1" {
                            if b.ret1[i].is_finite() && br[i].is_finite() {
                                b.ret1[i] - br[i]
                            } else {
                                f64::NAN
                            }
                        } else {
                            br[i]
                        }
                    })
                    .collect()
            }
            "btc_r20" | "rel_btc_r20" => {
                let r20 = ret_k(s, 20);
                (0..n)
                    .map(|i| {
                        let m = Cross::at(&ctx.cross.r20, s.times[i]);
                        if name == "rel_btc_r20" {
                            if r20[i].is_finite() && m.is_finite() {
                                r20[i] - m
                            } else {
                                f64::NAN
                            }
                        } else {
                            m
                        }
                    })
                    .collect()
            }
            "btc_vol20" => (0..n)
                .map(|i| Cross::at(&ctx.cross.vol20, s.times[i]))
                .collect(),
            "beta_btc_60" | "cross_beta20" => {
                let w = if name == "cross_beta20" { 20 } else { 60 };
                if s.symbol == ctx.cross.symbol {
                    (0..n)
                        .map(|i| {
                            if i + 1 >= w && contig(&s.times, i, w, s.step) {
                                1.0
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                } else {
                    beta_window(s, &b.ret1, &ctx.cross.r1, w)
                }
            }
            other => return Err(format!("unknown feature '{}'", other)),
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
/// The result is `(rank - 1) / (n_finite - 1)` in `[0, 1]`, with ties broken by
/// symbol name and then row order. A timestamp with fewer than two finite
/// values, or a non-finite source value, is `NaN`. Ranks are exact rational
/// values rounded once to f64, so repeats are bit-stable.
pub fn xs_rank(
    src: &[f64],
    times: &[i64],
    sym_key: &[u32],
    rows: usize,
) -> Result<Vec<f64>, String> {
    if src.len() != rows || times.len() != rows || sym_key.len() != rows {
        return Err("cross-sectional inputs have inconsistent lengths".to_string());
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
    let mut out = vec![f64::NAN; rows];
    let mut i = 0usize;
    while i < order.len() {
        let t = times[order[i] as usize];
        let mut j = i;
        while j < order.len() && times[order[j] as usize] == t {
            j += 1;
        }
        let mut group: Vec<usize> = (i..j).map(|k| order[k] as usize).collect();
        group.retain(|&r| src[r].is_finite());
        group.sort_unstable_by(|&a, &b| {
            src[a]
                .partial_cmp(&src[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(sym_key[a].cmp(&sym_key[b]))
                .then(a.cmp(&b))
        });
        let m = group.len();
        if m >= 2 {
            for (rank, &r) in group.iter().enumerate() {
                out[r] = rank as f64 / (m - 1) as f64;
            }
        }
        i = j;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

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
    let mut legacy = Vec::new();
    let mut own = Vec::new();
    let mut py15m_requested = Vec::new();
    for n in &requested {
        if matches!(
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
        } else if n.starts_with("xs_rank_") {
            // Computed in the cross-sectional pass, which needs every symbol.
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

    // Labels.
    let mut labels: Vec<(String, LabelSpec)> = Vec::new();
    for spec_src in &opts.labels {
        let tf = tf.ok_or_else(|| {
            format!(
                "label '{}' needs a recognised bar spacing (got {}s); pass --bar-seconds",
                spec_src, step
            )
        })?;
        let spec = LabelSpec::parse(spec_src, tf)?;
        let base = format!("fwd{}", spec.horizon);
        let mut name = base.clone();
        let mut k = 2;
        while labels.iter().any(|(n, _)| n == &name) || requested.contains(&name) {
            name = format!("{}_{}", base, k);
            k += 1;
        }
        labels.push((name, spec));
    }

    // Cross-market context: the benchmark symbol's own series.
    let cross_seg = segs
        .iter()
        .find(|s| s.symbol == opts.cross_symbol)
        .cloned()
        .or_else(|| {
            // An external cross store is optional; use it when the base store
            // does not carry the benchmark.
            None
        });
    let cross = match cross_seg {
        Some(seg) => Cross::from_series(&load_series(&store, &seg, &cols, step)?),
        None => match &opts.cross_data {
            Some(path) => {
                let cs = FileStore::open(path)?;
                let cnames = cs.header().names();
                let ccols = Cols {
                    time: find_time_col(&cnames, None).ok_or("cross store has no time column")?,
                    time_dtype: cs.header().dtypes()[find_time_col(&cnames, None).unwrap()],
                    open: find_col(&cnames, &["open", "o"]).ok_or("cross store has no open")?,
                    high: find_col(&cnames, &["high", "h"]).ok_or("cross store has no high")?,
                    low: find_col(&cnames, &["low", "l"]).ok_or("cross store has no low")?,
                    close: find_col(&cnames, &["close", "c"]).ok_or("cross store has no close")?,
                    volume: find_col(&cnames, &["volume", "vol", "v"]),
                    quote_volume: find_col(&cnames, &["quote_volume", "quotevolume", "qv"]),
                    symbol: find_symbol_col(&cnames, None),
                };
                let (csegs, _) = segments(&cs, ccols.time, ccols.symbol)?;
                let cseg = csegs
                    .iter()
                    .find(|s| s.symbol == opts.cross_symbol)
                    .or_else(|| csegs.first())
                    .ok_or("cross store has no rows")?;
                Cross::from_series(&load_series(&cs, cseg, &ccols, step)?)
            }
            None => Cross::default(),
        },
    };

    let needs_cross = requested.iter().any(|n| {
        n.starts_with("btc")
            || n.starts_with("rel_btc")
            || n.starts_with("beta_btc")
            || n.starts_with("cross_")
            || n.starts_with("mkt_")
            || n == "rel_r16"
            || n == "rel_r96"
    });
    if needs_cross && cross.symbol.is_empty() {
        return Err(format!(
            "btc/cross features need the benchmark series '{}' in --data or a --cross-data store",
            opts.cross_symbol
        ));
    }

    // GPU plan.
    let gpu = crate::backend::probe();
    let (gpu_available, gpu_device) = (gpu.available(), gpu.device.clone());
    let mut fam_counts: BTreeMap<&'static str, (usize, OpClass)> = BTreeMap::new();
    let classify = |name: &str| -> OpClass {
        if name.starts_with("xs_") {
            OpClass::Rank
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
            || name.starts_with("mkt_")
            || name.starts_with("cross_")
        {
            OpClass::Join
        } else {
            OpClass::Elementwise
        }
    };
    for n in requested.iter() {
        let fam = family_of(n);
        let e = fam_counts.entry(fam).or_insert((0, classify(n)));
        e.0 += 1;
    }
    if !py15m_requested.is_empty() {
        let e = fam_counts.entry("py15m").or_insert((0, OpClass::Elementwise));
        e.0 += py15m_requested.len();
    }
    let plans = plan_groups(&fam_counts, store.n_rows(), opts.gpu, gpu_available);

    // Input checksum (the same file is copied, not rebuilt).
    let input_sha = manifest::sha256_file(&opts.data)
        .or_else(|_| manifest::sha256_file(&input_path))
        .unwrap_or_else(|_| "unknown".to_string());

    // Lockbox guard for labelled builds: a label is an evaluation target.
    if !labels.is_empty() {
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

    // Cross-sectional source columns are computed even when the user did not
    // ask for them, then removed if they are not part of the output.
    let mut xs_sources: Vec<String> = Vec::new();
    for n in &requested {
        let src = if n == "xs_rank_r1" {
            "r1"
        } else if n == "xs_rank_r20" {
            "r20"
        } else if n == "xs_rank_vol20" {
            "vol20"
        } else {
            continue;
        };
        xs_sources.push(src.to_string());
        if !requested.contains(&src.to_string()) {
            temp.entry(src.to_string())
                .or_insert(TempColumn::create(work_dir, src, nominal_warmup(src), false)?);
        }
    }
    for src in &xs_sources {
        if !requested.contains(src) && !own.contains(src) {
            own.push(src.clone());
        }
    }

    let ctx = Context {
        legacy,
        own,
        py15m_requested,
        labels: labels.clone(),
        tf,
        cross: &cross,
    };

    // Feature pass, one symbol at a time in row order. `--chunk-rows` bounds
    // the read/compute batch; it never changes arithmetic or output order.
    let t_feat = Instant::now();
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
            compute_series(&s, &ctx, &mut temp)?;
        }
    }
    let feature_seconds = t_feat.elapsed().as_secs_f64();

    // The cross-sectional pass reads the source columns back, so flush the
    // buffered writers first (the files are row-complete at this point).
    for t in temp.values_mut() {
        t.flush()?;
    }

    // Cross-sectional pass.
    let t_xs = Instant::now();
    if !xs_sources.is_empty() {
        let mut times = Vec::with_capacity(store.n_rows());
        for seg in &segs {
            times.extend(segment_i64(&store, seg, time_col, cols.time_dtype)?);
        }
        let mut sym_key = vec![0u32; store.n_rows()];
        for seg in &segs {
            for r in seg.start..seg.start + seg.len {
                sym_key[r] = seg.key;
            }
        }
        for (rank_name, src) in requested
            .iter()
            .filter(|n| n.starts_with("xs_rank_"))
            .zip(xs_sources.iter())
        {
            let src_path = temp
                .get(src)
                .ok_or_else(|| format!("missing source column '{}'", src))?
                .path
                .clone();
            let values = read_f64_file(&src_path, store.n_rows())?;
            let ranks = xs_rank(&values, &times, &sym_key, store.n_rows())?;
            if let Some(t) = temp.get_mut(rank_name) {
                t.write_values(&ranks)?;
            }
        }
        // Remove sources that are not requested output columns.
        for src in &xs_sources {
            if !requested.contains(src) {
                if let Some(t) = temp.remove(src) {
                    let _ = std::fs::remove_file(t.path);
                }
            }
        }
    }
    let xs_seconds = t_xs.elapsed().as_secs_f64();

    // Finish temp columns and build the append list in the requested order.
    let mut reports: Vec<FeatureReport> = Vec::new();
    let mut extras: Vec<AppendColumn> = Vec::new();
    let order: Vec<String> = requested
        .iter()
        .chain(labels.iter().map(|(n, _)| n))
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
        .cloned()
        .collect();
    reports.sort_by_key(|r| order.iter().position(|n| n == &r.name).unwrap_or(usize::MAX));

    if opts.dry_run {
        for e in &extras {
            let _ = std::fs::remove_file(&e.file);
        }
        let mut report = Report {
            input: opts.data.clone(),
            output: out.to_string(),
            input_sha256: input_sha,
            output_sha256: String::new(),
            output_bytes: 0,
            rows: store.n_rows(),
            columns_in: store.n_cols(),
            columns_out: expected_columns_out(&store, &requested, &labels, opts.replace),
            groups: store.n_groups(),
            symbols: segs.len(),
            group_rows: store.group_rows(),
            chunk_rows: opts.chunk_rows,
            bar_seconds: step,
            features: reports,
            plans,
            backend: "cpu".to_string(),
            gpu_requested: format!("{:?}", opts.gpu).to_lowercase(),
            gpu_available,
            gpu_device,
            read_seconds,
            feature_seconds,
            xs_seconds,
            write_seconds: 0.0,
            total_seconds: started.elapsed().as_secs_f64(),
            peak_rss_bytes: peak_rss_bytes(),
        };
        report.backend = if report.plans.iter().any(|p| p.backend == "gpu") { "gpu" } else { "cpu" }.to_string();
        return Ok(report);
    }

    // Append pass: copy existing blocks and write the new columns.
    let t_write = Instant::now();
    let tmp_out = work_dir.join("out.rcsb").to_string_lossy().into_owned();
    store::write_store_appending(&tmp_out, &store, &extras)?;
    std::fs::rename(&tmp_out, out).map_err(|e| format!("{}: {}", out, e))?;
    let write_seconds = t_write.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    let output_sha = manifest::sha256_file(out).unwrap_or_else(|_| "unknown".to_string());

    let columns_out = FileStore::open(out)?.n_cols();
    let backend = if plans.iter().any(|p| p.backend == "gpu") { "gpu" } else { "cpu" };
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
        backend: backend.to_string(),
        gpu_requested: format!("{:?}", opts.gpu).to_lowercase(),
        gpu_available,
        gpu_device,
        read_seconds,
        feature_seconds,
        xs_seconds,
        write_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
        peak_rss_bytes: peak_rss_bytes(),
    })
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
    replace: bool,
) -> usize {
    let mut names: Vec<String> = store.header().names();
    for n in requested.iter().chain(labels.iter().map(|(n, _)| n)) {
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
            "--label" => opts.labels.push(val!()),
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
  --label <spec>          fwd:<tf|bars>,cost=<bp>,volnorm=<bars>,stride=<n>
                          (repeatable; labels are targets, not features)
  --allow-lockbox         permit labelled rows in 2026H2 (final test only)
  --gpu [auto|on|off]     dispatch policy; CPU is always available and is the
                          only arm in this increment (the cost model reports
                          where a registered exact GPU kernel would win)
  --no-gpu, --cpu         force CPU
  --dry-run               print the report and write nothing
  --json                  print the JSON report on stdout

FEATURE GROUPS:
  returns    r1..r120, log close-to-close over exactly k contiguous bars
  ma         ma5 ma10 ma20 ma50 ma100 ma200, close / trailing mean - 1
  vol        vol5 vol20 vol60 vol120, population std of log returns
  atr        atr14 (price units), natr14 (percent)
  rsi        rsi14 (0..100)
  bollinger  bb20_z, bb20_bw
  vz         vz96, z-score of log1p(volume)
  range      range1, (high - low) / close
  candle     clv1 body1 upper_wick1 lower_wick1 open_gap1
  donchian   don20 don96, close position in the trailing high/low channel
  calendar   dow_sin dow_cos hour_sin hour_cos month_sin month_cos dom_sin dom_cos
  lags       lag_r1_1 .. lag_r1_10
  xs         xs_rank_r1 xs_rank_r20 xs_rank_vol20 (per-timestamp cross-section)
  btc        btc_r1 btc_r20 btc_vol20 rel_btc_r1 rel_btc_r20 beta_btc_60
  py15m      the 35 columns added by the Python enrich_panel_15m.py step
  all        every group above except py15m

  #125 names and groups (ta:core, ta:full, exotic:cheap, candle:geometry) are
  also accepted and are computed by `rcsr::features`, not duplicated here.

The output store checksum and every feature column hash are written to
<out>.enrich.json. See docs/ohlcv.md for the catalogue, gap and warm-up rules.
"
    .to_string()
}

