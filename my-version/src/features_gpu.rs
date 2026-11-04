//! Fused flow and price-action device producers (#297).
//!
//! The CPU producers `enrich::flow_all` and `enrich::price_action_all` are the
//! reference. This module names the device columns, their parity tier, the raw
//! columns each one consumes, and the measured fused cost model. The CUDA
//! kernels live in `src/cuda/features.cu`; they reuse the resident
//! [`crate::enrich_gpu::EnrichGpu`] context, so `open/high/low/close/volume/
//! quote_volume/trades/taker_buy_base` are uploaded once per run.
//!
//! The flow registry family has 17 columns: the five raw flow inputs
//! (`volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`),
//! the ten canonical `FLOW_NAMES`, and the two engine columns `cmf20` and
//! `vwap20_dev`. The five raw inputs are pass-through context columns, not
//! producer outputs; the fused producer materialises the remaining twelve.
//! `price_action` has eleven materialised columns.
//!
//! Parity tiers are declared per column, not per family:
//!
//! * `bitwise` — arithmetic only; CUDA is compiled with `-fmad=false` and IEEE
//!   `sqrt`, so the bytes match the CPU;
//! * `1e-12` — the column takes a logarithm (`trade_intensity96`,
//!   `avg_trade_size96`, `amihud96`); host libm and CUDA `log` may differ in the
//!   last bit.
//!
//! `zero_ret_share96` stays bitwise because it tests the exact close ratio
//! rather than a rounded logarithm: `log(C_t/C_{t-1}) == 0` holds exactly when
//! the ratio is `1.0`.

/// Optional raw-column presence bits shared with `src/cuda/enrich_common.h`.
pub const PRESENT_VOLUME: u32 = 1 << 0;
pub const PRESENT_QUOTE_VOLUME: u32 = 1 << 1;
pub const PRESENT_TRADES: u32 = 1 << 2;
pub const PRESENT_TAKER_BUY: u32 = 1 << 3;
pub const PRESENT_TAKER_BUY_QV: u32 = 1 << 4;

/// Parity tier of one registered column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Bytes match the CPU reference exactly.
    Bitwise,
    /// Absolute-or-relative agreement within `1e-12`.
    Tolerance1e12,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Bitwise => "bitwise",
            Tier::Tolerance1e12 => "1e-12",
        }
    }
}

/// One device-produced column: name, parity tier, raw columns it needs.
pub struct DeviceColumn {
    pub name: &'static str,
    pub tier: Tier,
    pub requires: &'static [&'static str],
}

/// The twelve derived flow columns produced by the fused device pass, in the
/// C kernel's output order. The five raw flow inputs and the two pass-through
/// columns are not produced here.
pub const FLOW_DEVICE_COLUMNS: &[DeviceColumn] = &[
    DeviceColumn { name: "vwap", tier: Tier::Bitwise, requires: &["volume", "quote_volume"] },
    DeviceColumn { name: "taker_ratio96", tier: Tier::Bitwise, requires: &["volume", "taker_buy_base"] },
    DeviceColumn { name: "taker_imb16", tier: Tier::Bitwise, requires: &["volume", "taker_buy_base"] },
    DeviceColumn { name: "trade_intensity96", tier: Tier::Tolerance1e12, requires: &["volume", "trades"] },
    DeviceColumn { name: "avg_trade_size96", tier: Tier::Tolerance1e12, requires: &["quote_volume", "trades"] },
    DeviceColumn { name: "amihud96", tier: Tier::Tolerance1e12, requires: &["quote_volume"] },
    DeviceColumn { name: "zero_ret_share96", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "obv_z96", tier: Tier::Bitwise, requires: &["volume"] },
    DeviceColumn { name: "cvd_proxy", tier: Tier::Bitwise, requires: &["volume", "taker_buy_base"] },
    DeviceColumn { name: "vwap_dev96", tier: Tier::Bitwise, requires: &["volume", "quote_volume"] },
    DeviceColumn { name: "cmf20", tier: Tier::Bitwise, requires: &["volume"] },
    DeviceColumn { name: "vwap20_dev", tier: Tier::Bitwise, requires: &["volume", "quote_volume"] },
];

/// The eleven price-action columns, in the C kernel's output order.
pub const PRICE_ACTION_DEVICE_COLUMNS: &[DeviceColumn] = &[
    DeviceColumn { name: "clv16", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "body16", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "rng_atr", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "prev_range_pos", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "inside_s", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "outside_s", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "engulf_s", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "harami_s", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "run8", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "nr4", tier: Tier::Bitwise, requires: &[] },
    DeviceColumn { name: "trend96", tier: Tier::Bitwise, requires: &[] },
];

/// True when the fused flow producer materialises `name`.
pub fn is_device_flow(name: &str) -> bool {
    flow_index(name).is_some()
}

/// True when the fused price-action producer materialises `name`.
pub fn is_device_price_action(name: &str) -> bool {
    price_action_index(name).is_some()
}

/// Device output index of a flow column.
pub fn flow_index(name: &str) -> Option<usize> {
    FLOW_DEVICE_COLUMNS.iter().position(|c| c.name == name)
}

/// Device output index of a price-action column.
pub fn price_action_index(name: &str) -> Option<usize> {
    PRICE_ACTION_DEVICE_COLUMNS.iter().position(|c| c.name == name)
}

/// Declared parity tier for a fused device column.
pub fn device_tier(name: &str) -> Option<&'static str> {
    FLOW_DEVICE_COLUMNS
        .iter()
        .chain(PRICE_ACTION_DEVICE_COLUMNS.iter())
        .find(|c| c.name == name)
        .map(|c| c.tier.as_str())
}

/// Raw columns a fused device column consumes (used by the parity fixture and
/// the registry).
pub fn device_requires(name: &str) -> Option<&'static [&'static str]> {
    FLOW_DEVICE_COLUMNS
        .iter()
        .chain(PRICE_ACTION_DEVICE_COLUMNS.iter())
        .find(|c| c.name == name)
        .map(|c| c.requires)
}

/// Number of base kernels a fused family launches, independent of how many
/// output columns it produces. Flow = elementwise base + cumulative state;
/// price-action = geometry base + Wilder ATR. This is the measured `~6 vs ~21`
/// pass reduction from `04-feature-enrich.md` §3.7.
pub fn base_passes(family: &str) -> usize {
    match family {
        "flow" | "price_action" => 2,
        _ => 0,
    }
}

/// Measured fused cost of one family on this machine. The base pass is paid
/// once per run; each additional output column costs its consumer scan. This
/// replaces the linear `columns x cells` estimate for the two fused families.
#[derive(Clone, Copy, Debug)]
pub struct FusedCost {
    /// Measured milliseconds per million rows for the fused base kernels.
    pub base_ms_per_mrow: f64,
    /// Measured milliseconds per million rows per output column.
    pub column_ms_per_mrow: f64,
    /// Measured CPU reference cost per million rows per output column. The CPU
    /// producer has no fused base: every column recomputes its inputs.
    pub cpu_ms_per_mrow_column: f64,
}

/// Fused flow cost. Measured by `tests/features_gpu.rs` on a 1,000,075-row,
/// 109-symbol 15 m fixture on an RTX 3060; the segment-parallel consumers only
/// fill the device at a realistic symbol count. Re-measure and update both the
/// constant and `docs/ohlcv.md` when the kernels change.
pub const FLOW_COST: FusedCost = FusedCost {
    base_ms_per_mrow: 11.4,
    column_ms_per_mrow: 3.3,
    cpu_ms_per_mrow_column: 110.0,
};

/// Fused price-action cost, measured on the same fixture.
pub const PRICE_ACTION_COST: FusedCost = FusedCost {
    base_ms_per_mrow: 8.8,
    column_ms_per_mrow: 2.5,
    cpu_ms_per_mrow_column: 110.0,
};

/// Cost model lookup for a fused family.
pub fn fused_cost(family: &str) -> Option<FusedCost> {
    match family {
        "flow" => Some(FLOW_COST),
        "price_action" => Some(PRICE_ACTION_COST),
        _ => None,
    }
}

/// Estimated fused-family device time in milliseconds (base + consumers).
pub fn estimate_fused_ms(family: &str, rows: usize, columns: usize) -> Option<f64> {
    let cost = fused_cost(family)?;
    let mrow = rows as f64 / 1_000_000.0;
    Some(cost.base_ms_per_mrow * mrow + cost.column_ms_per_mrow * mrow * columns as f64)
}

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_int, c_void};

    extern "C" {
        pub fn rc_features_flow_base(ctx: *mut c_void) -> c_int;
        pub fn rc_features_flow_col(ctx: *mut c_void, which: c_int, out: *mut f64) -> c_int;
        pub fn rc_features_price_base(ctx: *mut c_void) -> c_int;
        pub fn rc_features_price_col(ctx: *mut c_void, which: c_int, out: *mut f64) -> c_int;
        pub fn rc_features_base_reads(ctx: *mut c_void) -> c_int;
        #[allow(dead_code)]
        pub fn rc_features_base_passes(ctx: *mut c_void) -> c_int;
    }
}

#[cfg(feature = "cuda")]
impl crate::enrich_gpu::EnrichGpu {
    /// Run the fused flow base once and download the requested columns.
    ///
    /// The base pass reads `trades` and `taker_buy_base` once; requesting more
    /// consumers does not add raw reads. Returns `None` if a name is not a
    /// device flow column or any kernel fails, so the caller can fall back to
    /// the CPU reference for the whole family.
    pub fn flow_all(&mut self, names: &[&str]) -> Option<Vec<(String, Vec<f64>)>> {
        let indices: Vec<usize> = names
            .iter()
            .map(|n| flow_index(n))
            .collect::<Option<Vec<_>>>()?;
        let started = std::time::Instant::now();
        let ctx = self.raw();
        let rc = unsafe { ffi::rc_features_flow_base(ctx) };
        self.record_fused(started, rc == 0);
        if rc != 0 {
            return None;
        }
        let mut out = Vec::with_capacity(names.len());
        for (name, idx) in names.iter().zip(indices) {
            let started = std::time::Instant::now();
            let mut buf = self.blank();
            let rc = unsafe { ffi::rc_features_flow_col(ctx, idx as i32, buf.as_mut_ptr()) };
            self.record_fused(started, rc == 0);
            if rc != 0 {
                return None;
            }
            out.push(((*name).to_string(), buf));
        }
        Some(out)
    }

    /// Run the fused price-action base (geometry + Wilder ATR14) once and
    /// download the requested columns.
    pub fn price_action_all(&mut self, names: &[&str]) -> Option<Vec<(String, Vec<f64>)>> {
        let indices: Vec<usize> = names
            .iter()
            .map(|n| price_action_index(n))
            .collect::<Option<Vec<_>>>()?;
        let started = std::time::Instant::now();
        let ctx = self.raw();
        let rc = unsafe { ffi::rc_features_price_base(ctx) };
        self.record_fused(started, rc == 0);
        if rc != 0 {
            return None;
        }
        let mut out = Vec::with_capacity(names.len());
        for (name, idx) in names.iter().zip(indices) {
            let started = std::time::Instant::now();
            let mut buf = self.blank();
            let rc = unsafe { ffi::rc_features_price_col(ctx, idx as i32, buf.as_mut_ptr()) };
            self.record_fused(started, rc == 0);
            if rc != 0 {
                return None;
            }
            out.push(((*name).to_string(), buf));
        }
        Some(out)
    }

    /// Counted reads of the optional raw flow columns in the fused base pass.
    #[allow(dead_code)]
    pub fn base_reads(&self) -> usize {
        unsafe { ffi::rc_features_base_reads(self.raw()) as usize }
    }

    /// Counted base kernel launches (flow + price-action).
    #[allow(dead_code)]
    pub fn base_passes(&self) -> usize {
        unsafe { ffi::rc_features_base_passes(self.raw()) as usize }
    }
}
