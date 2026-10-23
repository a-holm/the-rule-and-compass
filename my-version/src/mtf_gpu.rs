//! Device multi-timeframe resample and causal as-of join (issue #296).
//!
//! The CPU `enrich::mtf_all` columns are the reference. [`compute`] uploads the
//! resident base columns once, then for each requested timeframe resamples the
//! base rows into `floor(t/tf)` buckets, evaluates the eleven value leaves on
//! the higher series and joins them back with a monotone causal cursor. The
//! kernels live in `src/cuda/mtf.cu`; this module only owns the context and the
//! statistics ledger.
//!
//! A partial or current higher bucket is never visible (`higher.time + tf <=
//! base.time`), and a stale bucket is `NaN` for every value leaf while
//! `bar_age` remains the exact staleness in seconds.

use crate::enrich::{MTF_LEAVES, MTF_TIMEFRAMES};
use crate::gpu_stats::GpuStats;
use crate::ohlcv::Timeframe;
use std::collections::BTreeMap;
use std::os::raw::c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawStats {
    kernel_ms: f64,
    launch_ms: f64,
    sync_ms: f64,
    alloc_ms: f64,
    h2d_ms: f64,
    d2h_ms: f64,
    wall_ms: f64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    batch: i32,
    failed: i32,
}

enum Raw {}

extern "C" {
    fn rc_mtf_create(
        times: *const i64,
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        n: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        nseg: c_int,
        step: i64,
    ) -> *mut Raw;
    fn rc_mtf_destroy(ctx: *mut Raw);
    fn rc_mtf_last_stats(ctx: *mut Raw, out: *mut RawStats);
    fn rc_mtf_prepare(ctx: *mut Raw, tf_seconds: i64) -> c_int;
    fn rc_mtf_gather(ctx: *mut Raw, leaf: c_int, out: *mut f64) -> c_int;
}

/// One resident MTF context. It is per enrichment run, not per column.
pub struct MtfGpu {
    ctx: *mut Raw,
    n: usize,
    stats: GpuStats,
}

impl MtfGpu {
    /// Upload the base columns and segment table once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        times: &[i64],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        seg_start: &[u32],
        seg_len: &[u32],
        step: i64,
    ) -> Option<Self> {
        if times.is_empty()
            || seg_start.is_empty()
            || seg_start.len() != seg_len.len()
            || open.len() != times.len()
            || high.len() != times.len()
            || low.len() != times.len()
            || close.len() != times.len()
        {
            return None;
        }
        let starts: Vec<c_int> = seg_start.iter().map(|v| *v as c_int).collect();
        let lens: Vec<c_int> = seg_len.iter().map(|v| *v as c_int).collect();
        let ctx = unsafe {
            rc_mtf_create(
                times.as_ptr(),
                open.as_ptr(),
                high.as_ptr(),
                low.as_ptr(),
                close.as_ptr(),
                times.len() as c_int,
                starts.as_ptr(),
                lens.as_ptr(),
                starts.len() as c_int,
                step,
            )
        };
        if ctx.is_null() {
            return None;
        }
        let mut g = MtfGpu {
            ctx,
            n: times.len(),
            stats: GpuStats::default(),
        };
        g.record_creation();
        Some(g)
    }

    fn record_creation(&mut self) {
        let mut raw = RawStats::default();
        unsafe { rc_mtf_last_stats(self.ctx, &mut raw) };
        self.stats.add_batch(
            raw.kernel_ms,
            raw.launch_ms,
            raw.sync_ms,
            raw.alloc_ms,
            raw.h2d_ms,
            raw.d2h_ms,
            0.0,
            raw.wall_ms.max(raw.alloc_ms + raw.h2d_ms),
            raw.h2d_bytes,
            raw.d2h_bytes,
            0,
            self.n as u64,
            raw.failed != 0,
        );
    }

    fn record(&mut self, started: std::time::Instant, ok: bool) {
        let mut raw = RawStats::default();
        unsafe { rc_mtf_last_stats(self.ctx, &mut raw) };
        let outer_ms = started.elapsed().as_secs_f64() * 1e3;
        let windows = raw.alloc_ms + raw.h2d_ms + raw.launch_ms + raw.sync_ms + raw.d2h_ms;
        self.stats.add_batch(
            raw.kernel_ms,
            raw.launch_ms,
            raw.sync_ms,
            raw.alloc_ms,
            raw.h2d_ms,
            raw.d2h_ms,
            (outer_ms - windows).max(0.0),
            outer_ms.max(raw.wall_ms),
            raw.h2d_bytes,
            raw.d2h_bytes,
            0,
            self.n as u64,
            !ok || raw.failed != 0,
        );
    }

    /// Bucketise, resample and evaluate all eleven leaves for one timeframe.
    pub fn prepare(&mut self, tf_seconds: i64) -> bool {
        let started = std::time::Instant::now();
        let rc = unsafe { rc_mtf_prepare(self.ctx, tf_seconds) };
        let ok = rc == 0;
        self.record(started, ok);
        ok
    }

    /// Copy one base-length leaf (0..10) or `bar_age` (11) out of the context.
    pub fn gather(&mut self, leaf: usize) -> Option<Vec<f64>> {
        let started = std::time::Instant::now();
        let mut out = vec![0.0; self.n];
        let rc = unsafe { rc_mtf_gather(self.ctx, leaf as c_int, out.as_mut_ptr()) };
        let ok = rc == 0;
        self.record(started, ok);
        if ok {
            Some(out)
        } else {
            None
        }
    }

    pub fn stats(&self) -> GpuStats {
        self.stats
    }
}

impl Drop for MtfGpu {
    fn drop(&mut self) {
        unsafe { rc_mtf_destroy(self.ctx) };
    }
}

fn leaf_index(leaf: &str) -> Option<usize> {
    if leaf == "bar_age" {
        return Some(11);
    }
    MTF_LEAVES.iter().position(|name| *name == leaf)
}

fn mtf_parts(name: &str) -> Option<(&str, &str)> {
    let (tf, leaf) = name.strip_prefix("mtf_")?.split_once('_')?;
    if MTF_TIMEFRAMES.contains(&tf) && MTF_LEAVES.contains(&leaf) {
        Some((tf, leaf))
    } else {
        None
    }
}

/// Compute every requested MTF column on the device.
///
/// Returns the columns in request-independent name order plus the measured
/// device ledger. `None` means the context or a launch failed; the caller must
/// fall back to the CPU reference rather than emit a partial family.
#[allow(clippy::too_many_arguments)]
pub fn compute(
    times: &[i64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    seg_start: &[u32],
    seg_len: &[u32],
    step: i64,
    requested: &[String],
) -> Option<(BTreeMap<String, Vec<f64>>, GpuStats)> {
    let mut gpu = MtfGpu::new(times, open, high, low, close, seg_start, seg_len, step)?;
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for tf_name in MTF_TIMEFRAMES {
        let tf = Timeframe::parse(tf_name).ok()?;
        let leaves: Vec<&str> = requested
            .iter()
            .filter_map(|name| {
                let (tf, leaf) = mtf_parts(name)?;
                if tf == *tf_name {
                    Some(leaf)
                } else {
                    None
                }
            })
            .collect();
        if leaves.is_empty() {
            continue;
        }
        if !gpu.prepare(tf.seconds()) {
            return None;
        }
        for leaf in leaves {
            let index = leaf_index(leaf)?;
            let values = gpu.gather(index)?;
            out.insert(format!("mtf_{}_{}", tf_name, leaf), values);
        }
    }
    Some((out, gpu.stats()))
}
