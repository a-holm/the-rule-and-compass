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
        max_buckets: c_int,
    ) -> *mut Raw;
    fn rc_mtf_destroy(ctx: *mut Raw);
    fn rc_mtf_last_stats(ctx: *mut Raw, out: *mut RawStats);
    fn rc_mtf_prepare(ctx: *mut Raw, tf_seconds: i64) -> c_int;
    fn rc_mtf_total_buckets(ctx: *mut Raw) -> c_int;
    fn rc_mtf_gather(ctx: *mut Raw, leaf: c_int, out: *mut f64) -> c_int;
}

/// Why the device MTF path was abandoned: the failing stage and its return code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtfError {
    pub stage: &'static str,
    pub code: i32,
}

impl std::fmt::Display for MtfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage {} code {}", self.stage, self.code)
    }
}

/// Upper bound on the `1h` bucket count: the hour span of each segment, capped
/// at its row count. Every higher timeframe is a multiple of 3600 s, so it
/// never needs more buckets than `1h`.
pub fn bucket_capacity(times: &[i64], seg_start: &[u32], seg_len: &[u32]) -> usize {
    let mut total = 0usize;
    for (start, len) in seg_start.iter().zip(seg_len) {
        let (start, len) = (*start as usize, *len as usize);
        let Some(rows) = times.get(start..start.saturating_add(len)) else {
            return times.len();
        };
        let (Some(lo), Some(hi)) = (rows.iter().min(), rows.iter().max()) else {
            continue;
        };
        let span = hi.div_euclid(3600) - lo.div_euclid(3600) + 1;
        total = total.saturating_add((span as usize).min(len));
    }
    total.clamp(1, times.len().max(1))
}

/// One resident MTF context. It is per enrichment run, not per column.
pub struct MtfGpu {
    ctx: *mut Raw,
    n: usize,
    capacity: usize,
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
    ) -> Result<Self, MtfError> {
        let create_failed = MtfError { stage: "create", code: 0 };
        if times.is_empty()
            || seg_start.is_empty()
            || seg_start.len() != seg_len.len()
            || open.len() != times.len()
            || high.len() != times.len()
            || low.len() != times.len()
            || close.len() != times.len()
        {
            return Err(create_failed);
        }
        let capacity = bucket_capacity(times, seg_start, seg_len);
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
                capacity as c_int,
            )
        };
        if ctx.is_null() {
            return Err(create_failed);
        }
        let mut g = MtfGpu {
            ctx,
            n: times.len(),
            capacity,
            stats: GpuStats::default(),
        };
        g.record_creation();
        Ok(g)
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
    pub fn prepare(&mut self, tf_seconds: i64) -> Result<(), MtfError> {
        let started = std::time::Instant::now();
        let rc = unsafe { rc_mtf_prepare(self.ctx, tf_seconds) };
        let ok = rc == 0;
        self.record(started, ok);
        if ok {
            Ok(())
        } else {
            Err(MtfError { stage: "prepare", code: rc })
        }
    }

    /// Bucket capacity the device buffers were sized for.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bucket count found by the last `prepare`.
    pub fn total_buckets(&self) -> usize {
        unsafe { rc_mtf_total_buckets(self.ctx) }.max(0) as usize
    }

    /// Copy one base-length leaf (0..10) or `bar_age` (11) out of the context.
    pub fn gather(&mut self, leaf: usize) -> Result<Vec<f64>, MtfError> {
        let started = std::time::Instant::now();
        let mut out = vec![0.0; self.n];
        let rc = unsafe { rc_mtf_gather(self.ctx, leaf as c_int, out.as_mut_ptr()) };
        let ok = rc == 0;
        self.record(started, ok);
        if ok {
            Ok(out)
        } else {
            Err(MtfError { stage: "gather", code: rc })
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
/// device ledger. An error means the context or a launch failed; the caller
/// must fall back to the CPU reference rather than emit a partial family.
///
/// `RCSR_MTF_GPU_FORCE_TF_SECONDS` replaces the first timeframe handed to the
/// device. It exists so tests can force an unsupported timeframe.
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
) -> Result<(BTreeMap<String, Vec<f64>>, GpuStats), MtfError> {
    let mut gpu = MtfGpu::new(times, open, high, low, close, seg_start, seg_len, step)?;
    let forced: Option<i64> = std::env::var("RCSR_MTF_GPU_FORCE_TF_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for tf_name in MTF_TIMEFRAMES {
        let tf = Timeframe::parse(tf_name).map_err(|_| MtfError { stage: "timeframe", code: 0 })?;
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
        gpu.prepare(forced.unwrap_or(tf.seconds()))?;
        for leaf in leaves {
            let index = leaf_index(leaf).ok_or(MtfError { stage: "leaf", code: 0 })?;
            let values = gpu.gather(index)?;
            out.insert(format!("mtf_{}_{}", tf_name, leaf), values);
        }
    }
    Ok((out, gpu.stats()))
}
