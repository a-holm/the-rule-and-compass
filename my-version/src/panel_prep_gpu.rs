//! #301: resident device panel preparation (fused duplicate scan, 15 m → 1 h
//! resample and gap tags).
//!
//! The CPU reference is [`crate::panel_prep::prepare`]; this module owns the
//! CUDA context and its statistics ledger. The kernel is `src/cuda/prep.cu`.
//!
//! The context consumes one **already decoded** symbol-major block. Symbol
//! segments are contiguous; within each segment the device performs a stable
//! sort by `time`, an adjacent duplicate scan, a compacting four-bar fold and a
//! gap pass. No text parsing, unit decision or file I/O happens here.
//!
//! All outputs are deterministic: conflict offsets are exclusive scans, the
//! only sort is stable, and no atomic touches a result. The same input with
//! any launch block size must produce identical bytes.

use crate::gpu_stats::GpuStats;
use crate::panel_prep::{
    validate_segments, PrepBar, PrepBucket, PrepConflict, PrepOutput, PrepSegment,
};
use std::os::raw::c_int;

const FLAG_COMPLETE: u32 = 1;
const FLAG_TRADED: u32 = 2;
const FLAG_PARTIAL: u32 = 4;
const FLAG_CONFLICT: u32 = 8;
const FLAG_OUTAGE: u32 = 16;

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

#[repr(C)]
#[derive(Default)]
struct HostOut {
    time: *mut i64,
    open: *mut f64,
    high: *mut f64,
    low: *mut f64,
    close: *mut f64,
    volume: *mut f64,
    quote_volume: *mut f64,
    trades: *mut f64,
    taker_buy_base: *mut f64,
    taker_buy_quote: *mut f64,
    cov15m: *mut u32,
    dup_rows: *mut u32,
    gap: *mut u32,
    flags: *mut u32,
    chain: *mut i64,
    segment: *mut u32,
    conf_time: *mut i64,
    conf_segment: *mut u32,
    conf_chain: *mut i64,
    conf_dup_rows: *mut u32,
    conf_first_row: *mut u32,
    conf_len: *mut u32,
    n_out: c_int,
    n_conflict: c_int,
}

enum Raw {}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn rc_prep_create(
        times: *const i64,
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        volume: *const f64,
        quote_volume: *const f64,
        trades: *const f64,
        taker_buy_base: *const f64,
        taker_buy_quote: *const f64,
        complete: *const u8,
        traded: *const u8,
        chain: *const i64,
        n: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        nseg: c_int,
    ) -> *mut Raw;
    fn rc_prep_destroy(ctx: *mut Raw);
    fn rc_prep_last_stats(ctx: *mut Raw, out: *mut RawStats);
    fn rc_prep_run(
        ctx: *mut Raw,
        in_step: i64,
        out_step: i64,
        keep_partial: c_int,
        threads: c_int,
    ) -> c_int;
    fn rc_prep_counts(
        ctx: *mut Raw,
        n_out: *mut c_int,
        n_conflict: *mut c_int,
        n_unique: *mut c_int,
    );
    fn rc_prep_download(ctx: *mut Raw, out: *mut HostOut) -> c_int;
}

/// One resident device block. Drop it to release the arena.
pub struct PrepGpu {
    ctx: *mut Raw,
    n: usize,
    stats: GpuStats,
}

impl PrepGpu {
    /// Upload one decoded symbol-major block. `segments` must be contiguous and
    /// cover `bars` exactly.
    pub fn new(bars: &[PrepBar], segments: &[PrepSegment]) -> Option<Self> {
        if bars.is_empty() || segments.is_empty() {
            return None;
        }
        validate_segments(bars.len(), segments).ok()?;
        let mut times = Vec::with_capacity(bars.len());
        let mut open = Vec::with_capacity(bars.len());
        let mut high = Vec::with_capacity(bars.len());
        let mut low = Vec::with_capacity(bars.len());
        let mut close = Vec::with_capacity(bars.len());
        let mut volume = Vec::with_capacity(bars.len());
        let mut quote_volume = Vec::with_capacity(bars.len());
        let mut trades = Vec::with_capacity(bars.len());
        let mut taker_buy_base = Vec::with_capacity(bars.len());
        let mut taker_buy_quote = Vec::with_capacity(bars.len());
        let mut complete = Vec::with_capacity(bars.len());
        let mut traded = Vec::with_capacity(bars.len());
        let mut chain = Vec::with_capacity(bars.len());
        for b in bars {
            times.push(b.time);
            open.push(b.open);
            high.push(b.high);
            low.push(b.low);
            close.push(b.close);
            volume.push(b.volume);
            quote_volume.push(b.quote_volume);
            trades.push(b.trades);
            taker_buy_base.push(b.taker_buy_base);
            taker_buy_quote.push(b.taker_buy_quote);
            complete.push(b.complete as u8);
            traded.push(b.traded as u8);
            chain.push(b.chain);
        }
        let starts: Vec<c_int> = segments.iter().map(|s| s.start as c_int).collect();
        let lens: Vec<c_int> = segments.iter().map(|s| s.len as c_int).collect();
        let ctx = unsafe {
            rc_prep_create(
                times.as_ptr(),
                open.as_ptr(),
                high.as_ptr(),
                low.as_ptr(),
                close.as_ptr(),
                volume.as_ptr(),
                quote_volume.as_ptr(),
                trades.as_ptr(),
                taker_buy_base.as_ptr(),
                taker_buy_quote.as_ptr(),
                complete.as_ptr(),
                traded.as_ptr(),
                chain.as_ptr(),
                bars.len() as c_int,
                starts.as_ptr(),
                lens.as_ptr(),
                starts.len() as c_int,
            )
        };
        if ctx.is_null() {
            return None;
        }
        let mut g = PrepGpu {
            ctx,
            n: bars.len(),
            stats: GpuStats::default(),
        };
        g.record_creation();
        Some(g)
    }

    fn record_creation(&mut self) {
        let mut raw = RawStats::default();
        unsafe { rc_prep_last_stats(self.ctx, &mut raw) };
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
        unsafe { rc_prep_last_stats(self.ctx, &mut raw) };
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

    /// Run the fused duplicate/resample/gap pass with the default 256 threads
    /// per block.
    pub fn run(&mut self, in_step: i64, out_step: i64, keep_partial: bool) -> bool {
        self.run_with_threads(in_step, out_step, keep_partial, 256)
    }

    /// Run with an explicit block size. Block size must not change the bytes;
    /// the test harness uses this to prove block-count invariance.
    pub fn run_with_threads(
        &mut self,
        in_step: i64,
        out_step: i64,
        keep_partial: bool,
        threads: usize,
    ) -> bool {
        if in_step <= 0 || out_step <= 0 || out_step % in_step != 0 {
            return false;
        }
        if out_step != 4 * in_step {
            return false;
        }
        if !(32..=1024).contains(&threads) || threads % 32 != 0 {
            return false;
        }
        let started = std::time::Instant::now();
        let rc = unsafe {
            rc_prep_run(
                self.ctx,
                in_step,
                out_step,
                keep_partial as c_int,
                threads as c_int,
            )
        };
        let ok = rc == 0;
        self.record(started, ok);
        ok
    }

    /// Download the result. The conflict list is already in deterministic
    /// `(segment, time)` order.
    pub fn output(&mut self) -> Option<PrepOutput> {
        let started = std::time::Instant::now();
        let (mut n_out, mut n_conflict, mut n_unique) = (0 as c_int, 0 as c_int, 0 as c_int);
        unsafe { rc_prep_counts(self.ctx, &mut n_out, &mut n_conflict, &mut n_unique) };
        if n_out < 0 || n_conflict < 0 || n_unique < 0 {
            self.record(started, false);
            return None;
        }
        let n = n_out as usize;
        let c = n_conflict as usize;
        let mut time = vec![0i64; n];
        let mut open = vec![0.0f64; n];
        let mut high = vec![0.0f64; n];
        let mut low = vec![0.0f64; n];
        let mut close = vec![0.0f64; n];
        let mut volume = vec![0.0f64; n];
        let mut quote_volume = vec![0.0f64; n];
        let mut trades = vec![0.0f64; n];
        let mut taker_buy_base = vec![0.0f64; n];
        let mut taker_buy_quote = vec![0.0f64; n];
        let mut cov15m = vec![0u32; n];
        let mut dup_rows = vec![0u32; n];
        let mut gap = vec![0u32; n];
        let mut flags = vec![0u32; n];
        let mut chain = vec![0i64; n];
        let mut segment = vec![0u32; n];
        let mut conf_time = vec![0i64; c];
        let mut conf_segment = vec![0u32; c];
        let mut conf_chain = vec![0i64; c];
        let mut conf_dup_rows = vec![0u32; c];
        let mut conf_first_row = vec![0u32; c];
        let mut conf_len = vec![0u32; c];
        let mut out = HostOut {
            time: time.as_mut_ptr(),
            open: open.as_mut_ptr(),
            high: high.as_mut_ptr(),
            low: low.as_mut_ptr(),
            close: close.as_mut_ptr(),
            volume: volume.as_mut_ptr(),
            quote_volume: quote_volume.as_mut_ptr(),
            trades: trades.as_mut_ptr(),
            taker_buy_base: taker_buy_base.as_mut_ptr(),
            taker_buy_quote: taker_buy_quote.as_mut_ptr(),
            cov15m: cov15m.as_mut_ptr(),
            dup_rows: dup_rows.as_mut_ptr(),
            gap: gap.as_mut_ptr(),
            flags: flags.as_mut_ptr(),
            chain: chain.as_mut_ptr(),
            segment: segment.as_mut_ptr(),
            conf_time: conf_time.as_mut_ptr(),
            conf_segment: conf_segment.as_mut_ptr(),
            conf_chain: conf_chain.as_mut_ptr(),
            conf_dup_rows: conf_dup_rows.as_mut_ptr(),
            conf_first_row: conf_first_row.as_mut_ptr(),
            conf_len: conf_len.as_mut_ptr(),
            n_out,
            n_conflict,
        };
        let rc = unsafe { rc_prep_download(self.ctx, &mut out) };
        let ok = rc == 0;
        self.record(started, ok);
        if !ok {
            return None;
        }
        let buckets = (0..n)
            .map(|i| PrepBucket {
                time: time[i],
                segment: segment[i],
                chain: chain[i],
                open: open[i],
                high: high[i],
                low: low[i],
                close: close[i],
                volume: volume[i],
                quote_volume: quote_volume[i],
                trades: trades[i],
                taker_buy_base: taker_buy_base[i],
                taker_buy_quote: taker_buy_quote[i],
                cov15m: cov15m[i],
                dup_rows: dup_rows[i],
                gap: gap[i],
                complete: flags[i] & FLAG_COMPLETE != 0,
                traded: flags[i] & FLAG_TRADED != 0,
                partial: flags[i] & FLAG_PARTIAL != 0,
                duplicate_conflict: flags[i] & FLAG_CONFLICT != 0,
                outage: flags[i] & FLAG_OUTAGE != 0,
            })
            .collect();
        let conflicts = (0..c)
            .map(|i| PrepConflict {
                time: conf_time[i],
                segment: conf_segment[i],
                chain: conf_chain[i],
                dup_rows: conf_dup_rows[i],
                first_row: conf_first_row[i],
                len: conf_len[i],
            })
            .collect();
        Some(PrepOutput {
            buckets,
            conflicts,
            input_rows: self.n,
            unique_rows: n_unique as usize,
        })
    }

    pub fn stats(&self) -> GpuStats {
        self.stats
    }
}

impl Drop for PrepGpu {
    fn drop(&mut self) {
        unsafe { rc_prep_destroy(self.ctx) };
    }
}

/// One-shot convenience wrapper: create, run and download. `None` means the
/// context or launch failed; the caller keeps the CPU reference path.
pub fn prepare(
    bars: &[PrepBar],
    segments: &[PrepSegment],
    in_step: i64,
    out_step: i64,
    keep_partial: bool,
) -> Option<(PrepOutput, GpuStats)> {
    let mut gpu = PrepGpu::new(bars, segments)?;
    if !gpu.run(in_step, out_step, keep_partial) {
        return None;
    }
    let out = gpu.output()?;
    Some((out, gpu.stats()))
}
