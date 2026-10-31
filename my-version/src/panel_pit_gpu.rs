//! Device point-in-time as-of joins (issue #302).
//!
//! [`crate::panel_pit::join`] is the CPU reference. [`PitGpu`] uploads one
//! symbol-major block (base times, input chains, per-symbol decision lists) and
//! runs the monotone merge in `src/cuda/pit.cu`. The membership decisions are
//! made by the CPU; only the join, the gap tags and the chain numbering run on
//! the device. The result must be bit-identical to the reference for any
//! block size and chunk size.

use crate::gpu_stats::GpuStats;
use crate::panel_pit::{PitBlock, PitOutput, PitParams};
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
    loads: u64,
    violations: u64,
    n_kept: i32,
    n_chunks: i32,
}

#[repr(C)]
struct HostOut {
    leaf: *mut i32,
    keep: *mut u8,
    found: *mut u8,
    as_of: *mut i64,
    gap: *mut u32,
    chain_ord: *mut i32,
    kept_rows: *mut i32,
}

enum Raw {}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn rc_pit_create(
        times: *const i64,
        chain: *const i64,
        n: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        leaf_start: *const c_int,
        leaf_len: *const c_int,
        nseg: c_int,
        key: *const i64,
        member: *const u8,
        nleaf: c_int,
    ) -> *mut Raw;
    fn rc_pit_destroy(ctx: *mut Raw);
    fn rc_pit_last_stats(ctx: *mut Raw, out: *mut RawStats);
    fn rc_pit_run(
        ctx: *mut Raw,
        step: i64,
        threads: c_int,
        chunk_rows: c_int,
    ) -> c_int;
    fn rc_pit_check(ctx: *mut Raw, idx: *const c_int, violations: *mut u64) -> c_int;
    fn rc_pit_download(ctx: *mut Raw, out: *mut HostOut) -> c_int;
}

/// Default rows per merge cursor.
pub const DEFAULT_CHUNK_ROWS: usize = 256;

/// One resident join block.
pub struct PitGpu {
    ctx: *mut Raw,
    n: usize,
    params: Option<PitParams>,
    stats: GpuStats,
    loads: u64,
    violations: u64,
}

impl PitGpu {
    pub fn new(block: &PitBlock) -> Option<Self> {
        if block.times.is_empty() || block.segments.is_empty() || block.validate().is_err() {
            return None;
        }
        let starts: Vec<c_int> = block.segments.iter().map(|s| s.start as c_int).collect();
        let lens: Vec<c_int> = block.segments.iter().map(|s| s.len as c_int).collect();
        let lstarts: Vec<c_int> = block.segments.iter().map(|s| s.leaf_start as c_int).collect();
        let llens: Vec<c_int> = block.segments.iter().map(|s| s.leaf_len as c_int).collect();
        let ctx = unsafe {
            rc_pit_create(
                block.times.as_ptr(),
                block.chain.as_ptr(),
                block.times.len() as c_int,
                starts.as_ptr(),
                lens.as_ptr(),
                lstarts.as_ptr(),
                llens.as_ptr(),
                starts.len() as c_int,
                block.keys.as_ptr(),
                block.member.as_ptr(),
                block.keys.len() as c_int,
            )
        };
        if ctx.is_null() {
            return None;
        }
        let mut g = PitGpu {
            ctx,
            n: block.times.len(),
            params: None,
            stats: GpuStats::default(),
            loads: 0,
            violations: 0,
        };
        g.record(None);
        Some(g)
    }

    fn record(&mut self, started: Option<std::time::Instant>) {
        let mut raw = RawStats::default();
        unsafe { rc_pit_last_stats(self.ctx, &mut raw) };
        let outer_ms = started.map(|s| s.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
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
            raw.failed != 0,
        );
    }

    /// Run with the default block (256 threads) and chunk size.
    pub fn run(&mut self, p: PitParams) -> bool {
        self.run_with(p, 256, DEFAULT_CHUNK_ROWS)
    }

    /// Run with an explicit block and chunk size. Neither may change the bytes.
    pub fn run_with(&mut self, p: PitParams, threads: usize, chunk_rows: usize) -> bool {
        if p.step <= 0 || chunk_rows == 0 {
            return false;
        }
        if !(32..=1024).contains(&threads) || threads % 32 != 0 {
            return false;
        }
        let started = std::time::Instant::now();
        let rc = unsafe {
            rc_pit_run(self.ctx, p.step, threads as c_int, chunk_rows as c_int)
        };
        let ok = rc == 0;
        let mut raw = RawStats::default();
        unsafe { rc_pit_last_stats(self.ctx, &mut raw) };
        self.loads = raw.loads;
        self.violations = raw.violations;
        self.record(Some(started));
        if ok {
            self.params = Some(p);
        }
        ok
    }

    /// Dependent key loads of the last run (binary-search probes plus cursor
    /// comparisons).
    pub fn loads(&self) -> u64 {
        self.loads
    }

    /// Count look-ahead violations of a caller-supplied leaf index column.
    pub fn lookahead_violations(&mut self, leaf: &[i32]) -> Option<u64> {
        if leaf.len() != self.n {
            return None;
        }
        let started = std::time::Instant::now();
        let mut v = 0u64;
        let rc = unsafe { rc_pit_check(self.ctx, leaf.as_ptr(), &mut v) };
        self.record(Some(started));
        if rc == 0 {
            Some(v)
        } else {
            None
        }
    }

    pub fn output(&mut self) -> Option<PitOutput> {
        self.params?;
        let started = std::time::Instant::now();
        let n = self.n;
        let mut raw = RawStats::default();
        unsafe { rc_pit_last_stats(self.ctx, &mut raw) };
        let n_kept = raw.n_kept.max(0) as usize;
        let mut leaf = vec![0i32; n];
        let mut keep = vec![0u8; n];
        let mut found = vec![0u8; n];
        let mut as_of = vec![0i64; n];
        let mut gap = vec![0u32; n];
        let mut chain_ord = vec![0i32; n];
        let mut kept = vec![0i32; n_kept];
        let mut out = HostOut {
            leaf: leaf.as_mut_ptr(),
            keep: keep.as_mut_ptr(),
            found: found.as_mut_ptr(),
            as_of: as_of.as_mut_ptr(),
            gap: gap.as_mut_ptr(),
            chain_ord: chain_ord.as_mut_ptr(),
            kept_rows: kept.as_mut_ptr(),
        };
        let rc = unsafe { rc_pit_download(self.ctx, &mut out) };
        self.record(Some(started));
        if rc != 0 {
            return None;
        }
        Some(PitOutput {
            leaf,
            keep,
            found,
            as_of,
            gap,
            chain_id: chain_ord.into_iter().map(|v| v as i64).collect(),
            kept_rows: kept.into_iter().map(|v| v as u32).collect(),
            lookahead_violations: self.violations,
        })
    }

    pub fn stats(&self) -> GpuStats {
        self.stats
    }
}

impl Drop for PitGpu {
    fn drop(&mut self) {
        unsafe { rc_pit_destroy(self.ctx) };
    }
}

/// One-shot create, run and download. `None` means the device path failed and
/// the caller keeps the CPU reference.
pub fn join(block: &PitBlock, p: PitParams) -> Option<(PitOutput, GpuStats)> {
    let mut gpu = PitGpu::new(block)?;
    if !gpu.run(p) {
        return None;
    }
    let out = gpu.output()?;
    Some((out, gpu.stats()))
}
