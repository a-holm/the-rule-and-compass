//! CUDA enrichment backend for `rcsr enrich` (issue #200).
//!
//! One [`EnrichGpu`] owns the resident device copy of the source columns and a
//! scratch arena. Every feature family reuses that context, so the same
//! `time/open/high/low/close/volume/quote_volume` columns are uploaded **once**
//! per run, not once per feature group. The C kernels live in
//! `src/cuda/enrich.cu`; this module is only compiled with `--features cuda`.
//!
//! The kernels are f64 and deterministic. Sequential families (rolling sums and
//! standard deviations, Wilder/EMA scans, Donchian, labels) run one thread per
//! symbol segment and use the same operation order as the CPU reference, so
//! their columns hash identically. Cross-sectional ranks use a stable segmented
//! radix sort with the CPU's `(time, symbol, row)` tie order.

use crate::gpu_stats::GpuStats;
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
    fn rc_enrich_create(
        times: *const i64,
        open: *const f64,
        high: *const f64,
        low: *const f64,
        close: *const f64,
        volume: *const f64,
        qvolume: *const f64,
        n: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        nseg: c_int,
        step: i64,
    ) -> *mut Raw;
    fn rc_enrich_destroy(ctx: *mut Raw);
    fn rc_enrich_last_stats(ctx: *mut Raw, out: *mut RawStats);
    fn rc_enrich_ret(ctx: *mut Raw, k: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_lag(ctx: *mut Raw, k: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_range(ctx: *mut Raw, out: *mut f64) -> c_int;
    fn rc_enrich_clv(ctx: *mut Raw, out: *mut f64) -> c_int;
    fn rc_enrich_calendar(ctx: *mut Raw, which: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_ma(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_vol(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_bb_width(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_vz(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_donchian(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_ema(ctx: *mut Raw, span: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_macd(
        ctx: *mut Raw,
        macd: *mut f64,
        sig: *mut f64,
        hist: *mut f64,
    ) -> c_int;
    fn rc_enrich_rsi_py(ctx: *mut Raw, period: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_atr_py(ctx: *mut Raw, period: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_label(
        ctx: *mut Raw,
        horizon: c_int,
        cost_bps: f64,
        vol_window: c_int,
        stride: c_int,
        out: *mut f64,
    ) -> c_int;
    fn rc_enrich_rank(
        keys: *const u64,
        rows: *const u32,
        offsets: *const c_int,
        gid: *const u32,
        n: c_int,
        ngroups: c_int,
        out: *mut f64,
    ) -> c_int;
    fn rc_enrich_rank_stats(out: *mut RawStats);
}

pub struct EnrichGpu {
    ctx: *mut Raw,
    n: usize,
    stats: GpuStats,
}

impl EnrichGpu {
    /// Upload the resident source columns and segment table once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        times: &[i64],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        quote_volume: &[f64],
        seg_start: &[u32],
        seg_len: &[u32],
        step: i64,
    ) -> Option<Self> {
        if times.is_empty() || seg_start.is_empty() {
            return None;
        }
        let starts: Vec<c_int> = seg_start.iter().map(|v| *v as c_int).collect();
        let lens: Vec<c_int> = seg_len.iter().map(|v| *v as c_int).collect();
        let ctx = unsafe {
            rc_enrich_create(
                times.as_ptr(),
                open.as_ptr(),
                high.as_ptr(),
                low.as_ptr(),
                close.as_ptr(),
                volume.as_ptr(),
                quote_volume.as_ptr(),
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
        let mut g = EnrichGpu {
            ctx,
            n: times.len(),
            stats: GpuStats::default(),
        };
        // The create call owns the one-time upload; account it as a batch with
        // zero programs so the transfer totals are honest.
        g.record_creation();
        Some(g)
    }

    fn record_creation(&mut self) {
        let mut raw = RawStats::default();
        unsafe { rc_enrich_last_stats(self.ctx, &mut raw) };
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
        unsafe { rc_enrich_last_stats(self.ctx, &mut raw) };
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

    fn out(&self) -> Vec<f64> {
        vec![0.0; self.n]
    }

    fn one<F>(&mut self, mut call: F) -> Option<Vec<f64>>
    where
        F: FnMut(*mut f64) -> c_int,
    {
        let started = std::time::Instant::now();
        let mut out = self.out();
        let rc = call(out.as_mut_ptr());
        let ok = rc == 0;
        self.record(started, ok);
        if ok {
            Some(out)
        } else {
            None
        }
    }

    pub fn ret(&mut self, k: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_ret(ctx, k as c_int, p) })
    }

    pub fn lag(&mut self, k: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_lag(ctx, k as c_int, p) })
    }

    pub fn range(&mut self) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_range(ctx, p) })
    }

    pub fn clv(&mut self) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_clv(ctx, p) })
    }

    pub fn calendar(&mut self, which: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_calendar(ctx, which as c_int, p) })
    }

    pub fn ma(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_ma(ctx, w as c_int, p) })
    }

    pub fn vol(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_vol(ctx, w as c_int, p) })
    }

    pub fn bb_width(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_bb_width(ctx, w as c_int, p) })
    }

    pub fn vz(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_vz(ctx, w as c_int, p) })
    }

    pub fn donchian(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_donchian(ctx, w as c_int, p) })
    }

    pub fn ema(&mut self, span: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_ema(ctx, span as c_int, p) })
    }

    pub fn macd(&mut self) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let started = std::time::Instant::now();
        let mut macd = self.out();
        let mut sig = self.out();
        let mut hist = self.out();
        let rc = unsafe {
            rc_enrich_macd(
                self.ctx,
                macd.as_mut_ptr(),
                sig.as_mut_ptr(),
                hist.as_mut_ptr(),
            )
        };
        let ok = rc == 0;
        self.record(started, ok);
        if ok {
            Some((macd, sig, hist))
        } else {
            None
        }
    }

    pub fn rsi_py(&mut self, period: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_rsi_py(ctx, period as c_int, p) })
    }

    pub fn atr_py(&mut self, period: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_atr_py(ctx, period as c_int, p) })
    }

    pub fn label(
        &mut self,
        horizon: usize,
        cost_bps: f64,
        vol_window: usize,
        stride: usize,
    ) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe {
            rc_enrich_label(
                ctx,
                horizon as c_int,
                cost_bps,
                vol_window as c_int,
                stride as c_int,
                p,
            )
        })
    }

    /// Cross-sectional ranks. The `(time, symbol, row)` permutation and the
    /// stable segmented sort reproduce [`crate::enrich::xs_rank`] exactly.
    pub fn rank(&mut self, src: &[f64], times: &[i64], sym_key: &[u32]) -> Option<Vec<f64>> {
        let n = src.len();
        if n == 0 || times.len() != n || sym_key.len() != n {
            return None;
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            times[a]
                .cmp(&times[b])
                .then(sym_key[a].cmp(&sym_key[b]))
                .then(a.cmp(&b))
        });
        let mut offsets: Vec<c_int> = Vec::new();
        let mut gid = vec![0u32; n];
        let mut i = 0usize;
        while i < order.len() {
            let t = times[order[i] as usize];
            let mut j = i;
            while j < order.len() && times[order[j] as usize] == t {
                j += 1;
            }
            offsets.push(i as c_int);
            let g = (offsets.len() - 1) as u32;
            for slot in gid.iter_mut().take(j).skip(i) {
                *slot = g;
            }
            i = j;
        }
        offsets.push(n as c_int);
        let keys: Vec<u64> = order
            .iter()
            .map(|&r| sort_key(src[r as usize]))
            .collect();
        let rows: Vec<u32> = order;
        let started = std::time::Instant::now();
        let mut out = self.out();
        let rc = unsafe {
            rc_enrich_rank(
                keys.as_ptr(),
                rows.as_ptr(),
                offsets.as_ptr(),
                gid.as_ptr(),
                n as c_int,
                (offsets.len() - 1) as c_int,
                out.as_mut_ptr(),
            )
        };
        let ok = rc == 0;
        let mut raw = RawStats::default();
        unsafe { rc_enrich_rank_stats(&mut raw) };
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
            n as u64,
            !ok || raw.failed != 0,
        );
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

impl Drop for EnrichGpu {
    fn drop(&mut self) {
        unsafe { rc_enrich_destroy(self.ctx) };
    }
}

fn sort_key(v: f64) -> u64 {
    if v.is_nan() {
        u64::MAX
    } else {
        let bits = v.to_bits();
        if bits & (1u64 << 63) != 0 {
            !bits
        } else {
            bits | (1u64 << 63)
        }
    }
}
