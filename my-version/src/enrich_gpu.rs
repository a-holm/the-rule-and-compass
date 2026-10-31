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
//! their columns hash identically. Cross-sectional ranks build one device
//! `(time, symbol, row)` permutation per run and reuse it for every rank
//! column; each column is a stable segmented radix sort by `(value, symbol,
//! row)` with the CPU's tie, NaN and PIT-member policy.

use crate::gpu_stats::GpuStats;
use crate::projected_ring::{ProjectedRing, TileFeature};
use crate::store::{FileStore, Projection};
use std::collections::BTreeMap;
use std::os::raw::{c_int, c_void};

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
        trades: *const f64,
        taker_buy_base: *const f64,
        taker_buy_quote: *const f64,
        present: u32,
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
    fn rc_enrich_ma(ctx: *mut Raw, w: c_int, mode: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_vol(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_bb_width(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_vz(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_donchian(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_donchian_centred(ctx: *mut Raw, w: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_stoch(ctx: *mut Raw, which: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_atr(ctx: *mut Raw, period: c_int, which: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_batr(ctx: *mut Raw, out: *mut f64) -> c_int;
    fn rc_enrich_gapatr(ctx: *mut Raw, out: *mut f64) -> c_int;
    fn rc_enrich_volrat(ctx: *mut Raw, out: *mut f64) -> c_int;
    fn rc_enrich_adx(ctx: *mut Raw, which: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_bb_z(ctx: *mut Raw, w: c_int, mode: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_ma_slope(ctx: *mut Raw, w: c_int, lag: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_tod(ctx: *mut Raw, want_sin: c_int, out: *mut f64) -> c_int;
    fn rc_enrich_exotic(ctx: *mut Raw, which: c_int, out: *mut f64) -> c_int;
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
    fn rc_enrich_xs_prepare(
        ctx: *mut Raw,
        sym_key: *const u32,
        member: *const u8,
    ) -> c_int;
    fn rc_enrich_create_arena(
        n: c_int,
        nseg: c_int,
        step: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        nan_mask: u32,
    ) -> *mut Raw;
    fn rc_enrich_times_ptr(ctx: *mut Raw) -> *mut i64;
    fn rc_enrich_value_ptr(ctx: *mut Raw, which: c_int) -> *mut f64;
    fn rc_enrich_new_output(ctx: *mut Raw) -> *mut f64;
    fn rc_enrich_release_output(p: *mut f64);
    fn rc_enrich_set_test_budget(bytes: usize);
    fn rc_enrich_set_test_alloc_fault(at: u64);
    fn rc_enrich_test_alloc_count() -> u64;
    fn rc_enrich_set_test_launch_fault(site: c_int);
    fn rc_enrich_test_launch_hits() -> u64;
    fn rc_enrich_download_ptr(src: *const f64, dst: *mut f64, n: c_int) -> c_int;
    fn rc_enrich_note_projected_upload(ctx: *mut Raw, bytes: u64, ms: f64);
    fn rc_enrich_note_projected_download(ctx: *mut Raw, bytes: u64, ms: f64);
    fn rc_enrich_batch_capacity(ctx: *mut Raw) -> c_int;
    fn rc_enrich_batch_host(ctx: *mut Raw) -> *mut f64;
    fn rc_enrich_batch_run(ctx: *mut Raw, ops: *const BatchOp, nops: c_int) -> c_int;
    fn rc_enrich_xs_rank(ctx: *mut Raw, src: *const f64, out: *mut f64) -> c_int;
}

/// Where one projected column lands in the enrichment context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceColumn {
    /// The exact `i64` time block.
    Time,
    /// `0` open, `1` high, `2` low, `3` close, `4` volume, `5` quote volume,
    /// `6` trades, `7` taker-buy base, `8` taker-buy quote.
    Value(i32),
}

/// One column of the device transport: a projected header index and its
/// destination in the enrichment arena.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedSource {
    pub header_index: usize,
    pub destination: DeviceColumn,
}

pub struct EnrichGpu {
    ctx: *mut Raw,
    n: usize,
    stats: GpuStats,
    precomputed: BTreeMap<String, *mut f64>,
    ring_bytes: u64,
}

/// #300: one feature for the batched output path.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BatchOp {
    kind: c_int,
    a: c_int,
    b: c_int,
    c: c_int,
    param: f64,
}

impl BatchOp {
    pub fn ret(k: usize) -> Self {
        Self { kind: 1, a: k as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn lag(k: usize) -> Self {
        Self { kind: 2, a: k as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn range() -> Self {
        Self { kind: 3, a: 0, b: 0, c: 0, param: 0.0 }
    }
    pub fn clv() -> Self {
        Self { kind: 4, a: 0, b: 0, c: 0, param: 0.0 }
    }
    pub fn calendar(which: usize) -> Self {
        Self { kind: 5, a: which as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn ma(w: usize, mode: i32) -> Self {
        Self { kind: 6, a: w as c_int, b: mode as c_int, c: 0, param: 0.0 }
    }
    pub fn vol(w: usize) -> Self {
        Self { kind: 7, a: w as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn bb_width(w: usize) -> Self {
        Self { kind: 8, a: w as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn vz(w: usize) -> Self {
        Self { kind: 9, a: w as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn donchian(w: usize) -> Self {
        Self { kind: 10, a: w as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn ema(span: usize) -> Self {
        Self { kind: 11, a: span as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn rsi(period: usize) -> Self {
        Self { kind: 12, a: period as c_int, b: 0, c: 0, param: 0.0 }
    }
    pub fn atr(period: usize) -> Self {
        Self { kind: 13, a: period as c_int, b: 0, c: 0, param: 0.0 }
    }
    #[allow(dead_code)]
    pub fn label(horizon: usize, cost_bps: f64, vol_window: usize, stride: usize) -> Self {
        Self {
            kind: 14,
            a: horizon as c_int,
            b: vol_window as c_int,
            c: stride as c_int,
            param: cost_bps,
        }
    }
    #[allow(dead_code)]
    pub fn macd() -> Self {
        Self { kind: 15, a: 0, b: 0, c: 0, param: 0.0 }
    }
    /// Number of output columns this operation writes.
    pub fn slots(&self) -> usize {
        if self.kind == 15 {
            3
        } else {
            1
        }
    }
}

/// #300: a completed family block, read directly from the pinned arena.
///
/// The borrow keeps the context alive; the next [`EnrichGpu::batch`] call
/// overwrites the arena, so callers must finish writing the block first.
pub struct BatchBlock<'a> {
    gpu: &'a mut EnrichGpu,
    count: usize,
    rows: usize,
}

impl BatchBlock<'_> {
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.count
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn column(&self, i: usize) -> &[f64] {
        assert!(i < self.count, "batch column index out of range");
        let ptr = unsafe { rc_enrich_batch_host(self.gpu.ctx) };
        assert!(!ptr.is_null(), "pinned batch arena is missing");
        unsafe { std::slice::from_raw_parts(ptr.add(i * self.rows), self.rows) }
    }
}


impl EnrichGpu {
    /// Upload the resident source columns and segment table once.
    ///
    /// This is the whole-column path with no optional flow columns; the fused
    /// flow/price-action producer uses [`EnrichGpu::new_with_flow`].
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
        let nan = vec![f64::NAN; times.len()];
        Self::new_with_flow(
            times,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            &nan,
            &nan,
            &nan,
            crate::features_gpu::PRESENT_VOLUME | crate::features_gpu::PRESENT_QUOTE_VOLUME,
            seg_start,
            seg_len,
            step,
        )
    }

    /// Whole-column upload including the optional flow columns (#297).
    ///
    /// An absent optional column is passed as an all-NaN slice and must have
    /// its presence bit clear; consumers then emit `NaN`, never `0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flow(
        times: &[i64],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        quote_volume: &[f64],
        trades: &[f64],
        taker_buy_base: &[f64],
        taker_buy_quote: &[f64],
        present: u32,
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
                trades.as_ptr(),
                taker_buy_base.as_ptr(),
                taker_buy_quote.as_ptr(),
                present,
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
            precomputed: BTreeMap::new(),
            ring_bytes: 0,
        };
        // The create call owns the one-time upload; account it as a batch with
        // zero programs so the transfer totals are honest.
        g.record_creation();
        Some(g)
    }

    /// Build a context whose source columns are uploaded group by group through
    /// the bounded pinned ring (issue #294).
    ///
    /// `sources` must be in projection order. `precompute` names are the
    /// elementwise features whose tile can run while the next group copies; any
    /// name not recognised there is simply left to the whole-column kernel.
    pub fn from_projected(
        store: &FileStore,
        projection: Projection,
        sources: &[ProjectedSource],
        seg_start: &[u32],
        seg_len: &[u32],
        step: i64,
        precompute: &[String],
        slots: usize,
    ) -> Option<Self> {
        if store.n_rows() == 0 || seg_start.is_empty() {
            return None;
        }
        if sources.len() != projection.len() {
            return None;
        }
        if projection
            .columns()
            .iter()
            .zip(sources)
            .any(|(c, s)| c.index != s.header_index)
        {
            return None;
        }
        let n = store.n_rows();
        if n > c_int::MAX as usize {
            return None;
        }
        let starts: Vec<c_int> = seg_start.iter().map(|v| *v as c_int).collect();
        let lens: Vec<c_int> = seg_len.iter().map(|v| *v as c_int).collect();
        // The context owns all six value arrays; arrays the projection does not
        // provide start as NaN, never as zero.
        let mut nan_mask = 0u32;
        for (bit, which) in [
            (0u32, 0i32),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
        ] {
            if !sources
                .iter()
                .any(|s| s.destination == DeviceColumn::Value(which))
            {
                nan_mask |= 1 << bit;
            }
        }
        let ctx = unsafe {
            rc_enrich_create_arena(
                n as c_int,
                starts.len() as c_int,
                step as c_int,
                starts.as_ptr(),
                lens.as_ptr(),
                nan_mask,
            )
        };
        if ctx.is_null() {
            return None;
        }

        let mut precomputed: BTreeMap<String, *mut f64> = BTreeMap::new();
        let mut tiles: Vec<(TileFeature, *mut f64)> = Vec::new();
        for name in precompute {
            if let Some(feature) = tile_feature(name) {
                let out = unsafe { rc_enrich_new_output(ctx) };
                if !out.is_null() {
                    precomputed.insert(name.clone(), out);
                    tiles.push((feature, out));
                }
            }
        }

        let mut ring = match ProjectedRing::create(store, projection.clone(), slots) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("# enrich: projected transport unavailable: {}", e);
                unsafe {
                    for (_, p) in precomputed {
                        rc_enrich_release_output(p);
                    }
                    rc_enrich_destroy(ctx);
                }
                return None;
            }
        };
        let mut dests: Vec<(*mut c_void, usize)> = Vec::with_capacity(sources.len());
        let group_rows = store.group_rows();
        let started = std::time::Instant::now();
        let mut upload_error: Option<String> = None;
        loop {
            let job = match ring.next_group() {
                Some(Ok(j)) => j,
                Some(Err(e)) => {
                    upload_error = Some(e);
                    break;
                }
                None => break,
            };
            let slot = job.slot;
            let len = job.len;
            let at = job.group * group_rows;
            dests.clear();
            for source in sources {
                let (ptr, off) = match source.destination {
                    DeviceColumn::Time => (
                        unsafe { rc_enrich_times_ptr(ctx) } as *mut c_void,
                        at * std::mem::size_of::<i64>(),
                    ),
                    DeviceColumn::Value(which) => (
                        unsafe { rc_enrich_value_ptr(ctx, which) } as *mut c_void,
                        at * std::mem::size_of::<f64>(),
                    ),
                };
                if ptr.is_null() {
                    upload_error = Some("projected transport: null device column".to_string());
                    break;
                }
                dests.push((ptr, off));
            }
            if upload_error.is_some() {
                break;
            }
            if let Err(e) = ring.upload(job, &dests) {
                upload_error = Some(e);
                break;
            }
            if let Err(e) = ring.compute_begin(slot) {
                upload_error = Some(e);
                break;
            }
            if let Err(e) = ring.launch_seg_id(ctx as *mut c_void, at, len) {
                upload_error = Some(e);
                break;
            }
            for (feature, out) in &tiles {
                if let Err(e) =
                    ring.launch_feature(ctx as *mut c_void, *feature, at, len, *out)
                {
                    upload_error = Some(e);
                    break;
                }
            }
            if upload_error.is_some() {
                break;
            }
            if let Err(e) = ring.compute_end(slot) {
                upload_error = Some(e);
                break;
            }
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let finish = ring.finish();
        let ring_stats = ring.stats();
        if let Some(e) = upload_error {
            eprintln!("# enrich: projected transport failed: {}; using the whole-column fallback", e);
            unsafe {
                for (_, p) in precomputed {
                    rc_enrich_release_output(p);
                }
                rc_enrich_destroy(ctx);
            }
            return None;
        }
        if let Err(e) = finish {
            eprintln!("# enrich: projected transport failed: {}; using the whole-column fallback", e);
            unsafe {
                for (_, p) in precomputed {
                    rc_enrich_release_output(p);
                }
                rc_enrich_destroy(ctx);
            }
            return None;
        }
        unsafe {
            rc_enrich_note_projected_upload(ctx, ring_stats.bytes_read, elapsed_ms);
        }
        let mut gpu = EnrichGpu {
            ctx,
            n,
            stats: GpuStats::default(),
            precomputed,
            ring_bytes: ring_stats.ring_bytes,
        };
        // One ledger entry for the projected upload: pinned ring bytes copied,
        // host wait time and the final join.
        gpu.stats.add_batch(
            0.0,
            0.0,
            0.0,
            0.0,
            elapsed_ms,
            0.0,
            0.0,
            elapsed_ms,
            ring_stats.bytes_read,
            0,
            0,
            n as u64,
            false,
        );
        Some(gpu)
    }

    /// Names of the tile-precomputed features still resident on the device.
    pub fn precomputed_names(&self) -> Vec<String> {
        self.precomputed.keys().cloned().collect()
    }

    /// Download and remove one tile-precomputed feature, if it exists.
    pub fn take_precomputed(&mut self, name: &str) -> Option<Vec<f64>> {
        let ptr = self.precomputed.remove(name)?;
        let mut out = vec![0.0f64; self.n];
        let started = std::time::Instant::now();
        let rc = unsafe { rc_enrich_download_ptr(ptr, out.as_mut_ptr(), self.n as c_int) };
        let ms = started.elapsed().as_secs_f64() * 1e3;
        unsafe {
            rc_enrich_note_projected_download(
                self.ctx,
                (self.n as u64) * 8,
                ms,
            );
            rc_enrich_release_output(ptr);
        }
        if rc != 0 {
            return None;
        }
        self.stats.add_batch(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            ms,
            0.0,
            ms,
            0,
            (self.n as u64) * 8,
            0,
            self.n as u64,
            false,
        );
        Some(out)
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

    pub fn ma(&mut self, w: usize, mode: i32) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_ma(ctx, w as c_int, mode as c_int, p) })
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

    /// Compact `don96`: the same blocked extrema, centred on zero.
    pub fn donchian_centred(&mut self, w: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_donchian_centred(ctx, w as c_int, p) })
    }

    /// Stochastic `%K` (`which = 0`) or its 3-bar mean `%D` (`which = 1`).
    pub fn stoch(&mut self, which: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_stoch(ctx, which as c_int, p) })
    }

    /// Parameterised ATR. `which`: 0 price, 1 NATR, 2 fraction, 3 legacy #125.
    pub fn atr(&mut self, period: usize, which: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_atr(ctx, period as c_int, which as c_int, p) })
    }

    pub fn batr(&mut self) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_batr(ctx, p) })
    }

    pub fn gapatr(&mut self) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_gapatr(ctx, p) })
    }

    pub fn volrat(&mut self) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_volrat(ctx, p) })
    }

    /// ADX/DI selector: 0 ADX, 1 +DI, 2 -DI, 3 DI difference.
    pub fn adx(&mut self, which: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_adx(ctx, which as c_int, p) })
    }

    /// Bollinger z-score (mode 0) or two-sigma clip (mode 1).
    pub fn bb_z(&mut self, w: usize, mode: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_bb_z(ctx, w as c_int, mode as c_int, p) })
    }

    pub fn ma_slope(&mut self, w: usize, lag: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_ma_slope(ctx, w as c_int, lag as c_int, p) })
    }

    pub fn tod(&mut self, want_sin: bool) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_tod(ctx, want_sin as c_int, p) })
    }

    /// Exotic selector: 0 rmean5, 1 rstd5, 2 close_pos20, 3 ou_hl96, 4 acf1,
    /// 5 vr4, 6 rskew96, 7 jump_share, 8 semivar_asym, 9 cusum_abs_z.
    pub fn exotic(&mut self, which: usize) -> Option<Vec<f64>> {
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_exotic(ctx, which as c_int, p) })
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

    /// Build the device `(time, symbol, row)` permutation and cached timestamp
    /// segments once. Every [`rank`](Self::rank) column reuses them.
    ///
    /// `times` are the resident columns uploaded by [`new`](Self::new); `member`
    /// is the point-in-time mask (`None` means every row is a member). Rows
    /// that are non-members are excluded exactly like the CPU reference.
    pub fn prepare_cross_section(
        &mut self,
        sym_key: &[u32],
        member: Option<&[bool]>,
    ) -> bool {
        if sym_key.len() != self.n {
            return false;
        }
        let mask: Option<Vec<u8>> =
            member.map(|m| m.iter().map(|&b| b as u8).collect());
        if let Some(m) = &mask {
            if m.len() != self.n {
                return false;
            }
        }
        let started = std::time::Instant::now();
        let rc = unsafe {
            rc_enrich_xs_prepare(
                self.ctx,
                sym_key.as_ptr(),
                mask.as_ref().map_or(std::ptr::null(), |m| m.as_ptr()),
            )
        };
        let ok = rc == 0;
        self.record(started, ok);
        ok
    }

    /// Cross-sectional ranks against the cached device permutation. The stable
    /// segmented sort keeps the CPU's `(time, symbol, row)` tie order and treats
    /// `-0.0`/`+0.0` as one tie; non-members and non-finite values are `NaN`,
    /// and a timestamp with fewer than 20 finite PIT members is all `NaN`.
    pub fn rank(&mut self, src: &[f64]) -> Option<Vec<f64>> {
        if src.len() != self.n {
            return None;
        }
        let ctx = self.ctx;
        self.one(|p| unsafe { rc_enrich_xs_rank(ctx, src.as_ptr(), p) })
    }

    /// #300: how many output columns fit in the 64 MiB per-family arena.
    /// Zero means the store is too wide for the arena and the caller keeps the
    /// per-column staging path.
    pub fn batch_capacity(&self) -> usize {
        let cap = unsafe { rc_enrich_batch_capacity(self.ctx) };
        cap.max(0) as usize
    }

    /// #300: launch several family kernels back-to-back and copy their outputs
    /// out with one pinned, chunked `cudaMemcpy2DAsync` per row chunk. The
    /// returned block borrows the context until the caller has written it.
    pub fn batch(&mut self, ops: &[BatchOp]) -> Option<BatchBlock<'_>> {
        if ops.is_empty() {
            return None;
        }
        let slots: usize = ops.iter().map(BatchOp::slots).sum();
        if slots == 0 || slots > self.batch_capacity() {
            return None;
        }
        let started = std::time::Instant::now();
        let rc = unsafe { rc_enrich_batch_run(self.ctx, ops.as_ptr(), ops.len() as c_int) };
        let ok = rc == 0;
        self.record(started, ok);
        if !ok {
            return None;
        }
        let rows = self.n;
        Some(BatchBlock {
            gpu: self,
            count: slots,
            rows,
        })
    }

    /// Opaque device context for the fused feature module (#297).
    pub(crate) fn raw(&self) -> *mut std::ffi::c_void {
        self.ctx as *mut std::ffi::c_void
    }

    /// A zeroed output buffer for one column.
    pub(crate) fn blank(&self) -> Vec<f64> {
        self.out()
    }

    /// Record one fused-family call in the same statistics ledger as the
    /// single-column kernels.
    pub(crate) fn record_fused(&mut self, started: std::time::Instant, ok: bool) {
        self.record(started, ok);

    }

    pub fn stats(&self) -> GpuStats {
        self.stats
    }

    /// Pinned ring bytes held by the projected transport (0 for the
    /// whole-column path).
    pub fn ring_bytes(&self) -> u64 {
        self.ring_bytes
    }
}

impl Drop for EnrichGpu {
    fn drop(&mut self) {
        for (_, p) in std::mem::take(&mut self.precomputed) {
            unsafe { rc_enrich_release_output(p) };
        }
        unsafe { rc_enrich_destroy(self.ctx) };
    }
}

/// Limits the bytes the context may add after creation (tile outputs and
/// fused planes). Zero removes the limit. Lets tests exhaust that share
/// without exhausting the device.
pub fn set_device_budget_for_tests(bytes: usize) {
    unsafe { rc_enrich_set_test_budget(bytes) };
}

/// Refuses the `at`-th device allocation made from now on, before cudaMalloc is
/// called. Zero turns the hook off. `u64::MAX` only counts. Resets the count.
pub fn set_alloc_fault_for_tests(at: u64) {
    unsafe { rc_enrich_set_test_alloc_fault(at) };
}

/// Allocations seen since `set_alloc_fault_for_tests` was last called.
pub fn alloc_count_for_tests() -> u64 {
    unsafe { rc_enrich_test_alloc_count() }
}

/// Launch sites that can be made to fail before any kernel is queued.
pub const LAUNCH_FAULT_SITES: [(&str, i32); 6] = [
    ("ring segment-id", 1),
    ("ring tile feature", 2),
    ("batch launch", 3),
    ("cross-section prepare", 4),
    ("cross-section rank", 5),
    ("element-wise and tail", 6),
];

/// Launches refused at the armed site since `set_launch_fault_for_tests`.
pub fn launch_fault_hits_for_tests() -> u64 {
    unsafe { rc_enrich_test_launch_hits() }
}

/// Makes every launch at `site` fail. Zero turns the hook off.
pub fn set_launch_fault_for_tests(site: i32) {
    unsafe { rc_enrich_set_test_launch_fault(site) };
}

/// Map a catalogue name to the tiled kernel the transport can run while the
/// next group copies.
pub fn tile_feature(name: &str) -> Option<TileFeature> {
    if let Some(k) = name.strip_prefix('r').and_then(|s| s.parse::<usize>().ok()) {
        return Some(TileFeature::Ret(k));
    }
    if let Some(k) = name
        .strip_prefix("lag_r1_")
        .and_then(|s| s.parse::<usize>().ok())
    {
        return Some(TileFeature::Lag(k));
    }
    match name {
        "range1" => Some(TileFeature::Range),
        "clv1" => Some(TileFeature::Clv),
        "dow_sin" => Some(TileFeature::Calendar(0)),
        "dow_cos" => Some(TileFeature::Calendar(1)),
        "hour_sin" => Some(TileFeature::Calendar(2)),
        "hour_cos" => Some(TileFeature::Calendar(3)),
        "month_sin" => Some(TileFeature::Calendar(4)),
        "month_cos" => Some(TileFeature::Calendar(5)),
        "dom_sin" => Some(TileFeature::Calendar(6)),
        "dom_cos" => Some(TileFeature::Calendar(7)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference: stable `(time, symbol, row)` groups, finite PIT members
    /// only, positional ranks scaled to `[-1, 1]`.
    fn cpu_rank(
        src: &[f64],
        times: &[i64],
        sym: &[u32],
        member: &[bool],
    ) -> Vec<f64> {
        let rows = src.len();
        let mut order: Vec<u32> = (0..rows as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            times[a]
                .cmp(&times[b])
                .then(sym[a].cmp(&sym[b]))
                .then(a.cmp(&b))
        });
        let mut out = vec![f64::NAN; rows];
        let mut i = 0;
        while i < order.len() {
            let t = times[order[i] as usize];
            let mut j = i;
            while j < order.len() && times[order[j] as usize] == t {
                j += 1;
            }
            let mut group: Vec<usize> = (i..j)
                .map(|k| order[k] as usize)
                .filter(|&r| src[r].is_finite() && member[r])
                .collect();
            group.sort_unstable_by(|&a, &b| {
                src[a]
                    .partial_cmp(&src[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(sym[a].cmp(&sym[b]))
                    .then(a.cmp(&b))
            });
            let m = group.len();
            if m >= 20 {
                for (rank, &r) in group.iter().enumerate() {
                    out[r] = 2.0 * rank as f64 / (m - 1) as f64 - 1.0;
                }
            }
            i = j;
        }
        out
    }

    fn fixture(rows_per_symbol: usize, symbols: usize) -> (Vec<i64>, Vec<u32>, Vec<f64>, Vec<bool>, Vec<u32>, Vec<u32>) {
        let n = rows_per_symbol * symbols;
        let mut times = Vec::with_capacity(n);
        let mut sym = Vec::with_capacity(n);
        let mut src = Vec::with_capacity(n);
        let mut member = Vec::with_capacity(n);
        let mut seg_start = Vec::with_capacity(symbols);
        let mut seg_len = Vec::with_capacity(symbols);
        for k in 0..symbols {
            seg_start.push((k * rows_per_symbol) as u32);
            seg_len.push(rows_per_symbol as u32);
            for i in 0..rows_per_symbol {
                times.push(1_700_000_000i64 + i as i64 * 900);
                sym.push(k as u32);
                // Quantised values create many ties; the 0/-0 rows make the
                // signed-zero policy observable.
                let v = if i % 4093 == 0 {
                    if k % 2 == 0 { 0.0 } else { -0.0 }
                } else if i % 1009 == 0 && k == 3 {
                    f64::NAN
                } else {
                    (((i * 7 + k * 3) % 31) as f64) / 10.0
                };
                src.push(v);
                // One symbol drops out on every 17th timestamp; with 21
                // symbols that leaves 20 at most timestamps and 19 whenever a
                // NaN or a second dropout is present.
                member.push(!(k == 5 && i % 17 == 0));
            }
        }
        (times, sym, src, member, seg_start, seg_len)
    }

    fn open_context(
        times: &[i64],
        seg_start: &[u32],
        seg_len: &[u32],
    ) -> EnrichGpu {
        let n = times.len();
        let close = vec![100.0f64; n];
        EnrichGpu::new(
            times,
            &close,
            &close,
            &close,
            &close,
            &vec![0.0; n],
            &vec![0.0; n],
            seg_start,
            seg_len,
            900,
        )
        .expect("CUDA context")
    }

    #[test]
    fn device_permutation_and_segmented_rank_match_cpu_reference() {
        if !crate::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let (times, sym, src, member, seg_start, seg_len) = fixture(20_000, 21);
        let mut gpu = open_context(&times, &seg_start, &seg_len);
        assert!(gpu.prepare_cross_section(&sym, Some(&member)));
        // The permutation is idempotent and shared by every column.
        assert!(gpu.prepare_cross_section(&sym, Some(&member)));
        let got = gpu.rank(&src).expect("device rank");
        let want = cpu_rank(&src, &times, &sym, &member);
        let finite = want.iter().filter(|v| v.is_finite()).count();
        assert!(finite > 0, "fixture must produce finite ranks");
        for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                a.is_nan(),
                b.is_nan(),
                "row {i}: CPU {a:?} vs GPU {b:?}"
            );
            if a.is_finite() {
                assert_eq!(a.to_bits(), b.to_bits(), "row {i}: CPU {a:?} vs GPU {b:?}");
            }
        }
        // A non-member's own rank is NaN even when its source is finite.
        assert!(got
            .iter()
            .enumerate()
            .filter(|(i, _)| !member[*i])
            .all(|(_, v)| v.is_nan()));
    }

    #[test]
    fn device_rank_is_identical_across_reuse_and_column_order() {
        if !crate::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let (times, sym, src, member, seg_start, seg_len) = fixture(10_000, 21);
        let mut gpu = open_context(&times, &seg_start, &seg_len);
        assert!(gpu.prepare_cross_section(&sym, Some(&member)));
        let first = gpu.rank(&src).unwrap();
        let second = gpu.rank(&src).unwrap();
        let other: Vec<f64> = src.iter().map(|v| -v).collect();
        let other_rank = gpu.rank(&other).unwrap();
        let third = gpu.rank(&src).unwrap();
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            third.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a different column must not change a cached permutation"
        );
        assert_eq!(
            other_rank.iter().filter(|v| v.is_finite()).count(),
            first.iter().filter(|v| v.is_finite()).count()
        );
    }
}
