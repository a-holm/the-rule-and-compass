//! Device benchmark joins, relative returns, rolling beta and per-timestamp
//! cross means (issue #299).
//!
//! The CPU producers in `src/enrich.rs` (`benchmark_return`, `benchmark_vol`,
//! `benchmark_beta`, `market_state_pass`) are the reference. The CUDA kernels
//! live in `src/cuda/bench.cu`.
//!
//! Benchmark side. The benchmark symbol (BTC, optionally ETH) is a small sorted
//! `(time, value)` array. [`BenchSeries`] is that array with its eight value
//! planes; it is uploaded once. Each base row finds its own timestamp by exact
//! equality, so a missing benchmark bar is `NaN`, never a stale value.
//!
//! Parity tiers are declared per column:
//!
//! * `bitwise`: the column is a gather of host-computed benchmark values, so it
//!   is the same bytes as the CPU map lookup;
//! * `1e-12`: the column also uses the asset's own log return (`rel_*`,
//!   `beta_*`), computed with the CUDA `log`. The rolling window is summed in the
//!   CPU order and the NaN mask depends only on finite tests of exact values, so
//!   the masks match and the values differ only through that `log`.
//!
//! Cross-section side. The market columns are per-timestamp means over the
//! point-in-time members. One device thread walks one timestamp group in the
//! CPU's `(time, symbol, row)` order, so the sums use the CPU addition order: for
//! an identical source column the kernels are bit-for-bit equal to the CPU. The
//! sources are the asset `r1/r16/r96/vol96` columns, which are themselves
//! `1e-12` columns when the device produced them, so those market columns are
//! declared `1e-12` end to end. `btc_volume_share` reads only raw volume and is
//! `bitwise`. The kernels reuse the permutation that #295 builds for the ranks.
//!
//! Derivative columns stay `available:false`; the compact `beta384` and `rel_btc96`
//! are device columns here because they share the benchmark join.

/// Value planes of a benchmark series, in the order the kernels index them.
pub const PLANE_R1: usize = 0;
pub const PLANE_R16: usize = 1;
pub const PLANE_R20: usize = 2;
pub const PLANE_R96: usize = 3;
pub const PLANE_R672: usize = 4;
pub const PLANE_VOL20: usize = 5;
pub const PLANE_VOL96: usize = 6;
pub const PLANE_VOLUME: usize = 7;
pub const PLANE_COUNT: usize = 8;

/// Which benchmark a column reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bench {
    Btc,
    Eth,
}

impl Bench {
    pub fn index(self) -> usize {
        match self {
            Bench::Btc => 0,
            Bench::Eth => 1,
        }
    }
}

/// One benchmark symbol: sorted unique timestamps and the value planes.
#[derive(Clone, Debug, Default)]
pub struct BenchSeries {
    pub times: Vec<i64>,
    pub planes: [Vec<f64>; PLANE_COUNT],
}

impl BenchSeries {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    fn consistent(&self) -> bool {
        self.planes.iter().all(|p| p.len() == self.times.len())
            && self.times.windows(2).all(|w| w[0] < w[1])
    }
}

/// Parity tier of one registered column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Bitwise,
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

/// What a benchmark device column computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Gather one benchmark plane at the row timestamp.
    Lookup(usize),
    /// Own k-bar log return minus the benchmark plane for horizon k.
    Rel(usize),
    /// Rolling beta over `w` bars with the CPU `cov/var`, clip +-5.
    Beta(usize),
    /// `beta384`: as `Beta` with the covariance clipped to +-0.05 first and no
    /// "benchmark against itself" shortcut.
    BetaClipCov(usize),
}

/// One device-produced benchmark column.
pub struct BenchColumn {
    pub name: &'static str,
    pub bench: Bench,
    pub kind: Kind,
    pub tier: Tier,
}

const fn lookup(name: &'static str, bench: Bench, plane: usize) -> BenchColumn {
    BenchColumn { name, bench, kind: Kind::Lookup(plane), tier: Tier::Bitwise }
}

const fn rel(name: &'static str, bench: Bench, k: usize) -> BenchColumn {
    BenchColumn { name, bench, kind: Kind::Rel(k), tier: Tier::Tolerance1e12 }
}

const fn beta(name: &'static str, bench: Bench, w: usize) -> BenchColumn {
    BenchColumn { name, bench, kind: Kind::Beta(w), tier: Tier::Tolerance1e12 }
}

/// The 30 benchmark device columns: the 13 BTC and 13 ETH columns of the #273
/// family, the legacy `cross_ret1` / `cross_beta20`, and the compact `rel_btc96`
/// / `beta384`. `btc_r96` and `btc_vol96` are shared with the compact profile.
pub const BENCH_DEVICE_COLUMNS: &[BenchColumn] = &[
    lookup("btc_r1", Bench::Btc, PLANE_R1),
    lookup("btc_r16", Bench::Btc, PLANE_R16),
    lookup("btc_r20", Bench::Btc, PLANE_R20),
    lookup("btc_r96", Bench::Btc, PLANE_R96),
    lookup("btc_r672", Bench::Btc, PLANE_R672),
    lookup("btc_vol20", Bench::Btc, PLANE_VOL20),
    lookup("btc_vol96", Bench::Btc, PLANE_VOL96),
    rel("rel_btc_r1", Bench::Btc, 1),
    rel("rel_btc_r16", Bench::Btc, 16),
    rel("rel_btc_r20", Bench::Btc, 20),
    rel("rel_btc_r96", Bench::Btc, 96),
    beta("beta_btc_60", Bench::Btc, 60),
    beta("beta_btc_384", Bench::Btc, 384),
    lookup("eth_r1", Bench::Eth, PLANE_R1),
    lookup("eth_r16", Bench::Eth, PLANE_R16),
    lookup("eth_r20", Bench::Eth, PLANE_R20),
    lookup("eth_r96", Bench::Eth, PLANE_R96),
    lookup("eth_r672", Bench::Eth, PLANE_R672),
    lookup("eth_vol20", Bench::Eth, PLANE_VOL20),
    lookup("eth_vol96", Bench::Eth, PLANE_VOL96),
    rel("rel_eth_r1", Bench::Eth, 1),
    rel("rel_eth_r16", Bench::Eth, 16),
    rel("rel_eth_r20", Bench::Eth, 20),
    rel("rel_eth_r96", Bench::Eth, 96),
    beta("beta_eth_60", Bench::Eth, 60),
    beta("beta_eth_384", Bench::Eth, 384),
    lookup("cross_ret1", Bench::Btc, PLANE_R1),
    beta("cross_beta20", Bench::Btc, 20),
    rel("rel_btc96", Bench::Btc, 96),
    BenchColumn {
        name: "beta384",
        bench: Bench::Btc,
        kind: Kind::BetaClipCov(384),
        tier: Tier::Tolerance1e12,
    },
];

/// Cross-section columns produced by the per-timestamp group kernels.
pub const MARKET_DEVICE_COLUMNS: &[(&str, Tier)] = &[
    ("mkt_r16", Tier::Tolerance1e12),
    ("mkt_ew_ret16", Tier::Tolerance1e12),
    ("breadth", Tier::Tolerance1e12),
    ("mkt_breadth", Tier::Tolerance1e12),
    ("dispersion", Tier::Tolerance1e12),
    ("mkt_dispersion", Tier::Tolerance1e12),
    ("rel_mkt16", Tier::Tolerance1e12),
    ("mkt_r96", Tier::Tolerance1e12),
    ("rel_mkt96", Tier::Tolerance1e12),
    ("mkt_vol96", Tier::Tolerance1e12),
    ("mkt_ew_ret1", Tier::Tolerance1e12),
    ("avg_corr", Tier::Tolerance1e12),
    ("btc_volume_share", Tier::Bitwise),
];

/// The benchmark column table entry for `name`.
pub fn bench_column(name: &str) -> Option<&'static BenchColumn> {
    BENCH_DEVICE_COLUMNS.iter().find(|c| c.name == name)
}

/// True when the benchmark device path materialises `name`.
pub fn is_device_bench(name: &str) -> bool {
    bench_column(name).is_some()
}

/// True when a per-timestamp market kernel materialises `name`.
pub fn is_device_market(name: &str) -> bool {
    MARKET_DEVICE_COLUMNS.iter().any(|(n, _)| *n == name)
}

/// Declared parity tier for a benchmark or market device column.
pub fn device_tier(name: &str) -> Option<&'static str> {
    if let Some(c) = bench_column(name) {
        return Some(c.tier.as_str());
    }
    MARKET_DEVICE_COLUMNS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t.as_str())
}

/// Group statistics of one source column over the point-in-time members of each
/// timestamp. Vectors that were not requested are empty.
#[derive(Debug, Default)]
pub struct GroupStats {
    pub mean: Vec<f64>,
    pub breadth: Vec<f64>,
    pub dispersion: Vec<f64>,
    pub rel: Vec<f64>,
}

pub const WANT_MEAN: u32 = 1;
pub const WANT_BREADTH: u32 = 2;
pub const WANT_DISPERSION: u32 = 4;
pub const WANT_REL: u32 = 8;

/// A device that computes per-timestamp cross means. The enrichment market pass
/// calls it first and falls back to the CPU reference when a call returns
/// `None`.
pub trait MarketDevice {
    fn group_stats(&mut self, src: &[f64], want: u32) -> Option<GroupStats>;
    fn volume_share(&mut self, volume: &[f64], bench: &BenchSeries) -> Option<Vec<f64>>;
    fn avg_corr(&mut self, r1: &[f64]) -> Option<Vec<f64>>;
}

#[cfg(feature = "cuda")]
pub use device::{compute, BenchGpu};

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::gpu_stats::GpuStats;
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

    #[repr(C)]
    struct RawSeries {
        nb: c_int,
        times: *const i64,
        planes: [*const f64; PLANE_COUNT],
    }

    enum Raw {}

    extern "C" {
        fn rc_bench_create(
            times: *const i64,
            close: *const f64,
            n: c_int,
            seg_start: *const c_int,
            seg_len: *const c_int,
            nseg: c_int,
            step: i64,
            bench0: *const RawSeries,
            bench1: *const RawSeries,
            self_flags: *const u8,
        ) -> *mut Raw;
        fn rc_bench_destroy(ctx: *mut Raw);
        fn rc_bench_last_stats(ctx: *mut Raw, out: *mut RawStats);
        fn rc_bench_column(
            ctx: *mut Raw,
            bench: c_int,
            kind: c_int,
            a: c_int,
            use_self: c_int,
            out: *mut f64,
        ) -> c_int;
        fn rc_bench_market_group(
            ctx: *mut c_void,
            src: *const f64,
            want: u32,
            mean: *mut f64,
            breadth: *mut f64,
            disp: *mut f64,
            rel: *mut f64,
        ) -> c_int;
        fn rc_bench_market_volume_share(
            ctx: *mut c_void,
            vol: *const f64,
            bt: *const i64,
            bvol: *const f64,
            nb: c_int,
            out: *mut f64,
        ) -> c_int;
        fn rc_bench_market_avg_corr(ctx: *mut c_void, r1: *const f64, out: *mut f64) -> c_int;
    }

    fn raw_series(s: Option<&BenchSeries>) -> RawSeries {
        match s {
            Some(s) if !s.is_empty() => RawSeries {
                nb: s.len() as c_int,
                times: s.times.as_ptr(),
                planes: std::array::from_fn(|p| s.planes[p].as_ptr()),
            },
            _ => RawSeries {
                nb: 0,
                times: std::ptr::null(),
                planes: [std::ptr::null(); PLANE_COUNT],
            },
        }
    }

    /// One resident benchmark-join context: the base times and close, the
    /// segment table and both benchmark series, uploaded once.
    pub struct BenchGpu {
        ctx: *mut Raw,
        n: usize,
        have: [bool; 2],
        stats: GpuStats,
    }

    impl BenchGpu {
        /// `self_flags[b * nseg + s]` is 1 when segment `s` is benchmark `b`
        /// itself (CPU `s.symbol == cross.symbol`).
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            times: &[i64],
            close: &[f64],
            seg_start: &[u32],
            seg_len: &[u32],
            step: i64,
            btc: Option<&BenchSeries>,
            eth: Option<&BenchSeries>,
            self_flags: &[u8],
        ) -> Option<Self> {
            let nseg = seg_start.len();
            if times.is_empty()
                || close.len() != times.len()
                || nseg == 0
                || seg_len.len() != nseg
                || self_flags.len() != 2 * nseg
                || btc.map_or(true, |b| b.is_empty() || !b.consistent())
                || eth.map_or(false, |b| !b.is_empty() && !b.consistent())
            {
                return None;
            }
            let starts: Vec<c_int> = seg_start.iter().map(|v| *v as c_int).collect();
            let lens: Vec<c_int> = seg_len.iter().map(|v| *v as c_int).collect();
            let r0 = raw_series(btc);
            let r1 = raw_series(eth);
            let ctx = unsafe {
                rc_bench_create(
                    times.as_ptr(),
                    close.as_ptr(),
                    times.len() as c_int,
                    starts.as_ptr(),
                    lens.as_ptr(),
                    nseg as c_int,
                    step,
                    &r0,
                    &r1,
                    self_flags.as_ptr(),
                )
            };
            if ctx.is_null() {
                return None;
            }
            let mut g = BenchGpu {
                ctx,
                n: times.len(),
                have: [r0.nb > 0, r1.nb > 0],
                stats: GpuStats::default(),
            };
            g.record(std::time::Instant::now(), true);
            Some(g)
        }

        fn record(&mut self, started: std::time::Instant, ok: bool) {
            let mut raw = RawStats::default();
            unsafe { rc_bench_last_stats(self.ctx, &mut raw) };
            let outer_ms = started.elapsed().as_secs_f64() * 1e3;
            let windows = raw.alloc_ms + raw.h2d_ms + raw.d2h_ms;
            self.stats.add_batch(
                raw.kernel_ms,
                raw.launch_ms,
                raw.sync_ms,
                raw.alloc_ms,
                raw.h2d_ms,
                raw.d2h_ms,
                (outer_ms - windows - raw.kernel_ms).max(0.0),
                outer_ms.max(raw.wall_ms),
                raw.h2d_bytes,
                raw.d2h_bytes,
                0,
                self.n as u64,
                !ok || raw.failed != 0,
            );
        }

        /// Compute one registered column and download it.
        pub fn column(&mut self, col: &BenchColumn) -> Option<Vec<f64>> {
            if !self.have[col.bench.index()] {
                return None;
            }
            let (kind, a, use_self) = match col.kind {
                Kind::Lookup(p) => (0, p as c_int, 0),
                Kind::Rel(k) => (1, k as c_int, 0),
                Kind::Beta(w) => (2, w as c_int, 1),
                Kind::BetaClipCov(w) => (3, w as c_int, 0),
            };
            let started = std::time::Instant::now();
            let mut out = vec![0.0; self.n];
            let rc = unsafe {
                rc_bench_column(self.ctx, col.bench.index() as c_int, kind, a, use_self, out.as_mut_ptr())
            };
            self.record(started, rc == 0);
            if rc == 0 {
                Some(out)
            } else {
                None
            }
        }

        pub fn stats(&self) -> GpuStats {
            self.stats
        }
    }

    impl Drop for BenchGpu {
        fn drop(&mut self) {
            unsafe { rc_bench_destroy(self.ctx) };
        }
    }

    /// Compute every requested benchmark column on the device.
    ///
    /// `None` means the context or a kernel failed (or a column needs a
    /// benchmark that was not supplied); the caller keeps the CPU reference for
    /// the whole family rather than emit a partial one.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        times: &[i64],
        close: &[f64],
        seg_start: &[u32],
        seg_len: &[u32],
        step: i64,
        btc: Option<&BenchSeries>,
        eth: Option<&BenchSeries>,
        self_flags: &[u8],
        requested: &[String],
    ) -> Option<(Vec<(String, Vec<f64>)>, GpuStats)> {
        let cols: Vec<&BenchColumn> = requested
            .iter()
            .map(|n| bench_column(n))
            .collect::<Option<Vec<_>>>()?;
        let mut gpu = BenchGpu::new(times, close, seg_start, seg_len, step, btc, eth, self_flags)?;
        let mut out = Vec::with_capacity(cols.len());
        for col in cols {
            out.push((col.name.to_string(), gpu.column(col)?));
        }
        Some((out, gpu.stats()))
    }

    impl crate::enrich_gpu::EnrichGpu {
        fn market_call<T>(&mut self, call: impl FnOnce(*mut c_void) -> Option<T>) -> Option<T> {
            let started = std::time::Instant::now();
            let out = call(self.raw());
            self.record_fused(started, out.is_some());
            out
        }
    }

    impl MarketDevice for crate::enrich_gpu::EnrichGpu {
        fn group_stats(&mut self, src: &[f64], want: u32) -> Option<GroupStats> {
            let n = src.len();
            let mut st = GroupStats::default();
            let alloc = |bit: u32| if want & bit != 0 { vec![0.0; n] } else { Vec::new() };
            st.mean = alloc(WANT_MEAN);
            st.breadth = alloc(WANT_BREADTH);
            st.dispersion = alloc(WANT_DISPERSION);
            st.rel = alloc(WANT_REL);
            let ptr = |v: &mut Vec<f64>| if v.is_empty() { std::ptr::null_mut() } else { v.as_mut_ptr() };
            self.market_call(|ctx| {
                let rc = unsafe {
                    rc_bench_market_group(
                        ctx,
                        src.as_ptr(),
                        want,
                        ptr(&mut st.mean),
                        ptr(&mut st.breadth),
                        ptr(&mut st.dispersion),
                        ptr(&mut st.rel),
                    )
                };
                if rc == 0 {
                    Some(())
                } else {
                    None
                }
            })?;
            Some(st)
        }

        fn volume_share(&mut self, volume: &[f64], bench: &BenchSeries) -> Option<Vec<f64>> {
            if bench.is_empty() || !bench.consistent() {
                return None;
            }
            let mut out = vec![0.0; volume.len()];
            self.market_call(|ctx| {
                let rc = unsafe {
                    rc_bench_market_volume_share(
                        ctx,
                        volume.as_ptr(),
                        bench.times.as_ptr(),
                        bench.planes[PLANE_VOLUME].as_ptr(),
                        bench.len() as c_int,
                        out.as_mut_ptr(),
                    )
                };
                if rc == 0 {
                    Some(())
                } else {
                    None
                }
            })?;
            Some(out)
        }

        fn avg_corr(&mut self, r1: &[f64]) -> Option<Vec<f64>> {
            let mut out = vec![0.0; r1.len()];
            self.market_call(|ctx| {
                let rc = unsafe { rc_bench_market_avg_corr(ctx, r1.as_ptr(), out.as_mut_ptr()) };
                if rc == 0 {
                    Some(())
                } else {
                    None
                }
            })?;
            Some(out)
        }
    }
}
