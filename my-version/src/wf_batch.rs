//! #309: batch walk-forward long-only P&L over `(candidate, window)`.
//!
//! One call scores a candidate set over several test windows. The rows of all
//! windows are laid out once in chain order. A chain is the run of rows that
//! share one position state:
//!
//! * [`Seam::Pooled`]: one chain per group, ascending in `(window, time)`. The
//!   position carries across the window seam. Every window block still pays
//!   the final exit, exactly like the scalar evaluator with a shared
//!   `previous` map.
//! * [`Seam::PerWindow`]: one chain per `(window, group)`. Every window starts
//!   flat.
//!
//! The layout is sorted by `(group, time)` inside a window whatever the input
//! row order, so a flattened `(time, group)` input never invents trades.
//!
//! Scores stream in through a callback, one `(candidate, window)` vector at a
//! time, and fill a declared tile of at most [`MAX_BATCH_TILE`] candidates.
//! The population is never materialised as `P x N`.
//!
//! Provenance. A cell computed by the device kernel is labelled
//! [`GPU_PROXY`]. The same arithmetic on the host is [`CPU_BATCH`]. A cell
//! whose candidate has a non-finite score on a finite-target row is computed
//! by the scalar evaluator ([`EXACT_CPU`]): that evaluator drops such rows
//! per candidate, which the shared layout cannot. Finalist values in the
//! report always come from the scalar evaluator.
//!
//! The baseline ladder is scored beside the candidates: cash, buy-and-hold and
//! the constant long/flat control have closed forms per window (exact CPU
//! `f64`), the matched-exposure random long control is priced from each
//! candidate's own positions with the exact CPU reduction. Every control
//! starts flat in every window.

use crate::baselines;
use crate::trading_gpu::{self, PnlParams, PnlTotals};
use crate::walkforward::{jnum, jstr, pnl_totals_json, reference_cell};
use std::collections::HashMap;
use std::time::Instant;

/// Label of a value computed by the device kernel.
pub const GPU_PROXY: &str = "gpu_proxy";
/// Label of the host mirror of the device kernel.
pub const CPU_BATCH: &str = "cpu_batch";
/// Label of a value computed by the scalar `f64` evaluator.
pub const EXACT_CPU: &str = "exact_cpu_f64";

/// Largest declared candidate tile.
pub const MAX_BATCH_TILE: usize = 64;
/// Default byte budget of the host and device score tile.
pub const DEFAULT_TILE_BYTES: usize = 256 << 20;
/// Default device block size.
pub const DEFAULT_THREADS: usize = 128;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seam {
    Pooled,
    PerWindow,
}

impl Seam {
    pub fn as_str(self) -> &'static str {
        match self {
            Seam::Pooled => "pooled",
            Seam::PerWindow => "per_window",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The device when this build and machine have one, the host otherwise.
    Auto,
    /// Force the host mirror.
    Cpu,
}

/// One test window's rows in input order.
#[derive(Clone, Debug)]
pub struct BatchWindow {
    pub label: String,
    pub time: Vec<f64>,
    pub group: Vec<u64>,
    pub target: Vec<f64>,
    /// Mean finite training target; sets the constant control's side.
    pub train_mean: f64,
}

pub struct BatchSpec<'a> {
    pub windows: &'a [BatchWindow],
    pub candidates: usize,
    /// `thresholds[candidate][window]`.
    pub thresholds: &'a [Vec<f64>],
    pub cost_side: f64,
    pub seam: Seam,
    pub backend: Backend,
    pub tile_bytes: usize,
    pub threads: usize,
    /// Seed of the matched-exposure random control; `None` skips it.
    pub random_seed: Option<u64>,
}

impl<'a> BatchSpec<'a> {
    pub fn new(
        windows: &'a [BatchWindow],
        candidates: usize,
        thresholds: &'a [Vec<f64>],
        cost_side: f64,
        seam: Seam,
    ) -> Self {
        BatchSpec {
            windows,
            candidates,
            thresholds,
            cost_side,
            seam,
            backend: Backend::Auto,
            tile_bytes: DEFAULT_TILE_BYTES,
            threads: DEFAULT_THREADS,
            random_seed: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub candidate: usize,
    pub window: usize,
    pub totals: PnlTotals,
    pub provenance: &'static str,
}

/// The ladder of one window.
#[derive(Clone, Debug, PartialEq)]
pub struct LadderWindow {
    pub window: usize,
    pub cash: PnlTotals,
    pub buy_and_hold: PnlTotals,
    pub constant: PnlTotals,
    pub constant_long: bool,
    /// Matched-exposure random long per candidate; `None` when it was not
    /// requested or the candidate has non-finite scores.
    pub random: Vec<Option<PnlTotals>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchStats {
    pub candidates: usize,
    pub windows: usize,
    pub cells: usize,
    /// Valid rows over all windows.
    pub rows: u64,
    /// `candidates x rows` evaluated by the batch.
    pub row_evals: u64,
    pub wall_s: f64,
    /// Seconds inside the tile evaluation (upload, kernel, download).
    pub device_s: f64,
    /// Seconds filling tiles (the score callback and the scatter).
    pub score_s: f64,
    pub layout_s: f64,
    pub reference_s: f64,
    pub candidates_per_s: f64,
    pub row_evals_per_s: f64,
    /// `device_s / wall_s`; zero on the host path.
    pub device_fraction: f64,
    pub tile_candidates: usize,
    /// Bytes allocated on the device by the context.
    pub device_bytes: u64,
    /// Bytes of the host score tile.
    pub host_tile_bytes: u64,
    pub device_tiles: usize,
    pub host_tiles: usize,
    pub fallback_candidates: usize,
}

#[derive(Clone, Debug)]
pub struct BatchResult {
    pub seam: Seam,
    /// `cuda` or `cpu`.
    pub backend: String,
    /// Why the host path ran, when it did.
    pub fallback_reason: Option<String>,
    /// Candidate-major: index `candidate * windows + window`.
    pub cells: Vec<Cell>,
    pub ladder: Vec<LadderWindow>,
    pub stats: BatchStats,
}

impl BatchResult {
    pub fn cell(&self, candidate: usize, window: usize) -> &Cell {
        &self.cells[candidate * self.stats.windows + window]
    }
}

/// `RcWfbCell` of `wf_batch.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RawCell {
    gross: f64,
    cost: f64,
    turnover: f64,
    net: f64,
    rows: u64,
    long_rows: u64,
    entries: u64,
    exits: u64,
    trades: u64,
    hits: u64,
    final_exits: u64,
    status: i32,
}

impl RawCell {
    fn totals(&self) -> PnlTotals {
        PnlTotals {
            rows: self.rows,
            long_rows: self.long_rows,
            entries: self.entries,
            exits: self.exits,
            trades: self.trades,
            hits: self.hits,
            final_exits: self.final_exits,
            gross: self.gross,
            cost: self.cost,
            turnover: self.turnover,
            net: self.net,
        }
    }
}

struct Layout {
    n: usize,
    returns: Vec<f64>,
    win: Vec<i32>,
    chain_off: Vec<i64>,
    chain_block0: Vec<i32>,
    block_window: Vec<usize>,
    /// Layout index of every input row, `NONE` for a dropped row.
    pos: Vec<Vec<u32>>,
    /// Per window: input row indices in `(group, time, index)` order.
    sorted: Vec<Vec<u32>>,
}

fn validate(spec: &BatchSpec) -> Result<(), String> {
    if spec.windows.is_empty() {
        return Err("batch needs at least one window".to_string());
    }
    if spec.candidates == 0 {
        return Err("batch needs at least one candidate".to_string());
    }
    if spec.thresholds.len() != spec.candidates {
        return Err(format!(
            "{} threshold rows for {} candidates",
            spec.thresholds.len(),
            spec.candidates
        ));
    }
    for (c, t) in spec.thresholds.iter().enumerate() {
        if t.len() != spec.windows.len() {
            return Err(format!(
                "candidate {}: {} thresholds for {} windows",
                c,
                t.len(),
                spec.windows.len()
            ));
        }
    }
    if !spec.cost_side.is_finite() || spec.cost_side < 0.0 {
        return Err(format!("cost per side must be finite and non-negative, got {}", spec.cost_side));
    }
    if spec.tile_bytes == 0 || spec.threads == 0 || spec.threads % 32 != 0 || spec.threads > 1024 {
        return Err(format!(
            "tile_bytes must be positive and threads a multiple of 32 up to 1024, got {} and {}",
            spec.tile_bytes, spec.threads
        ));
    }
    for (w, win) in spec.windows.iter().enumerate() {
        if win.group.len() != win.time.len() || win.target.len() != win.time.len() {
            return Err(format!(
                "window {}: {} times, {} groups, {} targets",
                w,
                win.time.len(),
                win.group.len(),
                win.target.len()
            ));
        }
        if win.time.iter().any(|t| !t.is_finite()) {
            return Err(format!("window {}: non-finite time", w));
        }
    }
    Ok(())
}

fn build_layout(windows: &[BatchWindow], seam: Seam) -> Result<Layout, String> {
    let mut keys: Vec<(u32, u32)> = Vec::new();
    for (w, win) in windows.iter().enumerate() {
        for i in 0..win.time.len() {
            if win.target[i].is_finite() {
                keys.push((w as u32, i as u32));
            }
        }
    }
    let n = keys.len();
    if n == 0 {
        return Err("no rows with a finite target".to_string());
    }
    if n > i32::MAX as usize || windows.len() > i32::MAX as usize {
        return Err("batch exceeds the device index range".to_string());
    }
    let cmp_in_window = |a: &(u32, u32), b: &(u32, u32)| {
        let wa = &windows[a.0 as usize];
        let wb = &windows[b.0 as usize];
        let (ia, ib) = (a.1 as usize, b.1 as usize);
        wa.group[ia]
            .cmp(&wb.group[ib])
            .then_with(|| wa.time[ia].total_cmp(&wb.time[ib]))
            .then_with(|| a.1.cmp(&b.1))
    };
    match seam {
        Seam::Pooled => keys.sort_by(|a, b| {
            let wa = &windows[a.0 as usize];
            let wb = &windows[b.0 as usize];
            wa.group[a.1 as usize]
                .cmp(&wb.group[b.1 as usize])
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| wa.time[a.1 as usize].total_cmp(&wb.time[b.1 as usize]))
                .then_with(|| a.1.cmp(&b.1))
        }),
        Seam::PerWindow => keys.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| cmp_in_window(a, b))),
    }

    let mut returns = Vec::with_capacity(n);
    let mut win = Vec::with_capacity(n);
    let mut chain_off: Vec<i64> = Vec::new();
    let mut chain_block0: Vec<i32> = Vec::new();
    let mut block_window: Vec<usize> = Vec::new();
    let mut pos: Vec<Vec<u32>> = windows.iter().map(|w| vec![NONE; w.time.len()]).collect();
    let mut prev: Option<(u32, u64)> = None;
    for (k, &(w, i)) in keys.iter().enumerate() {
        let g = windows[w as usize].group[i as usize];
        let new_chain = match (seam, prev) {
            (_, None) => true,
            (Seam::Pooled, Some((_, pg))) => pg != g,
            (Seam::PerWindow, Some((pw, pg))) => pw != w || pg != g,
        };
        let new_block = new_chain || prev.map(|(pw, _)| pw != w).unwrap_or(true);
        if new_chain {
            chain_off.push(k as i64);
            chain_block0.push(block_window.len() as i32);
        }
        if new_block {
            block_window.push(w as usize);
        }
        returns.push(windows[w as usize].target[i as usize]);
        win.push(w as i32);
        pos[w as usize][i as usize] = k as u32;
        prev = Some((w, g));
    }
    chain_off.push(n as i64);

    let mut sorted: Vec<Vec<u32>> = Vec::with_capacity(windows.len());
    for win in windows.iter() {
        let mut idx: Vec<u32> = (0..win.time.len() as u32)
            .filter(|&i| win.target[i as usize].is_finite())
            .collect();
        idx.sort_by(|&a, &b| {
            let (a, b) = (a as usize, b as usize);
            win.group[a]
                .cmp(&win.group[b])
                .then_with(|| win.time[a].total_cmp(&win.time[b]))
                .then_with(|| a.cmp(&b))
        });
        sorted.push(idx);
    }

    Ok(Layout {
        n,
        returns,
        win,
        chain_off,
        chain_block0,
        block_window,
        pos,
        sorted,
    })
}

/// The host mirror of `rc_wfb_kernel`: same row arithmetic, same fold order.
fn eval_tile_cpu(layout: &Layout, scores: &[f64], thr: &[f64], cands: usize, windows: usize, side: f64) -> Vec<RawCell> {
    let blocks = layout.block_window.len();
    let chains = layout.chain_off.len() - 1;
    let mut out = vec![RawCell::default(); cands * blocks];
    for c in 0..cands {
        let zc = &scores[c * layout.n..(c + 1) * layout.n];
        let tc = &thr[c * windows..(c + 1) * windows];
        for k in 0..chains {
            let lo = layout.chain_off[k] as usize;
            let hi = layout.chain_off[k + 1] as usize;
            let mut block = layout.chain_block0[k] as usize;
            let mut cur = layout.win[lo];
            let mut prev = 0.0f64;
            let mut acc = RawCell::default();
            for i in lo..hi {
                let w = layout.win[i];
                if w != cur {
                    out[c * blocks + block] = acc;
                    block += 1;
                    acc = RawCell::default();
                    cur = w;
                }
                let r = layout.returns[i];
                if !r.is_finite() {
                    acc.status = 2;
                    continue;
                }
                let s = zc[i];
                let p = if s.is_finite() && s > tc[w as usize] { 1.0 } else { 0.0 };
                let d = p - prev;
                let last = i + 1 == hi || layout.win[i + 1] != w;
                let gross = p * r;
                let mut cost = side * d.abs();
                if last && p != 0.0 {
                    cost += side * p.abs();
                    acc.final_exits += 1;
                    acc.exits += 1;
                }
                acc.rows += 1;
                if p != 0.0 {
                    acc.long_rows += 1;
                    if r > 0.0 {
                        acc.hits += 1;
                    }
                }
                if d > 0.0 {
                    acc.entries += 1;
                } else if d < 0.0 {
                    acc.exits += 1;
                }
                if d != 0.0 {
                    acc.trades += 1;
                }
                acc.gross += gross;
                acc.cost += cost;
                acc.turnover += d.abs();
                acc.net += gross - cost;
                prev = p;
            }
            out[c * blocks + block] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;

    extern "C" {
        fn rc_wfb_create(
            returns: *const f64,
            win: *const i32,
            chain_off: *const i64,
            chain_block0: *const i32,
            n: i64,
            chains: i32,
            blocks: i32,
            windows: i32,
            tile: i32,
        ) -> *mut std::ffi::c_void;
        fn rc_wfb_destroy(ctx: *mut std::ffi::c_void);
        fn rc_wfb_device_bytes(ctx: *mut std::ffi::c_void) -> i64;
        fn rc_wfb_eval(
            ctx: *mut std::ffi::c_void,
            scores: *const f64,
            thresholds: *const f64,
            candidates: i32,
            side: f64,
            threads: i32,
            out: *mut RawCell,
        ) -> i32;
    }

    pub struct Ctx {
        ctx: *mut std::ffi::c_void,
        pub tile: usize,
        blocks: usize,
        windows: usize,
        threads: usize,
    }

    impl Ctx {
        pub fn new(layout: &Layout, windows: usize, tile: usize, threads: usize) -> Option<Self> {
            let ctx = unsafe {
                rc_wfb_create(
                    layout.returns.as_ptr(),
                    layout.win.as_ptr(),
                    layout.chain_off.as_ptr(),
                    layout.chain_block0.as_ptr(),
                    layout.n as i64,
                    (layout.chain_off.len() - 1) as i32,
                    layout.block_window.len() as i32,
                    windows as i32,
                    tile as i32,
                )
            };
            if ctx.is_null() {
                return None;
            }
            Some(Ctx {
                ctx,
                tile,
                blocks: layout.block_window.len(),
                windows,
                threads,
            })
        }

        pub fn device_bytes(&self) -> u64 {
            unsafe { rc_wfb_device_bytes(self.ctx) }.max(0) as u64
        }

        pub fn eval(&mut self, scores: &[f64], thr: &[f64], cands: usize, side: f64) -> Result<Vec<RawCell>, i32> {
            if cands == 0 || cands > self.tile || thr.len() != cands * self.windows {
                return Err(1);
            }
            let mut out = vec![RawCell::default(); cands * self.blocks];
            let rc = unsafe {
                rc_wfb_eval(
                    self.ctx,
                    scores.as_ptr(),
                    thr.as_ptr(),
                    cands as i32,
                    side,
                    self.threads as i32,
                    out.as_mut_ptr(),
                )
            };
            if rc == 0 {
                Ok(out)
            } else {
                Err(rc)
            }
        }
    }

    impl Drop for Ctx {
        fn drop(&mut self) {
            unsafe { rc_wfb_destroy(self.ctx) };
        }
    }
}

/// True when a device batch can run: a CUDA build on a machine with a device.
pub fn device_available() -> bool {
    trading_gpu::device_available()
}

/// Score `spec.candidates` over `spec.windows`. `scores(candidate, window)`
/// returns the candidate's scores for the window's rows in input order. It is
/// called once per cell, and once more per window for a candidate that needs
/// the scalar fallback.
pub fn run(spec: &BatchSpec, scores: &mut dyn FnMut(usize, usize) -> Vec<f64>) -> Result<BatchResult, String> {
    validate(spec)?;
    let t_all = Instant::now();
    let n_windows = spec.windows.len();
    let t0 = Instant::now();
    let layout = build_layout(spec.windows, spec.seam)?;
    let mut stats = BatchStats {
        candidates: spec.candidates,
        windows: n_windows,
        cells: spec.candidates * n_windows,
        rows: layout.n as u64,
        row_evals: spec.candidates as u64 * layout.n as u64,
        layout_s: t0.elapsed().as_secs_f64(),
        ..BatchStats::default()
    };

    let row_bytes = layout.n * std::mem::size_of::<f64>();
    let tile = (spec.tile_bytes / row_bytes.max(1)).clamp(1, MAX_BATCH_TILE).min(spec.candidates);
    stats.tile_candidates = tile;
    stats.host_tile_bytes = (tile * row_bytes) as u64;

    #[cfg_attr(not(feature = "cuda"), allow(unused_assignments))]
    let mut fallback_reason: Option<String> = None;
    #[cfg(feature = "cuda")]
    let mut dev: Option<device::Ctx> = None;
    #[cfg(feature = "cuda")]
    {
        if spec.backend == Backend::Cpu {
            fallback_reason = Some("host backend requested".to_string());
        } else if !device_available() {
            fallback_reason = Some("no CUDA device".to_string());
        } else {
            dev = device::Ctx::new(&layout, n_windows, tile, spec.threads);
            match &dev {
                Some(d) => stats.device_bytes = d.device_bytes(),
                None => fallback_reason = Some("device context allocation refused".to_string()),
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        fallback_reason = Some("built without --features cuda".to_string());
    }

    let mut cells: Vec<Cell> = Vec::with_capacity(spec.candidates * n_windows);
    let mut ladder: Vec<LadderWindow> = (0..n_windows)
        .map(|w| ladder_window(spec, &layout, w))
        .collect();
    let params = PnlParams {
        per_side_bps: spec.cost_side * 10_000.0,
        ..PnlParams::default()
    };

    let mut tile_scores = vec![0.0f64; tile * layout.n];
    let mut tile_thr = vec![0.0f64; tile * n_windows];
    let mut base = 0usize;
    while base < spec.candidates {
        let count = tile.min(spec.candidates - base);
        let t_fill = Instant::now();
        let mut retained: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for ci in 0..count {
            let c = base + ci;
            let row = &mut tile_scores[ci * layout.n..(ci + 1) * layout.n];
            let mut flagged = false;
            let mut kept: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
            for w in 0..n_windows {
                let s = scores(c, w);
                if s.len() != spec.windows[w].time.len() {
                    return Err(format!(
                        "candidate {} window {}: {} scores for {} rows",
                        c,
                        w,
                        s.len(),
                        spec.windows[w].time.len()
                    ));
                }
                tile_thr[ci * n_windows + w] = spec.thresholds[c][w];
                for (i, &v) in s.iter().enumerate() {
                    let p = layout.pos[w][i];
                    if p != NONE {
                        if !v.is_finite() {
                            flagged = true;
                        }
                        row[p as usize] = v;
                    }
                }
                if let Some(seed) = spec.random_seed {
                    if !flagged {
                        let t = random_control(spec, &layout, w, &s, spec.thresholds[c][w], seed, &params)?;
                        ladder[w].random[c] = Some(t);
                    }
                }
                kept.push(s);
            }
            if flagged {
                for l in ladder.iter_mut() {
                    l.random[c] = None;
                }
                retained.insert(c, kept);
            }
        }
        stats.score_s += t_fill.elapsed().as_secs_f64();

        let t_dev = Instant::now();
        let scores_slice = &tile_scores[..count * layout.n];
        let thr_slice = &tile_thr[..count * n_windows];
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut raw: Option<(Vec<RawCell>, &'static str)> = None;
        #[cfg(feature = "cuda")]
        if let Some(d) = dev.as_mut() {
            match d.eval(scores_slice, thr_slice, count, spec.cost_side) {
                Ok(v) => raw = Some((v, GPU_PROXY)),
                Err(code) => {
                    fallback_reason = Some(format!("device batch failed (status {})", code));
                }
            }
        }
        let (raw, label) = match raw {
            Some(r) => {
                stats.device_tiles += 1;
                r
            }
            None => {
                stats.host_tiles += 1;
                (
                    eval_tile_cpu(&layout, scores_slice, thr_slice, count, n_windows, spec.cost_side),
                    CPU_BATCH,
                )
            }
        };
        let dt = t_dev.elapsed().as_secs_f64();
        if label == GPU_PROXY {
            stats.device_s += dt;
        }

        let blocks = layout.block_window.len();
        for ci in 0..count {
            let c = base + ci;
            if let Some(kept) = retained.remove(&c) {
                let t_ref = Instant::now();
                stats.fallback_candidates += 1;
                let mut previous: HashMap<u64, f64> = HashMap::new();
                for (w, s) in kept.iter().enumerate() {
                    if spec.seam == Seam::PerWindow {
                        previous.clear();
                    }
                    let win = &spec.windows[w];
                    let totals = reference_cell(
                        &win.time,
                        &win.group,
                        &win.target,
                        s,
                        spec.thresholds[c][w],
                        spec.cost_side,
                        &mut previous,
                    );
                    cells.push(Cell {
                        candidate: c,
                        window: w,
                        totals,
                        provenance: EXACT_CPU,
                    });
                }
                stats.reference_s += t_ref.elapsed().as_secs_f64();
                continue;
            }
            let mut per_window = vec![PnlTotals::default(); n_windows];
            for b in 0..blocks {
                let r = &raw[ci * blocks + b];
                if r.status != 0 {
                    return Err(format!("candidate {}: non-finite return in a laid-out row", c));
                }
                per_window[layout.block_window[b]].merge(&r.totals());
            }
            for (w, totals) in per_window.into_iter().enumerate() {
                cells.push(Cell {
                    candidate: c,
                    window: w,
                    totals,
                    provenance: label,
                });
            }
        }
        base += count;
    }

    stats.wall_s = t_all.elapsed().as_secs_f64();
    if stats.wall_s > 0.0 {
        stats.candidates_per_s = spec.candidates as f64 / stats.wall_s;
        stats.row_evals_per_s = stats.row_evals as f64 / stats.wall_s;
        stats.device_fraction = stats.device_s / stats.wall_s;
    }
    let backend = if stats.device_tiles > 0 { "cuda" } else { "cpu" };
    Ok(BatchResult {
        seam: spec.seam,
        backend: backend.to_string(),
        fallback_reason: if stats.host_tiles > 0 { fallback_reason } else { None },
        cells,
        ladder,
        stats,
    })
}

fn window_arrays(spec: &BatchSpec, layout: &Layout, w: usize) -> (Vec<f64>, Vec<u64>) {
    let win = &spec.windows[w];
    let idx = &layout.sorted[w];
    (
        idx.iter().map(|&i| win.target[i as usize]).collect(),
        idx.iter().map(|&i| win.group[i as usize]).collect(),
    )
}

fn ladder_window(spec: &BatchSpec, layout: &Layout, w: usize) -> LadderWindow {
    let (returns, groups) = window_arrays(spec, layout, w);
    let cash = trading_gpu::cash_totals(&returns, 0.0);
    let bh = trading_gpu::buy_and_hold_totals(&returns, &groups, spec.cost_side);
    let constant_long = baselines::constant_is_long(spec.windows[w].train_mean);
    LadderWindow {
        window: w,
        cash,
        buy_and_hold: bh,
        constant: if constant_long { bh } else { cash },
        constant_long,
        random: vec![None; spec.candidates],
    }
}

fn random_control(
    spec: &BatchSpec,
    layout: &Layout,
    w: usize,
    scores: &[f64],
    threshold: f64,
    seed: u64,
    params: &PnlParams,
) -> Result<PnlTotals, String> {
    let (returns, groups) = window_arrays(spec, layout, w);
    let positions: Vec<f64> = layout.sorted[w]
        .iter()
        .map(|&i| trading_gpu::long_position(scores[i as usize], threshold) as f64)
        .collect();
    let matched = baselines::matched_exposure_positions(&positions, &groups, seed)?;
    baselines::position_totals(&matched, &returns, &groups, params)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

impl BatchStats {
    /// Deterministic counters only: this block is inside the report hash.
    pub fn json(&self) -> String {
        format!(
            "{{\"candidates\":{},\"windows\":{},\"cells\":{},\"rows\":{},\"row_evals\":{},\
             \"tile_candidates\":{},\"device_bytes\":{},\"host_tile_bytes\":{},\
             \"device_tiles\":{},\"host_tiles\":{},\"fallback_candidates\":{}}}",
            self.candidates,
            self.windows,
            self.cells,
            self.rows,
            self.row_evals,
            self.tile_candidates,
            self.device_bytes,
            self.host_tile_bytes,
            self.device_tiles,
            self.host_tiles,
            self.fallback_candidates,
        )
    }

    /// Wall-clock seconds and rates: printed outside the report hash.
    pub fn timing_json(&self) -> String {
        format!(
            "{{\"wall_s\":{},\"device_s\":{},\"score_s\":{},\"layout_s\":{},\"reference_s\":{},\
             \"candidates_per_s\":{},\"row_evals_per_s\":{},\"device_fraction\":{}}}",
            jnum(self.wall_s),
            jnum(self.device_s),
            jnum(self.score_s),
            jnum(self.layout_s),
            jnum(self.reference_s),
            jnum(self.candidates_per_s),
            jnum(self.row_evals_per_s),
            jnum(self.device_fraction),
        )
    }
}

impl BatchResult {
    /// The cells and the ladder as JSON. `labels` names the windows.
    pub fn json(&self, labels: &[String]) -> String {
        let label = |w: usize| jstr(labels.get(w).map(String::as_str).unwrap_or(""));
        let cells: Vec<String> = self
            .cells
            .iter()
            .map(|c| {
                format!(
                    "{{\"candidate\":{},\"window\":{},\"value_label\":{},\"totals\":{}}}",
                    c.candidate + 1,
                    label(c.window),
                    jstr(c.provenance),
                    pnl_totals_json(&c.totals)
                )
            })
            .collect();
        let ladder: Vec<String> = self
            .ladder
            .iter()
            .map(|l| {
                let random: Vec<String> = l
                    .random
                    .iter()
                    .map(|r| r.as_ref().map_or("null".to_string(), pnl_totals_json))
                    .collect();
                format!(
                    "{{\"window\":{},\"value_label\":{},\"cash\":{},\"buy_and_hold\":{},\
                     \"constant\":{},\"constant_long\":{},\"matched_exposure_random_long\":[{}]}}",
                    label(l.window),
                    jstr(EXACT_CPU),
                    pnl_totals_json(&l.cash),
                    pnl_totals_json(&l.buy_and_hold),
                    pnl_totals_json(&l.constant),
                    l.constant_long,
                    random.join(",")
                )
            })
            .collect();
        format!(
            "{{\"seam\":{},\"backend\":{},\"fallback_reason\":{},\"cells\":[{}],\"ladder\":[{}],\"stats\":{}}}",
            jstr(self.seam.as_str()),
            jstr(&self.backend),
            self.fallback_reason.as_deref().map_or("null".to_string(), jstr),
            cells.join(","),
            ladder.join(","),
            self.stats.json()
        )
    }
}

/// The report block of `--batch-pnl`: both seams beside the exact finalists.
#[derive(Clone, Debug)]
pub struct BatchSummary {
    pub labels: Vec<String>,
    pub per_window: BatchResult,
    pub pooled: Option<BatchResult>,
    /// Per window: the selected candidate's exact scalar net, the batch net of
    /// the same cell and their difference.
    pub finalists: Vec<FinalistCheck>,
}

#[derive(Clone, Debug)]
pub struct FinalistCheck {
    pub window: String,
    pub candidate: usize,
    pub exact_net: f64,
    pub proxy_net: f64,
    pub delta: f64,
    pub provenance: &'static str,
}

impl BatchSummary {
    /// Timings of both seams, printed outside the report hash.
    pub fn timings_json(&self) -> String {
        format!(
            "{{\"per_window\":{},\"pooled\":{}}}",
            self.per_window.stats.timing_json(),
            self.pooled.as_ref().map_or("null".to_string(), |p| p.stats.timing_json())
        )
    }

    pub fn max_abs_net_delta(&self) -> f64 {
        self.finalists.iter().fold(0.0, |m, f| m.max(f.delta.abs()))
    }

    pub fn json(&self) -> String {
        let finalists: Vec<String> = self
            .finalists
            .iter()
            .map(|f| {
                format!(
                    "{{\"window\":{},\"candidate\":{},\"exact_net\":{},\"exact_label\":{},\
                     \"proxy_net\":{},\"proxy_label\":{},\"delta\":{}}}",
                    jstr(&f.window),
                    f.candidate + 1,
                    jnum(f.exact_net),
                    jstr(EXACT_CPU),
                    jnum(f.proxy_net),
                    jstr(f.provenance),
                    jnum(f.delta)
                )
            })
            .collect();
        format!(
            "{{\"reported_values\":{},\"batch_values\":{},\"max_abs_net_delta\":{},\
             \"finalists\":[{}],\"per_window\":{},\"pooled\":{}}}",
            jstr(EXACT_CPU),
            jstr(GPU_PROXY),
            jnum(self.max_abs_net_delta()),
            finalists.join(","),
            self.per_window.json(&self.labels),
            self.pooled
                .as_ref()
                .map_or("null".to_string(), |p| p.json(&self.labels))
        )
    }
}

/// Inputs the walk-forward report keeps for the batch pass, per window.
pub struct WindowScores {
    pub rows: BatchWindow,
    /// `scores[candidate]`, aligned with the window's rows.
    pub scores: Vec<Vec<f64>>,
    /// `thresholds[candidate]`.
    pub thresholds: Vec<f64>,
    /// The candidate the report selected for this window.
    pub selected: usize,
    /// The scalar evaluator's row-order net of the selected candidate.
    pub exact_net: f64,
}

/// Score every candidate on every window in one batch per seam and compare the
/// selected candidate with the scalar evaluator. `pooled` adds the pooled seam.
pub fn summarize(
    input: &[WindowScores],
    cost_side: f64,
    pooled: bool,
    random_seed: u64,
) -> Result<BatchSummary, String> {
    let first = input.first().ok_or("batch needs at least one window")?;
    let candidates = first.scores.len();
    let windows: Vec<BatchWindow> = input.iter().map(|w| w.rows.clone()).collect();
    let thresholds: Vec<Vec<f64>> = (0..candidates)
        .map(|c| input.iter().map(|w| w.thresholds[c]).collect())
        .collect();
    let run_seam = |seam: Seam| -> Result<BatchResult, String> {
        let mut spec = BatchSpec::new(&windows, candidates, &thresholds, cost_side, seam);
        spec.random_seed = Some(random_seed);
        let mut fetch = |c: usize, w: usize| input[w].scores[c].clone();
        run(&spec, &mut fetch)
    };
    let per_window = run_seam(Seam::PerWindow)?;
    let pooled = if pooled { Some(run_seam(Seam::Pooled)?) } else { None };
    let finalists = input
        .iter()
        .enumerate()
        .map(|(w, win)| {
            let cell = per_window.cell(win.selected, w);
            FinalistCheck {
                window: windows[w].label.clone(),
                candidate: win.selected,
                exact_net: win.exact_net,
                proxy_net: cell.totals.net,
                delta: cell.totals.net - win.exact_net,
                provenance: cell.provenance,
            }
        })
        .collect();
    Ok(BatchSummary {
        labels: windows.iter().map(|w| w.label.clone()).collect(),
        per_window,
        pooled,
        finalists,
    })
}
