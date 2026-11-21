//! #1258: the device arm of `rcsr enumerate` (`src/cuda/enumerate.h`).
//!
//! The device evaluates a batch of trees in f32 on the resident training
//! columns, finds each tree's cut grid from exact order statistics of its own
//! predictions, and prices every cut of every tree on every book of a batch
//! in one kernel. Everything after the f32 prediction is the CPU's f64
//! arithmetic: the grid is [`enumerate::rank_grid`] or
//! [`enumerate::midpoint_grid`] on the widened predictions, a row is long when
//! `p > threshold` for the finite f64 `p = pred - c` (or `c - pred`), and the
//! totals of every (tree, cut, book) equal `FitnessBook::block_totals` of that
//! prediction bit for bit. The host turns the totals into fitness with
//! `FitnessBook::fitness_of`. Any refusal or device failure is an error; there
//! is no CPU fallback.

use crate::enumerate::{self, long_interval, Cut};
use crate::expr::Node;
use crate::gpu::{self, GpuEvaluator};
use crate::trading::{BookFamily, FitnessBook, FitnessTotals};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

/// Most trees in one device batch.
pub const MAX_BATCH: usize = 256;

/// `RcEnumTotals` in `src/cuda/enumerate.h`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct RawTotals {
    gross: f64,
    flips: u32,
    exits: u32,
    longs: u32,
    pad: u32,
}

extern "C" {
    fn rc_enum_create(
        ctx: *mut c_void,
        flags: *const u8,
        bids: *const u8,
        nb: i32,
        tree_cap: i32,
        word_cap: i32,
        book_cap: i32,
        q: i32,
        ranks: *const u32,
        nq: i32,
    ) -> *mut c_void;
    fn rc_enum_destroy(h: *mut c_void);
    fn rc_enum_set_book(h: *mut c_void, slot: i32, excess: *const f64) -> i32;
    fn rc_enum_eval(
        h: *mut c_void,
        prog: *const u32,
        off: *const u32,
        ntrees: i32,
        nonfinite: *mut i32,
        dcount: *mut i32,
        dvals: *mut u32,
        sel: *mut u32,
    ) -> i32;
    fn rc_enum_score(
        h: *mut c_void,
        ntrees: i32,
        lo: *const u32,
        hi: *const u32,
        ncuts: *const i32,
        nbooks: i32,
        out: *mut RawTotals,
    ) -> i32;
    fn rc_enum_pred(h: *mut c_void, tree: i32, out: *mut u32) -> i32;
    fn rc_enum_kernel_ms(h: *mut c_void, eval_ms: *mut f64, score_ms: *mut f64) -> i32;
    fn rc_enum_last_error() -> *const c_char;
    fn rc_enum_local_bytes(nb: i32) -> usize;
    fn rc_launch_force_for_tests(prefix: *const c_char);
    fn rc_enum_red_atomic_for_tests(on: i32);
}

/// Limits of the memory plan; the defaults are the production plan.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceOptions {
    /// Most trees per batch.
    pub max_batch: usize,
    /// Most books resident at once; more books cycle through the slots.
    pub max_book_batch: usize,
    /// Device bytes left free for the desktop; `None` is the larger of 1 GiB
    /// and a tenth of the device.
    pub reserve: Option<u64>,
    /// A byte budget on the live allocations of the context (test hook).
    pub budget: Option<u64>,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        DeviceOptions { max_batch: MAX_BATCH, max_book_batch: usize::MAX, reserve: None, budget: None }
    }
}

/// The `device` member of the report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceReport {
    pub name: String,
    /// Trees per batch.
    pub batch: usize,
    /// Books resident at once.
    pub book_batch: usize,
    /// Book batches per tree batch.
    pub book_batches: usize,
    /// Peak live device bytes of the context.
    pub bytes: u64,
    pub reserve: u64,
    /// Trees with 2 to `Q - 1` distinct predictions, cut on the host grid.
    pub host_grids: u64,
    /// Device milliseconds of the evaluation and grid kernels.
    pub eval_ms: f64,
    /// Device milliseconds of the scoring kernels.
    pub score_ms: f64,
}

impl DeviceReport {
    pub fn json(&self) -> String {
        format!(
            "{{\"name\":{},\"batch\":{},\"book_batch\":{},\"book_batches\":{},\"bytes\":{},\"reserve\":{},\"host_grids\":{},\"kernel_ms\":{},\"eval_ms\":{},\"score_ms\":{}}}",
            crate::devlog::text(&self.name),
            self.batch,
            self.book_batch,
            self.book_batches,
            self.bytes,
            self.reserve,
            self.host_grids,
            crate::devlog::num(self.eval_ms + self.score_ms),
            crate::devlog::num(self.eval_ms),
            crate::devlog::num(self.score_ms)
        )
    }
}

/// What the device found for one tree of a batch.
#[derive(Clone, Debug, PartialEq)]
pub enum TreeGrid {
    /// Some f32 prediction is not finite.
    NonFinite,
    /// The cuts of the tree (none for a constant tree); `host` when it has
    /// fewer than `Q` distinct predictions and took the midpoint grid.
    Cuts { cuts: Vec<Cut>, host: bool },
}

fn enum_error() -> String {
    let raw = unsafe { rc_enum_last_error() };
    if raw.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

fn refused_allocation(what: &str) -> String {
    match (gpu::last_alloc_failure(), gpu::last_alloc_failure_tag()) {
        (Some(bytes), Some(tag)) => format!("device_memory: the device refused an allocation of {} bytes (tag {}) for {}", bytes, tag, what),
        _ => format!("device_memory: the device refused an allocation for {} (no size or tag recorded)", what),
    }
}

/// Test hook: refuse this thread's launches of kernels whose name starts with
/// `prefix` (empty clears it), as `RCSR_FORCE_LAUNCH_REFUSAL` does for a process.
pub fn force_launch_refusal_for_tests(prefix: &str) {
    let c = CString::new(prefix).expect("a kernel prefix holds no NUL");
    unsafe { rc_launch_force_for_tests(c.as_ptr()) };
}

/// The device state of one enumeration: the columns, the seam flags and
/// block ids, the resident books and the batch buffers.
pub struct Enumerator {
    ev: GpuEvaluator,
    h: *mut c_void,
    rows: usize,
    q: usize,
    blocks: usize,
    threshold: f64,
    ranks: Vec<usize>,
    words: usize,
    batch: usize,
    book_batch: usize,
    reserve: u64,
    last: usize,
}

impl Drop for Enumerator {
    fn drop(&mut self) {
        unsafe { rc_enum_destroy(self.h) };
    }
}

impl Enumerator {
    /// Uploads the columns (`cols[var][row]`, as f32) and the rows of `books`
    /// and plans the batches for `trees` trees of at most `max_size` nodes.
    /// Refuses books that are not one [`BookFamily`], a threshold that is not
    /// finite, and a plan with no room for one tree. With a finite threshold
    /// every cut constant is finite, and an f32 prediction minus a finite f64
    /// rounds to a finite value, so no device block is non-finite.
    pub fn new(cols: &[Vec<f64>], rows: usize, books: &[&FitnessBook], q: usize, max_size: usize, trees: usize, opts: &DeviceOptions) -> Result<Enumerator, String> {
        let family = BookFamily::new(books.to_vec())?;
        let first = family.books()[0];
        let threshold = first.threshold();
        if !threshold.is_finite() {
            return Err(format!("the device arm needs a finite --threshold, got {}", threshold));
        }
        if rows == 0 || first.len() != rows || cols.iter().any(|c| c.len() != rows) {
            return Err(format!("{} training rows for a book of {} rows", rows, first.len()));
        }
        if !(2..=enumerate::MAX_CUTS).contains(&q) || q % 2 != 0 {
            return Err(format!("--cuts must be even and 2 to {}, got {}", enumerate::MAX_CUTS, q));
        }
        let nvars = cols.len();
        let mut cols32 = Vec::with_capacity(nvars * rows);
        for c in cols {
            cols32.extend(c.iter().map(|&v| v as f32));
        }
        let y = vec![0.0f32; rows];
        let Some(mut ev) = GpuEvaluator::with_budget(&cols32, rows, nvars, &y, opts.budget) else {
            return Err(refused_allocation("the training columns"));
        };
        drop(cols32);
        let info = gpu::device_info().ok_or("the device memory query failed")?;
        let reserve = opts.reserve.unwrap_or_else(|| (1u64 << 30).max(info.total / 10));
        let blocks = first.blocks();
        let local = unsafe { rc_enum_local_bytes(blocks as i32) } as u64;
        let ranks = if rows >= 2 { enumerate::grid_ranks(rows, q) } else { Vec::new() };
        let nq = ranks.len() as u64;
        let words = 2 * max_size + 1;
        let (n, qq, nb) = (rows as u64, q as u64, blocks as u64);
        let per_book = 8 * n;
        let per_tree = |bb: u64| 4 * n + 4 * words as u64 + 4 + 24 + 4 * qq + 16 * nq + 8 * qq + 4 * 2048 + 1024 * nq + 24 * bb * qq * nb;
        let avail = info.free.saturating_sub(reserve).saturating_sub(local).saturating_sub(2 * n);
        let bb = (books.len() as u64).min(opts.max_book_batch.max(1) as u64).min((avail / 2 / per_book).max(1));
        let fit = avail.saturating_sub(bb * per_book) / per_tree(bb);
        let batch = (opts.max_batch.min(MAX_BATCH).min(trees.max(1)) as u64).min(fit) as usize;
        if batch == 0 {
            return Err(format!(
                "device_memory: {} free device bytes after a reserve of {} and {} bytes of kernel local memory hold no batch: {} resident books need {} bytes and one tree {} bytes",
                info.free,
                reserve,
                local,
                bb,
                bb * per_book,
                per_tree(bb)
            ));
        }
        let ids: &[u8] = if blocks > 1 { first.block_ids() } else { &[] };
        if blocks > 1 && ids.len() != rows {
            return Err(format!("{} block ids for {} rows", ids.len(), rows));
        }
        let red = std::env::var("RCSR_ENUM_RED").unwrap_or_default();
        unsafe { rc_enum_red_atomic_for_tests((red == "atomic") as i32) };
        let mut ranks32: Vec<u32> = ranks.iter().map(|&r| r as u32).collect();
        if red == "rank" && !ranks32.is_empty() && (ranks32[0] as usize) + 1 < rows {
            ranks32[0] += 1;
        }
        let h = unsafe {
            rc_enum_create(
                ev.raw_context(),
                first.flags().as_ptr(),
                if ids.is_empty() { std::ptr::null() } else { ids.as_ptr() },
                blocks as i32,
                batch as i32,
                (batch * words) as i32,
                bb as i32,
                q as i32,
                if ranks32.is_empty() { std::ptr::null() } else { ranks32.as_ptr() },
                ranks32.len() as i32,
            )
        };
        if h.is_null() {
            let why = enum_error();
            return Err(if why == "allocation refused" { refused_allocation("the enumerate batch buffers") } else { format!("the device refused the enumerate buffers: {}", why) });
        }
        Ok(Enumerator { ev, h, rows, q, blocks, threshold, ranks, words, batch, book_batch: bb as usize, reserve, last: 0 })
    }

    /// Trees per batch.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Books resident at once.
    pub fn book_batch(&self) -> usize {
        self.book_batch
    }

    pub fn reserve(&self) -> u64 {
        self.reserve
    }

    /// Device milliseconds of the evaluation and grid kernels and of the scoring kernels so far.
    pub fn kernel_ms(&self) -> (f64, f64) {
        let (mut e, mut s) = (0.0f64, 0.0f64);
        unsafe { rc_enum_kernel_ms(self.h, &mut e, &mut s) };
        (e, s)
    }

    /// Peak live device bytes of the context.
    pub fn peak_bytes(&self) -> u64 {
        self.ev.alloc_report().peak_live_bytes
    }

    /// Uploads `books` into the resident slots `0..books.len()`.
    pub fn set_books(&mut self, books: &[&FitnessBook]) -> Result<(), String> {
        if books.len() > self.book_batch {
            return Err(format!("{} books for {} resident slots", books.len(), self.book_batch));
        }
        for (slot, b) in books.iter().enumerate() {
            if b.len() != self.rows {
                return Err(format!("book {} has {} rows, the device {}", slot, b.len(), self.rows));
            }
            if unsafe { rc_enum_set_book(self.h, slot as i32, b.excess().as_ptr()) } != 0 {
                return Err(format!("the device refused a book upload: {}", enum_error()));
            }
        }
        Ok(())
    }

    /// Evaluates `trees` (at most [`Enumerator::batch`]) and returns the grid
    /// of each.
    pub fn eval(&mut self, trees: &[Node]) -> Result<Vec<TreeGrid>, String> {
        let t = trees.len();
        if t == 0 || t > self.batch {
            return Err(format!("{} trees for a batch of {}", t, self.batch));
        }
        let mut prog: Vec<u32> = Vec::with_capacity(t * self.words);
        let mut off: Vec<u32> = Vec::with_capacity(t + 1);
        off.push(0);
        for tree in trees {
            let c = gpu::compile_program(tree).map_err(|e| format!("the device interpreter cannot run a tree of the class: {}", e))?;
            prog.extend_from_slice(c.words());
            off.push(prog.len() as u32);
        }
        if prog.len() > t * self.words {
            return Err(format!("{} program words for a capacity of {}", prog.len(), t * self.words));
        }
        let (q, nq) = (self.q, self.ranks.len());
        let mut nonfinite = vec![0i32; t];
        let mut dcount = vec![0i32; t];
        let mut dvals = vec![0u32; t * q];
        let mut sel = vec![0u32; (t * nq).max(1)];
        let rc = unsafe {
            rc_enum_eval(self.h, prog.as_ptr(), off.as_ptr(), t as i32, nonfinite.as_mut_ptr(), dcount.as_mut_ptr(), dvals.as_mut_ptr(), sel.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!("the device evaluation failed: {}", enum_error()));
        }
        self.last = t;
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            if nonfinite[i] != 0 {
                out.push(TreeGrid::NonFinite);
                continue;
            }
            let d = dcount[i] as usize;
            if d < q {
                let distinct: Vec<f64> = dvals[i * q..i * q + d].iter().map(|&b| f32::from_bits(b) as f64).collect();
                out.push(TreeGrid::Cuts { cuts: enumerate::midpoint_grid(distinct), host: d >= 2 });
            } else {
                let found: Vec<(usize, f64)> = self.ranks.iter().zip(&sel[i * nq..(i + 1) * nq]).map(|(&r, &b)| (r, f32::from_bits(b) as f64)).collect();
                out.push(TreeGrid::Cuts { cuts: enumerate::rank_grid(self.rows, q, &found), host: false });
            }
        }
        Ok(out)
    }

    /// Prices every cut of the trees of the last [`Enumerator::eval`] on the
    /// resident books `0..nbooks`. Per tree, `totals[(b * cuts + k) * blocks + blk]`;
    /// empty for a tree without cuts.
    pub fn score(&mut self, grids: &[TreeGrid], nbooks: usize) -> Result<Vec<Vec<FitnessTotals>>, String> {
        let t = grids.len();
        if t != self.last || nbooks == 0 || nbooks > self.book_batch {
            return Err(format!("{} trees and {} books for the last batch of {} trees and {} slots", t, nbooks, self.last, self.book_batch));
        }
        let (q, nb) = (self.q, self.blocks);
        let mut lo = vec![1u32; t * q];
        let mut hi = vec![0u32; t * q];
        let mut ncuts = vec![0i32; t];
        for (i, g) in grids.iter().enumerate() {
            if let TreeGrid::Cuts { cuts, .. } = g {
                ncuts[i] = cuts.len() as i32;
                for (k, cut) in cuts.iter().enumerate() {
                    let (l, h) = if std::env::var("RCSR_ENUM_RED").as_deref() == Ok("f32") { red_f32_interval(cut.side, cut.constant(self.threshold), self.threshold) } else { long_interval(cut.side, cut.constant(self.threshold), self.threshold) };
                    lo[i * q + k] = l;
                    hi[i * q + k] = h;
                }
            }
        }
        let mut raw = vec![RawTotals::default(); t * nbooks * q * nb];
        let rc = unsafe { rc_enum_score(self.h, t as i32, lo.as_ptr(), hi.as_ptr(), ncuts.as_ptr(), nbooks as i32, raw.as_mut_ptr()) };
        if rc != 0 {
            return Err(format!("the device scoring failed: {}", enum_error()));
        }
        Ok((0..t).map(|i| totals_of_tree(&raw, i, ncuts[i] as usize, nbooks, q, nb)).collect())
    }

    /// Test hook: the f32 predictions of tree `i` of the last batch.
    pub fn predictions_for_tests(&mut self, i: usize) -> Result<Vec<f32>, String> {
        if i >= self.last {
            return Err(format!("tree {} of a batch of {}", i, self.last));
        }
        let mut bits = vec![0u32; self.rows];
        if unsafe { rc_enum_pred(self.h, i as i32, bits.as_mut_ptr()) } != 0 {
            return Err(format!("the prediction download failed: {}", enum_error()));
        }
        Ok(bits.into_iter().map(f32::from_bits).collect())
    }
}


/// WIP red: the key interval with `p` computed and compared in f32.
fn red_f32_interval(side: crate::enumerate::Side, c: f64, t: f64) -> (u32, u32) {
    let p = |k: u32| {
        let v = f32::from_bits(enumerate::f32_of_key(k));
        if side == crate::enumerate::Side::Above { v - c as f32 } else { c as f32 - v }
    };
    let first = |f: &dyn Fn(u32) -> bool| {
        let (mut a, mut z) = (enumerate::KEY_MIN as u64, enumerate::KEY_MAX as u64 + 1);
        while a < z {
            let m = a + (z - a) / 2;
            if f(m as u32) {
                z = m;
            } else {
                a = m + 1;
            }
        }
        a
    };
    let (lo, hi) = match side {
        crate::enumerate::Side::Above => (first(&|k| p(k) > t as f32), first(&|k| !(p(k) < f32::INFINITY)) - 1),
        crate::enumerate::Side::Below => (first(&|k| p(k) < f32::INFINITY), first(&|k| !(p(k) > t as f32)) - 1),
    };
    if lo > hi { (1, 0) } else { (lo as u32, hi as u32) }
}

/// Tree `i` of the device layout `[tree][book][q][block]` in the layout of
/// `BookFamily::cut_totals`, `[(book * cuts + k) * blocks + block]`.
fn totals_of_tree(raw: &[RawTotals], i: usize, cuts: usize, nbooks: usize, q: usize, nb: usize) -> Vec<FitnessTotals> {
    let mut out = Vec::with_capacity(nbooks * cuts * nb);
    for b in 0..nbooks {
        for k in 0..cuts {
            let at = ((i * nbooks + b) * q + k) * nb;
            out.extend(raw[at..at + nb].iter().map(|r| FitnessTotals {
                gross: r.gross,
                flips: r.flips as u64,
                exits: r.exits as u64,
                long_rows: r.longs as u64,
                nonfinite: false,
            }));
        }
    }
    out
}

#[cfg(test)]
mod host_tests {
    use super::*;

    #[test]
    fn device_layout_maps_to_the_cut_totals_layout() {
        let (t, nbooks, q, nb) = (3usize, 2usize, 4usize, 3usize);
        let raw: Vec<RawTotals> = (0..t * nbooks * q * nb).map(|j| RawTotals { gross: j as f64 + 0.5, flips: j as u32, exits: 2 * j as u32, longs: 3 * j as u32, pad: 0 }).collect();
        let got = totals_of_tree(&raw, 1, 3, nbooks, q, nb);
        assert_eq!(got.len(), nbooks * 3 * nb);
        for b in 0..nbooks {
            for k in 0..3 {
                for blk in 0..nb {
                    let j = ((nbooks + b) * q + k) * nb + blk;
                    let g = got[(b * 3 + k) * nb + blk];
                    assert_eq!((g.gross, g.flips, g.exits, g.long_rows, g.nonfinite), (j as f64 + 0.5, j as u64, 2 * j as u64, 3 * j as u64, false));
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "gpu_enumerate_tests.rs"]
mod device_tests;
