//! #1051: exact row decisions of the device book check (`src/cuda/check_rows.h`).
//!
//! The device evaluates a check program per row on exact f64 columns with an
//! interval and a NaN state ([`crate::flat_bound::rows`] is its host mirror),
//! and returns per slot two bit planes (enter, stay), a flag word and the rows
//! it could not decide. [`CheckRows::settle`] patches those rows with the CPU
//! scalar evaluator, so a settled slot has the planes of the CPU prediction.

use crate::flat_bound::rows::{compile_check_program, SlotState, Thresholds, CHECK_SLOT_CAP, SLOT_UNDECIDED_CAP};
use crate::expr::Node;
use crate::gpu::{self, GpuEvaluator};
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

/// Slot flag: a row is undecided.
pub const FLAG_UNDECIDED: u32 = 1;
/// Slot flag: a row is certainly non-finite.
pub const FLAG_NONFINITE: u32 = 2;

extern "C" {
    fn rc_check_create(ctx: *mut c_void, nvars: i32, slot_cap: i32, word_cap: i32) -> *mut c_void;
    fn rc_check_destroy(h: *mut c_void);
    fn rc_check_set_col(h: *mut c_void, k: i32, col: *const f64) -> i32;
    fn rc_check_batch(
        h: *mut c_void,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        t_enter: f64,
        t_stay: f64,
        out_flags: *mut u32,
        out_und_count: *mut u32,
        out_und_rows: *mut u32,
        out_enter: *mut u32,
        out_stay: *mut u32,
    ) -> i32;
    fn rc_check_iv_for_tests(
        h: *mut c_void,
        prog: *const u32,
        off: *const u32,
        nprog: i32,
        t_enter: f64,
        t_stay: f64,
        lo: *mut f64,
        hi: *mut f64,
        nan: *mut i32,
        cls: *mut i32,
    ) -> i32;
    fn rc_check_libm_for_tests(ctx: *mut c_void, x: *const f64, n: i32, mode: i32, y: *mut f64) -> i32;
    fn rc_check_last_error() -> *const c_char;
}

fn check_error() -> String {
    let raw = unsafe { rc_check_last_error() };
    if raw.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

fn refused_allocation(what: &str) -> String {
    match (gpu::last_alloc_failure(), gpu::last_alloc_failure_tag()) {
        (Some(bytes), Some(tag)) => format!("device_memory: the device refused an allocation of {bytes} bytes (tag {tag}) for {what}"),
        _ => format!("device_memory: the device refused an allocation for {what} (no size or tag recorded)"),
    }
}

/// What the device returned for one slot, before the host patch.
#[derive(Clone, Debug)]
pub struct DeviceSlot {
    pub flags: u32,
    /// Undecided rows the device found (may exceed the list).
    pub undecided_count: u32,
    /// The first [`SLOT_UNDECIDED_CAP`] undecided rows, ascending within a chunk.
    pub undecided: Vec<u32>,
    pub enter: Vec<u32>,
    pub stay: Vec<u32>,
}

/// A slot after the host patch: the planes of the CPU prediction when settled.
#[derive(Clone, Debug)]
pub struct SettledSlot {
    pub state: SlotState,
    pub enter: Vec<u32>,
    pub stay: Vec<u32>,
    pub undecided: usize,
}

/// The exact f64 columns and the batch buffers of the check, on the context of
/// `ev`.
pub struct CheckRows<'a> {
    _ev: &'a mut GpuEvaluator,
    ctx: *mut c_void,
    h: *mut c_void,
    rows: usize,
    nvars: usize,
    words: usize,
}

impl Drop for CheckRows<'_> {
    fn drop(&mut self) {
        unsafe { rc_check_destroy(self.h) };
    }
}

impl GpuEvaluator {
    /// Uploads the exact f64 columns (`cols[var][row]`, one column at a time)
    /// for the book check. A refused allocation names its tag and bytes and
    /// leaves the evaluator unchanged. `word_cap` bounds the program words of
    /// one batch.
    pub fn check_rows<'a>(&'a mut self, cols: &[Vec<f64>], word_cap: usize) -> Result<CheckRows<'a>, String> {
        let rows = self.rows();
        if cols.is_empty() || cols.iter().any(|c| c.len() != rows) {
            return Err(format!("the check columns do not match the {rows} resident rows"));
        }
        let ctx = self.raw_context();
        #[cfg(test)]
        let _alloc_gate = crate::gpu::alloc_state_gate::shared();
        let h = unsafe { rc_check_create(ctx, cols.len() as i32, CHECK_SLOT_CAP as i32, word_cap.max(1) as i32) };
        if h.is_null() {
            let msg = check_error();
            return Err(if msg.contains("refused") { refused_allocation("the exact check columns") } else { msg });
        }
        let out = CheckRows { _ev: self, ctx, h, rows, nvars: cols.len(), words: rows.div_ceil(32) };
        for (k, c) in cols.iter().enumerate() {
            if unsafe { rc_check_set_col(out.h, k as i32, c.as_ptr()) } != 0 {
                return Err(format!("uploading check column {k} failed"));
            }
        }
        Ok(out)
    }
}

fn pack(programs: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut words = Vec::new();
    let mut off = vec![0u32];
    for p in programs {
        words.extend_from_slice(p);
        off.push(words.len() as u32);
    }
    (words, off)
}

impl CheckRows<'_> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Encodes `tree` for this check, or `None` when it has no encoding.
    pub fn compile(&self, tree: &Node) -> Option<Vec<u32>> {
        compile_check_program(tree, self.nvars)
    }

    /// Runs `programs` (each ends in HALT) and returns the raw device result
    /// per slot, including both planes.
    pub fn check_batch(&mut self, programs: &[Vec<u32>], t: Thresholds) -> Result<Vec<DeviceSlot>, String> {
        let nprog = programs.len();
        if nprog == 0 {
            return Ok(Vec::new());
        }
        let (words, off) = pack(programs);
        let mut flags = vec![0u32; nprog];
        let mut count = vec![0u32; nprog];
        let mut und = vec![0u32; nprog * SLOT_UNDECIDED_CAP];
        let mut enter = vec![0u32; nprog * self.words];
        let mut stay = vec![0u32; nprog * self.words];
        #[cfg(test)]
        let _alloc_gate = crate::gpu::alloc_state_gate::shared();
        let rc = unsafe {
            rc_check_batch(
                self.h,
                words.as_ptr(),
                off.as_ptr(),
                nprog as i32,
                t.enter,
                t.stay,
                flags.as_mut_ptr(),
                count.as_mut_ptr(),
                und.as_mut_ptr(),
                enter.as_mut_ptr(),
                stay.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(check_error());
        }
        Ok((0..nprog)
            .map(|s| DeviceSlot {
                flags: flags[s],
                undecided_count: count[s],
                undecided: und[s * SLOT_UNDECIDED_CAP..s * SLOT_UNDECIDED_CAP + (count[s] as usize).min(SLOT_UNDECIDED_CAP)].to_vec(),
                enter: enter[s * self.words..(s + 1) * self.words].to_vec(),
                stay: stay[s * self.words..(s + 1) * self.words].to_vec(),
            })
            .collect())
    }

    /// The host patch: a slot with a certainly non-finite row is `Inf`; one
    /// with more undecided rows than the list holds is `Unsettled`; otherwise
    /// each listed row is resolved with `eval_row` (the CPU scalar evaluator)
    /// and its bits are set. A patched row that is non-finite makes the slot
    /// `Inf`.
    pub fn settle(&self, d: DeviceSlot, t: Thresholds, mut eval_row: impl FnMut(usize) -> f64) -> SettledSlot {
        let DeviceSlot { flags, undecided_count, undecided, mut enter, mut stay } = d;
        let n = undecided_count as usize;
        if flags & FLAG_NONFINITE != 0 {
            return SettledSlot { state: SlotState::Inf, enter, stay, undecided: n };
        }
        if n > SLOT_UNDECIDED_CAP {
            return SettledSlot { state: SlotState::Unsettled, enter, stay, undecided: n };
        }
        for &row in &undecided {
            let row = row as usize;
            let p = eval_row(row);
            if !p.is_finite() {
                return SettledSlot { state: SlotState::Inf, enter, stay, undecided: n };
            }
            let bit = 1u32 << (row & 31);
            if p > t.enter {
                enter[row >> 5] |= bit;
            }
            if p > t.stay {
                stay[row >> 5] |= bit;
            }
        }
        SettledSlot { state: SlotState::Settled, enter, stay, undecided: n }
    }

    /// Test-only: the device interval, NaN state and class code of every row
    /// of every slot (`class`: 0 non-finite, 1 enter, 2 stay, 3 neither, 4
    /// undecided).
    #[allow(clippy::type_complexity)]
    pub fn iv_for_tests(&mut self, programs: &[Vec<u32>], t: Thresholds) -> Result<(Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>), String> {
        let nprog = programs.len();
        let (words, off) = pack(programs);
        let cnt = nprog * self.rows;
        let (mut lo, mut hi) = (vec![0f64; cnt], vec![0f64; cnt]);
        let (mut nan, mut cls) = (vec![0i32; cnt], vec![0i32; cnt]);
        #[cfg(test)]
        let _alloc_gate = crate::gpu::alloc_state_gate::shared();
        let rc = unsafe {
            rc_check_iv_for_tests(
                self.h,
                words.as_ptr(),
                off.as_ptr(),
                nprog as i32,
                t.enter,
                t.stay,
                lo.as_mut_ptr(),
                hi.as_mut_ptr(),
                nan.as_mut_ptr(),
                cls.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(check_error());
        }
        Ok((lo, hi, nan, cls))
    }

    /// Test-only: device `exp` (`mode` 0) or `tanh` (`mode` 1) of `x`.
    pub fn libm_for_tests(&mut self, x: &[f64], mode: i32) -> Result<Vec<f64>, String> {
        let mut y = vec![0f64; x.len()];
        let rc = unsafe { rc_check_libm_for_tests(self.ctx, x.as_ptr(), x.len() as i32, mode, y.as_mut_ptr()) };
        if rc != 0 {
            return Err(check_error());
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TradingSlice;
    use crate::expr::{constant, input, node, Op};
    use crate::flat_bound::row_tests::{tree14, Rng};
    use crate::flat_bound::rows::{classify, row_iv, RowClass, NAN_CERTAIN, NAN_NO};
    use crate::trading::{BandSpec, BandUnits, FitnessBook, TradingRows, TradingSpec};

    /// Under compute-sanitizer the tests use the small inputs of the plan: at most 8,192 rows and 16 slots.
    fn small() -> bool {
        std::env::var_os("RCSR_UNDER_SANITIZER").is_some()
    }

    fn evaluator(n: usize, nv: usize) -> Option<GpuEvaluator> {
        let cols = vec![0f32; n * nv];
        let y = vec![0f32; n];
        match GpuEvaluator::new(&cols, n, nv, &y) {
            Some(ev) => Some(ev),
            None => {
                crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
                None
            }
        }
    }

    fn columns(r: &mut Rng, vars: usize, rows: usize) -> Vec<Vec<f64>> {
        (0..vars)
            .map(|_| {
                let pick: Vec<f64> = (0..1 + r.pick(8)).map(|_| r.special()).collect();
                (0..rows).map(|_| pick[r.pick(pick.len())]).collect()
            })
            .collect()
    }

    fn row_x(cols: &[Vec<f64>], row: usize) -> Vec<f64> {
        cols.iter().map(|c| c[row]).collect()
    }

    fn ulp_distance(a: f64, b: f64) -> u64 {
        let ord = |v: f64| {
            let i = v.to_bits() as i64;
            if i < 0 { i64::MIN.wrapping_sub(i) } else { i }
        };
        (ord(a) as i128 - ord(b) as i128).unsigned_abs() as u64
    }

    fn has_libm(tree: &Node) -> bool {
        matches!(tree.op, Op::Exp | Op::Tanh | Op::Logistic | Op::Gauss) || tree.kids.iter().any(has_libm)
    }

    use crate::expr::Node;

    /// Device interval, NaN state and class against the CPU evaluator, and rule
    /// for rule against the host mirror where no libm call is involved.
    #[test]
    fn device_row_states_contain_the_cpu_prediction() {
        let (trees, rows, vars) = if small() { (16usize, 1_024usize, 3usize) } else { (2_000usize, 4_096usize, 3usize) };
        let Some(mut ev) = evaluator(rows, vars) else { return };
        let mut r = Rng(0x7a3c_91d5_e402_6b17);
        let ts = [Thresholds::new(0.0, 0.0).unwrap(), Thresholds::new(0.25, 0.5).unwrap()];
        let (mut checked, mut exact, mut violations, mut done) = (0u64, 0u64, 0u64, 0usize);
        while done < trees {
            let cols = columns(&mut r, vars, rows);
            let mut check = ev.check_rows(&cols, 64 * 512).expect("the columns upload");
            let mut batch: Vec<(Node, Vec<u32>)> = Vec::new();
            while batch.len() < (if small() { 16 } else { 64 }) && done + batch.len() < trees {
                let mut pool = Vec::new();
                let t = tree14(&mut r, 4, vars, &mut pool);
                if let Some(w) = check.compile(&t) {
                    batch.push((t, w));
                }
            }
            done += batch.len();
            let programs: Vec<Vec<u32>> = batch.iter().map(|b| b.1.clone()).collect();
            for &t in &ts {
                let (lo, hi, nan, cls) = check.iv_for_tests(&programs, t).expect("device pass");
                for (s, (tree, words)) in batch.iter().enumerate() {
                    let pure = !has_libm(tree);
                    for row in 0..rows {
                        let x = row_x(&cols, row);
                        let v = crate::expr::eval(tree, &x);
                        let o = s * rows + row;
                        let (l, h, n, c) = (lo[o], hi[o], nan[o], cls[o]);
                        checked += 1;
                        let mut bad = |why: &str| {
                            violations += 1;
                            if violations <= 8 {
                                eprintln!("violation {why}: {tree:?} row {x:?} value {v:e} device ({l:e}, {h:e}, {n}, {c})");
                            }
                        };
                        if v.is_nan() && n == NAN_NO as i32 {
                            bad("NaN value, nan == 0");
                        }
                        if n == NAN_CERTAIN as i32 && !v.is_nan() {
                            bad("nan == 2, value not NaN");
                        }
                        if !v.is_nan() && n != NAN_CERTAIN as i32 && !(l <= v && v <= h) {
                            bad("value outside the interval");
                        }
                        let want = crate::flat_bound::rows::class_of_value(v, t);
                        let code = match want {
                            RowClass::Nonfinite => 0,
                            RowClass::Enter => 1,
                            RowClass::Stay => 2,
                            RowClass::Neither => 3,
                            RowClass::Undecided => 4,
                        };
                        if c != 4 && c != code {
                            bad("class differs from the CPU class");
                        }
                        if pure {
                            let m = row_iv(words, &|k| cols[k][row]);
                            let same = (m.nan as i32 == n)
                                && (n == NAN_CERTAIN as i32 || (m.lo.to_bits() == l.to_bits() || m.lo == l) && (m.hi.to_bits() == h.to_bits() || m.hi == h));
                            let mc = match classify(m, t) {
                                RowClass::Nonfinite => 0,
                                RowClass::Enter => 1,
                                RowClass::Stay => 2,
                                RowClass::Neither => 3,
                                RowClass::Undecided => 4,
                            };
                            if !same || mc != c {
                                bad("device differs from the host mirror on a libm-free tree");
                            } else {
                                exact += 1;
                            }
                        }
                    }
                }
            }
        }
        println!("device row states: {checked} rows, {exact} libm-free rows equal to the mirror, {violations} violations");
        assert_eq!(violations, 0);
        assert!(small() || (checked > 10_000_000 && exact > 1_000_000), "{checked} {exact}");
    }

    #[test]
    fn device_libm_stays_within_the_check_widening() {
        let Some(mut ev) = evaluator(64, 1) else { return };
        let cols = vec![vec![0f64; 64]];
        let mut check = ev.check_rows(&cols, 64).expect("columns");
        let mut r = Rng(0x1234_5678_9abc_def1);
        for (mode, name, span) in [(0, "exp", 745.0f64), (1, "tanh", 20.0)] {
            let points = if small() { 20_000 } else { 1_000_000 };
            let mut x: Vec<f64> = (0..points).map(|_| (r.f() * 2.0 - 1.0) * span).collect();
            x.extend((0..1_000).map(|_| (r.f() * 2.0 - 1.0) * 1e-3));
            x.extend([0.0, -0.0, 1.0, -1.0, 700.0, 709.78, -745.0, 1e-300, 5e-324, 20.0, -20.0, f64::INFINITY, f64::NEG_INFINITY]);
            let y = check.libm_for_tests(&x, mode).expect("device libm");
            let mut worst = 0u64;
            for (xi, yi) in x.iter().zip(&y) {
                let c = if mode == 0 { xi.exp() } else { xi.tanh() };
                if c.is_nan() || yi.is_nan() {
                    assert!(c.is_nan() && yi.is_nan(), "{name}({xi:e})");
                    continue;
                }
                if c.is_infinite() || yi.is_infinite() {
                    assert_eq!(c.to_bits(), yi.to_bits(), "{name}({xi:e})");
                    continue;
                }
                worst = worst.max(ulp_distance(c, *yi));
            }
            println!("device {name}: largest distance to the CPU libm {worst} ulps over {} points", x.len());
            assert!(worst <= 8, "{name}: {worst} ulps");
        }
    }

    fn fixture(n: usize, groups: usize, seed: u64) -> (Vec<Vec<f64>>, TradingSlice) {
        let mut r = Rng(seed);
        let per = n / groups;
        let mut rows = TradingRows::default();
        let mut ret = Vec::with_capacity(n);
        let mut cols = vec![Vec::with_capacity(n); 4];
        for i in 0..n {
            let g = (i / per).min(groups - 1);
            rows.group.push(g as u64);
            rows.time.push(1.6e9 + 900.0 * (i - g * per) as f64);
            let signal = r.f() - 0.5;
            cols[0].push(signal);
            cols[1].push(r.f() * 4.0 - 2.0);
            cols[2].push(r.f() * 4.0 - 2.0);
            cols[3].push(if r.pick(50) == 0 { (r.f() - 0.2) * 900.0 } else { r.f() * 6.0 - 3.0 });
            ret.push(0.004 * signal + 0.003 * (r.f() - 0.5));
        }
        (cols, TradingSlice { rows, ret })
    }

    fn named() -> Vec<Node> {
        let e = |x: Node| node(Op::Exp, vec![x]);
        let b = |op, x, y| node(op, vec![x, y]);
        vec![
            input(0),
            b(Op::Mul, constant(0.75), input(0)),
            node(Op::Gauss, vec![b(Op::Mul, constant(1e6), input(0))]),
            node(Op::Logistic, vec![b(Op::Sub, input(3), constant(3423.03))]),
            b(Op::Sub, input(1), input(1)),
            e(e(input(0))),
            b(Op::Sub, node(Op::Logistic, vec![b(Op::Sub, e(input(3)), e(input(2)))]), constant(2.0)),
            b(Op::Sub, constant(0.0), b(Op::Mul, e(input(3)), e(input(3)))),
            node(Op::Tanh, vec![b(Op::Div, input(0), b(Op::Sub, input(1), input(1)))]),
            node(Op::If, vec![b(Op::Greater, input(0), constant(0.1)), input(0), constant(-1.0)]),
            b(Op::Sub, node(Op::Logistic, vec![b(Op::Mul, constant(8.0), input(0))]), constant(0.5)),
            constant(-1.0),
        ]
    }

    /// Every slot the device settles has the enter and stay planes of the CPU
    /// prediction, and the CPU book prices them to the CPU totals.
    #[test]
    fn device_check_settles_the_bits_of_the_cpu_prediction() {
        let n = if small() { 8_192usize } else { 90_000usize };
        let (cols, slice) = fixture(n, 8, 17);
        let Some(mut ev) = evaluator(n, 4) else { return };
        let mut r = Rng(0xfeed_f00d);
        let mut trees = named();
        while trees.len() < (if small() { 16 } else { 60 }) {
            let mut pool = Vec::new();
            let t = tree14(&mut r, 4, 4, &mut pool);
            if crate::flat_bound::rows::compile_check_program(&t, 4).is_some() {
                trees.push(t);
            }
        }
        let mut check = ev.check_rows(&cols, 64 * 512).expect("the columns upload");
        let programs: Vec<Vec<u32>> = trees.iter().map(|t| check.compile(t).expect("encodable")).collect();
        let (mut settled, mut inf, mut unsettled, mut priced) = (0, 0, 0, 0);
        for (band, hold) in [(0.0, 12usize), (1e-3, 8)] {
            let spec = TradingSpec {
                cost_bps: 2.0,
                min_trades: 1,
                band: BandSpec { units: BandUnits::Score, resolved: band, ..BandSpec::score(band) },
                min_holding: hold,
                ..Default::default()
            };
            let book = FitnessBook::with_blocks(&slice, &spec, 900.0, 4, 1.0).expect("book");
            let t = Thresholds::new(book.threshold(), band).unwrap();
            let device = check.check_batch(&programs, t).expect("device batch");
            for (tree, d) in trees.iter().zip(device) {
                let pred: Vec<f64> = (0..n).map(|row| crate::expr::eval(tree, &row_x(&cols, row))).collect();
                let s = check.settle(d, t, |row| pred[row]);
                match s.state {
                    SlotState::Unsettled => unsettled += 1,
                    SlotState::Inf => {
                        inf += 1;
                        assert!(pred.iter().any(|p| !p.is_finite()), "{tree:?}: Inf with an all-finite prediction");
                    }
                    SlotState::Settled => {
                        settled += 1;
                        assert!(pred.iter().all(|p| p.is_finite()), "{tree:?}: settled with a non-finite row");
                        for row in 0..n {
                            let (e, st) = ((s.enter[row >> 5] >> (row & 31)) & 1, (s.stay[row >> 5] >> (row & 31)) & 1);
                            assert_eq!((e == 1, st == 1), (pred[row] > t.enter, pred[row] > t.stay), "{tree:?} row {row}");
                        }
                        let synth: Vec<f64> = (0..n)
                            .map(|row| match ((s.enter[row >> 5] >> (row & 31)) & 1, (s.stay[row >> 5] >> (row & 31)) & 1) {
                                (1, _) => f64::MAX,
                                (_, 1) => t.enter,
                                _ => f64::MIN,
                            })
                            .collect();
                        let (want, shape, sig) = book.regret_shape_signed(&pred);
                        let (got, gshape, gsig) = book.regret_shape_signed(&synth);
                        assert_eq!((got.to_bits(), gshape, gsig), (want.to_bits(), shape, sig), "{tree:?} band {band}");
                        priced += 1;
                    }
                }
            }
        }
        println!("device check: settled {settled} inf {inf} unsettled {unsettled} priced {priced}");
        assert!(priced == settled && (small() || (settled > 20 && inf > 0)));
    }

    #[test]
    fn a_refused_check_upload_names_its_tag_and_changes_nothing() {
        let (n, v) = (4_096usize, 3usize);
        let cols32 = vec![0f32; n * v];
        let y = vec![0f32; n];
        let need = (n * v * 8) as u64;
        let Some(open) = GpuEvaluator::with_budget(&cols32, n, v, &y, None) else {
            crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
            return;
        };
        let live = open.alloc_report().live_bytes;
        drop(open);
        let mut ev = GpuEvaluator::with_budget(&cols32, n, v, &y, Some(live + need / 2)).expect("the resident rows fit the budget");
        assert_eq!(ev.alloc_report().live_bytes, live);
        let cols = vec![vec![0f64; n]; v];
        let why = ev.check_rows(&cols, 64).err().expect("the f64 columns do not fit the budget");
        assert!(why.contains("device_memory") && why.contains("check_f64") && why.contains(&need.to_string()), "{why}");
        assert_eq!(ev.alloc_report().live_bytes, live, "no allocation survives a refusal");
        assert_eq!(ev.alloc_report().refused_tag, "check_f64");
        drop(ev);
        let mut ev = GpuEvaluator::with_budget(&cols32, n, v, &y, Some(live + need + 100)).expect("rows fit");
        let why = ev.check_rows(&cols, 64).err().expect("the planes do not fit beside the columns");
        assert!(why.contains("device_memory") && why.contains("check_bits"), "{why}");
        assert_eq!(ev.alloc_report().live_bytes, live, "the columns are freed again");
    }

    /// Cost at the campaign row count: ms per slot of the device pass and of the whole batch.
    #[test]
    #[ignore = "cost measurement, run by hand through the test lock"]
    fn device_check_cost_at_the_campaign_shape() {
        let (n, vars, slots) = (1_996_201usize, 4usize, 64usize);
        let Some(mut ev) = evaluator(n, vars) else { return };
        let mut r = Rng(0xc057_5eed);
        let cols: Vec<Vec<f64>> = (0..vars).map(|_| (0..n).map(|_| r.f() * 4.0 - 2.0).collect()).collect();
        let mut check = ev.check_rows(&cols, slots * 512).expect("upload");
        let mut programs = Vec::new();
        while programs.len() < slots {
            let mut pool = Vec::new();
            let t = tree14(&mut r, 4, vars, &mut pool);
            if let Some(w) = check.compile(&t) {
                programs.push(w);
            }
        }
        let t = Thresholds::new(0.0, 0.0).unwrap();
        check.check_batch(&programs, t).expect("warm up");
        let start = std::time::Instant::now();
        let reps = 5;
        let mut undecided = 0u64;
        for _ in 0..reps {
            for d in check.check_batch(&programs, t).expect("batch") {
                undecided += d.undecided_count as u64;
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / (reps * slots) as f64;
        println!("device check cost: {n} rows, {slots} slots per batch, {ms:.3} ms per slot (device pass and plane copy), {undecided} undecided rows in total");
    }

    #[test]
    fn a_malformed_program_is_refused_before_the_launch() {
        let Some(mut ev) = evaluator(64, 2) else { return };
        let cols = vec![vec![0.5f64; 64]; 2];
        let mut check = ev.check_rows(&cols, 64).expect("columns");
        let t = Thresholds::new(0.0, 0.0).unwrap();
        let bad: [Vec<u32>; 5] = [
            vec![0, 0, 0, 255, 255],
            vec![1, 7, 255],
            vec![2, 255],
            vec![0, 0, 0, 0, 0, 0, 255],
            vec![0, 0, 0],
        ];
        for p in bad {
            let e = check.check_batch(&[p.clone()], t).err().unwrap_or_else(|| panic!("{p:?} must be refused"));
            assert!(e.contains("malformed"), "{e}");
        }
    }
}
