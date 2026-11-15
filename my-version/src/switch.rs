//! #1185: `rcsr switch`, the exact gate-cut profile of `if(greater(g, c), A, B)`
//! for two fixed long/flat books `A` and `B` and a gate `g`, and the k-switch
//! hindsight bound of the pair.
//!
//! Both routines read a plain, unmasked [`FitnessBook`] (the canonical training
//! rows of a `--fitness pnl` run) and the two position vectors on those rows.
//! They never read a row the caller did not pass.
//!
//! Gate-cut profile ([`gate_sweep`]). The rows with a non-NaN gate value are
//! sorted by value; cut `t` puts the rows of rank `t` and above in state `A`
//! and every other row (a NaN gate included) in state `B`. Along the cut axis
//! the flip, entry and seam terms of one row pair are constant on three rank
//! intervals, so difference arrays and one pass give every cut.
//!
//! k-switch bound ([`switch_bound`]). A time-level state path holds `A` or `B`
//! at each distinct timestamp for every group at once. A dynamic program over
//! the timestamps gives the best net of any path with at most `k` switches. A
//! row pair of one group that skips a timestamp is charged with the cheaper of
//! the two earlier states, which keeps the value an upper bound.

use crate::data::TradingSlice;
use crate::devlog::{num, text};
use crate::regime_gate::GateSpec;
use crate::trading::{FitnessBook, FitnessTotals, TradingRows};
use crate::trading_gpu::order_key_f64;

/// Order key of a gate value: `None` for NaN; `-0.0` equals `+0.0`; `-inf` and
/// `+inf` order below and above every finite value, as `greater` compares them.
fn gate_key(v: f64) -> Option<u64> {
    if v.is_nan() {
        return None;
    }
    let bits = if v == 0.0 { 0 } else { v.to_bits() };
    Some(if bits >> 63 != 0 { !bits } else { bits | (1 << 63) })
}

/// The distinct timestamps and the timestamp index of every row.
fn stamp_index(time: &[f64]) -> Result<(usize, Vec<u32>), String> {
    let mut keys = Vec::with_capacity(time.len());
    for (i, &t) in time.iter().enumerate() {
        keys.push(order_key_f64(t).ok_or_else(|| format!("time {} is not finite", i))?);
    }
    let mut distinct = keys.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let index = keys.iter().map(|k| distinct.partition_point(|d| d < k) as u32).collect();
    Ok((distinct.len(), index))
}

/// Counts one on the cuts `lo..=hi` when `on`; `lo > hi` is empty.
fn add(d: &mut [i32], lo: i64, hi: i64, on: bool) {
    if on && lo <= hi {
        d[lo as usize] += 1;
        d[hi as usize + 1] -= 1;
    }
}

fn entry(out: &mut Vec<(u32, u32, u32)>, stamp: u32, lo: i64, hi: i64, on: bool) {
    if on && lo <= hi {
        out.push((stamp, lo as u32, hi as u32));
    }
}

fn check_rows(book: &FitnessBook, what: &str, lens: &[(&str, usize)]) -> Result<(), String> {
    let n = book.len();
    if let Some((name, len)) = lens.iter().find(|(_, len)| *len != n) {
        return Err(format!("{}: length mismatch: {} {} for {} rows", what, len, name, n));
    }
    if book.is_masked() {
        return Err(format!("{}: the book has a regime mask", what));
    }
    if !book.is_plain() {
        return Err(format!("{}: the book is not plain (it has a band or a minimum hold)", what));
    }
    if n >= u32::MAX as usize {
        return Err(format!("{}: {} rows do not fit a u32 row index", what, n));
    }
    Ok(())
}

/// The books of `if(greater(g, c), A, B)` at every cut of one gate on a plain
/// [`FitnessBook`]. Cut `t` in `0..=m`, `m` the rows with a non-NaN gate value,
/// holds `A` on the `m - t` rows of highest gate value and `B` on every other
/// row. `t` is valid when it does not split equal gate values (`-0.0` equals
/// `+0.0`); `t = 0` and `t = m` are always valid. Every count is a training-row
/// count.
#[derive(Clone, Debug)]
pub struct GateSweep {
    /// Rows with a non-NaN gate value by ascending value, ties by row index.
    order: Vec<u32>,
    rows: usize,
    /// Sum over blocks of `gross - side * (flips + exits)`; NaN where invalid.
    net: Vec<f64>,
    gross: Vec<f64>,
    /// [`FitnessBook::fitness_of`] of the per-block totals; `+inf` where invalid.
    fitness: Vec<f64>,
    flips: Vec<u32>,
    exits: Vec<u32>,
    long_rows: Vec<u32>,
    /// Distinct timestamps at which at least one row enters a long run.
    entry_times: Vec<u32>,
    /// Row pairs of one book run whose gate state differs.
    gate_flips: Vec<u32>,
    /// Of those, the pairs long on both sides (`J` of the seam identity).
    seam_pairs: Vec<u32>,
    /// State changes between consecutive distinct timestamps; `None` when the
    /// gate is not time-level (two rows of one timestamp differ in value).
    switches: Option<Vec<u32>>,
}

impl GateSweep {
    /// Rows of the book.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Rows with a non-NaN gate value (`m`); the cuts are `0..=m`.
    pub fn ordered_rows(&self) -> usize {
        self.order.len()
    }

    pub fn nan_rows(&self) -> usize {
        self.rows - self.order.len()
    }

    pub fn is_valid(&self, t: usize) -> bool {
        !self.net[t].is_nan()
    }

    pub fn net(&self, t: usize) -> f64 {
        self.net[t]
    }

    pub fn gross(&self, t: usize) -> f64 {
        self.gross[t]
    }

    pub fn fitness(&self, t: usize) -> f64 {
        self.fitness[t]
    }

    pub fn flips(&self, t: usize) -> u32 {
        self.flips[t]
    }

    pub fn exits(&self, t: usize) -> u32 {
        self.exits[t]
    }

    pub fn long_rows(&self, t: usize) -> u32 {
        self.long_rows[t]
    }

    pub fn entry_times(&self, t: usize) -> u32 {
        self.entry_times[t]
    }

    pub fn gate_flips(&self, t: usize) -> u32 {
        self.gate_flips[t]
    }

    pub fn seam_pairs(&self, t: usize) -> u32 {
        self.seam_pairs[t]
    }

    /// Time-level switches of cut `t`; `None` for a gate that is not time-level.
    pub fn switches(&self, t: usize) -> Option<u32> {
        self.switches.as_ref().map(|s| s[t])
    }

    /// Every timestamp holds one gate value (or only NaN), so the state of a
    /// valid cut is a time-level path and [`SwitchBound`] applies.
    pub fn is_time_level(&self) -> bool {
        self.switches.is_some()
    }

    /// Rows in state `A` at cut `t`.
    pub fn rows_a(&self, t: usize) -> usize {
        self.order.len() - t
    }

    /// The gate state of every row at cut `t` (`true` is `A`).
    pub fn state(&self, t: usize) -> Vec<bool> {
        let mut on = vec![false; self.rows];
        for &i in &self.order[t..] {
            on[i as usize] = true;
        }
        on
    }

    /// A cut value `c` that realises inner cut `t` under `greater(g, c)`: the
    /// midpoint of the two neighbouring gate values, or the lower one when the
    /// midpoint is not strictly below the upper one (`-f64::MAX` above `-inf`).
    /// `-inf` for `t = 0`, `+inf` for `t = m`.
    pub fn value(&self, gate: &[f64], t: usize) -> f64 {
        let m = self.order.len();
        if t == 0 {
            return f64::NEG_INFINITY;
        }
        if t >= m {
            return f64::INFINITY;
        }
        let lo = gate[self.order[t - 1] as usize];
        let hi = gate[self.order[t] as usize];
        if lo == f64::NEG_INFINITY {
            return if hi > -f64::MAX { -f64::MAX } else { lo };
        }
        let mid = lo + (hi - lo) * 0.5;
        if mid < hi {
            mid
        } else {
            lo
        }
    }

    /// The cut of value `c`: the rows with a non-NaN gate value `<= c`.
    pub fn index_of(&self, gate: &[f64], c: f64) -> usize {
        self.order.partition_point(|&i| gate[i as usize] <= c)
    }

    /// The smallest valid cut at or above `t` (`m` at the latest).
    pub fn snap(&self, t: usize) -> usize {
        let m = self.order.len();
        (t.min(m)..m).find(|&k| self.is_valid(k)).unwrap_or(m)
    }

    /// Valid inner cuts (`0 < t < m`).
    pub fn inner_cuts(&self) -> usize {
        (1..self.order.len()).filter(|&t| self.is_valid(t)).count()
    }

    /// Among the valid inner cuts of `cuts`, the highest net; ties go to the
    /// smaller cut. `None` when none is valid.
    pub fn best_of(&self, cuts: impl IntoIterator<Item = usize>) -> Option<usize> {
        let m = self.order.len();
        let mut best: Option<usize> = None;
        for t in cuts {
            if t == 0 || t >= m || !self.is_valid(t) {
                continue;
            }
            best = match best {
                Some(b) if !(self.net[t] > self.net[b] || (self.net[t] == self.net[b] && t < b)) => Some(b),
                _ => Some(t),
            };
        }
        best
    }

    /// The best of every inner cut.
    pub fn best(&self) -> Option<usize> {
        self.best_of(1..self.order.len())
    }

    /// `q - 1` uniform quantile cuts `floor(j m / q)`, `j = 1..q`, each moved up
    /// to the next valid cut; the always-`A` and always-`B` ends are dropped and
    /// repeats removed.
    pub fn grid(&self, q: usize) -> Vec<usize> {
        let m = self.order.len();
        let mut out: Vec<usize> = Vec::new();
        for j in 1..q.max(1) {
            let t = self.snap(j * m / q);
            if t > 0 && t < m && out.last() != Some(&t) {
                out.push(t);
            }
        }
        out
    }

    /// The widest run of valid inner cuts around `best` whose net stays within
    /// a tenth of the gain `net(best) - reference`; `(best, best)` when the gain
    /// is not positive. Returns the lowest and the highest cut of the run.
    pub fn plateau(&self, best: usize, reference: f64) -> (usize, usize) {
        let m = self.order.len();
        let top = self.net[best];
        let gain = top - reference;
        if !(gain > 0.0) {
            return (best, best);
        }
        let limit = top - 0.1 * gain;
        let mut lo = best;
        for k in (1..best).rev() {
            if !self.is_valid(k) {
                continue;
            }
            if !(self.net[k] >= limit) {
                break;
            }
            lo = k;
        }
        let mut hi = best;
        for k in best + 1..m {
            if !self.is_valid(k) {
                continue;
            }
            if !(self.net[k] >= limit) {
                break;
            }
            hi = k;
        }
        (lo, hi)
    }
}

/// #1185: every cut of the composite `if(greater(gate, c), A, B)` in one sort
/// and one sweep. `a` and `b` are the long rows of the two books, `time` the
/// row timestamps (it counts distinct entry timestamps and time-level
/// switches). A NaN gate value holds `B`. Errors: a length mismatch, a masked
/// or non-plain book, a non-finite time, fewer than 2 distinct non-NaN gate
/// values.
pub fn gate_sweep(book: &FitnessBook, a: &[bool], b: &[bool], gate: &[f64], time: &[f64]) -> Result<GateSweep, String> {
    check_rows(book, "gate sweep", &[("A positions", a.len()), ("B positions", b.len()), ("gate values", gate.len()), ("times", time.len())])?;
    let n = book.len();
    let (stamps, tix) = stamp_index(time).map_err(|e| format!("gate sweep: {}", e))?;
    let mut keyed: Vec<(u64, u32)> = (0..n).filter_map(|i| gate_key(gate[i]).map(|k| (k, i as u32))).collect();
    keyed.sort_unstable();
    let m = keyed.len();
    let distinct = if m == 0 { 0 } else { 1 + keyed.windows(2).filter(|w| w[0].0 != w[1].0).count() };
    if distinct < 2 {
        return Err(format!(
            "gate sweep: the gate has {} distinct non-NaN value(s) on the {} rows; a cut needs at least 2",
            distinct, n
        ));
    }
    let mut rank = vec![-1i64; n];
    for (r, &(_, i)) in keyed.iter().enumerate() {
        rank[i as usize] = r as i64;
    }
    let valid: Vec<bool> = (0..=m).map(|t| t == 0 || t == m || keyed[t - 1].0 != keyed[t].0).collect();
    let order: Vec<u32> = keyed.iter().map(|&(_, i)| i).collect();
    drop(keyed);

    let flags = book.flags();
    let excess = book.excess();
    let nb = book.blocks();
    let top = m as i64;
    let w = m + 2;
    // `*_d[t]`: change of a count where the cut reaches `t` (ascending).
    let mut flip_d = vec![0i32; nb * w];
    let mut exit_d = vec![0i32; nb * w];
    let mut gate_d = vec![0i32; w];
    let mut seam_d = vec![0i32; w];
    let mut entries: Vec<(u32, u32, u32)> = Vec::new();
    for i in 0..n {
        let blk = book.block_of(i);
        let ri = rank[i];
        let (ai, bi) = (a[i], b[i]);
        let d = &mut flip_d[blk * w..(blk + 1) * w];
        if flags[i] & TradingRows::FIRST != 0 {
            // A run starts flat: the row enters whenever it is long.
            add(d, 0, ri, ai);
            add(d, ri + 1, top, bi);
            entry(&mut entries, tix[i], 0, ri, ai);
            entry(&mut entries, tix[i], ri + 1, top, bi);
        } else {
            let rj = rank[i - 1];
            let (aj, bj) = (a[i - 1], b[i - 1]);
            let (lo, hi) = (ri.min(rj), ri.max(rj));
            // Inside `lo < t <= hi` the row of higher rank holds `A`, the other `B`.
            let (xi, xj) = if ri > rj { (ai, bj) } else { (bi, aj) };
            add(d, 0, lo, ai != aj);
            add(d, lo + 1, hi, xi != xj);
            add(d, hi + 1, top, bi != bj);
            add(&mut gate_d, lo + 1, hi, true);
            add(&mut seam_d, lo + 1, hi, xi && xj);
            entry(&mut entries, tix[i], 0, lo, ai && !aj);
            entry(&mut entries, tix[i], lo + 1, hi, xi && !xj);
            entry(&mut entries, tix[i], hi + 1, top, bi && !bj);
        }
        if flags[i] & TradingRows::LAST != 0 {
            let e = &mut exit_d[blk * w..(blk + 1) * w];
            add(e, 0, ri, ai);
            add(e, ri + 1, top, bi);
        }
    }
    // A timestamp counts once on the union of its rows' entry intervals.
    entries.sort_unstable();
    let mut entry_d = vec![0i32; w];
    let mut k = 0;
    while k < entries.len() {
        let (stamp, mut lo, mut hi) = entries[k];
        k += 1;
        while k < entries.len() && entries[k].0 == stamp {
            let (_, s, e) = entries[k];
            if s <= hi + 1 {
                hi = hi.max(e);
            } else {
                entry_d[lo as usize] += 1;
                entry_d[hi as usize + 1] -= 1;
                lo = s;
                hi = e;
            }
            k += 1;
        }
        entry_d[lo as usize] += 1;
        entry_d[hi as usize + 1] -= 1;
    }
    drop(entries);
    // Time-level: one gate key per timestamp. Any row of a timestamp stands for
    // it, since a valid cut never splits equal keys.
    let mut rep: Vec<Option<(Option<u64>, i64)>> = vec![None; stamps];
    let mut time_level = true;
    for i in 0..n {
        let u = tix[i] as usize;
        let key = gate_key(gate[i]);
        let current = rep[u];
        match current {
            None => rep[u] = Some((key, rank[i])),
            Some((k0, _)) if k0 != key => time_level = false,
            Some(_) => {}
        }
    }
    let switch_d: Option<Vec<i32>> = time_level.then(|| {
        let mut d = vec![0i32; w];
        for u in 1..stamps {
            let (r0, r1) = (rep[u - 1].map_or(-1, |r| r.1), rep[u].map_or(-1, |r| r.1));
            add(&mut d, r0.min(r1) + 1, r0.max(r1), true);
        }
        d
    });
    drop(rep);

    let empty = FitnessTotals { gross: 0.0, flips: 0, exits: 0, long_rows: 0, nonfinite: false };
    let mut totals = vec![empty; nb];
    // Cut 0: every row with a non-NaN gate value holds `A`.
    for i in 0..n {
        if if rank[i] >= 0 { a[i] } else { b[i] } {
            let t = &mut totals[book.block_of(i)];
            t.gross += excess[i];
            t.long_rows += 1;
        }
    }
    drop(rank);
    let mut net = vec![f64::NAN; m + 1];
    let mut gross = vec![f64::NAN; m + 1];
    let mut fitness = vec![f64::INFINITY; m + 1];
    let mut flips = vec![0u32; m + 1];
    let mut exits = vec![0u32; m + 1];
    let mut long_rows = vec![0u32; m + 1];
    let mut entry_times = vec![0u32; m + 1];
    let mut gate_flips = vec![0u32; m + 1];
    let mut seam_pairs = vec![0u32; m + 1];
    let mut switches = switch_d.as_ref().map(|_| vec![0u32; m + 1]);
    let mut run_f = vec![0i64; nb];
    let mut run_e = vec![0i64; nb];
    let (mut run_entry, mut run_gate, mut run_seam, mut run_switch) = (0i64, 0i64, 0i64, 0i64);
    for t in 0..=m {
        if t > 0 {
            // The row of rank `t - 1` leaves `A` for `B`.
            let i = order[t - 1] as usize;
            if a[i] != b[i] {
                let tb = &mut totals[book.block_of(i)];
                if a[i] {
                    tb.gross -= excess[i];
                    tb.long_rows -= 1;
                } else {
                    tb.gross += excess[i];
                    tb.long_rows += 1;
                }
            }
        }
        let (mut f, mut e, mut l) = (0u64, 0u64, 0u64);
        for blk in 0..nb {
            run_f[blk] += flip_d[blk * w + t] as i64;
            run_e[blk] += exit_d[blk * w + t] as i64;
            totals[blk].flips = run_f[blk] as u64;
            totals[blk].exits = run_e[blk] as u64;
            f += totals[blk].flips;
            e += totals[blk].exits;
            l += totals[blk].long_rows;
        }
        run_entry += entry_d[t] as i64;
        run_gate += gate_d[t] as i64;
        run_seam += seam_d[t] as i64;
        flips[t] = f as u32;
        exits[t] = e as u32;
        long_rows[t] = l as u32;
        entry_times[t] = run_entry as u32;
        gate_flips[t] = run_gate as u32;
        seam_pairs[t] = run_seam as u32;
        if let (Some(d), Some(s)) = (&switch_d, switches.as_mut()) {
            run_switch += d[t] as i64;
            s[t] = run_switch as u32;
        }
        if valid[t] {
            fitness[t] = book.fitness_of(&totals);
            net[t] = totals.iter().map(|x| book.net_of(x)).sum();
            gross[t] = totals.iter().map(|x| x.gross).sum();
        }
    }
    Ok(GateSweep { order, rows: n, net, gross, fitness, flips, exits, long_rows, entry_times, gate_flips, seam_pairs, switches })
}

/// #1185: the k-switch hindsight bound of two books: `V_k`, the best net of
/// any time-level path between `A` and `B` with at most `k` switches between
/// consecutive distinct timestamps, for `k = 0..=min(max_k, T - 1)`, and the
/// free bound. Exact when no row pair of one book run skips a timestamp
/// (`gap_pairs == 0`), an upper bound otherwise. It is a pruning bound, never
/// evidence of an edge.
#[derive(Clone, Debug)]
pub struct SwitchBound {
    /// Distinct timestamps `T`.
    pub timestamps: usize,
    /// Row pairs of one book run more than one distinct timestamp apart.
    pub gap_pairs: u64,
    values: Vec<f64>,
    /// The bound with no switch budget, equal to `V_(T-1)`.
    pub free: f64,
}

impl SwitchBound {
    /// `V_k`. A budget of `T - 1` or more is the free bound.
    pub fn value(&self, k: usize) -> f64 {
        let last = self.values.len() - 1;
        assert!(k <= last || self.values.len() == self.timestamps, "switch bound: V_{} was not computed (max {})", k, last);
        self.values[k.min(last)]
    }

    /// The largest budget computed.
    pub fn max_k(&self) -> usize {
        self.values.len() - 1
    }
}

/// #1185: [`SwitchBound`] on a plain [`FitnessBook`]: one pass over the rows for
/// the per-timestamp gross, entry, exit and flip terms of each state, then the
/// dynamic program `V_k(u, s) = W_u(s) + max(V_k(u-1, s) - side F_u(s, s),
/// V_(k-1)(u-1, s') - side F_u(s', s))` in `O(T k)`.
pub fn switch_bound(book: &FitnessBook, a: &[bool], b: &[bool], time: &[f64], max_k: usize) -> Result<SwitchBound, String> {
    check_rows(book, "switch bound", &[("A positions", a.len()), ("B positions", b.len()), ("times", time.len())])?;
    let n = book.len();
    let (stamps, tix) = stamp_index(time).map_err(|e| format!("switch bound: {}", e))?;
    let flags = book.flags();
    let excess = book.excess();
    let side = book.side();
    let x = |i: usize, s: usize| if s == 0 { a[i] } else { b[i] };
    // `wgt[u][s]`: what the rows of timestamp `u` earn in state `s`; `cross[u][p][s]`:
    // flips between a row at `u - 1` in state `p` and its successor at `u` in `s`.
    let mut wgt = vec![[0.0f64; 2]; stamps];
    let mut cross = vec![[[0u32; 2]; 2]; stamps];
    let mut gap_pairs = 0u64;
    for i in 0..n {
        let u = tix[i] as usize;
        let first = flags[i] & TradingRows::FIRST != 0;
        let last = flags[i] & TradingRows::LAST != 0;
        for s in 0..2 {
            if x(i, s) {
                wgt[u][s] += excess[i];
                if first {
                    wgt[u][s] -= side;
                }
                if last {
                    wgt[u][s] -= side;
                }
            }
        }
        if first {
            continue;
        }
        let j = i - 1;
        match u - tix[j] as usize {
            0 => {
                for s in 0..2 {
                    if x(i, s) != x(j, s) {
                        wgt[u][s] -= side;
                    }
                }
            }
            1 => {
                for p in 0..2 {
                    for s in 0..2 {
                        if x(i, s) != x(j, p) {
                            cross[u][p][s] += 1;
                        }
                    }
                }
            }
            _ => {
                gap_pairs += 1;
                for s in 0..2 {
                    if x(i, s) != x(j, 0) && x(i, s) != x(j, 1) {
                        wgt[u][s] -= side;
                    }
                }
            }
        }
    }
    let layers = max_k.min(stamps - 1);
    let mut v = vec![wgt[0]; layers + 1];
    let mut next = v.clone();
    let mut free = wgt[0];
    for u in 1..stamps {
        let c = |p: usize, s: usize| side * cross[u][p][s] as f64;
        for k in 0..=layers {
            for s in 0..2 {
                let o = 1 - s;
                let stay = v[k][s] - c(s, s);
                let switch = if k > 0 { v[k - 1][o] - c(o, s) } else { f64::NEG_INFINITY };
                next[k][s] = wgt[u][s] + stay.max(switch);
            }
        }
        std::mem::swap(&mut v, &mut next);
        free = [0, 1].map(|s| wgt[u][s] + (free[s] - c(s, s)).max(free[1 - s] - c(1 - s, s)));
    }
    Ok(SwitchBound { timestamps: stamps, gap_pairs, values: v.iter().map(|s| s[0].max(s[1])).collect(), free: free[0].max(free[1]) })
}

/// The per-block totals of the book that holds `a` on the rows with `on` and
/// `b` elsewhere, by the book's own reduction ([`FitnessBook::block_totals`]).
/// The threshold must be below `f64::MAX`.
pub fn composite_totals(book: &FitnessBook, a: &[bool], b: &[bool], on: &[bool]) -> Vec<FitnessTotals> {
    assert!(book.threshold() < f64::MAX, "composite totals: the threshold must be below f64::MAX");
    let pred: Vec<f64> = (0..book.len()).map(|i| if if on[i] { a[i] } else { b[i] } { f64::MAX } else { f64::MIN }).collect();
    book.block_totals(&pred)
}

/// Net of per-block totals.
pub fn net_of(book: &FitnessBook, totals: &[FitnessTotals]) -> f64 {
    totals.iter().map(|t| book.net_of(t)).sum()
}

/// The seam identity at one gate state: `composite = masked_a + masked_b + seam`
/// with `seam = 2 side J`, `J` the row pairs of one book run whose state
/// changes with a long row on both sides. `masked_a` holds `A` on the rows in
/// state `A` and is flat elsewhere, every run of those rows starting flat and
/// paying its exit; `masked_b` likewise for `B`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seam {
    pub net_a: f64,
    pub net_b: f64,
    pub masked_a: f64,
    pub masked_b: f64,
    pub pairs: u64,
    pub seam: f64,
    pub composite: f64,
}

pub fn seam(book: &FitnessBook, a: &[bool], b: &[bool], on: &[bool]) -> Seam {
    let n = book.len();
    let none = vec![false; n];
    let all = vec![true; n];
    let net = |a: &[bool], b: &[bool], on: &[bool]| net_of(book, &composite_totals(book, a, b, on));
    let flags = book.flags();
    let pos = |i: usize| if on[i] { a[i] } else { b[i] };
    let pairs = (1..n).filter(|&i| flags[i] & TradingRows::FIRST == 0 && on[i] != on[i - 1] && pos(i) && pos(i - 1)).count() as u64;
    Seam {
        net_a: net(a, a, &all),
        net_b: net(b, b, &all),
        masked_a: net(a, &none, on),
        masked_b: net(&none, b, on),
        pairs,
        seam: 2.0 * book.side() * pairs as f64,
        composite: net(a, b, on),
    }
}

// ---------------------------------------------------------------- the command

/// A branch of the composite: an expression of columns, or the flat book.
#[derive(Clone, Debug)]
pub enum Branch {
    Flat,
    Expr(GateSpec),
}

impl Branch {
    fn source(&self) -> &str {
        match self {
            Branch::Flat => "flat",
            Branch::Expr(g) => &g.source,
        }
    }

    fn columns(&self) -> &[String] {
        match self {
            Branch::Flat => &[],
            Branch::Expr(g) => &g.columns,
        }
    }
}

/// The flags of `rcsr switch` that the search parser does not read.
#[derive(Clone, Debug)]
pub struct Request {
    pub a: Branch,
    pub b: Branch,
    pub gates: Vec<GateSpec>,
    pub cuts: usize,
    pub switches: Vec<usize>,
    pub json: bool,
    pub out: Option<String>,
}

static REQUEST: std::sync::OnceLock<Request> = std::sync::OnceLock::new();

/// The `rcsr switch` request of this process, when the command is `switch`.
pub fn request() -> Option<&'static Request> {
    REQUEST.get()
}

impl Request {
    /// The columns each flag reads, for the `--pnl-mask` column checks.
    pub fn column_sets(&self) -> Vec<(&'static str, &[String])> {
        let mut out = vec![("--a", self.a.columns()), ("--b", self.b.columns())];
        out.extend(self.gates.iter().map(|g| ("--gate", g.columns.as_slice())));
        out
    }

    /// Every column read, in order of first use.
    pub fn columns(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, cols) in self.column_sets() {
            for c in cols {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
        }
        out
    }
}

pub fn usage_text() -> &'static str {
    "rcsr switch: the gate-cut profile of if(greater(g, c), A, B) and the k-switch bound of two books (#1185)

USAGE:
  rcsr switch <the flags of a --fitness pnl run> --a <expr|flat> --b <expr|flat>
              --gate <expr> [--gate <expr>...] [--cuts Q] [--switches k,k,...]
              [--json] [--out <file>]

  The data, target, split, objective, cost, threshold and block flags are the
  ones of a search run with --fitness pnl (it is implied); the command builds
  that run's training book and stops before any search. Every number it
  reports is a training-row value.

  --a <expr|flat>      branch A: an expression of columns in original units,
                       or flat; long where the value is finite and above
                       --threshold
  --b <expr|flat>      branch B, likewise; A and B must differ on the
                       training rows (exit 2 otherwise)
  --gate <expr>        a gate expression of columns, repeatable; any numeric
                       column may be read; a label-family column, the target,
                       its return column or a construction column exits 5
                       (the --pnl-mask column checks), a missing column exits
                       3, an aggregate or meta operator exits 2. A row holds A
                       when its gate value is above the cut c; a NaN gate
                       value holds B
  --cuts Q             grid of Q uniform gate quantiles (default 16, >= 2);
                       the best cut is reported on the grid and over all cuts
  --switches k,k,...   switch budgets of the bound (default 2,8,32); a budget
                       above T - 1 (T distinct timestamps) is clamped and
                       reported; the free bound is always reported
  --json               print the rcsr-switch/1 JSON report instead of text
  --out <file>         also write the JSON report to this file

  The k-switch bound V_k is the best net of any time-level switch between A
  and B with at most k switches; it bounds every time-level gate with at most
  k flips. It is a pruning bound, not evidence of an edge. A band, a minimum
  hold, --pnl-mask, --pnl-stat, --pnl-cvar-q and --gpu exit 2. See
  docs/walkforward.md.
"
}

fn fail(msg: &str) -> ! {
    eprintln!("error: rcsr switch: {}", msg);
    std::process::exit(2);
}

/// Takes the `rcsr switch` flags out of `args` (the arguments after `switch`),
/// keeps the request for [`request`], and returns the search flags with
/// `--fitness pnl` and `--cpu` added when absent. Usage errors exit 2.
pub fn claim(args: &[String]) -> Vec<String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage_text());
        std::process::exit(0);
    }
    let (mut a, mut b, mut gates, mut cuts, mut switches, mut json, mut out) = (None, None, Vec::new(), None, None, false, None);
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> String { args.get(i + 1).cloned().unwrap_or_else(|| fail(&format!("{} requires a value", flag))) };
        match flag {
            "--a" => a = Some(value()),
            "--b" => b = Some(value()),
            "--gate" => gates.push(value()),
            "--cuts" => cuts = Some(value()),
            "--switches" => switches = Some(value()),
            "--out" => out = Some(value()),
            "--json" => {
                json = true;
                i += 1;
                continue;
            }
            "--fitness" => {
                if value() != "pnl" {
                    fail("the command prices the --fitness pnl book: --fitness must be pnl");
                }
                rest.push(flag.to_string());
                rest.push("pnl".to_string());
            }
            "--gpu" => fail("--gpu: the command runs on the CPU and has no device arm"),
            "--pnl-mask" | "--pnl-mask-band" | "--pnl-mask-dwell" => {
                fail(&format!("{}: the sweep prices the unmasked book; the gate is the mask", flag))
            }
            "--pnl-stat" | "--pnl-cvar-q" => fail(&format!("{}: the sweep reports the mean-sd block fitness", flag)),
            _ => {
                rest.push(args[i].clone());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let branch = |flag: &str, src: Option<String>| -> Branch {
        let src = src.unwrap_or_else(|| fail(&format!("{} <expr|flat> is required", flag)));
        if src.trim() == "flat" {
            return Branch::Flat;
        }
        Branch::Expr(GateSpec::parse(&src, 0.0, 1).unwrap_or_else(|e| fail(&format!("{}: {}", flag, e))))
    };
    let a = branch("--a", a);
    let b = branch("--b", b);
    if gates.is_empty() {
        fail("--gate <expr> is required");
    }
    let gates: Vec<GateSpec> = gates.iter().map(|g| GateSpec::parse(g, 0.0, 1).unwrap_or_else(|e| fail(&format!("--gate: {}", e)))).collect();
    let cuts = match cuts {
        None => 16,
        Some(c) => match c.trim().parse::<usize>() {
            Ok(q) if q >= 2 => q,
            _ => fail(&format!("--cuts must be an integer >= 2, got '{}'", c)),
        },
    };
    let switches: Vec<usize> = match switches.as_deref() {
        None => vec![2, 8, 32],
        Some(s) => s
            .split(',')
            .map(|k| k.trim().parse::<usize>().unwrap_or_else(|_| fail(&format!("--switches must be a comma-separated list of integers >= 0, got '{}'", s))))
            .collect(),
    };
    if !rest.iter().any(|a| a == "--fitness") {
        rest.extend(["--fitness".to_string(), "pnl".to_string()]);
    }
    if !rest.iter().any(|a| a == "--cpu" || a == "--no-gpu") {
        rest.push("--cpu".to_string());
    }
    let _ = REQUEST.set(Request { a, b, gates, cuts, switches, json, out });
    rest
}

/// What the report header names besides the book.
pub struct Context<'a> {
    pub target: &'a str,
    pub per_side_bps: f64,
    pub segments: usize,
}

fn eval_branch(branch: &Branch, columns: &[String], values: &[Vec<f64>], threshold: f64, n: usize) -> (Vec<bool>, usize) {
    match branch {
        Branch::Flat => (vec![false; n], 0),
        Branch::Expr(g) => {
            let raw = eval_expr(g, columns, values, n);
            let nonfinite = raw.iter().filter(|v| !v.is_finite()).count();
            (raw.iter().map(|v| v.is_finite() && *v > threshold).collect(), nonfinite)
        }
    }
}

fn eval_expr(g: &GateSpec, columns: &[String], values: &[Vec<f64>], n: usize) -> Vec<f64> {
    let idx: Vec<usize> = g.columns.iter().map(|c| columns.iter().position(|k| k == c).expect("every column of the request is carried")).collect();
    let mut row = vec![0.0f64; idx.len()];
    (0..n)
        .map(|i| {
            for (k, &j) in idx.iter().enumerate() {
                row[k] = values[j][i];
            }
            crate::expr::eval(&g.tree, &row)
        })
        .collect()
}

/// A constant that every position rule with this threshold holds flat.
fn flat_constant(threshold: f64) -> f64 {
    threshold - threshold.abs().max(1.0)
}

fn branch_text(branch: &Branch, threshold: f64) -> String {
    match branch {
        Branch::Flat => format!("{:?}", flat_constant(threshold)),
        Branch::Expr(g) => crate::expr::to_string_exact(&g.tree, &g.columns),
    }
}

/// `if(greater(<gate>, c), <A>, <B>)` with every constant exact; `None` when
/// `c` is not finite.
fn composite_expression(req: &Request, gate: &GateSpec, c: f64, threshold: f64) -> Option<String> {
    c.is_finite().then(|| {
        format!(
            "if(greater({}, {:?}), {}, {})",
            crate::expr::to_string_exact(&gate.tree, &gate.columns),
            c,
            branch_text(&req.a, threshold),
            branch_text(&req.b, threshold)
        )
    })
}

fn cut_json(sw: &GateSweep, gate: &[f64], t: usize) -> String {
    format!(
        "\"index\":{},\"value\":{},\"rows_a\":{},\"rows_b\":{},\"net\":{},\"gross\":{},\"fitness\":{},\"flips\":{},\"exits\":{},\"long_rows\":{},\"entry_times\":{},\"gate_flips\":{},\"switches\":{},\"seam_pairs\":{}",
        t,
        num(sw.value(gate, t)),
        sw.rows_a(t),
        sw.rows() - sw.rows_a(t),
        num(sw.net(t)),
        num(sw.gross(t)),
        num(sw.fitness(t)),
        sw.flips(t),
        sw.exits(t),
        sw.long_rows(t),
        sw.entry_times(t),
        sw.gate_flips(t),
        sw.switches(t).map_or_else(|| "null".to_string(), |s| s.to_string()),
        sw.seam_pairs(t)
    )
}

struct Budget {
    requested: usize,
    used: usize,
    value: f64,
}

fn share(gain: f64, bound_gain: f64) -> String {
    if bound_gain > 0.0 {
        num(gain / bound_gain)
    } else {
        "null".to_string()
    }
}

/// The report of one request on the training book; `values` holds the raw
/// values of [`Request::columns`] on the book rows.
pub fn report(req: &Request, book: &FitnessBook, slice: &TradingSlice, values: &[Vec<f64>], ctx: &Context) -> Result<(String, String), String> {
    let n = book.len();
    let columns = req.columns();
    if slice.rows.time.len() != n || values.len() != columns.len() || values.iter().any(|v| v.len() != n) {
        return Err(format!("the carried columns do not match the {} training rows", n));
    }
    if book.is_masked() {
        return Err("the training book has a regime mask".to_string());
    }
    if !book.is_plain() {
        return Err("the training book is not plain: a band (--band) or a minimum hold (--min-holding) has no exact sweep".to_string());
    }
    let threshold = book.threshold();
    if !(threshold < f64::MAX) {
        return Err("the threshold must be below the largest finite number".to_string());
    }
    let time = &slice.rows.time;
    let (a, a_nonfinite) = eval_branch(&req.a, &columns, values, threshold, n);
    let (b, b_nonfinite) = eval_branch(&req.b, &columns, values, threshold, n);
    if a == b {
        return Err("A and B hold the same book on the training rows: there is nothing to switch".to_string());
    }
    let all = vec![true; n];
    let ta = composite_totals(book, &a, &a, &all);
    let tb = composite_totals(book, &b, &b, &all);
    let (net_a, net_b) = (net_of(book, &ta), net_of(book, &tb));
    let reference = net_a.max(net_b);
    let max_k = req.switches.iter().copied().max().unwrap_or(0);
    let bound = switch_bound(book, &a, &b, time, max_k)?;
    let budgets: Vec<Budget> = req
        .switches
        .iter()
        .map(|&k| {
            let used = k.min(bound.timestamps - 1);
            Budget { requested: k, used, value: bound.value(used) }
        })
        .collect();
    let groups = book.groups();
    let side_json = |name: &str, branch: &Branch, t: &[FitnessTotals], nonfinite: usize| {
        let sum = |f: &dyn Fn(&FitnessTotals) -> u64| t.iter().map(f).sum::<u64>();
        format!(
            "\"{}\":{{\"expr\":{},\"flat\":{},\"net\":{},\"fitness\":{},\"flips\":{},\"exits\":{},\"long_rows\":{},\"nonfinite_rows\":{}}}",
            name,
            text(branch.source()),
            matches!(branch, Branch::Flat),
            num(net_of(book, t)),
            num(book.fitness_of(t)),
            sum(&|x| x.flips),
            sum(&|x| x.exits),
            sum(&|x| x.long_rows),
            nonfinite
        )
    };
    let budget_json: Vec<String> = budgets
        .iter()
        .map(|b| {
            format!(
                "{{\"requested\":{},\"k\":{},\"clamped\":{},\"value\":{},\"gain\":{}}}",
                b.requested,
                b.used,
                b.used < b.requested,
                num(b.value),
                num(b.value - reference)
            )
        })
        .collect();
    let mut text_out = format!(
        "rcsr switch: {} training rows, {} groups, {} timestamps, {} block(s), {} bp per side, threshold {} (every number is a training-row value)\n",
        n,
        groups,
        bound.timestamps,
        book.blocks(),
        ctx.per_side_bps,
        threshold
    );
    text_out.push_str(&format!("A {:<40} net {}\nB {:<40} net {}\n", req.a.source(), num(net_a), req.b.source(), num(net_b)));
    text_out.push_str(&format!(
        "bound (pruning bound, not evidence; {} gap pairs, {}):",
        bound.gap_pairs,
        if bound.gap_pairs == 0 { "exact" } else { "upper bound" }
    ));
    for b in &budgets {
        text_out.push_str(&format!(" k={} {}", b.used, num(b.value)));
    }
    text_out.push_str(&format!(" free {}\n", num(bound.free)));

    let mut gates_json: Vec<String> = Vec::new();
    for gate in &req.gates {
        let g = eval_expr(gate, &columns, values, n);
        let sw = gate_sweep(book, &a, &b, &g, time).map_err(|e| format!("--gate '{}': {}", gate.source, e))?;
        let best = sw.best().ok_or_else(|| format!("--gate '{}': no valid inner cut", gate.source))?;
        let grid = sw.grid(req.cuts);
        let grid_best = sw.best_of(grid.iter().copied());
        let on = sw.state(best);
        let s = seam(book, &a, &b, &on);
        let (lo, hi) = sw.plateau(best, reference);
        let gain = sw.net(best) - reference;
        let shares: Vec<String> = budgets
            .iter()
            .map(|b| format!("{{\"k\":{},\"share\":{}}}", b.used, share(gain, b.value - reference)))
            .chain(std::iter::once(format!("{{\"k\":\"free\",\"share\":{}}}", share(gain, bound.free - reference))))
            .collect();
        let expr_json = |t: usize| composite_expression(req, gate, sw.value(&g, t), threshold).map_or_else(|| "null".to_string(), |e| text(&e));
        let grid_best_json = match grid_best {
            Some(t) => format!("{{{},\"gain\":{},\"expression\":{}}}", cut_json(&sw, &g, t), num(sw.net(t) - reference), expr_json(t)),
            None => "null".to_string(),
        };
        let profile: Vec<String> = grid.iter().map(|&t| format!("{{{}}}", cut_json(&sw, &g, t))).collect();
        gates_json.push(format!(
            "{{\"expr\":{},\"rows\":{},\"nan_rows\":{},\"ordered_rows\":{},\"cuts\":{},\"time_level\":{},\"bound_applies\":{},\
             \"best\":{{{},\"gain\":{},\"share\":[{}],\"plateau\":{{\"lo\":{},\"hi\":{},\"rows\":{},\"value_lo\":{},\"value_hi\":{},\"width\":{}}},\
             \"seam\":{{\"net_a\":{},\"net_b\":{},\"masked_a\":{},\"masked_b\":{},\"pairs\":{},\"seam\":{},\"sum\":{},\"composite\":{}}},\"expression\":{}}},\
             \"grid\":{{\"q\":{},\"best\":{},\"profile\":[{}]}}}}",
            text(&gate.source),
            n,
            sw.nan_rows(),
            sw.ordered_rows(),
            sw.inner_cuts(),
            sw.is_time_level(),
            sw.is_time_level(),
            cut_json(&sw, &g, best),
            num(gain),
            shares.join(","),
            lo,
            hi,
            hi - lo,
            num(sw.value(&g, lo)),
            num(sw.value(&g, hi)),
            num(sw.value(&g, hi) - sw.value(&g, lo)),
            num(s.net_a),
            num(s.net_b),
            num(s.masked_a),
            num(s.masked_b),
            s.pairs,
            num(s.seam),
            num(s.masked_a + s.masked_b + s.seam),
            num(s.composite),
            expr_json(best),
            req.cuts,
            grid_best_json,
            profile.join(",")
        ));
        text_out.push_str(&format!(
            "gate {}{}: best cut {} (index {} of {}): net {} gain {} gate flips {} switches {} plateau {}..{}\n",
            gate.source,
            if sw.is_time_level() { " (time-level)" } else { " (varies inside a timestamp: the bound does not apply)" },
            num(sw.value(&g, best)),
            best,
            sw.ordered_rows(),
            num(sw.net(best)),
            num(gain),
            sw.gate_flips(best),
            sw.switches(best).map_or_else(|| "-".to_string(), |k| k.to_string()),
            num(sw.value(&g, lo)),
            num(sw.value(&g, hi))
        ));
        if let Some(t) = grid_best {
            text_out.push_str(&format!("  grid of {}: best cut {} net {}\n", req.cuts, num(sw.value(&g, t)), num(sw.net(t))));
        }
        text_out.push_str(&format!(
            "  seam: masked A {} + masked B {} + seam {} = {} (composite {})\n",
            num(s.masked_a),
            num(s.masked_b),
            num(s.seam),
            num(s.masked_a + s.masked_b + s.seam),
            num(s.composite)
        ));
        if let Some(e) = composite_expression(req, gate, sw.value(&g, best), threshold) {
            text_out.push_str(&format!("  expression: {}\n", e));
        }
    }
    let json = format!(
        "{{\"schema\":\"rcsr-switch/1\",\"rows\":\"train\",\"target\":{},\"train_rows\":{},\"groups\":{},\"timestamps\":{},\"time_first\":{},\"time_last\":{},\
         \"book\":{{\"per_side_bps\":{},\"threshold\":{},\"blocks\":{},\"lambda\":{},\"min_trades\":{},\"segments\":{}}},{},{},\
         \"bound\":{{\"label\":\"pruning_bound\",\"is_evidence\":false,\"applies_to\":\"time-level gates\",\"exact\":{},\"gap_pairs\":{},\"timestamps\":{},\"reference\":{},\"free\":{},\"free_gain\":{},\"k\":[{}]}},\
         \"gates\":[{}]}}",
        text(ctx.target),
        n,
        groups,
        bound.timestamps,
        num(time.iter().copied().fold(f64::INFINITY, f64::min)),
        num(time.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        num(ctx.per_side_bps),
        num(threshold),
        book.blocks(),
        num(book.lambda()),
        book.min_trades(),
        ctx.segments,
        side_json("a", &req.a, &ta, a_nonfinite),
        side_json("b", &req.b, &tb, b_nonfinite),
        bound.gap_pairs == 0,
        bound.gap_pairs,
        bound.timestamps,
        num(reference),
        num(bound.free),
        num(bound.free - reference),
        budget_json.join(","),
        gates_json.join(",")
    );
    Ok((json, text_out))
}

/// Prints the report (JSON with `--json`, else text), writes `--out`, exits 0;
/// a refusal exits 2.
pub fn run(req: &Request, book: &FitnessBook, slice: &TradingSlice, values: &[Vec<f64>], ctx: &Context) -> ! {
    let (json, text_out) = report(req, book, slice, values, ctx).unwrap_or_else(|e| fail(&e));
    if let Some(path) = &req.out {
        if let Err(e) = std::fs::write(path, format!("{}\n", json)) {
            eprintln!("error: rcsr switch: cannot write --out '{}': {}", path, e);
            std::process::exit(3);
        }
    }
    if req.json {
        println!("{}", json);
    } else {
        print!("{}", text_out);
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::{BookMask, TradingSpec};
    use std::collections::BTreeMap;

    struct Mix(u64);

    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn uni(&mut self) -> f64 {
            ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn gauss(&mut self) -> f64 {
            let (u, v) = (self.uni(), self.uni());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// `groups` groups over `steps` hourly stamps in canonical order; a row is
    /// left out with probability `skip`, so a group can skip a timestamp.
    /// `step[i]` is the stamp of row `i`.
    struct Panel {
        slice: TradingSlice,
        step: Vec<usize>,
        steps: usize,
    }

    fn panel(rng: &mut Mix, groups: usize, steps: usize, skip: f64) -> Panel {
        let z: Vec<f64> = (0..steps).map(|_| rng.gauss()).collect();
        let (mut time, mut group, mut ret, mut step) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for g in 0..groups {
            for (k, zk) in z.iter().enumerate() {
                if skip > 0.0 && rng.uni() < skip {
                    continue;
                }
                time.push(k as f64 * 3600.0);
                group.push(g as u64);
                ret.push(0.004 * rng.gauss() + 0.001 * zk);
                step.push(k);
            }
        }
        Panel { slice: TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret }, step, steps }
    }

    fn spec(cost: f64) -> TradingSpec {
        TradingSpec { cost_bps: cost, min_trades: 1, ..TradingSpec::default() }
    }

    fn book(p: &Panel, blocks: usize, segments: &[f64], cost: f64) -> FitnessBook {
        FitnessBook::with_segments(&p.slice, &spec(cost), 0.0, blocks, 1.0, segments, None).unwrap()
    }

    fn persistent(rng: &mut Mix, n: usize, stay: f64) -> Vec<bool> {
        let mut s = rng.uni() < 0.5;
        (0..n)
            .map(|_| {
                if rng.uni() > stay {
                    s = !s;
                }
                s
            })
            .collect()
    }

    fn compose(a: &[bool], b: &[bool], on: &[bool]) -> Vec<bool> {
        (0..a.len()).map(|i| if on[i] { a[i] } else { b[i] }).collect()
    }

    /// The book's own reduction of a position vector (threshold 0).
    fn direct(book: &FitnessBook, pos: &[bool]) -> Vec<FitnessTotals> {
        let pred: Vec<f64> = pos.iter().map(|&p| if p { 1.0 } else { -1.0 }).collect();
        book.block_totals(&pred)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    /// Distinct timestamps and the stamp of every row, by a sorted map (the
    /// panel times are not negative, so their bits sort like the values).
    fn stamps_of(time: &[f64]) -> (usize, Vec<usize>) {
        let mut by: BTreeMap<u64, usize> = time.iter().map(|t| (t.to_bits(), 0)).collect();
        for (k, v) in by.values_mut().enumerate() {
            *v = k;
        }
        (by.len(), time.iter().map(|t| by[&t.to_bits()]).collect())
    }

    /// Entry timestamps, gate flips and seam pairs of a composite, row by row.
    fn direct_counts(book: &FitnessBook, pos: &[bool], on: &[bool], stamp: &[usize], stamps: usize) -> (u32, u32, u32) {
        let flags = book.flags();
        let mut seen = vec![false; stamps];
        let (mut entries, mut gate_flips, mut seam_pairs) = (0u32, 0u32, 0u32);
        for i in 0..pos.len() {
            let first = flags[i] & TradingRows::FIRST != 0;
            let prev = !first && pos[i - 1];
            if pos[i] && !prev && !seen[stamp[i]] {
                seen[stamp[i]] = true;
                entries += 1;
            }
            if !first && on[i] != on[i - 1] {
                gate_flips += 1;
                if pos[i] && pos[i - 1] {
                    seam_pairs += 1;
                }
            }
        }
        (entries, gate_flips, seam_pairs)
    }

    /// State changes between consecutive distinct timestamps; `None` when two
    /// rows of one timestamp disagree.
    fn time_switches(stamp: &[usize], stamps: usize, on: &[bool]) -> Option<u32> {
        let mut state: Vec<Option<bool>> = vec![None; stamps];
        for (i, &k) in stamp.iter().enumerate() {
            match state[k] {
                Some(s) if s != on[i] => return None,
                _ => state[k] = Some(on[i]),
            }
        }
        let states: Vec<bool> = state.into_iter().map(|s| s.expect("every stamp has a row")).collect();
        Some(states.windows(2).filter(|w| w[0] != w[1]).count() as u32)
    }

    /// Gate kinds: 0 a time-level gaussian, 1 a time-level quarter grid with
    /// NaN, infinities and both zeros, 2 a per-row gaussian with NaN.
    fn gate_of(rng: &mut Mix, p: &Panel, kind: usize) -> Vec<f64> {
        let per_step: Vec<f64> = (0..p.steps)
            .map(|_| {
                let v = rng.gauss();
                if kind != 1 {
                    return v;
                }
                match rng.next() % 40 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    3 => -0.0,
                    _ => (v * 4.0).round() / 4.0,
                }
            })
            .collect();
        match kind {
            2 => (0..p.step.len()).map(|_| if rng.next() % 25 == 0 { f64::NAN } else { rng.gauss() }).collect(),
            _ => p.step.iter().map(|&k| per_step[k]).collect(),
        }
    }

    fn check_sweep(book: &FitnessBook, a: &[bool], b: &[bool], g: &[f64], time: &[f64], time_level: bool) -> usize {
        let sw = gate_sweep(book, a, b, g, time).unwrap();
        assert_eq!(sw.is_time_level(), time_level);
        let (stamps, stamp) = stamps_of(time);
        let m = sw.ordered_rows();
        assert_eq!(m, g.iter().filter(|v| !v.is_nan()).count());
        let mut checked = 0;
        for t in 0..=m {
            if !sw.is_valid(t) {
                assert_eq!(sw.fitness(t), f64::INFINITY, "cut {t} inside a tie");
                continue;
            }
            let on = sw.state(t);
            if t > 0 && t < m {
                let c = sw.value(g, t);
                let by_value: Vec<bool> = g.iter().map(|&v| v > c).collect();
                assert_eq!(on, by_value, "greater(g, {c}) realises cut {t}");
                assert_eq!(sw.index_of(g, c), t);
            }
            let pos = compose(a, b, &on);
            let totals = direct(book, &pos);
            let sum = |f: &dyn Fn(&FitnessTotals) -> u64| totals.iter().map(f).sum::<u64>();
            assert_eq!(sw.flips(t) as u64, sum(&|x| x.flips), "flips of cut {t}");
            assert_eq!(sw.exits(t) as u64, sum(&|x| x.exits), "exits of cut {t}");
            assert_eq!(sw.long_rows(t) as u64, sum(&|x| x.long_rows), "long rows of cut {t}");
            let net = net_of(book, &totals);
            let gross: f64 = totals.iter().map(|x| x.gross).sum();
            assert!(close(sw.net(t), net, 1e-12), "net of cut {t}: sweep {} book {net}", sw.net(t));
            assert!(close(sw.gross(t), gross, 1e-12), "gross of cut {t}: sweep {} book {gross}", sw.gross(t));
            let fit = book.fitness_of(&totals);
            assert!(fit == sw.fitness(t) || close(fit, sw.fitness(t), 1e-9), "fitness of cut {t}: sweep {} book {fit}", sw.fitness(t));
            let (entries, gate_flips, seam_pairs) = direct_counts(book, &pos, &on, &stamp, stamps);
            assert_eq!(sw.entry_times(t), entries, "entry timestamps of cut {t}");
            assert_eq!(sw.gate_flips(t), gate_flips, "gate flips of cut {t}");
            assert_eq!(sw.seam_pairs(t), seam_pairs, "seam pairs of cut {t}");
            if time_level {
                assert_eq!(sw.switches(t), time_switches(&stamp, stamps, &on), "switches of cut {t}");
            }
            checked += 1;
        }
        checked
    }

    #[test]
    fn gate_sweep_equals_the_composite_book() {
        let mut rng = Mix(1185);
        let mut cuts = 0;
        // #1178: trial 6 has 12 blocks, more than the flags byte holds.
        for trial in 0..7 {
            let p = panel(&mut rng, 7, 300, if trial % 2 == 0 { 0.03 } else { 0.0 });
            let segments: Vec<f64> = if trial % 3 == 2 { vec![0.0, 140.0 * 3600.0] } else { Vec::new() };
            let blocks = if trial == 6 { 12 } else { [1usize, 2, 4][trial % 3] };
            let bk = book(&p, blocks, &segments, 5.0 + 5.0 * (trial % 2) as f64);
            let n = bk.len();
            let a = persistent(&mut rng, n, 0.9);
            let b = persistent(&mut rng, n, 0.85);
            for kind in 0..3 {
                let g = gate_of(&mut rng, &p, kind);
                cuts += check_sweep(&bk, &a, &b, &g, &p.slice.rows.time, kind != 2);
            }
            let flat = vec![false; n];
            let g = gate_of(&mut rng, &p, 0);
            cuts += check_sweep(&bk, &a, &flat, &g, &p.slice.rows.time, true);
        }
        assert!(cuts > 15_000, "{cuts} cuts checked");
    }

    #[test]
    fn gate_sweep_refuses_bad_input() {
        let mut rng = Mix(7);
        let p = panel(&mut rng, 2, 60, 0.0);
        let bk = book(&p, 1, &[], 5.0);
        let n = bk.len();
        let time = &p.slice.rows.time;
        let (a, b) = (vec![true; n], vec![false; n]);
        let g: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert!(gate_sweep(&bk, &a[1..], &b, &g, time).unwrap_err().contains("length"));
        assert!(gate_sweep(&bk, &a, &b, &vec![2.0; n], time).unwrap_err().contains("1 distinct"));
        assert!(gate_sweep(&bk, &a, &b, &vec![f64::NAN; n], time).unwrap_err().contains("0 distinct"));
        let mut zeros = vec![0.0; n];
        zeros[3] = -0.0;
        assert!(gate_sweep(&bk, &a, &b, &zeros, time).unwrap_err().contains("1 distinct"));
        let held = FitnessBook::with_segments(&p.slice, &TradingSpec { min_holding: 2, ..spec(5.0) }, 0.0, 1, 1.0, &[], None).unwrap();
        assert!(gate_sweep(&held, &a, &b, &g, time).unwrap_err().contains("plain"));
        assert!(switch_bound(&held, &a, &b, time, 2).unwrap_err().contains("plain"));
        let on: Vec<bool> = (0..n).map(|i| i % 7 != 0).collect();
        let masked = FitnessBook::with_segments(&p.slice, &spec(5.0), 0.0, 1, 1.0, &[], Some(BookMask { on: &on, identity: "m" })).unwrap();
        assert!(gate_sweep(&masked, &a, &b, &g, time).unwrap_err().contains("mask"));
        assert!(switch_bound(&bk, &a, &b[1..], time, 2).unwrap_err().contains("length"));
    }

    #[test]
    fn the_seam_identity_holds() {
        let mut rng = Mix(268);
        let mut used = 0;
        for trial in 0..200usize {
            let steps = 40 + (trial * 7) % 50;
            let p = panel(&mut rng, 2 + trial % 4, steps, if trial % 3 == 0 { 0.05 } else { 0.0 });
            let segments: Vec<f64> = if trial % 2 == 1 { vec![0.0, (steps / 2) as f64 * 3600.0] } else { Vec::new() };
            let blocks = 1 + trial % 2;
            let cost = 5.0 + 5.0 * (trial % 3) as f64;
            let plain = book(&p, blocks, &segments, cost);
            let n = plain.len();
            let a = persistent(&mut rng, n, 0.9);
            let b = persistent(&mut rng, n, 0.9);
            let path = persistent(&mut rng, steps, 0.9);
            let on: Vec<bool> = p.step.iter().map(|&k| path[k]).collect();
            let off: Vec<bool> = on.iter().map(|o| !o).collect();
            let masked = |pos: &[bool], mask: &[bool]| -> Option<f64> {
                let m = FitnessBook::with_segments(&p.slice, &spec(cost), 0.0, blocks, 1.0, &segments, Some(BookMask { on: mask, identity: "gate" })).ok()?;
                Some(net_of(&m, &direct(&m, pos)))
            };
            let (Some(ma), Some(mb)) = (masked(&a, &on), masked(&b, &off)) else { continue };
            let composite = net_of(&plain, &direct(&plain, &compose(&a, &b, &on)));
            let (stamps, stamp) = stamps_of(&p.slice.rows.time);
            let (_, _, j) = direct_counts(&plain, &compose(&a, &b, &on), &on, &stamp, stamps);
            let identity = ma + mb + 2.0 * plain.side() * j as f64;
            assert!((identity - composite).abs() <= 1e-9, "trial {trial}: {ma} + {mb} + seam({j}) = {identity}, composite {composite}");
            let s = seam(&plain, &a, &b, &on);
            assert_eq!(s.pairs, j as u64);
            assert!(close(s.masked_a, ma, 1e-12) && close(s.masked_b, mb, 1e-12), "trial {trial}: masked nets {} {} vs books {ma} {mb}", s.masked_a, s.masked_b);
            assert!(close(s.composite, composite, 1e-12));
            assert!((s.masked_a + s.masked_b + s.seam - s.composite).abs() <= 1e-9);
            used += 1;
        }
        assert!(used >= 180, "{used} of 200 panels priced both masked books");
    }

    /// Every one of the `2^T` time-level paths of a small panel, priced by the
    /// book; `best[k]` is the best net with at most `k` switches.
    fn brute_force(bk: &FitnessBook, p: &Panel, a: &[bool], b: &[bool]) -> Vec<f64> {
        let steps = p.steps;
        let mut best = vec![f64::NEG_INFINITY; steps];
        for mask in 0u32..(1 << steps) {
            let on: Vec<bool> = p.step.iter().map(|&k| mask >> k & 1 == 1).collect();
            let switches = (1..steps).filter(|&k| (mask >> k & 1) != (mask >> (k - 1) & 1)).count();
            let v = net_of(bk, &direct(bk, &compose(a, b, &on)));
            for slot in best.iter_mut().skip(switches) {
                *slot = slot.max(v);
            }
        }
        best
    }

    #[test]
    fn the_switch_bound_is_the_path_optimum() {
        let mut rng = Mix(32);
        for trial in 0..24usize {
            let steps = 6 + trial % 7;
            let p = panel(&mut rng, 3, steps, 0.0);
            let segments: Vec<f64> = if trial % 3 == 1 { vec![0.0, (steps / 2) as f64 * 3600.0] } else { Vec::new() };
            let bk = book(&p, 1 + trial % 2, &segments, 5.0);
            let n = bk.len();
            let a = persistent(&mut rng, n, 0.7);
            let b = persistent(&mut rng, n, 0.7);
            let bound = switch_bound(&bk, &a, &b, &p.slice.rows.time, steps + 3).unwrap();
            assert_eq!((bound.timestamps, bound.gap_pairs, bound.max_k()), (steps, 0, steps - 1));
            let best = brute_force(&bk, &p, &a, &b);
            for k in 0..steps {
                assert!(close(bound.value(k), best[k], 1e-12), "trial {trial}, k {k}: bound {} paths {}", bound.value(k), best[k]);
                if k > 0 {
                    assert!(bound.value(k) >= bound.value(k - 1));
                }
            }
            let all = vec![true; n];
            let (na, nb) = (net_of(&bk, &direct(&bk, &a)), net_of(&bk, &direct(&bk, &b)));
            assert!(close(bound.value(0), na.max(nb), 1e-12), "V_0 {} vs max({na}, {nb})", bound.value(0));
            assert!(close(net_of(&bk, &composite_totals(&bk, &a, &b, &all)), na, 1e-15));
            assert!(close(bound.free, best[steps - 1], 1e-12));
            assert_eq!(bound.value(steps + 3), bound.value(steps - 1));
            assert!(close(bound.free, bound.value(steps - 1), 1e-15));
        }
    }

    #[test]
    fn a_skipped_timestamp_keeps_the_bound_above_every_path() {
        let mut rng = Mix(5);
        let mut p = panel(&mut rng, 3, 10, 0.0);
        // Group 1 skips stamps 3 and 4, group 2 skips stamp 6.
        let keep: Vec<bool> = (0..p.step.len()).map(|i| !matches!((p.slice.rows.group[i], p.step[i]), (1, 3) | (1, 4) | (2, 6))).collect();
        let pick = |v: &[f64]| -> Vec<f64> { v.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| *x).collect() };
        p.slice.rows.time = pick(&p.slice.rows.time);
        p.slice.ret = pick(&p.slice.ret);
        p.slice.rows.group = p.slice.rows.group.iter().zip(&keep).filter(|(_, k)| **k).map(|(g, _)| *g).collect();
        p.step = p.step.iter().zip(&keep).filter(|(_, k)| **k).map(|(s, _)| *s).collect();
        for trial in 0..8 {
            let bk = book(&p, 1, &[], 5.0 + 5.0 * (trial % 2) as f64);
            let n = bk.len();
            let a = persistent(&mut rng, n, 0.6);
            let b = persistent(&mut rng, n, 0.6);
            let bound = switch_bound(&bk, &a, &b, &p.slice.rows.time, 9).unwrap();
            assert_eq!(bound.gap_pairs, 2);
            let best = brute_force(&bk, &p, &a, &b);
            for (k, v) in best.iter().enumerate() {
                assert!(bound.value(k) >= v - 1e-12, "trial {trial}, k {k}: bound {} below a path {v}", bound.value(k));
            }
        }
    }

    #[test]
    fn the_bound_dominates_every_gate_cut() {
        let mut rng = Mix(1094);
        let (mut checked, mut gapped) = (0usize, 0usize);
        for trial in 0..6usize {
            let p = panel(&mut rng, 7, 300, if trial % 2 == 0 { 0.04 } else { 0.0 });
            let segments: Vec<f64> = if trial % 3 == 1 { vec![0.0, 150.0 * 3600.0] } else { Vec::new() };
            let bk = book(&p, [1usize, 2, 4][trial % 3], &segments, 10.0);
            let n = bk.len();
            let a = persistent(&mut rng, n, 0.95);
            let b = if trial == 5 { vec![false; n] } else { persistent(&mut rng, n, 0.95) };
            let time = &p.slice.rows.time;
            for kind in 0..2 {
                let g = gate_of(&mut rng, &p, kind);
                let sw = gate_sweep(&bk, &a, &b, &g, time).unwrap();
                assert!(sw.is_time_level());
                let kmax = (0..=sw.ordered_rows()).filter(|&t| sw.is_valid(t)).map(|t| sw.switches(t).unwrap() as usize).max().unwrap();
                let bound = switch_bound(&bk, &a, &b, time, kmax).unwrap();
                if trial % 2 == 0 {
                    assert!(bound.gap_pairs > 0, "trial {trial} skips timestamps");
                    gapped += 1;
                }
                for t in (0..=sw.ordered_rows()).filter(|&t| sw.is_valid(t)) {
                    let k = sw.switches(t).unwrap() as usize;
                    assert!(sw.net(t) <= bound.value(k) + 1e-12, "trial {trial}, cut {t}: net {} above V_{k} {}", sw.net(t), bound.value(k));
                    checked += 1;
                }
            }
        }
        assert!(checked > 1_900 && gapped == 6, "{checked} cuts, {gapped} gapped gates");
    }

    /// `A` (always long) pays only while the gate is above 0.5; `B` is flat.
    #[test]
    fn the_sweep_finds_a_planted_regime() {
        let mut rng = Mix(11);
        let (groups, steps) = (4usize, 2000usize);
        let zeta: Vec<f64> = (0..steps).map(|_| rng.gauss()).collect();
        let (mut time, mut group, mut ret, mut g) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for k_group in 0..groups {
            for (k, zk) in zeta.iter().enumerate() {
                time.push(k as f64 * 3600.0);
                group.push(k_group as u64);
                ret.push(0.002 * rng.gauss() + if *zk > 0.5 { 0.002 } else { -0.001 });
                g.push(*zk);
            }
        }
        let p = Panel { slice: TradingSlice { rows: TradingRows { time: time.clone(), group, funding: Vec::new() }, ret }, step: Vec::new(), steps };
        let bk = book(&p, 4, &[], 5.0);
        let n = bk.len();
        let (a, b) = (vec![true; n], vec![false; n]);
        let sw = gate_sweep(&bk, &a, &b, &g, &time).unwrap();
        let best = sw.best().unwrap();
        let c = sw.value(&g, best);
        assert!((c - 0.5).abs() < 0.2, "best cut {c}");
        let reference = net_of(&bk, &composite_totals(&bk, &a, &b, &vec![true; n])).max(0.0);
        assert!(sw.net(best) > reference);
        let (lo, hi) = sw.plateau(best, reference);
        assert!(lo <= best && best <= hi && sw.value(&g, lo) <= c && c <= sw.value(&g, hi));
        let grid = sw.grid(16);
        assert_eq!(grid.len(), 15);
        let gb = sw.best_of(grid.iter().copied()).unwrap();
        assert!(sw.net(gb) <= sw.net(best));
        let s = seam(&bk, &a, &b, &sw.state(best));
        assert!(close(s.composite, sw.net(best), 1e-12));
        assert_eq!(s.pairs, 0, "B is flat: no pair is long on both sides");
    }

    /// Seconds of the sweep and of the bound at k = 32 at the campaign training
    /// shape: 306 groups x 7,403 hours, 4 blocks. A measurement, run by hand.
    #[test]
    #[ignore]
    fn switch_cost_at_the_campaign_shape() {
        let mut rng = Mix(11);
        let p = panel(&mut rng, 306, 7403, 0.0);
        let bk = book(&p, 4, &[], 10.0);
        let n = bk.len();
        let a = persistent(&mut rng, n, 0.98);
        let b = persistent(&mut rng, n, 0.98);
        let g = gate_of(&mut rng, &p, 0);
        let time = &p.slice.rows.time;
        let (mut sweep_s, mut bound_s) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let sw = gate_sweep(&bk, &a, &b, &g, time).unwrap();
            sweep_s = sweep_s.min(start.elapsed().as_secs_f64());
            assert_eq!(sw.rows(), 306 * 7403);
            let start = std::time::Instant::now();
            let bound = switch_bound(&bk, &a, &b, time, 32).unwrap();
            bound_s = bound_s.min(start.elapsed().as_secs_f64());
            assert_eq!(bound.timestamps, 7403);
        }
        println!("switch: {} rows, 306 groups, 4 blocks: gate sweep best of 3 {:.3} s, bound k=32 best of 3 {:.3} s", n, sweep_s, bound_s);
    }
}
