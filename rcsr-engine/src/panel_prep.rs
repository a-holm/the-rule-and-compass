//! #301: exact 15 m → 1 h panel preparation policy.
//!
//! This module is the CPU reference for the fused device pass in
//! [`crate::panel_prep_gpu`] / `src/cuda/prep.cu`. It consumes **already
//! decoded** symbol-major bars; ZIP inflate, CSV tokenisation, unit detection
//! and validation stay in [`crate::prep`].
//!
//! The policy is the one defined by #259:
//!
//! * stable sort by `(symbol, time)`; a symbol is one contiguous segment;
//! * equal-time runs whose nine payload fields and flags are bit-identical
//!   collapse to one bar with `dup_rows = run_len` (never summed);
//! * differing equal-time runs are aggregated exactly once:
//!   `open=first`, `high=max`, `low=min`, `close=last`, flows added
//!   left-to-right, `complete` AND, `traded` OR, and the run is reported as a
//!   `DUPLICATE_CONFLICT`;
//! * a bucket is complete only when it holds the four contiguous expected
//!   `in_step` instants; otherwise it is dropped or emitted with
//!   `cov15m < 4` and `PARTIAL_BUCKET` under `keep_partial`;
//! * OHLC/flow folds use the exact streaming order, with `i64` UTC seconds;
//! * the conflict list is ordered by `(symbol, time)` and is the direct
//!   prefix-sum order the device kernel also uses;
//! * `gap = max(0, (t[i] - t[i-1]) / out_step - 1)` inside one chain, reset at
//!   a listing/delisting (segment change), migration or `member=0` (chain
//!   change).
//!
//! Everything is deterministic: no hash-map iteration, no atomics and no
//! floating-point reassociation.

/// One decoded 15 m (or `in_step`) bar. `chain` is a caller-supplied chain id:
/// a migration or a `member=0` decision starts a new id; a listing/delisting is
/// a new segment. The id is metadata only and never enters OHLC/flow arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrepBar {
    pub time: i64,
    /// Chain id for gap resets (migration, membership exclusion). Segment
    /// boundaries reset the chain as well.
    pub chain: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades: f64,
    pub taker_buy_base: f64,
    pub taker_buy_quote: f64,
    /// All base bars in the source bucket were observed.
    pub complete: bool,
    /// At least one source trade/positive volume occurred.
    pub traded: bool,
}

impl PrepBar {
    /// A bar with flows zeroed, both flags true and chain zero.
    pub fn new(time: i64, open: f64, high: f64, low: f64, close: f64) -> Self {
        PrepBar {
            time,
            chain: 0,
            open,
            high,
            low,
            close,
            volume: 0.0,
            quote_volume: 0.0,
            trades: 0.0,
            taker_buy_base: 0.0,
            taker_buy_quote: 0.0,
            complete: true,
            traded: false,
        }
    }

    pub fn with_flow(mut self, volume: f64, quote_volume: f64, trades: f64) -> Self {
        self.volume = volume;
        self.quote_volume = quote_volume;
        self.trades = trades;
        self.traded = volume > 0.0 || trades > 0.0;
        self
    }

    fn same_payload(&self, other: &PrepBar) -> bool {
        // Bitwise equality is the byte contract: `-0.0` and `+0.0` are
        // different bytes, and a NaN payload must survive a duplicate scan.
        self.open.to_bits() == other.open.to_bits()
            && self.high.to_bits() == other.high.to_bits()
            && self.low.to_bits() == other.low.to_bits()
            && self.close.to_bits() == other.close.to_bits()
            && self.volume.to_bits() == other.volume.to_bits()
            && self.quote_volume.to_bits() == other.quote_volume.to_bits()
            && self.trades.to_bits() == other.trades.to_bits()
            && self.taker_buy_base.to_bits() == other.taker_buy_base.to_bits()
            && self.taker_buy_quote.to_bits() == other.taker_buy_quote.to_bits()
            && self.complete == other.complete
            && self.traded == other.traded
    }
}

/// A contiguous symbol segment in the symbol-major input. Segments must be
/// contiguous, cover the input exactly and preserve listing/delisting order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepSegment {
    pub start: u32,
    pub len: u32,
}

/// One emitted hourly (or `out_step`) bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrepBucket {
    pub time: i64,
    /// Symbol segment id from the input.
    pub segment: u32,
    /// Chain id of the bucket's first unique source bar.
    pub chain: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades: f64,
    pub taker_buy_base: f64,
    pub taker_buy_quote: f64,
    /// Number of observed input bars in the bucket (0..=4 for a complete hour).
    pub cov15m: u32,
    /// Source rows represented by the bucket: the sum of the duplicate run
    /// lengths of its unique bars. Four for a clean complete hour.
    pub dup_rows: u32,
    /// Missing `out_step` buckets since the previous bar in the same chain.
    pub gap: u32,
    pub complete: bool,
    pub traded: bool,
    /// Emitted under `keep_partial` but not structurally complete.
    pub partial: bool,
    /// At least one equal-time run in the bucket was a conflict.
    pub duplicate_conflict: bool,
    /// `gap > 0`; a convenience tag, not a separate state.
    pub outage: bool,
}

/// One deterministic duplicate conflict. The list is sorted by
/// `(segment, time)` exactly like the prefix-sum offsets on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepConflict {
    pub time: i64,
    pub segment: u32,
    /// Chain id of the first copy.
    pub chain: i64,
    /// Number of equal-time input rows in the conflicting run.
    pub dup_rows: u32,
    /// Original (pre-sort) input row index of the first copy.
    pub first_row: u32,
    /// Equal-time run length.
    pub len: u32,
}

/// The result of one fused duplicate/resample/gap pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrepOutput {
    pub buckets: Vec<PrepBucket>,
    pub conflicts: Vec<PrepConflict>,
    pub input_rows: usize,
    pub unique_rows: usize,
}

#[derive(Clone, Copy, Debug)]
struct Unique {
    segment: u32,
    time: i64,
    chain: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    quote_volume: f64,
    trades: f64,
    taker_buy_base: f64,
    taker_buy_quote: f64,
    complete: bool,
    traded: bool,
    dup_rows: u32,
    conflict: bool,
    first_row: u32,
    len: u32,
}

impl Unique {
    fn from_bar(segment: u32, bar: &PrepBar, first_row: u32, len: u32, conflict: bool) -> Self {
        Unique {
            segment,
            time: bar.time,
            chain: bar.chain,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            quote_volume: bar.quote_volume,
            trades: bar.trades,
            taker_buy_base: bar.taker_buy_base,
            taker_buy_quote: bar.taker_buy_quote,
            complete: bar.complete,
            traded: bar.traded,
            dup_rows: len,
            conflict,
            first_row,
            len,
        }
    }
}

fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn bucket_start(t: i64, step: i64) -> i64 {
    floor_div(t, step) * step
}

/// Validate that the segments are contiguous and cover every input row.
pub fn validate_segments(n: usize, segments: &[PrepSegment]) -> Result<(), String> {
    let mut want = 0usize;
    for (i, s) in segments.iter().enumerate() {
        if s.start as usize != want {
            return Err(format!(
                "segment {} starts at {} but the previous segment ends at {}",
                i, s.start, want
            ));
        }
        want = want
            .checked_add(s.len as usize)
            .ok_or_else(|| "segment length overflow".to_string())?;
    }
    if want != n {
        return Err(format!("segments cover {} rows but the input has {}", want, n));
    }
    Ok(())
}

/// The exact CPU reference: duplicate scan, 15 m → `out_step` fold and gap
/// tags. `in_step` is normally 900 and `out_step` 3600.
pub fn prepare(
    bars: &[PrepBar],
    segments: &[PrepSegment],
    in_step: i64,
    out_step: i64,
    keep_partial: bool,
) -> Result<PrepOutput, String> {
    if in_step <= 0 || out_step <= 0 {
        return Err("bar steps must be positive".to_string());
    }
    if out_step % in_step != 0 {
        return Err(format!(
            "output step {} is not a multiple of input step {}",
            out_step, in_step
        ));
    }
    if out_step != 4 * in_step {
        return Err(format!(
            "the panel fold is defined for four input bars per output bar ({} vs {})",
            in_step, out_step
        ));
    }
    validate_segments(bars.len(), segments)?;

    let mut unique: Vec<Unique> = Vec::with_capacity(bars.len());
    let mut conflicts = Vec::new();

    for (seg_id, seg) in segments.iter().enumerate() {
        let start = seg.start as usize;
        let end = start + seg.len as usize;
        // Stable: equal timestamps keep their pre-sort input order, which is
        // the tie order both the CPU and the device use.
        let mut order: Vec<usize> = (start..end).collect();
        order.sort_by_key(|&i| bars[i].time);

        let mut lo = 0usize;
        while lo < order.len() {
            let mut hi = lo + 1;
            while hi < order.len() && bars[order[hi]].time == bars[order[lo]].time {
                hi += 1;
            }
            let first = &bars[order[lo]];
            let identical = order[lo + 1..hi]
                .iter()
                .all(|&j| first.same_payload(&bars[j]));
            let mut u = Unique::from_bar(
                seg_id as u32,
                first,
                order[lo] as u32,
                (hi - lo) as u32,
                !identical,
            );
            if !identical {
                // The fold is in stable-sorted input order, so every flow sum is
                // the exact streaming left-to-right sum. Start them at zero:
                // the first copy is part of the run.
                u.volume = 0.0;
                u.quote_volume = 0.0;
                u.trades = 0.0;
                u.taker_buy_base = 0.0;
                u.taker_buy_quote = 0.0;
                let mut complete = true;
                let mut traded = false;
                for &j in &order[lo..hi] {
                    let b = &bars[j];
                    u.high = u.high.max(b.high);
                    u.low = u.low.min(b.low);
                    u.close = b.close;
                    u.volume += b.volume;
                    u.quote_volume += b.quote_volume;
                    u.trades += b.trades;
                    u.taker_buy_base += b.taker_buy_base;
                    u.taker_buy_quote += b.taker_buy_quote;
                    complete &= b.complete;
                    traded |= b.traded;
                }
                u.complete = complete;
                u.traded = traded;
            }
            if u.conflict {
                conflicts.push(PrepConflict {
                    time: u.time,
                    segment: u.segment,
                    chain: u.chain,
                    dup_rows: u.dup_rows,
                    first_row: u.first_row,
                    len: u.len,
                });
            }
            unique.push(u);
            lo = hi;
        }
    }

    let mut buckets = Vec::new();
    let mut i = 0usize;
    while i < unique.len() {
        let start = bucket_start(unique[i].time, out_step);
        let seg = unique[i].segment;
        let mut j = i + 1;
        while j < unique.len()
            && unique[j].segment == seg
            && bucket_start(unique[j].time, out_step) == start
        {
            j += 1;
        }
        let cov = (j - i) as u32;
        let structural = cov == 4
            && unique[i].time == start
            && (1..4).all(|k| unique[i + k].time == start + k as i64 * in_step);
        if !structural && !keep_partial {
            i = j;
            continue;
        }
        let mut b = PrepBucket {
            time: start,
            segment: seg,
            chain: unique[i].chain,
            open: unique[i].open,
            high: unique[i].high,
            low: unique[i].low,
            close: unique[i].close,
            volume: 0.0,
            quote_volume: 0.0,
            trades: 0.0,
            taker_buy_base: 0.0,
            taker_buy_quote: 0.0,
            cov15m: cov,
            dup_rows: 0,
            gap: 0,
            complete: true,
            traded: false,
            partial: !structural,
            duplicate_conflict: false,
            outage: false,
        };
        for u in &unique[i..j] {
            b.high = b.high.max(u.high);
            b.low = b.low.min(u.low);
            b.close = u.close;
            b.volume += u.volume;
            b.quote_volume += u.quote_volume;
            b.trades += u.trades;
            b.taker_buy_base += u.taker_buy_base;
            b.taker_buy_quote += u.taker_buy_quote;
            b.dup_rows += u.dup_rows;
            b.complete &= u.complete;
            b.duplicate_conflict |= u.conflict;
        }
        b.complete &= structural;
        b.traded = b.volume > 0.0 || b.trades > 0.0;
        buckets.push(b);
        i = j;
    }

    // Gap tags fuse into the output pass. A segment change is a listing or a
    // delisting; a chain change is a migration or `member=0`. Both reset.
    for k in 1..buckets.len() {
        let (prev_time, prev_seg, prev_chain) = {
            let p = &buckets[k - 1];
            (p.time, p.segment, p.chain)
        };
        let b = &mut buckets[k];
        if b.segment != prev_seg || b.chain != prev_chain {
            b.gap = 0;
        } else {
            let missing = (b.time - prev_time) / out_step - 1;
            b.gap = missing.max(0) as u32;
        }
        b.outage = b.gap > 0;
    }

    Ok(PrepOutput {
        buckets,
        conflicts,
        input_rows: bars.len(),
        unique_rows: unique.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(t: i64, p: f64) -> PrepBar {
        PrepBar::new(t, p, p + 1.0, p - 1.0, p + 0.25).with_flow(10.0, 20.0, 2.0)
    }

    #[test]
    fn four_contiguous_bars_make_one_complete_hour() {
        let bars: Vec<_> = (0..4)
            .map(|q| bar(q * 900, 100.0 + q as f64))
            .collect();
        let out = prepare(
            &bars,
            &[PrepSegment { start: 0, len: 4 }],
            900,
            3600,
            false,
        )
        .unwrap();
        assert_eq!(out.buckets.len(), 1);
        let b = &out.buckets[0];
        assert_eq!(b.time, 0);
        assert_eq!(b.cov15m, 4);
        assert!(b.complete && !b.partial);
        assert_eq!(b.open, 100.0);
        assert_eq!(b.close, 103.25);
        assert_eq!(b.high, 104.0);
        assert_eq!(b.low, 99.0);
        assert_eq!(b.volume, 40.0);
        assert_eq!(b.dup_rows, 4);
    }

    #[test]
    fn partial_buckets_are_dropped_or_flagged() {
        for cov in 1..4usize {
            let bars: Vec<_> = (0..cov)
                .map(|q| bar(q as i64 * 900, 100.0))
                .collect();
            let seg = [PrepSegment { start: 0, len: cov as u32 }];
            let dropped = prepare(&bars, &seg, 900, 3600, false).unwrap();
            assert!(dropped.buckets.is_empty());
            let kept = prepare(&bars, &seg, 900, 3600, true).unwrap();
            assert_eq!(kept.buckets.len(), 1);
            assert_eq!(kept.buckets[0].cov15m, cov as u32);
            assert!(kept.buckets[0].partial && !kept.buckets[0].complete);
        }
    }

    #[test]
    fn a_misaligned_four_bar_run_splits_into_two_partial_buckets() {
        let bars: Vec<_> = (0..4)
            .map(|q| bar(900 + q as i64 * 900, 100.0))
            .collect();
        let seg = [PrepSegment { start: 0, len: 4 }];
        assert!(prepare(&bars, &seg, 900, 3600, false).unwrap().buckets.is_empty());
        let out = prepare(&bars, &seg, 900, 3600, true).unwrap();
        let cov: Vec<_> = out.buckets.iter().map(|b| (b.time, b.cov15m, b.partial)).collect();
        assert_eq!(cov, vec![(0, 3, true), (3600, 1, true)]);
    }

    #[test]
    fn identical_duplicates_collapse_without_quadrupling_flow() {
        let mut bars = Vec::new();
        for _ in 0..4 {
            bars.push(bar(0, 100.0));
        }
        for q in 1..4 {
            bars.push(bar(q * 900, 100.0 + q as f64));
        }
        let out = prepare(
            &bars,
            &[PrepSegment { start: 0, len: 7 }],
            900,
            3600,
            false,
        )
        .unwrap();
        assert_eq!(out.buckets.len(), 1);
        assert_eq!(out.buckets[0].volume, 40.0, "not 70 or 160");
        assert_eq!(out.buckets[0].dup_rows, 7);
        assert!(!out.buckets[0].duplicate_conflict);
        assert!(out.conflicts.is_empty());
    }

    #[test]
    fn conflicting_duplicates_are_summed_once_and_reported_in_order() {
        let mut bars = vec![bar(0, 100.0), bar(0, 101.0), bar(0, 102.0)];
        for q in 1..4 {
            bars.push(bar(q * 900, 100.0 + q as f64));
        }
        let out = prepare(
            &bars,
            &[PrepSegment { start: 0, len: 6 }],
            900,
            3600,
            false,
        )
        .unwrap();
        assert_eq!(out.buckets[0].volume, 60.0);
        assert!(out.buckets[0].duplicate_conflict);
        assert_eq!(out.conflicts.len(), 1);
        assert_eq!(out.conflicts[0].time, 0);
        assert_eq!(out.conflicts[0].dup_rows, 3);
        assert_eq!(out.conflicts[0].len, 3);
    }
}
