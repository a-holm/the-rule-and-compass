//! #304: fused long-only `{0, +1}` P&L reduction.
//!
//! This module is the device batch evaluator for the default long-only rule:
//!
//! ```text
//! p_i     = 1 if isfinite(z_i) and z_i > theta else 0
//! d_i     = p_i - p_{i-1},              p_{g,0} = 0
//! gross_i = p_i * r_i
//! cost_i  = s * |d_i| + [i = last row of group] * s * p_i
//! net_i   = gross_i - cost_i
//! ```
//!
//! `s` is the per-side cost in basis points divided by 10,000. There is no
//! short leg: a position that is not `0` or `1` is rejected by
//! [`check_long_only`]. The final exit is charged once per group, on the last
//! row of that group; a flat book is free.
//!
//! # Canonical segments and determinism
//!
//! Rows are split into fixed [`DEFAULT_SEGMENT_ROWS`]-row canonical segments.
//! Inside a segment rows are partitioned into [`PNL_LANES`] virtual lanes by
//! `(row - segment_start) % PNL_LANES`; each lane accumulates in `f64`, the 32
//! lane partials are merged by a fixed pairwise tree, and the segment partials
//! are merged in ascending segment order. Nothing depends on the chunk size
//! used to feed the rows, on the launch geometry or on the batch composition:
//!
//! * a chunk that ends inside a segment continues the same lane accumulators,
//!   so alone, padded, 4 k and 64 k chunks produce identical bits;
//! * a segment is always split over the same 32 lanes, so 128, 256 and 512
//!   threads only change which warp owns a segment, never the summation order;
//! * there are no floating-point atomics and no cross-lane tree over rows.
//!
//! All accumulation is `f64`, including on the device. Expression evaluation
//! may stay `f32`; the score is converted once per row and every P&L term is
//! then added in `f64`.
//!
//! # Memory
//!
//! The device path never materialises a `P x N` score plane. Candidates are
//! streamed through a declared tile of at most [`MAX_TILE_CANDIDATES`]
//! candidates ([`crate::trading_gpu::MAX_TILE_CANDIDATES`]), so resident score
//! memory is `O(tile * N)` with a fixed tile, independent of the population
//! size `P`.
//!
//! The CPU reference in [`reduce`] is always compiled. The device path is
//! opt-in behind `--features cuda` and falls back to the same canonical CPU
//! reduction when no device is available.

//!
//! Exact trading threshold selection and the shared valid-row mask (#305).
//!
//! The long-only rule compares each score to a **discrete** threshold: one
//! rounding difference flips every row whose score sits at the boundary and can
//! move the reported P&L by a whole position. This module therefore selects the
//! type-7 (numpy default) training quantile exactly, by order-preserving radix
//! selection, instead of sorting or sketching:
//!
//! * every finite `f32`/`f64` maps to an unsigned key that orders exactly like
//!   the value; `-0.0` is canonicalised to `+0.0`;
//! * non-finite values are excluded, exactly as [`walkforward::quantile`]
//!   filters them;
//! * the two ranks bracketing `h = (n - 1) * q` are selected with a 256-bin
//!   radix count, and the interpolation is done in `f64` with the same
//!   expression the CPU sort path uses.
//!
//! Selection is a pure function of the multiset of finite values, so batch
//! composition, block boundaries and the number of threads that produced a
//! partial histogram cannot change the result. [`type7_quantile_f32_parts`]
//! exists to make that contract testable: the same rows handed over as one
//! slice, as 4 k/64 k chunks, or strided across 128/256/512 "threads" produce
//! the same bits.
//!
//! The shared [`ValidMask`] drops a row when **either** its score or its target
//! is non-finite, matching `walkforward::evaluate_window`. That is deliberately
//! stricter than `walkforward::position_for`, which only looks at the score: in
//! the pinned cell six rows have a finite score and a `NaN` target, and keeping
//! them would change the denominator of every P&L statistic.
//!
//! A sketch quantile is still useful for screening, but it must never place a
//! reported position. [`Threshold::screening_only`] carries that label and
//! [`Threshold::long_positions`] refuses it.
//!
//! `walkforward::quantile` delegates here, so the CPU report path and the CUDA
//! parity path share one definition.

use std::cmp::Ordering;
use std::collections::HashSet;

/// Canonical segment length in rows. Fixed so that a launch or chunk change
/// cannot change the summation tree.
pub const DEFAULT_SEGMENT_ROWS: usize = 8_192;

/// Virtual lanes per canonical segment. A segment's rows are partitioned by
/// `(row - segment_start) % LANES`, each lane accumulates in `f64`, and the 32
/// lane partials are merged by a fixed pairwise tree. This is the same partition
/// the device warp uses, so the CPU reference is bit-identical while the device
/// reads coefficients across 32 consecutive rows.
pub const PNL_LANES: usize = 32;

/// Declared score tile width. This is a constant, not the population size: the
/// device path never allocates `P x N`.
pub const MAX_TILE_CANDIDATES: usize = 8;

/// Default per-side cost in basis points, matching `walkforward`.
pub const DEFAULT_COST_BPS: f64 = 10.0;


/// The method that produced a reported threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdSource {
    /// Exact type-7 quantile of the training scores.
    TrainType7,
    /// Exact type-7 quantile of the validation scores.
    ValidationType7,
    /// A pre-registered constant threshold, never fitted to the data.
    Fixed,
    /// The top-tercile policy has no threshold (#493).
    None,
}

impl ThresholdSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ThresholdSource::None => "none",
            ThresholdSource::TrainType7 => "train_type7",
            ThresholdSource::ValidationType7 => "validation_type7",
            ThresholdSource::Fixed => "fixed",
        }
    }
}

// ---------------------------------------------------------------------------
// Order-preserving keys
// ---------------------------------------------------------------------------

/// Order-preserving key of a finite `f32`, or `None` for NaN/±inf.
///
/// Negative zero maps to the key of positive zero. The transform is the usual
/// sign-bit flip: positive keys have the top bit set, negative keys are
/// bitwise-inverted, so unsigned comparison is value comparison.
#[inline]
pub fn order_key_f32(v: f32) -> Option<u32> {
    if !v.is_finite() {
        return None;
    }
    let bits = if v == 0.0 { 0u32 } else { v.to_bits() };
    Some(if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    })
}

/// Inverse of [`order_key_f32`] for finite values.
#[inline]
pub fn f32_from_order_key(key: u32) -> f32 {
    let bits = if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// Order-preserving key of a finite `f64`, or `None` for NaN/±inf.
#[inline]
pub fn order_key_f64(v: f64) -> Option<u64> {
    if !v.is_finite() {
        return None;
    }
    let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
    Some(if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000_0000_0000
    })
}

/// Inverse of [`order_key_f64`] for finite values.
#[inline]
pub fn f64_from_order_key(key: u64) -> f64 {
    let bits = if key & 0x8000_0000_0000_0000 != 0 {
        key ^ 0x8000_0000_0000_0000
    } else {
        !key
    };
    f64::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Exact type-7 quantile
// ---------------------------------------------------------------------------

/// The exact type-7 quantile and the ranks it used.
///
/// `theta` is the value the report should place positions with. The remaining
/// fields make the selection auditable: `n` is the number of finite values,
/// `h = (n - 1) * q` (after clamping `q`), `lo`/`hi` are the bracketing ranks
/// and `lo_value`/`hi_value` are their order statistics.
#[derive(Clone, Copy, Debug)]
pub struct Type7Quantile {
    pub theta: f64,
    pub n: usize,
    pub q: f64,
    pub h: f64,
    pub lo: usize,
    pub hi: usize,
    pub lo_value: f64,
    pub hi_value: f64,
}

impl Type7Quantile {
    /// The empty sample's quantile is `NaN`, matching `numpy.quantile([])` and
    /// [`walkforward::quantile`].
    pub fn empty(q: f64) -> Self {
        Type7Quantile {
            theta: f64::NAN,
            n: 0,
            q,
            h: f64::NAN,
            lo: 0,
            hi: 0,
            lo_value: f64::NAN,
            hi_value: f64::NAN,
        }
    }

    /// Finish a selection from the two bracket values.
    ///
    /// The arithmetic is written in exactly the same order as
    /// `walkforward::quantile` used before #305:
    /// `lo + (h - lo as f64) * (hi - lo)`, in `f64`. Reordering it would change
    /// the last bit on a non-tied interpolation.
    fn finish(n: usize, q: f64, lo: usize, hi: usize, lo_value: f64, hi_value: f64) -> Self {
        if n == 0 {
            return Type7Quantile::empty(q);
        }
        if n == 1 {
            return Type7Quantile {
                theta: lo_value,
                n,
                q,
                h: 0.0,
                lo: 0,
                hi: 0,
                lo_value,
                hi_value: lo_value,
            };
        }
        let q = q.clamp(0.0, 1.0);
        let h = (n - 1) as f64 * q;
        let theta = if lo == hi {
            lo_value
        } else {
            lo_value + (h - lo as f64) * (hi_value - lo_value)
        };
        Type7Quantile {
            theta,
            n,
            q,
            h,
            lo,
            hi,
            lo_value,
            hi_value,
        }
    }

    /// Compute `h` and the bracketing ranks for a non-empty sample.
    fn ranks(n: usize, q: f64) -> (f64, usize, usize) {
        let q = q.clamp(0.0, 1.0);
        let h = (n - 1) as f64 * q;
        (h, h.floor() as usize, h.ceil() as usize)
    }
}

/// Exact type-7 quantile of `f32` scores.
pub fn type7_quantile_f32(scores: &[f32], q: f64) -> Type7Quantile {
    let keys: Vec<u32> = scores.iter().filter_map(|&v| order_key_f32(v)).collect();
    type7_from_keys_u32(&keys, q)
}

/// Exact type-7 quantile of `f64` scores.
pub fn type7_quantile_f64(scores: &[f64], q: f64) -> Type7Quantile {
    let keys: Vec<u64> = scores.iter().filter_map(|&v| order_key_f64(v)).collect();
    type7_from_keys_u64(&keys, q)
}

/// Exact type-7 quantile over a partition of `f32` scores.
///
/// Flattening the parts is deliberate: rank selection is a multiset operation,
/// so no part boundary or part order can influence the result. Tests use this
/// to pin batch-composition and thread-count invariance.
pub fn type7_quantile_f32_parts(parts: &[&[f32]], q: f64) -> Type7Quantile {
    let mut keys = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for part in parts {
        keys.extend(part.iter().filter_map(|&v| order_key_f32(v)));
    }
    type7_from_keys_u32(&keys, q)
}

/// Same as [`type7_quantile_f32_parts`] for `f64`.
pub fn type7_quantile_f64_parts(parts: &[&[f64]], q: f64) -> Type7Quantile {
    let mut keys = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for part in parts {
        keys.extend(part.iter().filter_map(|&v| order_key_f64(v)));
    }
    type7_from_keys_u64(&keys, q)
}

fn type7_from_keys_u32(keys: &[u32], q: f64) -> Type7Quantile {
    let n = keys.len();
    if n == 0 {
        return Type7Quantile::empty(q);
    }
    if n == 1 {
        let v = f32_from_order_key(keys[0]) as f64;
        return Type7Quantile::finish(n, q, 0, 0, v, v);
    }
    let (_, lo, hi) = Type7Quantile::ranks(n, q);
    let lo_key = radix_select_u32(keys, lo);
    let hi_key = if hi == lo {
        lo_key
    } else {
        radix_select_u32(keys, hi)
    };
    Type7Quantile::finish(
        n,
        q,
        lo,
        hi,
        f32_from_order_key(lo_key) as f64,
        f32_from_order_key(hi_key) as f64,
    )
}

fn type7_from_keys_u64(keys: &[u64], q: f64) -> Type7Quantile {
    let n = keys.len();
    if n == 0 {
        return Type7Quantile::empty(q);
    }
    if n == 1 {
        let v = f64_from_order_key(keys[0]);
        return Type7Quantile::finish(n, q, 0, 0, v, v);
    }
    let (_, lo, hi) = Type7Quantile::ranks(n, q);
    let lo_key = radix_select_u64(keys, lo);
    let hi_key = if hi == lo {
        lo_key
    } else {
        radix_select_u64(keys, hi)
    };
    Type7Quantile::finish(
        n,
        q,
        lo,
        hi,
        f64_from_order_key(lo_key),
        f64_from_order_key(hi_key),
    )
}

/// Select the `rank`-th smallest `u32` key (zero-based) with a 256-bin radix
/// count.
///
/// The key space is fixed (32 bits, four byte passes), so the work is bounded
/// and does not depend on the data distribution. Ties are indistinguishable by
/// construction: the returned integer is the order statistic itself.
pub fn radix_select_u32(keys: &[u32], rank: usize) -> u32 {
    assert!(
        rank < keys.len(),
        "rank {} out of bounds for {} keys",
        rank,
        keys.len()
    );
    let mut active: Vec<u32> = keys.to_vec();
    let mut rank = rank;
    let mut prefix = 0u32;
    let mut shift = 24i32;
    loop {
        let mut hist = [0usize; 256];
        for &key in &active {
            hist[((key >> shift) & 0xff) as usize] += 1;
        }
        let mut before = 0usize;
        let mut bucket = 0usize;
        for (b, &count) in hist.iter().enumerate() {
            if rank < before + count {
                bucket = b;
                break;
            }
            before += count;
        }
        rank -= before;
        prefix |= (bucket as u32) << shift;
        active.retain(|&key| ((key >> shift) & 0xff) as usize == bucket);
        if shift == 0 {
            return prefix;
        }
        shift -= 8;
    }
}

/// Select the `rank`-th smallest `u64` key (zero-based) with a 256-bin radix
/// count.
pub fn radix_select_u64(keys: &[u64], rank: usize) -> u64 {
    assert!(
        rank < keys.len(),
        "rank {} out of bounds for {} keys",
        rank,
        keys.len()
    );
    let mut active: Vec<u64> = keys.to_vec();
    let mut rank = rank;
    let mut prefix = 0u64;
    let mut shift = 56i32;
    loop {
        let mut hist = [0usize; 256];
        for &key in &active {
            hist[((key >> shift) & 0xff) as usize] += 1;
        }
        let mut before = 0usize;
        let mut bucket = 0usize;
        for (b, &count) in hist.iter().enumerate() {
            if rank < before + count {
                bucket = b;
                break;
            }
            before += count;
        }
        rank -= before;
        prefix |= (bucket as u64) << shift;
        active.retain(|&key| ((key >> shift) & 0xff) as usize == bucket);
        if shift == 0 {
            return prefix;
        }
        shift -= 8;
    }
}

// ---------------------------------------------------------------------------
// Shared valid-row mask
// ---------------------------------------------------------------------------

/// One bit per row: the row is usable only when its score **and** target are
/// finite.
///
/// This is the mask `walkforward::evaluate_window` applies before placing
/// positions. It is shared by the P&L, risk and baseline paths so they cannot
/// disagree about the denominator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidMask {
    words: Vec<u64>,
    len: usize,
    valid: usize,
}

impl ValidMask {
    /// All rows valid until [`ValidMask::set`] says otherwise.
    pub fn new(len: usize) -> Self {
        ValidMask {
            words: vec![0u64; (len + 63) / 64],
            len,
            valid: 0,
        }
    }

    /// Build from parallel score and target slices. The lengths must match.
    pub fn from_finites(scores: &[f64], targets: &[f64]) -> Self {
        assert_eq!(
            scores.len(),
            targets.len(),
            "score and target slices must have the same length"
        );
        let mut mask = ValidMask::new(scores.len());
        for i in 0..scores.len() {
            mask.set(i, scores[i].is_finite() && targets[i].is_finite());
        }
        mask
    }

    /// Score-only mask, i.e. the row a bare `position_for` would consider. It
    /// is exposed for the documented score-only/score-and-target difference;
    /// reported P&L must use [`ValidMask::from_finites`].
    pub fn from_scores(scores: &[f64]) -> Self {
        let mut mask = ValidMask::new(scores.len());
        for (i, &score) in scores.iter().enumerate() {
            mask.set(i, score.is_finite());
        }
        mask
    }

    /// Set one row's validity. Re-setting the same value is idempotent.
    pub fn set(&mut self, row: usize, valid: bool) {
        assert!(row < self.len, "row {} out of bounds", row);
        let word = row / 64;
        let bit = 1u64 << (row % 64);
        let was = self.words[word] & bit != 0;
        if valid && !was {
            self.words[word] |= bit;
            self.valid += 1;
        } else if !valid && was {
            self.words[word] &= !bit;
            self.valid -= 1;
        }
    }

    #[inline]
    pub fn is_valid(&self, row: usize) -> bool {
        assert!(row < self.len, "row {} out of bounds", row);
        self.words[row / 64] & (1u64 << (row % 64)) != 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of valid rows.
    #[inline]
    pub fn valid_count(&self) -> usize {
        self.valid
    }

    /// Number of dropped rows (`len - valid_count`).
    #[inline]
    pub fn dropped_count(&self) -> usize {
        self.len - self.valid
    }

    /// Valid row indices in ascending row order. This is the canonical order:
    /// it never depends on how the mask was produced.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.is_valid(i))
    }

    /// The packed bits, least-significant bit first. The CUDA mask kernel writes
    /// the same layout, so a device mask can be compared word for word.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Build a mask from packed words (the inverse of [`ValidMask::words`]).
    pub fn from_words(words: &[u64], len: usize) -> Self {
        assert!(words.len() == (len + 63) / 64, "word count does not match len");
        let mut mask = ValidMask {
            words: words.to_vec(),
            len,
            valid: 0,
        };
        let mut valid = 0usize;
        for i in 0..len {
            if mask.words[i / 64] & (1u64 << (i % 64)) != 0 {
                valid += 1;
            }
        }
        mask.valid = valid;
        mask
    }
}

// ---------------------------------------------------------------------------
// Thresholds and positions
// ---------------------------------------------------------------------------

/// A threshold together with the provenance the report must carry.
///
/// Only [`Threshold::train_type7`], [`Threshold::validation_type7`] and
/// [`Threshold::fixed`] are report-eligible. [`Threshold::screening_only`]
/// exists for a cheap pre-pass and is refused by
/// [`Threshold::long_positions`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Threshold {
    value: f64,
    source: ThresholdSource,
    screening: bool,
}

impl Threshold {
    pub fn train_type7(scores: &[f64], q: f64) -> Self {
        Threshold {
            value: type7_quantile_f64(scores, q).theta,
            source: ThresholdSource::TrainType7,
            screening: false,
        }
    }

    pub fn validation_type7(scores: &[f64], q: f64) -> Self {
        Threshold {
            value: type7_quantile_f64(scores, q).theta,
            source: ThresholdSource::ValidationType7,
            screening: false,
        }
    }

    /// A pre-registered constant threshold.
    pub fn fixed(value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("a fixed threshold must be finite".to_string());
        }
        Ok(Threshold {
            value,
            source: ThresholdSource::Fixed,
            screening: false,
        })
    }

    /// An approximate threshold from a sketch or a subsample.
    ///
    /// It may rank candidates during screening. It must never be used to
    /// place a reported position, which is why the type carries the label.
    pub fn screening_only(value: f64) -> Self {
        Threshold {
            value,
            source: ThresholdSource::TrainType7,
            screening: true,
        }
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    #[inline]
    pub fn source(&self) -> ThresholdSource {
        self.source
    }

    #[inline]
    pub fn is_screening_only(&self) -> bool {
        self.screening
    }

    /// True when this threshold may place reported positions.
    #[inline]
    pub fn is_report_eligible(&self) -> bool {
        !self.screening
    }

    /// The reported value, or the refusal a screening threshold earns.
    pub fn require_report_eligible(&self) -> Result<f64, String> {
        if self.screening {
            Err("a screening-only threshold may not place reported positions; \
                 select the exact type-7 threshold for the report"
                .to_string())
        } else {
            Ok(self.value)
        }
    }
}

/// Long-only position for one row: `1` iff the score is finite and strictly
/// greater than the threshold. `score == theta` is flat, not long.
#[inline]
pub fn long_position(score: f64, threshold: f64) -> u8 {
    if score.is_finite() && score > threshold {
        1
    } else {
        0
    }
}

/// Positions over the valid rows of a mask, in ascending row order.
///
/// This is the vector the report is built from. A non-finite score can never
/// be long even if the mask says the row is otherwise usable.
pub fn long_positions(
    scores: &[f64],
    mask: &ValidMask,
    threshold: &Threshold,
) -> Result<Vec<u8>, String> {
    let theta = threshold.require_report_eligible()?;
    Ok(mask
        .indices()
        .map(|i| long_position(scores[i], theta))
        .collect())
}

/// Compare the exact `f32` type-7 selection with a straightforward CPU sort,
/// for the parity tests. The sort is the reference, not a production path.
pub fn type7_quantile_f32_sorted_reference(scores: &[f32], q: f64) -> Type7Quantile {
    let mut v: Vec<f64> = scores
        .iter()
        .filter(|v| v.is_finite())
        // The exact key canonicalises `-0.0` to `+0.0`; the reference does the
        // same so a signed-zero order statistic is compared as one value.
        .map(|&v| {
            let x = v as f64;
            if x == 0.0 {
                0.0
            } else {
                x
            }
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if v.is_empty() {
        return Type7Quantile::empty(q);
    }
    if v.len() == 1 {
        return Type7Quantile::finish(1, q, 0, 0, v[0], v[0]);
    }
    let q = q.clamp(0.0, 1.0);
    let h = (v.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    Type7Quantile::finish(v.len(), q, lo, hi, v[lo], v[hi])
}

/// Parameters of one long-only reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PnlParams {
    /// Long when `score > threshold` and the score is finite.
    pub threshold: f64,
    /// Cost per side in basis points.
    pub per_side_bps: f64,
    /// Canonical segment length in rows.
    pub segment_rows: usize,
    /// Per-row cash return used by the cash benchmark (default `0`).
    pub cash_return: f64,
}

impl Default for PnlParams {
    fn default() -> Self {
        PnlParams {
            threshold: 0.0,
            per_side_bps: DEFAULT_COST_BPS,
            segment_rows: DEFAULT_SEGMENT_ROWS,
            cash_return: 0.0,
        }
    }
}

impl PnlParams {
    /// Per-side cost as a fraction of notional.
    pub fn side(&self) -> f64 {
        self.per_side_bps / 10_000.0
    }
}

/// One candidate's additive long-only P&L.
///
/// `net`, `gross` and `cost` are independent `f64` sums: `net` is accumulated
/// per row, so it can differ from `gross - cost` by a final rounding step.
/// `turnover` is `sum |position change|`, without the final-exit charge.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PnlTotals {
    pub rows: u64,
    pub long_rows: u64,
    pub entries: u64,
    /// Rule exits plus final exits: the fixture has 2 rule exits and 1 final
    /// exit in 3 groups, so `exits == 3`.
    pub exits: u64,
    /// Non-zero position changes, excluding the final-exit charge.
    pub trades: u64,
    /// Long rows with a strictly positive return.
    pub hits: u64,
    pub final_exits: u64,
    pub gross: f64,
    /// Entry/exit cost plus the final-exit charge.
    pub cost: f64,
    /// `sum |position change|`; the final exit is deliberately not included.
    pub turnover: f64,
    pub net: f64,
}

impl PnlTotals {
    /// Add one canonical segment partial. Used in ascending segment order.
    pub fn merge(&mut self, other: &PnlTotals) {
        self.rows += other.rows;
        self.long_rows += other.long_rows;
        self.entries += other.entries;
        self.exits += other.exits;
        self.trades += other.trades;
        self.hits += other.hits;
        self.final_exits += other.final_exits;
        self.gross += other.gross;
        self.cost += other.cost;
        self.turnover += other.turnover;
        self.net += other.net;
    }

    /// Long rows divided by valid rows (`NaN` when there are no rows).
    pub fn exposure(&self) -> f64 {
        if self.rows == 0 {
            f64::NAN
        } else {
            self.long_rows as f64 / self.rows as f64
        }
    }

    /// Long rows with a positive return divided by long rows.
    pub fn hit_rate(&self) -> f64 {
        if self.long_rows == 0 {
            f64::NAN
        } else {
            self.hits as f64 / self.long_rows as f64
        }
    }

    /// A stable bit fingerprint for determinism tests and provenance.
    pub fn bits(&self) -> [u64; 4] {
        [
            self.gross.to_bits(),
            self.cost.to_bits(),
            self.turnover.to_bits(),
            self.net.to_bits(),
        ]
    }
}

/// The strategy and the two required benchmarks for one candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct PnlReport {
    pub strategy: PnlTotals,
    /// `p = 0`: no cost, no gross; `net` is the declared cash return.
    pub cash: PnlTotals,
    /// `p = 1`: one entry and one charged exit per group,
    /// `net = sum(r) - 2s * groups`.
    pub buy_and_hold: PnlTotals,
}

/// One row's contribution, exposed for the arithmetic fixture and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowTerm {
    pub position: f64,
    pub order: f64,
    pub gross: f64,
    pub cost: f64,
    pub net: f64,
    pub final_exit: bool,
}

#[inline]
fn position_for(score: f64, threshold: f64) -> f64 {
    long_position(score, threshold) as f64
}

/// Reject any position that is not exactly `0` or `1`.
///
/// The default long-only rule cannot produce a negative position; a caller that
/// supplies an explicit position vector is checked through here. There is no
/// short path in this module.
pub fn check_long_only(positions: &[f64]) -> Result<(), String> {
    for (i, &p) in positions.iter().enumerate() {
        if p != 0.0 && p != 1.0 {
            return Err(format!(
                "row {}: position {} is not long-only ({{0, +1}}); short positions are not implemented",
                i, p
            ));
        }
    }
    Ok(())
}

/// Validate that each group is one contiguous run of rows. The device path
/// reconstructs `p_{i-1}` from the previous row, so a group that reappears
/// after another group has no well-defined seam.
pub fn check_group_contiguous(groups: &[u64]) -> Result<(), String> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut current: Option<u64> = None;
    for (i, &g) in groups.iter().enumerate() {
        if current != Some(g) {
            if !seen.insert(g) {
                return Err(format!(
                    "row {}: group {} reappears after another group; rows must be \
                     group-contiguous and time-ordered",
                    i, g
                ));
            }
            current = Some(g);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// #306: finite-state no-trade band / cooldown and hindsight DP
// ---------------------------------------------------------------------------

/// A long-only band/cooldown rule. The transition is deterministic:
///
/// ```text
/// flat  -> long iff score is finite and score > threshold + band
/// long  -> flat iff score is non-finite or score <= threshold - band
///                 and the minimum holding count has been reached
/// ```
///
/// With `band == 0` and `min_hold == 1` this is exactly the plain rule
/// `long iff score is finite and score > threshold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandParams {
    pub threshold: f64,
    /// Half-width of the no-trade band (`beta`), in score units.
    pub band: f64,
    /// Minimum consecutive long rows before an exit is allowed (>= 1).
    pub min_hold: usize,
}

impl Default for BandParams {
    fn default() -> Self {
        BandParams {
            threshold: 0.0,
            band: 0.0,
            min_hold: 1,
        }
    }
}

impl BandParams {
    pub fn new(threshold: f64, band: f64, min_hold: usize) -> Result<Self, String> {
        let p = BandParams {
            threshold,
            band,
            min_hold,
        };
        p.validate()?;
        Ok(p)
    }

    fn validate(&self) -> Result<(), String> {
        if !self.threshold.is_finite() {
            return Err(format!("threshold must be finite, got {}", self.threshold));
        }
        if !self.band.is_finite() || self.band < 0.0 {
            return Err(format!(
                "band width must be finite and non-negative, got {}",
                self.band
            ));
        }
        if self.min_hold == 0 {
            return Err("min_hold must be at least 1".to_string());
        }
        Ok(())
    }

    /// One deterministic state transition. `held` counts the entry row.
    #[inline]
    pub fn next(&self, state: BandState, score: f64) -> BandState {
        if state.long {
            let exit = !score.is_finite() || score <= self.threshold - self.band;
            if exit && state.held >= self.min_hold {
                BandState {
                    long: false,
                    held: 0,
                }
            } else {
                BandState {
                    long: true,
                    held: state.held.saturating_add(1),
                }
            }
        } else if score.is_finite() && score > self.threshold + self.band {
            BandState {
                long: true,
                held: 1,
            }
        } else {
            BandState {
                long: false,
                held: 0,
            }
        }
    }
}

/// The two-state long-only band state. `held` is only meaningful while long.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BandState {
    pub long: bool,
    pub held: usize,
}

impl BandState {
    pub fn position(self) -> f64 {
        if self.long {
            1.0
        } else {
            0.0
        }
    }
}

/// Plain threshold positions, exposed so a test can pin `band(beta=0)` against
/// it bit for bit.
pub fn threshold_positions(scores: &[f64], threshold: f64) -> Vec<f64> {
    scores
        .iter()
        .map(|&z| position_for(z, threshold))
        .collect()
}

/// Serial band/cooldown reference. Groups must be contiguous; every group
/// starts flat and its state is reset at the seam.
pub fn band_positions(
    scores: &[f64],
    groups: &[u64],
    params: &BandParams,
) -> Result<Vec<f64>, String> {
    if scores.len() != groups.len() {
        return Err(format!(
            "length mismatch: scores {}, groups {}",
            scores.len(),
            groups.len()
        ));
    }
    params.validate()?;
    check_group_contiguous(groups)?;
    let mut out = Vec::with_capacity(scores.len());
    let mut state = BandState::default();
    let mut current: Option<u64> = None;
    for (&score, &group) in scores.iter().zip(groups) {
        if current != Some(group) {
            state = BandState::default();
            current = Some(group);
        }
        state = params.next(state, score);
        out.push(state.position());
    }
    Ok(out)
}

/// The 2-state carry of issue #306. For each start state `s` it stores the end
/// state `e_s` and the reward `w_s` of the row block.
///
/// Composition is left-then-right and associative:
///
/// ```text
/// e(s) = right.e(left.e(s))
/// w(s) = left.w(s) + right.w(left.e(s))
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Carry2 {
    pub e0: usize,
    pub e1: usize,
    pub w0: f64,
    pub w1: f64,
}

impl Carry2 {
    /// The identity block: both start states end in themselves with zero
    /// reward. `e0 = 0`, `e1 = 1`, `w0 = w1 = 0`.
    pub const IDENTITY: Carry2 = Carry2 {
        e0: 0,
        e1: 1,
        w0: 0.0,
        w1: 0.0,
    };

    /// Build a block from its per-start-state end state and reward.
    pub fn from_row(end: [usize; 2], reward: [f64; 2]) -> Carry2 {
        Carry2 {
            e0: end[0],
            e1: end[1],
            w0: reward[0],
            w1: reward[1],
        }
    }

    #[inline]
    pub fn end(&self, start: usize) -> usize {
        if start == 0 {
            self.e0
        } else {
            self.e1
        }
    }

    #[inline]
    pub fn reward(&self, start: usize) -> f64 {
        if start == 0 {
            self.w0
        } else {
            self.w1
        }
    }

    /// Compose `self` (earlier rows) with `right` (later rows).
    pub fn then(&self, right: &Carry2) -> Carry2 {
        Carry2 {
            e0: right.end(self.e0),
            e1: right.end(self.e1),
            w0: self.w0 + right.reward(self.e0),
            w1: self.w1 + right.reward(self.e1),
        }
    }

    /// End state and reward for a starting state, as a tuple.
    pub fn apply(&self, start: usize) -> (usize, f64) {
        (self.end(start), self.reward(start))
    }

    /// Stable bit fingerprint for chunk-invariance tests.
    pub fn bits(&self) -> [u64; 4] {
        [self.e0 as u64, self.e1 as u64, self.w0.to_bits(), self.w1.to_bits()]
    }
}

/// A K+1-state carry for the minimum-holding/cooldown band. State 0 is flat;
/// state `h` is long with `held == h` (clamped to `K`).
#[derive(Clone, Debug, PartialEq)]
pub struct CooldownCarry {
    pub end: Vec<usize>,
    pub reward: Vec<f64>,
}

impl CooldownCarry {
    pub fn identity(states: usize) -> CooldownCarry {
        CooldownCarry {
            end: (0..states).collect(),
            reward: vec![0.0; states],
        }
    }

    pub fn then(&self, right: &CooldownCarry) -> CooldownCarry {
        assert_eq!(self.end.len(), right.end.len());
        let mut end = Vec::with_capacity(self.end.len());
        let mut reward = Vec::with_capacity(self.end.len());
        for s in 0..self.end.len() {
            let mid = self.end[s];
            end.push(right.end[mid]);
            reward.push(self.reward[s] + right.reward[mid]);
        }
        CooldownCarry { end, reward }
    }

    pub fn bits(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.end.len() * 2);
        out.extend(self.end.iter().map(|&s| s as u64));
        out.extend(self.reward.iter().map(|v| v.to_bits()));
        out
    }
}

#[inline]
fn carry_start_state(state: usize, min_hold: usize) -> BandState {
    if state == 0 {
        BandState::default()
    } else {
        BandState {
            long: true,
            held: state.min(min_hold),
        }
    }
}

#[inline]
fn carry_state_index(state: BandState, min_hold: usize) -> usize {
    if state.long {
        state.held.max(1).min(min_hold)
    } else {
        0
    }
}

fn band_row_carry(
    params: &BandParams,
    score: f64,
    ret: f64,
    side: f64,
    final_exit: bool,
) -> Carry2 {
    let starts = [0usize, 1usize];
    let mut end = [0usize; 2];
    let mut reward = [0.0f64; 2];
    for (k, &start) in starts.iter().enumerate() {
        let before = carry_start_state(start, 1);
        let after = params.next(before, score);
        let p0 = before.position();
        let p1 = after.position();
        let mut w = p1 * ret - side * (p1 - p0).abs();
        if final_exit && p1 != 0.0 {
            w -= side;
        }
        end[k] = if after.long { 1 } else { 0 };
        reward[k] = w;
    }
    Carry2::from_row(end, reward)
}

fn cooldown_row_carry(
    params: &BandParams,
    score: f64,
    ret: f64,
    side: f64,
    final_exit: bool,
) -> CooldownCarry {
    let states = params.min_hold + 1;
    let mut end = Vec::with_capacity(states);
    let mut reward = Vec::with_capacity(states);
    for s in 0..states {
        let before = carry_start_state(s, params.min_hold);
        let after = params.next(before, score);
        let p0 = before.position();
        let p1 = after.position();
        let mut w = p1 * ret - side * (p1 - p0).abs();
        if final_exit && p1 != 0.0 {
            w -= side;
        }
        end.push(carry_state_index(after, params.min_hold));
        reward.push(w);
    }
    CooldownCarry { end, reward }
}

/// A band scan's positions and one composed carry per group (ascending).
#[derive(Clone, Debug, PartialEq)]
pub struct BandScan {
    pub positions: Vec<f64>,
    pub carries: Vec<Carry2>,
}

impl BandScan {
    /// Additive net including the final exit of every group. Group carries are
    /// added in ascending order.
    pub fn net(&self) -> f64 {
        let mut total = 0.0f64;
        for carry in &self.carries {
            total += carry.w0;
        }
        total
    }
}

/// A cooldown scan's positions and one composed K+1-state carry per group.
#[derive(Clone, Debug, PartialEq)]
pub struct CooldownScan {
    pub positions: Vec<f64>,
    pub carries: Vec<CooldownCarry>,
}

impl CooldownScan {
    pub fn net(&self) -> f64 {
        let mut total = 0.0f64;
        for carry in &self.carries {
            total += carry.reward[0];
        }
        total
    }
}

fn validate_scan(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
    chunk_rows: usize,
) -> Result<(), String> {
    if scores.len() != returns.len() || groups.len() != returns.len() {
        return Err(format!(
            "length mismatch: scores {}, returns {}, groups {}",
            scores.len(),
            returns.len(),
            groups.len()
        ));
    }
    params.validate()?;
    if !side.is_finite() || side < 0.0 {
        return Err(format!("side must be finite and non-negative, got {}", side));
    }
    if chunk_rows == 0 {
        return Err("chunk_rows must be at least 1".to_string());
    }
    check_group_contiguous(groups)?;
    for (i, &r) in returns.iter().enumerate() {
        if !r.is_finite() {
            return Err(format!("row {}: non-finite return {}", i, r));
        }
    }
    Ok(())
}

fn require_two_state(params: &BandParams) -> Result<(), String> {
    if params.min_hold > 1 {
        return Err(format!(
            "band_scan is the 2-state carry; min_hold {} needs cooldown_scan ({} states)",
            params.min_hold,
            params.min_hold + 1
        ));
    }
    Ok(())
}

fn validate_dp(returns: &[f64], groups: &[u64], side: f64, chunk_rows: usize) -> Result<(), String> {
    if returns.len() != groups.len() {
        return Err(format!(
            "length mismatch: returns {}, groups {}",
            returns.len(),
            groups.len()
        ));
    }
    if !side.is_finite() || side < 0.0 {
        return Err(format!("side must be finite and non-negative, got {}", side));
    }
    if chunk_rows == 0 {
        return Err("chunk_rows must be at least 1".to_string());
    }
    check_group_contiguous(groups)?;
    for (i, &r) in returns.iter().enumerate() {
        if !r.is_finite() {
            return Err(format!("row {}: non-finite return {}", i, r));
        }
    }
    Ok(())
}

/// Band scan with the 2-state [`Carry2`]. This is the `min_hold == 1` form;
/// use [`cooldown_scan`] for a longer minimum hold.
///
/// `chunk_rows` only controls how many rows are fed at a time. A chunk that
/// ends inside a group continues the same carry, so the group carries and the
/// net are bit-identical for every positive chunk size.
pub fn band_scan(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
    chunk_rows: usize,
) -> Result<BandScan, String> {
    validate_scan(scores, returns, groups, params, side, chunk_rows)?;
    require_two_state(params)?;
    let positions = band_positions(scores, groups, params)?;
    let n = returns.len();
    let mut carries = Vec::new();
    let mut carry = Carry2::IDENTITY;
    let mut current: Option<u64> = None;
    let mut lo = 0usize;
    while lo < n {
        let hi = (lo + chunk_rows).min(n);
        for i in lo..hi {
            let group = groups[i];
            if current != Some(group) {
                if current.is_some() {
                    carries.push(carry);
                }
                carry = Carry2::IDENTITY;
                current = Some(group);
            }
            let final_exit = i + 1 == n || groups[i + 1] != group;
            let row = band_row_carry(params, scores[i], returns[i], side, final_exit);
            carry = carry.then(&row);
        }
        lo = hi;
    }
    if current.is_some() {
        carries.push(carry);
    }
    Ok(BandScan { positions, carries })
}

/// Minimum-holding/cooldown scan with the K+1-state [`CooldownCarry`].
///
/// The counter includes the entry row, so `min_hold == 1` is the plain band and
/// `min_hold == 2` forbids an exit on the row after entry. Chunk invariance has
/// the same contract as [`band_scan`].
pub fn cooldown_scan(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
    chunk_rows: usize,
) -> Result<CooldownScan, String> {
    validate_scan(scores, returns, groups, params, side, chunk_rows)?;
    let positions = band_positions(scores, groups, params)?;
    let n = returns.len();
    let states = params.min_hold + 1;
    let mut carries = Vec::new();
    let mut carry = CooldownCarry::identity(states);
    let mut current: Option<u64> = None;
    let mut lo = 0usize;
    while lo < n {
        let hi = (lo + chunk_rows).min(n);
        for i in lo..hi {
            let group = groups[i];
            if current != Some(group) {
                if current.is_some() {
                    carries.push(carry);
                }
                carry = CooldownCarry::identity(states);
                current = Some(group);
            }
            let final_exit = i + 1 == n || groups[i + 1] != group;
            let row = cooldown_row_carry(params, scores[i], returns[i], side, final_exit);
            carry = carry.then(&row);
        }
        lo = hi;
    }
    if current.is_some() {
        carries.push(carry);
    }
    Ok(CooldownScan { positions, carries })
}

/// The exact `O(2n)` hindsight DP as a 2x2 max-plus carry.
///
/// `M_t = [[0, -s], [r_t - s, r_t]]` (destination rows, source columns) and the
/// carry is the ordered max-plus product of the row matrices. The identity is
/// `[[0, -inf], [-inf, 0]]`. Only [`DpScan::bound`] is the diagnostic best
/// hindsight path; it is never a reported strategy P&L.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DpCarry {
    /// `m[destination][source]`.
    pub m: [[f64; 2]; 2],
}

impl DpCarry {
    pub const IDENTITY: DpCarry = DpCarry {
        m: [[0.0, f64::NEG_INFINITY], [f64::NEG_INFINITY, 0.0]],
    };

    /// The row matrix for one return and per-side cost.
    pub fn row(ret: f64, side: f64) -> DpCarry {
        DpCarry {
            m: [[0.0, -side], [ret - side, ret]],
        }
    }

    /// Max-plus product: `self` is applied first, then `right`.
    pub fn then(&self, right: &DpCarry) -> DpCarry {
        let mut m = [[f64::NEG_INFINITY; 2]; 2];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let mut best = f64::NEG_INFINITY;
                for k in 0..2 {
                    let candidate = right.m[i][k] + self.m[k][j];
                    // Fixed k order: the first maximum wins, so ties have a
                    // deterministic representative (no NaN can reach here).
                    if candidate > best {
                        best = candidate;
                    }
                }
                *slot = best;
            }
        }
        DpCarry { m }
    }

    /// Best value from a flat start, charging one final exit if the end state is
    /// long. `max(V_T(0), V_T(1) - s)`.
    pub fn bound(&self, side: f64) -> f64 {
        let flat = self.m[0][0];
        let long = self.m[1][0] - side;
        if long > flat {
            long
        } else {
            flat
        }
    }

    pub fn bits(&self) -> [u64; 4] {
        [
            self.m[0][0].to_bits(),
            self.m[0][1].to_bits(),
            self.m[1][0].to_bits(),
            self.m[1][1].to_bits(),
        ]
    }
}

/// One composed DP carry per group and the additive hindsight bound.
#[derive(Clone, Debug, PartialEq)]
pub struct DpScan {
    pub carries: Vec<DpCarry>,
}

impl DpScan {
    pub fn bound(&self, side: f64) -> f64 {
        let mut total = 0.0f64;
        for carry in &self.carries {
            total += carry.bound(side);
        }
        total
    }
}

/// Exact hindsight DP scan with group seams reset and a max-plus carry. Chunk
/// size does not change the carry bits or the bound.
pub fn dp_scan(
    returns: &[f64],
    groups: &[u64],
    side: f64,
    chunk_rows: usize,
) -> Result<DpScan, String> {
    validate_dp(returns, groups, side, chunk_rows)?;
    let n = returns.len();
    let mut carries = Vec::new();
    let mut carry = DpCarry::IDENTITY;
    let mut current: Option<u64> = None;
    let mut lo = 0usize;
    while lo < n {
        let hi = (lo + chunk_rows).min(n);
        for i in lo..hi {
            let group = groups[i];
            if current != Some(group) {
                if current.is_some() {
                    carries.push(carry);
                }
                carry = DpCarry::IDENTITY;
                current = Some(group);
            }
            carry = carry.then(&DpCarry::row(returns[i], side));
        }
        lo = hi;
    }
    if current.is_some() {
        carries.push(carry);
    }
    Ok(DpScan { carries })
}

/// Convenience scalar hindsight bound.
pub fn dp_bound(returns: &[f64], groups: &[u64], side: f64) -> Result<f64, String> {
    Ok(dp_scan(returns, groups, side, returns.len().max(1))?.bound(side))
}

/// Merge 32 lane partials with the fixed pairwise tree used by the device.
pub fn merge_lanes(lanes: &[PnlTotals; PNL_LANES]) -> PnlTotals {
    let mut buf = *lanes;
    let mut width = PNL_LANES;
    while width > 1 {
        let half = width / 2;
        for i in 0..half {
            let right = buf[i + half];
            buf[i].merge(&right);
        }
        width = half;
    }
    buf[0]
}

/// The cash benchmark: flat every row.
pub fn cash_totals(returns: &[f64], cash_return: f64) -> PnlTotals {
    let rows = returns.len() as u64;
    let net = cash_return * rows as f64;
    PnlTotals {
        rows,
        gross: net,
        net,
        ..PnlTotals::default()
    }
}

/// The buy-and-hold benchmark: long every row, one entry and one charged exit
/// per group, so `net = sum(r) - 2s * groups`.
pub fn buy_and_hold_totals(returns: &[f64], groups: &[u64], side: f64) -> PnlTotals {
    let mut gross = 0.0f64;
    let mut hits = 0u64;
    let mut current: Option<u64> = None;
    let mut entries = 0u64;
    for (i, (&r, &g)) in returns.iter().zip(groups).enumerate() {
        if !r.is_finite() {
            // The caller's valid rows are finite; keep the benchmark parallel
            // with the strategy by skipping a non-finite return here too.
            let _ = i;
            continue;
        }
        gross += r;
        if r > 0.0 {
            hits += 1;
        }
        if current != Some(g) {
            entries += 1;
            current = Some(g);
        }
    }
    let groups = entries;
    let rows = returns.len() as u64;
    let cost = 2.0 * side * groups as f64;
    PnlTotals {
        rows,
        long_rows: rows,
        entries: groups,
        exits: groups,
        trades: groups,
        hits,
        final_exits: groups,
        gross,
        cost,
        turnover: groups as f64,
        net: gross - cost,
    }
}

/// The two benchmarks required by the epic.
pub fn benchmarks(
    returns: &[f64],
    groups: &[u64],
    side: f64,
    cash_return: f64,
) -> (PnlTotals, PnlTotals) {
    (
        cash_totals(returns, cash_return),
        buy_and_hold_totals(returns, groups, side),
    )
}

/// One row's terms under the default rule. `i` is a global row index; `groups`
/// must be group-contiguous.
fn row_term(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
    i: usize,
) -> Result<RowTerm, String> {
    let r = returns[i];
    if !r.is_finite() {
        return Err(format!("row {}: non-finite return {}", i, r));
    }
    let p = position_for(scores[i], params.threshold);
    let prev = if i > 0 && groups[i] == groups[i - 1] {
        position_for(scores[i - 1], params.threshold)
    } else {
        0.0
    };
    let d = p - prev;
    let last = i + 1 == returns.len() || groups[i + 1] != groups[i];
    let final_exit = last && p != 0.0;
    let side = params.side();
    let mut cost = side * d.abs();
    if final_exit {
        cost += side * p.abs();
    }
    let gross = p * r;
    Ok(RowTerm {
        position: p,
        order: d,
        gross,
        cost,
        net: gross - cost,
        final_exit,
    })
}

/// Explicit per-row returns, positions, orders and costs for one candidate.
///
/// This is the fixture/debug observation path, not the batch reduction: it is
/// `O(N)` for one candidate and never allocates `P x N`.
pub fn row_terms(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Result<Vec<RowTerm>, String> {
    validate_inputs(scores, returns, groups, params)?;
    let mut out = Vec::with_capacity(returns.len());
    for i in 0..returns.len() {
        out.push(row_term(scores, returns, groups, params, i)?);
    }
    check_long_only_terms(&out)?;
    Ok(out)
}

fn check_long_only_terms(terms: &[RowTerm]) -> Result<(), String> {
    let positions: Vec<f64> = terms.iter().map(|t| t.position).collect();
    check_long_only(&positions)
}

fn validate_inputs(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Result<(), String> {
    if scores.len() != returns.len() || groups.len() != returns.len() {
        return Err(format!(
            "length mismatch: scores {}, returns {}, groups {}",
            scores.len(),
            returns.len(),
            groups.len()
        ));
    }
    if params.segment_rows == 0 {
        return Err("segment_rows must be at least 1".to_string());
    }
    if !params.per_side_bps.is_finite() || params.per_side_bps < 0.0 {
        return Err(format!(
            "per_side_bps must be finite and non-negative, got {}",
            params.per_side_bps
        ));
    }
    check_group_contiguous(groups)
}

/// A streaming canonical reducer.
///
/// `feed(lo, hi)` may be called with any contiguous ranges in ascending order.
/// Rows are added to their fixed canonical segment; a segment split across two
/// chunks continues the same `f64` accumulator, so the final bits do not depend
/// on the chunk boundaries.
pub struct PnlReducer<'a> {
    scores: &'a [f64],
    returns: &'a [f64],
    groups: &'a [u64],
    params: PnlParams,
    partials: Vec<[PnlTotals; PNL_LANES]>,
    next_row: usize,
}

impl<'a> PnlReducer<'a> {
    pub fn new(
        scores: &'a [f64],
        returns: &'a [f64],
        groups: &'a [u64],
        params: &PnlParams,
    ) -> Result<Self, String> {
        validate_inputs(scores, returns, groups, params)?;
        let segments = returns.len().div_ceil(params.segment_rows);
        Ok(PnlReducer {
            scores,
            returns,
            groups,
            params: *params,
            partials: vec![[PnlTotals::default(); PNL_LANES]; segments],
            next_row: 0,
        })
    }

    /// Add rows `[lo, hi)`. Ranges must be contiguous and ascending.
    pub fn feed(&mut self, lo: usize, hi: usize) -> Result<(), String> {
        if lo != self.next_row {
            return Err(format!(
                "chunk must start at row {} (contiguous ascending feeds), got {}",
                self.next_row, lo
            ));
        }
        if hi < lo || hi > self.returns.len() {
            return Err(format!("invalid chunk [{}, {}) for {} rows", lo, hi, self.returns.len()));
        }
        for i in lo..hi {
            let term = row_term(self.scores, self.returns, self.groups, &self.params, i)?;
            let seg = i / self.params.segment_rows;
            let lane = (i - seg * self.params.segment_rows) % PNL_LANES;
            let acc = &mut self.partials[seg][lane];
            acc.rows += 1;
            if term.position != 0.0 {
                acc.long_rows += 1;
                if self.returns[i] > 0.0 {
                    acc.hits += 1;
                }
            }
            if term.order > 0.0 {
                acc.entries += 1;
            } else if term.order < 0.0 {
                acc.exits += 1;
            }
            if term.order != 0.0 {
                acc.trades += 1;
            }
            if term.final_exit {
                acc.final_exits += 1;
                acc.exits += 1;
            }
            acc.gross += term.gross;
            acc.cost += term.cost;
            acc.turnover += term.order.abs();
            acc.net += term.net;
        }
        self.next_row = hi;
        Ok(())
    }

    /// Merge the canonical segments in ascending order and attach benchmarks.
    pub fn finish(self) -> Result<PnlReport, String> {
        if self.next_row != self.returns.len() {
            return Err(format!(
                "reducer stopped at row {} of {}",
                self.next_row,
                self.returns.len()
            ));
        }
        let mut strategy = PnlTotals::default();
        for partial in &self.partials {
            let segment = merge_lanes(partial);
            strategy.merge(&segment);
        }
        let (cash, buy_and_hold) = benchmarks(
            self.returns,
            self.groups,
            self.params.side(),
            self.params.cash_return,
        );
        Ok(PnlReport {
            strategy,
            cash,
            buy_and_hold,
        })
    }
}

/// Canonical CPU reference: all rows in one chunk.
pub fn reduce(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Result<PnlReport, String> {
    let mut reducer = PnlReducer::new(scores, returns, groups, params)?;
    reducer.feed(0, returns.len())?;
    reducer.finish()
}

/// Canonical CPU reference fed in `chunk_rows`-row chunks. The result must be
/// bit-identical to [`reduce`] for every positive chunk size.
pub fn reduce_chunked(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
    chunk_rows: usize,
) -> Result<PnlReport, String> {
    if chunk_rows == 0 {
        return Err("chunk_rows must be at least 1".to_string());
    }
    let mut reducer = PnlReducer::new(scores, returns, groups, params)?;
    let n = returns.len();
    let mut lo = 0usize;
    while lo < n {
        let hi = (lo + chunk_rows).min(n);
        reducer.feed(lo, hi)?;
        lo = hi;
    }
    reducer.finish()
}

// ---------------------------------------------------------------------------
// Device path
// ---------------------------------------------------------------------------

/// The device path is compiled only with `--features cuda`.
/// The device path is compiled only with `--features cuda`.
#[cfg(feature = "cuda")]
mod device {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    struct RawTotals {
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

    extern "C" {
        fn rc_pnl_create(
            returns: *const f64,
            groups: *const u64,
            n: i32,
            segment_rows: i32,
            tile_candidates: i32,
        ) -> *mut std::ffi::c_void;
        fn rc_pnl_destroy(ctx: *mut std::ffi::c_void);
        fn rc_pnl_tile_capacity(ctx: *mut std::ffi::c_void) -> i32;
        fn rc_pnl_segment_rows(ctx: *mut std::ffi::c_void) -> i32;
        fn rc_pnl_eval(
            ctx: *mut std::ffi::c_void,
            scores: *const f32,
            candidates: i32,
            thresholds: *const f64,
            side: f64,
            threads: i32,
            chunk_rows: i32,
            out: *mut RawTotals,
        ) -> i32;
    }

    /// A resident dataset plus a declared `tile_candidates x n` score tile.
    pub struct DevicePnl {
        ctx: *mut std::ffi::c_void,
        n: usize,
        tile: usize,
        segment_rows: usize,
    }

    impl DevicePnl {
        /// Upload returns and group ids once. Returns `None` when the device or
        /// an allocation is unavailable.
        pub fn new(returns: &[f64], groups: &[u64], segment_rows: usize) -> Option<Self> {
            if returns.len() != groups.len() || returns.is_empty() || segment_rows == 0 {
                return None;
            }
            let ctx = unsafe {
                rc_pnl_create(
                    returns.as_ptr(),
                    groups.as_ptr(),
                    returns.len() as i32,
                    segment_rows as i32,
                    MAX_TILE_CANDIDATES as i32,
                )
            };
            if ctx.is_null() {
                return None;
            }
            let tile = unsafe { rc_pnl_tile_capacity(ctx) };
            let seg = unsafe { rc_pnl_segment_rows(ctx) };
            if tile <= 0 || seg <= 0 {
                unsafe { rc_pnl_destroy(ctx) };
                return None;
            }
            Some(DevicePnl {
                ctx,
                n: returns.len(),
                tile: tile as usize,
                segment_rows: seg as usize,
            })
        }

        pub fn n_rows(&self) -> usize {
            self.n
        }

        /// The declared tile width; this is not the population size.
        pub fn tile_candidates(&self) -> usize {
            self.tile
        }

        pub fn segment_rows(&self) -> usize {
            self.segment_rows
        }

        /// Evaluate up to `tile_candidates` candidates. `scores` is candidate
        /// major (`candidate * n + row`), one `f32` score per row.
        pub fn eval(
            &mut self,
            scores: &[f32],
            thresholds: &[f64],
            per_side_bps: f64,
            threads: usize,
            chunk_rows: usize,
        ) -> Option<Vec<PnlTotals>> {
            let candidates = thresholds.len();
            if candidates == 0 || candidates > self.tile {
                return None;
            }
            if scores.len() != candidates.checked_mul(self.n)? {
                return None;
            }
            if threads == 0 || chunk_rows == 0 {
                return None;
            }
            let mut raw = vec![RawTotals::default(); candidates];
            let rc = unsafe {
                rc_pnl_eval(
                    self.ctx,
                    scores.as_ptr(),
                    candidates as i32,
                    thresholds.as_ptr(),
                    per_side_bps / 10_000.0,
                    threads as i32,
                    chunk_rows as i32,
                    raw.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return None;
            }
            let mut out = Vec::with_capacity(candidates);
            for (i, r) in raw.iter().enumerate() {
                if r.status != 0 {
                    // A non-finite return was seen; the CPU path is the
                    // labelled fallback, so do not return a partial result.
                    let _ = i;
                    return None;
                }
                out.push(PnlTotals {
                    rows: r.rows,
                    long_rows: r.long_rows,
                    entries: r.entries,
                    exits: r.exits,
                    trades: r.trades,
                    hits: r.hits,
                    final_exits: r.final_exits,
                    gross: r.gross,
                    cost: r.cost,
                    turnover: r.turnover,
                    net: r.net,
                });
            }
            Some(out)
        }
    }

    impl Drop for DevicePnl {
        fn drop(&mut self) {
            unsafe { rc_pnl_destroy(self.ctx) };
        }
    }

    fn report_for(
        totals: PnlTotals,
        returns: &[f64],
        groups: &[u64],
        params: &PnlParams,
    ) -> PnlReport {
        let (cash, buy_and_hold) = benchmarks(
            returns,
            groups,
            params.side(),
            params.cash_return,
        );
        PnlReport {
            strategy: totals,
            cash,
            buy_and_hold,
        }
    }

    /// Device batch reduction. A population larger than the declared tile is
    /// streamed through it; it is never materialised as a `P x N` tensor.
    pub fn reduce_batch(
        scores: &[&[f64]],
        returns: &[f64],
        groups: &[u64],
        thresholds: &[f64],
        params: &PnlParams,
        threads: usize,
        chunk_rows: usize,
    ) -> Option<Vec<PnlReport>> {
        if scores.len() != thresholds.len() {
            return None;
        }
        if scores.is_empty() {
            return Some(Vec::new());
        }
        for s in scores {
            if s.len() != returns.len() {
                return None;
            }
        }
        if params.segment_rows == 0 || returns.is_empty() {
            return None;
        }
        let mut dev = DevicePnl::new(returns, groups, params.segment_rows)?;
        let tile = dev.tile_candidates();
        let mut out = Vec::with_capacity(scores.len());
        let mut base = 0usize;
        while base < scores.len() {
            let count = tile.min(scores.len() - base);
            let mut tile_scores = Vec::with_capacity(count * returns.len());
            for s in &scores[base..base + count] {
                tile_scores.extend(s.iter().map(|&v| v as f32));
            }
            let totals = dev.eval(
                &tile_scores,
                &thresholds[base..base + count],
                params.per_side_bps,
                threads,
                chunk_rows,
            )?;
            for t in totals {
                out.push(report_for(t, returns, groups, params));
            }
            base += count;
        }
        Some(out)
    }

    /// Convenience single-candidate device reduction.
    pub fn reduce_device(
        scores: &[f64],
        returns: &[f64],
        groups: &[u64],
        params: &PnlParams,
        threads: usize,
        chunk_rows: usize,
    ) -> Option<PnlReport> {
        reduce_batch(
            &[scores],
            returns,
            groups,
            &[params.threshold],
            params,
            threads,
            chunk_rows,
        )
        .map(|mut v| v.remove(0))
    }

    /// True when this build can run the device kernel on a real device.
    pub fn available() -> bool {
        crate::gpu::probe().available()
    }
}

// ---------------------------------------------------------------------------
// CUDA parity path
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod threshold_device {
    use super::Type7Quantile;

    extern "C" {
        /// Exact order-statistic selection on the device. Returns `0` on
        /// success, a CUDA error code otherwise. `out_lo`/`out_hi` are the
        /// values at the ranks bracketing `h`, already in `f64`.
        fn rc_threshold_select_f32(
            scores: *const f32,
            n: i64,
            q: f64,
            out_finite: *mut i64,
            out_lo: *mut f64,
            out_hi: *mut f64,
        ) -> i32;
        /// Same selection with an explicit device block size (128/256/512),
        /// used by the thread-invariance test.
        fn rc_threshold_select_f32_threads(
            scores: *const f32,
            n: i64,
            q: f64,
            threads: i32,
            out_finite: *mut i64,
            out_lo: *mut f64,
            out_hi: *mut f64,
        ) -> i32;
        /// Packed valid-row mask: bit `i` is set iff score and target are both
        /// finite. `words` must hold `ceil(n / 64)` `u64`.
        fn rc_valid_mask_f32(
            scores: *const f32,
            targets: *const f32,
            n: i64,
            words: *mut u64,
        ) -> i32;
    }

    /// Device threshold when CUDA is compiled in and a device is present.
    ///
    /// `None` means "the device path is not available here"; the CPU exact
    /// result remains the reported authority either way.
    pub fn type7_quantile_f32_device(scores: &[f32], q: f64) -> Option<Type7Quantile> {
        if !crate::gpu::probe().available() {
            return None;
        }
        let n = i64::try_from(scores.len()).ok()?;
        let mut finite = 0i64;
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        let rc = unsafe {
            rc_threshold_select_f32(scores.as_ptr(), n, q, &mut finite, &mut lo, &mut hi)
        };
        finish_device(q, rc, finite, lo, hi)
    }

    fn device_with_threads(scores: &[f32], q: f64, threads: i32) -> Option<Type7Quantile> {
        if !crate::gpu::probe().available() {
            return None;
        }
        let n = i64::try_from(scores.len()).ok()?;
        let mut finite = 0i64;
        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        let rc = unsafe {
            rc_threshold_select_f32_threads(
                scores.as_ptr(),
                n,
                q,
                threads,
                &mut finite,
                &mut lo,
                &mut hi,
            )
        };
        finish_device(q, rc, finite, lo, hi)
    }

    fn finish_device(q: f64, rc: i32, finite: i64, lo: f64, hi: f64) -> Option<Type7Quantile> {
        if rc != 0 || finite < 0 {
            return None;
        }
        let n = finite as usize;
        if n == 0 {
            return Some(Type7Quantile::empty(q));
        }
        if n == 1 {
            return Some(Type7Quantile::finish(1, q, 0, 0, lo, lo));
        }
        let (_, rank_lo, rank_hi) = Type7Quantile::ranks(n, q);
        Some(Type7Quantile::finish(n, q, rank_lo, rank_hi, lo, hi))
    }

    /// Device selection with an explicit block size.
    pub fn type7_quantile_f32_device_threads(
        scores: &[f32],
        q: f64,
        threads: i32,
    ) -> Option<Type7Quantile> {
        device_with_threads(scores, q, threads)
    }

    /// Device packed mask, for the parity test. The words are in the same
    /// layout as [`super::ValidMask::words`].
    pub fn valid_mask_f32_device(scores: &[f32], targets: &[f32]) -> Option<Vec<u64>> {
        if !crate::gpu::probe().available() || scores.len() != targets.len() {
            return None;
        }
        let n = i64::try_from(scores.len()).ok()?;
        let mut words = vec![0u64; (scores.len() + 63) / 64];
        let rc = unsafe {
            rc_valid_mask_f32(scores.as_ptr(), targets.as_ptr(), n, words.as_mut_ptr())
        };
        if rc != 0 {
            return None;
        }
        Some(words)
    }

}
/// Device exact type-7 quantile, or `None` when CUDA is absent. The CPU
/// result is the report authority; this proves the device agrees bit for bit.
#[cfg(feature = "cuda")]
pub fn type7_quantile_f32_device(scores: &[f32], q: f64) -> Option<Type7Quantile> {
    threshold_device::type7_quantile_f32_device(scores, q)
}

/// Device exact type-7 quantile with an explicit block size, for the
/// thread-invariance contract.
#[cfg(feature = "cuda")]
pub fn type7_quantile_f32_device_threads(
    scores: &[f32],
    q: f64,
    threads: i32,
) -> Option<Type7Quantile> {
    threshold_device::type7_quantile_f32_device_threads(scores, q, threads)
}

/// Device packed valid-row mask, or `None` when CUDA is absent.
#[cfg(feature = "cuda")]
pub fn valid_mask_f32_device(scores: &[f32], targets: &[f32]) -> Option<Vec<u64>> {
    threshold_device::valid_mask_f32_device(scores, targets)
}

#[cfg(feature = "cuda")]
pub use device::{available as device_available, reduce_batch as reduce_device_batch, DevicePnl};
#[cfg(feature = "cuda")]
pub use device::reduce_device;

/// CPU-only builds have no device path.
#[cfg(not(feature = "cuda"))]
pub fn device_available() -> bool {
    false
}

// ---------------------------------------------------------------------------
// #306: device band/cooldown and hindsight DP scans
// ---------------------------------------------------------------------------
//
// The CPU scans fold each chain left to right: one chain per (candidate, group,
// start state) for the band, one per (group, source state) for the DP. The
// reward and DP sums are f64 additions, which are not associative, so the
// device does not reassociate them: one device thread folds one chain in
// ascending row order. Block size, grid size and chunk size only change which
// thread owns a chain and when it resumes, never the order of a chain's sums.
// The cross-group totals ([`BandScan::net`], [`CooldownScan::net`],
// [`DpScan::bound`]) are computed on the host by the same code as the CPU path.
//
// Every device entry point returns either the exact CPU bits or `Err(reason)`;
// the caller then runs the CPU reference and reports the reason.

/// Default device block size for the band/DP scans. Every size from 1 to
/// [`band_dp_max_threads`] produces the same bits.
pub const SCAN_THREADS: usize = 256;

/// Validate a band batch exactly like the CPU scans and return its common
/// `min_hold`.
fn validate_band_batch(
    scores: &[&[f64]],
    returns: &[f64],
    groups: &[u64],
    params: &[BandParams],
    side: f64,
    chunk_rows: usize,
) -> Result<usize, String> {
    if scores.len() != params.len() {
        return Err(format!(
            "length mismatch: {} score vectors, {} parameter sets",
            scores.len(),
            params.len()
        ));
    }
    let Some(first) = params.first() else {
        return Err("no candidates to scan".to_string());
    };
    for (s, p) in scores.iter().zip(params) {
        validate_scan(s, returns, groups, p, side, chunk_rows)?;
        if p.min_hold != first.min_hold {
            return Err(format!(
                "one device batch needs one min_hold; got {} and {}",
                first.min_hold, p.min_hold
            ));
        }
    }
    Ok(first.min_hold)
}

#[cfg(feature = "cuda")]
mod scan_device {
    use super::*;

    extern "C" {
        fn rc_band_dp_max_threads() -> i32;
        fn rc_band_scan(
            scores: *const f64,
            returns: *const f64,
            offsets: *const i64,
            groups: i32,
            n: i64,
            candidates: i32,
            thresholds: *const f64,
            bands: *const f64,
            states: i32,
            side: f64,
            threads: i32,
            chunk_rows: i64,
            positions: *mut u8,
            end_states: *mut i32,
            rewards: *mut f64,
        ) -> i32;
        fn rc_dp_scan(
            returns: *const f64,
            offsets: *const i64,
            groups: i32,
            n: i64,
            side: f64,
            threads: i32,
            chunk_rows: i64,
            values: *mut f64,
        ) -> i32;
    }

    fn status_reason(code: i32) -> String {
        let what = match code {
            1 => "the device scan rejected its arguments",
            2 => "CUDA device attributes unavailable",
            3 => "device allocation refused",
            4 => "host-to-device copy failed",
            5 => "kernel launch failed",
            6 => "kernel execution failed",
            7 => "device-to-host copy failed",
            _ => "unknown device scan status",
        };
        format!("{} (status {})", what, code)
    }

    /// Start offsets of each group run plus the row count: `[0, ..., n]`.
    fn group_offsets(groups: &[u64]) -> Vec<i64> {
        let mut offsets = Vec::new();
        for (i, g) in groups.iter().enumerate() {
            if i == 0 || groups[i - 1] != *g {
                offsets.push(i as i64);
            }
        }
        offsets.push(groups.len() as i64);
        offsets
    }

    pub fn max_threads() -> Option<usize> {
        if !crate::backend::probe().available() {
            return None;
        }
        let max = unsafe { rc_band_dp_max_threads() };
        if max > 0 {
            Some(max as usize)
        } else {
            None
        }
    }

    /// Device, thread count and size checks shared by both scans. Returns
    /// `(groups, n, threads, chunk_rows)` as the C types.
    fn launch_shape(
        offsets: &[i64],
        n: usize,
        threads: usize,
        chunk_rows: usize,
    ) -> Result<(i32, i64, i32, i64), String> {
        if n == 0 {
            return Err("no rows to scan".to_string());
        }
        let probe = crate::backend::probe();
        if !probe.available() {
            return Err(probe
                .reason
                .unwrap_or_else(|| "no CUDA device".to_string()));
        }
        let max = unsafe { rc_band_dp_max_threads() };
        if max <= 0 {
            return Err("CUDA device attributes unavailable".to_string());
        }
        let max = max as usize;
        if threads == 0 || threads > max {
            return Err(format!("threads {} outside 1..={}", threads, max));
        }
        let groups = i32::try_from(offsets.len() - 1)
            .map_err(|_| format!("{} groups exceed the device scan's i32 index", offsets.len() - 1))?;
        let n = i64::try_from(n).map_err(|_| "row count exceeds i64".to_string())?;
        let chunk = i64::try_from(chunk_rows).unwrap_or(i64::MAX);
        Ok((groups, n, threads as i32, chunk))
    }

    /// One composed K+1-state carry per group and the start-flat positions, per
    /// candidate. Candidates stream through the declared tile.
    pub fn band_chains(
        scores: &[&[f64]],
        returns: &[f64],
        groups: &[u64],
        params: &[BandParams],
        side: f64,
        threads: usize,
        chunk_rows: usize,
    ) -> Result<Vec<(Vec<f64>, Vec<CooldownCarry>)>, String> {
        let min_hold = validate_band_batch(scores, returns, groups, params, side, chunk_rows)?;
        let offsets = group_offsets(groups);
        let (g32, n64, t32, chunk) = launch_shape(&offsets, returns.len(), threads, chunk_rows)?;
        let states = min_hold
            .checked_add(1)
            .and_then(|s| i32::try_from(s).ok())
            .ok_or_else(|| format!("min_hold {} exceeds the device scan's i32 state", min_hold))?;
        let n = returns.len();
        let ngroups = offsets.len() - 1;
        let s = states as usize;
        let per_candidate = ngroups
            .checked_mul(s)
            .ok_or("group x state count overflows usize")?;
        let mut out = Vec::with_capacity(scores.len());
        for (tile_scores, tile_params) in scores
            .chunks(MAX_TILE_CANDIDATES)
            .zip(params.chunks(MAX_TILE_CANDIDATES))
        {
            let count = tile_scores.len();
            let mut flat = Vec::with_capacity(count * n);
            for v in tile_scores {
                flat.extend_from_slice(v);
            }
            let thresholds: Vec<f64> = tile_params.iter().map(|p| p.threshold).collect();
            let bands: Vec<f64> = tile_params.iter().map(|p| p.band).collect();
            let mut positions = vec![0u8; count * n];
            let mut end_states = vec![0i32; count * per_candidate];
            let mut rewards = vec![0.0f64; count * per_candidate];
            let rc = unsafe {
                rc_band_scan(
                    flat.as_ptr(),
                    returns.as_ptr(),
                    offsets.as_ptr(),
                    g32,
                    n64,
                    count as i32,
                    thresholds.as_ptr(),
                    bands.as_ptr(),
                    states,
                    side,
                    t32,
                    chunk,
                    positions.as_mut_ptr(),
                    end_states.as_mut_ptr(),
                    rewards.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return Err(status_reason(rc));
            }
            for c in 0..count {
                let pos: Vec<f64> = positions[c * n..(c + 1) * n]
                    .iter()
                    .map(|&p| if p != 0 { 1.0 } else { 0.0 })
                    .collect();
                let mut carries = Vec::with_capacity(ngroups);
                for g in 0..ngroups {
                    let base = c * per_candidate + g * s;
                    let mut end = Vec::with_capacity(s);
                    for &e in &end_states[base..base + s] {
                        let e = usize::try_from(e)
                            .ok()
                            .filter(|&e| e < s)
                            .ok_or_else(|| format!("device returned end state {} of {}", e, s))?;
                        end.push(e);
                    }
                    carries.push(CooldownCarry {
                        end,
                        reward: rewards[base..base + s].to_vec(),
                    });
                }
                out.push((pos, carries));
            }
        }
        Ok(out)
    }

    pub fn dp(
        returns: &[f64],
        groups: &[u64],
        side: f64,
        threads: usize,
        chunk_rows: usize,
    ) -> Result<DpScan, String> {
        validate_dp(returns, groups, side, chunk_rows)?;
        let offsets = group_offsets(groups);
        let (g32, n64, t32, chunk) = launch_shape(&offsets, returns.len(), threads, chunk_rows)?;
        let ngroups = offsets.len() - 1;
        let mut values = vec![0.0f64; ngroups * 4];
        let rc = unsafe {
            rc_dp_scan(
                returns.as_ptr(),
                offsets.as_ptr(),
                g32,
                n64,
                side,
                t32,
                chunk,
                values.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(status_reason(rc));
        }
        // values[(2g + j) * 2 + i] is m[i][j] of group g.
        let carries = values
            .chunks_exact(4)
            .map(|v| DpCarry {
                m: [[v[0], v[2]], [v[1], v[3]]],
            })
            .collect();
        Ok(DpScan { carries })
    }
}

/// Largest device block size the band/DP scans accept, or `None` without a
/// device.
pub fn band_dp_max_threads() -> Option<usize> {
    #[cfg(feature = "cuda")]
    {
        scan_device::max_threads()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

#[cfg(feature = "cuda")]
fn band_chains_device(
    scores: &[&[f64]],
    returns: &[f64],
    groups: &[u64],
    params: &[BandParams],
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<Vec<(Vec<f64>, Vec<CooldownCarry>)>, String> {
    scan_device::band_chains(scores, returns, groups, params, side, threads, chunk_rows)
}

#[cfg(not(feature = "cuda"))]
fn band_chains_device(
    scores: &[&[f64]],
    returns: &[f64],
    groups: &[u64],
    params: &[BandParams],
    side: f64,
    _threads: usize,
    chunk_rows: usize,
) -> Result<Vec<(Vec<f64>, Vec<CooldownCarry>)>, String> {
    validate_band_batch(scores, returns, groups, params, side, chunk_rows)?;
    Err(crate::backend::probe()
        .reason
        .unwrap_or_else(|| "built without --features cuda".to_string()))
}

/// Device [`band_scan`] for a batch of candidates (`min_hold == 1`). Each
/// result is bit-identical to the CPU scan of that candidate alone, or the
/// call returns `Err(reason)`.
pub fn band_scan_device_batch(
    scores: &[&[f64]],
    returns: &[f64],
    groups: &[u64],
    params: &[BandParams],
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<Vec<BandScan>, String> {
    for p in params {
        require_two_state(p)?;
    }
    let chains = band_chains_device(scores, returns, groups, params, side, threads, chunk_rows)?;
    Ok(chains
        .into_iter()
        .map(|(positions, carries)| BandScan {
            positions,
            carries: carries
                .into_iter()
                .map(|c| Carry2::from_row([c.end[0], c.end[1]], [c.reward[0], c.reward[1]]))
                .collect(),
        })
        .collect())
}

/// Device [`cooldown_scan`] for a batch of candidates sharing one `min_hold`.
pub fn cooldown_scan_device_batch(
    scores: &[&[f64]],
    returns: &[f64],
    groups: &[u64],
    params: &[BandParams],
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<Vec<CooldownScan>, String> {
    let chains = band_chains_device(scores, returns, groups, params, side, threads, chunk_rows)?;
    Ok(chains
        .into_iter()
        .map(|(positions, carries)| CooldownScan { positions, carries })
        .collect())
}

/// Device [`band_scan`] for one candidate.
pub fn band_scan_device(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<BandScan, String> {
    band_scan_device_batch(&[scores], returns, groups, &[*params], side, threads, chunk_rows)
        .map(|mut v| v.remove(0))
}

/// Device [`cooldown_scan`] for one candidate.
pub fn cooldown_scan_device(
    scores: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &BandParams,
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<CooldownScan, String> {
    cooldown_scan_device_batch(&[scores], returns, groups, &[*params], side, threads, chunk_rows)
        .map(|mut v| v.remove(0))
}

/// Device [`dp_scan`]: bit-identical carries and bound, or `Err(reason)`.
pub fn dp_scan_device(
    returns: &[f64],
    groups: &[u64],
    side: f64,
    threads: usize,
    chunk_rows: usize,
) -> Result<DpScan, String> {
    #[cfg(feature = "cuda")]
    {
        scan_device::dp(returns, groups, side, threads, chunk_rows)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = threads;
        validate_dp(returns, groups, side, chunk_rows)?;
        Err(crate::backend::probe()
            .reason
            .unwrap_or_else(|| "built without --features cuda".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        (
            vec![0.5, -0.1, 0.2, -0.3, 0.4],
            vec![0.01, -0.02, 0.03, 0.0, 0.015],
            vec![0, 0, 0, 0, 0],
        )
    }

    #[test]
    fn fixture_matches_the_hand_computation() {
        let (scores, returns, groups) = fixture();
        let params = PnlParams {
            threshold: 0.0,
            per_side_bps: 10.0,
            ..PnlParams::default()
        };
        let r = reduce(&scores, &returns, &groups, &params).unwrap();
        assert!((r.strategy.net - 0.049).abs() <= 1e-12);
        assert!((r.strategy.gross - 0.055).abs() <= 1e-12);
        assert!((r.strategy.cost - 0.006).abs() <= 1e-12);
        assert_eq!(r.strategy.rows, 5);
        assert_eq!(r.strategy.trades, 5);
        assert_eq!(r.strategy.entries, 3);
        assert_eq!(r.strategy.exits, 3);
        assert_eq!(r.strategy.final_exits, 1);
        assert_eq!(r.strategy.long_rows, 3);
        assert!((r.strategy.exposure() - 3.0 / 5.0).abs() <= 1e-15);
    }

    #[test]
    fn row_terms_are_the_fixture_arithmetic() {
        let (scores, returns, groups) = fixture();
        let params = PnlParams {
            threshold: 0.0,
            per_side_bps: 10.0,
            ..PnlParams::default()
        };
        let terms = row_terms(&scores, &returns, &groups, &params).unwrap();
        let order: Vec<f64> = terms.iter().map(|t| t.order).collect();
        assert_eq!(order, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
        let net: Vec<f64> = terms.iter().map(|t| t.net).collect();
        let expect = [0.009, -0.001, 0.029, -0.001, 0.013];
        for (a, b) in net.iter().zip(expect) {
            assert!((a - b).abs() <= 1e-12, "{} vs {}", a, b);
        }
    }

    #[test]
    fn chunked_reduction_is_bit_identical() {
        let n = 33_000usize;
        let mut scores = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        let mut groups = Vec::with_capacity(n);
        for i in 0..n {
            scores.push(((i * 37) % 101) as f64 / 50.0 - 1.0);
            returns.push(((i * 13) % 29) as f64 / 1000.0 - 0.01);
            groups.push((i / 7000) as u64);
        }
        let params = PnlParams::default();
        let base = reduce(&scores, &returns, &groups, &params).unwrap();
        for chunk in [1usize, 4096, 65536] {
            let got = reduce_chunked(&scores, &returns, &groups, &params, chunk).unwrap();
            assert_eq!(base.strategy.bits(), got.strategy.bits(), "chunk {}", chunk);
        }
    }

    #[test]
    fn negative_positions_are_rejected() {
        assert!(check_long_only(&[0.0, 1.0, 1.0]).is_ok());
        let err = check_long_only(&[0.0, -1.0]).unwrap_err();
        assert!(err.contains("short positions are not implemented"));
    }

    #[test]
    fn round_trip_costs_two_sides_and_flat_is_free() {
        let params = PnlParams {
            threshold: 0.0,
            per_side_bps: 10.0,
            ..PnlParams::default()
        };
        let r = reduce(&[1.0, -1.0], &[0.0, 0.0], &[0, 0], &params).unwrap();
        assert!((r.strategy.cost - 0.002).abs() <= 1e-15);
        let flat = reduce(&[-1.0, -1.0], &[0.0, 0.0], &[0, 0], &params).unwrap();
        assert_eq!(flat.strategy.cost, 0.0);
        assert_eq!(flat.strategy.trades, 0);
    }

    #[test]
    fn benchmarks_follow_the_report_definitions() {
        let returns = [0.01, -0.02, 0.03, 0.0, 0.015];
        let groups = [0u64, 0, 0, 1, 1];
        let (cash, bh) = benchmarks(&returns, &groups, 0.001, 0.0);
        assert_eq!(cash.net, 0.0);
        assert_eq!(cash.cost, 0.0);
        assert!((bh.net - (returns.iter().sum::<f64>() - 0.004)).abs() <= 1e-15);
        assert_eq!(bh.entries, 2);
        assert_eq!(bh.exits, 2);
    }
    #[test]
    fn order_keys_are_monotone_and_canonicalise_signed_zero() {
        let values = [
            f32::NEG_INFINITY,
            -1.0e30,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            1.0e30,
            f32::INFINITY,
        ];
        let mut last: Option<u32> = None;
        for &v in &values {
            if let Some(key) = order_key_f32(v) {
                assert_eq!(f32_from_order_key(key), if v == 0.0 { 0.0 } else { v });
                if let Some(prev) = last {
                    // `-0.0` and `+0.0` deliberately share one key; every
                    // other adjacent pair is strictly increasing.
                    assert!(prev <= key, "keys not monotone at {v}");
                }
                last = Some(key);
            }
        }
        assert_eq!(order_key_f32(-0.0), order_key_f32(0.0));
        assert_eq!(order_key_f32(f32::NAN), None);
        assert_eq!(order_key_f64(f64::NAN), None);
        assert_eq!(order_key_f64(-0.0), order_key_f64(0.0));
    }

    #[test]
    fn radix_select_returns_each_rank_of_a_duplicate_heavy_sample() {
        let keys = [5u32, 1, 5, 7, 0, 7, 7, 3];
        let mut sorted = keys;
        sorted.sort_unstable();
        for (rank, &want) in sorted.iter().enumerate() {
            assert_eq!(radix_select_u32(&keys, rank), want);
        }
        let keys64: Vec<u64> = keys.iter().map(|&k| k as u64).collect();
        for (rank, &want) in sorted.iter().enumerate() {
            assert_eq!(radix_select_u64(&keys64, rank), want as u64);
        }
    }

    #[test]
    fn exact_type7_matches_the_sorted_reference_including_signed_zero() {
        let mut scores = vec![
            0.0f32,
            -0.0,
            1.0,
            1.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.5,
        ];
        for i in 0..200 {
            scores.push((i as f32 * 0.37).sin());
        }
        for &q in &[0.0, 0.1, 0.25, 0.5, 0.7, 0.73, 0.999, 1.0] {
            let got = type7_quantile_f32(&scores, q);
            let want = type7_quantile_f32_sorted_reference(&scores, q);
            assert_eq!(got.theta.to_bits(), want.theta.to_bits(), "q={q}");
            assert_eq!(got.n, want.n);
        }
    }

    #[test]
    fn valid_mask_drops_both_score_and_target_non_finites() {
        let scores = [1.0, 2.0, f64::NAN, 4.0, 5.0];
        let targets = [0.1, f64::NAN, 0.3, 0.4, f64::INFINITY];
        let mask = ValidMask::from_finites(&scores, &targets);
        assert_eq!(mask.valid_count(), 2);
        assert_eq!(mask.dropped_count(), 3);
        assert_eq!(mask.indices().collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(ValidMask::from_scores(&scores).valid_count(), 4);
    }

    #[test]
    fn screening_threshold_is_refused_for_positions() {
        let mut mask = ValidMask::new(3);
        for i in 0..3 {
            mask.set(i, true);
        }
        let sketch = Threshold::screening_only(0.0);
        assert!(!sketch.is_report_eligible());
        assert!(sketch.require_report_eligible().is_err());
        assert!(long_positions(&[1.0, -1.0, 0.0], &mask, &sketch).is_err());
        let exact = Threshold::train_type7(&[1.0, -1.0, 0.0], 0.5);
        assert!(long_positions(&[1.0, -1.0, 0.0], &mask, &exact).is_ok());
    }

}
