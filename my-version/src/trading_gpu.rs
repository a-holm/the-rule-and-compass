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

/// The method that produced a reported threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdSource {
    /// Exact type-7 quantile of the training scores.
    TrainType7,
    /// Exact type-7 quantile of the validation scores.
    ValidationType7,
    /// A pre-registered constant threshold, never fitted to the data.
    Fixed,
}

impl ThresholdSource {
    pub fn as_str(self) -> &'static str {
        match self {
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

// ---------------------------------------------------------------------------
// CUDA parity path
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod device {
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

/// True when this build can run the device parity path.
pub fn device_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::probe().available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Device exact type-7 quantile, or `None` when CUDA is absent. The CPU result
/// is the report authority; this exists to prove the device agrees bit for bit.
#[cfg(feature = "cuda")]
pub fn type7_quantile_f32_device(scores: &[f32], q: f64) -> Option<Type7Quantile> {
    device::type7_quantile_f32_device(scores, q)
}

/// Device packed valid-row mask, or `None` when CUDA is absent.
/// Device exact type-7 quantile with an explicit block size, for the
/// thread-invariance contract.
#[cfg(feature = "cuda")]
pub fn type7_quantile_f32_device_threads(
    scores: &[f32],
    q: f64,
    threads: i32,
) -> Option<Type7Quantile> {
    device::type7_quantile_f32_device_threads(scores, q, threads)
}

#[cfg(feature = "cuda")]
pub fn valid_mask_f32_device(scores: &[f32], targets: &[f32]) -> Option<Vec<u64>> {
    device::valid_mask_f32_device(scores, targets)
}

// Keep the order relations honest in debug builds: the key transform must be
// monotone and the inverse must round-trip every finite bit pattern we use.
#[cfg(test)]
mod tests {
    use super::*;

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
