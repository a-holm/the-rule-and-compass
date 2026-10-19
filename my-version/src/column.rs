//! Resident column-major storage and row selectors (issue #70).
//!
//! A [`ColumnStore`] keeps **one contiguous `f64` block per column**. There is
//! no implicit row-major copy and, while the table is resident, no row-group
//! partitioning: grouping only bounds the working set for an out-of-core
//! backing (issue #71).
//!
//! The hot-path accessor is [`GroupView::col`]. [`ColumnStore::group_view`]
//! hands out slices straight into the resident buffers; a future file backing
//! implements the same shape by filling a [`GroupBuf`] in
//! [`ColumnStore::load_group`] with `pread`, so call sites never need to know
//! which backing is in use.
//!
//! [`RowSelector`] is a *view* over rows of the store: a contiguous range, a
//! pinned hash predicate, an explicit set, or every row. Splits, folds and
//! evaluation subsets are expressed as selectors so that the rows are not
//! copied. [`RowSelector::auto`] applies the measured policy from the issue:
//!
//! * contiguous rows are a `Range`;
//! * a **reused, resident** `Set` is used only up to ~7 % of the rows (with a
//!   cheap one-multiply hash) and an absolute cap of 1 000 000 rows;
//! * a one-shot `Set` never wins, because building it costs the scan it was
//!   meant to avoid;
//! * everything else is a `Hash` predicate.
//!
//! The hash is pinned to a fixed mixer (a 64-bit multiply, the cheap variant
//! the measurement recommended). `DefaultHasher` is deliberately not used: it
//! is unspecified, randomly seeded and not stable across releases.

use std::ops::Index;

/// One contiguous `f64` block per column, row count fixed at construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColumnStore {
    n_rows: usize,
    cols: Vec<Vec<f64>>,
}

impl ColumnStore {
    pub fn new(n_rows: usize, cols: Vec<Vec<f64>>) -> Self {
        debug_assert!(cols.iter().all(|c| c.len() == n_rows));
        ColumnStore { n_rows, cols }
    }

    pub fn empty(n_cols: usize) -> Self {
        ColumnStore {
            n_rows: 0,
            cols: vec![Vec::new(); n_cols],
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// The whole contiguous column. This is the resident accessor; out-of-core
    /// code must go through [`ColumnStore::load_group`] instead.
    pub fn col(&self, c: usize) -> &[f64] {
        &self.cols[c]
    }

    pub fn col_mut(&mut self, c: usize) -> &mut [f64] {
        &mut self.cols[c]
    }

    pub fn get(&self, row: usize, c: usize) -> f64 {
        self.cols[c][row]
    }

    pub fn set(&mut self, row: usize, c: usize, v: f64) {
        self.cols[c][row] = v;
    }

    /// Copy one row out. Hot loops should use [`ColumnStore::col`] instead.
    pub fn row(&self, row: usize) -> Vec<f64> {
        self.cols.iter().map(|c| c[row]).collect()
    }

    pub fn push_row(&mut self, row: &[f64]) {
        assert!(self.cols.is_empty() || row.len() == self.cols.len());
        if self.cols.is_empty() {
            self.cols = vec![Vec::with_capacity(64); row.len()];
        }
        for (c, &v) in row.iter().enumerate() {
            self.cols[c].push(v);
        }
        self.n_rows += 1;
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let n_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut cols = vec![vec![0.0; rows.len()]; n_cols];
        for (r, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                cols[c][r] = v;
            }
        }
        ColumnStore {
            n_rows: rows.len(),
            cols,
        }
    }

    /// A view containing only `rows`, in the given order.
    pub fn gather(&self, rows: &[u32]) -> Self {
        let cols = self
            .cols
            .iter()
            .map(|c| rows.iter().map(|&r| c[r as usize]).collect())
            .collect();
        ColumnStore {
            n_rows: rows.len(),
            cols,
        }
    }

    /// A view containing the rows where `keep[r]` is true, preserving order.
    pub fn filter(&self, keep: &[bool]) -> Self {
        let idx: Vec<u32> = (0..self.n_rows)
            .filter(|&r| keep[r])
            .map(|r| r as u32)
            .collect();
        self.gather(&idx)
    }

    /// Rows addressed by a selector.
    pub fn select(&self, selector: &RowSelector) -> Self {
        match selector {
            RowSelector::All => self.clone(),
            RowSelector::Range { start, end } => {
                let lo = (*start).min(self.n_rows);
                let hi = (*end).min(self.n_rows).max(lo);
                let cols = self
                    .cols
                    .iter()
                    .map(|c| c[lo..hi].to_vec())
                    .collect::<Vec<_>>();
                ColumnStore {
                    n_rows: hi - lo,
                    cols,
                }
            }
            other => self.gather(&other.indices(self.n_rows)),
        }
    }

    /// Read one group of `group_size` rows into a caller-owned buffer.
    ///
    /// The resident backing just copies from its slices. This is the signature
    /// a `pread`-backed store will implement (issue #71); the copy is what lets
    /// it hand out a [`GroupView`] without exposing a whole-column slice.
    pub fn load_group<'b>(
        &self,
        group: usize,
        group_size: usize,
        buf: &'b mut GroupBuf,
    ) -> GroupView<'b> {
        let lo = group.saturating_mul(group_size).min(self.n_rows);
        let hi = (lo + group_size).min(self.n_rows);
        buf.resize(hi - lo, self.n_cols());
        for (c, dst) in buf.cols.iter_mut().enumerate() {
            dst.copy_from_slice(&self.cols[c][lo..hi]);
        }
        buf.view()
    }

    /// Borrow one resident group without copying.
    pub fn group_view(&self, group: usize, group_size: usize) -> GroupView<'_> {
        let lo = group.saturating_mul(group_size).min(self.n_rows);
        let hi = (lo + group_size).min(self.n_rows);
        GroupView {
            cols: self.cols.iter().map(|c| &c[lo..hi]).collect(),
            len: hi - lo,
        }
    }
}

impl Index<usize> for ColumnStore {
    type Output = [f64];
    fn index(&self, c: usize) -> &[f64] {
        &self.cols[c]
    }
}

/// Reusable destination for [`ColumnStore::load_group`].
///
/// A double-buffered out-of-core reader keeps two of these and swaps them, so
/// group *i+1* can be read while group *i* is evaluated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupBuf {
    cols: Vec<Vec<f64>>,
    len: usize,
}

impl GroupBuf {
    pub fn new() -> Self {
        Self::default()
    }

    fn resize(&mut self, len: usize, n_cols: usize) {
        self.cols.resize_with(n_cols, Vec::new);
        for c in &mut self.cols {
            c.clear();
            c.resize(len, f64::NAN);
        }
        self.len = len;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn view(&self) -> GroupView<'_> {
        GroupView {
            cols: self.cols.iter().map(|c| c.as_slice()).collect(),
            len: self.len,
        }
    }
}

/// A window of rows with one slice per column.
pub struct GroupView<'a> {
    cols: Vec<&'a [f64]>,
    len: usize,
}

impl<'a> GroupView<'a> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    /// The only hot-path accessor for a group's column.
    pub fn col(&self, c: usize) -> &'a [f64] {
        self.cols[c]
    }
}

/// How rows of a store are selected. The variants are views; none stores a
/// second copy of the data.
#[derive(Clone, Debug, PartialEq)]
pub enum RowSelector {
    /// Every row.
    All,
    /// A contiguous half-open range `[start, end)`.
    Range { start: usize, end: usize },
    /// A pinned hash predicate: row `r` is selected when
    /// `mix(r, seed) as f64 / 2^64 < pct`.
    Hash { seed: u64, pct: f64 },
    /// Explicit row ids (dictionary codes, sorted ascending).
    Set(Vec<u32>),
}

impl RowSelector {
    /// Classify an already-materialised row set.
    ///
    /// This is the honest description of a split that was decided by the
    /// shuffle/group logic: a contiguous block is a `Range`, everything else is
    /// a `Set` when it is small enough to keep resident, otherwise a `Hash`
    /// predicate is the only O(1)-memory representation of "that many rows".
    /// It never changes which rows were chosen.
    pub fn from_rows(n: usize, rows: &[u32]) -> Self {
        if rows.is_empty() {
            return RowSelector::Range { start: 0, end: 0 };
        }
        if rows.len() == n && rows.iter().enumerate().all(|(i, &r)| r as usize == i) {
            return RowSelector::All;
        }
        let contiguous = rows
            .windows(2)
            .all(|w| w[1] == w[0].wrapping_add(1));
        if contiguous {
            return RowSelector::Range {
                start: rows[0] as usize,
                end: rows[rows.len() - 1] as usize + 1,
            };
        }
        // A materialised split that is too large for a `Set` still holds the
        // exact ids the split decided on; it cannot be replaced by a hash
        // without changing which rows are scored. Report the representation it
        // actually is, and let `auto` choose the hash for a from-scratch plan.
        RowSelector::Set(rows.to_vec())
    }

    /// Choose a selector from a row count and caller hints.
    ///
    /// `reused` means the same selector is evaluated more than once (folds,
    /// bootstrap, `--fit-samples`); `resident` means the table is in RAM.
    /// `small_set` is an already-known explicit subset, if any.
    pub fn auto(
        n: usize,
        reused: bool,
        resident: bool,
        small_set: Option<Vec<u32>>,
        contiguous: Option<(usize, usize)>,
        seed: u64,
        pct: f64,
    ) -> Self {
        if let Some((start, end)) = contiguous {
            if start == 0 && end >= n {
                return RowSelector::All;
            }
            return RowSelector::Range { start, end };
        }
        if let Some(set) = small_set {
            let small_enough = set.len() <= SET_CAP
                && (set.len() as f64) <= if resident { RAM_SET_FRACTION } else { DISK_SET_FRACTION } * n as f64;
            // A one-shot Set never wins: building it costs the scan.
            if reused && resident && small_enough {
                return RowSelector::Set(set);
            }
        }
        RowSelector::Hash { seed, pct }
    }

    /// A contiguous fold assignment expressed as a hash predicate would not be
    /// exact; `fold` is therefore a range when the fold is a block and a hash
    /// otherwise. This helper keeps the choice in one place.
    pub fn fold(n: usize, fold: usize, k: usize, seed: u64, contiguous: bool) -> Self {
        let k = k.max(1);
        if contiguous {
            let lo = n * fold / k;
            let hi = n * (fold + 1) / k;
            RowSelector::Range { start: lo, end: hi }
        } else {
            RowSelector::Hash {
                seed: seed ^ (fold as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                pct: 1.0 / k as f64,
            }
        }
    }

    /// Materialised row ids for this selector, in ascending order.
    ///
    /// A `Set` returns its own ids; `Range`/`All` are `O(|S|)`; `Hash` is one
    /// pass over `n`. Long-lived selectors should keep this vector rather than
    /// rebuild it per column.
    pub fn indices(&self, n: usize) -> Vec<u32> {
        match self {
            RowSelector::All => (0..n as u32).collect(),
            RowSelector::Range { start, end } => {
                let lo = (*start).min(n);
                let hi = (*end).min(n).max(lo);
                (lo as u32..hi as u32).collect()
            }
            RowSelector::Set(rows) => rows.iter().copied().filter(|&r| (r as usize) < n).collect(),
            RowSelector::Hash { seed, pct } => {
                let threshold = hash_threshold(*pct);
                (0..n)
                    .filter(|&r| hash_row(r as u64, *seed) < threshold)
                    .map(|r| r as u32)
                    .collect()
            }
        }
    }

    /// Evaluate the predicate once per group into a `ceil(group/64)`-word
    /// mask, then reuse the mask for every column in that group.
    pub fn mask(&self, n: usize) -> Vec<u64> {
        let mut mask = vec![0u64; (n + 63) / 64];
        match self {
            RowSelector::All => {
                for (i, w) in mask.iter_mut().enumerate() {
                    let base = i * 64;
                    *w = if base + 64 <= n {
                        !0u64
                    } else {
                        (1u64 << (n - base)) - 1
                    };
                }
            }
            RowSelector::Range { start, end } => {
                let lo = (*start).min(n);
                let hi = (*end).min(n).max(lo);
                for r in lo..hi {
                    mask[r / 64] |= 1u64 << (r % 64);
                }
            }
            RowSelector::Set(rows) => {
                for &r in rows {
                    if (r as usize) < n {
                        mask[r as usize / 64] |= 1u64 << (r as usize % 64);
                    }
                }
            }
            RowSelector::Hash { seed, pct } => {
                let threshold = hash_threshold(*pct);
                for r in 0..n {
                    if hash_row(r as u64, *seed) < threshold {
                        mask[r / 64] |= 1u64 << (r % 64);
                    }
                }
            }
        }
        mask
    }

    pub fn describe(&self, n: usize) -> String {
        match self {
            RowSelector::All => format!("all ({} rows)", n),
            RowSelector::Range { start, end } => {
                let lo = (*start).min(n);
                let hi = (*end).min(n).max(lo);
                format!("range {}..{} ({} rows)", lo, hi, hi - lo)
            }
            RowSelector::Hash { seed, pct } => format!(
                "hash seed={:#x} pct={} ({} rows)",
                seed,
                pct,
                self.indices(n).len()
            ),
            RowSelector::Set(rows) => format!("set ({} rows)", rows.len()),
        }
    }
}

/// Share of rows below which a reused resident `Set` beats a scan. Measured on
/// a cheap one-multiply hash: 7 % (a 10 % gather already loses to the scan).
pub const RAM_SET_FRACTION: f64 = 0.07;
/// Out of core a `Set` is only useful for a few times 10^4 rows; the fraction
/// is the NVMe 4 KiB-page rule (~0.2 %). Callers also apply [`SET_CAP`].
pub const DISK_SET_FRACTION: f64 = 0.002;
/// Absolute cap on a resident `Set`, independent of the fraction above.
pub const SET_CAP: usize = 1_000_000;

/// The pinned row hash: one 64-bit multiply and the high bits. It is cheap
/// (1.5 ns/row measured) and stable across releases.
#[inline]
pub fn hash_row(row: u64, seed: u64) -> u64 {
    (row ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[inline]
fn hash_threshold(pct: f64) -> u64 {
    let p = pct.clamp(0.0, 1.0);
    (p * u64::MAX as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ColumnStore {
        ColumnStore::new(
            5,
            vec![vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![10.0, 11.0, 12.0, 13.0, 14.0]],
        )
    }

    #[test]
    fn columns_are_contiguous_one_block_per_column() {
        let s = store();
        assert_eq!(s.n_rows(), 5);
        assert_eq!(s.n_cols(), 2);
        assert_eq!(s.col(1), &[10.0, 11.0, 12.0, 13.0, 14.0]);
        assert_eq!(s.get(3, 0), 3.0);
    }

    #[test]
    fn group_view_and_load_group_agree() {
        let s = store();
        let v = s.group_view(1, 2);
        assert_eq!(v.len(), 2);
        assert_eq!(v.col(0), &[2.0, 3.0]);
        assert_eq!(v.col(1), &[12.0, 13.0]);

        let mut buf = GroupBuf::new();
        let loaded = s.load_group(1, 2, &mut buf);
        assert_eq!(loaded.col(0), v.col(0));
        assert_eq!(loaded.col(1), v.col(1));
        // Last group is short, not padded.
        let tail = s.group_view(2, 2);
        assert_eq!(tail.col(0), &[4.0]);
    }

    #[test]
    fn range_select_is_a_view_and_preserves_values() {
        let s = store();
        let sub = s.select(&RowSelector::Range { start: 1, end: 4 });
        assert_eq!(sub.n_rows(), 3);
        assert_eq!(sub.col(0), &[1.0, 2.0, 3.0]);
        let set = s.select(&RowSelector::Set(vec![4, 0]));
        assert_eq!(set.col(0), &[4.0, 0.0]);
    }

    #[test]
    fn hash_mask_matches_indices() {
        let n = 10_000;
        for pct in [0.01, 0.07, 0.5, 1.0] {
            let sel = RowSelector::Hash { seed: 7, pct };
            let idx = sel.indices(n);
            let mask = sel.mask(n);
            let from_mask: Vec<u32> = (0..n)
                .filter(|&r| mask[r / 64] & (1u64 << (r % 64)) != 0)
                .map(|r| r as u32)
                .collect();
            assert_eq!(idx, from_mask);
            let frac = idx.len() as f64 / n as f64;
            assert!((frac - pct).abs() < 0.02, "pct={} frac={}", pct, frac);
        }
    }

    #[test]
    fn auto_uses_range_for_contiguous_rows() {
        let sel = RowSelector::auto(100, false, true, None, Some((10, 40)), 1, 0.1);
        assert_eq!(sel, RowSelector::Range { start: 10, end: 40 });
        let sel = RowSelector::auto(100, false, true, None, Some((0, 100)), 1, 1.0);
        assert_eq!(sel, RowSelector::All);
    }

    #[test]
    fn auto_set_only_when_reused_resident_and_small() {
        let rows: Vec<u32> = (0..50).map(|i| i * 2).collect();
        // 50 % of the rows: too large for a Set even when reused.
        assert!(matches!(
            RowSelector::auto(100, true, true, Some(rows.clone()), None, 1, 0.5),
            RowSelector::Hash { .. }
        ));
        // 5 %: a reused resident Set wins.
        assert_eq!(
            RowSelector::auto(1000, true, true, Some(rows), None, 1, 0.5),
            RowSelector::Set((0..50).map(|i| i * 2).collect())
        );
        // A one-shot Set never wins.
        assert!(matches!(
            RowSelector::auto(1000, false, true, Some(vec![1, 2, 3]), None, 1, 0.5),
            RowSelector::Hash { .. }
        ));
        // Out of core the same fraction is below the 0.2 % rule.
        assert!(matches!(
            RowSelector::auto(100_000, true, false, Some(vec![1, 2, 3]), None, 1, 0.0),
            RowSelector::Hash { .. }
        ));
    }

    #[test]
    fn set_cap_is_absolute_not_a_fraction() {
        let set: Vec<u32> = (0..SET_CAP as u32 + 1).collect();
        assert!(matches!(
            RowSelector::auto(
                set.len() * 100,
                true,
                true,
                Some(set),
                None,
                1,
                0.0
            ),
            RowSelector::Hash { .. }
        ));
    }

    #[test]
    fn from_rows_classifies_without_changing_them() {
        assert_eq!(RowSelector::from_rows(5, &[0, 1, 2, 3, 4]), RowSelector::All);
        assert_eq!(
            RowSelector::from_rows(10, &[3, 4, 5]),
            RowSelector::Range { start: 3, end: 6 }
        );
        assert_eq!(
            RowSelector::from_rows(10, &[1, 5]),
            RowSelector::Set(vec![1, 5])
        );
    }
}
