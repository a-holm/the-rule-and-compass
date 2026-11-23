//! #116: out-of-core / row-sharded fitting with exact `f64` merging.
//!
//! The historical LM loop borrows a resident `Vec<Vec<f64>>`. The streaming arm
//! keeps only the sufficient statistics of the weighted problem:
//!
//! ```text
//! S   = Σ w_i g_i g_iᵀ      (p × p)
//! b   = Σ w_i g_i r_i       (p)
//! sse = Σ w_i r_i²          (scalar)
//! ```
//!
//! `S`, `b` and `sse` are built a block at a time and merged by addition, so
//! the resident state is `O(p²)`, not `O(N p)`. Each row is read once per LM
//! iteration (plus one forward pass for the trial SSE); the Jacobian is never
//! materialised. The row source is a trait, so a `.rcsb` store, a memory-mapped
//! file or a generated/streamed table can feed the same exact merger without
//! the `Sample` being resident.
//!
//! Sums use Neumaier compensation. Merging block partials is therefore
//! numerically stable and, unlike naive `f32` atomics, deterministic: the block
//! boundaries and merge order are fixed, so the same source always produces the
//! same constants.

use super::*;
use crate::column::GroupBuf;
use crate::store::FileStore;
use std::sync::Arc;

/// Rows per accumulation block. The same value is used for every pass, making
/// the reduction order deterministic for a given row count.
pub const STREAM_BLOCK_ROWS: usize = 8192;

/// A read-only row source for the streaming fitter.
///
/// Implementors can read from disk or generate rows; the fitter only asks for
/// one row at a time and retains nothing beyond the current block's statistics.
/// Returning `false` from the callback stops iteration (used on the first
/// non-finite row, which routes the candidate to the exact fallback).
pub trait RowSource {
    /// Total number of logical rows.
    fn rows(&self) -> usize;
    /// Visit every row as `(x, y, weight)`. The callback returns `false` to
    /// abort. The return value is `true` when every row was visited.
    fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool;
}

/// The resident slice implementation, used by the CLI when the training sample
/// is already in memory. The algorithm is still blockwise and never allocates
/// a design matrix.
pub struct SliceRows<'a> {
    pub x: &'a [Vec<f64>],
    pub y: &'a [f64],
    pub w: &'a [f64],
}

impl RowSource for SliceRows<'_> {
    fn rows(&self) -> usize {
        self.y.len()
    }

    fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
        let n = self.y.len().min(self.x.len());
        for i in 0..n {
            let w = self.w.get(i).copied().unwrap_or(1.0);
            if !f(&self.x[i], self.y[i], w) {
                return false;
            }
        }
        true
    }
}

/// #116: a resolved `.rcsb` projection for the streaming fitter.
///
/// The spec is cheap to share between threads: it holds the opened
/// [`FileStore`] plus column indices, the opt-in row subset and the same affine
/// per-column transform the resident training sample received. Each fit creates
/// its own [`StoreRows`] with its own [`GroupBuf`], so concurrent islands never
/// share mutable reader state and never materialise the data.
#[derive(Clone, Debug)]
pub struct StoreSpec {
    store: Arc<FileStore>,
    columns: Vec<usize>,
    target: usize,
    weight: Option<usize>,
    means: Vec<f64>,
    stds: Vec<f64>,
    /// Sorted ascending raw row ids, or `None` for every row.
    rows: Option<Vec<usize>>,
}

impl StoreSpec {
    /// Project `columns` and `target` from an opened store. Every used column
    /// must be numeric; text/dictionary columns are rejected with a descriptive
    /// error so the caller falls back to the resident path.
    pub fn new(
        store: Arc<FileStore>,
        columns: Vec<usize>,
        target: usize,
    ) -> Result<Self, String> {
        let n_cols = store.n_cols();
        if columns.is_empty() {
            return Err("stream store source needs at least one input column".to_string());
        }
        for &c in columns.iter().chain(std::iter::once(&target)) {
            if c >= n_cols {
                return Err(format!(
                    "stream store column {} is out of range ({} columns)",
                    c, n_cols
                ));
            }
            if store.header().is_numeric(c) != Some(true) {
                return Err(format!(
                    "stream store column {} is not numeric; use the resident path",
                    c
                ));
            }
        }
        let k = columns.len();
        Ok(StoreSpec {
            store,
            columns,
            target,
            weight: None,
            means: vec![0.0; k],
            stds: vec![1.0; k],
            rows: None,
        })
    }

    /// Use a numeric weight column (one weight per row).
    pub fn with_weight(mut self, column: usize) -> Result<Self, String> {
        if column >= self.store.n_cols() {
            return Err(format!(
                "stream store weight column {} is out of range ({} columns)",
                column,
                self.store.n_cols()
            ));
        }
        if self.store.header().is_numeric(column) != Some(true) {
            return Err(format!(
                "stream store weight column {} is not numeric",
                column
            ));
        }
        self.weight = Some(column);
        Ok(self)
    }

    /// Apply the prepared training centring/scaling to the input columns.
    /// `means`/`stds` are in the same order as [`StoreSpec::columns`].
    pub fn with_standardisation(mut self, means: Vec<f64>, stds: Vec<f64>) -> Result<Self, String> {
        if means.len() != self.columns.len() || stds.len() != self.columns.len() {
            return Err(format!(
                "stream store transform has {} means and {} stds for {} columns",
                means.len(),
                stds.len(),
                self.columns.len()
            ));
        }
        if !means.iter().chain(&stds).all(|v| v.is_finite()) {
            return Err("stream store transform contains a non-finite value".to_string());
        }
        if stds.iter().any(|s| *s == 0.0) {
            return Err("stream store transform has a zero scale".to_string());
        }
        self.means = means;
        self.stds = stds;
        Ok(self)
    }

    /// Restrict the source to the prepared training rows. Ids are raw store
    /// row numbers; they are sorted and deduplicated so the read order (and
    /// therefore the reduction order) is deterministic.
    pub fn with_rows(mut self, mut rows: Vec<usize>) -> Result<Self, String> {
        rows.sort_unstable();
        rows.dedup();
        if let Some(&last) = rows.last() {
            if last >= self.store.n_rows() {
                return Err(format!(
                    "stream store row {} is out of range ({} rows)",
                    last,
                    self.store.n_rows()
                ));
            }
        }
        if rows.is_empty() {
            return Err("stream store row subset is empty".to_string());
        }
        self.rows = Some(rows);
        Ok(self)
    }

    pub fn rows(&self) -> usize {
        self.rows
            .as_ref()
            .map(|r| r.len())
            .unwrap_or_else(|| self.store.n_rows())
    }
}

/// #116: disk-backed [`RowSource`] over [`FileStore::load_group_with`].
///
/// One group is resident at a time (`GroupBuf`, default 512 KiB per `f64`
/// column) plus one scratch row and the decode buffer. The fitter keeps only
/// `O(p²)` statistics, so a store larger than RAM is consumed group by group.
pub struct StoreRows {
    spec: Arc<StoreSpec>,
    reader: std::cell::RefCell<StoreReader>,
}

struct StoreReader {
    buf: GroupBuf,
    scratch: Vec<u8>,
    row: Vec<f64>,
}

impl StoreRows {
    pub fn new(spec: Arc<StoreSpec>) -> Self {
        let k = spec.columns.len();
        StoreRows {
            spec,
            reader: std::cell::RefCell::new(StoreReader {
                buf: GroupBuf::new(),
                scratch: Vec::new(),
                row: vec![0.0; k],
            }),
        }
    }
}

impl RowSource for StoreRows {
    fn rows(&self) -> usize {
        self.spec.rows()
    }

    fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
        let store = &*self.spec.store;
        let mut next = 0usize;
        let mut reader = self.reader.borrow_mut();
        let StoreReader {
            buf,
            scratch,
            row,
        } = &mut *reader;
        for g in 0..store.n_groups() {
            if let Some(selected) = &self.spec.rows {
                if next >= selected.len() {
                    break;
                }
            }
            let view = match store.load_group_with(g, buf, scratch) {
                Ok(v) => v,
                Err(_) => return false,
            };
            let base = g.saturating_mul(store.group_rows());
            for r in 0..view.len() {
                let raw = base + r;
                if let Some(selected) = &self.spec.rows {
                    while next < selected.len() && selected[next] < raw {
                        next += 1;
                    }
                    if next >= selected.len() || selected[next] != raw {
                        continue;
                    }
                    next += 1;
                }
                for (j, &c) in self.spec.columns.iter().enumerate() {
                    row[j] = (view.col(c)[r] - self.spec.means[j]) / self.spec.stds[j];
                }
                let y = view.col(self.spec.target)[r];
                let w = self
                    .spec
                    .weight
                    .map(|c| view.col(c)[r])
                    .unwrap_or(1.0);
                if !f(row, y, w) {
                    return false;
                }
            }
        }
        true
    }
}

/// Neumaier-compensated accumulator.
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    sum: f64,
    comp: f64,
}

impl Acc {
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn get(self) -> f64 {
        self.sum + self.comp
    }
}

/// Mergeable sufficient statistics. `p` is the number of constants.
#[derive(Clone, Debug)]
struct Sufficient {
    s: Vec<Acc>,
    b: Vec<Acc>,
    sse: Acc,
    rows: usize,
    p: usize,
}

impl Sufficient {
    fn new(p: usize) -> Self {
        Sufficient {
            s: vec![Acc::default(); p * p],
            b: vec![Acc::default(); p],
            sse: Acc::default(),
            rows: 0,
            p,
        }
    }

    fn reset(&mut self) {
        for a in &mut self.s {
            *a = Acc::default();
        }
        for a in &mut self.b {
            *a = Acc::default();
        }
        self.sse = Acc::default();
        self.rows = 0;
    }

    /// Add one weighted row. `g` is `√w·J` and `r` is `√w·(f−y)`.
    fn add_row(&mut self, g: &[f64], r: f64) {
        for i in 0..self.p {
            self.b[i].add(g[i] * r);
            for j in 0..=i {
                self.s[i * self.p + j].add(g[i] * g[j]);
            }
        }
        self.sse.add(r * r);
        self.rows += 1;
    }

    /// Merge another block by addition. This is the exact f64 reduction step;
    /// no design matrix and no per-row data are needed.
    fn merge(&mut self, other: &Sufficient) {
        debug_assert_eq!(self.p, other.p);
        for (a, o) in self.s.iter_mut().zip(&other.s) {
            a.add(o.sum);
            a.add(o.comp);
        }
        for (a, o) in self.b.iter_mut().zip(&other.b) {
            a.add(o.sum);
            a.add(o.comp);
        }
        self.sse.add(other.sse.sum);
        self.sse.add(other.sse.comp);
        self.rows += other.rows;
    }

    fn normal(&self) -> Vec<Vec<f64>> {
        let p = self.p;
        let mut a = vec![vec![0.0; p]; p];
        for i in 0..p {
            for j in 0..=i {
                let v = self.s[i * p + j].get();
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        a
    }

    fn rhs(&self) -> Vec<f64> {
        self.b.iter().map(|a| -a.get()).collect()
    }
}

/// #116: fit a tree from any [`RowSource`] using block sufficient statistics.
///
/// Returns the weighted SSE, or `None` when the source cannot be read
/// consistently, a value is non-finite, or the normal equations are singular.
/// On `None` the tree is untouched and the caller runs the exact LM path.
pub fn fit_stream_with<S: RowSource>(tree: &mut Node, source: &S, iters: usize) -> Option<f64> {
    let f = flatten(tree);
    let k = f.nconst;
    if k == 0 || source.rows() == 0 {
        let mut v = vec![0.0; f.op.len().max(1)];
        return Some(sse_scan(&f, source, &[], &mut v)?);
    }
    let mut c = vec![0.0; k];
    for i in 0..f.op.len() {
        if f.cslot[i] >= 0 {
            c[f.cslot[i] as usize] = f.val[i];
        }
    }
    let mut best_c = c.clone();
    let mut best_sse = sse_scan(&f, source, &best_c, &mut vec![0.0; f.op.len()])?;
    let start = best_sse;
    let mut lambda = 1e-3;
    let mut aborted = false;

    for _ in 0..iters.max(1) {
        let Some(stats) = accumulate(&f, source, &c) else {
            aborted = true;
            break;
        };
        let a = stats.normal();
        let mut rhs = stats.rhs();
        let Some(step) = solve(&a, &mut rhs, lambda, k) else {
            break;
        };
        let trial: Vec<f64> = c
            .iter()
            .zip(&step)
            .map(|(v, d)| (v + d).clamp(-1e6, 1e6))
            .collect();
        if !trial.iter().all(|v| v.is_finite()) {
            aborted = true;
            break;
        }
        let Some(trial_sse) = sse_scan(&f, source, &trial, &mut vec![0.0; f.op.len()]) else {
            aborted = true;
            break;
        };
        if trial_sse < best_sse {
            c = trial;
            best_c = c.clone();
            best_sse = trial_sse;
            lambda = (lambda / 8.0).max(1e-12);
            if best_sse == 0.0 {
                break;
            }
        } else {
            lambda *= 6.0;
            if lambda > 1e9 {
                break;
            }
        }
    }
    if aborted || !best_sse.is_finite() || best_sse > start {
        return None;
    }
    write_constants(tree, &f, &best_c);
    Some(best_sse)
}

/// Resident slice entry point for the streaming arm.
pub fn fit_stream(tree: &mut Node, x: &[Vec<f64>], y: &[f64], w: &[f64], iters: usize) -> Option<f64> {
    if x.len() != y.len() || (!w.is_empty() && w.len() != y.len()) {
        return None;
    }
    fit_stream_with(tree, &SliceRows { x, y, w }, iters)
}

/// Accumulate one iteration's normal equations in fixed-size blocks.
fn accumulate(f: &Flat, source: &dyn RowSource, c: &[f64]) -> Option<Sufficient> {
    let k = f.nconst;
    let mut total = Sufficient::new(k);
    let mut block = Sufficient::new(k);
    let mut v = vec![0.0; f.op.len()];
    let mut adj = vec![0.0; f.op.len()];
    let mut grad = vec![0.0; k];
    let mut g = vec![0.0; k];
    let mut ok = true;
    let mut seen = 0usize;
    let complete = source.for_each_row(&mut |row, t, sw| {
        if sw == 0.0 {
            return true; // a zero-weight row contributes nothing
        }
        if !(sw.is_finite() && sw > 0.0) {
            ok = false;
            return false;
        }
        eval_row(f, row, c, &mut v);
        let root = sw.sqrt();
        let r = root * (v[f.root] - t);
        if !r.is_finite() {
            ok = false;
            return false;
        }
        backward(f, &v, &mut adj, &mut grad);
        for (dst, &src) in g.iter_mut().zip(&grad) {
            if !src.is_finite() {
                ok = false;
                return false;
            }
            *dst = root * src;
        }
        block.add_row(&g, r);
        seen += 1;
        if seen % STREAM_BLOCK_ROWS == 0 {
            total.merge(&block);
            block.reset();
        }
        true
    });
    if !ok || !complete {
        return None;
    }
    total.merge(&block);
    Some(total)
}

/// Forward-only weighted SSE over the source.
fn sse_scan(f: &Flat, source: &dyn RowSource, c: &[f64], v: &mut [f64]) -> Option<f64> {
    let mut acc = Acc::default();
    let mut ok = true;
    let complete = source.for_each_row(&mut |row, t, sw| {
        if sw == 0.0 {
            return true;
        }
        if !(sw.is_finite() && sw > 0.0) {
            ok = false;
            return false;
        }
        eval_row(f, row, c, v);
        let r = sw.sqrt() * (v[f.root] - t);
        if !r.is_finite() {
            ok = false;
            return false;
        }
        acc.add(r * r);
        true
    });
    if !ok || !complete {
        None
    } else {
        Some(acc.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    fn fitted(tree: &Node) -> Vec<f64> {
        let f = flatten(tree);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        c
    }

    /// A generated source is never materialised. This exercises the same code
    /// path a disk-backed out-of-core source would use: rows arrive one at a
    /// time, the fitter keeps only `p²` statistics.
    struct Generated {
        n: usize,
    }

    impl RowSource for Generated {
        fn rows(&self) -> usize {
            self.n
        }

        fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
            for i in 0..self.n {
                let x = -3.0 + 6.0 * i as f64 / (self.n - 1) as f64;
                // a*sin(b*x)+c, the classic separable law.
                let y = 2.0 * (0.9 * x).sin() + 0.3;
                if !f(&[x], y, 1.0) {
                    return false;
                }
            }
            true
        }
    }

    fn model() -> Node {
        node(
            Op::Add,
            vec![
                node(
                    Op::Mul,
                    vec![
                        constant(0.5),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                    ],
                ),
                constant(0.0),
            ],
        )
    }

    #[test]
    fn streaming_fit_matches_the_resident_fit() {
        let n = 20_000;
        let source = Generated { n };
        let mut streamed = model();
        let a = fit_stream_with(&mut streamed, &source, 16).expect("stream fit");
        // Materialise the same rows only for the comparison.
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        source.for_each_row(&mut |row, t, _| {
            x.push(row.to_vec());
            y.push(t);
            true
        });
        let mut resident = model();
        let b = fit_weighted(&mut resident, &x, &y, &[], 16);
        assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "stream {a} vs resident {b}");
        for (u, v) in fitted(&streamed).iter().zip(fitted(&resident)) {
            assert!((u - v).abs() <= 1e-6 * (1.0 + v.abs()), "constants {u} vs {v}");
        }
    }

    /// Block-size independence: merging 997-row blocks and 8192-row blocks
    /// gives the same statistics to rounding, so the merge is stable.
    #[test]
    fn sufficient_statistics_merge_additively() {
        let p = 3;
        let mut whole = Sufficient::new(p);
        let mut a = Sufficient::new(p);
        let mut b = Sufficient::new(p);
        for i in 0..3000 {
            let g = [i as f64, (i as f64).sqrt(), 1.0];
            let r = (i % 7) as f64 - 3.0;
            whole.add_row(&g, r);
            if i < 997 {
                a.add_row(&g, r);
            } else {
                b.add_row(&g, r);
            }
        }
        a.merge(&b);
        for i in 0..p * p {
            let (u, v) = (whole.s[i].get(), a.s[i].get());
            assert!((u - v).abs() <= 1e-9 * (1.0 + u.abs()), "S[{i}] {u} vs {v}");
        }
        assert!((whole.sse.get() - a.sse.get()).abs() <= 1e-9 * (1.0 + whole.sse.get()));
    }

    #[test]
    fn non_finite_rows_reject_the_streaming_arm() {
        struct Bad;
        impl RowSource for Bad {
            fn rows(&self) -> usize {
                3
            }
            fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
                f(&[0.0], f64::INFINITY, 1.0)
            }
        }
        let mut tree = model();
        assert!(fit_stream_with(&mut tree, &Bad, 8).is_none());
    }
}
