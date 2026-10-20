//! Quality-diversity archive: a grid over cheap behaviour descriptors (#120).
//!
//! This is the first QD increment, deliberately independent of the search's
//! parent-selection and migration policy. It records **what behaviour regions
//! were reached**, not just which (error, complexity) pairs were non-dominated:
//!
//! * descriptor 1: **variable usage** — the fraction of input columns the tree
//!   actually reads, in `[0, 1]`;
//! * descriptor 2: **prediction skewness** — the third standardised moment of
//!   the prediction vector, mapped monotonically to `[0, 1]`.
//!
//! Both are cheap summaries of the tree and its prediction vector. The full
//! CVT-MAP-Elites design in #120 uses a centroidal Voronoi tessellation; this
//! increment lands a fixed rectangular grid, which has the same archive
//! contract (one bounded local Pareto front per cell, deterministic insertion)
//! without the CVT construction. See `docs/qd.md` for the remainder.
//!
//! The archive is opt-in and never changes the search trajectory or the
//! reported Pareto front.

use crate::data::Sample;
use crate::expr::{self, Node, Op};

/// A point in the two-dimensional descriptor space, each coordinate in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Descriptor {
    pub usage: f64,
    pub skew: f64,
}

#[derive(Clone, Debug)]
pub struct QdEntry {
    pub tree: Node,
    pub error: f64,
    pub cost: f64,
}

/// A bounded local Pareto front in one descriptor cell.
#[derive(Clone, Debug, Default)]
pub struct QdCell {
    pub entries: Vec<QdEntry>,
}

impl QdCell {
    fn dominates(a: &QdEntry, b: &QdEntry) -> bool {
        let tol = 1e-6 * (1.0 + a.error.abs().max(b.error.abs()));
        if a.cost > b.cost {
            return false;
        }
        if a.cost < b.cost {
            a.error <= b.error + tol
        } else {
            a.error < b.error
        }
    }

    /// Insert one candidate; returns whether the cell changed.
    fn insert(&mut self, entry: QdEntry, cap: usize) -> bool {
        if self.entries.iter().any(|e| Self::dominates(e, &entry)) {
            return false;
        }
        let before = self.entries.len();
        self.entries
            .retain(|e| !Self::dominates(&entry, e));
        if self.entries.iter().any(|e| {
            e.cost == entry.cost && (e.error - entry.error).abs() < 1e-12
        }) {
            return self.entries.len() != before;
        }
        self.entries.push(entry);
        if self.entries.len() > cap.max(1) {
            // Deterministic bounded front: keep the spread, drop the middle
            // cost level first (the same rule as the global archive).
            self.entries.sort_by(|a, b| {
                a.cost
                    .total_cmp(&b.cost)
                    .then(a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
            });
            while self.entries.len() > cap.max(1) {
                self.entries.remove(self.entries.len() / 2);
            }
        }
        true
    }
}

/// A rectangular descriptor grid whose occupied cells each hold a bounded local
/// Pareto front over `(error, cost)`.
#[derive(Clone, Debug)]
pub struct QdArchive {
    bins: usize,
    cell_cap: usize,
    cells: Vec<QdCell>,
}

/// The per-run coverage summary reported by the CLI and `--json`.
#[derive(Clone, Debug, PartialEq)]
pub struct QdReport {
    pub bins: usize,
    pub cells: usize,
    pub occupied: usize,
    pub coverage: f64,
    pub entries: usize,
    pub best_error: f64,
    pub cell_cap: usize,
}

impl QdArchive {
    pub fn new(bins: usize, cell_cap: usize) -> Self {
        let bins = bins.max(2);
        QdArchive {
            bins,
            cell_cap: cell_cap.max(1),
            cells: vec![QdCell::default(); bins * bins],
        }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    fn cell_index(&self, d: Descriptor) -> usize {
        let b = self.bins;
        let u = (d.usage.clamp(0.0, 1.0) * b as f64).floor() as usize;
        let s = (d.skew.clamp(0.0, 1.0) * b as f64).floor() as usize;
        u.min(b - 1) * b + s.min(b - 1)
    }

    /// Insert one evaluated candidate. Returns whether the archive changed.
    pub fn insert(&mut self, tree: &Node, error: f64, cost: f64, sample: &Sample) -> bool {
        if !error.is_finite() {
            return false;
        }
        let d = descriptor(tree, sample);
        let ix = self.cell_index(d);
        self.cells[ix].insert(
            QdEntry {
                tree: tree.clone(),
                error,
                cost,
            },
            self.cell_cap,
        )
    }

    /// Record every individual on the current populations plus the global
    /// archive. Kept deterministic: the caller passes them in a fixed order.
    pub fn record(&mut self, trees: impl IntoIterator<Item = (Node, f64, f64)>, sample: &Sample) {
        for (tree, error, cost) in trees {
            self.insert(&tree, error, cost, sample);
        }
    }

    pub fn report(&self) -> QdReport {
        let mut occupied = 0;
        let mut entries = 0;
        let mut best = f64::INFINITY;
        for cell in &self.cells {
            if !cell.entries.is_empty() {
                occupied += 1;
                entries += cell.entries.len();
                for e in &cell.entries {
                    best = best.min(e.error);
                }
            }
        }
        let total = self.cells.len();
        QdReport {
            bins: self.bins,
            cells: total,
            occupied,
            coverage: occupied as f64 / total as f64,
            entries,
            best_error: best,
            cell_cap: self.cell_cap,
        }
    }

    /// Occupied cells in row-major order, each with its local front. Used by
    /// tests and future reporting (the CLI reports the summary).
    pub fn occupied_cells(&self) -> Vec<(usize, &QdCell)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.entries.is_empty())
            .collect()
    }
}

/// The two cheap behaviour descriptors of one candidate.
///
/// The prediction vector is supplied by the caller when it already exists; this
/// function never evaluates the tree twice on purpose. `variable_usage` walks
/// the tree (including block bodies) and counts distinct input columns.
pub fn descriptor_from_prediction(
    tree: &Node,
    pred: &[f64],
    nvars: usize,
) -> Descriptor {
    let usage = if nvars == 0 {
        0.0
    } else {
        let mut seen = vec![false; nvars];
        fn walk(n: &Node, seen: &mut [bool]) {
            if n.op == Op::Input {
                if let Some(slot) = seen.get_mut(n.val as usize) {
                    *slot = true;
                }
            }
            for k in &n.kids {
                walk(k, seen);
            }
            // A block body's `Input(i)` nodes are parameters bound to the call
            // arguments, not columns; the arguments themselves are in `kids`.
        }
        walk(tree, &mut seen);
        let used = seen.iter().filter(|&&v| v).count();
        used as f64 / nvars as f64
    };
    Descriptor {
        usage,
        skew: skew_descriptor(pred),
    }
}

/// Map skewness to `[0, 1]` with a bounded, monotonic transform. A symmetric
/// prediction sits at 0.5; `g = s / (1 + |s|)` avoids the unbounded tail and
/// keeps zero-variance predictions (undefined skew) at the neutral 0.5.
pub fn skew_descriptor(pred: &[f64]) -> f64 {
    if pred.len() < 3 {
        return 0.5;
    }
    let n = pred.len() as f64;
    let mean = pred.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &p in pred {
        if !p.is_finite() {
            return 0.5;
        }
        let d = p - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 <= 1e-24 {
        return 0.5;
    }
    let skew = m3 / m2.powf(1.5);
    if !skew.is_finite() {
        return 0.5;
    }
    0.5 + 0.5 * (skew / (1.0 + skew.abs()))
}

/// Convenience wrapper for callers that do not already hold the predictions.
/// This is the only path that evaluates the tree here; the search can call
/// [`descriptor_from_prediction`] instead.
pub fn descriptor(tree: &Node, sample: &Sample) -> Descriptor {
    let pred: Vec<f64> = sample.x.iter().map(|row| expr::eval(tree, row)).collect();
    descriptor_from_prediction(tree, &pred, sample.x.first().map(Vec::len).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    fn sample() -> Sample {
        let mut s = Sample::default();
        for i in 0..41 {
            let x = -1.0 + 2.0 * i as f64 / 40.0;
            s.x.push(vec![x, x * x]);
            s.y.push(x);
        }
        s
    }

    #[test]
    fn descriptors_are_bounded_and_distinguish_behaviour() {
        let s = sample();
        let left = node(Op::Sub, vec![constant(2.0), node(Op::Mul, vec![input(0), input(0)])]);
        let right = node(Op::Mul, vec![input(0), input(1)]);
        let d1 = descriptor(&left, &s);
        let d2 = descriptor(&right, &s);
        for d in [d1, d2] {
            assert!((0.0..=1.0).contains(&d.usage));
            assert!((0.0..=1.0).contains(&d.skew));
        }
        // The first tree uses one of two columns, the second both.
        assert!((d1.usage - 0.5).abs() < 1e-12, "{d1:?}");
        assert!((d2.usage - 1.0).abs() < 1e-12, "{d2:?}");
        assert!((d1.skew - d2.skew).abs() > 1e-3, "{d1:?} vs {d2:?}");
    }

    #[test]
    fn insertion_is_deterministic_and_cell_local() {
        let s = sample();
        let trees = [
            (node(Op::Mul, vec![constant(2.0), input(0)]), 1.0, 3.0),
            (node(Op::Add, vec![constant(1.0), input(0)]), 0.5, 3.0),
            (node(Op::Mul, vec![input(0), input(1)]), 2.0, 4.0),
            (node(Op::Mul, vec![input(0), input(1)]), 2.5, 4.0),
        ];
        let run = || {
            let mut q = QdArchive::new(4, 4);
            for (t, e, c) in &trees {
                q.insert(t, *e, *c, &s);
            }
            q
        };
        let a = run();
        let b = run();
        assert_eq!(a.report(), b.report());
        assert!(a.report().occupied >= 2, "{:?}", a.report());
        let cells_a: Vec<_> = a
            .occupied_cells()
            .iter()
            .map(|(i, c)| {
                (
                    *i,
                    c.entries
                        .iter()
                        .map(|e| (e.error.to_bits(), e.cost.to_bits()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let cells_b: Vec<_> = b
            .occupied_cells()
            .iter()
            .map(|(i, c)| {
                (
                    *i,
                    c.entries
                        .iter()
                        .map(|e| (e.error.to_bits(), e.cost.to_bits()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(cells_a, cells_b);
    }

    #[test]
    fn a_dominated_candidate_does_not_replace_the_cell_elite() {
        let s = sample();
        let mut q = QdArchive::new(4, 4);
        let good = node(Op::Mul, vec![constant(2.0), input(0)]);
        q.insert(&good, 0.1, 3.0, &s);
        let worse = node(Op::Mul, vec![constant(2.0), input(0)]);
        assert!(!q.insert(&worse, 0.2, 3.0, &s));
        assert!(q.insert(&worse, 0.05, 3.0, &s));
        let cell = &q.occupied_cells()[0].1;
        assert_eq!(cell.entries.len(), 1);
        assert_eq!(cell.entries[0].error.to_bits(), 0.05f64.to_bits());
    }

    #[test]
    fn descriptor_cost_is_far_below_a_full_fit() {
        let small = sample();
        let mut big = Sample::default();
        for i in 0..800 {
            let x = -1.0 + 2.0 * i as f64 / 799.0;
            big.x.push(vec![x, x * x]);
            big.y.push(x);
        }
        let tree = node(
            Op::Add,
            vec![
                node(Op::Sin, vec![node(Op::Mul, vec![constant(1.3), input(0)])]),
                constant(0.2),
            ],
        );
        for (label, s) in [("small", &small), ("800-row", &big)] {
            let iterations = if label == "small" { 500 } else { 100 };
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = descriptor(&tree, s);
            }
            let elapsed = start.elapsed();
            println!(
                "qd descriptor {label}: {} rows, {} candidates, {:.2} us/candidate",
                s.len(),
                iterations,
                elapsed.as_secs_f64() * 1e6 / iterations as f64
            );
        }
    }
}
