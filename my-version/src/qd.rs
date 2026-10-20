//! Quality-diversity archive: CVT-MAP-Elites over cheap behaviour descriptors (#120).
//!
//! The layer records **which behaviour regions were reached**, not just which
//! `(error, complexity)` pairs were non-dominated:
//!
//! * descriptor 1: **variable usage** — the fraction of input columns the tree
//!   actually reads, in `[0, 1]`;
//! * descriptor 2: **prediction skewness** — the third standardised moment of
//!   the prediction vector, mapped monotonically to `[0, 1]`.
//!
//! Two archives are provided:
//!
//! * [`QdArchive::new`] — the original rectangular grid, kept for backward
//!   compatibility and pinned tests;
//! * [`QdArchive::new_cvt`] — a CVT-MAP-Elites archive with `K` centroids
//!   initialised deterministically by a Halton sequence and updated by Lloyd
//!   steps from the descriptors the search actually visits.
//!
//! Each occupied cell holds a bounded local Pareto front over `(error, cost)`.
//! Insertion is deterministic. The optional search policy (UCB1 parent draws,
//! stratified migration and least-occupied restarts) lives behind
//! `--qd-policy`; with the layer disabled the search is bit-identical to before.

use crate::data::Sample;
use crate::expr::{self, Node, Op};
use crate::rng::Mt;

/// A point in the two-dimensional descriptor space, each coordinate in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Descriptor {
    pub usage: f64,
    pub skew: f64,
}

/// Which tessellation the archive uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QdMode {
    Grid,
    Cvt,
    /// Control: descriptors are hashed to a cell, so the partition carries no
    /// behavioural information. Used only for the random-descriptor control.
    Random,
}

impl QdMode {
    pub fn name(self) -> &'static str {
        match self {
            QdMode::Grid => "grid",
            QdMode::Cvt => "cvt",
            QdMode::Random => "random",
        }
    }
}

#[derive(Clone, Debug)]
pub struct QdEntry {
    pub tree: Node,
    pub error: f64,
    pub cost: f64,
    pub descriptor: Descriptor,
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
            a.error < b.error || (a.error == b.error && expr::structural_hash(&a.tree) < expr::structural_hash(&b.tree))
        }
    }

    /// Insert one candidate; returns whether the cell changed.
    fn insert(&mut self, entry: QdEntry, cap: usize) -> bool {
        if self.entries.iter().any(|e| Self::dominates(e, &entry)) {
            return false;
        }
        let before = self.entries.len();
        self.entries.retain(|e| !Self::dominates(&entry, e));
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
                    .then(expr::structural_hash(&a.tree).cmp(&expr::structural_hash(&b.tree)))
            });
            while self.entries.len() > cap.max(1) {
                self.entries.remove(self.entries.len() / 2);
            }
        }
        true
    }

    pub fn best(&self) -> Option<&QdEntry> {
        self.entries.iter().min_by(|a, b| {
            a.error
                .partial_cmp(&b.error)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cost.total_cmp(&b.cost))
        })
    }
}

/// The descriptor archive. `bins` is the grid side for [`QdMode::Grid`] and the
/// number of centroids `K` for [`QdMode::Cvt`].
#[derive(Clone, Debug)]
pub struct QdArchive {
    mode: QdMode,
    bins: usize,
    cell_cap: usize,
    cells: Vec<QdCell>,
    centroids: Vec<Descriptor>,
    /// How often each cell was offered a candidate (UCB exploration term).
    visits: Vec<u64>,
    /// Bounded deterministic sample used by the CVT Lloyd updates.
    observed: Vec<Descriptor>,
    records: usize,
}

/// The per-run coverage summary reported by the CLI and `--json`.
#[derive(Clone, Debug, PartialEq)]
pub struct QdReport {
    pub mode: &'static str,
    pub bins: usize,
    pub cells: usize,
    pub occupied: usize,
    pub coverage: f64,
    pub entries: usize,
    pub best_error: f64,
    pub cell_cap: usize,
    /// Dominated area of the archive's `(cost, error)` points, normalized by
    /// the observed cost and error range. Reported, not optimized directly.
    pub hypervolume: f64,
    /// Total candidates offered to the archive.
    pub evaluations: usize,
}

/// Below this many observed descriptors a CVT update has nothing to learn from.
const CVT_UPDATE_MIN: usize = 32;
/// Bound the Lloyd sample so a long run stays O(1) in descriptor memory.
const CVT_SAMPLE_CAP: usize = 4096;
/// Update centroids every this many `record` calls.
const CVT_UPDATE_EVERY: usize = 8;

fn halton(mut i: usize, base: usize) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

fn nearest_in(centroids: &[Descriptor], d: Descriptor) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let du = d.usage - c.usage;
            let ds = d.skew - c.skew;
            (i, du * du + ds * ds)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

impl QdArchive {
    /// Rectangular grid, `bins x bins` cells. The original increment.
    pub fn new(bins: usize, cell_cap: usize) -> Self {
        let bins = bins.max(2);
        QdArchive {
            mode: QdMode::Grid,
            bins,
            cell_cap: cell_cap.max(1),
            cells: vec![QdCell::default(); bins * bins],
            centroids: Vec::new(),
            visits: vec![0; bins * bins],
            observed: Vec::new(),
            records: 0,
        }
    }

    /// CVT-MAP-Elites with `cells` centroids. The initial centroids are a
    /// deterministic Halton set, so a run is reproducible without a separate
    /// centroid seed.
    pub fn new_cvt(cells: usize, cell_cap: usize) -> Self {
        let cells = cells.max(2);
        let centroids = (1..=cells)
            .map(|i| Descriptor {
                usage: halton(i, 2),
                skew: halton(i, 3),
            })
            .collect();
        QdArchive {
            mode: QdMode::Cvt,
            bins: cells,
            cell_cap: cell_cap.max(1),
            cells: vec![QdCell::default(); cells],
            centroids,
            visits: vec![0; cells],
            observed: Vec::new(),
            records: 0,
        }
    }

    /// Control: the same number of cells, but the descriptor is hashed to a
    /// cell. Any gain this control shows is search capacity, not behaviour.
    pub fn new_random(cells: usize, cell_cap: usize) -> Self {
        let cells = cells.max(2);
        QdArchive {
            mode: QdMode::Random,
            bins: cells,
            cell_cap: cell_cap.max(1),
            cells: vec![QdCell::default(); cells],
            centroids: Vec::new(),
            visits: vec![0; cells],
            observed: Vec::new(),
            records: 0,
        }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    pub fn mode(&self) -> QdMode {
        self.mode
    }

    pub fn centroids(&self) -> &[Descriptor] {
        &self.centroids
    }

    pub fn visits(&self) -> &[u64] {
        &self.visits
    }

    fn grid_index(&self, d: Descriptor) -> usize {
        let b = self.bins;
        let u = (d.usage.clamp(0.0, 1.0) * b as f64).floor() as usize;
        let s = (d.skew.clamp(0.0, 1.0) * b as f64).floor() as usize;
        u.min(b - 1) * b + s.min(b - 1)
    }

    fn nearest_centroid(&self, d: Descriptor) -> usize {
        nearest_in(&self.centroids, d)
    }

    fn cell_index(&self, d: Descriptor) -> usize {
        match self.mode {
            QdMode::Grid => self.grid_index(d),
            QdMode::Cvt => self.nearest_centroid(d),
            QdMode::Random => {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for word in [d.usage.to_bits(), d.skew.to_bits()] {
                    for b in word.to_le_bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                (h % self.cells.len() as u64) as usize
            }
        }
    }

    /// Insert one evaluated candidate. Returns whether the archive changed.
    pub fn insert(&mut self, tree: &Node, error: f64, cost: f64, sample: &Sample) -> bool {
        let d = descriptor(tree, sample);
        self.insert_descriptor(tree, error, cost, d)
    }

    /// Insert a candidate whose descriptor is already known. The search uses
    /// this after promotion so it does not evaluate the tree twice.
    pub fn insert_descriptor(
        &mut self,
        tree: &Node,
        error: f64,
        cost: f64,
        descriptor: Descriptor,
    ) -> bool {
        if !error.is_finite() {
            return false;
        }
        let ix = self.cell_index(descriptor);
        self.visits[ix] = self.visits[ix].saturating_add(1);
        if self.observed.len() < CVT_SAMPLE_CAP {
            self.observed.push(descriptor);
        } else {
            // Deterministic bounded reservoir: replace at a stride, never with
            // an RNG, so the centroid path is reproducible.
            let slot = self.records % CVT_SAMPLE_CAP;
            self.observed[slot] = descriptor;
        }
        self.records += 1;
        self.cells[ix].insert(
            QdEntry {
                tree: tree.clone(),
                error,
                cost,
                descriptor,
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
        self.maybe_recenter();
    }

    /// One deterministic Lloyd step after enough observations. Existing cell
    /// elites are re-assigned to the new nearest centroid; their `(error, cost)`
    /// fronts are preserved.
    pub fn recenter(&mut self) -> bool {
        if self.mode != QdMode::Cvt || self.observed.len() < CVT_UPDATE_MIN {
            return false;
        }
        let k = self.centroids.len();
        let mut moved = false;
        for _ in 0..4 {
            let mut sums = vec![(0.0f64, 0.0f64, 0usize); k];
            for d in &self.observed {
                let i = self.nearest_centroid(*d);
                sums[i].0 += d.usage;
                sums[i].1 += d.skew;
                sums[i].2 += 1;
            }
            for i in 0..k {
                if sums[i].2 > 0 {
                    let c = Descriptor {
                        usage: sums[i].0 / sums[i].2 as f64,
                        skew: sums[i].1 / sums[i].2 as f64,
                    };
                    if c != self.centroids[i] {
                        self.centroids[i] = c;
                        moved = true;
                    }
                }
            }
        }
        if moved {
            // Re-assign existing elites to the moved centroids without losing
            // their local Pareto fronts. The centroid copy ends the mutable
            // borrow of `self.cells` before the rebuild.
            let centroids = self.centroids.clone();
            let mut rebuilt: Vec<QdCell> = vec![QdCell::default(); k];
            let mut all: Vec<QdEntry> = Vec::new();
            for cell in &mut self.cells {
                all.append(&mut cell.entries);
            }
            for e in all {
                let ix = nearest_in(&centroids, e.descriptor);
                rebuilt[ix].entries.push(e);
            }
            for cell in &mut rebuilt {
                cell.entries.sort_by(|a, b| {
                    a.cost
                        .total_cmp(&b.cost)
                        .then(a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
                });
                while cell.entries.len() > self.cell_cap {
                    cell.entries.remove(cell.entries.len() / 2);
                }
            }
            self.cells = rebuilt;
        }
        moved
    }

    fn maybe_recenter(&mut self) {
        if self.mode == QdMode::Cvt && self.records % CVT_UPDATE_EVERY == 0 {
            let _ = self.recenter();
        }
    }

    pub fn report(&self) -> QdReport {
        let mut occupied = 0;
        let mut entries = 0;
        let mut best = f64::INFINITY;
        let mut points: Vec<(f64, f64)> = Vec::new();
        for cell in &self.cells {
            if !cell.entries.is_empty() {
                occupied += 1;
                entries += cell.entries.len();
                for e in &cell.entries {
                    best = best.min(e.error);
                    if e.error.is_finite() && e.cost.is_finite() {
                        points.push((e.cost, e.error));
                    }
                }
            }
        }
        let total = self.cells.len();
        QdReport {
            mode: self.mode.name(),
            bins: self.bins,
            cells: total,
            occupied,
            coverage: occupied as f64 / total as f64,
            entries,
            best_error: best,
            cell_cap: self.cell_cap,
            hypervolume: hypervolume(&points),
            evaluations: self.records,
        }
    }

    /// Occupied cells in index order, each with its local front.
    pub fn occupied_cells(&self) -> Vec<(usize, &QdCell)> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.entries.is_empty())
            .collect()
    }

    /// The UCB1-scored cell: lower local error is better, and a rarely offered
    /// cell earns an exploration bonus. Deterministic for a given archive.
    pub fn ucb_index(&self) -> Option<usize> {
        let occupied: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.cells[i].entries.is_empty())
            .collect();
        if occupied.is_empty() {
            return None;
        }
        let total = self.records.max(1) as f64;
        occupied.into_iter().min_by(|&a, &b| {
            let score = |i: usize| {
                let best = self.cells[i]
                    .best()
                    .map(|e| e.error)
                    .unwrap_or(f64::INFINITY);
                let bonus = 2.0 * ((total + 1.0).ln() / (self.visits[i] + 1) as f64).sqrt();
                best - bonus
            };
            score(a)
                .partial_cmp(&score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        })
    }

    /// A parent tree from the QD archive using the issue's mix: 60% UCB
    /// quality, 25% uniform occupied cell, 10% ring neighbour, 5% any elite.
    /// `None` falls back to the ordinary island tournament in the caller.
    pub fn select_parent(&self, rng: &mut Mt) -> Option<&Node> {
        self.select_parent_with_mix(rng, true)
    }

    /// Uniform-cell control: every occupied cell has equal probability.
    pub fn select_parent_uniform(&self, rng: &mut Mt) -> Option<&Node> {
        self.select_parent_with_mix(rng, false)
    }

    fn select_parent_with_mix(&self, rng: &mut Mt, mixed: bool) -> Option<&Node> {
        let occupied: Vec<usize> = (0..self.cells.len())
            .filter(|&i| !self.cells[i].entries.is_empty())
            .collect();
        if occupied.is_empty() {
            return None;
        }
        let roll = rng.next_f64();
        let ix = if !mixed {
            occupied[rng.usize_below(occupied.len())]
        } else if roll < 0.60 {
            self.ucb_index().unwrap_or(occupied[0])
        } else if roll < 0.85 {
            occupied[rng.usize_below(occupied.len())]
        } else if roll < 0.95 {
            let base = occupied[rng.usize_below(occupied.len())];
            (base + 1) % self.cells.len()
        } else {
            occupied[rng.usize_below(occupied.len())]
        };
        let cell = &self.cells[ix];
        if cell.entries.is_empty() {
            return None;
        }
        let j = rng.usize_below(cell.entries.len());
        Some(&cell.entries[j].tree)
    }

    /// The least-visited region's descriptor, used to aim restarts.
    pub fn least_occupied_descriptor(&self) -> Option<Descriptor> {
        match self.mode {
            QdMode::Grid => {
                let i = (0..self.visits.len()).min_by_key(|&i| self.visits[i])?;
                let b = self.bins;
                Some(Descriptor {
                    usage: (i / b) as f64 / b as f64 + 0.5 / b as f64,
                    skew: (i % b) as f64 / b as f64 + 0.5 / b as f64,
                })
            }
            QdMode::Cvt => {
                let i = (0..self.visits.len()).min_by_key(|&i| self.visits[i])?;
                self.centroids.get(i).copied()
            }
            QdMode::Random => {
                let i = (0..self.visits.len()).min_by_key(|&i| self.visits[i])?;
                Some(Descriptor {
                    usage: halton(i + 1, 2),
                    skew: halton(i + 1, 3),
                })
            }
        }
    }

    /// Rebuild the archive from its trees with an exact scorer. Called at the
    /// end of a run so every reported cell elite is an all-rows f64 value.
    pub fn refit_all<F>(&mut self, score: F)
    where
        F: Fn(&Node) -> (f64, f64),
    {
        let mut all: Vec<QdEntry> = Vec::new();
        for cell in &mut self.cells {
            all.append(&mut cell.entries);
        }
        self.visits.iter_mut().for_each(|v| *v = 0);
        for e in all {
            let (error, cost) = score(&e.tree);
            let ix = self.cell_index(e.descriptor);
            self.visits[ix] = self.visits[ix].saturating_add(1);
            self.cells[ix].insert(
                QdEntry {
                    tree: e.tree,
                    error,
                    cost,
                    descriptor: e.descriptor,
                },
                self.cell_cap,
            );
        }
    }
}

/// Normalized 2-D hypervolume of `(cost, error)` points. Both axes are mapped
/// to `[0,1]` by the observed extremes; a single point contributes zero area.
fn hypervolume(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let cmin = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let cmax = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let emin = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let emax = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let cspan = (cmax - cmin).max(1e-12);
    let espan = (emax - emin).max(1e-12);
    // A deterministic 64x64 occupancy count of the dominated region. It is
    // exact to one grid cell and cannot be fooled by duplicate costs or a
    // degenerate front, unlike a hand-rolled skyline sum.
    const G: usize = 64;
    let normalized: Vec<(f64, f64)> = points
        .iter()
        .map(|&(c, e)| ((c - cmin) / cspan, (e - emin) / espan))
        .collect();
    let mut dominated = 0usize;
    for gy in 0..G {
        let y = (gy as f64 + 0.5) / G as f64;
        for gx in 0..G {
            let x = (gx as f64 + 0.5) / G as f64;
            // Lower cost and lower error are better: a point at (x,y)
            // dominates every grid cell with gx >= x and gy >= y.
            if normalized.iter().any(|&(px, py)| px <= x && py <= y) {
                dominated += 1;
            }
        }
    }
    dominated as f64 / (G * G) as f64
}

/// Moment and sign statistics accumulated from one prediction vector. This is
/// the local/GPU-friendly form: one pass over `pred` yields everything the
/// descriptor needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PredictionStats {
    pub n: usize,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub sign_changes: usize,
}

impl PredictionStats {
    pub fn from_prediction(pred: &[f64]) -> Self {
        let n = pred.len();
        if n == 0 {
            return Self::default();
        }
        let mean = pred.iter().sum::<f64>() / n as f64;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        let mut sign_changes = 0;
        let mut prev = 0i8;
        for &p in pred {
            if !p.is_finite() {
                return Self::default();
            }
            let d = p - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
            let sign = if p > 0.0 { 1 } else if p < 0.0 { -1 } else { 0 };
            if prev != 0 && sign != 0 && sign != prev {
                sign_changes += 1;
            }
            if sign != 0 {
                prev = sign;
            }
        }
        Self {
            n,
            mean,
            m2: m2 / n as f64,
            m3: m3 / n as f64,
            m4: m4 / n as f64,
            sign_changes,
        }
    }

    pub fn skewness(&self) -> f64 {
        if self.m2 <= 1e-24 {
            0.0
        } else {
            self.m3 / self.m2.powf(1.5)
        }
    }

    pub fn kurtosis(&self) -> f64 {
        if self.m2 <= 1e-24 {
            0.0
        } else {
            self.m4 / (self.m2 * self.m2) - 3.0
        }
    }

    pub fn sign_change_rate(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.sign_changes as f64 / (self.n - 1) as f64
        }
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
    let stats = PredictionStats::from_prediction(pred);
    if stats.m2 <= 1e-24 {
        return 0.5;
    }
    let skew = stats.skewness();
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
    fn cvt_centroids_move_towards_observed_descriptors() {
        let s = sample();
        let mut q = QdArchive::new_cvt(16, 4);
        let before = q.centroids().to_vec();
        let tree = node(Op::Mul, vec![input(0), input(1)]);
        for i in 0..200 {
            let e = 1.0 + i as f64 * 1e-3;
            q.insert(&tree, e, 3.0, &s);
        }
        assert!(q.recenter(), "centroids must update after enough observations");
        assert_ne!(before, q.centroids().to_vec());
        assert_eq!(q.report().mode, "cvt");
        assert_eq!(q.report().cells, 16);
    }

    #[test]
    fn ucb_and_least_occupied_are_defined_and_deterministic() {
        let s = sample();
        let mut q = QdArchive::new_cvt(8, 2);
        for (e, t) in [
            (1.0, node(Op::Mul, vec![constant(2.0), input(0)])),
            (0.5, node(Op::Add, vec![constant(1.0), input(0)])),
            (2.0, node(Op::Mul, vec![input(0), input(1)])),
        ] {
            q.insert(&t, e, 3.0, &s);
        }
        assert!(q.ucb_index().is_some());
        assert!(q.least_occupied_descriptor().is_some());
        let mut r1 = Mt::new(7);
        let mut r2 = Mt::new(7);
        let a = q.select_parent(&mut r1).map(expr::structural_hash);
        let b = q.select_parent(&mut r2).map(expr::structural_hash);
        assert_eq!(a, b);
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
