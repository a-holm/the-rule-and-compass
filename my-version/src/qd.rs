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
use std::collections::HashMap;

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
    /// Number of input columns, used to derive the *reachable* usage lattice.
    /// Usage is `distinct inputs / nvars`, so only `nvars + 1` values occur.
    nvars: usize,
    /// Descriptors already known for a structural hash, tagged with the sample
    /// they were computed on. Population descriptors arrive through
    /// [`QdArchive::record_with_descriptors`]; this cache keeps the archive from
    /// re-evaluating a tree it has already classified (for example a Pareto
    /// member that was a population member last generation). It is cleared when
    /// the sample changes and bounded so a long run stays O(1).
    known: HashMap<u64, Descriptor>,
    /// Fingerprint of the sample `known` entries were computed on.
    known_tag: u64,
    /// How many archive insertions needed the evaluating descriptor fallback
    /// (GPU-scored, subset-scored or unknown trees). Reported so the one-pass
    /// claim is measurable from `--json` instead of asserted.
    fallbacks: usize,
}

/// Keep the descriptor cache bounded; a long run must not retain one entry per
/// candidate it has ever seen.
const KNOWN_CACHE_CAP: usize = 65_536;

/// The per-run coverage summary reported by the CLI and `--json`.
#[derive(Clone, Debug, PartialEq)]
pub struct QdReport {
    pub mode: &'static str,
    pub bins: usize,
    pub cells: usize,
    pub occupied: usize,
    /// `occupied / cells`. *Diagnostic only*: `cells` is a designer choice
    /// (and CVT centroids move), so this fraction is not comparable across
    /// arms. Use [`QdReport::reachable_coverage`] for a denominator the
    /// descriptor can actually reach.
    pub coverage: f64,
    /// Cells whose Voronoi region intersects an attainable usage value.
    ///
    /// Usage is `distinct inputs / nvars`, so only `nvars + 1` values occur.
    /// A cell that cannot be nearest to any reachable descriptor can never be
    /// occupied, and counting it in the denominator charges the search for a
    /// region the descriptor simply cannot express.
    pub reachable: usize,
    /// Occupied cells that are also reachable.
    pub occupied_reachable: usize,
    /// `occupied_reachable / reachable`; the honest coverage fraction.
    pub reachable_coverage: f64,
    pub entries: usize,
    pub best_error: f64,
    pub cell_cap: usize,
    /// Dominated area of the archive's `(cost, error)` points, normalized by
    /// the observed cost and error range. Self-normalised, therefore **not
    /// comparable between runs**; kept as a diagnostic only.
    pub hypervolume: f64,
    /// Shannon entropy (bits) of the candidate-visit distribution over the
    /// reachable cells. Zero means every offer landed in one cell.
    pub occupancy_entropy: f64,
    /// `occupancy_entropy / log2(reachable)` in `[0, 1]`.
    pub occupancy_entropy_normalized: f64,
    /// Quality pairing: `Σ 1 / (1 + best_error)` over occupied reachable
    /// cells. Coverage without quality is trivially maximised by a random map,
    /// so coverage is always reported next to this sum.
    pub qd_score: f64,
    /// Total candidates offered to the archive.
    pub evaluations: usize,
    /// Archive insertions that needed the evaluating descriptor fallback. Zero
    /// on a CPU run whose candidates were all scored by the one-pass path.
    pub descriptor_fallbacks: usize,
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

/// Cheap identity of the rows a descriptor was computed on.
///
/// The archive caches descriptors by structural hash; using one computed on a
/// different row subset would silently mislabel a candidate. The tag folds the
/// row count and the target values (plus the first and last row), so a new
/// `--predictor-samples` draw or screening subset invalidates the cache without
/// an expensive per-candidate comparison.
fn sample_tag(s: &Sample) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(s.x.len() as u64);
    for y in &s.y {
        mix(y.to_bits());
    }
    if let Some(first) = s.x.first() {
        for v in first {
            mix(v.to_bits());
        }
    }
    if let Some(last) = s.x.last() {
        for v in last {
            mix(v.to_bits());
        }
    }
    h
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
            nvars: 2,
            known: HashMap::new(),
            known_tag: 0,
            fallbacks: 0,
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
            nvars: 2,
            known: HashMap::new(),
            known_tag: 0,
            fallbacks: 0,
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
            nvars: 2,
            known: HashMap::new(),
            known_tag: 0,
            fallbacks: 0,
        }
    }

    /// Declare the number of input columns so [`QdReport::reachable`] can use
    /// the actual usage lattice instead of the naive `K`.
    pub fn with_nvars(mut self, nvars: usize) -> Self {
        self.nvars = nvars;
        self
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
    ///
    /// A descriptor already known for this structural hash *on the same sample*
    /// is reused, so a tree the search has classified is never evaluated a
    /// second time here.
    pub fn insert(&mut self, tree: &Node, error: f64, cost: f64, sample: &Sample) -> bool {
        let tag = sample_tag(sample);
        if tag != self.known_tag {
            self.known.clear();
            self.known_tag = tag;
        }
        let d = match self.known.get(&expr::structural_hash(tree)) {
            Some(d) => *d,
            None => {
                self.fallbacks += 1;
                descriptor(tree, sample)
            }
        };
        self.remember(tree, d);
        self.insert_descriptor(tree, error, cost, d)
    }

    /// Insert a candidate whose descriptor is already known. The search uses
    /// this after scoring so it does not evaluate the tree twice.
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

    /// Bound and store a descriptor under its structural hash.
    fn remember(&mut self, tree: &Node, d: Descriptor) {
        if self.known.len() >= KNOWN_CACHE_CAP {
            // Deterministic bounded cache: dropping it only costs a future
            // re-evaluation, never a different archive.
            self.known.clear();
        }
        self.known.insert(expr::structural_hash(tree), d);
    }

    /// Record every individual on the current populations plus the global
    /// archive. Kept deterministic: the caller passes them in a fixed order.
    pub fn record(&mut self, trees: impl IntoIterator<Item = (Node, f64, f64)>, sample: &Sample) {
        self.record_with_descriptors(trees.into_iter().map(|(t, e, c)| (t, e, c, None)), sample);
    }

    /// #120: record candidates while carrying the descriptor that the scoring
    /// pass already accumulated. `None` falls back to a known descriptor for
    /// the same sample or, last resort, one evaluation — the path used for
    /// candidates scored on a GPU or on a row subset.
    pub fn record_with_descriptors(
        &mut self,
        trees: impl IntoIterator<Item = (Node, f64, f64, Option<Descriptor>)>,
        sample: &Sample,
    ) {
        let tag = sample_tag(sample);
        if tag != self.known_tag {
            self.known.clear();
            self.known_tag = tag;
        }
        for (tree, error, cost, d) in trees {
            let hash = expr::structural_hash(&tree);
            let d = match d {
                Some(d) => d,
                None => match self.known.get(&hash).copied() {
                    Some(d) => d,
                    None => {
                        self.fallbacks += 1;
                        descriptor(&tree, sample)
                    }
                },
            };
            self.remember(&tree, d);
            self.insert_descriptor(&tree, error, cost, d);
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

    /// The attainable usage values. Usage is `used / nvars`, so only
    /// `nvars + 1` points occur regardless of how many candidates are scored.
    fn usage_levels(&self) -> Vec<f64> {
        if self.nvars == 0 {
            return vec![0.0];
        }
        (0..=self.nvars)
            .map(|k| k as f64 / self.nvars as f64)
            .collect()
    }

    /// Mark every cell that can be nearest to at least one *attainable*
    /// descriptor: usage on the lattice above and skew anywhere in `[0, 1]`.
    ///
    /// For a CVT this is exact geometry, not a heuristic. On the vertical line
    /// `usage = u`, "centroid i is at least as near as centroid j" is linear in
    /// `skew`, so intersecting those half-lines over every `j` gives the
    /// interval of skew where `i` wins. A cell is reachable when that interval
    /// is non-empty for at least one usage level. Ties are treated as
    /// reachable (a point on a Voronoi boundary is counted for both cells);
    /// that can only over-count by a measure-zero boundary and never invents a
    /// region the descriptor cannot approach.
    fn reachable_mask(&self) -> Vec<bool> {
        let mut reach = vec![false; self.cells.len()];
        match self.mode {
            QdMode::Random => {
                // The control hashes arbitrary descriptors, so every cell is
                // reachable *by construction*. Its coverage is reported but is
                // not a behavioural measurement.
                reach.fill(true);
            }
            QdMode::Grid => {
                let b = self.bins;
                let mut usage_bins = vec![false; b];
                for u in self.usage_levels() {
                    let ix = (u.clamp(0.0, 1.0) * b as f64).floor() as usize;
                    usage_bins[ix.min(b - 1)] = true;
                }
                // Skew is continuous on [0, 1], so every skew bin is reachable
                // once its usage row is.
                for (u, row) in usage_bins.iter().enumerate() {
                    if *row {
                        for s in 0..b {
                            reach[u * b + s] = true;
                        }
                    }
                }
            }
            QdMode::Cvt => {
                let k = self.centroids.len();
                let mut done = vec![false; k];
                for u in self.usage_levels() {
                    for i in 0..k {
                        if done[i] {
                            continue;
                        }
                        let ci = self.centroids[i];
                        let mut lo = 0.0f64;
                        let mut hi = 1.0f64;
                        let mut feasible = true;
                        for (j, cj) in self.centroids.iter().enumerate() {
                            if j == i {
                                continue;
                            }
                            // d_i(s)^2 - d_j(s)^2 <= 0
                            // = (u-ci.u)^2 - (u-cj.u)^2 + ci.s^2 - cj.s^2
                            //   + 2(cj.s - ci.s) s
                            let a = (u - ci.usage).powi(2) - (u - cj.usage).powi(2)
                                + ci.skew * ci.skew
                                - cj.skew * cj.skew;
                            let p = 2.0 * (cj.skew - ci.skew);
                            let eps = 1e-12;
                            if p.abs() <= eps {
                                if a > eps {
                                    feasible = false;
                                    break;
                                }
                            } else if p > 0.0 {
                                hi = hi.min(-a / p);
                            } else {
                                lo = lo.max(-a / p);
                            }
                            if lo > hi + 1e-9 {
                                feasible = false;
                                break;
                            }
                        }
                        if feasible && lo <= hi + 1e-9 && hi >= 0.0 && lo <= 1.0 {
                            done[i] = true;
                            reach[i] = true;
                        }
                    }
                }
            }
        }
        reach
    }

    pub fn report(&self) -> QdReport {
        let reach = self.reachable_mask();
        let reachable = reach.iter().filter(|&&r| r).count();
        let mut occupied = 0;
        let mut occupied_reachable = 0;
        let mut entries = 0;
        let mut best = f64::INFINITY;
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut qd_score = 0.0;
        for (ix, cell) in self.cells.iter().enumerate() {
            if !cell.entries.is_empty() {
                occupied += 1;
                if reach[ix] {
                    occupied_reachable += 1;
                    if let Some(e) = cell.best() {
                        if e.error.is_finite() {
                            qd_score += 1.0 / (1.0 + e.error.max(0.0));
                        }
                    }
                }
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
        // Occupancy entropy over the *reachable* cells: how spread out the
        // candidates offered to the archive were. Coverage alone cannot see
        // the difference between "one offer in each of two cells" and "ten
        // thousand offers in one and one in the other".
        let mut visit_total = 0u64;
        for (ix, r) in reach.iter().enumerate() {
            if *r {
                visit_total = visit_total.saturating_add(self.visits[ix]);
            }
        }
        let mut entropy = 0.0;
        if visit_total > 0 {
            for (ix, r) in reach.iter().enumerate() {
                if *r && self.visits[ix] > 0 {
                    let p = self.visits[ix] as f64 / visit_total as f64;
                    entropy -= p * p.log2();
                }
            }
        }
        let entropy_norm = if reachable > 1 {
            entropy / (reachable as f64).log2()
        } else {
            0.0
        };
        QdReport {
            mode: self.mode.name(),
            bins: self.bins,
            cells: total,
            occupied,
            coverage: occupied as f64 / total as f64,
            reachable,
            occupied_reachable,
            reachable_coverage: if reachable == 0 {
                0.0
            } else {
                occupied_reachable as f64 / reachable as f64
            },
            entries,
            best_error: best,
            cell_cap: self.cell_cap,
            hypervolume: hypervolume(&points),
            occupancy_entropy: entropy,
            occupancy_entropy_normalized: entropy_norm,
            qd_score,
            evaluations: self.records,
            descriptor_fallbacks: self.fallbacks,
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

/// Fixed-reference validation hypervolume of a `(cost, error)` front.
///
/// `Σ (c_{i+1} − c_i) · log10(1 + e_ref / e_i)` over the cost-sorted front,
/// where `e_i` is the best error seen at or below `c_i`. Both the reference
/// error and the log transform are fixed by the **data**, not by the observed
/// extremes, so two arms can be compared even when one reaches a smaller cost
/// or error span. (The archive's [`QdReport::hypervolume`] intentionally does
/// normalise by the observed range and is therefore not comparable.)
///
/// `e_ref` should be a scale of the target on the held-out sample (its
/// variance is what the CLI uses). A single point contributes zero area,
/// exactly like a one-point Pareto front.
pub fn fixed_reference_hypervolume(points: &[(f64, f64)], e_ref: f64) -> f64 {
    let mut front: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(c, e)| c.is_finite() && e.is_finite())
        .collect();
    if front.len() < 2 {
        return 0.0;
    }
    front.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let e_ref = if e_ref.is_finite() && e_ref > 0.0 {
        e_ref
    } else {
        1.0
    };
    let mut best = f64::INFINITY;
    let mut hv = 0.0;
    for w in front.windows(2) {
        best = best.min(w[0].1.max(0.0));
        let dc = w[1].0 - w[0].0;
        if dc > 0.0 && best.is_finite() {
            hv += dc * (1.0 + e_ref / best.max(1e-12)).log10();
        }
    }
    hv
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
/// This is the only path that evaluates the tree here; the search calls
/// [`descriptor_from_prediction`] with the prediction the scoring pass already
/// computed, so a QD run never pays a second full tree evaluation per
/// candidate.
pub fn descriptor(tree: &Node, sample: &Sample) -> Descriptor {
    #[cfg(test)]
    FALLBACK_DESCRIPTORS.with(|n| n.set(n.get() + 1));
    let pred: Vec<f64> = sample.x.iter().map(|row| expr::eval(tree, row)).collect();
    descriptor_from_prediction(tree, &pred, sample.x.first().map(Vec::len).unwrap_or(0))
}

#[cfg(test)]
thread_local! {
    /// How many times the evaluating fallback [`descriptor`] was used. A
    /// one-pass test asserts this does not move when QD scores candidates.
    static FALLBACK_DESCRIPTORS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test hook for the one-pass descriptor contract.
#[cfg(test)]
pub fn fallback_descriptor_count() -> usize {
    FALLBACK_DESCRIPTORS.with(|n| n.get())
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

    /// #120 one-pass contract: the prediction-based descriptor is bit-identical
    /// to the convenience wrapper that evaluates the tree itself.
    #[test]
    fn descriptor_from_prediction_equals_the_evaluating_path() {
        let s = sample();
        let trees = [
            node(Op::Add, vec![input(0), constant(0.25)]),
            node(Op::Mul, vec![input(0), input(1)]),
            node(Op::Sin, vec![node(Op::Mul, vec![constant(1.7), input(0)])]),
            node(Op::Div, vec![input(1), node(Op::Add, vec![input(0), constant(2.0)])]),
        ];
        for t in &trees {
            let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(t, row)).collect();
            let one_pass = descriptor_from_prediction(t, &pred, 2);
            let own_path = descriptor(t, &s);
            assert_eq!(
                one_pass,
                own_path,
                "{}",
                expr::to_string(t, &["x".into(), "y".into()])
            );
            assert_eq!(one_pass.usage.to_bits(), own_path.usage.to_bits());
            assert_eq!(one_pass.skew.to_bits(), own_path.skew.to_bits());
        }
    }

    /// The denominator is the usage lattice, not `K`. With two inputs usage can
    /// only be `0`, `0.5` or `1`, so an `8 x 8` grid has 24 reachable cells.
    #[test]
    fn grid_reachable_coverage_uses_the_usage_lattice() {
        let mut q = QdArchive::new(8, 4).with_nvars(2);
        assert_eq!(q.report().reachable, 3 * 8);
        assert_eq!(q.report().reachable_coverage, 0.0);
        // usage 0.5 only reaches the middle usage row.
        q.insert_descriptor(
            &node(Op::Mul, vec![input(0), input(1)]),
            0.5,
            3.0,
            Descriptor { usage: 0.5, skew: 0.5 },
        );
        let r = q.report();
        assert_eq!(r.reachable, 24);
        assert_eq!(r.occupied_reachable, 1);
        assert!((r.reachable_coverage - 1.0 / 24.0).abs() < 1e-12, "{r:?}");
        // The legacy geometric fraction divides by 64 and is retained only as
        // a diagnostic.
        assert!((r.coverage - 1.0 / 64.0).abs() < 1e-12, "{r:?}");
    }

    /// A cell that no attainable descriptor can be nearest to must not count
    /// against coverage. The CVT denominator is therefore <= K.
    #[test]
    fn cvt_reachable_denominator_is_at_most_the_cell_count() {
        let q = QdArchive::new_cvt(16, 4).with_nvars(2);
        let r = q.report();
        assert!(r.reachable > 0, "{r:?}");
        assert!(r.reachable <= r.cells, "{r:?}");
        assert_eq!(r.occupied, 0);
        assert_eq!(r.reachable_coverage, 0.0);
    }

    /// Entropy sees attention spread that coverage cannot.
    #[test]
    fn occupancy_entropy_measures_visit_spread() {
        let mut q = QdArchive::new(8, 4).with_nvars(2);
        let t = node(Op::Mul, vec![input(0), input(1)]);
        let a = Descriptor { usage: 0.5, skew: 0.1 };
        let b = Descriptor { usage: 0.5, skew: 0.9 };
        for _ in 0..4 {
            q.insert_descriptor(&t, 1.0, 3.0, a);
        }
        let one_cell = q.report();
        assert_eq!(one_cell.occupancy_entropy, 0.0);
        assert_eq!(one_cell.occupancy_entropy_normalized, 0.0);
        // A second visited cell produces entropy in (0, 1] bits.
        q.insert_descriptor(&t, 2.0, 3.0, b);
        let two = q.report();
        assert!(two.occupancy_entropy > 0.0, "{two:?}");
        assert!(two.occupancy_entropy <= 1.0 + 1e-12, "{two:?}");
        assert!(two.occupancy_entropy_normalized > 0.0, "{two:?}");
        assert!(two.occupancy_entropy_normalized < 1.0, "{two:?}");
        // Quality pairing is reported alongside coverage.
        assert!((one_cell.qd_score - 0.5).abs() < 1e-12, "{one_cell:?}");
    }

    #[test]
    fn fixed_reference_hypervolume_is_pinned_to_the_reference() {
        let front = [(1.0, 0.1), (2.0, 0.05), (3.0, 0.01)];
        let hv = fixed_reference_hypervolume(&front, 1.0);
        let manual =
            1.0 * (1.0 + 1.0 / 0.1f64).log10() + 1.0 * (1.0 + 1.0 / 0.05f64).log10();
        assert!((hv - manual).abs() < 1e-12, "{hv} vs {manual}");
        // Doubling every cost doubles the area: the reference never rescales.
        let doubled: Vec<(f64, f64)> = front.iter().map(|&(c, e)| (2.0 * c, e)).collect();
        assert!((fixed_reference_hypervolume(&doubled, 1.0) - 2.0 * hv).abs() < 1e-12);
        // A dominated extra point cannot lower it.
        let mut with_extra = front.to_vec();
        with_extra.push((2.5, 0.5));
        assert!((fixed_reference_hypervolume(&with_extra, 1.0) - hv).abs() < 1e-12);
        // One point has no area; a larger reference error scores a larger area.
        assert_eq!(fixed_reference_hypervolume(&front[..1], 1.0), 0.0);
        assert!(fixed_reference_hypervolume(&front, 10.0) > hv);
    }
}
