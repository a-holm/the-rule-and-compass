//! Convergence statistics over the Pareto front (issue #18).
//!
//! Four bounded numbers describe whether the *search* has settled:
//!
//! * **confidence** — how little the best error moved over the last `w`
//!   generations;
//! * **stability** — how much the structural forms on the front survive
//!   between windows;
//! * **maturity** — how long the oldest continuously present form has been on
//!   the front, normalised by the generation;
//! * **percent converged** — a weighted headline combining the three.
//!
//! All four are computed from the archive alone, once per generation: the cost
//! is O(front size) and no training row is touched. The definitions, the
//! window, the weights and the evidence on easy/hard problems are documented
//! in `docs/convergence.md`.
//!
//! Nothing here changes the search: the caller calls [`ProgressStats::observe`]
//! once per generation and reports the returned [`Convergence`].

use crate::pareto::Archive;
use crate::stability::shape;
use std::collections::{HashMap, HashSet, VecDeque};

/// The default comparison window `w`, in generations.
///
/// The window is deliberately a documented constant rather than a CLI flag:
/// convergence is a *reading* of the search, not a tuning knob. Twenty
/// generations is long enough that one lucky generation does not dominate and
/// short enough to react on the default 200-generation budget.
pub const DEFAULT_WINDOW: usize = 20;

/// Weight of [`Convergence::confidence`] in `percent_converged`.
pub const CONFIDENCE_WEIGHT: f64 = 0.5;
/// Weight of [`Convergence::stability`] in `percent_converged`.
pub const STABILITY_WEIGHT: f64 = 0.3;
/// Weight of [`Convergence::maturity`] in `percent_converged`.
pub const MATURITY_WEIGHT: f64 = 0.2;

/// `percent_converged` at or above this value is called "converged".
pub const CONVERGED_THRESHOLD: f64 = 0.75;

/// Scale of the relative-improvement map used by confidence.
///
/// Confidence is `exp(-r / 1e-3)`, where `r` is the relative improvement of
/// the best error over the window. A window that improves by 0.1 % therefore
/// scores `e^-1 ≈ 0.37`; no improvement scores `1`. For comparison, the
/// search's own patience test calls `r <= 1e-9` "no improvement".
pub const CONFIDENCE_SCALE: f64 = 1e-3;

/// The error at or below which the search considers the problem solved (the
/// same floor its early-stop rule uses). A solved front has nothing left to
/// improve, so confidence is 1 even if the last window contained the jump to
/// the exact answer.
pub const SOLVED_ERROR: f64 = 1e-24;

/// The four convergence readings for one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Convergence {
    /// `0..=1`: 1 when the best error did not move over the window, 0 when it
    /// improved by a large relative amount (or when there is no history yet).
    pub confidence: f64,
    /// `0..=1`: mean Jaccard overlap of the front's structural forms between
    /// each generation in the window and the generation `w` before it.
    pub stability: f64,
    /// `0..=1`: age of the oldest continuously surviving form on the front,
    /// divided by the current generation.
    pub maturity: f64,
    /// `0..=1`: `0.5*confidence + 0.3*stability + 0.2*maturity`.
    pub percent_converged: f64,
}

impl Convergence {
    /// Combine the three components with the documented weights, clamping
    /// every input (and the result) to `0..=1`.
    pub fn from_parts(confidence: f64, stability: f64, maturity: f64) -> Self {
        let confidence = clamp01(confidence);
        let stability = clamp01(stability);
        let maturity = clamp01(maturity);
        Convergence {
            confidence,
            stability,
            maturity,
            percent_converged: clamp01(
                CONFIDENCE_WEIGHT * confidence
                    + STABILITY_WEIGHT * stability
                    + MATURITY_WEIGHT * maturity,
            ),
        }
    }

    /// True when the headline number has reached [`CONVERGED_THRESHOLD`], or
    /// when the problem is **solved** (`confidence == 1.0`, i.e. the best error
    /// is below `SOLVED_ERROR`).
    ///
    /// Solving is a sufficient condition because a converged search can keep
    /// discovering alternative forms with the same error (`x`, `exp(c*x)`,
    /// `c + c*x`), which lowers the structural `stability` without meaning the
    /// search is still moving. Issue #84.
    pub fn is_converged(&self) -> bool {
        self.percent_converged >= CONVERGED_THRESHOLD || self.confidence >= 1.0
    }

    /// The JSON object shared by `--json`, `--ndjson` and `--progress`.
    pub fn json(&self) -> String {
        format!(
            "{{\"confidence\":{},\"stability\":{},\"maturity\":{},\"percent_converged\":{}}}",
            num(self.confidence),
            num(self.stability),
            num(self.maturity),
            num(self.percent_converged)
        )
    }
}

/// The state of a [`ProgressStats`] after some observations, in a form a
/// checkpoint stores exactly: every float is kept as is (the checkpoint writes
/// bit patterns) and every set is sorted, so equal states give equal bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgressSnapshot {
    pub window: usize,
    pub best: Vec<(usize, f64)>,
    /// One sorted list of form hashes per remembered generation.
    pub fronts: Vec<Vec<u64>>,
    pub jaccards: Vec<f64>,
    pub jaccard_sum: f64,
    /// `(form, first generation of its current run)`, sorted by form.
    pub seen: Vec<(u64, usize)>,
    pub current: Convergence,
}

/// Incremental convergence tracker. Cheap by construction: each observation
/// hashes the current front (O(front size)) and keeps at most `w + 1` small
/// snapshots.
pub struct ProgressStats {
    window: usize,
    /// `(generation, best error)` for at most `w + 1` generations.
    best: VecDeque<(usize, f64)>,
    /// Structural-form sets for at most `w + 1` generations.
    fronts: VecDeque<HashSet<u64>>,
    /// The last `w` Jaccard values, and their running sum.
    jaccards: VecDeque<f64>,
    jaccard_sum: f64,
    /// First generation of each structural form's *current* continuous run on
    /// the front. Forms that leave and return start again.
    seen: HashMap<u64, usize>,
    current: Convergence,
}

impl ProgressStats {
    /// A tracker with comparison window `window` generations (`>= 1`).
    pub fn new(window: usize) -> Self {
        ProgressStats {
            window: window.max(1),
            best: VecDeque::new(),
            fronts: VecDeque::new(),
            jaccards: VecDeque::new(),
            jaccard_sum: 0.0,
            seen: HashMap::new(),
            current: Convergence::default(),
        }
    }

    /// The comparison window in generations.
    pub fn window(&self) -> usize {
        self.window
    }

    /// The full tracker state, for a checkpoint.
    pub fn snapshot(&self) -> ProgressSnapshot {
        let mut seen: Vec<(u64, usize)> = self.seen.iter().map(|(k, v)| (*k, *v)).collect();
        seen.sort_unstable();
        ProgressSnapshot {
            window: self.window,
            best: self.best.iter().copied().collect(),
            fronts: self
                .fronts
                .iter()
                .map(|set| {
                    let mut forms: Vec<u64> = set.iter().copied().collect();
                    forms.sort_unstable();
                    forms
                })
                .collect(),
            jaccards: self.jaccards.iter().copied().collect(),
            jaccard_sum: self.jaccard_sum,
            seen,
            current: self.current,
        }
    }

    /// Continue from a saved state. `None` when the state was kept under a
    /// different window (it would not continue the same readings).
    pub fn restore(snapshot: &ProgressSnapshot, window: usize) -> Option<Self> {
        if snapshot.window != window.max(1) {
            return None;
        }
        Some(ProgressStats {
            window: snapshot.window,
            best: snapshot.best.iter().copied().collect(),
            fronts: snapshot
                .fronts
                .iter()
                .map(|forms| forms.iter().copied().collect())
                .collect(),
            jaccards: snapshot.jaccards.iter().copied().collect(),
            jaccard_sum: snapshot.jaccard_sum,
            seen: snapshot.seen.iter().copied().collect(),
            current: snapshot.current,
        })
    }

    /// True once [`ProgressStats::observe`] has run at least once.
    pub fn has_observed(&self) -> bool {
        !self.best.is_empty()
    }

    /// The tracker a search starts with after `generations_done` completed
    /// generations, and whether its readings are exact. They are exact when
    /// nothing has run yet or when the saved state restores; a checkpoint from
    /// before the state was saved leaves a fresh tracker and `false`.
    pub fn resume(
        saved: Option<&ProgressSnapshot>,
        window: usize,
        generations_done: usize,
    ) -> (Self, bool) {
        match saved.and_then(|s| ProgressStats::restore(s, window)) {
            Some(stats) => (stats, true),
            None => (ProgressStats::new(window), generations_done == 0),
        }
    }

    /// Observe one generation: its best error and its Pareto front.
    ///
    /// Call at most once per generation, at the generation boundary. The
    /// return value is the same reading as [`ProgressStats::current`].
    pub fn observe(&mut self, generation: usize, best_error: f64, archive: &Archive) -> Convergence {
        let forms: HashSet<u64> = archive
            .items
            .iter()
            .map(|s| form_hash(&s.tree))
            .collect();

        // --- confidence: relative movement of the best error over the window.
        self.best.push_back((generation, best_error));
        while self.best.len() > self.window + 1 {
            self.best.pop_front();
        }
        let confidence = if best_error.is_finite() && best_error <= SOLVED_ERROR {
            // Exact (machine-precision) solution: there is nothing left to
            // improve, so the best error is as converged as it can get.
            1.0
        } else {
            match self.best.front().copied() {
                Some((first_generation, first_error)) => {
                    let span = generation.saturating_sub(first_generation);
                    if span == 0 || !best_error.is_finite() || !first_error.is_finite() {
                        0.0
                    } else {
                        let relative =
                            ((first_error - best_error) / (1.0 + first_error.abs())).max(0.0);
                        (-relative / CONFIDENCE_SCALE).exp()
                    }
                }
                None => 0.0,
            }
        };

        // --- stability: mean Jaccard overlap of structural forms.
        self.fronts.push_back(forms.clone());
        while self.fronts.len() > self.window + 1 {
            self.fronts.pop_front();
        }
        if self.fronts.len() >= 2 {
            let j = jaccard(
                self.fronts.front().expect("len >= 2"),
                self.fronts.back().expect("len >= 2"),
            );
            self.jaccards.push_back(j);
            self.jaccard_sum += j;
            while self.jaccards.len() > self.window {
                if let Some(old) = self.jaccards.pop_front() {
                    self.jaccard_sum -= old;
                }
            }
        }
        let stability = if self.jaccards.is_empty() {
            0.0
        } else {
            self.jaccard_sum / self.jaccards.len() as f64
        };

        // --- maturity: age of the oldest continuously surviving form.
        let mut next_seen: HashMap<u64, usize> = HashMap::with_capacity(forms.len());
        for form in &forms {
            let first = self.seen.get(form).copied().unwrap_or(generation);
            next_seen.insert(*form, first);
        }
        self.seen = next_seen;
        let maturity = if forms.is_empty() || generation == 0 {
            0.0
        } else {
            let oldest = self
                .seen
                .values()
                .copied()
                .min()
                .unwrap_or(generation);
            (generation - oldest) as f64 / generation as f64
        };

        self.current = Convergence::from_parts(confidence, stability, maturity);
        self.current
    }

    /// The most recent reading (all zero before the first observation).
    pub fn current(&self) -> Convergence {
        self.current
    }
}

impl Default for ProgressStats {
    fn default() -> Self {
        ProgressStats::new(DEFAULT_WINDOW)
    }
}

/// A structural form: the expression with every fitted constant collapsed to
/// `c` (and commutative arguments sorted), exactly the form reported by
/// `--stability` (see [`crate::stability::shape`]).
fn form_hash(tree: &crate::expr::Node) -> u64 {
    // FNV-1a over the canonical shape string. `shape` already sorts commutative
    // arguments and folds polynomial spellings, so numerically refitted but
    // structurally identical members keep one form.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in shape(tree).as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn clamp01(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "null".to_string()
    }
}

/// True when a generation counts as "no improvement" for the patience test.
///
/// The denominator is `1 + |best_prev|` and the threshold 1e-9. A non-finite
/// `best_prev` is always an improvement.
pub fn no_improvement(best_prev: f64, gen_best: f64) -> bool {
    (best_prev - gen_best) / (1.0 + best_prev.abs()) <= 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_patience_keeps_the_old_rule() {
        assert!(no_improvement(1.0, 1.0));
        assert!(!no_improvement(f64::INFINITY, 3.0));
        assert!(no_improvement(1e-6, 1e-6 - 1e-12));
    }
}
