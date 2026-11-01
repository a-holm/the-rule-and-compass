//! Adaptive edit weights for the node-path mutation (`--adaptive-mix`, #471).
//!
//! Each island keeps, for the last [`WINDOW`] generations, how often each edit
//! kind produced a child and how often that child beat its first parent. The
//! weight of kind `k` is `max(FLOOR, (s_k + 1) / (a_k + 2))`, normalised. No
//! RNG is used, so the update is a pure function of the counts.

use crate::expr::{MixWeights, EDIT_KINDS};
use std::collections::VecDeque;

/// Generations of counts that feed the weights.
pub const WINDOW: usize = 10;

/// Lowest weight an edit kind can fall to before normalisation.
pub const FLOOR: f64 = 0.02;

/// Per-kind `(attempts, successes)` of one generation.
pub type MixCounts = [(u64, u64); EDIT_KINDS];

/// The counts of the last [`WINDOW`] generations of one island, oldest first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MixWindow {
    gens: VecDeque<MixCounts>,
}

impl MixWindow {
    pub fn new() -> Self {
        MixWindow::default()
    }

    /// Append one generation, dropping the oldest beyond [`WINDOW`].
    pub fn push(&mut self, counts: MixCounts) {
        if self.gens.len() == WINDOW {
            self.gens.pop_front();
        }
        self.gens.push_back(counts);
    }

    pub fn len(&self) -> usize {
        self.gens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gens.is_empty()
    }

    /// The weights for the next generation. With no attempt in the window the
    /// `base` mix stands.
    pub fn weights(&self, base: &MixWeights) -> MixWeights {
        let mut total = [(0u64, 0u64); EDIT_KINDS];
        for g in &self.gens {
            for (t, c) in total.iter_mut().zip(g.iter()) {
                t.0 += c.0;
                t.1 += c.1;
            }
        }
        if total.iter().all(|t| t.0 == 0) {
            return *base;
        }
        let mut w = [0.0; EDIT_KINDS];
        for (wk, t) in w.iter_mut().zip(total.iter()) {
            *wk = (((t.1 + 1) as f64) / ((t.0 + 2) as f64)).max(FLOOR);
        }
        let sum: f64 = w.iter().sum();
        for wk in w.iter_mut() {
            *wk /= sum;
        }
        w
    }

    /// The generations as flat numbers, `2 * EDIT_KINDS` per generation.
    pub fn encode(&self) -> Vec<u64> {
        self.gens
            .iter()
            .flat_map(|g| g.iter().flat_map(|c| [c.0, c.1]))
            .collect()
    }

    pub fn decode(values: &[u64]) -> Result<MixWindow, String> {
        let per = 2 * EDIT_KINDS;
        if values.len() % per != 0 {
            return Err(format!(
                "mix record has {} number(s), expected a multiple of {}",
                values.len(),
                per
            ));
        }
        if values.len() / per > WINDOW {
            return Err(format!(
                "mix record has {} generation(s), the window holds {}",
                values.len() / per,
                WINDOW
            ));
        }
        let mut w = MixWindow::new();
        for chunk in values.chunks(per) {
            let mut counts = [(0u64, 0u64); EDIT_KINDS];
            for (k, c) in counts.iter_mut().enumerate() {
                *c = (chunk[2 * k], chunk[2 * k + 1]);
                if c.1 > c.0 {
                    return Err("mix record has more successes than attempts".to_string());
                }
            }
            w.push(counts);
        }
        Ok(w)
    }
}
