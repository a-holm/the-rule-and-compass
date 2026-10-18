//! Multi-objective archive: the error/complexity Pareto front.

use crate::expr::Node;

#[derive(Clone, Debug)]
pub struct Sol {
    pub tree: Node,
    pub error: f64,
    pub complexity: usize,
}

pub struct Archive {
    pub items: Vec<Sol>,
    pub cap: usize,
}

impl Archive {
    pub fn new(cap: usize) -> Self {
        Archive {
            items: Vec::new(),
            cap: cap.max(1),
        }
    }

    fn dominates(a_e: f64, a_c: usize, b_e: f64, b_c: usize) -> bool {
        a_e <= b_e && a_c <= b_c && (a_e < b_e || a_c < b_c)
    }

    pub fn try_insert(&mut self, s: Sol) {
        if !s.error.is_finite() {
            return;
        }
        for it in &self.items {
            if Self::dominates(it.error, it.complexity, s.error, s.complexity) {
                return;
            }
        }
        self.items.retain(|it| {
            !Self::dominates(s.error, s.complexity, it.error, it.complexity)
        });
        if self
            .items
            .iter()
            .any(|it| it.complexity == s.complexity && (it.error - s.error).abs() < 1e-12)
        {
            return;
        }
        self.items.push(s);
        if self.items.len() > self.cap {
            self.prune();
        }
    }

    /// Keep the front readable: one entry per complexity level, then trim.
    fn prune(&mut self) {
        self.items.sort_by(|a, b| {
            a.complexity
                .cmp(&b.complexity)
                .then(a.error.partial_cmp(&b.error).unwrap())
        });
        self.items.dedup_by(|a, b| a.complexity == b.complexity);
        while self.items.len() > self.cap {
            self.items.remove(self.items.len() / 2);
        }
    }

    pub fn sorted(&self) -> Vec<Sol> {
        let mut v = self.items.clone();
        v.sort_by(|a, b| {
            a.complexity
                .cmp(&b.complexity)
                .then(a.error.partial_cmp(&b.error).unwrap())
        });
        v
    }
}
