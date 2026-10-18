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

    /// ε-dominance (Laumanns et al. 2002): errors within a relative tolerance
    /// count as equal, so a marginally smaller error no longer justifies a much
    /// larger expression. This keeps the front small, well-spread and readable
    /// instead of filling it with numerically-equivalent, bloated variants.
    fn dominates(a_e: f64, a_c: usize, b_e: f64, b_c: usize) -> bool {
        if a_c > b_c {
            return false;
        }
        let tol = 1e-6 * (1.0 + a_e.abs().max(b_e.abs()));
        if a_c < b_c {
            a_e <= b_e + tol
        } else {
            a_e < b_e
        }
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
