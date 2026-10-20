//! Multi-objective archive: the error/complexity Pareto front.

use crate::expr::Node;

#[derive(Clone, Debug)]
pub struct Sol {
    pub tree: Node,
    pub error: f64,
    /// Structural node count. Retained as a stable, integer measure for
    /// callers that report tree size (bench and stability summaries).
    pub complexity: usize,
    /// The configured complexity from
    /// [`crate::complexity::ComplexityModel`]. This is the value the front is
    /// ranked and pruned on, and the value the CLI reports.
    pub cost: f64,
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
    fn dominates(a_e: f64, a_c: f64, b_e: f64, b_c: f64) -> bool {
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
        let Sol {
            tree,
            error,
            complexity,
            cost,
        } = s;
        self.try_insert_view(&tree, error, complexity, cost);
    }

    /// The same insertion as [`Archive::try_insert`], but borrowing the tree.
    ///
    /// The search visits every individual of every generation. Most are
    /// dominated immediately; cloning their trees first was pure waste. This
    /// clones only the candidate that actually enters the front, so the
    /// resulting archive is bit-identical to the owned-tree path.
    pub fn try_insert_view(
        &mut self,
        tree: &Node,
        error: f64,
        complexity: usize,
        cost: f64,
    ) {
        if !error.is_finite() {
            return;
        }
        for it in &self.items {
            if Self::dominates(it.error, it.cost, error, cost) {
                return;
            }
        }
        self.items
            .retain(|it| !Self::dominates(error, cost, it.error, it.cost));
        if self
            .items
            .iter()
            .any(|it| it.cost == cost && (it.error - error).abs() < 1e-12)
        {
            return;
        }
        self.items.push(Sol {
            tree: tree.clone(),
            error,
            complexity,
            cost,
        });
        if self.items.len() > self.cap {
            self.prune();
        }
    }

    /// Keep the front readable: one entry per complexity level, then trim.
    fn prune(&mut self) {
        self.items.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then(a.error.partial_cmp(&b.error).unwrap())
        });
        self.items.dedup_by(|a, b| a.cost == b.cost);
        while self.items.len() > self.cap {
            self.items.remove(self.items.len() / 2);
        }
    }

    pub fn sorted(&self) -> Vec<Sol> {
        let mut v = self.items.clone();
        v.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then(a.error.partial_cmp(&b.error).unwrap())
        });
        v
    }
}
