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

    /// Whether `(error, cost)` would survive the dominance test.
    ///
    /// The postfix generation path calls this before spending a tree rebuild
    /// on a candidate: the archive only ever stores survivors, so the common
    /// case costs two comparisons per front member and no allocation.
    pub fn admits(&self, error: f64, cost: f64) -> bool {
        if !error.is_finite() {
            return false;
        }
        !self
            .items
            .iter()
            .any(|it| Self::dominates(it.error, it.cost, error, cost))
    }

    /// Rebuild the front in the canonical staircase order (#122).
    ///
    /// Epsilon-dominance is not transitive, so a front built by incremental
    /// insertion depends on arrival order. This pass sorts by
    /// `(cost, error, structural hash)`, keeps the first entry per cost and
    /// then keeps only entries whose error strictly improves as cost rises.
    /// The result is a monotone staircase and a pure function of the member
    /// set, which makes the archive hash a valid determinism gate.
    pub fn canonicalize(&mut self) {
        self.items.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then(a.error.total_cmp(&b.error))
                .then(
                    crate::expr::structural_hash(&a.tree)
                        .cmp(&crate::expr::structural_hash(&b.tree)),
                )
        });
        let mut staircase: Vec<Sol> = Vec::with_capacity(self.items.len());
        for sol in self.items.drain(..) {
            let keep = match staircase.last() {
                None => true,
                Some(prev) => {
                    if prev.cost == sol.cost {
                        // One entry per cost: the sorted order already put the
                        // lowest error first.
                        false
                    } else {
                        sol.error < prev.error
                    }
                }
            };
            if keep {
                staircase.push(sol);
            }
        }
        // The cap trims deterministically: keep an even prefix spread so the
        // low-cost end, which is what a person reads, is never dropped.
        if staircase.len() > self.cap {
            let cap = self.cap.max(1);
            let n = staircase.len();
            let mut trimmed = Vec::with_capacity(cap);
            for k in 0..cap {
                trimmed.push(staircase[k * n / cap].clone());
            }
            staircase = trimmed;
        }
        self.items = staircase;
    }

    /// FNV-1a over the canonical staircase: cost bits, error bits and the
    /// structural hash of each tree. Two archives with the same members in
    /// canonical order hash identically, independent of insertion order.
    pub fn hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut order: Vec<&Sol> = self.items.iter().collect();
        order.sort_by(|a, b| {
            a.cost
                .total_cmp(&b.cost)
                .then(a.error.total_cmp(&b.error))
                .then(
                    crate::expr::structural_hash(&a.tree)
                        .cmp(&crate::expr::structural_hash(&b.tree)),
                )
        });
        for sol in order {
            for word in [
                sol.cost.to_bits(),
                sol.error.to_bits(),
                crate::expr::structural_hash(&sol.tree),
            ] {
                h ^= word;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input};

    fn sol(cost: f64, error: f64, v: f64) -> Sol {
        Sol {
            tree: crate::expr::node(
                crate::expr::Op::Add,
                vec![input(0), constant(v)],
            ),
            error,
            complexity: cost as usize,
            cost,
        }
    }

    #[test]
    fn canonical_staircase_is_order_independent() {
        let members = vec![
            sol(1.0, 0.9, 1.0),
            sol(2.0, 0.7, 2.0),
            sol(2.0, 0.6, 3.0),
            sol(3.0, 0.8, 4.0),
            sol(4.0, 0.5, 5.0),
        ];
        let mut a = Archive::new(100);
        a.items = members.clone();
        a.canonicalize();
        let mut b = Archive::new(100);
        b.items = members.into_iter().rev().collect();
        b.canonicalize();
        assert_eq!(a.items.len(), 3, "one per cost, then a monotone staircase");
        assert_eq!(a.items.len(), b.items.len());
        for (x, y) in a.items.iter().zip(&b.items) {
            assert_eq!((x.cost, x.error.to_bits()), (y.cost, y.error.to_bits()));
        }
        assert_eq!(a.hash(), b.hash());
        // Strictly improving errors as cost rises.
        for pair in a.items.windows(2) {
            assert!(pair[1].cost > pair[0].cost);
            assert!(pair[1].error < pair[0].error);
        }
    }

    #[test]
    fn admits_matches_the_insertion_test() {
        let mut a = Archive::new(100);
        a.try_insert(sol(1.0, 0.5, 1.0));
        assert!(!a.admits(0.6, 2.0), "dominated by (1, 0.5)");
        assert!(!a.admits(f64::NAN, 0.1));
        assert!(a.admits(0.4, 2.0));
    }
}
