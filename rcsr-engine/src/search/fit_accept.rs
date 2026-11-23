//! #1117: counts of the `--fit-accept fitness` decisions, by path.

use std::sync::atomic::{AtomicU64, Ordering};

/// The place a gated fit ran. The names are the keys of `search.fit_accept.by_path`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitPath {
    Offspring,
    Restart,
    Carried,
    Device,
    LostBatch,
    Survivor,
    Finalist,
    Prune,
    Seed,
}

impl FitPath {
    pub const ALL: [FitPath; 9] = [
        FitPath::Offspring,
        FitPath::Restart,
        FitPath::Carried,
        FitPath::Device,
        FitPath::LostBatch,
        FitPath::Survivor,
        FitPath::Finalist,
        FitPath::Prune,
        FitPath::Seed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FitPath::Offspring => "offspring",
            FitPath::Restart => "restart",
            FitPath::Carried => "carried",
            FitPath::Device => "device",
            FitPath::LostBatch => "lost_batch",
            FitPath::Survivor => "survivor",
            FitPath::Finalist => "finalist",
            FitPath::Prune => "prune",
            FitPath::Seed => "seed",
        }
    }
}

/// Atomic `[accepted, rejected]` pairs, one per [`FitPath`]. A count is a pure
/// function of the gate results, so it does not depend on the thread count.
#[derive(Debug, Default)]
pub struct FitAcceptCounts {
    slots: [[AtomicU64; 2]; 9],
}

impl FitAcceptCounts {
    /// Record one decision: `kept` is true when the fitted candidate won.
    pub fn record(&self, path: FitPath, kept: bool) {
        self.slots[path as usize][usize::from(!kept)].fetch_add(1, Ordering::Relaxed);
    }

    /// Add every count of `other` into `self`.
    pub fn absorb(&self, other: &FitAcceptCounts) {
        for p in FitPath::ALL {
            for k in 0..2 {
                let n = other.slots[p as usize][k].load(Ordering::Relaxed);
                self.slots[p as usize][k].fetch_add(n, Ordering::Relaxed);
            }
        }
    }

    pub fn snapshot(&self) -> FitAcceptReport {
        let mut by_path = [[0u64; 2]; 9];
        for p in FitPath::ALL {
            for k in 0..2 {
                by_path[p as usize][k] = self.slots[p as usize][k].load(Ordering::Relaxed);
            }
        }
        FitAcceptReport { by_path }
    }
}

/// A copy of the counts, as reported in the run JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FitAcceptReport {
    by_path: [[u64; 2]; 9],
}

impl FitAcceptReport {
    /// `(accepted, rejected)` summed over the paths.
    pub fn totals(&self) -> (u64, u64) {
        self.by_path.iter().fold((0, 0), |(a, r), p| (a + p[0], r + p[1]))
    }

    /// `(accepted, rejected)` of the path named `name`, or `None` for an unknown name.
    pub fn path(&self, name: &str) -> Option<(u64, u64)> {
        FitPath::ALL.iter().find(|p| p.name() == name).map(|p| {
            let c = self.by_path[*p as usize];
            (c[0], c[1])
        })
    }

    /// The `search.fit_accept` object.
    pub fn json(&self) -> String {
        let (accepted, rejected) = self.totals();
        let paths: Vec<String> = FitPath::ALL
            .iter()
            .map(|p| {
                let c = self.by_path[*p as usize];
                format!("\"{}\":[{},{}]", p.name(), c[0], c[1])
            })
            .collect();
        format!("{{\"accepted\":{},\"rejected\":{},\"by_path\":{{{}}}}}", accepted, rejected, paths.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_accept_report_totals_are_the_sums_over_the_paths() {
        let counts = FitAcceptCounts::default();
        counts.record(FitPath::Offspring, true);
        counts.record(FitPath::Offspring, false);
        counts.record(FitPath::Seed, false);
        let other = FitAcceptCounts::default();
        other.record(FitPath::Prune, true);
        counts.absorb(&other);
        let report = counts.snapshot();
        assert_eq!(report.totals(), (2, 2));
        assert_eq!(report.path("offspring"), Some((1, 1)));
        assert_eq!(report.path("seed"), Some((0, 1)));
        assert_eq!(report.path("nothing"), None);
        assert_eq!(
            report.json(),
            "{\"accepted\":2,\"rejected\":2,\"by_path\":{\"offspring\":[1,1],\"restart\":[0,0],\"carried\":[0,0],\
             \"device\":[0,0],\"lost_batch\":[0,0],\"survivor\":[0,0],\"finalist\":[0,0],\"prune\":[1,0],\"seed\":[0,1]}}"
        );
    }
}
