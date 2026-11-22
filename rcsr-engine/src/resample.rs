//! #92: index-based resampling with a bounded BLB cap.
//!
//! The diagnostics used to clone a whole row for every bootstrap draw. A
//! resample is an *index* selection, so the draw itself is `O(rows)` integers
//! and only the selected rows are materialised. Above a threshold the draw is
//! bounded with the bag of little bootstraps (Kleiner, Talwalkar, Sarkar &
//! Jordan, 2014): take a subsample of `b = n^0.7` distinct rows and resample
//! within it. That changes the estimator (and is reported as a cap), so below
//! the threshold the exact bootstrap is kept bit-for-bit.

use crate::data::Sample;
use crate::rng::Mt;

/// BLB exponent from Kleiner et al. (2014): `b = n^0.7`.
pub const BLB_GAMMA: f64 = 0.7;

/// Smallest BLB subsample we are willing to resample from.
///
/// BLB consistency needs `b -> inf`; below `b = 1_000` the exact bootstrap is
/// affordable and is the better estimator, so the cap switches on only when
/// `n^0.7 >= 1_000` (about `n = 19_307`). This is a property of the method, not
/// a magic row count.
pub const BLB_MIN_ROWS: usize = 1_000;

/// What a resample will materialise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Rows in the source sample.
    pub source_rows: usize,
    /// Rows in each resample: `source_rows` below the cap, `n^0.7` above it.
    pub rows: usize,
    /// True when `rows < source_rows` (the BLB cap is active).
    pub capped: bool,
}

/// The resampling plan for a source sample of `n` rows.
pub fn plan(n: usize) -> Plan {
    if n == 0 {
        return Plan {
            source_rows: 0,
            rows: 0,
            capped: false,
        };
    }
    let b = (n as f64).powf(BLB_GAMMA).floor() as usize;
    if b >= BLB_MIN_ROWS && b < n {
        Plan {
            source_rows: n,
            rows: b,
            capped: true,
        }
    } else {
        Plan {
            source_rows: n,
            rows: n,
            capped: false,
        }
    }
}

/// `b` distinct row indices from `0..n`, uniformly, with reservoir sampling.
///
/// Uses `O(b)` memory, not `O(n)`: the point of the cap is that a diagnostic
/// never allocates one index per source row.
pub fn subsample_indices(n: usize, b: usize, rng: &mut Mt) -> Vec<usize> {
    let b = b.min(n);
    let mut out: Vec<usize> = (0..b).collect();
    for i in b..n {
        let j = rng.usize_below(i + 1);
        if j < b {
            out[j] = i;
        }
    }
    out
}

/// `draws` indices with replacement into `0..pool`, in draw order.
pub fn draw_indices(pool: usize, draws: usize, rng: &mut Mt) -> Vec<usize> {
    let mut out = Vec::with_capacity(draws);
    for _ in 0..draws {
        out.push(rng.usize_below(pool));
    }
    out
}

/// Materialise only the selected rows of `s`.
pub fn gather(s: &Sample, rows: &[usize]) -> Sample {
    let mut out = Sample::default();
    out.x.reserve(rows.len());
    out.y.reserve(rows.len());
    for &r in rows {
        out.x.push(s.x[r].clone());
        out.y.push(s.y[r]);
    }
    if !s.w.is_empty() {
        out.w.reserve(rows.len());
        for &r in rows {
            out.w.push(s.w[r]);
        }
    }
    if !s.ys.is_empty() {
        out.ys = s
            .ys
            .iter()
            .map(|col| rows.iter().map(|&r| col[r]).collect())
            .collect();
    }
    out
}

/// A bootstrap resample: `plan.rows` rows drawn **with replacement**.
pub fn resample(s: &Sample, rng: &mut Mt) -> Sample {
    let p = plan(s.len());
    if p.capped {
        let base = subsample_indices(p.source_rows, p.rows, rng);
        let rows: Vec<usize> = draw_indices(p.rows, p.rows, rng)
            .into_iter()
            .map(|j| base[j])
            .collect();
        gather(s, &rows)
    } else {
        let rows = draw_indices(p.source_rows, p.rows, rng);
        gather(s, &rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_is_n_pow_07_and_reports_its_size() {
        // Below the threshold the plan is exact.
        let small = plan(100);
        assert_eq!(small.rows, 100);
        assert!(!small.capped);

        // At 200 000 rows the BLB draw is floor(n^0.7), not a magic constant.
        let big = plan(200_000);
        assert!(big.capped);
        assert_eq!(big.source_rows, 200_000);
        assert_eq!(big.rows, (200_000f64).powf(BLB_GAMMA).floor() as usize);
        assert_eq!(big.rows, 5_137);
        assert!(big.rows < big.source_rows);
    }

    #[test]
    fn subsample_has_b_distinct_in_bounds_rows() {
        let mut rng = Mt::new(7);
        let rows = subsample_indices(50_000, 777, &mut rng);
        assert_eq!(rows.len(), 777);
        let mut seen = vec![false; 50_000];
        for &r in &rows {
            assert!(r < 50_000);
            assert!(!seen[r], "reservoir drew row {r} twice");
            seen[r] = true;
        }
    }

    #[test]
    fn gather_copies_only_the_selected_rows() {
        let s = Sample::weighted(
            vec![vec![0.0], vec![1.0], vec![2.0]],
            vec![10.0, 11.0, 12.0],
            vec![1.0, 2.0, 3.0],
        );
        let g = gather(&s, &[2, 0, 2]);
        assert_eq!(g.len(), 3);
        assert_eq!(g.x, vec![vec![2.0], vec![0.0], vec![2.0]]);
        assert_eq!(g.y, vec![12.0, 10.0, 12.0]);
        assert_eq!(g.w, vec![3.0, 1.0, 3.0]);
    }
}
