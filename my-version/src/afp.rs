//! Age-fitness Pareto survivor selection (#465).
//!
//! Survivors are chosen from parents + offspring by NSGA-II on three minimised
//! objectives `(error, complexity, age)`. Every step is a deterministic
//! function of the input order: ties go to the lower index.

use crate::rng::Mt;
use std::cmp::Ordering;

/// Objective vector of one candidate: error, complexity, age.
pub type Point = [f64; 3];

/// `a` dominates `b`: no worse everywhere, better somewhere. Equal vectors are
/// resolved by index, so the lower index dominates and survives.
fn dominates(a: &Point, ia: usize, b: &Point, ib: usize) -> bool {
    let mut better = false;
    for k in 0..3 {
        match a[k].total_cmp(&b[k]) {
            Ordering::Greater => return false,
            Ordering::Less => better = true,
            Ordering::Equal => {}
        }
    }
    better || ia < ib
}

/// Non-dominated sorting. Returns the fronts as lists of indices, each in
/// ascending index order.
pub fn fronts(points: &[Point]) -> Vec<Vec<usize>> {
    let n = points.len();
    let mut dominated_by = vec![0usize; n];
    let mut dominating: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&points[i], i, &points[j], j) {
                dominating[i].push(j);
                dominated_by[j] += 1;
            } else if dominates(&points[j], j, &points[i], i) {
                dominating[j].push(i);
                dominated_by[i] += 1;
            }
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|&i| dominated_by[i] == 0).collect();
    while !current.is_empty() {
        let mut next: Vec<usize> = Vec::new();
        for &i in &current {
            for &j in &dominating[i] {
                dominated_by[j] -= 1;
                if dominated_by[j] == 0 {
                    next.push(j);
                }
            }
        }
        next.sort_unstable();
        out.push(std::mem::replace(&mut current, next));
    }
    out
}

/// NSGA-II crowding distance of the members of one front, indexed like
/// `front`. Boundary members get infinity; an objective whose range is not
/// finite and positive adds nothing.
fn crowding(points: &[Point], front: &[usize]) -> Vec<f64> {
    let m = front.len();
    let mut dist = vec![0.0f64; m];
    if m <= 2 {
        return vec![f64::INFINITY; m];
    }
    for k in 0..3 {
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| {
            points[front[a]][k]
                .total_cmp(&points[front[b]][k])
                .then(front[a].cmp(&front[b]))
        });
        let lo = points[front[order[0]]][k];
        let hi = points[front[order[m - 1]]][k];
        dist[order[0]] = f64::INFINITY;
        dist[order[m - 1]] = f64::INFINITY;
        let range = hi - lo;
        if !(range.is_finite() && range > 0.0) {
            continue;
        }
        for w in 1..m - 1 {
            let d = (points[front[order[w + 1]]][k] - points[front[order[w - 1]]][k]) / range;
            if d.is_finite() {
                dist[order[w]] += d;
            }
        }
    }
    dist
}

/// Pareto rank and crowding distance of every point.
pub fn rank_and_crowd(points: &[Point]) -> (Vec<u32>, Vec<f64>) {
    let mut rank = vec![0u32; points.len()];
    let mut crowd = vec![0.0f64; points.len()];
    for (r, front) in fronts(points).iter().enumerate() {
        let c = crowding(points, front);
        for (pos, &i) in front.iter().enumerate() {
            rank[i] = r as u32;
            crowd[i] = c[pos];
        }
    }
    (rank, crowd)
}

/// Indices of the `n` survivors, fronts first, the last front split by
/// crowding distance (larger first, then lower index). Returned in ascending
/// index order.
pub fn survivors(points: &[Point], n: usize) -> Vec<usize> {
    let mut keep: Vec<usize> = Vec::with_capacity(n);
    for front in fronts(points) {
        if keep.len() + front.len() <= n {
            keep.extend(front);
        } else {
            let c = crowding(points, &front);
            let mut order: Vec<usize> = (0..front.len()).collect();
            order.sort_by(|&a, &b| c[b].total_cmp(&c[a]).then(front[a].cmp(&front[b])));
            keep.extend(order.into_iter().take(n - keep.len()).map(|p| front[p]));
        }
        if keep.len() >= n {
            break;
        }
    }
    keep.sort_unstable();
    keep
}

/// Tournament of size `k`: lower rank wins, then larger crowding distance, then
/// lower index.
pub fn tournament(rank: &[u32], crowd: &[f64], rng: &mut Mt, k: usize) -> usize {
    let n = rank.len();
    if n == 0 {
        return 0;
    }
    let beats = |a: usize, b: usize| {
        rank[a]
            .cmp(&rank[b])
            .then(crowd[b].total_cmp(&crowd[a]))
            .then(a.cmp(&b))
            == Ordering::Less
    };
    let mut best = rng.usize_below(n);
    for _ in 1..k {
        let c = rng.usize_below(n);
        if beats(c, best) {
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fronts_sort_by_dominance() {
        let p = [[1.0, 5.0, 0.0], [2.0, 4.0, 0.0], [2.0, 6.0, 0.0], [3.0, 7.0, 1.0]];
        let f = fronts(&p);
        assert_eq!(f, vec![vec![0, 1], vec![2], vec![3]]);
    }

    #[test]
    fn equal_vectors_keep_the_lower_index() {
        let p = [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 0.0]];
        assert_eq!(survivors(&p, 1), vec![0]);
        assert_eq!(survivors(&p, 2), vec![0, 1]);
    }

    #[test]
    fn crowding_splits_the_last_front() {
        // One front on a line; keep 3 of 5: both ends and the most isolated.
        let p = [
            [0.0, 4.0, 0.0],
            [1.0, 3.0, 0.0],
            [1.1, 2.9, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        assert_eq!(survivors(&p, 3), vec![0, 3, 4]);
    }

    #[test]
    fn non_finite_errors_fall_back_to_complexity_and_age() {
        let inf = f64::INFINITY;
        let p = [[inf, 5.0, 3.0], [inf, 2.0, 3.0], [inf, 2.0, 1.0]];
        assert_eq!(survivors(&p, 1), vec![2]);
    }

    #[test]
    fn tournament_prefers_rank_then_crowding_then_index() {
        let rank = [1, 0, 0];
        let crowd = [9.0, 1.0, 2.0];
        let mut rng = Mt::new(1);
        for _ in 0..50 {
            let w = tournament(&rank, &crowd, &mut rng, 8);
            assert_ne!(w, 0);
        }
    }
}
