//! Dependence-robust noise for paired losses and score sums (#257).
//!
//! Rows that share a timestamp are cross-correlated and losses are serially
//! correlated, so the iid standard error of a mean is too small on a panel.
//! The Driscoll-Kraay estimator sums the per-row influence over the cross
//! section at each time, then applies a Newey-West (Bartlett) sum across
//! time. Everything here is single-threaded with a fixed summation order:
//! per-timestamp sums run over rows in input order, timestamps in sorted key
//! order, lags `1..=L` in order. The result is a pure function of its inputs.

use crate::metrics::Metric;

/// Mean of a per-row quantity with its dependence-robust standard error.
#[derive(Clone, Copy, Debug)]
pub struct MeanSe {
    pub mean: f64,
    pub se: f64,
    pub se_iid: f64,
    pub design_effect: f64,
    pub effective_n: f64,
    pub rows: usize,
    pub timestamps: usize,
    pub lag: usize,
}

impl MeanSe {
    pub fn empty() -> MeanSe {
        MeanSe {
            mean: f64::NAN,
            se: f64::NAN,
            se_iid: f64::NAN,
            design_effect: f64::NAN,
            effective_n: f64::NAN,
            rows: 0,
            timestamps: 0,
            lag: 0,
        }
    }
}

/// Where the Bartlett lag came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagSource {
    /// `--noise-lag`.
    Flag,
    /// The label horizon in timestamps minus one.
    Horizon,
    /// The Newey-West (1994) rule of thumb.
    RuleOfThumb,
    /// One row per key and lag zero (random split).
    Iid,
}

impl LagSource {
    pub fn name(self) -> &'static str {
        match self {
            LagSource::Flag => "flag",
            LagSource::Horizon => "horizon",
            LagSource::RuleOfThumb => "rule_of_thumb",
            LagSource::Iid => "iid",
        }
    }
}

/// `max(h_steps - 1, floor(4 (T/100)^(2/9)))` for `T` distinct timestamps.
pub fn newey_west_lag(timestamps: usize, horizon_steps: Option<usize>) -> usize {
    let rule = if timestamps == 0 {
        0
    } else {
        (4.0 * (timestamps as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize
    };
    let h = horizon_steps.map(|h| h.saturating_sub(1)).unwrap_or(0);
    rule.max(h)
}

/// Rows grouped by key: `group[i]` is the position of row `i`'s key among the
/// sorted distinct finite keys. A row with a non-finite key forms its own
/// group after all finite keys, in row order.
#[derive(Clone, Debug)]
pub struct KeyIndex {
    pub group: Vec<usize>,
    pub groups: usize,
}

impl KeyIndex {
    pub fn new(keys: &[f64]) -> KeyIndex {
        // `+ 0.0` maps -0.0 to 0.0 so equal keys share a group.
        let mut distinct: Vec<f64> = keys
            .iter()
            .filter(|k| k.is_finite())
            .map(|k| k + 0.0)
            .collect();
        distinct.sort_by(|a, b| a.total_cmp(b));
        distinct.dedup();
        let finite_groups = distinct.len();
        let mut extra = 0usize;
        let group = keys
            .iter()
            .map(|&k| {
                if k.is_finite() {
                    let k = k + 0.0;
                    distinct
                        .binary_search_by(|a| a.total_cmp(&k))
                        .unwrap_or_else(|p| p)
                } else {
                    extra += 1;
                    finite_groups + extra - 1
                }
            })
            .collect();
        KeyIndex {
            group,
            groups: finite_groups + extra,
        }
    }

    /// One group per row, in row order (the iid case).
    pub fn per_row(n: usize) -> KeyIndex {
        KeyIndex {
            group: (0..n).collect(),
            groups: n,
        }
    }
}

/// Driscoll-Kraay standard error of the weighted mean of `values`.
///
/// `weights` empty means every row weighs 1. `keys` has one entry per row.
pub fn dk_mean_se(values: &[f64], weights: &[f64], keys: &[f64], lag: usize) -> MeanSe {
    dk_mean_se_indexed(values, weights, &KeyIndex::new(keys), lag)
}

/// As [`dk_mean_se`] with the key grouping precomputed.
pub fn dk_mean_se_indexed(values: &[f64], weights: &[f64], index: &KeyIndex, lag: usize) -> MeanSe {
    let n = values.len();
    if n == 0 || index.group.len() != n || (!weights.is_empty() && weights.len() != n) {
        return MeanSe::empty();
    }
    let weight = |i: usize| if weights.is_empty() { 1.0 } else { weights[i] };
    let mut total = 0.0;
    let mut sum = 0.0;
    let mut rows = 0usize;
    for i in 0..n {
        let w = weight(i);
        total += w;
        sum += w * values[i];
        if w > 0.0 {
            rows += 1;
        }
    }
    if !(total > 0.0) {
        return MeanSe::empty();
    }
    let mu = sum / total;
    let mut sums = vec![0.0; index.groups];
    let mut v_iid = 0.0;
    for i in 0..n {
        let psi = weight(i) * (values[i] - mu) / total;
        sums[index.group[i]] += psi;
        v_iid += psi * psi;
    }
    let lag = lag.min(index.groups.saturating_sub(1));
    let mut v = 0.0;
    for s in &sums {
        v += s * s;
    }
    for l in 1..=lag {
        let bartlett = 1.0 - l as f64 / (lag as f64 + 1.0);
        let mut acc = 0.0;
        for t in l..sums.len() {
            acc += sums[t] * sums[t - l];
        }
        v += 2.0 * bartlett * acc;
    }
    let design_effect = if v_iid > 0.0 { v / v_iid } else { 1.0 };
    let effective_n = if design_effect > 0.0 {
        rows as f64 / design_effect
    } else {
        f64::INFINITY
    };
    MeanSe {
        mean: mu,
        se: v.max(0.0).sqrt(),
        se_iid: v_iid.sqrt(),
        design_effect,
        effective_n,
        rows,
        timestamps: index.groups,
        lag,
    }
}

/// Driscoll-Kraay long-run covariance of `k` per-row score series.
///
/// `scores[j][i]` is the score of row `i` for series `j`. Returns the `k x k`
/// matrix `sum_t S_t S_t' + sum_l bartlett_l (G_l + G_l')`, where `S_t` sums the
/// scores of the rows at timestamp `t`.
pub fn dk_score_cov(scores: &[Vec<f64>], keys: &[f64], lag: usize) -> Vec<Vec<f64>> {
    let index = KeyIndex::new(keys);
    let k = scores.len();
    let mut sums = vec![0.0; index.groups * k];
    for (j, series) in scores.iter().enumerate() {
        for (i, &s) in series.iter().enumerate() {
            sums[index.group[i] * k + j] += s;
        }
    }
    dk_cov_from_group_sums(&sums, index.groups, k, lag)
}

/// The covariance of [`dk_score_cov`] from per-timestamp sums laid out as
/// `sums[t * k + j]`.
pub fn dk_cov_from_group_sums(sums: &[f64], groups: usize, k: usize, lag: usize) -> Vec<Vec<f64>> {
    let mut omega = vec![vec![0.0; k]; k];
    if groups == 0 || k == 0 {
        return omega;
    }
    for t in 0..groups {
        for a in 0..k {
            for b in 0..k {
                omega[a][b] += sums[t * k + a] * sums[t * k + b];
            }
        }
    }
    let lag = lag.min(groups - 1);
    for l in 1..=lag {
        let bartlett = 1.0 - l as f64 / (lag as f64 + 1.0);
        for t in l..groups {
            for a in 0..k {
                for b in 0..k {
                    omega[a][b] +=
                        bartlett * (sums[t * k + a] * sums[(t - l) * k + b] + sums[(t - l) * k + a] * sums[t * k + b]);
                }
            }
        }
    }
    omega
}

/// True when `metric` is a mean of per-row losses that [`row_losses`] defines.
pub fn row_decomposable(metric: Metric) -> bool {
    matches!(
        metric,
        Metric::Squared
            | Metric::Absolute
            | Metric::LogSquashed
            | Metric::LogLoss
            | Metric::Hinge
            | Metric::Aic
            | Metric::Aicc
            | Metric::Bic
    )
}

/// Per-row loss of a prediction. `aic`, `aicc` and `bic` use the squared row
/// loss. A non-finite prediction gives `+inf`. `None` when the metric is not
/// row-decomposable. Classification labels follow the metric's own rule: a row
/// is positive when its target exceeds the midpoint of the observed extremes
/// (over positive-weight rows when weighted).
pub fn row_losses(pred: &[f64], y: &[f64], w: &[f64], metric: Metric) -> Option<Vec<f64>> {
    use crate::metrics::{binary_labels, binary_labels_weighted, classification_row_loss, ScoreObjective};
    if pred.len() != y.len() {
        return None;
    }
    let finite = |loss: f64| if loss.is_finite() { loss } else { f64::INFINITY };
    let regression = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
        pred.iter().zip(y).map(|(p, t)| finite(f(p - t))).collect()
    };
    match metric {
        Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic => Some(regression(&|e| e * e)),
        Metric::Absolute => Some(regression(&|e| e.abs())),
        Metric::LogSquashed => Some(regression(&|e| (1.0 + e * e).ln())),
        Metric::LogLoss | Metric::Hinge => {
            let objective = if metric == Metric::LogLoss {
                ScoreObjective::LogLoss
            } else {
                ScoreObjective::HingeSigmoid
            };
            let labels = if w.is_empty() {
                binary_labels(y)
            } else {
                binary_labels_weighted(y, w)?
            };
            Some(
                pred.iter()
                    .zip(&labels)
                    .map(|(&p, &l)| finite(classification_row_loss(p, l, objective)))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Standard normal quantile by fixed-iteration bisection on `Phi`, built on
/// [`crate::expr::erfc`]. Deterministic and accurate to about 1e-12.
pub fn normal_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return if p <= 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    let phi = |x: f64| 0.5 * crate::expr::erfc(-x / std::f64::consts::SQRT_2);
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if phi(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn dk_se_iid_one_row_per_key() {
        let v: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let keys: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let m = dk_mean_se(&v, &[], &keys, 0);
        assert!(close(m.mean, 4.5, 1e-15));
        assert!(close(m.se, 0.8100925873, 1e-10), "{}", m.se);
        assert!(close(m.se * m.se, 0.65625, 1e-15));
    }

    #[test]
    fn dk_se_sums_rows_within_a_timestamp() {
        let v = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let keys = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let m = dk_mean_se(&v, &[], &keys, 0);
        assert!(close(m.se, 0.5590169944, 1e-10), "{}", m.se);
        assert!(close(m.se_iid * m.se_iid, 0.15625, 1e-15));
        assert!(close(m.design_effect, 2.0, 1e-12), "{}", m.design_effect);
        assert!(close(m.effective_n, 4.0, 1e-12), "{}", m.effective_n);
        assert_eq!(m.timestamps, 4);
    }

    #[test]
    fn dk_se_bartlett_lag_one() {
        let v = [1.0, 2.0, 3.0, 4.0];
        let keys = [0.0, 1.0, 2.0, 3.0];
        let m = dk_mean_se(&v, &[], &keys, 1);
        assert!(close(m.se * m.se, 0.390625, 1e-15), "{}", m.se * m.se);
        assert!(close(m.se, 0.625, 1e-15));
    }

    #[test]
    fn newey_west_lag_rule() {
        assert_eq!(newey_west_lag(100, None), 4);
        assert_eq!(newey_west_lag(3490, None), 8);
        assert_eq!(newey_west_lag(4344, None), 9);
        assert_eq!(newey_west_lag(34900, None), 14);
        assert_eq!(newey_west_lag(100, Some(24)), 23);
        assert_eq!(newey_west_lag(0, None), 0);
    }

    #[test]
    fn dk_se_is_invariant_to_timestamp_interleaving() {
        // Same rows per timestamp, in the same relative order, interleaved
        // differently across timestamps.
        let a_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_keys = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let b_vals = [3.0, 1.0, 5.0, 2.0, 4.0, 6.0];
        let b_keys = [1.0, 0.0, 2.0, 0.0, 1.0, 2.0];
        let a = dk_mean_se(&a_vals, &[], &a_keys, 2);
        let b = dk_mean_se(&b_vals, &[], &b_keys, 2);
        assert_eq!(a.se.to_bits(), b.se.to_bits());
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
    }

    #[test]
    fn dk_se_is_deterministic_across_calls() {
        let v: Vec<f64> = (0..500).map(|i| ((i * 37 % 101) as f64).sin()).collect();
        let keys: Vec<f64> = (0..500).map(|i| (i / 7) as f64).collect();
        let a = dk_mean_se(&v, &[], &keys, 5);
        let b = dk_mean_se(&v, &[], &keys, 5);
        assert_eq!(a.se.to_bits(), b.se.to_bits());
    }

    #[test]
    fn non_finite_keys_form_one_group_per_row_after_finite_keys() {
        let idx = KeyIndex::new(&[2.0, f64::NAN, 1.0, f64::INFINITY, 1.0, -0.0, 0.0]);
        assert_eq!(idx.group, vec![2, 3, 1, 4, 1, 0, 0]);
        assert_eq!(idx.groups, 5);
    }

    #[test]
    fn normal_quantile_matches_reference_values() {
        for (p, want) in [
            (0.95, 1.6448536270),
            (0.975, 1.9599639845),
            (1.0 - 0.05 / 3.0, 2.1280452342),
            (1.0 - 0.05 / 6.0, 2.3939797998),
        ] {
            let got = normal_quantile(p);
            assert!(close(got, want, 1e-9), "p={p} got={got} want={want}");
        }
    }

    #[test]
    fn score_covariance_sums_rows_within_a_timestamp() {
        let scores = vec![vec![1.0, 1.0, 2.0, 2.0], vec![2.0, 2.0, 1.0, 1.0]];
        let keys = [0.0, 0.0, 1.0, 1.0];
        let o = dk_score_cov(&scores, &keys, 0);
        // S_0 = (2, 4), S_1 = (4, 2).
        assert_eq!(o[0][0], 20.0);
        assert_eq!(o[0][1], 16.0);
        assert_eq!(o[1][0], 16.0);
        assert_eq!(o[1][1], 20.0);
    }

    #[test]
    fn row_losses_cover_the_decomposable_metrics() {
        let p = [1.0, 2.0];
        let y = [0.0, 4.0];
        assert_eq!(row_losses(&p, &y, &[], Metric::Squared).unwrap(), vec![1.0, 4.0]);
        assert_eq!(row_losses(&p, &y, &[], Metric::Absolute).unwrap(), vec![1.0, 2.0]);
        assert!(row_losses(&p, &y, &[], Metric::Correlation).is_none());
        assert!(row_losses(&[f64::NAN, 1.0], &y, &[], Metric::Squared).unwrap()[0].is_infinite());
        let ll = row_losses(&[0.0, 0.0], &[0.0, 1.0], &[], Metric::LogLoss).unwrap();
        assert!(close(ll[0], std::f64::consts::LN_2, 1e-15));
    }
}
