//! Rung family on the declared structural term (#257).
//!
//! #254 gives the zero-parameter rung `S`, the label a zero return produces
//! (`TargetSpec::null_prediction`, a function of the scale column `g`). This
//! module adds the rungs that fit parameters on fit-train only:
//!
//! * `affine`: `a + b*S` by (weighted) least squares, 2 parameters;
//! * `binned20`: the train mean of `y` in 20 equal-count bins of `S`, edges
//!   from train, ties ordered by `(S, row position)`, 20 parameters;
//! * `construction_ols`: least squares of `y` on the referenced column `g`
//!   plus an intercept, 2 parameters.
//!
//! A spec has a structural term when [`structural`] says so: a scale column and
//! a non-zero offset or an asymmetric clip. Otherwise `S` is constant and the
//! reference is the mean.

use crate::target::TargetSpec;

pub const BINS: usize = 20;

/// True when the spec declares a structural term that varies by row.
pub fn structural(spec: &TargetSpec) -> bool {
    spec.structural_term()
}

/// `S` for each scale value.
pub fn structure_values(spec: &TargetSpec, g: &[f64]) -> Vec<f64> {
    g.iter().map(|&v| spec.null_prediction(v)).collect()
}

/// Fitted rung parameters.
#[derive(Clone, Debug)]
pub struct RungModels {
    pub a: f64,
    pub b: f64,
    /// Interior bin edges (19), ascending; a value goes to the bin equal to
    /// the number of edges at or below it.
    pub edges: Vec<f64>,
    /// Mean of `y` per bin (20); an empty bin carries the overall mean.
    pub bin_means: Vec<f64>,
    pub c0: f64,
    pub c1: f64,
    pub fit_rows: usize,
    /// Rows left out of the fit because `S`, `g` or `y` was not finite.
    pub non_finite_rows: usize,
}

/// The four rung predictions on one sample, aligned with its rows. A row with
/// a non-finite `S` or `g` predicts NaN in every rung.
#[derive(Clone, Debug)]
pub struct RungPredictions {
    pub zero_param: Vec<f64>,
    pub affine: Vec<f64>,
    pub binned20: Vec<f64>,
    pub construction_ols: Vec<f64>,
}

fn weight(w: &[f64], i: usize) -> f64 {
    if w.is_empty() {
        1.0
    } else {
        w[i]
    }
}

/// Weighted least squares line `y = c0 + c1 x`; `(mean, 0)` when `x` is constant.
fn line(x: &[f64], y: &[f64], w: &[f64], rows: &[usize]) -> (f64, f64) {
    let mut sw = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &i in rows {
        let wi = weight(w, i);
        sw += wi;
        sx += wi * x[i];
        sy += wi * y[i];
    }
    if !(sw > 0.0) {
        return (f64::NAN, 0.0);
    }
    let (mx, my) = (sx / sw, sy / sw);
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &i in rows {
        let wi = weight(w, i);
        let dx = x[i] - mx;
        sxx += wi * dx * dx;
        sxy += wi * dx * (y[i] - my);
    }
    if sxx > 0.0 {
        let b = sxy / sxx;
        (my - b * mx, b)
    } else {
        (my, 0.0)
    }
}

/// Fit the rungs on the fit-train rows. `s`, `g`, `y` and `w` (empty means
/// all ones) are aligned; a zero-weight row is ignored.
pub fn fit(s: &[f64], g: &[f64], y: &[f64], w: &[f64]) -> RungModels {
    let n = s.len().min(g.len()).min(y.len());
    let mut rows: Vec<usize> = Vec::with_capacity(n);
    let mut non_finite = 0usize;
    for i in 0..n {
        if !(s[i].is_finite() && g[i].is_finite() && y[i].is_finite()) {
            non_finite += 1;
        } else if weight(w, i) > 0.0 {
            rows.push(i);
        }
    }
    let (a, b) = line(s, y, w, &rows);
    let (c0, c1) = line(g, y, w, &rows);

    let mut order = rows.clone();
    order.sort_by(|&p, &q| s[p].total_cmp(&s[q]).then(p.cmp(&q)));
    let m = order.len();
    let mut edges = Vec::with_capacity(BINS - 1);
    let mut start = 0usize;
    for bin in 1..BINS {
        // First rank whose bin index is `bin`.
        while start < m && start * BINS / m < bin {
            start += 1;
        }
        if start < m {
            edges.push(s[order[start]]);
        }
    }
    let bin_of = |v: f64| edges.partition_point(|&e| e <= v);
    let mut sum = vec![0.0; BINS];
    let mut mass = vec![0.0; BINS];
    let mut total = 0.0;
    let mut total_mass = 0.0;
    for &i in &rows {
        let wi = weight(w, i);
        let bin = bin_of(s[i]);
        sum[bin] += wi * y[i];
        mass[bin] += wi;
        total += wi * y[i];
        total_mass += wi;
    }
    let overall = if total_mass > 0.0 { total / total_mass } else { f64::NAN };
    let bin_means = (0..BINS)
        .map(|k| if mass[k] > 0.0 { sum[k] / mass[k] } else { overall })
        .collect();
    RungModels {
        a,
        b,
        edges,
        bin_means,
        c0,
        c1,
        fit_rows: m,
        non_finite_rows: non_finite,
    }
}

/// Predict every rung on one sample.
pub fn predict(m: &RungModels, s: &[f64], g: &[f64]) -> RungPredictions {
    let n = s.len().min(g.len());
    let mut out = RungPredictions {
        zero_param: Vec::with_capacity(n),
        affine: Vec::with_capacity(n),
        binned20: Vec::with_capacity(n),
        construction_ols: Vec::with_capacity(n),
    };
    for i in 0..n {
        if s[i].is_finite() && g[i].is_finite() {
            out.zero_param.push(s[i]);
            out.affine.push(m.a + m.b * s[i]);
            out.binned20
                .push(m.bin_means[m.edges.partition_point(|&e| e <= s[i])]);
            out.construction_ols.push(m.c0 + m.c1 * g[i]);
        } else {
            out.zero_param.push(f64::NAN);
            out.affine.push(f64::NAN);
            out.binned20.push(f64::NAN);
            out.construction_ols.push(f64::NAN);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Four rows: y = clip((r - 0.002) / (6 s), -1, 1), S = clip(-0.002 / (6 s)).
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let g = vec![0.001, 0.002, 0.004, 0.008];
        let r = [0.01f64, -0.01, 0.0, 0.005];
        let y: Vec<f64> = g
            .iter()
            .zip(r)
            .map(|(s, r)| ((r - 0.002) / (6.0 * *s)).max(-1.0).min(1.0))
            .collect();
        let s: Vec<f64> = g.iter().map(|s| (-0.002f64 / (6.0 * *s)).max(-1.0).min(1.0)).collect();
        (s, g, y)
    }

    fn mse(p: &[f64], y: &[f64]) -> f64 {
        p.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / p.len() as f64
    }

    #[test]
    fn zero_param_rung_matches_hand_computed_mse() {
        let (s, g, y) = fixture();
        let m = fit(&s, &g, &y, &[]);
        let p = predict(&m, &s, &g);
        assert!((mse(&p.zero_param, &y) - 0.6207682292).abs() <= 1e-10);
        let mean = y.iter().sum::<f64>() / 4.0;
        assert!((mse(&[mean; 4], &y) - 0.5026855469).abs() <= 1e-10);
    }

    #[test]
    fn affine_rung_matches_closed_form() {
        let (s, g, y) = fixture();
        let m = fit(&s, &g, &y, &[]);
        assert!((m.a - -0.5235507246).abs() <= 1e-10, "{}", m.a);
        assert!((m.b - -3.3173913043).abs() <= 1e-10, "{}", m.b);
        let p = predict(&m, &s, &g);
        assert!((mse(&p.affine, &y) - 0.3653608092).abs() <= 1e-10);
    }

    #[test]
    fn binned_rung_edges_come_from_train_only() {
        let s: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
        let g: Vec<f64> = s.iter().map(|v| v + 1.0).collect();
        let y: Vec<f64> = s.iter().map(|v| v * 2.0).collect();
        let a = fit(&s, &g, &y, &[]);
        assert_eq!(a.edges.len(), BINS - 1);
        // Validation or test values never enter `fit`; predicting on other
        // values leaves the edges and bin means untouched.
        let b = fit(&s, &g, &y, &[]);
        let _ = predict(&b, &[0.5, 9.0], &[1.5, 10.0]);
        for (x, z) in a.edges.iter().zip(&b.edges) {
            assert_eq!(x.to_bits(), z.to_bits());
        }
        for (x, z) in a.bin_means.iter().zip(&b.bin_means) {
            assert_eq!(x.to_bits(), z.to_bits());
        }
        // Equal-count bins: five rows each.
        assert!((a.edges[0] - 0.05).abs() <= 1e-15);
        assert!((a.bin_means[0] - 0.04).abs() <= 1e-15);
    }

    #[test]
    fn non_finite_structure_rows_are_counted() {
        let s = vec![0.1, f64::NAN, 0.3, 0.4];
        let g = vec![1.0, 2.0, f64::INFINITY, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let m = fit(&s, &g, &y, &[]);
        assert_eq!(m.non_finite_rows, 2);
        assert_eq!(m.fit_rows, 2);
        let p = predict(&m, &s, &g);
        assert!(p.affine[1].is_nan() && p.zero_param[2].is_nan());
        assert!(p.affine[0].is_finite());
    }

    #[test]
    fn ties_in_the_structure_share_a_bin_and_mean() {
        let s = vec![1.0; 40];
        let g: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..40).map(|i| (i % 2) as f64).collect();
        let m = fit(&s, &g, &y, &[]);
        assert_eq!((m.a - 0.5).abs() < 1e-15, true);
        assert_eq!(m.b, 0.0);
        let p = predict(&m, &[1.0], &[0.0]);
        assert!((p.binned20[0] - 0.5).abs() < 1e-15);
    }
}
