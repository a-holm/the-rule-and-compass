//! Error metrics.
//!
//! Every metric returns a number where **lower is better** and `+inf` means
//! "this prediction is unusable" (empty input, a non-finite prediction, or a
//! target the metric cannot interpret). The search minimises the metric, and
//! the Pareto front is built against it, so the choice of metric decides what
//! "a good model" means.
//!
//! Metrics split into three families:
//!
//! * **regression** — functions of `(prediction, target)`;
//! * **classification** — the target is read as two classes and the model
//!   output as a logit (see [`Metric::is_classification`]); the label is
//!   `y > midpoint` so the metric is unaffected by the affine normalisation the
//!   engine applies to the target;
//! * **information criteria** — fit measured against the number of fitted
//!   constants, for model selection ([`Metric::is_parameter_aware`]).
//!
//! Two **implicit** metrics ([`Metric::is_implicit`], issue #14) score a
//! candidate `f(x1..xn, y)` that should vanish on the data rather than predict
//! `y`: the plain residual `|f|` and the distance to the level set
//! `|f| / ||∇f||`. Both are computed from the tree and a row by
//! [`crate::implicit`], not from a prediction vector.
//!
//! See `docs/overfitting.md` for when to prefer which and `docs/implicit.md`
//! for the implicit model.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Mean absolute error.
    Absolute,
    /// Mean squared error.
    Squared,
    /// `1 - |Pearson r|`.
    Correlation,
    /// Median absolute error.
    Median,
    /// Worst-case (maximum) absolute error.
    Maximum,
    /// Absolute mean signed error: penalises a consistent bias.
    Signed,
    /// Mean `ln(1 + e^2)`: squared error that does not chase outliers.
    LogSquashed,
    /// `1 - |Spearman rank correlation|`.
    RankCorrelation,
    /// Mean absolute error over the middle 50% of residuals.
    Interquartile,
    /// `1 - AUC` (classification).
    Auc,
    /// Mean logistic (cross-entropy) loss (classification).
    LogLoss,
    /// Mean hinge loss (classification).
    Hinge,
    /// Mean absolute error of the first difference: how well slopes match.
    Slope,
    /// Mean distance from each point to the candidate's zero level set,
    /// `|f(p)| / ||∇f(p)||` (implicit model, issue #14).
    ImplicitDerivative,
    /// Mean `|f(p)|` for an implicit model `f(inputs, y) = 0` (issue #14).
    ImplicitResidual,
    /// Akaike information criterion.
    Aic,
    /// Small-sample-corrected AIC.
    Aicc,
    /// Bayesian information criterion.
    Bic,
}

impl Metric {
    /// Every metric, in a sensible presentation order.
    pub fn all() -> &'static [Metric] {
        use Metric::*;
        &[
            Absolute,
            Squared,
            Correlation,
            Median,
            Maximum,
            Signed,
            LogSquashed,
            RankCorrelation,
            Interquartile,
            Slope,
            ImplicitDerivative,
            ImplicitResidual,
            Auc,
            LogLoss,
            Hinge,
            Aic,
            Aicc,
            Bic,
        ]
    }

    /// Parse a metric name, returning `None` for an unknown name so the caller
    /// can report it. Accepts the usual aliases.
    pub fn parse_opt(s: &str) -> Option<Metric> {
        use Metric::*;
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "absolute" | "abs" | "mae" | "l1" => Absolute,
            "squared" | "square" | "mse" | "l2" => Squared,
            "correlation" | "corr" | "r" => Correlation,
            "median" | "medae" => Median,
            "maximum" | "max" | "max_error" | "worst_case" => Maximum,
            "signed" | "bias" => Signed,
            "log_squashed" | "logsquashed" | "log" => LogSquashed,
            "rank_correlation" | "rank" | "spearman" => RankCorrelation,
            "interquartile" | "iqr" | "iqmae" => Interquartile,
            "auc" | "roc" => Auc,
            "log_loss" | "logloss" | "cross_entropy" => LogLoss,
            "hinge" => Hinge,
            "slope" | "derivative" => Slope,
            "implicit" | "implicit_derivative" | "implicit_deriv" | "level_set"
            | "levelset" => ImplicitDerivative,
            "implicit_residual" | "implicit_resid" | "implicit_abs" | "implicit_l1" => {
                ImplicitResidual
            }
            "aic" => Aic,
            "aicc" => Aicc,
            "bic" => Bic,
            _ => return None,
        })
    }

    /// Lenient parse used where a default is acceptable (falls back to
    /// [`Metric::Absolute`]). The CLI uses [`Metric::parse_opt`] so that a typo
    /// is an error rather than a silent change of objective.
    pub fn parse(s: &str) -> Metric {
        Self::parse_opt(s).unwrap_or(Metric::Absolute)
    }

    pub fn name(self) -> &'static str {
        use Metric::*;
        match self {
            Absolute => "absolute",
            Squared => "squared",
            Correlation => "correlation",
            Median => "median",
            Maximum => "maximum",
            Signed => "signed",
            LogSquashed => "log_squashed",
            RankCorrelation => "rank_correlation",
            Interquartile => "interquartile",
            Auc => "auc",
            LogLoss => "log_loss",
            Hinge => "hinge",
            Slope => "slope",
            ImplicitDerivative => "implicit",
            ImplicitResidual => "implicit_residual",
            Aic => "aic",
            Aicc => "aicc",
            Bic => "bic",
        }
    }

    /// True for metrics that read the target as two classes. The label is
    /// `y > (min+max)/2`; the model output is treated as a logit.
    pub fn is_classification(self) -> bool {
        matches!(self, Metric::Auc | Metric::LogLoss | Metric::Hinge)
    }

    /// True for metrics that depend on the number of fitted constants.
    pub fn is_parameter_aware(self) -> bool {
        matches!(self, Metric::Aic | Metric::Aicc | Metric::Bic)
    }

    /// True for the implicit metrics (issue #14). They are scored from the tree
    /// and a row by [`crate::implicit`], because they need `f` and its gradient
    /// rather than a prediction vector.
    pub fn is_implicit(self) -> bool {
        matches!(self, Metric::ImplicitDerivative | Metric::ImplicitResidual)
    }

    /// True for the level-set distance `|f| / ||∇f||`. Only this metric changes
    /// how the search fits constants (see [`crate::implicit::fit`]).
    pub fn is_implicit_derivative(self) -> bool {
        self == Metric::ImplicitDerivative
    }
}

/// Score with the number of fitted constants set to zero. Used by callers that
/// do not know (or do not need) the model's parameter count.
pub fn score(pred: &[f64], y: &[f64], m: Metric) -> f64 {
    score_with_constants(pred, y, m, 0)
}

/// Score a prediction against a target. Lower is better; `+inf` is "invalid".
///
/// `k` is the number of fitted constants, needed only by the information
/// criteria ([`Metric::is_parameter_aware`]).
pub fn score_with_constants(pred: &[f64], y: &[f64], m: Metric, k: usize) -> f64 {
    let n = pred.len();
    if n == 0 || y.len() != n {
        return f64::INFINITY;
    }
    for &p in pred {
        if !p.is_finite() {
            return f64::INFINITY;
        }
    }
    let resid: Vec<f64> = pred.iter().zip(y).map(|(p, t)| p - t).collect();
    match m {
        Metric::Absolute => mean_abs(&resid),
        Metric::Squared => mean_sq(&resid),
        Metric::Correlation => 1.0 - correlation(pred, y).abs(),
        Metric::Median => median_abs(&resid),
        Metric::Maximum => resid.iter().fold(0.0f64, |a, e| a.max(e.abs())),
        Metric::Signed => mean(&resid).abs(),
        Metric::LogSquashed => resid.iter().map(|e| (1.0 + e * e).ln()).sum::<f64>() / n as f64,
        Metric::RankCorrelation => 1.0 - rank_correlation(pred, y).abs(),
        Metric::Interquartile => interquartile_mean_abs(&resid),
        Metric::Slope => {
            if n < 2 {
                return f64::INFINITY;
            }
            let mut acc = 0.0;
            for i in 1..n {
                acc += ((pred[i] - pred[i - 1]) - (y[i] - y[i - 1])).abs();
            }
            acc / (n - 1) as f64
        }
        Metric::Auc => 1.0 - auc(pred, y),
        Metric::LogLoss => log_loss(pred, y),
        Metric::Hinge => hinge(pred, y),
        Metric::Aic => aic(n, rss(&resid), k),
        Metric::Aicc => aicc(n, rss(&resid), k),
        Metric::Bic => bic(n, rss(&resid), k),
        // Implicit metrics need the expression tree (for ∇f), not a prediction
        // vector. [`crate::search::eval_error`] routes them to
        // `crate::implicit::score` before this function is called; reaching
        // here means the caller has no tree, which is unusable.
        Metric::ImplicitDerivative | Metric::ImplicitResidual => f64::INFINITY,
    }
}

/// True when `m` has a defined weighted form. The CLI refuses `--weight` for
/// every other metric rather than silently ignoring the weights.
///
/// Supported: `absolute`, `squared`, `correlation`, `signed`, `log_squashed`,
/// `implicit`, `implicit_residual`, `aic`, `aicc` and `bic`. The order-statistic,
/// classification, rank and first-difference metrics have no standard weighted
/// definition (see `docs/weights.md`).
pub fn weights_supported(m: Metric) -> bool {
    matches!(
        m,
        Metric::Absolute
            | Metric::Squared
            | Metric::Correlation
            | Metric::Signed
            | Metric::LogSquashed
            | Metric::ImplicitDerivative
            | Metric::ImplicitResidual
            | Metric::Aic
            | Metric::Aicc
            | Metric::Bic
    )
}

/// Names of the metrics that accept weights, for error messages.
pub fn weighted_metric_names() -> &'static [&'static str] {
    &[
        "absolute",
        "squared",
        "correlation",
        "signed",
        "log_squashed",
        "implicit",
        "implicit_residual",
        "aic",
        "aicc",
        "bic",
    ]
}

/// Score a prediction against a target with per-row weights.
///
/// With an empty `w` this is exactly [`score_with_constants`]: the unweighted
/// code path is unchanged, bit for bit. Otherwise every supported metric is a
/// **weighted mean**:
///
/// ```text
/// absolute    Σ wᵢ·|eᵢ| / Σ wᵢ
/// squared     Σ wᵢ·eᵢ² / Σ wᵢ
/// signed      |Σ wᵢ·eᵢ / Σ wᵢ|
/// log_squashed Σ wᵢ·ln(1 + eᵢ²) / Σ wᵢ
/// correlation 1 - |r_w|, the weighted Pearson correlation
/// aic/aicc/bic weighted RSS, n = number of rows with wᵢ > 0
/// ```
///
/// Normalising by `Σ wᵢ` (not by the row count) is what makes a weight of 2
/// mean "this row twice": both the numerator and the denominator grow by that
/// row's loss and one unit of weight. For an unsupported metric the weights
/// cannot be applied, so the result is `+inf`; the CLI refuses such a
/// combination up front.
///
/// A zero-weight row is invisible to both the fitter and the scorer: it is
/// skipped before any arithmetic, and its prediction and target may be
/// non-finite without making the candidate invalid. The finiteness guard
/// therefore applies only to rows with `wᵢ > 0` (the unweighted path in
/// [`score_with_constants`] still requires every prediction to be finite).
pub fn score_weighted(pred: &[f64], y: &[f64], w: &[f64], m: Metric, k: usize) -> f64 {
    if w.is_empty() {
        return score_with_constants(pred, y, m, k);
    }
    let n = pred.len();
    if n == 0 || y.len() != n || w.len() != n {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    for &wi in w {
        if !wi.is_finite() || wi < 0.0 {
            return f64::INFINITY;
        }
        total += wi;
    }
    if !(total > 0.0) {
        return f64::INFINITY;
    }
    for (i, &p) in pred.iter().enumerate() {
        if w[i] > 0.0 && !p.is_finite() {
            return f64::INFINITY;
        }
    }
    // A masked row's residual is forced to zero so that no `wᵢ * loss`
    // product can become `0 * inf = NaN`; the row then contributes nothing.
    let resid: Vec<f64> = pred
        .iter()
        .zip(y)
        .zip(w)
        .map(|((p, t), &wi)| if wi > 0.0 { p - t } else { 0.0 })
        .collect();
    match m {
        Metric::Absolute => resid
            .iter()
            .zip(w)
            .map(|(e, wi)| wi * e.abs())
            .sum::<f64>()
            / total,
        Metric::Squared => resid
            .iter()
            .zip(w)
            .map(|(e, wi)| wi * e * e)
            .sum::<f64>()
            / total,
        Metric::Correlation => 1.0 - weighted_correlation(pred, y, w).abs(),
        Metric::Signed => {
            (resid.iter().zip(w).map(|(e, wi)| wi * e).sum::<f64>() / total).abs()
        }
        Metric::LogSquashed => resid
            .iter()
            .zip(w)
            .map(|(e, wi)| wi * (1.0 + e * e).ln())
            .sum::<f64>()
            / total,
        Metric::Aic => {
            let npos = w.iter().filter(|&&v| v > 0.0).count();
            aic(npos, weighted_rss(&resid, w), k)
        }
        Metric::Aicc => {
            let npos = w.iter().filter(|&&v| v > 0.0).count();
            aicc(npos, weighted_rss(&resid, w), k)
        }
        Metric::Bic => {
            let npos = w.iter().filter(|&&v| v > 0.0).count();
            bic(npos, weighted_rss(&resid, w), k)
        }
        // No standard weighted form: refuse rather than ignore the weights.
        _ => f64::INFINITY,
    }
}

fn weighted_rss(v: &[f64], w: &[f64]) -> f64 {
    v.iter()
        .zip(w)
        .filter(|(_, &wi)| wi > 0.0)
        .map(|(e, wi)| wi * e * e)
        .sum()
}

/// Weighted Pearson correlation with weighted means and covariance.
///
/// Rows with zero weight are skipped before their values are read, so a
/// non-finite value in a masked row cannot poison the moments.
fn weighted_correlation(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    if a.len() != b.len() || w.len() != a.len() || a.is_empty() {
        return 0.0;
    }
    let total: f64 = w.iter().filter(|&&wi| wi > 0.0).sum();
    if !(total > 0.0) {
        return 0.0;
    }
    let mut ma = 0.0;
    let mut mb = 0.0;
    for i in 0..a.len() {
        let wi = w[i];
        if wi > 0.0 {
            ma += wi * a[i];
            mb += wi * b[i];
        }
    }
    ma /= total;
    mb /= total;
    let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let wi = w[i];
        if wi > 0.0 {
            let da = a[i] - ma;
            let db = b[i] - mb;
            sab += wi * da * db;
            sa += wi * da * da;
            sb += wi * db * db;
        }
    }
    if sa <= 0.0 || sb <= 0.0 {
        0.0
    } else {
        sab / (sa.sqrt() * sb.sqrt())
    }
}

/// Gaussian log-likelihood term `n·ln(RSS/n)`, guarding the perfect fit
/// (`RSS = 0`, which sends every criterion to `-inf`).
fn loglik(n: usize, rss: f64) -> f64 {
    if rss <= 0.0 {
        f64::NEG_INFINITY
    } else {
        n as f64 * (rss / n as f64).ln()
    }
}

fn aic(n: usize, rss: f64, k: usize) -> f64 {
    loglik(n, rss) + 2.0 * k as f64
}

fn aicc(n: usize, rss: f64, k: usize) -> f64 {
    let base = loglik(n, rss);
    if !base.is_finite() {
        return base;
    }
    let denom = n as f64 - k as f64 - 1.0;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    base + 2.0 * k as f64 + 2.0 * k as f64 * (k as f64 + 1.0) / denom
}

fn bic(n: usize, rss: f64, k: usize) -> f64 {
    loglik(n, rss) + k as f64 * (n as f64).ln()
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn mean_abs(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|e| e.abs()).sum::<f64>() / v.len() as f64
}

fn mean_sq(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64
}

fn rss(v: &[f64]) -> f64 {
    v.iter().map(|e| e * e).sum()
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

/// Median of the absolute errors. `+inf` is impossible here (inputs are finite
/// and non-empty), so the empty case returns 0 by convention.
fn median_abs(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let s = sorted(&v.iter().map(|e| e.abs()).collect::<Vec<_>>());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Mean of the absolute errors that fall between the 25th and 75th percentile —
/// the "interquartile mean absolute error". Robust to extreme residuals without
/// throwing most of the data away, as the median does.
fn interquartile_mean_abs(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let s = sorted(&v.iter().map(|e| e.abs()).collect::<Vec<_>>());
    let n = s.len();
    if n < 4 {
        return mean(&s);
    }
    let lo = n / 4;
    let hi = (3 * n) / 4;
    let slice = &s[lo..hi.max(lo + 1)];
    mean(slice)
}

/// Average ranks (1-based), with ties sharing their mean rank.
fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // ranks i+1 .. j share the average
        let avg = 0.5 * ((i + 1) as f64 + j as f64);
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

/// Spearman rank correlation.
pub fn rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    if a.len() == 1 {
        return 0.0;
    }
    correlation(&ranks(a), &ranks(b))
}

/// Pearson correlation.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        sa += da * da;
        sb += db * db;
    }
    if sa <= 0.0 || sb <= 0.0 {
        0.0
    } else {
        sab / (sa.sqrt() * sb.sqrt())
    }
}

/// Split the target into two classes at its midpoint. The engine normalises the
/// target, so an absolute 0.5 threshold would be wrong; the midpoint of the two
/// observed extremes is invariant under the affine normalisation.
fn binary_labels(y: &[f64]) -> Vec<f64> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in y {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let mid = 0.5 * (lo + hi);
    y.iter().map(|&v| if v > mid { 1.0 } else { 0.0 }).collect()
}

fn sigmoid(x: f64) -> f64 {
    // Numerically stable logistic.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Area under the ROC curve, by the rank (Mann–Whitney U) formula.
///
/// Returns `0.5` (**chance**) when either class is empty, so the resulting
/// `1 - AUC` is a `0.5` error rather than a free win.
pub fn auc(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return 0.5;
    }
    let labels = binary_labels(y);
    let npos = labels.iter().filter(|&&l| l > 0.5).count();
    let nneg = labels.len() - npos;
    if npos == 0 || nneg == 0 {
        return 0.5;
    }
    let r = ranks(pred);
    let sum_pos_ranks: f64 = r
        .iter()
        .zip(&labels)
        .filter(|(_, &l)| l > 0.5)
        .map(|(r, _)| *r)
        .sum();
    let u = sum_pos_ranks - (npos * (npos + 1)) as f64 / 2.0;
    u / (npos * nneg) as f64
}

/// Mean logistic loss, treating the prediction as a logit.
pub fn log_loss(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let eps = 1e-12;
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        let q = sigmoid(*p).clamp(eps, 1.0 - eps);
        acc += -(l * q.ln() + (1.0 - l) * (1.0 - q).ln());
    }
    acc / pred.len() as f64
}

/// Mean hinge loss, treating the prediction as a logit mapped to `(-1, 1)`.
pub fn hinge(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        let t = 2.0 * l - 1.0;
        let s = 2.0 * sigmoid(*p) - 1.0;
        acc += (1.0 - t * s).max(0.0);
    }
    acc / pred.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn parse_covers_all_names_and_rejects_typos() {
        for m in Metric::all() {
            assert_eq!(Metric::parse_opt(m.name()), Some(*m), "{}", m.name());
        }
        assert_eq!(Metric::parse_opt("nonsense"), None);
        assert_eq!(Metric::parse_opt("MAE"), Some(Metric::Absolute));
        assert_eq!(Metric::parse_opt("mse"), Some(Metric::Squared));
    }

    #[test]
    fn absolute_and_squared_are_hand_checkable() {
        // residuals: -1, 2, -3  -> MAE = 2, MSE = 14/3
        let p = [0.0, 3.0, 0.0];
        let y = [1.0, 1.0, 3.0];
        assert!(approx(score(&p, &y, Metric::Absolute), 2.0, 1e-12));
        assert!(approx(score(&p, &y, Metric::Squared), 14.0 / 3.0, 1e-12));
    }

    #[test]
    fn median_max_signed_logsq_and_iqr() {
        // residuals: -3, -1, 2, 4
        let p = [0.0, 0.0, 0.0, 0.0];
        let y = [3.0, 1.0, -2.0, -4.0];
        assert!(approx(score(&p, &y, Metric::Median), 2.5, 1e-12)); // |e| = 3,1,2,4 -> med 2.5
        assert!(approx(score(&p, &y, Metric::Maximum), 4.0, 1e-12));
        assert!(approx(
            score(&p, &y, Metric::Signed),
            (3.0f64 + 1.0 - 2.0 - 4.0).abs() / 4.0,
            1e-12
        ));
        let logsq =
            ((1.0 + 9.0f64).ln() + (1.0 + 1.0f64).ln() + (1.0 + 4.0f64).ln() + (1.0 + 16.0f64).ln()) / 4.0;
        assert!(approx(score(&p, &y, Metric::LogSquashed), logsq, 1e-12));
        // |e| sorted: 1,2,3,4 ; middle 50% is indices [1,3) = {2,3} -> 2.5
        assert!(approx(score(&p, &y, Metric::Interquartile), 2.5, 1e-12));
    }

    #[test]
    fn correlation_and_rank_correlation() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0]; // perfect positive
        assert!(approx(score(&a, &b, Metric::Correlation), 0.0, 1e-12));
        // Rank correlation is 1 for any strictly increasing transform.
        let c = [1.0, 4.0, 9.0, 16.0];
        assert!(approx(score(&a, &c, Metric::RankCorrelation), 0.0, 1e-12));
        // Reversed -> |r| = 1 as well.
        let d = [4.0, 3.0, 2.0, 1.0];
        assert!(approx(score(&a, &d, Metric::Correlation), 0.0, 1e-12));
    }

    #[test]
    fn slope_metric_compares_first_differences() {
        // pred differences: 1,1 ; target differences: 1,1 -> 0
        let p = [0.0, 1.0, 2.0];
        let y = [5.0, 6.0, 7.0];
        assert!(approx(score(&p, &y, Metric::Slope), 0.0, 1e-12));
        // target flat, prediction rising -> mean |1 - 0| over 2 steps = 1
        let y2 = [5.0, 5.0, 5.0];
        assert!(approx(score(&p, &y2, Metric::Slope), 1.0, 1e-12));
    }

    #[test]
    fn classification_metrics() {
        // Two separable classes: ideal ranking.
        let y = [0.0, 0.0, 1.0, 1.0];
        let pred = [-2.0, -1.0, 1.0, 2.0];
        assert!(approx(auc(&pred, &y), 1.0, 1e-12));
        assert!(approx(score(&pred, &y, Metric::Auc), 0.0, 1e-12));
        // Reversed ranking -> AUC 0, error 1.
        let bad = [2.0, 1.0, -1.0, -2.0];
        assert!(approx(auc(&bad, &y), 0.0, 1e-12));
        // Log loss and hinge get smaller as the margin grows.
        let strong = [-4.0, -3.0, 3.0, 4.0];
        assert!(score(&strong, &y, Metric::LogLoss) < score(&pred, &y, Metric::LogLoss));
        assert!(score(&strong, &y, Metric::Hinge) < score(&pred, &y, Metric::Hinge));
        // Empty class -> chance (0.5), not a free win.
        let one_class = [0.0, 0.0];
        assert!(approx(auc(&pred[..2], &one_class), 0.5, 1e-12));
    }

    #[test]
    fn information_criteria_metrics() {
        // residuals 1,1 -> rss 2, n 2. AIC = 2 ln(1) + 2k = 2k.
        let p = [0.0, 0.0];
        let y = [1.0, 1.0];
        assert!(approx(score_with_constants(&p, &y, Metric::Aic, 1), 2.0, 1e-12));
        // AICc adds 2k(k+1)/(n-k-1); with n=2,k=1 the denominator is 0 -> inf.
        assert!(!score_with_constants(&p, &y, Metric::Aicc, 1).is_finite());
        // A perfect fit -> -inf.
        assert_eq!(score(&p, &p, Metric::Aic), f64::NEG_INFINITY);
    }

    #[test]
    fn non_finite_prediction_is_invalid() {
        let y = [1.0, 2.0];
        assert_eq!(score(&[f64::NAN, 1.0], &y, Metric::Absolute), f64::INFINITY);
        assert_eq!(score(&[1.0], &y, Metric::Absolute), f64::INFINITY);
    }

    #[test]
    fn weighted_means_are_hand_checkable() {
        // residuals -1, 2, -3 with weights 1, 2, 1: Σw = 4.
        let p = [0.0, 3.0, 0.0];
        let y = [1.0, 1.0, 3.0];
        let w = [1.0, 2.0, 1.0];
        assert!(approx(
            score_weighted(&p, &y, &w, Metric::Absolute, 0),
            (1.0 + 4.0 + 3.0) / 4.0,
            1e-12
        ));
        assert!(approx(
            score_weighted(&p, &y, &w, Metric::Squared, 0),
            (1.0 + 8.0 + 9.0) / 4.0,
            1e-12
        ));
        // Signed: (−1 + 4 − 3)/4 = 0.
        assert!(approx(
            score_weighted(&p, &y, &w, Metric::Signed, 0),
            0.0,
            1e-12
        ));
    }

    #[test]
    fn empty_weights_are_exactly_the_unweighted_score() {
        let p = [0.2, -1.0, 3.5, 2.0];
        let y = [1.0, 2.0, -0.5, 0.0];
        for m in Metric::all() {
            assert_eq!(
                score_weighted(&p, &y, &[], *m, 2).to_bits(),
                score_with_constants(&p, &y, *m, 2).to_bits(),
                "empty weights changed {}",
                m.name()
            );
        }
    }

    #[test]
    fn weights_supported_is_the_documented_set() {
        let supported: Vec<&str> = Metric::all()
            .iter()
            .filter(|m| weights_supported(**m))
            .map(|m| m.name())
            .collect();
        assert_eq!(supported, weighted_metric_names());
        assert!(!weights_supported(Metric::Median));
        assert!(!weights_supported(Metric::Auc));
        assert_eq!(
            score_weighted(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 1.0], Metric::Median, 0),
            f64::INFINITY,
            "an unsupported metric must refuse weights, not ignore them"
        );
    }

    #[test]
    fn weighted_correlation_matches_duplicating_the_heavier_row() {
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [0.0, 2.0, 2.5, 4.0];
        let w = [1.0, 2.0, 1.0, 1.0];
        let dup_a = [0.0, 1.0, 1.0, 2.0, 3.0];
        let dup_b = [0.0, 2.0, 2.0, 2.5, 4.0];
        assert!(approx(
            score_weighted(&a, &b, &w, Metric::Correlation, 0),
            score(&dup_a, &dup_b, Metric::Correlation),
            1e-12
        ));
    }
}
