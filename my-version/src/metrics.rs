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
    /// `1 - |Pearson r|`: shape, either direction.
    Correlation,
    /// `1 - Pearson r`: shape **and** direction (an inverted model is
    /// penalised, not rewarded).
    CorrelationSigned,
    /// Median absolute error.
    Median,
    /// Worst-case (maximum) absolute error.
    Maximum,
    /// Absolute mean signed error: penalises a consistent bias.
    Signed,
    /// Mean `ln(1 + e^2)`: squared error that does not chase outliers.
    LogSquashed,
    /// `1 - |Spearman rank correlation|`: monotone shape, either direction.
    RankCorrelation,
    /// `1 - Spearman rank correlation`: monotone shape **and** direction.
    RankCorrelationSigned,
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
            CorrelationSigned,
            Median,
            Maximum,
            Signed,
            LogSquashed,
            RankCorrelation,
            RankCorrelationSigned,
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
            "correlation_signed" | "corr_signed" | "signed_correlation" | "r_signed" => {
                CorrelationSigned
            }
            "median" | "medae" => Median,
            "maximum" | "max" | "max_error" | "worst_case" => Maximum,
            "signed" | "bias" => Signed,
            "log_squashed" | "logsquashed" | "log" => LogSquashed,
            "rank_correlation" | "rank" | "spearman" => RankCorrelation,
            "rank_correlation_signed" | "rank_signed" | "spearman_signed" => {
                RankCorrelationSigned
            }
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
            CorrelationSigned => "correlation_signed",
            Median => "median",
            Maximum => "maximum",
            Signed => "signed",
            LogSquashed => "log_squashed",
            RankCorrelation => "rank_correlation",
            RankCorrelationSigned => "rank_correlation_signed",
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

    /// True for metrics that read the target as **two** classes. The label is
    /// `y > (min+max)/2`; the model output is treated as a logit. More than two
    /// distinct target values are refused by the CLI
    /// ([`classification_target_error`]) rather than silently collapsed to
    /// `positive = {largest class}`.
    pub fn is_classification(self) -> bool {
        matches!(self, Metric::Auc | Metric::LogLoss | Metric::Hinge)
    }

    /// True for metrics that depend on the number of fitted constants.
    pub fn is_parameter_aware(self) -> bool {
        matches!(self, Metric::Aic | Metric::Aicc | Metric::Bic)
    }

    /// True for the metrics whose fitted objective **is** weighted least
    /// squares: `squared` and the information criteria, which fit by least
    /// squares and add their parameter penalty when scoring (#223).
    ///
    /// This is the family whose fitted objective is exactly weighted least
    /// squares. The batched GPU LM also honours `log_squashed` through weighted
    /// IRLS since #229 and `absolute` through Lawson/L1 IRLS with true-MAE
    /// acceptance since #222 ([`crate::fit::fit_loss`]); every other metric
    /// fits by logistic IRLS or the coordinate polish on the CPU, which is a
    /// different objective from the one the kernel would minimise.
    pub fn is_least_squares(self) -> bool {
        matches!(self, Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic)
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
        Metric::CorrelationSigned => 1.0 - correlation(pred, y),
        Metric::Median => median_abs(&resid),
        Metric::Maximum => resid.iter().fold(0.0f64, |a, e| a.max(e.abs())),
        Metric::Signed => mean(&resid).abs(),
        Metric::LogSquashed => resid.iter().map(|e| (1.0 + e * e).ln()).sum::<f64>() / n as f64,
        Metric::RankCorrelation => 1.0 - rank_correlation(pred, y).abs(),
        Metric::RankCorrelationSigned => 1.0 - rank_correlation(pred, y),
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
/// Supported: `absolute`, `squared`, `correlation`, `correlation_signed`,
/// `signed`, `log_squashed`, `implicit`, `implicit_residual`, `aic`, `aicc` and
/// `bic`. The order-statistic, classification, rank and first-difference metrics
/// have no standard weighted definition (see `docs/weights.md`).
pub fn weights_supported(m: Metric) -> bool {
    matches!(
        m,
        Metric::Absolute
            | Metric::Squared
            | Metric::Correlation
            | Metric::CorrelationSigned
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
        "correlation_signed",
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
/// correlation_signed 1 - r_w (direction preserved)
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
        Metric::CorrelationSigned => 1.0 - weighted_correlation(pred, y, w),
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

/// #230: twice the centred average rank, `A_i = 2·r_i − (n + 1)`.
///
/// Average ranks are half-integers and the average rank of any input is
/// exactly `(n + 1)/2` even with ties, so every `A_i` is an integer in
/// `[−(n−1), n−1]` and `Σ A_i = 0`. This is the value the exact rank-correlation
/// path ([`RankSums`]) accumulates, on the CPU and in the CUDA kernel alike.
///
/// Ties are **exact `f64` equality**, so `-0.0` and `+0.0` share a rank (both
/// compare `Equal` and are `==`), and the run order inside a tie block never
/// matters because every member receives the same average rank. Returns `None`
/// when any value is not finite: the rank stage is only defined on finite
/// values, and the scorers reject a non-finite prediction before it gets here.
pub fn centred_ranks_doubled(v: &[f64]) -> Option<Vec<i64>> {
    let n = v.len();
    if v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut a = vec![0i64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Run `[i, j)` holds sorted positions `i+1 ..= j`, whose average rank is
        // `(i + 1 + j)/2`; doubled and centred that is exactly `i + j - n`.
        let centred = (i + j) as i64 - n as i64;
        for &k in &idx[i..j] {
            a[k] = centred;
        }
        i = j;
    }
    Some(a)
}

/// #230: exact integer ingredients of the (tie-corrected) Spearman statistic.
///
/// `ρ = Σ A_i B_i / sqrt(Σ A_i² · Σ B_i²)` with the integer vectors from
/// [`centred_ranks_doubled`]. Every term is exact; only the final division and
/// square root are floating point, so the value does not depend on the order in
/// which the sums were reduced. The CUDA rank stage returns exactly these three
/// integers, which is what makes CPU and GPU scores bit-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RankSums {
    /// `Σ A_i B_i`.
    pub ab: i128,
    /// `Σ A_i²`.
    pub aa: i128,
    /// `Σ B_i²`.
    pub bb: i128,
}

impl RankSums {
    /// `Σ A_i B_i`, `Σ A_i²` and `Σ B_i²` for two equal-length rank vectors.
    pub fn between(a: &[i64], b: &[i64]) -> Option<RankSums> {
        if a.len() != b.len() {
            return None;
        }
        let mut s = RankSums::default();
        for (&x, &y) in a.iter().zip(b) {
            let xv = x as i128;
            let yv = y as i128;
            s.ab += xv * yv;
            s.aa += xv * xv;
            s.bb += yv * yv;
        }
        Some(s)
    }

    /// The correlation itself. Zero variance on either side (a constant
    /// prediction or a constant target) is `0.0`, never `0/0`.
    pub fn correlation(&self) -> f64 {
        if self.aa <= 0 || self.bb <= 0 {
            return 0.0;
        }
        (self.ab as f64) / ((self.aa as f64) * (self.bb as f64)).sqrt()
    }
}

/// #230: the rank-correlation statistic computed exactly from integer ranks.
///
/// Equivalent to [`rank_correlation`] (same tie handling, same value up to the
/// rounding of one division) but independent of reduction order. `None` when an
/// input is empty, of unequal length, or not finite.
pub fn rank_correlation_exact(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let ra = centred_ranks_doubled(a)?;
    let rb = centred_ranks_doubled(b)?;
    let sums = RankSums::between(&ra, &rb)?;
    Some(sums.correlation())
}

/// #230: the target side of the exact rank score, cached per fit sample.
///
/// `coordinate_polish` used to re-sort the target on every trial
/// (`metric_score` → `metrics::ranks`); sorting `y` once per fit removes half
/// the sorts and cannot change the score, because `B` depends only on `y`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RankTarget {
    b: Vec<i64>,
    bb: i128,
}

impl RankTarget {
    /// Rank the target once. `None` when `y` holds a non-finite value.
    pub fn new(y: &[f64]) -> Option<RankTarget> {
        let b = centred_ranks_doubled(y)?;
        let mut bb = 0i128;
        for &v in &b {
            let v = v as i128;
            bb += v * v;
        }
        Some(RankTarget { b, bb })
    }

    pub fn len(&self) -> usize {
        self.b.len()
    }

    pub fn is_empty(&self) -> bool {
        self.b.is_empty()
    }

    /// Cached `B` values, indexed by row.
    pub fn ranks(&self) -> &[i64] {
        &self.b
    }

    /// `Σ B²`, zero when the target is constant.
    pub fn sum_bb(&self) -> i128 {
        self.bb
    }

    /// Exact rank sums of `pred` against this target.
    pub fn sums(&self, pred: &[f64]) -> Option<RankSums> {
        if pred.len() != self.b.len() {
            return None;
        }
        let a = centred_ranks_doubled(pred)?;
        let mut ab = 0i128;
        let mut aa = 0i128;
        for (&x, &y) in a.iter().zip(&self.b) {
            let xv = x as i128;
            ab += xv * (y as i128);
            aa += xv * xv;
        }
        Some(RankSums { ab, aa, bb: self.bb })
    }

    /// Exact score of `pred`: `1 − |ρ|` or `1 − ρ`, with the same degenerate
    /// conventions as [`score_with_constants`] (`n < 2` → `1.0`, a constant
    /// prediction → `1.0`, a non-finite prediction → `+inf`).
    pub fn score(&self, pred: &[f64], signed: bool) -> f64 {
        score_rank_sums(self.sums(pred), signed)
    }
}

/// #230: turn exact rank sums into the reported score.
///
/// `None` means "the prediction or its length was unusable", which the metric
/// contract maps to `+inf`. A one-row sample scores `1.0` like the f64 path.
pub fn score_rank_sums(sums: Option<RankSums>, signed: bool) -> f64 {
    match sums {
        None => f64::INFINITY,
        Some(s) => {
            if s.aa == 0 && s.bb == 0 && s.ab == 0 {
                // Only reachable with `n ≤ 1` (or an all-zero row), where the
                // f64 path also returns 1.0.
                return 1.0;
            }
            let rho = s.correlation();
            if signed {
                1.0 - rho
            } else {
                1.0 - rho.abs()
            }
        }
    }
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

/// Why `y` cannot be used with a binary classification metric, if it cannot.
///
/// `auc`, `log_loss` and `hinge` label a row `y > (min+max)/2`. With three or
/// more distinct values that rule silently merges classes: on `{-1, 0, +1}` it
/// calls `+1` positive and `{0, -1}` negative, so a model that separates
/// `-1` from `{0, +1}` scores as if it had learned nothing. The CLI therefore
/// refuses such a target instead of reporting a meaningless number.
///
/// Exactly two distinct values are valid (including an imbalanced or a
/// single-class sample, which the metric treats as chance). `None` also means
/// the target is empty and has nothing to classify.
pub fn classification_target_error(y: &[f64]) -> Option<String> {
    let mut vals: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.len() != y.len() {
        return Some("the target contains non-finite values".to_string());
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup();
    if vals.len() <= 2 {
        return None;
    }
    let shown: Vec<String> = vals.iter().take(4).map(|v| format!("{v:.6}")).collect();
    Some(format!(
        "the target has {} distinct values ({}); binary metrics need at most two",
        vals.len(),
        shown.join(", ")
    ))
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

    /// #223: exactly the family whose fitted objective is weighted least
    /// squares. The GPU LM dispatch and the CLI refusal both read this
    /// predicate, so the two cannot disagree about which metrics it covers.
    #[test]
    fn the_least_squares_family_is_squared_and_the_information_criteria() {
        let mut seen = Vec::new();
        for m in Metric::all() {
            if m.is_least_squares() {
                seen.push(m.name());
            }
        }
        assert_eq!(seen, ["squared", "aic", "aicc", "bic"]);
        // The information criteria are the parameter-aware least-squares fit;
        // every parameter-aware metric must be in the family.
        for m in Metric::all() {
            if m.is_parameter_aware() {
                assert!(m.is_least_squares(), "{}", m.name());
            }
        }
        // The other solvers are not: IRLS, logistic IRLS, coordinate polish.
        for m in [
            Metric::Absolute,
            Metric::LogSquashed,
            Metric::Auc,
            Metric::LogLoss,
            Metric::Hinge,
            Metric::Median,
            Metric::Maximum,
            Metric::Correlation,
            Metric::RankCorrelation,
            Metric::Slope,
            Metric::ImplicitDerivative,
        ] {
            assert!(!m.is_least_squares(), "{}", m.name());
        }
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
    fn signed_correlation_penalises_an_inverted_model_that_unsigned_cannot() {
        let y = [1.0, 2.0, 3.0, 4.0, 8.0];
        let inverted = [-1.0, -2.0, -3.0, -4.0, -8.0];
        // The unsigned variants cannot tell the right direction from the wrong
        // one: both are a perfect fit of |r|.
        assert!(approx(score(&inverted, &y, Metric::Correlation), 0.0, 1e-12));
        assert!(approx(
            score(&inverted, &y, Metric::RankCorrelation),
            0.0,
            1e-12
        ));
        // The signed variants must reject the inversion.
        assert!(approx(
            score(&inverted, &y, Metric::CorrelationSigned),
            2.0,
            1e-12
        ));
        assert!(approx(
            score(&inverted, &y, Metric::RankCorrelationSigned),
            2.0,
            1e-12
        ));
        // A model with the right sign but a different level is still rewarded.
        let shifted = [11.0, 12.0, 13.0, 14.0, 18.0];
        assert!(approx(
            score(&shifted, &y, Metric::CorrelationSigned),
            0.0,
            1e-12
        ));
        // Even a noisy prediction and its reflection have identical *unsigned*
        // error but opposite signed errors.
        let noisy = [1.0, 2.0, 4.0, 3.0, 9.0];
        let reflected = [9.0, 8.0, 6.0, 7.0, 1.0]; // 2*mean - noisy, mean = 5
        assert!(approx(
            score(&noisy, &y, Metric::Correlation),
            score(&reflected, &y, Metric::Correlation),
            1e-12
        ));
        assert!(
            score(&noisy, &y, Metric::CorrelationSigned) < 0.5
                && score(&reflected, &y, Metric::CorrelationSigned) > 1.5,
            "signed metric must distinguish the reflection"
        );
    }

    #[test]
    fn classification_targets_with_three_values_are_rejected() {
        assert!(classification_target_error(&[0.0, 1.0, 1.0]).is_none());
        assert!(classification_target_error(&[0.0, 0.0]).is_none());
        let err = classification_target_error(&[-1.0, 0.0, 1.0]).unwrap();
        assert!(err.contains("3 distinct values"), "{err}");
        assert!(err.contains("binary"), "{err}");
        // Non-finite values are refused rather than silently dropped.
        assert!(classification_target_error(&[0.0, f64::NAN]).is_some());
    }

    #[test]
    fn signed_correlation_supports_weights() {
        assert!(weights_supported(Metric::CorrelationSigned));
        let p = [0.0, 1.0, 2.0, 3.0];
        let y = [3.0, 2.0, 1.0, 0.0];
        let w = [1.0, 2.0, 1.0, 1.0];
        let dup_p = [0.0, 1.0, 1.0, 2.0, 3.0];
        let dup_y = [3.0, 2.0, 2.0, 1.0, 0.0];
        assert!(approx(
            score_weighted(&p, &y, &w, Metric::CorrelationSigned, 0),
            score(&dup_p, &dup_y, Metric::CorrelationSigned),
            1e-12
        ));
        // Both are perfectly inverted, so the signed score is 2 under weights.
        assert!(approx(
            score_weighted(&p, &y, &w, Metric::CorrelationSigned, 0),
            2.0,
            1e-12
        ));
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

    // --- #230: exact integer ranks -------------------------------------

    /// Deterministic pseudo-random `f64`s in `[0, scale)` (LCG; no crate).
    fn lcg(n: usize, mut s: u64, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * scale
            })
            .collect()
    }

    fn ulps(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let ia = a.to_bits() as i64;
        let ib = b.to_bits() as i64;
        (ia - ib).unsigned_abs()
    }

    #[test]
    fn doubled_centred_ranks_are_integers_that_sum_to_zero() {
        // Strictly increasing: ranks 1..n, so A = 2i - (n-1) in reverse order.
        let v = [10.0, 20.0, 30.0];
        assert_eq!(centred_ranks_doubled(&v).unwrap(), vec![-2, 0, 2]);
        // All tied: every rank is (n+1)/2, so every A is zero.
        assert_eq!(centred_ranks_doubled(&[5.0; 4]).unwrap(), vec![0; 4]);
        // A partial tie block [1, 2, 2] ranks 1, 2.5, 2.5 -> A = -2, 1, 1.
        assert_eq!(centred_ranks_doubled(&[1.0, 2.0, 2.0]).unwrap(), vec![-2, 1, 1]);
        for n in [1usize, 2, 3, 17, 1000] {
            let v = lcg(n, 7 + n as u64, 5.0);
            let a = centred_ranks_doubled(&v).unwrap();
            assert_eq!(a.iter().map(|&x| x as i128).sum::<i128>(), 0, "n={n}");
            assert!(a.iter().all(|&x| (x as i64).abs() < n as i64 || n <= 1));
        }
    }

    #[test]
    fn negative_zero_is_a_tie_and_duplicates_share_the_average_rank() {
        let a = [-0.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let doubled = centred_ranks_doubled(&a).unwrap();
        // Sorted: -0.0/0.0 (tie, ranks 1.5) then 1.0 x3 (ranks 3, 4, 5 -> 4)
        // then 2.0 (rank 6). Doubled centred (n = 6, centre 3.5):
        // -0.0 -> -4, 0.0 -> -4, the three 1.0 rows -> 1, 2.0 -> 5.
        assert_eq!(doubled, vec![-4, -4, 1, 1, 1, 5]);
        // The f64 path agrees: a perfect monotone match is still rho = 1.
        assert_eq!(rank_correlation_exact(&a, &a), Some(1.0));
    }

    #[test]
    fn exact_rank_correlation_agrees_with_the_f64_pearson_path() {
        for (seed, n, scale) in [(1u64, 2usize, 3.0), (2, 17, 1e-9), (3, 1000, 5.0)] {
            let a = lcg(n, seed, scale);
            let b: Vec<f64> = a.iter().map(|v| (v * 0.7).sin() + v / scale).collect();
            let slow = rank_correlation(&a, &b);
            let exact = rank_correlation_exact(&a, &b).unwrap();
            assert!(
                ulps(slow, exact) <= 4,
                "n={n}: f64 {slow:?} vs exact {exact:?} ({} ulps)",
                ulps(slow, exact)
            );
        }
        // Heavy ties: all four values come from two levels, so the tie-corrected
        // form is the only correct one.
        let a = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let b = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let slow = rank_correlation(&a, &b);
        let exact = rank_correlation_exact(&a, &b).unwrap();
        assert!(approx(slow, exact, 1e-15), "{slow} vs {exact}");
    }

    #[test]
    fn exact_rank_score_matches_the_metric_on_unambiguous_cases() {
        let y = [1.0, 2.5, 3.0, 4.0, 9.0];
        let target = RankTarget::new(&y).unwrap();
        // Strictly increasing: rho = 1.
        assert_eq!(target.score(&[-5.0, 0.0, 7.0, 8.0, 100.0], false), 0.0);
        // Strictly decreasing: rho = -1, so the unsigned score is 0 and the
        // signed score is 2.
        assert_eq!(target.score(&[5.0, 4.0, 3.0, 2.0, 1.0], false), 0.0);
        assert_eq!(target.score(&[5.0, 4.0, 3.0, 2.0, 1.0], true), 2.0);
        // Constant prediction: zero variance -> rho = 0 -> 1.0, never NaN.
        let flat = target.score(&[3.0; 5], false);
        assert_eq!(flat, 1.0);
        assert!(flat.is_finite());
        // A constant *target* is also 1.0.
        let flat_target = RankTarget::new(&[2.0; 5]).unwrap();
        assert_eq!(flat_target.score(&[1.0, 2.0, 3.0, 4.0, 5.0], false), 1.0);
        // One row is 1.0 by contract.
        let one = RankTarget::new(&[1.0]).unwrap();
        assert_eq!(one.score(&[4.0], false), 1.0);
        // Non-finite predictions are invalid, not a rank.
        assert_eq!(target.score(&[f64::NAN, 0.0, 7.0, 8.0, 100.0], false), f64::INFINITY);
        assert_eq!(target.score(&[0.0, 1.0], false), f64::INFINITY, "length mismatch");
    }

    #[test]
    fn exact_rank_sums_are_order_independent() {
        // The three integer sums are what the CUDA kernel returns; summing them
        // in a different order (or on a different machine) cannot change them.
        let a = lcg(500, 11, 9.0);
        let b = lcg(500, 12, 9.0);
        let ra = centred_ranks_doubled(&a).unwrap();
        let rb = centred_ranks_doubled(&b).unwrap();
        let forward = RankSums::between(&ra, &rb).unwrap();
        let mut rev_a = ra.clone();
        let mut rev_b = rb.clone();
        rev_a.reverse();
        rev_b.reverse();
        let reverse = RankSums::between(&rev_a, &rev_b).unwrap();
        assert_eq!(forward, reverse);
        // Covariance under a strictly increasing map (rho is invariant).
        let scaled: Vec<f64> = a.iter().map(|v| v * 3.0 + 1.0).collect();
        assert_eq!(
            RankSums::between(&centred_ranks_doubled(&scaled).unwrap(), &rb).unwrap(),
            forward
        );
    }
}
