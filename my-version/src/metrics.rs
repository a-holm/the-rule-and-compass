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
    /// Mean Brier score `(σ(p) − l)²` (classification).
    Brier,
    /// Mean **bounded sigmoid** hinge surrogate `2σ(−t·p)` (classification).
    ///
    /// This is the historical `hinge` metric, not the SVM hinge: it is smooth,
    /// bounded in `(0, 2)` and never exactly zero, so it has no finite optimum
    /// on separable data. [`Metric::HingeOrdinary`] and
    /// [`Metric::HingeSquared`] are the separate SVM forms.
    Hinge,
    /// Mean ordinary (SVM) hinge `max(0, 1 − t·p)` (classification).
    HingeOrdinary,
    /// Mean squared hinge `max(0, 1 − t·p)²` (classification).
    HingeSquared,
    /// Mean absolute error of the first difference over adjacent-row pairs
    /// (same group, consecutive in the original data order): how well slopes
    /// match. See [`SLOPE_MIN_PAIRS`] and [`score_slope_pairs`].
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
            Brier,
            Hinge,
            HingeOrdinary,
            HingeSquared,
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
            "brier" | "brier_score" => Brier,
            "hinge" | "hinge_sigmoid" | "hinge_logistic" => Hinge,
            "hinge_ordinary" | "hinge_linear" | "hinge_svm" => HingeOrdinary,
            "hinge_squared" | "squared_hinge" => HingeSquared,
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
            Brier => "brier",
            Hinge => "hinge",
            HingeOrdinary => "hinge_ordinary",
            HingeSquared => "hinge_squared",
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
        matches!(
            self,
            Metric::Auc
                | Metric::LogLoss
                | Metric::Brier
                | Metric::Hinge
                | Metric::HingeOrdinary
                | Metric::HingeSquared
        )
    }

    /// True when `best_error` means the run is solved and the search may stop.
    ///
    /// Error metrics are non-negative, so solved means the error reached the
    /// floor [`crate::progress_stats::SOLVED_ERROR`]. The information criteria
    /// go negative on a good fit and the search sites hold only the criterion,
    /// not its residual sum of squares, so they are never solved.
    pub fn is_solved(self, best_error: f64) -> bool {
        !self.is_parameter_aware()
            && best_error >= 0.0
            && best_error <= crate::progress_stats::SOLVED_ERROR
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

/// #287: the device/native scoring statistic for a metric.
///
/// This is deliberately **not** [`crate::fit::DeviceLoss`]: fitting and scoring
/// are separate decisions. `log_loss` has a device *fit* objective and, since
/// #289, a native *score*; a metric may have a native score while its constants
/// still fit by a warm-start objective (Brier and the hinge forms). The
/// discriminants are the `kind` argument of `rc_score_batch`/`rc_pool_score` in
/// `src/cuda/eval.cu` (`RC_SCORE_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreObjective {
    /// Mean `r²`.
    Squared,
    /// Mean `|r|`.
    Absolute,
    /// Mean `ln(1 + r²)`.
    LogSquashed,
    /// `|mean r|`.
    Signed,
    /// `max |r|`, with the lowest-row tie broken on the `|r|` bit pattern.
    Maximum,
    /// `1 − |ρ|`, two-pass centred moments.
    Correlation,
    /// `1 − ρ`, two-pass centred moments.
    CorrelationSigned,
    /// `n·ln(RSS/n) + 2k`.
    Aic,
    /// Small-sample-corrected AIC.
    Aicc,
    /// `n·ln(RSS/n) + k·ln n`.
    Bic,
    /// #289: mean stable softplus `max(−u, 0) + log1p(exp(−|u|))`, `u = t·p`.
    LogLoss,
    /// #289: mean `(σ(p) − l)²`.
    Brier,
    /// #289: the historical bounded sigmoid hinge surrogate `2σ(−u)`.
    HingeSigmoid,
    /// #289: `max(0, 1 − u)`.
    HingeOrdinary,
    /// #289: `max(0, 1 − u)²`.
    HingeSquared,
    /// #236: mean `|(pᵢ − pᵢ₋₁) − (yᵢ − yᵢ₋₁)|` over the consecutive rows of
    /// the resident sample, divided by `n − 1`. Admitted only when the
    /// sample's slope pairs are exactly the consecutive pairs.
    Slope,
}

impl ScoreObjective {
    /// The `RC_SCORE_*` code the CUDA scoring entry points take.
    ///
    /// The regression block is 0–9 (#287); classification is 10–14 (#289);
    /// slope is 15 (#236). A fit code (`DeviceLoss`) and a score code can never
    /// be confused, and every code in this enum has its own kernel branch.
    pub fn code(self) -> i32 {
        match self {
            ScoreObjective::Squared => 0,
            ScoreObjective::Absolute => 1,
            ScoreObjective::LogSquashed => 2,
            ScoreObjective::Signed => 3,
            ScoreObjective::Maximum => 4,
            ScoreObjective::Correlation => 5,
            ScoreObjective::CorrelationSigned => 6,
            ScoreObjective::Aic => 7,
            ScoreObjective::Aicc => 8,
            ScoreObjective::Bic => 9,
            ScoreObjective::LogLoss => 10,
            ScoreObjective::Brier => 11,
            ScoreObjective::HingeSigmoid => 12,
            ScoreObjective::HingeOrdinary => 13,
            ScoreObjective::HingeSquared => 14,
            ScoreObjective::Slope => 15,
        }
    }

    /// Inverse of [`Self::code`], used when a status report is read back.
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => ScoreObjective::Squared,
            1 => ScoreObjective::Absolute,
            2 => ScoreObjective::LogSquashed,
            3 => ScoreObjective::Signed,
            4 => ScoreObjective::Maximum,
            5 => ScoreObjective::Correlation,
            6 => ScoreObjective::CorrelationSigned,
            7 => ScoreObjective::Aic,
            8 => ScoreObjective::Aicc,
            9 => ScoreObjective::Bic,
            10 => ScoreObjective::LogLoss,
            11 => ScoreObjective::Brier,
            12 => ScoreObjective::HingeSigmoid,
            13 => ScoreObjective::HingeOrdinary,
            14 => ScoreObjective::HingeSquared,
            15 => ScoreObjective::Slope,
            _ => return None,
        })
    }

    /// A stable name for reports and schema output.
    pub fn name(self) -> &'static str {
        match self {
            ScoreObjective::Squared => "squared",
            ScoreObjective::Absolute => "absolute",
            ScoreObjective::LogSquashed => "log_squashed",
            ScoreObjective::Signed => "signed",
            ScoreObjective::Maximum => "maximum",
            ScoreObjective::Correlation => "correlation",
            ScoreObjective::CorrelationSigned => "correlation_signed",
            ScoreObjective::Aic => "aic",
            ScoreObjective::Aicc => "aicc",
            ScoreObjective::Bic => "bic",
            ScoreObjective::LogLoss => "log_loss",
            ScoreObjective::Brier => "brier",
            ScoreObjective::HingeSigmoid => "hinge",
            ScoreObjective::HingeOrdinary => "hinge_ordinary",
            ScoreObjective::HingeSquared => "hinge_squared",
            ScoreObjective::Slope => "slope",
        }
    }

    /// The metric this objective scores.
    pub fn metric(self) -> Metric {
        match self {
            ScoreObjective::Squared => Metric::Squared,
            ScoreObjective::Absolute => Metric::Absolute,
            ScoreObjective::LogSquashed => Metric::LogSquashed,
            ScoreObjective::Signed => Metric::Signed,
            ScoreObjective::Maximum => Metric::Maximum,
            ScoreObjective::Correlation => Metric::Correlation,
            ScoreObjective::CorrelationSigned => Metric::CorrelationSigned,
            ScoreObjective::Aic => Metric::Aic,
            ScoreObjective::Aicc => Metric::Aicc,
            ScoreObjective::Bic => Metric::Bic,
            ScoreObjective::LogLoss => Metric::LogLoss,
            ScoreObjective::Brier => Metric::Brier,
            ScoreObjective::HingeSigmoid => Metric::Hinge,
            ScoreObjective::HingeOrdinary => Metric::HingeOrdinary,
            ScoreObjective::HingeSquared => Metric::HingeSquared,
            ScoreObjective::Slope => Metric::Slope,
        }
    }

    /// The native scoring objective for a classification metric, or `None`.
    ///
    /// `auc` deliberately returns `None`: it is a rank statistic with its own
    /// segmented-sort stage (#288), not a per-row loss.
    pub fn for_metric(metric: Metric) -> Option<Self> {
        match metric {
            Metric::LogLoss => Some(ScoreObjective::LogLoss),
            Metric::Brier => Some(ScoreObjective::Brier),
            Metric::Hinge => Some(ScoreObjective::HingeSigmoid),
            Metric::HingeOrdinary => Some(ScoreObjective::HingeOrdinary),
            Metric::HingeSquared => Some(ScoreObjective::HingeSquared),
            _ => None,
        }
    }

    /// True when the objective is parameter-aware and needs `k`.
    pub fn needs_constants(self) -> bool {
        matches!(
            self,
            ScoreObjective::Aic | ScoreObjective::Aicc | ScoreObjective::Bic
        )
    }

    /// True when the scorer accumulates a two-pass centred moment.
    pub fn is_correlation(self) -> bool {
        matches!(
            self,
            ScoreObjective::Correlation | ScoreObjective::CorrelationSigned
        )
    }

    /// True for the label-aware classification row losses (#289).
    pub fn is_classification(self) -> bool {
        matches!(
            self,
            ScoreObjective::LogLoss
                | ScoreObjective::Brier
                | ScoreObjective::HingeSigmoid
                | ScoreObjective::HingeOrdinary
                | ScoreObjective::HingeSquared
        )
    }
}

/// #287: one row of the score dispatch table.
///
/// Every [`Metric`] maps here exactly once. `Native` means a device kernel
/// computes the configured statistic; `Refused` carries a named, user-facing
/// reason instead of silently substituting squared error (the #236 defect).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDispatch {
    /// A native one-pass/two-pass statistic.
    Native(ScoreObjective),
    /// #288: an exact `f64` order-statistic or rank statistic scored through
    /// the plan batch entry point (flattened plans, sorted residual bits).
    Exact(ScoreStat),
    /// No native scorer in this issue; the name says why.
    Refused(&'static str),
}

impl ScoreDispatch {
    /// The native objective, or `None` when the metric is refused.
    pub fn objective(self) -> Option<ScoreObjective> {
        match self {
            ScoreDispatch::Native(o) => Some(o),
            ScoreDispatch::Exact(_) | ScoreDispatch::Refused(_) => None,
        }
    }

    /// #288: the exact order/rank statistic, or `None`.
    pub fn stat(self) -> Option<ScoreStat> {
        match self {
            ScoreDispatch::Exact(s) => Some(s),
            ScoreDispatch::Native(_) | ScoreDispatch::Refused(_) => None,
        }
    }

    /// The refusal reason, or `None` for a native metric.
    pub fn refusal(self) -> Option<&'static str> {
        match self {
            ScoreDispatch::Native(_) | ScoreDispatch::Exact(_) => None,
            ScoreDispatch::Refused(reason) => Some(reason),
        }
    }
}

/// #288: an exact order-statistic or rank statistic the device can score
/// natively.
///
/// This is the **scoring** counterpart of [`crate::fit::DeviceLoss`] and
/// [`crate::fit::OrderStat`]: it names the statistic a batch of *arbitrary*
/// candidates is ranked on, not the objective a fit arm moves constants
/// against. Every member is computed from the exact `f64` forward pass over the
/// resident rows, so a value depends only on the candidate's prediction vector
/// and the target, never on the batch it was scored in, the wave split, or the
/// number of islands.
///
/// The statistics split into two families:
///
/// * **order** — `median`, `interquartile` and `maximum` of `|pred − y|`. The
///   first two are the segmented sort/pick already used by the fit arms; the
///   maximum is the order-free integer max scan. All three are bit-identical to
///   the CPU statistic for the same prediction vector.
/// * **rank** — `rank_correlation(_signed)` and `auc`. Both are exact integer
///   sums over average ranks (`A_i = 2r_i − (n+1)`); the only floating-point
///   operations are the final divide (and square root for rho), so ties share
///   one integer and `−0.0`/`+0.0` are one tie exactly like the CPU's `==`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreStat {
    /// Median of `|pred − y|`; even `n` averages the two middle values.
    Median,
    /// Mean of `|pred − y|` over the sorted ranks `[⌊n/4⌋, ⌊3n/4⌋)`.
    Interquartile,
    /// Exact `max |pred − y|` (lowest-row tie, order-free).
    Maximum,
    /// `1 − |ρ_s|` from exact integer rank sums.
    RankCorrelation,
    /// `1 − ρ_s` from exact integer rank sums.
    RankCorrelationSigned,
    /// `1 − AUC`, the rank (Mann–Whitney U) form of the area under the ROC.
    Auc,
}

impl ScoreStat {
    /// The exact statistic the [`score_dispatch`] table names for a metric, or
    /// `None` when the table scores it with the objective kernel or refuses it.
    pub fn for_metric(metric: Metric) -> Option<Self> {
        score_dispatch(metric).stat()
    }

    /// The configured CPU metric this statistic is the definition of.
    pub fn metric(self) -> Metric {
        match self {
            ScoreStat::Median => Metric::Median,
            ScoreStat::Interquartile => Metric::Interquartile,
            ScoreStat::Maximum => Metric::Maximum,
            ScoreStat::RankCorrelation => Metric::RankCorrelation,
            ScoreStat::RankCorrelationSigned => Metric::RankCorrelationSigned,
            ScoreStat::Auc => Metric::Auc,
        }
    }

    /// Stable name for the ledger, the schema text and the fallback reason.
    pub fn name(self) -> &'static str {
        self.metric().name()
    }

    /// The measured minimum training-row count below which the exact CPU
    /// statistic is cheaper than the device launch and sort.
    ///
    /// The sort-based statistics reuse the fit arms' measured floor
    /// ([`crate::fit::MEDIAN_GPU_MIN_ROWS`] = 4096 rows); the order-free
    /// maximum uses its own 512-row floor
    /// ([`crate::fit::MAXIMUM_GPU_MIN_ROWS`]). The rank family is sort-based
    /// too, so it shares 4096. Below the floor the run keeps the CPU scorer and
    /// records this reason; the device may still fit constants through the
    /// fit-only arm.
    pub fn min_rows(self) -> usize {
        match self {
            ScoreStat::Maximum => crate::fit::MAXIMUM_GPU_MIN_ROWS,
            ScoreStat::Median
            | ScoreStat::Interquartile
            | ScoreStat::RankCorrelation
            | ScoreStat::RankCorrelationSigned
            | ScoreStat::Auc => crate::fit::MEDIAN_GPU_MIN_ROWS,
        }
    }

    /// The `'static` refusal used when the sample is below [`Self::min_rows`].
    pub fn small_sample_reason(self) -> &'static str {
        match self {
            ScoreStat::Maximum => {
                "maximum scoring: the sample is below the 512-row device floor"
            }
            _ => "order/rank scoring: the sample is below the 4096-row device floor",
        }
    }

    /// True for the two Spearman statistics.
    pub fn is_rank(self) -> bool {
        matches!(
            self,
            ScoreStat::RankCorrelation | ScoreStat::RankCorrelationSigned
        )
    }
}

/// #287: the total `Metric -> ScoreDispatch` table.
///
/// The match is exhaustive, so adding a `Metric` without deciding how it is
/// scored is a compile error rather than a silent fallback. The table is the
/// only scoring gate: [`crate::search::gpu_blocker`], the CUDA entry points and
/// this module all read it.
pub fn score_dispatch(m: Metric) -> ScoreDispatch {
    use Metric::*;
    use ScoreObjective as O;
    match m {
        Squared => ScoreDispatch::Native(O::Squared),
        Absolute => ScoreDispatch::Native(O::Absolute),
        LogSquashed => ScoreDispatch::Native(O::LogSquashed),
        Signed => ScoreDispatch::Native(O::Signed),
        Maximum => ScoreDispatch::Native(O::Maximum),
        Correlation => ScoreDispatch::Native(O::Correlation),
        CorrelationSigned => ScoreDispatch::Native(O::CorrelationSigned),
        Aic => ScoreDispatch::Native(O::Aic),
        Aicc => ScoreDispatch::Native(O::Aicc),
        Bic => ScoreDispatch::Native(O::Bic),
        // #289: the label-aware classification family.
        LogLoss => ScoreDispatch::Native(O::LogLoss),
        Brier => ScoreDispatch::Native(O::Brier),
        Hinge => ScoreDispatch::Native(O::HingeSigmoid),
        HingeOrdinary => ScoreDispatch::Native(O::HingeOrdinary),
        HingeSquared => ScoreDispatch::Native(O::HingeSquared),
        Median => ScoreDispatch::Exact(ScoreStat::Median),
        Interquartile => ScoreDispatch::Exact(ScoreStat::Interquartile),
        RankCorrelation => ScoreDispatch::Exact(ScoreStat::RankCorrelation),
        RankCorrelationSigned => ScoreDispatch::Exact(ScoreStat::RankCorrelationSigned),
        // #236: consecutive-row pairs; `search::gpu_blocker` refuses a sample
        // whose pair list is anything else.
        Slope => ScoreDispatch::Native(O::Slope),
        ImplicitDerivative => ScoreDispatch::Refused(
            "implicit needs an expression-tree gradient pass",
        ),
        // The target of an implicit sample is 0, so `sum w|f| / sum w` is the
        // absolute metric bit for bit.
        ImplicitResidual => ScoreDispatch::Native(O::Absolute),
        Auc => ScoreDispatch::Exact(ScoreStat::Auc),
    }
}

/// #287: the native scoring objective for a metric, or `None` when refused.
pub fn native_score_objective(m: Metric) -> Option<ScoreObjective> {
    score_dispatch(m).objective()
}

/// #287: the named reason a metric has no native scorer, or `None`.
pub fn score_refusal(m: Metric) -> Option<&'static str> {
    score_dispatch(m).refusal()
}

/// #287: a per-candidate scoring status, mirrored by the CUDA ABI.
///
/// `Ok` is `0`; `NonFinite` is `2` (the existing order-statistic/maximum ABI);
/// the CPU-only labels use higher values. A non-`Ok` status always carries
/// `value = +inf`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreStatus {
    /// A finite (or explicitly `-inf`, for a perfect AIC fit) score.
    Ok = 0,
    /// A prediction was not finite (or exceeded the device's hard magnitude).
    NonFinite = 2,
    /// Lengths disagreed or an input weight was negative/non-finite.
    Invalid = 3,
    /// Every row was masked by a zero weight.
    AllZeroWeight = 4,
    /// The device launch failed; the caller must fall back to the CPU.
    DeviceError = 5,
}

impl ScoreStatus {
    /// The wire code for the CUDA status array.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// Decode a device status; unknown values are a device failure.
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => ScoreStatus::Ok,
            2 => ScoreStatus::NonFinite,
            3 => ScoreStatus::Invalid,
            4 => ScoreStatus::AllZeroWeight,
            _ => ScoreStatus::DeviceError,
        }
    }

    /// True when `value` is usable.
    pub fn is_ok(self) -> bool {
        self == ScoreStatus::Ok
    }
}

/// #287: the CPU reference result of a native reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeScore {
    /// The metric value; `+inf` for every non-`Ok` status.
    pub value: f64,
    /// Why the value is unusable, if it is.
    pub status: ScoreStatus,
    /// Lowest maximiser row for [`ScoreObjective::Maximum`].
    pub maximum_row: Option<usize>,
}

impl NativeScore {
    fn ok(value: f64) -> Self {
        NativeScore {
            value,
            status: ScoreStatus::Ok,
            maximum_row: None,
        }
    }

    fn invalid(status: ScoreStatus) -> Self {
        NativeScore {
            value: f64::INFINITY,
            status,
            maximum_row: None,
        }
    }
}

/// #287: CPU reference for a native score objective, without weights.
///
/// This is the exact `f64` reducer the CUDA kernel mirrors. It exists so the
/// statistic can be tested independently of the general metric dispatcher and
/// so a device readback can be compared against the same formula given the same
/// (f32-rounded) prediction vector.
pub fn score_native(pred: &[f64], y: &[f64], objective: ScoreObjective, k: usize) -> NativeScore {
    score_native_weighted(pred, y, &[], objective, k)
}

/// #287: CPU reference for a native score objective with optional weights.
///
/// Zero-weight rows are masked **before** any prediction is read, exactly like
/// [`score_weighted`]. A non-finite prediction in a masked row is therefore
/// invisible. `maximum` and `slope` have no weighted form and refuse weights.
///
/// #289: the classification objectives read `pred` as a **logit**, derive the
/// label `l = [y > weighted midpoint]` and accumulate the configured row loss.
/// The midpoint is taken over positive-weight rows, matching
/// [`score_weighted`]'s weighted classification branch.
pub fn score_native_weighted(
    pred: &[f64],
    y: &[f64],
    w: &[f64],
    objective: ScoreObjective,
    k: usize,
) -> NativeScore {
    let n = pred.len();
    if n == 0 || y.len() != n || (!w.is_empty() && w.len() != n) {
        return NativeScore::invalid(ScoreStatus::Invalid);
    }
    let weighted = !w.is_empty();
    if weighted {
        if objective == ScoreObjective::Maximum || objective == ScoreObjective::Slope {
            return NativeScore::invalid(ScoreStatus::Invalid);
        }
        let mut total = 0.0;
        for &wi in w {
            if !wi.is_finite() || wi < 0.0 {
                return NativeScore::invalid(ScoreStatus::Invalid);
            }
            total += wi;
        }
        if !(total > 0.0) {
            return NativeScore::invalid(ScoreStatus::AllZeroWeight);
        }
    }
    for (i, &p) in pred.iter().enumerate() {
        if (!weighted || w[i] > 0.0) && !p.is_finite() {
            return NativeScore::invalid(ScoreStatus::NonFinite);
        }
    }
    // Masked rows have a zero residual, so no `w * loss` product can become
    // `0 * inf = NaN`.
    let active = |i: usize| !weighted || w[i] > 0.0;
    let weight = |i: usize| if weighted { w[i] } else { 1.0 };
    let denom = if weighted {
        w.iter().sum::<f64>()
    } else {
        n as f64
    };

    // #289: the label-aware objectives are not functions of `pred - y`.
    if objective.is_classification() {
        let labels = if weighted {
            match binary_labels_weighted(y, w) {
                Some(labels) => labels,
                None => return NativeScore::invalid(ScoreStatus::NonFinite),
            }
        } else {
            binary_labels(y)
        };
        let mut sum = 0.0;
        for i in 0..n {
            if active(i) {
                sum += weight(i) * classification_row_loss(pred[i], labels[i], objective);
            }
        }
        return NativeScore::ok(sum / denom);
    }

    match objective {
        ScoreObjective::Slope => {
            if n < 2 {
                return NativeScore::invalid(ScoreStatus::NonFinite);
            }
            let mut sum = 0.0;
            for i in 1..n {
                sum += ((pred[i] - pred[i - 1]) - (y[i] - y[i - 1])).abs();
            }
            NativeScore::ok(sum / (n - 1) as f64)
        }
        ScoreObjective::Maximum => {
            let mut best_bits = 0u64;
            let mut best_row = 0usize;
            for i in 0..n {
                let key = (pred[i] - y[i]).abs().to_bits();
                if key > best_bits {
                    best_bits = key;
                    best_row = i;
                }
            }
            NativeScore {
                value: f64::from_bits(best_bits),
                status: ScoreStatus::Ok,
                maximum_row: Some(best_row),
            }
        }
        ScoreObjective::Correlation | ScoreObjective::CorrelationSigned => {
            let rho = if weighted {
                weighted_correlation(pred, y, w)
            } else {
                correlation(pred, y)
            };
            NativeScore::ok(if objective == ScoreObjective::Correlation {
                1.0 - rho.abs()
            } else {
                1.0 - rho
            })
        }
        ScoreObjective::Signed => {
            let mut sum = 0.0;
            for i in 0..n {
                if active(i) {
                    sum += weight(i) * (pred[i] - y[i]);
                }
            }
            NativeScore::ok((sum / denom).abs())
        }
        ScoreObjective::Aic | ScoreObjective::Aicc | ScoreObjective::Bic => {
            let mut rss = 0.0;
            for i in 0..n {
                if active(i) {
                    let r = pred[i] - y[i];
                    rss += weight(i) * r * r;
                }
            }
            let rows = if weighted {
                w.iter().filter(|&&wi| wi > 0.0).count()
            } else {
                n
            };
            NativeScore::ok(match objective {
                ScoreObjective::Aic => aic(rows, rss, k),
                ScoreObjective::Aicc => aicc(rows, rss, k),
                _ => bic(rows, rss, k),
            })
        }
        objective => {
            let mut sum = 0.0;
            for i in 0..n {
                if active(i) {
                    let r = pred[i] - y[i];
                    let loss = match objective {
                        ScoreObjective::Squared => r * r,
                        ScoreObjective::Absolute => r.abs(),
                        ScoreObjective::LogSquashed => (1.0 + r * r).ln(),
                        _ => unreachable!("maximum, correlation and classification are handled above"),
                    };
                    sum += weight(i) * loss;
                }
            }
            NativeScore::ok(sum / denom)
        }
    }
}

/// Fewest adjacent-row pairs the training side may keep for `slope`; below it
/// the CLI refuses the metric. Matches the 32-row floors used elsewhere
/// (`FUSED_MIN_SEGMENTS`, the QD update floor, the 32-rows-per-constant rule).
pub const SLOPE_MIN_PAIRS: usize = 32;

/// Mean absolute error of the first difference over explicit row pairs
/// `[a, b]` (positions in `pred` and `y`). No pairs, or a non-finite
/// prediction, scores `+inf`.
pub fn score_slope_pairs(pred: &[f64], y: &[f64], pairs: &[[u32; 2]]) -> f64 {
    if pairs.is_empty() || pred.len() != y.len() || pred.iter().any(|p| !p.is_finite()) {
        return f64::INFINITY;
    }
    let mut acc = 0.0;
    for &[a, b] in pairs {
        let (a, b) = (a as usize, b as usize);
        acc += ((pred[b] - pred[a]) - (y[b] - y[a])).abs();
    }
    acc / pairs.len() as f64
}

/// Score a prediction on a sample: [`score_weighted`], except that `slope`
/// differences the sample's adjacent-row pairs when it carries them.
pub fn score_sample(pred: &[f64], s: &crate::data::Sample, m: Metric, k: usize) -> f64 {
    match (m, &s.pairs) {
        (Metric::Slope, Some(pairs)) => score_slope_pairs(pred, &s.y, pairs),
        _ => score_weighted(pred, &s.y, &s.w, m, k),
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
    let mut resid: Vec<f64> = pred.iter().zip(y).map(|(p, t)| p - t).collect();
    match m {
        Metric::Absolute | Metric::ImplicitResidual => mean_abs(&resid),
        Metric::Squared => mean_sq(&resid),
        Metric::Correlation => 1.0 - correlation(pred, y).abs(),
        Metric::CorrelationSigned => 1.0 - correlation(pred, y),
        Metric::Median => median_abs_in_place(&mut resid),
        Metric::Maximum => resid.iter().fold(0.0f64, |a, e| a.max(e.abs())),
        Metric::Signed => mean(&resid).abs(),
        Metric::LogSquashed => resid.iter().map(|e| (1.0 + e * e).ln()).sum::<f64>() / n as f64,
        Metric::RankCorrelation => 1.0 - rank_correlation(pred, y).abs(),
        Metric::RankCorrelationSigned => 1.0 - rank_correlation(pred, y),
        Metric::Interquartile => interquartile_mean_abs_in_place(&mut resid),
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
        Metric::Brier => brier(pred, y),
        Metric::Hinge => hinge(pred, y),
        Metric::HingeOrdinary => hinge_ordinary(pred, y),
        Metric::HingeSquared => hinge_squared(pred, y),
        Metric::Aic => aic(n, rss(&resid), k),
        Metric::Aicc => aicc(n, rss(&resid), k),
        Metric::Bic => bic(n, rss(&resid), k),
        // The level-set distance needs the expression tree (for ∇f), not a
        // prediction vector. [`crate::search::eval_error`] routes it to
        // `crate::implicit::score`; reaching here means the caller has no tree.
        // `implicit_residual` is `absolute` on an implicit sample (target 0).
        Metric::ImplicitDerivative => f64::INFINITY,
    }
}

/// True when `m` has a defined weighted form. The CLI refuses `--weight` for
/// every other metric rather than silently ignoring the weights.
///
/// Supported: `absolute`, `squared`, `correlation`, `correlation_signed`,
/// `signed`, `log_squashed`, `log_loss`, `brier`, the three hinge forms,
/// `implicit`, `implicit_residual`, `aic`, `aicc` and `bic`. The order-statistic,
/// rank and first-difference metrics have no standard weighted definition (see
/// `docs/weights.md`); `auc` is a rank statistic and needs its own class-count
/// definition, so it stays unweighted here.
pub fn weights_supported(m: Metric) -> bool {
    matches!(
        m,
        Metric::Absolute
            | Metric::Squared
            | Metric::Correlation
            | Metric::CorrelationSigned
            | Metric::Signed
            | Metric::LogSquashed
            | Metric::LogLoss
            | Metric::Brier
            | Metric::Hinge
            | Metric::HingeOrdinary
            | Metric::HingeSquared
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
        // `Metric::all` presents the implicit family before the
        // classification family; keep this list in the same order so the
        // "supported" test and the CLI text cannot drift.
        "implicit",
        "implicit_residual",
        "log_loss",
        "brier",
        "hinge",
        "hinge_ordinary",
        "hinge_squared",
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
    // #289: the weighted classification family. A zero-weight row is skipped
    // **before** its prediction or label is read, so a masked `NaN`/infinite
    // branch cannot poison the loss, and the midpoint is taken over the
    // positive-weight rows only. `auc` has no per-row weighted form here.
    if m.is_classification() {
        let Some(objective) = ScoreObjective::for_metric(m) else {
            return f64::INFINITY;
        };
        let Some(labels) = binary_labels_weighted(y, w) else {
            return f64::INFINITY;
        };
        let mut acc = 0.0;
        for ((&p, &l), &wi) in pred.iter().zip(&labels).zip(w) {
            if wi <= 0.0 {
                continue;
            }
            acc += wi * classification_row_loss(p, l, objective);
        }
        return acc / total;
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
        Metric::Absolute | Metric::ImplicitResidual => resid
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
    let (mut ha, mut hb) = (0.0f64, 0.0f64);
    for i in 0..a.len() {
        if w[i] > 0.0 {
            ha = ha.max((a[i] - ma).abs());
            hb = hb.max((b[i] - mb).abs());
        }
    }
    if !(ha >= f64::MIN_POSITIVE && hb >= f64::MIN_POSITIVE && ha.is_finite() && hb.is_finite()) {
        return 0.0;
    }
    let (ha, hb) = (pow2_floor(ha), pow2_floor(hb));
    let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let wi = w[i];
        if wi > 0.0 {
            let da = (a[i] - ma) / ha;
            let db = (b[i] - mb) / hb;
            sab += wi * da * db;
            sa += wi * da * da;
            sb += wi * db * db;
        }
    }
    clamp_correlation(sab, sa, sb)
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

/// Median of the absolute errors. `+inf` is impossible here (inputs are finite
/// and non-empty), so the empty case returns 0 by convention.
/// The buffer is owned by the caller: its values are replaced by their
/// absolute values and reordered.
fn median_abs_in_place(s: &mut [f64]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    for e in s.iter_mut() {
        *e = e.abs();
    }
    let n = s.len();
    let (left, upper, _) = s.select_nth_unstable_by(n / 2, f64::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        upper
    } else {
        let lower = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

/// Mean of the absolute errors that fall between the 25th and 75th percentile —
/// the "interquartile mean absolute error". Robust to extreme residuals without
/// throwing most of the data away, as the median does.
/// The buffer is owned by the caller: its values are replaced by their absolute
/// values and reordered.
fn interquartile_mean_abs_in_place(s: &mut [f64]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    for e in s.iter_mut() {
        *e = e.abs();
    }
    let n = s.len();
    if n < 4 {
        s.sort_unstable_by(f64::total_cmp);
        return mean(s);
    }
    let lo = n / 4;
    let end = ((3 * n) / 4).max(lo + 1);
    let (_, _, right) = s.select_nth_unstable_by(lo, f64::total_cmp);
    right.select_nth_unstable_by(end - lo - 1, f64::total_cmp);
    let slice = &mut s[lo..end];
    slice.sort_unstable_by(f64::total_cmp);
    mean(slice)
}

/// Average ranks (1-based), with ties sharing their mean rank.
fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut order: Vec<(f64, usize)> = v.iter().copied().zip(0..n).collect();
    order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && order[j].0 == order[i].0 {
            j += 1;
        }
        // ranks i+1 .. j share the average
        let avg = 0.5 * ((i + 1) as f64 + j as f64);
        for &(_, k) in &order[i..j] {
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
    let mut order: Vec<(f64, usize)> = v.iter().copied().zip(0..n).collect();
    order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut a = vec![0i64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && order[j].0 == order[i].0 {
            j += 1;
        }
        // Run `[i, j)` holds sorted positions `i+1 ..= j`, whose average rank is
        // `(i + 1 + j)/2`; doubled and centred that is exactly `i + j - n`.
        let centred = (i + j) as i64 - n as i64;
        for &(_, k) in &order[i..j] {
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
    let (mut ha, mut hb) = (0.0f64, 0.0f64);
    for i in 0..a.len() {
        ha = ha.max((a[i] - ma).abs());
        hb = hb.max((b[i] - mb).abs());
    }
    if !(ha >= f64::MIN_POSITIVE && hb >= f64::MIN_POSITIVE && ha.is_finite() && hb.is_finite()) {
        return 0.0;
    }
    let (ha, hb) = (pow2_floor(ha), pow2_floor(hb));
    let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let da = (a[i] - ma) / ha;
        let db = (b[i] - mb) / hb;
        sab += da * db;
        sa += da * da;
        sb += db * db;
    }
    clamp_correlation(sab, sa, sb)
}

/// The power of two at or below a normal positive `h`. Dividing by it is
/// exact, so scaling the deviations changes no digit of a well-scaled input.
pub(crate) fn pow2_floor(h: f64) -> f64 {
    f64::from_bits(h.to_bits() & 0x7ff0_0000_0000_0000)
}

/// `sab / sqrt(sa * sb)` clamped to `[-1, 1]`. Zero variance or a non-finite
/// result is `0.0`, the metric's worst value.
fn clamp_correlation(sab: f64, sa: f64, sb: f64) -> f64 {
    if !(sa > 0.0 && sb > 0.0) {
        return 0.0;
    }
    let r = sab / (sa.sqrt() * sb.sqrt());
    if r.is_finite() {
        r.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Split the target into two classes at its midpoint. The engine normalises the
/// target, so an absolute 0.5 threshold would be wrong; the midpoint of the two
/// observed extremes is invariant under the affine normalisation.
pub fn binary_labels(y: &[f64]) -> Vec<f64> {
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

/// #289: the midpoint labels of the **positive-weight** rows.
///
/// A zero-weight row is invisible, exactly as it is to the fitter and to every
/// other weighted metric: changing its target (even to `±inf` or `NaN`) cannot
/// move the midpoint. `None` means a positive-weight target is non-finite.
pub fn binary_labels_weighted(y: &[f64], w: &[f64]) -> Option<Vec<f64>> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&v, &wi) in y.iter().zip(w) {
        if wi > 0.0 {
            if !v.is_finite() {
                return None;
            }
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let mid = 0.5 * (lo + hi);
    Some(
        y.iter()
            .zip(w)
            .map(|(&v, &wi)| if wi > 0.0 && v > mid { 1.0 } else { 0.0 })
            .collect(),
    )
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

/// #289: the Lin–Lin–Weng softplus `ln(1 + exp(−u))`, the exact stable form of
/// the logistic loss. It never overflows and is exact for small `|u|`, where
/// the naive `ln(1 + exp(−u))` cancels.
fn softplus_neg(u: f64) -> f64 {
    let a = u.abs();
    (-u).max(0.0) + (-a).exp().ln_1p()
}

/// #289: one row of a classification metric, with `p` the logit and `l ∈ {0,1}`.
pub fn classification_row_loss(p: f64, l: f64, objective: ScoreObjective) -> f64 {
    let t = 2.0 * l - 1.0;
    let u = t * p;
    match objective {
        ScoreObjective::LogLoss => softplus_neg(u),
        ScoreObjective::Brier => {
            let d = sigmoid(p) - l;
            d * d
        }
        ScoreObjective::HingeSigmoid => {
            // The historical `1 − t·(2σ(p) − 1)`, kept in this exact form so the
            // pre-#289 `hinge` numbers do not move.
            let s = 2.0 * sigmoid(p) - 1.0;
            (1.0 - t * s).max(0.0)
        }
        ScoreObjective::HingeOrdinary => (1.0 - u).max(0.0),
        ScoreObjective::HingeSquared => {
            let h = (1.0 - u).max(0.0);
            h * h
        }
        // Only the five classification objectives reach this function; a
        // regression objective here is a dispatch bug, not a number.
        _ => f64::NAN,
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
///
/// #289: evaluated as the stable softplus `max(−u, 0) + ln(1 + exp(−|u|))` with
/// `u = t·p`. It is exactly the binary cross-entropy for every finite logit and
/// needs no arbitrary probability clamp, so the CPU reference and the device
/// scorer agree at `|p| = 10⁶` as well as near `p = 0`.
pub fn log_loss(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        acc += classification_row_loss(*p, *l, ScoreObjective::LogLoss);
    }
    acc / pred.len() as f64
}

/// Mean Brier score `(σ(p) − l)²`, treating the prediction as a logit.
///
/// This is a **proper** scoring rule on the probability `σ(p)`, not a squared
/// error on the raw logit. It is bounded in `[0, 1]` and its constant-model
/// optimum is `p* = ln(n₊/n₋)` for log-loss but the probability `σ(p*) = n₊/n`
/// for Brier.
pub fn brier(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        acc += classification_row_loss(*p, *l, ScoreObjective::Brier);
    }
    acc / pred.len() as f64
}

/// Mean **bounded sigmoid** hinge surrogate, `2σ(−t·p)` on the signed margin.
///
/// This is the historical `hinge` metric. It is *not* the SVM hinge: it is
/// smooth and bounded in `(0, 2)`, never exactly zero, and it applies the
/// logistic link twice when the model output is already a probability. Use
/// [`hinge_ordinary`] or [`hinge_squared`] for the linear-margin forms.
pub fn hinge(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        acc += classification_row_loss(*p, *l, ScoreObjective::HingeSigmoid);
    }
    acc / pred.len() as f64
}

/// Mean ordinary (SVM) hinge `max(0, 1 − t·p)` on the signed margin.
pub fn hinge_ordinary(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        acc += classification_row_loss(*p, *l, ScoreObjective::HingeOrdinary);
    }
    acc / pred.len() as f64
}

/// Mean squared hinge `max(0, 1 − t·p)²` on the signed margin.
pub fn hinge_squared(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || pred.is_empty() {
        return f64::INFINITY;
    }
    let labels = binary_labels(y);
    let mut acc = 0.0;
    for (p, l) in pred.iter().zip(&labels) {
        acc += classification_row_loss(*p, *l, ScoreObjective::HingeSquared);
    }
    acc / pred.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_reference(v: &[f64]) -> Vec<f64> {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s
    }

    fn median_reference(v: &[f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let s = sorted_reference(&v.iter().map(|e| e.abs()).collect::<Vec<_>>());
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            0.5 * (s[n / 2 - 1] + s[n / 2])
        }
    }

    fn interquartile_reference(v: &[f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let s = sorted_reference(&v.iter().map(|e| e.abs()).collect::<Vec<_>>());
        let n = s.len();
        if n < 4 {
            return mean(&s);
        }
        let lo = n / 4;
        let hi = (3 * n) / 4;
        mean(&s[lo..hi.max(lo + 1)])
    }

    /// Residual vectors with ties, both zeros and mixed signs.
    fn selection_cases() -> Vec<Vec<f64>> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut cases = Vec::new();
        for n in [1usize, 2, 3, 4, 5, 1000, 1001] {
            let continuous: Vec<f64> = (0..n)
                .map(|_| (f64::from(next() % 20_001) - 10_000.0) / 137.0)
                .collect();
            let tied: Vec<f64> = (0..n)
                .map(|_| match next() % 6 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1.5,
                    3 => -1.5,
                    4 => 2.25,
                    _ => -3.0,
                })
                .collect();
            cases.push(continuous);
            cases.push(tied);
            cases.push(vec![0.5; n]);
        }
        cases
    }

    #[test]
    fn median_by_selection_matches_sort() {
        for v in selection_cases() {
            assert_eq!(
                median_abs_in_place(&mut v.clone()).to_bits(),
                median_reference(&v).to_bits(),
                "n = {}",
                v.len()
            );
        }
    }

    #[test]
    fn interquartile_by_selection_matches_sort() {
        for v in selection_cases() {
            assert_eq!(
                interquartile_mean_abs_in_place(&mut v.clone()).to_bits(),
                interquartile_reference(&v).to_bits(),
                "n = {}",
                v.len()
            );
        }
    }

    #[test]
    fn ranks_with_ties_and_signed_zeros_share_the_mean_rank() {
        let v = [3.0, 0.0, -0.0, 1.0, 3.0, 3.0, -2.0];
        assert_eq!(ranks(&v), vec![6.0, 2.5, 2.5, 4.0, 6.0, 6.0, 1.0]);
        assert_eq!(
            centred_ranks_doubled(&v),
            Some(vec![4, -3, -3, 0, 4, 4, -6])
        );
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// #402: solved means "error is zero" for the non-negative error metrics
    /// and never for the information criteria, whose value goes negative on a
    /// good fit.
    #[test]
    fn solved_is_defined_per_metric() {
        for m in Metric::all() {
            let name = m.name();
            if m.is_parameter_aware() {
                for v in [0.0, -1.0, -1e6, f64::NEG_INFINITY, 1e-30] {
                    assert!(!m.is_solved(v), "{name} must not be solved at {v}");
                }
            } else {
                assert!(m.is_solved(0.0), "{name} is solved at zero error");
                assert!(m.is_solved(1e-25), "{name} is solved below the floor");
                assert!(!m.is_solved(1e-6), "{name} is not solved at 1e-6");
                assert!(!m.is_solved(0.5), "{name} is not solved at 0.5");
            }
        }
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
            Metric::Brier,
            Metric::Hinge,
            Metric::HingeOrdinary,
            Metric::HingeSquared,
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

    fn tiny_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..200).map(|i| 2.0 * i as f64 / 199.0).collect();
        let b: Vec<f64> = xs.iter().map(|x| (x + 1.0).ln() + (x * x + 1.0).ln()).collect();
        let tiny: Vec<f64> = xs.iter().map(|x| 1e-162 * x.exp()).collect();
        let unit: Vec<f64> = xs.iter().map(|x| x.exp()).collect();
        (tiny, unit, b)
    }

    #[test]
    fn correlation_tiny_magnitude() {
        let (tiny, unit, b) = tiny_fixture();
        let r = correlation(&tiny, &b);
        assert!(r.abs() <= 1.0, "r={r}");
        assert!((r - correlation(&unit, &b)).abs() < 1e-9, "r={r}");
        let w = vec![1.0; tiny.len()];
        let rw = weighted_correlation(&tiny, &b, &w);
        assert!(rw.abs() <= 1.0, "rw={rw}");
        assert!((rw - correlation(&unit, &b)).abs() < 1e-9, "rw={rw}");
        assert!(score(&tiny, &b, Metric::Correlation) >= 0.0);
    }

    #[test]
    fn correlation_large_mean_small_spread() {
        let xs: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let big: Vec<f64> = xs.iter().map(|x| 1e8 + 1e-6 * x).collect();
        let want = correlation(&xs, &xs.iter().map(|x| x * x).collect::<Vec<_>>());
        let y: Vec<f64> = xs.iter().map(|x| x * x).collect();
        assert!((correlation(&big, &y) - want).abs() < 1e-6);
        let w: Vec<f64> = (0..200).map(|i| 1.0 + (i % 3) as f64).collect();
        let want_w = weighted_correlation(&xs, &y, &w);
        assert!((weighted_correlation(&big, &y, &w) - want_w).abs() < 1e-6);
    }

    #[test]
    fn bounded_metric_never_solves_negative() {
        for m in [Metric::Correlation, Metric::CorrelationSigned, Metric::Squared, Metric::RankCorrelation] {
            assert!(!m.is_solved(-0.5), "{m:?}");
            assert!(!m.is_solved(-1e-30), "{m:?}");
            assert!(m.is_solved(0.0), "{m:?}");
        }
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

    /// #289: the three hinge forms are different functions, and only the
    /// historical one is the bounded sigmoid surrogate.
    #[test]
    fn hinge_forms_are_distinct_and_match_their_definitions() {
        // y = [0, 1]; predictions are logits.
        let y = [0.0, 1.0];
        let p = [0.0, 0.0];
        // At p = 0: surrogate = 2σ(0) - 0? 1-t*0 = 1 for both rows -> 1.
        assert!(approx(hinge(&p, &y), 1.0, 1e-12));
        // Ordinary hinge at u = 0 is 1 for both rows too.
        assert!(approx(hinge_ordinary(&p, &y), 1.0, 1e-12));
        assert!(approx(hinge_squared(&p, &y), 1.0, 1e-12));
        // Correct margins: surrogate is never zero; ordinary/squared are.
        let good = [-30.0, 30.0];
        assert!(hinge(&good, &y) > 0.0, "the surrogate has no finite optimum");
        assert!(approx(hinge_ordinary(&good, &y), 0.0, 1e-12));
        assert!(approx(hinge_squared(&good, &y), 0.0, 1e-12));
        // A margin inside the ordinary hinge's active set: u = 0.5 gives 0.5
        // and 0.25 for the linear and squared forms.
        let half = [-0.5, 0.5];
        assert!(approx(hinge_ordinary(&half, &y), 0.5, 1e-12));
        assert!(approx(hinge_squared(&half, &y), 0.25, 1e-12));
        // The ordinary hinge has a kink; squared is C1. Both must be finite at
        // the kink u = 1 instead of relying on a compiler branch.
        let kink = [-1.0, 1.0];
        assert!(approx(hinge_ordinary(&kink, &y), 0.0, 1e-12));
        assert!(approx(hinge_squared(&kink, &y), 0.0, 1e-12));
        // Every classification metric is in the classification family.
        for m in [
            Metric::Auc,
            Metric::LogLoss,
            Metric::Brier,
            Metric::Hinge,
            Metric::HingeOrdinary,
            Metric::HingeSquared,
        ] {
            assert!(m.is_classification(), "{}", m.name());
        }
    }

    /// #289: log-loss is the stable softplus, so an extreme logit gives the
    /// analytic value instead of a clamped constant.
    #[test]
    fn log_loss_is_the_stable_softplus_at_extreme_logits() {
        let y = [0.0, 1.0];
        // Correct, confident rows: loss ~ exp(-1e6) = 0.
        assert_eq!(log_loss(&[-1e6, 1e6], &y), 0.0);
        // Wrong, confident rows: loss is the margin itself, not a 1e-12 clamp.
        let wrong = log_loss(&[1e6, -1e6], &y);
        assert!(approx(wrong, 1e6, 1e-6), "{wrong}");
        // At p = 0 the value is ln 2.
        assert!(approx(log_loss(&[0.0, 0.0], &y), 2.0f64.ln(), 1e-15));
    }

    /// #289 §6.1: the constant-model optima in probability/logit space.
    #[test]
    fn constant_model_optima_for_log_loss_and_brier() {
        // 3 positives, 1 negative.
        let y = [1.0, 1.0, 1.0, 0.0];
        let want_logit = (3.0f64 / 1.0).ln();
        let want_p = 3.0 / 4.0;
        let at = |b: f64| [b; 4];
        // Log-loss is minimised at b* = ln(n+/n-).
        let best = log_loss(&at(want_logit), &y);
        for d in [-0.25, -0.01, 0.01, 0.25] {
            assert!(
                log_loss(&at(want_logit + d), &y) >= best - 1e-15,
                "log-loss moved at {d}"
            );
        }
        // Brier's probability optimum is n+/n = sigma(b*), so the same logit
        // minimises it too; its raw-logit squared error does not.
        let best_brier = brier(&at(want_logit), &y);
        for d in [-0.25, -0.01, 0.01, 0.25] {
            assert!(
                brier(&at(want_logit + d), &y) >= best_brier - 1e-15,
                "brier moved at {d}"
            );
        }
        // Brier is bounded in [0, 1]; its value at the optimum is the
        // irreducible variance p(1-p).
        assert!(approx(best_brier, want_p * (1.0 - want_p), 1e-12));
        // A single class is still scoreable (it is a degenerate but valid
        // distribution); the *fit* is what refuses it.
        assert!(brier(&[0.0, 0.0], &[0.0, 0.0]).is_finite());
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
        for m in [
            Metric::LogLoss,
            Metric::Brier,
            Metric::Hinge,
            Metric::HingeOrdinary,
            Metric::HingeSquared,
        ] {
            assert!(weights_supported(m), "{} must accept weights", m.name());
        }
        assert_eq!(
            score_weighted(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 1.0], Metric::Median, 0),
            f64::INFINITY,
            "an unsupported metric must refuse weights, not ignore them"
        );
    }

    /// #289: every weighted classification metric is exactly the score of the
    /// table with the heavier row duplicated, and zero-weight rows vanish.
    #[test]
    fn weighted_classification_is_duplicate_row_parity() {
        let y = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let p = [-1.5, 0.3, 2.2, -0.4, 0.8, 0.1];
        let w = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0];
        let mut dup_p = Vec::new();
        let mut dup_y = Vec::new();
        for i in 0..p.len() {
            for _ in 0..w[i] as usize {
                dup_p.push(p[i]);
                dup_y.push(y[i]);
            }
        }
        for m in [
            Metric::LogLoss,
            Metric::Brier,
            Metric::Hinge,
            Metric::HingeOrdinary,
            Metric::HingeSquared,
        ] {
            let weighted = score_weighted(&p, &y, &w, m, 0);
            let duplicated = score(&dup_p, &dup_y, m);
            assert!(
                approx(weighted, duplicated, 1e-12),
                "{}: weighted {weighted} vs duplicated {duplicated}",
                m.name()
            );
        }
        // Scaling every weight is a no-op (Σw normalisation).
        let scaled: Vec<f64> = w.iter().map(|v| v * 7.5).collect();
        for m in [Metric::LogLoss, Metric::Brier, Metric::Hinge] {
            assert!(approx(
                score_weighted(&p, &y, &w, m, 0),
                score_weighted(&p, &y, &scaled, m, 0),
                1e-12
            ));
        }
    }

    /// #289: a zero-weight row is invisible even when its prediction or target
    /// is non-finite, and cannot move the class midpoint.
    #[test]
    fn zero_weight_classification_rows_are_invisible() {
        let y = [0.0, 1.0, 0.0, 1.0];
        let p = [0.2, -0.4, 0.9, 1.3];
        let w = [1.0, 1.0, 0.0, 1.0];
        for m in [
            Metric::LogLoss,
            Metric::Brier,
            Metric::Hinge,
            Metric::HingeOrdinary,
            Metric::HingeSquared,
        ] {
            let base = score_weighted(&p, &y, &w, m, 0);
            // A masked row with a NaN prediction, a NaN target, an infinite
            // target and a completely different value contributes nothing.
            let bad_p = [0.2, -0.4, f64::NAN, 1.3];
            let bad_y = [0.0, 1.0, f64::NAN, 1.0];
            let inf_y = [0.0, 1.0, f64::INFINITY, 1.0];
            assert_eq!(score_weighted(&bad_p, &y, &w, m, 0).to_bits(), base.to_bits());
            assert_eq!(score_weighted(&p, &bad_y, &w, m, 0).to_bits(), base.to_bits());
            assert_eq!(score_weighted(&p, &inf_y, &w, m, 0).to_bits(), base.to_bits());
        }
        // A positive-weight non-finite target or prediction is invalid.
        let w_bad = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            score_weighted(&p, &[0.0, 1.0, f64::NAN, 1.0], &w_bad, Metric::LogLoss, 0),
            f64::INFINITY
        );
        assert_eq!(
            score_weighted(&[0.2, f64::NAN, 0.9, 1.3], &y, &w_bad, Metric::Brier, 0),
            f64::INFINITY
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

    // --- #287: the score-objective table and native reductions ----------

    /// Every metric appears exactly once and every refusal is named. The match
    /// in `score_dispatch` is exhaustive, so an unhandled metric cannot compile;
    /// this test additionally pins the coverage and the native family.
    #[test]
    fn score_table_covers_every_metric_and_names_every_refusal() {
        let mut seen = std::collections::BTreeSet::new();
        for metric in Metric::all() {
            assert!(seen.insert(metric.name()), "duplicate table row: {}", metric.name());
            match score_dispatch(*metric) {
                ScoreDispatch::Native(o) => {
                    assert_eq!(o, native_score_objective(*metric).unwrap());
                    assert_eq!(ScoreObjective::from_code(o.code()), Some(o));
                    assert!(score_refusal(*metric).is_none());
                }
                ScoreDispatch::Exact(s) => {
                    assert_eq!(s.metric(), *metric);
                    assert_eq!(ScoreStat::for_metric(*metric), Some(s));
                    assert!(native_score_objective(*metric).is_none());
                    assert!(score_refusal(*metric).is_none());
                }
                ScoreDispatch::Refused(reason) => {
                    assert!(!reason.is_empty(), "{} has an empty refusal", metric.name());
                    assert!(reason.contains(metric.name()) || metric.name().contains("implicit"),
                        "{} refusal must name the metric: {reason}", metric.name());
                    assert!(native_score_objective(*metric).is_none());
                }
            }
        }
        assert_eq!(seen.len(), Metric::all().len());
        let native: Vec<&str> = Metric::all()
            .iter()
            .filter(|m| native_score_objective(**m).is_some())
            .map(|m| m.name())
            .collect();
        assert_eq!(
            native,
            [
                "absolute",
                "squared",
                "correlation",
                "correlation_signed",
                "maximum",
                "signed",
                "log_squashed",
                "slope",
                "implicit_residual",
                "log_loss",
                "brier",
                "hinge",
                "hinge_ordinary",
                "hinge_squared",
                "aic",
                "aicc",
                "bic",
            ]
        );
        let exact: Vec<&str> = Metric::all()
            .iter()
            .filter(|m| ScoreStat::for_metric(**m).is_some())
            .map(|m| m.name())
            .collect();
        assert_eq!(
            exact,
            ["median", "rank_correlation", "rank_correlation_signed", "interquartile", "auc"]
        );
        // An unhandled metric must be a hard failure, never an MSE fallback.
        assert_eq!(
            score_refusal(Metric::ImplicitDerivative).unwrap(),
            "implicit needs an expression-tree gradient pass"
        );
        assert_eq!(score_dispatch(Metric::Slope).objective(), Some(ScoreObjective::Slope));
        assert_eq!(score_dispatch(Metric::Auc).objective(), None);
        assert_eq!(score_dispatch(Metric::Maximum).stat(), None);
    }

    /// T1/T2 parity between the native reducer and the historical CPU metric,
    /// on exactly the fixtures named by #287.
    #[test]
    fn native_reductions_match_the_cpu_metric() {
        let fixtures: &[(&[f64], &[f64])] = &[
            (&[0.0, 3.0, 0.0], &[1.0, 1.0, 3.0]), // residuals -1, 2, -3
            (&[0.0, 0.0, 0.0, 0.0], &[3.0, 1.0, -2.0, -4.0]), // |r| 3,1,2,4
            (&[4.0], &[4.0]),                      // n = 1
            (&[0.0, 0.0], &[1.0, 1.0]),
        ];
        for (pred, y) in fixtures {
            for m in Metric::all() {
                let Some(o) = native_score_objective(*m) else {
                    continue;
                };
                let got = score_native(pred, y, o, 2);
                if o == ScoreObjective::Slope && pred.len() < 2 {
                    assert_eq!(got.status, ScoreStatus::NonFinite);
                    assert_eq!(got.value, f64::INFINITY);
                    assert_eq!(score_with_constants(pred, y, *m, 2), f64::INFINITY);
                    continue;
                }
                assert_eq!(got.status, ScoreStatus::Ok, "{}", o.name());
                let want = score_with_constants(pred, y, *m, 2);
                if o == ScoreObjective::Maximum {
                    assert_eq!(got.value.to_bits(), want.to_bits(), "maximum must be bit-exact");
                } else if want.is_infinite() {
                    assert_eq!(got.value, want, "{}", o.name());
                } else {
                    let rel = (got.value - want).abs() / (1.0 + want.abs());
                    assert!(
                        rel <= 1e-12,
                        "{} {pred:?}/{y:?}: native {} vs cpu {} (relative {rel:e})",
                        o.name(),
                        got.value,
                        want
                    );
                }
            }
        }
    }

    /// #287: correlation absolute `ρ` error ≤ 1e-9, including the near-perfect
    /// case where a relative score tolerance would be meaningless.
    #[test]
    fn native_correlation_matches_the_pearson_path_and_the_constant_rule() {
        let y: Vec<f64> = (0..97).map(|i| (i as f64 * 0.37).sin() * 4.0 + 10.0).collect();
        let near: Vec<f64> = y.iter().map(|v| v + 1e-12).collect();
        for (signed, m) in [
            (false, Metric::Correlation),
            (true, Metric::CorrelationSigned),
        ] {
            let o = native_score_objective(m).unwrap();
            let got = score_native(&near, &y, o, 0);
            let want = score_with_constants(&near, &y, m, 0);
            let want_rho = correlation(&near, &y);
            let got_rho = 1.0 - (if signed { got.value } else { got.value });
            assert!((got_rho - want_rho).abs() <= 1e-9,
                "rho error on signed={signed}: got {got_rho} want {want_rho}");
            assert!((got.value - want).abs() <= 1e-9, "score error");
            assert!(got_rho.is_finite());
        }
        // Constant prediction: CPU rule is rho = 0 -> score 1.0, never NaN.
        for pred in [vec![3.0; y.len()], vec![-0.0; y.len()]] {
            let got = score_native(&pred, &y, ScoreObjective::Correlation, 0);
            assert_eq!(got.status, ScoreStatus::Ok);
            assert_eq!(got.value, 1.0);
            let signed = score_native(&pred, &y, ScoreObjective::CorrelationSigned, 0);
            assert_eq!(signed.value, 1.0);
        }
        // Zero variance target is also rho = 0.
        let flat = vec![7.0; 40];
        let p: Vec<f64> = (0..40).map(|i| i as f64).collect();
        assert_eq!(score_native(&p, &flat, ScoreObjective::Correlation, 0).value, 1.0);
    }

    /// Maximum is an integer comparison on the `|r|` bit patterns; the first
    /// (lowest-index) row wins a tie and `-0.0` canonicalises to `+0.0`.
    #[test]
    fn native_maximum_is_bit_exact_and_breaks_ties_on_the_lowest_row() {
        let pred = [0.0, 3.0, -3.0, 1.0, 3.0, -0.0];
        let y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let got = score_native(&pred, &y, ScoreObjective::Maximum, 0);
        assert_eq!(got.value.to_bits(), 3.0f64.to_bits());
        assert_eq!(got.maximum_row, Some(1), "lowest row among |r| = 3");
        // Large magnitudes compare by bits exactly like `f64::max`.
        let huge = [1e300, -1e300, 1e-300];
        let z = [0.0; 3];
        let got = score_native(&huge, &z, ScoreObjective::Maximum, 0);
        assert_eq!(got.value, 1e300);
        assert_eq!(got.maximum_row, Some(0));
        // Signed zeros are a tie at +0, and the lowest row wins.
        let zeros = [-0.0, 0.0];
        let got = score_native(&zeros, &z[..2], ScoreObjective::Maximum, 0);
        assert_eq!(got.value.to_bits(), 0.0f64.to_bits());
        assert_eq!(got.maximum_row, Some(0));
    }

    /// Non-finite predictions yield `+inf` with a status, never a finite
    /// `1e30/n`-style score. Zero-weight masked rows are invisible.
    #[test]
    fn native_non_finite_and_zero_weight_statuses() {
        let y = [1.0, 2.0];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for o in [
                ScoreObjective::Squared,
                ScoreObjective::Absolute,
                ScoreObjective::LogSquashed,
                ScoreObjective::Signed,
                ScoreObjective::Maximum,
                ScoreObjective::Correlation,
                ScoreObjective::CorrelationSigned,
                ScoreObjective::Aic,
                ScoreObjective::Aicc,
                ScoreObjective::Bic,
                ScoreObjective::LogLoss,
                ScoreObjective::Brier,
                ScoreObjective::HingeSigmoid,
                ScoreObjective::HingeOrdinary,
                ScoreObjective::HingeSquared,
            ] {
                let got = score_native(&[bad, 1.0], &y, o, 1);
                assert_eq!(got.status, ScoreStatus::NonFinite, "{}", o.name());
                assert_eq!(got.value, f64::INFINITY, "{}", o.name());
                assert!(got.value.is_infinite(), "never a finite sentinel");
            }
        }
        // The same bad row is masked away by weight 0.
        let got = score_native_weighted(
            &[f64::NAN, 800.0],
            &y,
            &[0.0, 1.0],
            ScoreObjective::Squared,
            0,
        );
        assert_eq!(got.status, ScoreStatus::Ok);
        assert_eq!(got.value, (800.0 - 2.0) * (800.0 - 2.0));
        // All masked is a labelled refusal, not a divide by zero.
        let all_zero = score_native_weighted(&[1.0, 2.0], &y, &[0.0, 0.0], ScoreObjective::Squared, 0);
        assert_eq!(all_zero.status, ScoreStatus::AllZeroWeight);
        assert_eq!(all_zero.value, f64::INFINITY);
        // Weighted parity with the existing weighted path.
        let p = [0.0, 3.0, 0.0];
        let target = [1.0, 1.0, 3.0];
        let w = [1.0, 2.0, 1.0];
        for m in [
            Metric::Absolute,
            Metric::Squared,
            Metric::Signed,
            Metric::LogSquashed,
            Metric::Correlation,
            Metric::CorrelationSigned,
            Metric::Aic,
            Metric::Aicc,
            Metric::Bic,
        ] {
            let o = native_score_objective(m).unwrap();
            let got = score_native_weighted(&p, &target, &w, o, 2);
            let want = score_weighted(&p, &target, &w, m, 2);
            assert_eq!(got.status, ScoreStatus::Ok, "{}", m.name());
            if want.is_infinite() {
                assert_eq!(got.value, want, "{}", m.name());
            } else {
                assert!(
                    (got.value - want).abs() <= 1e-12 * (1.0 + want.abs()),
                    "{}: {} vs {}",
                    m.name(),
                    got.value,
                    want
                );
            }
        }
    }

    /// A batch is just independent reducers: the same candidate must produce
    /// the same bits alone and in a larger composition, on repeated runs.
    #[test]
    fn native_reductions_are_composition_and_order_stable() {
        let y = lcg(257, 91, 3.0);
        let pred: Vec<f64> = y.iter().map(|v| v * 0.5 + (v * 7.0).cos()).collect();
        for o in [
            ScoreObjective::Squared,
            ScoreObjective::Absolute,
            ScoreObjective::LogSquashed,
            ScoreObjective::Signed,
            ScoreObjective::Maximum,
            ScoreObjective::Correlation,
            ScoreObjective::CorrelationSigned,
            ScoreObjective::Aic,
            ScoreObjective::Aicc,
            ScoreObjective::Bic,
        ] {
            let alone = score_native(&pred, &y, o, 3);
            let again = score_native(&pred, &y, o, 3);
            assert_eq!(alone.value.to_bits(), again.value.to_bits(), "{}", o.name());
            // Composition is a host-side concept, but ensure no hidden
            // whole-batch state changes the reducer: run other candidates
            // first, then this one, and compare.
            let _ = score_native(&y, &pred, o, 3);
            let composed = score_native(&pred, &y, o, 3);
            assert_eq!(alone.value.to_bits(), composed.value.to_bits(), "{}", o.name());
            assert_eq!(alone.maximum_row, composed.maximum_row, "{}", o.name());
        }
    }

    /// `rho = 1 - 1e-12`, `1e30`, subnormals and signed zeros must not trip a
    /// magnitude sentinel or lose the finite result.
    #[test]
    fn native_edge_fixtures_are_finite_and_exact_where_required() {
        let z = [0.0; 3];
        // 1e30 residuals are finite in f64 and must stay finite.
        let p = [1e30, -1e30, 0.0];
        let sq = score_native(&p, &z, ScoreObjective::Squared, 0);
        assert!(sq.value.is_finite() && sq.value > 0.0);
        let abs = score_native(&p, &z, ScoreObjective::Absolute, 0);
        assert!(abs.value.is_finite());
        // Subnormals survive the same reduction.
        let tiny = [f64::MIN_POSITIVE / 4.0, 0.0, 0.0];
        let got = score_native(&tiny, &z, ScoreObjective::Absolute, 0);
        assert_eq!(got.value.to_bits(), (tiny[0] / 3.0).to_bits());
        // Perfect correlation fixture.
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p: Vec<f64> = y.iter().map(|v| v * 2.0 + 3.0).collect();
        let rho = 1.0 - score_native(&p, &y, ScoreObjective::Correlation, 0).value;
        assert!((rho - 1.0).abs() <= 1e-9, "rho = {rho}");
    }
}
