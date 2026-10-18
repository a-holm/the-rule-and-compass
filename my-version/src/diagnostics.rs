//! Overfitting diagnostics.
//!
//! A symbolic search is powerful enough to fit noise, so "the error is tiny"
//! is not evidence of a real law. These helpers supply the standard checks:
//!
//! * **Information criteria** (AIC / AICc / BIC) trade fit against the number
//!   of free constants, so a slightly better fit that needs three extra
//!   constants is punished. AICc is the small-sample correction (Burnham &
//!   Anderson 2002).
//! * **Durbin–Watson** tests whether the residuals are still autocorrelated
//!   (Durbin & Watson 1950/1951). For ordered data a value near 2 means "no
//!   leftover structure"; far below 2 means the model left something in.
//! * **Permutation (y-randomisation) control**: run the *same* search on a
//!   shuffled target. If it finds a fit as good as the real one, the real fit
//!   is what the search can manufacture from noise (Rücker et al. 2007;
//!   the standard "y-scrambling" test).
//!
//! None of these proves a model is right; together they make "perfect fit"
//! falsifiable.

/// Least-squares information criteria for a model with `k` free constants fit
/// to `n` points with residual sum of squares `rss`.
///
/// Returns `(aic, aicc, bic)`, all "lower is better".
pub fn information_criteria(n: usize, rss: f64, k: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let kf = k as f64;
    if n == 0 || rss <= 0.0 {
        // A perfect fit drives the log-likelihood to +inf, so the criteria go
        // to -inf. Report exactly that instead of a NaN.
        return (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    }
    let aic = nf * (rss / nf).ln() + 2.0 * kf;
    let denom = nf - kf - 1.0;
    let aicc = if denom > 0.0 {
        aic + 2.0 * kf * (kf + 1.0) / denom
    } else {
        // More constants than the data can support: undefined.
        f64::INFINITY
    };
    let bic = nf * (rss / nf).ln() + kf * nf.ln();
    (aic, aicc, bic)
}

/// Durbin–Watson statistic `d = Σ(e_t − e_{t−1})² / Σ e_t²`, in row order.
///
/// `d ≈ 2` no autocorrelation, `d → 0` positive, `d → 4` negative. Only
/// meaningful when the rows are in a natural order (a temporal split); for a
/// shuffled split read it as a curiosity. NaN for <2 points or a perfect fit.
pub fn durbin_watson(resid: &[f64]) -> f64 {
    if resid.len() < 2 {
        return f64::NAN;
    }
    let num: f64 = resid.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = resid.iter().map(|e| e * e).sum();
    if den <= 0.0 {
        return f64::NAN;
    }
    num / den
}

/// Summary of a permutation (y-randomisation) control.
#[derive(Clone, Debug)]
pub struct NullControl {
    pub runs: usize,
    /// Best training error achieved on each shuffled target.
    pub best: Vec<f64>,
    /// Best (smallest) error any shuffled run reached.
    pub null_min: f64,
    /// Median of the shuffled runs' best errors.
    pub null_median: f64,
    /// `(#{null ≤ observed} + 1) / (runs + 1)`. Small ⇒ the real fit is not
    /// something the search can produce from noise.
    pub p_value: f64,
}

impl NullControl {
    pub fn summarise(observed: f64, mut best: Vec<f64>) -> NullControl {
        best.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = best.len();
        let null_min = best.first().copied().unwrap_or(f64::NAN);
        let null_median = if n == 0 {
            f64::NAN
        } else if n % 2 == 1 {
            best[n / 2]
        } else {
            0.5 * (best[n / 2 - 1] + best[n / 2])
        };
        let count = best.iter().filter(|&&e| e <= observed).count();
        let p_value = (count as f64 + 1.0) / (n as f64 + 1.0);
        NullControl {
            runs: n,
            best,
            null_min,
            null_median,
            p_value,
        }
    }

    /// A plain-language verdict. With `r` runs the smallest attainable p-value
    /// is `1/(r+1)`, so a p-value at that floor means "beats every control" and
    /// the honest advice is to run more controls.
    pub fn verdict(&self) -> &'static str {
        let floor = 1.0 / (self.runs as f64 + 1.0);
        if self.p_value <= 0.05 {
            "the real fit beats the noise control (unlikely to be chance)"
        } else if self.p_value <= floor + 1e-9 {
            "the real fit beats every control run; add more --null-runs to sharpen p"
        } else if self.p_value <= 0.2 {
            "the real fit is only weakly better than the noise control"
        } else {
            "the search fits shuffled targets as well: suspect overfitting"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aicc_punishes_extra_constants() {
        // Same fit quality, more constants: AICc must rise.
        let n = 50;
        let rss = 1.0;
        let (_, small, _) = information_criteria(n, rss, 2);
        let (_, big, _) = information_criteria(n, rss, 10);
        assert!(big > small);
    }

    #[test]
    fn aicc_larger_than_aic() {
        let (aic, aicc, bic) = information_criteria(30, 4.0, 3);
        assert!(aicc > aic, "small-sample correction adds a penalty");
        assert!(bic > aic, "BIC penalises more than AIC at n>7");
    }

    #[test]
    fn durbin_watson_detects_structure() {
        // Alternating residuals: strongly negative autocorrelation -> d near 4.
        let alt: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!(durbin_watson(&alt) > 3.0);
        // White-ish residuals around 2.
        let mut x = 0.0f64;
        let mut rng = crate::rng::Mt::new(1);
        let noise: Vec<f64> = (0..2000)
            .map(|_| {
                x = 0.9 * x + (rng.next_f64() - 0.5);
                x
            })
            .collect();
        let d = durbin_watson(&noise);
        assert!(d < 2.0, "positively autocorrelated noise has d < 2, got {}", d);
    }

    #[test]
    fn permutation_p_value_is_small_when_observed_wins() {
        let null_best = vec![1.0, 1.2, 0.9, 1.1, 1.3];
        let c = NullControl::summarise(0.01, null_best);
        assert!(c.p_value < 0.2, "p={}", c.p_value);
        let c2 = NullControl::summarise(5.0, vec![1.0, 1.2, 0.9]);
        assert!(c2.p_value > 0.9, "p={}", c2.p_value);
    }
}
