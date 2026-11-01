//! Block uncertainty and multiple-testing statistics for long-only P&L series.
//!
//! Every statistic is exact CPU `f64` over a per-period net return series (or a
//! matrix of such series, one column per candidate). Nothing here touches the
//! test block: callers pass series that are already fixed.
//!
//! # Conventions
//!
//! * Sharpe inside this module is the **per-period** `mean / sd` (ddof = 1)
//!   unless a function says it annualises.
//! * CVaR at level `a` is the mean return of the worst `ceil(n (1 - a))`
//!   periods. It is a return, so it is negative for a losing tail.
//! * A circular block bootstrap draws `ceil(n / L)` blocks of `L` consecutive
//!   periods (the last block is truncated to `n` periods in total) from the
//!   series wrapped end to start. `L = 1` is the iid row bootstrap.
//! * Randomness is the crate's MT19937 ([`crate::rng::Mt`]) from a `u32` seed,
//!   so every draw is reproducible.
//! * Null p-values use `(1 + #{null >= observed}) / (B + 1)`.
//! * The SPA test is Hansen (2005) with a joint circular block bootstrap, the
//!   studentised statistic and the three recentering variants.
//! * DSR is Bailey and Lopez de Prado (2014); PBO is the CSCV procedure of
//!   Bailey, Borwein, Lopez de Prado and Zhu (2017).

use crate::rng::Mt;

/// Euler-Mascheroni constant used by the expected-maximum approximation.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Default circular block bootstrap draws.
pub const DEFAULT_DRAWS: usize = 10_000;

/// Standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * crate::expr::erfc(-x / std::f64::consts::SQRT_2)
}

pub use crate::noise::normal_quantile;

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        f64::NAN
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Sample standard deviation (ddof = 1).
pub fn sample_sd(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
}

/// Population skewness `m3 / m2^1.5`.
pub fn skewness(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 3 {
        return f64::NAN;
    }
    let m = mean(x);
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    if m2 > 0.0 {
        m3 / m2.powf(1.5)
    } else {
        f64::NAN
    }
}

/// Population raw kurtosis `m4 / m2^2` (3 for a normal).
pub fn kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 4 {
        return f64::NAN;
    }
    let m = mean(x);
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    if m2 > 0.0 {
        m4 / (m2 * m2)
    } else {
        f64::NAN
    }
}

/// Per-period Sharpe `mean / sd`, `NaN` for fewer than two periods or zero sd.
pub fn sharpe_per_period(x: &[f64]) -> f64 {
    let sd = sample_sd(x);
    if sd > 0.0 {
        mean(x) / sd
    } else {
        f64::NAN
    }
}

/// Type-7 quantile of an ascending slice.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n as f64 - 1.0) * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn sorted_copy(x: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

// ---------------------------------------------------------------------------
// Tail risk, drawdown ratio, cost sensitivity
// ---------------------------------------------------------------------------

/// Mean of the worst `ceil(n (1 - level))` returns (at least one).
pub fn cvar(returns: &[f64], level: f64) -> f64 {
    let s = sorted_copy(returns);
    if s.is_empty() {
        return f64::NAN;
    }
    let k = ((s.len() as f64) * (1.0 - level) - 1e-9).ceil().max(1.0) as usize;
    let k = k.min(s.len());
    s[..k].iter().sum::<f64>() / k as f64
}

/// `cagr / |max drawdown|`; `NaN` without a drawdown.
pub fn calmar(cagr: f64, max_drawdown: f64) -> f64 {
    if max_drawdown < 0.0 && cagr.is_finite() {
        cagr / -max_drawdown
    } else {
        f64::NAN
    }
}

/// Per-side cost in basis points at which the mean net return is zero.
///
/// The cost charged per period is `gross - net`, linear in the per-side cost,
/// so the break-even is `base_bps * mean(gross) / mean(gross - net)`. `NaN`
/// when the series paid no cost.
pub fn break_even_bps(net: &[f64], gross: &[f64], base_bps: f64) -> f64 {
    let n = net.len().min(gross.len());
    if n == 0 {
        return f64::NAN;
    }
    let g = gross[..n].iter().sum::<f64>();
    let c = gross[..n].iter().zip(&net[..n]).map(|(g, n)| g - n).sum::<f64>();
    if c > 0.0 {
        base_bps * g / c
    } else {
        f64::NAN
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LadderRow {
    pub multiplier: f64,
    pub cost_bps_per_side: f64,
    pub mean_return: f64,
    pub sharpe: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
}

/// Net series re-priced at `multiplier` times the base per-side cost.
pub fn cost_ladder(
    net: &[f64],
    gross: &[f64],
    base_bps: f64,
    multipliers: &[f64],
    periods_per_year: f64,
) -> Vec<LadderRow> {
    let n = net.len().min(gross.len());
    multipliers
        .iter()
        .map(|&m| {
            let repriced: Vec<f64> = (0..n).map(|i| gross[i] - m * (gross[i] - net[i])).collect();
            let r = crate::risk::risk_stats(&repriced, periods_per_year, crate::risk::DEFAULT_SORTINO_MAR);
            LadderRow {
                multiplier: m,
                cost_bps_per_side: base_bps * m,
                mean_return: r.mean,
                sharpe: r.sharpe,
                total_return: r.total_return,
                max_drawdown: r.max_drawdown,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Circular block bootstrap
// ---------------------------------------------------------------------------

/// Smallest block length that respects the label overlap and the holding
/// period, floored at `ceil(n^(1/3))` and capped at `n`.
pub fn auto_block_len(n: usize, horizon_periods: usize, hold_periods: usize) -> usize {
    let cube = (n as f64).cbrt().ceil() as usize;
    horizon_periods.max(hold_periods).max(cube).max(1).min(n.max(1))
}

/// Draws of the resampled mean and per-period Sharpe.
#[derive(Clone, Debug)]
pub struct Draws {
    pub means: Vec<f64>,
    pub sharpes: Vec<f64>,
}

/// Circular block bootstrap of the mean and Sharpe of `x`.
///
/// Block sums come from prefix sums over the series wrapped twice, so a draw
/// costs `O(n / L)`. The series is centred before the prefix sums so the
/// variance has no cancellation.
pub fn bootstrap_draws(x: &[f64], block: usize, draws: usize, seed: u32) -> Draws {
    let n = x.len();
    let mut out = Draws {
        means: Vec::with_capacity(draws),
        sharpes: Vec::with_capacity(draws),
    };
    if n < 2 {
        return out;
    }
    let block = block.clamp(1, n);
    let m0 = mean(x);
    let mut p1 = vec![0.0f64; 2 * n + 1];
    let mut p2 = vec![0.0f64; 2 * n + 1];
    for i in 0..2 * n {
        let y = x[i % n] - m0;
        p1[i + 1] = p1[i] + y;
        p2[i + 1] = p2[i] + y * y;
    }
    let mut rng = Mt::new(seed);
    let nf = n as f64;
    for _ in 0..draws {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        let mut remaining = n;
        while remaining > 0 {
            let len = block.min(remaining);
            let s = rng.usize_below(n);
            sum += p1[s + len] - p1[s];
            sumsq += p2[s + len] - p2[s];
            remaining -= len;
        }
        let var = ((sumsq - sum * sum / nf) / (nf - 1.0)).max(0.0);
        let m = m0 + sum / nf;
        out.means.push(m);
        out.sharpes.push(if var > 0.0 { m / var.sqrt() } else { f64::NAN });
    }
    out
}

fn draw_variance(v: &[f64]) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let sd = sample_sd(&f);
    sd * sd
}

#[derive(Clone, Debug)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    pub se: f64,
}

fn interval(v: &[f64], level: f64) -> Interval {
    let s = sorted_copy(v);
    let tail = (1.0 - level) / 2.0;
    Interval {
        lo: quantile_sorted(&s, tail),
        hi: quantile_sorted(&s, 1.0 - tail),
        se: sample_sd(&s),
    }
}

/// Block bootstrap summary of one return series.
#[derive(Clone, Debug)]
pub struct BootstrapSummary {
    pub n: usize,
    pub block_len: usize,
    pub draws: usize,
    pub seed: u32,
    pub level: f64,
    pub mean: f64,
    pub mean_ci: Interval,
    pub sharpe_per_period: f64,
    /// Annualised with `sqrt(periods_per_year)`.
    pub sharpe_annual: f64,
    pub sharpe_ci_annual: Interval,
    pub periods_per_year: f64,
    /// Variance of the bootstrap mean under blocks over the same under iid rows.
    pub design_effect: f64,
    /// `n / design_effect`, in periods (hours on an hourly grid).
    pub effective_n: f64,
    /// One-sided `P(mean <= 0)` from the centred block bootstrap.
    pub p_mean_positive: f64,
}

pub fn block_bootstrap(
    x: &[f64],
    block: usize,
    draws: usize,
    seed: u32,
    periods_per_year: f64,
    level: f64,
) -> BootstrapSummary {
    let n = x.len();
    let blocks = bootstrap_draws(x, block, draws, seed);
    let iid = bootstrap_draws(x, 1, draws, seed);
    let ann = periods_per_year.max(0.0).sqrt();
    let sharpe_ci = {
        let scaled: Vec<f64> = blocks.sharpes.iter().map(|s| s * ann).collect();
        interval(&scaled, level)
    };
    let v_block = draw_variance(&blocks.means);
    let v_iid = draw_variance(&iid.means);
    let design_effect = if v_iid > 0.0 { v_block / v_iid } else { f64::NAN };
    let m0 = mean(x);
    let sr = sharpe_per_period(x);
    BootstrapSummary {
        n,
        block_len: block.clamp(1, n.max(1)),
        draws,
        seed,
        level,
        mean: m0,
        mean_ci: interval(&blocks.means, level),
        sharpe_per_period: sr,
        sharpe_annual: sr * ann,
        sharpe_ci_annual: sharpe_ci,
        periods_per_year,
        design_effect,
        effective_n: if design_effect > 0.0 { n as f64 / design_effect } else { f64::NAN },
        p_mean_positive: bootstrap_p_mean_positive(x, &blocks),
    }
}

/// One-sided p-value for `mean > 0`: the share of centred draws at least as
/// large as the observed mean, `(1 + count) / (B + 1)`.
pub fn bootstrap_p_mean_positive(x: &[f64], d: &Draws) -> f64 {
    let m0 = mean(x);
    let count = d.means.iter().filter(|&&m| m - m0 >= m0).count();
    null_p_count(count, d.means.len())
}

/// `(1 + count) / (B + 1)`.
pub fn null_p_count(count: usize, draws: usize) -> f64 {
    (1.0 + count as f64) / (draws as f64 + 1.0)
}

/// `(1 + #{null >= observed}) / (B + 1)`.
pub fn null_p(observed: f64, null: &[f64]) -> f64 {
    let count = null.iter().filter(|&&v| v >= observed).count();
    null_p_count(count, null.len())
}

// ---------------------------------------------------------------------------
// Effective number of trials, DSR
// ---------------------------------------------------------------------------

/// Pearson correlation matrix of the columns. A zero-variance column is
/// uncorrelated with every other column.
pub fn correlation_matrix(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = cols.len();
    let means: Vec<f64> = cols.iter().map(|c| mean(c)).collect();
    let sds: Vec<f64> = cols.iter().map(|c| sample_sd(c)).collect();
    let mut m = vec![vec![0.0; k]; k];
    for i in 0..k {
        m[i][i] = 1.0;
        for j in 0..i {
            let n = cols[i].len().min(cols[j].len());
            let ok = n >= 2 && sds[i] > 0.0 && sds[j] > 0.0;
            let c = if ok {
                let cov = (0..n)
                    .map(|t| (cols[i][t] - means[i]) * (cols[j][t] - means[j]))
                    .sum::<f64>()
                    / (n as f64 - 1.0);
                (cov / (sds[i] * sds[j])).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            m[i][j] = c;
            m[j][i] = c;
        }
    }
    m
}

/// Participation ratio of the correlation spectrum, `(sum l)^2 / sum l^2`,
/// computed as `N^2 / sum_ij c_ij^2` (trace identities, no eigensolver).
/// Identical candidates give 1; uncorrelated candidates give `N`.
pub fn participation_ratio(corr: &[Vec<f64>]) -> f64 {
    let n = corr.len();
    if n == 0 {
        return f64::NAN;
    }
    let ss: f64 = corr.iter().flat_map(|r| r.iter()).map(|c| c * c).sum();
    (n * n) as f64 / ss
}

/// Expected maximum Sharpe of `n_trials` independent zero-skill trials whose
/// Sharpe estimates have variance `var_sr`.
pub fn expected_max_sharpe(n_trials: f64, var_sr: f64) -> f64 {
    if !(n_trials > 1.0) || !(var_sr >= 0.0) {
        return 0.0;
    }
    var_sr.sqrt()
        * ((1.0 - EULER_GAMMA) * normal_quantile(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * normal_quantile(1.0 - 1.0 / (n_trials * std::f64::consts::E)))
}

/// Inputs and result of one deflated Sharpe computation.
#[derive(Clone, Debug, PartialEq)]
pub struct Dsr {
    pub sharpe: f64,
    pub t: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub n_trials: f64,
    pub var_sr: f64,
    pub sr0: f64,
    /// `P(true Sharpe > 0)` after deflating for the trials.
    pub dsr: f64,
}

/// Deflated Sharpe ratio. `sharpe` is per period, `t` the number of periods
/// (or the effective number), `kurtosis` the raw kurtosis.
pub fn deflated_sharpe(sharpe: f64, t: f64, skew: f64, kurt: f64, n_trials: f64, var_sr: f64) -> Dsr {
    let sr0 = expected_max_sharpe(n_trials, var_sr);
    let denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe;
    let dsr = if denom > 0.0 && t > 1.0 {
        normal_cdf((sharpe - sr0) * (t - 1.0).sqrt() / denom.sqrt())
    } else {
        f64::NAN
    };
    Dsr {
        sharpe,
        t,
        skew,
        kurtosis: kurt,
        n_trials,
        var_sr,
        sr0,
        dsr,
    }
}

/// Large-sample variance of a per-period Sharpe estimate (Mertens).
pub fn sharpe_estimator_variance(sharpe: f64, t: f64, skew: f64, kurt: f64) -> f64 {
    (1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe) / (t - 1.0)
}

// ---------------------------------------------------------------------------
// PBO by combinatorially symmetric cross-validation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Pbo {
    pub candidates: usize,
    pub blocks: usize,
    pub rows_per_block: usize,
    pub combinations: usize,
    pub pbo: f64,
    pub mean_logit: f64,
    pub logits: Vec<f64>,
}

#[derive(Clone, Copy, Default)]
struct Moments {
    n: f64,
    sum: f64,
    sumsq: f64,
}

fn moments_sharpe(m: Moments) -> f64 {
    if m.n < 2.0 {
        return f64::NEG_INFINITY;
    }
    let var = ((m.sumsq - m.sum * m.sum / m.n) / (m.n - 1.0)).max(0.0);
    if var > 0.0 {
        (m.sum / m.n) / var.sqrt()
    } else {
        f64::NEG_INFINITY
    }
}

/// Probability of backtest overfitting. `cols[k]` is candidate `k`'s aligned
/// series; `s` (even) contiguous blocks are formed from the first
/// `s * floor(T / s)` periods and every choice of `s / 2` blocks is the
/// in-sample half. The performance measure is the per-period Sharpe of the raw
/// series; nothing is demeaned. Ties: `rank = #{oos <= oos[best]}`, so a
/// candidate tied with the winner counts as not worse than it.
pub fn pbo_cscv(cols: &[Vec<f64>], s: usize) -> Result<Pbo, String> {
    let k = cols.len();
    if k < 2 {
        return Err("PBO needs at least two candidates".to_string());
    }
    if s < 2 || s % 2 != 0 || s > 24 {
        return Err("PBO block count must be even and in 2..=24".to_string());
    }
    let t = cols.iter().map(|c| c.len()).min().unwrap_or(0);
    let rows = t / s;
    if rows < 2 {
        return Err(format!("PBO needs at least {} periods for {} blocks", 2 * s, s));
    }
    let mut block = vec![vec![Moments::default(); s]; k];
    for (c, col) in cols.iter().enumerate() {
        for b in 0..s {
            let mut m = Moments::default();
            for &y in &col[b * rows..(b + 1) * rows] {
                m.n += 1.0;
                m.sum += y;
                m.sumsq += y * y;
            }
            block[c][b] = m;
        }
    }
    let mut logits = Vec::new();
    let full = (1u32 << s) - 1;
    for mask in 0..=full {
        if mask.count_ones() as usize != s / 2 {
            continue;
        }
        let mut is_perf = vec![0.0; k];
        let mut oos_perf = vec![0.0; k];
        for c in 0..k {
            let mut a = Moments::default();
            let mut b = Moments::default();
            for (i, m) in block[c].iter().enumerate() {
                let dst = if mask >> i & 1 == 1 { &mut a } else { &mut b };
                dst.n += m.n;
                dst.sum += m.sum;
                dst.sumsq += m.sumsq;
            }
            is_perf[c] = moments_sharpe(a);
            oos_perf[c] = moments_sharpe(b);
        }
        let mut best = 0;
        for c in 1..k {
            if is_perf[c] > is_perf[best] {
                best = c;
            }
        }
        let rank = oos_perf.iter().filter(|&&p| p <= oos_perf[best]).count();
        let omega = rank as f64 / (k as f64 + 1.0);
        logits.push((omega / (1.0 - omega)).ln());
    }
    let combos = logits.len();
    let overfit = logits.iter().filter(|&&l| l <= 0.0).count();
    Ok(Pbo {
        candidates: k,
        blocks: s,
        rows_per_block: rows,
        combinations: combos,
        pbo: overfit as f64 / combos as f64,
        mean_logit: mean(&logits),
        logits,
    })
}

// ---------------------------------------------------------------------------
// Superior predictive ability
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Spa {
    pub models: usize,
    pub n: usize,
    pub block_len: usize,
    pub draws: usize,
    pub seed: u32,
    /// `max(0, max_k sqrt(n) dbar_k / omega_k)`.
    pub statistic: f64,
    /// Index of the model with the largest studentised mean (`None` when all are non-positive).
    pub best_model: Option<usize>,
    pub p_consistent: f64,
    pub p_lower: f64,
    pub p_upper: f64,
}

/// Hansen's SPA test. `d[k][t]` is model `k`'s performance minus the
/// benchmark's at period `t` (higher is better); the null is that no model
/// beats the benchmark.
pub fn spa_test(d: &[Vec<f64>], block: usize, draws: usize, seed: u32) -> Result<Spa, String> {
    let k = d.len();
    if k == 0 {
        return Err("SPA needs at least one model".to_string());
    }
    let n = d.iter().map(|c| c.len()).min().unwrap_or(0);
    if n < 3 {
        return Err("SPA needs at least 3 periods".to_string());
    }
    let block = block.clamp(1, n);
    let nf = n as f64;
    let dbar: Vec<f64> = d.iter().map(|c| mean(&c[..n])).collect();
    let mut prefix = vec![vec![0.0f64; 2 * n + 1]; k];
    for (c, col) in d.iter().enumerate() {
        for i in 0..2 * n {
            prefix[c][i + 1] = prefix[c][i] + (col[i % n] - dbar[c]);
        }
    }
    let mut rng = Mt::new(seed);
    let mut star = vec![vec![0.0f64; k]; draws];
    let mut starts: Vec<(usize, usize)> = Vec::new();
    for row in star.iter_mut() {
        starts.clear();
        let mut remaining = n;
        while remaining > 0 {
            let len = block.min(remaining);
            starts.push((rng.usize_below(n), len));
            remaining -= len;
        }
        for c in 0..k {
            let s: f64 = starts.iter().map(|&(s, len)| prefix[c][s + len] - prefix[c][s]).sum();
            row[c] = s / nf;
        }
    }
    let omega: Vec<f64> = (0..k)
        .map(|c| {
            let v: Vec<f64> = star.iter().map(|r| r[c]).collect();
            nf.sqrt() * sample_sd(&v)
        })
        .collect();
    let lil = (2.0 * nf.ln().ln()).sqrt();
    let t_k: Vec<f64> = (0..k)
        .map(|c| {
            if omega[c] > 0.0 {
                nf.sqrt() * dbar[c] / omega[c]
            } else if dbar[c] > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let mut best_model = None;
    let mut statistic = 0.0f64;
    for (c, &t) in t_k.iter().enumerate() {
        if t > statistic {
            statistic = t;
            best_model = Some(c);
        }
    }
    let p_for = |mu: &dyn Fn(usize) -> f64| -> f64 {
        let mut count = 0usize;
        for row in &star {
            let mut tb = 0.0f64;
            for c in 0..k {
                if omega[c] > 0.0 {
                    let z = nf.sqrt() * (row[c] + mu(c)) / omega[c];
                    if z > tb {
                        tb = z;
                    }
                }
            }
            if tb >= statistic {
                count += 1;
            }
        }
        null_p_count(count, draws)
    };
    let mu_c = |c: usize| if omega[c] > 0.0 && t_k[c] <= -lil { dbar[c] } else { 0.0 };
    let mu_l = |c: usize| dbar[c].min(0.0);
    let mu_u = |_c: usize| 0.0;
    Ok(Spa {
        models: k,
        n,
        block_len: block,
        draws,
        seed,
        statistic,
        best_model,
        p_consistent: p_for(&mu_c),
        p_lower: p_for(&mu_l),
        p_upper: p_for(&mu_u),
    })
}

// ---------------------------------------------------------------------------
// Benjamini-Hochberg
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Bh {
    pub q: f64,
    pub rejected: Vec<bool>,
    pub adjusted: Vec<f64>,
    pub n_rejected: usize,
}

/// Benjamini-Hochberg step-up at false discovery rate `q`.
pub fn bh_fdr(p: &[f64], q: f64) -> Bh {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cutoff = 0usize;
    for (rank, &i) in order.iter().enumerate() {
        if p[i] <= (rank + 1) as f64 / m as f64 * q {
            cutoff = rank + 1;
        }
    }
    let mut rejected = vec![false; m];
    for &i in &order[..cutoff] {
        rejected[i] = true;
    }
    let mut adjusted = vec![0.0; m];
    let mut running = f64::INFINITY;
    for (rank, &i) in order.iter().enumerate().rev() {
        running = running.min(p[i] * m as f64 / (rank + 1) as f64).min(1.0);
        adjusted[i] = running;
    }
    Bh {
        q,
        rejected,
        adjusted,
        n_rejected: cutoff,
    }
}
