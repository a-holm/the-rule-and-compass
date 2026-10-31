//! Long-only economics, block uncertainty and nulls for the walk-forward report.
//!
//! Everything is exact CPU `f64` over the per-period net series the evaluator
//! already produces. See [`crate::inference`] for the statistics and
//! `docs/walkforward.md` for the report fields.

use crate::baselines;
use crate::inference::{self, Bh, BootstrapSummary, Dsr, LadderRow, Pbo, Spa};
use crate::risk::{risk_stats, BetaAlpha, DEFAULT_SORTINO_MAR};
use crate::rng::Mt;

use crate::walkforward::{jnum, jstr};

/// Cost multipliers of the report's cost ladder.
pub const COST_LADDER: [f64; 4] = [0.5, 1.0, 2.0, 3.0];

/// One window's rows in group-contiguous order, kept for the nulls.
#[derive(Clone, Debug)]
pub struct NullBook {
    pub positions: Vec<f64>,
    pub returns: Vec<f64>,
    pub groups: Vec<u64>,
    /// Period index of each row.
    pub period: Vec<u32>,
    /// Rows per period.
    pub period_count: Vec<u32>,
    /// Observed cost per period (`gross - net`), held fixed under the nulls.
    pub period_cost: Vec<f64>,
    pub side: f64,
}

// ---------------------------------------------------------------------------
// Economics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BenchRow {
    pub name: String,
    pub mean_return: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    /// Long rows over valid rows.
    pub exposure: f64,
    /// Beta of the series on the equal-weight universe.
    pub beta: f64,
    /// Mean per-period return above cash (cash returns zero).
    pub mean_excess_over_cash: f64,
}

#[derive(Clone, Debug)]
pub struct Daily {
    pub days: usize,
    pub sharpe: f64,
    pub sortino: f64,
}

#[derive(Clone, Debug)]
pub struct Economics {
    pub cost_bps: f64,
    pub periods_per_year: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub cvar95: f64,
    pub cvar99: f64,
    pub break_even_bps_per_side: f64,
    pub ladder: Vec<LadderRow>,
    pub daily: Option<Daily>,
    /// Strategy, cash, buy-and-hold, equal-weight universe, constant long/flat.
    pub benchmarks: Vec<BenchRow>,
    /// The best of cash, buy-and-hold and constant long/flat by mean return.
    pub best_baseline: String,
}

pub struct EconomicsInput<'a> {
    pub net: &'a [f64],
    pub gross: &'a [f64],
    pub times: &'a [f64],
    pub cost_bps: f64,
    pub periods_per_year: f64,
    pub sortino: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub exposure: f64,
    /// Strategy series with every window started flat, like the controls.
    pub fresh: &'a [f64],
    pub market: &'a [f64],
    pub buy_and_hold: &'a [f64],
    pub constant: &'a [f64],
    pub bh_exposure: f64,
    pub constant_exposure: f64,
}

fn bench_row(name: &str, series: &[f64], exposure: f64, market: &[f64], ppy: f64) -> BenchRow {
    let r = risk_stats(series, ppy, DEFAULT_SORTINO_MAR);
    let mut ba = BetaAlpha::identity();
    for (x, y) in market.iter().zip(series) {
        ba.push(*x, *y);
    }
    BenchRow {
        name: name.to_string(),
        mean_return: r.mean,
        sharpe: r.sharpe,
        sortino: r.sortino,
        total_return: r.total_return,
        max_drawdown: r.max_drawdown,
        exposure,
        beta: ba.beta(),
        mean_excess_over_cash: r.mean,
    }
}

/// Daily series from epoch-second period stamps: the periods of one UTC day
/// compound into one return.
pub fn daily_returns(times: &[f64], net: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    let mut day = i64::MIN;
    for (t, r) in times.iter().zip(net) {
        let d = (t / 86_400.0).floor() as i64;
        if d != day {
            out.push(0.0);
            day = d;
        }
        let last = out.last_mut().unwrap();
        *last = (1.0 + *last) * (1.0 + r) - 1.0;
    }
    out
}

pub fn economics(i: &EconomicsInput) -> Economics {
    let zeros = vec![0.0; i.fresh.len()];
    let mut benchmarks = vec![
        bench_row("strategy", i.fresh, i.exposure, i.market, i.periods_per_year),
        bench_row("cash", &zeros, 0.0, i.market, i.periods_per_year),
        bench_row("buy_and_hold", i.buy_and_hold, i.bh_exposure, i.market, i.periods_per_year),
        bench_row("equal_weight_universe", i.market, 1.0, i.market, i.periods_per_year),
        bench_row("constant_long_flat", i.constant, i.constant_exposure, i.market, i.periods_per_year),
    ];
    // Cash, buy-and-hold, constant long/flat. A tie goes to the earlier entry.
    let mut best = 1usize;
    for k in [2usize, 4] {
        if benchmarks[k].mean_return > benchmarks[best].mean_return {
            best = k;
        }
    }
    let best_baseline = benchmarks[best].name.clone();
    benchmarks.push(BenchRow {
        name: "best_long_flat_baseline".to_string(),
        ..benchmarks[best].clone()
    });
    let daily_ok = i.periods_per_year >= 2.0 && i.times.first().map_or(false, |&t| t > 1e8);
    let daily = if daily_ok {
        let d = daily_returns(i.times, i.net);
        let r = risk_stats(&d, 365.0, DEFAULT_SORTINO_MAR);
        Some(Daily {
            days: d.len(),
            sharpe: r.sharpe,
            sortino: r.sortino,
        })
    } else {
        None
    };
    Economics {
        cost_bps: i.cost_bps,
        periods_per_year: i.periods_per_year,
        sortino: i.sortino,
        calmar: inference::calmar(i.cagr, i.max_drawdown),
        cvar95: inference::cvar(i.net, 0.95),
        cvar99: inference::cvar(i.net, 0.99),
        break_even_bps_per_side: inference::break_even_bps(i.net, i.gross, i.cost_bps),
        ladder: inference::cost_ladder(i.net, i.gross, i.cost_bps, &COST_LADDER, i.periods_per_year),
        daily,
        benchmarks,
        best_baseline,
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Multiple {
    pub trials: usize,
    pub candidates: usize,
    pub participation_ratio: f64,
    pub n_eff: f64,
    pub t_used: f64,
    pub dsr_n_eff: Dsr,
    pub dsr_10_n_eff: Dsr,
    pub pbo: Option<Pbo>,
    pub pbo_error: Option<String>,
    pub p_values: Vec<f64>,
    pub bh: Bh,
}

#[derive(Clone, Debug)]
pub struct NullRow {
    pub p_mean: f64,
    pub p_sharpe: f64,
    pub null_mean_q95: f64,
    pub null_sharpe_q95: f64,
}

#[derive(Clone, Debug)]
pub struct NullTests {
    pub draws: usize,
    pub block_len: usize,
    pub seed: u32,
    pub periods: usize,
    pub observed_mean: f64,
    pub observed_sharpe: f64,
    pub block_permutation: NullRow,
    pub matched_exposure_random: NullRow,
    pub always_long_mean: f64,
    pub always_long_sharpe: f64,
}

#[derive(Clone, Debug)]
pub struct Inference {
    pub bootstrap: Option<BootstrapSummary>,
    pub multiple: Option<Multiple>,
    pub spa: Option<(String, Spa)>,
    pub nulls: Option<NullTests>,
}

pub struct InferenceInput<'a> {
    pub draws: usize,
    pub null_draws: usize,
    pub block_len: usize,
    pub seed: u32,
    pub control_seed: u64,
    pub periods_per_year: f64,
    pub pooled: &'a [f64],
    pub fresh: &'a [f64],
    /// One column per candidate on the strategy's periods.
    pub candidates: &'a [Vec<f64>],
    pub trials: usize,
    pub pbo_blocks: usize,
    pub fdr_q: f64,
    pub best_baseline: &'a str,
    pub baseline_series: &'a [f64],
    pub buy_and_hold: &'a [f64],
    pub books: &'a [NullBook],
}

pub fn infer(i: &InferenceInput) -> Result<Inference, String> {
    let mut out = Inference {
        bootstrap: None,
        multiple: None,
        spa: None,
        nulls: None,
    };
    if i.draws > 0 {
        let summary = inference::block_bootstrap(i.pooled, i.block_len, i.draws, i.seed, i.periods_per_year, 0.95);
        let t_used = if summary.effective_n.is_finite() && summary.effective_n > 1.0 {
            summary.effective_n.min(i.pooled.len() as f64)
        } else {
            i.pooled.len() as f64
        };
        let sr = inference::sharpe_per_period(i.pooled);
        let skew = inference::skewness(i.pooled);
        let kurt = inference::kurtosis(i.pooled);
        let k = i.candidates.len();
        let var_sr = if k >= 2 {
            let s: Vec<f64> = i.candidates.iter().map(|c| inference::sharpe_per_period(c)).filter(|v| v.is_finite()).collect();
            let sd = inference::sample_sd(&s);
            sd * sd
        } else {
            inference::sharpe_estimator_variance(sr, t_used, skew, kurt)
        };
        let pr = if k >= 2 {
            inference::participation_ratio(&inference::correlation_matrix(i.candidates))
        } else {
            1.0
        };
        let trials = i.trials.max(k).max(1);
        let n_eff = if k >= 2 { trials as f64 * pr / k as f64 } else { trials as f64 };
        let (pbo, pbo_error) = match inference::pbo_cscv(i.candidates, i.pbo_blocks) {
            Ok(p) => (Some(p), None),
            Err(e) => (None, Some(e)),
        };
        let p_values: Vec<f64> = if i.candidates.is_empty() {
            vec![summary.p_mean_positive]
        } else {
            i.candidates
                .iter()
                .enumerate()
                .map(|(c, col)| {
                    let d = inference::bootstrap_draws(col, i.block_len, i.draws, i.seed.wrapping_add(c as u32 + 1));
                    inference::bootstrap_p_mean_positive(col, &d)
                })
                .collect()
        };
        out.bootstrap = Some(summary.clone());
        out.multiple = Some(Multiple {
            trials,
            candidates: k.max(1),
            participation_ratio: pr,
            n_eff,
            t_used,
            dsr_n_eff: inference::deflated_sharpe(sr, t_used, skew, kurt, n_eff, var_sr),
            dsr_10_n_eff: inference::deflated_sharpe(sr, t_used, skew, kurt, 10.0 * n_eff, var_sr),
            pbo,
            pbo_error,
            bh: inference::bh_fdr(&p_values, i.fdr_q),
            p_values,
        });
        let models: Vec<&Vec<f64>> = if i.candidates.is_empty() { Vec::new() } else { i.candidates.iter().collect() };
        let mut diffs: Vec<Vec<f64>> = models
            .iter()
            .map(|c| c.iter().zip(i.baseline_series).map(|(a, b)| a - b).collect())
            .collect();
        if diffs.is_empty() {
            diffs.push(i.fresh.iter().zip(i.baseline_series).map(|(a, b)| a - b).collect());
        }
        out.spa = Some((
            i.best_baseline.to_string(),
            inference::spa_test(&diffs, i.block_len, i.draws, i.seed)?,
        ));
    }
    if i.null_draws > 0 {
        out.nulls = Some(run_nulls(i)?);
    }
    Ok(out)
}

struct Stat {
    n: f64,
    sum: f64,
    sumsq: f64,
}

impl Stat {
    fn new() -> Self {
        Stat { n: 0.0, sum: 0.0, sumsq: 0.0 }
    }
    fn push(&mut self, v: f64) {
        self.n += 1.0;
        self.sum += v;
        self.sumsq += v * v;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n
    }
    fn sharpe(&self) -> f64 {
        let var = ((self.sumsq - self.sum * self.sum / self.n) / (self.n - 1.0)).max(0.0);
        if var > 0.0 {
            self.mean() / var.sqrt()
        } else {
            f64::NAN
        }
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Dense target matrix of one window for the block permutation.
struct Dense {
    gid: Vec<u32>,
    target: Vec<f64>,
    periods: usize,
}

fn dense(book: &NullBook) -> Dense {
    let p = book.period_count.len();
    let mut gid = Vec::with_capacity(book.groups.len());
    let mut current = u32::MAX;
    let mut last: Option<u64> = None;
    for &g in &book.groups {
        if last != Some(g) {
            current = current.wrapping_add(1);
            last = Some(g);
        }
        gid.push(current);
    }
    let n_groups = gid.last().map_or(0, |&g| g as usize + 1);
    let mut target = vec![0.0; n_groups * p];
    for (i, &g) in gid.iter().enumerate() {
        target[g as usize * p + book.period[i] as usize] = book.returns[i];
    }
    Dense { gid, target, periods: p }
}

fn permuted_sources(periods: usize, block: usize, rng: &mut Mt) -> Vec<u32> {
    let nb = periods.div_ceil(block);
    let mut perm: Vec<usize> = (0..nb).collect();
    for i in (1..nb).rev() {
        perm.swap(i, rng.usize_below(i + 1));
    }
    let mut src = vec![0u32; periods];
    for b in 0..nb {
        let start = b * block;
        let len = block.min(periods - start);
        let s_start = perm[b] * block;
        let s_len = block.min(periods - s_start);
        for off in 0..len {
            src[start + off] = (s_start + off % s_len) as u32;
        }
    }
    src
}

fn run_nulls(i: &InferenceInput) -> Result<NullTests, String> {
    if i.books.is_empty() {
        return Err("nulls need the window books".to_string());
    }
    let block = i.block_len.max(1);
    let dense_books: Vec<Dense> = i.books.iter().map(dense).collect();
    let mut obs = Stat::new();
    for &v in i.fresh {
        obs.push(v);
    }
    let (obs_mean, obs_sharpe) = (obs.mean(), obs.sharpe());
    let mut perm_rng = Mt::new(i.seed.wrapping_add(7919));
    let mut perm_mean = Vec::with_capacity(i.null_draws);
    let mut perm_sharpe = Vec::with_capacity(i.null_draws);
    let mut rand_mean = Vec::with_capacity(i.null_draws);
    let mut rand_sharpe = Vec::with_capacity(i.null_draws);
    for b in 0..i.null_draws {
        let mut p_stat = Stat::new();
        let mut r_stat = Stat::new();
        for (w, (book, d)) in i.books.iter().zip(&dense_books).enumerate() {
            let periods = d.periods;
            let src = permuted_sources(periods, block, &mut perm_rng);
            let mut sum = vec![0.0f64; periods];
            for (row, &g) in d.gid.iter().enumerate() {
                let pos = book.positions[row];
                if pos != 0.0 {
                    let k = book.period[row] as usize;
                    sum[k] += pos * d.target[g as usize * periods + src[k] as usize];
                }
            }
            for k in 0..periods {
                p_stat.push(sum[k] / book.period_count[k] as f64 - book.period_cost[k]);
            }
            let seed = splitmix(i.control_seed ^ splitmix((b as u64) << 20 | w as u64));
            let pos = baselines::matched_exposure_positions(&book.positions, &book.groups, seed)?;
            let net = baselines::net_rows(&pos, &book.returns, &book.groups, book.side);
            let mut bins = vec![0.0f64; periods];
            for (row, &v) in net.iter().enumerate() {
                bins[book.period[row] as usize] += v;
            }
            for k in 0..periods {
                r_stat.push(bins[k] / book.period_count[k] as f64);
            }
        }
        perm_mean.push(p_stat.mean());
        perm_sharpe.push(p_stat.sharpe());
        rand_mean.push(r_stat.mean());
        rand_sharpe.push(r_stat.sharpe());
    }
    let q95 = |v: &[f64]| {
        let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        inference::quantile_sorted(&s, 0.95)
    };
    let row = |mean: &[f64], sharpe: &[f64]| NullRow {
        p_mean: inference::null_p(obs_mean, mean),
        p_sharpe: inference::null_p(obs_sharpe, sharpe),
        null_mean_q95: q95(mean),
        null_sharpe_q95: q95(sharpe),
    };
    let mut bh = Stat::new();
    for &v in i.buy_and_hold {
        bh.push(v);
    }
    Ok(NullTests {
        draws: i.null_draws,
        block_len: block,
        seed: i.seed,
        periods: i.fresh.len(),
        observed_mean: obs_mean,
        observed_sharpe: obs_sharpe,
        block_permutation: row(&perm_mean, &perm_sharpe),
        matched_exposure_random: row(&rand_mean, &rand_sharpe),
        always_long_mean: bh.mean(),
        always_long_sharpe: bh.sharpe(),
    })
}

// ---------------------------------------------------------------------------
// JSON and Markdown
// ---------------------------------------------------------------------------

fn interval_json(lo: f64, hi: f64, se: f64) -> String {
    format!("{{\"lo\":{},\"hi\":{},\"se\":{}}}", jnum(lo), jnum(hi), jnum(se))
}

fn dsr_json(d: &Dsr) -> String {
    format!(
        "{{\"n_trials\":{},\"sharpe_per_period\":{},\"t\":{},\"skew\":{},\"kurtosis\":{},\
         \"var_sr\":{},\"sr0\":{},\"dsr\":{}}}",
        jnum(d.n_trials),
        jnum(d.sharpe),
        jnum(d.t),
        jnum(d.skew),
        jnum(d.kurtosis),
        jnum(d.var_sr),
        jnum(d.sr0),
        jnum(d.dsr)
    )
}

fn null_row_json(r: &NullRow) -> String {
    format!(
        "{{\"p_mean\":{},\"p_sharpe\":{},\"null_mean_q95\":{},\"null_sharpe_q95\":{}}}",
        jnum(r.p_mean),
        jnum(r.p_sharpe),
        jnum(r.null_mean_q95),
        jnum(r.null_sharpe_q95)
    )
}

impl Economics {
    pub fn json(&self) -> String {
        let ladder: Vec<String> = self
            .ladder
            .iter()
            .map(|r| {
                format!(
                    "{{\"multiplier\":{},\"cost_bps_per_side\":{},\"mean_return\":{},\"sharpe\":{},\
                     \"total_return\":{},\"max_drawdown\":{}}}",
                    jnum(r.multiplier),
                    jnum(r.cost_bps_per_side),
                    jnum(r.mean_return),
                    jnum(r.sharpe),
                    jnum(r.total_return),
                    jnum(r.max_drawdown)
                )
            })
            .collect();
        let bench: Vec<String> = self
            .benchmarks
            .iter()
            .map(|b| {
                format!(
                    "{{\"name\":{},\"mean_return\":{},\"sharpe\":{},\"sortino\":{},\"total_return\":{},\
                     \"max_drawdown\":{},\"exposure\":{},\"beta_vs_equal_weight\":{},\
                     \"mean_excess_over_cash\":{}}}",
                    jstr(&b.name),
                    jnum(b.mean_return),
                    jnum(b.sharpe),
                    jnum(b.sortino),
                    jnum(b.total_return),
                    jnum(b.max_drawdown),
                    jnum(b.exposure),
                    jnum(b.beta),
                    jnum(b.mean_excess_over_cash)
                )
            })
            .collect();
        let daily = self.daily.as_ref().map_or("null".to_string(), |d| {
            format!(
                "{{\"days\":{},\"periods_per_year\":365,\"sharpe\":{},\"sortino\":{}}}",
                d.days,
                jnum(d.sharpe),
                jnum(d.sortino)
            )
        });
        format!(
            "{{\"book\":\"long_flat_vs_cash\",\"cost_bps_per_side\":{},\"periods_per_year\":{},\
             \"sortino\":{},\"calmar\":{},\"cvar_95\":{},\"cvar_99\":{},\
             \"break_even_bps_per_side\":{},\"cost_ladder\":[{}],\"daily\":{},\
             \"benchmark_series\":\"every window started flat\",\"best_baseline\":{},\"benchmarks\":[{}]}}",
            jnum(self.cost_bps),
            jnum(self.periods_per_year),
            jnum(self.sortino),
            jnum(self.calmar),
            jnum(self.cvar95),
            jnum(self.cvar99),
            jnum(self.break_even_bps_per_side),
            ladder.join(","),
            daily,
            jstr(&self.best_baseline),
            bench.join(",")
        )
    }

    pub fn markdown(&self) -> String {
        let f = |v: f64| if v.is_finite() { format!("{:.4}", v) } else { "n/a".to_string() };
        let mut s = String::from("## Long-only economics\n\n");
        s.push_str(&format!(
            "Sortino {} | Calmar {} | CVaR 95 {} | CVaR 99 {} | break-even {} bp/side (cost {} bp/side)\n\n",
            f(self.sortino),
            f(self.calmar),
            f(self.cvar95),
            f(self.cvar99),
            f(self.break_even_bps_per_side),
            f(self.cost_bps)
        ));
        if let Some(d) = &self.daily {
            s.push_str(&format!(
                "Daily ({} days, sqrt 365): Sharpe {} | Sortino {}\n\n",
                d.days,
                f(d.sharpe),
                f(d.sortino)
            ));
        }
        s.push_str("| cost x | bp/side | mean | Sharpe | total return | max drawdown |\n|---|---|---|---|---|---|\n");
        for r in &self.ladder {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                r.multiplier,
                f(r.cost_bps_per_side),
                f(r.mean_return),
                f(r.sharpe),
                f(r.total_return),
                f(r.max_drawdown)
            ));
        }
        s.push_str(&format!(
            "\nBenchmarks (every window started flat; best long/flat baseline: {}):\n\n",
            self.best_baseline
        ));
        s.push_str("| series | mean | Sharpe | total return | max drawdown | exposure | beta | excess over cash |\n|---|---|---|---|---|---|---|---|\n");
        for b in &self.benchmarks {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                b.name,
                f(b.mean_return),
                f(b.sharpe),
                f(b.total_return),
                f(b.max_drawdown),
                f(b.exposure),
                f(b.beta),
                f(b.mean_excess_over_cash)
            ));
        }
        s
    }
}

impl Inference {
    pub fn json(&self) -> String {
        let bootstrap = self.bootstrap.as_ref().map_or("null".to_string(), |b| {
            format!(
                "{{\"n\":{},\"block_len\":{},\"draws\":{},\"seed\":{},\"level\":{},\"mean\":{},\
                 \"mean_ci\":{},\"sharpe_per_period\":{},\"sharpe_annual\":{},\"sharpe_ci_annual\":{},\
                 \"periods_per_year\":{},\"design_effect\":{},\"effective_n\":{},\"p_mean_positive\":{}}}",
                b.n,
                b.block_len,
                b.draws,
                b.seed,
                jnum(b.level),
                jnum(b.mean),
                interval_json(b.mean_ci.lo, b.mean_ci.hi, b.mean_ci.se),
                jnum(b.sharpe_per_period),
                jnum(b.sharpe_annual),
                interval_json(b.sharpe_ci_annual.lo, b.sharpe_ci_annual.hi, b.sharpe_ci_annual.se),
                jnum(b.periods_per_year),
                jnum(b.design_effect),
                jnum(b.effective_n),
                jnum(b.p_mean_positive)
            )
        });
        let multiple = self.multiple.as_ref().map_or("null".to_string(), |m| {
            let pbo = m.pbo.as_ref().map_or("null".to_string(), |p| {
                format!(
                    "{{\"blocks\":{},\"rows_per_block\":{},\"combinations\":{},\"candidates\":{},\
                     \"pbo\":{},\"mean_logit\":{}}}",
                    p.blocks,
                    p.rows_per_block,
                    p.combinations,
                    p.candidates,
                    jnum(p.pbo),
                    jnum(p.mean_logit)
                )
            });
            format!(
                "{{\"trials\":{},\"candidates\":{},\"participation_ratio\":{},\"n_eff\":{},\
                 \"t_used\":{},\"dsr_at_n_eff\":{},\"dsr_at_10_n_eff\":{},\"pbo\":{},\"pbo_error\":{},\
                 \"p_values\":[{}],\"bh_fdr\":{{\"q\":{},\"rejected\":{},\"adjusted\":[{}]}}}}",
                m.trials,
                m.candidates,
                jnum(m.participation_ratio),
                jnum(m.n_eff),
                jnum(m.t_used),
                dsr_json(&m.dsr_n_eff),
                dsr_json(&m.dsr_10_n_eff),
                pbo,
                m.pbo_error.as_ref().map_or("null".to_string(), |e| jstr(e)),
                m.p_values.iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(","),
                jnum(m.bh.q),
                m.bh.n_rejected,
                m.bh.adjusted.iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(",")
            )
        });
        let spa = self.spa.as_ref().map_or("null".to_string(), |(name, s)| {
            format!(
                "{{\"benchmark\":{},\"variant\":\"hansen_2005_circular_block\",\"models\":{},\"n\":{},\
                 \"block_len\":{},\"draws\":{},\"statistic\":{},\"best_model\":{},\
                 \"p_consistent\":{},\"p_lower\":{},\"p_upper\":{}}}",
                jstr(name),
                s.models,
                s.n,
                s.block_len,
                s.draws,
                jnum(s.statistic),
                s.best_model.map_or("null".to_string(), |v| v.to_string()),
                jnum(s.p_consistent),
                jnum(s.p_lower),
                jnum(s.p_upper)
            )
        });
        let nulls = self.nulls.as_ref().map_or("null".to_string(), |n| {
            format!(
                "{{\"draws\":{},\"block_len\":{},\"seed\":{},\"periods\":{},\"observed_mean\":{},\
                 \"observed_sharpe_per_period\":{},\"p_formula\":\"(1+count)/(B+1)\",\
                 \"block_permutation\":{},\"matched_exposure_random\":{},\
                 \"always_long\":{{\"mean\":{},\"sharpe_per_period\":{}}}}}",
                n.draws,
                n.block_len,
                n.seed,
                n.periods,
                jnum(n.observed_mean),
                jnum(n.observed_sharpe),
                null_row_json(&n.block_permutation),
                null_row_json(&n.matched_exposure_random),
                jnum(n.always_long_mean),
                jnum(n.always_long_sharpe)
            )
        });
        format!(
            "{{\"bootstrap\":{},\"multiple_testing\":{},\"spa\":{},\"nulls\":{}}}",
            bootstrap, multiple, spa, nulls
        )
    }

    pub fn markdown(&self) -> String {
        let f = |v: f64| if v.is_finite() { format!("{:.4}", v) } else { "n/a".to_string() };
        let mut s = String::from("## Block uncertainty and multiple testing\n\n");
        if let Some(b) = &self.bootstrap {
            s.push_str(&format!(
                "Circular block bootstrap: L = {} periods, B = {}, n = {}. Mean {} (95% CI {} to {}). \
                 Sharpe {} (95% CI {} to {}). Design effect vs iid rows {}, effective n {}. \
                 One-sided p(mean > 0) {}.\n\n",
                b.block_len,
                b.draws,
                b.n,
                f(b.mean),
                f(b.mean_ci.lo),
                f(b.mean_ci.hi),
                f(b.sharpe_annual),
                f(b.sharpe_ci_annual.lo),
                f(b.sharpe_ci_annual.hi),
                f(b.design_effect),
                f(b.effective_n),
                f(b.p_mean_positive)
            ));
        }
        if let Some(m) = &self.multiple {
            s.push_str(&format!(
                "Trials N = {} over {} candidate(s); participation ratio {}, N_eff {}; T used {}.\n\n",
                m.trials,
                m.candidates,
                f(m.participation_ratio),
                f(m.n_eff),
                f(m.t_used)
            ));
            s.push_str("| DSR at | N | SR0 | DSR | Sharpe/period | skew | kurtosis | T |\n|---|---|---|---|---|---|---|---|\n");
            for (name, d) in [("N_eff", &m.dsr_n_eff), ("10 N_eff", &m.dsr_10_n_eff)] {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    name,
                    f(d.n_trials),
                    f(d.sr0),
                    f(d.dsr),
                    f(d.sharpe),
                    f(d.skew),
                    f(d.kurtosis),
                    f(d.t)
                ));
            }
            s.push('\n');
            match (&m.pbo, &m.pbo_error) {
                (Some(p), _) => s.push_str(&format!(
                    "PBO (CSCV, S = {}, {} combinations): {}.\n\n",
                    p.blocks,
                    p.combinations,
                    f(p.pbo)
                )),
                (None, Some(e)) => s.push_str(&format!("PBO not computed: {}.\n\n", e)),
                _ => {}
            }
            s.push_str(&format!(
                "BH-FDR at q = {}: {} of {} rejected.\n\n",
                f(m.bh.q),
                m.bh.n_rejected,
                m.p_values.len()
            ));
        }
        if let Some((name, sp)) = &self.spa {
            s.push_str(&format!(
                "SPA vs {} (Hansen, block {}, B = {}): statistic {}, p consistent {} (lower {}, upper {}).\n\n",
                name,
                sp.block_len,
                sp.draws,
                f(sp.statistic),
                f(sp.p_consistent),
                f(sp.p_lower),
                f(sp.p_upper)
            ));
        }
        if let Some(n) = &self.nulls {
            s.push_str(&format!(
                "Nulls ({} draws, p = (1+count)/(B+1)): observed mean {}, Sharpe/period {}.\n\n",
                n.draws,
                f(n.observed_mean),
                f(n.observed_sharpe)
            ));
            s.push_str("| null | p (mean) | p (Sharpe) | null mean q95 |\n|---|---|---|---|\n");
            for (name, r) in [
                ("hour-block permutation", &n.block_permutation),
                ("matched-exposure random long/flat", &n.matched_exposure_random),
            ] {
                s.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    name,
                    f(r.p_mean),
                    f(r.p_sharpe),
                    f(r.null_mean_q95)
                ));
            }
            s.push_str(&format!(
                "\nAlways-long: mean {}, Sharpe/period {}.\n",
                f(n.always_long_mean),
                f(n.always_long_sharpe)
            ));
        }
        s
    }
}
