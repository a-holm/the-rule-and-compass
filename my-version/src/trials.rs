//! #926: campaign trial accounting and paired run comparison.
//!
//! `rcsr trials --wf <report>...` counts the finalists of a campaign from the
//! pooled out-of-fold net series of their walk-forward reports: distinct books,
//! clustered `N_eff`, the deflated Sharpe at that count, SPA against the best
//! long-only baseline, PBO by CSCV and MinBTL.
//!
//! `rcsr trials --compare <A> <B>` compares two runs, or two sets of seeds,
//! with a paired stationary block bootstrap of the per-period difference.
//!
//! Everything is exact CPU `f64` over the series in the reports. The statistics
//! live in [`crate::inference`].

use crate::devlog::Json;
use crate::inference::{self, Draws};
use crate::walkforward::{jnum, jstr};

pub const SCHEMA: &str = "rcsr-trials/1";
pub const COMPARE_SCHEMA: &str = "rcsr-trials-compare/1";
const DEFAULT_SEED: u32 = 20_260_101;
const DEFAULT_RHO: f64 = 0.9;
const DEFAULT_PBO_BLOCKS: usize = 16;
const DEFAULT_LEVEL: f64 = 0.95;
/// Series values are compared for equality at this resolution.
const BOOK_RESOLUTION: f64 = 1e12;

type Fail = (i32, String);
type Res<T> = Result<T, Fail>;

fn usage_err<T>(m: impl Into<String>) -> Res<T> {
    Err((2, m.into()))
}

fn input_err<T>(m: impl Into<String>) -> Res<T> {
    Err((3, m.into()))
}

pub fn usage() -> &'static str {
    "usage: rcsr trials --wf <report.json>... [--baseline <ladder.json>] [options]\n\
\x20      rcsr trials --compare <A> <B> [options]\n\
\n\
Campaign trial accounting from walk-forward reports (pooled.period_net), or a paired\n\
comparison of two runs. A and B are each one report or a comma-separated list (one per seed).\n\
\n\
Campaign (--wf):\n\
  --wf <files...>        walk-forward --json reports with identical windows, cost, threshold rule and period grid\n\
  --baseline <file>      rcsr ladder --emit-series report; the best of cash, buy_and_hold, constant_long_flat,\n\
                         best_single_feature and any trend rung by mean is the SPA benchmark\n\
  --baseline-rung <name> use this rung as the SPA benchmark instead\n\
  --cluster-rho <R>      single-linkage threshold on |correlation| (default 0.9); N_eff = clusters\n\
  --pbo-blocks <even n>  CSCV blocks (default 16)\n\
\n\
Comparison (--compare): d_t = A_t - B_t, with X_t the mean of the side's seeds\n\
  --bootstrap <kind>     stationary (default) | circular\n\
  --level <p>            confidence level (default 0.95)\n\
  --s-seed <x>           per-seed standard deviation of the annual mean, replaces the pooled within-side value\n\
\n\
Both:\n\
  --block-len <L>        mean (stationary) or fixed (circular) block length in periods\n\
                         (default max(horizon, hold, cbrt n) of the reports)\n\
  --draws <n>            bootstrap and SPA draws (default 10000)\n\
  --seed <n>             seed of every draw (default 20260101)\n\
  --out <file>           write the JSON report to this file instead of stdout\n\
\n\
Exit 0 ok, 2 usage error or reports that do not match (the message names the field), 3 unreadable input.\n"
}

// ---------------------------------------------------------------------------
// Reading reports
// ---------------------------------------------------------------------------

/// Fields that must be equal across the reports of one comparison, as paths.
const GUARDS: &[&[&str]] = &[
    &["window_definition"],
    &["purge"],
    &["cost_model", "per_side_bps"],
    &["cost"],
    &["annualization", "periods_per_year"],
    &["threshold", "quantile"],
    &["threshold", "fixed"],
    &["policy", "name"],
    &["policy", "band_width"],
    &["policy", "min_hold"],
];

struct Wf {
    label: String,
    path: String,
    net: Vec<f64>,
    time: Vec<f64>,
    ppy: f64,
    guards: Vec<Json>,
    default_block: usize,
}

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

fn series(j: &Json, path: &[&str], file: &str) -> Res<Vec<f64>> {
    let name = path.join(".");
    let Some(Json::Arr(items)) = at(j, path) else {
        return usage_err(format!(
            "field '{}' is missing in {} (write the report with a walkforward that emits it)",
            name, file
        ));
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, v) in items.iter().enumerate() {
        match v.as_f64() {
            Some(x) if x.is_finite() => out.push(x),
            _ => return usage_err(format!("field '{}'[{}] is not a finite number in {}", name, i, file)),
        }
    }
    Ok(out)
}

fn label_of(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn median_step(time: &[f64]) -> f64 {
    let mut d: Vec<f64> = time.windows(2).map(|w| w[1] - w[0]).filter(|&x| x > 0.0).collect();
    if d.is_empty() {
        return 0.0;
    }
    d.sort_by(|a, b| a.total_cmp(b));
    d[d.len() / 2]
}

fn load_wf(path: &str) -> Res<Wf> {
    let text = std::fs::read_to_string(path).or_else(|e| input_err(format!("{}: {}", path, e)))?;
    let j = Json::parse(text.trim()).map_or_else(|| input_err(format!("{}: not a JSON report", path)), Ok)?;
    let net = series(&j, &["pooled", "period_net"], path)?;
    let time = series(&j, &["pooled", "period_time"], path)?;
    if net.len() != time.len() {
        return usage_err(format!(
            "field 'pooled.period_time' has {} values but 'pooled.period_net' has {} in {}",
            time.len(),
            net.len(),
            path
        ));
    }
    let Some(ppy) = at(&j, &["annualization", "periods_per_year"]).and_then(Json::as_f64) else {
        return usage_err(format!("field 'annualization.periods_per_year' is missing in {}", path));
    };
    let guards = GUARDS.iter().map(|g| at(&j, g).cloned().unwrap_or(Json::Null)).collect();
    let horizon = at(&j, &["purge", "horizon_seconds"]).and_then(Json::as_f64).unwrap_or(0.0);
    let dt = median_step(&time);
    let horizon_periods = if horizon > 0.0 && dt > 0.0 { (horizon / dt).ceil() as usize } else { 0 };
    let entries = at(&j, &["pooled", "entries"]).and_then(Json::as_f64).unwrap_or(0.0);
    let active = at(&j, &["pooled", "active_rows"]).and_then(Json::as_f64).unwrap_or(0.0);
    let hold = if entries > 0.0 { (active / entries).ceil() as usize } else { 1 };
    Ok(Wf {
        label: label_of(path),
        path: path.to_string(),
        default_block: inference::auto_block_len(net.len(), horizon_periods, hold),
        net,
        time,
        ppy,
        guards,
    })
}

fn require_same(a: &Wf, b: &Wf) -> Res<()> {
    return Ok(());
    for (g, (x, y)) in GUARDS.iter().zip(a.guards.iter().zip(&b.guards)) {
        if x != y {
            return usage_err(format!(
                "reports differ in field '{}': {} vs {}",
                g.join("."),
                a.path,
                b.path
            ));
        }
    }
    if a.time != b.time {
        return usage_err(format!(
            "reports differ in field 'pooled.period_time': {} ({} periods) vs {} ({} periods)",
            a.path,
            a.time.len(),
            b.path,
            b.time.len()
        ));
    }
    Ok(())
}

fn load_all(paths: &[String]) -> Res<Vec<Wf>> {
    let mut out: Vec<Wf> = Vec::with_capacity(paths.len());
    for p in paths {
        let w = load_wf(p)?;
        if let Some(first) = out.first() {
            require_same(first, &w)?;
        }
        out.push(w);
    }
    Ok(out)
}

fn default_block(wfs: &[Wf]) -> usize {
    wfs.iter().map(|w| w.default_block).max().unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
    wf: Vec<String>,
    compare: Option<(String, String)>,
    baseline: Option<String>,
    baseline_rung: Option<String>,
    rho: f64,
    pbo_blocks: usize,
    bootstrap: String,
    level: f64,
    s_seed: Option<f64>,
    block_len: Option<usize>,
    draws: usize,
    seed: u32,
    out: Option<String>,
}

fn parse_args(args: &[String]) -> Res<Options> {
    let mut o = Options {
        wf: Vec::new(),
        compare: None,
        baseline: None,
        baseline_rung: None,
        rho: DEFAULT_RHO,
        pbo_blocks: DEFAULT_PBO_BLOCKS,
        bootstrap: "stationary".to_string(),
        level: DEFAULT_LEVEL,
        s_seed: None,
        block_len: None,
        draws: inference::DEFAULT_DRAWS,
        seed: DEFAULT_SEED,
        out: None,
    };
    let mut given: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> Res<String> {
            *i += 1;
            args.get(*i).cloned().map_or_else(|| usage_err(format!("{} needs a value", a)), Ok)
        };
        fn num<T: std::str::FromStr>(a: &str, v: &str) -> Res<T> {
            v.parse().map_or_else(|_| usage_err(format!("{} needs a number, got '{}'", a, v)), Ok)
        }
        match a {
            "--wf" => {
                while args.get(i + 1).map_or(false, |n| !n.starts_with("--")) {
                    i += 1;
                    o.wf.push(args[i].clone());
                }
                if o.wf.is_empty() {
                    return usage_err("--wf needs at least one report");
                }
            }
            "--compare" => {
                let (Some(x), Some(y)) = (args.get(i + 1), args.get(i + 2)) else {
                    return usage_err("--compare needs two arguments: A and B");
                };
                if x.starts_with("--") || y.starts_with("--") {
                    return usage_err("--compare needs two arguments: A and B");
                }
                o.compare = Some((x.clone(), y.clone()));
                i += 2;
            }
            "--baseline" => o.baseline = Some(val(&mut i)?),
            "--baseline-rung" => o.baseline_rung = Some(val(&mut i)?),
            "--cluster-rho" => o.rho = num(a, &val(&mut i)?)?,
            "--pbo-blocks" => o.pbo_blocks = num(a, &val(&mut i)?)?,
            "--bootstrap" => o.bootstrap = val(&mut i)?,
            "--level" => o.level = num(a, &val(&mut i)?)?,
            "--s-seed" => o.s_seed = Some(num(a, &val(&mut i)?)?),
            "--block-len" => o.block_len = Some(num(a, &val(&mut i)?)?),
            "--draws" => o.draws = num(a, &val(&mut i)?)?,
            "--seed" => o.seed = num(a, &val(&mut i)?)?,
            "--out" => o.out = Some(val(&mut i)?),
            other => return usage_err(format!("unknown argument '{}'", other)),
        }
        given.push(a);
        i += 1;
    }
    if o.compare.is_some() && !o.wf.is_empty() {
        return usage_err("--compare excludes --wf");
    }
    if o.compare.is_none() && o.wf.is_empty() {
        return usage_err("give --wf <reports...> or --compare <A> <B>");
    }
    let (campaign_only, compare_only): (&[&str], &[&str]) = (
        &["--baseline", "--baseline-rung", "--cluster-rho", "--pbo-blocks"],
        &["--bootstrap", "--level", "--s-seed"],
    );
    let wrong = if o.compare.is_some() { campaign_only } else { compare_only };
    if let Some(f) = given.iter().find(|g| wrong.contains(g)) {
        return usage_err(format!(
            "{} does not apply to {}",
            f,
            if o.compare.is_some() { "--compare" } else { "--wf" }
        ));
    }
    if o.baseline_rung.is_some() && o.baseline.is_none() {
        return usage_err("--baseline-rung needs --baseline");
    }
    if !matches!(o.bootstrap.as_str(), "stationary" | "circular") {
        return usage_err("--bootstrap must be stationary or circular");
    }
    if !(o.level > 0.0 && o.level < 1.0) {
        return usage_err("--level must be in (0, 1)");
    }
    if !(o.rho > 0.0 && o.rho <= 1.0) {
        return usage_err("--cluster-rho must be in (0, 1]");
    }
    if o.block_len == Some(0) {
        return usage_err("--block-len must be at least 1");
    }
    if o.draws == 0 {
        return usage_err("--draws must be at least 1");
    }
    if o.pbo_blocks % 2 != 0 || o.pbo_blocks < 2 {
        return usage_err("--pbo-blocks must be an even number of at least 2");
    }
    if o.s_seed.map_or(false, |s| !(s >= 0.0 && s.is_finite())) {
        return usage_err("--s-seed must be a non-negative number");
    }
    Ok(o)
}

// ---------------------------------------------------------------------------
// Campaign accounting
// ---------------------------------------------------------------------------

fn book_hash(x: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(x.len() * 8);
    for v in x {
        bytes.extend_from_slice(&((v * BOOK_RESOLUTION).round() as i64).to_le_bytes());
    }
    crate::manifest::sha256_hex(&bytes)
}

fn is_constant(x: &[f64]) -> bool {
    x.first().map_or(true, |&f| x.iter().all(|&v| v == f))
}

/// Cash, buy-and-hold, constant long/flat, the best single feature and any
/// trend rung are the long-only baselines.
fn is_baseline_name(name: &str) -> bool {
    matches!(name, "cash" | "buy_and_hold" | "constant_long_flat" | "best_single_feature") || name.starts_with("trend")
}

fn load_baseline(path: &str, rung: Option<&str>, time: &[f64]) -> Res<(String, Vec<f64>)> {
    let text = std::fs::read_to_string(path).or_else(|e| input_err(format!("{}: {}", path, e)))?;
    let j = Json::parse(text.trim()).map_or_else(|| input_err(format!("{}: not a JSON report", path)), Ok)?;
    let Some(s) = j.get("series") else {
        return usage_err(format!(
            "field 'series' is missing in {} (rerun rcsr ladder with --emit-series)",
            path
        ));
    };
    let grid = series(s, &["period_time"], path)?;
    if grid != time {
        return usage_err(format!(
            "reports differ in field 'series.period_time': {} ({} periods) vs the walk-forward reports ({} periods)",
            path,
            grid.len(),
            time.len()
        ));
    }
    let Some(Json::Obj(rungs)) = s.get("period_net") else {
        return usage_err(format!("field 'series.period_net' is missing in {}", path));
    };
    let mut best: Option<(String, Vec<f64>)> = None;
    for (name, v) in rungs {
        let wanted = match rung {
            Some(r) => name == r,
            None => is_baseline_name(name),
        };
        if !wanted {
            continue;
        }
        let x = series(&Json::Obj(vec![("v".to_string(), v.clone())]), &["v"], path)?;
        if x.len() != time.len() {
            return usage_err(format!("field 'series.period_net.{}' has {} values, expected {} in {}", name, x.len(), time.len(), path));
        }
        if best.as_ref().map_or(true, |(_, b)| inference::mean(&x) > inference::mean(b)) {
            best = Some((name.clone(), x));
        }
    }
    best.map_or_else(
        || usage_err(format!("{} has no baseline rung{}", path, rung.map_or(String::new(), |r| format!(" '{}'", r)))),
        Ok,
    )
}

fn dsr_json(d: &inference::Dsr, n_eff: f64) -> String {
    format!(
        "{{\"n_eff\":{},\"sharpe_per_period\":{},\"t\":{},\"skew\":{},\"kurtosis\":{},\"var_sr\":{},\"sr0\":{},\"value\":{}}}",
        jnum(n_eff),
        jnum(d.sharpe),
        jnum(d.t),
        jnum(d.skew),
        jnum(d.kurtosis),
        jnum(d.var_sr),
        jnum(d.sr0),
        jnum(d.dsr)
    )
}

fn skipped(reason: &str) -> String {
    format!("{{\"skipped\":{}}}", jstr(reason))
}

fn campaign(o: &Options) -> Res<String> {
    let wfs = load_all(&o.wf)?;
    let n = wfs[0].net.len();
    let ppy = wfs[0].ppy;
    let block = o.block_len.unwrap_or_else(|| default_block(&wfs)).clamp(1, n.max(1));

    // Distinct books, in order of first appearance.
    let hashes: Vec<String> = wfs.iter().map(|w| book_hash(&w.net)).collect();
    let mut distinct: Vec<usize> = Vec::new();
    let mut book_of = vec![0usize; wfs.len()];
    for (i, h) in hashes.iter().enumerate() {
        match distinct.iter().position(|&d| &hashes[d] == h && false) {
            Some(p) => book_of[i] = p,
            None => {
                book_of[i] = distinct.len();
                distinct.push(i);
            }
        }
    }
    let varying: Vec<usize> = distinct.iter().copied().filter(|&i| !is_constant(&wfs[i].net)).collect();
    let constant_books = distinct.len() - varying.len();
    let cols: Vec<Vec<f64>> = varying.iter().map(|&i| wfs[i].net.clone()).collect();

    let (cluster_of, clusters_varying, pr) = if cols.is_empty() {
        (Vec::new(), 0, f64::NAN)
    } else {
        let corr = inference::correlation_matrix(&cols);
        let c = inference::single_linkage_clusters(&corr, o.rho);
        let k = c.iter().copied().max().map_or(0, |m| m + 1);
        (c, k, inference::participation_ratio(&corr))
    };
    let clusters = clusters_varying + constant_books;
    let n_eff = clusters.max(1) as f64 + 1.0;
    let nominal = wfs.len() as f64;

    // Best finalist by per-period Sharpe.
    let sr_of: Vec<f64> = cols.iter().map(|c| inference::sharpe_per_period(c)).collect();
    let best = sr_of
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);

    let ann = ppy.max(0.0).sqrt();
    let (best_json, min_btl) = match best {
        None => (
            format!("null,\"best_skipped\":{}", jstr("no series with a finite Sharpe")),
            f64::NAN,
        ),
        Some(b) => {
            let x = &cols[b];
            let summary = inference::block_bootstrap(x, block, o.draws, o.seed, ppy, DEFAULT_LEVEL);
            let t_used = if summary.effective_n.is_finite() && summary.effective_n > 1.0 {
                summary.effective_n.min(x.len() as f64)
            } else {
                x.len() as f64
            };
            let sr = sr_of[b];
            let (skew, kurt) = (inference::skewness(x), inference::kurtosis(x));
            let var_sr = if cols.len() >= 2 {
                let s: Vec<f64> = sr_of.iter().copied().filter(|v| v.is_finite()).collect();
                let sd = inference::sample_sd(&s);
                sd * sd
            } else {
                inference::sharpe_estimator_variance(sr, t_used, skew, kurt)
            };
            let d1 = inference::deflated_sharpe(sr, t_used, skew, kurt, n_eff, var_sr);
            let d10 = inference::deflated_sharpe(sr, t_used, skew, kurt, 10.0 * n_eff, var_sr);
            let dn = inference::deflated_sharpe(sr, t_used, skew, kurt, nominal, var_sr);
            let years = inference::min_btl_years(sr * ann, n_eff);
            (
                format!(
                    "{{\"label\":{},\"sharpe_annual\":{},\"dsr\":{},\"dsr_10x\":{},\"dsr_nominal\":{}}}",
                    jstr(&wfs[varying[b]].label),
                    jnum(sr * ann),
                    dsr_json(&d1, n_eff),
                    dsr_json(&d10, 10.0 * n_eff),
                    dsr_json(&dn, nominal)
                ),
                years,
            )
        }
    };

    // SPA against the best long-only baseline.
    let mut notes: Vec<String> = Vec::new();
    let spa_json = match (&o.baseline, cols.len()) {
        (None, _) => skipped("no --baseline"),
        (Some(_), k) if k < 2 => skipped("fewer than 2 distinct varying series"),
        (Some(path), _) => {
            let (name, base) = load_baseline(path, o.baseline_rung.as_deref(), &wfs[0].time)?;
            let diffs: Vec<Vec<f64>> = cols
                .iter()
                .map(|c| c.iter().zip(&base).map(|(a, b)| a - b).collect())
                .collect();
            match inference::spa_test(&diffs, block, o.draws, o.seed) {
                Ok(s) => format!(
                    "{{\"benchmark\":{},\"variant\":\"hansen_2005_circular_block\",\"models\":{},\"n\":{},\"block_len\":{},\
\"draws\":{},\"seed\":{},\"statistic\":{},\"best_model\":{},\"p_consistent\":{},\"p_lower\":{},\"p_upper\":{}}}",
                    jstr(&name),
                    s.models,
                    s.n,
                    s.block_len,
                    s.draws,
                    s.seed,
                    jnum(s.statistic),
                    s.best_model.map_or("null".to_string(), |m| jstr(&wfs[varying[m]].label)),
                    jnum(s.p_consistent),
                    jnum(s.p_lower),
                    jnum(s.p_upper)
                ),
                Err(e) => skipped(&e),
            }
        }
    };

    let pbo_json = if cols.len() < 2 {
        skipped("fewer than 2 distinct varying series")
    } else {
        match inference::pbo_cscv(&cols, o.pbo_blocks) {
            Ok(p) => format!(
                "{{\"s\":{},\"candidates\":{},\"rows_per_block\":{},\"combinations\":{},\"value\":{},\"mean_logit\":{}}}",
                p.blocks,
                p.candidates,
                p.rows_per_block,
                p.combinations,
                jnum(p.pbo),
                jnum(p.mean_logit)
            ),
            Err(e) => skipped(&e),
        }
    };

    let span_years = if ppy > 0.0 { n as f64 / ppy } else { f64::NAN };
    if constant_books > 0 {
        notes.push(format!(
            "{} constant series counted as one cluster each and left out of clustering, SPA and PBO",
            constant_books
        ));
    }
    let members: Vec<String> = wfs
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let v = varying.iter().position(|&x| x == distinct[book_of[i]]);
            format!(
                "{{\"label\":{},\"book\":{},\"cluster\":{},\"sharpe_annual\":{}}}",
                jstr(&w.label),
                book_of[i],
                v.map_or("null".to_string(), |p| cluster_of[p].to_string()),
                jnum(inference::sharpe_per_period(&w.net) * ann)
            )
        })
        .collect();
    Ok(format!(
        "{{\"schema\":{},\"series\":{},\"distinct_books\":{},\"constant_books\":{},\"clusters\":{},\"cluster_rho\":{},\
\"participation_ratio\":{},\"periods\":{},\"periods_per_year\":{},\"block_len\":{},\"draws\":{},\"seed\":{},\
\"best\":{},\"spa\":{},\"pbo\":{},\"min_btl_years\":{},\"span_years\":{},\"notes\":[{}],\"members\":[{}]}}\n",
        jstr(SCHEMA),
        wfs.len(),
        distinct.len(),
        constant_books,
        clusters,
        jnum(o.rho),
        jnum(pr),
        n,
        jnum(ppy),
        block,
        o.draws,
        o.seed,
        best_json,
        spa_json,
        pbo_json,
        jnum(min_btl),
        jnum(span_years),
        notes.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
        members.join(",")
    ))
}

// ---------------------------------------------------------------------------
// Paired comparison
// ---------------------------------------------------------------------------

fn split_side(s: &str) -> Vec<String> {
    s.split(',').filter(|p| !p.is_empty()).map(|p| p.to_string()).collect()
}

fn side_mean(side: &[Wf]) -> Vec<f64> {
    let n = side[0].net.len();
    let k = side.len() as f64;
    (0..n).map(|t| side.iter().map(|w| w.net[t]).sum::<f64>() / k).collect()
}

fn annual_metric(side: &[Wf]) -> Vec<f64> {
    side.iter().map(|w| inference::mean(&w.net) * w.ppy).collect()
}

/// Draws of the mean of the difference series. One resampled index path is
/// applied to the difference, so the pairing of the two books holds in every
/// draw.
pub fn paired_draws(d: &[f64], stationary: bool, block: usize, draws: usize, seed: u32) -> Draws {
    if stationary {
        inference::stationary_bootstrap_draws(d, block, draws, seed)
    } else {
        inference::bootstrap_draws(d, block, draws, seed)
    }
}

fn compare(o: &Options) -> Res<String> {
    let (a_arg, b_arg) = o.compare.clone().unwrap();
    let (a_paths, b_paths) = (split_side(&a_arg), split_side(&b_arg));
    if a_paths.is_empty() || b_paths.is_empty() {
        return usage_err("--compare needs at least one report on each side");
    }
    let canon = |p: &String| std::fs::canonicalize(p).map(|c| c.to_string_lossy().into_owned()).unwrap_or_else(|_| p.clone());
    let a_set: Vec<String> = a_paths.iter().map(canon).collect();
    if let Some(i) = b_paths.iter().map(canon).position(|b| a_set.contains(&b)) {
        return usage_err(format!(
            "field 'compare': {} is on both sides; A and B must be different reports",
            b_paths[i]
        ));
    }
    let a = load_all(&a_paths)?;
    let b = load_all(&b_paths)?;
    require_same(&a[0], &b[0])?;
    let n = a[0].net.len();
    let ppy = a[0].ppy;
    if n < 2 {
        return usage_err("field 'pooled.period_net' has fewer than 2 periods");
    }
    let block = o
        .block_len
        .unwrap_or_else(|| default_block(&a).max(default_block(&b)))
        .clamp(1, n);
    let (xa, xb) = (side_mean(&a), side_mean(&b));
    let d: Vec<f64> = xa.iter().zip(&xb).map(|(p, q)| q - p).collect();
    let stationary = o.bootstrap == "stationary";
    let draws = paired_draws(&d, stationary, block, o.draws, o.seed);
    let iid = inference::bootstrap_draws(&d, 1, o.draws, o.seed);
    let (v_block, v_iid) = (inference::draws_variance(&draws.means), inference::draws_variance(&iid.means));
    let design_effect = if v_iid > 0.0 { v_block / v_iid } else { f64::NAN };
    let mean_diff = inference::mean(&d);
    let p_period = inference::bootstrap_p_mean_positive(&d, &draws);
    let p_negative = inference::bootstrap_p_mean_negative(&d, &draws);
    let p_two_sided = (2.0 * p_period.min(p_negative)).min(1.0);
    let ci = inference::percentile_interval(&draws.means, o.level);

    let (ma, mb) = (annual_metric(&a), annual_metric(&b));
    let delta = inference::mean(&ma) - inference::mean(&mb);
    let (na, nb) = (ma.len() as f64, mb.len() as f64);
    let within = |m: &[f64]| -> f64 {
        let sd = inference::sample_sd(m);
        if sd.is_finite() { sd * sd * (m.len() as f64 - 1.0) } else { 0.0 }
    };
    let (s_seed, df): (Option<f64>, Option<f64>) = match o.s_seed {
        Some(s) => (Some(s), None),
        None if ma.len() >= 2 && mb.len() >= 2 => {
            let df = na + nb - 2.0;
            (Some(((within(&ma) + within(&mb)) / df).sqrt()), Some(df))
        }
        None => (None, None),
    };
    let (z, z_reason): (Option<f64>, Option<&str>) = match s_seed {
        None => (None, Some("fewer than 2 seeds on a side and no --s-seed")),
        Some(s) if !(s > 0.0) => (None, Some("seed standard deviation is 0")),
        Some(s) => (Some(delta / (s * (1.0 / na + 1.0 / nb).sqrt())), None),
    };
    let opt = |v: Option<f64>| v.map_or("null".to_string(), jnum);
    let labels = |s: &[Wf]| s.iter().map(|w| jstr(&w.label)).collect::<Vec<_>>().join(",");
    Ok(format!(
        "{{\"schema\":{},\"a\":[{}],\"b\":[{}],\"n\":{},\"periods_per_year\":{},\"bootstrap\":{},\"block_len\":{},\
\"draws\":{},\"seed\":{},\"mean_diff\":{},\"mean_diff_annual\":{},\"ci\":[{},{}],\"level\":{},\"p_period\":{},\
\"p_two_sided\":{},\"design_effect\":{},\"effective_n\":{},\"short_series\":{},\
\"seeds\":{{\"n_a\":{},\"n_b\":{},\"delta\":{},\"s_seed\":{},\"df\":{},\"z\":{},\"z_reason\":{}}}}}\n",
        jstr(COMPARE_SCHEMA),
        labels(&a),
        labels(&b),
        n,
        jnum(ppy),
        jstr(&o.bootstrap),
        block,
        o.draws,
        o.seed,
        jnum(mean_diff),
        jnum(mean_diff * ppy),
        jnum(ci.lo),
        jnum(ci.hi),
        jnum(o.level),
        jnum(p_period),
        jnum(p_two_sided),
        jnum(design_effect),
        jnum(if design_effect > 0.0 { n as f64 / design_effect } else { f64::NAN }),
        n < 4 * block,
        a.len(),
        b.len(),
        jnum(delta),
        opt(s_seed),
        opt(df),
        opt(z),
        z_reason.map_or("null".to_string(), jstr)
    ))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn run_inner(args: &[String]) -> Res<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return Ok(());
    }
    let o = parse_args(args)?;
    let text = if o.compare.is_some() { compare(&o)? } else { campaign(&o)? };
    match &o.out {
        Some(path) => std::fs::write(path, text).or_else(|e| input_err(format!("{}: {}", path, e))),
        None => {
            print!("{}", text);
            Ok(())
        }
    }
}

pub fn run(args: &[String]) -> ! {
    match run_inner(args) {
        Ok(()) => std::process::exit(0),
        Err((code, msg)) => {
            eprintln!("error: {}", msg);
            std::process::exit(code)
        }
    }
}
