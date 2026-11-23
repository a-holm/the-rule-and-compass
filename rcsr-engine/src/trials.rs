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
\x20      rcsr trials --cpcv-assemble <report.json>... --splits <csv> --out-dir <dir> [--summary <file>]\n\
\x20      rcsr trials --wf <report.json>... --table [--rule <rule.json>] [options]\n\
\n\
Campaign trial accounting from walk-forward reports (pooled.period_net), or a paired\n\
comparison of two runs. A and B are each one report or a comma-separated list (one per seed).\n\
\n\
CPCV paths (--cpcv-assemble, #1174): the walk-forward reports of one configuration, one per split\n\
of a `rcsr protocol cpcv` table (--splits), each priced with --pool-reset-flat (and --period-grid target, so\n\
the splits of a group share its period grid whatever NaN coverage or gate their formulas have). A report belongs to\n\
the split with its test intervals. Path j of group g takes the j-th split, in lexicographic order of\n\
the test groups, that tests g (López de Prado 2018, ch. 12); every group of a path is one book\n\
segment that starts flat and pays its final exit. Writes <dir>/path_<j>.json in the walk-forward\n\
schema (pooled.period_time, pooled.period_net, window_definition = the CPCV scheme hash, so\n\
--wf reads them) and the summary (rcsr-cpcv-summary/1: per-path net, Sharpe and max drawdown,\n\
the path Sharpe distribution, the share of positive paths, per-group net) to --summary or stdout.\n\
A missing or extra split report, a report of no split, reset_flat not true, or reports that differ\n\
in cost, purge, annualization, threshold rule, policy, period grid rule (pooled.grid.rule), missing-score rule\n\
(book.missing_score) or a group's\n\
period grid exit 2 naming the split or field. Target-grid reports add pooled.grid and pooled.period_rows\n\
to every path and the empty periods per path and per group to the summary. Each of pooled.period_gross,\n\
period_exposure, period_entries, period_bh, period_cost, period_entry_stamp and period_bh_net that the reports\n\
carry (walkforward --period-series or --emit-series, #1236) goes into every path as the concatenation of the\n\
chosen group series, and pooled.entry_times as the sum of the chosen groups' windows[].entry_times, so --table\n\
gates paths on net at m x cost, exposure, entry_times and alpha_bh. A field on some reports only, a series of\n\
another length than period_net or with a non-finite value, pooled.entry_times without windows[].entry_times, or\n\
window counts that do not sum to pooled.entry_times exit 2 naming the field and the report.\n\
\n\
Campaign (--wf):\n\
  --wf <files...>        walk-forward --json reports with identical windows, cost, threshold rule (threshold.quantile, threshold.fixed,\n\
                         threshold.fit, threshold.cut_min_entries, threshold.cut_blocks, threshold.cut_lambda), policy\n\
                         and period grid (pooled.grid.rule, absent for the score grid, and pooled.period_time) and missing-score\n\
                         rule (book.missing_score, absent in a report of a build that carried the position); a report\n\
                         with null_store (priced on a null store) exits 2 naming its label and seed (#1304)\n\
  --members <mode>       selected (default): one series per report, pooled.period_net; all: one series per\n\
                         members[] row of a walkforward --front --all-members report (members[].series), named\n\
                         <report>#<archive_rank>; a report without members exits 2\n\
  --baseline <file>      rcsr ladder --emit-series report; the best of cash, buy_and_hold, constant_long_flat,\n\
                         best_single_feature and any trend rung by mean is the SPA benchmark\n\
  --baseline-rung <name> use this rung as the SPA benchmark instead\n\
  --baseline-wf <files...> with --table: walk-forward or CPCV path reports (pooled.period_net) as the SPA benchmark rungs;\n\
                         excludes --baseline. Rungs: cash (zeros), buy_and_hold (pooled.period_bh_net of the first\n\
                         report), one per report named by its label or file stem; the best by mean is the benchmark\n\
  --cluster-rho <R>      single-linkage threshold on |correlation| (default 0.9); N_eff = clusters\n\
  --pbo-blocks <even n>  CSCV blocks (default 16)\n\
  --regimes <calendar>   per-stratum metrics of every series in its member entry. Calendar rows label,start,end\n\
                         (UTC half-open, no overlap, any label set; `rcsr protocol regimes` writes one) are tested\n\
                         against the period stamps; periods outside every row are `undated`. Each stratum reports\n\
                         periods, years, net, Sharpe, Sortino, drawdown and the block-bootstrap interval of the mean\n\
                         (--draws, --seed, --block-len). With pooled.period_gross, period_exposure, period_entries and\n\
                         period_bh in the reports (walkforward --period-series) it also reports net at twice the cost,\n\
                         exposure, entries, buy-and-hold and the alpha over exposure-matched buy-and-hold\n\
\n\
Candidate table (--table, #1177): one row per --wf series, with the campaign options above. Rows pass the gates,\n\
the Pareto front ranks the passers, the average rank orders them inside a front, and a walk in the order front,\n\
average rank, complexity, trades, input order picks rows below the diversity cap. Every picked row gets PSR, DSR\n\
at N_eff, 10 N_eff and the nominal N (every row of the table, plus --prior-trials), MinBTL, the Harvey-Liu haircut\n\
(Bonferroni, Holm, BHY over the rows) and its StepM and BH-FDR verdicts; SPA, StepM, BH-FDR and PBO run over the\n\
distinct varying rows against the --baseline or --baseline-wf best rung, else cash at 0. With --family-table the deflation also\n\
counts the sibling horizon tables. Output rcsr-trials-table/1.\n\
  --rule <file>          rcsr-trials-rule/1 JSON with cost_multiples, gates, relaxed_gates (apply when no row passes\n\
                         gates), pareto, rank, pick, diversity_rho, min_front, max_composites and freeze; its sha256 is\n\
                         in the output; it excludes --gates, --pareto, --rank, --pick, --diversity-rho, --min-front,\n\
                         --cost-multiples. max_composites K (#1237, D8): the walk skips a composite row (a report with\n\
                         the composite marker of an rcsr compose run: selection.composite, or composite of a CPCV path)\n\
                         once the picks hold K composites (skipped.reason max_composites). freeze (#1237):\n\
                         {\"cpcv\":[N,k],\"cost_bps\":B,\"s4\":{\"window_days\":W,\"percentile\":P},\"p1\":{...},\n\
                         \"drawdown_cap\":{\"window_days\":W,\"percentile\":P,\"floor\":F}} freezes per pick the sealed-read\n\
                         thresholds over every CPCV(N,k) path of the pick's configuration (needs --freeze-map)\n\
  --freeze-map <file>    rcsr-trials-freeze-map/1 JSON {\"rows\":[{\"row\",\"configuration\",\"paths\",\"run\"}]}: per\n\
                         row (its report as given to --wf) the configuration name, its CPCV path reports and the\n\
                         run.json to seal (optional). Every pick gets freeze: s4_floor and p1_floor (nearest-rank\n\
                         percentile, k = (P n + 99) / 100, of the annualised mean nets at B bp of the windows\n\
                         [t, t + W days) that start at each period and end inside one path, pooled over the paths),\n\
                         drawdown_cap (max(F, the percentile of the windows' compounding max drawdown magnitudes)),\n\
                         window counts and ranks, the path files with sha256 and run_sha256. A pick without a map\n\
                         entry, a missing or doubled cpcv.path, paths of two splits tables, a split with two\n\
                         report_sha256, cpcv.groups or cpcv.test_groups other than freeze.cpcv, a locked period, a\n\
                         path priced at another cost without pooled.period_cost, or no window that fits exit 2\n\
  --gates <file>         JSON object {metric: [op, value]}, op >, >=, <, <=, or [\"between\", lo, hi]; all must hold;\n\
                         a NaN value fails; every failed gate is listed per row\n\
  --pareto <m[:dir],...> Pareto metrics over the gate passers, dir max (default) or min; NaN is the worst value\n\
  --rank <m[:dir],...>   average rank over the gate passers (1 = best, ties share the mean place) and the Friedman\n\
                         chi2 over rows x metrics\n\
  --pick <K>             pick K rows (default 0: none)\n\
  --diversity-rho <R>    skip a row whose net series correlates at R or more with a picked row (default 1)\n\
  --min-front <M>        keep the first fronts until at least M rows remain (default 0: every front)\n\
  --cost-multiples <m,..> net<bps> metrics at m x cost_model.per_side_bps (default 1,2: net10 and net20 at 10 bp)\n\
  --null-wf <files...>   null walk-forward reports; null_q = share of their annual Sharpes below the row's. A report\n\
                         priced on a null store (null_store, rcsr panel permute-labels, #1272) is read only here: as\n\
                         --wf (with or without --table, also with --members all), in --compare, as a --baseline-wf\n\
                         rung or as a freeze population path it exits 2 (#1304)\n\
  --prior-trials <N0>    add N0 earlier trials to the nominal N of the picked rows' DSR\n\
  --family-table <files...> rcsr trials --table outputs of the sibling horizons over the same development span\n\
                         (#1269, repeatable). DSR, DSR 10 N_eff and MinBTL use N_eff plus each sibling's own\n\
                         trials.n_eff, DSR nominal the nominal N plus --prior-trials plus each sibling's trials.nominal,\n\
                         always with this table's cross-trial Sharpe variance; spa.p_family = min(1, (1 + k)\n\
                         spa.p_consistent) for k siblings; trials.family lists each sibling's path, sha256, n_eff,\n\
                         nominal and spa_p, the family sha256 and the totals. A sibling's own trials.family is not\n\
                         read. Two siblings with one sha256, a sibling that is the --out file, a sibling that is not\n\
                         an rcsr-trials-table/1 output or lacks trials.n_eff, trials.nominal, spa.p_consistent or\n\
                         inputs.reports, a sibling with a locked read, and a sibling whose inputs.reports share a report\n\
                         with --wf or with an earlier sibling (by canonical path when the path resolves from the current\n\
                         directory, else as given) exit 2 naming the file (and the shared report and the other table)\n\
  --alpha <a>            family error rate of StepM and q of BH-FDR (default 0.05)\n\
  --locked-test <S,E>    refuse a report with a period or test window in [S, E) (repeatable)\n\
  --occam <files...>     run.json of a --select-rule occam re-selection or of rcsr compose: bits, p_occam and\n\
                         bit_budget per row (report only), matched by the report's run sha256 (selection.run_sha256, else cpcv.run_sha256, else run_sha256) and\n\
                         archive rank (a --from-run report: the run finalist); a CPCV path row gets null\n\
  --markdown <file>      also write the table as Markdown\n\
  Metrics: net<bps> (annualised mean net over cash at 0), sharpe, sortino (MAR 0), calmar (net at the report cost\n\
  over |max_drawdown|), max_drawdown, exposure, entry_times, alpha_bh (mean of net - exposure x period_bh_net,\n\
  annualised), psr, groups and groups_pos (test windows or CPCV groups with positive net); with --regimes\n\
  bull_net, bear_excess (net minus buy-and-hold net in bear), worst_stratum (lowest stratum net), net:<label>,\n\
  excess:<label>, alpha_bh:<label> and a strata block per row; null_q with --null-wf; baseline_excess with\n\
  --baseline. A rule metric a report cannot give exits 2 naming the field and the report. A report with a locked\n\
  read (protocol.test_read, protocol.lockbox.rows, evaluated or read_marker) or a period or test window inside a\n\
  locked range (protocol.lockbox, its declared and also ranges, lockbox, --locked-test) exits 2 naming it.\n\
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
  --threshold-arms       let the reports differ in the fitted-cut rule (threshold.fit, threshold.cut_min_entries,\n\
                         threshold.cut_blocks, threshold.cut_lambda) and record the rule of each report; every other\n\
                         guard stays. The fitted and the fixed arm are separate trials\n\
  --out <file>           write the JSON report to this file instead of stdout\n\
\n\
Annualisation (#1208): every report (and every --cpcv-assemble split report) must have annualization.periods_per_year within 1 % of 31536000 / annualization.period_seconds (when present) and, for a report with a period grid (pooled.grid.rule) or without period_seconds, of 31536000 / the median step of pooled.period_time (a score-grid report with period_seconds skips the step check, #1328); a report that does not, or has a non-positive value, exits 2 naming the field, the value and the implied value (rerun the walk-forward with a current build; there is no rescaling and no bypass flag). A report with fewer than 3 periods skips the step check and prints a note on stderr.\n\
Exit 0 ok, 2 usage error or reports that do not match (the message names the field), 3 unreadable input.\n"
}

// ---------------------------------------------------------------------------
// Reading reports
// ---------------------------------------------------------------------------

/// Fields that must be equal across the reports of one comparison, as paths.
pub(crate) const GUARDS: &[&[&str]] = &[
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
    GRID_RULE,
    MISSING_SCORE,
];

/// #1199: the period grid rule of `walkforward --period-grid`; a report without it is the score grid.
pub(crate) const GRID_RULE: &[&str] = &["pooled", "grid", "rule"];

/// #1203: the missing-score rule of the book. A report without it was priced by a build that carried the position
/// across a row with no finite score (`carry`), so it never mixes with a `flat` or `hold` report.
pub(crate) const MISSING_SCORE: &[&str] = &["book", "missing_score"];

/// The fitted-cut rule of `walkforward --threshold-fit` (#1094). Reports written without the flag, and before it
/// existed, read all four as `null` and match each other. `--threshold-arms` skips these paths only.
pub(crate) const FIT_GUARDS: &[&[&str]] = &[
    &["threshold", "fit"],
    &["threshold", "cut_min_entries"],
    &["threshold", "cut_blocks"],
    &["threshold", "cut_lambda"],
];

pub(crate) struct Wf {
    pub(crate) label: String,
    pub(crate) path: String,
    /// `pooled`, or `series` for a `members[]` entry (#1176).
    pub(crate) prefix: &'static str,
    pub(crate) net: Vec<f64>,
    /// #1175: the per-period series a walkforward report carries with `--period-series`.
    pub(crate) gross: Option<Vec<f64>>,
    pub(crate) exposure: Option<Vec<f64>>,
    entries: Option<Vec<f64>>,
    bh: Option<Vec<f64>>,
    pub(crate) time: Vec<f64>,
    pub(crate) ppy: f64,
    guards: Vec<Json>,
    fit_guards: Vec<Json>,
    rule: String,
    pub(crate) default_block: usize,
}

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

pub(crate) fn series(j: &Json, path: &[&str], file: &str) -> Res<Vec<f64>> {
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

pub(crate) fn label_of(path: &str) -> String {
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

/// #1208: relative tolerance between `periods_per_year` and the value its period width implies.
pub(crate) const ANNUALIZATION_TOLERANCE: f64 = 0.01;
const SECONDS_PER_YEAR: f64 = 31_536_000.0;

/// #1208, #1328: `Err((2, msg))` when `annualization.periods_per_year` contradicts `annualization.period_seconds` or the
/// median step of `time`; the median step is skipped for a score-grid report that has `period_seconds`, because
/// its `period_time` holds the scored rows only and NaN scores thin it; `Ok(Some(note))` when the series is too short to check.
pub(crate) fn check_annualization(path: &str, j: &Json, time: &[f64], ppy: f64) -> Res<Option<String>> {
    if !(ppy.is_finite() && ppy > 0.0) {
        return usage_err(format!("field 'annualization.periods_per_year' is {} in {} (a positive number is required)", jnum(ppy), path));
    }
    let off = |implied: f64| ((ppy - implied) / implied).abs() > ANNUALIZATION_TOLERANCE;
    let period_seconds = at(j, &["annualization", "period_seconds"]).and_then(Json::as_f64).filter(|s| s.is_finite() && *s > 0.0);
    if let Some(s) = period_seconds {
        let implied = SECONDS_PER_YEAR / s;
        if off(implied) {
            return usage_err(format!(
                "field 'annualization.periods_per_year' is {} but 'annualization.period_seconds' is {} s ({} periods per year) in {}",
                jnum(ppy),
                jnum(s),
                jnum(implied),
                path
            ));
        }
    }
    let period_grid = at(j, GRID_RULE).is_some_and(|r| !matches!(r, Json::Null));
    if period_seconds.is_some() && !period_grid {
        return Ok(None);
    }
    let step = median_step(time);
    if step <= 0.0 || time.windows(2).filter(|w| w[1] > w[0]).count() < 2 {
        return Ok(Some(format!("note: {}: too few periods to check annualization.periods_per_year", path)));
    }
    let implied = SECONDS_PER_YEAR / step;
    if off(implied) {
        return usage_err(format!(
            "field 'annualization.periods_per_year' is {} but the median step of 'pooled.period_time' is {} s ({} periods per year) in {}",
            jnum(ppy),
            jnum(step),
            jnum(implied),
            path
        ));
    }
    Ok(None)
}

pub(crate) fn read_report(path: &str) -> Res<Json> {
    let text = std::fs::read_to_string(path).or_else(|e| input_err(format!("{}: {}", path, e)))?;
    Json::parse(text.trim()).map_or_else(|| input_err(format!("{}: not a JSON report", path)), Ok)
}

/// #1304: a report priced on a null store is never a campaign or comparison series.
fn load_wf(path: &str) -> Res<Wf> {
    let j = read_report(path)?;
    crate::trials_table::refuse_null_store(&j, path)?;
    wf_entry(&j, &j, "pooled", path, label_of(path))
}

/// #1176: one entry per `members[]` row of a `walkforward --all-members` report,
/// labelled `<report>#<archive_rank>`, with the series of `members[].series`.
fn load_members(path: &str) -> Res<Vec<Wf>> {
    let j = read_report(path)?;
    crate::trials_table::refuse_null_store(&j, path)?;
    let Some(Json::Arr(members)) = j.get("members") else {
        return usage_err(format!(
            "field 'members' is missing in {} (write the report with walkforward --front --all-members, or read it with --members selected)",
            path
        ));
    };
    members
        .iter()
        .enumerate()
        .map(|(k, m)| {
            let rank = m.get("archive_rank").and_then(Json::as_f64).map_or_else(|| (k + 1).to_string(), jnum);
            wf_entry(&j, m, "series", &format!("{}#{}", path, rank), format!("{}#{}", label_of(path), rank))
        })
        .collect()
}

/// `root.<prefix>` holds the series; the guards, the annualisation and the rule come
/// from the report `j`, the hold length from `root.pooled`.
pub(crate) fn wf_entry(j: &Json, root: &Json, prefix: &'static str, path: &str, label: String) -> Res<Wf> {
    let net = series(root, &[prefix, "period_net"], path)?;
    let time = series(root, &[prefix, "period_time"], path)?;
    if net.len() != time.len() {
        return usage_err(format!(
            "field '{}.period_time' has {} values but '{}.period_net' has {} in {}",
            prefix,
            time.len(),
            prefix,
            net.len(),
            path
        ));
    }
    let optional = |key: &str| -> Res<Option<Vec<f64>>> {
        let v = at(root, &[prefix, key]).is_some().then(|| series(root, &[prefix, key], path)).transpose()?;
        match v {
            Some(x) if x.len() != net.len() => usage_err(format!("field '{}.{}' has {} values but '{}.period_net' has {} in {}", prefix, key, x.len(), prefix, net.len(), path)),
            v => Ok(v),
        }
    };
    let (gross, exposure, entry_series, bh) = (optional("period_gross")?, optional("period_exposure")?, optional("period_entries")?, optional("period_bh")?);
    let Some(ppy) = at(j, &["annualization", "periods_per_year"]).and_then(Json::as_f64) else {
        return usage_err(format!("field 'annualization.periods_per_year' is missing in {}", path));
    };
    if let Some(note) = check_annualization(path, j, &time, ppy)? {
        eprintln!("{}", note);
    }
    let guards = GUARDS.iter().map(|g| at(j, g).cloned().unwrap_or(Json::Null)).collect();
    let fit_guards: Vec<Json> = FIT_GUARDS.iter().map(|g| at(j, g).cloned().unwrap_or(Json::Null)).collect();
    let rule = threshold_rule(j);
    let horizon = at(j, &["purge", "horizon_seconds"]).and_then(Json::as_f64).unwrap_or(0.0);
    let dt = median_step(&time);
    let horizon_periods = if horizon > 0.0 && dt > 0.0 { (horizon / dt).ceil() as usize } else { 0 };
    let entries = at(root, &["pooled", "entries"]).and_then(Json::as_f64).unwrap_or(0.0);
    let active = at(root, &["pooled", "active_rows"]).and_then(Json::as_f64).unwrap_or(0.0);
    let hold = if entries > 0.0 { (active / entries).ceil() as usize } else { 1 };
    Ok(Wf {
        label,
        path: path.to_string(),
        prefix,
        default_block: inference::auto_block_len(net.len(), horizon_periods, hold),
        net,
        gross,
        exposure,
        entries: entry_series,
        bh,
        time,
        ppy,
        guards,
        fit_guards,
        rule,
    })
}

/// The `threshold_rule` member of a report entry, only with `--threshold-arms`.
fn rule_field(arms: bool, rule: &str) -> String {
    if arms { format!(",\"threshold_rule\":{}", jstr(rule)) } else { String::new() }
}

fn threshold_rule(j: &Json) -> String {
    let t = at(j, &["threshold"]);
    let num = |k: &str| t.and_then(|t| t.get(k)).and_then(Json::as_f64);
    if t.and_then(|t| t.get("fit")).is_some_and(|f| *f != Json::Null) {
        let (e, b, l) = (num("cut_min_entries"), num("cut_blocks"), num("cut_lambda"));
        let f = |v: Option<f64>| v.map_or("?".to_string(), jnum);
        return format!("fitness(min_entries={},blocks={},lambda={})", f(e), f(b), f(l));
    }
    if num("quantile").is_some() { "quantile" } else { "fixed" }.to_string()
}

/// `arms`: the fit paths may differ (`--threshold-arms`).
pub(crate) fn require_same(a: &Wf, b: &Wf, arms: bool) -> Res<()> {
    let fit = FIT_GUARDS.iter().zip(a.fit_guards.iter().zip(&b.fit_guards)).filter(|_| !arms);
    let all = GUARDS.iter().zip(a.guards.iter().zip(&b.guards)).chain(fit);
    for (g, (x, y)) in all {
        if x != y {
            let hint = if FIT_GUARDS.contains(g) {
                "; pass --threshold-arms to compare a fitted cut with a fixed cut"
            } else if g == &MISSING_SCORE {
                "; price every report with the same walkforward --missing-score (null is a report of a build that carried the position)"
            } else if g == &GRID_RULE {
                "; price every report with the same walkforward --period-grid (null is the score grid)"
            } else {
                ""
            };
            return usage_err(format!(
                "reports differ in field '{}': {} vs {}{}",
                g.join("."),
                a.path,
                b.path,
                hint
            ));
        }
    }
    if a.time != b.time {
        let (field, hint) = if a.prefix == "pooled" && b.prefix == "pooled" {
            ("pooled.period_time".to_string(), "")
        } else {
            (format!("{}.period_time", if a.prefix == b.prefix { a.prefix } else { "series" }), "; price the members with walkforward --period-grid target")
        };
        return usage_err(format!(
            "reports differ in field '{}': {} ({} periods) vs {} ({} periods){}",
            field,
            a.path,
            a.time.len(),
            b.path,
            b.time.len(),
            hint
        ));
    }
    Ok(())
}

fn load_all(paths: &[String]) -> Res<Vec<Wf>> {
    load_set(paths, false, false)
}

/// `members`: one entry per `members[]` row of every report (`--members all`, #1176).
fn load_set(paths: &[String], arms: bool, members: bool) -> Res<Vec<Wf>> {
    let mut out: Vec<Wf> = Vec::with_capacity(paths.len());
    for p in paths {
        let entries = if members { load_members(p)? } else { vec![load_wf(p)?] };
        for w in entries {
            if let Some(first) = out.first() {
                require_same(first, &w, arms)?;
            }
            out.push(w);
        }
    }
    Ok(out)
}

fn default_block(wfs: &[Wf]) -> usize {
    wfs.iter().map(|w| w.default_block).max().unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

pub(crate) struct Options {
    pub(crate) wf: Vec<String>,
    pub(crate) compare: Option<(String, String)>,
    pub(crate) baseline: Option<String>,
    pub(crate) baseline_rung: Option<String>,
    pub(crate) rho: f64,
    pub(crate) pbo_blocks: usize,
    bootstrap: String,
    level: f64,
    s_seed: Option<f64>,
    pub(crate) block_len: Option<usize>,
    pub(crate) draws: usize,
    pub(crate) seed: u32,
    pub(crate) out: Option<String>,
    pub(crate) threshold_arms: bool,
    /// #1176: `--members all`: one series per `members[]` row.
    pub(crate) all_members: bool,
    pub(crate) regimes: Option<String>,
}

pub(crate) fn parse_args(args: &[String]) -> Res<Options> {
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
        threshold_arms: false,
        all_members: false,
        regimes: None,
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
            "--threshold-arms" => o.threshold_arms = true,
            "--members" => {
                let v = val(&mut i)?;
                o.all_members = match v.as_str() {
                    "all" => true,
                    "selected" => false,
                    _ => return usage_err(format!("--members must be all or selected, got '{}'", v)),
                };
            }
            "--regimes" => o.regimes = Some(val(&mut i)?),
            other if crate::trials_table::FLAGS.contains(&other) => return usage_err(format!("{} needs --table", other)),
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
        &["--baseline", "--baseline-rung", "--cluster-rho", "--pbo-blocks", "--regimes", "--members"],
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

pub(crate) fn book_hash(x: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(x.len() * 8);
    for v in x {
        bytes.extend_from_slice(&((v * BOOK_RESOLUTION).round() as i64).to_le_bytes());
    }
    crate::manifest::sha256_hex(&bytes)
}

pub(crate) fn is_constant(x: &[f64]) -> bool {
    x.first().map_or(true, |&f| x.iter().all(|&v| v == f))
}

/// Cash, buy-and-hold, constant long/flat, the best single feature and any
/// trend rung are the long-only baselines.
fn is_baseline_name(name: &str) -> bool {
    matches!(name, "cash" | "buy_and_hold" | "constant_long_flat" | "best_single_feature") || name.starts_with("trend")
}

pub(crate) fn load_baseline(path: &str, rung: Option<&str>, time: &[f64]) -> Res<(String, Vec<f64>)> {
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

pub(crate) fn dsr_json(d: &inference::Dsr, n_eff: f64) -> String {
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
    let wfs = load_set(&o.wf, o.threshold_arms, o.all_members)?;
    let n = wfs[0].net.len();
    let ppy = wfs[0].ppy;
    let block = o.block_len.unwrap_or_else(|| default_block(&wfs)).clamp(1, n.max(1));

    // Distinct books, in order of first appearance.
    let hashes: Vec<String> = wfs.iter().map(|w| book_hash(&w.net)).collect();
    let mut distinct: Vec<usize> = Vec::new();
    let mut book_of = vec![0usize; wfs.len()];
    for (i, h) in hashes.iter().enumerate() {
        match distinct.iter().position(|&d| &hashes[d] == h) {
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
    let n_eff = clusters.max(1) as f64;
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
    let calendar = match o.regimes.as_deref() {
        None => None,
        Some(path) => {
            let text = std::fs::read_to_string(path).or_else(|e| input_err(format!("{}: {}", path, e)))?;
            Some(crate::regimes::Calendar::parse(&text).or_else(|e| usage_err(format!("--regimes {}: {}", path, e)))?)
        }
    };
    let strata_of = |w: &Wf| -> String {
        use crate::regimes::{strata, strata_json, Series, StatOpts};
        let Some(cal) = &calendar else { return String::new() };
        let (names, assign) = cal.assign(&w.time);
        let s = Series { net: &w.net, gross: w.gross.as_deref(), exposure: w.exposure.as_deref(), entries: w.entries.as_deref(), bh: w.bh.as_deref() };
        let so = StatOpts { periods_per_year: w.ppy, draws: o.draws, block_len: Some(block), seed: o.seed };
        format!(",\"regimes\":{}", strata_json(&strata(&names, "calendar", &assign, &s, &so)))
    };
    let members: Vec<String> = wfs
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let v = varying.iter().position(|&x| x == distinct[book_of[i]]);
            format!(
                "{{\"label\":{},\"book\":{},\"cluster\":{},\"sharpe_annual\":{}{}{}}}",
                jstr(&w.label),
                book_of[i],
                v.map_or("null".to_string(), |p| cluster_of[p].to_string()),
                jnum(inference::sharpe_per_period(&w.net) * ann),
                rule_field(o.threshold_arms, &w.rule),
                strata_of(w)
            )
        })
        .collect();
    Ok(format!(
        "{{\"schema\":{},\"series\":{},\"distinct_books\":{},\"constant_books\":{},\"clusters\":{},\"cluster_rho\":{},\
\"participation_ratio\":{},\"periods\":{},\"periods_per_year\":{},\"block_len\":{},\"draws\":{},\"seed\":{},\
\"best\":{},\"spa\":{},\"pbo\":{},\"min_btl_years\":{},\"span_years\":{},\"notes\":[{}],\"members\":[{}]{}}}\n",
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
        members.join(","),
        format!(
            "{}{}",
            if o.threshold_arms { ",\"threshold_arms\":true" } else { "" },
            calendar.as_ref().map_or(String::new(), |c| format!(
                ",\"regimes_calendar\":{{\"sha256\":{},\"rows\":{},\"labels\":[{}]}}",
                jstr(&c.sha256),
                c.rows.len(),
                c.labels().iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",")
            ))
        )
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
    require_same(&a[0], &b[0], o.threshold_arms)?;
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
    let d: Vec<f64> = xa.iter().zip(&xb).map(|(p, q)| p - q).collect();
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
\"seeds\":{{\"n_a\":{},\"n_b\":{},\"delta\":{},\"s_seed\":{},\"df\":{},\"z\":{},\"z_reason\":{}}}{}}}\n",
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
        z_reason.map_or("null".to_string(), jstr),
        if o.threshold_arms {
            let rules = |s: &[Wf]| s.iter().map(|w| jstr(&w.rule)).collect::<Vec<_>>().join(",");
            format!(",\"threshold_arms\":true,\"threshold_rule_a\":[{}],\"threshold_rule_b\":[{}]", rules(&a), rules(&b))
        } else {
            String::new()
        }
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
    if args.iter().any(|a| a == "--cpcv-assemble") {
        return crate::cpcv::assemble_cli(args);
    }
    if args.iter().any(|a| a == "--table") {
        return crate::trials_table::cli(args);
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
