//! #211: `rcsr dev-report` — read a development log and explain the run.
//!
//! The analyzer is part of the binary on purpose: the log schema, the reader and
//! the summary ship together, so there is no separate Python evaluator to drift
//! from the writer. It reads only the events [`crate::devlog`] writes.
//!
//! The report answers, in order:
//!
//! * what ran (command, seed, backend, data fingerprint, config hash, stop
//!   reason, wall, peak RSS, device-hours);
//! * where the generation wall went, phase by phase, as totals and a per
//!   generation median, plus a ranked list of the disjoint windows;
//! * how much of the candidate CPU time was **fitting** and how much was
//!   **evaluation** — the split that decides whether a run is fit-bound;
//! * how much of the variation effort the structural de-duplication and the
//!   complexity limit threw away, and how many offers the archive admitted;
//! * which CPU fallback causes fired on the GPU path, and how many seconds they
//!   cost;
//! * whether the search converged, and how the Pareto front grew;
//! * a ranked list of bottlenecks, plus cross-cutting findings that are not
//!   wall-clock phases (fallback rate, duplicate rate, GPU idle, a convergence
//!   plateau).
//!
//! A malformed or truncated line is skipped and counted: a killed run's log is
//! still readable up to its last complete event.

use crate::devlog::{Json, SCHEMA};
use std::path::Path;

const EXIT_USAGE: i32 = 2;
const EXIT_INPUT: i32 = 3;

/// One generation, peeled out of its JSON event.
#[derive(Clone, Debug, Default)]
pub struct GenPoint {
    pub generation: usize,
    pub wall: f64,
    pub prep: f64,
    pub search: f64,
    pub eval: f64,
    pub select: f64,
    pub archive: f64,
    /// #256: the constant-fit stage window (absent in older logs: 0).
    pub fit: f64,
    pub fit_cpu: f64,
    pub eval_cpu: f64,
    pub best: f64,
    pub median: f64,
    pub pareto: usize,
    pub hypervolume: f64,
    pub hash: String,
    pub offers: usize,
    pub admissions: usize,
    pub attempts: usize,
    pub duplicates: usize,
    pub dedup_rate: f64,
    pub too_complex: usize,
    pub offspring: usize,
    pub fitted: usize,
    pub refits: usize,
    pub fit_attempts: usize,
    pub fit_gpu: usize,
    pub fit_fallback: usize,
    pub fallback_causes: Vec<(String, usize)>,
    pub fallback_s: f64,
    pub fallback_workers: usize,
    pub gpu_kernel_s: f64,
    pub gpu_wall_s: f64,
    pub gpu_duty: f64,
    pub gpu_batches: u64,
    pub qd_coverage: Option<f64>,
    pub qd_entropy: Option<f64>,
    pub percent_converged: f64,
    pub islands: Vec<f64>,
}

fn f(j: Option<&Json>) -> f64 {
    j.and_then(|v| v.as_f64()).unwrap_or(f64::NAN)
}

fn n(j: Option<&Json>) -> usize {
    j.and_then(|v| v.as_usize()).unwrap_or(0)
}

fn s(j: Option<&Json>) -> String {
    j.and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn opt_f(j: Option<&Json>) -> Option<f64> {
    j.and_then(|v| v.as_f64())
}

fn peel_generation(ev: &Json) -> GenPoint {
    let phases = ev.get("phases");
    let ph = |k: &str| f(phases.and_then(|p| p.get(k)));
    let err = ev.get("error");
    let archive = ev.get("archive");
    let var = ev.get("variation");
    let fit = ev.get("fit");
    let fallback = ev.get("fallback");
    let gpu = ev.get("gpu");
    let qd = ev.get("qd");
    let conv = ev.get("convergence");
    let islands = match ev.get("islands") {
        Some(Json::Arr(items)) => items.iter().filter_map(|v| v.as_f64()).collect(),
        _ => Vec::new(),
    };
    let causes = match fit.and_then(|x| x.get("fallback_by_cause")).and_then(|v| v.as_obj()) {
        Some(fields) => fields
            .iter()
            .map(|(k, v)| (k.clone(), v.as_usize().unwrap_or(0)))
            .collect(),
        None => Vec::new(),
    };
    GenPoint {
        generation: n(ev.get("generation")),
        wall: f(ev.get("wall")),
        prep: ph("prep"),
        search: ph("search"),
        eval: ph("eval"),
        select: ph("select"),
        archive: ph("archive"),
        fit: ph("fit"),
        fit_cpu: ph("fit_cpu"),
        eval_cpu: ph("eval_cpu"),
        best: f(err.and_then(|e| e.get("best"))),
        median: f(err.and_then(|e| e.get("median"))),
        pareto: n(archive.and_then(|a| a.get("size"))),
        hypervolume: f(archive.and_then(|a| a.get("hypervolume"))),
        hash: s(archive.and_then(|a| a.get("hash"))),
        offers: n(archive.and_then(|a| a.get("offers"))),
        admissions: n(archive.and_then(|a| a.get("admissions"))),
        attempts: n(var.and_then(|v| v.get("attempts"))),
        duplicates: n(var.and_then(|v| v.get("duplicates"))),
        dedup_rate: f(var.and_then(|v| v.get("dedup_rate"))),
        too_complex: n(var.and_then(|v| v.get("too_complex"))),
        offspring: n(var.and_then(|v| v.get("offspring"))),
        fitted: n(var.and_then(|v| v.get("fitted"))),
        refits: n(var.and_then(|v| v.get("refits"))),
        fit_attempts: n(fit.and_then(|v| v.get("attempts"))),
        fit_gpu: n(fit.and_then(|v| v.get("gpu"))),
        fit_fallback: n(fit.and_then(|v| v.get("fallback"))),
        fallback_causes: causes,
        fallback_s: f(fallback.and_then(|v| v.get("seconds"))),
        fallback_workers: n(fallback.and_then(|v| v.get("workers"))),
        gpu_kernel_s: f(gpu.and_then(|v| v.get("kernel_s"))),
        gpu_wall_s: f(gpu.and_then(|v| v.get("wall_s"))),
        gpu_duty: f(gpu.and_then(|v| v.get("duty"))),
        gpu_batches: gpu
            .and_then(|v| v.get("batches"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u64,
        qd_coverage: opt_f(qd.and_then(|v| v.get("coverage"))),
        qd_entropy: opt_f(qd.and_then(|v| v.get("entropy"))),
        percent_converged: f(conv.and_then(|v| v.get("percent_converged"))),
        islands,
    }
}

/// One disjoint wall window of the generation loop, ranked by total time.
#[derive(Clone, Debug, PartialEq)]
pub struct Bottleneck {
    pub name: &'static str,
    pub seconds: f64,
    pub share: f64,
    pub reading: &'static str,
}

/// One cross-cutting finding that is not a wall-clock phase.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    /// `0..=1`, larger means "look here first".
    pub severity: f64,
    pub reading: String,
}

/// The whole summary.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    pub path: String,
    pub schema: String,
    /// `summary` or `full`, from `run_start` (or the first generation event).
    pub level: String,
    pub events: usize,
    pub skipped: usize,
    pub truncated: bool,
    pub command: String,
    pub seed: u32,
    pub backend: String,
    pub device: String,
    pub threads: usize,
    pub data_path: String,
    pub train_rows: usize,
    pub test_rows: usize,
    pub train_fingerprint: String,
    pub test_fingerprint: String,
    pub config_hash: String,
    pub islands_cfg: usize,
    pub generations_requested: usize,
    pub generations: usize,
    pub stop: String,
    pub wall_s: f64,
    pub search_s: f64,
    pub peak_rss_bytes: Option<u64>,
    pub device_hours: Option<f64>,
    pub run_wall_s: Option<f64>,
    pub device_busy_s: Option<f64>,
    pub duty_run: Option<f64>,
    pub unaccounted_s: Option<f64>,
    pub fit_kernel_s: Option<f64>,
    pub fit_fallback_s: Option<f64>,
    pub fit_refit_s: Option<f64>,
    pub fit_refits: Option<usize>,
    pub start_utc: String,
    pub end_utc: String,
    pub final_hash: String,
    pub final_pareto: usize,
    pub final_best: f64,
    pub log_events: u64,
    pub log_bytes: u64,
    pub log_dropped: u64,
    pub checkpoints: usize,
    // Totals over generations.
    pub gen_wall: f64,
    pub phases: Vec<(&'static str, f64)>,
    pub fit_cpu: f64,
    pub eval_cpu: f64,
    pub attempts: usize,
    pub duplicates: usize,
    pub too_complex: usize,
    pub offspring: usize,
    pub fitted: usize,
    pub refits: usize,
    pub offers: usize,
    pub admissions: usize,
    pub fallback_causes: Vec<(String, usize)>,
    pub fallback_s: f64,
    pub fallback_workers: usize,
    pub fit_attempts: usize,
    pub fit_gpu: usize,
    pub fit_fallback: usize,
    pub gpu_wall_s: f64,
    pub gpu_kernel_s: f64,
    pub gpu_batches: u64,
    pub gpu_duty_mean: Option<f64>,
    pub gpu_duty_min: Option<f64>,
    pub qd_last: Option<(f64, f64)>,
    pub conv_first: Option<usize>,
    pub conv_last: f64,
    pub best_first: f64,
    pub best_last: f64,
    pub best_last_gen: usize,
    pub pareto_first: usize,
    pub pareto_last: usize,
    pub pareto_max: usize,
    pub hv_first: f64,
    pub hv_last: f64,
    pub islands_last: Vec<f64>,
    pub bottlenecks: Vec<Bottleneck>,
    pub findings: Vec<Finding>,
    pub points: Vec<GenPoint>,
}

/// `0.75` is the convergence threshold documented in `docs/convergence.md`.
const CONVERGED_THRESHOLD: f64 = 0.75;

/// Build the summary from parsed events. `skipped` counts lines that were not
/// this schema's events (a torn trailing line after a kill, or a foreign file).
pub fn analyse(path: &str, events: &[Json], skipped: usize) -> Analysis {
    let mut a = Analysis {
        path: path.to_string(),
        schema: String::new(),
        events: events.len(),
        skipped,
        ..Analysis::default()
    };
    let mut start: Option<&Json> = None;
    let mut end: Option<&Json> = None;
    for ev in events {
        match ev.get("event").and_then(|v| v.as_str()) {
            Some("run_start") => {
                if a.schema.is_empty() {
                    a.schema = s(ev.get("schema"));
                }
                start = Some(ev);
            }
            Some("run_end") => end = Some(ev),
            Some("checkpoint") => a.checkpoints += 1,
            Some("generation") => a.points.push(peel_generation(ev)),
            _ => {}
        }
    }
    a.points.sort_by_key(|p| p.generation);
    if a.level.is_empty() {
        for ev in events {
            if ev.get("event").and_then(|v| v.as_str()) == Some("generation") {
                if let Some(l) = ev.get("level").and_then(|v| v.as_str()) {
                    a.level = l.to_string();
                    break;
                }
            }
        }
    }

    if let Some(s) = start {
        a.command = string_field(s, "command");
        a.level = string_field(s, "level");
        a.seed = s.get("seed").and_then(|v| v.as_usize()).unwrap_or(0) as u32;
        a.backend = string_field(s, "backend");
        a.device = string_field(s, "device");
        a.threads = s.get("threads").and_then(|v| v.as_usize()).unwrap_or(0);
        a.start_utc = string_field(s, "start");
        a.config_hash = s.get("config_hash").and_then(|v| v.as_str()).unwrap_or("").to_string();
        a.islands_cfg = s
            .get("config")
            .and_then(|c| c.get("islands"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        a.generations_requested = s
            .get("config")
            .and_then(|c| c.get("generations"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        if let Some(data) = s.get("data") {
            a.data_path = string_field(data, "path");
            a.train_rows = n(data.get("train").and_then(|t| t.get("rows")));
            a.test_rows = n(data.get("test").and_then(|t| t.get("rows")));
            a.train_fingerprint = string_field(
                data.get("train").unwrap_or(&Json::Null),
                "fingerprint",
            );
            a.test_fingerprint =
                string_field(data.get("test").unwrap_or(&Json::Null), "fingerprint");
        }
    }
    if let Some(e) = end {
        a.generations = e.get("generations").and_then(|v| v.as_usize()).unwrap_or(0);
        a.stop = string_field(e, "stop");
        a.wall_s = f(e.get("wall"));
        a.search_s = f(e.get("search"));
        a.peak_rss_bytes = opt_f(e.get("peak_rss_bytes")).map(|v| v as u64);
        a.device_hours = opt_f(e.get("device_hours"));
        a.run_wall_s = opt_f(e.get("run_wall_s"));
        a.device_busy_s = opt_f(e.get("device_busy_s"));
        a.duty_run = opt_f(e.get("duty_run"));
        a.unaccounted_s = opt_f(e.get("unaccounted_s"));
        a.fit_kernel_s = opt_f(e.get("fit_kernel_s"));
        a.fit_fallback_s = opt_f(e.get("fit_fallback_s"));
        a.fit_refit_s = opt_f(e.get("fit_refit_s"));
        a.fit_refits = opt_f(e.get("fit_refits")).map(|v| v as usize);
        a.end_utc = string_field(e, "end");
        a.log_events = opt_f(e.get("events")).map(|v| v as u64).unwrap_or(0);
        a.log_bytes = opt_f(e.get("bytes")).map(|v| v as u64).unwrap_or(0);
        a.log_dropped = opt_f(e.get("dropped")).map(|v| v as u64).unwrap_or(0);
        if let Some(ar) = e.get("archive") {
            a.final_hash = string_field(ar, "hash");
            a.final_pareto = n(ar.get("size"));
            a.final_best = f(ar.get("best_error"));
        }
    }

    // Totals over the logged generations.
    let mut prep = 0.0;
    let mut search = 0.0;
    let mut eval = 0.0;
    let mut select = 0.0;
    let mut archive = 0.0;
    let mut fit = 0.0;
    let mut duties = Vec::new();
    let mut causes: Vec<(String, usize)> = Vec::new();
    for p in &a.points {
        a.gen_wall += clean(p.wall);
        prep += clean(p.prep);
        search += clean(p.search);
        eval += clean(p.eval);
        select += clean(p.select);
        archive += clean(p.archive);
        fit += clean(p.fit);
        a.fit_cpu += clean(p.fit_cpu);
        a.eval_cpu += clean(p.eval_cpu);
        a.attempts += p.attempts;
        a.duplicates += p.duplicates;
        a.too_complex += p.too_complex;
        a.offspring += p.offspring;
        a.fitted += p.fitted;
        a.refits += p.refits;
        a.offers += p.offers;
        a.admissions += p.admissions;
        a.fallback_s += clean(p.fallback_s);
        a.fallback_workers = a.fallback_workers.max(p.fallback_workers);
        a.fit_attempts += p.fit_attempts;
        a.fit_gpu += p.fit_gpu;
        a.fit_fallback += p.fit_fallback;
        a.gpu_wall_s += clean(p.gpu_wall_s);
        a.gpu_kernel_s += clean(p.gpu_kernel_s);
        a.gpu_batches += p.gpu_batches;
        if p.gpu_wall_s.is_finite() && p.gpu_wall_s > 0.0 {
            duties.push(p.gpu_duty);
        }
        for (k, v) in &p.fallback_causes {
            match causes.iter_mut().find(|(c, _)| c == k) {
                Some((_, total)) => *total += v,
                None => causes.push((k.clone(), *v)),
            }
        }
        if let (Some(cov), Some(ent)) = (p.qd_coverage, p.qd_entropy) {
            a.qd_last = Some((cov, ent));
        }
    }
    causes.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    a.fallback_causes = causes;
    a.phases = vec![
        ("search", search),
        ("eval", eval),
        ("prep", prep),
        ("select", select),
        ("archive", archive),
        ("fit", fit),
    ];
    if !duties.is_empty() {
        a.gpu_duty_mean = Some(duties.iter().sum::<f64>() / duties.len() as f64);
        a.gpu_duty_min = Some(duties.iter().cloned().fold(f64::INFINITY, f64::min));
    }

    // Error / archive / convergence trajectory.
    if let Some(first) = a.points.first() {
        a.best_first = first.best;
        a.pareto_first = first.pareto;
        a.hv_first = first.hypervolume;
    }
    if let Some(last) = a.points.last() {
        a.best_last = last.best;
        a.pareto_last = last.pareto;
        a.hv_last = last.hypervolume;
        a.conv_last = last.percent_converged;
        a.islands_last = last.islands.clone();
        if a.final_hash.is_empty() {
            a.final_hash = last.hash.clone();
        }
    }
    let mut best_so_far = f64::INFINITY;
    for p in &a.points {
        if p.pareto > a.pareto_max {
            a.pareto_max = p.pareto;
        }
        if p.percent_converged >= CONVERGED_THRESHOLD && a.conv_first.is_none() {
            a.conv_first = Some(p.generation);
        }
        if p.best.is_finite() && p.best < best_so_far {
            best_so_far = p.best;
            a.best_last_gen = p.generation;
        }
    }

    // Ranked, disjoint wall windows.
    let mut ranked: Vec<(&'static str, f64, &'static str)> = vec![
        ("search", search, "island variation: selection, crossover, mutation, fitting, scoring"),
        ("eval", eval, "the batched evaluator call and its CPU fallback rescore (fit is its own phase)"),
        ("prep", prep, "per-generation sample draws and the carried-population refit"),
        ("select", select, "screened successive-halving promotion and population selection"),
        ("archive", archive, "Pareto insertions and the best/quantile bookkeeping"),
        ("fit", fit, "constant fitting: the GPU fit call, the exact CPU fallback and write-back"),
    ];
    // #495: the exact finalist refit runs after the last generation, so it is
    // its own window beside the generation phases.
    let refit_s = a.fit_refit_s.filter(|v| v.is_finite()).map(|v| v.max(0.0));
    if let Some(r) = refit_s {
        ranked.push((
            "refit",
            r,
            "post-loop exact CPU f64 refit of the finalists (outside the generation wall)",
        ));
    }
    ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
    let total = a.gen_wall + refit_s.unwrap_or(0.0);
    let wall = if total > 0.0 { total } else { 1.0 };
    a.bottlenecks = ranked
        .into_iter()
        .map(|(name, seconds, reading)| Bottleneck {
            name,
            seconds,
            share: seconds / wall,
            reading,
        })
        .collect();

    // Cross-cutting findings, ranked by severity.
    let mut findings = Vec::new();
    let candidate_cpu = a.fit_cpu + a.eval_cpu;
    if candidate_cpu > 0.0 {
        let fit_share = a.fit_cpu / candidate_cpu;
        findings.push(Finding {
            severity: fit_share,
            reading: format!(
                "constant fitting is {:.1}% of candidate CPU time ({:.3} s of {:.3} s); the run is {}",
                100.0 * fit_share,
                a.fit_cpu,
                candidate_cpu,
                if fit_share >= 0.5 { "fit-bound" } else { "evaluation-bound" }
            ),
        });
    }
    if a.attempts > 0 {
        let dup = a.duplicates as f64 / a.attempts as f64;
        findings.push(Finding {
            severity: dup,
            reading: format!(
                "structural de-duplication rejected {} of {} attempts ({:.1}%): {}",
                a.duplicates,
                a.attempts,
                100.0 * dup,
                if dup >= 0.5 {
                    "the population is saturated; more diversity pressure may pay"
                } else {
                    "diversity is not the limit"
                }
            ),
        });
    }
    if a.offers > 0 && a.attempts > 0 {
        let adm = a.admissions as f64 / a.offers as f64;
        findings.push(Finding {
            severity: 1.0 - adm,
            reading: format!(
                "the archive admitted {} of {} population offers ({:.2}%)",
                a.admissions,
                a.offers,
                100.0 * adm
            ),
        });
    }
    if let Some(duty) = a.gpu_duty_mean {
        findings.push(Finding {
            severity: 1.0 - duty,
            reading: format!(
                "GPU kernel-time duty averaged {:.1}% (min {:.1}%): {:.3} s kernel of {:.3} s full path",
                100.0 * duty,
                100.0 * a.gpu_duty_min.unwrap_or(duty),
                a.gpu_kernel_s,
                a.gpu_wall_s
            ),
        });
    }
    if a.fit_attempts > 0 && a.fit_fallback > 0 {
        let share = a.fit_fallback as f64 / a.fit_attempts as f64;
        findings.push(Finding {
            severity: share,
            reading: format!(
                "the batched GPU fit fell back to the CPU for {} of {} candidates ({:.1}%), costing {:.3} s",
                a.fit_fallback, a.fit_attempts, 100.0 * share, a.fallback_s
            ),
        });
    }
    if !a.points.is_empty() {
        let since = a.generations.saturating_sub(a.best_last_gen);
        let plateau = if a.generations > 0 {
            since as f64 / a.generations as f64
        } else {
            0.0
        };
        if a.conv_last < CONVERGED_THRESHOLD || plateau > 0.5 {
            findings.push(Finding {
                severity: plateau.max(1.0 - a.conv_last),
                reading: format!(
                    "the best error last improved at generation {} of {} \
                     ({} generation(s) of stalling); percent_converged {:.3}",
                    a.best_last_gen, a.generations, since, a.conv_last
                ),
            });
        }
    }
    findings.sort_by(|x, y| y.severity.total_cmp(&x.severity));
    a.findings = findings;
    a
}

fn clean(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        0.0
    }
}

fn string_field(j: &Json, key: &str) -> String {
    match j.get(key) {
        Some(Json::Str(s)) => s.clone(),
        Some(Json::Num(n)) => format!("{}", n),
        _ => String::new(),
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

fn median_of(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}

/// Render the analysis as the human-readable report `dev-report` prints.
pub fn render(a: &Analysis) -> String {
    let mut out = String::new();
    out.push_str("# rcsr development log\n\n");
    out.push_str(&format!(
        "`{}` — schema `{}`, `{}` level, {} event(s), {} of {} generation(s) logged, \
         {} checkpoint(s)\n\n",
        a.path,
        if a.schema.is_empty() { "(none)" } else { &a.schema },
        if a.level.is_empty() { "unknown" } else { &a.level },
        a.events,
        a.points.len(),
        a.generations_requested.max(a.points.len()),
        a.checkpoints
    ));
    if a.level == "summary" {
        out.push_str(
            "> This is a **summary** log: one event every few generations with the \
             monitoring fields. Per-generation quantiles, variation counters, archive \
             hypervolume/hash and RNG keys are only in `--dev-log-level full`.\n\n",
        );
    }
    if a.skipped > 0 {
        out.push_str(&format!(
            "> {} line(s) were skipped: not complete `{}` events. A killed run's \
             log is read up to its last complete event.\n\n",
            a.skipped, SCHEMA
        ));
    }
    if a.points.is_empty() {
        out.push_str("No `generation` events: there is nothing to analyse.\n");
        return out;
    }

    out.push_str("## Run\n\n| item | value |\n|---|---|\n");
    if !a.command.is_empty() {
        out.push_str(&format!("| command | `{}` |\n", a.command));
    }
    out.push_str(&format!(
        "| data | `{}` ({} train + {} test rows) |\n",
        a.data_path, a.train_rows, a.test_rows
    ));
    out.push_str(&format!(
        "| data fingerprint | train `{}`, test `{}` |\n",
        a.train_fingerprint, a.test_fingerprint
    ));
    out.push_str(&format!("| config hash | `{}` |\n", a.config_hash));
    out.push_str(&format!("| seed | {} |\n", a.seed));
    out.push_str(&format!(
        "| backend | {} |\n",
        match (a.backend.is_empty(), a.device.is_empty()) {
            (true, _) => "?".to_string(),
            (false, true) => a.backend.clone(),
            (false, false) => format!("{} on {}", a.backend, a.device),
        }
    ));
    out.push_str(&format!("| host threads | {} |\n", a.threads));
    out.push_str(&format!(
        "| generations | {} of {} requested |\n",
        a.generations, a.generations_requested
    ));
    out.push_str(&format!("| stop reason | `{}` |\n", a.stop));
    out.push_str(&format!(
        "| wall | {:.3} s (search {:.3} s) |\n",
        a.wall_s, a.search_s
    ));
    match a.peak_rss_bytes {
        Some(b) => out.push_str(&format!("| peak RSS | {} |\n", human_bytes(b))),
        None => out.push_str("| peak RSS | not available on this platform |\n"),
    }
    if let Some(h) = a.device_hours {
        out.push_str(&format!("| device-hours (evaluator window) | {:.6} |\n", h));
    }
    if let Some(w) = a.run_wall_s {
        out.push_str(&format!("| run ledger wall | {:.3} s |\n", w));
    }
    if let Some(b) = a.device_busy_s {
        out.push_str(&format!("| device-busy (interval union) | {:.3} s |\n", b));
    }
    if let Some(d) = a.duty_run {
        out.push_str(&format!("| run duty (union / wall) | {:.2}% |\n", 100.0 * d));
    }
    if let Some(u) = a.unaccounted_s {
        let pct = a
            .run_wall_s
            .filter(|w| *w > 0.0)
            .map(|w| 100.0 * u / w)
            .unwrap_or(0.0);
        out.push_str(&format!("| ledger unaccounted | {:.3} s ({:.2}%) |\n", u, pct));
    }
    if let Some(k) = a.fit_kernel_s {
        out.push_str(&format!("| fit kernel seconds | {:.3} s |\n", k));
    }
    if let Some(f) = a.fit_fallback_s {
        out.push_str(&format!("| fit CPU fallback | {:.3} s |\n", f));
    }
    if let Some(r) = a.fit_refit_s {
        out.push_str(&format!(
            "| exact finalist refit | {:.3} s, {} refit(s) |\n",
            r,
            a.fit_refits.unwrap_or(0)
        ));
    }
    out.push_str(&format!("| started / ended | {} / {} |\n", a.start_utc, a.end_utc));
    out.push_str(&format!(
        "| log | {} event(s), {}, {} dropped by the cap |\n\n",
        a.log_events,
        human_bytes(a.log_bytes),
        a.log_dropped
    ));

    // Phases.
    out.push_str("## Where the generation wall went\n\n");
    out.push_str("| phase | total s | median s/gen | share of generation wall |\n|---|---:|---:|---:|\n");
    for (name, total) in &a.phases {
        let mut per: Vec<f64> = a
            .points
            .iter()
            .map(|p| match *name {
                "search" => p.search,
                "eval" => p.eval,
                "prep" => p.prep,
                "select" => p.select,
                "fit" => p.fit,
                _ => p.archive,
            })
            .collect();
        let m = median_of(&mut per);
        let share = if a.gen_wall > 0.0 { total / a.gen_wall } else { 0.0 };
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.1}% |\n",
            name,
            total,
            m,
            100.0 * share
        ));
    }
    out.push_str(&format!(
        "| **generation wall** | {:.4} | {:.4} | 100.0% |\n\n",
        a.gen_wall,
        {
            let mut per: Vec<f64> = a.points.iter().map(|p| p.wall).collect();
            median_of(&mut per)
        }
    ));
    out.push_str(&format!(
        "Totals cover the {} logged generation(s). `search` is the island-thread scope, \
         so the CPU work inside it (`fit_cpu`, `eval_cpu`) overlaps other islands and \
         is not a phase of its own; the five windows above are disjoint on the main \
         thread.\n\n",
        a.points.len()
    ));

    // Fit vs eval.
    let candidate_cpu = a.fit_cpu + a.eval_cpu;
    out.push_str("## Fit vs evaluation\n\n| item | value |\n|---|---:|\n");
    out.push_str(&format!("| constant fitting (CPU, summed over islands) | {:.4} s |\n", a.fit_cpu));
    out.push_str(&format!("| candidate evaluation (CPU, summed over islands) | {:.4} s |\n", a.eval_cpu));
    if candidate_cpu > 0.0 {
        out.push_str(&format!(
            "| fit share of candidate CPU time | {:.1}% |\n",
            100.0 * a.fit_cpu / candidate_cpu
        ));
    }
    if a.search_s > 0.0 && a.islands_cfg > 0 {
        let capacity = a.search_s * a.islands_cfg as f64;
        out.push_str(&format!(
            "| fit share of the island CPU budget | {:.1}% |\n",
            100.0 * (a.fit_cpu / capacity).min(1.0)
        ));
    }
    if a.gpu_wall_s > 0.0 {
        out.push_str(&format!(
            "| GPU evaluator wall | {:.4} s in {} batch(es) |\n",
            a.gpu_wall_s, a.gpu_batches
        ));
    }
    if a.level == "full" || a.fitted > 0 {
        out.push_str(&format!(
            "| constants fitted on the CPU | {} candidate(s) |\n",
            a.fitted
        ));
    }
    out.push('\n');

    // Variation effort exists only in a full log; a summary log does not carry
    // per-generation variation counters.
    out.push_str("## Variation effort\n\n");
    if a.attempts == 0 && a.offers == 0 {
        out.push_str(
            "Not recorded at `summary` level: re-run with `--dev-log-level full` for \
             attempts, duplicates, too-complex rejects, archive offers/admissions and \
             refits.\n\n",
        );
    } else {
        out.push_str("| counter | total |\n|---|---:|\n");
        out.push_str(&format!("| offspring produced | {} |\n", a.offspring));
        out.push_str(&format!("| attempts | {} |\n", a.attempts));
        out.push_str(&format!("| rejected: duplicate | {} |\n", a.duplicates));
        out.push_str(&format!("| rejected: too complex | {} |\n", a.too_complex));
        if a.attempts > 0 {
            out.push_str(&format!(
                "| dedup rate | {:.1}% |\n",
                100.0 * a.duplicates as f64 / a.attempts as f64
            ));
        }
        out.push_str(&format!("| archive offers | {} |\n", a.offers));
        out.push_str(&format!("| archive admissions | {} |\n", a.admissions));
        out.push_str(&format!("| exact CPU refits | {} |\n\n", a.refits));
    }

    // Fallbacks.
    out.push_str("## Fallback causes\n\n");
    if a.fit_attempts == 0 && a.fallback_causes.is_empty() {
        out.push_str("No batched GPU fit or CPU fallback ran in this log.\n\n");
    } else {
        out.push_str("| item | value |\n|---|---:|\n");
        out.push_str(&format!("| batched fit attempts | {} |\n", a.fit_attempts));
        out.push_str(&format!("| fitted on the GPU | {} |\n", a.fit_gpu));
        out.push_str(&format!("| CPU fallback | {} |\n", a.fit_fallback));
        out.push_str(&format!("| CPU fallback seconds | {:.4} |\n", a.fallback_s));
        out.push_str(&format!("| evaluator fallback workers | {} |\n\n", a.fallback_workers));
        if !a.fallback_causes.is_empty() {
            out.push_str("| cause | candidates |\n|---|---:|\n");
            for (cause, count) in &a.fallback_causes {
                out.push_str(&format!("| `{}` | {} |\n", cause, count));
            }
            out.push('\n');
        } else if a.level != "full" {
            out.push_str(
                "The per-cause breakdown is only in `--dev-log-level full`; the counts \
                 above are the summary ledger.\n\n",
            );
        }
    }

    // Convergence.
    out.push_str("## Convergence\n\n| item | value |\n|---|---:|\n");
    out.push_str(&format!("| best error, first logged generation | {:.6e} |\n", a.best_first));
    out.push_str(&format!("| best error, last logged generation | {:.6e} |\n", a.best_last));
    out.push_str(&format!("| best error last improved at generation | {} |\n", a.best_last_gen));
    out.push_str(&format!("| percent_converged at the end | {:.3} |\n", a.conv_last));
    match a.conv_first {
        Some(g) => out.push_str(&format!(
            "| first generation at or above {:.2} converged | {} |\n\n",
            CONVERGED_THRESHOLD, g
        )),
        None => out.push_str(&format!(
            "| first generation at or above {:.2} converged | never |\n\n",
            CONVERGED_THRESHOLD
        )),
    }

    // Archive growth.
    out.push_str("## Archive growth\n\n| item | value |\n|---|---:|\n");
    out.push_str(&format!(
        "| Pareto size, first / last / max | {} / {} / {} |\n",
        a.pareto_first, a.pareto_last, a.pareto_max
    ));
    if a.hv_first.is_finite() && a.hv_last.is_finite() {
        out.push_str(&format!(
            "| hypervolume (reference 1.0), first / last | {:.6} / {:.6} |\n",
            a.hv_first, a.hv_last
        ));
    } else {
        out.push_str("| hypervolume | only in `--dev-log-level full` |\n");
    }
    out.push_str(&format!("| final archive hash | `{}` |\n\n", a.final_hash));
    if !a.islands_last.is_empty() {
        out.push_str("| island | best error (last generation) |\n|---:|---:|\n");
        for (i, v) in a.islands_last.iter().enumerate() {
            out.push_str(&format!("| {} | {:.6e} |\n", i, v));
        }
        out.push('\n');
    }
    if let Some((cov, ent)) = a.qd_last {
        out.push_str(&format!(
            "Quality-diversity coverage {:.1}%, occupancy entropy {:.3} bits at the end.\n\n",
            100.0 * cov, ent
        ));
    }

    // Bottlenecks.
    out.push_str("## Bottleneck ranking\n\n");
    for (i, b) in a.bottlenecks.iter().enumerate() {
        out.push_str(&format!(
            "{}. **{}** — {:.3} s ({:.1}% of {}): {}\n",
            i + 1,
            b.name,
            b.seconds,
            100.0 * b.share,
            if a.fit_refit_s.is_some() { "generation wall plus refit" } else { "generation wall" },
            b.reading
        ));
    }
    out.push('\n');
    out.push_str("## Other findings\n\n");
    if a.findings.is_empty() {
        out.push_str("Nothing to flag.\n");
    } else {
        out.push_str(
            "Ranked by severity (0–1; larger means look here first). These are cross-cutting \
             readings, not wall-clock windows.\n\n",
        );
        for f in &a.findings {
            out.push_str(&format!("- **(severity {:.2})** {}\n", f.severity, f.reading));
        }
    }
    out
}

/// `rcsr dev-report <log>`.
pub fn run(args: &[String]) -> ! {
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown dev-report argument '{}'\n{}", other, usage());
                std::process::exit(EXIT_USAGE);
            }
            other => {
                if path.replace(other).is_some() {
                    eprintln!("error: dev-report takes one log path\n{}", usage());
                    std::process::exit(EXIT_USAGE);
                }
            }
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("error: dev-report needs a log path\n{}", usage());
        std::process::exit(EXIT_USAGE);
    };
    let body = match std::fs::read_to_string(Path::new(path)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot read dev log '{}': {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    };
    let (events, skipped) = parse_log(&body);
    if events.is_empty() {
        eprintln!(
            "error: '{}' contains no {} events ({} line(s) skipped)",
            path,
            SCHEMA,
            skipped
        );
        std::process::exit(EXIT_INPUT);
    }
    let analysis = analyse(path, &events, skipped);
    print!("{}", render(&analysis));
    std::process::exit(0);
}

/// Parse every line, keeping the events of this schema. Blank lines are ignored
/// rather than counted, so a log with a trailing newline is clean.
pub fn parse_log(body: &str) -> (Vec<Json>, usize) {
    let mut events = Vec::new();
    let mut skipped = 0;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match Json::parse(trimmed) {
            Some(v) if v.get("schema").and_then(|s| s.as_str()) == Some(SCHEMA) => events.push(v),
            _ => skipped += 1,
        }
    }
    (events, skipped)
}

pub fn usage() -> String {
    "usage: rcsr dev-report <log>\n  \
     Summarise a --dev-log NDJSON file: phases, fit/eval share, fallback causes,\n  \
     convergence, archive growth and a ranked bottleneck list. Reads only the\n  \
     events the engine wrote; a torn final line is skipped.\n"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devlog::{run_end_json, run_start_json, GenRecord, RunEnd, RunStart};

    fn line(gen: usize, fit_cpu: f64, eval_cpu: f64, search: f64, dup: usize, att: usize) -> String {
        GenRecord {
            generation: gen,
            total_generations: 3,
            t: gen as f64 * (search + 0.01),
            wall_s: search + 0.01,
            prep_s: 0.005,
            search_s: search,
            eval_s: 0.004,
            select_s: 0.0005,
            archive_s: 0.0005,
            fit_s: 0.001,
            fit_cpu_s: fit_cpu,
            eval_cpu_s: eval_cpu,
            phase_b_s: 0.0,
            post_fit_drops: 0,
            best_error: 1.0 / gen as f64,
            median: 2.0,
            p05: 0.0,
            p25: 0.0,
            p75: 0.0,
            p95: 0.0,
            pareto_size: gen + 2,
            hypervolume: 3.0 * gen as f64,
            archive_hash: gen as u64,
            offers: 20,
            admissions: gen,
            attempts: att,
            duplicates: dup,
            too_complex: 1,
            offspring: 20,
            fitted: 20,
            refits: 0,
            gpu: None,
            fit: Default::default(),
            fallback_s: 0.0,
            fallback_workers: 0,
            qd: None,
            confidence: 0.5,
            stability: 0.5,
            maturity: 0.5,
            percent_converged: if gen >= 2 { 0.8 } else { 0.1 },
            island_best: vec![1.0 / gen as f64],
            rng_key: 7,
        }
        .json()
    }

    fn sample_log() -> String {
        let start = run_start_json(&RunStart {
            version: "0.1.0".into(),
            commit: "abc".into(),
            config: vec![
                ("generations".into(), "3".into()),
                ("islands".into(), "1".into()),
                ("seed".into(), "5".into()),
            ],
            seed: 5,
            backend: "cpu".into(),
            threads: 2,
            ..RunStart::default()
        });
        let end = run_end_json(&RunEnd {
            generations: 3,
            stop: "patience".into(),
            wall_s: 1.0,
            search_s: 0.9,
            archive_hash: 3,
            ..RunEnd::default()
        });
        format!(
            "{}\n{}\n{}\n{}\n{}\n",
            start,
            line(1, 0.1, 0.9, 1.0, 5, 10),
            line(2, 0.6, 0.4, 1.2, 6, 10),
            line(3, 0.7, 0.3, 1.1, 7, 10),
            end
        )
    }

    /// The report names the phases, ranks the disjoint windows by time and
    /// reports the fit share — the three questions the analyzer exists for.
    #[test]
    fn report_summarises_phases_fit_share_and_ranks_bottlenecks() {
        let (events, skipped) = parse_log(&sample_log());
        assert_eq!(skipped, 0);
        let a = analyse("x.ndjson", &events, skipped);
        assert_eq!(a.points.len(), 3);
        assert_eq!(a.generations, 3);
        assert_eq!(a.stop, "patience");
        assert!((a.fit_cpu - 1.4).abs() < 1e-12);
        assert!((a.eval_cpu - 1.6).abs() < 1e-12);
        assert!((a.fit_cpu / (a.fit_cpu + a.eval_cpu) - 1.4 / 3.0).abs() < 1e-12);
        // search is much larger than eval/prep/select/archive: it must rank first.
        assert_eq!(a.bottlenecks[0].name, "search");
        assert_eq!(a.bottlenecks.last().unwrap().name, "archive");
        assert!(a.bottlenecks[0].share > a.bottlenecks[1].share);
        // Totals across generations.
        assert!((a.gen_wall - (1.01 + 1.21 + 1.11)).abs() < 1e-9);
        // Convergence and archive trajectory.
        assert_eq!(a.conv_first, Some(2));
        assert_eq!(a.pareto_max, 5);
        assert_eq!(a.best_last_gen, 3);
        assert!((a.hv_last - 9.0).abs() < 1e-12);
        let text = render(&a);
        assert!(text.contains("## Where the generation wall went"));
        assert!(text.contains("## Fit vs evaluation"));
        assert!(text.contains("## Bottleneck ranking"));
        assert!(text.contains("1. **search**"));
        assert!(text.contains("| dedup rate |"));
        assert!(text.contains("first generation at or above 0.75 converged | 2"));
        assert!(text.contains("## Archive growth"));
        assert!(text.contains("0x0000000000000003"));
    }

    /// #495: `run_end` carries the exact finalist refit; the report ranks it
    /// beside the generation phases and without it the ranking is unchanged.
    #[test]
    fn report_ranks_the_post_loop_refit() {
        let log = sample_log();
        let end_line = log.lines().last().unwrap().to_string();
        let with_refit = run_end_json(&RunEnd {
            generations: 3,
            stop: "patience".into(),
            wall_s: 1.0,
            search_s: 0.9,
            archive_hash: 3,
            fit_refit_s: Some(2.5),
            fit_refits: Some(16),
            ..RunEnd::default()
        });
        let log = log.replace(&end_line, &with_refit);
        let (events, skipped) = parse_log(&log);
        let a = analyse("x.ndjson", &events, skipped);
        assert_eq!(a.fit_refit_s, Some(2.5));
        assert_eq!(a.fit_refits, Some(16));
        let names: Vec<&str> = a.bottlenecks.iter().map(|b| b.name).collect();
        assert_eq!(names[0], "search");
        assert_eq!(names[1], "refit");
        let text = render(&a);
        assert!(text.contains("2. **refit** — 2.500 s (42.9% of generation wall plus refit)"), "{text}");
        assert!(text.contains("| exact finalist refit | 2.500 s, 16 refit(s) |"));
        let (events, _) = parse_log(&sample_log());
        let plain = analyse("x.ndjson", &events, 0);
        assert!(plain.bottlenecks.iter().all(|b| b.name != "refit"));
        assert!(!render(&plain).contains("exact finalist refit"));
    }

    /// A `summary` log renders the monitoring sections, says what the level
    /// omits and never prints `NaN` for the fields a summary does not carry.
    #[test]
    fn a_summary_log_is_reported_as_such() {
        let start = run_start_json(&RunStart {
            version: "0.1.0".into(),
            level: crate::devlog::Level::Summary,
            every: 10,
            seed: 5,
            ..RunStart::default()
        });
        let mut body = format!("{}\n", start);
        for gen in 1..=3 {
            let mut rec = GenRecord {
                generation: gen * 10,
                total_generations: 30,
                t: gen as f64,
                wall_s: 0.5,
                search_s: 0.4,
                best_error: 1.0 / gen as f64,
                median: 2.0,
                pareto_size: gen + 1,
                percent_converged: 0.5,
                ..Default::default()
            };
            rec.offers = 0;
            body.push_str(&rec.json_at(crate::devlog::Level::Summary));
            body.push('\n');
        }
        let (events, skipped) = parse_log(&body);
        assert_eq!(skipped, 0);
        let a = analyse("summary.ndjson", &events, skipped);
        assert_eq!(a.level, "summary");
        assert_eq!(a.points.len(), 3);
        let text = render(&a);
        assert!(text.contains("`summary` level"), "{}", text);
        assert!(text.contains("Not recorded at `summary` level"), "{}", text);
        assert!(text.contains("| hypervolume | only in `--dev-log-level full` |"));
        assert!(text.contains("## Bottleneck ranking"));
        assert!(!text.contains("NaN"), "{}", text);
        // A summary log carries no dedup/admission counters, so no finding may
        // claim them.
        assert!(!text.contains("structural de-duplication rejected"), "{}", text);
    }

    /// A torn trailing line is skipped and the report still renders.
    #[test]
    fn a_torn_final_line_is_skipped_not_fatal() {
        let mut body = sample_log();
        body.push_str("{\"schema\":\"rcsr-devlog/1\",\"event\":\"generation\",\"generation\":4,");
        let (events, skipped) = parse_log(&body);
        assert_eq!(skipped, 1);
        assert_eq!(events.len(), 5);
        let a = analyse("x.ndjson", &events, skipped);
        let text = render(&a);
        assert!(text.contains("1 line(s) were skipped"));
    }

    /// Foreign lines are ignored, so the analyzer never invents events.
    #[test]
    fn foreign_lines_are_skipped() {
        let body = format!("not json\n{{\"a\":1}}\n{}{}", sample_log(), "\n");
        let (events, skipped) = parse_log(&body);
        assert_eq!(skipped, 2);
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn usage_mentions_the_log_and_the_summary() {
        let u = usage();
        assert!(u.contains("rcsr dev-report <log>"));
        assert!(u.contains("bottleneck"));
    }
}
