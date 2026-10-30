//! Glue between a finished run and [`crate::gates`] (#257).
//!
//! Builds the reference rungs from the prepared rows, scores the archive
//! members on the validation and test slices, runs the gates and renders the
//! JSON and text pieces `main` splices into its report. Everything is exact
//! CPU `f64`, single-threaded, and a pure function of the prepared rows.

use crate::data::{Prepared, Sample};
use crate::devlog::{num, text};
use crate::expr::{self, Node};
use crate::gates::{self, Cmp, Comparison, GateConfig, GateReport, MemberInput, References, RowsView, Verdict};
use crate::metrics::{self, Metric};
use crate::noise::{newey_west_lag, KeyIndex, LagSource};
use crate::target::TargetSpec;
use crate::target_structure::{self, RungPredictions};

pub const DEFAULT_Z: f64 = 1.0;
pub const DEFAULT_ALPHA: f64 = 0.05;

pub struct Options {
    pub metric: Metric,
    pub z: f64,
    pub alpha: f64,
    /// `--noise-lag`; `None` is `auto`.
    pub noise_lag: Option<usize>,
    /// Temporal split or a time column: keys and the lag rule apply.
    pub time_ordered: bool,
}

pub struct MemberFacts<'a> {
    pub archive_rank: usize,
    pub cost: f64,
    pub tree: &'a Node,
    pub expression: &'a str,
    /// Structural identification of the constants (true without constants).
    pub identified: bool,
    pub practically_identified: Option<bool>,
}

/// What `main` splices into the report.
pub struct Output {
    pub report: GateReport,
    /// Per member, in the order given: JSON members with a leading comma.
    pub member_json: Vec<String>,
    /// `,"selection":...,"noise":...` for the top level.
    pub top_json: String,
    /// Replacement for the `baselines.target_structure` object, when a
    /// structure is declared and the zero-parameter rung exists.
    pub structure_json: Option<String>,
    /// `,"validation_error":x` pieces for the mean and linear baselines.
    pub mean_validation_json: String,
    pub linear_validation_json: String,
    pub text: Vec<String>,
    pub seconds: f64,
}

struct View {
    /// Indices into the sample of the rows kept (finite structure).
    kept: Vec<usize>,
    y: Vec<f64>,
    w: Vec<f64>,
    g: Vec<f64>,
    keys: Vec<f64>,
}

fn median_step(keys: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = keys.iter().copied().filter(|k| k.is_finite()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup();
    let mut d: Vec<f64> = v.windows(2).map(|p| p[1] - p[0]).filter(|x| *x > 0.0).collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(|a, b| a.total_cmp(b));
    Some(d[d.len() / 2])
}

pub fn lag_for(keys: &[f64], opts: &Options, spec: Option<&TargetSpec>) -> (usize, LagSource) {
    if let Some(l) = opts.noise_lag {
        return (l, LagSource::Flag);
    }
    if !opts.time_ordered {
        return (0, LagSource::Iid);
    }
    let timestamps = KeyIndex::new(keys).groups;
    let rule = newey_west_lag(timestamps, None);
    let horizon_steps = spec.and_then(|s| {
        median_step(keys).map(|step| (s.horizon_seconds / step).ceil().max(1.0) as usize)
    });
    let lag = newey_west_lag(timestamps, horizon_steps);
    let source = if lag > rule { LagSource::Horizon } else { LagSource::RuleOfThumb };
    (lag, source)
}

fn weight_of(s: &Sample, i: usize) -> f64 {
    if s.w.is_empty() {
        1.0
    } else {
        s.w[i]
    }
}

/// Predictions of a tree on the kept rows of a sample.
fn predict_kept(tree: &Node, s: &Sample, kept: &[usize]) -> Vec<f64> {
    kept.iter().map(|&i| expr::eval(tree, &s.x[i])).collect()
}

/// Mean of fold errors (or the single error) over finite predictions.
fn score_finite(pred: &[f64], s: &Sample, metric: Metric) -> f64 {
    let mut p = Vec::with_capacity(pred.len());
    let mut y = Vec::with_capacity(pred.len());
    let mut w = Vec::new();
    for (i, &v) in pred.iter().enumerate() {
        if v.is_finite() {
            p.push(v);
            y.push(s.y[i]);
            if !s.w.is_empty() {
                w.push(s.w[i]);
            }
        }
    }
    if p.is_empty() {
        return f64::NAN;
    }
    metrics::score_weighted(&p, &y, &w, metric, 0)
}

fn test_error(pred: &[f64], prepared: &Prepared, metric: Metric) -> f64 {
    if prepared.fold_samples.is_empty() {
        return score_finite(pred, &prepared.test, metric);
    }
    let mut at = 0usize;
    let mut errs = Vec::new();
    for f in &prepared.fold_samples {
        let end = (at + f.y.len()).min(pred.len());
        errs.push(score_finite(&pred[at.min(end)..end], f, metric));
        at += f.y.len();
    }
    errs.iter().sum::<f64>() / errs.len() as f64
}

struct Preds {
    train: RungPredictions,
    validation: Option<RungPredictions>,
    test: RungPredictions,
}

fn pick(p: &RungPredictions, which: usize) -> &Vec<f64> {
    match which {
        0 => &p.zero_param,
        1 => &p.affine,
        2 => &p.binned20,
        _ => &p.construction_ols,
    }
}

/// Weighted Pearson correlation and `1 - SSE/SS0` of `r_hat` against `r`.
fn return_space(
    spec: &TargetSpec,
    pred: &[f64],
    view: &View,
) -> Option<(usize, usize, f64, f64)> {
    let (mut sw, mut sa, mut sb) = (0.0, 0.0, 0.0);
    let mut rows = 0usize;
    let mut clipped = 0usize;
    let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(pred.len());
    for i in 0..pred.len() {
        let wi = if view.w.is_empty() { 1.0 } else { view.w[i] };
        if !(wi > 0.0) {
            continue;
        }
        let g = if view.g.is_empty() { 0.0 } else { view.g[i] };
        if let Some((lo, hi)) = spec.clip {
            if view.y[i] <= lo || view.y[i] >= hi {
                clipped += 1;
                continue;
            }
        }
        let r = spec.return_from_target(view.y[i], g);
        let rh = spec.return_from_target(pred[i], g);
        if !(r.is_finite() && rh.is_finite()) {
            continue;
        }
        sw += wi;
        sa += wi * r;
        sb += wi * rh;
        rows += 1;
        pts.push((wi, r, rh));
    }
    if rows < 2 || !(sw > 0.0) {
        return None;
    }
    let (ma, mb) = (sa / sw, sb / sw);
    let (mut saa, mut sbb, mut sab, mut sse, mut ss0) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(wi, r, rh) in &pts {
        saa += wi * (r - ma) * (r - ma);
        sbb += wi * (rh - mb) * (rh - mb);
        sab += wi * (r - ma) * (rh - mb);
        sse += wi * (r - rh) * (r - rh);
        ss0 += wi * r * r;
    }
    let corr = if saa > 0.0 && sbb > 0.0 { sab / (saa * sbb).sqrt() } else { f64::NAN };
    let r2 = if ss0 > 0.0 { 1.0 - sse / ss0 } else { f64::NAN };
    Some((rows, clipped, corr, r2))
}

fn comparisons_json(list: &[Comparison]) -> String {
    let items: Vec<String> = list
        .iter()
        .map(|c| {
            format!(
                "{}:{{\"delta\":{},\"se\":{},\"verdict\":{}}}",
                text(c.reference),
                num(c.delta),
                num(c.se),
                text(c.verdict.name())
            )
        })
        .collect();
    format!("{{{}}}", items.join(","))
}

fn beats(list: &[Comparison], name: &str) -> String {
    match list.iter().find(|c| c.reference == name) {
        Some(c) => (c.verdict == Cmp::Better).to_string(),
        None => "null".to_string(),
    }
}

fn opt_usize(v: Option<usize>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string())
}

/// Run the gates for the archive `members` (in archive order).
pub fn run(
    prepared: &Prepared,
    spec: Option<&TargetSpec>,
    members: &[MemberFacts],
    opts: &Options,
    linear_coef: Option<&[f64]>,
) -> Output {
    let start = std::time::Instant::now();
    let structural = spec.is_some_and(target_structure::structural) && prepared.target_scale.is_some();
    let scale = prepared.target_scale.as_ref();
    let empty: Vec<f64> = Vec::new();
    let (g_train, g_val, g_test): (&[f64], Option<&[f64]>, &[f64]) = match scale {
        Some(ts) => (&ts.train, ts.validation.as_deref(), &ts.test),
        None => (&empty, None, &empty),
    };
    let s_of = |g: &[f64]| -> Vec<f64> {
        match spec {
            Some(sp) if structural => target_structure::structure_values(sp, g),
            _ => Vec::new(),
        }
    };
    let (s_train, s_test) = (s_of(g_train), s_of(g_test));
    let s_val = g_val.map(|g| s_of(g));

    let models = if structural {
        Some(target_structure::fit(&s_train, g_train, &prepared.train.y, &prepared.train.w))
    } else {
        None
    };
    let preds = models.as_ref().map(|m| Preds {
        train: target_structure::predict(m, &s_train, g_train),
        validation: s_val
            .as_ref()
            .zip(g_val)
            .map(|(s, g)| target_structure::predict(m, s, g)),
        test: target_structure::predict(m, &s_test, g_test),
    });

    let keys_of = |all: Option<&Vec<f64>>, n: usize| -> Vec<f64> {
        match (opts.time_ordered, all) {
            (true, Some(k)) => k.clone(),
            _ => (0..n).map(|i| i as f64).collect(),
        }
    };
    let make_view = |s: &Sample, sv: &[f64], g: &[f64], keys: Vec<f64>| -> View {
        let n = s.y.len();
        let kept: Vec<usize> = (0..n)
            .filter(|&i| {
                let gv = if g.is_empty() { f64::NAN } else { g[i] };
                let sval = if sv.is_empty() { f64::NAN } else { sv[i] };
                !structural || (sval.is_finite() && gv.is_finite())
            })
            .collect();
        View {
            y: kept.iter().map(|&i| s.y[i]).collect(),
            w: if s.w.is_empty() { Vec::new() } else { kept.iter().map(|&i| s.w[i]).collect() },
            g: if g.is_empty() { Vec::new() } else { kept.iter().map(|&i| g[i]).collect() },
            keys: kept.iter().map(|&i| keys[i]).collect(),
            kept,
        }
    };
    let train_keys = prepared.keys.as_ref();
    let val_view = prepared.validation.as_ref().map(|v| {
        let keys = keys_of(train_keys.and_then(|k| k.validation.as_ref()), v.y.len());
        make_view(v, s_val.as_deref().unwrap_or(&[]), g_val.unwrap_or(&[]), keys)
    });
    let mut test_view = {
        let keys = keys_of(train_keys.map(|k| &k.test), prepared.test.y.len());
        make_view(&prepared.test, &s_test, g_test, keys)
    };
    // With several folds the pooled test error is the mean of fold errors:
    // every row carries its sample weight over the fold's kept weight total.
    if !prepared.fold_samples.is_empty() {
        let folds = prepared.fold_samples.len() as f64;
        let mut fold_of = Vec::with_capacity(prepared.test.y.len());
        for (f, s) in prepared.fold_samples.iter().enumerate() {
            fold_of.extend(std::iter::repeat(f).take(s.y.len()));
        }
        let mut totals = vec![0.0; prepared.fold_samples.len()];
        for &i in &test_view.kept {
            totals[fold_of[i]] += weight_of(&prepared.test, i);
        }
        test_view.w = test_view
            .kept
            .iter()
            .map(|&i| weight_of(&prepared.test, i) / (folds * totals[fold_of[i]]))
            .collect();
    }

    // References on the kept rows.
    let ybar = if prepared.train.y.is_empty() {
        f64::NAN
    } else {
        prepared.train.y.iter().sum::<f64>() / prepared.train.y.len() as f64
    };
    let coef: Option<Vec<f64>> = linear_coef.map(|c| c.to_vec());
    let lin = |s: &Sample, kept: &[usize]| -> Vec<f64> {
        let c = coef.as_ref().expect("linear coefficients");
        kept.iter()
            .map(|&i| {
                let mut v = c[0];
                for (j, &xj) in s.x[i].iter().enumerate() {
                    if j + 1 < c.len() {
                        v += c[j + 1] * xj;
                    }
                }
                v
            })
            .collect()
    };
    let mut names: Vec<&'static str> = vec!["mean"];
    let nv = val_view.as_ref().map_or(0, |v| v.kept.len());
    let mut ref_val: Vec<Vec<f64>> = vec![vec![ybar; nv]];
    let mut ref_test: Vec<Vec<f64>> = vec![vec![ybar; test_view.kept.len()]];
    if coef.is_some() {
        names.push("linear");
        ref_val.push(match (&prepared.validation, &val_view) {
            (Some(v), Some(vv)) => lin(v, &vv.kept),
            _ => Vec::new(),
        });
        ref_test.push(lin(&prepared.test, &test_view.kept));
    }
    let mut gate_ref = 0usize;
    if let Some(p) = &preds {
        let sel = |which: usize| -> (Vec<f64>, Vec<f64>) {
            let v = match (&p.validation, &val_view) {
                (Some(pv), Some(vv)) => vv.kept.iter().map(|&i| pick(pv, which)[i]).collect(),
                _ => Vec::new(),
            };
            let t = test_view.kept.iter().map(|&i| pick(&p.test, which)[i]).collect();
            (v, t)
        };
        let (zv, zt) = sel(0);
        names.push("target_structure.zero_param");
        ref_val.push(zv);
        ref_test.push(zt);
        let (av, at) = sel(1);
        names.push("target_structure.affine");
        gate_ref = names.len() - 1;
        ref_val.push(av);
        ref_test.push(at);
    }
    let refs = References { names: names.clone(), val: ref_val, test: ref_test, gate: gate_ref };

    // Views for the gate.
    let (val_rows, test_rows) = {
        let vr = match &val_view {
            Some(v) => {
                let (lag, src) = lag_for(&v.keys, opts, spec);
                RowsView::new(v.y.clone(), v.w.clone(), &v.keys, lag, src)
            }
            None => RowsView::iid(Vec::new(), Vec::new()),
        };
        let (lag, src) = lag_for(&test_view.keys, opts, spec);
        let tr = RowsView::new(test_view.y.clone(), test_view.w.clone(), &test_view.keys, lag, src);
        (vr, tr)
    };

    let val_preds: Vec<Vec<f64>> = members
        .iter()
        .map(|m| match (&prepared.validation, &val_view) {
            (Some(v), Some(vv)) => predict_kept(m.tree, v, &vv.kept),
            _ => Vec::new(),
        })
        .collect();
    let test_preds: Vec<Vec<f64>> = members
        .iter()
        .map(|m| predict_kept(m.tree, &prepared.test, &test_view.kept))
        .collect();
    let hashes: Vec<u64> = members.iter().map(|m| expr::structural_hash(m.tree)).collect();
    let inputs: Vec<MemberInput> = members
        .iter()
        .enumerate()
        .map(|(i, m)| MemberInput {
            rank: m.archive_rank,
            cost: m.cost,
            hash: hashes[i],
            expression: m.expression,
            identified: m.identified,
            val_pred: &val_preds[i],
            test_pred: &test_preds[i],
        })
        .collect();
    let cfg = GateConfig { metric: opts.metric, z: opts.z, alpha: opts.alpha };
    let report = gates::evaluate(&inputs, &refs, &val_rows, &test_rows, &cfg);

    // Return-space metrics, scoring-only.
    let ret = |i: usize, view: Option<&View>, pred: &[f64]| -> String {
        match (spec, view) {
            (Some(sp), Some(v)) => match return_space(sp, pred, v) {
                Some((rows, clipped, corr, r2)) => format!(
                    "{{\"rows\":{},\"clipped_rows\":{},\"corr\":{},\"r2_vs_zero\":{}}}",
                    rows,
                    clipped,
                    num(corr),
                    num(r2)
                ),
                None => "null".to_string(),
            },
            _ => {
                let _ = i;
                "null".to_string()
            }
        }
    };

    let selected = report.selected_rank;
    let member_json: Vec<String> = members
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let g = &report.members[i];
            let nb = match &g.not_better_than {
                Some((label, d, s)) => format!(
                    "{{\"ref\":{},\"delta\":{},\"se\":{}}}",
                    text(label),
                    num(*d),
                    num(*s)
                ),
                None => "null".to_string(),
            };
            let structure_known = preds.is_some();
            let zp = |list: &[Comparison]| {
                if structure_known {
                    beats(list, "target_structure.zero_param")
                } else {
                    "null".to_string()
                }
            };
            let return_space_json = if spec.is_some() {
                format!(
                    "{{\"validation\":{},\"test\":{}}}",
                    ret(i, val_view.as_ref(), &val_preds[i]),
                    ret(i, Some(&test_view), &test_preds[i])
                )
            } else {
                "null".to_string()
            };
            format!(
                ",\"gates\":{{\"archive_rank\":{},\"identified\":{},\"practically_identified\":{},\"supported\":{},\"not_better_than\":{}}},\"comparisons\":{{\"validation\":{},\"test\":{}}},\"beats_mean_test\":{},\"beats_zero_param_test\":{},\"beats_mean_validation\":{},\"beats_zero_param_validation\":{},\"validated\":{},\"return_space\":{}",
                m.archive_rank,
                g.identified,
                m.practically_identified.map(|b| b.to_string()).unwrap_or_else(|| "null".to_string()),
                g.supported,
                nb,
                comparisons_json(&g.validation),
                comparisons_json(&g.test),
                beats(&g.test, "mean"),
                zp(&g.test),
                beats(&g.validation, "mean"),
                zp(&g.validation),
                selected == Some(m.archive_rank),
                return_space_json
            )
        })
        .collect();

    // Top-level selection and noise objects.
    let selected_expression = members
        .iter()
        .find(|m| Some(m.archive_rank) == selected)
        .map(|m| text(m.expression))
        .unwrap_or_else(|| "null".to_string());
    let finite_range = |keys: &[f64]| -> (String, String) {
        let f: Vec<f64> = keys.iter().copied().filter(|k| k.is_finite()).collect();
        if f.is_empty() {
            ("null".into(), "null".into())
        } else {
            (
                num(f.iter().cloned().fold(f64::INFINITY, f64::min)),
                num(f.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
            )
        }
    };
    let validation_json = match &val_view {
        Some(v) => {
            let (lo, hi) = if opts.time_ordered { finite_range(&v.keys) } else { ("null".into(), "null".into()) };
            format!(
                "{{\"rows\":{},\"timestamps\":{},\"time_min\":{},\"time_max\":{},\"purge_seconds_effective\":{}}}",
                v.kept.len(),
                report.noise.timestamps,
                lo,
                hi,
                prepared
                    .report
                    .validation_purge_seconds
                    .map(num)
                    .unwrap_or_else(|| "null".to_string())
            )
        }
        None => "null".to_string(),
    };
    let top_json = format!(
        ",\"selection\":{{\"criterion\":\"validation_gates\",\"sample\":{},\"test_role\":\"scoring_only\",\"verdict\":{},\"selected_rank\":{},\"selected_expression\":{},\"reference\":{},\"z\":{},\"z_null\":{},\"alpha\":{},\"eligible\":{},\"supported_ranks\":[{}],\"validation\":{}}},\"noise\":{}",
        if val_view.is_some() { "\"validation\"" } else { "null" },
        text(report.verdict.name()),
        opt_usize(report.selected_rank),
        selected_expression,
        text(report.reference),
        num(report.z),
        num(report.z_null),
        num(report.alpha),
        report.eligible,
        report.supported_ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(","),
        validation_json,
        if matches!(report.verdict, Verdict::Signal | Verdict::NoSignal) {
            format!(
                "{{\"estimator\":{},\"lag\":{},\"lag_source\":{},\"rows\":{},\"timestamps\":{},\"design_effect\":{},\"effective_n\":{}}}",
                text(if report.lag_source == LagSource::Iid { "iid" } else { "driscoll_kraay_bartlett" }),
                report.noise.lag,
                text(report.lag_source.name()),
                report.noise.rows,
                report.noise.timestamps,
                num(report.noise.design_effect),
                num(report.noise.effective_n)
            )
        } else {
            "null".to_string()
        }
    );

    // Rung table.
    let (structure_json, rung_text) = match (&models, &preds, spec) {
        (Some(m), Some(p), Some(sp)) => {
            let mut parts = Vec::new();
            let mut text_parts = Vec::new();
            for (which, name, params) in [(0usize, "zero_param", 0usize), (1, "affine", 2), (2, "binned20", BINS), (3, "construction_ols", 2)] {
                let tr = score_finite(pick(&p.train, which), &prepared.train, opts.metric);
                let va = match (&p.validation, &prepared.validation) {
                    (Some(pv), Some(v)) => Some(score_finite(pick(pv, which), v, opts.metric)),
                    _ => None,
                };
                let te = test_error(pick(&p.test, which), prepared, opts.metric);
                let extra = match which {
                    1 => format!(",\"a\":{},\"b\":{}", num(m.a), num(m.b)),
                    3 => format!(",\"intercept\":{},\"slope\":{}", num(m.c0), num(m.c1)),
                    _ => String::new(),
                };
                parts.push(format!(
                    "{}:{{\"params\":{},\"train_error\":{},\"validation_error\":{},\"test_error\":{}{}}}",
                    text(name),
                    params,
                    num(tr),
                    va.map(num).unwrap_or_else(|| "null".to_string()),
                    num(te),
                    extra
                ));
                text_parts.push(format!("{}={:.9}", name, te));
            }
            let columns = sp.scale_column().map(|c| text(c)).unwrap_or_default();
            (
                Some(format!(
                    ",\"source\":\"target_spec\",\"columns\":[{}],\"fit_rows\":{},\"non_finite_rows\":{},\"rungs\":{{{}}}",
                    columns,
                    m.fit_rows,
                    m.non_finite_rows,
                    parts.join(",")
                )),
                Some(format!(
                    "# target structure rungs (test): {} | mean={:.9}",
                    text_parts.join(" "),
                    test_error(&vec![ybar; prepared.test.y.len()], prepared, opts.metric)
                )),
            )
        }
        _ => (None, None),
    };

    let mean_validation_json = match &prepared.validation {
        Some(v) => format!(",\"validation_error\":{}", num(score_finite(&vec![ybar; v.y.len()], v, opts.metric))),
        None => String::new(),
    };
    let linear_validation_json = match (&prepared.validation, &coef) {
        (Some(v), Some(_)) => {
            let all: Vec<usize> = (0..v.y.len()).collect();
            format!(",\"validation_error\":{}", num(score_finite(&lin(v, &all), v, opts.metric)))
        }
        _ => String::new(),
    };

    // Text lines.
    let mut lines = Vec::new();
    lines.push(format!(
        "# selection: validation gates (z={:.2}, reference {}; test errors are scoring-only)",
        report.z, report.reference
    ));
    if let Some(t) = rung_text {
        lines.push(t);
    }
    if matches!(report.verdict, Verdict::Signal | Verdict::NoSignal) {
        let mut parts = Vec::new();
        for (i, m) in members.iter().enumerate() {
            let g = &report.members[i];
            if g.supported {
                parts.push(format!("rank {} supported", m.archive_rank));
            } else if let Some((label, d, s)) = &g.not_better_than {
                parts.push(format!(
                    "rank {} not better than {} (delta={:.3e}, se={:.3e})",
                    m.archive_rank, label, d, s
                ));
            }
        }
        lines.push(format!("# supported: {}", if parts.is_empty() { "none".to_string() } else { parts.join("; ") }));
    }
    let unidentified: Vec<String> = members
        .iter()
        .enumerate()
        .filter(|(i, _)| !report.members[*i].identified)
        .map(|(_, m)| format!("rank {}", m.archive_rank))
        .collect();
    if !unidentified.is_empty() {
        lines.push(format!(
            "# unidentified (never supported or selected): {}",
            unidentified.join(", ")
        ));
    }
    lines.push(match report.verdict {
        Verdict::Signal => {
            let r = report.selected_rank.unwrap_or(0);
            format!(
                "# verdict: signal: rank {} beats {} on validation beyond z_null={:.3}",
                r, report.reference, report.z_null
            )
        }
        Verdict::NoSignal => format!(
            "# verdict: no_signal: no member beats {} on validation beyond noise; quote {}; the front is a menu",
            report.reference, report.reference
        ),
        Verdict::Unvalidated => {
            "# verdict: unvalidated: no validation slice; the front is a menu; add --validation-frac".to_string()
        }
        Verdict::NotGated => format!(
            "# verdict: not_gated: metric {} has no per-row loss; no member is selected",
            opts.metric.name()
        ),
    });

    Output {
        report,
        member_json,
        top_json,
        structure_json,
        mean_validation_json,
        linear_validation_json,
        text: lines,
        seconds: start.elapsed().as_secs_f64(),
    }
}

use target_structure::BINS;
