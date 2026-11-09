//! #482: the parts of a `run.json` (`rcsr run --json`) that later commands
//! read: the Pareto front, the variables, the position rule and the time range
//! the search trained or selected on.

use crate::devlog::Json;

/// #807: which string of a run.json an evaluator prices. `Exact` is
/// `expression_exact`, the lossless form; `Rounded` is the 4-decimal display
/// `expression` of a run written before that field existed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpressionSource {
    Exact,
    Rounded,
}

impl ExpressionSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ExpressionSource::Exact => "exact",
            ExpressionSource::Rounded => "rounded",
        }
    }

    /// The exact form when present, else the display form.
    pub(crate) fn pick(j: &Json) -> Option<(String, ExpressionSource)> {
        if let Some(e) = j.get("expression_exact").and_then(Json::as_str) {
            return Some((e.to_string(), ExpressionSource::Exact));
        }
        j.get("expression")
            .and_then(Json::as_str)
            .map(|e| (e.to_string(), ExpressionSource::Rounded))
    }

    /// Rounded if either is.
    pub fn worst(self, other: ExpressionSource) -> ExpressionSource {
        if self == ExpressionSource::Rounded || other == ExpressionSource::Rounded {
            ExpressionSource::Rounded
        } else {
            ExpressionSource::Exact
        }
    }
}

/// One `pareto[k]` member, in archive order.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontMember {
    pub rank: usize,
    pub expression: String,
    pub source: ExpressionSource,
    /// #946: `validation_book.band_resolved` and `band_sd`, written under sd units.
    pub band_resolved: Option<f64>,
    pub band_sd: Option<f64>,
}

/// #855: the refusal for a run.json written by an `--interim-chunk` run.
pub const INTERIM_REFUSAL: &str = "run.json comes from an --interim-chunk run: it holds no report; use the run.json of the last chunk";

#[derive(Clone, Debug)]
pub struct RunFile {
    pub path: String,
    pub sha256: String,
    pub variables: Vec<String>,
    pub front: Vec<FrontMember>,
    /// #807: `Rounded` when any front member has no `expression_exact`.
    pub front_source: ExpressionSource,
    /// `target_spec.return_column`.
    pub return_column: Option<String>,
    /// `objective.spec.threshold`: the objective's long rule is `1[pred > threshold]`.
    pub threshold: Option<f64>,
    /// `objective.spec.per_side_bps`.
    pub per_side_bps: Option<f64>,
    /// `objective.spec.position`: the position map the search priced.
    pub position: crate::trading::PositionMap,
    /// #946: the band the search priced (`objective.spec`); band 0 in score units without it.
    pub band: crate::trading::BandSpec,
    /// `objective.spec.min_holding`; 1 without it.
    pub min_holding: usize,
    /// `split.train_time`: `[first, last]` raw time keys.
    pub train_time: Option<(f64, f64)>,
    /// `split.validation_time`: `[first, last]`, absent without a validation slice.
    pub validation_time: Option<(f64, f64)>,
    /// `split.required_purge_seconds`, else `split.purge_seconds`.
    pub purge_seconds: Option<f64>,
    /// `search.candidates_evaluated`.
    pub candidates_evaluated: Option<u64>,
    /// `elapsed_seconds`.
    pub elapsed_seconds: Option<f64>,
    /// `search.cumulative.elapsed_seconds` when the block is complete (#923).
    pub cumulative_elapsed_seconds: Option<f64>,
    /// The parsed document, for views such as `RunFile::book`.
    doc: Json,
}

fn pair(j: Option<&Json>) -> Option<(f64, f64)> {
    match j? {
        Json::Arr(a) if a.len() == 2 => Some((a[0].as_f64()?, a[1].as_f64()?)),
        _ => None,
    }
}

/// `position` of an `objective.spec` object; a run without it priced the threshold rule.
pub(crate) fn position_of(objective_spec: Option<&Json>) -> crate::trading::PositionMap {
    match objective_spec.and_then(|s| s.get("position")).and_then(Json::as_str) {
        Some("top_tercile") => crate::trading::PositionMap::TopTercile,
        _ => crate::trading::PositionMap::Threshold,
    }
}

impl RunFile {
    pub fn load(path: &str) -> Result<RunFile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let sha256 = crate::manifest::sha256_hex(text.as_bytes());
        let mut run = RunFile::parse(&text).map_err(|e| format!("{}: {}", path, e))?;
        run.path = path.to_string();
        run.sha256 = sha256;
        Ok(run)
    }

    pub fn parse(text: &str) -> Result<RunFile, String> {
        let j = Json::parse(text.trim()).ok_or_else(|| "not a run.json document".to_string())?;
        if matches!(j.get("interim"), Some(Json::Bool(true))) {
            return Err(INTERIM_REFUSAL.to_string());
        }
        let variables = match j.get("variables") {
            Some(Json::Arr(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        };
        let front: Vec<FrontMember> = match j.get("pareto") {
            Some(Json::Arr(a)) => a
                .iter()
                .enumerate()
                .filter_map(|(k, m)| {
                    ExpressionSource::pick(m)
                        .map(|(expression, source)| {
                        let vb = m.get("validation_book");
                        let at = |key: &str| vb.and_then(|b| b.get(key)).and_then(Json::as_f64);
                        FrontMember { rank: k + 1, expression, source, band_resolved: at("band_resolved"), band_sd: at("band_sd") }
                    })
                })
                .collect(),
            _ => return Err("no 'pareto' array (not a single-target run.json)".to_string()),
        };
        let front_source = front
            .iter()
            .fold(ExpressionSource::Exact, |acc, m| acc.worst(m.source));
        let split = j.get("split");
        let objective_spec = j.get("objective").and_then(|o| o.get("spec"));
        let (band, min_holding) = crate::run_book::band_of(objective_spec)?;
        let purge = |key: &str| split.and_then(|s| s.get(key)).and_then(Json::as_f64);
        Ok(RunFile {
            path: String::new(),
            sha256: String::new(),
            variables,
            front,
            front_source,
            return_column: j
                .get("target_spec")
                .and_then(|t| t.get("return_column"))
                .and_then(Json::as_str)
                .map(str::to_string),
            threshold: objective_spec.and_then(|s| s.get("threshold")).and_then(Json::as_f64),
            per_side_bps: objective_spec.and_then(|s| s.get("per_side_bps")).and_then(Json::as_f64),
            position: position_of(objective_spec),
            band,
            min_holding,
            train_time: pair(split.and_then(|s| s.get("train_time"))),
            validation_time: pair(split.and_then(|s| s.get("validation_time"))),
            purge_seconds: purge("required_purge_seconds").or_else(|| purge("purge_seconds")),
            candidates_evaluated: j
                .get("search")
                .and_then(|s| s.get("candidates_evaluated"))
                .and_then(Json::as_f64)
                .map(|v| v as u64),
            elapsed_seconds: j.get("elapsed_seconds").and_then(Json::as_f64),
            cumulative_elapsed_seconds: j
                .get("search")
                .and_then(|s| s.get("cumulative"))
                .filter(|c| c.get("complete").map_or(false, |v| matches!(v, Json::Bool(true))))
                .and_then(|c| c.get("elapsed_seconds"))
                .and_then(Json::as_f64),
            doc: j.clone(),
        })
    }

    /// The book the run selected (#487): finalist, fixed threshold, cost and target spec, band and minimum hold (#946).
    pub fn band_book(&self) -> Result<crate::run_book::RunBook, String> {
        crate::run_book::RunBook::from_json(&self.doc)
    }

    /// The declared target spec of the run, when it holds a full one.
    pub fn target_spec(&self) -> Option<crate::target::TargetSpec> {
        crate::target::parse_object(self.doc.get("target_spec")?).ok()
    }

    /// `objective.cost.source`; a run without the block counts as a flag cost.
    pub fn cost_source(&self) -> crate::target::CostSource {
        crate::run_book::cost_source_of(&self.doc)
    }

    /// Last time key the search trained or selected on.
    pub fn last_seen(&self) -> Option<f64> {
        let train = self.train_time?.1;
        Some(self.validation_time.map_or(train, |v| train.max(v.1)))
    }

    /// The first test start that does not overlap what the run saw, or `None`
    /// when the run has no time range.
    pub fn earliest_test_start(&self) -> Option<(f64, f64, f64)> {
        let seen = self.last_seen()?;
        let purge = self.purge_seconds.unwrap_or(0.0);
        Some((seen + purge, seen, purge))
    }
}

impl RunFile {
    /// The #482 rule: a test window that starts before the last row the run saw
    /// plus the purge overlaps what the run trained or selected on. A run without
    /// a time range cannot prove otherwise and counts as in-sample.
    pub fn in_sample(&self, test_start: f64) -> bool {
        self.earliest_test_start().map_or(true, |(earliest, _, _)| test_start < earliest)
    }

    /// Out-of-fold flag per `(label, test_start)` fold. An in-sample fold is an
    /// error naming it, unless `allow`.
    pub fn oof_flags(&self, folds: &[(String, f64)], allow: bool) -> Result<Vec<bool>, String> {
        let flags: Vec<bool> = folds.iter().map(|(_, t)| !self.in_sample(*t)).collect();
        if !allow {
            let bad: Vec<&str> = folds
                .iter()
                .zip(&flags)
                .filter(|(_, ok)| !**ok)
                .map(|((l, _), _)| l.as_str())
                .collect();
            if !bad.is_empty() {
                let why = match self.earliest_test_start() {
                    Some((earliest, seen, purge)) => {
                        format!("test_start < last seen {} + purge {} = {}", seen, purge, earliest)
                    }
                    None => "the run has no split.train_time (no time column)".to_string(),
                };
                return Err(format!(
                    "fold(s) {} test rows the run trained or selected on ({}); \
                     use a run trained before the fold, or --allow-in-sample to label them oof:false",
                    bad.join(", "),
                    why
                ));
            }
        }
        Ok(flags)
    }
}

/// #807: the sealed read prices only the selected tree, so it refuses a run
/// whose strings are the 4-decimal display form.
pub fn rounded_refusal(path: &str) -> String {
    format!(
        "{}: the run has no expression_exact, so the sealed read would price the rounded display string, not the selected tree; \
         rerun the search with a build that writes expression_exact",
        path
    )
}
