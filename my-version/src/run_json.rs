//! #482: the parts of a `run.json` (`rcsr run --json`) that later commands
//! read: the Pareto front, the variables, the position rule and the time range
//! the search trained or selected on.

use crate::devlog::Json;

/// One `pareto[k]` member, in archive order.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontMember {
    pub rank: usize,
    pub expression: String,
}

#[derive(Clone, Debug)]
pub struct RunFile {
    pub path: String,
    pub sha256: String,
    pub variables: Vec<String>,
    pub front: Vec<FrontMember>,
    /// `target_spec.return_column`.
    pub return_column: Option<String>,
    /// `objective.spec.threshold`: the objective's long rule is `1[pred > threshold]`.
    pub threshold: Option<f64>,
    /// `objective.spec.per_side_bps`.
    pub per_side_bps: Option<f64>,
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
    /// The parsed document, for views such as `RunFile::book`.
    doc: Json,
}

fn pair(j: Option<&Json>) -> Option<(f64, f64)> {
    match j? {
        Json::Arr(a) if a.len() == 2 => Some((a[0].as_f64()?, a[1].as_f64()?)),
        _ => None,
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
        let variables = match j.get("variables") {
            Some(Json::Arr(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        };
        let front = match j.get("pareto") {
            Some(Json::Arr(a)) => a
                .iter()
                .enumerate()
                .filter_map(|(k, m)| {
                    m.get("expression")
                        .and_then(Json::as_str)
                        .map(|e| FrontMember { rank: k + 1, expression: e.to_string() })
                })
                .collect(),
            _ => return Err("no 'pareto' array (not a single-target run.json)".to_string()),
        };
        let split = j.get("split");
        let objective_spec = j.get("objective").and_then(|o| o.get("spec"));
        let purge = |key: &str| split.and_then(|s| s.get(key)).and_then(Json::as_f64);
        Ok(RunFile {
            path: String::new(),
            sha256: String::new(),
            variables,
            front,
            return_column: j
                .get("target_spec")
                .and_then(|t| t.get("return_column"))
                .and_then(Json::as_str)
                .map(str::to_string),
            threshold: objective_spec.and_then(|s| s.get("threshold")).and_then(Json::as_f64),
            per_side_bps: objective_spec.and_then(|s| s.get("per_side_bps")).and_then(Json::as_f64),
            train_time: pair(split.and_then(|s| s.get("train_time"))),
            validation_time: pair(split.and_then(|s| s.get("validation_time"))),
            purge_seconds: purge("required_purge_seconds").or_else(|| purge("purge_seconds")),
            candidates_evaluated: j
                .get("search")
                .and_then(|s| s.get("candidates_evaluated"))
                .and_then(Json::as_f64)
                .map(|v| v as u64),
            elapsed_seconds: j.get("elapsed_seconds").and_then(Json::as_f64),
            doc: j.clone(),
        })
    }

    /// The book the run selected (#487): finalist, fixed threshold, cost and target spec.
    pub fn book(&self) -> Result<crate::run_book::RunBook, String> {
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
