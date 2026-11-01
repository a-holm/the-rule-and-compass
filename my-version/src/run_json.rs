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
            doc: j.clone(),
        })
    }

    /// The book the run selected (#487): finalist, fixed threshold, cost and target spec.
    pub fn book(&self) -> Result<crate::run_book::RunBook, String> {
        crate::run_book::RunBook::from_json(&self.doc)
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
