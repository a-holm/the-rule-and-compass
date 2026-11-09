//! Post-loop re-selection (#928): score a finished front under a selection
//! rule without running a search.

use crate::data::Sample;
use crate::devlog::{text, Json};
use crate::metrics::Metric;
use crate::pareto::Archive;
use crate::provenance::BuildIdentity;
use crate::search::Outcome;

/// One `pareto[]` member of a run.json.
#[derive(Clone, Debug, PartialEq)]
pub struct RunMember {
    pub archive_rank: usize,
    pub expression_exact: String,
    pub train_error: f64,
    /// The reported complexity (the configured model's value).
    pub complexity: f64,
}

/// The parts of a run.json that re-selection reads.
#[derive(Clone, Debug)]
pub struct RunFront {
    pub path: String,
    pub sha256: String,
    pub variables: Vec<String>,
    pub split_mode: Option<String>,
    pub train_rows: Option<usize>,
    pub generations_run: usize,
    pub candidates_evaluated: u64,
    /// In `archive_rank` order.
    pub members: Vec<RunMember>,
}

impl RunFront {
    pub fn load(path: &str) -> Result<RunFront, String> {
        let body = std::fs::read_to_string(path).map_err(|e| format!("cannot read run.json '{}': {}", path, e))?;
        let sha256 = crate::manifest::sha256_hex(body.as_bytes());
        let j = Json::parse(body.trim()).ok_or_else(|| format!("'{}' is not a run.json document", path))?;
        if matches!(j.get("interim"), Some(Json::Bool(true))) {
            return Err(crate::run_json::INTERIM_REFUSAL.to_string());
        }
        let variables = match j.get("variables") {
            Some(Json::Arr(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => return Err(format!("'{}' has no 'variables' array", path)),
        };
        let Some(Json::Arr(pareto)) = j.get("pareto") else {
            return Err(format!("'{}' has no 'pareto' array (not a single-target run.json)", path));
        };
        let mut members = Vec::with_capacity(pareto.len());
        for (k, m) in pareto.iter().enumerate() {
            let archive_rank = m
                .get("archive_rank")
                .and_then(Json::as_usize)
                .ok_or_else(|| format!("'{}' pareto[{}] has no archive_rank", path, k))?;
            let expression_exact = m
                .get("expression_exact")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("'{}' pareto[{}] has no expression_exact", path, k))?
                .to_string();
            let train_error = m.get("train_error").and_then(Json::as_f64).unwrap_or(f64::INFINITY);
            let complexity = m
                .get("complexity")
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("'{}' pareto[{}] has no complexity", path, k))?;
            members.push(RunMember { archive_rank, expression_exact, train_error, complexity });
        }
        members.sort_by_key(|m| m.archive_rank);
        let split = j.get("split");
        Ok(RunFront {
            path: path.to_string(),
            sha256,
            variables,
            split_mode: split.and_then(|s| s.get("mode")).and_then(Json::as_str).map(str::to_string),
            train_rows: j.get("train_rows").and_then(Json::as_usize),
            generations_run: j.get("generations_run").and_then(Json::as_usize).unwrap_or(0),
            candidates_evaluated: j
                .get("search")
                .and_then(|s| s.get("candidates_evaluated"))
                .and_then(Json::as_f64)
                .map_or(0, |v| v as u64),
            members,
        })
    }
}

/// An [`Outcome`] for a front that no search produced. Nothing runs: the
/// counters are the ones of the run that made the front.
pub fn outcome(
    archive: Archive,
    test: &Sample,
    metric: Metric,
    generations: usize,
    candidates_evaluated: u64,
) -> Outcome {
    let test_errors = archive
        .sorted()
        .iter()
        .map(|s| crate::search::eval_error(&s.tree, test, metric))
        .collect();
    Outcome {
        archive,
        test_errors,
        generations,
        converged_early: false,
        seeds: Vec::new(),
        convergence: Default::default(),
        fit_samples: 0,
        fit_rows: 0,
        archive_refit: false,
        predictor_samples: 0,
        backend: crate::backend::Kind::Cpu,
        backend_reason: None,
        device: None,
        gpu: None,
        dispatch: Default::default(),
        qd: None,
        stop_reason: "generations",
        prune: Default::default(),
        candidates_evaluated,
        fixed_position_fallback: false,
        interim: false,
    }
}

fn identity_json(i: &BuildIdentity) -> String {
    format!(
        "{{\"version\":{},\"commit\":{},\"binary_sha256\":{},\"device\":{},\"driver\":{}}}",
        text(&i.version),
        text(&i.commit),
        text(&i.binary_sha256),
        text(&i.device),
        text(&i.driver)
    )
}

/// `objective.selection_rule.reselected_from` of a re-selected checkpoint.
/// `saved` is `None` for a checkpoint written before the identity existed.
pub fn from_checkpoint_json(path: &str, sha256: &str, saved: Option<&BuildIdentity>, now: &BuildIdentity) -> String {
    let differs = saved.is_some_and(|s| !s.refusing_differences(now).is_empty());
    format!(
        "{{\"kind\":\"checkpoint\",\"path\":{},\"sha256\":{},\"build_differs\":{},\"saved_build\":{},\"current_build\":{}}}",
        text(path),
        text(sha256),
        differs,
        saved.map_or_else(|| "null".to_string(), identity_json),
        identity_json(now)
    )
}

/// `objective.selection_rule.reselected_from` of a re-selected run.json.
pub fn from_run_json(path: &str, sha256: &str) -> String {
    format!("{{\"kind\":\"run\",\"path\":{},\"sha256\":{}}}", text(path), text(sha256))
}

/// The front a checkpoint reported: its post-refit exact `final_archive`,
/// never the search archive.
pub fn front_of(cp: &crate::checkpoint::Checkpoint) -> Result<Archive, String> {
    Some(&cp.archive)
        .map(|a| Archive { items: a.items.clone(), cap: a.cap })
        .ok_or_else(|| "the checkpoint has no final archive (a mid-run checkpoint): run it to the end first".to_string())
}
