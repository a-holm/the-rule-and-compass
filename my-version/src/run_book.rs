//! The book a search run selected (#487), a view over [`crate::run_json::RunFile`].
//!
//! `rcsr --objective pnl|sharpe|log_growth --json` writes the finalist
//! expression, the position rule (`objective.spec`) and the cost beside the
//! declared target spec. [`RunBook::parse`] reads them back so an evaluator
//! prices the same book instead of a book rebuilt from flags.

use crate::devlog::Json;
use crate::target::{self, CostDecl, CostSource, TargetSpec};

#[derive(Clone, Debug)]
pub struct RunBook {
    pub expression: String,
    pub variables: Vec<String>,
    /// Fixed threshold: the position is `1[score > threshold]`.
    pub threshold: f64,
    /// `objective.spec.position`; only the walk-forward prices the top-tercile map.
    pub position: crate::trading::PositionMap,
    /// Total charge per side (cost, fee, half spread and slippage).
    pub per_side_bps: f64,
    pub source: CostSource,
    pub spec: TargetSpec,
    /// The finalist reads no variable.
    pub constant: bool,
}

/// `objective.cost.source`; a run without the block counts as a flag cost.
pub(crate) fn cost_source_of(j: &Json) -> CostSource {
    match j
        .get("objective")
        .and_then(|o| o.get("cost"))
        .and_then(|c| c.get("source"))
        .and_then(Json::as_str)
    {
        Some("target_spec") => CostSource::TargetSpec,
        Some("default") => CostSource::Default,
        _ => CostSource::Flag,
    }
}

impl RunBook {
    pub fn parse(text: &str) -> Result<RunBook, String> {
        crate::run_json::RunFile::parse(text)?.book()
    }

    pub(crate) fn from_json(j: &Json) -> Result<RunBook, String> {
        let obj = match j.get("objective") {
            Some(o @ Json::Obj(_)) => o,
            _ => {
                return Err("run.json has no objective block: run the search with --objective \
                            pnl, sharpe or log_growth"
                    .to_string())
            }
        };
        let spec_json = obj
            .get("spec")
            .ok_or_else(|| "run.json objective has no spec".to_string())?;
        let num = |key: &str| -> Result<f64, String> {
            spec_json
                .get(key)
                .and_then(Json::as_f64)
                .ok_or_else(|| format!("run.json objective.spec has no number '{}'", key))
        };
        if spec_json.get("tau").is_some_and(|t| !matches!(t, Json::Null)) {
            return Err("run.json objective.spec has a tau: only the indicator position is priced here".to_string());
        }
        if num("max_position")? != 1.0 {
            return Err("run.json objective.spec has max_position != 1: only the unit indicator is priced here".to_string());
        }
        if num("min_holding")? != 1.0 {
            return Err("run.json objective.spec has min_holding > 1: the walk-forward threshold rule has no minimum hold".to_string());
        }
        if num("cash_return")? != 0.0 {
            return Err("run.json objective.spec has a cash_return: the walk-forward book earns nothing while flat".to_string());
        }
        if spec_json.get("funding_col").is_some_and(|t| !matches!(t, Json::Null)) {
            return Err("run.json objective.spec has a funding column: not priced here".to_string());
        }
        let finalist = obj
            .get("finalist")
            .ok_or_else(|| "run.json objective has no finalist".to_string())?;
        let expression = finalist
            .get("expression")
            .and_then(Json::as_str)
            .ok_or_else(|| "run.json objective.finalist has no expression".to_string())?
            .to_string();
        let variables = match j.get("variables") {
            Some(Json::Arr(a)) => a
                .iter()
                .map(|v| v.as_str().map(str::to_string).ok_or_else(|| "'variables' must hold strings".to_string()))
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("run.json has no variables list".to_string()),
        };
        let target_spec = j
            .get("target_spec")
            .filter(|t| matches!(t, Json::Obj(_)))
            .ok_or_else(|| "run.json has no target_spec".to_string())?;
        let spec = target::parse_object(target_spec).map_err(|e| format!("run.json target_spec: {}", e))?;
        let source = cost_source_of(j);
        let constant = finalist
            .get("constant")
            .map(|c| matches!(c, Json::Bool(true)))
            .unwrap_or(false);
        Ok(RunBook {
            expression,
            variables,
            threshold: num("threshold")?,
            position: crate::run_json::position_of(Some(spec_json)),
            per_side_bps: num("per_side_bps")?,
            source,
            spec,
            constant,
        })
    }

    pub fn cost(&self) -> CostDecl {
        CostDecl::new(Some(&self.spec), self.per_side_bps, self.source)
    }
}
