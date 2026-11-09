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
    /// #807: `Exact` when the finalist came from `expression_exact`.
    pub expression_source: crate::run_json::ExpressionSource,
    pub variables: Vec<String>,
    /// Fixed threshold: the position is `1[score > threshold]`.
    pub threshold: f64,
    /// `objective.spec.position`; only the walk-forward prices the top-tercile map.
    pub position: crate::trading::PositionMap,
    /// Total charge per side (cost, fee, half spread and slippage).
    pub per_side_bps: f64,
    pub source: CostSource,
    pub spec: TargetSpec,
    /// #946: the band the search priced; `resolved` is its half-width in score
    /// units, taken as recorded and never resolved again from other rows.
    pub band: crate::trading::BandSpec,
    /// `objective.spec.min_holding`.
    pub min_holding: usize,
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

/// #946: the band and the minimum hold of an `objective.spec`. A run written
/// before #925 has neither key and reads as band 0 in score units, hold 1.
pub(crate) fn band_of(spec: Option<&Json>) -> Result<(crate::trading::BandSpec, usize), String> {
    use crate::trading::{BandSpec, BandUnits};
    let Some(spec) = spec else { return Ok((BandSpec::default(), 1)) };
    let number = |key: &str| -> Result<Option<f64>, String> {
        match spec.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) => Ok(Some(x)),
                None => Err(format!("run.json objective.spec '{}' is not a number", key)),
            },
        }
    };
    let min_holding = match number("min_holding")? {
        None => 1,
        Some(h) if h >= 1.0 && h.fract() == 0.0 && h < 1e9 => h as usize,
        Some(h) => return Err(format!("run.json objective.spec min_holding {} is not an integer >= 1", h)),
    };
    let multiple = number("band")?.unwrap_or(0.0);
    let resolved = number("band_resolved")?;
    let units = match spec.get("band_units") {
        None | Some(Json::Null) => {
            if resolved.is_some() {
                return Err("run.json objective.spec has band_resolved but no band_units".to_string());
            }
            BandUnits::Score
        }
        Some(u) => BandUnits::parse(u.as_str().unwrap_or_default()).map_err(|e| format!("run.json objective.spec band_units: {}", e))?,
    };
    if !multiple.is_finite() || multiple < 0.0 {
        return Err(format!("run.json objective.spec band {} is not finite and >= 0", multiple));
    }
    let resolved = match (units, resolved) {
        (BandUnits::Score, r) => r.unwrap_or(multiple),
        (BandUnits::Sd, Some(r)) => r,
        (BandUnits::Sd, None) => return Err("run.json objective.spec has band_units sd but no band_resolved".to_string()),
    };
    if !resolved.is_finite() || resolved < 0.0 {
        return Err(format!("run.json objective.spec band_resolved {} is not finite and >= 0", resolved));
    }
    let sd = number("band_sd")?;
    let sd_rows = number("band_sd_rows")?.map(|r| r as usize);
    let sd_max_time = number("band_sd_max_time")?;
    Ok((BandSpec { units, multiple, sd, sd_rows, sd_max_time, resolved }, min_holding))
}

impl RunBook {
    /// The book of a run for evaluators that price the plain threshold rule
    /// only: a run that priced a band or a minimum hold is refused.
    pub fn parse(text: &str) -> Result<RunBook, String> {
        crate::run_json::RunFile::parse(text)?.book()
    }

    /// `Err` when the run priced a band or a minimum hold (#946).
    pub(crate) fn threshold_only(self) -> Result<RunBook, String> {
        if self.has_band() {
            return Err("run.json objective.spec has a band or a min_holding > 1: only walkforward --from-run prices that book".to_string());
        }
        Ok(self)
    }

    /// The run priced a no-trade band or a minimum hold.
    pub fn has_band(&self) -> bool {
        self.band.resolved > 0.0 || self.min_holding > 1
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
        if num("cash_return")? != 0.0 {
            return Err("run.json objective.spec has a cash_return: the walk-forward book earns nothing while flat".to_string());
        }
        if spec_json.get("funding_col").is_some_and(|t| !matches!(t, Json::Null)) {
            return Err("run.json objective.spec has a funding column: not priced here".to_string());
        }
        let (band, min_holding) = band_of(Some(spec_json))?;
        let finalist = obj
            .get("finalist")
            .ok_or_else(|| "run.json objective has no finalist".to_string())?;
        let (expression, expression_source) = crate::run_json::ExpressionSource::pick(finalist)
            .ok_or_else(|| "run.json objective.finalist has no expression".to_string())?;
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
            expression_source,
            variables,
            threshold: num("threshold")?,
            position: crate::run_json::position_of(Some(spec_json)),
            per_side_bps: num("per_side_bps")?,
            source,
            spec,
            band,
            min_holding,
            constant,
        })
    }

    pub fn cost(&self) -> CostDecl {
        CostDecl::new(Some(&self.spec), self.per_side_bps, self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(spec: &str) -> Result<(crate::trading::BandSpec, usize), String> {
        let j = Json::parse(spec).unwrap();
        band_of(Some(&j))
    }

    #[test]
    fn a_spec_without_band_keys_reads_as_the_threshold_book() {
        let (b, hold) = band("{\"min_holding\":1}").unwrap();
        assert_eq!((b, hold), (crate::trading::BandSpec::default(), 1));
        assert_eq!(band_of(None).unwrap().1, 1);
    }

    #[test]
    fn band_keys_are_read_as_recorded() {
        let (b, hold) = band(
            "{\"min_holding\":24,\"band\":1.5,\"band_units\":\"sd\",\"band_sd\":0.5,\"band_resolved\":0.75,\
             \"band_sd_rows\":100,\"band_sd_max_time\":9.0}",
        )
        .unwrap();
        assert_eq!(hold, 24);
        assert_eq!(b.units, crate::trading::BandUnits::Sd);
        assert_eq!((b.multiple, b.sd, b.resolved, b.sd_rows, b.sd_max_time), (1.5, Some(0.5), 0.75, Some(100), Some(9.0)));
        let (s, _) = band("{\"band\":0.2,\"band_units\":\"score\"}").unwrap();
        assert_eq!(s.resolved, 0.2);
    }

    #[test]
    fn a_malformed_band_block_is_refused() {
        for bad in [
            "{\"band_resolved\":0.5}",
            "{\"band\":1.0,\"band_units\":\"sd\"}",
            "{\"band\":1.0,\"band_units\":\"furlongs\",\"band_resolved\":1.0}",
            "{\"band\":-1.0,\"band_units\":\"score\",\"band_resolved\":-1.0}",
            "{\"min_holding\":0}",
            "{\"min_holding\":2.5}",
        ] {
            assert!(band(bad).is_err(), "{bad}");
        }
    }
}
