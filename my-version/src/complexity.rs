//! Configurable expression complexity (issue #13).
//!
//! The default model reproduces the historical node count exactly:
//!
//! ```text
//! complexity(tree) = sum(op_cost) + var_weight * n_vars
//!                  + const_weight * n_consts + depth_weight * max_depth
//! ```
//!
//! `sum(op_cost)` runs over **operator** nodes only (every node that is neither
//! [`Op::Const`] nor [`Op::Input`]); terminals are priced by the variable and
//! constant weights. `max_depth` is the number of edges on the longest path
//! from the root to a node, so a single terminal has depth `0`. With the
//! defaults — every operator cost `1`, `var_weight = const_weight = 1` and
//! `depth_weight = 0` — the value is the number of nodes, bit for bit.
//!
//! The model is stored with the search configuration and evaluated on every
//! candidate, so [`ComplexityModel::complexity`] never allocates.

use crate::expr::{Node, Op};

/// A per-operator, per-terminal, depth-weighted complexity model.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexityModel {
    /// Cost of each operator node, indexed by `Op as usize`. Terminals
    /// (`Const`, `Input`) are ignored here; they use the weights below.
    op_costs: Vec<f64>,
    var_weight: f64,
    const_weight: f64,
    depth_weight: f64,
}

impl Default for ComplexityModel {
    fn default() -> Self {
        ComplexityModel {
            op_costs: vec![1.0; Op::COUNT],
            var_weight: 1.0,
            const_weight: 1.0,
            depth_weight: 0.0,
        }
    }
}

impl ComplexityModel {
    /// The default model: one per node, no depth penalty.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cost of one operator node. An operator outside the table (which cannot
    /// happen for the current [`Op`] set) falls back to the default of `1`.
    pub fn op_cost(&self, op: Op) -> f64 {
        self.op_costs.get(op as usize).copied().unwrap_or(1.0)
    }

    pub fn var_weight(&self) -> f64 {
        self.var_weight
    }

    pub fn const_weight(&self) -> f64 {
        self.const_weight
    }

    pub fn depth_weight(&self) -> f64 {
        self.depth_weight
    }

    pub fn set_op_cost(&mut self, op: Op, cost: f64) {
        if let Some(slot) = self.op_costs.get_mut(op as usize) {
            *slot = cost;
        }
    }

    /// Set the nesting-depth weight. Rejects negative, non-finite values.
    pub fn set_depth_weight(&mut self, weight: f64) -> Result<(), String> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "the depth weight must be a finite number >= 0, got '{}'",
                weight
            ));
        }
        self.depth_weight = weight;
        Ok(())
    }

    /// Parse a `--op-complexity`-style spec: `pow=2,sin=1.5,if=3`.
    ///
    /// Operator names use the same table as `--ops` ([`Op::from_name`]);
    /// unlisted operators keep their default cost of `1`. An unknown name,
    /// a missing `=`, an empty entry, or a cost that is not positive and
    /// finite is an error naming the offending token.
    pub fn apply_op_costs(&mut self, spec: &str) -> Result<(), String> {
        for raw in spec.split(',') {
            let token = raw.trim();
            if token.is_empty() {
                return Err(format!(
                    "empty entry in --op-complexity '{}'; expected name=cost",
                    spec
                ));
            }
            let (name, value) = token.split_once('=').ok_or_else(|| {
                format!(
                    "malformed --op-complexity entry '{}': expected name=cost",
                    token
                )
            })?;
            let name = name.trim();
            let op = Op::from_name(name)
                .ok_or_else(|| format!("unknown operator '{}' in '{}'", name, token))?;
            let cost = parse_positive(value).map_err(|why| {
                format!(
                    "invalid cost in --op-complexity entry '{}': {}",
                    token, why
                )
            })?;
            self.set_op_cost(op, cost);
        }
        Ok(())
    }

    /// Parse a `--complexity-weights`-style spec: `var=1,const=1`.
    ///
    /// Keys are `var` (aliases `variable`, `input`), `const` (`constant`) and
    /// `depth` (the same as `--depth-weight`). Weights must be positive and
    /// finite; the depth weight may also be zero.
    pub fn apply_weights(&mut self, spec: &str) -> Result<(), String> {
        for raw in spec.split(',') {
            let token = raw.trim();
            if token.is_empty() {
                return Err(format!(
                    "empty entry in --complexity-weights '{}'; expected name=weight",
                    spec
                ));
            }
            let (name, value) = token.split_once('=').ok_or_else(|| {
                format!(
                    "malformed --complexity-weights entry '{}': expected name=weight",
                    token
                )
            })?;
            let key = name.trim().to_ascii_lowercase();
            match key.as_str() {
                "var" | "variable" | "input" => {
                    self.var_weight = parse_positive(value).map_err(|why| {
                        format!(
                            "invalid weight in --complexity-weights entry '{}': {}",
                            token, why
                        )
                    })?;
                }
                "const" | "constant" => {
                    self.const_weight = parse_positive(value).map_err(|why| {
                        format!(
                            "invalid weight in --complexity-weights entry '{}': {}",
                            token, why
                        )
                    })?;
                }
                "depth" => {
                    self.set_depth_weight(parse_non_negative(value).map_err(|why| {
                        format!(
                            "invalid weight in --complexity-weights entry '{}': {}",
                            token, why
                        )
                    })?)?;
                }
                _ => {
                    return Err(format!(
                        "unknown weight '{}' in '{}'; expected var, const or depth",
                        name.trim(),
                        token
                    ));
                }
            }
        }
        Ok(())
    }

    /// The configured complexity of a tree. No allocation: one recursive walk.
    pub fn complexity(&self, n: &Node) -> f64 {
        let (node_cost, max_depth) = self.walk(n, 0);
        node_cost + self.depth_weight * max_depth as f64
    }

    /// `(sum of node costs, deepest depth reached)`, depth counted as edges.
    fn walk(&self, n: &Node, depth: usize) -> (f64, usize) {
        let cost = match n.op {
            Op::Const => self.const_weight,
            Op::Input => self.var_weight,
            op => self.op_cost(op),
        };
        let mut total = cost;
        let mut deepest = depth;
        for kid in &n.kids {
            let (child_cost, child_depth) = self.walk(kid, depth + 1);
            total += child_cost;
            if child_depth > deepest {
                deepest = child_depth;
            }
        }
        (total, deepest)
    }
}

fn parse_positive(raw: &str) -> Result<f64, String> {
    let v: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("'{}' is not a number", raw.trim()))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!(
            "'{}' must be a positive finite number",
            raw.trim()
        ));
    }
    Ok(v)
}

fn parse_non_negative(raw: &str) -> Result<f64, String> {
    let v: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("'{}' is not a number", raw.trim()))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!(
            "'{}' must be a finite number >= 0",
            raw.trim()
        ));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_count_covers_every_variant() {
        // `Op::COUNT` indexes a table by discriminant; a new variant at the end
        // must bump it. `Op::Constrain` is the last variant today.
        assert_eq!(Op::Constrain as usize + 1, Op::COUNT);
    }

    #[test]
    fn parsing_names_the_bad_token() {
        let mut m = ComplexityModel::new();
        let err = m.apply_op_costs("pow=2,nope=1").unwrap_err();
        assert!(err.contains("nope"), "{}", err);
        let err = m.apply_op_costs("pow").unwrap_err();
        assert!(err.contains("pow"), "{}", err);
        let err = m.apply_op_costs("pow=-1").unwrap_err();
        assert!(err.contains("pow=-1"), "{}", err);
        let err = m.apply_weights("var=0").unwrap_err();
        assert!(err.contains("var=0"), "{}", err);
        let err = m.apply_weights("bogus=1").unwrap_err();
        assert!(err.contains("bogus"), "{}", err);
    }
}
