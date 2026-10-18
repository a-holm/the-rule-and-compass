//! Dimensional analysis for expression trees (issue #25).
//!
//! A **unit** is a vector of integer exponents over seven SI base dimensions:
//! `m`, `s`, `kg`, `A`, `K`, `mol` and `cd`. Multiplication adds exponents,
//! division subtracts them, and an integer power multiplies them. Anything
//! that is not a declared input has the zero vector, i.e. is dimensionless.
//!
//! [`infer`] is a single bottom-up pass over a [`Node`]. It returns either the
//! root unit or the first [`Inconsistency`] it meets. The search asks exactly
//! one question through [`Units::accepts`]: does the root dimension equal the
//! target's? See `docs/units.md` for the full algebra and the caveats.
//!
//! The parser accepts the base dimensions above, `*`, `/`, integer powers with
//! `^`, parentheses and the empty unit `1`:
//!
//! ```
//! use rcsr::units::Unit;
//! assert_eq!(Unit::parse("m/s^2").unwrap().to_string(), "m/s^2");
//! // The canonical form follows BASE_DIMENSIONS, so kg comes after m and s.
//! assert_eq!(Unit::parse("kg*m^2/s^2").unwrap().to_string(), "m^2*kg/s^2");
//! assert_eq!(Unit::parse("1").unwrap(), Unit::DIMENSIONLESS);
//! ```

use crate::expr::{Node, Op};

/// The seven base dimensions, in the fixed order of [`Unit::exponents`].
pub const BASE_DIMENSIONS: [&str; 7] = ["m", "s", "kg", "A", "K", "mol", "cd"];

/// An SI dimension: one integer exponent per entry of [`BASE_DIMENSIONS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct Unit {
    exps: [i32; 7],
}

impl Unit {
    /// The empty unit, written `1`.
    pub const DIMENSIONLESS: Unit = Unit { exps: [0; 7] };

    /// Build a unit directly from exponents in [`BASE_DIMENSIONS`] order.
    pub const fn from_exponents(exps: [i32; 7]) -> Unit {
        Unit { exps }
    }

    /// The exponents in [`BASE_DIMENSIONS`] order.
    pub fn exponents(&self) -> [i32; 7] {
        self.exps
    }

    /// The unit named by one base dimension, or `None` for an unknown name.
    pub fn base(name: &str) -> Option<Unit> {
        let i = BASE_DIMENSIONS.iter().position(|b| *b == name)?;
        let mut exps = [0; 7];
        exps[i] = 1;
        Some(Unit { exps })
    }

    /// True when every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        self.exps == [0; 7]
    }

    /// `self * rhs`: add the exponents.
    pub fn mul(self, rhs: Unit) -> Unit {
        let mut out = [0; 7];
        for i in 0..7 {
            out[i] = self.exps[i].saturating_add(rhs.exps[i]);
        }
        Unit { exps: out }
    }

    /// `self / rhs`: subtract the exponents.
    pub fn div(self, rhs: Unit) -> Unit {
        let mut out = [0; 7];
        for i in 0..7 {
            out[i] = self.exps[i].saturating_sub(rhs.exps[i]);
        }
        Unit { exps: out }
    }

    /// `self ^ n`: multiply every exponent by `n`.
    pub fn pow(self, n: i32) -> Unit {
        let mut out = [0; 7];
        for i in 0..7 {
            out[i] = self.exps[i].saturating_mul(n);
        }
        Unit { exps: out }
    }

    /// Parse an SI-ish unit expression.
    ///
    /// Grammar:
    ///
    /// ```text
    /// expr   := factor (('*' | '/') factor)*
    /// factor := atom ('^' [+-]? digits)?
    /// atom   := base | '1' | '(' expr ')'
    /// ```
    pub fn parse(src: &str) -> Result<Unit, String> {
        let mut p = Parser {
            src: src.as_bytes(),
            pos: 0,
            text: src,
        };
        let u = p.expr()?;
        p.skip_ws();
        if p.pos != p.src.len() {
            return Err(format!(
                "unexpected '{}' at column {} in unit '{}'",
                &src[p.pos..],
                p.pos + 1,
                src
            ));
        }
        Ok(u)
    }

    /// Canonical text form, e.g. `kg*m^2/s^2` or `1`.
    pub fn encode(self) -> String {
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        for (i, &e) in self.exps.iter().enumerate() {
            if e > 0 {
                positive.push(power(BASE_DIMENSIONS[i], e));
            } else if e < 0 {
                negative.push(power(BASE_DIMENSIONS[i], -e));
            }
        }
        let mut out = if positive.is_empty() {
            "1".to_string()
        } else {
            positive.join("*")
        };
        // One `/` per negative factor keeps the form unambiguous under the
        // left-associative grammar: `mol/(A*K)` prints as `mol/A/K`, which
        // parses back to the same exponents.
        for n in negative {
            out.push('/');
            out.push_str(&n);
        }
        out
    }
}

fn power(name: &str, e: i32) -> String {
    if e == 1 {
        name.to_string()
    } else {
        format!("{}^{}", name, e)
    }
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&Unit::encode(*self))
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    text: &'a str,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }

    fn expr(&mut self) -> Result<Unit, String> {
        let mut out = self.factor()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    out = out.mul(self.factor()?);
                }
                Some(b'/') => {
                    self.pos += 1;
                    out = out.div(self.factor()?);
                }
                _ => return Ok(out),
            }
        }
    }

    fn factor(&mut self) -> Result<Unit, String> {
        let mut out = self.atom()?;
        if self.peek() == Some(b'^') {
            self.pos += 1;
            out = out.pow(self.integer()?);
        }
        Ok(out)
    }

    fn atom(&mut self) -> Result<Unit, String> {
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let inner = self.expr()?;
                if self.peek() != Some(b')') {
                    return Err(format!(
                        "missing ')' at column {} in unit '{}'",
                        self.pos + 1,
                        self.text
                    ));
                }
                self.pos += 1;
                Ok(inner)
            }
            Some(b'1') => {
                self.pos += 1;
                Ok(Unit::DIMENSIONLESS)
            }
            Some(c) if (c as char).is_ascii_alphabetic() => {
                let start = self.pos;
                while self.pos < self.src.len()
                    && (self.src[self.pos] as char).is_ascii_alphabetic()
                {
                    self.pos += 1;
                }
                let name = &self.text[start..self.pos];
                Unit::base(name).ok_or_else(|| {
                    format!(
                        "unknown unit '{}' at column {} (base dimensions: {})",
                        name,
                        start + 1,
                        BASE_DIMENSIONS.join(", ")
                    )
                })
            }
            _ => Err(format!(
                "expected a unit name, '1' or '(' at column {} in unit '{}'",
                self.pos + 1,
                self.text
            )),
        }
    }

    fn integer(&mut self) -> Result<i32, String> {
        self.skip_ws();
        let start = self.pos;
        if matches!(self.src.get(self.pos), Some(b'+') | Some(b'-')) {
            self.pos += 1;
        }
        let digits = self.pos;
        while self.pos < self.src.len() && (self.src[self.pos] as char).is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == digits {
            return Err(format!(
                "expected an integer exponent at column {} in unit '{}'",
                start + 1,
                self.text
            ));
        }
        self.text[start..self.pos].parse::<i32>().map_err(|_| {
            format!(
                "exponent at column {} is out of range in unit '{}'",
                start + 1,
                self.text
            )
        })
    }
}

/// Why a tree has no single dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Inconsistency {
    /// A binary or branch operator required equal dimensions.
    Mismatch {
        op: Op,
        left: Unit,
        right: Unit,
    },
    /// The operator requires a dimensionless argument.
    NotDimensionless { op: Op, actual: Unit },
    /// `pow` with a dimensional base needs a constant integer exponent.
    NonIntegerPower { base: Unit },
    /// `sqrt` of a unit whose exponents are not all even.
    NonIntegerRoot { actual: Unit },
}

impl std::fmt::Display for Inconsistency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inconsistency::Mismatch { op, left, right } => write!(
                f,
                "{} combines incompatible dimensions {} and {}",
                op.symbol(),
                left,
                right
            ),
            Inconsistency::NotDimensionless { op, actual } => write!(
                f,
                "{} requires a dimensionless argument, got {}",
                op.symbol(),
                actual
            ),
            Inconsistency::NonIntegerPower { base } => write!(
                f,
                "a dimensional base ({}) needs a constant integer exponent",
                base
            ),
            Inconsistency::NonIntegerRoot { actual } => write!(
                f,
                "sqrt({}) has no integer exponents (all exponents must be even)",
                actual
            ),
        }
    }
}

/// Infer the root unit of `tree`, with input `i` taking `vars[i]`.
///
/// An input index outside `vars` is treated as dimensionless; evaluation of
/// such a tree returns `NaN`, so the dimensional question is moot.
pub fn infer(tree: &Node, vars: &[Unit]) -> Result<Unit, Inconsistency> {
    use Op::*;
    let d = |n: &Node| infer(n, vars);
    let equal = |op: Op, a: Unit, b: Unit| -> Result<Unit, Inconsistency> {
        if a == b {
            Ok(a)
        } else {
            Err(Inconsistency::Mismatch {
                op,
                left: a,
                right: b,
            })
        }
    };
    let dimensionless = |op: Op, u: Unit| -> Result<Unit, Inconsistency> {
        if u.is_dimensionless() {
            Ok(u)
        } else {
            Err(Inconsistency::NotDimensionless { op, actual: u })
        }
    };
    match tree.op {
        Const => Ok(Unit::DIMENSIONLESS),
        Input => Ok(vars
            .get(tree.val as usize)
            .copied()
            .unwrap_or(Unit::DIMENSIONLESS)),
        // Sign-preserving / magnitude-preserving operators.
        Neg | Abs => d(&tree.kids[0]),
        // `sqrt` halves every exponent; all exponents must stay integral.
        Sqrt => {
            let a = d(&tree.kids[0])?;
            if a.exps.iter().any(|e| e % 2 != 0) {
                return Err(Inconsistency::NonIntegerRoot { actual: a });
            }
            let mut out = [0; 7];
            for i in 0..7 {
                out[i] = a.exps[i] / 2;
            }
            Ok(Unit { exps: out })
        }
        Square => Ok(d(&tree.kids[0])?.pow(2)),
        Cube => Ok(d(&tree.kids[0])?.pow(3)),
        // Transcendental and shape-based primitives need a pure number.
        Exp | Ln | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Gauss | Step | Floor
        | Ceil | Round | Sign | Erf | Erfc | Asinh | Acosh | Atanh | Not | Require
        | Constrain => {
            let a = d(&tree.kids[0])?;
            dimensionless(tree.op, a)
        }
        // Same-dimension binary operators.
        Add | Sub | Min | Max | Mod => {
            let a = d(&tree.kids[0])?;
            let b = d(&tree.kids[1])?;
            equal(tree.op, a, b)
        }
        Mul => Ok(d(&tree.kids[0])?.mul(d(&tree.kids[1])?)),
        Div => Ok(d(&tree.kids[0])?.div(d(&tree.kids[1])?)),
        // The exponent is a pure number. A dimensionless base stays
        // dimensionless; a dimensional base needs a constant integer power.
        Pow => {
            let exponent = d(&tree.kids[1])?;
            dimensionless(Pow, exponent)?;
            let base = d(&tree.kids[0])?;
            if base.is_dimensionless() {
                return Ok(base);
            }
            match constant_integer(&tree.kids[1]) {
                Some(n) => Ok(base.pow(n)),
                None => Err(Inconsistency::NonIntegerPower { base }),
            }
        }
        // atan2 takes two lengths of the same dimension and returns an angle,
        // which is dimensionless here.
        Atan2 => {
            let a = d(&tree.kids[0])?;
            let b = d(&tree.kids[1])?;
            equal(Atan2, a, b)?;
            Ok(Unit::DIMENSIONLESS)
        }
        // Conditions are truth values; both branches must agree.
        If => {
            let c = d(&tree.kids[0])?;
            dimensionless(If, c)?;
            let a = d(&tree.kids[1])?;
            let b = d(&tree.kids[2])?;
            equal(If, a, b)
        }
        // Comparisons return 0/1, so their result is dimensionless even
        // though their operands must share a dimension.
        Equal | Less | LessOrEqual | Greater | GreaterOrEqual => {
            let a = d(&tree.kids[0])?;
            let b = d(&tree.kids[1])?;
            equal(tree.op, a, b)?;
            Ok(Unit::DIMENSIONLESS)
        }
        And | Or | Xor => {
            let a = d(&tree.kids[0])?;
            dimensionless(tree.op, a)?;
            let b = d(&tree.kids[1])?;
            dimensionless(tree.op, b)?;
            Ok(Unit::DIMENSIONLESS)
        }
        // The bounds and the tested value are all truth-valued constraints;
        // they must be pure numbers, and the operator yields `v`.
        Between => {
            let v = d(&tree.kids[0])?;
            dimensionless(Between, v)?;
            let lo = d(&tree.kids[1])?;
            dimensionless(Between, lo)?;
            let hi = d(&tree.kids[2])?;
            dimensionless(Between, hi)?;
            Ok(Unit::DIMENSIONLESS)
        }
        // Meta operators describe the tree's shape, not a row, and yield a
        // pure number. Their argument is deliberately not dimension-checked.
        Complexity | Variables | Constants | Contains | ContainsOp => Ok(Unit::DIMENSIONLESS),
    }
}

/// The value of a constant integer subtree, if it is one.
fn constant_integer(n: &Node) -> Option<i32> {
    if n.op != Op::Const || !n.val.is_finite() {
        return None;
    }
    let r = n.val.round();
    if (n.val - r).abs() > 1e-9 || r < i32::MIN as f64 || r > i32::MAX as f64 {
        return None;
    }
    Some(r as i32)
}

/// Units declared for one search: one per input index, plus the target's.
///
/// An input without a declaration is [`Unit::DIMENSIONLESS`]; this is why the
/// whole feature is opt-in.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Units {
    /// Input units in input order. Missing entries are dimensionless.
    pub vars: Vec<Unit>,
    /// The unit of the target column.
    pub target: Unit,
}

impl Units {
    /// Construct a declaration.
    pub fn new(vars: Vec<Unit>, target: Unit) -> Units {
        Units { vars, target }
    }

    /// The unit of the tree's root, or the first inconsistency.
    pub fn root_dimension(&self, tree: &Node) -> Result<Unit, Inconsistency> {
        infer(tree, &self.vars)
    }

    /// The single predicate the search calls: a tree is accepted exactly when
    /// its root dimension equals the target's. Inconsistent trees are rejected.
    pub fn accepts(&self, tree: &Node) -> bool {
        match self.root_dimension(tree) {
            Ok(d) => d == self.target,
            Err(_) => false,
        }
    }

    /// Text form used by the checkpoint format: one unit per input, separated
    /// by spaces, or `none` when no input has a unit.
    pub fn encode_vars(&self) -> String {
        if self.vars.is_empty() {
            "none".to_string()
        } else {
            self.vars
                .iter()
                .map(|u| u.encode())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    /// Parse the text form written by [`Units::encode_vars`].
    pub fn decode_vars(s: &str) -> Result<Vec<Unit>, String> {
        if s == "none" {
            return Ok(Vec::new());
        }
        s.split_whitespace().map(Unit::parse).collect()
    }
}
