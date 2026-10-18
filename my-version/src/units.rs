//! Dimensional analysis for expression trees (issue #25).
//!
//! A **dimension** is a vector of integer exponents over named base
//! dimensions. The seven SI names (`m`, `s`, `kg`, `A`, `K`, `mol`, `cd`) are
//! conventional, but they are not special: any other identifier in a unit
//! expression becomes its own base dimension, so tabular quantities such as
//! `USD` and `shares` work too.
//!
//! Only **dimensions** are modelled. There is deliberately no unit table and no
//! scale factor: `m^3` and a volume in litres are the same dimension, and the
//! fitted constant absorbs the 1e-3. Likewise `K` and `degC` share a dimension
//! and an intercept absorbs the affine offset.
//!
//! [`infer`] is a single bottom-up pass over a [`Node`]. It returns either the
//! root dimension or the first [`Inconsistency`] it meets. The search asks
//! exactly one question through [`Units::accepts`]: does the root dimension
//! equal the target's? [`constant_dimensions`] is a separate, best-effort
//! top-down pass used only when reporting a fitted law.
//!
//! The parser accepts identifiers, `*`, `/`, integer powers with `^`,
//! parentheses and the empty unit `1`:
//!
//! ```
//! use rcsr::units::Unit;
//! assert_eq!(Unit::parse("m/s^2").unwrap().to_string(), "m/s^2");
//! assert_eq!(Unit::parse("kg*m^2/s^2").unwrap().to_string(), "kg*m^2/s^2");
//! assert_eq!(Unit::parse("1").unwrap(), Unit::DIMENSIONLESS);
//! // An unknown name is its own base dimension.
//! assert_ne!(Unit::parse("USD").unwrap(), Unit::parse("shares").unwrap());
//! assert_eq!(Unit::parse("m*m*m").unwrap(), Unit::parse("m^3").unwrap());
//! ```

use crate::expr::{Node, Op};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The conventional SI base dimensions, in the order used by
/// [`Unit::from_exponents`].
pub const BASE_DIMENSIONS: [&str; 7] = ["m", "s", "kg", "A", "K", "mol", "cd"];

/// A dimension: nonzero integer exponents over named base dimensions, sorted
/// by name. There is no unit identity and no scale factor.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Unit {
    exps: Vec<(String, i32)>,
}

impl Unit {
    /// The empty unit, written `1`.
    pub const DIMENSIONLESS: Unit = Unit { exps: Vec::new() };

    /// One base dimension with exponent 1. Any name is accepted; this is how
    /// `USD`, `shares` or an SI name become a dimension.
    pub fn base(name: &str) -> Unit {
        Unit {
            exps: vec![(name.to_string(), 1)],
        }
    }

    /// Build a dimension from exponents over [`BASE_DIMENSIONS`]. Used by the
    /// tests and by callers that already work in the seven SI names.
    pub fn from_exponents(exps: [i32; 7]) -> Unit {
        let mut out = Vec::new();
        for (name, e) in BASE_DIMENSIONS.iter().zip(exps) {
            if e != 0 {
                out.push((name.to_string(), e));
            }
        }
        normalize(out)
    }

    /// True when every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        self.exps.is_empty()
    }

    /// `self * rhs`: add the exponents.
    pub fn mul(mut self, rhs: Unit) -> Unit {
        self.exps.extend(rhs.exps);
        normalize(self.exps)
    }

    /// `self / rhs`: subtract the exponents.
    pub fn div(self, rhs: Unit) -> Unit {
        self.mul(rhs.pow(-1))
    }

    /// `self ^ n`: multiply every exponent by `n`.
    pub fn pow(self, n: i32) -> Unit {
        normalize(
            self.exps
                .into_iter()
                .map(|(name, e)| (name, e.saturating_mul(n)))
                .collect(),
        )
    }

    /// The exact `n`-th root, when every exponent is divisible by `n`.
    pub fn root(self, n: i32) -> Option<Unit> {
        if n == 0 {
            return None;
        }
        let mut out = Vec::new();
        for (name, e) in self.exps {
            if e % n != 0 {
                return None;
            }
            let r = e / n;
            if r != 0 {
                out.push((name, r));
            }
        }
        Some(Unit { exps: out })
    }

    /// Parse a dimension expression.
    ///
    /// Grammar:
    ///
    /// ```text
    /// expr   := factor (('*' | '/') factor)*
    /// factor := atom ('^' [+-]? digits)?
    /// atom   := name | '1' | '(' expr ')'
    /// ```
    ///
    /// A name is an identifier: an ASCII letter followed by letters, digits or
    /// underscores. Every name is a base dimension; unknown names are not an
    /// error. Only malformed syntax and non-integral exponents are errors.
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
    ///
    /// Exponents are printed in name order; one `/` is written per negative
    /// factor so the left-associative grammar parses the text back to the same
    /// dimension (`mol/(A*K)` prints as `mol/A/K`).
    pub fn encode(&self) -> String {
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        for (name, e) in &self.exps {
            if *e > 0 {
                positive.push(power(name, *e));
            } else if *e < 0 {
                negative.push(power(name, -*e));
            }
        }
        let mut out = if positive.is_empty() {
            "1".to_string()
        } else {
            positive.join("*")
        };
        for n in negative {
            out.push('/');
            out.push_str(&n);
        }
        out
    }
}

/// Merge duplicate names, drop zeros and sort by name.
fn normalize(exps: Vec<(String, i32)>) -> Unit {
    let mut map: BTreeMap<String, i32> = BTreeMap::new();
    for (name, e) in exps {
        let slot = map.entry(name).or_insert(0);
        *slot = slot.saturating_add(e);
    }
    Unit {
        exps: map.into_iter().filter(|(_, e)| *e != 0).collect(),
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
        f.write_str(&self.encode())
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
                self.pos += 1;
                while self.pos < self.src.len() {
                    let c = self.src[self.pos] as char;
                    if c.is_ascii_alphanumeric() || c == '_' {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                Ok(Unit::base(&self.text[start..self.pos]))
            }
            _ => Err(format!(
                "expected a dimension name, '1' or '(' at column {} in unit '{}'",
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
    /// `sqrt` of a dimension whose exponents are not all even.
    NonIntegerRoot { actual: Unit },
    /// The constraints on the constants' dimensions have no solution.
    Unsatisfiable { op: Op, detail: String },
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
            Inconsistency::Unsatisfiable { op, detail } => {
                write!(f, "{} has no consistent constant dimensions: {}", op.symbol(), detail)
            }
        }
    }
}

/// Infer the root dimension of `tree`, with input `i` taking `vars[i]`.
///
/// This is the strict bottom-up algebra with **constants treated as
/// dimensionless**. The search predicate [`Units::accepts`] instead solves for
/// the dimensions of the fitted constants, so an intercept can absorb the
/// target's dimension. An input index outside `vars` is treated as
/// dimensionless; evaluation of such a tree returns `NaN`, so the dimensional
/// question is moot.
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
            .cloned()
            .unwrap_or(Unit::DIMENSIONLESS)),
        // Sign-preserving / magnitude-preserving operators.
        Neg | Abs => d(&tree.kids[0]),
        // `sqrt` halves every exponent; all exponents must stay integral.
        Sqrt => d(&tree.kids[0])?
            .root(2)
            .ok_or_else(|| Inconsistency::NonIntegerRoot {
                actual: d(&tree.kids[0]).unwrap_or(Unit::DIMENSIONLESS),
            }),
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
    /// Input dimensions in input order. Missing entries are dimensionless.
    pub vars: Vec<Unit>,
    /// The dimension of the target column.
    pub target: Unit,
}

impl Units {
    /// Construct a declaration.
    pub fn new(vars: Vec<Unit>, target: Unit) -> Units {
        Units { vars, target }
    }

    /// The dimension of the tree's root, or the first inconsistency.
    pub fn root_dimension(&self, tree: &Node) -> Result<Unit, Inconsistency> {
        infer(tree, &self.vars)
    }

    /// The single predicate the search calls.
    ///
    /// Unlike [`infer`], which treats every constant as dimensionless, this
    /// solves for the dimension each fitted constant must carry. That is what
    /// lets a coefficient absorb a unit ratio and an intercept absorb the
    /// target's unit (and an affine offset); an inconsistent tree has no
    /// integer solution and is rejected.
    pub fn accepts(&self, tree: &Node) -> bool {
        solve(tree, &self.vars, &self.target).is_ok()
    }

    /// Text form used by the checkpoint format: one dimension per input,
    /// separated by spaces, or `none` when no input has one.
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

/// Fitted values of the constant leaves, in pre-order tree order. This is the
/// order [`constant_dimensions`] uses.
pub fn constant_values(tree: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

/// Best-effort dimensions of the constant leaves, in pre-order tree order
/// (the same order as [`constant_values`]).
///
/// This is a reporting aid, not the search predicate. It runs the same solve
/// as [`Units::accepts`] and returns one [`Unit`] per constant. When the tree
/// has no consistent integer assignment the whole vector is `None`.
pub fn constant_dimensions(tree: &Node, vars: &[Unit], target: &Unit) -> Vec<Option<Unit>> {
    let k = crate::expr::count_constants(tree);
    match solve(tree, vars, target) {
        Ok((units, order)) => order
            .iter()
            .map(|&i| Some(units[i].clone()))
            .collect(),
        Err(_) => vec![None; k],
    }
}

/// A dimension expression that is affine in the unknown dimensions:
/// `base + sum_j coeffs[j] * U_j`.
///
/// Most unknowns are fitted constants. A `sqrt` node contributes one extra
/// unknown for its own result, which is how `sqrt(c*x)` can be consistent with
/// a length target (choose `c` so `c*x` is an area).
#[derive(Clone, Debug)]
struct SymDim {
    base: Unit,
    coeffs: Vec<i32>,
}

impl SymDim {
    fn from_unit(base: Unit, nvars: usize) -> SymDim {
        SymDim {
            base,
            coeffs: vec![0; nvars],
        }
    }

    fn dimensionless(nvars: usize) -> SymDim {
        SymDim::from_unit(Unit::DIMENSIONLESS, nvars)
    }

    fn is_zero(&self) -> bool {
        self.base.is_dimensionless() && self.coeffs.iter().all(|&c| c == 0)
    }

    fn add(mut self, rhs: SymDim) -> SymDim {
        self.base = self.base.mul(rhs.base);
        for (a, b) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *a = a.saturating_add(b);
        }
        self
    }

    fn sub(mut self, rhs: SymDim) -> SymDim {
        self.base = self.base.div(rhs.base);
        for (a, b) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *a = a.saturating_sub(b);
        }
        self
    }

    fn scale(mut self, n: i32) -> SymDim {
        self.base = self.base.pow(n);
        for a in &mut self.coeffs {
            *a = a.saturating_mul(n);
        }
        self
    }
}

/// Variable slots for the solve: one per constant plus one per `sqrt` result.
struct VarMap {
    constants: HashMap<usize, usize>,
    sqrts: HashMap<usize, usize>,
    /// Constant slots in pre-order, the order the report uses.
    const_order: Vec<usize>,
    nvars: usize,
}

fn index_unknowns(n: &Node, map: &mut VarMap) {
    let ptr = n as *const Node as usize;
    if n.op == Op::Const {
        map.constants.insert(ptr, map.nvars);
        map.const_order.push(map.nvars);
        map.nvars += 1;
    } else if n.op == Op::Sqrt {
        // The result gets its own unknown; `child = result^2` is an equation.
        map.sqrts.insert(ptr, map.nvars);
        map.nvars += 1;
    }
    for k in &n.kids {
        index_unknowns(k, map);
    }
}

/// Add the equation `a == b`.
fn equation(
    op: Op,
    a: SymDim,
    b: SymDim,
    equations: &mut Vec<SymDim>,
) -> Result<(), Inconsistency> {
    let d = a.sub(b);
    if d.is_zero() {
        return Ok(());
    }
    // A pure base mismatch with no unknown can never be solved; keeping it as
    // an equation would work too, but this gives a clearer operator.
    if !d.base.is_dimensionless() && d.coeffs.iter().all(|&c| c == 0) {
        return Err(Inconsistency::Unsatisfiable {
            op,
            detail: format!(
                "{} != {} with no fitted constant or root to absorb the difference",
                d.base,
                Unit::DIMENSIONLESS
            ),
        });
    }
    equations.push(d);
    Ok(())
}

fn solve(
    tree: &Node,
    vars: &[Unit],
    target: &Unit,
) -> Result<(Vec<Unit>, Vec<usize>), Inconsistency> {
    let mut map = VarMap {
        constants: HashMap::new(),
        sqrts: HashMap::new(),
        const_order: Vec::new(),
        nvars: 0,
    };
    index_unknowns(tree, &mut map);
    let mut equations = Vec::new();
    let root = symbolic(tree, vars, &map, &mut equations)?;
    equation(
        Op::Const,
        root,
        SymDim::from_unit(target.clone(), map.nvars),
        &mut equations,
    )?;
    let units = solve_equations(&equations, map.nvars)?;
    Ok((units, map.const_order))
}

fn symbolic(
    n: &Node,
    vars: &[Unit],
    map: &VarMap,
    equations: &mut Vec<SymDim>,
) -> Result<SymDim, Inconsistency> {
    use Op::*;
    let nvars = map.nvars;
    let sym = |child: &Node, eqs: &mut Vec<SymDim>| symbolic(child, vars, map, eqs);
    let dimless = SymDim::dimensionless(nvars);
    match n.op {
        Const => {
            let mut coeffs = vec![0; nvars];
            let id = *map
                .constants
                .get(&(n as *const Node as usize))
                .expect("constant index allocated in the prepass");
            coeffs[id] = 1;
            Ok(SymDim {
                base: Unit::DIMENSIONLESS,
                coeffs,
            })
        }
        Input => Ok(SymDim::from_unit(
            vars.get(n.val as usize)
                .cloned()
                .unwrap_or(Unit::DIMENSIONLESS),
            nvars,
        )),
        Neg | Abs => sym(&n.kids[0], equations),
        Sqrt => {
            let child = sym(&n.kids[0], equations)?;
            let id = *map
                .sqrts
                .get(&(n as *const Node as usize))
                .expect("sqrt index allocated in the prepass");
            let mut coeffs = vec![0; nvars];
            coeffs[id] = 1;
            let result = SymDim {
                base: Unit::DIMENSIONLESS,
                coeffs,
            };
            // child == result^2, exactly.
            equation(Sqrt, child, result.clone().scale(2), equations)?;
            Ok(result)
        }
        Square => Ok(sym(&n.kids[0], equations)?.scale(2)),
        Cube => Ok(sym(&n.kids[0], equations)?.scale(3)),
        Exp | Ln | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Gauss | Step | Floor
        | Ceil | Round | Sign | Erf | Erfc | Asinh | Acosh | Atanh | Not | Require
        | Constrain => {
            let child = sym(&n.kids[0], equations)?;
            equation(n.op, child, dimless.clone(), equations)?;
            Ok(dimless)
        }
        Add | Sub | Min | Max | Mod => {
            let left = sym(&n.kids[0], equations)?;
            let right = sym(&n.kids[1], equations)?;
            equation(n.op, left.clone(), right, equations)?;
            Ok(left)
        }
        Mul => {
            let left = sym(&n.kids[0], equations)?;
            let right = sym(&n.kids[1], equations)?;
            Ok(left.add(right))
        }
        Div => {
            let left = sym(&n.kids[0], equations)?;
            let right = sym(&n.kids[1], equations)?;
            Ok(left.sub(right))
        }
        Pow => {
            // Left first, so unknown indices stay in printed order.
            let base = sym(&n.kids[0], equations)?;
            let exponent = sym(&n.kids[1], equations)?;
            equation(Pow, exponent, dimless.clone(), equations)?;
            // A dimensionless base (or one made only of constants) can take
            // any dimensionless exponent. A base containing inputs needs a
            // constant integer power.
            if base.base.is_dimensionless() {
                return Ok(dimless);
            }
            match constant_integer(&n.kids[1]) {
                Some(p) => Ok(base.scale(p)),
                None => Err(Inconsistency::NonIntegerPower { base: base.base }),
            }
        }
        Atan2 => {
            let left = sym(&n.kids[0], equations)?;
            let right = sym(&n.kids[1], equations)?;
            equation(Atan2, left, right, equations)?;
            Ok(dimless)
        }
        If => {
            let condition = sym(&n.kids[0], equations)?;
            equation(If, condition, dimless.clone(), equations)?;
            let then = sym(&n.kids[1], equations)?;
            let otherwise = sym(&n.kids[2], equations)?;
            equation(If, then.clone(), otherwise, equations)?;
            Ok(then)
        }
        Equal | Less | LessOrEqual | Greater | GreaterOrEqual => {
            let left = sym(&n.kids[0], equations)?;
            let right = sym(&n.kids[1], equations)?;
            equation(n.op, left, right, equations)?;
            Ok(dimless)
        }
        And | Or | Xor => {
            let left = sym(&n.kids[0], equations)?;
            equation(n.op, left, dimless.clone(), equations)?;
            let right = sym(&n.kids[1], equations)?;
            equation(n.op, right, dimless.clone(), equations)?;
            Ok(dimless)
        }
        Between => {
            for child in &n.kids {
                let d = sym(child, equations)?;
                equation(Between, d, dimless.clone(), equations)?;
            }
            Ok(dimless)
        }
        // Meta operators describe shape, not a value; their constants are
        // structural and stay unconstrained (and solve to dimensionless).
        Complexity | Variables | Constants | Contains | ContainsOp => Ok(dimless),
    }
}

/// Solve the accumulated linear constraints for the unknown dimensions.
///
/// Each base dimension is independent: the same coefficient matrix with a
/// different right-hand side. Gaussian elimination over `f64` is enough for
/// the small systems a fitted expression produces; a solution that is not
/// integral is rejected because dimensions are integer exponent vectors.
fn solve_equations(equations: &[SymDim], nvars: usize) -> Result<Vec<Unit>, Inconsistency> {
    if nvars == 0 {
        for eq in equations {
            if !eq.base.is_dimensionless() {
                return Err(Inconsistency::Unsatisfiable {
                    op: Op::Const,
                    detail: "the expression has no unknown to absorb the mismatch".to_string(),
                });
            }
        }
        return Ok(Vec::new());
    }

    let mut names = BTreeSet::new();
    for eq in equations {
        for (name, _) in &eq.base.exps {
            names.insert(name.clone());
        }
    }

    let mut solved = vec![Vec::new(); nvars];
    for name in names {
        let mut rows = Vec::with_capacity(equations.len());
        for eq in equations {
            let mut row = Vec::with_capacity(nvars + 1);
            row.extend(eq.coeffs.iter().map(|&c| c as f64));
            let rhs = -eq
                .base
                .exps
                .iter()
                .find(|(n, _)| n == &name)
                .map(|(_, e)| *e as f64)
                .unwrap_or(0.0);
            row.push(rhs);
            rows.push(row);
        }
        let x = gaussian(rows, nvars).ok_or_else(|| Inconsistency::Unsatisfiable {
            op: Op::Const,
            detail: format!("no integer dimensions for '{}'", name),
        })?;
        for (j, value) in x.into_iter().enumerate() {
            if value != 0.0 {
                solved[j].push((name.clone(), value as i32));
            }
        }
    }

    Ok(solved.into_iter().map(|exps| Unit { exps }).collect())
}

/// Reduced row echelon solve of `A x = b`; free variables are zero.
fn gaussian(mut rows: Vec<Vec<f64>>, k: usize) -> Option<Vec<f64>> {
    const EPS: f64 = 1e-9;
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut pivot_row = 0;
    for col in 0..k {
        if pivot_row == rows.len() {
            break;
        }
        let mut best = pivot_row;
        for r in pivot_row..rows.len() {
            if rows[r][col].abs() > rows[best][col].abs() {
                best = r;
            }
        }
        if rows[best][col].abs() <= EPS {
            continue;
        }
        rows.swap(pivot_row, best);
        let pivot = rows[pivot_row][col];
        for j in col..=k {
            rows[pivot_row][j] /= pivot;
        }
        for r in 0..rows.len() {
            if r == pivot_row {
                continue;
            }
            let factor = rows[r][col];
            if factor.abs() <= EPS {
                continue;
            }
            for j in col..=k {
                rows[r][j] -= factor * rows[pivot_row][j];
            }
        }
        pivots.push((pivot_row, col));
        pivot_row += 1;
    }
    for row in &rows {
        if row[..k].iter().all(|v| v.abs() <= EPS) && row[k].abs() > EPS {
            return None;
        }
    }
    let mut x = vec![0.0; k];
    for (r, c) in pivots {
        x[c] = rows[r][k];
    }
    // A particular solution must be integral: dimensions are integer vectors.
    for value in &x {
        let rounded = value.round();
        if (*value - rounded).abs() > 1e-6 {
            return None;
        }
    }
    Some(x.into_iter().map(|v| v.round()).collect())
}
