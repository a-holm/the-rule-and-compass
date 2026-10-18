//! Expression trees: the search representation.
//!
//! The primitive set covers the usual building blocks of symbolic regression
//! (see ../SPEC.md section 4).

use crate::rng::Mt;

/// The expression parser, re-exported so callers can write [`crate::expr::parse`].
pub use crate::parse::{parse, ParseError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Const,
    Input,
    Neg,
    Exp,
    Ln,
    Abs,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Logistic,
    Step,
    Gauss,
    Square,
    Cube,
    Floor,
    Ceil,
    Round,
    Sign,
    Erf,
    Erfc,
    Asinh,
    Acosh,
    Atanh,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Min,
    Max,
    Atan2,
    If,
    // Comparisons (0/1), logical operators and feasibility constraints.
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
    Xor,
    Not,
    Require,
    Between,
    Constrain,
}

impl Op {
    /// Number of operator variants. `Op` is a fieldless enum, so a discriminant
    /// is a valid dense index into a cost table; `complexity::ComplexityModel`
    /// relies on this. A new variant must bump the count (a test guards it).
    pub const COUNT: usize = 49;

    pub fn arity(self) -> usize {
        use Op::*;
        match self {
            Const | Input => 0,
            Neg | Exp | Ln | Abs | Sqrt | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Step
            | Gauss | Square | Cube | Floor | Ceil | Round | Sign | Erf | Erfc | Asinh | Acosh
            | Atanh | Not | Require | Constrain => 1,
            Add | Sub | Mul | Div | Pow | Mod | Min | Max | Atan2 | Equal | Less | LessOrEqual
            | Greater | GreaterOrEqual | And | Or | Xor => 2,
            If | Between => 3,
        }
    }

    pub fn symbol(self) -> &'static str {
        use Op::*;
        match self {
            Const => "c",
            Input => "x",
            Neg => "-",
            Exp => "exp",
            Ln => "ln",
            Abs => "abs",
            Sqrt => "sqrt",
            Sin => "sin",
            Cos => "cos",
            Tan => "tan",
            Sinh => "sinh",
            Cosh => "cosh",
            Tanh => "tanh",
            Logistic => "logistic",
            Step => "step",
            Gauss => "gauss",
            Square => "square",
            Cube => "cube",
            Floor => "floor",
            Ceil => "ceil",
            Round => "round",
            Sign => "sign",
            Erf => "erf",
            Erfc => "erfc",
            Asinh => "asinh",
            Acosh => "acosh",
            Atanh => "atanh",
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Pow => "^",
            Mod => "mod",
            Min => "min",
            Max => "max",
            Atan2 => "atan2",
            If => "if",
            Equal => "==",
            Less => "<",
            LessOrEqual => "<=",
            Greater => ">",
            GreaterOrEqual => ">=",
            And => "and",
            Or => "or",
            Xor => "xor",
            Not => "not",
            Require => "require",
            Between => "between",
            Constrain => "constrain",
        }
    }

    /// Default operator set for a search (all non-terminal primitives).
    ///
    /// The extended primitives (`Floor`…`Atanh`, `Mod`/`Min`/`Max`/`Atan2`) and
    /// the conditional family (comparisons, `and`/`or`/`xor`/`not`, and the
    /// constraints `require`/`between`/`constrain`) are not in the default set:
    /// they are opt-in through `--ops` so that broad searches are not diluted by
    /// them. Use [`Op::names`] for the full vocabulary.
    pub fn default_set() -> Vec<Op> {
        use Op::*;
        vec![
            Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Gauss, Square,
            Cube, Add, Sub, Mul, Div, Pow, If,
        ]
    }

    /// Canonical names accepted by [`Op::parse_list`], in a stable order. Used by
    /// the `schema` output so callers can discover the full primitive set.
    pub fn names() -> &'static [&'static str] {
        &[
            "neg", "exp", "ln", "abs", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "logistic", "step", "gauss", "square", "cube", "floor", "ceil", "round", "sign",
            "erf", "erfc", "asinh", "acosh", "atanh", "mod", "min", "max", "atan2", "add", "sub",
            "mul", "div", "pow", "if", "equal", "less", "less_or_equal", "greater",
            "greater_or_equal", "and", "or", "xor", "not", "require", "between", "constrain",
        ]
    }

    pub fn parse_list(s: &str) -> Vec<Op> {
        let mut out = Vec::new();
        for tok in s.split(|c| c == ',' || c == ' ') {
            if let Some(op) = Op::from_name(tok) {
                out.push(op);
            }
        }
        if out.is_empty() {
            Op::default_set()
        } else {
            out
        }
    }

    /// Map a primitive name (canonical name or an accepted alias) to an [`Op`].
    ///
    /// The inverse of [`Op::symbol`] for the alphabetic primitives, plus the
    /// short aliases (`log`, `sig`, `-`, `+`, …) that `--ops` has always taken.
    /// Used by [`Op::parse_list`] and by the expression parser, so the two
    /// never disagree about what a name means.
    pub fn from_name(s: &str) -> Option<Op> {
        use Op::*;
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "neg" | "-" => Neg,
            "exp" => Exp,
            "ln" | "log" => Ln,
            "abs" => Abs,
            "sqrt" => Sqrt,
            "sin" => Sin,
            "cos" => Cos,
            "tan" => Tan,
            "sinh" => Sinh,
            "cosh" => Cosh,
            "tanh" => Tanh,
            "logistic" | "sig" => Logistic,
            "step" => Step,
            "gauss" => Gauss,
            "square" => Square,
            "cube" => Cube,
            "floor" => Floor,
            "ceil" => Ceil,
            "round" => Round,
            "sign" => Sign,
            "erf" => Erf,
            "erfc" => Erfc,
            "asinh" => Asinh,
            "acosh" => Acosh,
            "atanh" => Atanh,
            "add" | "+" => Add,
            "sub" => Sub,
            "mul" | "*" => Mul,
            "div" | "/" => Div,
            "pow" | "^" => Pow,
            "mod" => Mod,
            "min" => Min,
            "max" => Max,
            "atan2" => Atan2,
            "if" => If,
            "equal" | "eq" | "==" => Equal,
            "less" | "lt" | "<" => Less,
            "less_or_equal" | "le" | "<=" => LessOrEqual,
            "greater" | "gt" | ">" => Greater,
            "greater_or_equal" | "ge" | ">=" => GreaterOrEqual,
            "and" | "&&" => And,
            "or" | "||" => Or,
            "xor" => Xor,
            "not" | "!" => Not,
            "require" => Require,
            "between" => Between,
            "constrain" => Constrain,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    /// `Const` -> value; `Input` -> variable index.
    pub val: f64,
    pub kids: Vec<Node>,
}

pub fn constant(v: f64) -> Node {
    Node {
        op: Op::Const,
        val: v,
        kids: Vec::new(),
    }
}

pub fn input(i: usize) -> Node {
    Node {
        op: Op::Input,
        val: i as f64,
        kids: Vec::new(),
    }
}

pub fn node(op: Op, kids: Vec<Node>) -> Node {
    Node {
        op,
        val: 0.0,
        kids,
    }
}

/// Taylor/continued-fraction accuracy target and the smallest representable
/// positive number for the continued fraction (nothing here is a hard domain
/// bound: the function is finite for every finite input).
const ERF_EPS: f64 = 1e-15;
const ERF_FPMIN: f64 = 1e-300;

/// Regularised lower incomplete gamma `P(1/2, x)` for `x >= 0`, which is
/// `erf(sqrt(x))`. Series for the small branch, continued fraction for the
/// large one (both converge to machine precision).
fn gammp_half(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return 1.0;
    }
    if x < 1.5 {
        let a = 0.5;
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..300 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * ERF_EPS {
                break;
            }
        }
        (sum * (-x + a * x.ln() - 0.5 * std::f64::consts::PI.ln()).exp()).clamp(0.0, 1.0)
    } else {
        (1.0 - gcf_half(x)).clamp(0.0, 1.0)
    }
}

/// Regularised upper incomplete gamma `Q(1/2, x) = 1 - P(1/2, x) = erfc(sqrt(x))`,
/// computed directly on the large branch so `erfc` never suffers cancellation.
fn gammq_half(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    if x < 1.5 {
        (1.0 - gammp_half(x)).clamp(0.0, 1.0)
    } else {
        gcf_half(x).clamp(0.0, 1.0)
    }
}

fn gcf_half(x: f64) -> f64 {
    let a = 0.5;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / ERF_FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..300 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < ERF_FPMIN {
            d = ERF_FPMIN;
        }
        c = b + an / c;
        if c.abs() < ERF_FPMIN {
            c = ERF_FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < ERF_EPS {
            break;
        }
    }
    (-x + a * x.ln() - 0.5 * std::f64::consts::PI.ln()).exp() * h
}

/// The error function `erf(x) = 2/sqrt(pi) * integral_0^x e^{-t^2} dt`, finite
/// for every input; `erf(NaN) = NaN`, `erf(±inf) = ±1`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let p = gammp_half(x * x);
    if x < 0.0 {
        -p
    } else {
        p
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`, evaluated directly
/// on its tail so large `x` does not cancel; finite for every input.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let q = gammq_half(x * x);
    if x < 0.0 {
        2.0 - q
    } else {
        q
    }
}

/// Comparison helper for the `equal`…`greater_or_equal` primitives: `1.0` when
/// `f(a, b)` holds, `0.0` otherwise, and `NaN` when either operand is `NaN`
/// (which the search treats as a hard failure).
fn cmp(a: f64, b: f64, f: impl Fn(f64, f64) -> bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if f(a, b) {
        1.0
    } else {
        0.0
    }
}

/// Evaluate a tree against one input row.
pub fn eval(n: &Node, x: &[f64]) -> f64 {
    use Op::*;
    match n.op {
        Const => n.val,
        Input => {
            let i = n.val as usize;
            if i < x.len() {
                x[i]
            } else {
                f64::NAN
            }
        }
        Neg => -eval(&n.kids[0], x),
        Exp => {
            let a = eval(&n.kids[0], x);
            if a > 700.0 {
                f64::INFINITY
            } else {
                a.exp()
            }
        }
        Ln => {
            let a = eval(&n.kids[0], x);
            if a > 0.0 {
                a.ln()
            } else {
                f64::NAN
            }
        }
        Abs => eval(&n.kids[0], x).abs(),
        Sqrt => {
            let a = eval(&n.kids[0], x);
            if a >= 0.0 {
                a.sqrt()
            } else {
                f64::NAN
            }
        }
        Sin => eval(&n.kids[0], x).sin(),
        Cos => eval(&n.kids[0], x).cos(),
        Tan => {
            let t = eval(&n.kids[0], x).tan();
            if t.is_finite() {
                t
            } else {
                f64::NAN
            }
        }
        Sinh => eval(&n.kids[0], x).sinh(),
        Cosh => eval(&n.kids[0], x).cosh(),
        Tanh => eval(&n.kids[0], x).tanh(),
        Logistic => {
            let a = eval(&n.kids[0], x);
            1.0 / (1.0 + (-a).exp())
        }
        Step => {
            let a = eval(&n.kids[0], x);
            if a > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Gauss => {
            let a = eval(&n.kids[0], x);
            (-a * a).exp()
        }
        Square => {
            let a = eval(&n.kids[0], x);
            a * a
        }
        Cube => {
            let a = eval(&n.kids[0], x);
            a * a * a
        }
        Floor => eval(&n.kids[0], x).floor(),
        Ceil => eval(&n.kids[0], x).ceil(),
        Round => eval(&n.kids[0], x).round(),
        Sign => {
            let a = eval(&n.kids[0], x);
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else if a == 0.0 {
                0.0
            } else {
                f64::NAN
            }
        }
        Erf => erf(eval(&n.kids[0], x)),
        Erfc => erfc(eval(&n.kids[0], x)),
        Asinh => eval(&n.kids[0], x).asinh(),
        Acosh => {
            let a = eval(&n.kids[0], x);
            if a >= 1.0 {
                a.acosh()
            } else {
                f64::NAN
            }
        }
        Atanh => {
            let a = eval(&n.kids[0], x);
            if a.abs() < 1.0 {
                a.atanh()
            } else {
                f64::NAN
            }
        }
        Add => eval(&n.kids[0], x) + eval(&n.kids[1], x),
        Sub => eval(&n.kids[0], x) - eval(&n.kids[1], x),
        Mul => eval(&n.kids[0], x) * eval(&n.kids[1], x),
        Div => {
            let b = eval(&n.kids[1], x);
            if b.abs() < 1e-12 {
                f64::NAN
            } else {
                eval(&n.kids[0], x) / b
            }
        }
        Pow => {
            let a = eval(&n.kids[0], x);
            let b = eval(&n.kids[1], x);
            if a < 0.0 && (b - b.round()).abs() > 1e-9 {
                f64::NAN
            } else {
                a.powf(b)
            }
        }
        Mod => {
            let a = eval(&n.kids[0], x);
            let b = eval(&n.kids[1], x);
            if !a.is_finite() || !b.is_finite() || b.abs() < 1e-12 {
                f64::NAN
            } else {
                a - b * (a / b).floor()
            }
        }
        Min => {
            let a = eval(&n.kids[0], x);
            let b = eval(&n.kids[1], x);
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
        Max => {
            let a = eval(&n.kids[0], x);
            let b = eval(&n.kids[1], x);
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        Atan2 => {
            let y = eval(&n.kids[0], x);
            let xx = eval(&n.kids[1], x);
            if y.is_nan() || xx.is_nan() {
                f64::NAN
            } else {
                y.atan2(xx)
            }
        }
        If => {
            let c = eval(&n.kids[0], x);
            if c > 0.0 {
                eval(&n.kids[1], x)
            } else {
                eval(&n.kids[2], x)
            }
        }
        // Comparisons: 1/0, `NaN` if an operand is `NaN`.
        Equal => cmp(eval(&n.kids[0], x), eval(&n.kids[1], x), |a, b| a == b),
        Less => cmp(eval(&n.kids[0], x), eval(&n.kids[1], x), |a, b| a < b),
        LessOrEqual => cmp(eval(&n.kids[0], x), eval(&n.kids[1], x), |a, b| a <= b),
        Greater => cmp(eval(&n.kids[0], x), eval(&n.kids[1], x), |a, b| a > b),
        GreaterOrEqual => cmp(eval(&n.kids[0], x), eval(&n.kids[1], x), |a, b| a >= b),
        // Logical: a value is true when it is strictly positive (`0` is false,
        // `NaN` propagates). `and`/`or`/`xor` return `1.0`/`0.0`.
        And => {
            let (a, b) = (eval(&n.kids[0], x), eval(&n.kids[1], x));
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if a > 0.0 && b > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Or => {
            let (a, b) = (eval(&n.kids[0], x), eval(&n.kids[1], x));
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if a > 0.0 || b > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Xor => {
            let (a, b) = (eval(&n.kids[0], x), eval(&n.kids[1], x));
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if (a > 0.0) != (b > 0.0) {
                1.0
            } else {
                0.0
            }
        }
        Not => {
            let a = eval(&n.kids[0], x);
            if a.is_nan() {
                f64::NAN
            } else if a > 0.0 {
                0.0
            } else {
                1.0
            }
        }
        // Constraints: return their argument when the condition holds, and a
        // non-finite value (a hard failure for the whole candidate) when it is
        // violated. See the module docs / SPEC.md 4.1 for the exact contract.
        Require | Constrain => {
            let a = eval(&n.kids[0], x);
            if a > 0.0 {
                a
            } else {
                f64::NAN
            }
        }
        Between => {
            let (v, lo, hi) = (
                eval(&n.kids[0], x),
                eval(&n.kids[1], x),
                eval(&n.kids[2], x),
            );
            if lo <= v && v <= hi {
                v
            } else {
                f64::NAN
            }
        }
    }
}

/// Complexity = node count. This is the default of
/// [`crate::complexity::ComplexityModel`], which supports per-operator costs
/// and variable/constant/depth weights; it stays the canonical structural
/// measure for tree traversal (`nth`, `mutate`, `crossover`).
pub fn complexity(n: &Node) -> usize {
    1 + n.kids.iter().map(complexity).sum::<usize>()
}

/// Number of constant nodes: the free parameters the fitter tunes. Used by the
/// information criteria in [`crate::diagnostics`].
pub fn count_constants(n: &Node) -> usize {
    let me = if n.op == Op::Const { 1 } else { 0 };
    me + n.kids.iter().map(count_constants).sum::<usize>()
}

pub fn nth<'a>(n: &'a Node, i: usize) -> Option<&'a Node> {
    if i == 0 {
        return Some(n);
    }
    let mut idx = i - 1;
    for k in &n.kids {
        let c = complexity(k);
        if idx < c {
            return nth(k, idx);
        }
        idx -= c;
    }
    None
}

pub fn nth_mut<'a>(n: &'a mut Node, i: usize) -> Option<&'a mut Node> {
    if i == 0 {
        return Some(n);
    }
    let mut idx = i - 1;
    for k in n.kids.iter_mut() {
        let c = complexity(k);
        if idx < c {
            return nth_mut(k, idx);
        }
        idx -= c;
    }
    None
}

fn round_sig(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
    (v * 1000.0).round() / 1000.0
}

fn random_terminal(rng: &mut Mt, nvars: usize) -> Node {
    if nvars > 0 && rng.bool(0.65) {
        input(rng.usize_below(nvars))
    } else {
        constant(round_sig(rng.range(-2.0, 2.0)))
    }
}

pub fn random_tree(rng: &mut Mt, nvars: usize, ops: &[Op], depth: usize) -> Node {
    if depth == 0 || rng.bool(0.15) {
        return random_terminal(rng, nvars);
    }
    let op = *rng.pick(ops);
    let k = op.arity();
    let kids = (0..k)
        .map(|_| random_tree(rng, nvars, ops, depth - 1))
        .collect();
    node(op, kids)
}

fn perturb(rng: &mut Mt, v: f64) -> f64 {
    match rng.usize_below(4) {
        0 => round_sig(v * rng.range(0.3, 1.8).max(0.05)),
        1 => round_sig(v + rng.range(-1.0, 1.0)),
        2 => round_sig(rng.range(-2.0, 2.0)),
        _ => round_sig(v + rng.range(-0.2, 0.2)),
    }
}

/// In-place mutation: point-change an operator, perturb a constant, or replace a
/// subtree (grow / shrink). [I]
pub fn mutate(n: &mut Node, rng: &mut Mt, nvars: usize, ops: &[Op]) {
    let total = complexity(n);
    if total == 0 {
        return;
    }
    let i = rng.usize_below(total);
    let r = rng.next_f64();
    if r < 0.25 {
        if let Some(t) = nth_mut(n, i) {
            let ar = t.op.arity();
            if ar == 0 {
                // flip terminal kind, or re-value
                if t.op == Op::Const {
                    t.val = perturb(rng, t.val);
                } else if nvars > 0 {
                    t.val = rng.usize_below(nvars) as f64;
                }
            } else {
                let cands: Vec<Op> = ops
                    .iter()
                    .copied()
                    .filter(|o| o.arity() == ar && *o != t.op)
                    .collect();
                if !cands.is_empty() {
                    t.op = cands[rng.usize_below(cands.len())];
                    t.kids.truncate(t.op.arity());
                    while t.kids.len() < t.op.arity() {
                        t.kids.push(constant(1.0));
                    }
                }
            }
        }
    } else if r < 0.6 {
        if let Some(t) = nth_mut(n, i) {
            if t.op == Op::Const {
                t.val = perturb(rng, t.val);
            } else if t.op == Op::Input && nvars > 0 {
                t.val = rng.usize_below(nvars) as f64;
            }
        }
    } else {
        let depth = if r < 0.85 { 2 } else { 1 };
        if let Some(t) = nth_mut(n, i) {
            *t = random_tree(rng, nvars, ops, depth);
        }
    }
}

/// Replace a random subtree of `a` with a random subtree of `b` (crossover). [I]
pub fn crossover(a: &mut Node, b: &Node, rng: &mut Mt) {
    let ta = complexity(a);
    let tb = complexity(b);
    if ta == 0 || tb == 0 {
        return;
    }
    let ia = rng.usize_below(ta);
    let ib = rng.usize_below(tb);
    if let Some(sub) = nth(b, ib).cloned() {
        if let Some(t) = nth_mut(a, ia) {
            *t = sub;
        }
    }
}

// ---------------- display & simplification ----------------

fn fmt_f(v: f64) -> String {
    if v == v.round() && v.abs() < 1e12 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn to_string(n: &Node, vars: &[String]) -> String {
    use Op::*;
    match n.op {
        Const => fmt_f(n.val),
        Input => {
            let i = n.val as usize;
            vars.get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i))
        }
        Neg => format!("(-{})", to_string(&n.kids[0], vars)),
        Add | Sub | Mul | Div | Pow => {
            let s = match n.op {
                Add => " + ",
                Sub => " - ",
                Mul => " * ",
                Div => " / ",
                _ => " ^ ",
            };
            format!(
                "({}{}{})",
                to_string(&n.kids[0], vars),
                s,
                to_string(&n.kids[1], vars)
            )
        }
        If => format!(
            "if({}, {}, {})",
            to_string(&n.kids[0], vars),
            to_string(&n.kids[1], vars),
            to_string(&n.kids[2], vars)
        ),
        Mod | Min | Max | Atan2 => format!(
            "{}({}, {})",
            n.op.symbol(),
            to_string(&n.kids[0], vars),
            to_string(&n.kids[1], vars)
        ),
        // Comparisons and the binary logical operators print infix inside
        // parentheses, so the printed form parses back unambiguously.
        Equal | Less | LessOrEqual | Greater | GreaterOrEqual | And | Or | Xor => {
            let s = match n.op {
                Equal => " == ",
                Less => " < ",
                LessOrEqual => " <= ",
                Greater => " > ",
                GreaterOrEqual => " >= ",
                And => " and ",
                Or => " or ",
                _ => " xor ",
            };
            format!(
                "({}{}{})",
                to_string(&n.kids[0], vars),
                s,
                to_string(&n.kids[1], vars)
            )
        }
        Not => format!("(not {})", to_string(&n.kids[0], vars)),
        Between => format!(
            "between({}, {}, {})",
            to_string(&n.kids[0], vars),
            to_string(&n.kids[1], vars),
            to_string(&n.kids[2], vars)
        ),
        _ => format!("{}({})", n.op.symbol(), to_string(&n.kids[0], vars)),
    }
}

fn cval(n: &Node) -> Option<f64> {
    if n.op == Op::Const {
        Some(n.val)
    } else {
        None
    }
}

fn is_val(n: &Node, v: f64) -> bool {
    cval(n).map(|c| (c - v).abs() < 1e-12).unwrap_or(false)
}

/// Relative comparison against a "nice" value (0, 1). Constants of a fitted
/// model differ from the ideal by more than machine epsilon whenever the data
/// is rounded, so multiplicative identity rules need a relative tolerance.
fn near_val(n: &Node, v: f64, rel: f64) -> bool {
    cval(n)
        .map(|c| (c - v).abs() <= rel * (1.0 + v.abs()))
        .unwrap_or(false)
}

/// Structural equality (exact shape and constant values).
pub fn same(a: &Node, b: &Node) -> bool {
    a.op == b.op
        && a.val.to_bits() == b.val.to_bits()
        && a.kids.len() == b.kids.len()
        && a.kids.iter().zip(&b.kids).all(|(x, y)| same(x, y))
}

/// FNV-1a structural hash, for de-duplicating equivalent candidates before
/// they are evaluated.
pub fn structural_hash(n: &Node) -> u64 {
    fn mix(h: &mut u64, x: u64) {
        *h ^= x;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn go(n: &Node, h: &mut u64) {
        mix(h, n.op as u64);
        mix(h, n.val.to_bits());
        mix(h, n.kids.len() as u64);
        for k in &n.kids {
            go(k, h);
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    go(n, &mut h);
    h
}

/// `k * u`, pulled into canonical form by absorbing any constant factor that
/// already sits at the top of `u`. Keeps `0.8447*(x)` rather than `x/1.1839`.
fn mul_const(k: f64, u: Node) -> Node {
    if k == 0.0 {
        return constant(0.0);
    }
    if (k - 1.0).abs() <= 1e-8 {
        return u;
    }
    if u.op == Op::Const {
        return constant(k * u.val);
    }
    if u.op == Op::Mul && u.kids[0].op == Op::Const {
        return mul_const(k * u.kids[0].val, u.kids[1].clone());
    }
    if u.op == Op::Div && u.kids[1].op == Op::Const {
        return mul_const(k / u.kids[1].val, u.kids[0].clone());
    }
    node(Op::Mul, vec![constant(k), u])
}

/// `k + u`, the additive counterpart of [`mul_const`].
fn add_const(k: f64, u: Node) -> Node {
    if k.abs() <= 1e-12 {
        return u;
    }
    if u.op == Op::Const {
        return constant(k + u.val);
    }
    if u.op == Op::Add && u.kids[0].op == Op::Const {
        return add_const(k + u.kids[0].val, u.kids[1].clone());
    }
    if u.op == Op::Sub && u.kids[1].op == Op::Const {
        return add_const(k - u.kids[1].val, u.kids[0].clone());
    }
    node(Op::Add, vec![constant(k), u])
}

/// Bottom-up constant folding, algebraic identities, and affine normalisation.
///
/// Equivalent expressions evaluate identically, so collapsing them removes work
/// that buys nothing (see `docs/optimisation.md`, "the kernel of evaluation").
/// Normalising constant factors also makes the printed equation readable:
/// `x/1.1839` becomes `0.8447*x`, and `(x/c1)/c2` becomes `1.6894*x`.
pub fn simplify(n: &Node) -> Node {
    use Op::*;
    let m = Node {
        op: n.op,
        val: n.val,
        kids: n.kids.iter().map(simplify).collect(),
    };
    if m.op == Const || m.op == Input {
        return m;
    }
    // Fold any operator whose children are all constants.
    if m.kids.iter().all(|k| k.op == Const) {
        let v = eval(&m, &[]);
        if v.is_finite() {
            return constant(v);
        }
    }
    match m.op {
        Neg => {
            if m.kids[0].op == Neg && m.kids[0].kids.len() == 1 {
                return m.kids[0].kids[0].clone();
            }
        }
        Add => {
            if is_val(&m.kids[0], 0.0) {
                return m.kids[1].clone();
            }
            if is_val(&m.kids[1], 0.0) {
                return m.kids[0].clone();
            }
            if m.kids[0].op == Const {
                return add_const(m.kids[0].val, m.kids[1].clone());
            }
            if m.kids[1].op == Const {
                return add_const(m.kids[1].val, m.kids[0].clone());
            }
        }
        Sub => {
            if is_val(&m.kids[1], 0.0) {
                return m.kids[0].clone();
            }
            if same(&m.kids[0], &m.kids[1]) {
                return constant(0.0);
            }
            if m.kids[1].op == Const {
                return add_const(-m.kids[1].val, m.kids[0].clone());
            }
        }
        Mul => {
            if is_val(&m.kids[0], 0.0) || is_val(&m.kids[1], 0.0) {
                return constant(0.0);
            }
            if m.kids[0].op == Const {
                return mul_const(m.kids[0].val, m.kids[1].clone());
            }
            if m.kids[1].op == Const {
                return mul_const(m.kids[1].val, m.kids[0].clone());
            }
        }
        Div => {
            if near_val(&m.kids[1], 1.0, 1e-12) {
                return m.kids[0].clone();
            }
            if same(&m.kids[0], &m.kids[1]) && m.kids[0].op != Const {
                return constant(1.0);
            }
            if m.kids[1].op == Const {
                return mul_const(1.0 / m.kids[1].val, m.kids[0].clone());
            }
        }
        Pow => {
            if is_val(&m.kids[1], 1.0) {
                return m.kids[0].clone();
            }
            if is_val(&m.kids[1], 0.0) {
                return constant(1.0);
            }
        }
        If => {
            // A constant condition makes one branch dead code.
            if m.kids[0].op == Const {
                return if m.kids[0].val > 0.0 {
                    m.kids[1].clone()
                } else {
                    m.kids[2].clone()
                };
            }
        }
        _ => {}
    }
    m
}

/// Rewrite `Input(i)` from normalised space back to original units:
/// `X_i -> (x_i - mean_i) / std_i`. [C: Expression_denormalizer]
pub fn denormalize(n: &Node, means: &[f64], stds: &[f64]) -> Node {
    match n.op {
        Op::Input => {
            let i = n.val as usize;
            let m = means.get(i).copied().unwrap_or(0.0);
            let s = stds.get(i).copied().unwrap_or(1.0);
            node(
                Op::Div,
                vec![node(Op::Sub, vec![input(i), constant(m)]), constant(s)],
            )
        }
        _ => Node {
            op: n.op,
            val: n.val,
            kids: n.kids.iter().map(|k| denormalize(k, means, stds)).collect(),
        },
    }
}
