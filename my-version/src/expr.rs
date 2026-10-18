//! Expression trees: the search representation.
//!
//! The primitive set covers the usual building blocks of symbolic regression
//! (see ../SPEC.md section 4).

use crate::rng::Mt;

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
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    If,
}

impl Op {
    pub fn arity(self) -> usize {
        use Op::*;
        match self {
            Const | Input => 0,
            Neg | Exp | Ln | Abs | Sqrt | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Step
            | Gauss | Square | Cube => 1,
            Add | Sub | Mul | Div | Pow => 2,
            If => 3,
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
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Pow => "^",
            If => "if",
        }
    }

    /// Default operator set for a search (all non-terminal primitives).
    pub fn default_set() -> Vec<Op> {
        use Op::*;
        vec![
            Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Gauss, Square,
            Cube, Add, Sub, Mul, Div, Pow, If,
        ]
    }

    pub fn parse_list(s: &str) -> Vec<Op> {
        use Op::*;
        let mut out = Vec::new();
        for tok in s.split(|c| c == ',' || c == ' ') {
            let t = tok.trim().to_ascii_lowercase();
            let op = match t.as_str() {
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
                "add" | "+" => Add,
                "sub" => Sub,
                "mul" | "*" => Mul,
                "div" | "/" => Div,
                "pow" | "^" => Pow,
                "if" => If,
                _ => continue,
            };
            out.push(op);
        }
        if out.is_empty() {
            Op::default_set()
        } else {
            out
        }
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
        If => {
            let c = eval(&n.kids[0], x);
            if c > 0.0 {
                eval(&n.kids[1], x)
            } else {
                eval(&n.kids[2], x)
            }
        }
    }
}

/// Complexity = node count. A per-operator cost table can refine this;
/// v1 uses 1.0 per node.
pub fn complexity(n: &Node) -> usize {
    1 + n.kids.iter().map(complexity).sum::<usize>()
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

/// Bottom-up constant folding and algebraic identities. [C flags / I algorithm]
pub fn simplify(n: &Node) -> Node {
    use Op::*;
    let kids: Vec<Node> = n.kids.iter().map(simplify).collect();
    let m = Node {
        op: n.op,
        val: n.val,
        kids,
    };
    match m.op {
        Const | Input => m,
        Neg => {
            if let Some(a) = cval(&m.kids[0]) {
                constant(-a)
            } else if m.kids[0].op == Neg {
                m.kids[0].kids[0].clone()
            } else {
                m
            }
        }
        Add => {
            if is_val(&m.kids[0], 0.0) {
                m.kids[1].clone()
            } else if is_val(&m.kids[1], 0.0) {
                m.kids[0].clone()
            } else if let (Some(a), Some(b)) = (cval(&m.kids[0]), cval(&m.kids[1])) {
                constant(a + b)
            } else {
                m
            }
        }
        Sub => {
            if is_val(&m.kids[1], 0.0) {
                m.kids[0].clone()
            } else if let (Some(a), Some(b)) = (cval(&m.kids[0]), cval(&m.kids[1])) {
                constant(a - b)
            } else {
                m
            }
        }
        Mul => {
            if is_val(&m.kids[0], 0.0) || is_val(&m.kids[1], 0.0) {
                constant(0.0)
            } else if is_val(&m.kids[0], 1.0) {
                m.kids[1].clone()
            } else if is_val(&m.kids[1], 1.0) {
                m.kids[0].clone()
            } else if let (Some(a), Some(b)) = (cval(&m.kids[0]), cval(&m.kids[1])) {
                constant(a * b)
            } else {
                m
            }
        }
        Div => {
            if is_val(&m.kids[1], 1.0) {
                m.kids[0].clone()
            } else if let (Some(a), Some(b)) = (cval(&m.kids[0]), cval(&m.kids[1])) {
                if b.abs() > 1e-12 {
                    constant(a / b)
                } else {
                    m
                }
            } else {
                m
            }
        }
        Pow => {
            if is_val(&m.kids[1], 1.0) {
                m.kids[0].clone()
            } else if is_val(&m.kids[1], 0.0) {
                constant(1.0)
            } else if let (Some(a), Some(b)) = (cval(&m.kids[0]), cval(&m.kids[1])) {
                constant(a.powf(b))
            } else {
                m
            }
        }
        _ => m,
    }
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
