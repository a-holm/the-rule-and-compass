//! Expression trees: the search representation.
//!
//! The primitive set covers the usual building blocks of symbolic regression
//! (see ../SPEC.md section 4).

use crate::rng::Mt;

/// The expression parser, re-exported so callers can write [`crate::expr::parse`].
pub use crate::parse::{parse, parse_block, parse_blocks, parse_with, ParseError};

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
    // Meta operators: they describe the *shape* of their argument, not a row.
    // Opt-in only, and their value is cached when the node is built.
    Complexity,
    Variables,
    Constants,
    Contains,
    ContainsOp,
    // Dataset aggregates: a reduction over one input column, computed once
    // from the training rows and cached when the node is built. Opt-in only.
    Agg(AggKind),
    // User-defined building blocks (issue #10). Appended after the aggregates so
    // the aggregate indices stay where issue #8 put them; the four slots keep
    // every earlier discriminant meaningful.
    Custom0,
    Custom1,
    Custom2,
    Custom3,
}

/// The dataset-wide reductions behind [`Op::Agg`] (issue #8).
///
/// Every variant except [`AggKind::Count`] takes one input column as its child;
/// `count` is zero-arity and returns the number of training rows. The value is
/// a property of the **training sample**, not of a row, so it is computed once
/// into an [`AggTable`] and cached in [`Node::val`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggKind {
    Mean,
    Std,
    Median,
    Iqr,
    Sum,
    Min,
    Max,
    Count,
}

impl AggKind {
    /// Number of aggregate kinds; `Op::Agg` uses a dense index per kind.
    pub const COUNT: usize = 8;

    /// Every aggregate, in index order (the order of [`AggKind::index`]).
    pub const ALL: [AggKind; AggKind::COUNT] = [
        AggKind::Mean,
        AggKind::Std,
        AggKind::Median,
        AggKind::Iqr,
        AggKind::Sum,
        AggKind::Min,
        AggKind::Max,
        AggKind::Count,
    ];

    /// Canonical name used by `--ops`, the parser and `to_string`.
    ///
    /// `min` and `max` are already the binary primitives, so the column
    /// reductions are named `amin`/`amax`; the parser also accepts `min(x1)` /
    /// `max(x1)` with a single argument.
    pub fn name(self) -> &'static str {
        match self {
            AggKind::Mean => "mean",
            AggKind::Std => "std",
            AggKind::Median => "median",
            AggKind::Iqr => "iqr",
            AggKind::Sum => "sum",
            AggKind::Min => "amin",
            AggKind::Max => "amax",
            AggKind::Count => "count",
        }
    }

    pub fn arity(self) -> usize {
        if self == AggKind::Count {
            0
        } else {
            1
        }
    }

    /// Dense index in `0..COUNT`, stable for the checkpoint format.
    pub fn index(self) -> usize {
        match self {
            AggKind::Mean => 0,
            AggKind::Std => 1,
            AggKind::Median => 2,
            AggKind::Iqr => 3,
            AggKind::Sum => 4,
            AggKind::Min => 5,
            AggKind::Max => 6,
            AggKind::Count => 7,
        }
    }

    /// Map an `--ops` / parser name to an aggregate kind.
    pub fn from_name(s: &str) -> Option<AggKind> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "mean" => AggKind::Mean,
            "std" | "stddev" => AggKind::Std,
            "median" => AggKind::Median,
            "iqr" => AggKind::Iqr,
            "sum" => AggKind::Sum,
            "amin" | "agg_min" | "min_over" => AggKind::Min,
            "amax" | "agg_max" | "max_over" => AggKind::Max,
            "count" => AggKind::Count,
            _ => return None,
        })
    }
}

impl Op {
    /// Dense index into a cost table (`complexity::ComplexityModel`). Replaces
    /// the enum discriminant, which stopped being usable when [`Op::Agg`]
    /// gained a payload. A new variant must bump [`Op::COUNT`] (a test guards
    /// it) and must be appended to the checkpoint table in that order.
    pub const AGG_BASE: usize = 54;
    /// First user-defined building block (`Custom0`); the four slots are
    /// contiguous (issue #10).
    pub const CUSTOM_BASE: usize = Self::AGG_BASE + AggKind::COUNT;
    pub const COUNT: usize = Self::CUSTOM_BASE + 4;

    /// Dense index in `0..COUNT`; `Op::Agg(kind)` occupies one index per kind.
    pub fn index(self) -> usize {
        use Op::*;
        match self {
            Const => 0,
            Input => 1,
            Neg => 2,
            Exp => 3,
            Ln => 4,
            Abs => 5,
            Sqrt => 6,
            Sin => 7,
            Cos => 8,
            Tan => 9,
            Sinh => 10,
            Cosh => 11,
            Tanh => 12,
            Logistic => 13,
            Step => 14,
            Gauss => 15,
            Square => 16,
            Cube => 17,
            Floor => 18,
            Ceil => 19,
            Round => 20,
            Sign => 21,
            Erf => 22,
            Erfc => 23,
            Asinh => 24,
            Acosh => 25,
            Atanh => 26,
            Add => 27,
            Sub => 28,
            Mul => 29,
            Div => 30,
            Pow => 31,
            Mod => 32,
            Min => 33,
            Max => 34,
            Atan2 => 35,
            If => 36,
            Equal => 37,
            Less => 38,
            LessOrEqual => 39,
            Greater => 40,
            GreaterOrEqual => 41,
            And => 42,
            Or => 43,
            Xor => 44,
            Not => 45,
            Require => 46,
            Between => 47,
            Constrain => 48,
            Complexity => 49,
            Variables => 50,
            Constants => 51,
            Contains => 52,
            ContainsOp => 53,
            Agg(kind) => Self::AGG_BASE + kind.index(),
            Custom0 => Self::CUSTOM_BASE,
            Custom1 => Self::CUSTOM_BASE + 1,
            Custom2 => Self::CUSTOM_BASE + 2,
            Custom3 => Self::CUSTOM_BASE + 3,
        }
    }

    /// Whether this is a dataset aggregate. Like a meta operator it returns a
    /// cached value that does not depend on the row, but its cache comes from
    /// the data ([`AggTable`]) rather than from the tree's shape.
    pub fn is_agg(self) -> bool {
        matches!(self, Op::Agg(_))
    }

    /// The `Op::Custom(slot)` spelling used throughout the code for a
    /// user-defined building block; the slot is one of `0..=3`.
    ///
    /// The **arity** of a custom operator comes from the thread-local block
    /// registry installed with [`set_blocks`], because it is a property of the
    /// user's definition, not of the variant. A custom operator with no
    /// installed block has arity `0` and evaluates to `NaN`.
    #[allow(non_snake_case)]
    pub const fn Custom(slot: usize) -> Op {
        match slot {
            0 => Op::Custom0,
            1 => Op::Custom1,
            2 => Op::Custom2,
            _ => Op::Custom3,
        }
    }

    /// The slot of a user-defined block, or `None` for a primitive.
    pub fn custom_slot(self) -> Option<usize> {
        match self {
            Op::Custom0 => Some(0),
            Op::Custom1 => Some(1),
            Op::Custom2 => Some(2),
            Op::Custom3 => Some(3),
            _ => None,
        }
    }

    pub fn is_custom(self) -> bool {
        self.custom_slot().is_some()
    }

    /// Whether this operator inspects its argument's structure instead of
    /// evaluating it per row. Meta nodes cache their result in [`Node::val`];
    /// see [`meta`].
    pub fn is_meta(self) -> bool {
        use Op::*;
        matches!(
            self,
            Complexity | Variables | Constants | Contains | ContainsOp
        )
    }

    pub fn arity(self) -> usize {
        use Op::*;
        match self {
            Const | Input => 0,
            Agg(kind) => kind.arity(),
            Custom0 | Custom1 | Custom2 | Custom3 => {
                block_for_slot(self.custom_slot().unwrap())
                    .map(|b| b.params.len())
                    .unwrap_or(0)
            }
            Neg | Exp | Ln | Abs | Sqrt | Sin | Cos | Tan | Sinh | Cosh | Tanh | Logistic | Step
            | Gauss | Square | Cube | Floor | Ceil | Round | Sign | Erf | Erfc | Asinh | Acosh
            | Atanh | Not | Require | Constrain | Complexity | Variables | Constants | Contains
            | ContainsOp => 1,
            Add | Sub | Mul | Div | Pow | Mod | Min | Max | Atan2 | Equal | Less | LessOrEqual
            | Greater | GreaterOrEqual | And | Or | Xor => 2,
            If | Between => 3,
        }
    }

    pub fn symbol(self) -> &'static str {
        use Op::*;
        match self {
            Agg(kind) => kind.name(),
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
            Complexity => "complexity",
            Variables => "variables",
            Constants => "constants",
            Contains => "contains",
            ContainsOp => "contains_op",
            Custom0 => "custom0",
            Custom1 => "custom1",
            Custom2 => "custom2",
            Custom3 => "custom3",
        }
    }

    /// Default operator set for a search (all non-terminal primitives).
    ///
    /// The extended primitives (`Floor`…`Atanh`, `Mod`/`Min`/`Max`/`Atan2`),
    /// the conditional family (comparisons, `and`/`or`/`xor`/`not`, and the
    /// constraints `require`/`between`/`constrain`) and the meta operators
    /// (`Complexity`…`ContainsOp`) are not in the default set: they are opt-in
    /// through `--ops` so that broad searches are not diluted by them. Use
    /// [`Op::names`] for the full vocabulary.
    pub fn default_set() -> Vec<Op> {
        use Op::*;
        vec![
            Neg, Exp, Ln, Abs, Sqrt, Sin, Cos, Tan, Sinh, Cosh, Tanh, Logistic, Gauss, Square,
            Cube, Add, Sub, Mul, Div, Pow, If,
        ]
    }

    /// Canonical names accepted by [`Op::parse_list`], in a stable order. Used by
    /// the `schema` output so callers can discover the full primitive set.
    ///
    /// The four `custom0..custom3` slots are deliberately absent: they are
    /// enabled by `--function` and their arity comes from the supplied
    /// definition, so they are not part of the static vocabulary.
    pub fn names() -> &'static [&'static str] {
        &[
            "neg", "exp", "ln", "abs", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "logistic", "step", "gauss", "square", "cube", "floor", "ceil", "round", "sign",
            "erf", "erfc", "asinh", "acosh", "atanh", "mod", "min", "max", "atan2", "add", "sub",
            "mul", "div", "pow", "if", "equal", "less", "less_or_equal", "greater",
            "greater_or_equal", "and", "or", "xor", "not", "require", "between", "constrain",
            "complexity", "variables", "constants", "contains", "contains_op", "mean", "std",
            "median", "iqr", "sum", "amin", "amax", "count",
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

    /// Strict variant of [`Op::parse_list`] for command-line parsing: every
    /// non-empty token must name an operator, otherwise the whole list is
    /// rejected. An empty list still means the default set.
    pub fn parse_list_checked(s: &str) -> Result<Vec<Op>, String> {
        let mut out = Vec::new();
        for tok in s.split(|c| c == ',' || c == ' ') {
            if tok.is_empty() {
                continue;
            }
            match Op::from_name(tok) {
                Some(op) => out.push(op),
                None => return Err(format!("unknown operator '{tok}'")),
            }
        }
        if out.is_empty() {
            Ok(Op::default_set())
        } else {
            Ok(out)
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
            "complexity" => Complexity,
            "variables" => Variables,
            "constants" => Constants,
            "contains" => Contains,
            "contains_op" => ContainsOp,
            "custom0" => Custom0,
            "custom1" => Custom1,
            "custom2" => Custom2,
            "custom3" => Custom3,
            _ => {
                return AggKind::from_name(s).map(Op::Agg);
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub op: Op,
    /// `Const` -> value; `Input` -> variable index; a meta operator -> its
    /// cached structural value; an aggregate -> its cached dataset value
    /// (`NaN` until [`bind_aggregates`] or the sampler resolves it). Build meta
    /// nodes through [`meta`] or [`node`], and aggregate nodes through
    /// [`aggregate`], never from a literal.
    pub val: f64,
    pub kids: Vec<Node>,
    /// For `Op::Custom(slot)` only: the user's definition. Every use of the
    /// block carries its own copy so a candidate can hold fitted constants
    /// without mutating a process-wide table; the fitter keeps the copies of
    /// one slot in step (see `docs/building-blocks.md`).
    pub block: Option<Box<Block>>,
}

/// A user-defined building block (`--function`).
///
/// `params` are the names used inside `body`; the body refers to them through
/// [`Op::Input`] with the parameter's index, exactly like a compiled function.
/// A use of the block is an [`Op::Custom`] node whose children are the argument
/// expressions; [`eval`] substitutes their values for the parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    /// The name the user wrote (or `customN` when the definition had no name).
    pub name: String,
    /// Parameter names, in order of first appearance for `name=expr`.
    pub params: Vec<String>,
    /// Body in parameter space (`Input(i)` is `params[i]`).
    pub body: Node,
}

impl Block {
    /// The canonical slot name, used when a definition omits a parameter list.
    pub fn slot_name(slot: usize) -> String {
        format!("custom{slot}")
    }
}

thread_local! {
    /// The blocks visible to the current thread. Search islands inherit a copy
    /// explicitly (they are new threads); the CLI installs the command-line
    /// definitions before the run. Empty means "no `--function`", which keeps
    /// every historical code path unchanged.
    static BLOCKS: std::cell::RefCell<Vec<Block>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Install the process's user-defined blocks on the current thread.
pub fn set_blocks(blocks: Vec<Block>) {
    BLOCKS.with(|cell| *cell.borrow_mut() = blocks);
}

/// A copy of the blocks visible on the current thread.
pub fn current_blocks() -> Vec<Block> {
    BLOCKS.with(|cell| cell.borrow().clone())
}

fn block_for_slot(slot: usize) -> Option<Block> {
    BLOCKS.with(|cell| cell.borrow().get(slot).cloned())
}

/// Arity of the custom operator in `slot` under the current thread's blocks.
pub fn block_arity(slot: usize) -> usize {
    block_for_slot(slot).map(|b| b.params.len()).unwrap_or(0)
}

fn block_arity_of(op: Op) -> usize {
    op.arity()
}

pub fn constant(v: f64) -> Node {
    Node {
        op: Op::Const,
        val: v,
        kids: Vec::new(),
        block: None,
    }
}

pub fn input(i: usize) -> Node {
    Node {
        op: Op::Input,
        val: i as f64,
        kids: Vec::new(),
        block: None,
    }
}

/// Build an operator node. For a meta operator the structural value is
/// computed immediately and stored in `val`, so every constructor path (the
/// parser, `random_tree`, mutation and crossover) produces a consistent node.
/// An aggregate starts **unresolved** (`val = NaN`): its value is data, so the
/// sampler or [`bind_aggregates`] must supply it.
///
/// An [`Op::Custom`] node automatically carries a copy of the current thread's
/// block definition, which is what makes a generated call evaluable.
pub fn node(op: Op, kids: Vec<Node>) -> Node {
    let val = if op.is_meta() {
        debug_assert_eq!(kids.len(), 1, "meta operators are unary");
        meta_value(op, &kids[0])
    } else if op.is_agg() {
        f64::NAN
    } else {
        0.0
    };
    let block = op
        .custom_slot()
        .and_then(block_for_slot)
        .map(Box::new);
    Node {
        op,
        val,
        kids,
        block,
    }
}

/// Build a custom call with an explicit definition, for the parser while it
/// validates a new `--function` (the new block is not yet on the thread).
pub fn custom_node(op: Op, kids: Vec<Node>, block: Block) -> Node {
    debug_assert!(op.is_custom(), "{op:?} is not a custom operator");
    Node {
        op,
        val: 0.0,
        kids,
        block: Some(Box::new(block)),
    }
}

/// Build an aggregate node with its cached value.
///
/// `kind.arity() == 1` requires `column`; `count` takes none. Passing `None`
/// for a columned aggregate (or a column for `count`) is a programming error
/// and produces the correct shape with an unresolved value.
pub fn aggregate(kind: AggKind, column: Option<usize>, value: f64) -> Node {
    let kids = match (kind.arity(), column) {
        (1, Some(i)) => vec![input(i)],
        _ => Vec::new(),
    };
    Node {
        op: Op::Agg(kind),
        val: value,
        kids,
        block: None,
    }
}

/// The per-column aggregate values a search computed from its **training**
/// sample, once, before the first generation (issue #8).
///
/// `values[column][kind.index()]` is the cached statistic; `count` is
/// `rows`. Missing/non-finite cells are ignored by every aggregate except
/// `count`; a column with no finite value at all yields `NaN`, which makes the
/// candidate infeasible through the ordinary domain guard.
#[derive(Clone, Debug, PartialEq)]
pub struct AggTable {
    values: Vec<[f64; AggKind::COUNT]>,
    rows: usize,
}

impl AggTable {
    /// Build the table from rows in the sample's own space. This is the only
    /// place an aggregate is ever computed; nothing downstream recomputes it
    /// per row or per test row.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let nrows = rows.len();
        let nvars = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut values = Vec::with_capacity(nvars);
        for col in 0..nvars {
            let mut finite: Vec<f64> = rows
                .iter()
                .filter_map(|r| r.get(col).copied())
                .filter(|v| v.is_finite())
                .collect();
            let mut out = [f64::NAN; AggKind::COUNT];
            if !finite.is_empty() {
                let n = finite.len() as f64;
                let sum: f64 = finite.iter().sum();
                let mean = sum / n;
                let var = finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                out[AggKind::Mean.index()] = mean;
                out[AggKind::Std.index()] = var.sqrt();
                out[AggKind::Sum.index()] = sum;
                finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                out[AggKind::Min.index()] = finite[0];
                out[AggKind::Max.index()] = finite[finite.len() - 1];
                out[AggKind::Median.index()] = quantile(&finite, 0.5);
                out[AggKind::Iqr.index()] =
                    quantile(&finite, 0.75) - quantile(&finite, 0.25);
            }
            values.push(out);
        }
        AggTable { values, rows: nrows }
    }

    /// Build the table from a [`crate::data::Sample`]'s input columns. Only the
    /// rows in `sample` contribute; test data never does.
    pub fn from_sample(sample: &crate::data::Sample) -> Self {
        Self::from_rows(&sample.x)
    }

    /// Cached value of `kind` on `column`; `NaN` when the column is unknown or
    /// has no finite value. `count` is always the number of rows.
    pub fn get(&self, kind: AggKind, column: usize) -> f64 {
        if kind == AggKind::Count {
            return self.rows as f64;
        }
        self.values
            .get(column)
            .map(|v| v[kind.index()])
            .unwrap_or(f64::NAN)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn nvars(&self) -> usize {
        self.values.len()
    }
}

/// Linear-interpolated quantile (`p` in `[0, 1]`) of a sorted slice, the same
/// convention as the data-preparation code. Empty slices return `NaN`.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let f = pos - lo as f64;
        sorted[lo] * (1.0 - f) + sorted[hi] * f
    }
}

/// Resolve every aggregate in `tree` against `table`, bottom-up.
///
/// The parser cannot know the data, so it leaves each aggregate with `val =
/// NaN`; this is the step that turns `mean(x1)` into a number. An aggregate
/// whose child is not a bare input column, or a column outside the table, is an
/// error: the value must never be invented.
pub fn bind_aggregates(tree: &mut Node, table: &AggTable) -> Result<(), String> {
    for k in &mut tree.kids {
        bind_aggregates(k, table)?;
    }
    if let Op::Agg(kind) = tree.op {
        if kind.arity() == 0 {
            tree.val = table.get(kind, 0);
            return Ok(());
        }
        let child = tree
            .kids
            .first()
            .ok_or_else(|| format!("{}() is missing its input column", kind.name()))?;
        if child.op != Op::Input {
            return Err(format!(
                "{}() takes a single input column (for example {}(x1)), not an expression",
                kind.name(),
                kind.name()
            ));
        }
        let column = child.val as usize;
        if column >= table.nvars() {
            return Err(format!(
                "{}() refers to column x{}, but the data has {} input column(s)",
                kind.name(),
                column,
                table.nvars()
            ));
        }
        tree.val = table.get(kind, column);
    }
    Ok(())
}

/// Recompute every aggregate's cached value after a structural edit. With no
/// table the cache is left alone (an unbound tree keeps its `NaN`).
///
/// A structural edit can also replace the column child itself. An aggregate
/// must reduce a real column, so a malformed child is repaired to the first
/// input column (the only well-formed child) rather than left behind.
fn refresh_aggregates(n: &mut Node, table: Option<&AggTable>) {
    for k in &mut n.kids {
        refresh_aggregates(k, table);
    }
    if let (Some(table), Op::Agg(kind)) = (table, n.op) {
        if kind.arity() == 0 {
            n.val = table.get(kind, 0);
            return;
        }
        let column = n
            .kids
            .first()
            .filter(|c| c.op == Op::Input)
            .map(|c| c.val as usize)
            .filter(|&c| c < table.nvars());
        match column {
            Some(c) => n.val = table.get(kind, c),
            None => {
                if table.nvars() > 0 {
                    n.kids = vec![input(0)];
                    n.val = table.get(kind, 0);
                } else {
                    n.val = f64::NAN;
                }
            }
        }
    }
}

/// Build a meta operator node from `kind` and its one child.
///
/// The returned node caches the structural value in [`Node::val`];
/// [`eval`] returns that cache without walking the child. The result depends
/// only on the child's shape (and, for `variables`, its input indices), never
/// on a data row.
pub fn meta(kind: Op, child: Node) -> Node {
    debug_assert!(kind.is_meta(), "{kind:?} is not a meta operator");
    node(kind, vec![child])
}

/// The cached structural value of `kind` applied to `child`.
///
/// | operator | value |
/// |---|---|
/// | `complexity(e)` | node count of `e` |
/// | `variables(e)` | number of distinct input variables in `e` |
/// | `constants(e)` | number of constant leaves in `e` |
/// | `contains(e)` | `1` if `e` contains an input variable, else `0` |
/// | `contains_op(e)` | `1` if `e` contains an operator node, else `0` |
fn meta_value(kind: Op, child: &Node) -> f64 {
    use Op::*;
    match kind {
        Complexity => complexity(child) as f64,
        Variables => distinct_variables(child) as f64,
        Constants => count_constants(child) as f64,
        Contains => {
            if has_input(child) {
                1.0
            } else {
                0.0
            }
        }
        ContainsOp => {
            if has_operator(child) {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Number of distinct input indices appearing in `n`.
fn distinct_variables(n: &Node) -> usize {
    fn collect(n: &Node, out: &mut Vec<usize>) {
        if n.op == Op::Input {
            out.push(n.val as usize);
        }
        for k in &n.kids {
            collect(k, out);
        }
    }
    let mut out = Vec::new();
    collect(n, &mut out);
    out.sort_unstable();
    out.dedup();
    out.len()
}

fn has_input(n: &Node) -> bool {
    n.op == Op::Input || n.kids.iter().any(has_input)
}

/// Whether the tree contains any operator node. A bare leaf (`Const` or
/// `Input`) has none; any other root is an operator.
fn has_operator(n: &Node) -> bool {
    n.op != Op::Const && n.op != Op::Input
}

/// Recompute the cached value of every meta node bottom-up. Used after a
/// structural edit (mutation, crossover) that may have changed a meta
/// argument's shape.
fn refresh_meta(n: &mut Node) {
    for k in &mut n.kids {
        refresh_meta(k);
    }
    if n.op.is_meta() {
        debug_assert_eq!(n.kids.len(), 1, "meta operators are unary");
        n.val = meta_value(n.op, &n.kids[0]);
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
        // A block call: evaluate the arguments, then evaluate the body with
        // `Input(i)` bound to argument `i`. The body is parameter space, so no
        // row is involved beyond the arguments' own inputs.
        Custom0 | Custom1 | Custom2 | Custom3 => {
            let Some(block) = &n.block else {
                return f64::NAN;
            };
            let args: Vec<f64> = n.kids.iter().map(|k| eval(k, x)).collect();
            eval(&block.body, &args)
        }
        Input => {
            let i = n.val as usize;
            if i < x.len() {
                x[i]
            } else {
                f64::NAN
            }
        }
        // Meta operators describe the tree, not the row: return the value
        // cached when the node was built. Never walk the child here; `eval` is
        // the innermost loop of the engine.
        Complexity | Variables | Constants | Contains | ContainsOp => {
            debug_assert_eq!(n.kids.len(), 1, "meta operators are unary");
            debug_assert!(
                n.val.is_finite(),
                "meta operator {op:?} has a non-finite cached value",
                op = n.op
            );
            n.val
        }
        // An aggregate is a dataset fact: return the value cached when the node
        // was built. Never walk the column child here, and never look at the
        // row. `NaN` means the statistic is undefined (e.g. an empty column),
        // which the metric turns into infeasible.
        Agg(_) => n.val,
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
///
/// A meta operator is never counted itself. Constant leaves *inside* its
/// argument are real leaves and stay counted, but the zero gradient through a
/// meta node means the fitter cannot identify them from the model output.
/// An aggregate has no constant leaf at all: it is a dataset fact, not a fitted
/// parameter, so it never contributes to `k`.
///
/// Constants inside a block body are **shared** by every use of that block, so
/// they are counted once per distinct slot no matter how many calls appear.
pub fn count_constants(n: &Node) -> usize {
    let mut seen = [false; 4];
    count_constants_seen(n, &mut seen)
}

fn count_constants_seen(n: &Node, seen: &mut [bool; 4]) -> usize {
    let mut total = usize::from(n.op == Op::Const);
    total += n
        .kids
        .iter()
        .map(|k| count_constants_seen(k, seen))
        .sum::<usize>();
    if let Some(slot) = n.op.custom_slot() {
        if !seen[slot] {
            seen[slot] = true;
            if let Some(block) = &n.block {
                total += count_constants_seen(&block.body, seen);
            }
        }
    }
    total
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
    random_tree_with(rng, nvars, ops, depth, None)
}

/// [`random_tree`] with the training-sample aggregate table.
///
/// When the sampler emits [`Op::Agg`] it reads the precomputed statistic from
/// `aggs` (one lookup, no pass over the rows). With no table the node is built
/// unresolved (`NaN`), which the metric treats as infeasible.
pub fn random_tree_with(
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    depth: usize,
    aggs: Option<&AggTable>,
) -> Node {
    random_tree_opts(rng, nvars, ops, depth, aggs, false)
}

/// [`random_tree_with`] with the operator draw of `--tree-arity`. `balanced`
/// draws the arity class first (binary 0.5, unary 0.4, ternary 0.1, renormalised
/// over the classes present in `ops`) and an operator uniformly inside it.
pub fn random_tree_opts(
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    depth: usize,
    aggs: Option<&AggTable>,
    balanced: bool,
) -> Node {
    if depth == 0 || rng.bool(0.15) {
        return random_terminal(rng, nvars);
    }
    let op = if balanced { pick_balanced(rng, ops) } else { *rng.pick(ops) };
    if let Op::Agg(kind) = op {
        return random_aggregate(rng, nvars, kind, aggs);
    }
    let k = op.arity();
    let kids = (0..k)
        .map(|_| random_tree_opts(rng, nvars, ops, depth - 1, aggs, balanced))
        .collect();
    node(op, kids)
}

/// A column aggregate with its value read from `aggs`; `count()` needs none.
/// With no input column there is nothing to reduce, so the sampler falls back
/// to a constant rather than emitting a malformed node.
fn random_aggregate(
    rng: &mut Mt,
    nvars: usize,
    kind: AggKind,
    aggs: Option<&AggTable>,
) -> Node {
    if kind.arity() == 0 {
        return aggregate(
            kind,
            None,
            aggs.map(|t| t.get(kind, 0)).unwrap_or(f64::NAN),
        );
    }
    if nvars == 0 {
        return constant(0.0);
    }
    let column = rng.usize_below(nvars);
    aggregate(
        kind,
        Some(column),
        aggs.map(|t| t.get(kind, column)).unwrap_or(f64::NAN),
    )
}

fn perturb(rng: &mut Mt, v: f64) -> f64 {
    match rng.usize_below(4) {
        0 => round_sig(v * rng.range(0.3, 1.8).max(0.05)),
        1 => round_sig(v + rng.range(-1.0, 1.0)),
        2 => round_sig(rng.range(-2.0, 2.0)),
        _ => round_sig(v + rng.range(-0.2, 0.2)),
    }
}

/// A perturbed constant that is guaranteed to change its bits.
fn perturb_changed(rng: &mut Mt, v: f64) -> f64 {
    for _ in 0..4 {
        let w = perturb(rng, v);
        if w.to_bits() != v.to_bits() {
            return w;
        }
    }
    // Four draws can all round back to `v`; take a fresh value instead.
    let w = round_sig(rng.range(-2.0, 2.0));
    if w.to_bits() != v.to_bits() {
        w
    } else {
        round_sig(w + 1.0)
    }
}

/// One structural edit of the node-path mutation (#471).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditKind {
    Point,
    Terminal,
    Subtree,
    Insert,
    Delete,
    Swap,
    Rotate,
}

/// The number of edit kinds; the length of a [`MixWeights`] array.
pub const EDIT_KINDS: usize = 7;

/// Edit weights in [`EditKind::ALL`] order.
pub type MixWeights = [f64; EDIT_KINDS];

/// The historical draw: point 25 %, terminal 35 %, subtree 40 %.
pub const CLASSIC_MIX: MixWeights = [0.25, 0.35, 0.40, 0.0, 0.0, 0.0, 0.0];

/// `--mutation-mix extended`.
pub const EXTENDED_MIX: MixWeights = [0.20, 0.30, 0.20, 0.10, 0.10, 0.05, 0.05];

impl EditKind {
    pub const ALL: [EditKind; EDIT_KINDS] = [
        EditKind::Point,
        EditKind::Terminal,
        EditKind::Subtree,
        EditKind::Insert,
        EditKind::Delete,
        EditKind::Swap,
        EditKind::Rotate,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            EditKind::Point => "point",
            EditKind::Terminal => "terminal",
            EditKind::Subtree => "subtree",
            EditKind::Insert => "insert",
            EditKind::Delete => "delete",
            EditKind::Swap => "swap",
            EditKind::Rotate => "rotate",
        }
    }
}

/// How the node path varies a tree: edit weights, the random-tree operator
/// draw and the crossover cut-point bias. The default is the historical
/// behaviour bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariationOpts {
    pub mix: MixWeights,
    /// Arity-balanced operator draw in random trees (`--tree-arity balanced`).
    pub balanced: bool,
    /// Probability that a crossover cut point is an internal node
    /// (`--crossover-internal`); `0` keeps the uniform choice.
    pub crossover_internal: f64,
}

impl Default for VariationOpts {
    fn default() -> Self {
        VariationOpts {
            mix: CLASSIC_MIX,
            balanced: false,
            crossover_internal: 0.0,
        }
    }
}

/// The kind a uniform draw `r` selects under `w`.
fn draw_kind(r: f64, w: &MixWeights) -> EditKind {
    let mut acc = 0.0;
    let mut last = EditKind::Subtree;
    for k in EditKind::ALL {
        let wk = w[k.index()];
        if wk <= 0.0 {
            continue;
        }
        acc += wk;
        last = k;
        if r < acc {
            return k;
        }
    }
    last
}

/// Operators that are not commutative, so swapping the operands changes the value.
fn noncommutative(op: Op) -> bool {
    matches!(
        op,
        Op::Sub
            | Op::Div
            | Op::Pow
            | Op::Mod
            | Op::Atan2
            | Op::Less
            | Op::LessOrEqual
            | Op::Greater
            | Op::GreaterOrEqual
    )
}

/// A built-in binary operator that rotate and swap may rebuild.
fn plain_binary(t: &Node) -> bool {
    t.kids.len() == 2 && t.op.arity() == 2 && !t.op.is_custom() && !t.op.is_agg()
}

/// Whether `op` may wrap `site` in an insert edit.
fn insertable(op: Op, site: &Node) -> bool {
    op.arity() >= 1
        && !op.is_agg()
        && !op.is_meta()
        && !(op == Op::Neg && site.op == Op::Neg)
}

/// Draw an operator from `ops` with the `--tree-arity balanced` class weights.
fn pick_balanced(rng: &mut Mt, ops: &[Op]) -> Op {
    const WEIGHT: [f64; 3] = [0.5, 0.4, 0.1];
    fn class(op: Op) -> usize {
        match op.arity() {
            2 => 0,
            a if a >= 3 => 2,
            _ => 1,
        }
    }
    let mut count = [0usize; 3];
    for &o in ops {
        count[class(o)] += 1;
    }
    let total: f64 = (0..3).filter(|&c| count[c] > 0).map(|c| WEIGHT[c]).sum();
    if total <= 0.0 {
        return *rng.pick(ops);
    }
    let r = rng.next_f64() * total;
    let mut acc = 0.0;
    let mut chosen = 0;
    for c in 0..3 {
        if count[c] == 0 {
            continue;
        }
        acc += WEIGHT[c];
        chosen = c;
        if r < acc {
            break;
        }
    }
    let mut k = rng.usize_below(count[chosen]);
    for &o in ops {
        if class(o) == chosen {
            if k == 0 {
                return o;
            }
            k -= 1;
        }
    }
    unreachable!("a counted class has members")
}

/// Whether a point change can actually change this node: it must be an
/// operator and `ops` must offer another operator of the same arity.
fn point_changeable(t: &Node, nvars: usize, ops: &[Op]) -> bool {
    let ar = t.op.arity();
    ar > 0
        && ops.iter().any(|&o| {
            block_arity_of(o) == ar && o != t.op && !(o.is_agg() && nvars == 0)
        })
}

/// Whether a terminal edit can change this node. A univariate input has no
/// other column to move to, so it is handled by a point edit instead.
fn terminal_changeable(t: &Node, nvars: usize) -> bool {
    t.op == Op::Const || (t.op == Op::Input && nvars > 1)
}

/// Pre-order positions of the nodes `kind` can edit, in one pass. `mutate_with`
/// therefore still costs O(nodes), like [`nth_mut`].
fn mutation_sites(n: &Node, kind: EditKind, nvars: usize, ops: &[Op]) -> Vec<usize> {
    fn walk(
        n: &Node,
        under_agg: bool,
        kind: EditKind,
        nvars: usize,
        ops: &[Op],
        i: &mut usize,
        out: &mut Vec<usize>,
    ) {
        let editable = match kind {
            EditKind::Point => point_changeable(n, nvars, ops),
            EditKind::Terminal => terminal_changeable(n, nvars),
            EditKind::Subtree => true,
            // An aggregate keeps its input-column child.
            EditKind::Insert => !under_agg && ops.iter().any(|&o| insertable(o, n)),
            EditKind::Delete => !n.kids.is_empty() && !n.op.is_agg(),
            EditKind::Swap => {
                plain_binary(n) && noncommutative(n.op) && n.kids[0] != n.kids[1]
            }
            EditKind::Rotate => plain_binary(n) && plain_binary(&n.kids[1]),
        };
        if editable {
            out.push(*i);
        }
        *i += 1;
        for k in &n.kids {
            walk(k, n.op.is_agg(), kind, nvars, ops, i, out);
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    walk(n, false, kind, nvars, ops, &mut i, &mut out);
    out
}

/// Point-change an operator to a different operator of the same arity.
fn change_operator(
    t: &mut Node,
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    aggs: Option<&AggTable>,
) {
    let ar = t.op.arity();
    let cands: Vec<Op> = ops
        .iter()
        .copied()
        .filter(|o| block_arity_of(*o) == ar && *o != t.op)
        .filter(|o| !(o.is_agg() && nvars == 0))
        .collect();
    if cands.is_empty() {
        return;
    }
    let new_op = cands[rng.usize_below(cands.len())];
    if let Op::Agg(kind) = new_op {
        // An aggregate's single child is its column, never an arbitrary
        // expression.
        let column = rng.usize_below(nvars);
        t.op = new_op;
        t.kids = vec![input(column)];
        t.val = aggs
            .map(|a| a.get(kind, column))
            .unwrap_or(f64::NAN);
        t.block = None;
    } else {
        t.op = new_op;
        // `val` belongs to the old operator; meta operators get a fresh
        // structural value in the refresh below.
        t.val = 0.0;
        // #10: a custom operator carries its block definition.
        t.block = t.op.custom_slot().and_then(block_for_slot).map(Box::new);
        t.kids.truncate(t.op.arity());
        while t.kids.len() < t.op.arity() {
            t.kids.push(constant(1.0));
        }
    }
}

/// Perturb a constant, or move an input to a different column. Both paths
/// change the node whenever they are called; [`terminal_changeable`] only
/// offers sites where that is possible.
fn change_terminal(t: &mut Node, rng: &mut Mt, nvars: usize) {
    if t.op == Op::Const {
        t.val = perturb_changed(rng, t.val);
    } else if t.op == Op::Input && nvars > 1 {
        let cur = t.val as usize;
        let mut column = rng.usize_below(nvars - 1);
        if column >= cur {
            column += 1;
        }
        t.val = column as f64;
    }
}

/// A leaf that differs structurally from `t`.
fn different_leaf(t: &Node) -> Node {
    if t.op == Op::Const && t.val == 0.0 {
        constant(1.0)
    } else {
        constant(0.0)
    }
}

/// Replace `t` with a random subtree; retry rather than return the same shape.
fn replace_subtree(
    t: &mut Node,
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    aggs: Option<&AggTable>,
    balanced: bool,
) {
    let old = structural_hash(t);
    let depth = if rng.next_f64() < 0.625 { 2 } else { 1 };
    for _ in 0..8 {
        let cand = if ops.is_empty() {
            different_leaf(t)
        } else {
            random_tree_opts(rng, nvars, ops, depth, aggs, balanced)
        };
        if structural_hash(&cand) != old {
            *t = cand;
            return;
        }
    }
    *t = different_leaf(t);
}

/// Wrap `t` in a random operator, `o(t, new...)`, the other operands being
/// random terminals.
fn insert_operator(t: &mut Node, rng: &mut Mt, nvars: usize, ops: &[Op]) {
    let cands: Vec<Op> = ops.iter().copied().filter(|&o| insertable(o, t)).collect();
    if cands.is_empty() {
        return;
    }
    let op = cands[rng.usize_below(cands.len())];
    let arity = op.arity();
    let slot = rng.usize_below(arity);
    let site = std::mem::replace(t, constant(0.0));
    let mut site = Some(site);
    let kids = (0..arity)
        .map(|i| {
            if i == slot {
                site.take().expect("the site is placed once")
            } else {
                random_terminal(rng, nvars)
            }
        })
        .collect();
    *t = node(op, kids);
}

/// Replace `t` by one of its children, chosen uniformly.
fn delete_operator(t: &mut Node, rng: &mut Mt) {
    if t.kids.is_empty() {
        return;
    }
    let i = rng.usize_below(t.kids.len());
    let child = std::mem::replace(&mut t.kids[i], constant(0.0));
    *t = child;
}

/// `o1(a, o2(b, c))` becomes `o2(o1(a, b), c)`.
fn rotate_tree(t: &mut Node) {
    let o1 = t.op;
    let mut q = t.kids.pop().expect("a rotate site is binary");
    let a = t.kids.pop().expect("a rotate site is binary");
    let o2 = q.op;
    let c = q.kids.pop().expect("a rotate child is binary");
    let b = q.kids.pop().expect("a rotate child is binary");
    *t = node(o2, vec![node(o1, vec![a, b]), c]);
}

/// In-place mutation: point-change an operator, perturb a constant, or replace a
/// subtree (grow / shrink). [I]
pub fn mutate(n: &mut Node, rng: &mut Mt, nvars: usize, ops: &[Op]) {
    mutate_with(n, rng, nvars, ops, None)
}

/// [`mutate`] with the training-sample aggregate table, so a node switched to
/// [`Op::Agg`] (or an input column changed under one) gets the right cached
/// value immediately.
///
/// The edit kind is drawn first — point change (25 %), terminal perturb (35 %)
/// or subtree replacement (40 %) — and only nodes that kind can actually edit
/// are candidate sites, so a mutation never returns the parent unchanged. If a
/// kind has no compatible site, a fixed fallback order is used; subtree
/// replacement covers every non-empty tree.
pub fn mutate_with(
    n: &mut Node,
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    aggs: Option<&AggTable>,
) {
    mutate_with_opts(n, rng, nvars, ops, aggs, &VariationOpts::default());
}

/// [`mutate_with`] with the edit weights and tree draw of `opts`. Returns the
/// kind that changed the tree, or `None` when only the last-resort leaf
/// replacement (or nothing, for an empty tree) applied.
pub fn mutate_with_opts(
    n: &mut Node,
    rng: &mut Mt,
    nvars: usize,
    ops: &[Op],
    aggs: Option<&AggTable>,
    opts: &VariationOpts,
) -> Option<EditKind> {
    let total = complexity(n);
    if total == 0 {
        return None;
    }
    let before = structural_hash(n);
    let drawn = draw_kind(rng.next_f64(), &opts.mix);
    let order: &[EditKind] = match drawn {
        EditKind::Point => &[EditKind::Point, EditKind::Subtree],
        EditKind::Terminal => &[EditKind::Terminal, EditKind::Point, EditKind::Subtree],
        EditKind::Subtree => &[EditKind::Subtree, EditKind::Point, EditKind::Terminal],
        EditKind::Insert => &[EditKind::Insert, EditKind::Subtree, EditKind::Point, EditKind::Terminal],
        EditKind::Delete => &[EditKind::Delete, EditKind::Subtree, EditKind::Point, EditKind::Terminal],
        EditKind::Swap => &[EditKind::Swap, EditKind::Subtree, EditKind::Point, EditKind::Terminal],
        EditKind::Rotate => &[EditKind::Rotate, EditKind::Subtree, EditKind::Point, EditKind::Terminal],
    };
    for &kind in order {
        let sites = mutation_sites(n, kind, nvars, ops);
        if sites.is_empty() {
            continue;
        }
        let i = sites[rng.usize_below(sites.len())];
        if let Some(t) = nth_mut(n, i) {
            match kind {
                EditKind::Point => change_operator(t, rng, nvars, ops, aggs),
                EditKind::Terminal => change_terminal(t, rng, nvars),
                EditKind::Subtree => replace_subtree(t, rng, nvars, ops, aggs, opts.balanced),
                EditKind::Insert => insert_operator(t, rng, nvars, ops),
                EditKind::Delete => delete_operator(t, rng),
                EditKind::Swap => t.kids.swap(0, 1),
                EditKind::Rotate => rotate_tree(t),
            }
        }
        // A structural edit anywhere below a meta node invalidates its cache.
        refresh_meta(n);
        // The same is true for an aggregate when the edit changed its column
        // (or replaced the node with a malformed child).
        refresh_aggregates(n, aggs);
        if structural_hash(n) != before {
            return Some(kind);
        }
    }
    // Every kind fell through, or an aggregate repair undid the edit. Replacing
    // the root with a different constant cannot be a no-op.
    let leaf = different_leaf(n);
    *n = leaf;
    None
}

/// Replace a random subtree of `a` with a random subtree of `b` (crossover). [I]
pub fn crossover(a: &mut Node, b: &Node, rng: &mut Mt) {
    crossover_with(a, b, rng, None)
}

/// [`crossover`] with the training-sample aggregate table.
pub fn crossover_with(a: &mut Node, b: &Node, rng: &mut Mt, aggs: Option<&AggTable>) {
    crossover_with_opts(a, b, rng, aggs, 0.0)
}

/// Pre-order positions of the internal nodes of `n`.
fn internal_sites(n: &Node) -> Vec<usize> {
    fn walk(n: &Node, i: &mut usize, out: &mut Vec<usize>) {
        if !n.kids.is_empty() {
            out.push(*i);
        }
        *i += 1;
        for k in &n.kids {
            walk(k, i, out);
        }
    }
    let mut out = Vec::new();
    let mut i = 0;
    walk(n, &mut i, &mut out);
    out
}

/// A cut point: an internal node with probability `internal` when the tree has
/// one, else uniform over all nodes. `internal == 0` draws exactly as the
/// uniform crossover does.
fn cut_point(n: &Node, total: usize, rng: &mut Mt, internal: f64) -> usize {
    if internal > 0.0 && rng.bool(internal) {
        let sites = internal_sites(n);
        if !sites.is_empty() {
            return sites[rng.usize_below(sites.len())];
        }
    }
    rng.usize_below(total)
}

/// [`crossover_with`] with `--crossover-internal`: the probability that each
/// cut point is an internal node.
pub fn crossover_with_opts(
    a: &mut Node,
    b: &Node,
    rng: &mut Mt,
    aggs: Option<&AggTable>,
    internal: f64,
) {
    let ta = complexity(a);
    let tb = complexity(b);
    if ta == 0 || tb == 0 {
        return;
    }
    let ia = cut_point(a, ta, rng, internal);
    let ib = cut_point(b, tb, rng, internal);
    if let Some(sub) = nth(b, ib).cloned() {
        if let Some(t) = nth_mut(a, ia) {
            *t = sub;
        }
    }
    // The graft may sit below a meta node, whose cached value must follow the
    // new shape.
    refresh_meta(a);
    // Likewise an aggregate whose column child was replaced.
    refresh_aggregates(a, aggs);
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
        // A block prints by the user's name (or `customN`), with its arguments.
        Custom0 | Custom1 | Custom2 | Custom3 => {
            let name = n
                .block
                .as_ref()
                .map(|b| b.name.clone())
                .unwrap_or_else(|| n.op.symbol().to_string());
            let args: Vec<String> = n.kids.iter().map(|k| to_string(k, vars)).collect();
            format!("{}({})", name, args.join(", "))
        }
        Input => {
            let i = n.val as usize;
            vars.get(i)
                .cloned()
                .unwrap_or_else(|| format!("x{}", i))
        }
        Neg => format!("(-{})", to_string(&n.kids[0], vars)),
        Op::Agg(kind) => {
            if kind.arity() == 0 {
                format!("{}()", kind.name())
            } else {
                format!("{}({})", kind.name(), to_string(&n.kids[0], vars))
            }
        }
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

/// Structural equality (exact shape, constant values and a custom block's
/// definition).
pub fn same(a: &Node, b: &Node) -> bool {
    a.op == b.op
        && a.val.to_bits() == b.val.to_bits()
        && a.kids.len() == b.kids.len()
        && a.kids.iter().zip(&b.kids).all(|(x, y)| same(x, y))
        && a.block == b.block
}

/// #477: whether the tree can depend on the row: an `Input` occurs outside
/// every aggregate and meta operator. Aggregates and meta operators evaluate to
/// constants, and a user-defined block is assumed to read the row.
pub fn row_dependent(n: &Node) -> bool {
    match n.op {
        Op::Input => true,
        op if op.is_agg() || op.is_meta() => false,
        _ => n.block.is_some() || n.kids.iter().any(row_dependent),
    }
}

/// FNV-1a structural hash, for de-duplicating equivalent candidates before
/// they are evaluated.
pub fn structural_hash(n: &Node) -> u64 {
    fn mix(h: &mut u64, x: u64) {
        *h ^= x;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn go(n: &Node, h: &mut u64) {
        mix(h, n.op.index() as u64);
        mix(h, n.val.to_bits());
        mix(h, n.kids.len() as u64);
        for k in &n.kids {
            go(k, h);
        }
        // A custom call's fitted body constants are part of its structure.
        if let Some(block) = &n.block {
            mix(h, 1);
            for b in block.name.bytes() {
                mix(h, b as u64);
            }
            mix(h, block.params.len() as u64);
            go(&block.body, h);
        } else {
            mix(h, 0);
        }
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    go(n, &mut h);
    h
}


// ---------------------------------------------------------------------------
// Definedness and range analysis (issue #143)
// ---------------------------------------------------------------------------

/// A conservative non-relational abstract domain. `Fin` means every value in
/// the interval is finite; `MayNaN` also covers infinities and failed guards.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Domain {
    Fin { lo: f64, hi: f64 },
    MayNaN,
}

/// Interval plus a proof that zero is outside the interval.
#[derive(Clone, Copy, Debug)]
struct Info {
    dom: Domain,
    nonzero: bool,
}

impl Info {
    fn unknown() -> Self {
        Info {
            dom: Domain::MayNaN,
            nonzero: false,
        }
    }

    fn fin(lo: f64, hi: f64) -> Option<Self> {
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            Some(Info {
                dom: Domain::Fin { lo, hi },
                nonzero: lo > 0.0 || hi < 0.0,
            })
        } else {
            None
        }
    }

    fn total(self) -> bool {
        matches!(self.dom, Domain::Fin { .. })
    }

    fn bounds(self) -> Option<(f64, f64)> {
        match self.dom {
            Domain::Fin { lo, hi } => Some((lo, hi)),
            Domain::MayNaN => None,
        }
    }
}

fn next_down(x: f64) -> f64 {
    if x.is_finite() {
        x.next_down()
    } else {
        x
    }
}

fn next_up(x: f64) -> f64 {
    if x.is_finite() {
        x.next_up()
    } else {
        x
    }
}

fn add_bounds(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    let lo = a.0 + b.0;
    let hi = a.1 + b.1;
    if lo.is_finite() && hi.is_finite() {
        Some((next_down(lo), next_up(hi)))
    } else {
        None
    }
}

fn sub_bounds(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    add_bounds(a, (-b.1, -b.0))
}

fn mul_bounds(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    if p.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let lo = p.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((next_down(lo), next_up(hi)))
}

fn div_bounds(a: (f64, f64), b: (f64, f64)) -> Option<(f64, f64)> {
    // `eval` returns NaN when |denominator| < 1e-12, so a total division needs
    // the whole denominator interval on one side of that guard.
    if !(b.0 > 1e-12 || b.1 < -1e-12) {
        return None;
    }
    let q = [a.0 / b.0, a.0 / b.1, a.1 / b.0, a.1 / b.1];
    if q.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let lo = q.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((next_down(lo), next_up(hi)))
}

fn pow_int_bounds(base: (f64, f64), n: i32) -> Option<(f64, f64)> {
    if n == 0 {
        return Some((1.0, 1.0));
    }
    let mut acc = (1.0, 1.0);
    let mut b = base;
    let mut k = n.unsigned_abs();
    while k > 0 {
        if k & 1 == 1 {
            acc = mul_bounds(acc, b)?;
        }
        k >>= 1;
        if k > 0 {
            b = mul_bounds(b, b)?;
        }
    }
    if n < 0 {
        div_bounds((1.0, 1.0), acc)
    } else {
        Some(acc)
    }
}

fn join(a: Info, b: Info) -> Info {
    match (a.dom, b.dom) {
        (Domain::Fin { lo: al, hi: ah }, Domain::Fin { lo: bl, hi: bh }) => {
            Info::fin(al.min(bl), ah.max(bh)).unwrap_or_else(Info::unknown)
        }
        _ => Info::unknown(),
    }
}

fn round_info(x: Info, lo: f64, hi: f64) -> Info {
    if x.total() {
        Info::fin(lo, hi).unwrap_or_else(Info::unknown)
    } else {
        Info::unknown()
    }
}

/// Abstract interpretation of definedness and ranges for one tree.
///
/// `bounds` supplies `[lo, hi]` for input columns; `None` treats an input as
/// possibly non-finite (the conservative choice for the no-data simplifier).
/// `env` supplies parameter intervals for a block body. The analysis is
/// non-relational and O(nodes): it is the proof obligation behind the guarded
/// cancellations in [`simplify`].
fn analyze(n: &Node, bounds: Option<&[(f64, f64)]>, env: Option<&[Domain]>) -> Info {
    use Op::*;
    if n.op.is_meta() || n.op.is_agg() {
        return if n.val.is_finite() {
            Info::fin(n.val, n.val).unwrap_or_else(Info::unknown)
        } else {
            Info::unknown()
        };
    }
    if n.op.is_custom() {
        let Some(block) = &n.block else {
            return Info::unknown();
        };
        let args: Vec<Domain> = n
            .kids
            .iter()
            .map(|k| analyze(k, bounds, env).dom)
            .collect();
        return analyze(&block.body, None, Some(&args));
    }
    match n.op {
        Const => {
            if n.val.is_finite() {
                Info::fin(n.val, n.val).unwrap_or_else(Info::unknown)
            } else {
                Info::unknown()
            }
        }
        Input => {
            let i = n.val as usize;
            if let Some(env) = env {
                return match env.get(i).copied().unwrap_or(Domain::MayNaN) {
                    Domain::Fin { lo, hi } => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                    Domain::MayNaN => Info::unknown(),
                };
            }
            let Some((lo, hi)) = bounds.and_then(|b| b.get(i)).copied() else {
                return Info::unknown();
            };
            Info::fin(lo, hi).unwrap_or_else(Info::unknown)
        }
        Neg => match analyze(&n.kids[0], bounds, env).bounds() {
            Some((lo, hi)) => Info::fin(-hi, -lo).unwrap_or_else(Info::unknown),
            None => Info::unknown(),
        },
        Abs => match analyze(&n.kids[0], bounds, env).bounds() {
            Some((lo, hi)) if hi < 0.0 => Info::fin(-hi, -lo).unwrap_or_else(Info::unknown),
            Some((lo, hi)) if lo > 0.0 => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
            Some((lo, hi)) => Info::fin(0.0, (-lo).max(hi)).unwrap_or_else(Info::unknown),
            None => Info::unknown(),
        },
        Exp => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if hi <= 700.0 => {
                    Info::fin(next_down(lo.exp()), next_up(hi.exp())).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Ln => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if lo > 0.0 => {
                    Info::fin(next_down(lo.ln()), next_up(hi.ln())).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Sqrt => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if lo >= 0.0 => {
                    Info::fin(0.0, next_up(hi.sqrt())).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Sin | Cos => round_info(analyze(&n.kids[0], bounds, env), -1.0, 1.0),
        Tan => Info::unknown(),
        Sinh => match analyze(&n.kids[0], bounds, env).bounds() {
            Some((lo, hi)) if lo.abs() <= 700.0 && hi.abs() <= 700.0 => {
                Info::fin(next_down(lo.sinh()), next_up(hi.sinh())).unwrap_or_else(Info::unknown)
            }
            _ => Info::unknown(),
        },
        Cosh => match analyze(&n.kids[0], bounds, env).bounds() {
            Some((lo, hi)) if lo.abs() <= 700.0 && hi.abs() <= 700.0 => {
                let (a, b) = (lo.cosh(), hi.cosh());
                let min = if lo <= 0.0 && hi >= 0.0 { 1.0 } else { a.min(b) };
                Info::fin(next_down(min), next_up(a.max(b))).unwrap_or_else(Info::unknown)
            }
            _ => Info::unknown(),
        },
        Tanh => round_info(analyze(&n.kids[0], bounds, env), -1.0, 1.0),
        Logistic => round_info(analyze(&n.kids[0], bounds, env), 0.0, 1.0),
        // `step(NaN)` and `step(inf)` are both finite (0 or 1).
        Step => Info::fin(0.0, 1.0).unwrap(),
        Gauss => round_info(analyze(&n.kids[0], bounds, env), 0.0, 1.0),
        Erf => round_info(analyze(&n.kids[0], bounds, env), -1.0, 1.0),
        Erfc => round_info(analyze(&n.kids[0], bounds, env), 0.0, 2.0),
        Square => match analyze(&n.kids[0], bounds, env).bounds() {
            Some(v) => match mul_bounds(v, v) {
                Some((lo, hi)) => Info::fin(lo.max(0.0), hi).unwrap_or_else(Info::unknown),
                None => Info::unknown(),
            },
            None => Info::unknown(),
        },
        Cube => match analyze(&n.kids[0], bounds, env).bounds() {
            Some(v) => match mul_bounds(v, v).and_then(|(l, h)| mul_bounds((l, h), v)) {
                Some((lo, hi)) => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                None => Info::unknown(),
            },
            None => Info::unknown(),
        },
        Floor | Ceil | Round => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) => {
                    let (lo, hi) = match n.op {
                        Floor => (lo.floor(), hi.floor()),
                        Ceil => (lo.ceil(), hi.ceil()),
                        _ => (lo.round(), hi.round()),
                    };
                    Info::fin(lo, hi).unwrap_or_else(Info::unknown)
                }
                None => Info::unknown(),
            }
        }
        Sign => round_info(analyze(&n.kids[0], bounds, env), -1.0, 1.0),
        Asinh => match analyze(&n.kids[0], bounds, env).bounds() {
            Some((lo, hi)) => {
                Info::fin(next_down(lo.asinh()), next_up(hi.asinh())).unwrap_or_else(Info::unknown)
            }
            None => Info::unknown(),
        },
        Acosh => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if lo >= 1.0 => {
                    Info::fin(next_down(lo.acosh()), next_up(hi.acosh())).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Atanh => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if lo > -1.0 && hi < 1.0 => {
                    Info::fin(next_down(lo.atanh()), next_up(hi.atanh())).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Add => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some(a), Some(b)) => match add_bounds(a, b) {
                    Some((lo, hi)) => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                    None => Info::unknown(),
                },
                _ => Info::unknown(),
            }
        }
        Sub => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some(a), Some(b)) => match sub_bounds(a, b) {
                    Some((lo, hi)) => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                    None => Info::unknown(),
                },
                _ => Info::unknown(),
            }
        }
        Mul => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some(a), Some(b)) => match mul_bounds(a, b) {
                    Some((lo, hi)) => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                    None => Info::unknown(),
                },
                _ => Info::unknown(),
            }
        }
        Div => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some(a), Some(b)) => match div_bounds(a, b) {
                    Some((lo, hi)) => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                    None => Info::unknown(),
                },
                _ => Info::unknown(),
            }
        }
        Pow => {
            let b = &n.kids[1];
            // `pow(x, 0)` is 1 even for NaN and ±inf; `pow(x, 1)` is x.
            if b.op == Const && b.val.to_bits() == 0.0f64.to_bits() {
                return Info::fin(1.0, 1.0).unwrap();
            }
            if b.op == Const && b.val.to_bits() == 1.0f64.to_bits() {
                return analyze(&n.kids[0], bounds, env);
            }
            let a = analyze(&n.kids[0], bounds, env);
            if let (Some(ab), Some(n)) = (a.bounds(), b.op.eq(&Op::Const).then_some(b.val)) {
                if n.fract() == 0.0 && n.abs() <= 1024.0 {
                    if let Some((lo, hi)) = pow_int_bounds(ab, n as i32) {
                        return Info::fin(lo, hi).unwrap_or_else(Info::unknown);
                    }
                }
            }
            Info::unknown()
        }
        Mod => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some(a), Some(b)) if div_bounds(a, b).is_some() => {
                    let m = b.0.abs().max(b.1.abs());
                    Info::fin(-m, m).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Atan2 => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            if a.total() && b.total() {
                Info::fin(
                    -std::f64::consts::PI,
                    std::f64::consts::PI,
                )
                .unwrap_or_else(Info::unknown)
            } else {
                Info::unknown()
            }
        }
        Min => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some((al, ah)), Some((bl, bh))) => {
                    Info::fin(al.min(bl), ah.min(bh)).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        Max => {
            let a = analyze(&n.kids[0], bounds, env);
            let b = analyze(&n.kids[1], bounds, env);
            match (a.bounds(), b.bounds()) {
                (Some((al, ah)), Some((bl, bh))) => {
                    Info::fin(al.max(bl), ah.max(bh)).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        If => {
            let c = analyze(&n.kids[0], bounds, env);
            let a = analyze(&n.kids[1], bounds, env);
            let b = analyze(&n.kids[2], bounds, env);
            match c.bounds() {
                Some((lo, _)) if lo > 0.0 => a,
                Some((_, hi)) if hi <= 0.0 => b,
                Some(_) if a.total() && b.total() => join(a, b),
                _ => Info::unknown(),
            }
        }
        Equal | Less | LessOrEqual | Greater | GreaterOrEqual | And | Or | Xor | Not => {
            // Every comparison and logical operator propagates NaN, so all
            // operands must be finite for the result to be total.
            if n.kids.iter().all(|k| analyze(k, bounds, env).total()) {
                Info::fin(0.0, 1.0).unwrap()
            } else {
                Info::unknown()
            }
        }
        Require | Constrain => {
            let a = analyze(&n.kids[0], bounds, env);
            match a.bounds() {
                Some((lo, hi)) if lo > 0.0 => Info::fin(lo, hi).unwrap_or_else(Info::unknown),
                _ => Info::unknown(),
            }
        }
        Between => {
            let v = analyze(&n.kids[0], bounds, env);
            let lo = analyze(&n.kids[1], bounds, env);
            let hi = analyze(&n.kids[2], bounds, env);
            match (v.bounds(), lo.bounds(), hi.bounds()) {
                (Some((vl, vh)), Some((_, lh)), Some((hl, _))) if lh <= vl && vh <= hl => {
                    Info::fin(vl, vh).unwrap_or_else(Info::unknown)
                }
                _ => Info::unknown(),
            }
        }
        _ => Info::unknown(),
    }
}

/// Total and range analysis of `n`, using `bounds` for input columns.
/// `None` means an input may be non-finite.
pub fn analyze_total(n: &Node, bounds: Option<&[(f64, f64)]>) -> bool {
    analyze(n, bounds, None).total()
}

/// Canonical structural order, used to orient the commutative `Add`/`Mul`
/// children. It is a total order on nodes (op index, value bits, arity, children,
/// block definition), so the normal form is deterministic.
fn cmp_nodes(a: &Node, b: &Node) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    a.op
        .index()
        .cmp(&b.op.index())
        .then_with(|| a.val.to_bits().cmp(&b.val.to_bits()))
        .then_with(|| a.kids.len().cmp(&b.kids.len()))
        .then_with(|| {
            for (x, y) in a.kids.iter().zip(&b.kids) {
                let c = cmp_nodes(x, y);
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        })
        .then_with(|| match (&a.block, &b.block) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(x), Some(y)) => x
                .name
                .cmp(&y.name)
                .then_with(|| x.params.len().cmp(&y.params.len()))
                .then_with(|| cmp_nodes(&x.body, &y.body)),
        })
}

fn exact_val(n: &Node, bits: u64) -> bool {
    n.op == Op::Const && n.val.to_bits() == bits
}

fn zero_literal(n: &Node) -> Option<f64> {
    if n.op == Op::Const && n.val == 0.0 {
        Some(n.val)
    } else {
        None
    }
}

/// Sound, value-preserving simplification: the normal form used for scoring,
/// de-duplication and CSE.
///
/// Every rule is exact on IEEE-754 values, including `NaN`, `±inf` and signed
/// zero. Cancellations that need a proof (`x-x`, `x/x`, `0*x`) are applied only
/// when [`analyze`] proves the operand finite (and, for `x/x`, non-zero); with
/// no data bounds an input is treated as possibly non-finite, so those rules
/// stay off. Associativity, distributivity and affine re-association are
/// deliberately absent: they change f64 rounding and belong to
/// [`simplify_display`].
pub fn simplify(n: &Node) -> Node {
    simplify_with_bounds(n, None)
}

/// [`simplify`] with per-input `[lo, hi]` bounds, which enable the guarded
/// cancellation rules for subtrees that contain inputs.
pub fn simplify_with_bounds(n: &Node, bounds: Option<&[(f64, f64)]>) -> Node {
    use Op::*;
    let mut m = Node {
        op: n.op,
        val: n.val,
        kids: n
            .kids
            .iter()
            .map(|k| simplify_with_bounds(k, bounds))
            .collect(),
        block: n.block.clone(),
    };
    if m.op == Const || m.op == Input {
        return m;
    }
    if m.op.is_custom() {
        if let Some(block) = &mut m.block {
            block.body = simplify(&block.body);
        }
        return m;
    }
    if m.op.is_meta() {
        m.val = meta_value(m.op, &m.kids[0]);
    }
    if !m.op.is_agg() && m.kids.iter().all(|k| k.op == Const) {
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
            // `x + -0` is exactly `x` for every IEEE value (including -0).
            let neg_zero = (-0.0f64).to_bits();
            if exact_val(&m.kids[0], neg_zero) {
                return m.kids[1].clone();
            }
            if exact_val(&m.kids[1], neg_zero) {
                return m.kids[0].clone();
            }
            // `x + +0` is an identity unless x can be -0. A strictly signed
            // interval (or a concrete aggregate/meta value) proves that.
            for (zero, other) in [(&m.kids[0], &m.kids[1]), (&m.kids[1], &m.kids[0])] {
                if exact_val(zero, 0.0f64.to_bits()) {
                    if let Some((lo, hi)) = analyze(other, bounds, None).bounds() {
                        if lo > 0.0 || hi < 0.0 {
                            return other.clone();
                        }
                    }
                }
            }
        }
        Sub => {
            // `x - +0` is exactly `x`; `x - -0` is not (it turns -0 into +0).
            if exact_val(&m.kids[1], 0.0f64.to_bits()) {
                return m.kids[0].clone();
            }
            if same(&m.kids[0], &m.kids[1]) {
                let info = analyze(&m.kids[0], bounds, None);
                if info.total() {
                    return constant(0.0);
                }
            }
        }
        Mul => {
            if exact_val(&m.kids[0], 1.0f64.to_bits())
                || exact_val(&m.kids[1], 1.0f64.to_bits())
            {
                return if exact_val(&m.kids[0], 1.0f64.to_bits()) {
                    m.kids[1].clone()
                } else {
                    m.kids[0].clone()
                };
            }
            // `0 * x` is only a constant zero when x is finite and has a known
            // sign, because the sign of the zero result follows x.
            for (zero, other) in [(&m.kids[0], &m.kids[1]), (&m.kids[1], &m.kids[0])] {
                if let Some(z) = zero_literal(zero) {
                    let info = analyze(other, bounds, None);
                    if let Some((lo, hi)) = info.bounds() {
                        if lo > 0.0 || hi < 0.0 {
                            let negative = z.is_sign_negative() ^ (hi < 0.0);
                            return constant(if negative { -0.0 } else { 0.0 });
                        }
                    }
                }
            }
        }
        Div => {
            if exact_val(&m.kids[1], 1.0f64.to_bits()) {
                return m.kids[0].clone();
            }
            if same(&m.kids[0], &m.kids[1]) {
                let info = analyze(&m.kids[0], bounds, None);
                if info.total() && info.nonzero {
                    return constant(1.0);
                }
            }
        }
        Pow => {
            if m.kids[1].op == Const {
                if m.kids[1].val.to_bits() == 1.0f64.to_bits() {
                    return m.kids[0].clone();
                }
                if m.kids[1].val.to_bits() == 0.0f64.to_bits() {
                    return constant(1.0);
                }
            }
        }
        If => {
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
    if m.op == Add || m.op == Mul {
        m.kids.sort_by(cmp_nodes);
    }
    m
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

/// Display-only aggressive simplification: constant folding, algebraic
/// identities and affine normalisation.
///
/// Equivalent expressions evaluate identically, so collapsing them removes work
/// that buys nothing (see `docs/optimisation.md`, "the kernel of evaluation").
/// Normalising constant factors also makes the printed equation readable:
/// `x/1.1839` becomes `0.8447*x`, and `(x/c1)/c2` becomes `1.6894*x`.
///
/// **This is not value-preserving at IEEE edge cases** (it applies
/// `x + 0 -> x`, `0*x -> 0`, tolerance-based identities and affine
/// re-association). Use [`simplify`] for scoring, dedup and CSE; use this only
/// when producing a human-readable equation.
pub fn simplify_display(n: &Node) -> Node {
    use Op::*;
    let mut m = Node {
        op: n.op,
        val: n.val,
        kids: n.kids.iter().map(simplify_display).collect(),
        block: n.block.clone(),
    };
    if m.op == Const || m.op == Input {
        return m;
    }
    // A block call is kept named (that is the point of supplying it), but its
    // body is simplified in parameter space. Its constants stay shared by every
    // use; the fitter synchronises the copies (see `docs/building-blocks.md`).
    if m.op.is_custom() {
        if let Some(block) = &mut m.block {
            block.body = simplify_display(&block.body);
        }
        return m;
    }
    // Simplifying the argument can change its shape, so a meta node must take
    // a fresh structural value before the fold check below reads it.
    if m.op.is_meta() {
        m.val = meta_value(m.op, &m.kids[0]);
    }
    // Fold any operator whose children are all constants. An aggregate is
    // never folded: its name is the point (and `count()` has no children at
    // all, so the test below would otherwise be vacuously true).
    if !m.op.is_agg() && m.kids.iter().all(|k| k.op == Const) {
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
///
/// A meta operator or an aggregate is carried over verbatim: a meta value
/// describes the *searched* (normalised) tree, and an aggregate names a dataset
/// column whose cache came from the searched sample. Rewriting the inputs
/// inside either would change its meaning and make the printed value disagree
/// with the model. `to_string` still prints input names for the untouched
/// argument.
pub fn denormalize(n: &Node, means: &[f64], stds: &[f64]) -> Node {
    // A block body is in parameter space: parameter `Input(i)` is bound to an
    // argument, not to a data column, so only the calls' arguments are
    // rewritten. The definition is carried over unchanged.
    if n.op.is_custom() {
        return Node {
            op: n.op,
            val: n.val,
            kids: n.kids.iter().map(|k| denormalize(k, means, stds)).collect(),
            block: n.block.clone(),
        };
    }
    if n.op.is_meta() || n.op.is_agg() {
        return Node {
            op: n.op,
            val: n.val,
            kids: n.kids.clone(),
            block: None,
        };
    }
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
            block: n.block.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_dependent_rules() {
        let x1 = || input(0);
        let x2 = || input(1);
        let mean = || aggregate(AggKind::Mean, Some(0), 0.5);
        assert!(row_dependent(&x1()));
        assert!(!row_dependent(&node(Op::Exp, vec![constant(1.3)])));
        let m = node(Op::Add, vec![mean(), constant(0.2)]);
        assert!(!row_dependent(&m));
        let m = node(Op::Mul, vec![mean(), x2()]);
        assert!(row_dependent(&m));
        assert!(!row_dependent(&meta(Op::Complexity, x1())));
    }

    /// Bit-exact value equality except that all NaNs compare equal.
    fn same_value(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    /// Evaluate a meta operator on `child`; `row` must not matter.
    fn meta_val(kind: Op, child: Node, row: &[f64]) -> f64 {
        eval(&meta(kind, child), row)
    }

    /// Assert all five operators against one known shape, on two very different
    /// rows (the value is structural, so both must agree at every operator).
    fn check_shape(child: Node, want: (f64, f64, f64, f64, f64)) {
        let rows = [[0.0, 0.0], [3.5, -7.25]];
        for row in rows {
            assert_eq!(meta_val(Op::Complexity, child.clone(), &row), want.0);
            assert_eq!(meta_val(Op::Variables, child.clone(), &row), want.1);
            assert_eq!(meta_val(Op::Constants, child.clone(), &row), want.2);
            assert_eq!(meta_val(Op::Contains, child.clone(), &row), want.3);
            assert_eq!(meta_val(Op::ContainsOp, child.clone(), &row), want.4);
        }
    }

    #[test]
    fn meta_operators_describe_the_shape_of_hand_built_trees() {
        // (complexity, variables, constants, contains, contains_op)

        // A bare input.
        check_shape(input(0), (1.0, 1.0, 0.0, 1.0, 0.0));
        // A bare constant.
        check_shape(constant(7.5), (1.0, 0.0, 1.0, 0.0, 0.0));

        // x1 * x1: the same variable twice still counts once.
        let repeated = node(Op::Mul, vec![input(0), input(0)]);
        check_shape(repeated, (3.0, 1.0, 0.0, 1.0, 1.0));

        // x1 * x2 + 3.5
        let mixed = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![input(0), input(1)]),
                constant(3.5),
            ],
        );
        check_shape(mixed, (5.0, 2.0, 1.0, 1.0, 1.0));

        // sin(x1) ^ 2
        let nested = node(
            Op::Pow,
            vec![node(Op::Sin, vec![input(0)]), constant(2.0)],
        );
        check_shape(nested, (4.0, 1.0, 1.0, 1.0, 1.0));

        // A meta operator nested in another meta operator is counted like any
        // other operator node: complexity(complexity(x1*x2)) = 5 nodes (both
        // meta nodes, the product and its two inputs).
        let inner = meta(Op::Complexity, node(Op::Mul, vec![input(0), input(1)]));
        let nested_meta = meta(Op::Complexity, inner);
        check_shape(nested_meta, (5.0, 2.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn meta_values_are_cached_at_construction() {
        for kind in [
            Op::Complexity,
            Op::Variables,
            Op::Constants,
            Op::Contains,
            Op::ContainsOp,
        ] {
            let n = meta(kind, node(Op::Mul, vec![input(0), input(1)]));
            assert_eq!(n.val, eval(&n, &[1.0, 2.0]), "{kind:?}");
            assert!(n.val.is_finite());
        }

        // `eval` must return the cache without walking the child. Corrupting
        // the child's shape behind the constructor's back would change a fresh
        // walk (3 nodes); the cached value must stay at 1.
        let mut n = meta(Op::Complexity, input(0));
        n.kids[0] = node(Op::Add, vec![input(0), input(1)]);
        assert_eq!(eval(&n, &[]), 1.0);
    }

    #[test]
    fn simplify_refreshes_a_meta_value_after_the_argument_shrinks() {
        let tree = meta(
            Op::Complexity,
            node(Op::Add, vec![input(0), constant(0.0)]),
        );
        assert_eq!(tree.val, 3.0);
        let simple = simplify_display(&tree);
        assert_eq!(simple.op, Op::Complexity);
        // `x + 0` becomes `x` in the display normal form, so the argument is
        // one node.
        assert_eq!(simple.val, 1.0);
        assert_eq!(eval(&simple, &[4.0]), 1.0);
        // The sound core deliberately keeps `x + +0` (it is not an identity at
        // x = -0), so the cached shape stays at 3 nodes there.
        assert_eq!(simplify(&tree).val, 3.0);
    }

    #[test]
    fn constants_operator_counts_leaves_in_the_argument_only() {
        // constants(3) is 1, not 2: the meta node itself is not a constant.
        let n = meta(Op::Constants, constant(3.0));
        assert_eq!(n.val, 1.0);
        assert_eq!(count_constants(&n), 1);
        assert_eq!(count_constants(&meta(Op::Complexity, input(0))), 0);
    }

    #[test]
    fn denormalize_leaves_a_meta_argument_alone() {
        let tree = meta(Op::Complexity, node(Op::Mul, vec![input(0), constant(2.0)]));
        let den = denormalize(&tree, &[10.0], &[2.0]);
        assert_eq!(den.op, Op::Complexity);
        assert_eq!(den.val, 3.0, "the searched structural value is preserved");
        assert_eq!(den.kids[0].op, Op::Mul);
        assert_eq!(den.kids[0].kids[0].op, Op::Input);
    }

    #[test]
    fn meta_operators_are_unary_opt_in_and_named() {
        for op in [
            Op::Complexity,
            Op::Variables,
            Op::Constants,
            Op::Contains,
            Op::ContainsOp,
        ] {
            assert_eq!(op.arity(), 1, "{op:?}");
            assert!(op.is_meta(), "{op:?}");
            assert!(!Op::default_set().contains(&op), "{op:?} must be opt-in");
            assert!(Op::names().contains(&op.symbol()), "{op:?}");
            assert_eq!(Op::from_name(op.symbol()), Some(op));
        }
        // The five are the only meta operators.
        let count = Op::names()
            .iter()
            .filter(|n| Op::from_name(n).unwrap().is_meta())
            .count();
        assert_eq!(count, 5);
        assert_eq!(
            Op::parse_list_checked("complexity,variables"),
            Ok(vec![Op::Complexity, Op::Variables])
        );
        assert!(Op::parse_list_checked("complexity,nope").is_err());
    }

    /// One column, five rows: mean/median = 3, population std = sqrt(2),
    /// q1 = 2, q3 = 4, sum = 15, min = 1, max = 5, count = 5.
    fn hand_table() -> AggTable {
        AggTable::from_rows(&[
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ])
    }

    #[test]
    fn aggregate_values_are_exact_and_row_independent() {
        let t = hand_table();
        let cases = [
            (AggKind::Mean, 3.0),
            (AggKind::Std, 2.0f64.sqrt()),
            (AggKind::Median, 3.0),
            (AggKind::Iqr, 2.0),
            (AggKind::Sum, 15.0),
            (AggKind::Min, 1.0),
            (AggKind::Max, 5.0),
        ];
        for (kind, want) in cases {
            let n = aggregate(kind, Some(0), t.get(kind, 0));
            assert!((eval(&n, &[1.0]) - want).abs() < 1e-12, "{kind:?}");
            assert!((eval(&n, &[999.0]) - want).abs() < 1e-12, "{kind:?}");
            assert_eq!(n.kids.len(), 1, "{kind:?} has one column child");
            assert_eq!(n.kids[0].op, Op::Input);
        }
        let c = aggregate(AggKind::Count, None, t.get(AggKind::Count, 0));
        assert!(c.kids.is_empty(), "count() is zero-arity");
        assert_eq!(eval(&c, &[]), 5.0);
        assert_eq!(eval(&c, &[1.0, 2.0]), 5.0);
    }

    #[test]
    fn aggregate_eval_returns_the_cache_without_walking_the_child() {
        let mut n = aggregate(AggKind::Mean, Some(0), 4.25);
        // Corrupting the child behind the constructor's back must not change
        // the value: `eval` may not touch the column or the row.
        n.kids[0] = node(Op::Add, vec![input(0), input(1)]);
        assert_eq!(eval(&n, &[0.0, 0.0]), 4.25);
        assert_eq!(eval(&n, &[1e30, -1e30]), 4.25);
    }

    #[test]
    fn aggregates_are_opt_in_unary_and_named() {
        for kind in AggKind::ALL {
            let op = Op::Agg(kind);
            assert_eq!(op.arity(), kind.arity(), "{kind:?}");
            assert!(op.is_agg() && !op.is_meta(), "{kind:?}");
            assert!(!Op::default_set().contains(&op), "{kind:?} must be opt-in");
            assert!(Op::names().contains(&op.symbol()), "{kind:?}");
            assert_eq!(Op::from_name(op.symbol()), Some(op));
        }
        assert_eq!(Op::Agg(AggKind::Std).arity(), 1);
        assert_eq!(Op::Agg(AggKind::Count).arity(), 0);
        // `min`/`max` stay binary in the shared name table; the column
        // reductions have their own canonical names.
        assert_eq!(Op::from_name("min"), Some(Op::Min));
        assert_eq!(Op::from_name("amax"), Some(Op::Agg(AggKind::Max)));
        assert_eq!(
            Op::parse_list_checked("mean,count"),
            Ok(vec![Op::Agg(AggKind::Mean), Op::Agg(AggKind::Count)])
        );
        assert!(Op::parse_list_checked("mean,nope").is_err());
        // Dense indices are unique and in `0..COUNT`.
        let mut seen = std::collections::HashSet::new();
        for kind in AggKind::ALL {
            let i = Op::Agg(kind).index();
            assert!(i < Op::COUNT);
            assert!(seen.insert(i), "duplicate operator index {i}");
        }
    }

    #[test]
    fn simplify_preserves_an_aggregate_cache_and_binds_correctly() {
        let t = hand_table();
        let mean = aggregate(AggKind::Mean, Some(0), t.get(AggKind::Mean, 0));
        let expr = node(Op::Add, vec![mean.clone(), constant(0.0)]);
        let simple = simplify(&expr);
        assert_eq!(simple.op, Op::Agg(AggKind::Mean));
        assert_eq!(simple.val, 3.0, "the cached value survives simplification");
        assert_eq!(eval(&simple, &[123.0]), 3.0);

        // `count()` has no children; simplification must not fold it into a
        // bare constant (which would lose the name and the zero-arity shape).
        let c = aggregate(AggKind::Count, None, 5.0);
        let simple = simplify(&c);
        assert_eq!(simple.op, Op::Agg(AggKind::Count));
        assert!(simple.kids.is_empty());
        assert_eq!(simple.val, 5.0);

        // Binding resolves a parser-built node against the table.
        let mut parsed = node(Op::Agg(AggKind::Iqr), vec![input(0)]);
        assert!(parsed.val.is_nan());
        bind_aggregates(&mut parsed, &t).unwrap();
        assert_eq!(parsed.val, 2.0);
    }

    #[test]
    fn bind_rejects_expressions_and_unknown_columns() {
        let t = hand_table();
        let mut not_a_column = node(
            Op::Agg(AggKind::Mean),
            vec![node(Op::Add, vec![input(0), constant(1.0)])],
        );
        assert!(bind_aggregates(&mut not_a_column, &t).is_err());
        let mut out_of_range = node(Op::Agg(AggKind::Mean), vec![input(3)]);
        assert!(bind_aggregates(&mut out_of_range, &t).is_err());
    }

    #[test]
    fn sound_core_keeps_unsafe_cancellations() {
        let x = input(0);
        let div_zero = node(Op::Div, vec![constant(1.0), constant(0.0)]);
        let cases = [
            node(Op::Sub, vec![x.clone(), x.clone()]),
            node(Op::Div, vec![x.clone(), x.clone()]),
            node(Op::Mul, vec![constant(0.0), x.clone()]),
            node(Op::Add, vec![x.clone(), constant(0.0)]),
            node(Op::Sub, vec![div_zero.clone(), div_zero.clone()]),
        ];
        for t in cases {
            let s = simplify(&t);
            // None of these may become a finite constant without a proof.
            assert!(
                !(s.op == Op::Const && s.val.is_finite()),
                "unsafe rewrite of {}",
                to_string(&t, &["x".to_string()])
            );
            for row in [[-0.0], [0.0], [1.0], [f64::INFINITY], [f64::NAN]] {
                assert!(
                    same_value(eval(&t, &row), eval(&s, &row)),
                    "{} at {:?}: {} -> {}",
                    to_string(&t, &["x".to_string()]),
                    row,
                    eval(&t, &row),
                    eval(&s, &row)
                );
            }
        }
        // `x - x` stays NaN at a non-finite input.
        let t = node(Op::Sub, vec![x.clone(), x.clone()]);
        assert!(eval(&simplify(&t), &[f64::INFINITY]).is_nan());
        // The display relation is allowed to make the aggressive rewrite.
        assert_eq!(simplify_display(&t).op, Op::Const);
    }

    #[test]
    fn sound_core_uses_proofs_to_enable_safe_cancellations() {
        let x = input(0);
        let bounds = [(-2.0, 2.0)];
        let exp_x = node(Op::Exp, vec![x.clone()]);
        // exp is finite and strictly positive on the bounded column.
        assert!(analyze_total(&exp_x, Some(&bounds)));
        assert_eq!(simplify_with_bounds(&node(Op::Sub, vec![exp_x.clone(), exp_x.clone()]), Some(&bounds)).op, Op::Const);
        assert_eq!(simplify_with_bounds(&node(Op::Div, vec![exp_x.clone(), exp_x.clone()]), Some(&bounds)).op, Op::Const);
        // 0*exp(x) is +0 because exp(x) > 0.
        let z = simplify_with_bounds(&node(Op::Mul, vec![constant(0.0), exp_x]), Some(&bounds));
        assert_eq!(z.op, Op::Const);
        assert_eq!(z.val.to_bits(), 0.0f64.to_bits());
        // Without bounds the same rewrites are withheld.
        let t = node(Op::Sub, vec![input(0), input(0)]);
        assert_eq!(simplify(&t).op, Op::Sub);
    }

    #[test]
    fn sound_core_is_signed_zero_exact() {
        let x = input(0);
        // x + -0 is an identity at every input, including -0.
        let t = node(Op::Add, vec![x.clone(), constant(-0.0)]);
        assert_eq!(simplify(&t).op, Op::Input);
        // x - +0 is an identity; x - -0 is not.
        assert_eq!(simplify(&node(Op::Sub, vec![x.clone(), constant(0.0)])).op, Op::Input);
        assert_ne!(simplify(&node(Op::Sub, vec![x.clone(), constant(-0.0)])).op, Op::Input);
        // x * 1 and x / 1 are exact for NaN, inf and both zeros.
        for t in [
            node(Op::Mul, vec![x.clone(), constant(1.0)]),
            node(Op::Div, vec![x.clone(), constant(1.0)]),
        ] {
            assert_eq!(simplify(&t).op, Op::Input);
        }
        // A constant one ulp away is not an identity.
        let almost_one = 1.0f64 + f64::EPSILON;
        assert_ne!(
            simplify(&node(Op::Mul, vec![x.clone(), constant(almost_one)])).op,
            Op::Input
        );
    }

    #[test]
    fn sound_core_canonicalises_add_and_mul_and_is_idempotent() {
        let x = input(0);
        let y = input(1);
        let z = node(Op::Sin, vec![input(0)]);
        for (a, b) in [
            (node(Op::Add, vec![x.clone(), y.clone()]), node(Op::Add, vec![y.clone(), x.clone()])),
            (node(Op::Mul, vec![x.clone(), y.clone()]), node(Op::Mul, vec![y.clone(), x.clone()])),
            (
                node(Op::Add, vec![x.clone(), node(Op::Add, vec![y.clone(), z.clone()])]),
                node(Op::Add, vec![node(Op::Add, vec![z.clone(), y.clone()]), x.clone()]),
            ),
        ] {
            let sa = simplify(&a);
            let sb = simplify(&b);
            assert_eq!(structural_hash(&sa), structural_hash(&sb));
            assert!(same(&sa, &sb));
            assert_eq!(structural_hash(&simplify(&sa)), structural_hash(&sa));
            assert_eq!(structural_hash(&simplify(&sb)), structural_hash(&sb));
        }
    }

    /// Each fixture is mutated with many seeds; a mutation that returns the
    /// parent unchanged (a wasted fit and score) is a failure. The table mixes
    /// hand-built univariate trees with generated univariate and bivariate
    /// ones, so both the terminal-edit fall-through and the ordinary paths are
    /// covered.
    #[test]
    fn mutation_always_changes_generated_trees() {
        let ops = Op::default_set();
        let mut fixtures: Vec<(usize, Node)> = vec![
            (
                1,
                node(
                    Op::Add,
                    vec![
                        node(Op::Sin, vec![input(0)]),
                        node(Op::Cos, vec![input(0)]),
                    ],
                ),
            ),
            (1, input(0)),
            (1, constant(0.5)),
        ];
        for tseed in 0..6u32 {
            for &nvars in &[1usize, 2] {
                let mut rng = crate::rng::Mt::new(0x7700 ^ (tseed << 4) ^ nvars as u32);
                fixtures.push((nvars, random_tree_with(&mut rng, nvars, &ops, 3, None)));
            }
        }
        for (k, (nvars, tree)) in fixtures.iter().enumerate() {
            for mseed in 0..128u32 {
                let mut rng = crate::rng::Mt::new(mseed.wrapping_mul(0x9E37_79B9) ^ k as u32);
                let mut t = tree.clone();
                let before = structural_hash(&t);
                mutate_with(&mut t, &mut rng, *nvars, &ops, None);
                assert_ne!(
                    structural_hash(&t),
                    before,
                    "fixture {k} (nvars {nvars}), mutation seed {mseed}: no change"
                );
            }
        }
    }
}
