//! #1182: `rcsr --gate-search`, the search of a regime gate between two frozen
//! branches.
//!
//! The genome is the gate `g`. The branches come from finished runs (their
//! finalist with the run's threshold folded in), from expressions, or are cash;
//! their training positions are computed once and never change. A candidate
//! gate is scored at its best cut of the 16-quantile grid of its training
//! values: for every inner grid cut `c` the state of a row is the hysteresis
//! and dwell kernel of `--pnl-mask` over `g - c` (`rcsr-gate-state/1` on the
//! training rows: per group and training segment), the composite holds the on
//! branch in state on and the off branch elsewhere, and the fitness is the
//! `--fitness pnl` statistic of that one book. A gate with fewer than two
//! distinct training values is constant: it scores the better branch alone.
//!
//! After the search every front member becomes a searched gate `(g) - (c)` of
//! one composite, and the choice, the pricing and the run.json are those of
//! `rcsr compose` (#1181): the robust rule with the per-state alpha condition
//! on the validation rows, the gate states of `rcsr-gate-state/1`, and the
//! `switch` object that walk-forward, the ladder, the seal and `--reselect-run`
//! price.

use crate::compose::{self, Member, Pool, PoolRun};
use crate::data::{TradingSlice, TradingSplit};
use crate::devlog::{num, text, Json};
use crate::expr::{self, Node, Op};
use crate::occam::{self, CodeSpec};
use crate::regime_gate::{self, GateSpec};
use crate::trading::{BlockStat, FitnessBook, PnlExcess, PositionLayer, TradingSpec};
use std::sync::{Arc, OnceLock};

/// Quantiles of the cut grid (the 4-bit cut of the composite code).
pub const QUANTILES: usize = 16;
pub const SCHEMA: &str = "rcsr-gate-search/1";
/// The cash branch: below the composite threshold 0 on every row.
const CASH: &str = "-1.0";

/// Where a branch comes from.
#[derive(Clone, Debug)]
pub enum Source {
    Cash,
    Expr,
    /// A run.json: its finalist, with the run checked against this invocation.
    Run(Box<PoolRun>),
}

/// A frozen branch: long where `text` is above 0.
#[derive(Clone, Debug)]
pub struct Branch {
    pub source: Source,
    /// The finalist or the expression as given; empty for cash.
    pub expression_exact: String,
    /// `expression_exact` with the run's threshold subtracted (`(f) - (t)`), or `-1.0` for cash.
    pub text: String,
    pub columns: Vec<String>,
    pub archive_rank: usize,
}

impl Branch {
    pub fn is_cash(&self) -> bool {
        matches!(self.source, Source::Cash)
    }

    fn json(&self) -> String {
        match &self.source {
            Source::Cash => format!("{{\"kind\":\"cash\",\"branch\":{}}}", text(&self.text)),
            Source::Expr => format!("{{\"kind\":\"expr\",\"expression_exact\":{},\"branch\":{}}}", text(&self.expression_exact), text(&self.text)),
            Source::Run(r) => format!(
                "{{\"kind\":\"run\",\"path\":{},\"sha256\":{},\"archive_rank\":{},\"expression_exact\":{},\"threshold\":{},\"branch\":{}}}",
                text(&r.path),
                text(&r.sha256),
                self.archive_rank,
                text(&self.expression_exact),
                num(r.threshold),
                text(&self.text)
            ),
        }
    }
}

/// The flags of `--gate-search` that the search parser does not read.
#[derive(Clone, Debug)]
pub struct Request {
    pub on: Branch,
    pub off: Branch,
    pub band: f64,
    pub dwell: usize,
    pub json: bool,
    pub out: Option<String>,
}

static REQUEST: OnceLock<Request> = OnceLock::new();
static LAYER: OnceLock<Arc<GateLayer>> = OnceLock::new();

/// The `--gate-search` request of this process.
pub fn request() -> Option<&'static Request> {
    REQUEST.get()
}

impl Request {
    /// The columns each flag reads, for the `--pnl-mask` column checks.
    pub fn column_sets<'a>(&'a self, vars: &'a [String]) -> Vec<(&'static str, &'a [String])> {
        vec![("--vars", vars), ("--branch-on", self.on.columns.as_slice()), ("--branch-off", self.off.columns.as_slice())]
    }

    /// The columns whose raw values travel with the rows: the gate variables, then the branch columns.
    pub fn columns(&self, vars: &[String]) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, cols) in self.column_sets(vars) {
            for c in cols {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
        }
        out
    }

    fn form(&self) -> &'static str {
        if self.on.is_cash() || self.off.is_cash() {
            "C1"
        } else {
            "C2"
        }
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: rcsr --gate-search: {}", msg);
    std::process::exit(2);
}

fn exit_with((code, msg): (i32, String)) -> ! {
    eprintln!("error: rcsr --gate-search: {}", msg);
    std::process::exit(code);
}

const OWN_FLAGS: [&str; 4] = ["--branch-on", "--branch-off", "--gate-band", "--gate-dwell"];

/// Takes the `--gate-search` flags out of `args`, loads the branches, keeps the
/// request for [`request`], and returns the search flags with `--fitness pnl`,
/// `--select-rule robust` and `--cpu` added when absent. Without
/// `--gate-search` the arguments come back unchanged; a gate-search flag
/// without it exits 2. Usage errors and mismatched runs exit 2, an unreadable
/// run exits 3.
pub fn claim(args: Vec<String>) -> Vec<String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return args;
    }
    if !args.iter().any(|a| a == "--gate-search") {
        if let Some(flag) = args.iter().find(|a| OWN_FLAGS.contains(&a.as_str())) {
            fail(&format!("{} needs --gate-search", flag));
        }
        return args;
    }
    if compose::request().is_some() {
        fail("--gate-search with rcsr compose or --reselect-run of a composed run: the gates come from the run");
    }
    let (mut on, mut off, mut band, mut dwell, mut json, mut out) = (None, None, None, None, false, None);
    let mut rest: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> String { args.get(i + 1).cloned().unwrap_or_else(|| fail(&format!("{} requires a value", flag))) };
        match flag {
            "--gate-search" => {
                i += 1;
                continue;
            }
            "--json" => {
                json = true;
                rest.push(flag.to_string());
                i += 1;
                continue;
            }
            "--branch-on" => on = Some(value()),
            "--branch-off" => off = Some(value()),
            "--gate-band" => band = Some(value()),
            "--gate-dwell" => dwell = Some(value()),
            "--out" => out = Some(value()),
            "--fitness" => {
                if value() != "pnl" {
                    fail("the gate is scored by the --fitness pnl book of the composite: --fitness must be pnl");
                }
                rest.push(flag.to_string());
                rest.push("pnl".to_string());
            }
            "--select-rule" => {
                if value().trim() != "robust" {
                    fail("the choice is the robust rule with a regime condition: --select-rule must be robust");
                }
                rest.push(flag.to_string());
                rest.push("robust".to_string());
            }
            "--gpu" => fail("--gpu: the gate search runs on the CPU; the device book carries no gate layer"),
            "--pnl-mask" | "--pnl-mask-band" | "--pnl-mask-dwell" => fail(&format!("{}: the composite prices the unmasked branches under the searched gate", flag)),
            "--pnl-excess" => fail("--pnl-excess: the composite is priced on the absolute long/flat book"),
            "--threshold" | "--position" | "--book-target" | "--band" | "--band-units" | "--min-holding" | "--max-position" | "--cash-return" | "--funding" | "--allow-short" => fail(&format!(
                "{}: the composite is the unit long/flat book at threshold 0; each branch carries its run's threshold",
                flag
            )),
            "--save" | "--save-every" | "--resume" | "--interim-chunk" | "--reselect" | "--reselect-run" => {
                fail(&format!("{}: the gate search runs to its end in one process and has no checkpoint", flag))
            }
            "--eval-expr" | "--function" | "--implicit" | "--implicit-derivative" => fail(&format!("{}: the genome is one causal gate formula", flag)),
            "--format" | "--ndjson" | "--json-out" | "--report" | "--report-out" => {
                fail(&format!("{}: the gate search prints the composed run.json (--json) or its summary", flag))
            }
            _ => {
                rest.push(args[i].clone());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let band = match band {
        None => 0.0,
        Some(t) => match t.trim().parse::<f64>() {
            Ok(b) if b.is_finite() && b >= 0.0 => b,
            _ => fail(&format!("--gate-band must be a finite number >= 0, got '{}'", t)),
        },
    };
    let dwell = match dwell {
        None => 1,
        Some(t) => match t.trim().parse::<usize>() {
            Ok(d) if d >= 1 => d,
            _ => fail(&format!("--gate-dwell must be an integer >= 1, got '{}'", t)),
        },
    };
    let on = load_branch("--branch-on", Pool::Bull, &on.unwrap_or_else(|| fail("--branch-on <run.json|expr|cash> is required"))).unwrap_or_else(|e| exit_with(e));
    let off = load_branch("--branch-off", Pool::Bear, &off.unwrap_or_else(|| fail("--branch-off <run.json|expr|cash> is required"))).unwrap_or_else(|e| exit_with(e));
    if on.text == off.text {
        fail(&format!("--branch-on and --branch-off are the same branch ('{}'): nothing to switch", on.text));
    }
    if let (Source::Run(a), Source::Run(b)) = (&on.source, &off.source) {
        if let Some(what) = differs(a, b.target.as_deref(), Some(&b.target_sha), b.per_side_bps, &b.split) {
            fail(&format!("--branch-on '{}' and --branch-off '{}' differ in {}", a.path, b.path, what));
        }
    }
    if !rest.iter().any(|a| a == "--fitness") {
        rest.extend(["--fitness".to_string(), "pnl".to_string()]);
    }
    if !rest.iter().any(|a| a == "--select-rule") {
        rest.extend(["--select-rule".to_string(), "robust".to_string()]);
    }
    if !rest.iter().any(|a| a == "--cpu" || a == "--no-gpu") {
        rest.push("--cpu".to_string());
    }
    let _ = REQUEST.set(Request { on, off, band, dwell, json, out });
    rest
}

/// What a branch run does not share with the target, target spec, cost and
/// split of another run or of this invocation; `None` when it shares all.
fn differs(run: &PoolRun, target: Option<&str>, target_sha: Option<&str>, per_side_bps: f64, split: &Json) -> Option<String> {
    if run.target.as_deref() != target {
        return Some(format!("target ({:?} vs {:?})", run.target, target));
    }
    if Some(run.target_sha.as_str()) != target_sha {
        return Some("target spec".to_string());
    }
    if run.per_side_bps != per_side_bps {
        return Some(format!("cost ({} vs {} bp per side)", run.per_side_bps, per_side_bps));
    }
    compose::SPLIT_KEYS.iter().find(|k| compose::split_differs(&run.split, split, k)).map(|k| format!("split.{}", k))
}

/// A branch from `cash`, a run.json (a file of that name) or an expression.
fn load_branch(flag: &str, pool: Pool, value: &str) -> Result<Branch, (i32, String)> {
    let bad = |e: String| (2, format!("{} '{}': {}", flag, value, e));
    if value.trim() == "cash" {
        return Ok(Branch { source: Source::Cash, expression_exact: String::new(), text: CASH.to_string(), columns: Vec::new(), archive_rank: 0 });
    }
    if !std::path::Path::new(value).is_file() {
        let columns = regime_gate::referenced_columns(value);
        expr::parse(value, &columns).map_err(|e| bad(format!("neither a run.json file nor an expression: {}", e)))?;
        return Ok(Branch { source: Source::Expr, expression_exact: value.to_string(), text: value.to_string(), columns, archive_rank: 0 });
    }
    let body = std::fs::read_to_string(value).map_err(|e| (3, format!("{} '{}': {}", flag, value, e)))?;
    let run = crate::run_json::RunFile::parse(&body).map_err(|e| if e.contains(crate::run_json::STALE_COMPOSE_REFUSAL) { (3, format!("{} '{}': {}", flag, value, e)) } else { bad(e) })?;
    let book = run.band_book().map_err(bad)?;
    if book.switch.is_some() || run.front.iter().any(|m| m.switch.is_some()) {
        return Err(bad("the run is a composed run: a composite cannot be a branch, nested switches are not supported".to_string()));
    }
    if book.position != crate::trading::PositionMap::Threshold {
        return Err(bad("the run priced the top_tercile book; a branch is a threshold book".to_string()));
    }
    if book.has_band() {
        return Err(bad("the run priced a band or a minimum hold; a branch is a plain threshold book".to_string()));
    }
    if !book.threshold.is_finite() {
        return Err(bad(format!("objective.spec.threshold {} is not finite", book.threshold)));
    }
    if book.expression_source != crate::run_json::ExpressionSource::Exact {
        return Err(bad("the finalist has no expression_exact (the rounded display string would price another formula)".to_string()));
    }
    let e = book.expression.clone();
    let columns = regime_gate::referenced_columns(&e);
    expr::parse(&e, &columns).map_err(|err| bad(format!("cannot parse the finalist '{}': {}", e, err)))?;
    let doc = Json::parse(body.trim()).ok_or_else(|| bad("not a run.json document".to_string()))?;
    let archive_rank = doc.get("objective").and_then(|o| o.get("finalist")).and_then(|f| f.get("archive_rank")).and_then(Json::as_usize).unwrap_or(1);
    let text = if book.threshold == 0.0 { e.clone() } else { format!("({}) - ({:?})", e, book.threshold) };
    let (pnl_excess, stat) = compose::run_fitness(&doc).map_err(bad)?;
    let pool_run = PoolRun {
        pool,
        path: value.to_string(),
        sha256: crate::manifest::sha256_hex(body.as_bytes()),
        target: doc.get("target").and_then(Json::as_str).map(str::to_string),
        target_sha: book.spec.sha256(),
        threshold: book.threshold,
        per_side_bps: book.per_side_bps,
        mask: book.mask.clone(),
        split: doc.get("split").cloned().unwrap_or(Json::Null),
        candidates_evaluated: run.candidates_evaluated,
        front_size: run.front.len(),
        pnl_excess,
        stat,
    };
    Ok(Branch { source: Source::Run(Box::new(pool_run)), expression_exact: e, text, columns, archive_rank })
}

// ---------------------------------------------------------------- the fitness

/// The cut a gate is scored at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cut {
    /// Inner cut of rank `t` of the sorted training values at value `value`:
    /// the rows of value above it are on.
    Grid { t: usize, value: f64 },
    /// Fewer than two distinct training values: the better branch alone.
    Constant { on: bool },
}

/// The best cut of one gate on the training rows.
#[derive(Clone, Debug)]
pub struct Choice {
    pub cut: Cut,
    pub fitness: f64,
    pub states: Vec<bool>,
}

/// #1182: the composite of two frozen branches under a gate, as a position
/// layer of the training book: the prediction is the gate value.
#[derive(Debug)]
pub struct GateLayer {
    /// The plain training book, without a layer.
    inner: FitnessBook,
    on: Vec<bool>,
    off: Vec<bool>,
    /// State group of every row: its group and training segment, numbered from 0.
    group: Vec<u32>,
    groups: usize,
    band: f64,
    dwell: usize,
    identity: String,
}

impl GateLayer {
    /// `on` and `off` are the long rows of the two branches on the book's rows;
    /// `state_group` numbers each row's group and training segment from 0.
    pub fn new(inner: FitnessBook, on: Vec<bool>, off: Vec<bool>, state_group: Vec<u32>, band: f64, dwell: usize, identity: &str) -> Result<GateLayer, String> {
        let n = inner.len();
        if on.len() != n || off.len() != n || state_group.len() != n {
            return Err(format!("gate layer: {} on, {} off and {} group entries for {} rows", on.len(), off.len(), state_group.len(), n));
        }
        if inner.has_layer() || inner.is_masked() || !inner.is_plain() {
            return Err("gate layer: the training book must be plain, unmasked and without a layer".to_string());
        }
        let groups = state_group.iter().map(|&g| g as usize + 1).max().unwrap_or(0);
        let identity = format!("gate|{}|band={:016x}|dwell={}|q={}", identity, band.to_bits(), dwell, QUANTILES);
        Ok(GateLayer { inner, on, off, group: state_group, groups, band, dwell, identity })
    }

    pub fn inner(&self) -> &FitnessBook {
        &self.inner
    }

    fn composite_into(&self, states: &[bool], pred: &mut [f64]) {
        for (i, p) in pred.iter_mut().enumerate() {
            *p = if if states[i] { self.on[i] } else { self.off[i] } { f64::MAX } else { f64::MIN };
        }
    }

    /// The composite positions of a state vector (`f64::MAX` long, `f64::MIN` flat).
    pub fn composite(&self, states: &[bool]) -> Vec<f64> {
        let mut pred = vec![0.0f64; states.len()];
        self.composite_into(states, &mut pred);
        pred
    }

    /// The states of the cut at value `c`: the kernel over `g - c`.
    pub fn states_at(&self, g: &[f64], c: f64) -> Vec<bool> {
        let mut st = vec![false; g.len()];
        regime_gate::hysteresis_states_dense(|i| g[i] - c, &self.group, self.groups, self.band, self.dwell, &mut st);
        st
    }

    /// The best cut of gate values `g` on the training rows: the lowest
    /// fitness over the inner cuts of the 16-quantile grid (ties to the lower
    /// cut), or the better branch alone for a constant gate (ties to on).
    /// `None` when a value is not finite.
    pub fn choose(&self, g: &[f64]) -> Option<Choice> {
        let n = self.inner.len();
        assert_eq!(g.len(), n, "gate layer: {} gate values for {} rows", g.len(), n);
        if g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut pred = vec![0.0f64; n];
        let mut st = vec![false; n];
        let mut best: Option<Choice> = None;
        for (t, c) in grid_cuts(g, QUANTILES) {
            regime_gate::hysteresis_states_dense(|i| g[i] - c, &self.group, self.groups, self.band, self.dwell, &mut st);
            self.composite_into(&st, &mut pred);
            let f = self.inner.regret(&pred);
            if best.as_ref().map_or(true, |b| f < b.fitness) {
                best = Some(Choice { cut: Cut::Grid { t, value: c }, fitness: f, states: st.clone() });
            }
        }
        if best.is_some() {
            return best;
        }
        let side = |on: bool| -> (f64, Vec<bool>) {
            let states = vec![on; n];
            (self.inner.regret(&self.composite(&states)), states)
        };
        let (fa, sa) = side(true);
        let (fb, sb) = side(false);
        Some(if fa <= fb { Choice { cut: Cut::Constant { on: true }, fitness: fa, states: sa } } else { Choice { cut: Cut::Constant { on: false }, fitness: fb, states: sb } })
    }
}

impl PositionLayer for GateLayer {
    fn positions(&self, pred: &[f64]) -> Vec<f64> {
        match self.choose(pred) {
            Some(c) => self.composite(&c.states),
            None => vec![f64::NAN; pred.len()],
        }
    }

    fn identity(&self) -> String {
        self.identity.clone()
    }
}

/// The inner cuts of the `q`-quantile grid of finite values `g`: rank
/// `floor(j m / q)` for `j = 1..q`, moved up to the next rank that does not
/// split equal values (`-0.0` equals `0.0`), the ends and repeats dropped
/// (the grid of `switch::GateSweep::grid`). The value of rank `t` is the
/// midpoint of its two neighbouring values, or the lower one when the midpoint
/// is not strictly below the upper one (`switch::GateSweep::value`).
pub fn grid_cuts(g: &[f64], q: usize) -> Vec<(usize, f64)> {
    let mut sorted = g.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite gate values"));
    let m = sorted.len();
    let valid = |t: usize| t == 0 || t == m || sorted[t - 1] != sorted[t];
    let mut out: Vec<(usize, f64)> = Vec::new();
    for j in 1..q.max(1) {
        let t = (j * m / q).min(m);
        let t = (t..m).find(|&k| valid(k)).unwrap_or(m);
        if t > 0 && t < m && out.last().map(|l| l.0) != Some(t) {
            let (lo, hi) = (sorted[t - 1], sorted[t]);
            let mid = lo + (hi - lo) * 0.5;
            out.push((t, if mid < hi { mid } else { lo }));
        }
    }
    out
}

/// The long rows of a branch on a slice: its value is finite and above 0.
fn long_rows(branch: &Branch, columns: &[String], values: &[Vec<f64>], n: usize) -> Result<Vec<bool>, String> {
    if branch.is_cash() {
        return Ok(vec![false; n]);
    }
    let tree = expr::parse(&branch.text, columns).map_err(|e| format!("cannot parse the branch '{}': {}", branch.text, e))?;
    let mut row = vec![0.0f64; columns.len()];
    Ok((0..n)
        .map(|i| {
            for (k, c) in values.iter().enumerate() {
                row[k] = c[i];
            }
            let v = expr::eval(&tree, &row);
            v.is_finite() && v > 0.0
        })
        .collect())
}

/// Operators of a tree, without the leaves.
fn tree_ops(n: &Node, out: &mut Vec<Op>) {
    if n.op != Op::Const && n.op != Op::Input && !out.contains(&n.op) {
        out.push(n.op);
    }
    for k in &n.kids {
        tree_ops(k, out);
    }
}

/// What the search loop needs besides the rows.
pub struct Setup<'a> {
    pub train: &'a TradingSlice,
    /// Raw values of [`Request::columns`] on the training rows.
    pub train_values: &'a [Vec<f64>],
    pub train_segments: &'a [f64],
    pub vars: &'a [String],
    pub ops: &'a [Op],
    pub target: &'a str,
    pub target_sha: Option<&'a str>,
    pub per_side_bps: f64,
    pub split_json: &'a str,
    /// The choice ranks a P&L objective on validation rows.
    pub trading_objective: bool,
    pub validation: bool,
}

/// Checks the branch runs against this invocation, builds the frozen branch
/// positions on the training rows and returns the training book with the gate
/// layer in front. Exits 2 on a refusal.
pub fn setup(req: &Request, inner: &FitnessBook, s: &Setup) -> Arc<FitnessBook> {
    if !s.trading_objective {
        fail("the choice ranks the validation P&L book: add --objective pnl|sharpe|log_growth");
    }
    if !s.validation {
        fail("the choice reads the validation rows: add --validation-frac <f> or validation intervals");
    }
    if s.vars.is_empty() {
        fail("--vars is required: it lists the gate's variables");
    }
    if let Some(op) = s.ops.iter().find(|o| o.is_agg() || o.is_meta() || o.is_custom()) {
        fail(&format!(
            "--ops holds '{:?}': an aggregate, meta or building-block operator reads the whole column, future rows included, so the gate would not be causal",
            op
        ));
    }
    let split = Json::parse(s.split_json).unwrap_or_else(|| fail("the split of this invocation is not JSON"));
    for (flag, b) in [("--branch-on", &req.on), ("--branch-off", &req.off)] {
        if let Source::Run(r) = &b.source {
            if let Some(what) = differs(r, Some(s.target), s.target_sha, s.per_side_bps, &split) {
                fail(&format!("{} '{}' differs from this invocation in {}", flag, r.path, what));
            }
        }
    }
    let columns = req.columns(s.vars);
    let n = inner.len();
    if s.train.rows.time.len() != n || s.train_values.len() != columns.len() || s.train_values.iter().any(|v| v.len() != n) {
        fail(&format!("the carried columns do not match the {} training rows", n));
    }
    let on = long_rows(&req.on, &columns, s.train_values, n).unwrap_or_else(|e| fail(&e));
    let off = long_rows(&req.off, &columns, s.train_values, n).unwrap_or_else(|e| fail(&e));
    if on == off {
        fail("--branch-on and --branch-off hold the same positions on every training row: nothing to switch");
    }
    let groups: Vec<u32> = compose::state_groups(&s.train.rows, s.train_segments).into_iter().map(|g| g as u32).collect();
    let identity = format!("on={}|off={}", req.on.text, req.off.text);
    let layer = GateLayer::new(inner.clone(), on, off, groups, req.band, req.dwell, &identity).unwrap_or_else(|e| fail(&e));
    let layer = Arc::new(layer);
    let _ = LAYER.set(layer.clone());
    let book = inner.clone().with_layer(layer).unwrap_or_else(|e| fail(&e));
    Arc::new(book)
}

/// One front member as a searched gate.
struct Gate {
    expression: String,
    /// The tree as searched, over the normalised variables (the gate code reads it).
    search_tree: String,
    complexity: f64,
    search_fitness: f64,
    choice: Choice,
    gate: GateSpec,
    gate_bits: f64,
    train_net: f64,
}

/// What the search hands back besides the rows.
pub struct Finish<'a> {
    pub front: &'a [crate::pareto::Sol],
    pub candidates_evaluated: u64,
    pub generations: usize,
    pub vars: &'a [String],
    pub ops: &'a [Op],
    /// The centre and scale of every search variable: the front is printed in raw units.
    pub means: &'a [f64],
    pub stds: &'a [f64],
}

/// Turns the front into searched gates, hands them to compose's choice and
/// pricing, prints the composed run.json (or its summary) and exits.
pub fn finish(req: &Request, f: &Finish, trading: Option<&TradingSplit>, tspec: &TradingSpec, command: &str, inputs: &compose::Inputs) -> ! {
    let layer = LAYER.get().unwrap_or_else(|| fail("the gate layer was not set up"));
    let inner = layer.inner();
    let carried_names = req.columns(f.vars);
    let (means, stds) = (f.means, f.stds);
    let mut split = trading.cloned().unwrap_or_else(|| fail("no trading rows"));
    let gate_rows = split.gate_rows.take().unwrap_or_else(|| fail("the gate rows were not kept"));
    let train_values = split.train_values.clone();
    let n = inner.len();
    let pick = |name: &str| -> &Vec<f64> { &train_values[carried_names.iter().position(|c| c == name).expect("a carried column")] };

    // The code of the composite: the search alphabet and variables plus what the branches read.
    let mut code_vars: Vec<String> = f.vars.to_vec();
    for c in req.on.columns.iter().chain(&req.off.columns) {
        if !code_vars.contains(c) {
            code_vars.push(c.clone());
        }
    }
    let mut code_ops: Vec<Op> = f.ops.to_vec();
    let mut member_trees: Vec<(Pool, &Branch, Node)> = Vec::new();
    for (pool, b) in [(Pool::Bull, &req.on), (Pool::Bear, &req.off)] {
        if b.is_cash() {
            continue;
        }
        let tree = expr::parse(&b.expression_exact, &code_vars).unwrap_or_else(|e| fail(&format!("cannot parse the branch '{}': {}", b.expression_exact, e)));
        tree_ops(&tree, &mut code_ops);
        member_trees.push((pool, b, tree));
    }
    let code = CodeSpec::v1(&code_ops, code_vars.len());

    let mut gates: Vec<Gate> = Vec::new();
    for sol in f.front {
        let den = expr::simplify_display(&expr::denormalize(&expr::simplify_display(&sol.tree), means, stds));
        let src = expr::to_string_exact(&den, f.vars);
        let raw = GateSpec::parse(&src, req.band, req.dwell).unwrap_or_else(|e| fail(&format!("front member '{}': {}", src, e)));
        let cols: Vec<Vec<f64>> = raw.columns.iter().map(|c| pick(c).clone()).collect();
        let g = raw.values(&cols, n).unwrap_or_else(|e| fail(&e));
        let Some(choice) = layer.choose(&g) else { continue };
        let gate = match choice.cut {
            Cut::Grid { value, .. } => GateSpec::parse(&format!("({}) - ({:?})", src, value), req.band, req.dwell),
            Cut::Constant { on } => GateSpec::parse(if on { "1.0" } else { CASH }, 0.0, 1),
        }
        .unwrap_or_else(|e| fail(&format!("front member '{}': {}", src, e)));
        let gate_bits = compose::tree_bits(&code, &sol.tree).unwrap_or_else(|e| fail(&format!("front member '{}': {}", src, e))).0;
        let totals = inner.block_totals(&layer.composite(&choice.states));
        let train_net: f64 = totals.iter().map(|t| inner.net_of(t)).sum();
        gates.push(Gate { expression: src, search_tree: expr::to_string_exact(&sol.tree, f.vars), complexity: sol.cost, search_fitness: sol.error, choice, gate, gate_bits, train_net });
    }

    // The compose request: the searched gates as the menu, the branches as the members.
    let split_doc = Json::parse(inputs.split_json).unwrap_or(Json::Null);
    let mut runs: Vec<PoolRun> = Vec::new();
    let mut members: Vec<Member> = Vec::new();
    let mut member_bits: Vec<(f64, bool)> = Vec::new();
    for (pool, b, tree) in &member_trees {
        let run = match &b.source {
            Source::Run(r) => (**r).clone(),
            _ => PoolRun {
                pool: *pool,
                path: String::new(),
                sha256: String::new(),
                target: Some(inputs.target.to_string()),
                target_sha: inputs.target_sha.unwrap_or_default().to_string(),
                threshold: 0.0,
                per_side_bps: tspec.per_side_bps(),
                mask: None,
                split: split_doc.clone(),
                candidates_evaluated: None,
                front_size: 1,
                pnl_excess: PnlExcess::None,
                stat: BlockStat::MeanSd,
            },
        };
        member_bits.push(compose::tree_bits(&code, tree).unwrap_or_else(|e| fail(&format!("branch '{}': {}", b.expression_exact, e))));
        members.push(Member { run: runs.len(), archive_rank: b.archive_rank.max(1), expression_exact: b.expression_exact.clone(), columns: b.columns.clone(), branch: b.text.clone() });
        runs.push(run);
    }
    let menu = compose::Menu {
        path: String::new(),
        sha256: String::new(),
        gates: gates
            .iter()
            .enumerate()
            .map(|(k, g)| compose::MenuGate { name: format!("searched{}", k), codeword: "searched".to_string(), index: k, gate: g.gate.clone() })
            .collect(),
        switch_budget: None,
    };
    let branch = |m: usize| occam::Branch::Tree(member_bits[m].0);
    let bits: Vec<f64> = gates
        .iter()
        .map(|g| {
            let gate = occam::Gate::Searched(occam::Branch::Tree(g.gate_bits));
            let shape = match (req.on.is_cash(), req.off.is_cash()) {
                (false, false) => occam::Composite::C2 { gate, on: branch(0), off: branch(1) },
                _ => occam::Composite::C1 { gate, branch: branch(0) },
            };
            code.composite_bits(&shape).unwrap_or_else(|e| fail(&e))
        })
        .collect();
    let bound = inner.bound();
    let side_fitness = |b: &Branch| -> (f64, f64) {
        let states = vec![std::ptr::eq(b, &req.on); n];
        let totals = inner.block_totals(&layer.composite(&states));
        (inner.fitness_of(&totals), totals.iter().map(|t| inner.net_of(t)).sum())
    };
    let (on_fit, on_net) = side_fitness(&req.on);
    let (off_fit, off_net) = side_fitness(&req.off);
    let gates_json: Vec<String> = gates
        .iter()
        .enumerate()
        .map(|(k, g)| {
            let (cut, rank, constant) = match g.choice.cut {
                Cut::Grid { t, value } => (num(value), t.to_string(), "null".to_string()),
                Cut::Constant { on } => ("null".to_string(), "null".to_string(), text(if on { "on" } else { "off" })),
            };
            let on_rows = g.choice.states.iter().filter(|&&s| s).count();
            format!(
                "{{\"composite_id\":{},\"expression\":{},\"search_tree\":{},\"complexity\":{},\"search_fitness\":{},\"fitness\":{},\"train_net\":{},\"cut\":{},\"cut_rank\":{},\"constant\":{},\"on_share\":{},\"gate_bits\":{},\"bits\":{}}}",
                k,
                text(&g.expression),
                text(&g.search_tree),
                num(g.complexity),
                num(g.search_fitness),
                num(g.choice.fitness),
                num(g.train_net),
                cut,
                rank,
                constant,
                num(on_rows as f64 / n.max(1) as f64),
                num(g.gate_bits),
                num(bits[k])
            )
        })
        .collect();
    let json = format!(
        "{{\"schema\":\"{}\",\"branch_on\":{},\"branch_off\":{},\"band\":{},\"dwell\":{},\"form\":\"{}\",\"variables\":[{}],\
         \"cut\":{{\"rule\":\"lowest training fitness over the inner cuts of the quantile grid of the gate's training values; a gate with fewer than two distinct values is constant and holds the better branch\",\"quantiles\":{}}},\
         \"generations\":{},\"candidates_evaluated\":{},\"front\":{},\"bound\":{},\
         \"branches_train\":{{\"on\":{{\"fitness\":{},\"net\":{}}},\"off\":{{\"fitness\":{},\"net\":{}}}}},\"gates\":[{}]}}",
        SCHEMA,
        req.on.json(),
        req.off.json(),
        num(req.band),
        req.dwell,
        req.form(),
        f.vars.iter().map(|v| text(v)).collect::<Vec<_>>().join(","),
        QUANTILES,
        f.generations,
        f.candidates_evaluated,
        f.front.len(),
        num(bound),
        num(on_fit),
        num(on_net),
        num(off_fit),
        num(off_net),
        gates_json.join(",")
    );
    let pareto_extra: Vec<String> = gates
        .iter()
        .map(|g| {
            format!(
                ",\"gate_search\":{{\"expression\":{},\"cut\":{},\"constant\":{},\"search_fitness\":{},\"fitness\":{},\"train_net\":{}}}",
                text(&g.expression),
                match g.choice.cut {
                    Cut::Grid { value, .. } => num(value),
                    Cut::Constant { .. } => "null".to_string(),
                },
                matches!(g.choice.cut, Cut::Constant { .. }),
                num(g.search_fitness),
                num(g.choice.fitness),
                num(g.train_net)
            )
        })
        .collect();
    let compose_req = compose::Request {
        menu,
        runs,
        members,
        top: 1,
        json: req.json,
        out: req.out.clone(),
        reselect: None,
        searched: Some(compose::Searched { candidates_evaluated: f.candidates_evaluated, generations: f.generations, member_bits, bits, code: code.json(), json, pareto_extra }),
    };

    // The rows compose reads: the carried values in its column order and the
    // rcsr-gate-state/1 states of every searched gate.
    let columns = compose_req.columns();
    let align = |values: &[Vec<f64>]| -> Vec<Vec<f64>> {
        columns.iter().map(|c| values.get(carried_names.iter().position(|k| k == c).expect("a carried column")).cloned().unwrap_or_default()).collect()
    };
    split.train_values = align(&train_values);
    if !split.validation_values.is_empty() {
        split.validation_values = align(&split.validation_values);
    }
    split.test_values = align(&split.test_values);
    split.gate_states = compose_req
        .menu
        .gates
        .iter()
        .map(|g| gate_rows.states(&g.gate))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| exit_with((3, e)));
    compose::start_split(&compose_req, command, Some(&split), Some(inner), Some(tspec), inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::{TradingRows, TradingSpec};

    /// Two groups of `n` rows on one clock, with a regime that flips every 8 stamps.
    fn book(n: usize, blocks: usize) -> (FitnessBook, TradingSlice) {
        let mut time = Vec::new();
        let mut group = Vec::new();
        let mut ret = Vec::new();
        for g in 0..2u64 {
            for k in 0..n {
                time.push(k as f64 * 3600.0);
                group.push(g);
                let up = (k / 8) % 2 == 0;
                let x = ((k * 7 + g as usize * 3) % 11) as f64 / 11.0 - 0.45;
                ret.push(if up { 0.01 * x } else { -0.01 * x } + 0.0001 * g as f64);
            }
        }
        let slice = TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret };
        let spec = TradingSpec { cost_bps: 5.0, ..Default::default() };
        let b = FitnessBook::with_blocks(&slice, &spec, 3600.0, blocks, 1.0).expect("a profitable book");
        (b, slice)
    }

    fn branches(slice: &TradingSlice) -> (Vec<bool>, Vec<bool>, Vec<f64>) {
        let n = slice.ret.len() / 2;
        let x: Vec<f64> = (0..2 * n).map(|i| ((i % n) * 7 + (i / n) * 3) as f64 % 11.0 / 11.0 - 0.45).collect();
        let a: Vec<bool> = x.iter().map(|v| *v > 0.0).collect();
        let b: Vec<bool> = x.iter().map(|v| -*v > 0.0).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| if ((i % n) / 8) % 2 == 0 { 0.6 } else { -0.4 } + 0.001 * ((i * 13) % 17) as f64).collect();
        (a, b, g)
    }

    fn groups(slice: &TradingSlice) -> Vec<u32> {
        compose::state_groups(&slice.rows, &[]).into_iter().map(|g| g as u32).collect()
    }

    #[test]
    fn grid_cuts_follow_the_exact_sweep_grid() {
        let (b, slice) = book(64, 1);
        let (a, bb, g) = branches(&slice);
        let sweep = crate::switch::gate_sweep(&b, &a, &bb, &g, &slice.rows.time).unwrap();
        let cuts = grid_cuts(&g, QUANTILES);
        let ranks: Vec<usize> = cuts.iter().map(|c| c.0).collect();
        assert_eq!(ranks, sweep.grid(QUANTILES));
        let layer = GateLayer::new(b.clone(), a.clone(), bb.clone(), groups(&slice), 0.0, 1, "t").unwrap();
        for &(t, c) in &cuts {
            assert_eq!(c.to_bits(), sweep.value(&g, t).to_bits(), "cut value of rank {t}");
            // Band 0 and dwell 1: the kernel state of g - c is the sweep's state of rank t.
            let st = layer.states_at(&g, c);
            assert_eq!(st, sweep.state(t), "states of rank {t}");
            let f = b.regret(&layer.composite(&st));
            assert!((f - sweep.fitness(t)).abs() <= 1e-12, "rank {t}: {f} vs the sweep {}", sweep.fitness(t));
        }
        let best = layer.choose(&g).unwrap();
        let want = cuts.iter().map(|&(t, _)| sweep.fitness(t)).fold(f64::INFINITY, f64::min);
        assert!((best.fitness - want).abs() <= 1e-12, "{} vs {want}", best.fitness);
    }

    #[test]
    fn the_layer_book_prices_the_best_cut_composite() {
        for (blocks, band, dwell) in [(1, 0.0, 1), (4, 0.0, 1), (1, 0.05, 3), (4, 0.05, 2)] {
            let (b, slice) = book(96, blocks);
            let (a, bb, g) = branches(&slice);
            let layer = Arc::new(GateLayer::new(b.clone(), a, bb, groups(&slice), band, dwell, "t").unwrap());
            let choice = layer.choose(&g).unwrap();
            let layered = b.clone().with_layer(layer.clone()).unwrap();
            assert_ne!(layered.hash(), b.hash(), "the layer joins the hash");
            assert_eq!(layered.regret(&g).to_bits(), choice.fitness.to_bits(), "blocks {blocks} band {band} dwell {dwell}");
            assert_eq!(layered.block_totals(&g), b.block_totals(&layer.composite(&choice.states)));
            assert_eq!(layered.case_errors(&g), b.case_errors(&layer.composite(&choice.states)));
            // Every grid cut scores at least the chosen one.
            for (_, c) in grid_cuts(&g, QUANTILES) {
                assert!(b.regret(&layer.composite(&layer.states_at(&g, c))) >= choice.fitness);
            }
            assert!(layered.cut_sweep(&g, &slice.rows.time).is_err());
        }
    }

    #[test]
    fn cash_off_branch_is_flat_in_off_states() {
        let (b, slice) = book(64, 1);
        let (a, _, g) = branches(&slice);
        let n = a.len();
        let layer = GateLayer::new(b, a.clone(), vec![false; n], groups(&slice), 0.0, 1, "t").unwrap();
        let choice = layer.choose(&g).unwrap();
        let pos = layer.composite(&choice.states);
        for i in 0..n {
            if !choice.states[i] {
                assert_eq!(pos[i], f64::MIN, "row {i} is off and flat");
            } else {
                assert_eq!(pos[i] == f64::MAX, a[i], "row {i} is on and holds the on branch");
            }
        }
    }

    #[test]
    fn a_constant_gate_holds_the_better_branch_and_a_non_finite_gate_scores_inf() {
        let (b, slice) = book(64, 1);
        let (a, bb, _) = branches(&slice);
        let n = a.len();
        let layer = Arc::new(GateLayer::new(b.clone(), a.clone(), bb.clone(), groups(&slice), 0.0, 1, "t").unwrap());
        let choice = layer.choose(&vec![0.25; n]).unwrap();
        let fa = b.regret(&layer.composite(&vec![true; n]));
        let fb = b.regret(&layer.composite(&vec![false; n]));
        assert_eq!(choice.cut, Cut::Constant { on: fa <= fb });
        assert_eq!(choice.fitness.to_bits(), fa.min(fb).to_bits());
        let mut g = vec![0.25; n];
        g[n - 1] = f64::NAN;
        assert!(layer.choose(&g).is_none());
        let layered = b.with_layer(layer).unwrap();
        assert_eq!(layered.regret(&g), f64::INFINITY);
    }

    #[test]
    fn the_layer_refuses_a_masked_or_banded_book() {
        let (b, slice) = book(32, 1);
        let n = b.len();
        let spec = TradingSpec { cost_bps: 5.0, min_holding: 2, ..Default::default() };
        let held = FitnessBook::new(&slice, &spec, 3600.0).unwrap();
        assert!(GateLayer::new(held, vec![true; n], vec![false; n], groups(&slice), 0.0, 1, "t").is_err());
        assert!(GateLayer::new(b, vec![true; n - 1], vec![false; n], groups(&slice), 0.0, 1, "t").is_err());
    }
}
