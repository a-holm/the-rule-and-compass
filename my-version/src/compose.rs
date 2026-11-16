//! #1181: `rcsr compose`, the regime-switching composition of specialist pools
//! into one exact formula.
//!
//! Pools are the top K members of run.json fronts: bull and bear specialists
//! (searched with `--pnl-mask` or plainly) and all-weather members. A
//! pre-registered gate menu fixes the expression, band, dwell and codeword of
//! every gate. The command builds every composite of the menu: C0 an
//! all-weather member alone, C1 a specialist in one gate state and cash in the
//! other (`if(gate, f_bull, cash)` and `if(gate, cash, f_bear)`), C2 a bull
//! and a bear branch (`if(gate, f_bull, f_bear)`). A composite is one
//! expression of the existing language, long where its value is above 0: a
//! branch carries its run's threshold folded in (`(f) - (t)`) and cash is the
//! constant -1.
//!
//! The choice reads the training and validation rows only; the test rows are
//! read once the finalist is fixed. A C1 or C2 composite is a switch
//! ([`regime_gate::Switch`]): its validation and test rows take the gate states
//! of the evaluation rule `rcsr-gate-state/1` (warm-up over the full readable
//! prefix of the data file; the validation rows never read the test intervals),
//! its training rows the per-segment kernel of the `--pnl-mask` fitness (#1190).
//! The run carries the switch of every C1 and C2 composite, so every reader
//! prices the same book.

use crate::data::TradingSlice;
use crate::devlog::{num, text, Json};
use crate::expr::{self, Node};
use crate::occam::{self, CodeSpec};
use crate::regime_gate::{self, GateSpec, Regime};
use crate::selection::{RobustMember, RobustMeta, RobustParams, RobustSelection};
use crate::switch;
use crate::trading::{self, FitnessBook, PathStats, TradingRows, TradingSpec};
use std::collections::HashMap;

pub const SCHEMA: &str = "rcsr-compose/1";
pub const MENU_SCHEMA: &str = "rcsr-gate-menu/1";
/// The cash branch: below the composite threshold 0 on every row.
const FLAT: &str = "-1.0";
const FLAT_VALUE: f64 = -1.0;

type Fail = (i32, String);

fn usage<T>(m: impl Into<String>) -> Result<T, Fail> {
    Err((2, m.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Bull,
    Bear,
    AllWeather,
}

impl Pool {
    pub fn flag(self) -> &'static str {
        match self {
            Pool::Bull => "--bull",
            Pool::Bear => "--bear",
            Pool::AllWeather => "--all-weather",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Pool::Bull => "bull",
            Pool::Bear => "bear",
            Pool::AllWeather => "all_weather",
        }
    }
}

/// One entry of the pre-registered gate menu.
#[derive(Clone, Debug)]
pub struct MenuGate {
    pub name: String,
    pub codeword: String,
    /// The codeword read as a binary number: the menu index of the composite code.
    pub index: usize,
    pub gate: GateSpec,
}

impl MenuGate {
    /// Band 0 and dwell 1: the state of a row is `greater(gate, 0)` of the row.
    pub fn pure(&self) -> bool {
        self.gate.band == 0.0 && self.gate.dwell == 1
    }
}

/// The gate menu file (`rcsr-gate-menu/1`).
#[derive(Clone, Debug)]
pub struct Menu {
    pub path: String,
    pub sha256: String,
    pub gates: Vec<MenuGate>,
    /// Switch budget k of the pruning bound; `None` prunes nothing.
    pub switch_budget: Option<usize>,
}

/// Entries of the composite code's gate menu (`composite.menu_size` of the code spec).
fn menu_size() -> usize {
    Json::parse(occam::SPEC_V1.trim())
        .and_then(|j| j.get("composite").and_then(|c| c.get("menu_size")).and_then(Json::as_usize))
        .expect("the embedded occam code spec names composite.menu_size")
}

impl Menu {
    pub fn parse(text: &str) -> Result<Menu, String> {
        let j = Json::parse(text.trim()).ok_or("the gate menu is not JSON")?;
        if j.get("schema").and_then(Json::as_str) != Some(MENU_SCHEMA) {
            return Err(format!("the gate menu needs \"schema\":\"{}\"", MENU_SCHEMA));
        }
        let size = menu_size();
        if !size.is_power_of_two() || size < 2 {
            return Err(format!("the code registers {} menu gates, not a power of two", size));
        }
        let width = size.trailing_zeros() as usize;
        let Some(Json::Arr(list)) = j.get("gates") else {
            return Err("the gate menu has no 'gates' array".to_string());
        };
        if list.is_empty() || list.len() > size {
            return Err(format!("the gate menu holds {} gates; the composite code registers 1 to {}", list.len(), size));
        }
        let mut gates: Vec<MenuGate> = Vec::with_capacity(list.len());
        for (k, g) in list.iter().enumerate() {
            let field = |key: &str| g.get(key).ok_or_else(|| format!("gate {} has no '{}'", k, key));
            let name = field("name")?.as_str().filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("gate {}: 'name' must be a non-empty string", k))?;
            let codeword = field("codeword")?.as_str().ok_or_else(|| format!("gate {}: 'codeword' must be a string", k))?;
            if codeword.len() != width || !codeword.bytes().all(|b| b == b'0' || b == b'1') {
                return Err(format!("gate '{}': codeword '{}' must be {} binary digits (a menu of {} gates)", name, codeword, width, size));
            }
            let index = usize::from_str_radix(codeword, 2).expect("checked binary digits");
            let expr = field("expr")?.as_str().ok_or_else(|| format!("gate '{}': 'expr' must be a string", name))?;
            let band = field("band")?.as_f64().ok_or_else(|| format!("gate '{}': 'band' must be a number", name))?;
            let dwell = field("dwell")?.as_f64().filter(|d| *d >= 1.0 && d.fract() == 0.0 && *d < 1e9).ok_or_else(|| format!("gate '{}': 'dwell' must be an integer >= 1", name))?;
            let gate = GateSpec::parse(expr, band, dwell as usize).map_err(|e| format!("gate '{}': {}", name, e))?;
            if gates.iter().any(|o| o.name == name) {
                return Err(format!("gate name '{}' appears twice", name));
            }
            if let Some(o) = gates.iter().find(|o| o.codeword == codeword) {
                return Err(format!("gates '{}' and '{}' share codeword {}", o.name, name, codeword));
            }
            gates.push(MenuGate { name: name.to_string(), codeword: codeword.to_string(), index, gate });
        }
        let switch_budget = match j.get("switch_budget") {
            None | Some(Json::Null) => None,
            Some(v) => Some(
                v.as_f64()
                    .filter(|k| *k >= 0.0 && k.fract() == 0.0 && *k < 1e9)
                    .ok_or("'switch_budget' must be an integer >= 0 or null")? as usize,
            ),
        };
        Ok(Menu { path: String::new(), sha256: crate::manifest::sha256_hex(text.as_bytes()), gates, switch_budget })
    }

    fn load(path: &str) -> Result<Menu, Fail> {
        let text = std::fs::read_to_string(path).map_err(|e| (3, format!("--gates '{}': {}", path, e)))?;
        let mut menu = Menu::parse(&text).map_err(|e| (2, format!("--gates '{}': {}", path, e)))?;
        menu.path = path.to_string();
        Ok(menu)
    }
}

/// An input run of one pool.
#[derive(Clone, Debug)]
pub struct PoolRun {
    pub pool: Pool,
    pub path: String,
    pub sha256: String,
    pub target: Option<String>,
    pub target_sha: String,
    pub threshold: f64,
    pub per_side_bps: f64,
    pub mask: Option<GateSpec>,
    pub split: Json,
    pub candidates_evaluated: Option<u64>,
    pub front_size: usize,
}

/// One pool member: a front member of an input run.
#[derive(Clone, Debug)]
pub struct Member {
    pub run: usize,
    pub archive_rank: usize,
    pub expression_exact: String,
    pub columns: Vec<String>,
    /// Long where the value is above 0: the member with its run's threshold subtracted.
    pub branch: String,
}

/// The split fields every input run and the compose invocation must share.
const SPLIT_KEYS: [&str; 12] = [
    "mode",
    "train_rows",
    "validation_rows",
    "test_rows",
    "train_time",
    "validation_time",
    "required_purge_seconds",
    "purge_seconds",
    "split_id",
    "intervals",
    "interval_purge_seconds",
    "embargo_seconds",
];

fn load_run(pool: Pool, path: &str, top: usize, index: usize, members: &mut Vec<Member>) -> Result<PoolRun, Fail> {
    let flag = pool.flag();
    let body = std::fs::read_to_string(path).map_err(|e| (3, format!("{} '{}': {}", flag, path, e)))?;
    let bad = |e: String| (2, format!("{} '{}': {}", flag, path, e));
    let run = crate::run_json::RunFile::parse(&body).map_err(|e| {
        if e.contains(crate::run_json::STALE_COMPOSE_REFUSAL) {
            (3, format!("{} '{}': {}", flag, path, e))
        } else {
            bad(e)
        }
    })?;
    if run.front.iter().any(|m| m.switch.is_some()) {
        return Err(bad("a member carries a switch (a composed run): a composite cannot be a branch, nested switches are not supported".to_string()));
    }
    let book = run.band_book().map_err(bad)?;
    if book.position != trading::PositionMap::Threshold {
        return Err(bad("the run priced the top_tercile book; compose composes threshold books".to_string()));
    }
    if book.has_band() {
        return Err(bad("the run priced a band or a minimum hold; compose composes plain threshold books".to_string()));
    }
    if !book.threshold.is_finite() {
        return Err(bad(format!("objective.spec.threshold {} is not finite", book.threshold)));
    }
    let doc = Json::parse(body.trim()).ok_or_else(|| bad("not a run.json document".to_string()))?;
    let Some(Json::Arr(pareto)) = doc.get("pareto") else {
        return Err(bad("no 'pareto' array".to_string()));
    };
    let mut front: Vec<(usize, String)> = Vec::with_capacity(pareto.len());
    for (k, m) in pareto.iter().enumerate() {
        let rank = m.get("archive_rank").and_then(Json::as_usize).ok_or_else(|| bad(format!("pareto[{}] has no archive_rank", k)))?;
        let e = m
            .get("expression_exact")
            .and_then(Json::as_str)
            .ok_or_else(|| bad(format!("pareto[{}] has no expression_exact (the rounded display string would price another formula)", k)))?;
        front.push((rank, e.to_string()));
    }
    if front.is_empty() {
        return Err(bad("the Pareto front is empty".to_string()));
    }
    front.sort_by_key(|m| m.0);
    for (rank, e) in front.iter().take(top) {
        let columns = regime_gate::referenced_columns(e);
        expr::parse(e, &columns).map_err(|err| bad(format!("archive_rank {}: cannot parse '{}': {}", rank, e, err)))?;
        let branch = if book.threshold == 0.0 { e.clone() } else { format!("({}) - ({:?})", e, book.threshold) };
        members.push(Member { run: index, archive_rank: *rank, expression_exact: e.clone(), columns, branch });
    }
    Ok(PoolRun {
        pool,
        path: path.to_string(),
        sha256: crate::manifest::sha256_hex(body.as_bytes()),
        target: doc.get("target").and_then(Json::as_str).map(str::to_string),
        target_sha: book.spec.sha256(),
        threshold: book.threshold,
        per_side_bps: book.per_side_bps,
        mask: book.mask.clone(),
        split: doc.get("split").cloned().unwrap_or(Json::Null),
        candidates_evaluated: run.candidates_evaluated,
        front_size: front.len(),
    })
}

/// The split field `key` of two split objects differs (absent on both is equal).
fn split_differs(a: &Json, b: &Json, key: &str) -> bool {
    a.get(key) != b.get(key)
}

/// The flags of `rcsr compose` that the search parser does not read.
#[derive(Clone, Debug)]
pub struct Request {
    pub menu: Menu,
    pub runs: Vec<PoolRun>,
    pub members: Vec<Member>,
    pub top: usize,
    pub json: bool,
    pub out: Option<String>,
    /// #1214: `--reselect-run` of a composed run: the choice reruns over the run's composites.
    pub reselect: Option<Reselect>,
}

static REQUEST: std::sync::OnceLock<Request> = std::sync::OnceLock::new();

/// The `rcsr compose` request of this process, when the command is `compose`.
pub fn request() -> Option<&'static Request> {
    REQUEST.get()
}

/// #1214: what a re-selection keeps from the composed run: the bits, the
/// pruning and the code of every composite, which this invocation's `--ops`
/// and `--vars` could not reproduce.
#[derive(Clone, Debug)]
pub struct Reselect {
    pub path: String,
    pub sha256: String,
    pub member_bits: Vec<(f64, bool)>,
    pub bits: Vec<f64>,
    pub pruned: Vec<bool>,
    pub bound_gain: Vec<Option<f64>>,
    pub code: String,
}

/// The `--reselect-run` path of `args` when that run.json has a `compose`
/// block; a run that cannot be read is left to the search parser.
pub fn reselect_target(args: &[String]) -> Option<String> {
    let at = args.iter().position(|a| a == "--reselect-run")?;
    let path = args.get(at + 1)?;
    let body = std::fs::read_to_string(path).ok()?;
    Json::parse(body.trim())?.get("compose")?;
    Some(path.clone())
}

/// #1214: `rcsr <compose flags> --reselect-run <composed.run.json>`: compose's
/// choice over the run's composites on this invocation's rows. The menu, the
/// members and the composites come from the run; the pool and menu flags exit 2.
pub fn claim_reselect(args: &[String], path: &str) -> Vec<String> {
    let mut rest: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--reselect-run" {
            i += 2;
            continue;
        }
        rest.push(args[i].clone());
        i += 1;
    }
    claim_with(&rest, Some(path))
}

/// Compact JSON text of a parsed value.
fn json_text(j: &Json) -> String {
    match j {
        Json::Null => "null".to_string(),
        Json::Bool(b) => b.to_string(),
        Json::Num(v) => num(*v),
        Json::Str(s) => text(s),
        Json::Arr(a) => format!("[{}]", a.iter().map(json_text).collect::<Vec<_>>().join(",")),
        Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}:{}", text(k), json_text(v))).collect::<Vec<_>>().join(",")),
    }
}

fn from_composed(path: &str) -> Result<Request, Fail> {
    let body = std::fs::read_to_string(path).map_err(|e| (3, format!("--reselect-run '{}': {}", path, e)))?;
    let bad = |e: String| (3, format!("--reselect-run '{}': {}", path, e));
    let doc = Json::parse(body.trim()).ok_or_else(|| bad("not a run.json document".to_string()))?;
    let c = doc.get("compose").ok_or_else(|| bad("no compose block".to_string()))?;
    if c.get("schema").and_then(Json::as_str) != Some(SCHEMA) {
        return Err(bad(format!("compose.schema is not {}", SCHEMA)));
    }
    let run = crate::run_json::RunFile::parse(&body).map_err(bad)?;
    let book = run.band_book().map_err(bad)?;
    let list = |j: Option<&Json>, what: &str| -> Result<Vec<Json>, Fail> {
        match j {
            Some(Json::Arr(a)) => Ok(a.clone()),
            _ => Err(bad(format!("compose.{} is not a list", what))),
        }
    };
    let menu_j = c.get("menu").ok_or_else(|| bad("no compose.menu".to_string()))?;
    let mut gates = Vec::new();
    for (k, g) in list(menu_j.get("gates"), "menu.gates")?.iter().enumerate() {
        let s = |key: &str| g.get(key).and_then(Json::as_str).map(str::to_string).ok_or_else(|| bad(format!("compose.menu.gates[{}] has no '{}'", k, key)));
        let n = |key: &str| g.get(key).and_then(Json::as_f64).ok_or_else(|| bad(format!("compose.menu.gates[{}] has no '{}'", k, key)));
        let gate = GateSpec::parse(&s("expr")?, n("band")?, n("dwell")? as usize).map_err(|e| bad(format!("compose.menu.gates[{}]: {}", k, e)))?;
        gates.push(MenuGate { name: s("name")?, codeword: s("codeword")?, index: n("index")? as usize, gate });
    }
    let menu = Menu {
        path: menu_j.get("path").and_then(Json::as_str).unwrap_or_default().to_string(),
        sha256: menu_j.get("sha256").and_then(Json::as_str).unwrap_or_default().to_string(),
        gates,
        switch_budget: menu_j.get("switch_budget").and_then(Json::as_usize),
    };
    let mut runs = Vec::new();
    for (k, r) in list(c.get("runs"), "runs")?.iter().enumerate() {
        let pool = match r.get("pool").and_then(Json::as_str) {
            Some("bull") => Pool::Bull,
            Some("bear") => Pool::Bear,
            Some("all_weather") => Pool::AllWeather,
            other => return Err(bad(format!("compose.runs[{}].pool {:?}", k, other))),
        };
        let mask = match r.get("mask") {
            Some(m @ Json::Obj(_)) => {
                let e = m.get("expr").and_then(Json::as_str).unwrap_or_default();
                let b = m.get("band").and_then(Json::as_f64).unwrap_or(0.0);
                let d = m.get("dwell").and_then(Json::as_f64).unwrap_or(1.0) as usize;
                Some(GateSpec::parse(e, b, d).map_err(|e| bad(format!("compose.runs[{}].mask: {}", k, e)))?)
            }
            _ => None,
        };
        runs.push(PoolRun {
            pool,
            path: r.get("path").and_then(Json::as_str).unwrap_or_default().to_string(),
            sha256: r.get("sha256").and_then(Json::as_str).unwrap_or_default().to_string(),
            target: doc.get("target").and_then(Json::as_str).map(str::to_string),
            target_sha: book.spec.sha256(),
            threshold: r.get("threshold").and_then(Json::as_f64).unwrap_or(0.0),
            per_side_bps: book.per_side_bps,
            mask,
            split: doc.get("split").cloned().unwrap_or(Json::Null),
            candidates_evaluated: r.get("candidates_evaluated").and_then(Json::as_f64).map(|v| v as u64),
            front_size: r.get("front_size").and_then(Json::as_usize).unwrap_or(0),
        });
    }
    let mut members = Vec::new();
    let mut member_bits = Vec::new();
    for (k, m) in list(c.get("members"), "members")?.iter().enumerate() {
        let s = |key: &str| m.get(key).and_then(Json::as_str).map(str::to_string).ok_or_else(|| bad(format!("compose.members[{}] has no '{}'", k, key)));
        let run_index = m.get("run").and_then(Json::as_usize).filter(|&r| r < runs.len()).ok_or_else(|| bad(format!("compose.members[{}].run", k)))?;
        let expression_exact = s("expression_exact")?;
        member_bits.push((m.get("bits").and_then(Json::as_f64).unwrap_or(f64::NAN), matches!(m.get("exact_codeword"), Some(Json::Bool(true)))));
        members.push(Member {
            run: run_index,
            archive_rank: m.get("archive_rank").and_then(Json::as_usize).unwrap_or(0),
            columns: regime_gate::referenced_columns(&expression_exact),
            expression_exact,
            branch: s("branch")?,
        });
    }
    let req = Request { menu, runs, members, top: c.get("top").and_then(Json::as_usize).unwrap_or(0), json: false, out: None, reselect: None };
    let composites = enumerate(&req);
    let listed = list(c.get("composites"), "composites")?;
    if listed.len() != composites.len() {
        return Err(bad(format!("compose.composites holds {} entries; the menu and the members give {}", listed.len(), composites.len())));
    }
    let (mut bits, mut pruned, mut bound_gain) = (Vec::new(), Vec::new(), Vec::new());
    for (k, (j, made)) in listed.iter().zip(&composites).enumerate() {
        let same = j.get("id").and_then(Json::as_usize) == Some(made.id)
            && j.get("expression").and_then(Json::as_str) == Some(made.text.as_str())
            && j.get("type").and_then(Json::as_str) == Some(made.kind.label());
        if !same {
            return Err(bad(format!("compose.composites[{}] is not the composite the menu and the members give ('{}')", k, made.text)));
        }
        bits.push(j.get("bits").and_then(Json::as_f64).unwrap_or(f64::NAN));
        pruned.push(matches!(j.get("pruned"), Some(Json::Bool(true))));
        bound_gain.push(j.get("bound_gain").and_then(Json::as_f64));
    }
    let code = c.get("code").map_or_else(|| "null".to_string(), json_text);
    let sha256 = crate::manifest::sha256_hex(body.as_bytes());
    Ok(Request {
        reselect: Some(Reselect { path: path.to_string(), sha256, member_bits, bits, pruned, bound_gain, code }),
        ..req
    })
}

impl Request {
    fn members_of(&self, pool: Pool) -> Vec<usize> {
        (0..self.members.len()).filter(|&m| self.runs[self.members[m].run].pool == pool).collect()
    }

    /// The columns each flag reads, for the `--pnl-mask` column checks.
    pub fn column_sets(&self) -> Vec<(&'static str, Vec<String>)> {
        let mut out = vec![("--gates", self.menu.gates.iter().flat_map(|g| g.gate.columns.iter().cloned()).collect())];
        for pool in [Pool::Bull, Pool::Bear, Pool::AllWeather] {
            let cols: Vec<String> = self.members_of(pool).iter().flat_map(|&m| self.members[m].columns.iter().cloned()).collect();
            out.push((pool.flag(), cols));
        }
        out
    }

    /// The menu gates, whose `rcsr-gate-state/1` states the validation and test slices carry.
    pub fn state_gates(&self) -> Vec<GateSpec> {
        self.menu.gates.iter().map(|g| g.gate.clone()).collect()
    }

    /// Every column read, gate columns first, in order of first use.
    pub fn columns(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, cols) in self.column_sets() {
            for c in cols {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out
    }
}

pub fn usage_text() -> &'static str {
    "rcsr compose: regime-switching composition of specialist pools into one exact formula (#1181)

USAGE:
  rcsr compose <the flags of a --fitness pnl run> --gates <menu.json>
               --bull <run.json>... --bear <run.json>... [--all-weather <run.json>...]
               [--top K] [--json] [--out <composed.run.json>]

  The data, target, vars, ops, split, objective, cost and block flags are the
  ones of the pool searches (--fitness pnl, --select-rule robust and --cpu are
  implied); the command builds that run's rows and stops before any search.
  Every input run must share the target, the target spec, the per-side cost
  and the split (mode, row counts, train and validation time, purge, split id,
  intervals, embargo) with this invocation; a difference exits 2 naming it.

  --gates <menu.json>      the pre-registered gate menu (rcsr-gate-menu/1):
                           {\"schema\":\"rcsr-gate-menu/1\",\"switch_budget\":k|null,
                           \"gates\":[{\"name\",\"codeword\",\"expr\",\"band\",\"dwell\"}]};
                           at most 8 gates, each codeword 3 binary digits, distinct
                           (the menu index of the composite code); a row is on above
                           +band, off below -band, else its previous state, a change
                           waiting for dwell rows per group (the --pnl-mask rule);
                           a non-finite value is off. A label-family
                           column, the target, its return column or a construction
                           column exits 5; a missing column exits 3
  --bull <run.json>        a bull pool run, repeatable: its members hold the gate-on
                           branch of C1 if(gate, f, cash) and C2 if(gate, f, f_bear)
  --bear <run.json>        a bear pool run, repeatable: its members hold the gate-off
                           branch of C1 if(gate, cash, f) and C2
  --all-weather <run.json> an all-weather pool run, repeatable: its members are C0
                           composites alone
  --top K                  members per run by archive rank (default 5)
  --json                   print the composed run.json instead of the text summary
  --out <file>             write the composed run.json to this file

  Composites: every all-weather member (C0), and per menu gate every bull member
  with cash (C1), every bear member with cash (C1) and every bull x bear pair (C2):
  gates x (bull x bear + bull + bear) + all-weather; all of them are counted in
  search.candidates_evaluated and listed in compose.composites with the bits of
  the composite code (docs/occam-code-v1.json; report-only). A branch is its
  member with the run's threshold subtracted; the composite threshold is 0.
  With switch_budget k, a C1 or C2 pair whose k-switch bound gain over its better
  branch on the training rows is not positive is pruned (counted, not scored).

  Choice (training and validation rows only): the robust rule of --select-rule
  robust on the validation rows, plus alpha over exposure-matched buy-and-hold
  >= 0 in the gate-on and in the gate-off rows of a C1 or C2; no eligible
  composite gives verdict no_signal and the best all-weather member (else the
  best composite) by the validation objective. The test rows are priced after
  the finalist is fixed. The report gives per gate state net, exposure,
  buy-and-hold and alpha, gate switches per year and mean dwell, the cost paid
  in switch periods against the total cost, each branch alone and the best
  all-weather member on the same rows.

  Every C1 and C2 composite is a switch: run.json carries objective.finalist.switch
  and pareto[].switch (gate, on branch, off branch), and pareto lists every
  composite with pruned. The validation and test rows take the gate states of
  rcsr-gate-state/1: the kernel over every row of the data file with a finite
  time, per group from its first readable row, never reading a locked range the
  command does not open (with its lookback embargo); the validation rows also
  never read a test interval (with the split's embargo). The training rows keep
  the per-segment start of the --pnl-mask fitness. compose.chosen.validation.regime
  and .test.regime carry the warm-up (start). walkforward --from-run and --front,
  ladder --sr-run, the seal and the locked read price the switch.

  rcsr <these flags without the menu and pool flags> --reselect-run <composed.run.json>
  reruns the choice over the run's composites on this invocation's rows (the
  run's pruning, bits and code kept); the split must match the run (exit 2).

  A band, a minimum hold, --threshold, --position, --book-target, --pnl-mask,
  --sample-rows, --pnl-excess, --pnl-stat, --pnl-cvar-q and --gpu exit 2. A pool run whose
  member carries a switch exits 2 (nested switch); a run with
  compose.gate.carried and no switch exits 3. See docs/compose.md.
"
}

fn fail(msg: &str) -> ! {
    eprintln!("error: rcsr compose: {}", msg);
    std::process::exit(2);
}

fn exit_with((code, msg): Fail) -> ! {
    eprintln!("error: rcsr compose: {}", msg);
    std::process::exit(code);
}

/// Takes the `rcsr compose` flags out of `args` (the arguments after `compose`),
/// loads the gate menu and the input runs, keeps the request for [`request`], and
/// returns the search flags with `--fitness pnl`, `--select-rule robust` and
/// `--cpu` added when absent. Usage errors and mismatched runs exit 2, an
/// unreadable file exits 3.
pub fn claim(args: &[String]) -> Vec<String> {
    claim_with(args, None)
}

fn claim_with(args: &[String], reselect: Option<&str>) -> Vec<String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage_text());
        std::process::exit(0);
    }
    let (mut gates, mut top, mut json, mut out) = (None, None, false, None);
    let mut paths: Vec<(Pool, String)> = Vec::new();
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> String { args.get(i + 1).cloned().unwrap_or_else(|| fail(&format!("{} requires a value", flag))) };
        match flag {
            "--gates" | "--bull" | "--bear" | "--all-weather" | "--top" if reselect.is_some() => {
                fail(&format!("{} with --reselect-run of a composed run: the menu and the pools come from the run", flag))
            }
            "--gates" => gates = Some(value()),
            "--bull" => paths.push((Pool::Bull, value())),
            "--bear" => paths.push((Pool::Bear, value())),
            "--all-weather" => paths.push((Pool::AllWeather, value())),
            "--top" => top = Some(value()),
            "--out" => out = Some(value()),
            "--json" => {
                json = true;
                i += 1;
                continue;
            }
            "--fitness" => {
                if value() != "pnl" {
                    fail("the command prices the --fitness pnl books: --fitness must be pnl");
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
            "--gpu" => fail("--gpu: the command runs on the CPU and has no device arm"),
            "--pnl-mask" | "--pnl-mask-band" | "--pnl-mask-dwell" => {
                fail(&format!("{}: a composite prices the unmasked branches under its own gate; the masks belong to the pool searches", flag))
            }
            "--pnl-excess" => fail("--pnl-excess: a composite is priced on the absolute long/flat book"),
            "--pnl-stat" | "--pnl-cvar-q" => fail(&format!("{}: a composite is priced on the net of its long/flat book, not on a block statistic", flag)),
            "--threshold" | "--position" | "--book-target" | "--band" | "--band-units" | "--min-holding" | "--max-position" | "--cash-return" | "--funding" => fail(&format!(
                "{}: a composite is the unit long/flat book at threshold 0; each branch carries its run's threshold",
                flag
            )),
            _ => {
                rest.push(args[i].clone());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    let top = match top {
        None => 5,
        Some(k) => match k.trim().parse::<usize>() {
            Ok(k) if k >= 1 => k,
            _ => fail(&format!("--top must be an integer >= 1, got '{}'", k)),
        },
    };
    let req = match reselect {
        Some(path) => Request { json, out, ..from_composed(path).unwrap_or_else(|e| exit_with(e)) },
        None => {
            let gates = gates.unwrap_or_else(|| fail("--gates <menu.json> is required"));
            if !paths.iter().any(|(p, _)| *p == Pool::Bull) && !paths.iter().any(|(p, _)| *p == Pool::Bear) {
                fail("--bull <run.json> or --bear <run.json> is required");
            }
            let menu = Menu::load(&gates).unwrap_or_else(|e| exit_with(e));
            let mut members: Vec<Member> = Vec::new();
            let mut runs: Vec<PoolRun> = Vec::new();
            for (pool, path) in &paths {
                let run = load_run(*pool, path, top, runs.len(), &mut members).unwrap_or_else(|e| exit_with(e));
                runs.push(run);
            }
            if let Err(e) = check_runs_agree(&runs) {
                exit_with(e);
            }
            Request { menu, runs, members, top, json, out, reselect: None }
        }
    };
    if !rest.iter().any(|a| a == "--fitness") {
        rest.extend(["--fitness".to_string(), "pnl".to_string()]);
    }
    if !rest.iter().any(|a| a == "--select-rule") {
        rest.extend(["--select-rule".to_string(), "robust".to_string()]);
    }
    if !rest.iter().any(|a| a == "--cpu" || a == "--no-gpu") {
        rest.push("--cpu".to_string());
    }
    let _ = REQUEST.set(req);
    rest
}

/// Every input run shares the target, the target spec, the cost and the split of the first.
fn check_runs_agree(runs: &[PoolRun]) -> Result<(), Fail> {
    let Some(first) = runs.first() else { return Ok(()) };
    for r in &runs[1..] {
        let differ = |what: &str| usage(format!("{} '{}' and {} '{}' differ in {}", first.pool.flag(), first.path, r.pool.flag(), r.path, what));
        if r.target != first.target {
            return differ("target");
        }
        if r.target_sha != first.target_sha {
            return differ("target spec");
        }
        if r.per_side_bps != first.per_side_bps {
            return differ("objective.spec.per_side_bps (cost)");
        }
        for key in SPLIT_KEYS {
            if split_differs(&first.split, &r.split, key) {
                return differ(&format!("split.{}", key));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- composites

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    C0,
    C1On,
    C1Off,
    C2,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::C0 => "C0",
            Kind::C1On | Kind::C1Off => "C1",
            Kind::C2 => "C2",
        }
    }

    fn form(self) -> &'static str {
        match self {
            Kind::C0 => "f",
            Kind::C1On => "if(gate, f, cash)",
            Kind::C1Off => "if(gate, cash, f)",
            Kind::C2 => "if(gate, f_on, f_off)",
        }
    }
}

/// One composite of the menu.
#[derive(Clone, Debug)]
pub struct Composite {
    pub id: usize,
    pub kind: Kind,
    /// Index into the menu gates.
    pub gate: Option<usize>,
    /// The member held where the gate is on (the member of a C0).
    pub on: Option<usize>,
    /// The member held where the gate is off.
    pub off: Option<usize>,
    /// The exact expression.
    pub text: String,
}

/// Every composite in id order: C0 members, then per gate the bull C1, the bear C1 and the bull x bear C2.
pub fn enumerate(req: &Request) -> Vec<Composite> {
    let bull = req.members_of(Pool::Bull);
    let bear = req.members_of(Pool::Bear);
    let mut out: Vec<Composite> = Vec::new();
    let mut push = |kind: Kind, gate: Option<usize>, on: Option<usize>, off: Option<usize>| {
        let branch = |m: Option<usize>| m.map_or(FLAT.to_string(), |m| req.members[m].branch.clone());
        let text = match gate {
            None => branch(on),
            Some(g) => format!("if(greater({}, 0), {}, {})", req.menu.gates[g].gate.source, branch(on), branch(off)),
        };
        let id = out.len();
        out.push(Composite { id, kind, gate, on, off, text });
    };
    for m in req.members_of(Pool::AllWeather) {
        push(Kind::C0, None, Some(m), None);
    }
    for g in 0..req.menu.gates.len() {
        for &m in &bull {
            push(Kind::C1On, Some(g), Some(m), None);
        }
        for &m in &bear {
            push(Kind::C1Off, Some(g), None, Some(m));
        }
        for &a in &bull {
            for &b in &bear {
                push(Kind::C2, Some(g), Some(a), Some(b));
            }
        }
    }
    out
}

/// Branch values and gate states of every member and gate on one slice.
struct SliceVals {
    branch: Vec<Vec<f64>>,
    state: Vec<Vec<bool>>,
}

/// State groups of a slice: per group, and per training segment when there are several.
fn state_groups(rows: &TradingRows, starts: &[f64]) -> Vec<u64> {
    let mut ids: HashMap<(u64, usize), u64> = HashMap::new();
    rows.group
        .iter()
        .zip(&rows.time)
        .map(|(&g, &t)| {
            let seg = if starts.len() > 1 { starts.partition_point(|s| *s <= t) } else { 0 };
            let next = ids.len() as u64;
            *ids.entry((g, seg)).or_insert(next)
        })
        .collect()
}

/// Branch values and gate states on one slice. `states` holds the gate states of
/// the evaluation rule per menu gate; without it (the training rows) the kernel
/// runs over the slice rows, per group and training segment.
fn slice_vals(req: &Request, columns: &[String], values: &[Vec<f64>], rows: &TradingRows, starts: &[f64], states: Option<&[Vec<bool>]>) -> Result<SliceVals, String> {
    let n = rows.time.len();
    if values.len() != columns.len() || values.iter().any(|v| v.len() != n) {
        return Err(format!("the carried columns do not match the {} rows", n));
    }
    let parse = |src: &str| expr::parse(src, columns).map_err(|e| format!("cannot parse '{}': {}", src, e));
    let branch_trees: Vec<Node> = req.members.iter().map(|m| parse(&m.branch)).collect::<Result<_, _>>()?;
    let gate_trees: Vec<Node> = req.menu.gates.iter().map(|g| parse(&g.gate.source)).collect::<Result<_, _>>()?;
    let mut branch = vec![vec![0.0f64; n]; branch_trees.len()];
    let mut gate = vec![vec![0.0f64; n]; gate_trees.len()];
    let mut row = vec![0.0f64; columns.len()];
    for i in 0..n {
        for (k, c) in values.iter().enumerate() {
            row[k] = c[i];
        }
        for (m, t) in branch_trees.iter().enumerate() {
            branch[m][i] = expr::eval(t, &row);
        }
        for (g, t) in gate_trees.iter().enumerate() {
            gate[g][i] = expr::eval(t, &row);
        }
    }
    if let Some(states) = states {
        if states.len() != gate.len() || states.iter().any(|s| s.len() != n) {
            return Err(format!("the gate states do not match the {} rows", n));
        }
        return Ok(SliceVals { branch, state: states.to_vec() });
    }
    let groups = state_groups(rows, starts);
    let mut last: HashMap<u64, f64> = HashMap::new();
    for i in 0..n {
        if let Some(prev) = last.insert(groups[i], rows.time[i]) {
            if rows.time[i] < prev {
                return Err(format!("rows of group {} are not in time order at row {}", rows.group[i], i));
            }
        }
    }
    let state = req
        .menu
        .gates
        .iter()
        .zip(&gate)
        .map(|(g, v)| regime_gate::hysteresis_states(v, &groups, g.gate.band, g.gate.dwell))
        .collect();
    Ok(SliceVals { branch, state })
}

fn pred(c: &Composite, v: &SliceVals) -> Vec<f64> {
    let side = |m: Option<usize>, i: usize| m.map_or(FLAT_VALUE, |m| v.branch[m][i]);
    let n = v.branch.first().map_or(0, Vec::len);
    match c.gate {
        None => (0..n).map(|i| side(c.on, i)).collect(),
        Some(g) => (0..n).map(|i| if v.state[g][i] { side(c.on, i) } else { side(c.off, i) }).collect(),
    }
}

// ---------------------------------------------------------------- pricing

fn book_json(s: &Result<PathStats, String>) -> String {
    match s {
        Ok(s) => format!(
            "\"net\":{},\"sharpe\":{},\"exposure\":{},\"trades\":{},\"total_cost\":{},\"buy_and_hold_net\":{}",
            num(s.net_pnl),
            num(s.sharpe),
            num(s.exposure),
            s.trades,
            num(s.total_cost),
            num(s.buy_and_hold.net_pnl)
        ),
        Err(e) => format!("\"unusable\":{}", text(e)),
    }
}

// ---------------------------------------------------------------- the run

/// What the run.json carries besides the composites, from the search parser.
pub struct Context<'a> {
    pub command: &'a str,
    pub target: &'a str,
    pub target_sha: Option<&'a str>,
    pub target_spec_json: &'a str,
    /// The `split` object of this invocation.
    pub split_json: &'a str,
    pub objective: trading::Objective,
    /// The book of every composite: threshold 0.
    pub spec: TradingSpec,
    pub cost: crate::target::CostDecl,
    pub min_group_step: Option<f64>,
    pub horizon_seconds: f64,
    pub robust: RobustParams,
    /// `--select-blocks intervals`: the validation interval starts.
    pub cuts: Option<&'a [f64]>,
    pub train_segments: &'a [f64],
    pub validation_segments: &'a [f64],
    pub test_segments: &'a [f64],
    pub ops: &'a [expr::Op],
    /// `--vars`: the variables of the branch code.
    pub vars: &'a [String],
}

/// The rows of the three slices and the raw values of [`Request::columns`] on them.
pub struct Slices<'a> {
    pub train: &'a TradingSlice,
    pub train_values: &'a [Vec<f64>],
    pub validation: &'a TradingSlice,
    pub validation_values: &'a [Vec<f64>],
    pub test: &'a TradingSlice,
    pub test_values: &'a [Vec<f64>],
    /// #1214: the `rcsr-gate-state/1` states and warm-ups of every menu gate.
    pub gates: &'a [crate::data::SliceGateStates],
}

struct Scored {
    stats: Result<PathStats, String>,
    objective: f64,
    regime: Option<Result<Regime, String>>,
}


/// The composed run.json and the text summary.
pub fn compose(req: &Request, book: &FitnessBook, slices: &Slices, ctx: &Context) -> Result<(String, String), Fail> {
    check_against(req, ctx)?;
    let columns = req.columns();
    let code = CodeSpec::v1(ctx.ops, ctx.vars.len());
    let composites = enumerate(req);
    let complexity_model = crate::complexity::ComplexityModel::new();
    let mut complexity: Vec<f64> = Vec::with_capacity(composites.len());
    for c in &composites {
        let tree = expr::parse(&c.text, &columns).map_err(|e| (2, format!("composite {}: cannot parse '{}': {}", c.id, c.text, e)))?;
        complexity.push(complexity_model.complexity(&tree));
    }
    let (member_bits, bits) = match &req.reselect {
        Some(r) => (r.member_bits.clone(), r.bits.clone()),
        None => {
            let mut member_bits: Vec<(f64, bool)> = Vec::with_capacity(req.members.len());
            for m in &req.members {
                let run = &req.runs[m.run];
                let tree = expr::parse(&m.expression_exact, ctx.vars).map_err(|e| {
                    (2, format!("{} '{}' archive_rank {}: the branch code reads --vars: {}", run.pool.flag(), run.path, m.archive_rank, e))
                })?;
                member_bits.push(tree_bits(&code, &tree).map_err(|e| (2, format!("{} '{}' archive_rank {}: {}", run.pool.flag(), run.path, m.archive_rank, e)))?);
            }
            let mut bits: Vec<f64> = Vec::with_capacity(composites.len());
            for c in &composites {
                let b = |m: Option<usize>| occam::Branch::Tree(member_bits[m.expect("a branch")].0);
                let gate = c.gate.map(|g| occam::Gate::Menu(req.menu.gates[g].index));
                let shape = match c.kind {
                    Kind::C0 => occam::Composite::C0(b(c.on)),
                    Kind::C1On => occam::Composite::C1 { gate: gate.expect("a gate"), branch: b(c.on) },
                    Kind::C1Off => occam::Composite::C1 { gate: gate.expect("a gate"), branch: b(c.off) },
                    Kind::C2 => occam::Composite::C2 { gate: gate.expect("a gate"), on: b(c.on), off: b(c.off) },
                };
                bits.push(code.composite_bits(&shape).map_err(|e| (2, format!("composite {}: {}", c.id, e)))?);
            }
            (member_bits, bits)
        }
    };

    // Training rows: the pruning bound of each pair (a re-selection keeps the run's pruning).
    let train = slice_vals(req, &columns, slices.train_values, &slices.train.rows, ctx.train_segments, None).map_err(|e| (2, format!("training rows: {}", e)))?;
    let n_train = book.len();
    if slices.train.rows.time.len() != n_train {
        return usage(format!("the training book holds {} rows for {} training trading rows", n_train, slices.train.rows.time.len()));
    }
    let long = |v: &[f64]| -> Vec<bool> { v.iter().map(|x| x.is_finite() && *x > 0.0).collect() };
    let train_long: Vec<Vec<bool>> = train.branch.iter().map(|v| long(v)).collect();
    let flat = vec![false; n_train];
    let all = vec![true; n_train];
    let mut bound_gain: Vec<Option<f64>> = vec![None; composites.len()];
    match (&req.reselect, req.menu.switch_budget) {
        (Some(r), _) => bound_gain = r.bound_gain.clone(),
        (None, Some(k)) => {
            let train_net: Vec<f64> = train_long.iter().map(|a| switch::net_of(book, &switch::composite_totals(book, a, a, &all))).collect();
            let mut cache: HashMap<(Option<usize>, Option<usize>), f64> = HashMap::new();
            for c in composites.iter().filter(|c| c.gate.is_some()) {
                let key = (c.on, c.off);
                let gain = match cache.get(&key) {
                    Some(g) => *g,
                    None => {
                        let a = c.on.map_or(&flat, |m| &train_long[m]);
                        let b = c.off.map_or(&flat, |m| &train_long[m]);
                        let bound = switch::switch_bound(book, a, b, &slices.train.rows.time, k).map_err(|e| (2, format!("training rows: {}", e)))?;
                        let value = bound.value(k.min(bound.timestamps.saturating_sub(1)));
                        let reference = c.on.map_or(0.0, |m| train_net[m]).max(c.off.map_or(0.0, |m| train_net[m]));
                        cache.insert(key, value - reference);
                        value - reference
                    }
                };
                bound_gain[c.id] = Some(gain);
            }
        }
        (None, None) => {}
    }
    let pruned: Vec<bool> = match &req.reselect {
        Some(r) => r.pruned.clone(),
        None => bound_gain.iter().map(|g| g.is_some_and(|g| g <= 0.0)).collect(),
    };

    // Validation rows: every composite that is not pruned.
    let validation_states: Vec<Vec<bool>> = slices.gates.iter().map(|g| g.validation.clone().unwrap_or_default()).collect();
    let val = slice_vals(req, &columns, slices.validation_values, &slices.validation.rows, &[], Some(&validation_states)).map_err(|e| (2, format!("validation rows: {}", e)))?;
    let vspec = TradingSpec { segment_starts: ctx.validation_segments.to_vec(), ..ctx.spec.clone() };
    let mut scored: Vec<Option<Scored>> = Vec::with_capacity(composites.len());
    for c in &composites {
        if pruned[c.id] {
            scored.push(None);
            continue;
        }
        let p = pred(c, &val);
        let stats = trading::evaluate(&p, &slices.validation.ret, &slices.validation.rows, &vspec);
        let objective = stats.as_ref().map_or(f64::INFINITY, |s| ctx.objective.score(s, vspec.min_trades));
        let regime = match (c.gate, &stats) {
            (Some(g), Ok(s)) => Some(regime_gate::regime(&p, &val.state[g], slices.validation, &vspec, s)),
            _ => None,
        };
        scored.push(Some(Scored { stats, objective, regime }));
    }
    let ids: Vec<usize> = (0..composites.len()).filter(|&i| scored[i].is_some()).collect();
    let metas: Vec<RobustMeta> = ids
        .iter()
        .map(|&i| {
            let s = scored[i].as_ref().expect("scored");
            RobustMeta { archive_rank: i + 1, complexity: complexity[i], usable: s.stats.is_ok() && s.objective.is_finite(), objective: s.objective }
        })
        .collect();
    let picked = if metas.is_empty() {
        RobustSelection { params: ctx.robust, members: Vec::new(), chosen: None }
    } else {
        crate::selection::select_robust_specs_cuts(&metas, ctx.robust, ctx.cuts, &mut |_| vspec.clone(), &slices.validation.ret, &slices.validation.rows, &mut |k| {
            pred(&composites[ids[k]], &val)
        })
        .map_err(|e| (2, e))?
    };
    let regime_ok = |i: usize| -> bool {
        match scored[i].as_ref().and_then(|s| s.regime.as_ref()) {
            None => composites[i].gate.is_none(),
            Some(Ok(r)) => r.on.alpha >= 0.0 && r.off.alpha >= 0.0,
            Some(Err(_)) => false,
        }
    };
    let robust_eligible: Vec<bool> = picked.members.iter().map(|m| m.eligible).collect();
    let members: Vec<RobustMember> = picked
        .members
        .iter()
        .zip(&ids)
        .map(|(m, &i)| RobustMember { eligible: m.eligible && regime_ok(i), ..m.clone() })
        .collect();
    let chosen_k = if ctx.robust.blocks == 1 && ctx.robust.se_z == 0.0 {
        metas
            .iter()
            .enumerate()
            .filter(|(k, m)| m.usable && m.objective.is_finite() && regime_ok(ids[*k]))
            .min_by(|(i, a), (j, b)| a.objective.total_cmp(&b.objective).then(i.cmp(j)))
            .map(|(k, _)| k)
    } else {
        crate::selection::choose_robust(&members, ctx.robust.se_z)
    };
    let usable: Vec<usize> = ids.iter().zip(&metas).filter(|(_, m)| m.usable).map(|(&i, _)| i).collect();
    let best_by_objective = |pool: &dyn Fn(usize) -> bool| -> Option<usize> {
        usable
            .iter()
            .copied()
            .filter(|&i| pool(i))
            .min_by(|&a, &b| scored[a].as_ref().unwrap().objective.total_cmp(&scored[b].as_ref().unwrap().objective).then(a.cmp(&b)))
    };
    let best_c0 = best_by_objective(&|i| composites[i].kind == Kind::C0);
    let finalist = match chosen_k {
        Some(k) => ids[k],
        None => best_c0.or_else(|| best_by_objective(&|_| true)).ok_or((2, "no composite has a usable validation book".to_string()))?,
    };
    let verdict = if chosen_k.is_some() { "selected" } else { "no_signal" };

    // Rank: the finalist, the eligible by pooled alpha, the usable by objective, the unusable, the pruned.
    let member_of: HashMap<usize, usize> = ids.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let class = |i: usize| -> u8 {
        if i == finalist {
            0
        } else if member_of.get(&i).is_some_and(|&k| members[k].eligible) {
            1
        } else if usable.contains(&i) {
            2
        } else if !pruned[i] {
            3
        } else {
            4
        }
    };
    let mut order: Vec<usize> = (0..composites.len()).collect();
    order.sort_by(|&a, &b| {
        class(a).cmp(&class(b)).then_with(|| match class(a) {
            1 => members[member_of[&b]].pooled_alpha.total_cmp(&members[member_of[&a]].pooled_alpha),
            2 => scored[a].as_ref().unwrap().objective.total_cmp(&scored[b].as_ref().unwrap().objective),
            _ => std::cmp::Ordering::Equal,
        })
        .then(a.cmp(&b))
    });
    let mut rank = vec![0usize; composites.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r + 1;
    }

    // The test rows, read now that the finalist is fixed.
    let test_states: Vec<Vec<bool>> = slices.gates.iter().map(|g| g.test.clone()).collect();
    let test = slice_vals(req, &columns, slices.test_values, &slices.test.rows, &[], Some(&test_states)).map_err(|e| (2, format!("test rows: {}", e)))?;
    let tspec = TradingSpec { segment_starts: ctx.test_segments.to_vec(), ..ctx.spec.clone() };
    let c = &composites[finalist];
    let vpred = pred(c, &val);
    let tpred = pred(c, &test);
    let validation_stats = trading::evaluate(&vpred, &slices.validation.ret, &slices.validation.rows, &vspec);
    let test_stats = trading::evaluate(&tpred, &slices.test.ret, &slices.test.rows, &tspec);
    let test_regime = match (c.gate, &test_stats) {
        (Some(g), Ok(s)) => Some(regime_gate::regime(&tpred, &test.state[g], slices.test, &tspec, s)),
        _ => None,
    };
    let branch_book = |m: usize, v: &SliceVals, slice: &TradingSlice, spec: &TradingSpec| trading::evaluate(&v.branch[m], &slice.ret, &slice.rows, spec);

    // JSON.
    let gate_json = |g: &MenuGate| {
        format!(
            "{{\"name\":{},\"codeword\":{},\"index\":{},\"expr\":{},\"band\":{},\"dwell\":{},\"columns\":[{}],\"pure\":{}}}",
            text(&g.name),
            text(&g.codeword),
            g.index,
            text(&g.gate.source),
            num(g.gate.band),
            g.gate.dwell,
            g.gate.columns.iter().map(|c| text(c)).collect::<Vec<_>>().join(","),
            g.pure()
        )
    };
    let runs_json: Vec<String> = req
        .runs
        .iter()
        .map(|r| {
            format!(
                "{{\"pool\":{},\"path\":{},\"sha256\":{},\"threshold\":{},\"per_side_bps\":{},\"mask\":{},\"front_size\":{},\"candidates_evaluated\":{}}}",
                text(r.pool.name()),
                text(&r.path),
                text(&r.sha256),
                num(r.threshold),
                num(r.per_side_bps),
                r.mask.as_ref().map_or("null".to_string(), |g| format!("{{\"expr\":{},\"band\":{},\"dwell\":{}}}", text(&g.source), num(g.band), g.dwell)),
                r.front_size,
                r.candidates_evaluated.map_or("null".to_string(), |n| n.to_string())
            )
        })
        .collect();
    let members_json: Vec<String> = req
        .members
        .iter()
        .enumerate()
        .map(|(k, m)| {
            format!(
                "{{\"id\":{},\"pool\":{},\"run\":{},\"archive_rank\":{},\"expression_exact\":{},\"branch\":{},\"bits\":{},\"exact_codeword\":{}}}",
                k,
                text(req.runs[m.run].pool.name()),
                m.run,
                m.archive_rank,
                text(&m.expression_exact),
                text(&m.branch),
                num(member_bits[k].0),
                member_bits[k].1
            )
        })
        .collect();
    let opt = |v: Option<usize>| v.map_or("null".to_string(), |m| m.to_string());
    let composites_json: Vec<String> = composites
        .iter()
        .map(|c| {
            let validation = match &scored[c.id] {
                None => "null".to_string(),
                Some(s) => {
                    let robust = member_of.get(&c.id).map(|&k| &members[k]);
                    let (alpha_on, alpha_off) = match &s.regime {
                        Some(Ok(r)) => (num(r.on.alpha), num(r.off.alpha)),
                        _ => ("null".to_string(), "null".to_string()),
                    };
                    format!(
                        "{{\"usable\":{},{},\"objective\":{},\"pooled_alpha\":{},\"se\":{},\"robust_eligible\":{},\"alpha_on\":{},\"alpha_off\":{},\"eligible\":{}}}",
                        s.stats.is_ok() && s.objective.is_finite(),
                        book_json(&s.stats),
                        num(s.objective),
                        robust.map_or("null".to_string(), |m| num(m.pooled_alpha)),
                        robust.map_or("null".to_string(), |m| num(m.se)),
                        member_of.get(&c.id).is_some_and(|&k| robust_eligible[k]),
                        alpha_on,
                        alpha_off,
                        robust.is_some_and(|m| m.eligible)
                    )
                }
            };
            format!(
                "{{\"id\":{},\"rank\":{},\"type\":{},\"form\":{},\"gate\":{},\"codeword\":{},\"on\":{},\"off\":{},\"expression\":{},\"complexity\":{},\"bits\":{},\"pruned\":{},\"bound_gain\":{},\"validation\":{}}}",
                c.id,
                rank[c.id],
                text(c.kind.label()),
                text(c.kind.form()),
                c.gate.map_or("null".to_string(), |g| text(&req.menu.gates[g].name)),
                c.gate.map_or("null".to_string(), |g| text(&req.menu.gates[g].codeword)),
                opt(c.on),
                opt(c.off),
                text(&c.text),
                num(complexity[c.id]),
                num(bits[c.id]),
                pruned[c.id],
                bound_gain[c.id].map_or("null".to_string(), num),
                validation
            )
        })
        .collect();
    let seam_json = match c.gate {
        None => "null".to_string(),
        Some(g) => {
            let a = c.on.map_or(&flat, |m| &train_long[m]);
            let b = c.off.map_or(&flat, |m| &train_long[m]);
            let s = switch::seam(book, a, b, &train.state[g]);
            format!(
                "{{\"rows\":\"train\",\"net_on_branch\":{},\"net_off_branch\":{},\"masked_on\":{},\"masked_off\":{},\"pairs\":{},\"seam\":{},\"sum\":{},\"composite\":{}}}",
                num(s.net_a),
                num(s.net_b),
                num(s.masked_a),
                num(s.masked_b),
                s.pairs,
                num(s.seam),
                num(s.masked_a + s.masked_b + s.seam),
                num(s.composite)
            )
        }
    };
    let regime_or_null = |r: &Option<Result<Regime, String>>, start: Option<&regime_gate::WarmUp>| match r {
        None => "null".to_string(),
        Some(Ok(r)) => {
            let mut s = regime_gate::regime_json(r);
            if let Some(w) = start {
                s.pop();
                s.push_str(&format!(",\"start\":{}}}", w.json()));
            }
            s
        }
        Some(Err(e)) => format!("{{\"unusable\":{}}}", text(e)),
    };
    let branches_json: Vec<String> = [("on", c.on), ("off", c.off)]
        .iter()
        .filter_map(|(side, m)| m.map(|m| (side, m)))
        .filter(|_| c.gate.is_some())
        .map(|(side, m)| {
            format!(
                "{{\"side\":{},\"member\":{},\"validation\":{{{}}},\"test\":{{{}}}}}",
                text(side),
                m,
                book_json(&branch_book(m, &val, slices.validation, &vspec)),
                book_json(&branch_book(m, &test, slices.test, &tspec))
            )
        })
        .collect();
    let best_aw_json = match best_c0 {
        None => "null".to_string(),
        Some(i) => {
            let m = composites[i].on.expect("a C0 member");
            format!(
                "{{\"composite\":{},\"member\":{},\"validation\":{{{}}},\"test\":{{{}}}}}",
                i,
                m,
                book_json(&branch_book(m, &val, slices.validation, &vspec)),
                book_json(&branch_book(m, &test, slices.test, &tspec))
            )
        }
    };
    let n_bull = req.members_of(Pool::Bull).len();
    let n_bear = req.members_of(Pool::Bear).len();
    let n_aw = req.members_of(Pool::AllWeather).len();
    let pruned_count = pruned.iter().filter(|&&p| p).count();
    let eligible_count = members.iter().filter(|m| m.eligible).count();
    let chosen_gate = c.gate.map(|g| &req.menu.gates[g]);
    let compose_json = format!(
        "{{\"schema\":\"{}\",\"menu\":{{\"path\":{},\"sha256\":{},\"schema\":\"{}\",\"switch_budget\":{},\"gates\":[{}]}},\"code\":{},\
         \"top\":{},\"runs\":[{}],\"members\":[{}],\
         \"count\":{{\"gates\":{},\"bull\":{},\"bear\":{},\"all_weather\":{},\"formula\":\"gates x (bull x bear + bull + bear) + all_weather\",\"composites\":{}}},\
         \"candidates_evaluated\":{},\"pruned\":{},\"scored\":{},\"usable\":{},\"rows\":{{\"train\":{},\"validation\":{},\"test\":{}}},\
         \"selection\":{{\"rule\":\"robust\",\"blocks\":{},\"se_z\":{},\"block_len\":{},\"blocks_source\":{},\"regime_condition\":\"alpha_on >= 0 and alpha_off >= 0 on the validation rows (C1 and C2)\",\"eligible\":{},\"verdict\":\"{}\",\"chosen\":{}}},\
         \"gate\":{},\
         \"chosen\":{{\"id\":{},\"type\":{},\"form\":{},\"gate\":{},\"expression\":{},\"complexity\":{},\"bits\":{},\
         \"validation\":{{{},\"regime\":{}}},\"test\":{{{},\"regime\":{}}},\"train_seam\":{},\"branches\":[{}],\"best_all_weather\":{}}},\
         \"composites\":[{}]}}",
        SCHEMA,
        text(&req.menu.path),
        text(&req.menu.sha256),
        MENU_SCHEMA,
        req.menu.switch_budget.map_or("null".to_string(), |k| k.to_string()),
        req.menu.gates.iter().map(gate_json).collect::<Vec<_>>().join(","),
        req.reselect.as_ref().map_or_else(|| code.json(), |r| r.code.clone()),
        req.top,
        runs_json.join(","),
        members_json.join(","),
        req.menu.gates.len(),
        n_bull,
        n_bear,
        n_aw,
        composites.len(),
        composites.len(),
        pruned_count,
        ids.len(),
        usable.len(),
        slices.train.rows.time.len(),
        slices.validation.rows.time.len(),
        slices.test.rows.time.len(),
        ctx.robust.blocks,
        num(ctx.robust.se_z),
        ctx.robust.block_len.map_or("null".to_string(), |l| l.to_string()),
        text(if ctx.cuts.is_some() { "intervals" } else { "equal" }),
        eligible_count,
        verdict,
        chosen_k.map_or("null".to_string(), |k| ids[k].to_string()),
        chosen_gate.map_or("null".to_string(), gate_json),
        c.id,
        text(c.kind.label()),
        text(c.kind.form()),
        chosen_gate.map_or("null".to_string(), |g| text(&g.name)),
        text(&c.text),
        num(complexity[c.id]),
        num(bits[c.id]),
        book_json(&validation_stats),
        regime_or_null(&scored[c.id].as_ref().and_then(|s| s.regime.clone()), c.gate.and_then(|g| slices.gates[g].validation_start.as_ref())),
        book_json(&test_stats),
        regime_or_null(&test_regime, c.gate.map(|g| &slices.gates[g].test_start)),
        seam_json,
        branches_json.join(","),
        best_aw_json,
        composites_json.join(",")
    );

    // The objective block of a search run, with the composite as the finalist.
    let tree = expr::parse(&c.text, &columns).map_err(|e| (2, format!("finalist: {}", e)))?;
    let mut relabelled: Vec<RobustMember> = members
        .iter()
        .zip(&ids)
        .map(|(m, &i)| RobustMember { archive_rank: rank[i], ..m.clone() })
        .collect();
    relabelled.sort_by_key(|m| m.archive_rank);
    let rule = RobustSelection { params: ctx.robust, members: relabelled, chosen: chosen_k.map(|_| 0) };
    let mut selection_rule = rule.fields_json();
    if ctx.cuts.is_some() {
        selection_rule.push_str(",\"blocks_source\":\"intervals\"");
    }
    selection_rule.push_str(",\"composed\":true,\"regime_condition\":\"alpha_on >= 0 and alpha_off >= 0 on the validation rows\"");
    let report = trading::FinalistReport {
        objective: ctx.objective,
        spec: ctx.spec.clone(),
        min_group_step: ctx.min_group_step,
        horizon_seconds: ctx.horizon_seconds,
        cost: ctx.cost.clone(),
        constant: trading::is_constant(&tree),
        fixed_position: trading::fixed_position(&tree, &ctx.spec),
        fixed_position_realised: validation_stats.as_ref().ok().and_then(|s| trading::realised_position(s.exposure, &ctx.spec)),
        candidates: composites.len(),
        usable: usable.len(),
        edge_rejected: None,
        expression: c.text.clone(),
        expression_exact: c.text.clone(),
        archive_rank: 1,
        fit_metric: "none".to_string(),
        fit_error: f64::NAN,
        validation_error: f64::NAN,
        test_error: f64::NAN,
        validation_value: scored[c.id].as_ref().map_or(f64::INFINITY, |s| s.objective),
        validation: validation_stats.clone(),
        search_fitness: None,
        selection: "robust",
        selection_rule,
        reselected_from: req.reselect.as_ref().map(|r| crate::reselect::from_run_json(&r.path, &r.sha256)),
        book_target: None,
        test: test_stats.clone(),
        mask: None,
    };
    // Every composite, in rank order; a C1 or C2 carries the switch that defines its book.
    let pareto: Vec<String> = order
        .iter()
        .map(|&i| {
            let k = &composites[i];
            format!(
                "{{\"archive_rank\":{},\"composite_id\":{},\"type\":{},\"expression\":{},\"expression_exact\":{},\"complexity\":{},\"bits\":{},\"pruned\":{}{}}}",
                rank[i],
                i,
                text(k.kind.label()),
                text(&k.text),
                text(&k.text),
                num(complexity[i]),
                num(bits[i]),
                pruned[i],
                switch_of(req, k).map_or(String::new(), |s| format!(",\"switch\":{}", s.json()))
            )
        })
        .collect();
    let train_rows = Json::parse(ctx.split_json).and_then(|s| s.get("train_rows").and_then(Json::as_usize)).unwrap_or(0);
    let test_rows = Json::parse(ctx.split_json).and_then(|s| s.get("test_rows").and_then(Json::as_usize)).unwrap_or(0);
    let run_json = format!(
        "{{\"tool\":\"rcsr\",\"version\":{},\"command\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{},\"target_spec\":{},\
         \"generations_run\":0,\"search\":{{\"candidates_evaluated\":{}}},\"objective\":{},\"compose\":{},\"pareto\":[{}]}}",
        text(env!("CARGO_PKG_VERSION")),
        text(ctx.command),
        text(ctx.target),
        columns.iter().map(|c| text(c)).collect::<Vec<_>>().join(","),
        train_rows,
        test_rows,
        ctx.split_json,
        ctx.target_spec_json,
        composites.len(),
        finalist_with_switch(report.json("cpu", None), switch_of(req, c).as_ref()),
        compose_json,
        pareto.join(",")
    );

    let mut summary = format!(
        "rcsr compose: {} composites ({} gates x ({} bull x {} bear + {} + {}) + {} all-weather), {} pruned, {} usable on {} validation rows; verdict {}\n",
        composites.len(),
        req.menu.gates.len(),
        n_bull,
        n_bear,
        n_bull,
        n_bear,
        n_aw,
        pruned_count,
        usable.len(),
        slices.validation.rows.time.len(),
        verdict
    );
    summary.push_str(&format!(
        "finalist {} {}{} (bits {:.1}): {}\n",
        c.kind.label(),
        c.kind.form(),
        chosen_gate.map_or(String::new(), |g| format!(" gate {}", g.name)),
        bits[c.id],
        c.text
    ));
    let line = |name: &str, s: &Result<PathStats, String>, r: Option<&Result<Regime, String>>| -> String {
        let mut out = match s {
            Ok(s) => format!("{}: net {} sharpe {} exposure {} cost {}", name, num(s.net_pnl), num(s.sharpe), num(s.exposure), num(s.total_cost)),
            Err(e) => format!("{}: unusable ({})", name, e),
        };
        if let Some(Ok(r)) = r {
            out.push_str(&format!(
                "; alpha on {} off {}; {} switches ({} per year); switch cost {} of {}",
                num(r.on.alpha),
                num(r.off.alpha),
                r.switches.switches,
                num(r.switches.per_year),
                num(r.switch_cost),
                num(r.total_cost)
            ));
        }
        out.push('\n');
        out
    };
    summary.push_str(&line("validation", &validation_stats, scored[c.id].as_ref().and_then(|s| s.regime.as_ref())));
    summary.push_str(&line("test", &test_stats, test_regime.as_ref()));
    Ok((run_json, summary))
}

/// #1214: the switch of a C1 or C2 composite; `None` for a C0.
fn switch_of(req: &Request, c: &Composite) -> Option<regime_gate::Switch> {
    let g = c.gate?;
    let branch = |m: Option<usize>| m.map_or(FLAT.to_string(), |m| req.members[m].branch.clone());
    Some(regime_gate::Switch { gate: req.menu.gates[g].gate.clone(), on: branch(c.on), off: branch(c.off) })
}

/// #1214: the objective block with `finalist.switch` added.
fn finalist_with_switch(objective: String, sw: Option<&regime_gate::Switch>) -> String {
    let Some(sw) = sw else { return objective };
    let close = objective
        .find("\"validation_objective\":")
        .and_then(|at| objective[at..].find("},\"validation\":").map(|k| at + k))
        .expect("the objective block has a finalist followed by validation");
    let mut out = objective;
    out.insert_str(close, &format!(",\"switch\":{}", sw.json()));
    out
}

/// Bits of a branch tree under the #1184 code, and whether the tree is its own
/// codeword. A constant no tier holds is written at the code's precision (the
/// exact-label candidate of [`CodeSpec::candidates`]).
fn tree_bits(code: &CodeSpec, tree: &Node) -> Result<(f64, bool), String> {
    match code.description_bits(tree) {
        Ok(b) => Ok((b.total, true)),
        Err(e) => code.candidates(tree).into_iter().max_by_key(|c| c.tier).map(|c| (c.bits.total, false)).ok_or(e),
    }
}

/// The input runs share the target, the target spec, the cost and the split of this invocation.
fn check_against(req: &Request, ctx: &Context) -> Result<(), Fail> {
    let split = Json::parse(ctx.split_json).ok_or((2, "the split of this invocation is not JSON".to_string()))?;
    let per_side = ctx.spec.per_side_bps();
    for r in &req.runs {
        let differ = |what: String| usage(format!("{} '{}' differs from this invocation in {}", r.pool.flag(), r.path, what));
        if r.target.as_deref() != Some(ctx.target) {
            return differ(format!("target ({:?} vs '{}')", r.target, ctx.target));
        }
        if ctx.target_sha != Some(r.target_sha.as_str()) {
            return differ("target spec".to_string());
        }
        if r.per_side_bps != per_side {
            return differ(format!("cost ({} vs {} bp per side)", r.per_side_bps, per_side));
        }
        for key in SPLIT_KEYS {
            if split_differs(&r.split, &split, key) {
                return differ(format!("split.{}", key));
            }
        }
    }
    Ok(())
}

/// What the command takes from the search parser besides the rows.
pub struct Inputs<'a> {
    pub target: &'a str,
    pub target_sha: Option<&'a str>,
    pub target_spec_json: &'a str,
    pub split_json: &'a str,
    pub objective: trading::Objective,
    pub cost: Option<crate::target::CostDecl>,
    pub min_group_step: Option<f64>,
    pub horizon_seconds: f64,
    pub robust: RobustParams,
    pub cuts: Option<&'a [f64]>,
    pub interval_split: Option<&'a crate::intervals::IntervalSplit>,
    pub ops: &'a [expr::Op],
    pub vars: &'a [String],
    pub book_target: bool,
}

/// Composes the pools on the training, validation and test rows the search
/// parser prepared, and exits.
pub fn start(req: &Request, command: &str, prepared: &crate::data::Prepared, book: Option<&FitnessBook>, tspec: Option<&TradingSpec>, inputs: &Inputs) -> ! {
    if inputs.book_target {
        fail("--book-target: the composite prices the return column of the fit label");
    }
    if !inputs.objective.is_trading() {
        fail("the choice ranks the validation P&L book: add --objective pnl|sharpe|log_growth");
    }
    let (Some(book), Some(tspec), Some(split)) = (book, tspec, prepared.trading.as_ref()) else {
        fail("no training book");
    };
    let (Some(train), Some(validation)) = (split.train.as_ref(), split.validation.as_ref()) else {
        fail("the choice reads the validation rows: add --validation-frac <f> or validation intervals");
    };
    if tspec.position != trading::PositionMap::Threshold
        || tspec.tau.is_some()
        || tspec.max_position != 1.0
        || tspec.cash_return != 0.0
        || tspec.funding_col.is_some()
        || tspec.band.resolved > 0.0
        || tspec.band.multiple > 0.0
        || tspec.min_holding > 1
    {
        fail("a composite is the unit long/flat threshold book: a position map, tau, max position, cash return, funding, band or minimum hold is not supported");
    }
    let spec = TradingSpec { threshold: 0.0, ..tspec.clone() };
    let segments = |f: fn(&crate::intervals::IntervalSplit) -> Vec<f64>| inputs.interval_split.map_or_else(Vec::new, f);
    let (train_segments, validation_segments, test_segments) =
        (segments(|s| s.segment_starts()), segments(|s| s.validation_segment_starts()), segments(|s| s.test_segment_starts()));
    let ctx = Context {
        command,
        target: inputs.target,
        target_sha: inputs.target_sha,
        target_spec_json: inputs.target_spec_json,
        split_json: inputs.split_json,
        objective: inputs.objective,
        cost: inputs.cost.clone().unwrap_or_else(|| crate::target::CostDecl::new(None, spec.per_side_bps(), crate::target::CostSource::Flag)),
        spec,
        min_group_step: inputs.min_group_step,
        horizon_seconds: inputs.horizon_seconds,
        robust: inputs.robust,
        cuts: inputs.cuts,
        train_segments: &train_segments,
        validation_segments: &validation_segments,
        test_segments: &test_segments,
        ops: inputs.ops,
        vars: inputs.vars,
    };
    let slices = Slices {
        train,
        train_values: &split.train_values,
        validation,
        validation_values: &split.validation_values,
        test: &split.test,
        test_values: &split.test_values,
        gates: &split.gate_states,
    };
    run(req, book, &slices, &ctx)
}

/// Prints the summary (the run.json with `--json`), writes `--out`, exits 0; a
/// refusal exits 2.
pub fn run(req: &Request, book: &FitnessBook, slices: &Slices, ctx: &Context) -> ! {
    let (json, summary) = compose(req, book, slices, ctx).unwrap_or_else(|e| exit_with(e));
    if let Some(path) = &req.out {
        if let Err(e) = crate::atomic::write_text_atomic(path, &format!("{}\n", json)) {
            eprintln!("error: rcsr compose: cannot write --out '{}': {}", path, e);
            std::process::exit(3);
        }
    }
    if req.json {
        println!("{}", json);
    } else {
        print!("{}", summary);
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu(gates: &str) -> Result<Menu, String> {
        Menu::parse(&format!("{{\"schema\":\"rcsr-gate-menu/1\",\"gates\":[{}]}}", gates))
    }

    #[test]
    fn the_menu_fixes_codewords_and_refuses_bad_ones() {
        let m = menu("{\"name\":\"a\",\"codeword\":\"101\",\"expr\":\"g\",\"band\":0,\"dwell\":1},{\"name\":\"b\",\"codeword\":\"000\",\"expr\":\"h - 0.5\",\"band\":0.02,\"dwell\":3}").unwrap();
        assert_eq!(m.gates.len(), 2);
        assert_eq!((m.gates[0].index, m.gates[1].index), (5, 0));
        assert!(m.gates[0].pure() && !m.gates[1].pure());
        assert_eq!(m.switch_budget, None);
        for (bad, needle) in [
            ("{\"name\":\"a\",\"codeword\":\"10\",\"expr\":\"g\",\"band\":0,\"dwell\":1}", "3 binary digits"),
            ("{\"name\":\"a\",\"codeword\":\"102\",\"expr\":\"g\",\"band\":0,\"dwell\":1}", "3 binary digits"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"g\",\"band\":0,\"dwell\":1},{\"name\":\"b\",\"codeword\":\"001\",\"expr\":\"h\",\"band\":0,\"dwell\":1}", "share codeword"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"g\",\"band\":0,\"dwell\":1},{\"name\":\"a\",\"codeword\":\"010\",\"expr\":\"h\",\"band\":0,\"dwell\":1}", "twice"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"mean(g)\",\"band\":0,\"dwell\":1}", "aggregate"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"g\",\"band\":-1,\"dwell\":1}", "band"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"g\",\"band\":0,\"dwell\":0}", "dwell"),
            ("{\"name\":\"a\",\"codeword\":\"001\",\"expr\":\"g\",\"dwell\":1}", "'band'"),
        ] {
            let e = menu(bad).unwrap_err();
            assert!(e.contains(needle), "{bad}: {e}");
        }
        let nine: Vec<String> = (0..9).map(|k| format!("{{\"name\":\"g{k}\",\"codeword\":\"000\",\"expr\":\"g\",\"band\":0,\"dwell\":1}}")).collect();
        assert!(menu(&nine.join(",")).unwrap_err().contains("1 to 8"));
        assert!(Menu::parse("{\"schema\":\"other\",\"gates\":[]}").unwrap_err().contains("rcsr-gate-menu/1"));
    }
}
