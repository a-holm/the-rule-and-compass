//! #1177: the multi-metric candidate table of `rcsr trials --table`.
//!
//! One row per series: a `--wf` report, a `members[]` row with `--members all`,
//! or a CPCV path report. The rule (gates, Pareto metrics, rank metrics, pick
//! count, diversity cap) is one input whose SHA-256 is in the output. Rows pass
//! the gates; the Pareto front ranks the passers; the average rank orders them
//! inside a front; a walk in that order picks rows whose net series correlates
//! below the cap with every earlier pick. A picked row gets its deflation at
//! the trial count of the whole table. With `max_composites` (#1237, D8) the
//! walk skips a composite row once the picks hold that many; with `freeze`
//! every pick gets the sealed-read thresholds of [`crate::trials_freeze`]. With
//! `--family-table` (#1269) the sibling horizon tables add their trials to the
//! deflation and the SPA p gets a Bonferroni factor over the tables.
//!
//! The table reads development reports only. A report with a locked read, or
//! with a period or test window inside a locked range, exits 2.

use crate::devlog::Json;
use crate::inference;
use crate::regimes::{Calendar, UNDATED};
use crate::trials::{self, Options, Wf};
use crate::trials_freeze::{self, FreezeMap, FreezeRule};
use crate::walkforward::{jnum, jstr};

type Res<T> = Result<T, (i32, String)>;

pub const SCHEMA: &str = "rcsr-trials-table/1";
pub const RULE_SCHEMA: &str = "rcsr-trials-rule/1";
const DEFAULT_ALPHA: f64 = 0.05;
const DEFAULT_MULTIPLES: [f64; 2] = [1.0, 2.0];

/// The flags of the table mode; each needs `--table`.
pub const FLAGS: &[&str] = &[
    "--rule",
    "--gates",
    "--pareto",
    "--rank",
    "--pick",
    "--diversity-rho",
    "--min-front",
    "--cost-multiples",
    "--null-wf",
    "--prior-trials",
    "--alpha",
    "--markdown",
    "--locked-test",
    "--occam",
    "--freeze-map",
    "--family-table",
    "--baseline-wf",
];

/// Metrics every row has (null where the series is too short or flat).
const BASE_METRICS: &[&str] =
    &["sharpe", "sortino", "calmar", "max_drawdown", "exposure", "entry_times", "alpha_bh", "psr", "groups", "groups_pos"];

fn usage<T>(m: impl Into<String>) -> Res<T> {
    Err((2, m.into()))
}

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

fn num_at(j: &Json, path: &[&str]) -> Option<f64> {
    at(j, path).and_then(Json::as_f64)
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Gt,
    Ge,
    Lt,
    Le,
    Between,
}

/// One gate: `metric op lo`, or `lo <= metric <= hi` for `between`.
#[derive(Clone, Debug)]
pub struct Gate {
    pub metric: String,
    pub op: Op,
    pub lo: f64,
    pub hi: f64,
}

impl Gate {
    /// A non-finite value fails every gate.
    pub fn passes(&self, v: f64) -> bool {
        if !v.is_finite() {
            return false;
        }
        match self.op {
            Op::Gt => v > self.lo,
            Op::Ge => v >= self.lo,
            Op::Lt => v < self.lo,
            Op::Le => v <= self.lo,
            Op::Between => v >= self.lo && v <= self.hi,
        }
    }

    fn op_str(&self) -> &'static str {
        match self.op {
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Between => "between",
        }
    }

    fn spec_json(&self) -> String {
        match self.op {
            Op::Between => format!("[\"between\",{},{}]", jnum(self.lo), jnum(self.hi)),
            _ => format!("[{},{}]", jstr(self.op_str()), jnum(self.lo)),
        }
    }

    fn threshold_json(&self) -> String {
        match self.op {
            Op::Between => format!("[{},{}]", jnum(self.lo), jnum(self.hi)),
            _ => jnum(self.lo),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dir {
    Max,
    Min,
}

impl Dir {
    fn name(self) -> &'static str {
        match self {
            Dir::Max => "max",
            Dir::Min => "min",
        }
    }

    /// The value oriented so that higher is better.
    fn orient(self, v: f64) -> f64 {
        match self {
            Dir::Max => v,
            Dir::Min => -v,
        }
    }
}

/// The selection rule: one input, hashed into the output.
#[derive(Clone, Debug)]
pub struct Rule {
    pub cost_multiples: Vec<f64>,
    pub gates: Vec<Gate>,
    pub relaxed_gates: Option<Vec<Gate>>,
    pub pareto: Vec<(String, Dir)>,
    pub rank: Vec<(String, Dir)>,
    /// 0: no row is picked.
    pub pick: usize,
    pub diversity_rho: f64,
    /// 0: every front is kept.
    pub min_front: usize,
    /// #1237 (D8): at most this many composite rows are picked.
    pub max_composites: Option<usize>,
    /// #1237: the sealed-read thresholds frozen per pick.
    pub freeze: Option<FreezeRule>,
}

impl Default for Rule {
    fn default() -> Self {
        Rule {
            cost_multiples: DEFAULT_MULTIPLES.to_vec(),
            gates: Vec::new(),
            relaxed_gates: None,
            pareto: Vec::new(),
            rank: Vec::new(),
            pick: 0,
            diversity_rho: 1.0,
            min_front: 0,
            max_composites: None,
            freeze: None,
        }
    }
}

fn gates_json(g: &[Gate]) -> String {
    format!("{{{}}}", g.iter().map(|x| format!("{}:{}", jstr(&x.metric), x.spec_json())).collect::<Vec<_>>().join(","))
}

fn dirs_json(d: &[(String, Dir)]) -> String {
    format!("{{{}}}", d.iter().map(|(m, x)| format!("{}:{}", jstr(m), jstr(x.name()))).collect::<Vec<_>>().join(","))
}

impl Rule {
    /// The rule as canonical JSON; its SHA-256 is the rule hash of a flag-built rule.
    /// `max_composites` and `freeze` appear only when set, so a rule without them keeps its hash.
    pub fn json(&self) -> String {
        let mut s = self.base_json();
        if self.max_composites.is_some() || self.freeze.is_some() {
            s.pop();
            if let Some(m) = self.max_composites {
                s.push_str(&format!(",\"max_composites\":{}", m));
            }
            if let Some(f) = &self.freeze {
                s.push_str(&format!(",\"freeze\":{}", f.json()));
            }
            s.push('}');
        }
        s
    }

    fn base_json(&self) -> String {
        format!(
            "{{\"schema\":{},\"cost_multiples\":[{}],\"gates\":{},\"relaxed_gates\":{},\"pareto\":{},\"rank\":{},\"pick\":{},\
\"diversity_rho\":{},\"min_front\":{}}}",
            jstr(RULE_SCHEMA),
            self.cost_multiples.iter().map(|m| jnum(*m)).collect::<Vec<_>>().join(","),
            gates_json(&self.gates),
            self.relaxed_gates.as_ref().map_or("null".to_string(), |g| gates_json(g)),
            dirs_json(&self.pareto),
            dirs_json(&self.rank),
            self.pick,
            jnum(self.diversity_rho),
            self.min_front
        )
    }

    /// Every metric the rule reads, in rule order, each once.
    fn metrics(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let gates = self.gates.iter().chain(self.relaxed_gates.iter().flatten()).map(|g| &g.metric);
        for m in gates.chain(self.pareto.iter().map(|p| &p.0)).chain(self.rank.iter().map(|r| &r.0)) {
            if !out.contains(m) {
                out.push(m.clone());
            }
        }
        out
    }

    fn validate(&self, src: &str) -> Res<()> {
        if self.cost_multiples.is_empty() || self.cost_multiples.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
            return usage(format!("{}: cost_multiples must be positive numbers", src));
        }
        if !(self.diversity_rho > -1.0 && self.diversity_rho <= 1.0) {
            return usage(format!("{}: diversity_rho must be in (-1, 1]", src));
        }
        if self.relaxed_gates.is_some() && self.gates.is_empty() {
            return usage(format!("{}: relaxed_gates need gates", src));
        }
        Ok(())
    }
}

fn parse_gate(metric: &str, v: &Json, src: &str) -> Res<Gate> {
    let bad = || usage(format!("{}: gate '{}' must be [op, value] with op >, >=, <, <=, or [\"between\", lo, hi]", src, metric));
    let Json::Arr(a) = v else { return bad() };
    let op = match a.first().and_then(Json::as_str) {
        Some(">") => Op::Gt,
        Some(">=") => Op::Ge,
        Some("<") => Op::Lt,
        Some("<=") => Op::Le,
        Some("between") => Op::Between,
        _ => return bad(),
    };
    let n = |i: usize| a.get(i).and_then(Json::as_f64).filter(|x| x.is_finite());
    match (op, a.len(), n(1), n(2)) {
        (Op::Between, 3, Some(lo), Some(hi)) if lo <= hi => Ok(Gate { metric: metric.to_string(), op, lo, hi }),
        (Op::Between, ..) => bad(),
        (_, 2, Some(lo), _) => Ok(Gate { metric: metric.to_string(), op, lo, hi: lo }),
        _ => bad(),
    }
}

fn parse_gates(v: &Json, src: &str) -> Res<Vec<Gate>> {
    let Json::Obj(fields) = v else {
        return usage(format!("{}: gates must be an object {{metric: [op, value]}}", src));
    };
    fields.iter().map(|(k, g)| parse_gate(k, g, src)).collect()
}

/// `metric` or `metric:max` / `metric:min`.
fn parse_dir_item(s: &str, src: &str) -> Res<(String, Dir)> {
    let (m, d) = match s.rsplit_once(':') {
        Some((m, "max")) => (m, Dir::Max),
        Some((m, "min")) => (m, Dir::Min),
        _ => (s, Dir::Max),
    };
    if m.is_empty() {
        return usage(format!("{}: empty metric name in '{}'", src, s));
    }
    Ok((m.to_string(), d))
}

fn parse_dirs(v: &Json, src: &str) -> Res<Vec<(String, Dir)>> {
    let out: Vec<(String, Dir)> = match v {
        Json::Arr(items) => items
            .iter()
            .map(|x| x.as_str().map_or_else(|| usage(format!("{}: a metric list holds names", src)), |s| parse_dir_item(s, src)))
            .collect::<Res<_>>()?,
        Json::Obj(fields) => fields
            .iter()
            .map(|(k, d)| match d.as_str() {
                Some("max") => Ok((k.clone(), Dir::Max)),
                Some("min") => Ok((k.clone(), Dir::Min)),
                _ => usage(format!("{}: the direction of '{}' must be \"max\" or \"min\"", src, k)),
            })
            .collect::<Res<_>>()?,
        _ => return usage(format!("{}: a metric list is an array of names or an object {{metric: \"max\"|\"min\"}}", src)),
    };
    for (i, (m, _)) in out.iter().enumerate() {
        if out[..i].iter().any(|(x, _)| x == m) {
            return usage(format!("{}: metric '{}' is listed twice", src, m));
        }
    }
    Ok(out)
}

fn parse_list(s: &str, src: &str) -> Res<Vec<(String, Dir)>> {
    let items: Vec<Json> = s.split(',').filter(|x| !x.trim().is_empty()).map(|x| Json::Str(x.trim().to_string())).collect();
    parse_dirs(&Json::Arr(items), src)
}

fn whole(v: &Json, key: &str, src: &str) -> Res<usize> {
    match v.as_f64() {
        Some(x) if x >= 0.0 && x.fract() == 0.0 && x < 1e9 => Ok(x as usize),
        _ => usage(format!("{}: {} must be a whole number", src, key)),
    }
}

fn parse_multiples(v: &Json, src: &str) -> Res<Vec<f64>> {
    let Json::Arr(a) = v else { return usage(format!("{}: cost_multiples must be an array of numbers", src)) };
    a.iter().map(|x| x.as_f64().map_or_else(|| usage(format!("{}: cost_multiples must be an array of numbers", src)), Ok)).collect()
}

/// The rule of a `--rule` file and the SHA-256 of its bytes.
fn rule_from_file(path: &str) -> Res<(Rule, String)> {
    let bytes = std::fs::read(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
    let src = format!("--rule {}", path);
    let text = String::from_utf8_lossy(&bytes);
    let Some(Json::Obj(fields)) = Json::parse(text.trim()) else {
        return usage(format!("{}: not a JSON object", src));
    };
    let mut r = Rule::default();
    for (k, v) in &fields {
        match k.as_str() {
            "schema" => {
                if v.as_str() != Some(RULE_SCHEMA) {
                    return usage(format!("{}: schema must be {}", src, RULE_SCHEMA));
                }
            }
            "cost_multiples" => r.cost_multiples = parse_multiples(v, &src)?,
            "gates" => r.gates = parse_gates(v, &src)?,
            "relaxed_gates" => r.relaxed_gates = Some(parse_gates(v, &src)?),
            "pareto" => r.pareto = parse_dirs(v, &src)?,
            "rank" => r.rank = parse_dirs(v, &src)?,
            "pick" => r.pick = whole(v, "pick", &src)?,
            "min_front" => r.min_front = whole(v, "min_front", &src)?,
            "diversity_rho" => r.diversity_rho = v.as_f64().map_or_else(|| usage(format!("{}: diversity_rho must be a number", src)), Ok)?,
            "max_composites" => r.max_composites = Some(whole(v, "max_composites", &src)?),
            "freeze" => r.freeze = Some(FreezeRule::parse(v, &src)?),
            other => return usage(format!("{}: unknown key '{}'", src, other)),
        }
    }
    r.validate(&src)?;
    Ok((r, crate::manifest::sha256_hex(&bytes)))
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TableArgs {
    rule: Option<String>,
    gates: Option<String>,
    pareto: Option<String>,
    rank: Option<String>,
    pick: Option<usize>,
    diversity_rho: Option<f64>,
    min_front: Option<usize>,
    cost_multiples: Option<String>,
    null_wf: Vec<String>,
    prior_trials: usize,
    alpha: Option<f64>,
    markdown: Option<String>,
    locked: Vec<(f64, f64)>,
    occam: Vec<String>,
    freeze_map: Option<String>,
    family: Vec<String>,
    baseline_wf: Vec<String>,
}

fn value(args: &[String], i: &mut usize, flag: &str) -> Res<String> {
    *i += 1;
    args.get(*i).cloned().map_or_else(|| usage(format!("{} needs a value", flag)), Ok)
}

fn number<T: std::str::FromStr>(flag: &str, v: &str) -> Res<T> {
    v.parse().map_or_else(|_| usage(format!("{} needs a number, got '{}'", flag, v)), Ok)
}

fn files(args: &[String], i: &mut usize, flag: &str) -> Res<Vec<String>> {
    let mut out = Vec::new();
    while args.get(*i + 1).is_some_and(|n| !n.starts_with("--")) {
        *i += 1;
        out.push(args[*i].clone());
    }
    if out.is_empty() {
        return usage(format!("{} needs at least one file", flag));
    }
    Ok(out)
}

/// The table flags, and every other argument for the campaign parser.
fn split_args(args: &[String]) -> Res<(TableArgs, Vec<String>)> {
    let mut t = TableArgs::default();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--table" => {}
            "--rule" => t.rule = Some(value(args, &mut i, a)?),
            "--gates" => t.gates = Some(value(args, &mut i, a)?),
            "--pareto" => t.pareto = Some(value(args, &mut i, a)?),
            "--rank" => t.rank = Some(value(args, &mut i, a)?),
            "--pick" => t.pick = Some(number(a, &value(args, &mut i, a)?)?),
            "--diversity-rho" => t.diversity_rho = Some(number(a, &value(args, &mut i, a)?)?),
            "--min-front" => t.min_front = Some(number(a, &value(args, &mut i, a)?)?),
            "--cost-multiples" => t.cost_multiples = Some(value(args, &mut i, a)?),
            "--null-wf" => t.null_wf = files(args, &mut i, a)?,
            "--prior-trials" => t.prior_trials = number(a, &value(args, &mut i, a)?)?,
            "--alpha" => t.alpha = Some(number(a, &value(args, &mut i, a)?)?),
            "--markdown" => t.markdown = Some(value(args, &mut i, a)?),
            "--occam" => t.occam = files(args, &mut i, a)?,
            "--freeze-map" => t.freeze_map = Some(value(args, &mut i, a)?),
            "--family-table" => t.family.extend(files(args, &mut i, a)?),
            "--baseline-wf" => t.baseline_wf.extend(files(args, &mut i, a)?),
            "--locked-test" => {
                let v = value(args, &mut i, a)?;
                let parsed = v.split_once(',').and_then(|(s, e)| Some((crate::data::parse_time(s)?, crate::data::parse_time(e)?)));
                match parsed {
                    Some((s, e)) if s.is_finite() && e.is_finite() && e > s => t.locked.push((s, e)),
                    _ => return usage(format!("--locked-test needs START,END with END after START, got '{}'", v)),
                }
            }
            _ => rest.push(args[i].clone()),
        }
        i += 1;
    }
    if t.alpha.is_some_and(|x| !(x > 0.0 && x < 1.0)) {
        return usage("--alpha must be in (0, 1)");
    }
    if !t.baseline_wf.is_empty() && rest.iter().any(|a| a == "--baseline") {
        return usage("--baseline-wf excludes --baseline");
    }
    Ok((t, rest))
}

/// The rule, its SHA-256 and where it came from.
fn build_rule(t: &TableArgs) -> Res<(Rule, String, String)> {
    let flags = [
        ("--gates", t.gates.is_some()),
        ("--pareto", t.pareto.is_some()),
        ("--rank", t.rank.is_some()),
        ("--pick", t.pick.is_some()),
        ("--diversity-rho", t.diversity_rho.is_some()),
        ("--min-front", t.min_front.is_some()),
        ("--cost-multiples", t.cost_multiples.is_some()),
    ];
    if let Some(path) = &t.rule {
        if let Some((f, _)) = flags.iter().find(|(_, given)| *given) {
            return usage(format!("--rule holds the whole rule; {} cannot be combined with it", f));
        }
        let (r, sha) = rule_from_file(path)?;
        return Ok((r, sha, path.clone()));
    }
    let mut r = Rule::default();
    if let Some(path) = &t.gates {
        let text = std::fs::read_to_string(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
        let src = format!("--gates {}", path);
        let j = Json::parse(text.trim()).map_or_else(|| usage(format!("{}: not a JSON object", src)), Ok)?;
        let g = j.get("gates").unwrap_or(&j);
        r.gates = parse_gates(g, &src)?;
    }
    if let Some(s) = &t.pareto {
        r.pareto = parse_list(s, "--pareto")?;
    }
    if let Some(s) = &t.rank {
        r.rank = parse_list(s, "--rank")?;
    }
    r.pick = t.pick.unwrap_or(0);
    r.diversity_rho = t.diversity_rho.unwrap_or(1.0);
    r.min_front = t.min_front.unwrap_or(0);
    if let Some(s) = &t.cost_multiples {
        r.cost_multiples = s.split(',').map(|x| number("--cost-multiples", x.trim())).collect::<Res<_>>()?;
    }
    r.validate("the rule")?;
    let sha = crate::manifest::sha256_hex(r.json().as_bytes());
    Ok((r, sha, "flags".to_string()))
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// One series and the fields of its report the table reads beyond the campaign.
struct Entry {
    wf: Wf,
    per_side_bps: Option<f64>,
    cost: Result<Vec<f64>, String>,
    bh_net: Result<Vec<f64>, String>,
    entry_times: Result<f64, String>,
    groups: Result<Vec<(f64, f64)>, String>,
    trades: Option<f64>,
    complexity: Option<f64>,
    expression: Option<String>,
    archive_rank: Option<f64>,
    run_sha256: Option<String>,
    /// #1253: a `--from-run` report (a run, no `selection.source`): its book is the run's finalist.
    from_run: bool,
    /// #1237 (D19): the compose marker of the report, `composite` of a CPCV path or
    /// `selection.composite` of a walk-forward report.
    composite: Option<Json>,
}

/// Why a report reads locked data, or `None` for a development report. The
/// reason names counts and range bounds, never a time inside a range.
fn locked_reason(j: &Json, extra: &[(f64, f64)]) -> Option<String> {
    let prot = j.get("protocol");
    let pl = prot.and_then(|p| p.get("lockbox"));
    let truth = |v: Option<&Json>| v == Some(&Json::Bool(true));
    if truth(prot.and_then(|p| p.get("test_read"))) {
        return Some("protocol.test_read is true (a locked read)".to_string());
    }
    if truth(pl.and_then(|l| l.get("evaluated"))) {
        return Some("protocol.lockbox.evaluated is true".to_string());
    }
    if let Some(r) = pl.and_then(|l| l.get("rows")).and_then(Json::as_f64).filter(|r| *r > 0.0) {
        return Some(format!("protocol.lockbox.rows is {}", jnum(r)));
    }
    if pl.and_then(|l| l.get("read_marker")).is_some() {
        return Some("protocol.lockbox.read_marker is set".to_string());
    }
    let lb = j.get("lockbox");
    if truth(lb.and_then(|l| l.get("evaluated"))) {
        if let Some(Json::Arr(ws)) = lb.and_then(|l| l.get("windows")) {
            if ws.iter().any(|w| truth(w.get("test_overlap"))) {
                return Some("lockbox.evaluated is true and a test window overlaps the lockbox".to_string());
            }
        }
    }
    let mut ranges: Vec<(f64, f64, &str)> = Vec::new();
    let mut add = |v: Option<&Json>, name: &'static str| {
        if let (Some(s), Some(e)) = (v.and_then(|x| num_at(x, &["start"])), v.and_then(|x| num_at(x, &["end"]))) {
            if e > s {
                ranges.push((s, e, name));
            }
        }
    };
    add(pl, "protocol.lockbox");
    add(pl.and_then(|l| l.get("declared")), "protocol.lockbox.declared");
    add(pl.and_then(|l| l.get("also")), "protocol.lockbox.also");
    add(lb, "lockbox");
    ranges.extend(extra.iter().map(|&(s, e)| (s, e, "--locked-test")));
    let mut times: Vec<&Json> = Vec::new();
    if let Some(t) = at(j, &["pooled", "period_time"]) {
        times.push(t);
    }
    if let Some(Json::Arr(ms)) = j.get("members") {
        times.extend(ms.iter().filter_map(|m| at(m, &["series", "period_time"])));
    }
    let mut windows: Vec<(f64, f64)> = Vec::new();
    if let Some(Json::Arr(ws)) = j.get("window_definition") {
        windows.extend(ws.iter().filter_map(|w| Some((num_at(w, &["test_start"])?, num_at(w, &["test_end"])?))));
    }
    if let Some(Json::Arr(ss)) = at(j, &["cpcv", "segments"]) {
        windows.extend(ss.iter().filter_map(|w| Some((num_at(w, &["start"])?, num_at(w, &["end"])?))));
    }
    for (s, e, name) in ranges {
        let inside: usize = times
            .iter()
            .map(|t| match t {
                Json::Arr(v) => v.iter().filter_map(Json::as_f64).filter(|&x| x >= s && x < e).count(),
                _ => 0,
            })
            .sum();
        if inside > 0 {
            return Some(format!("{} periods lie inside the locked range {}..{} ({})", inside, jnum(s), jnum(e), name));
        }
        if windows.iter().any(|&(a, b)| a < e && b > s) {
            return Some(format!("a test window overlaps the locked range {}..{} ({})", jnum(s), jnum(e), name));
        }
    }
    None
}

fn groups_of(j: &Json, report: &str) -> Result<Vec<(f64, f64)>, String> {
    let pair = |x: &Json, a: &str, b: &str| Some((num_at(x, &[a])?, num_at(x, &[b])?));
    if let Some(Json::Arr(segs)) = at(j, &["cpcv", "segments"]) {
        return segs
            .iter()
            .map(|s| pair(s, "start", "end").ok_or_else(|| format!("field 'cpcv.segments' has a group without start and end in {}", report)))
            .collect();
    }
    match j.get("window_definition") {
        Some(Json::Arr(ws)) => ws
            .iter()
            .map(|w| {
                pair(w, "test_start", "test_end")
                    .ok_or_else(|| format!("field 'window_definition' has a window without test_start and test_end in {}", report))
            })
            .collect(),
        _ => Err(format!("field 'window_definition' (test windows) or 'cpcv.segments' is missing in {}", report)),
    }
}

const SERIES_HINT: &str = "write the report with walkforward --emit-series, or read a walkforward --all-members report with --members all";

fn entry(j: &Json, root: &Json, prefix: &'static str, path: String, label: String) -> Res<Entry> {
    let wf = trials::wf_entry(j, root, prefix, &path, label)?;
    let n = wf.net.len();
    let optional = |key: &str| -> Res<Option<Vec<f64>>> {
        if at(root, &[prefix, key]).is_none() {
            return Ok(None);
        }
        let v = trials::series(root, &[prefix, key], &path)?;
        if v.len() != n {
            return usage(format!("field '{}.{}' has {} values but '{}.period_net' has {} in {}", prefix, key, v.len(), prefix, n, path));
        }
        Ok(Some(v))
    };
    let cost = match (optional("period_cost")?, &wf.gross) {
        (Some(c), _) => Ok(c),
        (None, Some(g)) => Ok(g.iter().zip(&wf.net).map(|(g, x)| g - x).collect()),
        (None, None) => Err(format!("field '{}.period_cost' (or '{}.period_gross') is missing in {} ({})", prefix, prefix, path, SERIES_HINT)),
    };
    let bh_net = optional("period_bh_net")?.ok_or_else(|| format!("field '{}.period_bh_net' is missing in {} ({})", prefix, path, SERIES_HINT));
    let entry_times = num_at(root, &["pooled", "entry_times"]).ok_or_else(|| format!("field 'pooled.entry_times' is missing in {} ({})", path, SERIES_HINT));
    Ok(Entry {
        per_side_bps: num_at(j, &["cost_model", "per_side_bps"]),
        cost,
        bh_net,
        entry_times,
        groups: groups_of(j, &path),
        trades: num_at(root, &["pooled", "trades"]),
        complexity: num_at(root, &["complexity"]),
        expression: root.get("expression").and_then(Json::as_str).map(str::to_string),
        archive_rank: num_at(root, &["archive_rank"]),
        run_sha256: [&["selection", "run_sha256"][..], &["cpcv", "run_sha256"], &["run_sha256"]].iter().find_map(|k| at(j, k).and_then(Json::as_str)).map(str::to_string),
        from_run: at(j, &["selection", "run_sha256"]).is_some() && at(j, &["selection", "source"]).is_none(),
        composite: j.get("composite").or_else(|| at(j, &["selection", "composite"])).cloned(),
        wf,
    })
}

/// Exit 2 for a report `path` that reads locked data.
pub(crate) fn refuse_locked(j: &Json, path: &str, locked: &[(f64, f64)]) -> Res<()> {
    match locked_reason(j, locked) {
        Some(why) => usage(format!("{} reads locked data: {}; the table reads development reports only", path, why)),
        None => Ok(()),
    }
}

/// #1272: exit 2 for a report priced on a null store (`null_store`, written by `rcsr panel permute-labels`).
pub(crate) fn refuse_null_store(j: &Json, path: &str) -> Res<()> {
    match j.get("null_store") {
        Some(m) => usage(format!(
            "{} is priced on a null store (null_store label {}, seed {}); a null store is never a candidate: pass it with --null-wf",
            path,
            m.get("label").and_then(Json::as_str).unwrap_or("?"),
            m.get("seed").and_then(Json::as_f64).map_or("?".to_string(), jnum)
        )),
        None => Ok(()),
    }
}

/// A development report of real data: [`refuse_locked`] and [`refuse_null_store`].
fn read_checked(path: &str, locked: &[(f64, f64)]) -> Res<Json> {
    let j = read_null(path, locked)?;
    refuse_null_store(&j, path)?;
    Ok(j)
}

/// A `--null-wf` report: a development report, of real data or of a null store.
fn read_null(path: &str, locked: &[(f64, f64)]) -> Res<Json> {
    let j = trials::read_report(path)?;
    refuse_locked(&j, path, locked)?;
    Ok(j)
}

fn load_entries(o: &Options, locked: &[(f64, f64)]) -> Res<Vec<Entry>> {
    let mut out: Vec<Entry> = Vec::new();
    for p in &o.wf {
        let j = read_checked(p, locked)?;
        let mut found = Vec::new();
        if o.all_members {
            let Some(Json::Arr(members)) = j.get("members") else {
                return usage(format!(
                    "field 'members' is missing in {} (write the report with walkforward --front --all-members, or read it with --members selected)",
                    p
                ));
            };
            for (k, m) in members.iter().enumerate() {
                let rank = m.get("archive_rank").and_then(Json::as_f64).map_or_else(|| (k + 1).to_string(), jnum);
                found.push(entry(&j, m, "series", format!("{}#{}", p, rank), format!("{}#{}", trials::label_of(p), rank))?);
            }
        } else {
            found.push(entry(&j, &j, "pooled", p.clone(), trials::label_of(p))?);
        }
        for e in found {
            if let Some(first) = out.first() {
                trials::require_same(&first.wf, &e.wf, o.threshold_arms)?;
            }
            out.push(e);
        }
    }
    Ok(out)
}

/// #1271: the `--baseline-wf` benchmark: the best rung by mean, and the `inputs.baseline` object.
struct WfBaseline {
    best: (String, Vec<f64>),
    json: String,
}

/// Rungs: `cash` (zeros), `buy_and_hold` (`pooled.period_bh_net` of the first report) and one per report, named by
/// its `label` or file stem (`<directory>/<name>` for each report whose name agrees with another). Reports share the table's period grid and
/// per-side cost; policy, band and hold may differ.
fn load_baseline_wf(paths: &[String], time: &[f64], bps: Option<f64>, locked: &[(f64, f64)]) -> Res<WfBaseline> {
    struct Rung {
        path: String,
        name: String,
        sha256: String,
        net: Vec<f64>,
    }
    let mut rungs: Vec<Rung> = Vec::new();
    let mut bh: Option<Vec<f64>> = None;
    for p in paths {
        let bytes = std::fs::read(p).map_err(|e| (3, format!("{}: {}", p, e)))?;
        let j = read_checked(p, locked)?;
        let grid = trials::series(&j, &["pooled", "period_time"], p)?;
        if grid != time {
            return usage(format!(
                "--baseline-wf {} differs in field 'pooled.period_time' ({} periods) from the table grid ({} periods)",
                p,
                grid.len(),
                time.len()
            ));
        }
        let got = num_at(&j, &["cost_model", "per_side_bps"]);
        if got != bps {
            let show = |v: Option<f64>| v.map_or("missing".to_string(), jnum);
            return usage(format!(
                "--baseline-wf {} differs in field 'cost_model.per_side_bps': {} vs {} in the table reports",
                p,
                show(got),
                show(bps)
            ));
        }
        let net = trials::series(&j, &["pooled", "period_net"], p)?;
        if net.len() != time.len() {
            return usage(format!("field 'pooled.period_net' has {} values but 'pooled.period_time' has {} in {}", net.len(), time.len(), p));
        }
        if bh.is_none() {
            let v = trials::series(&j, &["pooled", "period_bh_net"], p)?;
            if v.len() != time.len() {
                return usage(format!("field 'pooled.period_bh_net' has {} values but 'pooled.period_time' has {} in {}", v.len(), time.len(), p));
            }
            bh = Some(v);
        }
        let name = j.get("label").and_then(Json::as_str).map_or_else(|| trials::label_of(p), str::to_string);
        rungs.push(Rung { path: p.clone(), name, sha256: crate::manifest::sha256_hex(&bytes), net });
    }
    let stems: Vec<String> = rungs.iter().map(|r| r.name.clone()).collect();
    for r in rungs.iter_mut().filter(|r| stems.iter().filter(|s| **s == r.name).count() > 1) {
        let dir = std::path::Path::new(&r.path).parent().and_then(|d| d.file_name()).map(|d| d.to_string_lossy().into_owned());
        if let Some(d) = dir {
            r.name = format!("{}/{}", d, r.name);
        }
    }
    let mut all: Vec<(String, Option<String>, Vec<f64>)> = vec![
        ("cash".to_string(), None, vec![0.0; time.len()]),
        ("buy_and_hold".to_string(), rungs.first().map(|r| r.sha256.clone()), bh.unwrap_or_default()),
    ];
    for r in &rungs {
        if let Some(prev) = all.iter().position(|(n, _, _)| *n == r.name) {
            return usage(format!(
                "--baseline-wf {}: the rung name '{}' is taken{}; give the report a distinct label or file name",
                r.path,
                r.name,
                if prev < 2 { " by a built-in rung" } else { " by another report" }
            ));
        }
        all.push((r.name.clone(), Some(r.sha256.clone()), r.net.clone()));
    }
    let mut best = 0;
    for k in 1..all.len() {
        if inference::mean(&all[k].2) > inference::mean(&all[best].2) {
            best = k;
        }
    }
    let rung_json: Vec<String> = all
        .iter()
        .map(|(n, s, _)| format!("{{\"name\":{},\"sha256\":{}}}", jstr(n), s.as_ref().map_or("null".to_string(), |s| jstr(s))))
        .collect();
    let json = format!("{{\"source\":\"wf\",\"rungs\":[{}],\"best\":{}}}", rung_json.join(","), jstr(&all[best].0));
    let (name, _, net) = all.swap_remove(best);
    Ok(WfBaseline { best: (name, net), json })
}

/// `--occam`: one re-selected or composed run.json.
struct OccamRun {
    path: String,
    sha256: String,
    from: Option<String>,
    pareto: Vec<Json>,
    /// `objective.finalist.archive_rank`: the member a `--from-run` report prices.
    finalist_rank: Option<f64>,
}

fn load_occam(paths: &[String]) -> Res<Vec<OccamRun>> {
    paths
        .iter()
        .map(|p| -> Res<OccamRun> {
            let bytes = std::fs::read(p).map_err(|e| (3, format!("{}: {}", p, e)))?;
            let text = String::from_utf8_lossy(&bytes);
            let j = Json::parse(text.trim()).ok_or_else(|| (3, format!("{}: not a JSON run", p)))?;
            let Some(Json::Arr(pareto)) = j.get("pareto") else {
                return usage(format!("field 'pareto' is missing in {} (--occam reads a run.json)", p));
            };
            Ok(OccamRun {
                path: p.clone(),
                sha256: crate::manifest::sha256_hex(&bytes),
                from: at(&j, &["objective", "selection_rule", "reselected_from", "sha256"]).and_then(Json::as_str).map(str::to_string),
                pareto: pareto.clone(),
                finalist_rank: at(&j, &["objective", "finalist", "archive_rank"]).and_then(Json::as_f64),
            })
        })
        .collect()
}

/// The `occam` cell of a row: the run whose sha256 (or `reselected_from.sha256`) is the report's
/// run sha256, its pareto member of the row's archive rank (else of the row's expression, a `--from-run` report: the run finalist first, its expression only when the run has no matching finalist).
fn occam_cell(e: &Entry, runs: &[OccamRun]) -> String {
    let Some(sha) = &e.run_sha256 else { return "null".to_string() };
    for run in runs.iter().filter(|r| &r.sha256 == sha || r.from.as_ref() == Some(sha)) {
        // A report's archive rank is the 1-based position in the run's pareto array (`FrontMember::rank`); the
        // run's own `archive_rank` field names the finalist of `objective.finalist`.
        let by_position = |rank: f64| run.pareto.get((rank as usize).wrapping_sub(1)).filter(|_| rank >= 1.0 && rank.fract() == 0.0);
        let by_field = |rank: f64| run.pareto.iter().find(|m| num_at(m, &["archive_rank"]) == Some(rank));
        let by_expression = || run.pareto.iter().find(|m| m.get("expression").and_then(Json::as_str) == e.expression.as_deref());
        let member = match e.archive_rank {
            Some(rank) => by_position(rank),
            None if e.from_run => run.finalist_rank.and_then(by_field).or_else(by_expression),
            None => by_expression(),
        };
        let Some(m) = member else { continue };
        let (bits, p, budget, source) = match m.get("occam") {
            Some(oc) => (num_at(oc, &["bits"]), num_at(oc, &["p_occam"]), num_at(oc, &["bit_budget"]), "occam"),
            None => (num_at(m, &["bits"]), None, None, "compose"),
        };
        let opt = |v: Option<f64>| v.map_or("null".to_string(), jnum);
        return format!(
            "{{\"run\":{},\"source\":{},\"bits\":{},\"p_occam\":{},\"bit_budget\":{}}}",
            jstr(&run.path),
            jstr(source),
            opt(bits),
            opt(p),
            opt(budget)
        );
    }
    "null".to_string()
}

/// #1269: one `--family-table`, the table of a sibling horizon over the same development span.
struct Sibling {
    path: String,
    sha256: String,
    n_eff: f64,
    nominal: f64,
    spa_p: f64,
    /// [`report_key`] of every `inputs.reports` entry.
    reports: Vec<String>,
}

fn same_file(a: &str, b: &str) -> bool {
    a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
}

/// A report path as compared across tables: its canonical path when it resolves from the
/// current directory, else the path as given. Tables record no report sha256.
fn report_key(p: &str) -> String {
    std::fs::canonicalize(p).map_or_else(|_| p.to_string(), |c| c.to_string_lossy().into_owned())
}

/// The sibling tables with their own `trials.n_eff`, `trials.nominal` and `spa.p_consistent`;
/// a sibling's own `trials.family` is never read, so tables that list each other count once.
/// A report of this table (`own`) or of an earlier sibling in a sibling's `inputs.reports`
/// would count its rows twice.
fn load_family(paths: &[String], own: &[String], out: Option<&str>, locked: &[(f64, f64)]) -> Res<Vec<Sibling>> {
    let own_keys: Vec<String> = if paths.is_empty() { Vec::new() } else { own.iter().map(|r| report_key(r)).collect() };
    let mut family: Vec<Sibling> = Vec::new();
    for p in paths {
        if let Some(o) = out.filter(|o| same_file(p, o)) {
            return usage(format!("--family-table {}: the file is the --out of this run ({}); a table cannot count itself", p, o));
        }
        let bytes = std::fs::read(p).map_err(|e| (3, format!("{}: {}", p, e)))?;
        let sha256 = crate::manifest::sha256_hex(&bytes);
        let j = Json::parse(String::from_utf8_lossy(&bytes).trim()).ok_or_else(|| (3, format!("{}: not a JSON table", p)))?;
        refuse_locked(&j, p, locked)?;
        if j.get("schema").and_then(Json::as_str) != Some(SCHEMA) {
            return usage(format!("--family-table {}: schema is not {} (the output of rcsr trials --table)", p, SCHEMA));
        }
        if let Some(twin) = family.iter().find(|s| s.sha256 == sha256) {
            return usage(format!("--family-table {}: sha256 {} is the sha256 of the family table {}; each sibling table counts once", p, sha256, twin.path));
        }
        let field = |path: &[&str], ok: fn(f64) -> bool| {
            num_at(&j, path).filter(|v| v.is_finite() && ok(*v)).ok_or_else(|| {
                (2, format!("--family-table {}: field '{}' is missing or invalid (a table with a computed SPA is needed)", p, path.join(".")))
            })
        };
        let n_eff = field(&["trials", "n_eff"], |v| v >= 1.0)?;
        let nominal = field(&["trials", "nominal"], |v| v >= 1.0)?;
        let spa_p = field(&["spa", "p_consistent"], |v| (0.0..=1.0).contains(&v))?;
        let listed: Vec<&str> = match at(&j, &["inputs", "reports"]) {
            Some(Json::Arr(a)) if !a.is_empty() && a.iter().all(|x| x.as_str().is_some()) => a.iter().filter_map(Json::as_str).collect(),
            _ => return usage(format!("--family-table {}: field 'inputs.reports' is missing or invalid (a list of report paths)", p)),
        };
        let reports: Vec<String> = listed.iter().map(|r| report_key(r)).collect();
        for (r, key) in listed.iter().zip(&reports) {
            let other = if own_keys.contains(key) {
                Some("this table (--wf)".to_string())
            } else {
                family.iter().find(|s| s.reports.contains(key)).map(|s| format!("the family table {}", s.path))
            };
            if let Some(other) = other {
                return usage(format!(
                    "--family-table {}: its report {} is also a report of {} (compared by canonical path); a report counts in one table only",
                    p, r, other
                ));
            }
        }
        family.push(Sibling { path: p.clone(), sha256, n_eff, nominal, spa_p, reports });
    }
    Ok(family)
}

/// The SHA-256 of the sorted sibling sha256 values, each followed by a newline.
fn family_sha(family: &[Sibling]) -> String {
    let mut shas: Vec<&str> = family.iter().map(|s| s.sha256.as_str()).collect();
    shas.sort_unstable();
    crate::manifest::sha256_hex(shas.iter().map(|s| format!("{}\n", s)).collect::<String>().as_bytes())
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// A metric value, or why the report cannot give it (a missing field).
#[derive(Clone, Debug)]
enum Val {
    Num(f64),
    Missing(String),
}

impl Val {
    fn num(&self) -> f64 {
        match self {
            Val::Num(v) => *v,
            Val::Missing(_) => f64::NAN,
        }
    }
}

fn from_res(r: &Result<f64, String>) -> Val {
    match r {
        Ok(v) => Val::Num(*v),
        Err(e) => Val::Missing(e.clone()),
    }
}

/// One calendar stratum of a row.
struct Cell {
    label: String,
    periods: usize,
    net: f64,
    bh: Val,
    excess: Val,
    alpha: Val,
}

struct Row {
    metrics: Vec<(String, Val)>,
    strata: Vec<Cell>,
}

impl Row {
    fn get(&self, name: &str) -> Option<Val> {
        if let Some((_, v)) = self.metrics.iter().find(|(m, _)| m == name) {
            return Some(v.clone());
        }
        let cell = |l: &str| self.strata.iter().find(|c| c.label == l);
        if let Some(l) = name.strip_prefix("net:") {
            return cell(l).map(|c| Val::Num(c.net));
        }
        if let Some(l) = name.strip_prefix("excess:") {
            return cell(l).map(|c| c.excess.clone());
        }
        if let Some(l) = name.strip_prefix("alpha_bh:") {
            return cell(l).map(|c| c.alpha.clone());
        }
        None
    }

    fn num(&self, name: &str) -> f64 {
        self.get(name).map_or(f64::NAN, |v| v.num())
    }
}

/// What the rows are computed against.
struct Ctx<'a> {
    cost_names: Vec<(String, f64)>,
    calendar: Option<&'a Calendar>,
    null_sharpes: Option<&'a [f64]>,
    baseline: Option<&'a [f64]>,
}

fn ann(v: &[f64], ppy: f64) -> f64 {
    inference::mean(v) * ppy
}

fn exposure_reason(e: &Entry) -> String {
    format!(
        "field '{}.period_exposure' is missing in {} (write the report with walkforward --period-series or --emit-series)",
        e.wf.prefix, e.wf.path
    )
}

fn row_of(e: &Entry, ctx: &Ctx, psr: f64) -> Row {
    let ppy = e.wf.ppy;
    let net = &e.wf.net;
    let n = net.len();
    let mut m: Vec<(String, Val)> = Vec::new();
    for (name, mult) in &ctx.cost_names {
        let v = if *mult == 1.0 {
            Val::Num(ann(net, ppy))
        } else {
            match &e.cost {
                Ok(c) => Val::Num(ann(&net.iter().zip(c).map(|(x, c)| x - (mult - 1.0) * c).collect::<Vec<_>>(), ppy)),
                Err(r) => Val::Missing(r.clone()),
            }
        };
        m.push((name.clone(), v));
    }
    let base = ann(net, ppy);
    let mdd = crate::walkforward::equity_metrics(net, ppy).2;
    let sortino = crate::risk::risk_stats(net, ppy, crate::risk::DEFAULT_SORTINO_MAR).sortino;
    let sharpe = crate::walkforward::sharpe(net, ppy);
    m.push(("sharpe".to_string(), Val::Num(sharpe)));
    m.push(("sortino".to_string(), Val::Num(sortino)));
    m.push(("calmar".to_string(), Val::Num(if mdd < 0.0 { base / -mdd } else { f64::NAN })));
    m.push(("max_drawdown".to_string(), Val::Num(mdd)));
    let exposure = e.wf.exposure.as_ref();
    m.push(("exposure".to_string(), exposure.map_or_else(|| Val::Missing(exposure_reason(e)), |x| Val::Num(inference::mean(x)))));
    m.push(("entry_times".to_string(), from_res(&e.entry_times)));
    let alpha_of = |idx: &[usize]| -> Val {
        match (&e.bh_net, exposure) {
            (Err(r), _) => Val::Missing(r.clone()),
            (_, None) => Val::Missing(exposure_reason(e)),
            (Ok(b), Some(x)) => Val::Num(ann(&idx.iter().map(|&t| net[t] - x[t] * b[t]).collect::<Vec<_>>(), ppy)),
        }
    };
    let all: Vec<usize> = (0..n).collect();
    m.push(("alpha_bh".to_string(), alpha_of(&all)));
    m.push(("psr".to_string(), Val::Num(psr)));
    match &e.groups {
        Ok(g) => {
            let sums: Vec<f64> = g
                .iter()
                .map(|&(s, end)| (0..n).filter(|&t| e.wf.time[t] >= s && e.wf.time[t] < end).map(|t| net[t]).sum())
                .collect();
            m.push(("groups".to_string(), Val::Num(g.len() as f64)));
            m.push(("groups_pos".to_string(), Val::Num(sums.iter().filter(|&&x| x > 0.0).count() as f64)));
        }
        Err(r) => {
            m.push(("groups".to_string(), Val::Missing(r.clone())));
            m.push(("groups_pos".to_string(), Val::Missing(r.clone())));
        }
    }
    let mut strata = Vec::new();
    if let Some(cal) = ctx.calendar {
        let (names, idx) = cal.assign(&e.wf.time);
        for (k, label) in names.iter().enumerate() {
            let p: Vec<usize> = (0..n).filter(|&t| idx[t] == k).collect();
            let snet = ann(&p.iter().map(|&t| net[t]).collect::<Vec<_>>(), ppy);
            let bh = match &e.bh_net {
                Ok(b) => Val::Num(ann(&p.iter().map(|&t| b[t]).collect::<Vec<_>>(), ppy)),
                Err(r) => Val::Missing(r.clone()),
            };
            let excess = match &bh {
                Val::Num(b) => Val::Num(snet - b),
                other => other.clone(),
            };
            strata.push(Cell { label: label.clone(), periods: p.len(), net: snet, bh, excess, alpha: alpha_of(&p) });
        }
        let labels = cal.labels();
        let cell = |l: &str| strata.iter().find(|c| c.label == l);
        if labels.iter().any(|l| l == "bull") {
            m.push(("bull_net".to_string(), Val::Num(cell("bull").map_or(f64::NAN, |c| c.net))));
        }
        if labels.iter().any(|l| l == "bear") {
            m.push(("bear_excess".to_string(), cell("bear").map_or(Val::Num(f64::NAN), |c| c.excess.clone())));
        }
        let worst = strata.iter().filter(|c| c.label != UNDATED && c.periods > 0).map(|c| c.net).fold(f64::NAN, f64::min);
        m.push(("worst_stratum".to_string(), Val::Num(worst)));
    }
    if let Some(null) = ctx.null_sharpes {
        let q = if sharpe.is_finite() && !null.is_empty() {
            null.iter().filter(|&&s| s < sharpe).count() as f64 / null.len() as f64
        } else {
            f64::NAN
        };
        m.push(("null_q".to_string(), Val::Num(q)));
    }
    if let Some(b) = ctx.baseline {
        m.push(("baseline_excess".to_string(), Val::Num((inference::mean(net) - inference::mean(b)) * ppy)));
    }
    Row { metrics: m, strata }
}

/// What a rule metric needs, checked before any row is computed.
struct Names<'a> {
    cost: Vec<String>,
    labels: Option<Vec<String>>,
    regimes: Option<&'a str>,
    null: bool,
    baseline: bool,
}

fn check_metric(name: &str, nm: &Names) -> Res<()> {
    if nm.cost.iter().any(|c| c == name) || BASE_METRICS.contains(&name) {
        return Ok(());
    }
    let label = |l: &str| -> Res<()> {
        match &nm.labels {
            None => usage(format!("metric '{}' needs --regimes", name)),
            Some(ls) if ls.iter().any(|x| x == l) => Ok(()),
            Some(_) => usage(format!(
                "metric '{}' needs the calendar label '{}', which --regimes {} does not have",
                name,
                l,
                nm.regimes.unwrap_or("")
            )),
        }
    };
    match name {
        "bull_net" => return label("bull"),
        "bear_excess" => return label("bear"),
        "worst_stratum" => return if nm.labels.is_some() { Ok(()) } else { usage("metric 'worst_stratum' needs --regimes") },
        "null_q" => return if nm.null { Ok(()) } else { usage("metric 'null_q' needs --null-wf") },
        "baseline_excess" => return if nm.baseline { Ok(()) } else { usage("metric 'baseline_excess' needs --baseline or --baseline-wf") },
        _ => {}
    }
    for prefix in ["net:", "excess:", "alpha_bh:"] {
        if let Some(l) = name.strip_prefix(prefix) {
            return label(l);
        }
    }
    usage(format!(
        "unknown metric '{}'; known: {}, {}; with --regimes bull_net, bear_excess, worst_stratum, net:<label>, excess:<label>, \
alpha_bh:<label>; null_q with --null-wf; baseline_excess with --baseline or --baseline-wf (the net metrics are cost_multiples x cost_model.per_side_bps)",
        name,
        nm.cost.join(", "),
        BASE_METRICS.join(", ")
    ))
}

// ---------------------------------------------------------------------------
// Pareto front, average rank, Friedman
// ---------------------------------------------------------------------------

/// NaN ranks below every number.
fn nan_low(v: f64) -> f64 {
    if v.is_nan() {
        f64::NEG_INFINITY
    } else {
        v
    }
}

/// Front rank (1 = first front) of every row of `values[row][metric]`, each
/// metric oriented so that higher is better. A row is in front `k` when no row
/// left after the first `k - 1` fronts dominates it (at least as good on every
/// metric and better on one). Rows equal on every metric share a front.
pub fn pareto_fronts(values: &[Vec<f64>]) -> Vec<usize> {
    let v: Vec<Vec<f64>> = values.iter().map(|r| r.iter().map(|&x| nan_low(x)).collect()).collect();
    let dominates = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x >= y) && a.iter().zip(b).any(|(x, y)| x > y);
    let mut front = vec![0usize; v.len()];
    let mut left: Vec<usize> = (0..v.len()).collect();
    let mut k = 0;
    while !left.is_empty() {
        k += 1;
        let now: Vec<usize> = left.iter().copied().filter(|&i| !left.iter().any(|&j| j != i && dominates(&v[j], &v[i]))).collect();
        for &i in &now {
            front[i] = k;
        }
        left.retain(|i| !now.contains(i));
    }
    front
}

/// Rank of every value, 1 for the highest; tied values share the mean of their
/// places; NaN ranks last.
pub fn column_ranks(col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let key: Vec<f64> = col.iter().map(|&x| nan_low(x)).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| key[b].total_cmp(&key[a]).then(a.cmp(&b)));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && key[idx[j + 1]] == key[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &t in &idx[i..=j] {
            ranks[t] = r;
        }
        i = j + 1;
    }
    ranks
}

/// `values[row][metric]` oriented so that higher is better: the rank of every
/// row on every metric (`ranks[row][metric]`) and the mean rank of every row.
pub fn average_ranks(values: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let m = values.first().map_or(0, |r| r.len());
    let mut ranks = vec![vec![0.0; m]; values.len()];
    for c in 0..m {
        let col: Vec<f64> = values.iter().map(|r| r[c]).collect();
        for (i, r) in column_ranks(&col).into_iter().enumerate() {
            ranks[i][c] = r;
        }
    }
    let avg = ranks.iter().map(|r| if m == 0 { f64::NAN } else { r.iter().sum::<f64>() / m as f64 }).collect();
    (ranks, avg)
}

/// The Friedman statistic over rows (treatments) by metrics (blocks).
#[derive(Clone, Debug)]
pub struct Friedman {
    pub chi2: f64,
    pub df: usize,
    pub p: f64,
    pub rows: usize,
    pub metrics: usize,
}

/// `chi2 = 12 N / (k (k + 1)) (sum_j R_j^2 - k (k + 1)^2 / 4)` with `N` metrics,
/// `k` rows and `R_j` the mean rank of row `j` (Demšar 2006, no tie
/// correction); `p` from the chi-square tail with `k - 1` degrees of freedom.
pub fn friedman(ranks: &[Vec<f64>]) -> Friedman {
    let k = ranks.len();
    let nm = ranks.first().map_or(0, |r| r.len());
    if k < 2 || nm == 0 {
        return Friedman { chi2: f64::NAN, df: k.saturating_sub(1), p: f64::NAN, rows: k, metrics: nm };
    }
    let (kf, nf) = (k as f64, nm as f64);
    let sum_sq: f64 = ranks
        .iter()
        .map(|r| {
            let a = r.iter().sum::<f64>() / nf;
            a * a
        })
        .sum();
    let chi2 = 12.0 * nf / (kf * (kf + 1.0)) * (sum_sq - kf * (kf + 1.0).powi(2) / 4.0);
    Friedman { chi2, df: k - 1, p: crate::diagnostics::chi2_sf(chi2, k - 1), rows: k, metrics: nm }
}

/// Pearson correlation; 0 when either series is flat.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (sa, sb) = (inference::sample_sd(&a[..n]), inference::sample_sd(&b[..n]));
    if !(n >= 2 && sa > 0.0 && sb > 0.0) {
        return 0.0;
    }
    let (ma, mb) = (inference::mean(&a[..n]), inference::mean(&b[..n]));
    let cov = (0..n).map(|t| (a[t] - ma) * (b[t] - mb)).sum::<f64>() / (n as f64 - 1.0);
    (cov / (sa * sb)).clamp(-1.0, 1.0)
}

/// Harvey and Liu (2015) adjusted p-values over all `p`: Bonferroni, Holm and BHY.
fn hl_adjust(p: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = p.len();
    let mf = m as f64;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let bonf: Vec<f64> = p.iter().map(|&x| (mf * x).min(1.0)).collect();
    let mut holm = vec![0.0; m];
    let mut running = 0.0f64;
    for (l, &i) in order.iter().enumerate() {
        running = running.max(((mf - l as f64) * p[i]).min(1.0));
        holm[i] = running;
    }
    let c: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();
    let mut bhy = vec![0.0; m];
    let mut running = 1.0f64;
    for (l, &i) in order.iter().enumerate().rev() {
        running = running.min((mf * c * p[i] / (l + 1) as f64).min(1.0));
        bhy[i] = running;
    }
    (bonf, holm, bhy)
}

fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).clamp(1, n.max(1));
    if threads <= 1 || n < 2 {
        return (0..n).map(f).collect();
    }
    let size = n.div_ceil(threads);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(size).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(f(c * size + k));
                }
            });
        }
    });
    out.into_iter().map(|v| v.expect("every book is computed")).collect()
}

/// Per-period Sharpe, the walk-forward `T` (block-bootstrap effective periods), skew and raw kurtosis.
struct BookStat {
    sr: f64,
    t: f64,
    skew: f64,
    kurt: f64,
}

fn book_stat(x: &[f64], block: usize, draws: usize, seed: u32, ppy: f64) -> BookStat {
    let summary = inference::block_bootstrap(x, block, draws, seed, ppy, 0.95);
    let t = if summary.effective_n.is_finite() && summary.effective_n > 1.0 { summary.effective_n.min(x.len() as f64) } else { x.len() as f64 };
    BookStat { sr: inference::sharpe_per_period(x), t, skew: inference::skewness(x), kurt: inference::kurtosis(x) }
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

struct Out {
    json: String,
    markdown: String,
}

/// Why the walk passed over a kept row.
#[derive(Clone, Copy, Debug)]
enum Skip {
    /// The row's net series correlates at `diversity_rho` or more with this picked row.
    Diversity(usize, f64),
    /// #1237 (D8): the picks already hold `max_composites` composite rows.
    Composites(usize),
}

fn opt_json(v: Option<f64>) -> String {
    v.map_or("null".to_string(), jnum)
}

fn skipped(reason: &str) -> String {
    format!("{{\"skipped\":{}}}", jstr(reason))
}

fn cell(v: f64) -> String {
    if v.is_finite() {
        format!("{:.4}", v)
    } else {
        "n/a".to_string()
    }
}

fn metric_cell(name: &str, v: &Val) -> String {
    match v {
        Val::Num(x) if x.is_finite() && matches!(name, "entry_times" | "groups" | "groups_pos") => format!("{}", x),
        Val::Num(x) => cell(*x),
        Val::Missing(_) => "n/a".to_string(),
    }
}

fn table(o: &Options, t: &TableArgs, rule: &Rule, rule_sha: &str, rule_source: &str) -> Res<Out> {
    let map: Option<FreezeMap> = t.freeze_map.as_deref().map(trials_freeze::load_map).transpose()?;
    let family = load_family(&t.family, &o.wf, o.out.as_deref(), &t.locked)?;
    let entries = load_entries(o, &t.locked)?;
    let first = &entries[0];
    let n = first.wf.net.len();
    let ppy = first.wf.ppy;
    let ann_scale = ppy.max(0.0).sqrt();
    let block = o.block_len.unwrap_or_else(|| entries.iter().map(|e| e.wf.default_block).max().unwrap_or(1)).clamp(1, n.max(1));
    let alpha = t.alpha.unwrap_or(DEFAULT_ALPHA);

    // Net metric names from the report cost.
    let Some(bps) = first.per_side_bps else {
        return usage(format!("field 'cost_model.per_side_bps' is missing in {} (the net metric names come from it)", first.wf.path));
    };
    let mut cost_names: Vec<(String, f64)> = Vec::new();
    for &m in &rule.cost_multiples {
        let name = format!("net{}", jnum(bps * m));
        if let Some((_, other)) = cost_names.iter().find(|(x, _)| *x == name) {
            return usage(format!("cost multiples {} and {} both give the metric {} (cost_model.per_side_bps is {})", jnum(*other), jnum(m), name, jnum(bps)));
        }
        cost_names.push((name, m));
    }

    let calendar = match o.regimes.as_deref() {
        None => None,
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
            Some(Calendar::parse(&text).map_err(|e| (2, format!("--regimes {}: {}", path, e)))?)
        }
    };
    let names = Names {
        cost: cost_names.iter().map(|(x, _)| x.clone()).collect(),
        labels: calendar.as_ref().map(|c| c.labels()),
        regimes: o.regimes.as_deref(),
        null: !t.null_wf.is_empty(),
        baseline: o.baseline.is_some() || !t.baseline_wf.is_empty(),
    };
    let used = rule.metrics();
    for m in &used {
        check_metric(m, &names)?;
    }

    // The null set, the baseline and the occam runs.
    let mut null_sharpes: Vec<f64> = Vec::new();
    for p in &t.null_wf {
        let j = read_null(p, &t.locked)?;
        let w = trials::wf_entry(&j, &j, "pooled", p, trials::label_of(p))?;
        trials::require_same(&first.wf, &w, o.threshold_arms)?;
        null_sharpes.push(crate::walkforward::sharpe(&w.net, w.ppy));
    }
    let wf_baseline = if t.baseline_wf.is_empty() { None } else { Some(load_baseline_wf(&t.baseline_wf, &first.wf.time, first.per_side_bps, &t.locked)?) };
    let baseline = match (&o.baseline, &wf_baseline) {
        (_, Some(w)) => Some(w.best.clone()),
        (None, None) => None,
        (Some(path), None) => Some(trials::load_baseline(path, o.baseline_rung.as_deref(), &first.wf.time)?),
    };
    let occam = load_occam(&t.occam)?;

    // Distinct books and clusters, as the campaign counts them.
    let rows_n = entries.len();
    let hashes: Vec<String> = entries.iter().map(|e| trials::book_hash(&e.wf.net)).collect();
    let mut distinct: Vec<usize> = Vec::new();
    let mut book_of = vec![0usize; rows_n];
    for (i, h) in hashes.iter().enumerate() {
        match distinct.iter().position(|&d| &hashes[d] == h) {
            Some(p) => book_of[i] = p,
            None => {
                book_of[i] = distinct.len();
                distinct.push(i);
            }
        }
    }
    let varying: Vec<usize> = distinct.iter().copied().filter(|&i| !trials::is_constant(&entries[i].wf.net)).collect();
    let constant_books = distinct.len() - varying.len();
    let cols: Vec<Vec<f64>> = varying.iter().map(|&i| entries[i].wf.net.clone()).collect();
    let vpos: Vec<Option<usize>> = (0..rows_n).map(|i| varying.iter().position(|&x| x == distinct[book_of[i]])).collect();
    let (cluster_of, clusters_varying, pr) = if cols.is_empty() {
        (Vec::new(), 0, f64::NAN)
    } else {
        let corr = inference::correlation_matrix(&cols);
        let c = inference::single_linkage_clusters(&corr, o.rho);
        let k = c.iter().copied().max().map_or(0, |m| m + 1);
        (c, k, inference::participation_ratio(&corr))
    };
    let n_eff = (clusters_varying + constant_books).max(1) as f64;
    let nominal = rows_n as f64;
    let nominal_prior = nominal + t.prior_trials as f64;
    let n_eff_total = family.iter().fold(n_eff, |a, s| a + s.n_eff);
    let nominal_total = family.iter().fold(nominal_prior, |a, s| a + s.nominal);

    // Block statistics of every varying book.
    let stats: Vec<BookStat> = par_map(cols.len(), |v| book_stat(&cols[v], block, o.draws, o.seed, ppy));
    let srs: Vec<f64> = stats.iter().map(|s| s.sr).filter(|v| v.is_finite()).collect();
    let var_all = if cols.len() >= 2 {
        let sd = inference::sample_sd(&srs);
        sd * sd
    } else {
        f64::NAN
    };
    let var_of = |s: &BookStat| if cols.len() >= 2 { var_all } else { inference::sharpe_estimator_variance(s.sr, s.t, s.skew, s.kurt) };
    let psr_of = |r: usize| vpos[r].map_or(f64::NAN, |v| {
        let s = &stats[v];
        inference::deflated_sharpe(s.sr, s.t, s.skew, s.kurt, 1.0, var_of(s)).dsr
    });

    // Rows.
    let ctx = Ctx {
        cost_names: cost_names.clone(),
        calendar: calendar.as_ref(),
        null_sharpes: if t.null_wf.is_empty() { None } else { Some(null_sharpes.as_slice()) },
        baseline: baseline.as_ref().map(|(_, b)| b.as_slice()),
    };
    let rows: Vec<Row> = entries.iter().enumerate().map(|(i, e)| row_of(e, &ctx, psr_of(i))).collect();
    for m in &used {
        for (i, r) in rows.iter().enumerate() {
            if let Some(Val::Missing(why)) = r.get(m) {
                return usage(format!("metric '{}' of row {} cannot be computed: {}", m, entries[i].wf.label, why));
            }
        }
    }

    // Gates.
    let fails = |gates: &[Gate]| -> Vec<Vec<(usize, f64)>> {
        rows.iter()
            .map(|r| gates.iter().enumerate().filter_map(|(g, x)| { let v = r.num(&x.metric); (!x.passes(v)).then_some((g, v)) }).collect::<Vec<_>>())
            .collect()
    };
    let strict = fails(rule.gates.as_slice());
    let strict_pass = strict.iter().filter(|f| f.is_empty()).count();
    let mut notes: Vec<String> = Vec::new();
    let relaxed = match &rule.relaxed_gates {
        Some(g) if strict_pass == 0 => {
            notes.push("no row passes the gates: the relaxed gates apply".to_string());
            Some(fails(g.as_slice()))
        }
        _ => None,
    };
    let gate_set = if relaxed.is_some() { "relaxed" } else if rule.gates.is_empty() { "none" } else { "strict" };
    let applied = relaxed.as_ref().unwrap_or(&strict);
    let passers: Vec<usize> = (0..rows_n).filter(|&i| applied[i].is_empty()).collect();
    let few = passers.len() < 2;
    if few && (!rule.pareto.is_empty() || !rule.rank.is_empty()) {
        notes.push(format!("{} gate passer(s): the Pareto and rank steps are skipped", passers.len()));
    }
    let oriented = |metrics: &[(String, Dir)], r: usize| -> Vec<f64> { metrics.iter().map(|(m, d)| d.orient(rows[r].num(m))).collect() };

    // Pareto front.
    let mut front: Vec<Option<usize>> = vec![None; rows_n];
    let mut kept: Vec<usize> = passers.clone();
    let pareto_json = if rule.pareto.is_empty() {
        "null".to_string()
    } else if few {
        skipped("fewer than 2 gate passers")
    } else {
        let vals: Vec<Vec<f64>> = passers.iter().map(|&r| oriented(rule.pareto.as_slice(), r)).collect();
        let f = pareto_fronts(&vals);
        for (k, &r) in passers.iter().enumerate() {
            front[r] = Some(f[k]);
        }
        let fronts = f.iter().copied().max().unwrap_or(0);
        let mut keep_to = fronts;
        if rule.min_front > 0 {
            for k in 1..=fronts {
                if f.iter().filter(|&&x| x <= k).count() >= rule.min_front {
                    keep_to = k;
                    break;
                }
            }
        }
        kept.retain(|&r| front[r].is_some_and(|x| x <= keep_to));
        format!(
            "{{\"metrics\":{},\"fronts\":{},\"first_front\":{},\"kept_fronts\":{},\"kept_rows\":{}}}",
            dirs_json(&rule.pareto),
            fronts,
            f.iter().filter(|&&x| x == 1).count(),
            keep_to,
            kept.len()
        )
    };

    // Average rank.
    let mut avg_rank: Vec<Option<f64>> = vec![None; rows_n];
    let rank_json = if rule.rank.is_empty() {
        "null".to_string()
    } else if few {
        skipped("fewer than 2 gate passers")
    } else {
        let vals: Vec<Vec<f64>> = passers.iter().map(|&r| oriented(rule.rank.as_slice(), r)).collect();
        let (ranks, avg) = average_ranks(&vals);
        for (k, &r) in passers.iter().enumerate() {
            avg_rank[r] = Some(avg[k]);
        }
        let f = friedman(&ranks);
        format!(
            "{{\"metrics\":{},\"friedman\":{{\"chi2\":{},\"df\":{},\"p\":{},\"rows\":{},\"metrics\":{},\"ties\":\"mean rank, no tie correction\"}}}}",
            dirs_json(&rule.rank),
            jnum(f.chi2),
            f.df,
            jnum(f.p),
            f.rows,
            f.metrics
        )
    };

    // Walk order: front, average rank, complexity, trades, input order.
    let key = |v: Option<f64>| v.filter(|x| !x.is_nan()).unwrap_or(f64::INFINITY);
    let mut order = kept.clone();
    order.sort_by(|&a, &b| {
        front[a]
            .unwrap_or(0)
            .cmp(&front[b].unwrap_or(0))
            .then(key(avg_rank[a]).total_cmp(&key(avg_rank[b])))
            .then(key(entries[a].complexity).total_cmp(&key(entries[b].complexity)))
            .then(key(entries[a].trades).total_cmp(&key(entries[b].trades)))
            .then(a.cmp(&b))
    });
    let mut order_of: Vec<Option<usize>> = vec![None; rows_n];
    for (k, &r) in order.iter().enumerate() {
        order_of[r] = Some(k + 1);
    }
    let mut picks: Vec<usize> = Vec::new();
    let mut pick_of: Vec<Option<usize>> = vec![None; rows_n];
    let mut skip_of: Vec<Option<Skip>> = vec![None; rows_n];
    if rule.pick == 0 {
        notes.push("no pick count in the rule: no row is picked".to_string());
    }
    let mut composites = 0usize;
    for &r in &order {
        if picks.len() >= rule.pick {
            break;
        }
        let composite = entries[r].composite.is_some();
        if let Some(limit) = rule.max_composites.filter(|&l| composite && composites >= l) {
            skip_of[r] = Some(Skip::Composites(limit));
            continue;
        }
        let hit = picks.iter().map(|&p| (p, pearson(&entries[r].wf.net, &entries[p].wf.net))).find(|(_, c)| *c >= rule.diversity_rho);
        match hit {
            Some((p, c)) => skip_of[r] = Some(Skip::Diversity(p, c)),
            None => {
                picks.push(r);
                pick_of[r] = Some(picks.len());
                composites += usize::from(composite);
            }
        }
    }
    if rule.pick > 0 && picks.len() < rule.pick {
        notes.push(format!("{} of {} picks: the kept rows ran out", picks.len(), rule.pick));
    }

    // #1237: the sealed-read thresholds of every pick, frozen before any family test runs.
    let mut frozen: Vec<trials_freeze::Frozen> = Vec::new();
    if let (Some(fr), Some(m)) = (&rule.freeze, &map) {
        for (k, &r) in picks.iter().enumerate() {
            let e = &entries[r];
            let Some(row) = m.row(&e.wf.path) else {
                return usage(format!(
                    "pick {} (row {}, label '{}', report {}) has no entry in --freeze-map {}: rows[].row names the report as given to --wf",
                    k + 1,
                    r,
                    e.wf.label,
                    e.wf.path,
                    m.path
                ));
            };
            frozen.push(trials_freeze::freeze_pick(row, fr, m, &t.locked)?);
        }
    }

    // Family tests against the benchmark: the --baseline best rung, else cash at 0.
    let (bench_name, bench): (String, Vec<f64>) = match &baseline {
        Some((name, b)) => (name.clone(), b.clone()),
        None => ("cash".to_string(), vec![0.0; n]),
    };
    let diffs: Vec<Vec<f64>> = cols.iter().map(|c| c.iter().zip(&bench).map(|(a, b)| a - b).collect()).collect();
    let p_family = |p: f64| ((1 + family.len()) as f64 * p).min(1.0);
    let spa_json = if cols.is_empty() {
        skipped("no varying series")
    } else {
        match inference::spa_test(&diffs, block, o.draws, o.seed) {
            Ok(s) => format!(
                "{{\"benchmark\":{},\"variant\":\"hansen_2005_circular_block\",\"models\":{},\"n\":{},\"block_len\":{},\"draws\":{},\"seed\":{},\
\"statistic\":{},\"best_model\":{},\"p_consistent\":{},\"p_lower\":{},\"p_upper\":{}{}}}",
                jstr(&bench_name),
                s.models,
                s.n,
                s.block_len,
                s.draws,
                s.seed,
                jnum(s.statistic),
                s.best_model.map_or("null".to_string(), |m| jstr(&entries[varying[m]].wf.label)),
                jnum(s.p_consistent),
                jnum(s.p_lower),
                jnum(s.p_upper),
                if family.is_empty() { String::new() } else { format!(",\"p_family\":{}", jnum(p_family(s.p_consistent))) }
            ),
            Err(e) => skipped(&e),
        }
    };
    let stepm = if cols.is_empty() { Err("no varying series".to_string()) } else { inference::stepm(&diffs, block, o.draws, o.seed, alpha) };
    let bh = stepm.as_ref().ok().map(|s| inference::bh_fdr(&s.p_values, alpha));
    let stepm_json = match &stepm {
        Ok(s) => format!(
            "{{\"benchmark\":{},\"variant\":\"romano_wolf_2005_studentized_circular_block\",\"alpha\":{},\"models\":{},\"n\":{},\"block_len\":{},\
\"draws\":{},\"seed\":{},\"rejected\":{},\"steps\":{},\"critical\":[{}]}}",
            jstr(&bench_name),
            jnum(s.alpha),
            s.models,
            s.n,
            s.block_len,
            s.draws,
            s.seed,
            s.rejected.iter().filter(|&&r| r).count(),
            s.critical.len(),
            s.critical.iter().map(|c| jnum(*c)).collect::<Vec<_>>().join(",")
        ),
        Err(e) => skipped(e),
    };
    let bh_json = match &bh {
        Some(b) => format!("{{\"benchmark\":{},\"q\":{},\"models\":{},\"rejected\":{}}}", jstr(&bench_name), jnum(b.q), b.rejected.len(), b.n_rejected),
        None => skipped("no StepM p-values"),
    };
    let pbo_json = if cols.len() < 2 {
        skipped("fewer than 2 distinct varying series")
    } else {
        match inference::pbo_cscv(&cols, o.pbo_blocks) {
            Ok(p) => format!(
                "{{\"s\":{},\"candidates\":{},\"rows_per_block\":{},\"combinations\":{},\"value\":{},\"mean_logit\":{}}}",
                p.blocks,
                p.candidates,
                p.rows_per_block,
                p.combinations,
                jnum(p.pbo),
                jnum(p.mean_logit)
            ),
            Err(e) => skipped(&e),
        }
    };

    // Harvey-Liu: two-sided p of t = SR sqrt(T) of every row, adjusted over the rows.
    let t_of = |r: usize| vpos[r].map_or(f64::NAN, |v| stats[v].sr * stats[v].t.sqrt());
    let p_hl: Vec<f64> = (0..rows_n)
        .map(|r| {
            let t = t_of(r);
            if t.is_finite() { crate::expr::erfc(t.abs() / std::f64::consts::SQRT_2) } else { 1.0 }
        })
        .collect();
    let (p_bonf, p_holm, p_bhy) = hl_adjust(&p_hl);

    // Picks.
    let span_years = if ppy > 0.0 { n as f64 / ppy } else { f64::NAN };
    let pick_json: Vec<String> = picks
        .iter()
        .enumerate()
        .map(|(k, &r)| {
            let head = format!("{{\"pick\":{},\"row\":{},\"label\":{}", k + 1, r, jstr(&entries[r].wf.label));
            let Some(v) = vpos[r] else {
                return format!("{},\"skipped\":\"a constant series has no deflation\"}}", head);
            };
            let s = &stats[v];
            let var_sr = var_of(s);
            let d = |trials_n: f64| inference::deflated_sharpe(s.sr, s.t, s.skew, s.kurt, trials_n, var_sr);
            let sharpe = s.sr * ann_scale;
            let t = t_of(r);
            let haircut = |p_adj: f64| -> String {
                let t_adj = if p_adj <= p_hl[r] { t.abs() } else { -crate::inference::normal_quantile(p_adj / 2.0) };
                let adj = if t.abs() > 0.0 { sharpe * t_adj / t.abs() } else { f64::NAN };
                format!("{{\"p\":{},\"sharpe\":{},\"haircut\":{}}}", jnum(p_adj), jnum(adj), jnum(if sharpe != 0.0 { 1.0 - adj / sharpe } else { f64::NAN }))
            };
            let tests = match (&stepm, &bh) {
                (Ok(sm), Some(b)) => format!(
                    ",\"stepm\":{{\"rejected\":{},\"p\":{}}},\"bh_fdr\":{{\"rejected\":{},\"p_adjusted\":{}}}",
                    sm.rejected[v],
                    jnum(sm.p_values[v]),
                    b.rejected[v],
                    jnum(b.adjusted[v])
                ),
                _ => ",\"stepm\":null,\"bh_fdr\":null".to_string(),
            };
            format!(
                "{},\"sharpe_annual\":{},\"psr\":{},\"dsr\":{},\"dsr_10x\":{},\"dsr_nominal\":{},\"min_btl_years\":{},\"span_years\":{},\
\"harvey_liu\":{{\"t\":{},\"p\":{},\"tests\":{},\"t_above_3\":{},\"bonferroni\":{},\"holm\":{},\"bhy\":{}}}{}}}",
                head,
                jnum(sharpe),
                jnum(d(1.0).dsr),
                trials::dsr_json(&d(n_eff_total), n_eff_total),
                trials::dsr_json(&d(10.0 * n_eff_total), 10.0 * n_eff_total),
                trials::dsr_json(&d(nominal_total), nominal_total),
                jnum(inference::min_btl_years(sharpe, n_eff_total)),
                jnum(span_years),
                jnum(t),
                jnum(p_hl[r]),
                rows_n,
                t.is_finite() && t > 3.0,
                haircut(p_bonf[r]),
                haircut(p_holm[r]),
                haircut(p_bhy[r]),
                tests
            )
        })
        .collect();
    let marker = |r: usize| entries[r].composite.as_ref().map_or("null".to_string(), crate::cpcv::to_json);
    let pick_json: Vec<String> = pick_json
        .into_iter()
        .enumerate()
        .map(|(k, mut s)| {
            s.pop();
            s.push_str(&format!(",\"composite\":{}", marker(picks[k])));
            if let Some(f) = frozen.get(k) {
                s.push_str(&format!(",\"freeze\":{}", f.json));
            }
            s.push('}');
            s
        })
        .collect();

    if constant_books > 0 {
        notes.push(format!("{} constant series counted as one cluster each and left out of clustering, SPA, StepM and PBO", constant_books));
    }
    if baseline.is_none() {
        notes.push("no --baseline: SPA, StepM and BH-FDR test against cash at 0".to_string());
    }

    // Row JSON.
    let row_json: Vec<String> = (0..rows_n)
        .map(|i| {
            let e = &entries[i];
            let r = &rows[i];
            let metrics: Vec<String> = r.metrics.iter().map(|(m, v)| format!("{}:{}", jstr(m), jnum(v.num()))).collect();
            let unavailable: Vec<String> = r.metrics.iter().filter(|(_, v)| matches!(v, Val::Missing(_))).map(|(m, _)| jstr(m)).collect();
            let strata = if calendar.is_some() {
                let cells: Vec<String> = r
                    .strata
                    .iter()
                    .map(|c| {
                        format!(
                            "{{\"label\":{},\"periods\":{},\"net\":{},\"bh\":{},\"excess\":{},\"alpha_bh\":{}}}",
                            jstr(&c.label),
                            c.periods,
                            jnum(c.net),
                            jnum(c.bh.num()),
                            jnum(c.excess.num()),
                            jnum(c.alpha.num())
                        )
                    })
                    .collect();
                format!(",\"strata\":[{}]", cells.join(","))
            } else {
                String::new()
            };
            let gate_obj = |gates: &[Gate], f: &[(usize, f64)]| {
                let failed: Vec<String> = f
                    .iter()
                    .map(|&(g, v)| {
                        let x = &gates[g];
                        format!("{{\"metric\":{},\"op\":{},\"threshold\":{},\"value\":{}}}", jstr(&x.metric), jstr(x.op_str()), x.threshold_json(), jnum(v))
                    })
                    .collect();
                format!("{{\"pass\":{},\"failed\":[{}]}}", f.is_empty(), failed.join(","))
            };
            let relaxed_obj = match (&relaxed, &rule.relaxed_gates) {
                (Some(rf), Some(g)) => format!(",\"relaxed_gates\":{}", gate_obj(g.as_slice(), rf[i].as_slice())),
                _ => String::new(),
            };
            let occam_obj = if t.occam.is_empty() { String::new() } else { format!(",\"occam\":{}", occam_cell(e, &occam)) };
            format!(
                "{{\"row\":{},\"label\":{},\"report\":{},\"book\":{},\"cluster\":{},\"archive_rank\":{},\"complexity\":{},\"trades\":{},\
\"expression\":{},\"composite\":{},\"metrics\":{{{}}},\"unavailable\":[{}]{}{},\"gates\":{}{},\"passes\":{},\"front\":{},\"avg_rank\":{},\
\"order\":{},\"pick\":{},\"skipped\":{}}}",
                i,
                jstr(&e.wf.label),
                jstr(&e.wf.path),
                book_of[i],
                vpos[i].map_or("null".to_string(), |p| cluster_of[p].to_string()),
                opt_json(e.archive_rank),
                opt_json(e.complexity),
                opt_json(e.trades),
                e.expression.as_deref().map_or("null".to_string(), jstr),
                marker(i),
                metrics.join(","),
                unavailable.join(","),
                strata,
                occam_obj,
                gate_obj(rule.gates.as_slice(), strict[i].as_slice()),
                relaxed_obj,
                applied[i].is_empty(),
                front[i].map_or("null".to_string(), |f| f.to_string()),
                opt_json(avg_rank[i]),
                order_of[i].map_or("null".to_string(), |k| k.to_string()),
                pick_of[i].map_or("null".to_string(), |k| k.to_string()),
                skip_of[i].map_or("null".to_string(), |s| match s {
                    Skip::Diversity(p, c) => format!(
                        "{{\"reason\":\"diversity\",\"row\":{},\"label\":{},\"rho\":{}}}",
                        p,
                        jstr(&entries[p].wf.label),
                        jnum(c)
                    ),
                    Skip::Composites(limit) => format!("{{\"reason\":\"max_composites\",\"limit\":{}}}", limit),
                })
            )
        })
        .collect();

    let calendar_json = calendar.as_ref().map_or("null".to_string(), |c| {
        format!(
            "{{\"sha256\":{},\"rows\":{},\"labels\":[{}]}}",
            jstr(&c.sha256),
            c.rows.len(),
            c.labels().iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",")
        )
    });
    let mut rule_obj = rule.json();
    rule_obj.pop();
    rule_obj.push_str(&format!(",\"source\":{},\"sha256\":{}}}", jstr(rule_source), jstr(rule_sha)));
    let freeze_obj = match (&map, &rule.freeze) {
        (Some(m), Some(_)) => format!(",\"freeze\":{}", trials_freeze::top_json(rule_sha, m)),
        _ => String::new(),
    };
    let family_obj = if family.is_empty() {
        String::new()
    } else {
        let tables: Vec<String> = family
            .iter()
            .map(|s| {
                format!("{{\"path\":{},\"sha256\":{},\"n_eff\":{},\"nominal\":{},\"spa_p\":{}}}", jstr(&s.path), jstr(&s.sha256), jnum(s.n_eff), jnum(s.nominal), jnum(s.spa_p))
            })
            .collect();
        format!(
            ",\"family\":{{\"tables\":[{}],\"sha256\":{},\"n_eff_total\":{},\"nominal_total\":{}}}",
            tables.join(","),
            jstr(&family_sha(&family)),
            jnum(n_eff_total),
            jnum(nominal_total)
        )
    };
    let json = format!(
        "{{\"schema\":{},\"rule\":{}{},\"inputs\":{{\"reports\":[{}],\"members\":{},\"null_reports\":{},\"null_sharpes\":[{}],\"calendar\":{},\
\"baseline\":{},\"occam_runs\":{},\"locked_test\":[{}]}},\
\"trials\":{{\"nominal\":{},\"prior_trials\":{},\"nominal_with_prior\":{},\"distinct_books\":{},\"constant_books\":{},\"n_eff\":{},\
\"cluster_rho\":{},\"participation_ratio\":{}{}}},\
\"periods\":{},\"periods_per_year\":{},\"span_years\":{},\"block_len\":{},\"draws\":{},\"seed\":{},\"alpha\":{},\"benchmark\":{},\
\"gate_set\":{},\"gate_passers\":{},\"pareto\":{},\"rank\":{},\"picks\":[{}],\"spa\":{},\"stepm\":{},\"bh_fdr\":{},\"pbo\":{},\
\"notes\":[{}],\"rows\":[{}]}}\n",
        jstr(SCHEMA),
        rule_obj,
        freeze_obj,
        o.wf.iter().map(|p| jstr(p)).collect::<Vec<_>>().join(","),
        jstr(if o.all_members { "all" } else { "selected" }),
        t.null_wf.len(),
        null_sharpes.iter().map(|s| jnum(*s)).collect::<Vec<_>>().join(","),
        calendar_json,
        wf_baseline.as_ref().map(|w| w.json.clone()).or_else(|| baseline.as_ref().map(|(name, _)| jstr(name))).unwrap_or_else(|| "null".to_string()),
        t.occam.len(),
        t.locked.iter().map(|(s, e)| format!("[{},{}]", jnum(*s), jnum(*e))).collect::<Vec<_>>().join(","),
        rows_n,
        t.prior_trials,
        jnum(nominal_prior),
        distinct.len(),
        constant_books,
        jnum(n_eff),
        jnum(o.rho),
        jnum(pr),
        family_obj,
        n,
        jnum(ppy),
        jnum(span_years),
        block,
        o.draws,
        o.seed,
        jnum(alpha),
        jstr(&bench_name),
        jstr(gate_set),
        passers.len(),
        pareto_json,
        rank_json,
        pick_json.join(","),
        spa_json,
        stepm_json,
        bh_json,
        pbo_json,
        notes.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
        row_json.join(",")
    );

    // Markdown.
    let mut md = String::from("# Candidate table\n\n");
    md.push_str(&format!(
        "Rule sha256 `{}` ({}). Rows {} (nominal N), prior trials {}, distinct books {}, N_eff {} (single linkage at |rho| >= {}). \
Gate set: {}, {} of {} rows pass. Benchmark: {}.\n\n",
        rule_sha,
        rule_source,
        rows_n,
        t.prior_trials,
        distinct.len(),
        jnum(n_eff),
        jnum(o.rho),
        gate_set,
        passers.len(),
        rows_n,
        bench_name
    ));
    if !family.is_empty() {
        let spa_p = Json::parse(&spa_json).and_then(|j| num_at(&j, &["p_consistent"]));
        md.push_str(&format!(
            "Family: {} sibling horizon table(s) (family sha256 `{}`). DSR, DSR 10 N_eff and MinBTL use N_eff {} (this table {} plus {}), \
DSR nominal uses N {} (this table {} plus {}); the expected maximum keeps this table's cross-trial Sharpe variance, the family adds \
trials, not variance. {}.\n\n",
            family.len(),
            family_sha(&family),
            jnum(n_eff_total),
            jnum(n_eff),
            family.iter().map(|s| jnum(s.n_eff)).collect::<Vec<_>>().join(" + "),
            jnum(nominal_total),
            jnum(nominal_prior),
            family.iter().map(|s| jnum(s.nominal)).collect::<Vec<_>>().join(" + "),
            spa_p.map_or("SPA skipped: no family p".to_string(), |p| format!(
                "SPA family p {} = min(1, {} x p {})",
                cell(p_family(p)),
                family.len() + 1,
                cell(p)
            ))
        ));
    }
    let dir_list = |d: &[(String, Dir)]| d.iter().map(|(m, x)| format!("{} ({})", m, x.name())).collect::<Vec<_>>().join(", ");
    if !rule.pareto.is_empty() {
        md.push_str(&format!("Pareto over {}: {}.\n\n", dir_list(rule.pareto.as_slice()), match front.iter().flatten().max() {
            Some(f) => format!("{} fronts, {} rows kept", f, kept.len()),
            None => "skipped".to_string(),
        }));
    }
    if !rule.rank.is_empty() {
        md.push_str(&format!("Average rank over {}", dir_list(rule.rank.as_slice())));
        if avg_rank.iter().any(Option::is_some) {
            let vals: Vec<Vec<f64>> = passers.iter().map(|&r| oriented(rule.rank.as_slice(), r)).collect();
            let f = friedman(&average_ranks(&vals).0);
            md.push_str(&format!(": Friedman chi2 {} (df {}, p {})", cell(f.chi2), f.df, cell(f.p)));
        } else {
            md.push_str(": skipped");
        }
        md.push_str(".\n\n");
    }
    let metric_names: Vec<&str> = rows[0].metrics.iter().map(|(m, _)| m.as_str()).collect();
    md.push_str(&format!("| row | label | gates | front | avg rank | pick | {} |", metric_names.join(" | ")));
    if !t.occam.is_empty() {
        md.push_str(" occam |");
    }
    md.push('\n');
    md.push_str(&format!("|{}\n", "---|".repeat(6 + metric_names.len() + usize::from(!t.occam.is_empty()))));
    for i in 0..rows_n {
        let gates = if applied[i].is_empty() {
            "pass".to_string()
        } else {
            let g = rule.relaxed_gates.as_ref().filter(|_| relaxed.is_some()).unwrap_or(&rule.gates);
            format!("fail: {}", applied[i].iter().map(|&(k, _)| g[k].metric.as_str()).collect::<Vec<_>>().join(", "))
        };
        let pick = match (pick_of[i], skip_of[i]) {
            (Some(k), _) => k.to_string(),
            (None, Some(Skip::Diversity(p, c))) => format!("skip (rho {} with row {})", cell(c), p),
            (None, Some(Skip::Composites(limit))) => format!("skip (max_composites {})", limit),
            _ => "-".to_string(),
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            i,
            entries[i].wf.label,
            gates,
            front[i].map_or("-".to_string(), |f| f.to_string()),
            avg_rank[i].map_or("-".to_string(), |a| format!("{:.2}", a)),
            pick,
            rows[i].metrics.iter().map(|(m, v)| metric_cell(m, v)).collect::<Vec<_>>().join(" | ")
        ));
        if !t.occam.is_empty() {
            md.push_str(&format!(" {} |", occam_cell(&entries[i], &occam)));
        }
        md.push('\n');
    }
    md.push('\n');
    if let Some(cal) = &calendar {
        let (names, _) = cal.assign(&first.wf.time);
        md.push_str("## Strata\n\nAnnualised net, net minus buy-and-hold net and alpha over exposure-matched buy-and-hold per calendar label.\n\n");
        md.push_str(&format!("| row | label | {} |\n", names.iter().map(|l| format!("{l} net | {l} excess | {l} alpha")).collect::<Vec<_>>().join(" | ")));
        md.push_str(&format!("|{}\n", "---|".repeat(2 + 3 * names.len())));
        for i in 0..rows_n {
            let cells: Vec<String> = rows[i].strata.iter().map(|c| format!("{} | {} | {}", cell(c.net), cell(c.excess.num()), cell(c.alpha.num()))).collect();
            md.push_str(&format!("| {} | {} | {} |\n", i, entries[i].wf.label, cells.join(" | ")));
        }
        md.push('\n');
    }
    md.push_str("## Picks\n\n");
    if picks.is_empty() {
        md.push_str("No row is picked.\n\n");
    } else {
        md.push_str("| pick | row | label | Sharpe | PSR | DSR N_eff | DSR 10 N_eff | DSR nominal | MinBTL years | HL Sharpe (BHY) | t | StepM | BH-FDR |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (k, &r) in picks.iter().enumerate() {
            let Some(v) = vpos[r] else {
                md.push_str(&format!("| {} | {} | {} | constant series | | | | | | | | | |\n", k + 1, r, entries[r].wf.label));
                continue;
            };
            let s = &stats[v];
            let var_sr = var_of(s);
            let d = |trials_n: f64| inference::deflated_sharpe(s.sr, s.t, s.skew, s.kurt, trials_n, var_sr).dsr;
            let sharpe = s.sr * ann_scale;
            let t = t_of(r);
            let t_adj = if p_bhy[r] <= p_hl[r] { t.abs() } else { -crate::inference::normal_quantile(p_bhy[r] / 2.0) };
            let hl = if t.abs() > 0.0 { sharpe * t_adj / t.abs() } else { f64::NAN };
            let yes = |b: Option<bool>| b.map_or("n/a", |x| if x { "reject" } else { "keep" });
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                k + 1,
                r,
                entries[r].wf.label,
                cell(sharpe),
                cell(d(1.0)),
                cell(d(n_eff_total)),
                cell(d(10.0 * n_eff_total)),
                cell(d(nominal_total)),
                cell(inference::min_btl_years(sharpe, n_eff_total)),
                cell(hl),
                cell(t),
                yes(stepm.as_ref().ok().map(|s| s.rejected[v])),
                yes(bh.as_ref().map(|b| b.rejected[v]))
            ));
        }
        md.push('\n');
    }
    if let (Some(fr), Some(m)) = (&rule.freeze, &map) {
        md.push_str(&format!(
            "## Frozen thresholds\n\nRule sha256 `{}`, map `{}` sha256 `{}`. Nearest-rank percentiles over the rolling windows of every \
CPCV({}, {}) path of the pick's configuration at {} bp: S4 floor P{} of {}-day annualised nets, P1 floor P{} of {}-day annualised nets, \
D9 cap max({}, P{} of {}-day max drawdowns).\n\n",
            rule_sha,
            m.path,
            m.sha256,
            fr.groups,
            fr.test_groups,
            jnum(fr.cost_bps),
            fr.s4.percentile,
            fr.s4.days,
            fr.p1.percentile,
            fr.p1.days,
            jnum(fr.cap_floor),
            fr.cap.percentile,
            fr.cap.days
        ));
        md.push_str("| pick | row | label | configuration | paths | S4 floor | P1 floor | D9 cap | run sha256 |\n|---|---|---|---|---|---|---|---|---|\n");
        for (k, (&r, f)) in picks.iter().zip(&frozen).enumerate() {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                k + 1,
                r,
                entries[r].wf.label,
                f.configuration,
                f.paths,
                cell(f.s4_floor),
                cell(f.p1_floor),
                cell(f.drawdown_cap),
                f.run_sha256.as_deref().unwrap_or("-")
            ));
        }
        md.push('\n');
    }
    let tests_line = format!(
        "SPA against {}: {}. StepM at alpha {}: {}. BH-FDR at q {}: {}. PBO: {}.\n\n",
        bench_name,
        Json::parse(&spa_json).and_then(|j| num_at(&j, &["p_consistent"])).map_or("skipped".to_string(), |p| format!("p {}", cell(p))),
        jnum(alpha),
        stepm.as_ref().map_or("skipped".to_string(), |s| format!("{} of {} rejected", s.rejected.iter().filter(|&&x| x).count(), s.models)),
        jnum(alpha),
        bh.as_ref().map_or("skipped".to_string(), |b| format!("{} of {} rejected", b.n_rejected, b.rejected.len())),
        Json::parse(&pbo_json).and_then(|j| num_at(&j, &["value"])).map_or("skipped".to_string(), cell)
    );
    md.push_str(&tests_line);
    if !notes.is_empty() {
        md.push_str("Notes:\n\n");
        for x in &notes {
            md.push_str(&format!("- {}\n", x));
        }
    }
    Ok(Out { json, markdown: md })
}

/// `rcsr trials --table ...`.
pub fn cli(args: &[String]) -> Res<()> {
    let (t, rest) = split_args(args)?;
    let o = trials::parse_args(&rest)?;
    if o.compare.is_some() {
        return usage("--table needs --wf, not --compare");
    }
    let (rule, sha, source) = build_rule(&t)?;
    match (&rule.freeze, &t.freeze_map) {
        (Some(_), None) => return usage("the rule has freeze: give --freeze-map <map.json> (rcsr-trials-freeze-map/1) with the CPCV paths of every row that may be picked"),
        (None, Some(_)) => return usage("--freeze-map needs a rule with freeze (a --rule file with the freeze key)"),
        _ => {}
    }
    let out = table(&o, &t, &rule, &sha, &source)?;
    match &o.out {
        Some(path) => std::fs::write(path, &out.json).map_err(|e| (3, format!("{}: {}", path, e)))?,
        None => print!("{}", out.json),
    }
    if let Some(path) = &t.markdown {
        std::fs::write(path, &out.markdown).map_err(|e| (3, format!("{}: {}", path, e)))?;
    }
    Ok(())
}
