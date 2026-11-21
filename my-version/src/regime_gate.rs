//! #1180: the causal regime gate.
//!
//! A gate is an expression of columns of the row, read with hysteresis and a
//! minimum dwell per group. Its state per row is the regime mask of
//! `--pnl-mask`; the composition and the gate search read the same function, so
//! the switch a specialist is trained under is the switch the composite uses.
//!
//! Causality: the value of row `t` reads only the columns of row `t`, and the
//! state of row `t` reads only the values of rows `0..=t` of its group. An
//! aggregate, meta or building-block operator is refused: an aggregate reads a
//! whole column, the future of every row included.
//!
//! Book semantics at the mask edges (see [`crate::trading::FitnessBook`]): a row
//! with state off is flat; every maximal run of on rows of a group is its own
//! book that starts flat and pays its exit at the run's last row.

use crate::data::TradingSlice;
use crate::expr::{self, Node};
use crate::trading::{self, PathStats, TradingSpec};
use std::collections::HashMap;

/// A gate: expression source, the columns it reads, hysteresis and dwell.
#[derive(Clone, Debug, PartialEq)]
pub struct GateSpec {
    pub source: String,
    /// Column names in the order of the expression's inputs.
    pub columns: Vec<String>,
    pub tree: Node,
    /// Hysteresis half-width: on above `+band`, off below `-band`, else the
    /// previous state. 0 is the plain sign gate.
    pub band: f64,
    /// Minimum rows per state, per group, from the group's first row (>= 1).
    pub dwell: usize,
}

/// Names of the columns an expression reads, in order of first use. A name
/// followed by `(` is a function; `pi`, `e` and the reserved words are not
/// columns.
pub fn referenced_columns(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                }
            }
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let name = &src[start..i];
            let mut j = i;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            let call = b.get(j) == Some(&b'(');
            let reserved = matches!(name, "pi" | "e" | "and" | "or" | "xor" | "not");
            if !call && !reserved && !out.iter().any(|n| n == name) {
                out.push(name.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn reads_whole_columns(n: &Node) -> bool {
    n.op.is_agg() || n.op.is_meta() || n.op.is_custom() || n.kids.iter().any(reads_whole_columns)
}

impl GateSpec {
    pub fn parse(source: &str, band: f64, dwell: usize) -> Result<GateSpec, String> {
        if !band.is_finite() || band < 0.0 {
            return Err(format!("the gate band must be finite and >= 0, got {}", band));
        }
        if dwell == 0 {
            return Err("the gate dwell must be at least 1 row".to_string());
        }
        let columns = referenced_columns(source);
        let tree = crate::parse::parse(source, &columns).map_err(|e| format!("cannot parse the gate expression '{}': {}", source, e))?;
        if reads_whole_columns(&tree) {
            return Err(format!(
                "the gate expression '{}' uses an aggregate, meta or building-block operator: an aggregate reads the whole column, future rows included, so the gate would not be causal",
                source
            ));
        }
        Ok(GateSpec { source: source.to_string(), columns, tree, band, dwell })
    }

    /// Text that identifies the gate in a book hash.
    pub fn identity(&self) -> String {
        format!("{}|{:016x}|{}", self.source, self.band.to_bits(), self.dwell)
    }

    /// The gate value of every row from the column values (one vector per
    /// [`GateSpec::columns`] entry, all of length `n`).
    pub fn values(&self, columns: &[Vec<f64>], n: usize) -> Result<Vec<f64>, String> {
        if columns.len() != self.columns.len() || columns.iter().any(|c| c.len() != n) {
            return Err("the gate columns do not match the rows".to_string());
        }
        let mut row = vec![0.0f64; columns.len()];
        Ok((0..n)
            .map(|i| {
                for (k, c) in columns.iter().enumerate() {
                    row[k] = c[i];
                }
                expr::eval(&self.tree, &row)
            })
            .collect())
    }
}

/// The gate state of every row. Rows of one group must be in time order;
/// groups may interleave. State and dwell start at the group's first row.
pub fn gate_states(spec: &GateSpec, group: &[u64], time: &[f64], columns: &[Vec<f64>]) -> Result<Vec<bool>, String> {
    let n = group.len();
    if time.len() != n {
        return Err(format!("{} times for {} rows", time.len(), n));
    }
    let values = spec.values(columns, n)?;
    let mut last: HashMap<u64, f64> = HashMap::new();
    for i in 0..n {
        if let Some(prev) = last.insert(group[i], time[i]) {
            if time[i] < prev {
                return Err(format!("rows of group {} are not in time order at row {}", group[i], i));
            }
        }
    }
    Ok(hysteresis_states(&values, group, spec.band, spec.dwell))
}

/// Hysteresis and dwell over a value series, state per group.
///
/// Raw target: with `band == 0`, on iff value > 0; otherwise on above `+band`,
/// off below `-band`, else the previous state. A non-finite value is off. A
/// group starts off; its first row takes the raw target at once. After that a
/// state change waits until the current state has held `dwell` rows (the entry
/// row counts).
pub fn hysteresis_states(values: &[f64], group: &[u64], band: f64, dwell: usize) -> Vec<bool> {
    let mut state: HashMap<u64, (bool, usize)> = HashMap::new();
    let mut out = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let next = transition(state.get(&group[i]).copied(), v, band, dwell);
        state.insert(group[i], next);
        out.push(next.0);
    }
    out
}

/// #1182: [`hysteresis_states`] over the values `value(i)` with the groups
/// numbered `0..groups`, into `out`; a gate search prices many cuts of one gate.
pub fn hysteresis_states_dense(value: impl Fn(usize) -> f64, group: &[u32], groups: usize, band: f64, dwell: usize, out: &mut [bool]) {
    let mut state: Vec<Option<(bool, usize)>> = vec![None; groups];
    for (i, o) in out.iter_mut().enumerate() {
        let g = group[i] as usize;
        let next = transition(state[g], value(i), band, dwell);
        state[g] = Some(next);
        *o = next.0;
    }
}

/// One step of the kernel: `(state, rows held)` after a row of value `value`,
/// from the group's `(state, rows held)` before it (`None` at its first row).
fn transition(prev: Option<(bool, usize)>, value: f64, band: f64, dwell: usize) -> (bool, usize) {
    let held = prev.map_or(false, |(s, _)| s);
    let target = if !value.is_finite() {
        false
    } else if band == 0.0 {
        value > 0.0
    } else if value > band {
        true
    } else if value < -band {
        false
    } else {
        held
    };
    match prev {
        None => (target, 1),
        Some((s, age)) if target != s && age >= dwell => (target, 1),
        Some((s, age)) => (s, age + 1),
    }
}

// ---------------------------------------------------------------- rcsr-gate-state/1

/// #1214: the evaluation rule of every out-of-sample and selection book. The
/// gate rows are every row of the data file with a finite time, per group in
/// file order; a row inside a barrier is not read, and the state restarts at a
/// group's first readable row after a barrier. A row's state is a property of
/// the row, the data file and the barrier set.
pub const STATE_RULE: &str = "rcsr-gate-state/1";
/// The schema of a gate object in a run.json.
pub const GATE_SCHEMA: &str = "rcsr-gate/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierKind {
    /// A locked range the command does not open, with its lookback embargo.
    Locked,
    /// A test interval of the split with the split's embargo (selection side).
    Test,
}

impl BarrierKind {
    pub fn name(self) -> &'static str {
        match self {
            BarrierKind::Locked => "locked",
            BarrierKind::Test => "test",
        }
    }
}

/// A half-open time range `[start, end)` whose gate rows are not read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Barrier {
    pub start: f64,
    pub end: f64,
    pub kind: BarrierKind,
}

impl Barrier {
    pub fn contains(&self, t: f64) -> bool {
        t >= self.start && t < self.end
    }

    pub fn json(&self) -> String {
        let at = |v: f64| if v.is_finite() { crate::devlog::num(v) } else { "null".to_string() };
        format!("{{\"start\":{},\"end\":{},\"kind\":\"{}\"}}", at(self.start), at(self.end), self.kind.name())
    }
}

/// The gate state carried into a slice or window: what the warm-up read.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WarmUp {
    /// Readable gate rows since each group's last restart, before the start.
    pub rows: usize,
    pub first_time: Option<f64>,
    pub last_time: Option<f64>,
    /// Groups that carry a state into the start.
    pub groups: usize,
    pub on: usize,
    /// Groups whose state has held fewer than `dwell` rows: a change still waits.
    pub dwell_running: usize,
}

impl WarmUp {
    pub fn json(&self) -> String {
        let t = |v: Option<f64>| v.map_or("null".to_string(), crate::devlog::num);
        format!(
            "{{\"warmup_rows\":{},\"warmup_first_time\":{},\"warmup_last_time\":{},\"groups\":{},\"on_at_start\":{},\"dwell_running_at_start\":{}}}",
            self.rows,
            t(self.first_time),
            t(self.last_time),
            self.groups,
            self.on,
            self.dwell_running
        )
    }
}

/// One group of a [`GateTape`].
#[derive(Clone, Copy, Debug)]
struct Lane {
    state: Option<(bool, usize)>,
    /// Latest time of any row of the group (the order check).
    seen: f64,
    /// Latest time of a readable row (the restart check).
    last: f64,
    rows: usize,
    first: f64,
    latest: f64,
    /// Next entry of the sorted starts this group has not crossed.
    next_start: usize,
}

impl Default for Lane {
    fn default() -> Self {
        Lane { state: None, seen: f64::NEG_INFINITY, last: f64::NEG_INFINITY, rows: 0, first: f64::NAN, latest: f64::NAN, next_start: 0 }
    }
}

/// The evaluation rule online: feed every gate row of the data file in file
/// order. Snapshots at the given start times give the warm-up of each slice or
/// window.
#[derive(Clone, Debug)]
pub struct GateTape {
    band: f64,
    dwell: usize,
    barriers: Vec<Barrier>,
    lanes: HashMap<u64, Lane>,
    rows_read: usize,
    /// Start times in ascending order and their index in the caller's order.
    starts: Vec<(f64, usize)>,
    snaps: Vec<HashMap<u64, Lane>>,
}

impl GateTape {
    pub fn new(spec: &GateSpec, barriers: Vec<Barrier>) -> GateTape {
        GateTape::with_starts(spec, barriers, &[])
    }

    /// A tape that records the state of every group just before its first row
    /// at or after each of `starts`.
    pub fn with_starts(spec: &GateSpec, barriers: Vec<Barrier>, starts: &[f64]) -> GateTape {
        let mut sorted: Vec<(f64, usize)> = starts.iter().copied().enumerate().map(|(k, t)| (t, k)).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        GateTape { band: spec.band, dwell: spec.dwell, barriers, lanes: HashMap::new(), rows_read: 0, starts: sorted, snaps: vec![HashMap::new(); starts.len()] }
    }

    /// The row lies inside a barrier.
    pub fn barred(&self, time: f64) -> bool {
        self.barriers.iter().any(|b| b.contains(time))
    }

    /// The state of the row; `None` for a row with no finite time or inside a
    /// barrier, whose value is never computed.
    pub fn step_with(&mut self, group: u64, time: f64, value: impl FnOnce() -> f64) -> Result<Option<bool>, String> {
        if !time.is_finite() {
            return Ok(None);
        }
        let barred = self.barred(time);
        let lane = self.lanes.entry(group).or_default();
        if time < lane.seen {
            return Err(format!("rows of group {} are not in time order at time {}", group, time));
        }
        lane.seen = time;
        while lane.next_start < self.starts.len() && self.starts[lane.next_start].0 <= time {
            let start = self.starts[lane.next_start].0;
            let carried = if restarts(&self.barriers, lane.last, start) { Lane::default() } else { *lane };
            self.snaps[self.starts[lane.next_start].1].insert(group, carried);
            lane.next_start += 1;
        }
        if barred {
            return Ok(None);
        }
        let last = lane.last;
        if lane.state.is_some() && restarts(&self.barriers, last, time) {
            *lane = Lane { seen: lane.seen, next_start: lane.next_start, ..Lane::default() };
        }
        let next = transition(lane.state, value(), self.band, self.dwell);
        lane.state = Some(next);
        lane.rows += 1;
        if lane.rows == 1 {
            lane.first = time;
        }
        lane.latest = time;
        lane.last = time;
        self.rows_read += 1;
        Ok(Some(next.0))
    }

    pub fn step(&mut self, group: u64, time: f64, value: f64) -> Result<Option<bool>, String> {
        self.step_with(group, time, || value)
    }

    /// `(state, rows held)` before the group's next row.
    pub fn peek(&self, group: u64) -> Option<(bool, usize)> {
        self.lanes.get(&group).and_then(|l| l.state)
    }

    /// Readable gate rows stepped so far.
    pub fn rows_read(&self) -> usize {
        self.rows_read
    }

    /// The warm-up at every start, in the caller's order: groups that never
    /// reached a start carry the state of their last row.
    pub fn warm_ups(&self) -> Vec<WarmUp> {
        let mut out = vec![WarmUp::default(); self.snaps.len()];
        let mut at = vec![0.0f64; self.snaps.len()];
        for &(t, k) in &self.starts {
            at[k] = t;
        }
        for (k, snap) in self.snaps.iter().enumerate() {
            let mut lanes: Vec<(u64, Lane)> = snap.iter().map(|(&g, &l)| (g, l)).collect();
            for (&g, l) in &self.lanes {
                if !snap.contains_key(&g) && !restarts(&self.barriers, l.last, at[k]) {
                    lanes.push((g, *l));
                }
            }
            let w = &mut out[k];
            for (_, l) in lanes {
                let Some((on, age)) = l.state else { continue };
                w.groups += 1;
                w.rows += l.rows;
                w.on += usize::from(on);
                w.dwell_running += usize::from(age < self.dwell);
                w.first_time = Some(w.first_time.map_or(l.first, |f| f.min(l.first)));
                w.last_time = Some(w.last_time.map_or(l.latest, |f| f.max(l.latest)));
            }
        }
        out
    }
}

/// A barrier starts after the readable row at `last` and at or before `t`: the
/// state at `t` restarts.
fn restarts(barriers: &[Barrier], last: f64, t: f64) -> bool {
    barriers.iter().any(|b| last < b.start && b.start <= t)
}

/// The evaluation rule in batch form: the states of the rows `want` (indices
/// into the gate rows, which are in file order). A wanted row inside a barrier
/// or without a finite time is an error.
pub fn prefix_states(spec: &GateSpec, group: &[u64], time: &[f64], columns: &[Vec<f64>], barriers: &[Barrier], want: &[usize]) -> Result<Vec<bool>, String> {
    let n = group.len();
    if time.len() != n || columns.len() != spec.columns.len() || columns.iter().any(|c| c.len() != n) {
        return Err("the gate columns do not match the rows".to_string());
    }
    let mut tape = GateTape::new(spec, barriers.to_vec());
    let mut states: Vec<Option<bool>> = vec![None; n];
    let mut row = vec![0.0f64; columns.len()];
    for i in 0..n {
        states[i] = tape.step_with(group[i], time[i], || {
            for (k, c) in columns.iter().enumerate() {
                row[k] = c[i];
            }
            expr::eval(&spec.tree, &row)
        })?;
    }
    want.iter().map(|&i| states.get(i).copied().flatten().ok_or_else(|| format!("row {} is not a readable gate row", i))).collect()
}

/// D13 for gate columns (#1180, #1214): the message of the exit 5 refusal for
/// the first gate column that is a target, a label-family column or a
/// construction column of a declared target; `None` when every column is clean.
pub fn gate_column_leak(columns: &[String], specs: &[&crate::target::TargetSpec], target_names: &[String], allow_structure: bool) -> Option<String> {
    for c in columns {
        if target_names.iter().any(|t| t == c) {
            return Some(format!("D13a: target '{}' is a mask column", c));
        }
        if crate::registry::lookup(c).is_some_and(|r| r.family == "label") {
            return Some(format!("D13a: label-family column '{}' reads bars after the decision time", c));
        }
    }
    for spec in specs {
        match crate::target::check(spec, columns, allow_structure) {
            Ok(check) if !check.construction_in_vars.is_empty() => {
                return Some(format!("D13: construction column(s) of target '{}' are mask columns: {}", spec.column, check.construction_in_vars.join(", ")))
            }
            Ok(_) => {}
            Err(report) => return Some(report.result().err().map_or_else(|| "D13 finding".to_string(), |f| f.to_string())),
        }
    }
    None
}

/// The gate object of a run.json (`rcsr-gate/1`).
pub fn gate_json(g: &GateSpec) -> String {
    format!(
        "{{\"schema\":\"{}\",\"expr\":{},\"columns\":[{}],\"band\":{},\"dwell\":{},\"state_rule\":\"{}\"}}",
        GATE_SCHEMA,
        crate::devlog::text(&g.source),
        g.columns.iter().map(|c| crate::devlog::text(c)).collect::<Vec<_>>().join(","),
        crate::devlog::num(g.band),
        g.dwell,
        STATE_RULE
    )
}

/// #1214: a C1 or C2 composite: the book holds `on` where the gate state is on,
/// else `off`, and is long where the held branch is above 0. Not an expression:
/// every row evaluator assumes a row-pure tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Switch {
    pub gate: GateSpec,
    pub on: String,
    pub off: String,
}

impl Switch {
    pub fn json(&self) -> String {
        format!("{{\"gate\":{},\"on\":{},\"off\":{}}}", gate_json(&self.gate), crate::devlog::text(&self.on), crate::devlog::text(&self.off))
    }

    /// A `switch` object of a run.json.
    pub fn parse(j: &crate::devlog::Json) -> Result<Switch, String> {
        use crate::devlog::Json;
        let g = j.get("gate").ok_or("switch has no gate")?;
        if g.get("schema").and_then(Json::as_str) != Some(GATE_SCHEMA) {
            return Err(format!("switch.gate needs \"schema\":\"{}\"", GATE_SCHEMA));
        }
        match g.get("state_rule").and_then(Json::as_str) {
            Some(STATE_RULE) => {}
            other => return Err(format!("switch.gate.state_rule {:?} is not {}", other, STATE_RULE)),
        }
        let expr = g.get("expr").and_then(Json::as_str).ok_or("switch.gate has no expr")?;
        let band = g.get("band").and_then(Json::as_f64).ok_or("switch.gate has no band")?;
        let dwell = g.get("dwell").and_then(Json::as_f64).filter(|d| *d >= 1.0 && d.fract() == 0.0 && *d < 1e9).ok_or("switch.gate has no integer dwell >= 1")?;
        let gate = GateSpec::parse(expr, band, dwell as usize).map_err(|e| format!("switch.gate: {}", e))?;
        if let Some(Json::Arr(cols)) = g.get("columns") {
            let listed: Vec<&str> = cols.iter().filter_map(Json::as_str).collect();
            if listed != gate.columns.iter().map(String::as_str).collect::<Vec<_>>() {
                return Err(format!("switch.gate.columns {:?} are not the columns of '{}' ({:?})", listed, expr, gate.columns));
            }
        }
        let side = |key: &str| j.get(key).and_then(Json::as_str).map(str::to_string).ok_or_else(|| format!("switch has no '{}' branch", key));
        Ok(Switch { gate, on: side("on")?, off: side("off")? })
    }

    /// The procedure-spec text of the switch (#1214 seal binding).
    pub fn identity(&self) -> String {
        format!("{}|{}|{:016x}|{}|on={}|off={}", STATE_RULE, self.gate.source, self.gate.band.to_bits(), self.gate.dwell, self.on, self.off)
    }

    /// Columns of the gate and both branches, gate columns first.
    pub fn columns(&self) -> Vec<String> {
        let mut out = self.gate.columns.clone();
        for c in referenced_columns(&self.on).into_iter().chain(referenced_columns(&self.off)) {
            if !out.contains(&c) {
                out.push(c);
            }
        }
        out
    }
}

/// #1278: the group of every row of a masked book, in row order: a distinct id
/// for every maximal run of priced on rows of a group and for every priced off
/// row. A row that is not priced (no finite return: an unlabelled row of a
/// strided label) neither starts nor ends a run, whatever its state; it takes
/// the id of the run or off row before it in its group. Rows of one group may
/// interleave with other groups.
pub fn masked_groups(group: &[u64], on: &[bool], priced: &[bool]) -> Vec<u64> {
    let mut open: HashMap<u64, bool> = HashMap::new();
    let mut current: HashMap<u64, u64> = HashMap::new();
    let mut next = 0u64;
    let mut out = Vec::with_capacity(group.len());
    for i in 0..group.len() {
        if !priced[i] {
            let id = *current.entry(group[i]).or_insert_with(|| {
                next += 1;
                next - 1
            });
            out.push(id);
            continue;
        }
        let continues = on[i] && open.get(&group[i]).copied().unwrap_or(false);
        if !continues {
            current.insert(group[i], next);
            next += 1;
        }
        open.insert(group[i], on[i]);
        out.push(current[&group[i]]);
    }
    out
}

/// #1278: the rows a book prices: a finite return, time and funding.
pub fn priced_rows(slice: &TradingSlice) -> Vec<bool> {
    let r = &slice.rows;
    (0..slice.ret.len()).map(|i| slice.ret[i].is_finite() && r.time[i].is_finite() && (r.funding.is_empty() || r.funding[i].is_finite())).collect()
}

/// Priced rows on, and maximal on-runs per group ([`masked_groups`]).
pub fn run_count(group: &[u64], on: &[bool], priced: &[bool]) -> (usize, usize) {
    let ids = masked_groups(group, on, priced);
    let mut seen = std::collections::HashSet::new();
    let mut rows = 0;
    for i in 0..on.len() {
        if on[i] && priced[i] {
            rows += 1;
            seen.insert(ids[i]);
        }
    }
    (rows, seen.len())
}

/// State changes of a state series and the length of its runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Switches {
    pub periods: usize,
    pub switches: usize,
    /// `switches` per year of the `periods - 1` intervals between the states.
    pub per_year: f64,
    pub mean_on_run: f64,
    pub mean_off_run: f64,
}

/// Switch statistics of a period state series in time order. The observation
/// span is the `periods - 1` intervals between the states, so a gate that
/// flips every 24 hourly periods over 8760 intervals switches 365 times a year.
pub fn switch_stats(states: &[bool], periods_per_year: f64) -> Switches {
    let n = states.len();
    let switches = states.windows(2).filter(|w| w[0] != w[1]).count();
    let per_year = if n >= 2 && periods_per_year > 0.0 {
        switches as f64 * periods_per_year / (n - 1) as f64
    } else {
        f64::NAN
    };
    let (mut on_runs, mut on_len, mut off_runs, mut off_len) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && states[j] == states[i] {
            j += 1;
        }
        if states[i] {
            on_runs += 1;
            on_len += j - i;
        } else {
            off_runs += 1;
            off_len += j - i;
        }
        i = j;
    }
    let mean = |len: usize, runs: usize| if runs == 0 { f64::NAN } else { len as f64 / runs as f64 };
    Switches { periods: n, switches, per_year, mean_on_run: mean(on_len, on_runs), mean_off_run: mean(off_len, off_runs) }
}

/// The slice priced under a mask: every on-run is its own group (flat start,
/// exit paid at the run's last row) and the return of an off row is NaN, so the
/// pricing drops it as an unusable row. #1278: the runs are those of
/// [`masked_groups`] over the priced rows, the groups walk-forward prices.
pub fn masked_slice(slice: &TradingSlice, on: &[bool]) -> TradingSlice {
    let mut out = slice.clone();
    if on.iter().all(|&o| o) {
        return out;
    }
    out.rows.group = masked_groups(&slice.rows.group, on, &priced_rows(slice));
    for (r, &keep) in out.ret.iter_mut().zip(on) {
        if !keep {
            *r = f64::NAN;
        }
    }
    out
}

// ---------------------------------------------------------------- the state view of a book

/// Net, exposure and buy-and-hold of the rows of one gate state.
#[derive(Clone, Debug)]
pub struct StatePart {
    pub rows: usize,
    pub net: f64,
    pub exposure: f64,
    pub bh_net: f64,
    pub alpha: f64,
}

/// The gate-state view of a composite book on one slice.
#[derive(Clone, Debug)]
pub struct Regime {
    pub rows: usize,
    pub on: StatePart,
    pub off: StatePart,
    pub seam: f64,
    pub switches: Switches,
    pub switch_cost: f64,
    pub total_cost: f64,
}

/// A score the book holds flat, and one it holds long.
fn flat_score(spec: &TradingSpec) -> f64 {
    spec.threshold - 1.0 - spec.threshold.abs()
}

fn long_score(spec: &TradingSpec) -> f64 {
    spec.threshold + 1.0 + spec.threshold.abs()
}

/// Splits the book of `pred` on `slice` by gate state. The book of a state
/// holds the composite's score on the rows of that state and a flat score
/// elsewhere, with the composite's missing scores, priced by the book code;
/// `seam` is the composite net minus both (`2 side J` of the seam identity). A
/// state's alpha is its net minus its exposure times the net of the
/// buy-and-hold book of its rows. A period is in the state most of its priced
/// rows hold; the switch cost is the cost of the position changes in the
/// periods where that state changes.
pub fn regime(pred: &[f64], on: &[bool], slice: &TradingSlice, spec: &TradingSpec, book: &PathStats) -> Result<Regime, String> {
    let r = &slice.rows;
    let mut cand: Vec<usize> = (0..pred.len())
        .filter(|&i| slice.ret[i].is_finite() && r.time[i].is_finite() && (r.funding.is_empty() || r.funding[i].is_finite()))
        .collect();
    cand.sort_by(|&a, &b| r.group[a].cmp(&r.group[b]).then(r.time[a].total_cmp(&r.time[b])).then(a.cmp(&b)));
    let books: Vec<(u64, usize)> = cand.iter().map(|&i| (r.group[i], spec.segment_starts.partition_point(|s| *s <= r.time[i]))).collect();
    let scored: Vec<bool> = cand.iter().map(|&i| pred[i].is_finite()).collect();
    let runs = trading::score_runs(&books, &scored, spec.missing_score);
    let mut pos: Vec<f64> = cand.iter().map(|&i| if pred[i].is_finite() { trading::position_of(pred[i], spec) } else { 0.0 }).collect();
    trading::hold_positions(&books, &scored, &mut pos, |_| 0.0);
    let priced: Vec<usize> = (0..cand.len()).filter(|&k| runs.run[k].is_some()).collect();
    if priced.is_empty() {
        return Err("no usable rows".to_string());
    }
    let n = priced.len();
    let (flat, long) = (flat_score(spec), long_score(spec));
    let part = |state: bool| -> Result<StatePart, String> {
        let rows = priced.iter().filter(|&&k| on[cand[k]] == state).count();
        if rows == 0 {
            return Ok(StatePart { rows: 0, net: 0.0, exposure: f64::NAN, bh_net: 0.0, alpha: 0.0 });
        }
        let pick = |v: f64| -> Vec<f64> { (0..pred.len()).map(|i| if !pred[i].is_finite() { pred[i] } else if on[i] == state { if v.is_nan() { pred[i] } else { v } } else { flat }).collect() };
        let net = trading::evaluate(&pick(f64::NAN), &slice.ret, r, spec)?.net_pnl;
        let bh_net = trading::evaluate(&pick(long), &slice.ret, r, spec)?.net_pnl;
        let exposure = priced.iter().filter(|&&k| on[cand[k]] == state).map(|&k| pos[k]).sum::<f64>() / rows as f64;
        Ok(StatePart { rows, net, exposure, bh_net, alpha: net - exposure * bh_net })
    };
    let (on_part, off_part) = (part(true)?, part(false)?);
    let bucket = spec.period_seconds.filter(|s| *s > 0.0);
    let key = |t: f64| bucket.map_or(t, |s| (t / s).floor());
    let row_key: Vec<f64> = priced.iter().map(|&k| key(r.time[cand[k]])).collect();
    let st: Vec<bool> = priced.iter().map(|&k| on[cand[k]]).collect();
    let mut period_key = row_key.clone();
    period_key.sort_by(|a, b| a.total_cmp(b));
    period_key.dedup_by(|a, b| *a == *b);
    let (states, _) = crate::regimes::period_states(&row_key, &st, &vec![true; n], &period_key);
    let switches = switch_stats(&states, book.periods_per_year);
    let flip_at = |k: f64| {
        let p = period_key.partition_point(|x| *x < k);
        p > 0 && p < states.len() && states[p] != states[p - 1]
    };
    let side = spec.per_side_bps() * 1e-4;
    let mut prev: HashMap<(u64, usize, u32), f64> = HashMap::new();
    let mut switch_cost = 0.0f64;
    for (j, &k) in priced.iter().enumerate() {
        let run = runs.run[k].expect("a priced row has a run");
        let p = prev.insert((books[k].0, books[k].1, run), pos[k]).unwrap_or(0.0);
        if flip_at(row_key[j]) {
            switch_cost += side * (pos[k] - p).abs();
        }
    }
    Ok(Regime {
        rows: n,
        seam: book.net_pnl - on_part.net - off_part.net,
        on: on_part,
        off: off_part,
        switches,
        switch_cost,
        total_cost: book.total_cost,
    })
}

fn part_json(p: &StatePart, rows: usize) -> String {
    format!(
        "{{\"rows\":{},\"share\":{},\"net\":{},\"exposure\":{},\"buy_and_hold_net\":{},\"alpha\":{}}}",
        p.rows,
        crate::devlog::num(p.rows as f64 / rows.max(1) as f64),
        crate::devlog::num(p.net),
        crate::devlog::num(p.exposure),
        crate::devlog::num(p.bh_net),
        crate::devlog::num(p.alpha)
    )
}

pub fn regime_json(r: &Regime) -> String {
    format!(
        "{{\"rows\":{},\"states\":{{\"on\":{},\"off\":{}}},\"seam\":{},\"switches\":{{\"periods\":{},\"switches\":{},\"per_year\":{},\"mean_on_periods\":{},\"mean_off_periods\":{}}},\
         \"switch_cost\":{},\"total_cost\":{},\"switch_cost_share\":{}}}",
        r.rows,
        part_json(&r.on, r.rows),
        part_json(&r.off, r.rows),
        crate::devlog::num(r.seam),
        r.switches.periods,
        r.switches.switches,
        crate::devlog::num(r.switches.per_year),
        crate::devlog::num(r.switches.mean_on_run),
        crate::devlog::num(r.switches.mean_off_run),
        crate::devlog::num(r.switch_cost),
        crate::devlog::num(r.total_cost),
        if r.total_cost > 0.0 { crate::devlog::num(r.switch_cost / r.total_cost) } else { "null".to_string() }
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_names_skip_functions_numbers_and_constants() {
        assert_eq!(referenced_columns("btc_r672"), vec!["btc_r672"]);
        assert_eq!(referenced_columns("0 - btc_r672"), vec!["btc_r672"]);
        assert_eq!(referenced_columns("greater(a, 1e-3) and not b"), vec!["a", "b"]);
        assert_eq!(referenced_columns("max(x1, x2) + pi * e"), vec!["x1", "x2"]);
        assert!(referenced_columns("1").is_empty());
    }

    #[test]
    fn the_gate_refuses_aggregates() {
        assert!(GateSpec::parse("mean(a)", 0.0, 1).is_err());
        assert!(GateSpec::parse("a - mean(a)", 0.0, 1).is_err());
        assert!(GateSpec::parse("a", -1.0, 1).is_err());
        assert!(GateSpec::parse("a", 0.0, 0).is_err());
        assert!(GateSpec::parse("a +", 0.0, 1).is_err());
    }

    #[test]
    fn gate_hysteresis_and_dwell() {
        let g = vec![0u64; 12];
        // band 0.02, dwell 3. Row 0 starts on at once; row 1 (-0.5) wants off
        // but the state has held 1 row; rows 1 and 2 hold; row 3 flips off.
        let v = [0.5, -0.5, -0.5, -0.5, 0.01, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.0];
        let s = hysteresis_states(&v, &g, 0.02, 3);
        let expect = [true, true, true, false, false, false, true, true, true, false, false, false];
        assert_eq!(s, expect);
        // no dwell, no band: the plain sign, non-finite off
        let v = [1.0, -1.0, 0.0, f64::NAN, 2.0];
        assert_eq!(hysteresis_states(&v, &[0; 5], 0.0, 1), vec![true, false, false, false, true]);
        // the band keeps the previous state inside +-band
        let v = [0.5, 0.01, -0.01, -0.5, -0.01, 0.01, 0.5];
        assert_eq!(hysteresis_states(&v, &[0; 7], 0.02, 1), vec![true, true, true, false, false, false, true]);
    }

    #[test]
    fn dwell_and_state_are_per_group() {
        let g = [1u64, 2, 1, 2, 1, 2];
        let v = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        // dwell 2: each group keeps its first state for 2 rows and flips at its third row
        let s = hysteresis_states(&v, &g, 0.0, 2);
        assert_eq!(s, vec![true, false, true, false, false, true]);
    }

    #[test]
    fn dense_groups_give_the_hashed_kernel_states() {
        let ids = [7u64, 3, 7, 9, 3, 3, 7, 9, 9, 7, 3, 7];
        let dense: Vec<u32> = ids.iter().map(|g| [7u64, 3, 9].iter().position(|h| h == g).unwrap() as u32).collect();
        let v = [0.5, -0.03, -0.5, 0.01, f64::NAN, 0.4, 0.0, -0.2, 0.3, 0.02, -0.6, -0.01];
        for (band, dwell) in [(0.0, 1), (0.02, 1), (0.0, 3), (0.02, 2)] {
            let mut out = vec![false; v.len()];
            hysteresis_states_dense(|i| v[i], &dense, 3, band, dwell, &mut out);
            assert_eq!(out, hysteresis_states(&v, &ids, band, dwell), "band {band} dwell {dwell}");
        }
    }

    #[test]
    fn the_state_of_a_row_reads_only_earlier_rows() {
        let spec = GateSpec::parse("a - b", 0.05, 4).unwrap();
        let n = 300;
        let group: Vec<u64> = (0..n).map(|i| (i % 3) as u64).collect();
        let time: Vec<f64> = (0..n).map(|i| (i / 3) as f64).collect();
        let a: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i * 13 % 97) as f64 - 48.0) / 80.0).collect();
        let full = gate_states(&spec, &group, &time, &[a.clone(), b.clone()]).unwrap();
        for k in [1usize, 7, 100, 299] {
            let part = gate_states(&spec, &group[..k], &time[..k], &[a[..k].to_vec(), b[..k].to_vec()]).unwrap();
            assert_eq!(part, full[..k].to_vec(), "prefix {}", k);
        }
    }

    #[test]
    fn a_group_out_of_time_order_is_refused() {
        let spec = GateSpec::parse("a", 0.0, 1).unwrap();
        let e = gate_states(&spec, &[0, 0], &[2.0, 1.0], &[vec![1.0, 1.0]]).unwrap_err();
        assert!(e.contains("time order"), "{}", e);
    }

    #[test]
    fn runs_split_at_off_rows_and_group_changes() {
        let g = [1u64, 1, 1, 1, 2, 2];
        let on = [true, true, false, true, true, true];
        let ids = masked_groups(&g, &on, &[true; 6]);
        assert_eq!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[2], ids[3]);
        assert_ne!(ids[3], ids[4]);
        assert_eq!(ids[4], ids[5]);
        assert_eq!(run_count(&g, &on, &[true; 6]), (5, 3));
    }

    #[test]
    fn masked_groups_skips_unlabelled_rows() {
        // on, unlabelled, on, off, on: the unlabelled row neither ends nor starts a run, whatever its state.
        for state in [true, false] {
            let ids = masked_groups(&[7; 5], &[true, state, true, false, true], &[true, false, true, true, true]);
            assert_eq!(ids, vec![0, 0, 0, 1, 2], "unlabelled row {state}");
        }
        // Two interleaved groups: group 1 is on, unlabelled, on, off; group 2 starts unlabelled.
        let g = [1u64, 2, 1, 2, 1, 2, 1, 2];
        let on = [true, false, false, true, true, false, false, true];
        let priced = [true, false, false, true, true, false, true, true];
        assert_eq!(masked_groups(&g, &on, &priced), vec![0, 1, 0, 2, 0, 2, 3, 2]);
        assert_eq!(run_count(&g, &on, &priced), (4, 2));
    }

    /// The run ids before #1278: every row counts.
    fn every_row_run_ids(group: &[u64], on: &[bool]) -> Vec<u64> {
        let mut open: HashMap<u64, bool> = HashMap::new();
        let mut current: HashMap<u64, u64> = HashMap::new();
        let mut next = 0u64;
        let mut out = Vec::with_capacity(group.len());
        for i in 0..group.len() {
            if !(on[i] && open.get(&group[i]).copied().unwrap_or(false)) {
                current.insert(group[i], next);
                next += 1;
            }
            open.insert(group[i], on[i]);
            out.push(current[&group[i]]);
        }
        out
    }

    #[test]
    fn masked_slice_unchanged_for_labelled_rows() {
        let mut seed = 0x1278u64;
        for case in 0..300 {
            let groups = 1 + case % 4;
            let n = 1 + (uniform(&mut seed) * 120.0) as usize;
            let group: Vec<u64> = (0..n).map(|_| (uniform(&mut seed) * groups as f64) as u64).collect();
            let on: Vec<bool> = (0..n).map(|_| uniform(&mut seed) < 0.6).collect();
            let slice = TradingSlice {
                rows: trading::TradingRows { time: (0..n).map(|i| i as f64).collect(), group: group.clone(), funding: Vec::new() },
                ret: (0..n).map(|_| uniform(&mut seed) - 0.5).collect(),
            };
            let book = masked_slice(&slice, &on);
            let want = if on.iter().all(|&o| o) { group.clone() } else { every_row_run_ids(&group, &on) };
            assert_eq!(book.rows.group, want, "case {case}");
            let ret: Vec<u64> = slice.ret.iter().zip(&on).map(|(r, &o)| if o { r.to_bits() } else { f64::NAN.to_bits() }).collect();
            assert_eq!(book.ret.iter().map(|r| r.to_bits()).collect::<Vec<_>>(), ret, "case {case}");
        }
    }

    #[test]
    fn switch_count_of_a_daily_flip() {
        let states: Vec<bool> = (0..=24 * 365).map(|i| (i / 24) % 2 == 0).collect();
        let s = switch_stats(&states, 8760.0);
        assert_eq!(s.switches, 365);
        assert!((s.per_year - 365.0).abs() < 1e-9, "{}", s.per_year);
        assert!((s.mean_on_run - 24.0).abs() < 0.2 && (s.mean_off_run - 24.0).abs() < 0.2);
        let flat = switch_stats(&[true, true, true], 8760.0);
        assert_eq!((flat.switches, flat.per_year), (0, 0.0));
        assert!(flat.mean_off_run.is_nan());
        assert!(switch_stats(&[true], 8760.0).per_year.is_nan());
    }

    fn uniform(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn warm_up_state_equals_the_full_prefix_state() {
        let mut seed = 0x1214u64;
        let mut inside_dwell = 0usize;
        for case in 0..1000 {
            let band = if case % 4 == 0 { 0.0 } else { uniform(&mut seed) * 0.3 };
            let dwell = 1 + (uniform(&mut seed) * 8.0) as usize;
            let groups = 1 + case % 3;
            let n = 60 + (uniform(&mut seed) * 60.0) as usize;
            let group: Vec<u64> = (0..n).map(|i| (i % groups) as u64).collect();
            let time: Vec<f64> = (0..n).map(|i| (i / groups) as f64).collect();
            let a: Vec<f64> = (0..n).map(|_| uniform(&mut seed) - 0.5).collect();
            let spec = GateSpec::parse("a", band, dwell).unwrap();
            let all: Vec<usize> = (0..n).collect();
            let full = prefix_states(&spec, &group, &time, &[a.clone()], &[], &all).unwrap();
            assert_eq!(full, hysteresis_states(&a, &group, band, dwell), "case {case}: one transition for the batch and the kernel");
            let cut = 1 + (uniform(&mut seed) * (n - 1) as f64) as usize;
            let mut tape = GateTape::new(&spec, Vec::new());
            for i in 0..cut {
                tape.step(group[i], time[i], a[i]).unwrap();
            }
            for g in 0..groups as u64 {
                let rows: Vec<usize> = (0..cut).filter(|&i| group[i] == g).collect();
                let Some(&last) = rows.last() else {
                    assert_eq!(tape.peek(g), None);
                    continue;
                };
                let held = rows.iter().rev().take_while(|&&i| full[i] == full[last]).count();
                assert_eq!(tape.peek(g), Some((full[last], held)), "case {case} group {g} cut {cut}");
                inside_dwell += usize::from(held < dwell);
            }
            for i in cut..n {
                assert_eq!(tape.step(group[i], time[i], a[i]).unwrap(), Some(full[i]), "case {case} row {i}");
            }
        }
        assert!(inside_dwell >= 200, "{inside_dwell} cuts fall inside a running dwell");
    }

    #[test]
    fn the_state_restarts_after_a_barrier() {
        let spec = GateSpec::parse("a", 0.0, 4).unwrap();
        // Off for rows 0..6, on from row 6, off again from row 11; [8, 11) is a barrier.
        let a: Vec<f64> = (0..16).map(|i| if (6..11).contains(&i) { 1.0 } else { -1.0 }).collect();
        let barrier = Barrier { start: 8.0, end: 11.0, kind: BarrierKind::Locked };
        let mut tape = GateTape::with_starts(&spec, vec![barrier], &[7.5, 11.0, 13.0]);
        let got: Vec<Option<bool>> = (0..16).map(|i| tape.step(0, i as f64, a[i]).unwrap()).collect();
        let f = Some(false);
        let t = Some(true);
        // Without the restart row 11 would hold the on state (2 rows held, dwell 4).
        assert_eq!(got, vec![f, f, f, f, f, f, t, t, None, None, None, f, f, f, f, f]);
        assert_eq!(tape.rows_read(), 13);
        let w = tape.warm_ups();
        assert_eq!(w[0], WarmUp { rows: 8, first_time: Some(0.0), last_time: Some(7.0), groups: 1, on: 1, dwell_running: 1 });
        assert_eq!(w[1], WarmUp::default(), "a window that starts at the first row after a barrier carries no state");
        assert_eq!(w[2], WarmUp { rows: 2, first_time: Some(11.0), last_time: Some(12.0), groups: 1, on: 0, dwell_running: 1 });
        // The batch form gives the same states and refuses a wanted row inside the barrier.
        let time: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let want: Vec<usize> = (0..16).filter(|i| !(8..11).contains(i)).collect();
        let batch = prefix_states(&spec, &[0; 16], &time, &[a.clone()], &[barrier], &want).unwrap();
        assert_eq!(batch, want.iter().map(|&i| got[i].unwrap()).collect::<Vec<_>>());
        assert!(prefix_states(&spec, &[0; 16], &time, &[a], &[barrier], &[9]).is_err());
        let e = GateTape::new(&spec, Vec::new()).step(0, f64::NAN, 1.0).unwrap();
        assert_eq!(e, None, "a row without a finite time is no gate row");
        let mut tape = GateTape::new(&spec, Vec::new());
        tape.step(3, 5.0, 1.0).unwrap();
        assert!(tape.step(3, 4.0, 1.0).unwrap_err().contains("group 3"));
    }

    #[test]
    fn a_switch_round_trips_through_its_json() {
        let sw = Switch { gate: GateSpec::parse("h - 0.1", 0.02, 5).unwrap(), on: "(x1) - (0.1)".to_string(), off: "-1.0".to_string() };
        let j = crate::devlog::Json::parse(&sw.json()).unwrap();
        let back = Switch::parse(&j).unwrap();
        assert_eq!((back.gate.identity(), back.on.as_str(), back.off.as_str()), (sw.gate.identity(), sw.on.as_str(), sw.off.as_str()));
        assert_eq!(sw.columns(), vec!["h", "x1"]);
        let other = sw.json().replace(STATE_RULE, "window-restart/1");
        assert!(Switch::parse(&crate::devlog::Json::parse(&other).unwrap()).unwrap_err().contains("state_rule"));
        assert_ne!(sw.identity(), Switch { off: sw.on.clone(), on: sw.off.clone(), ..sw.clone() }.identity());
    }
}
