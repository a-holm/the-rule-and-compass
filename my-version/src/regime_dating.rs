//! #1175: `rcsr protocol regimes`, the regime calendar writer.
//!
//! The rule is a threshold rule on daily closes (Pagan and Sossounov 2003;
//! Lunde and Timmermann 2004): a bull phase ends at the highest close after
//! which the price falls by `down`, a bear phase ends at the lowest close
//! after which it rises by `up`, and a phase shorter than `min-days` merges
//! into its neighbour unless its move exceeds twice the threshold. Inside a
//! phase, a span of at least `sideways-days` days whose closes stay within a
//! `sideways-band` band (max over min) is labelled `sideways`.
//!
//! Two modes, never mixed:
//!
//! * `ex-post` (default): the phases come from the whole series, so the label
//!   of a day uses closes after it. The calendar only stratifies evaluation.
//! * `causal`: the label of a day uses the closes at or before it.
//!
//! Locked ranges: a day inside a locked range, its purge before and its
//! embargo after, is dropped at read time (its price is never parsed) and the
//! span is written as `not_dated`. Only `--after-read` opens the primary range,
//! and only when the ledger holds every read its final seal allows.

use crate::protocol::{self, Ledger, LockedRange};
use crate::regimes::CALENDAR_HEADER;
use std::collections::BTreeMap;

pub const DAY: f64 = 86_400.0;
pub const NOT_DATED: &str = "not_dated";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub up: f64,
    pub down: f64,
    pub min_days: f64,
    /// 0 turns the sideways label off.
    pub sideways_days: f64,
    pub sideways_band: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Day {
    /// UTC midnight of the day.
    pub t: f64,
    pub close: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub label: String,
    pub start: f64,
    pub end: f64,
}

fn row(label: &str, start: f64, end: f64) -> Row {
    Row { label: label.to_string(), start, end }
}

/// End of the data of a segment: the day after its last day.
fn tail(days: &[Day]) -> f64 {
    days[days.len() - 1].t + DAY
}

// ---------------------------------------------------------------------------
// Ex-post dating
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Phase {
    bull: bool,
    /// Index of the first day.
    from: usize,
    /// Index of the first day of the next phase, or the number of days.
    to: usize,
}

fn turning_points(days: &[Day], p: &Params) -> Vec<(usize, bool)> {
    let c = |i: usize| days[i].close;
    let mut out = Vec::new();
    let (mut hi, mut lo) = (0usize, 0usize);
    // None until the first threshold move; then the current phase is bull or bear.
    let mut state: Option<bool> = None;
    let (mut peak, mut trough) = (0usize, 0usize);
    for i in 1..days.len() {
        match state {
            None => {
                if c(i) > c(hi) {
                    hi = i;
                }
                if c(i) < c(lo) {
                    lo = i;
                }
                if c(i) <= c(hi) * (1.0 - p.down) {
                    out.push((hi, false));
                    state = Some(false);
                    trough = i;
                } else if c(i) >= c(lo) * (1.0 + p.up) {
                    out.push((lo, true));
                    state = Some(true);
                    peak = i;
                }
            }
            Some(true) => {
                if c(i) > c(peak) {
                    peak = i;
                } else if c(i) <= c(peak) * (1.0 - p.down) {
                    out.push((peak, false));
                    state = Some(false);
                    trough = i;
                }
            }
            Some(false) => {
                if c(i) < c(trough) {
                    trough = i;
                } else if c(i) >= c(trough) * (1.0 + p.up) {
                    out.push((trough, true));
                    state = Some(true);
                    peak = i;
                }
            }
        }
    }
    out
}

fn phases_of(days: &[Day], turns: &[(usize, bool)]) -> Vec<Phase> {
    let n = days.len();
    let mut out = Vec::new();
    let Some(&(first, first_bull)) = turns.first() else { return out };
    if first > 0 {
        out.push(Phase { bull: !first_bull, from: 0, to: first });
    }
    for (k, &(at, bull)) in turns.iter().enumerate() {
        let to = turns.get(k + 1).map_or(n, |t| t.0);
        out.push(Phase { bull, from: at, to });
    }
    out
}

fn phase_start_t(days: &[Day], i: usize) -> f64 {
    days[i].t
}

fn phase_end_t(days: &[Day], to: usize) -> f64 {
    if to < days.len() { days[to].t } else { tail(days) }
}

/// The move of a phase: the rise of a bull, the fall of a bear, from its first
/// close to the close of the turning point that ends it (the last close for
/// the last phase).
fn phase_move(days: &[Day], ph: &Phase) -> f64 {
    let start = days[ph.from].close;
    let end = days[if ph.to < days.len() { ph.to } else { days.len() - 1 }].close;
    if ph.bull { end / start - 1.0 } else { 1.0 - end / start }
}

fn merge_short(days: &[Day], mut phases: Vec<Phase>, p: &Params) -> Vec<Phase> {
    loop {
        if phases.len() < 2 {
            return phases;
        }
        let short = phases.iter().position(|ph| {
            let len = (phase_end_t(days, ph.to) - phase_start_t(days, ph.from)) / DAY;
            let bar = 2.0 * if ph.bull { p.up } else { p.down };
            len < p.min_days && phase_move(days, ph) < bar
        });
        let Some(k) = short else { return phases };
        if k == 0 {
            let next = phases[1];
            phases[1] = Phase { from: phases[0].from, ..next };
            phases.remove(0);
        } else if k + 1 == phases.len() {
            let prev = phases[k - 1];
            phases[k - 1] = Phase { to: phases[k].to, ..prev };
            phases.pop();
        } else {
            let (prev, next) = (phases[k - 1], phases[k + 1]);
            phases[k - 1] = Phase { to: next.to, ..prev };
            phases.drain(k..=k + 1);
        }
    }
}

/// Spans `[i, j]` of day indexes inside `[from, to)` that last at least
/// `sideways_days` and keep max close over min close within the band. Greedy
/// from the left: the longest span from the first start that qualifies, then
/// on from the day after it.
fn sideways_spans(days: &[Day], from: usize, to: usize, p: &Params) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if p.sideways_days <= 0.0 {
        return out;
    }
    let mut i = from;
    while i < to {
        let (mut lo, mut hi) = (days[i].close, days[i].close);
        let mut j = i;
        while j + 1 < to {
            let c = days[j + 1].close;
            if c.max(hi) > c.min(lo) * (1.0 + p.sideways_band) {
                break;
            }
            lo = lo.min(c);
            hi = hi.max(c);
            j += 1;
        }
        let end = phase_end_t(days, j + 1).min(phase_end_t(days, to));
        if (end - days[i].t) / DAY >= p.sideways_days {
            out.push((i, j));
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Phases of one contiguous series, from the whole series (ex post).
pub fn date_ex_post(days: &[Day], p: &Params) -> Vec<Row> {
    if days.len() < 2 {
        return Vec::new();
    }
    let phases = merge_short(days, phases_of(days, &turning_points(days, p)), p);
    let mut rows = Vec::new();
    for ph in &phases {
        let label = if ph.bull { "bull" } else { "bear" };
        let mut at = days[ph.from].t;
        let stop = phase_end_t(days, ph.to);
        for (i, j) in sideways_spans(days, ph.from, ph.to, p) {
            let (s, e) = (days[i].t, phase_end_t(days, j + 1).min(stop));
            if s > at {
                rows.push(row(label, at, s));
            }
            rows.push(row("sideways", s, e));
            at = e;
        }
        if stop > at {
            rows.push(row(label, at, stop));
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// Causal dating
// ---------------------------------------------------------------------------

/// The label of every day from the closes at or before it. `None` until the
/// first threshold move. A switch inside `min_days` of the last one is ignored
/// unless the move exceeds twice the threshold. Sideways: the closes of the
/// trailing `sideways_days` days stay within the band.
pub fn causal_labels(days: &[Day], p: &Params) -> Vec<Option<&'static str>> {
    let c = |i: usize| days[i].close;
    let mut out = Vec::with_capacity(days.len());
    let (mut hi, mut lo) = (0usize, 0usize);
    let mut state: Option<bool> = None;
    let (mut peak, mut trough) = (0usize, 0usize);
    let mut last_switch = f64::NEG_INFINITY;
    let mut first = 0usize;
    for i in 0..days.len() {
        if i > 0 {
            let aged = |at: f64| (days[i].t - at) / DAY >= p.min_days;
            match state {
                None => {
                    if c(i) > c(hi) {
                        hi = i;
                    }
                    if c(i) < c(lo) {
                        lo = i;
                    }
                    if c(i) <= c(hi) * (1.0 - p.down) {
                        state = Some(false);
                        trough = i;
                        last_switch = days[i].t;
                    } else if c(i) >= c(lo) * (1.0 + p.up) {
                        state = Some(true);
                        peak = i;
                        last_switch = days[i].t;
                    }
                }
                Some(true) => {
                    if c(i) > c(peak) {
                        peak = i;
                    } else if c(i) <= c(peak) * (1.0 - p.down) && (aged(last_switch) || c(i) <= c(peak) * (1.0 - 2.0 * p.down)) {
                        state = Some(false);
                        trough = i;
                        last_switch = days[i].t;
                    }
                }
                Some(false) => {
                    if c(i) < c(trough) {
                        trough = i;
                    } else if c(i) >= c(trough) * (1.0 + p.up) && (aged(last_switch) || c(i) >= c(trough) * (1.0 + 2.0 * p.up)) {
                        state = Some(true);
                        peak = i;
                        last_switch = days[i].t;
                    }
                }
            }
        }
        let label = state.map(|bull| if bull { "bull" } else { "bear" });
        let sideways = label.is_some() && p.sideways_days > 0.0 && {
            while days[i].t - days[first].t > (p.sideways_days - 1.0) * DAY {
                first += 1;
            }
            let window = &days[first..=i];
            let covers = days[i].t - window[0].t >= (p.sideways_days - 1.0) * DAY;
            let lo = window.iter().map(|d| d.close).fold(f64::INFINITY, f64::min);
            let hi = window.iter().map(|d| d.close).fold(f64::NEG_INFINITY, f64::max);
            covers && hi <= lo * (1.0 + p.sideways_band)
        };
        out.push(if sideways { Some("sideways") } else { label });
    }
    out
}

pub fn date_causal(days: &[Day], p: &Params) -> Vec<Row> {
    let labels = causal_labels(days, p);
    let mut rows: Vec<Row> = Vec::new();
    for (i, l) in labels.iter().enumerate() {
        let Some(l) = l else { continue };
        let end = days.get(i + 1).map_or(tail(days), |d| d.t);
        match rows.last_mut() {
            Some(r) if r.label == *l && r.end == days[i].t => r.end = end,
            _ => rows.push(row(l, days[i].t, end)),
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// Series from rows
// ---------------------------------------------------------------------------

/// The last close of every UTC day of every symbol, in order.
#[derive(Default)]
pub struct DailyCloses {
    symbols: Vec<String>,
    /// `(symbol, day number)` to `(stamp of the last row, close)`.
    last: BTreeMap<(usize, i64), (f64, f64)>,
    pub rows_used: usize,
    pub rows_dropped_locked: usize,
}

impl DailyCloses {
    /// `skip` is true for a row that reaches a zone that must stay unread; its
    /// price is never looked at.
    pub fn add(&mut self, symbol: &str, t: f64, price: impl FnOnce() -> f64, skip: bool) {
        if !t.is_finite() {
            return;
        }
        if skip {
            self.rows_dropped_locked += 1;
            return;
        }
        let close = price();
        if !close.is_finite() || close <= 0.0 {
            return;
        }
        let s = match self.symbols.iter().position(|x| x == symbol) {
            Some(s) => s,
            None => {
                self.symbols.push(symbol.to_string());
                self.symbols.len() - 1
            }
        };
        let day = (t / DAY).floor() as i64;
        let e = self.last.entry((s, day)).or_insert((f64::NEG_INFINITY, f64::NAN));
        if t >= e.0 {
            *e = (t, close);
        }
        self.rows_used += 1;
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    pub fn symbol_days(&self, symbol: &str) -> Vec<Day> {
        let Some(s) = self.symbols.iter().position(|x| x == symbol) else { return Vec::new() };
        self.last.range((s, i64::MIN)..=(s, i64::MAX)).map(|(&(_, d), &(_, c))| Day { t: d as f64 * DAY, close: c }).collect()
    }

    /// Equal-weight universe index: the level compounds the mean of the daily
    /// returns of the symbols that have a close on the day and the day before.
    pub fn equal_weight_index(&self) -> Vec<Day> {
        let mut by_day: BTreeMap<i64, Vec<(usize, f64)>> = BTreeMap::new();
        for (&(s, d), &(_, c)) in &self.last {
            by_day.entry(d).or_default().push((s, c));
        }
        let mut out: Vec<Day> = Vec::new();
        let mut level = 1.0f64;
        let mut previous: Option<(i64, &Vec<(usize, f64)>)> = None;
        for (&d, closes) in &by_day {
            match previous {
                None => out.push(Day { t: d as f64 * DAY, close: level }),
                Some((pd, prev)) if pd + 1 == d => {
                    let r: Vec<f64> = closes
                        .iter()
                        .filter_map(|&(s, c)| prev.iter().find(|&&(ps, _)| ps == s).map(|&(_, pc)| c / pc - 1.0))
                        .collect();
                    if !r.is_empty() {
                        level *= 1.0 + r.iter().sum::<f64>() / r.len() as f64;
                        out.push(Day { t: d as f64 * DAY, close: level });
                    }
                }
                Some(_) => {}
            }
            previous = Some((d, closes));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Calendar assembly
// ---------------------------------------------------------------------------

/// A span that must stay undated, in epoch seconds, aligned outward to days.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zone {
    pub start: f64,
    pub end: f64,
}

impl Zone {
    /// The span of a locked range with its purge before and embargo after.
    pub fn of(start: i64, end: i64, purge: i64, embargo: i64) -> Zone {
        Zone {
            start: ((start - purge) as f64 / DAY).floor() * DAY,
            end: ((end + embargo) as f64 / DAY).ceil() * DAY,
        }
    }

    pub fn contains(&self, t: f64) -> bool {
        t >= self.start && t < self.end
    }
}

pub fn merge_zones(mut zones: Vec<Zone>) -> Vec<Zone> {
    zones.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut out: Vec<Zone> = Vec::new();
    for z in zones {
        match out.last_mut() {
            Some(l) if z.start <= l.end => l.end = l.end.max(z.end),
            _ => out.push(z),
        }
    }
    out
}

pub enum Mode {
    ExPost,
    Causal,
}

/// The calendar of a day series: each segment between zones is dated on its
/// own and every zone is written as `not_dated`. With `only` the series is
/// cut to that span first.
pub fn calendar_rows(days: &[Day], zones: &[Zone], mode: &Mode, p: &Params) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    let mut at = 0usize;
    let mut segments: Vec<&[Day]> = Vec::new();
    for z in zones {
        let upto = days[at..].iter().position(|d| d.t >= z.start).map_or(days.len(), |k| at + k);
        segments.push(&days[at..upto]);
        at = upto;
        while at < days.len() && days[at].t < z.end {
            at += 1;
        }
    }
    segments.push(&days[at..]);
    for (k, seg) in segments.iter().enumerate() {
        if !seg.is_empty() {
            rows.extend(match mode {
                Mode::ExPost => date_ex_post(seg, p),
                Mode::Causal => date_causal(seg, p),
            });
        }
        if let Some(z) = zones.get(k) {
            rows.push(row(NOT_DATED, z.start, z.end));
        }
    }
    rows.sort_by(|a, b| a.start.total_cmp(&b.start));
    rows
}

pub fn calendar_csv(rows: &[Row], header_comment: &str) -> String {
    let mut s = String::new();
    for l in header_comment.lines() {
        s.push_str(&format!("# {}\n", l));
    }
    s.push_str(CALENDAR_HEADER);
    s.push('\n');
    for r in rows {
        s.push_str(&format!("{},{}Z,{}Z\n", r.label, protocol::iso(r.start), protocol::iso(r.end)));
    }
    s
}

// ---------------------------------------------------------------------------
// Reading the data
// ---------------------------------------------------------------------------

struct Source<'a> {
    path: &'a str,
    time_col: &'a str,
    price_col: &'a str,
    symbol_col: Option<&'a str>,
}

fn skip_of(zones: &[Zone], keep: Option<(f64, f64)>, t: f64) -> bool {
    zones.iter().any(|z| z.contains(t)) || keep.is_some_and(|(s, e)| !(t >= s && t < e))
}

fn read_csv(src: &Source, zones: &[Zone], keep: Option<(f64, f64)>) -> Result<DailyCloses, String> {
    use std::io::BufRead;
    let f = std::fs::File::open(src.path).map_err(|e| format!("{}: {}", src.path, e))?;
    let mut lines = std::io::BufReader::new(f).lines();
    let header = lines.next().ok_or_else(|| format!("{}: empty file", src.path))?.map_err(|e| e.to_string())?;
    let names: Vec<String> = header.split(',').map(|s| s.trim().trim_matches('"').to_string()).collect();
    let find = |c: &str| names.iter().position(|n| n == c).ok_or_else(|| format!("{}: no column '{}'", src.path, c));
    let (ti, pi) = (find(src.time_col)?, find(src.price_col)?);
    let si = src.symbol_col.map(find).transpose()?;
    let mut out = DailyCloses::default();
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|s| s.trim().trim_matches('"')).collect();
        let Some(t) = f.get(ti).and_then(|s| crate::data::parse_time(s)) else { continue };
        let sym = si.and_then(|i| f.get(i).copied()).unwrap_or("");
        let skip = skip_of(zones, keep, t);
        out.add(sym, t, || f.get(pi).and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN), skip);
    }
    Ok(out)
}

fn read_store(src: &Source, zones: &[Zone], keep: Option<(f64, f64)>) -> Result<DailyCloses, String> {
    use crate::store::{self, FileStore};
    let fs = FileStore::open(src.path)?;
    let header = fs.header();
    let names = header.names();
    let numeric = header.numeric();
    let find = |c: &str| names.iter().position(|n| n == c).ok_or_else(|| format!("{}: no column '{}'", src.path, c));
    let (ti, pi) = (find(src.time_col)?, find(src.price_col)?);
    let si = src.symbol_col.map(find).transpose()?;
    if !numeric[pi] {
        return Err(format!("{}: price column '{}' is not numeric", src.path, src.price_col));
    }
    let time_i64 = header.dtypes()[ti] == store::DType::I64;
    let time_f64 = numeric[ti] && !time_i64;
    let sym_numeric = si.is_some_and(|s| numeric[s]);
    let mut needed = vec![pi];
    if time_f64 {
        needed.push(ti);
    }
    if sym_numeric {
        needed.extend(si);
    }
    let projection = store::Projection::from_indices(&fs, &needed)?;
    let at = |index: usize| projection.columns().iter().position(|c| c.index == index).expect("projected column");
    let mut pg = store::ProjectedGroup::new();
    let time_levels = header.levels(ti);
    let mut out = DailyCloses::default();
    for g in 0..fs.n_groups() {
        let len = header.group_len(g);
        let ints = if time_i64 { Some(fs.read_i64(g, ti)?) } else { None };
        let codes = if !time_i64 && !time_f64 { Some(fs.read_codes(g, ti)?) } else { None };
        let sym_codes = match si {
            Some(s) if !numeric[s] => Some(fs.read_codes(g, s)?),
            _ => None,
        };
        fs.read_group_projected(g, &projection, &mut pg)?;
        let price = pg.col(at(pi));
        for r in 0..len {
            let t = match (&ints, &codes) {
                (Some(v), _) => v.get(r).copied().unwrap_or(i64::MIN) as f64,
                (None, None) => pg.col(at(ti))[r],
                (None, Some(c)) => time_levels
                    .and_then(|l| l.get(c.get(r).copied().unwrap_or(0) as usize))
                    .and_then(|s| crate::data::parse_time(s))
                    .unwrap_or(f64::NAN),
            };
            let sym = match (si, &sym_codes) {
                (None, _) => String::new(),
                (Some(s), Some(c)) => header.levels(s).and_then(|l| l.get(c.get(r).copied().unwrap_or(0) as usize)).cloned().unwrap_or_default(),
                (Some(s), None) => format!("{}", pg.col(at(s))[r]),
            };
            let skip = skip_of(zones, keep, t);
            out.add(&sym, t, || price[r], skip);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

pub fn usage() -> &'static str {
    "rcsr protocol regimes --data <store.rcsb|csv> --time-col <col> --price-col <col> --out <calendar.csv>
        (--symbol <name> --symbol-col <col> | --index equal-weight --symbol-col <col>)
        [--rule threshold] [--up 0.30] [--down 0.30] [--min-days 90]
        [--sideways-days 60] [--sideways-band 0.25] [--mode ex-post|causal]
        [--locked-test <START,END|none>] [--lockbox-embargo <SECONDS>] [--retired <record>]
        [--purge-seconds <N>]
        [--after-read --ledger <file> --sealed <file> --expect-reads <N>]

  Dates bull, bear and sideways phases on the daily closes of one symbol or of
  the equal-weight universe index and writes a regime calendar (label,start,end,
  UTC half-open, no overlap) for `--regimes` of walkforward and trials.

  rule threshold: a bull phase ends at the highest close after which the price
  falls by --down, a bear phase at the lowest close after which it rises by --up.
  A phase shorter than --min-days merges into its neighbour unless its move
  exceeds twice the threshold. Inside a phase, a span of at least --sideways-days
  days whose closes stay within a band --sideways-band (max over min) is labelled
  sideways and splits the phase. --sideways-days 0 turns the label off.

  mode ex-post (default): the label of a day uses closes after it. The calendar
  is for stratifying reports only, never a gate and never a search input.
  mode causal: the label of a day uses only the closes at or before it; a switch
  within --min-days of the last one is ignored unless its move exceeds twice the
  threshold; sideways is the trailing --sideways-days window.

  A locked range (--locked-test, the range the data declares, or the protocol
  default) with --purge-seconds before it (default 219600) and its embargo after
  it is never read: those rows are dropped before their price is parsed and the
  span is written as not_dated. --after-read dates the primary locked range only,
  and only when the ledger holds exactly --expect-reads locked reads under one
  final seal (the `protocol verify --expect-reads` check, with --data and
  --sealed); it appends a regime_dating line with the calendar hash to the ledger.
  The calendar then covers the locked range only. A null store (rcsr panel
  permute-labels) with --after-read exits 2 before the ledger is read (#1304).
"
}

fn fail(msg: String, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code)
}

pub fn run(args: &[String]) -> ! {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        std::process::exit(0);
    }
    let mut o: BTreeMap<&str, String> = BTreeMap::new();
    let mut after_read = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--after-read" => after_read = true,
            "--data" | "--time-col" | "--price-col" | "--out" | "--symbol" | "--symbol-col" | "--index" | "--rule" | "--up" | "--down"
            | "--min-days" | "--sideways-days" | "--sideways-band" | "--mode" | "--locked-test" | "--lockbox" | "--lockbox-embargo"
            | "--retired" | "--purge-seconds" | "--ledger" | "--sealed" | "--expect-reads" => {
                let v = args.get(i + 1).cloned().unwrap_or_else(|| fail(format!("{} requires a value", a), 2));
                o.insert(if a == "--lockbox" { "--locked-test" } else { a }, v);
                i += 1;
            }
            other => fail(format!("unknown option '{}' for protocol regimes\n\n{}", other, usage()), 2),
        }
        i += 1;
    }
    let need = |k: &str| o.get(k).cloned().unwrap_or_else(|| fail(format!("protocol regimes needs {}", k), 2));
    let number = |k: &str, default: f64, min: f64| -> f64 {
        match o.get(k) {
            None => default,
            Some(s) => s.trim().parse::<f64>().ok().filter(|v| v.is_finite() && *v >= min).unwrap_or_else(|| fail(format!("{} needs a number >= {}", k, min), 2)),
        }
    };
    let data = need("--data");
    let out_path = need("--out");
    if o.get("--rule").is_some_and(|r| r != "threshold") {
        fail("--rule: only threshold exists".to_string(), 2);
    }
    let mode = match o.get("--mode").map(String::as_str) {
        None | Some("ex-post") => Mode::ExPost,
        Some("causal") => Mode::Causal,
        Some(m) => fail(format!("--mode takes ex-post or causal, got '{}'", m), 2),
    };
    let params = Params {
        up: number("--up", 0.30, f64::MIN_POSITIVE),
        down: number("--down", 0.30, f64::MIN_POSITIVE),
        min_days: number("--min-days", 90.0, 0.0),
        sideways_days: number("--sideways-days", 60.0, 0.0),
        sideways_band: number("--sideways-band", 0.25, f64::MIN_POSITIVE),
    };
    if params.down >= 1.0 {
        fail("--down must be below 1".to_string(), 2);
    }
    let purge = number("--purge-seconds", protocol::DEFAULT_PURGE_SECONDS, 0.0) as i64;
    let symbol = o.get("--symbol").cloned();
    let index = o.get("--index").cloned();
    if symbol.is_some() == index.is_some() {
        fail("give exactly one of --symbol and --index equal-weight".to_string(), 2);
    }
    if index.as_deref().is_some_and(|i| i != "equal-weight") {
        fail("--index: only equal-weight exists".to_string(), 2);
    }
    let symbol_col = o.get("--symbol-col").cloned();
    if symbol_col.is_none() {
        fail("protocol regimes needs --symbol-col".to_string(), 2);
    }
    let resolved = crate::lockbox::resolve_with(o.get("--locked-test").map(String::as_str), o.get("--lockbox-embargo").map(String::as_str), &data)
        .unwrap_or_else(|e| fail(e.message, e.exit));
    let ranges = resolved.ranges();
    let embargo = resolved.embargo.seconds;
    let (mut opened, mut keep): (Vec<crate::lockbox::Lockbox>, Option<(f64, f64)>) = (Vec::new(), None);
    let mut after: Option<(Ledger, LockedRange)> = None;
    if after_read {
        if let Some(m) = crate::permute_labels::marker(&data) {
            fail(crate::permute_labels::refusal(&data, &m, "--after-read dates the lock calendar of the protocol ledger from real data only"), 2);
        }
        let ledger = Ledger::new(need("--ledger"));
        let sealed = need("--sealed");
        let expect: u64 = need("--expect-reads").parse().unwrap_or_else(|_| fail("--expect-reads needs a non-negative integer".to_string(), 2));
        let Some(primary) = resolved.range else { fail("--after-read needs a locked range: drop --locked-test none".to_string(), 2) };
        if let Err(e) = ledger.verify_expected(expect, &data, Some(std::path::Path::new(&sealed))) {
            fail(format!("--after-read refused, the ledger does not hold every read the final seal allows: {}", e), protocol::LEDGER_FAULT_EXIT);
        }
        opened.push(primary);
        keep = Some((primary.start as f64, primary.end as f64));
        after = Some((ledger, LockedRange::new(&primary, resolved.source.as_str())));
    } else if o.contains_key("--ledger") || o.contains_key("--sealed") || o.contains_key("--expect-reads") {
        fail("--ledger, --sealed and --expect-reads need --after-read".to_string(), 2);
    }
    let zones = merge_zones(
        ranges
            .iter()
            .filter(|r| !opened.contains(r))
            .map(|r| Zone::of(r.start, r.end, purge, embargo))
            .collect(),
    );
    let src = Source { path: &data, time_col: &need("--time-col"), price_col: &need("--price-col"), symbol_col: symbol_col.as_deref() };
    let closes = if data.ends_with(".rcsb") { read_store(&src, &zones, keep) } else { read_csv(&src, &zones, keep) }.unwrap_or_else(|e| fail(e, 3));
    let days = match &symbol {
        Some(s) => {
            let d = closes.symbol_days(s);
            if d.is_empty() {
                fail(format!("no usable row of symbol '{}' outside the locked zones", s), 3);
            }
            d
        }
        None => closes.equal_weight_index(),
    };
    if days.len() < 2 {
        fail("fewer than 2 daily closes to date".to_string(), 3);
    }
    let rows = calendar_rows(&days, &zones, &mode, &params);
    let series = match (&symbol, &index) {
        (Some(s), _) => format!("symbol {}", s),
        (None, _) => "equal-weight index".to_string(),
    };
    let comment = format!(
        "mode={} rule=threshold up={} down={} min_days={} sideways_days={} sideways_band={} series={} price={}\n{}",
        match mode {
            Mode::ExPost => "ex_post",
            Mode::Causal => "causal",
        },
        params.up,
        params.down,
        params.min_days,
        params.sideways_days,
        params.sideways_band,
        series,
        src.price_col,
        match mode {
            Mode::ExPost => "ex_post: a label uses closes after its date; it stratifies reports only and is never a gate or a search input",
            Mode::Causal => "causal: a label uses closes at or before its date",
        }
    );
    let csv = calendar_csv(&rows, &comment);
    crate::atomic::write_text_atomic(&out_path, &csv).unwrap_or_else(|e| fail(e, 1));
    let sha = crate::manifest::sha256_hex(csv.as_bytes());
    if let Some((ledger, range)) = after {
        let params_sha = crate::manifest::sha256_hex(comment.as_bytes());
        ledger.record_regime_dating(&sha, &params_sha, &range).unwrap_or_else(|e| fail(e, protocol::LEDGER_FAULT_EXIT));
    }
    println!("calendar_sha256 {}", sha);
    println!(
        "rows {} (used {} price rows, dropped {} rows of locked zones)",
        rows.len(),
        closes.rows_used,
        closes.rows_dropped_locked
    );
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn days(closes: &[f64]) -> Vec<Day> {
        closes.iter().enumerate().map(|(i, &c)| Day { t: i as f64 * DAY, close: c }).collect()
    }

    fn p() -> Params {
        Params { up: 0.3, down: 0.3, min_days: 5.0, sideways_days: 0.0, sideways_band: 0.25 }
    }

    /// A path with a planted trough at day 20, peak at day 50 and trough at day 80.
    fn path() -> Vec<f64> {
        let mut v = Vec::new();
        for i in 0..=100usize {
            let c = if i <= 20 {
                100.0 - 2.0 * i as f64
            } else if i <= 50 {
                60.0 + 2.0 * (i - 20) as f64
            } else if i <= 80 {
                120.0 - 2.0 * (i - 50) as f64
            } else {
                60.0 + 1.5 * (i - 80) as f64
            };
            v.push(c);
        }
        v
    }

    #[test]
    fn the_threshold_rule_finds_planted_turns() {
        let d = days(&path());
        let r = date_ex_post(&d, &p());
        let got: Vec<(&str, f64, f64)> = r.iter().map(|r| (r.label.as_str(), r.start / DAY, r.end / DAY)).collect();
        assert_eq!(
            got,
            vec![("bear", 0.0, 20.0), ("bull", 20.0, 50.0), ("bear", 50.0, 80.0), ("bull", 80.0, 101.0)]
        );
    }

    #[test]
    fn a_short_phase_merges_unless_its_move_is_large() {
        // 100 -> 200 -> 130 (-35%) -> 400: the middle bear lasts 3 days.
        let mut c = vec![100.0];
        for i in 1..=30 {
            c.push(100.0 + i as f64 * 100.0 / 30.0);
        }
        c.extend([185.0, 160.0, 130.0]);
        for i in 1..=30 {
            c.push(130.0 + i as f64 * 9.0);
        }
        let d = days(&c);
        let merged = date_ex_post(&d, &Params { min_days: 10.0, ..p() });
        assert_eq!(merged.len(), 1, "{:?}", merged);
        assert_eq!(merged[0].label, "bull");
        // a move above twice the threshold keeps a short phase
        let kept = date_ex_post(&d, &Params { min_days: 10.0, down: 0.15, ..p() });
        assert_eq!(kept.len(), 3, "{:?}", kept);
    }

    #[test]
    fn a_sideways_span_splits_its_phase() {
        let mut c: Vec<f64> = (0..=30).map(|i| 100.0 + 5.0 * i as f64).collect();
        let top = *c.last().unwrap();
        for i in 1..=70 {
            c.push(top + if i % 2 == 0 { 3.0 } else { -3.0 });
        }
        for i in 1..=30 {
            c.push(top + 3.0 + 5.0 * i as f64);
        }
        let d = days(&c);
        let r = date_ex_post(&d, &Params { sideways_days: 60.0, ..p() });
        assert!(r.iter().any(|x| x.label == "sideways"), "{:?}", r);
        for w in r.windows(2) {
            assert!(w[0].end <= w[1].start, "rows overlap: {:?}", r);
        }
        let side = r.iter().find(|x| x.label == "sideways").unwrap();
        assert!((side.end - side.start) / DAY >= 60.0);
        assert!(r.iter().filter(|x| x.label == "bull").count() >= 2, "the phase is split around the span");
    }

    #[test]
    fn the_causal_label_of_a_day_ignores_later_closes() {
        let c = path();
        let all = causal_labels(&days(&c), &Params { sideways_days: 20.0, ..p() });
        for k in [5usize, 30, 55, 90, 100] {
            let part = causal_labels(&days(&c[..k]), &Params { sideways_days: 20.0, ..p() });
            assert_eq!(part, all[..k].to_vec(), "prefix {}", k);
        }
        assert!(all.iter().any(|l| *l == Some("bull")) && all.iter().any(|l| *l == Some("bear")));
        assert_eq!(all[0], None);
    }

    #[test]
    fn zones_split_segments_and_write_not_dated() {
        let d = days(&path());
        let zones = vec![Zone { start: 40.0 * DAY, end: 60.0 * DAY }];
        let r = calendar_rows(&d.iter().copied().filter(|x| !zones[0].contains(x.t)).collect::<Vec<_>>(), &zones, &Mode::ExPost, &p());
        assert!(r.iter().any(|x| x.label == NOT_DATED && x.start == 40.0 * DAY && x.end == 60.0 * DAY));
        for w in r.windows(2) {
            assert!(w[0].end <= w[1].start, "{:?}", r);
        }
    }
}
