//! #1238: `rcsr protocol verdict`, the pre-registered verdict of the sealed reads (broad-pilot
//! design 8.3, decisions D2, D9 and D15-D18).
//!
//! The verdict opens the ledger, the final sealed list, the files frozen in the ledger before
//! the final seal (`freeze` events: the candidate table with the thresholds of #1237, an
//! optional development table, the subwindows and years calendars), the locked read reports
//! and the lock calendar dated after the reads (`regime_dating`). It opens no development
//! data and recomputes no threshold. A read binds to one pick of the frozen table by the
//! SHA-256 of the run file it priced (`selection.run_sha256` against the pick's
//! `freeze.run_sha256`); its file must hash to the `protocol.report_sha256` that a
//! `test_read` event of its candidate recorded.

use crate::devlog::Json;
use crate::inference;
use crate::manifest::sha256_hex;
use crate::protocol::{Event, Ledger, LEDGER_FAULT_EXIT};
use crate::regimes::Calendar;
use crate::walkforward::{jnum, jstr};

pub const SCHEMA: &str = "rcsr-verdict/1";
/// The roles of `rcsr protocol freeze`.
pub const ROLES: [&str; 4] = ["thresholds", "development", "subwindows", "years"];
pub const DEFAULT_DSR_TRIALS: f64 = 51.0;

const S5_ALPHA: f64 = 0.05;
const PROMISING_S5_ALPHA: f64 = 0.10;
const D1_MIN: f64 = 0.95;
const PROMISING_DSR_MIN: f64 = 0.80;
const D2_ALPHA: f64 = 0.05;
const D3_MAX: f64 = 0.2;
const HOLM_ALPHA: f64 = 0.05;
const BULL: &str = "bull";
const BEAR: &str = "bear";
const SERIES_HINT: &str = "run the locked read with walkforward --emit-series --bootstrap N";

type Res<T> = Result<T, (i32, String)>;

fn refuse<T>(m: impl Into<String>) -> Res<T> {
    Err((2, m.into()))
}

fn read_file(path: &str) -> Res<Vec<u8>> {
    std::fs::read(path).map_err(|e| (3, format!("{}: {}", path, e)))
}

// ---------------------------------------------------------------------------
// The hashed text of a walk-forward report
// ---------------------------------------------------------------------------

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\n' | b'\r' | b'\t') {
        i += 1;
    }
    i
}

/// Index just past the string that opens at `i`.
fn skip_string(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

/// Index just past the value that starts at `i`.
fn skip_value(b: &[u8], i: usize) -> Option<usize> {
    match *b.get(i)? {
        b'"' => skip_string(b, i),
        b'{' | b'[' => {
            let (mut j, mut depth) = (i, 0usize);
            while j < b.len() {
                match b[j] {
                    b'"' => {
                        j = skip_string(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            None
        }
        _ => {
            let mut j = i;
            while j < b.len() && !matches!(b[j], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t') {
                j += 1;
            }
            (j > i).then_some(j)
        }
    }
}

/// One member of a raw JSON object: its key, where the key starts, where the value starts
/// and the index just past the value.
struct Member {
    key: String,
    start: usize,
    value: usize,
    end: usize,
}

fn members(b: &[u8], open: usize) -> Option<Vec<Member>> {
    if b.get(open) != Some(&b'{') {
        return None;
    }
    let mut out = Vec::new();
    let mut i = skip_ws(b, open + 1);
    if b.get(i) == Some(&b'}') {
        return Some(out);
    }
    loop {
        if b.get(i) != Some(&b'"') {
            return None;
        }
        let key_end = skip_string(b, i)?;
        let key = std::str::from_utf8(&b[i + 1..key_end - 1]).ok()?.to_string();
        let colon = skip_ws(b, key_end);
        if b.get(colon) != Some(&b':') {
            return None;
        }
        let value = skip_ws(b, colon + 1);
        let end = skip_value(b, value)?;
        out.push(Member { key, start: i, value, end });
        i = skip_ws(b, end);
        match *b.get(i)? {
            b',' => i = skip_ws(b, i + 1),
            b'}' => return Some(out),
            _ => return None,
        }
    }
}

/// `text` without the member `path[last]` of the object at `path[..last]`, byte for byte
/// otherwise. An absent member or parent leaves the text as it is.
pub fn remove_member(text: &str, path: &[&str]) -> Result<String, String> {
    let Some((last, parents)) = path.split_last() else {
        return Ok(text.to_string());
    };
    let b = text.as_bytes();
    let mut open = skip_ws(b, 0);
    for (depth, k) in parents.iter().enumerate() {
        let ms = members(b, open).ok_or_else(|| format!("'{}' is not one JSON object", path[..depth].join(".")))?;
        match ms.iter().find(|m| m.key == *k) {
            Some(m) => open = m.value,
            None => return Ok(text.to_string()),
        }
    }
    let ms = members(b, open).ok_or_else(|| format!("'{}' is not one JSON object", parents.join(".")))?;
    let Some(pos) = ms.iter().position(|m| m.key == *last) else {
        return Ok(text.to_string());
    };
    let (from, to) = if pos > 0 {
        (ms[pos - 1].end, ms[pos].end)
    } else if ms.len() > 1 {
        (ms[0].start, ms[1].start)
    } else {
        (ms[0].start, ms[0].end)
    };
    Ok(format!("{}{}", &text[..from], &text[to..]))
}

/// The text whose SHA-256 is `protocol.report_sha256` of a walk-forward report file: the
/// file without the members outside the reproduction hash (`protocol`, `batch_timings`,
/// `scan`, `ledger`, the file path `selection.run`, `split.fit_label` and `split.labelled_rows`).
pub fn hashed_report_text(text: &str) -> Result<String, String> {
    let mut t = text.trim().to_string();
    for path in [&["protocol"][..], &["batch_timings"], &["scan"], &["ledger"], &["selection", "run"], &["split", "fit_label"], &["split", "labelled_rows"]] {
        t = remove_member(&t, path)?;
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// Frozen inputs
// ---------------------------------------------------------------------------

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

fn num_at(j: &Json, path: &[&str]) -> Option<f64> {
    at(j, path).and_then(Json::as_f64).filter(|v| v.is_finite())
}

fn str_at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a str> {
    at(j, path).and_then(Json::as_str)
}

fn whole_at(j: &Json, path: &[&str]) -> Option<u64> {
    num_at(j, path).filter(|v| *v >= 0.0 && v.fract() == 0.0 && *v < 1e15).map(|v| v as u64)
}

fn parse_json(bytes: &[u8], what: &str) -> Res<Json> {
    let text = std::str::from_utf8(bytes).map_err(|_| (2, format!("{}: not UTF-8 text", what)))?;
    match Json::parse(text.trim()) {
        Some(j @ Json::Obj(_)) => Ok(j),
        _ => refuse(format!("{}: not one JSON object", what)),
    }
}

fn table_doc(bytes: &[u8], what: &str) -> Res<Json> {
    let schema = crate::trials_table::SCHEMA;
    let j = parse_json(bytes, what).map_err(|(c, m)| (c, format!("{} (expected the {} output of rcsr trials --table)", m, schema)))?;
    if str_at(&j, &["schema"]) != Some(schema) {
        return refuse(format!("{}: schema is not {} (the output of rcsr trials --table)", what, schema));
    }
    Ok(j)
}

fn calendar(bytes: &[u8], what: &str) -> Res<Calendar> {
    let text = std::str::from_utf8(bytes).map_err(|_| (2, format!("{}: not UTF-8 text", what)))?;
    Calendar::parse(text).map_err(|e| (2, format!("{}: {}", what, e)))
}

/// SPA and PBO of a development table (D2 and D3).
struct Dev {
    spa_p: Option<f64>,
    /// #1269: `family` when the table carries `trials.family` (D2 reads `spa.p_family`), else `consistent`.
    p_source: &'static str,
    benchmark: Option<String>,
    pbo: Option<f64>,
}

fn dev_of(j: &Json) -> Dev {
    let family = matches!(at(j, &["trials", "family"]), Some(Json::Obj(_)));
    Dev {
        spa_p: num_at(j, &["spa", if family { "p_family" } else { "p_consistent" }]),
        p_source: if family { "family" } else { "consistent" },
        benchmark: str_at(j, &["spa", "benchmark"]).map(str::to_string),
        pbo: num_at(j, &["pbo", "value"]),
    }
}

/// One pick of the frozen candidate table.
struct Pick {
    pick: usize,
    row: Option<usize>,
    label: String,
    dsr: Option<f64>,
    run_sha256: Option<String>,
    cost_bps: f64,
    ppy: f64,
    s4: f64,
    p1: f64,
    cap: f64,
    bits: Option<f64>,
    p_occam: Option<f64>,
    p_family: Option<f64>,
}

struct Thresholds {
    rule_sha256: String,
    picks: Vec<Pick>,
    dev: Dev,
}

fn parse_thresholds(bytes: &[u8], what: &str) -> Res<Thresholds> {
    let freeze_schema = crate::trials_freeze::SCHEMA;
    let j = table_doc(bytes, what)?;
    if str_at(&j, &["freeze", "schema"]) != Some(freeze_schema) {
        return refuse(format!(
            "{}: the table has no top-level freeze object of {}: write it with a rule that has freeze and --freeze-map (#1237)",
            what, freeze_schema
        ));
    }
    let rule_sha256 = str_at(&j, &["freeze", "rule_sha256"]).filter(|s| !s.is_empty()).map(str::to_string);
    let Some(rule_sha256) = rule_sha256 else {
        return refuse(format!("{}: freeze.rule_sha256 is missing", what));
    };
    let list = match j.get("picks") {
        Some(Json::Arr(a)) if !a.is_empty() => a,
        _ => return refuse(format!("{}: the table has no picks", what)),
    };
    let rows: &[Json] = match j.get("rows") {
        Some(Json::Arr(r)) => r,
        _ => &[],
    };
    let mut picks: Vec<Pick> = Vec::new();
    for (i, p) in list.iter().enumerate() {
        let place = format!("{}: picks[{}]", what, i);
        let Some(pick) = whole_at(p, &["pick"]).filter(|k| *k >= 1).map(|k| k as usize) else {
            return refuse(format!("{} has no pick number", place));
        };
        let f = match p.get("freeze") {
            Some(f @ Json::Obj(_)) if str_at(f, &["schema"]) == Some(freeze_schema) => f,
            _ => return refuse(format!("{} (pick {}) has no freeze object of {}: the table froze no thresholds for it", place, pick, freeze_schema)),
        };
        let need = |path: &[&str]| {
            num_at(f, path).ok_or_else(|| (2, format!("{} (pick {}): freeze.{} is missing or not a finite number", place, pick, path.join("."))))
        };
        let row = whole_at(p, &["row"]).map(|r| r as usize);
        let occam = row.and_then(|r| rows.iter().find(|x| whole_at(x, &["row"]) == Some(r as u64))).and_then(|x| x.get("occam"));
        let report = |key: &str| occam.and_then(|o| num_at(o, &[key]));
        picks.push(Pick {
            pick,
            row,
            label: str_at(p, &["label"]).unwrap_or_default().to_string(),
            dsr: num_at(p, &["dsr", "value"]),
            run_sha256: str_at(f, &["run_sha256"]).filter(|s| !s.is_empty()).map(str::to_string),
            cost_bps: need(&["cost_bps"])?,
            ppy: need(&["periods_per_year"])?,
            s4: need(&["s4_floor", "value"])?,
            p1: need(&["p1_floor", "value"])?,
            cap: need(&["drawdown_cap", "value"])?,
            bits: report("bits"),
            p_occam: report("p_occam"),
            p_family: report("p_family"),
        });
    }
    picks.sort_by_key(|p| p.pick);
    if let Some(w) = picks.windows(2).find(|w| w[0].pick == w[1].pick) {
        return refuse(format!("{}: pick {} is listed twice", what, w[0].pick));
    }
    if picks[0].pick != 1 {
        return refuse(format!("{}: the table has no pick 1 (the primary)", what));
    }
    Ok(Thresholds { rule_sha256, picks, dev: dev_of(&j) })
}

/// The content check of `rcsr protocol freeze`: a file that the verdict cannot read is
/// refused before it is frozen (one file per role).
pub fn check_frozen(role: &str, bytes: &[u8]) -> Result<(), String> {
    let what = format!("role {}", role);
    let checked = match role {
        "thresholds" => parse_thresholds(bytes, &what).map(|_| ()),
        "development" => table_doc(bytes, &what).map(|_| ()),
        "subwindows" | "years" => calendar(bytes, &what).map(|_| ()),
        _ => refuse(format!("unknown role '{}': one of {}", role, ROLES.join(", "))),
    };
    checked.map_err(|(_, m)| m)
}

/// The ledger's freeze event of `role` and the file `path` checked against it.
fn frozen(events: &[Event], final_at: usize, role: &str, flag: &str, path: Option<&str>) -> Res<Option<(Vec<u8>, String)>> {
    let found: Vec<(usize, &String, u64)> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Event::Freeze { role: r, sha256, bytes } if r == role => Some((i, sha256, *bytes)),
            _ => None,
        })
        .collect();
    if found.len() > 1 {
        return refuse(format!("the ledger freezes role {} {} times; one file per role", role, found.len()));
    }
    match (found.first(), path) {
        (None, None) => Ok(None),
        (None, Some(p)) => refuse(format!(
            "{} {}: the ledger has no freeze event for role {}; files are frozen with rcsr protocol freeze before the final seal",
            flag, p, role
        )),
        (Some((_, sha, _)), None) => refuse(format!("the ledger froze a {} file (sha256 {}): pass it with {}", role, sha, flag)),
        (Some(&(line, sha, n)), Some(p)) => {
            let bytes = read_file(p)?;
            if line > final_at {
                return refuse(format!(
                    "{} {}: its freeze event (ledger line {}) is after the final seal (line {}): a file frozen after the seal is not pre-registered",
                    flag,
                    p,
                    line + 1,
                    final_at + 1
                ));
            }
            let got = sha256_hex(&bytes);
            if got != *sha || bytes.len() as u64 != n {
                return refuse(format!(
                    "{} {}: sha256 {} ({} bytes) differs from the frozen sha256 {} ({} bytes) of role {}",
                    flag,
                    p,
                    got,
                    bytes.len(),
                    sha,
                    n,
                    role
                ));
            }
            Ok(Some((bytes, got)))
        }
    }
}

// ---------------------------------------------------------------------------
// Locked reads
// ---------------------------------------------------------------------------

struct Read {
    path: String,
    candidate_id: String,
    report_sha256: String,
    run_sha256: String,
    cost_bps: f64,
    ppy: f64,
    time: Vec<f64>,
    net: Vec<f64>,
    cost: Vec<f64>,
    exposure: Vec<f64>,
    bh: Vec<f64>,
    stamp: Option<Vec<f64>>,
    p: f64,
    block_len: usize,
    draws: usize,
    seed: u32,
}

fn pooled_series(j: &Json, key: &str, place: &str) -> Res<Vec<f64>> {
    let Some(Json::Arr(items)) = at(j, &["pooled", key]) else {
        return refuse(format!("{}: field 'pooled.{}' is missing ({})", place, key, SERIES_HINT));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, v)| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| (2, format!("{}: field 'pooled.{}'[{}] is not a finite number", place, key, i))))
        .collect()
}

fn load_read(path: &str) -> Res<Read> {
    let place = format!("--read {}", path);
    let bytes = read_file(path)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| (2, format!("{}: not UTF-8 text", place)))?;
    let j = Json::parse(text.trim()).ok_or_else(|| (2, format!("{}: not a JSON report", place)))?;
    if at(&j, &["protocol", "test_read"]) != Some(&Json::Bool(true)) {
        return refuse(format!("{}: protocol.test_read is not true: the verdict reads locked reads only, never a development report", place));
    }
    let need_str = |p: &[&str]| -> Res<String> {
        str_at(&j, p).filter(|s| !s.is_empty()).map(str::to_string).ok_or_else(|| (2, format!("{}: field '{}' is missing", place, p.join("."))))
    };
    let candidate_id = need_str(&["protocol", "candidate_id"])?;
    let report_sha256 = need_str(&["protocol", "report_sha256"])?;
    let hashed = hashed_report_text(text).map_err(|e| (2, format!("{}: {}", place, e)))?;
    let got = sha256_hex(hashed.as_bytes());
    if got != report_sha256 {
        return refuse(format!(
            "{}: the report does not hash to its protocol.report_sha256 ({} vs {}): the file changed after the read",
            place, got, report_sha256
        ));
    }
    let Some(run_sha256) = str_at(&j, &["selection", "run_sha256"]).filter(|s| !s.is_empty()).map(str::to_string) else {
        return refuse(format!(
            "{}: field 'selection.run_sha256' is missing: the read names no run file (an --expr read), so it binds to no pick; read the sealed run with --from-run or --front",
            place
        ));
    };
    let cost_bps = num_at(&j, &["cost_model", "per_side_bps"]).ok_or_else(|| (2, format!("{}: field 'cost_model.per_side_bps' is missing", place)))?;
    let ppy = num_at(&j, &["annualization", "periods_per_year"])
        .filter(|v| *v > 0.0)
        .ok_or_else(|| (2, format!("{}: field 'annualization.periods_per_year' is missing or not positive", place)))?;
    let time = pooled_series(&j, "period_time", &place)?;
    let net = pooled_series(&j, "period_net", &place)?;
    let cost = pooled_series(&j, "period_cost", &place)?;
    let exposure = pooled_series(&j, "period_exposure", &place)?;
    let bh = pooled_series(&j, "period_bh_net", &place)?;
    let stamp = match at(&j, &["pooled", "period_entry_stamp"]) {
        Some(_) => Some(pooled_series(&j, "period_entry_stamp", &place)?),
        None => None,
    };
    if time.is_empty() {
        return refuse(format!("{}: the read has no period", place));
    }
    for (key, v) in [("period_net", &net), ("period_cost", &cost), ("period_exposure", &exposure), ("period_bh_net", &bh)] {
        if v.len() != time.len() {
            return refuse(format!("{}: field 'pooled.{}' has {} values but 'pooled.period_time' has {}", place, key, v.len(), time.len()));
        }
    }
    if let Some(s) = &stamp {
        if s.len() != time.len() {
            return refuse(format!("{}: field 'pooled.period_entry_stamp' has {} values but 'pooled.period_time' has {}", place, s.len(), time.len()));
        }
    }
    let b = match at(&j, &["inference", "bootstrap"]) {
        Some(b @ Json::Obj(_)) => b,
        _ => return refuse(format!("{}: field 'inference.bootstrap' is missing or null ({})", place, SERIES_HINT)),
    };
    let bad = |key: &str| (2, format!("{}: field 'inference.bootstrap.{}' is missing or invalid ({})", place, key, SERIES_HINT));
    let p = num_at(b, &["p_mean_positive"]).ok_or_else(|| bad("p_mean_positive"))?;
    let block_len = whole_at(b, &["block_len"]).filter(|v| *v >= 1).ok_or_else(|| bad("block_len"))? as usize;
    let draws = whole_at(b, &["draws"]).filter(|v| *v >= 1).ok_or_else(|| bad("draws"))? as usize;
    let seed = whole_at(b, &["seed"]).filter(|v| *v <= u32::MAX as u64).ok_or_else(|| bad("seed"))? as u32;
    Ok(Read { path: path.to_string(), candidate_id, report_sha256, run_sha256, cost_bps, ppy, time, net, cost, exposure, bh, stamp, p, block_len, draws, seed })
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn max_drawdown(v: &[f64]) -> f64 {
    let mut d = crate::risk::Drawdown::identity();
    for &x in v {
        d.push(x);
    }
    d.max_drawdown()
}

impl Read {
    fn sel(&self, v: &[f64], idx: &[usize]) -> Vec<f64> {
        idx.iter().map(|&i| v[i]).collect()
    }

    fn ann(&self, v: impl Iterator<Item = f64>) -> f64 {
        let v: Vec<f64> = v.collect();
        mean(&v) * self.ppy
    }

    /// Annualised mean net at the read cost (10 bp), over cash at 0.
    fn net10(&self, idx: &[usize]) -> f64 {
        self.ann(idx.iter().map(|&i| self.net[i]))
    }

    /// Annualised mean net at twice the read cost.
    fn net20(&self, idx: &[usize]) -> f64 {
        self.ann(idx.iter().map(|&i| self.net[i] - self.cost[i]))
    }

    /// Annualised exposure-matched alpha over buy-and-hold of the same rows.
    fn alpha(&self, idx: &[usize]) -> f64 {
        self.ann(idx.iter().map(|&i| self.net[i] - self.exposure[i] * self.bh[i]))
    }

    fn bh_ann(&self, idx: &[usize]) -> f64 {
        self.ann(idx.iter().map(|&i| self.bh[i]))
    }

    fn bootstrap_p(&self, x: &[f64]) -> Option<f64> {
        (x.len() >= 2).then(|| inference::bootstrap_p_mean_positive(x, &inference::bootstrap_draws(x, self.block_len, self.draws, self.seed)))
    }

    /// DSR at `trials` with the single-series variance of the candidate table.
    fn dsr(&self, idx: &[usize], trials: f64) -> Option<f64> {
        let x = self.sel(&self.net, idx);
        if x.len() < 2 {
            return None;
        }
        let s = inference::block_bootstrap(&x, self.block_len, self.draws, self.seed, self.ppy, 0.95);
        let t = if s.effective_n.is_finite() && s.effective_n > 1.0 { s.effective_n.min(x.len() as f64) } else { x.len() as f64 };
        let (sr, skew, kurt) = (inference::sharpe_per_period(&x), inference::skewness(&x), inference::kurtosis(&x));
        let var = inference::sharpe_estimator_variance(sr, t, skew, kurt);
        Some(inference::deflated_sharpe(sr, t, skew, kurt, trials, var).dsr).filter(|d| d.is_finite())
    }

    /// The report-only figures of one part, year or stratum.
    fn figures(&self, label: &str, idx: &[usize]) -> Vec<(&'static str, String)> {
        let x = self.sel(&self.net, idx);
        let entries = self.stamp.as_ref().map(|s| idx.iter().map(|&i| s[i]).sum::<f64>());
        vec![
            ("label", jstr(label)),
            ("periods", idx.len().to_string()),
            ("net10", jnum(self.net10(idx))),
            ("net20", jnum(self.net20(idx))),
            ("sharpe", jnum(crate::walkforward::sharpe(&x, self.ppy))),
            ("alpha_bh", jnum(self.alpha(idx))),
            ("max_drawdown", jnum(max_drawdown(&x))),
            ("exposure", jnum(mean(&self.sel(&self.exposure, idx)))),
            ("entries", opt(entries)),
            ("bh", jnum(self.bh_ann(idx))),
            ("cash", "0".to_string()),
            ("bootstrap_p", opt(self.bootstrap_p(&x))),
        ]
    }
}

// ---------------------------------------------------------------------------
// Gates and the verdict
// ---------------------------------------------------------------------------

fn opt(v: Option<f64>) -> String {
    v.map_or("null".to_string(), jnum)
}

fn obj(fields: &[(&str, String)]) -> String {
    format!("{{{}}}", fields.iter().map(|(k, v)| format!("{}:{}", jstr(k), v)).collect::<Vec<_>>().join(","))
}

struct Gate {
    name: &'static str,
    fields: Vec<(&'static str, String)>,
    pass: bool,
}

/// A gate over the periods `idx` of a calendar label; no period fails with reason `no periods`.
fn stratum_gate(name: &'static str, idx: &[usize], keys: &[&'static str], eval: impl Fn(&[usize]) -> (Vec<f64>, bool), tail: Vec<(&'static str, String)>) -> Gate {
    let mut fields: Vec<(&'static str, String)> = Vec::new();
    let pass = if idx.is_empty() {
        fields.extend(keys.iter().map(|k| (*k, "null".to_string())));
        false
    } else {
        let (values, pass) = eval(idx);
        fields.extend(keys.iter().zip(values).map(|(k, v)| (*k, jnum(v))));
        pass
    };
    fields.extend(tail);
    fields.push(("periods", idx.len().to_string()));
    fields.push(("pass", pass.to_string()));
    if idx.is_empty() {
        fields.push(("reason", jstr("no periods")));
    }
    Gate { name, fields, pass }
}

fn value_gate(name: &'static str, mut fields: Vec<(&'static str, String)>, value: Option<f64>, pass: bool) -> Gate {
    fields.push(("pass", pass.to_string()));
    if value.is_none() {
        fields.push(("reason", jstr("missing in the frozen table")));
    }
    Gate { name, fields, pass }
}

struct Outcome {
    gates: Vec<Gate>,
    verdict: &'static str,
}

struct Strata<'a> {
    bull: &'a [usize],
    bear: &'a [usize],
    pristine: &'a [usize],
}

fn evaluate_gates(r: &Read, pick: &Pick, dev: &Dev, s: &Strata) -> Outcome {
    let all: Vec<usize> = (0..r.net.len()).collect();
    let (net10, net20, alpha) = (r.net10(&all), r.net20(&all), r.alpha(&all));
    let dd = max_drawdown(&r.net);
    let s1 = net10 > 0.0 && net20 > 0.0;
    let s2 = alpha > 0.0;
    let s4 = net10 >= pick.s4;
    let d2 = dev.spa_p.is_some_and(|p| p <= D2_ALPHA);
    let d1 = pick.dsr.is_some_and(|d| d >= D1_MIN);
    let gates = vec![
        Gate { name: "S1", fields: vec![("net10", jnum(net10)), ("net20", jnum(net20)), ("pass", s1.to_string())], pass: s1 },
        Gate { name: "S2", fields: vec![("alpha_bh", jnum(alpha)), ("pass", s2.to_string())], pass: s2 },
        {
            let pass = -dd <= pick.cap;
            Gate { name: "S3", fields: vec![("max_drawdown", jnum(dd)), ("cap", jnum(pick.cap)), ("pass", pass.to_string())], pass }
        },
        Gate { name: "S4", fields: vec![("net10", jnum(net10)), ("floor", jnum(pick.s4)), ("pass", s4.to_string())], pass: s4 },
        {
            let pass = r.p <= S5_ALPHA;
            Gate { name: "S5", fields: vec![("p", jnum(r.p)), ("alpha", jnum(S5_ALPHA)), ("pass", pass.to_string())], pass }
        },
        stratum_gate(
            "R1",
            s.bull,
            &["net10", "net20", "alpha_bh"],
            |idx| {
                let (a, b, c) = (r.net10(idx), r.net20(idx), r.alpha(idx));
                (vec![a, b, c], a > 0.0 && b > 0.0 && c >= 0.0)
            },
            Vec::new(),
        ),
        stratum_gate(
            "R2",
            s.bear,
            &["excess", "alpha_bh"],
            |idx| {
                let (e, a) = (r.net10(idx) - r.bh_ann(idx), r.alpha(idx));
                (vec![e, a], e >= 0.0 && a >= 0.0)
            },
            Vec::new(),
        ),
        stratum_gate(
            "P1",
            s.pristine,
            &["net10"],
            |idx| {
                let n = r.net10(idx);
                (vec![n], n >= pick.p1)
            },
            vec![("floor", jnum(pick.p1))],
        ),
        value_gate("D1", vec![("dsr", opt(pick.dsr)), ("min", jnum(D1_MIN))], pick.dsr, d1),
        value_gate(
            "D2",
            vec![
                ("p", opt(dev.spa_p)),
                ("p_source", jstr(dev.p_source)),
                ("alpha", jnum(D2_ALPHA)),
                ("benchmark", dev.benchmark.as_deref().map_or("null".to_string(), jstr)),
            ],
            dev.spa_p,
            d2,
        ),
        value_gate("D3", vec![("pbo", opt(dev.pbo)), ("max", jnum(D3_MAX))], dev.pbo, dev.pbo.is_some_and(|p| p <= D3_MAX)),
    ];
    let promising = net10 > 0.0 && s2 && s4 && r.p <= PROMISING_S5_ALPHA && d2 && pick.dsr.is_some_and(|d| d >= PROMISING_DSR_MIN);
    let verdict = if gates.iter().all(|g| g.pass) {
        "good"
    } else if promising {
        "promising"
    } else if net10 >= 0.0 {
        "not_confirmed"
    } else {
        "falsified"
    };
    Outcome { gates, verdict }
}

/// Holm-adjusted p-values (step-down, monotone, capped at 1) in the order of `p`.
pub fn holm(p: &[f64]) -> Vec<f64> {
    let m = p.len() as f64;
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let mut out = vec![0.0; p.len()];
    let mut running = 0.0f64;
    for (l, &i) in order.iter().enumerate() {
        running = running.max(((m - l as f64) * p[i]).min(1.0));
        out[i] = running;
    }
    out
}

fn groups(cal: &Calendar, time: &[f64]) -> Vec<(String, Vec<usize>)> {
    let (names, idx) = cal.assign(time);
    names
        .iter()
        .enumerate()
        .map(|(k, name)| (name.clone(), idx.iter().enumerate().filter(|(_, g)| **g == k).map(|(i, _)| i).collect()))
        .collect()
}

fn indices_of<'a>(g: &'a [(String, Vec<usize>)], label: &str) -> &'a [usize] {
    g.iter().find(|(l, _)| l == label).map_or(&[], |(_, v)| v.as_slice())
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// The inputs of `rcsr protocol verdict`.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub ledger: String,
    pub sealed: String,
    pub thresholds: String,
    pub development: Option<String>,
    pub reads: Vec<String>,
    pub subwindows: String,
    pub pristine_label: String,
    pub years: Option<String>,
    pub regimes: String,
    pub dsr_trials: f64,
}

/// The verdict JSON and its Markdown.
pub struct Verdict {
    pub json: String,
    pub markdown: String,
    /// `(pick, role, verdict)` per pick; the verdict is `None` for a pick without a read.
    pub summary: Vec<(usize, &'static str, Option<&'static str>)>,
}

struct Scored<'a> {
    pick: &'a Pick,
    read: Option<(Read, Outcome, Vec<(String, Vec<usize>)>)>,
}

pub fn evaluate(o: &Inputs) -> Result<Verdict, (i32, String)> {
    let ledger_bytes = read_file(&o.ledger)?;
    let ledger = Ledger::new(&o.ledger);
    ledger.verify_chain().map_err(|e| (LEDGER_FAULT_EXIT, e))?;
    let events = ledger.events().map_err(|e| (LEDGER_FAULT_EXIT, e))?;
    let Some((final_at, final_sha)) = events.iter().enumerate().find_map(|(i, e)| match e {
        Event::Seal { sealed_sha256, final_seal: Some(_) } => Some((i, sealed_sha256.clone())),
        _ => None,
    }) else {
        return refuse(format!("{}: the ledger has no final seal; the verdict reads sealed reads only", o.ledger));
    };
    let sealed_bytes = read_file(&o.sealed)?;
    let sealed_sha = sha256_hex(&sealed_bytes);
    if sealed_sha != final_sha {
        return refuse(format!("--sealed {}: sha256 {} is not the list of the final seal ({})", o.sealed, sealed_sha, final_sha));
    }
    let sealed_ids: Vec<String> = match parse_json(&sealed_bytes, &format!("--sealed {}", o.sealed))?.get("candidates") {
        Some(Json::Arr(a)) => a.iter().filter_map(|c| str_at(c, &["id"]).map(str::to_string)).collect(),
        _ => return refuse(format!("--sealed {}: no candidates array", o.sealed)),
    };

    let (t_bytes, t_sha) = frozen(&events, final_at, "thresholds", "--thresholds", Some(o.thresholds.as_str()))?.expect("a passed file");
    let thresholds = parse_thresholds(&t_bytes, &format!("--thresholds {}", o.thresholds))?;
    let development = frozen(&events, final_at, "development", "--development", o.development.as_deref())?;
    let (sw_bytes, sw_sha) = frozen(&events, final_at, "subwindows", "--subwindows", Some(o.subwindows.as_str()))?.expect("a passed file");
    let years = frozen(&events, final_at, "years", "--years", o.years.as_deref())?;
    let dev = match &development {
        Some((bytes, _)) => dev_of(&table_doc(bytes, &format!("--development {}", o.development.as_deref().unwrap_or_default()))?),
        None => thresholds.dev,
    };
    let subwindows = calendar(&sw_bytes, &format!("--subwindows {}", o.subwindows))?;
    if !subwindows.labels().contains(&o.pristine_label) {
        return refuse(format!(
            "--pristine-label {} is not a label of the subwindows calendar {} (labels: {})",
            o.pristine_label,
            o.subwindows,
            subwindows.labels().join(", ")
        ));
    }
    let years_cal = match (&years, &o.years) {
        (Some((bytes, _)), Some(p)) => Some(calendar(bytes, &format!("--years {}", p))?),
        _ => None,
    };
    let reg_bytes = read_file(&o.regimes)?;
    let reg_sha = sha256_hex(&reg_bytes);
    let dated = events
        .iter()
        .enumerate()
        .any(|(i, e)| i > final_at && matches!(e, Event::RegimeDating { calendar_sha256, .. } if *calendar_sha256 == reg_sha));
    if !dated {
        return refuse(format!(
            "--regimes {}: sha256 {} matches no regime_dating event after the final seal; pass the calendar that rcsr protocol regimes --after-read wrote",
            o.regimes, reg_sha
        ));
    }
    let lock_cal = calendar(&reg_bytes, &format!("--regimes {}", o.regimes))?;

    let mut bound: Vec<Option<Read>> = (0..thresholds.picks.len()).map(|_| None).collect();
    for path in &o.reads {
        let r = load_read(path)?;
        let place = format!("--read {}", path);
        if !sealed_ids.contains(&r.candidate_id) {
            return refuse(format!("{}: candidate {} is not in the sealed list {}", place, r.candidate_id, o.sealed));
        }
        let recorded = events
            .iter()
            .any(|e| matches!(e, Event::TestRead { id, report_sha256, .. } if *id == r.candidate_id && *report_sha256 == r.report_sha256));
        if !recorded {
            return refuse(format!(
                "{}: the ledger records no test_read of candidate {} with report sha256 {}",
                place, r.candidate_id, r.report_sha256
            ));
        }
        let matches: Vec<usize> = (0..thresholds.picks.len()).filter(|&k| thresholds.picks[k].run_sha256.as_deref() == Some(r.run_sha256.as_str())).collect();
        let k = match matches.as_slice() {
            [k] => *k,
            [] => {
                return refuse(format!(
                    "{}: selection.run_sha256 {} matches no pick's freeze.run_sha256: the read is unbound (another candidate or procedure)",
                    place, r.run_sha256
                ))
            }
            many => {
                let names: Vec<String> = many.iter().map(|&k| thresholds.picks[k].pick.to_string()).collect();
                return refuse(format!("{}: selection.run_sha256 {} matches picks {}: the binding is ambiguous", place, r.run_sha256, names.join(" and ")));
            }
        };
        let pick = &thresholds.picks[k];
        if let Some(other) = &bound[k] {
            return refuse(format!("{}: pick {} is bound by two reads ({} and {})", place, pick.pick, other.path, path));
        }
        if r.cost_bps != pick.cost_bps {
            return refuse(format!(
                "{}: cost_model.per_side_bps {} differs from the frozen cost_bps {} of pick {}: the floors and the cap hold at that cost",
                place,
                jnum(r.cost_bps),
                jnum(pick.cost_bps),
                pick.pick
            ));
        }
        if r.ppy != pick.ppy {
            return refuse(format!(
                "{}: annualization.periods_per_year {} differs from the frozen periods_per_year {} of pick {}",
                place,
                jnum(r.ppy),
                jnum(pick.ppy),
                pick.pick
            ));
        }
        bound[k] = Some(r);
    }
    if bound[0].is_none() {
        return refuse(format!(
            "pick 1 (the primary) has no read: pass its locked read with --read (its frozen run sha256 is {})",
            thresholds.picks[0].run_sha256.as_deref().unwrap_or("not recorded")
        ));
    }

    let scored: Vec<Scored> = thresholds
        .picks
        .iter()
        .zip(bound)
        .map(|(pick, r)| Scored {
            pick,
            read: r.map(|r| {
                let sub = groups(&subwindows, &r.time);
                let strata = groups(&lock_cal, &r.time);
                let s = Strata { bull: indices_of(&strata, BULL), bear: indices_of(&strata, BEAR), pristine: indices_of(&sub, &o.pristine_label) };
                let outcome = evaluate_gates(&r, pick, &dev, &s);
                (r, outcome, sub)
            }),
        })
        .collect();
    let primary_good = scored[0].read.as_ref().is_some_and(|(_, out, _)| out.verdict == "good");
    let secondaries: Vec<(usize, f64)> =
        scored.iter().enumerate().skip(1).filter_map(|(k, s)| s.read.as_ref().map(|(r, _, _)| (k, r.p))).collect();
    let tested = primary_good && !secondaries.is_empty();
    let adjusted = if tested { holm(&secondaries.iter().map(|(_, p)| *p).collect::<Vec<_>>()) } else { Vec::new() };
    let p_holm = |k: usize| secondaries.iter().position(|(j, _)| *j == k).and_then(|i| adjusted.get(i).copied());
    let reason = if tested {
        None
    } else if !primary_good {
        Some("the primary is not good")
    } else {
        Some("no secondary pick was read")
    };

    let mut candidates: Vec<String> = Vec::new();
    let mut md = String::from("# Sealed-read verdict\n\n");
    md.push_str(&format!(
        "Ledger sha256 `{}`; sealed list `{}` (final seal); rule `{}`. Frozen before the seal: thresholds `{}`, development {}, subwindows `{}`, years {}. Lock calendar `{}`, dated after the reads. D2 and D3 read the {} table. Benchmarks: cash at 0 and buy-and-hold of the same rows.\n",
        sha256_hex(&ledger_bytes),
        sealed_sha,
        thresholds.rule_sha256,
        t_sha,
        development.as_ref().map_or("none".to_string(), |(_, h)| format!("`{}`", h)),
        sw_sha,
        years.as_ref().map_or("none".to_string(), |(_, h)| format!("`{}`", h)),
        reg_sha,
        if development.is_some() { "development" } else { "thresholds" }
    ));
    let mut summary = Vec::new();
    for (k, s) in scored.iter().enumerate() {
        let p = s.pick;
        let role = if k == 0 { "primary" } else { "secondary" };
        let mut f: Vec<(&str, String)> = vec![
            ("pick", p.pick.to_string()),
            ("role", jstr(role)),
            ("row", p.row.map_or("null".to_string(), |r| r.to_string())),
            ("label", jstr(&p.label)),
        ];
        let secondary = |f: &mut Vec<(&str, String)>| {
            if k > 0 {
                f.push(("tested", (tested && s.read.is_some()).to_string()));
                f.push(("p_holm", opt(p_holm(k))));
            }
        };
        let Some((r, out, sub)) = &s.read else {
            f.push(("read", "false".to_string()));
            f.push(("run_sha256", p.run_sha256.as_deref().map_or("null".to_string(), jstr)));
            secondary(&mut f);
            f.push(("verdict", "null".to_string()));
            candidates.push(obj(&f));
            md.push_str(&format!("\n## Pick {} ({}): not read\n", p.pick, role));
            summary.push((p.pick, role, None));
            continue;
        };
        f.push(("read", "true".to_string()));
        f.push(("candidate_id", jstr(&r.candidate_id)));
        f.push(("run_sha256", jstr(&r.run_sha256)));
        f.push(("read_report_sha256", jstr(&r.report_sha256)));
        f.push(("cost_bps", jnum(r.cost_bps)));
        f.push(("periods_per_year", jnum(r.ppy)));
        f.push(("periods", r.net.len().to_string()));
        let gates: Vec<(&str, String)> = out.gates.iter().map(|g| (g.name, obj(&g.fields))).collect();
        f.push(("gates", obj(&gates)));
        f.push(("verdict", jstr(out.verdict)));
        secondary(&mut f);
        let all: Vec<usize> = (0..r.net.len()).collect();
        let mut parts: Vec<(String, Vec<usize>)> = vec![("full".to_string(), all)];
        parts.extend(sub.iter().cloned());
        let years_g = years_cal.as_ref().map_or_else(Vec::new, |c| groups(c, &r.time));
        let strata_g = groups(&lock_cal, &r.time);
        let rows = |g: &[(String, Vec<usize>)]| -> Vec<Vec<(&'static str, String)>> { g.iter().map(|(l, idx)| r.figures(l, idx)).collect() };
        let (part_rows, year_rows, stratum_rows) = (rows(&parts), rows(&years_g), rows(&strata_g));
        let list = |v: &[Vec<(&str, String)>]| format!("[{}]", v.iter().map(|x| obj(x)).collect::<Vec<_>>().join(","));
        f.push(("parts", list(&part_rows)));
        f.push(("years", list(&year_rows)));
        f.push(("strata", list(&stratum_rows)));
        let dsr: Vec<String> = parts
            .iter()
            .filter(|(l, _)| *l != o.pristine_label && l != crate::regimes::UNDATED)
            .map(|(l, idx)| obj(&[("part", jstr(l)), ("value", opt(r.dsr(idx, o.dsr_trials)))]))
            .collect();
        f.push((
            "report_only",
            obj(&[
                ("dsr_trials", jnum(o.dsr_trials)),
                ("dsr", format!("[{}]", dsr.join(","))),
                ("bits", opt(p.bits)),
                ("p_occam", opt(p.p_occam)),
                ("p_family", opt(p.p_family)),
            ]),
        ));
        candidates.push(obj(&f));
        summary.push((p.pick, role, Some(out.verdict)));

        md.push_str(&format!(
            "\n## Pick {} ({}): {}\n\nCandidate `{}`, run `{}`, read report `{}`; {} bp per side, {} periods a year, {} periods.\n\n| gate | values | pass |\n|---|---|---|\n",
            p.pick,
            role,
            out.verdict,
            r.candidate_id,
            r.run_sha256,
            r.report_sha256,
            jnum(r.cost_bps),
            jnum(r.ppy),
            r.net.len()
        ));
        for g in &out.gates {
            let values: Vec<String> = g.fields.iter().filter(|(k, _)| *k != "pass").map(|(k, v)| format!("{} {}", k, v.trim_matches('"'))).collect();
            md.push_str(&format!("| {} | {} | {} |\n", g.name, values.join(", "), if g.pass { "yes" } else { "no" }));
        }
        md.push_str("\n| part | periods | net10 | net20 | sharpe | alpha_bh | max_drawdown | exposure | entries | buy-and-hold | cash | bootstrap p |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (kind, rows) in [("", &part_rows), ("year ", &year_rows), ("stratum ", &stratum_rows)] {
            for row in rows.iter() {
                let cells: Vec<String> = row.iter().map(|(_, v)| v.trim_matches('"').to_string()).collect();
                md.push_str(&format!("| {}{} |\n", kind, cells.join(" | ")));
            }
        }
    }
    let holm_rows: Vec<String> = secondaries
        .iter()
        .zip(adjusted.iter())
        .map(|(&(k, p), &ph)| {
            obj(&[("pick", scored[k].pick.pick.to_string()), ("p", jnum(p)), ("p_holm", jnum(ph)), ("pass", (ph <= HOLM_ALPHA).to_string())])
        })
        .collect();
    md.push_str("\n## Secondaries\n\n");
    match reason {
        Some(why) => md.push_str(&format!("Not tested: {}. Their numbers above are context and never change the primary.\n", why)),
        None => {
            md.push_str("Holm over the S5 p of the secondaries (the primary is good):\n\n| pick | p | p_holm | pass |\n|---|---|---|---|\n");
            for (&(k, p), &ph) in secondaries.iter().zip(adjusted.iter()) {
                md.push_str(&format!("| {} | {} | {} | {} |\n", scored[k].pick.pick, jnum(p), jnum(ph), if ph <= HOLM_ALPHA { "yes" } else { "no" }));
            }
        }
    }
    let frozen_json = obj(&[
        ("thresholds", jstr(&t_sha)),
        ("development", development.as_ref().map_or("null".to_string(), |(_, h)| jstr(h))),
        ("subwindows", jstr(&sw_sha)),
        ("years", years.as_ref().map_or("null".to_string(), |(_, h)| jstr(h))),
    ]);
    let json = obj(&[
        ("schema", jstr(SCHEMA)),
        ("ledger_sha256", jstr(&sha256_hex(&ledger_bytes))),
        ("sealed_sha256", jstr(&sealed_sha)),
        ("frozen", frozen_json),
        ("rule_sha256", jstr(&thresholds.rule_sha256)),
        ("regimes_calendar_sha256", jstr(&reg_sha)),
        ("pristine_label", jstr(&o.pristine_label)),
        ("development_source", jstr(if development.is_some() { "development" } else { "thresholds" })),
        ("candidates", format!("[{}]", candidates.join(","))),
        (
            "secondaries",
            obj(&[
                ("tested", tested.to_string()),
                ("reason", reason.map_or("null".to_string(), jstr)),
                ("alpha", jnum(HOLM_ALPHA)),
                ("holm", format!("[{}]", holm_rows.join(","))),
            ]),
        ),
    ]) + "\n";
    Ok(Verdict { json, markdown: md, summary })
}

pub fn usage() -> &'static str {
    "  verdict --ledger FILE --sealed FILE --thresholds FILE [--development FILE]
          --read FILE [--read FILE ...] --subwindows FILE --pristine-label LABEL
          [--years FILE] --regimes FILE [--dsr-trials 51] [--out FILE] [--markdown FILE]
      #1238: the pre-registered verdict of the sealed reads, rcsr-verdict/1 JSON
      (stdout without --out) and Markdown. It opens the ledger, the final sealed
      list, the files frozen in the ledger before the final seal (each must match
      its freeze event: --thresholds the candidate table with the frozen thresholds,
      --development an optional development table for D2 and D3, --subwindows and
      --years the calendars), the locked reads and the lock calendar dated after the
      reads (--regimes, its sha256 equal to a regime_dating event after the final
      seal). It opens no development data and recomputes no threshold. A read needs
      protocol.test_read true, a file that hashes to its protocol.report_sha256, its
      candidate in the sealed list, a test_read event of that candidate and hash,
      selection.run_sha256 equal to exactly one pick's freeze.run_sha256, the frozen
      cost_bps and periods_per_year, and the series of --emit-series --bootstrap N.
      Gates on the read (x net at the read cost, c period_cost, e exposure, b
      buy-and-hold net, ann = mean x periods_per_year): S1 ann(x) > 0 and
      ann(x - c) > 0; S2 ann(x - e b) > 0; S3 -max_drawdown <= the frozen cap; S4
      ann(x) >= the frozen S4 floor; S5 inference.bootstrap.p_mean_positive <= 0.05;
      R1 on bull periods of the lock calendar ann(x) > 0, ann(x - c) > 0,
      ann(x - e b) >= 0; R2 on bear periods ann(x) - ann(b) >= 0 and
      ann(x - e b) >= 0; P1 on the --pristine-label periods ann(x) >= the frozen P1
      floor (a stratum without periods fails); D1 the pick's DSR at N_eff >= 0.95;
      D2 SPA p <= 0.05 and D3 PBO <= 0.2 of the development table (else the
      thresholds table); D2 reads spa.p_family when that table carries
      trials.family (rcsr trials --family-table, #1269), else spa.p_consistent,
      and p_source names it. Verdict: good (every gate), promising (S1 at 10 bp, S2,
      S4, S5 p <= 0.10, D2, DSR >= 0.80), not_confirmed (ann(x) >= 0), falsified.
      Pick 1 is the primary; secondaries get Holm-adjusted S5 p only when the
      primary is good. Report-only: the full lock and each subwindows part, the
      years and the lock strata (net at 10 and 20 bp, Sharpe, alpha, drawdown,
      exposure, entries, buy-and-hold, cash 0, block-bootstrap p), DSR at
      --dsr-trials of the full lock and the non-pristine parts, bits, p_occam and
      p_family of the frozen row. Exit 1 on a ledger fault, 2 on a refused binding
      or input, 3 on an I/O error.
"
}

fn fail(msg: String, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code)
}

/// `rcsr protocol verdict ...`.
pub fn run(args: &[String]) -> ! {
    let mut o = Inputs { dsr_trials: DEFAULT_DSR_TRIALS, ..Inputs::default() };
    let (mut ledger, mut sealed, mut thresholds, mut subwindows, mut pristine, mut regimes) = (None, None, None, None, None, None);
    let (mut out, mut markdown): (Option<String>, Option<String>) = (None, None);
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args.get(i + 1).cloned().unwrap_or_else(|| fail(format!("{} requires a value", flag), 2));
        match flag {
            "--ledger" => ledger = Some(value),
            "--sealed" => sealed = Some(value),
            "--thresholds" => thresholds = Some(value),
            "--development" => o.development = Some(value),
            "--read" => o.reads.push(value),
            "--subwindows" => subwindows = Some(value),
            "--pristine-label" => pristine = Some(value),
            "--years" => o.years = Some(value),
            "--regimes" => regimes = Some(value),
            "--dsr-trials" => {
                o.dsr_trials = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 1.0)
                    .unwrap_or_else(|| fail("--dsr-trials needs a number of at least 1".to_string(), 2))
            }
            "--out" => out = Some(value),
            "--markdown" => markdown = Some(value),
            other => fail(format!("unknown option '{}' for protocol verdict\n\n{}", other, usage()), 2),
        }
        i += 2;
    }
    let need = |v: Option<String>, flag: &str| v.unwrap_or_else(|| fail(format!("protocol verdict needs {}", flag), 2));
    o.ledger = need(ledger, "--ledger");
    o.sealed = need(sealed, "--sealed");
    o.thresholds = need(thresholds, "--thresholds");
    o.subwindows = need(subwindows, "--subwindows");
    o.pristine_label = need(pristine, "--pristine-label");
    o.regimes = need(regimes, "--regimes");
    if o.reads.is_empty() {
        fail("protocol verdict needs at least one --read".to_string(), 2);
    }
    let v = evaluate(&o).unwrap_or_else(|(code, msg)| fail(msg, code));
    if let Some(p) = &markdown {
        crate::atomic::write_text_atomic(p, &v.markdown).unwrap_or_else(|e| fail(e, 3));
    }
    match &out {
        Some(p) => {
            crate::atomic::write_text_atomic(p, &v.json).unwrap_or_else(|e| fail(e, 3));
            for (pick, role, verdict) in &v.summary {
                println!("pick {} ({}): {}", pick, role, verdict.unwrap_or("not read"));
            }
        }
        None => print!("{}", v.json),
    }
    std::process::exit(0)
}
