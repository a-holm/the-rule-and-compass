//! #1237: the sealed-read thresholds that `rcsr trials --table` freezes per pick before the
//! seal (broad-pilot decisions D15-D18): the S4 floor and the P1 floor, nearest-rank
//! percentiles of rolling annualised nets, and the D9 drawdown cap, the larger of a floor and
//! a nearest-rank percentile of rolling compounding max drawdowns. The population is every
//! CPCV path of the pick's configuration, development rows only; a window starts at every
//! period and lies inside one path. The sealed-read verdict reads the frozen values and never
//! recomputes them from development data.

use crate::devlog::Json;
use crate::risk::Drawdown;
use crate::walkforward::{jnum, jstr};

type Res<T> = Result<T, (i32, String)>;

/// The version of the frozen objects (`freeze` at the top level and in every pick).
pub const SCHEMA: &str = "rcsr-trials-freeze/1";
pub const MAP_SCHEMA: &str = "rcsr-trials-freeze-map/1";
const DAY: f64 = 86_400.0;
const MAX_WINDOW_DAYS: f64 = 1_000_000.0;

const METHOD: &str = "{\"population\":\"every CPCV path of the configuration (freeze.cpcv) named by the map, development rows only\",\
\"windows\":\"[t_i, t_i + 86400 window_days) from every period start t_i of a path, used when t_i + 86400 window_days is at most the end \
of the path's last cpcv.segments entry\",\
\"net\":\"mean period net at cost_bps times periods_per_year, over cash at 0\",\
\"drawdown\":\"magnitude of the compounding max drawdown from equity 1 inside the window (pooled.max_drawdown with the sign flipped)\",\
\"percentile\":\"nearest rank over the windows of every path pooled: the k-th smallest value, k = (P n + 99) / 100\",\
\"drawdown_cap\":\"max(floor, percentile_value)\",\
\"cost\":\"period_net, or period_net - (cost_bps / cost_model.per_side_bps - 1) period_cost for a path at another cost\"}";

fn usage<T>(m: impl Into<String>) -> Res<T> {
    Err((2, m.into()))
}

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

fn whole(v: Option<&Json>) -> Option<usize> {
    v.and_then(Json::as_f64).filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0 && *x < 1e15).map(|x| x as usize)
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

/// One rolling-window statistic: the window length in calendar days and the percentile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowRule {
    pub days: u32,
    pub percentile: u32,
}

impl WindowRule {
    fn json(&self) -> String {
        format!("{{\"window_days\":{},\"percentile\":{}}}", self.days, self.percentile)
    }
}

/// The `freeze` key of an `rcsr-trials-rule/1` rule.
#[derive(Clone, Debug, PartialEq)]
pub struct FreezeRule {
    /// `cpcv`: the (N, k) of the population's CPCV assembly.
    pub groups: usize,
    pub test_groups: usize,
    pub cost_bps: f64,
    pub s4: WindowRule,
    pub p1: WindowRule,
    pub cap: WindowRule,
    pub cap_floor: f64,
}

fn fields_of<'a>(v: &'a Json, key: &str, allowed: &[&str], src: &str) -> Res<&'a [(String, Json)]> {
    let Json::Obj(fields) = v else {
        return usage(format!("{}: {} must be an object with {}", src, key, allowed.join(", ")));
    };
    if let Some((k, _)) = fields.iter().find(|(k, _)| !allowed.contains(&k.as_str())) {
        return usage(format!("{}: unknown key '{}.{}'", src, key, k));
    }
    Ok(fields)
}

fn need<'a>(v: &'a Json, key: &str, field: &str, src: &str) -> Res<&'a Json> {
    v.get(field).map_or_else(|| usage(format!("{}: {}.{} is missing", src, key, field)), Ok)
}

fn whole_in(v: &Json, name: &str, lo: f64, hi: f64, src: &str) -> Res<u32> {
    match v.as_f64() {
        Some(x) if x.fract() == 0.0 && x >= lo && x <= hi => Ok(x as u32),
        _ => usage(format!("{}: {} must be a whole number in {}..={}", src, name, jnum(lo), jnum(hi))),
    }
}

fn window_rule(v: &Json, key: &str, extra: &[&str], src: &str) -> Res<WindowRule> {
    let mut allowed = vec!["window_days", "percentile"];
    allowed.extend_from_slice(extra);
    fields_of(v, key, &allowed, src)?;
    Ok(WindowRule {
        days: whole_in(need(v, key, "window_days", src)?, &format!("{}.window_days", key), 1.0, MAX_WINDOW_DAYS, src)?,
        percentile: whole_in(need(v, key, "percentile", src)?, &format!("{}.percentile", key), 1.0, 99.0, src)?,
    })
}

impl FreezeRule {
    /// The `freeze` object of a rule file; `src` names the file in messages.
    pub fn parse(v: &Json, src: &str) -> Res<FreezeRule> {
        fields_of(v, "freeze", &["cpcv", "cost_bps", "s4", "p1", "drawdown_cap"], src)?;
        let (groups, test_groups) = match need(v, "freeze", "cpcv", src)? {
            Json::Arr(a) if a.len() == 2 => match (whole(a.first()), whole(a.get(1))) {
                (Some(n), Some(k)) if (2..=crate::cpcv::MAX_GROUPS).contains(&n) && k >= 1 && k < n => (n, k),
                _ => {
                    return usage(format!(
                        "{}: freeze.cpcv must be [N, k] with whole numbers 2 <= N <= {} and 1 <= k < N",
                        src,
                        crate::cpcv::MAX_GROUPS
                    ))
                }
            },
            _ => return usage(format!("{}: freeze.cpcv must be [N, k], the groups and test groups of the CPCV paths", src)),
        };
        let cost_bps = match need(v, "freeze", "cost_bps", src)?.as_f64() {
            Some(x) if x.is_finite() && x >= 0.0 => x,
            _ => return usage(format!("{}: freeze.cost_bps must be a number of basis points per side, 0 or more", src)),
        };
        let cap_obj = need(v, "freeze", "drawdown_cap", src)?;
        let cap = window_rule(cap_obj, "freeze.drawdown_cap", &["floor"], src)?;
        let cap_floor = match need(cap_obj, "freeze.drawdown_cap", "floor", src)?.as_f64() {
            Some(x) if (0.0..1.0).contains(&x) => x,
            _ => return usage(format!("{}: freeze.drawdown_cap.floor must be a fraction in [0, 1)", src)),
        };
        Ok(FreezeRule {
            groups,
            test_groups,
            cost_bps,
            s4: window_rule(need(v, "freeze", "s4", src)?, "freeze.s4", &[], src)?,
            p1: window_rule(need(v, "freeze", "p1", src)?, "freeze.p1", &[], src)?,
            cap,
            cap_floor,
        })
    }

    /// The canonical JSON of the key.
    pub fn json(&self) -> String {
        format!(
            "{{\"cpcv\":[{},{}],\"cost_bps\":{},\"s4\":{},\"p1\":{},\"drawdown_cap\":{{\"window_days\":{},\"percentile\":{},\"floor\":{}}}}}",
            self.groups,
            self.test_groups,
            jnum(self.cost_bps),
            self.s4.json(),
            self.p1.json(),
            self.cap.days,
            self.cap.percentile,
            jnum(self.cap_floor)
        )
    }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/// One `rows[]` entry of the freeze map.
#[derive(Clone, Debug)]
pub struct MapRow {
    /// The table row: its report as given to `--wf` (`<report>#<archive_rank>` with `--members all`).
    pub row: String,
    pub configuration: String,
    /// The configuration's CPCV path reports.
    pub paths: Vec<String>,
    /// The run.json that is sealed for the row.
    pub run: Option<String>,
}

/// `--freeze-map`: a pre-written input whose SHA-256 goes into the output.
#[derive(Clone, Debug)]
pub struct FreezeMap {
    pub path: String,
    pub sha256: String,
    pub rows: Vec<MapRow>,
}

impl FreezeMap {
    pub fn row(&self, report: &str) -> Option<&MapRow> {
        self.rows.iter().find(|r| r.row == report)
    }
}

pub fn load_map(path: &str) -> Res<FreezeMap> {
    let bytes = std::fs::read(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
    let src = format!("--freeze-map {}", path);
    let text = String::from_utf8_lossy(&bytes);
    let Some(doc) = Json::parse(text.trim()) else {
        return usage(format!("{}: not a JSON object", src));
    };
    fields_of(&doc, "map", &["schema", "rows"], &src)?;
    if doc.get("schema").and_then(Json::as_str) != Some(MAP_SCHEMA) {
        return usage(format!("{}: schema must be {}", src, MAP_SCHEMA));
    }
    let Some(Json::Arr(list)) = doc.get("rows") else {
        return usage(format!("{}: rows must be an array of {{row, configuration, paths, run}}", src));
    };
    let mut rows: Vec<MapRow> = Vec::with_capacity(list.len());
    for (i, r) in list.iter().enumerate() {
        let key = format!("rows[{}]", i);
        fields_of(r, &key, &["row", "configuration", "paths", "run"], &src)?;
        let text = |f: &str| -> Res<String> {
            match r.get(f).and_then(Json::as_str) {
                Some(s) if !s.is_empty() => Ok(s.to_string()),
                _ => usage(format!("{}: {}.{} must be a non-empty string", src, key, f)),
            }
        };
        let paths: Vec<String> = match r.get("paths") {
            Some(Json::Arr(p)) if !p.is_empty() => p
                .iter()
                .map(|x| x.as_str().filter(|s| !s.is_empty()).map(str::to_string))
                .collect::<Option<_>>()
                .map_or_else(|| usage(format!("{}: {}.paths must hold file names", src, key)), Ok)?,
            _ => return usage(format!("{}: {}.paths must be a non-empty array of CPCV path reports", src, key)),
        };
        let run = match r.get("run") {
            None | Some(Json::Null) => None,
            Some(_) => Some(text("run")?),
        };
        let row = MapRow { row: text("row")?, configuration: text("configuration")?, paths, run };
        if rows.iter().any(|x| x.row == row.row) {
            return usage(format!("{}: row '{}' is listed twice", src, row.row));
        }
        rows.push(row);
    }
    Ok(FreezeMap { path: path.to_string(), sha256: crate::manifest::sha256_hex(&bytes), rows })
}

// ---------------------------------------------------------------------------
// Rolling windows
// ---------------------------------------------------------------------------

/// The windows `[t_i, t_i + seconds)` of one path as period index ranges `[start, end)`: one
/// per period start `t_i` with `t_i + seconds <= path_end`. `time` is strictly increasing.
pub fn windows(time: &[f64], path_end: f64, seconds: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut end = 0;
    for (i, &t) in time.iter().enumerate() {
        if t + seconds > path_end {
            break;
        }
        end = end.max(i + 1);
        while end < time.len() && time[end] < t + seconds {
            end += 1;
        }
        out.push((i, end));
    }
    out
}

/// Range queries of the compounding drawdown carry over one series: a segment tree over
/// [`Drawdown::merge`], `O(log n)` merges per range.
pub struct DrawdownTree {
    size: usize,
    node: Vec<Drawdown>,
}

impl DrawdownTree {
    pub fn new(x: &[f64]) -> DrawdownTree {
        let size = x.len().next_power_of_two();
        let mut node = vec![Drawdown::identity(); 2 * size];
        for (i, &v) in x.iter().enumerate() {
            node[size + i].push(v);
        }
        for i in (1..size).rev() {
            node[i] = Drawdown::merge(&node[2 * i], &node[2 * i + 1]);
        }
        DrawdownTree { size, node }
    }

    /// The carry of `x[a..b]` (`a <= b <= x.len()`), equal to pushing those values in order up
    /// to rounding.
    pub fn query(&self, a: usize, b: usize) -> Drawdown {
        let (mut l, mut r) = (a + self.size, b + self.size);
        let (mut left, mut right) = (Drawdown::identity(), Drawdown::identity());
        while l < r {
            if l & 1 == 1 {
                left = Drawdown::merge(&left, &self.node[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                right = Drawdown::merge(&self.node[r], &right);
            }
            l >>= 1;
            r >>= 1;
        }
        Drawdown::merge(&left, &right)
    }
}

/// One rolling window of a path.
#[derive(Clone, Copy, Debug)]
pub struct RollingWindow {
    pub start: usize,
    pub end: usize,
    /// Mean period net (not annualised).
    pub mean: f64,
    /// Magnitude of the compounding max drawdown from equity 1, in [0, 1].
    pub drawdown: f64,
}

/// Every window of `days` calendar days of one path ([`windows`]) with its mean from prefix
/// sums and its drawdown from a [`DrawdownTree`].
pub fn rolling_windows(time: &[f64], x: &[f64], path_end: f64, days: u32) -> Vec<RollingWindow> {
    let spans = windows(time, path_end, DAY * days as f64);
    if spans.is_empty() {
        return Vec::new();
    }
    let mut prefix = Vec::with_capacity(x.len() + 1);
    let mut acc = 0.0;
    prefix.push(acc);
    for &v in x {
        acc += v;
        prefix.push(acc);
    }
    let tree = DrawdownTree::new(x);
    spans
        .into_iter()
        .map(|(a, b)| RollingWindow {
            start: a,
            end: b,
            mean: (prefix[b] - prefix[a]) / (b - a) as f64,
            drawdown: 0.0 - tree.query(a, b).max_drawdown(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------

struct PathFile {
    file: String,
    sha256: String,
    index: usize,
    time: Vec<f64>,
    /// Period net at the rule's cost.
    x: Vec<f64>,
    /// `end` of the last `cpcv.segments` entry.
    end: f64,
}

struct Population {
    paths: usize,
    splits_sha256: String,
    scheme_sha256: String,
    ppy: f64,
    period_seconds: Option<f64>,
    files: Vec<PathFile>,
}

fn text_at(j: &Json, path: &[&str], file: &str) -> Res<String> {
    at(j, path).and_then(Json::as_str).map(str::to_string).map_or_else(|| usage(format!("field '{}' is missing in {}", path.join("."), file)), Ok)
}

fn whole_at(j: &Json, path: &[&str], file: &str) -> Res<usize> {
    whole(at(j, path)).map_or_else(|| usage(format!("field '{}' is not a whole number in {}", path.join("."), file)), Ok)
}

/// The period net of a path at `cost_bps`.
fn priced(j: &Json, net: Vec<f64>, cost_bps: f64, file: &str) -> Res<Vec<f64>> {
    let Some(bps) = at(j, &["cost_model", "per_side_bps"]).and_then(Json::as_f64) else {
        return usage(format!("field 'cost_model.per_side_bps' is missing in {}", file));
    };
    if bps == cost_bps {
        return Ok(net);
    }
    if !(bps > 0.0) {
        return usage(format!(
            "field 'cost_model.per_side_bps' is {} in {}: a path without cost cannot be repriced to freeze.cost_bps {}",
            jnum(bps),
            file,
            jnum(cost_bps)
        ));
    }
    if at(j, &["pooled", "period_cost"]).is_none() {
        return usage(format!(
            "field 'pooled.period_cost' is missing in {}: the path is priced at {} bp per side and freeze.cost_bps is {} \
(assemble the paths from walkforward --emit-series split reports, #1236)",
            file,
            jnum(bps),
            jnum(cost_bps)
        ));
    }
    let cost = crate::trials::series(j, &["pooled", "period_cost"], file)?;
    if cost.len() != net.len() {
        return usage(format!("field 'pooled.period_cost' has {} values but 'pooled.period_net' has {} in {}", cost.len(), net.len(), file));
    }
    let m = cost_bps / bps - 1.0;
    Ok(net.iter().zip(&cost).map(|(x, c)| x - m * c).collect())
}

fn load_population(row: &MapRow, rule: &FreezeRule, map: &str, locked: &[(f64, f64)]) -> Res<Population> {
    let conf = &row.configuration;
    let want = crate::cpcv::path_count(rule.groups, rule.test_groups);
    let mut pop: Option<Population> = None;
    let mut splits: Vec<(String, String, String)> = Vec::new();
    for f in &row.paths {
        let bytes = std::fs::read(f).map_err(|e| (3, format!("{}: {}", f, e)))?;
        let text = String::from_utf8_lossy(&bytes);
        let j = Json::parse(text.trim()).ok_or_else(|| (3, format!("{}: not a JSON report", f)))?;
        crate::trials_table::refuse_locked(&j, f, locked)?;
        crate::trials_table::refuse_null_store(&j, f)?;
        if j.get("mode").and_then(Json::as_str) != Some("cpcv_path") {
            return usage(format!(
                "field 'mode' is not cpcv_path in {}: the population of configuration {} holds the path reports of rcsr trials --cpcv-assemble",
                f, conf
            ));
        }
        for (field, asked) in [("groups", rule.groups), ("test_groups", rule.test_groups)] {
            let got = whole_at(&j, &["cpcv", field], f)?;
            if got != asked {
                return usage(format!("field 'cpcv.{}' is {} in {} but freeze.cpcv asks for {}", field, got, f, asked));
            }
        }
        let paths = whole_at(&j, &["cpcv", "paths"], f)?;
        if paths != want {
            return usage(format!("field 'cpcv.paths' is {} in {} but CPCV({}, {}) has {} paths", paths, f, rule.groups, rule.test_groups, want));
        }
        let index = whole_at(&j, &["cpcv", "path"], f)?;
        if !(1..=paths).contains(&index) {
            return usage(format!("field 'cpcv.path' is {} in {}: not in 1..={}", index, f, paths));
        }
        let ppy = at(&j, &["annualization", "periods_per_year"]).and_then(Json::as_f64).filter(|v| v.is_finite() && *v > 0.0);
        let Some(ppy) = ppy else {
            return usage(format!("field 'annualization.periods_per_year' is missing or not positive in {}", f));
        };
        let period_seconds = at(&j, &["annualization", "period_seconds"]).and_then(Json::as_f64);
        let (splits_sha256, scheme_sha256) = (text_at(&j, &["cpcv", "splits_sha256"], f)?, text_at(&j, &["cpcv", "scheme_sha256"], f)?);
        if let Some(p) = &pop {
            let differs = [
                ("cpcv.splits_sha256", p.splits_sha256 != splits_sha256),
                ("cpcv.scheme_sha256", p.scheme_sha256 != scheme_sha256),
                ("annualization.periods_per_year", p.ppy != ppy),
                ("annualization.period_seconds", p.period_seconds != period_seconds),
            ];
            if let Some((field, _)) = differs.iter().find(|(_, d)| *d) {
                return usage(format!(
                    "the paths of configuration {} differ in field '{}': {} vs {} (the population is the paths of one CPCV assembly)",
                    conf, field, p.files[0].file, f
                ));
            }
            if let Some(other) = p.files.iter().find(|x| x.index == index) {
                return usage(format!("two files of configuration {} have cpcv.path {}: {} and {}", conf, index, other.file, f));
            }
        }
        let Some(Json::Arr(segments)) = at(&j, &["cpcv", "segments"]) else {
            return usage(format!("field 'cpcv.segments' is missing in {}", f));
        };
        let place = format!("cpcv.segments of {}", f);
        let mut end = None;
        for s in segments {
            let split = text_at(s, &["split"], &place)?;
            let sha = text_at(s, &["report_sha256"], &place)?;
            match splits.iter().find(|(x, _, _)| *x == split) {
                Some((_, other, file)) if *other != sha => {
                    return usage(format!(
                        "split {} has two report_sha256 in the population of configuration {}: {} in {} and {} in {}",
                        split, conf, other, file, sha, f
                    ));
                }
                Some(_) => {}
                None => splits.push((split, sha, f.clone())),
            }
            end = Some(whole_at(s, &["end"], &place)? as f64);
        }
        let Some(end) = end else {
            return usage(format!("field 'cpcv.segments' is empty in {}", f));
        };
        let time = crate::trials::series(&j, &["pooled", "period_time"], f)?;
        let net = crate::trials::series(&j, &["pooled", "period_net"], f)?;
        if time.len() != net.len() {
            return usage(format!("field 'pooled.period_time' has {} values but 'pooled.period_net' has {} in {}", time.len(), net.len(), f));
        }
        if let Some(i) = time.windows(2).position(|w| w[1] <= w[0]) {
            return usage(format!("field 'pooled.period_time' is not strictly increasing at period {} in {}", i + 1, f));
        }
        let x = priced(&j, net, rule.cost_bps, f)?;
        let file = PathFile { file: f.clone(), sha256: crate::manifest::sha256_hex(&bytes), index, time, x, end };
        match &mut pop {
            Some(p) => p.files.push(file),
            None => pop = Some(Population { paths, splits_sha256, scheme_sha256, ppy, period_seconds, files: vec![file] }),
        }
    }
    let Some(mut pop) = pop else {
        return usage(format!("--freeze-map {}: configuration {} lists no path", map, conf));
    };
    if let Some(k) = (1..=pop.paths).find(|k| !pop.files.iter().any(|p| p.index == *k)) {
        return usage(format!(
            "cpcv.path {} of {} is missing from the population of configuration {} in --freeze-map {}: the map lists every path of the \
configuration's CPCV({}, {}) assembly",
            k, pop.paths, conf, map, rule.groups, rule.test_groups
        ));
    }
    pop.files.sort_by_key(|p| p.index);
    Ok(pop)
}

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

struct Threshold {
    rule: WindowRule,
    windows: usize,
    rank: usize,
    value: f64,
}

impl Threshold {
    fn head(&self) -> String {
        format!("\"window_days\":{},\"percentile\":{},\"windows\":{},\"rank\":{}", self.rule.days, self.rule.percentile, self.windows, self.rank)
    }
}

fn threshold(pop: &Population, w: WindowRule, key: &str, conf: &str, drawdown: bool) -> Res<Threshold> {
    let mut values: Vec<f64> = Vec::new();
    for p in &pop.files {
        let ws = rolling_windows(&p.time, &p.x, p.end, w.days);
        if ws.is_empty() {
            return usage(format!(
                "no window of {} days ({}.window_days) fits inside path {} of configuration {} ({}): every path needs at least one",
                w.days, key, p.index, conf, p.file
            ));
        }
        values.extend(ws.iter().map(|r| if drawdown { r.drawdown } else { r.mean * pop.ppy }));
    }
    values.sort_by(f64::total_cmp);
    let rank = crate::inference::nearest_rank(values.len(), w.percentile);
    Ok(Threshold { rule: w, windows: values.len(), rank, value: values[rank - 1] })
}

/// The frozen thresholds of one pick.
pub struct Frozen {
    /// The pick's `freeze` object.
    pub json: String,
    pub configuration: String,
    pub paths: usize,
    pub s4_floor: f64,
    pub p1_floor: f64,
    pub drawdown_cap: f64,
    pub run_sha256: Option<String>,
}

/// The SHA-256 of a run.json's bytes, as `RunFile::load` computes it, after the run parses.
fn run_sha256(path: &str, map: &str) -> Res<String> {
    let text = std::fs::read_to_string(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
    crate::run_json::RunFile::parse(&text).map_err(|e| (2, format!("--freeze-map {}: run {}: {}", map, path, e)))?;
    Ok(crate::manifest::sha256_hex(text.as_bytes()))
}

/// The thresholds of the pick whose map entry is `row`.
pub fn freeze_pick(row: &MapRow, rule: &FreezeRule, map: &FreezeMap, locked: &[(f64, f64)]) -> Res<Frozen> {
    let conf = &row.configuration;
    let pop = load_population(row, rule, &map.path, locked)?;
    let s4 = threshold(&pop, rule.s4, "freeze.s4", conf, false)?;
    let p1 = threshold(&pop, rule.p1, "freeze.p1", conf, false)?;
    let cap = threshold(&pop, rule.cap, "freeze.drawdown_cap", conf, true)?;
    let cap_value = rule.cap_floor.max(cap.value);
    let run_sha = row.run.as_deref().map(|r| run_sha256(r, &map.path)).transpose()?;
    let files: Vec<String> = pop
        .files
        .iter()
        .map(|p| format!("{{\"file\":{},\"sha256\":{},\"path\":{},\"periods\":{}}}", jstr(&p.file), jstr(&p.sha256), p.index, p.time.len()))
        .collect();
    let json = format!(
        "{{\"schema\":{},\"configuration\":{},\"cost_bps\":{},\"periods_per_year\":{},\"period_seconds\":{},\
\"population\":{{\"groups\":{},\"test_groups\":{},\"paths\":{},\"splits_sha256\":{},\"scheme_sha256\":{},\"files\":[{}]}},\
\"s4_floor\":{{{},\"value\":{}}},\"p1_floor\":{{{},\"value\":{}}},\
\"drawdown_cap\":{{{},\"percentile_value\":{},\"floor\":{},\"value\":{}}},\"run\":{},\"run_sha256\":{}}}",
        jstr(SCHEMA),
        jstr(conf),
        jnum(rule.cost_bps),
        jnum(pop.ppy),
        pop.period_seconds.map_or("null".to_string(), jnum),
        rule.groups,
        rule.test_groups,
        pop.paths,
        jstr(&pop.splits_sha256),
        jstr(&pop.scheme_sha256),
        files.join(","),
        s4.head(),
        jnum(s4.value),
        p1.head(),
        jnum(p1.value),
        cap.head(),
        jnum(cap.value),
        jnum(rule.cap_floor),
        jnum(cap_value),
        row.run.as_deref().map_or("null".to_string(), jstr),
        run_sha.as_deref().map_or("null".to_string(), jstr)
    );
    Ok(Frozen {
        json,
        configuration: conf.clone(),
        paths: pop.paths,
        s4_floor: s4.value,
        p1_floor: p1.value,
        drawdown_cap: cap_value,
        run_sha256: run_sha,
    })
}

/// The top-level `freeze` object of a table with a freeze rule.
pub fn top_json(rule_sha256: &str, map: &FreezeMap) -> String {
    format!(
        "{{\"schema\":{},\"rule_sha256\":{},\"map\":{},\"map_sha256\":{},\"method\":{}}}",
        jstr(SCHEMA),
        jstr(rule_sha256),
        jstr(&map.path),
        jstr(&map.sha256),
        METHOD
    )
}
