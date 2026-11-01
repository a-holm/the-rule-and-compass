//! Statistics and the paired comparison behind `rcsr-bench compare`.
//!
//! Everything is pure `std`: a small JSON reader for the NDJSON the bench
//! writes, Wilson intervals, the exact McNemar test, the front hypervolume and
//! the pairing logic.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

const Z95: f64 = 1.959_963_984_540_054;

/// Wilson 95 % interval of `k` successes in `n` trials. `(0, 1)` when `n == 0`.
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n = n as f64;
    let p = k as f64 / n;
    let z2 = Z95 * Z95;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Exact two-sided McNemar p-value `min(1, 2 P(X <= min(b, c)))` with
/// `X ~ Binomial(b + c, 1/2)`. `1` when there are no discordant pairs.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let lo = b.min(c);
    let mut ln_fact = vec![0.0f64; n + 1];
    for i in 1..=n {
        ln_fact[i] = ln_fact[i - 1] + (i as f64).ln();
    }
    let ln_half_n = -(n as f64) * std::f64::consts::LN_2;
    let mut tail = 0.0;
    for k in 0..=lo {
        tail += (ln_fact[n] - ln_fact[k] - ln_fact[n - k] + ln_half_n).exp();
    }
    (2.0 * tail).min(1.0)
}

/// Median of an unsorted slice; `None` when empty.
pub fn median_of(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    })
}

/// Hypervolume dominated by a front in the plane
/// `(min(1, max(0, 1 - R²)), cost / max_complexity)` with reference point
/// `(1, 1)`. Each point is `(test R², cost)`. Higher is better.
pub fn front_hv(points: &[(f64, f64)], max_complexity: f64) -> f64 {
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .map(|&(r2, cost)| {
            let err = if r2.is_finite() {
                (1.0 - r2).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let cx = (cost / max_complexity).clamp(0.0, 1.0);
            (cx, err)
        })
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut best_err = 1.0f64;
    let mut area = 0.0;
    for i in 0..pts.len() {
        best_err = best_err.min(pts[i].1);
        let next = if i + 1 < pts.len() { pts[i + 1].0 } else { 1.0 };
        area += (1.0 - best_err) * (next - pts[i].0);
    }
    area
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(v) => Some(v),
            _ => None,
        }
    }
}

/// Parse one JSON document.
pub fn parse_json(text: &str) -> Result<Json, String> {
    let b = text.as_bytes();
    let mut i = 0;
    let v = parse_value(b, &mut i)?;
    skip_ws(b, &mut i);
    if i != b.len() {
        return Err(format!("trailing characters at byte {}", i));
    }
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') {
        *i += 1;
    }
}

fn parse_value(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    let c = *b.get(*i).ok_or("unexpected end of input")?;
    match c {
        b'{' => {
            *i += 1;
            let mut kv = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b'}') {
                *i += 1;
                return Ok(Json::Obj(kv));
            }
            loop {
                skip_ws(b, i);
                let k = match parse_value(b, i)? {
                    Json::Str(s) => s,
                    _ => return Err("object key is not a string".to_string()),
                };
                skip_ws(b, i);
                if b.get(*i) != Some(&b':') {
                    return Err(format!("expected ':' at byte {}", i));
                }
                *i += 1;
                kv.push((k, parse_value(b, i)?));
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b'}') => {
                        *i += 1;
                        return Ok(Json::Obj(kv));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {}", i)),
                }
            }
        }
        b'[' => {
            *i += 1;
            let mut items = Vec::new();
            skip_ws(b, i);
            if b.get(*i) == Some(&b']') {
                *i += 1;
                return Ok(Json::Arr(items));
            }
            loop {
                items.push(parse_value(b, i)?);
                skip_ws(b, i);
                match b.get(*i) {
                    Some(b',') => *i += 1,
                    Some(b']') => {
                        *i += 1;
                        return Ok(Json::Arr(items));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {}", i)),
                }
            }
        }
        b'"' => {
            *i += 1;
            let mut out: Vec<u8> = Vec::new();
            loop {
                let ch = *b.get(*i).ok_or("unterminated string")?;
                *i += 1;
                match ch {
                    b'"' => break,
                    b'\\' => {
                        let e = *b.get(*i).ok_or("unterminated escape")?;
                        *i += 1;
                        match e {
                            b'"' => out.push(b'"'),
                            b'\\' => out.push(b'\\'),
                            b'/' => out.push(b'/'),
                            b'n' => out.push(b'\n'),
                            b'r' => out.push(b'\r'),
                            b't' => out.push(b'\t'),
                            b'b' => out.push(8),
                            b'f' => out.push(12),
                            b'u' => {
                                let hex = b
                                    .get(*i..*i + 4)
                                    .and_then(|h| std::str::from_utf8(h).ok())
                                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                                    .ok_or("bad \\u escape")?;
                                *i += 4;
                                let ch = char::from_u32(hex).unwrap_or('\u{fffd}');
                                let mut buf = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                            _ => return Err("bad escape".to_string()),
                        }
                    }
                    other => out.push(other),
                }
            }
            String::from_utf8(out)
                .map(Json::Str)
                .map_err(|_| "string is not UTF-8".to_string())
        }
        b't' if b[*i..].starts_with(b"true") => {
            *i += 4;
            Ok(Json::Bool(true))
        }
        b'f' if b[*i..].starts_with(b"false") => {
            *i += 5;
            Ok(Json::Bool(false))
        }
        b'n' if b[*i..].starts_with(b"null") => {
            *i += 4;
            Ok(Json::Null)
        }
        _ => {
            let start = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            std::str::from_utf8(&b[start..*i])
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .map(Json::Num)
                .ok_or_else(|| format!("unexpected character at byte {}", start))
        }
    }
}

// ---------------------------------------------------------------------------
// Paired comparison
// ---------------------------------------------------------------------------

/// Config keys that may differ between the two sides of a comparison.
const KNOB_KEYS: [&str; 2] = ["knobs", "build"];

/// The fields of one NDJSON run line that the comparison reads.
#[derive(Clone, Debug)]
pub struct RunRec {
    pub suite: String,
    pub problem: String,
    pub noise: Option<f64>,
    pub seed: u64,
    pub recovered: Option<bool>,
    pub oracle_recovered: Option<bool>,
    pub signal_recovered: Option<bool>,
    pub hit_s: Option<f64>,
    pub hit_evals: Option<f64>,
    pub front_hv: Option<f64>,
    pub seconds: f64,
}

impl RunRec {
    fn key(&self) -> (String, String, String, u64) {
        let noise = match self.noise {
            Some(n) => format!("{}", n),
            None => "null".to_string(),
        };
        (self.suite.clone(), self.problem.clone(), noise, self.seed)
    }

    fn criterion(&self, c: Criterion) -> Option<bool> {
        match c {
            Criterion::Recovered => self.recovered,
            Criterion::OracleRecovered => self.oracle_recovered,
            Criterion::SignalRecovered => self.signal_recovered,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Criterion {
    Recovered,
    OracleRecovered,
    SignalRecovered,
}

impl Criterion {
    pub fn parse(s: &str) -> Option<Criterion> {
        match s {
            "recovered" => Some(Criterion::Recovered),
            "oracle_recovered" => Some(Criterion::OracleRecovered),
            "signal_recovered" => Some(Criterion::SignalRecovered),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Criterion::Recovered => "recovered",
            Criterion::OracleRecovered => "oracle_recovered",
            Criterion::SignalRecovered => "signal_recovered",
        }
    }
}

fn opt_f64(j: &Json, key: &str) -> Option<f64> {
    j.get(key).and_then(Json::as_f64)
}
fn opt_bool(j: &Json, key: &str) -> Option<bool> {
    j.get(key).and_then(Json::as_bool)
}

/// Read the run lines of one NDJSON file. Returns the runs and their configs.
pub fn load_runs(text: &str, name: &str) -> Result<(Vec<RunRec>, Vec<Json>), String> {
    let mut runs = Vec::new();
    let mut configs = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let j = parse_json(line).map_err(|e| format!("{}:{}: {}", name, n + 1, e))?;
        if j.get("type").and_then(Json::as_str) != Some("run") {
            continue;
        }
        let need = |key: &str| -> Result<String, String> {
            j.get(key)
                .and_then(Json::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("{}:{}: run line without '{}'", name, n + 1, key))
        };
        runs.push(RunRec {
            suite: need("suite")?,
            problem: need("problem")?,
            noise: opt_f64(&j, "noise"),
            seed: opt_f64(&j, "seed")
                .ok_or_else(|| format!("{}:{}: run line without 'seed'", name, n + 1))?
                as u64,
            recovered: opt_bool(&j, "recovered"),
            oracle_recovered: opt_bool(&j, "oracle_recovered"),
            signal_recovered: opt_bool(&j, "signal_recovered"),
            hit_s: opt_f64(&j, "hit_s"),
            hit_evals: opt_f64(&j, "hit_evals"),
            front_hv: opt_f64(&j, "front_hv"),
            seconds: opt_f64(&j, "seconds").unwrap_or(0.0),
        });
        configs.push(
            j.get("config")
                .cloned()
                .ok_or_else(|| format!("{}:{}: run line without 'config'", name, n + 1))?,
        );
    }
    Ok((runs, configs))
}

fn strip_knobs(config: &Json) -> Json {
    match config {
        Json::Obj(kv) => Json::Obj(
            kv.iter()
                .filter(|(k, _)| !KNOB_KEYS.contains(&k.as_str()))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Per-problem line of the comparison.
#[derive(Clone, Debug)]
pub struct ProblemRow {
    pub suite: String,
    pub problem: String,
    pub pairs: usize,
    pub base_hits: usize,
    pub treat_hits: usize,
    pub wins: usize,
    pub losses: usize,
}

#[derive(Clone, Debug)]
pub struct CompareReport {
    pub criterion: Criterion,
    pub pairs: usize,
    pub base_hits: usize,
    pub treat_hits: usize,
    pub base_ci: (f64, f64),
    pub treat_ci: (f64, f64),
    /// Pairs hit by the base only.
    pub b: usize,
    /// Pairs hit by the treatment only.
    pub c: usize,
    pub p_value: f64,
    pub hit_s_ratio: Option<f64>,
    pub hit_s_pairs: usize,
    pub hit_evals_ratio: Option<f64>,
    pub hit_evals_pairs: usize,
    pub front_hv_diff: Option<f64>,
    pub base_seconds: f64,
    pub treat_seconds: f64,
    pub rows: Vec<ProblemRow>,
    /// Keys of pairs left out because the criterion is undefined on a side.
    pub dropped: Vec<String>,
}

/// Pair the runs of two sides and compare them. Every error is a usage error.
pub fn compare(
    base: &[(String, String)],
    treat: &[(String, String)],
    criterion: Criterion,
) -> Result<CompareReport, String> {
    let load_side = |side: &[(String, String)],
                     label: &str|
     -> Result<(BTreeMap<(String, String, String, u64), RunRec>, Vec<Json>), String> {
        let mut map = BTreeMap::new();
        let mut cfgs = Vec::new();
        for (name, text) in side {
            let (runs, c) = load_runs(text, name)?;
            cfgs.extend(c);
            for r in runs {
                let k = r.key();
                if map.insert(k.clone(), r).is_some() {
                    return Err(format!(
                        "{}: duplicate run {}/{}/noise {}/seed {}",
                        label, k.0, k.1, k.2, k.3
                    ));
                }
            }
        }
        if map.is_empty() {
            return Err(format!("{}: no run lines", label));
        }
        Ok((map, cfgs))
    };
    let (bm, bc) = load_side(base, "base")?;
    let (tm, tc) = load_side(treat, "treat")?;

    let reference = strip_knobs(&bc[0]);
    for (side, cfgs) in [("base", &bc), ("treat", &tc)] {
        if let Some(bad) = cfgs.iter().position(|c| strip_knobs(c) != reference) {
            return Err(format!(
                "{} run {} has a different data configuration (not a search knob)",
                side, bad
            ));
        }
    }
    let only_base: Vec<_> = bm.keys().filter(|k| !tm.contains_key(*k)).collect();
    let only_treat: Vec<_> = tm.keys().filter(|k| !bm.contains_key(*k)).collect();
    if !only_base.is_empty() || !only_treat.is_empty() {
        let show = |k: &(String, String, String, u64)| format!("{}/{}/noise {}/seed {}", k.0, k.1, k.2, k.3);
        return Err(format!(
            "the two sides do not hold the same runs: {} only in base (first {}), {} only in treat (first {})",
            only_base.len(),
            only_base.first().map(|k| show(k)).unwrap_or_default(),
            only_treat.len(),
            only_treat.first().map(|k| show(k)).unwrap_or_default()
        ));
    }

    let mut rows: BTreeMap<(String, String), ProblemRow> = BTreeMap::new();
    let (mut base_hits, mut treat_hits, mut b, mut c) = (0, 0, 0, 0);
    let mut hit_s_ratios = Vec::new();
    let mut evals_ratios = Vec::new();
    let mut hv_diffs = Vec::new();
    let (mut base_seconds, mut treat_seconds) = (0.0, 0.0);
    for (k, br) in &bm {
        let tr = &tm[k];
        let (bv, tv) = match (br.criterion(criterion), tr.criterion(criterion)) {
            (Some(x), Some(y)) => (x, y),
            _ => {
                return Err(format!(
                    "criterion {} is not defined for {}/{}/noise {}/seed {}",
                    criterion.name(),
                    k.0,
                    k.1,
                    k.2,
                    k.3
                ))
            }
        };
        let row = rows
            .entry((k.0.clone(), k.1.clone()))
            .or_insert_with(|| ProblemRow {
                suite: k.0.clone(),
                problem: k.1.clone(),
                pairs: 0,
                base_hits: 0,
                treat_hits: 0,
                wins: 0,
                losses: 0,
            });
        row.pairs += 1;
        if bv {
            base_hits += 1;
            row.base_hits += 1;
        }
        if tv {
            treat_hits += 1;
            row.treat_hits += 1;
        }
        if bv && !tv {
            b += 1;
            row.losses += 1;
        }
        if tv && !bv {
            c += 1;
            row.wins += 1;
        }
        if let (Some(x), Some(y)) = (br.hit_s, tr.hit_s) {
            if x > 0.0 {
                hit_s_ratios.push(y / x);
            }
        }
        if let (Some(x), Some(y)) = (br.hit_evals, tr.hit_evals) {
            if x > 0.0 {
                evals_ratios.push(y / x);
            }
        }
        if let (Some(x), Some(y)) = (br.front_hv, tr.front_hv) {
            hv_diffs.push(y - x);
        }
        base_seconds += br.seconds;
        treat_seconds += tr.seconds;
    }
    let pairs = bm.len();
    Ok(CompareReport {
        criterion,
        pairs,
        base_hits,
        treat_hits,
        base_ci: wilson(base_hits, pairs),
        treat_ci: wilson(treat_hits, pairs),
        b,
        c,
        p_value: mcnemar_exact(b, c),
        hit_s_pairs: hit_s_ratios.len(),
        hit_s_ratio: median_of(&hit_s_ratios),
        hit_evals_pairs: evals_ratios.len(),
        hit_evals_ratio: median_of(&evals_ratios),
        front_hv_diff: median_of(&hv_diffs),
        base_seconds,
        treat_seconds,
        rows: rows.into_values().collect(),
        dropped: Vec::new(),
    })
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{:.4}", x)).unwrap_or_else(|| "n/a".to_string())
}

fn json_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{}", x),
        _ => "null".to_string(),
    }
}

impl CompareReport {
    pub fn to_text(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!(
            "rcsr-bench compare  criterion={}  pairs={}\n",
            self.criterion.name(),
            self.pairs
        ));
        let pct = |k: usize| 100.0 * k as f64 / self.pairs as f64;
        o.push_str(&format!(
            "base   {}/{} ({:.1}%)  Wilson 95% [{:.3}, {:.3}]\n",
            self.base_hits,
            self.pairs,
            pct(self.base_hits),
            self.base_ci.0,
            self.base_ci.1
        ));
        o.push_str(&format!(
            "treat  {}/{} ({:.1}%)  Wilson 95% [{:.3}, {:.3}]\n",
            self.treat_hits,
            self.pairs,
            pct(self.treat_hits),
            self.treat_ci.0,
            self.treat_ci.1
        ));
        o.push_str(&format!(
            "discordant: b (base only) = {}, c (treat only) = {}\n",
            self.b, self.c
        ));
        o.push_str(&format!("McNemar exact two-sided p = {:.5}\n", self.p_value));
        o.push_str(&format!(
            "median paired hit_s ratio treat/base = {} over {} pairs\n",
            fmt_opt(self.hit_s_ratio),
            self.hit_s_pairs
        ));
        o.push_str(&format!(
            "median paired hit_evals ratio treat/base = {} over {} pairs\n",
            fmt_opt(self.hit_evals_ratio),
            self.hit_evals_pairs
        ));
        o.push_str(&format!(
            "median paired front_hv difference treat-base = {}\n",
            fmt_opt(self.front_hv_diff)
        ));
        o.push_str(&format!(
            "summed seconds: base {:.1}, treat {:.1}\n",
            self.base_seconds, self.treat_seconds
        ));
        o.push_str(&format!(
            "\n{:<8} {:<18} {:>5} {:>5} {:>5} {:>5} {:>6}\n",
            "suite", "problem", "pairs", "base", "treat", "wins", "losses"
        ));
        for r in &self.rows {
            o.push_str(&format!(
                "{:<8} {:<18} {:>5} {:>5} {:>5} {:>5} {:>6}\n",
                r.suite, r.problem, r.pairs, r.base_hits, r.treat_hits, r.wins, r.losses
            ));
        }
        o
    }

    pub fn to_json(&self) -> String {
        let mut o = String::new();
        o.push_str("{\n");
        o.push_str(&format!("\"criterion\": \"{}\",\n", self.criterion.name()));
        o.push_str(&format!("\"pairs\": {},\n", self.pairs));
        o.push_str(&format!(
            "\"base\": {{\"hits\": {}, \"rate\": {}, \"wilson_lo\": {}, \"wilson_hi\": {}}},\n",
            self.base_hits,
            self.base_hits as f64 / self.pairs as f64,
            self.base_ci.0,
            self.base_ci.1
        ));
        o.push_str(&format!(
            "\"treat\": {{\"hits\": {}, \"rate\": {}, \"wilson_lo\": {}, \"wilson_hi\": {}}},\n",
            self.treat_hits,
            self.treat_hits as f64 / self.pairs as f64,
            self.treat_ci.0,
            self.treat_ci.1
        ));
        o.push_str(&format!("\"b\": {}, \"c\": {},\n", self.b, self.c));
        o.push_str(&format!("\"mcnemar_p\": {},\n", self.p_value));
        o.push_str(&format!(
            "\"hit_s_ratio_median\": {}, \"hit_s_pairs\": {},\n",
            json_opt(self.hit_s_ratio),
            self.hit_s_pairs
        ));
        o.push_str(&format!(
            "\"hit_evals_ratio_median\": {}, \"hit_evals_pairs\": {},\n",
            json_opt(self.hit_evals_ratio),
            self.hit_evals_pairs
        ));
        o.push_str(&format!(
            "\"front_hv_diff_median\": {},\n",
            json_opt(self.front_hv_diff)
        ));
        o.push_str(&format!(
            "\"base_seconds\": {}, \"treat_seconds\": {},\n",
            self.base_seconds, self.treat_seconds
        ));
        o.push_str("\"problems\": [\n");
        for (i, r) in self.rows.iter().enumerate() {
            o.push_str(&format!(
                "  {{\"suite\": \"{}\", \"problem\": \"{}\", \"pairs\": {}, \"base_hits\": {}, \"treat_hits\": {}, \"wins\": {}, \"losses\": {}}}{}\n",
                r.suite,
                r.problem,
                r.pairs,
                r.base_hits,
                r.treat_hits,
                r.wins,
                r.losses,
                if i + 1 < self.rows.len() { "," } else { "" }
            ));
        }
        o.push_str("]\n}\n");
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_wilson_interval() {
        let (lo, hi) = wilson(7, 10);
        assert!((lo - 0.397).abs() < 5e-4, "lo = {lo}");
        assert!((hi - 0.892).abs() < 5e-4, "hi = {hi}");
        assert_eq!(wilson(0, 0), (0.0, 1.0));
    }

    #[test]
    fn bench_mcnemar_exact() {
        assert!((mcnemar_exact(0, 6) - 0.03125).abs() < 1e-12);
        assert_eq!(mcnemar_exact(3, 3), 1.0);
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert!((mcnemar_exact(6, 0) - 0.03125).abs() < 1e-12);
    }

    #[test]
    fn bench_front_hv() {
        let one = front_hv(&[(1.0, 3.0)], 30.0);
        assert!((one - 0.9).abs() < 1e-12, "{one}");
        let two = front_hv(&[(1.0, 3.0), (0.5, 1.0)], 30.0);
        assert!((two - (0.9 + 0.5 * 2.0 / 30.0)).abs() < 1e-12, "{two}");
        assert_eq!(format!("{:.4}", two), "0.9333");
        assert_eq!(front_hv(&[], 30.0), 0.0);
        assert_eq!(front_hv(&[(f64::NEG_INFINITY, 3.0)], 30.0), 0.0);
    }

    #[test]
    fn json_round_trip_reads_numbers_strings_and_null() {
        let j = parse_json(r#"{"a": 1.5e-3, "b": "x\"yé", "c": null, "d": [true, false]}"#).unwrap();
        assert_eq!(j.get("a").unwrap().as_f64(), Some(1.5e-3));
        assert_eq!(j.get("b").unwrap().as_str(), Some("x\"y\u{e9}"));
        assert_eq!(j.get("c"), Some(&Json::Null));
        assert!(parse_json("{\"a\": }").is_err());
    }

    fn line(problem: &str, seed: u64, rec: bool, gen: &str) -> String {
        format!(
            "{{\"type\":\"run\",\"suite\":\"hard\",\"problem\":\"{}\",\"noise\":0,\"seed\":{},\"recovered\":{},\"oracle_recovered\":{},\"signal_recovered\":null,\"hit_s\":1.0,\"hit_evals\":100,\"front_hv\":0.5,\"seconds\":2.0,\"config\":{{\"generator\":\"{}\",\"knobs\":{{\"generations\":{}}}}}}}\n",
            problem, seed, rec, rec, gen, seed
        )
    }

    fn side(lines: &[String]) -> Vec<(String, String)> {
        vec![("f".to_string(), lines.concat())]
    }

    #[test]
    fn compare_with_itself_has_no_discordant_pairs() {
        let s = side(&[line("a", 1, true, "g"), line("a", 2, false, "g")]);
        let r = compare(&s, &s, Criterion::Recovered).unwrap();
        assert_eq!((r.b, r.c, r.p_value), (0, 0, 1.0));
        assert_eq!(r.pairs, 2);
        assert_eq!(r.hit_s_ratio, Some(1.0));
    }

    #[test]
    fn compare_counts_wins_and_losses() {
        let base = side(&[line("a", 1, false, "g"), line("a", 2, true, "g"), line("b", 1, false, "g")]);
        let treat = side(&[line("a", 1, true, "g"), line("a", 2, true, "g"), line("b", 1, true, "g")]);
        let r = compare(&base, &treat, Criterion::Recovered).unwrap();
        assert_eq!((r.b, r.c), (0, 2));
        assert_eq!(r.rows[0].wins, 1);
    }

    #[test]
    fn bench_compare_rejects_mismatched_key_sets() {
        let base = side(&[line("a", 1, true, "g"), line("a", 2, true, "g")]);
        let treat = side(&[line("a", 1, true, "g")]);
        let e = compare(&base, &treat, Criterion::Recovered).unwrap_err();
        assert!(e.contains("do not hold the same runs"), "{e}");
    }

    #[test]
    fn bench_compare_rejects_changed_data_generator() {
        let base = side(&[line("a", 1, true, "g1")]);
        let treat = side(&[line("a", 1, true, "g2")]);
        let e = compare(&base, &treat, Criterion::Recovered).unwrap_err();
        assert!(e.contains("different data configuration"), "{e}");
    }

    fn line_oracle(problem: &str, seed: u64, oracle: &str) -> String {
        line(problem, seed, true, "g").replace("\"oracle_recovered\":true", &format!("\"oracle_recovered\":{}", oracle))
    }

    #[test]
    fn compare_drops_pairs_with_undefined_criterion() {
        let base = side(&[line_oracle("a", 1, "true"), line_oracle("a", 2, "null"), line_oracle("b", 1, "false")]);
        let treat = side(&[line_oracle("a", 1, "false"), line_oracle("a", 2, "true"), line_oracle("b", 1, "true")]);
        let r = compare(&base, &treat, Criterion::OracleRecovered).unwrap();
        assert_eq!(r.pairs, 2);
        assert_eq!((r.b, r.c), (1, 1));
        assert_eq!(r.dropped, vec!["hard/a/noise 0/seed 2".to_string()]);
        assert!(r.to_text().contains("dropped pairs (criterion undefined on a side): 1"));
        assert!(r.to_text().contains("hard/a/noise 0/seed 2"));
        assert!(r.to_json().contains("\"dropped\": {\"count\": 1"));
    }

    #[test]
    fn compare_rejects_a_criterion_undefined_for_every_pair() {
        let s = side(&[line_oracle("a", 1, "null"), line_oracle("a", 2, "null")]);
        let e = compare(&s, &s, Criterion::OracleRecovered).unwrap_err();
        assert!(e.contains("not defined for any pair"), "{e}");
    }

    #[test]
    fn compare_allows_changed_search_knobs() {
        let base = side(&[line("a", 1, true, "g")]);
        let mut t = line("a", 1, true, "g");
        t = t.replace("\"generations\":1", "\"generations\":99");
        let r = compare(&base, &side(&[t]), Criterion::Recovered);
        assert!(r.is_ok());
    }
}
