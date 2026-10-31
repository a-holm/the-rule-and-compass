//! Reserved, gated derivatives feature family: funding, open interest, basis,
//! long/short ratio, taker flow and liquidations.
//!
//! No such data exists locally, so every column is `available:false`. A column
//! becomes available only through [`verify`], which needs a local snapshot file
//! that passes a coverage-by-symbol/date check and a point-in-time join check.
//! Nothing here downloads data, and no code path writes a zero, an empty column
//! or a forward-filled value: a missing value is `NaN`.
//!
//! Snapshot file: CSV with header `field,symbol,publish_time,value`. `field` is
//! a raw vendor field (see [`Spec::fields`]), `publish_time` is when the value
//! became known (epoch seconds or ISO 8601), `value` is a finite number. A row
//! with an empty or non-finite value is dropped. Two rows with the same
//! `(field, symbol, publish_time)` and different values are an error.

use crate::manifest::sha256_hex;
use std::collections::BTreeMap;

/// How a column is derived from its raw fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Join {
    /// Latest value with `publish_time <= t`, `NaN` when older than `max_age`.
    Asof,
    /// Sum of the three latest published values.
    Carry3,
    /// `(mark / spot - 1) * 1e4` from rows with `publish_time == t` for both legs.
    Basis,
    /// `v(t) / v(t - window) - 1`, both sides as-of known snapshots.
    Change(f64),
    /// No join is implemented; the column stays unavailable.
    None,
}

pub struct Spec {
    pub name: &'static str,
    pub fields: &'static [&'static str],
    pub source: &'static str,
    pub publication_lag: &'static str,
    pub cadence_secs: f64,
    pub max_age_secs: f64,
    pub join: Join,
}

const H: f64 = 3600.0;

pub const SPECS: &[Spec] = &[
    Spec { name: "funding_rate", fields: &["fundingRate"], source: "Binance USD-M fundingRate history",
        publication_lag: "known at payment time (8h cadence)", cadence_secs: 8.0 * H, max_age_secs: 16.0 * H, join: Join::Asof },
    Spec { name: "funding_carry_24h", fields: &["fundingRate"], source: "Binance USD-M fundingRate history",
        publication_lag: "three latest published payments", cadence_secs: 8.0 * H, max_age_secs: 16.0 * H, join: Join::Carry3 },
    Spec { name: "basis_bps", fields: &["markPrice", "spotClose"], source: "Binance USD-M mark price klines and spot klines",
        publication_lag: "none, both legs at the same instant", cadence_secs: H, max_age_secs: 0.0, join: Join::Basis },
    Spec { name: "oi", fields: &["sumOpenInterest"], source: "Binance openInterestHist",
        publication_lag: "snapshot publish_time per row", cadence_secs: H, max_age_secs: 2.0 * H, join: Join::Asof },
    Spec { name: "oi_chg_1h", fields: &["sumOpenInterest"], source: "Binance openInterestHist",
        publication_lag: "known snapshots only", cadence_secs: H, max_age_secs: 2.0 * H, join: Join::Change(H) },
    Spec { name: "oi_chg_24h", fields: &["sumOpenInterest"], source: "Binance openInterestHist",
        publication_lag: "known snapshots only", cadence_secs: H, max_age_secs: 2.0 * H, join: Join::Change(24.0 * H) },
    Spec { name: "ls_ratio", fields: &["longShortRatio"], source: "Binance global long/short account ratio",
        publication_lag: "vendor publication lag, publish_time per row", cadence_secs: H, max_age_secs: 2.0 * H, join: Join::Asof },
    Spec { name: "taker_buy_sell_ratio", fields: &["buySellRatio"], source: "Binance futures taker buy/sell volume",
        publication_lag: "vendor publication lag, publish_time per row", cadence_secs: H, max_age_secs: 2.0 * H, join: Join::Asof },
    Spec { name: "liq_long", fields: &["liqLongNotional"], source: "Binance liquidation stream archive",
        publication_lag: "event timestamp, no revision", cadence_secs: H, max_age_secs: 0.0, join: Join::None },
    Spec { name: "liq_short", fields: &["liqShortNotional"], source: "Binance liquidation stream archive",
        publication_lag: "event timestamp, no revision", cadence_secs: H, max_age_secs: 0.0, join: Join::None },
];

pub const NAMES: &[&str] = &[
    "funding_rate", "funding_carry_24h", "basis_bps", "oi", "oi_chg_1h", "oi_chg_24h",
    "ls_ratio", "taker_buy_sell_ratio", "liq_long", "liq_short",
];

/// Minimum fraction of calendar days, per symbol and field, with a published row.
pub const MIN_COVERAGE: f64 = 0.99;

pub fn spec(name: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.name == name)
}

fn flip_condition(s: &Spec) -> String {
    let join = if s.join == Join::None { "no join is implemented, so the column cannot flip" } else { "PIT join check passes" };
    format!(
        "a local snapshot has raw field(s) {} for every panel symbol on at least {:.0}% of days, and the {}",
        s.fields.join("+"),
        MIN_COVERAGE * 100.0,
        join
    )
}

/// Refusal text for a requested derivatives column; names the missing raw field.
pub fn refusal(name: &str) -> Option<String> {
    let s = spec(name)?;
    Some(format!(
        "no local snapshot: missing raw field {} ({}); intentionally not filled with zero",
        s.fields.join("+"),
        s.source
    ))
}

/// Refusal for the whole `derivatives` group.
pub fn group_refusal() -> String {
    let fields: Vec<&str> = SPECS.iter().flat_map(|s| s.fields.iter().copied()).collect();
    format!(
        "feature group 'derivatives' is unavailable: no local snapshot, missing raw fields {}; \
         intentionally not filled with zero",
        dedup(fields).join(", ")
    )
}

fn dedup(mut v: Vec<&str>) -> Vec<&str> {
    v.sort_unstable();
    v.dedup();
    v
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Extra registry JSON members (leading comma) for a derivatives entry.
pub fn registry_extra_json(name: &str) -> String {
    match spec(name) {
        Some(s) => format!(
            ",\"required_source\":{},\"publication_lag\":{},\"required_raw_field\":{},\"flip_condition\":{}",
            jstr(s.source),
            jstr(s.publication_lag),
            jstr(&s.fields.join("+")),
            jstr(&flip_condition(s))
        ),
        None => String::new(),
    }
}

/// Derivatives inventory: source, lag and coverage condition per name.
pub fn inventory_json() -> String {
    let items: Vec<String> = SPECS
        .iter()
        .map(|s| {
            format!(
                "{{\"name\":{},\"family\":\"derivatives\",\"available\":false,\"source\":null,\
                 \"required_source\":{},\"required_raw_field\":{},\"publication_lag\":{},\
                 \"coverage_condition\":{},\"flip_condition\":{}}}",
                jstr(s.name),
                jstr(s.source),
                jstr(&s.fields.join("+")),
                jstr(s.publication_lag),
                jstr(&format!("{:.0}% of days per symbol and field", MIN_COVERAGE * 100.0)),
                jstr(&flip_condition(s))
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// A snapshot that passed [`verify`]. Only this value can make columns available.
#[derive(Clone, Debug)]
pub struct Verified {
    pub path: String,
    pub sha256: String,
    pub columns: Vec<String>,
}

impl Verified {
    pub fn available(&self, name: &str) -> bool {
        self.columns.iter().any(|c| c == name)
    }
}

/// Manifest block. Default (`None`): not available, no source, no columns.
pub fn manifest_json(v: Option<&Verified>) -> String {
    match v {
        None => "{\"available\":false,\"source\":null,\"columns\":[]}".to_string(),
        Some(v) => format!(
            "{{\"available\":true,\"source\":{},\"sha256\":{},\"columns\":[{}]}}",
            jstr(&v.path),
            jstr(&v.sha256),
            v.columns.iter().map(|c| jstr(c)).collect::<Vec<_>>().join(",")
        ),
    }
}

/// Resolve requested derivatives names. Without a verified snapshot, or for a
/// column it does not cover, this refuses and names the missing raw field.
pub fn resolve(names: &[&str], v: Option<&Verified>) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for &n in names {
        if n == "derivatives" {
            return Err(group_refusal());
        }
        if spec(n).is_none() {
            return Err(format!("unknown derivatives feature '{n}'"));
        }
        match v {
            Some(v) if v.available(n) => out.push(n.to_string()),
            _ => return Err(format!("registered feature '{n}' is unavailable: {}", refusal(n).unwrap())),
        }
    }
    Ok(out)
}

type Series = Vec<(f64, f64)>;

/// Local snapshot, keyed by `(field, symbol)`, rows sorted by publish time.
#[derive(Default, Debug)]
pub struct Snapshot {
    pub series: BTreeMap<(String, String), Series>,
}

impl Snapshot {
    pub fn from_csv(text: &str) -> Result<Snapshot, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty derivatives snapshot")?;
        if header.trim() != "field,symbol,publish_time,value" {
            return Err(format!("derivatives snapshot header must be 'field,symbol,publish_time,value', got '{}'", header.trim()));
        }
        let mut snap = Snapshot::default();
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            if c.len() != 4 {
                return Err(format!("derivatives snapshot line {}: expected 4 columns", i + 2));
            }
            let t = crate::data::parse_time(c[2])
                .filter(|t| t.is_finite())
                .ok_or_else(|| format!("derivatives snapshot line {}: bad publish_time '{}'", i + 2, c[2]))?;
            let v = match c[3].trim().parse::<f64>() {
                Ok(v) if v.is_finite() => v,
                _ => continue,
            };
            snap.series.entry((c[0].trim().to_string(), c[1].trim().to_string())).or_default().push((t, v));
        }
        for ((f, s), rows) in snap.series.iter_mut() {
            rows.sort_by(|a, b| a.0.total_cmp(&b.0));
            for w in rows.windows(2) {
                if w[0].0 == w[1].0 && w[0].1 != w[1].1 {
                    return Err(format!("derivatives snapshot: conflicting values for {f} {s} at {}", w[0].0));
                }
            }
            rows.dedup_by(|a, b| a.0 == b.0);
        }
        Ok(snap)
    }

    pub fn load(path: &str) -> Result<Snapshot, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read derivatives snapshot {path}: {e}"))?;
        Snapshot::from_csv(&text)
    }

    fn get(&self, field: &str, symbol: &str) -> &[(f64, f64)] {
        self.series.get(&(field.to_string(), symbol.to_string())).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Index of the latest row with `publish_time <= t`.
fn asof_index(rows: &[(f64, f64)], t: f64) -> Option<usize> {
    rows.partition_point(|r| r.0 <= t).checked_sub(1)
}

fn asof_value(rows: &[(f64, f64)], t: f64, max_age: f64) -> f64 {
    match asof_index(rows, t) {
        Some(i) if t - rows[i].0 <= max_age => rows[i].1,
        _ => f64::NAN,
    }
}

/// One derivatives column for `symbol` at the given bar times (seconds).
/// A bar sees only rows with `publish_time <= t`; anything else is `NaN`.
pub fn column(snap: &Snapshot, name: &str, symbol: &str, bar_times: &[f64]) -> Result<Vec<f64>, String> {
    let s = spec(name).ok_or_else(|| format!("unknown derivatives feature '{name}'"))?;
    let rows = snap.get(s.fields[0], symbol);
    Ok(bar_times
        .iter()
        .map(|&t| match s.join {
            Join::Asof => asof_value(rows, t, s.max_age_secs),
            Join::Carry3 => match asof_index(rows, t) {
                Some(i) if i >= 2 && t - rows[i].0 <= s.max_age_secs => rows[i - 2..=i].iter().map(|r| r.1).sum(),
                _ => f64::NAN,
            },
            Join::Change(w) => {
                let now = asof_value(rows, t, s.max_age_secs);
                let then = asof_value(rows, t - w, s.max_age_secs);
                if now.is_finite() && then.is_finite() && then > 0.0 { now / then - 1.0 } else { f64::NAN }
            }
            Join::Basis => {
                let at = |f: &str| {
                    let r = snap.get(f, symbol);
                    r.binary_search_by(|x| x.0.total_cmp(&t)).ok().map(|i| r[i].1)
                };
                match (at(s.fields[0]), at(s.fields[1])) {
                    (Some(mark), Some(spot)) if spot > 0.0 && mark > 0.0 => (mark / spot - 1.0) * 1e4,
                    _ => f64::NAN,
                }
            }
            Join::None => f64::NAN,
        })
        .collect())
}

/// Fraction of calendar days in `[start, end]` (seconds) with a published row.
fn day_coverage(rows: &[(f64, f64)], start: f64, end: f64) -> f64 {
    let d0 = (start / 86_400.0).floor() as i64;
    let d1 = (end / 86_400.0).floor() as i64;
    let days: std::collections::BTreeSet<i64> = rows
        .iter()
        .filter(|r| r.0 >= start && r.0 <= end)
        .map(|r| (r.0 / 86_400.0).floor() as i64)
        .collect();
    days.len() as f64 / (d1 - d0 + 1) as f64
}

/// PIT check on the loaded rows: a bar just before a row's publish time never
/// sees that row, and a bar at its publish time does.
fn pit_ok(rows: &[(f64, f64)]) -> bool {
    rows.iter().enumerate().all(|(i, r)| {
        asof_index(rows, r.0 - 1e-3).map_or(true, |j| j < i) && asof_index(rows, r.0) == Some(i)
    })
}

/// Verify a snapshot against the panel symbols and date range. A column is
/// covered when every symbol has every required field on at least
/// [`MIN_COVERAGE`] of the days and the PIT join check passes. Errors when no
/// column is covered, naming what is missing.
pub fn verify(snap: &Snapshot, path: &str, bytes: &[u8], symbols: &[String], start: f64, end: f64) -> Result<Verified, String> {
    let mut columns = Vec::new();
    let mut problems = Vec::new();
    for s in SPECS {
        if s.join == Join::None {
            problems.push(format!("{}: no join implemented", s.name));
            continue;
        }
        let mut bad: Option<String> = None;
        'f: for f in s.fields {
            for sym in symbols {
                let rows = snap.get(f, sym);
                if rows.is_empty() {
                    bad = Some(format!("missing raw field {f} for {sym}"));
                    break 'f;
                }
                let cov = day_coverage(rows, start, end);
                if cov < MIN_COVERAGE {
                    bad = Some(format!("raw field {f} for {sym} covers {:.1}% of days", cov * 100.0));
                    break 'f;
                }
                if !pit_ok(rows) {
                    bad = Some(format!("PIT join check failed for {f} {sym}"));
                    break 'f;
                }
            }
        }
        match bad {
            None => columns.push(s.name.to_string()),
            Some(b) => problems.push(format!("{}: {b}", s.name)),
        }
    }
    if columns.is_empty() {
        return Err(format!("derivatives snapshot {path} covers no column: {}", problems.join("; ")));
    }
    Ok(Verified { path: path.to_string(), sha256: sha256_hex(bytes), columns })
}
