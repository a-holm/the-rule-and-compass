//! `rcsr prep` — raw OHLCV archives and CSVs straight to a `.rcsb` store
//! (issue #123), with no intermediate CSV.
//!
//! The pipeline is:
//!
//! ```text
//! discover (CSV / ZIP, per symbol)
//!   -> parse + timestamp normalisation + duplicate aggregation
//!   -> optional migration splices (verified, overlap removed)
//!   -> resample (calendar-aware) + causal multi-timeframe joins
//!   -> causal features + forward label
//!   -> column store + JSON manifest
//! ```
//!
//! It reads Binance Vision headerless klines (12 columns), Bitstamp klines
//! (6/7 columns) and aliased OHLCV headers, streams one archive at a time,
//! parallelises by symbol, sorts symbols deterministically, and writes the store
//! directly through [`crate::store::write_store`]. `--dry-run` prints the audit
//! and writes nothing.

use crate::data::{Column, Table};
use crate::features::{self, LabelSpec};
use crate::ohlcv::{
    apply_migrations, join_column_causal, resample_calendar, Bar, Calendar,
    Migration, Timeframe,
};
use crate::{manifest, store, zip};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const EXIT_USAGE: i32 = 2;
const EXIT_INPUT: i32 = 3;

static SPOOL_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Auto,
    Binance,
    Bitstamp,
    Ohlcv,
}

impl Layout {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Layout::Auto),
            "binance" => Ok(Layout::Binance),
            "bitstamp" => Ok(Layout::Bitstamp),
            "ohlcv" => Ok(Layout::Ohlcv),
            other => Err(format!(
                "unknown --format '{}' (expected auto, binance, bitstamp or ohlcv)",
                other
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Layout::Auto => "auto",
            Layout::Binance => "binance",
            Layout::Bitstamp => "bitstamp",
            Layout::Ohlcv => "ohlcv",
        }
    }
}

const CANONICAL: &[&str] = &[
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
];

fn alias(name: &str) -> Option<&'static str> {
    let n = name
        .trim()
        .trim_matches('"')
        .to_ascii_lowercase()
        .replace(' ', "_");
    Some(match n.as_str() {
        "time" | "timestamp" | "date" | "datetime" | "open_time" | "opentime" | "unix"
        | "ts" => "time",
        "open" | "o" => "open",
        "high" | "h" => "high",
        "low" | "l" => "low",
        "close" | "c" => "close",
        "volume" | "vol" | "v" | "base_volume" | "basevolume" => "volume",
        "quote_volume" | "quotevolume" | "volume_quote" | "qv" => "quote_volume",
        "trades" | "count" | "n_trades" | "number_of_trades" => "trades",
        "taker_buy_base" | "taker_buy_volume" | "takerbuybase" => "taker_buy_base",
        "taker_buy_quote" | "taker_buy_quote_volume" | "takerbuyquote" => "taker_buy_quote",
        "ignore" | "close_time" | "closetime" => return None,
        _ => return None,
    })
}

/// What one build did; printed by `--dry-run` and recorded in the manifest.
#[derive(Clone, Debug, Default)]
pub struct Audit {
    pub layout: String,
    pub symbols: usize,
    pub files: usize,
    pub rows_in: usize,
    pub rows_out: usize,
    pub duplicates: usize,
    pub non_monotonic: usize,
    pub invalid: usize,
    pub unit: String,
    pub unit_converted: usize,
    pub resample: String,
    pub gaps: usize,
    pub zero_volume: usize,
    pub features: Vec<String>,
    pub feature_zero_variance: Vec<String>,
    pub feature_nan_rows: usize,
    pub label: String,
    pub label_gap_rows: usize,
    pub label_stride_rows: usize,
    pub migrations: usize,
    pub columns: Vec<String>,
    pub notes: Vec<String>,
}

impl Audit {
    pub fn text_lines(&self) -> Vec<String> {
        let mut out = vec![
            format!(
                "# prep: layout={} symbols={} files={} rows_in={} rows_out={}",
                self.layout, self.symbols, self.files, self.rows_in, self.rows_out
            ),
            format!(
                "# prep: duplicates={} non_monotonic={} invalid={} unit={} converted={}",
                self.duplicates, self.non_monotonic, self.invalid, self.unit, self.unit_converted
            ),
            format!(
                "# prep: resample={} gaps={} zero_volume={} migrations={}",
                self.resample, self.gaps, self.zero_volume, self.migrations
            ),
            format!("# prep: features={}", self.features.join(",")),
            format!(
                "# prep: label={} gap_rows={} stride_rows={}",
                self.label, self.label_gap_rows, self.label_stride_rows
            ),
            format!("# prep: columns={}", self.columns.join(",")),
            format!(
                "# prep: feature_nan_rows={} zero_variance={}",
                self.feature_nan_rows,
                self.feature_zero_variance.join(",")
            ),
        ];
        out.extend(self.notes.iter().map(|n| format!("# prep: {}", n)));
        out
    }

    pub fn json(&self) -> String {
        let arr = |v: &[String]| {
            v.iter()
                .map(|s| jstr(s))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut s = String::from("{");
        let _ = write!(
            s,
            "\"layout\":{},\"symbols\":{},\"files\":{},\"rows_in\":{},\"rows_out\":{},",
            jstr(&self.layout),
            self.symbols,
            self.files,
            self.rows_in,
            self.rows_out
        );
        let _ = write!(
            s,
            "\"duplicates\":{},\"non_monotonic\":{},\"invalid\":{},\"unit\":{},\"unit_converted\":{},",
            self.duplicates, self.non_monotonic, self.invalid, jstr(&self.unit), self.unit_converted
        );
        let _ = write!(
            s,
            "\"resample\":{},\"gaps\":{},\"zero_volume\":{},\"migrations\":{},",
            jstr(&self.resample),
            self.gaps,
            self.zero_volume,
            self.migrations
        );
        let _ = write!(
            s,
            "\"features\":[{}],\"feature_zero_variance\":[{}],\"feature_nan_rows\":{},",
            arr(&self.features),
            arr(&self.feature_zero_variance),
            self.feature_nan_rows
        );
        let _ = write!(
            s,
            "\"label\":{},\"label_gap_rows\":{},\"label_stride_rows\":{},",
            jstr(&self.label),
            self.label_gap_rows,
            self.label_stride_rows
        );
        let _ = write!(
            s,
            "\"columns\":[{}],\"notes\":[{}]}}",
            arr(&self.columns),
            arr(&self.notes)
        );
        s
    }
}

/// How a raw timestamp is interpreted. UTC seconds are the canonical unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tz {
    Utc,
    /// Fixed offset east of UTC, in seconds.
    Fixed(i64),
    /// A civil zone resolved through a calendar's DST rules.
    Civil(Calendar),
}

impl Tz {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("utc") || s == "Z" {
            return Ok(Tz::Utc);
        }
        if let Some(cal) = Calendar::parse(s) {
            return Ok(if cal == Calendar::Crypto247 {
                Tz::Utc
            } else {
                Tz::Civil(cal)
            });
        }
        if s.eq_ignore_ascii_case("america/new_york") {
            return Ok(Tz::Civil(Calendar::Nyse));
        }
        if s.eq_ignore_ascii_case("europe/london") {
            return Ok(Tz::Civil(Calendar::Lse));
        }
        let (sign, rest) = match s.as_bytes().first() {
            Some(b'+') => (1i64, &s[1..]),
            Some(b'-') => (-1i64, &s[1..]),
            _ => return Err(format!("unknown timezone '{}'", s)),
        };
        let mut it = rest.split(':');
        let h: i64 = it
            .next()
            .and_then(|x| x.parse().ok())
            .ok_or_else(|| format!("bad timezone offset '{}'", s))?;
        let m: i64 = it.next().unwrap_or("0").parse().unwrap_or(0);
        Ok(Tz::Fixed(sign * (h * 3600 + m * 60)))
    }

    pub fn name(self) -> String {
        match self {
            Tz::Utc => "UTC".to_string(),
            Tz::Fixed(s) => format!("{}{:02}:{:02}", if s < 0 { '-' } else { '+' }, s.abs() / 3600, (s.abs() % 3600) / 60),
            Tz::Civil(c) => c.timezone().to_string(),
        }
    }

    fn to_utc(self, local: i64) -> i64 {
        match self {
            Tz::Utc => local,
            Tz::Fixed(s) => local - s,
            Tz::Civil(c) => {
                let a = local - c.utc_offset(local);
                local - c.utc_offset(a)
            }
        }
    }
}

/// Everything a build needs. Constructed from the CLI or directly in Rust.
#[derive(Clone, Debug)]
pub struct Options {
    pub inputs: Vec<String>,
    pub format: Layout,
    pub symbol: Option<String>,
    pub tz: Tz,
    pub calendar: Calendar,
    pub resample: Option<Timeframe>,
    pub mtf: Vec<Timeframe>,
    pub features: Vec<String>,
    pub label: Option<LabelSpec>,
    pub migrations: Vec<Migration>,
    pub out: Option<String>,
    pub spec_path: Option<String>,
    pub dry_run: bool,
    pub jobs: usize,
    pub group_rows: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            inputs: Vec::new(),
            format: Layout::Auto,
            symbol: None,
            tz: Tz::Utc,
            calendar: Calendar::Crypto247,
            resample: None,
            mtf: Vec::new(),
            features: Vec::new(),
            label: None,
            migrations: Vec::new(),
            out: None,
            spec_path: None,
            dry_run: false,
            jobs: 1,
            group_rows: store::DEFAULT_GROUP_ROWS,
        }
    }
}

/// The result of a build: the table, the audit and the manifest text.
#[derive(Clone, Debug)]
pub struct Build {
    pub table: Table,
    pub audit: Audit,
    pub manifest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub path: PathBuf,
    pub symbol: String,
}

/// Expand files and directories into a deterministic symbol-sorted source
/// list. A ZIP or CSV is one source; a directory is searched recursively.
pub fn discover(inputs: &[String], override_symbol: Option<&str>) -> Result<Vec<Source>, String> {
    let mut out = Vec::new();
    for input in inputs {
        let p = PathBuf::from(input);
        if p.is_dir() {
            let mut files = Vec::new();
            walk(&p, &mut files)?;
            for f in files {
                let symbol = symbol_of(&f, Some(&p), override_symbol);
                out.push(Source { path: f, symbol });
            }
        } else if p.is_file() {
            let symbol = symbol_of(&p, None, override_symbol);
            out.push(Source { path: p, symbol });
        } else {
            return Err(format!("input '{}' does not exist", input));
        }
    }
    if out.is_empty() {
        return Err("no .csv/.zip inputs found".to_string());
    }
    out.sort_by(|a, b| a.symbol.cmp(&b.symbol).then(a.path.cmp(&b.path)));
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out)?;
        } else if is_data_file(&p) {
            out.push(p);
        }
    }
    Ok(())
}

fn is_data_file(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("csv") | Some("tsv") | Some("txt") | Some("zip")
    )
}

fn symbol_of(path: &Path, root: Option<&Path>, override_symbol: Option<&str>) -> String {
    if let Some(s) = override_symbol {
        return s.to_ascii_uppercase();
    }
    // A Binance Vision layout is <root>/<SYMBOL>/<SYMBOL>-<tf>-<month>.zip;
    // the parent directory is authoritative when it is below the input root.
    if let Some(root) = root {
        if let Some(parent) = path.parent() {
            if parent != root {
                if let Some(name) = parent.file_name().and_then(|x| x.to_str()) {
                    if !name.is_empty() && name != "." {
                        return name.to_ascii_uppercase();
                    }
                }
            }
        }
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("symbol");
    let base = stem
        .split(['-', '_'])
        .next()
        .unwrap_or(stem)
        .trim();
    if base.is_empty() {
        "SYMBOL".to_string()
    } else {
        base.to_ascii_uppercase()
    }
}

/// Parse one source into raw bars. Header detection is by aliases, so a
/// headerless Binance/Bitstamp export and a named OHLCV CSV both work.
fn parse_source(
    src: &Source,
    layout: Layout,
    tz: Tz,
    audit: &mut Audit,
) -> Result<Vec<Bar>, String> {
    let mut texts: Vec<(String, String)> = Vec::new();
    if src.path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("zip")) == Some(true) {
        let archive = zip::ZipArchive::open(&src.path.to_string_lossy())?;
        for (name, bytes) in archive.read_extension(".csv")? {
            texts.push((name.clone(), String::from_utf8_lossy(&bytes).to_string()));
        }
        for (name, bytes) in archive.read_extension(".txt")? {
            texts.push((name.clone(), String::from_utf8_lossy(&bytes).to_string()));
        }
        if texts.is_empty() {
            return Err(format!("{}: ZIP has no .csv entry", src.path.display()));
        }
    } else {
        let text = std::fs::read_to_string(&src.path)
            .map_err(|e| format!("{}: {}", src.path.display(), e))?;
        texts.push((src.path.display().to_string(), text));
    }
    let mut bars = Vec::new();
    for (name, text) in texts {
        bars.extend(parse_text(&name, &text, layout, tz, audit)?);
    }
    Ok(bars)
}

fn parse_text(
    name: &str,
    text: &str,
    layout: Layout,
    tz: Tz,
    audit: &mut Audit,
) -> Result<Vec<Bar>, String> {
    let mut lines = text.lines().enumerate().filter_map(|(i, l)| {
        let l = l.trim_end_matches('\r').trim();
        if l.is_empty() || l.starts_with('#') {
            None
        } else {
            Some((i + 1, l))
        }
    });
    let first = lines.next().ok_or_else(|| format!("{}: empty file", name))?;
    let delim = if first.1.contains(',') {
        ','
    } else if first.1.contains('\t') {
        '\t'
    } else if first.1.contains(';') {
        ';'
    } else {
        ' '
    };
    let split = |s: &str| -> Vec<String> {
        s.split(delim)
            .map(|x| x.trim().trim_matches('"').to_string())
            .collect()
    };
    let fields = split(first.1);
    let has_header = fields.iter().any(|f| alias(f).is_some() || f.parse::<f64>().is_err());
    let layout = if layout == Layout::Auto {
        if has_header {
            Layout::Ohlcv
        } else {
            match fields.len() {
                12 => Layout::Binance,
                6 | 7 => Layout::Bitstamp,
                n => {
                    return Err(format!(
                        "{}: unrecognised headerless layout with {} columns; pass --format",
                        name, n
                    ))
                }
            }
        }
    } else {
        layout
    };
    let mut index: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut rows: Vec<(usize, &str)> = Vec::new();
    match layout {
        Layout::Binance => {
            for (k, v) in [
                ("time", 0),
                ("open", 1),
                ("high", 2),
                ("low", 3),
                ("close", 4),
                ("volume", 5),
                ("quote_volume", 7),
                ("trades", 8),
                ("taker_buy_base", 9),
                ("taker_buy_quote", 10),
            ] {
                index.insert(k, v);
            }
            rows.push(first);
        }
        Layout::Bitstamp => {
            for (k, v) in [
                ("time", 0),
                ("open", 1),
                ("high", 2),
                ("low", 3),
                ("close", 4),
                ("volume", 5),
            ] {
                index.insert(k, v);
            }
            rows.push(first);
        }
        Layout::Ohlcv | Layout::Auto => {
            if !has_header {
                return Err(format!("{}: --format ohlcv needs a header row", name));
            }
            for (i, f) in fields.iter().enumerate() {
                if let Some(k) = alias(f) {
                    index.insert(k, i);
                }
            }
        }
    }
    for req in ["time", "open", "high", "low", "close"] {
        if !index.contains_key(req) {
            return Err(format!("{}: missing required column '{}'", name, req));
        }
    }
    rows.extend(lines);
    let mut bars = Vec::with_capacity(rows.len());
    let mut unit_seen = String::from("s");
    for (line_no, line) in rows {
        let f = split(line);
        let min = index.values().copied().max().unwrap_or(0);
        if f.len() <= min {
            return Err(format!(
                "{}:{}: {} columns, need at least {}",
                name,
                line_no,
                f.len(),
                min + 1
            ));
        }
        let num = |k: &str| -> Result<f64, String> {
            let Some(&i) = index.get(k) else {
                return Ok(0.0);
            };
            let raw = &f[i];
            raw.parse::<f64>()
                .map_err(|_| format!("{}:{}: '{}' is not numeric for '{}'", name, line_no, raw, k))
        };
        let raw_time = num("time")?;
        let (secs, div, unit) = normalise_time(raw_time);
        if !secs.is_finite() {
            return Err(format!("{}:{}: non-finite timestamp", name, line_no));
        }
        if unit != "s" {
            audit.unit_converted += 1;
            if rank(unit) > rank(&unit_seen) {
                unit_seen = unit.to_string();
            }
        }
        let local = secs as i64;
        let time = tz.to_utc(local);
        let _ = div;
        let open = num("open")?;
        let high = num("high")?;
        let low = num("low")?;
        let close = num("close")?;
        if !(open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()) {
            audit.invalid += 1;
            continue;
        }
        if high < low || open <= 0.0 || close <= 0.0 {
            return Err(format!(
                "{}:{}: invalid OHLC open={} high={} low={} close={}",
                name, line_no, open, high, low, close
            ));
        }
        let volume = num("volume").unwrap_or(0.0);
        let quote_volume = num("quote_volume").unwrap_or(0.0);
        let trades = num("trades").unwrap_or(0.0);
        let taker_buy_base = num("taker_buy_base").unwrap_or(0.0);
        let taker_buy_quote = num("taker_buy_quote").unwrap_or(0.0);
        if volume == 0.0 {
            audit.zero_volume += 1;
        }
        bars.push(Bar {
            time,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trades,
            taker_buy_base,
            taker_buy_quote,
            complete: true,
            traded: volume > 0.0 || trades > 0.0,
        });
        audit.rows_in += 1;
    }
    audit.unit = unit_seen;
    Ok(bars)
}

fn rank(u: &str) -> u8 {
    match u {
        "ms" => 1,
        "us" => 2,
        "ns" => 3,
        _ => 0,
    }
}

/// Normalise a numeric epoch to whole seconds. The unit is inferred from the
/// magnitude: seconds < 1e11, ms < 1e14, us < 1e17, otherwise ns.
fn normalise_time(t: f64) -> (f64, i64, &'static str) {
    let a = t.abs();
    let (div, unit) = if a < 1e11 {
        (1i64, "s")
    } else if a < 1e14 {
        (1_000, "ms")
    } else if a < 1e17 {
        (1_000_000, "us")
    } else {
        (1_000_000_000, "ns")
    };
    ((t / div as f64).floor(), div, unit)
}

/// Sort and aggregate duplicate instants: open = first, high = max, low = min,
/// close = last, flows summed. Non-monotonic input is counted, not trusted.
fn normalise_bars(mut bars: Vec<Bar>, audit: &mut Audit) -> Vec<Bar> {
    let mut non_monotonic = 0usize;
    for w in bars.windows(2) {
        if w[1].time < w[0].time {
            non_monotonic += 1;
        }
    }
    audit.non_monotonic += non_monotonic;
    bars.sort_by_key(|b| b.time);
    let mut out: Vec<Bar> = Vec::with_capacity(bars.len());
    for b in bars {
        match out.last_mut() {
            Some(prev) if prev.time == b.time => {
                audit.duplicates += 1;
                prev.high = prev.high.max(b.high);
                prev.low = prev.low.min(b.low);
                prev.close = b.close;
                prev.volume += b.volume;
                prev.quote_volume += b.quote_volume;
                prev.trades += b.trades;
                prev.taker_buy_base += b.taker_buy_base;
                prev.taker_buy_quote += b.taker_buy_quote;
                prev.traded |= b.traded;
                prev.complete &= b.complete;
            }
            _ => out.push(b),
        }
    }
    out
}

/// Apply a spec file (`key=value` per line). Explicit CLI options win; the
/// file fills the fields still at their defaults.
pub fn load_spec(opts: &mut Options, path: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line
            .split_once('=')
            .ok_or_else(|| format!("{}:{}: expected key=value", path, i + 1))?;
        map.insert(k.trim().to_string(), v.trim().to_string());
    }
    // Fixed order: the label horizon may be a timeframe that depends on
    // `resample`, so the file's line order must not matter.
    if opts.resample.is_none() {
        if let Some(v) = map.get("resample") {
            opts.resample = Some(Timeframe::parse(v)?);
        }
    }
    if opts.mtf.is_empty() {
        if let Some(v) = map.get("mtf") {
            opts.mtf = parse_mtf(v)?;
        }
    }
    if opts.calendar == Calendar::Crypto247 {
        if let Some(v) = map.get("calendar") {
            opts.calendar = Calendar::parse(v)
                .ok_or_else(|| format!("unknown calendar '{}'", v))?;
        }
    }
    if opts.tz == Tz::Utc {
        if let Some(v) = map.get("tz") {
            opts.tz = Tz::parse(v)?;
        }
    }
    if opts.migrations.is_empty() {
        if let Some(v) = map.get("migrations") {
            opts.migrations = parse_migrations(v)?;
        }
    }
    if opts.features.is_empty() {
        if let Some(v) = map.get("features") {
            opts.features = features::expand(v)?;
        }
    }
    if opts.label.is_none() {
        if let Some(v) = map.get("label") {
            let tf = opts.resample.unwrap_or(Timeframe::M1);
            opts.label = Some(LabelSpec::parse(v, tf)?);
        }
    }
    opts.spec_path = Some(path.to_string());
    Ok(())
}

pub fn parse_mtf(s: &str) -> Result<Vec<Timeframe>, String> {
    let mut out = Vec::new();
    for p in s.split(',') {
        if p.trim().is_empty() {
            continue;
        }
        let tf = Timeframe::parse(p.trim())?;
        if !out.contains(&tf) {
            out.push(tf);
        }
    }
    Ok(out)
}

/// `OLD->NEW:k[,OLD->NEW:k...]`, optionally `:tol` as a fourth field.
pub fn parse_migrations(s: &str) -> Result<Vec<Migration>, String> {
    let mut out = Vec::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (pair, k) = item
            .split_once(':')
            .ok_or_else(|| format!("migration '{}' needs OLD->NEW:k", item))?;
        let (old, new) = pair
            .split_once("->")
            .ok_or_else(|| format!("migration '{}' needs OLD->NEW:k", item))?;
        let mut parts = k.split(':');
        let k: f64 = parts
            .next()
            .unwrap_or("")
            .parse()
            .map_err(|_| format!("migration '{}': bad k", item))?;
        if !(k.is_finite() && k > 0.0) {
            return Err(format!("migration '{}': k must be positive", item));
        }
        let mut m = Migration::new(old.trim(), new.trim(), k);
        if let Some(t) = parts.next() {
            m.tolerance = t
                .parse()
                .map_err(|_| format!("migration '{}': bad tolerance", item))?;
        }
        out.push(m);
    }
    Ok(out)
}

struct SymbolPanel {
    symbol: String,
    /// Bucket times in seconds; the canonical OHLCV values live in the spool.
    times: Vec<i64>,
    rows: usize,
    /// Extra (feature/label/MTF) column names, in order, after the canonical 10.
    names: Vec<String>,
    spool: PathBuf,
    warmup: usize,
    gap_rows: usize,
    stride_rows: usize,
    notes: Vec<String>,
}

/// Write one symbol's canonical OHLCV and extra columns to a column-major
/// spool. The spool is what keeps peak RSS near the final store: the worker's
/// bars and feature columns are dropped as soon as the file is written.
fn write_spool(path: &Path, symbol: &str, bars: &[Bar], values: &[(String, Vec<f64>)]) -> Result<(), String> {
    use std::io::Write;
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut w = std::io::BufWriter::new(file);
    let mut names: Vec<(&str, &[f64])> = vec![("ts", &[])];
    for n in &CANONICAL[1..] {
        names.push((n, &[]));
    }
    for (n, _) in values {
        names.push((n.as_str(), &[]));
    }
    let _ = symbol;
    w.write_all(&(names.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for (n, _) in &names {
        w.write_all(&(n.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        w.write_all(n.as_bytes()).map_err(|e| e.to_string())?;
    }
    w.write_all(&(bars.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    let write_col = |w: &mut std::io::BufWriter<std::fs::File>, f: &dyn Fn(usize) -> f64| -> Result<(), String> {
        let mut buf = Vec::with_capacity(bars.len() * 8);
        for i in 0..bars.len() {
            buf.extend_from_slice(&f(i).to_le_bytes());
        }
        w.write_all(&buf).map_err(|e| e.to_string())
    };
    write_col(&mut w, &|i| bars[i].time as f64)?;
    for (i, field) in [
        (0usize, "open"),
        (1, "high"),
        (2, "low"),
        (3, "close"),
        (4, "volume"),
        (5, "quote_volume"),
        (6, "trades"),
        (7, "taker_buy_base"),
        (8, "taker_buy_quote"),
    ] {
        let _ = field;
        write_col(&mut w, &|r| match i {
            0 => bars[r].open,
            1 => bars[r].high,
            2 => bars[r].low,
            3 => bars[r].close,
            4 => bars[r].volume,
            5 => bars[r].quote_volume,
            6 => bars[r].trades,
            7 => bars[r].taker_buy_base,
            _ => bars[r].taker_buy_quote,
        })?;
    }
    for (_, v) in values {
        write_col(&mut w, &|r| v[r])?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn read_spool(
    path: &Path,
    rows: usize,
    global: &mut [Vec<f64>],
    index: &BTreeMap<String, usize>,
) -> Result<(), String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut r = std::io::BufReader::new(file);
    let mut u32b = [0u8; 4];
    r.read_exact(&mut u32b).map_err(|e| e.to_string())?;
    let n = u32::from_le_bytes(u32b) as usize;
    let mut names = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut u32b).map_err(|e| e.to_string())?;
        let len = u32::from_le_bytes(u32b) as usize;
        let mut b = vec![0u8; len];
        r.read_exact(&mut b).map_err(|e| e.to_string())?;
        names.push(String::from_utf8_lossy(&b).to_string());
    }
    let mut u64b = [0u8; 8];
    r.read_exact(&mut u64b).map_err(|e| e.to_string())?;
    let nrows = u64::from_le_bytes(u64b) as usize;
    if nrows != rows {
        return Err(format!("{}: spool has {} rows, expected {}", path.display(), nrows, rows));
    }
    let mut bytes = vec![0u8; rows * 8];
    for name in names {
        r.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        // `ts` is pre-filled by the assembler from the panel metadata.
        if name == "ts" {
            continue;
        }
        if let Some(&j) = index.get(&name) {
            global[j].reserve(rows);
            for chunk in bytes.chunks_exact(8) {
                global[j].push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
    }
    Ok(())
}

impl Audit {
    fn merge(&mut self, other: &Audit) {
        self.rows_in += other.rows_in;
        self.duplicates += other.duplicates;
        self.non_monotonic += other.non_monotonic;
        self.invalid += other.invalid;
        self.unit_converted += other.unit_converted;
        self.zero_volume += other.zero_volume;
        if rank(&other.unit) > rank(&self.unit) {
            self.unit = other.unit.clone();
        }
    }
}

/// Split `items` into at most `jobs` deterministic round-robin chunks.
///
/// Every parallel pass uses this helper, so `--jobs` bounds the number of
/// workers (and therefore the number of in-flight per-symbol matrices) rather
/// than only the parse pass. With more jobs than items, one worker per item is
/// enough.
fn bounded_chunks<T>(items: Vec<T>, jobs: usize) -> Vec<Vec<T>> {
    let workers = jobs.max(1).min(items.len().max(1));
    let mut chunks: Vec<Vec<T>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, item) in items.into_iter().enumerate() {
        chunks[i % workers].push(item);
    }
    chunks
}

#[cfg(test)]
static PASS4_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static PASS4_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
struct Pass4WorkerGuard;

#[cfg(test)]
impl Drop for Pass4WorkerGuard {
    fn drop(&mut self) {
        PASS4_ACTIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
fn enter_pass4_worker() -> Pass4WorkerGuard {
    let now = PASS4_ACTIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    PASS4_MAX.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
    Pass4WorkerGuard
}

#[cfg(not(test))]
fn enter_pass4_worker() {}

fn log_rss(tag: &str) {
    if std::env::var_os("RCSR_PREP_RSS_LOG").is_none() {
        return;
    }
    let kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        })
        .unwrap_or(0);
    eprintln!("prep-rss: {} rss_kib={}", tag, kib);
}

/// Build the table, audit and manifest without writing anything.
pub fn build(opts: &Options) -> Result<Build, String> {
    let sources = discover(&opts.inputs, opts.symbol.as_deref())?;
    let mut audit = Audit {
        layout: opts.format.name().to_string(),
        files: sources.len(),
        unit: "s".to_string(),
        resample: opts
            .resample
            .map(|t| t.name().to_string())
            .unwrap_or_else(|| "none".to_string()),
        label: opts
            .label
            .map(|l| format!("fwd:{} cost={}bp volnorm={} stride={}", l.horizon, l.cost_bps, l.vol_window, l.stride))
            .unwrap_or_else(|| "none".to_string()),
        ..Default::default()
    };

    // Group by symbol so a worker owns every month of a symbol.
    let mut by_symbol: BTreeMap<String, Vec<Source>> = BTreeMap::new();
    for s in sources.clone() {
        by_symbol.entry(s.symbol.clone()).or_default().push(s);
    }
    let groups: Vec<(String, Vec<Source>)> = by_symbol.into_iter().collect();
    let chunks = bounded_chunks(groups, opts.jobs);

    // Pass 1: parse + timestamp/duplicate normalisation, parallel by symbol.
    let mut parsed: Vec<(String, Vec<Bar>)> = Vec::new();
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::new();
        for chunk in chunks {
            let format = opts.format;
            let tz = opts.tz;
            handles.push(scope.spawn(move || -> Result<(Vec<(String, Vec<Bar>)>, Audit), String> {
                let mut local = Audit::default();
                local.unit = "s".to_string();
                let mut out = Vec::new();
                for (symbol, srcs) in chunk {
                    let mut raw = Vec::new();
                    for src in srcs {
                        raw.extend(parse_source(&src, format, tz, &mut local)?);
                    }
                    out.push((symbol, normalise_bars(raw, &mut local)));
                }
                Ok((out, local))
            }));
        }
        for h in handles {
            let (part, a) = h.join().map_err(|_| "prep worker panicked".to_string())??;
            audit.merge(&a);
            parsed.extend(part);
        }
        Ok(())
    })?;
    parsed.sort_by(|a, b| a.0.cmp(&b.0));
    audit.symbols = parsed.len();
    let mut series: BTreeMap<String, Vec<Bar>> = parsed.into_iter().collect();

    // Pass 2: verified splices, before resampling.
    if !opts.migrations.is_empty() {
        series = apply_migrations(&series, &opts.migrations)?;
        audit.migrations = opts.migrations.len();
    }

    // Pass 3: resample each symbol (calendar-aware), parallel.
    let symbols: Vec<(String, Vec<Bar>)> = series.into_iter().collect();
    let mut resampled: Vec<(String, Vec<Bar>)> = Vec::new();
    let resample_chunks = bounded_chunks(symbols, opts.jobs);
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::new();
        for chunk in resample_chunks {
            let resample = opts.resample;
            let calendar = opts.calendar;
            handles.push(scope.spawn(move || -> Result<Vec<(String, Vec<Bar>)>, String> {
                let mut out = Vec::with_capacity(chunk.len());
                for (symbol, bars) in chunk {
                    let bars = match resample {
                        Some(tf) => resample_calendar(&bars, tf, calendar)?,
                        None => bars,
                    };
                    out.push((symbol, bars));
                }
                Ok(out)
            }));
        }
        for h in handles {
            resampled.extend(h.join().map_err(|_| "prep worker panicked".to_string())??);
        }
        Ok(())
    })?;
    resampled.sort_by(|a, b| a.0.cmp(&b.0));
    let tf = opts.resample.unwrap_or(Timeframe::M1);

    // A cross-asset feature uses BTC when the panel contains it. The benchmark
    // is joined by timestamp, never by row index.
    // Share one immutable benchmark copy across all workers instead of cloning
    // the whole series per symbol.
    let cross: Option<Arc<[Bar]>> = resampled
        .iter()
        .find(|(s, _)| s == "BTCUSDT" || s == "BTCUSD" || s == "BTC")
        .map(|(_, b)| Arc::from(b.as_slice()));

    // Pass 4: features, MTF joins and the label, parallel by symbol.
    let mut panels: Vec<SymbolPanel> = Vec::new();
    let feature_chunks = bounded_chunks(resampled, opts.jobs);
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::new();
        for chunk in feature_chunks {
            let spec = opts.features.clone();
            let mtf = opts.mtf.clone();
            let label = opts.label;
            let calendar = opts.calendar;
            let cross = cross.clone();
            handles.push(scope.spawn(move || -> Result<Vec<SymbolPanel>, String> {
                let mut out = Vec::with_capacity(chunk.len());
                for (symbol, bars) in chunk {
                    let _guard = enter_pass4_worker();
                    let mut values: Vec<(String, Vec<f64>)> = Vec::new();
                    let mut warmup = 0usize;
                    let mut notes = Vec::new();
                    if !spec.is_empty() {
                        let cols = features::compute_with_cross(
                            &bars,
                            tf,
                            &spec,
                            cross.as_deref(),
                        )?;
                        for c in cols {
                            warmup = warmup.max(c.warmup);
                            values.push((c.name, c.values));
                        }
                    }
                    for &htf in &mtf {
                        let higher = resample_calendar(&bars, htf, calendar)?;
                        if spec.is_empty() {
                            let vals: Vec<f64> = higher.iter().map(|b| b.close).collect();
                            let joined = join_column_causal(&bars, &higher, htf, &vals)?;
                            values.push((
                                format!("mtf_{}_close", htf.name()),
                                joined.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                            ));
                        } else {
                            let cols = features::compute(&higher, htf, &spec)?;
                            for c in &cols {
                                let joined = join_column_causal(&bars, &higher, htf, &c.values)?;
                                values.push((
                                    format!("mtf_{}_{}", htf.name(), c.name),
                                    joined.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                                ));
                            }
                        }
                    }
                    let (label_values, gap_rows, stride_rows) = match label {
                        Some(spec) => {
                            let l = features::forward_label(&bars, tf, &spec);
                            (l.values, l.gap_rows, l.stride_rows)
                        }
                        None => (Vec::new(), 0, 0),
                    };
                    if bars.is_empty() {
                        notes.push(format!("symbol {} has no bars after resampling", symbol));
                    }
                    if !label_values.is_empty() {
                        values.push(("label".to_string(), label_values));
                    }
                    let names: Vec<String> = values.iter().map(|(n, _)| n.clone()).collect();
                    let seq = SPOOL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let spool = std::env::temp_dir().join(format!(
                        "rcsr-prep-{}-{}-{}.spool",
                        std::process::id(),
                        seq,
                        symbol.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
                    ));
                    write_spool(&spool, &symbol, &bars, &values)?;
                    let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
                    out.push(SymbolPanel {
                        symbol,
                        times,
                        rows: bars.len(),
                        names,
                        spool,
                        warmup,
                        gap_rows,
                        stride_rows,
                        notes,
                    });
                }
                Ok(out)
            }));
        }
        for h in handles {
            panels.extend(h.join().map_err(|_| "prep worker panicked".to_string())??);
        }
        Ok(())
    })?;
    panels.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    // Assemble the table with a fixed, deterministic column order. A panel
    // carries a `symbol` column; a single series does not (the canonical OHLCV
    // layout stays byte-identical to `rcsr import`).
    let with_symbol = panels.len() > 1;
    let mut names: Vec<String> = Vec::new();
    names.push("ts".to_string());
    if with_symbol {
        names.push("symbol".to_string());
    }
    for n in CANONICAL.iter().skip(1) {
        names.push(n.to_string());
    }
    for p in &panels {
        for n in &p.names {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
    }
    let total: usize = panels.iter().map(|p| p.rows).sum();
    // `vec![empty; n]` clones the *length* (zero), not the capacity, so the
    // per-column buffers would each grow from zero and leave reallocated
    // blocks behind. Reserve every column exactly once here.
    log_rss("assembly:start");
    let mut columns: Vec<Vec<f64>> = (0..names.len())
        .map(|_| Vec::with_capacity(total))
        .collect();
    log_rss("assembly:columns-reserved");
    let mut times: Vec<i64> = Vec::with_capacity(total);
    let canon_start = if with_symbol { 2 } else { 1 };
    let mut symbol_codes: Vec<u32> = Vec::new();
    let mut symbol_levels: Vec<String> = Vec::new();
    let index: BTreeMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let extra_start = canon_start + CANONICAL.len() - 1;
    let mut panel_no = 0usize;
    for p in panels {
        panel_no += 1;
        if panel_no % 10 == 0 || panel_no + 1 == audit.symbols {
            log_rss(&format!("assembly:panel-{panel_no}-of-{}", audit.symbols));
        }
        let code = if with_symbol {
            let c = symbol_levels
                .iter()
                .position(|s| s == &p.symbol)
                .unwrap_or_else(|| {
                    symbol_levels.push(p.symbol.clone());
                    symbol_levels.len() - 1
                }) as u32;
            symbol_codes.extend(std::iter::repeat(c).take(p.rows));
            c
        } else {
            0
        };
        let _ = code;
        times.extend_from_slice(&p.times);
        columns[0].extend(p.times.iter().map(|t| *t as f64));
        // A feature this symbol did not produce is missing, not zero.
        for (j, name) in names.iter().enumerate().skip(extra_start) {
            if !p.names.contains(name) {
                columns[j].extend(std::iter::repeat(f64::NAN).take(p.rows));
            }
        }
        read_spool(&p.spool, p.rows, &mut columns, &index)?;
        let _ = std::fs::remove_file(&p.spool);
        audit.rows_out += p.rows;
        audit.gaps += gap_count_times(&p.times, tf);
        audit.label_gap_rows += p.gap_rows;
        audit.label_stride_rows += p.stride_rows;
        if p.warmup > 0 {
            audit.notes.push(format!(
                "symbol {} warm-up {} rows; first usable {}",
                p.symbol,
                p.warmup,
                p.times.get(p.warmup.saturating_sub(1)).copied().unwrap_or(0)
            ));
        }
        audit.notes.extend(p.notes.clone());
    }
    if total == 0 {
        return Err("prep produced no rows".to_string());
    }
    audit.features = names
        .iter()
        .skip(canon_start + CANONICAL.len() - 1)
        .filter(|n| *n != "label")
        .cloned()
        .collect();
    audit.columns = names.clone();
    for (i, n) in names.iter().enumerate().skip(canon_start + CANONICAL.len() - 1) {
        let v = &columns[i];
        let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        audit.feature_nan_rows += v.len() - finite.len();
        if finite.len() > 1 {
            let first = finite[0];
            if finite.iter().all(|x| *x == first) {
                audit.feature_zero_variance.push(n.clone());
            }
        }
    }
    log_rss("assembly:panels-read");
    let mut out_columns: Vec<Column> = Vec::with_capacity(names.len());
    let mut numeric: Vec<bool> = Vec::with_capacity(names.len());
    for (j, _name) in names.iter().enumerate() {
        if with_symbol && j == 1 {
            numeric.push(false);
            out_columns.push(Column {
                values: Vec::new(),
                ints: None,
                dict: Some(crate::data::DictColumn {
                    codes: symbol_codes.clone(),
                    levels: symbol_levels.clone(),
                }),
            });
            continue;
        }
        let ints = if j == 0 { Some(times.clone()) } else { None };
        let values = std::mem::take(&mut columns[j]);
        let dict = importer_dict(&values, ints.as_deref());
        numeric.push(true);
        out_columns.push(Column { values, ints, dict });
    }
    log_rss("assembly:out-columns");
    let table = Table::from_parts(names.clone(), numeric, out_columns, total, 0);
    log_rss("assembly:table-built");

    let input_hashes: Vec<(String, String, u64)> = sources
        .iter()
        .map(|s| {
            let p = s.path.to_string_lossy().to_string();
            let size = std::fs::metadata(&s.path).map(|m| m.len()).unwrap_or(0);
            let hash = manifest::sha256_file(&s.path).unwrap_or_else(|_| "unknown".to_string());
            (p, hash, size)
        })
        .collect();
    let mut manifest_text = String::from("{\n");
    let _ = writeln!(manifest_text, "  \"tool\":\"rcsr prep\",");
    let _ = writeln!(manifest_text, "  \"format\":{},", jstr(opts.format.name()));
    let _ = writeln!(
        manifest_text,
        "  \"calendar\":{},\"timezone\":{},\"time_unit\":\"s\",",
        jstr(opts.calendar.name()),
        jstr(&opts.tz.name())
    );
    let _ = writeln!(
        manifest_text,
        "  \"resample\":{},\"mtf\":[{}],",
        jstr(&audit.resample),
        opts.mtf.iter().map(|t| jstr(t.name())).collect::<Vec<_>>().join(",")
    );
    let _ = writeln!(
        manifest_text,
        "  \"features\":[{}],\"label\":{},",
        opts.features.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","),
        opts.label
            .map(|l| format!(
                "{{\"horizon\":{},\"cost_bps\":{},\"vol_window\":{},\"stride\":{}}}",
                l.horizon, l.cost_bps, l.vol_window, l.stride
            ))
            .unwrap_or_else(|| "null".to_string())
    );
    let _ = writeln!(
        manifest_text,
        "  \"migrations\":[{}],",
        opts.migrations
            .iter()
            .map(|m| format!(
                "{{\"old\":{},\"new\":{},\"k\":{},\"tolerance\":{}}}",
                jstr(&m.old),
                jstr(&m.new),
                m.k,
                m.tolerance
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let spec_hash = opts
        .spec_path
        .as_deref()
        .and_then(|p| manifest::sha256_file(p).ok())
        .unwrap_or_else(|| {
            manifest::sha256_hex(
                format!(
                    "{}|{}|{}|{:?}|{:?}|{}",
                    opts.format.name(),
                    opts.calendar.name(),
                    audit.resample,
                    opts.mtf,
                    opts.features,
                    audit.label
                )
                .as_bytes(),
            )
        });
    let _ = writeln!(manifest_text, "  \"spec_hash\":{},", jstr(&spec_hash));
    let _ = writeln!(
        manifest_text,
        "  \"spec_path\":{},",
        opts.spec_path.as_deref().map(jstr).unwrap_or_else(|| "null".to_string())
    );
    let _ = writeln!(manifest_text, "  \"inputs\":[");
    for (i, (p, hash, size)) in input_hashes.iter().enumerate() {
        let comma = if i + 1 == input_hashes.len() { "" } else { "," };
        let _ = writeln!(
            manifest_text,
            "    {{\"path\":{},\"sha256\":{},\"bytes\":{}}}{}",
            jstr(p),
            jstr(hash),
            size,
            comma
        );
    }
    let _ = writeln!(manifest_text, "  ],");
    let _ = writeln!(manifest_text, "  \"audit\":{},", audit.json());
    let _ = writeln!(
        manifest_text,
        "  \"output\":{{\"rows\":{},\"columns\":{},\"symbols\":{},\"first_ts\":{},\"last_ts\":{}}}",
        total,
        names.len(),
        audit.symbols,
        times.first().copied().unwrap_or(0),
        times.last().copied().unwrap_or(0)
    );
    manifest_text.push_str("}\n");

    Ok(Build {
        table,
        audit,
        manifest: manifest_text,
    })
}

/// Reproduce the importer's low-cardinality numeric dictionary so a prep store
/// is byte-identical to `rcsr import` on the canonical CSV. The reader keeps a
/// dictionary while a column has at most [`crate::data::NUMERIC_DICT_MAX`]
/// distinct cells; above that it drops it, and so do we.
fn importer_dict(values: &[f64], ints: Option<&[i64]>) -> Option<crate::data::DictColumn> {
    use crate::data::{DictColumn, NUMERIC_DICT_MAX};
    if values.len() > NUMERIC_DICT_MAX {
        return None;
    }
    let mut levels: Vec<String> = Vec::new();
    let mut map: BTreeMap<String, u32> = BTreeMap::new();
    let mut codes = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
        let text = if let Some(ints) = ints {
            ints[i].to_string()
        } else if v.is_nan() {
            "NaN".to_string()
        } else {
            format!("{}", v)
        };
        let code = match map.get(&text) {
            Some(&c) => c,
            None => {
                let c = levels.len() as u32;
                levels.push(text.clone());
                map.insert(text, c);
                c
            }
        };
        codes.push(code);
    }
    if levels.len() > NUMERIC_DICT_MAX {
        return None;
    }
    Some(DictColumn { codes, levels })
}

/// Missing buckets between the first and last bar of a time-only series.
fn gap_count_times(times: &[i64], tf: Timeframe) -> usize {
    if times.len() < 2 {
        return 0;
    }
    let first = tf.bucket_start(times[0]);
    let last = tf.bucket_start(times[times.len() - 1]);
    let expected = (last - first) / tf.seconds() + 1;
    let mut present = 1i64;
    for w in times.windows(2) {
        if tf.bucket_start(w[1]) != tf.bucket_start(w[0]) {
            present += 1;
        }
    }
    (expected - present).max(0) as usize
}

/// Render a table as canonical CSV. This is the reference path for the
/// byte-identity test (`prep` output vs the existing importer); `prep` itself
/// never writes this file.
pub fn table_csv(table: &Table) -> String {
    let mut out = String::new();
    out.push_str(&table.names.join(","));
    out.push('\n');
    for r in 0..table.len() {
        for c in 0..table.n_cols() {
            if c > 0 {
                out.push(',');
            }
            if let Some(text) = table.text(r, c) {
                out.push_str(text);
                continue;
            }
            let v = table.value(r, c);
            if v.is_nan() {
                out.push_str("NaN");
            } else if let Some(ints) = table.ints(c) {
                out.push_str(&ints[r].to_string());
            } else {
                let _ = write!(out, "{}", v);
            }
        }
        out.push('\n');
    }
    out
}

fn jstr(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The `rcsr prep` entry point (called from `main`).
pub fn run(args: &[String]) -> ! {
    let mut opts = Options::default();
    let mut spec_paths: Vec<String> = Vec::new();
    let mut i = 0usize;
    macro_rules! val {
        () => {{
            let v = args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: {} requires a value", args[i]);
                std::process::exit(EXIT_USAGE);
            });
            i += 1;
            v
        }};
    }
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--input" | "-i" => opts.inputs.push(val!()),
            "--format" => match Layout::parse(&val!()) {
                Ok(f) => opts.format = f,
                Err(e) => fail(&e),
            },
            "--symbol" => opts.symbol = Some(val!()),
            "--tz" => match Tz::parse(&val!()) {
                Ok(t) => opts.tz = t,
                Err(e) => fail(&e),
            },
            "--calendar" => match Calendar::parse(&val!()) {
                Some(c) => opts.calendar = c,
                None => fail("unknown calendar (24x7, nyse, lse)"),
            },
            "--resample" => match Timeframe::parse(&val!()) {
                Ok(t) => opts.resample = Some(t),
                Err(e) => fail(&e),
            },
            "--mtf" => match parse_mtf(&val!()) {
                Ok(t) => opts.mtf = t,
                Err(e) => fail(&e),
            },
            "--features" => match features::expand(&val!()) {
                Ok(f) => opts.features = f,
                Err(e) => fail(&e),
            },
            "--label" => {
                let v = val!();
                let tf = opts.resample.unwrap_or(Timeframe::M1);
                match LabelSpec::parse(&v, tf) {
                    Ok(l) => opts.label = Some(l),
                    Err(e) => fail(&e),
                }
            }
            "--spec" => spec_paths.push(val!()),
            "--migrations" | "--splice" => match parse_migrations(&val!()) {
                Ok(m) => opts.migrations = m,
                Err(e) => fail(&e),
            },
            "--out" | "-o" => opts.out = Some(val!()),
            "--dry-run" => opts.dry_run = true,
            "--jobs" | "-j" => {
                let v = val!();
                opts.jobs = v.parse().unwrap_or(0);
                if opts.jobs == 0 {
                    fail("--jobs needs a positive integer");
                }
            }
            "--group-rows" => {
                let v = val!();
                opts.group_rows = v.parse().unwrap_or(0);
                if opts.group_rows == 0 {
                    fail("--group-rows needs a positive integer");
                }
            }
            other => fail(&format!("unknown prep option '{}'", other)),
        }
        i += 1;
    }
    for p in spec_paths {
        if let Err(e) = load_spec(&mut opts, &p) {
            fail(&e);
        }
    }
    if opts.inputs.is_empty() {
        fail("prep needs --input <file|dir>");
    }
    if opts.out.is_none() && !opts.dry_run {
        fail("prep needs --out <store.rcsb> (or --dry-run)");
    }
    let build = match build(&opts) {
        Ok(b) => b,
        Err(e) => fail(&e),
    };
    for line in build.audit.text_lines() {
        println!("{}", line);
    }
    if opts.dry_run {
        println!("{}", build.manifest);
        std::process::exit(0);
    }
    let out = opts.out.clone().unwrap();
    if let Err(e) = store::write_store_with_time(&out, &build.table, opts.group_rows, Some(0)) {
        fail(&e);
    }
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let hash = manifest::sha256_file(&out).unwrap_or_else(|_| "unknown".to_string());
    // Append the output hash to the manifest atomically next to the store.
    let manifest_path = format!("{}.manifest.json", out);
    let mut text = build.manifest;
    if let Some(pos) = text.rfind("\"output\":") {
        let insert = format!(
            "\"output_sha256\":{},\"output_bytes\":{},\n  ",
            jstr(&hash),
            bytes
        );
        text.insert_str(pos, &insert);
    }
    if let Err(e) = std::fs::write(&manifest_path, text) {
        fail(&format!("{}: {}", manifest_path, e));
    }
    eprintln!(
        "prep: {} rows x {} cols -> {} ({} bytes, {:.1} MiB), manifest {}",
        build.table.len(),
        build.table.n_cols(),
        out,
        bytes,
        bytes as f64 / (1024.0 * 1024.0),
        manifest_path
    );
    std::process::exit(0);
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(EXIT_INPUT);
}

pub fn usage() -> String {
    "rcsr prep — raw OHLCV CSV/ZIP archives to a .rcsb store, no intermediate CSV

USAGE:
  rcsr prep --input <file|dir> [--out store.rcsb] [options]

INPUT:
  --input, -i <path>      CSV/TSV/ZIP file or a directory searched recursively
                          (Binance Vision monthly ZIPs and Bitstamp CSVs)
  --format <kind>         auto (default), binance, bitstamp or ohlcv
  --symbol <name>         override the symbol for a single-file input
  --tz <zone>             UTC (default), +HH:MM, America/New_York, Europe/London
  --calendar <name>       24x7 (default), nyse or lse
  --migrations <table>    OLD->NEW:k[:tol],... with a measured splice ratio

TRANSFORMS:
  --resample <tf>         1m,3m,5m,15m,30m,1h,2h,4h,1d,1w (calendar-aware)
  --mtf <tf[,tf...]>      causal higher-timeframe features (last closed bar)
  --features <list>       ta:core, ta:full, exotic:cheap, candle:geometry,
                          cross:btc, calendar, or explicit feature names
  --label <spec>          fwd:<tf|bars>,cost=<bp>,volnorm=<bars>,stride=<n>
  --spec <file>           key=value spec (features/label/resample/mtf/...)
  --jobs, -j <n>          parallel symbol workers (default 1)
  --group-rows <n>        column-store group size (default 65536)
  --out, -o <store.rcsb>  output store (required unless --dry-run)
  --dry-run               print the audit and manifest, write nothing

FEATURES (explicit names):
  ret1 ret5 ma20_ratio ma60_ratio bb20_z atr14 natr14 rsi14 adx14 di_plus
  di_minus cmf20 mfi14 vol20_ratio vwap20_dev rmean5 rstd5 skew20 kurt20
  range1 close_pos20 body1 upper_wick1 lower_wick1 open_gap1
  cross_ret1 cross_beta20 dow_sin dow_cos hour_sin hour_cos

The store is written directly; no CSV intermediate is created. The manifest
(<out>.manifest.json) records the spec hash, input checksums, audit counters
and output checksum. The same inputs + spec reproduce the same bytes.
"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_units_are_detected_and_converted_to_seconds() {
        assert_eq!(normalise_time(1_700_000_000.0), (1_700_000_000.0, 1, "s"));
        assert_eq!(normalise_time(1_700_000_000_000.0), (1_700_000_000.0, 1_000, "ms"));
        assert_eq!(
            normalise_time(1_700_000_000_000_000.0),
            (1_700_000_000.0, 1_000_000, "us")
        );
        assert_eq!(
            normalise_time(1_700_000_000_000_000_000.0),
            (1_700_000_000.0, 1_000_000_000, "ns")
        );
    }

    #[test]
    fn duplicate_instants_aggregate_instead_of_dropping_a_bar() {
        let mut a = Audit::default();
        let bars = vec![
            Bar::new(60, 1.0, 2.0, 0.5, 1.5, 1.0),
            Bar::new(60, 1.5, 3.0, 1.0, 2.5, 2.0),
        ];
        let out = normalise_bars(bars, &mut a);
        assert_eq!(out.len(), 1);
        assert_eq!(a.duplicates, 1);
        assert_eq!(out[0].open, 1.0);
        assert_eq!(out[0].high, 3.0);
        assert_eq!(out[0].low, 0.5);
        assert_eq!(out[0].close, 2.5);
        assert_eq!(out[0].volume, 3.0);
    }

    #[test]
    fn migration_parser_accepts_a_tolerance() {
        let m = parse_migrations("TOMO->VIC:1.0169:0.005,GAL->G:60").unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].k, 1.0169);
        assert_eq!(m[1].k, 60.0);
        assert_eq!(m[1].tolerance, 0.005);
    }

    #[test]
    fn bounded_chunks_never_exceed_jobs_and_lose_no_item() {
        for jobs in [1usize, 2, 3, 8] {
            let chunks = bounded_chunks((0..109).collect::<Vec<_>>(), jobs);
            assert!(chunks.len() <= jobs, "jobs={jobs} spawned {} chunks", chunks.len());
            let mut seen: Vec<usize> = chunks.into_iter().flatten().collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..109).collect::<Vec<_>>());
        }
        // Fewer items than jobs: one worker per item, never idle empties.
        assert_eq!(bounded_chunks(vec![1, 2], 8).len(), 2);
        assert_eq!(bounded_chunks(Vec::<i32>::new(), 8).len(), 1);
    }

    #[test]
    fn pass_four_worker_count_is_bounded_by_jobs() {
        let dir = std::env::temp_dir().join(format!(
            "rcsr-prep-workers-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for s in 0..9 {
            let mut text = String::new();
            for i in 0..8 {
                let t = 1_704_067_200_000i64 + (i as i64) * 60_000;
                text.push_str(&format!(
                    "{t},100,101,99,100,10,1000,1,6,600,0\n"
                ));
            }
            std::fs::write(dir.join(format!("sym{s}.csv")), text).unwrap();
        }
        PASS4_ACTIVE.store(0, std::sync::atomic::Ordering::SeqCst);
        PASS4_MAX.store(0, std::sync::atomic::Ordering::SeqCst);
        let opts = Options {
            inputs: vec![dir.to_string_lossy().to_string()],
            format: Layout::Binance,
            jobs: 3,
            ..Default::default()
        };
        let build = build(&opts).unwrap();
        assert_eq!(build.table.len(), 72);
        let peak = PASS4_MAX.load(std::sync::atomic::Ordering::SeqCst);
        assert!(peak >= 1);
        assert!(
            peak <= 3,
            "pass 4 used {peak} concurrent workers with --jobs 3"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
