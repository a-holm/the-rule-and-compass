//! `rcsr prep` — raw OHLCV CSV to a typed column store, without an
//! intermediate CSV (#123).
//!
//! First increment of the pipeline in issue #123: layouts (Binance/Bitstamp
//! headerless and aliased headers), timestamp normalisation, validation,
//! numeric de-duplication, deterministic ordering, 1m -> 15m/1h/1d OHLCV
//! resampling, an audit and a manifest, writing `.rcsb` directly through
//! [`crate::store::write_store`]. Feature/label specs, ZIP archives, splice and
//! per-symbol parallelism are follow-up work; nothing here pretends otherwise.

use crate::data::{self, Column, Table};
use std::collections::HashSet;
use std::fmt::Write as _;

/// How the raw file is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Auto,
    Binance,
    Bitstamp,
    Ohlcv,
}

impl Layout {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
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

/// The canonical OHLCV columns the pipeline understands.
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
    let n = name.trim().to_ascii_lowercase();
    Some(match n.as_str() {
        "time" | "timestamp" | "date" | "datetime" | "open_time" | "opentime" | "unix" => "time",
        "open" | "o" => "open",
        "high" | "h" => "high",
        "low" | "l" => "low",
        "close" | "c" => "close",
        "volume" | "vol" | "base_volume" | "basevolume" => "volume",
        "quote_volume" | "quotevolume" | "volume_quote" => "quote_volume",
        "trades" | "count" | "number_of_trades" => "trades",
        "taker_buy_base" | "taker_buy_volume" | "takerbuybase" => "taker_buy_base",
        "taker_buy_quote" | "taker_buy_quote_volume" | "takerbuyquote" => "taker_buy_quote",
        "ignore" | "close_time" | "closetime" => return None,
        _ => return None,
    })
}

/// A raw bar before resampling (time in seconds).
struct Bar {
    time: f64,
    values: Vec<f64>,
}

/// What the pipeline read and did, for `--dry-run` and the manifest.
#[derive(Clone, Debug, Default)]
pub struct Audit {
    pub layout: String,
    pub rows_in: usize,
    pub rows_out: usize,
    pub rows_dropped_duplicate: usize,
    pub non_monotonic: usize,
    pub unit: String,
    pub unit_converted: usize,
    pub resample: String,
    pub gaps: usize,
    pub zero_volume: usize,
    pub columns: Vec<String>,
    pub notes: Vec<String>,
}

impl Audit {
    pub fn text_lines(&self) -> Vec<String> {
        let mut out = vec![
            format!(
                "# prep: layout={} rows_in={} rows_out={} unit={} resample={}",
                self.layout, self.rows_in, self.rows_out, self.unit, self.resample
            ),
            format!(
                "# prep: duplicates_dropped={} non_monotonic={} gaps={} zero_volume={}",
                self.rows_dropped_duplicate, self.non_monotonic, self.gaps, self.zero_volume
            ),
            format!("# prep: columns={}", self.columns.join(",")),
        ];
        out.extend(self.notes.iter().map(|n| format!("# prep: {}", n)));
        out
    }

    pub fn json(&self, spec_hash: u64) -> String {
        let notes: Vec<String> = self
            .notes
            .iter()
            .map(|n| format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect();
        format!(
            "{{\"layout\":\"{}\",\"rows_in\":{},\"rows_out\":{},\"rows_dropped_duplicate\":{},\"non_monotonic\":{},\"unit\":\"{}\",\"unit_converted\":{},\"resample\":\"{}\",\"gaps\":{},\"zero_volume\":{},\"columns\":[{}],\"spec_hash\":\"{:016x}\",\"notes\":[{}]}}",
            self.layout,
            self.rows_in,
            self.rows_out,
            self.rows_dropped_duplicate,
            self.non_monotonic,
            self.unit,
            self.unit_converted,
            self.resample,
            self.gaps,
            self.zero_volume,
            self.columns
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(","),
            spec_hash,
            notes.join(",")
        )
    }
}

/// Parse `1m`, `15m`, `1h`, `1d`, `1w` (also `s`) to seconds.
pub fn interval_seconds(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if t.is_empty() || t == "none" || t == "raw" {
        return Ok(0);
    }
    let (num, unit) = t.split_at(t.len() - 1);
    let n: u64 = num
        .parse()
        .map_err(|_| format!("bad --resample '{}' (expected e.g. 15m, 1h, 1d)", s))?;
    let mult = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        "d" => 86400,
        "w" => 604800,
        _ => return Err(format!("bad --resample '{}' (unit must be s/m/h/d/w)", s)),
    };
    if n == 0 {
        return Err("--resample interval must be positive".to_string());
    }
    Ok(n * mult)
}

/// Read the raw file and rewrite a headerless layout with canonical names.
fn canonicalise_header(path: &str, layout: Layout) -> Result<(Table, Layout), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let first = text
        .lines()
        .find(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .ok_or_else(|| format!("{}: empty input", path))?;
    let first_fields: Vec<&str> = first.split(',').map(|f| f.trim()).collect();
    let headerless = !first_fields.is_empty()
        && first_fields
            .iter()
            .all(|f| f.parse::<f64>().is_ok() && !f.is_empty());

    if headerless {
        let detected = match (layout, first_fields.len()) {
            (Layout::Binance, _) | (Layout::Auto, 12) => Layout::Binance,
            (Layout::Bitstamp, _) | (Layout::Auto, 6) => Layout::Bitstamp,
            (Layout::Auto, n) if n >= 6 => {
                return Err(format!(
                    "{}: headerless file has {} columns; expected 6 (Bitstamp) or 12 \
                     (Binance); pass --format",
                    path, n
                ))
            }
            (l, _) => l,
        };
        let names: &[&str] = match detected {
            Layout::Binance => &[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
            Layout::Bitstamp => &["time", "open", "high", "low", "close", "volume"],
            _ => CANONICAL,
        };
        let mut annotated = String::with_capacity(text.len() + 64);
        annotated.push_str(&names[..first_fields.len().min(names.len())].join(","));
        annotated.push('\n');
        annotated.push_str(&text);
        let t = data::parse_table(&annotated).map_err(|e| format!("{}: {}", path, e))?;
        return Ok((t, detected));
    }

    let t = data::read_table(path)?;
    let detected = match layout {
        Layout::Auto => {
            if t.names.iter().any(|n| n.eq_ignore_ascii_case("open_time")) {
                Layout::Binance
            } else {
                Layout::Ohlcv
            }
        }
        l => l,
    };
    Ok((t, detected))
}

/// Read raw klines/CSV and build the canonical table plus an audit.
///
/// `resample` is `0`/`none` for raw rows. The output is sorted by time and
/// numeric duplicates are dropped (first occurrence wins), so the same input
/// and options produce the same table.
pub fn build(path: &str, layout: Layout, resample: &str, group_rows: usize) -> Result<(Table, Audit), String> {
    let interval = interval_seconds(resample)?;
    let (raw, detected) = canonicalise_header(path, layout)?;
    let mut audit = Audit {
        layout: detected.name().to_string(),
        resample: if interval == 0 { "none".to_string() } else { resample.to_string() },
        rows_in: raw.len(),
        ..Default::default()
    };
    let _ = group_rows;

    // Map aliases to canonical column indices. Unknown columns are ignored.
    let mut map: Vec<(usize, &'static str)> = Vec::new();
    for (j, name) in raw.names.iter().enumerate() {
        if let Some(c) = alias(name) {
            if !map.iter().any(|(_, n)| *n == c) {
                map.push((j, c));
            }
        }
    }
    for required in ["time", "open", "high", "low", "close", "volume"] {
        if !map.iter().any(|(_, n)| *n == required) {
            return Err(format!(
                "{}: missing required column '{}' (found {:?}); expected time/open/high/\
                 low/close/volume or a Binance/Bitstamp layout",
                path,
                required,
                raw.names
            ));
        }
    }
    // Output order: time then every canonical column present, in CANONICAL
    // order, so the store schema is stable.
    let out_cols: Vec<(&'static str, usize)> = CANONICAL
        .iter()
        .filter_map(|c| map.iter().find(|(_, n)| n == c).map(|(j, _)| (*c, *j)))
        .collect();
    audit.columns = out_cols.iter().map(|(n, _)| n.to_string()).collect();

    let time_j = map.iter().find(|(_, n)| *n == "time").unwrap().0;
    for (j, name) in map.iter() {
        if !raw.numeric[*j] {
            return Err(format!(
                "{}: column '{}' is not numeric; OHLCV columns must be numbers",
                path, name
            ));
        }
    }
    let _ = time_j;

    // Timestamp unit detection (disjoint epoch windows, as in #130/#110).
    let times: Vec<f64> = raw.col(time_j).to_vec();
    let (lo, hi) = times
        .iter()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let (unit, factor) = if lo >= 0.0 && hi < 1.0e8 {
        ("s", 1.0)
    } else if lo >= 1.0e8 && hi <= 4.2e9 {
        ("s", 1.0)
    } else if lo >= 1.0e11 && hi <= 4.2e12 {
        ("ms", 1e-3)
    } else if lo >= 1.0e14 && hi <= 4.2e15 {
        ("us", 1e-6)
    } else if lo >= 1.0e17 && hi <= 4.2e18 {
        ("ns", 1e-9)
    } else {
        return Err(format!(
            "{}: time column spans [{} ..= {}] and fits no single epoch unit (s/ms/us/ns); \
             normalise it first",
            path, lo, hi
        ));
    };
    audit.unit = unit.to_string();
    if factor != 1.0 {
        audit.unit_converted = times.iter().filter(|v| v.is_finite()).count();
    }

    let mut bars: Vec<Bar> = Vec::with_capacity(raw.len());
    let mut seen: HashSet<u64> = HashSet::new();
    let mut prev: Option<f64> = None;
    for r in 0..raw.len() {
        let t = raw.value(r, time_j) * factor;
        if !t.is_finite() {
            return Err(format!("{}: row {} has a non-numeric time", path, r + 1));
        }
        if let Some(p) = prev {
            if t < p {
                audit.non_monotonic += 1;
            }
        }
        prev = Some(t);
        if !seen.insert(t.to_bits()) {
            audit.rows_dropped_duplicate += 1;
            continue;
        }
        let mut values = Vec::with_capacity(out_cols.len());
        for &(name, j) in &out_cols {
            values.push(if name == "time" {
                t
            } else {
                raw.value(r, j)
            });
        }
        let high = out_cols.iter().position(|(n, _)| *n == "high").unwrap();
        let low = out_cols.iter().position(|(n, _)| *n == "low").unwrap();
        if values[high] < values[low] {
            return Err(format!(
                "{}: row {} has high {} below low {}; not valid OHLCV",
                path,
                r + 1,
                values[high],
                values[low]
            ));
        }
        bars.push(Bar { time: t, values });
    }
    if bars.is_empty() {
        return Err(format!("{}: no usable bars", path));
    }
    if audit.non_monotonic > 0 {
        audit
            .notes
            .push(format!("{} non-monotonic step(s) in input row order", audit.non_monotonic));
    }
    bars.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));

    // Resample by aligned bucket: open=first, high=max, low=min, close=last,
    // every other column sums. Empty buckets are not invented.
    let rows: Vec<Bar> = if interval == 0 {
        bars
    } else {
        resample_bars(&bars, interval, &out_cols, &mut audit)
    };
    audit.rows_out = rows.len();
    let volume_pos = out_cols.iter().position(|(n, _)| *n == "volume");
    if let Some(v) = volume_pos {
        audit.zero_volume = rows.iter().filter(|b| b.values[v] == 0.0).count();
    }

    // Materialise the canonical table. This is the column store's input; no
    // CSV is written or re-read.
    let names: Vec<String> = out_cols.iter().map(|(n, _)| n.to_string()).collect();
    let columns: Vec<Column> = (0..out_cols.len())
        .map(|c| Column {
            values: rows.iter().map(|b| b.values[c]).collect(),
            dict: None,
        })
        .collect();
    let numeric = vec![true; names.len()];
    let table = Table::from_parts(names, numeric, columns, rows.len(), 0);
    Ok((table, audit))
}

fn resample_bars(bars: &[Bar], interval: u64, cols: &[(&str, usize)], audit: &mut Audit) -> Vec<Bar> {
    let step = interval as f64;
    let mut out: Vec<Bar> = Vec::new();
    let mut i = 0usize;
    while i < bars.len() {
        let bucket = (bars[i].time / step).floor();
        let mut j = i + 1;
        while j < bars.len() && (bars[j].time / step).floor() == bucket {
            j += 1;
        }
        let group = &bars[i..j];
        let mut values = vec![0.0; cols.len()];
        for (c, (name, _)) in cols.iter().enumerate() {
            values[c] = match *name {
                "time" => bucket * step,
                "open" => group[0].values[c],
                "high" => group.iter().map(|b| b.values[c]).fold(f64::NEG_INFINITY, f64::max),
                "low" => group.iter().map(|b| b.values[c]).fold(f64::INFINITY, f64::min),
                "close" => group[group.len() - 1].values[c],
                _ => group.iter().map(|b| b.values[c]).sum(),
            };
        }
        out.push(Bar {
            time: bucket * step,
            values,
        });
        i = j;
    }
    // Gaps: expected buckets between the first and last output bar.
    if out.len() > 1 {
        let first = (out[0].time / step).round() as i64;
        let last = (out[out.len() - 1].time / step).round() as i64;
        let present = out.len() as i64;
        audit.gaps = (last - first + 1 - present).max(0) as usize;
    }
    out
}

/// FNV-1a over the spec string. Recorded in the manifest so a reader can tell
/// whether the same options produced a store.
pub fn spec_hash(spec: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in spec.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The full pipeline: build, write `.rcsb`, write the manifest.
pub fn run(
    path: &str,
    layout: Layout,
    resample: &str,
    group_rows: usize,
    out: Option<&str>,
    manifest: Option<&str>,
    dry_run: bool,
) -> Result<Audit, String> {
    let (table, audit) = build(path, layout, resample, group_rows)?;
    let spec = format!(
        "input={}\nlayout={}\nresample={}\ngroup_rows={}",
        path,
        audit.layout,
        audit.resample,
        group_rows
    );
    let hash = spec_hash(&spec);
    if !dry_run {
        let out = out.ok_or_else(|| "--out is required unless --dry-run".to_string())?;
        crate::store::write_store(out, &table, group_rows.max(1))?;
        if let Some(m) = manifest {
            let mut s = String::new();
            let _ = write!(
                s,
                "{{\n  \"spec\":\"{}\",\n  \"spec_hash\":\"{:016x}\",\n  \"audit\":{},\n  \"group_rows\":{},\n  \"out\":\"{}\"\n}}\n",
                spec.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"),
                hash,
                audit.json(hash),
                group_rows,
                out.replace('\\', "\\\\").replace('"', "\\\"")
            );
            std::fs::write(m, s).map_err(|e| format!("{}: {}", m, e))?;
        }
    }
    Ok(audit)
}
