//! #259: the default hourly point-in-time panel (v2).
//!
//! Stages (`rcsr panel snapshot | base | membership | build | validate`):
//!
//! * P0 [`snapshot`]: SHA-256, size, mtime, first/last timestamp, timestamp
//!   unit histogram, duplicate scan and invalid-OHLC scan of every archive.
//!   No network.
//! * P1 [`build_base`]: decode 15m archives, apply the duplicate and full-bucket
//!   policy of [`crate::panel_prep`], write the symbol-major hourly base store.
//! * P2 is `rcsr panel membership` on the base store (unchanged).
//! * P4/P5 [`build_panel`]: select point-in-time member rows, number listings,
//!   refuse undeclared level jumps and the lockbox, write the time-major store
//!   and its manifest.
//! * P6 [`validate`]: V1-V12 as a machine-readable report.
//!
//! Enriched feature parts (P3) belong to the registry and projection work and
//! are not part of this store.

use crate::data::{Column, DictColumn, Table};
pub use crate::enrich::GpuMode;
use crate::lockbox::{Guard, Lockbox};
use crate::ohlcv::{self, Bar};
use crate::panel::{self, Membership, Panel, PanelRow};
use crate::panel_prep::{self, PrepBar, PrepSegment};
use crate::{manifest, prep, store, zip};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const FLAG_PARTIAL_BUCKET: u32 = 1;
pub const FLAG_DUPLICATE_CONFLICT: u32 = 2;

pub const UNIT_S: u8 = 0;
pub const UNIT_MS: u8 = 1;
pub const UNIT_US: u8 = 2;

pub const IN_STEP: i64 = 900;
pub const OUT_STEP: i64 = 3600;

/// Columns of the symbol-major base store.
pub const BASE_COLUMNS: [&str; 17] = [
    "ts",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "cov15m",
    "gap",
    "flags",
    "traded",
    "dup_rows",
    "archive_unit",
];

/// Columns of the time-major panel store.
pub const PANEL_COLUMNS: [&str; 20] = [
    "ts",
    "symbol",
    "listing_id",
    "member_asof",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "vwap",
    "gap",
    "cov15m",
    "flags",
    "traded",
    "dup_rows",
    "archive_unit",
];

fn date(ts: i64) -> String {
    panel::date(ts)
}

fn jstr(s: &str) -> String {
    panel::jstr(s)
}

// ---------------------------------------------------------------------------
// Archive decoding
// ---------------------------------------------------------------------------

/// One decoded 15m row.
#[derive(Clone, Copy, Debug)]
struct Raw {
    time: i64,
    unit: u8,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    quote_volume: f64,
    trades: f64,
    taker_buy_base: f64,
    taker_buy_quote: f64,
}

impl Raw {
    fn payload_bits(&self) -> [u64; 9] {
        [
            self.open.to_bits(),
            self.high.to_bits(),
            self.low.to_bits(),
            self.close.to_bits(),
            self.volume.to_bits(),
            self.quote_volume.to_bits(),
            self.trades.to_bits(),
            self.taker_buy_base.to_bits(),
            self.taker_buy_quote.to_bits(),
        ]
    }

    fn prep_bar(&self) -> PrepBar {
        let mut b = PrepBar::new(self.time, self.open, self.high, self.low, self.close)
            .with_flow(self.volume, self.quote_volume, self.trades);
        b.taker_buy_base = self.taker_buy_base;
        b.taker_buy_quote = self.taker_buy_quote;
        b
    }
}

/// An OHLC row refused by the integrity rule.
#[derive(Clone, Debug, PartialEq)]
pub struct InvalidBar {
    pub symbol: String,
    pub path: String,
    pub time: i64,
    pub reason: &'static str,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// The decoded content of one archive.
#[derive(Default)]
struct Decoded {
    rows: Vec<Raw>,
    invalid: Vec<InvalidBar>,
    units: [usize; 3],
}

fn unit_of(raw: i64) -> (i64, u8) {
    let a = raw.unsigned_abs();
    if a < 100_000_000_000 {
        (raw, UNIT_S)
    } else if a < 100_000_000_000_000 {
        (raw.div_euclid(1_000), UNIT_MS)
    } else {
        (raw.div_euclid(1_000_000), UNIT_US)
    }
}

fn ohlc_defect(o: f64, h: f64, l: f64, c: f64) -> Option<&'static str> {
    if !(o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite()) {
        Some("non_finite")
    } else if o <= 0.0 || h <= 0.0 || l <= 0.0 || c <= 0.0 {
        Some("non_positive_price")
    } else if h < l {
        Some("high_below_low")
    } else if h < o.max(c) {
        Some("high_below_body")
    } else if l > o.min(c) {
        Some("low_above_body")
    } else {
        None
    }
}

fn decode_text(name: &str, symbol: &str, text: &str, out: &mut Decoded) -> Result<(), String> {
    let mut first = true;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|x| x.trim().trim_matches('"')).collect();
        if first {
            first = false;
            if f[0].parse::<f64>().is_err() {
                continue;
            }
        }
        if f.len() < 11 {
            return Err(format!("{}:{}: {} columns, need 11", name, i + 1, f.len()));
        }
        let num = |k: usize| -> Result<f64, String> {
            f[k].parse::<f64>()
                .map_err(|_| format!("{}:{}: '{}' is not numeric", name, i + 1, f[k]))
        };
        let raw_t: i64 = f[0]
            .parse::<i64>()
            .or_else(|_| f[0].parse::<f64>().map(|v| v as i64))
            .map_err(|_| format!("{}:{}: bad timestamp '{}'", name, i + 1, f[0]))?;
        let (time, unit) = unit_of(raw_t);
        out.units[unit as usize] += 1;
        let (o, h, l, c) = (num(1)?, num(2)?, num(3)?, num(4)?);
        if let Some(reason) = ohlc_defect(o, h, l, c) {
            out.invalid.push(InvalidBar {
                symbol: symbol.to_string(),
                path: name.to_string(),
                time,
                reason,
                open: o,
                high: h,
                low: l,
                close: c,
            });
            continue;
        }
        out.rows.push(Raw {
            time,
            unit,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: num(5)?,
            quote_volume: num(7)?,
            trades: num(8)?,
            taker_buy_base: num(9)?,
            taker_buy_quote: num(10)?,
        });
    }
    Ok(())
}

fn decode_bytes(path: &Path, symbol: &str, bytes: Vec<u8>) -> Result<Decoded, String> {
    let name = path.display().to_string();
    let mut out = Decoded::default();
    let is_zip = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("zip"))
        .unwrap_or(false);
    if is_zip {
        let archive = zip::ZipArchive::from_bytes(bytes).map_err(|e| format!("{}: {}", name, e))?;
        let entries = archive.read_extension(".csv")?;
        if entries.is_empty() {
            return Err(format!("{}: ZIP has no .csv entry", name));
        }
        for (entry, data) in entries {
            let text = String::from_utf8_lossy(&data);
            decode_text(&format!("{}:{}", name, entry), symbol, &text, &mut out)?;
        }
    } else {
        let text = String::from_utf8_lossy(&bytes);
        decode_text(&name, symbol, &text, &mut out)?;
    }
    Ok(out)
}

fn is_archive(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("zip") | Some("csv")
    )
}

/// Archives below `input`, sorted by symbol then path.
fn discover(input: &str) -> Result<Vec<prep::Source>, String> {
    let mut sources = prep::discover(&[input.to_string()], None)?;
    sources.retain(|s| is_archive(&s.path));
    if sources.is_empty() {
        return Err(format!("{}: no .zip or .csv archives found", input));
    }
    Ok(sources)
}

fn relative(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

/// First and last second of the month named by `SYMBOL-<interval>-YYYY-MM.zip`.
fn month_of_name(path: &Path) -> Option<(i64, i64)> {
    let stem = path.file_stem()?.to_str()?;
    let mut it = stem.rsplit('-');
    let m: u32 = it.next()?.parse().ok()?;
    let y: i32 = it.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1990..=2200).contains(&y) {
        return None;
    }
    let start = ohlcv::days_from_civil(y, m, 1) * 86_400;
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    Some((start, ohlcv::days_from_civil(ny, nm, 1) * 86_400))
}

/// Refuse an archive whose month intersects the lockbox, before decoding it.
fn guard_archive_names(sources: &[prep::Source], guard: &Guard, what: &str) -> Result<(), String> {
    for s in sources {
        if let Some((a, b)) = month_of_name(&s.path) {
            guard.check(&format!("{} ({})", what, s.path.display()), a, b)?;
        }
    }
    Ok(())
}

fn parallel<T: Send, R: Send>(
    items: Vec<T>,
    jobs: usize,
    f: impl Fn(T) -> Result<R, String> + Sync,
) -> Result<Vec<R>, String> {
    let n = items.len();
    let slots: Vec<std::sync::Mutex<Option<T>>> =
        items.into_iter().map(|t| std::sync::Mutex::new(Some(t))).collect();
    let results: Vec<std::sync::Mutex<Option<Result<R, String>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(n.max(1)) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let item = slots[i].lock().unwrap().take().unwrap();
                *results[i].lock().unwrap() = Some(f(item));
            });
        }
    });
    let mut out = Vec::with_capacity(n);
    for r in results {
        out.push(r.into_inner().unwrap().ok_or("worker did not finish")??);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// P0: snapshot
// ---------------------------------------------------------------------------

struct FileScan {
    rel: String,
    symbol: String,
    bytes: u64,
    mtime: i64,
    sha256: String,
    rows: usize,
    first_ts: i64,
    last_ts: i64,
    units: [usize; 3],
    duplicate_rows: usize,
    duplicate_conflicts: usize,
    invalid: Vec<InvalidBar>,
}

fn scan_file(root: &Path, src: &prep::Source) -> Result<FileScan, String> {
    let bytes = std::fs::read(&src.path).map_err(|e| format!("{}: {}", src.path.display(), e))?;
    let meta = std::fs::metadata(&src.path).map_err(|e| format!("{}: {}", src.path.display(), e))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let sha256 = manifest::sha256_hex(&bytes);
    let size = bytes.len() as u64;
    let rel = relative(root, &src.path);
    let mut dec = decode_bytes(&src.path, &src.symbol, bytes)?;
    for b in &mut dec.invalid {
        b.path = rel.clone();
    }
    let mut order: Vec<usize> = (0..dec.rows.len()).collect();
    order.sort_by_key(|&i| dec.rows[i].time);
    let mut duplicate_rows = 0usize;
    let mut duplicate_conflicts = 0usize;
    let mut lo = 0usize;
    while lo < order.len() {
        let mut hi = lo + 1;
        while hi < order.len() && dec.rows[order[hi]].time == dec.rows[order[lo]].time {
            hi += 1;
        }
        if hi - lo > 1 {
            duplicate_rows += hi - lo - 1;
            let first = dec.rows[order[lo]].payload_bits();
            if order[lo + 1..hi].iter().any(|&j| dec.rows[j].payload_bits() != first) {
                duplicate_conflicts += 1;
            }
        }
        lo = hi;
    }
    let total = dec.rows.len() + dec.invalid.len();
    let first_ts = dec.rows.iter().map(|r| r.time).chain(dec.invalid.iter().map(|b| b.time)).min().unwrap_or(0);
    let last_ts = dec.rows.iter().map(|r| r.time).chain(dec.invalid.iter().map(|b| b.time)).max().unwrap_or(0);
    Ok(FileScan {
        rel,
        symbol: src.symbol.clone(),
        bytes: size,
        mtime,
        sha256,
        rows: total,
        first_ts,
        last_ts,
        units: dec.units,
        duplicate_rows,
        duplicate_conflicts,
        invalid: dec.invalid,
    })
}

fn units_json(u: &[usize; 3]) -> String {
    format!("{{\"s\":{},\"ms\":{},\"us\":{}}}", u[0], u[1], u[2])
}

/// Hash and scan every archive below `input`; the result is `snapshot.json`.
/// The text is a pure function of the archives, their sizes and mtimes.
pub fn snapshot_text(input: &str, jobs: usize) -> Result<String, String> {
    let root = PathBuf::from(input);
    let sources = discover(input)?;
    let scans = parallel(sources, jobs, |s| scan_file(&root, &s))?;
    let lb = Lockbox::crypto_2026h2();
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"tool\":\"rcsr panel snapshot\",\n  \"source\":\"local archive\",\n  \"network\":false,");
    let total_bytes: u64 = scans.iter().map(|f| f.bytes).sum();
    let symbols: BTreeSet<&str> = scans.iter().map(|f| f.symbol.as_str()).collect();
    let lockbox_rows: usize = 0;
    let _ = lockbox_rows;
    let first = scans.iter().map(|f| f.first_ts).min().unwrap_or(0);
    let last = scans.iter().map(|f| f.last_ts).max().unwrap_or(0);
    let lock_files = scans
        .iter()
        .filter(|f| f.rows > 0 && f.first_ts < lb.end && f.last_ts >= lb.start)
        .count();
    let _ = writeln!(
        s,
        "  \"coverage\":{{\"files\":{},\"symbols\":{},\"bytes\":{},\"first_ts\":{},\"last_ts\":{},\"first_date\":{},\"last_date\":{},\"lockbox_files\":{}}},",
        scans.len(),
        symbols.len(),
        total_bytes,
        first,
        last,
        jstr(&date(first)),
        jstr(&date(last)),
        lock_files
    );
    let _ = writeln!(s, "  \"entries\":[");
    for (i, f) in scans.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{\"path\":{},\"symbol\":{},\"bytes\":{},\"mtime\":{},\"sha256\":{},\"rows\":{},\"first_ts\":{},\"last_ts\":{},\"units\":{},\"duplicate_rows\":{},\"duplicate_conflicts\":{},\"invalid_ohlc\":{}}}{}",
            jstr(&f.rel),
            jstr(&f.symbol),
            f.bytes,
            f.mtime,
            jstr(&f.sha256),
            f.rows,
            f.first_ts,
            f.last_ts,
            units_json(&f.units),
            f.duplicate_rows,
            f.duplicate_conflicts,
            f.invalid.len(),
            if i + 1 == scans.len() { "" } else { "," }
        );
    }
    let _ = writeln!(s, "  ],");
    let dups: Vec<String> = scans
        .iter()
        .filter(|f| f.duplicate_rows > 0)
        .map(|f| {
            format!(
                "{{\"path\":{},\"symbol\":{},\"rows\":{},\"duplicate_rows\":{},\"identical\":{}}}",
                jstr(&f.rel),
                jstr(&f.symbol),
                f.rows,
                f.duplicate_rows,
                f.duplicate_conflicts == 0
            )
        })
        .collect();
    let _ = writeln!(s, "  \"duplicates\":[{}],", dups.join(","));
    let bad: Vec<String> = scans
        .iter()
        .flat_map(|f| f.invalid.iter())
        .map(|b| {
            format!(
                "{{\"path\":{},\"symbol\":{},\"ts\":{},\"date\":{},\"reason\":{},\"open\":{},\"high\":{},\"low\":{},\"close\":{}}}",
                jstr(&b.path),
                jstr(&b.symbol),
                b.time,
                jstr(&format!("{} {:02}:{:02}", date(b.time), b.time.rem_euclid(86_400) / 3600, b.time.rem_euclid(3600) / 60)),
                jstr(b.reason),
                b.open,
                b.high,
                b.low,
                b.close
            )
        })
        .collect();
    let _ = writeln!(s, "  \"invalid_ohlc\":[{}],", bad.join(","));
    let mut per: BTreeMap<&str, (usize, usize, i64, i64, [usize; 3])> = BTreeMap::new();
    for f in &scans {
        let e = per.entry(f.symbol.as_str()).or_insert((0, 0, i64::MAX, i64::MIN, [0; 3]));
        e.0 += 1;
        e.1 += f.rows;
        if f.rows > 0 {
            e.2 = e.2.min(f.first_ts);
            e.3 = e.3.max(f.last_ts);
        }
        for k in 0..3 {
            e.4[k] += f.units[k];
        }
    }
    let syms: Vec<String> = per
        .iter()
        .map(|(sym, (files, rows, a, b, u))| {
            format!(
                "{{\"symbol\":{},\"files\":{},\"rows\":{},\"first_ts\":{},\"last_ts\":{},\"units\":{},\"mixed_units\":{}}}",
                jstr(sym),
                files,
                rows,
                if *rows > 0 { *a } else { 0 },
                if *rows > 0 { *b } else { 0 },
                units_json(u),
                u.iter().filter(|c| **c > 0).count() > 1
            )
        })
        .collect();
    let _ = writeln!(s, "  \"symbols\":[{}]", syms.join(",\n    "));
    s.push_str("}\n");
    Ok(s)
}

// ---------------------------------------------------------------------------
// P1: hourly base
// ---------------------------------------------------------------------------

/// One hourly base row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseRow {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades: f64,
    pub taker_buy_base: f64,
    pub taker_buy_quote: f64,
    pub cov15m: u8,
    pub gap: u32,
    pub flags: u32,
    pub traded: bool,
    pub dup_rows: u32,
    pub unit: u8,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolLedger {
    pub input_rows: usize,
    pub unique_rows: usize,
    pub partial_dropped: usize,
    pub partial_kept: usize,
    pub zero_volume: usize,
    pub units: [usize; 3],
    pub conflicts: Vec<(i64, u32)>,
    pub invalid: Vec<InvalidBar>,
}

struct BaseSymbol {
    symbol: String,
    rows: Vec<BaseRow>,
    ledger: SymbolLedger,
}

#[derive(Clone, Debug)]
pub struct BaseOptions {
    pub input: String,
    pub out: String,
    pub keep_partial: bool,
    pub jobs: usize,
    pub allow_lockbox: bool,
    pub resume: Option<String>,
    pub group_rows: usize,
    pub interval_seconds: i64,
    pub gpu: GpuMode,
}

impl BaseOptions {
    pub fn new(input: &str, out: &str) -> Self {
        BaseOptions {
            input: input.to_string(),
            out: out.to_string(),
            keep_partial: false,
            jobs: 1,
            allow_lockbox: false,
            resume: None,
            group_rows: store::DEFAULT_GROUP_ROWS,
            interval_seconds: IN_STEP,
            gpu: GpuMode::Auto,
        }
    }
}

/// Decoded rows of a symbol to hourly base rows.
pub fn fold_symbol(
    symbol: &str,
    decoded: Vec<Raw2>,
    invalid: Vec<InvalidBar>,
    units: [usize; 3],
    keep_partial: bool,
) -> Result<(Vec<BaseRow>, SymbolLedger), String> {
    let raws: Vec<Raw> = decoded.into_iter().map(|r| r.0).collect();
    fold_cpu(symbol, &raws, invalid, units, keep_partial)
}

fn fold_cpu(
    symbol: &str,
    raws: &[Raw],
    invalid: Vec<InvalidBar>,
    units: [usize; 3],
    keep_partial: bool,
) -> Result<(Vec<BaseRow>, SymbolLedger), String> {
    let bars: Vec<PrepBar> = raws.iter().map(|r| r.prep_bar()).collect();
    let segments = [PrepSegment { start: 0, len: bars.len() as u32 }];
    let out = panel_prep::prepare(&bars, &segments, IN_STEP, OUT_STEP, keep_partial)
        .map_err(|e| format!("{}: {}", symbol, e))?;
    Ok(fold_output(raws, invalid, units, keep_partial, out))
}

/// Base rows and ledger of one symbol from its prepared buckets. `out` holds
/// only this symbol's buckets and conflicts, from the CPU or the device pass.
fn fold_output(
    raws: &[Raw],
    invalid: Vec<InvalidBar>,
    units: [usize; 3],
    keep_partial: bool,
    out: panel_prep::PrepOutput,
) -> (Vec<BaseRow>, SymbolLedger) {
    let mut unit_at: Vec<(i64, u8)> = raws.iter().map(|r| (r.time, r.unit)).collect();
    unit_at.sort_by_key(|x| x.0);
    let unique = unit_at.windows(2).filter(|w| w[0].0 != w[1].0).count() + usize::from(!unit_at.is_empty());
    let mut buckets_seen: BTreeSet<i64> = BTreeSet::new();
    for r in raws {
        buckets_seen.insert(r.time.div_euclid(OUT_STEP) * OUT_STEP);
    }
    let mut rows = Vec::with_capacity(out.buckets.len());
    let mut ledger = SymbolLedger {
        input_rows: raws.len(),
        unique_rows: unique,
        units,
        invalid,
        ..Default::default()
    };
    for b in &out.buckets {
        let k = unit_at.partition_point(|x| x.0 < b.time);
        let unit = unit_at.get(k).map(|x| x.1).unwrap_or(UNIT_S);
        let mut flags = 0u32;
        if b.partial {
            flags |= FLAG_PARTIAL_BUCKET;
            ledger.partial_kept += 1;
        }
        if b.duplicate_conflict {
            flags |= FLAG_DUPLICATE_CONFLICT;
        }
        if !b.traded {
            ledger.zero_volume += 1;
        }
        rows.push(BaseRow {
            time: b.time,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
            volume: b.volume,
            quote_volume: b.quote_volume,
            trades: b.trades,
            taker_buy_base: b.taker_buy_base,
            taker_buy_quote: b.taker_buy_quote,
            cov15m: b.cov15m as u8,
            gap: b.gap,
            flags,
            traded: b.traded,
            dup_rows: b.dup_rows,
            unit,
        });
    }
    if !keep_partial {
        ledger.partial_dropped = buckets_seen.len() - out.buckets.len();
    }
    ledger.conflicts = out.conflicts.iter().map(|c| (c.time, c.len)).collect();
    (rows, ledger)
}

/// A decoded row wrapper, public so tests can fold fixtures without archives.
#[derive(Clone, Copy, Debug)]
pub struct Raw2(Raw);

impl Raw2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(time: i64, unit: u8, ohlc: [f64; 4], flow: [f64; 5]) -> Self {
        Raw2(Raw {
            time,
            unit,
            open: ohlc[0],
            high: ohlc[1],
            low: ohlc[2],
            close: ohlc[3],
            volume: flow[0],
            quote_volume: flow[1],
            trades: flow[2],
            taker_buy_base: flow[3],
            taker_buy_quote: flow[4],
        })
    }
}

const CACHE_MAGIC: &[u8; 8] = b"RCSRBASE";
const CACHE_VERSION: u32 = 1;

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.at + n > self.b.len() {
            return Err("truncated base cache".to_string());
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "bad base cache text".to_string())
    }
}

fn put_string(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

fn encode_symbol(key: &str, s: &BaseSymbol) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(CACHE_MAGIC);
    put_u64(&mut b, CACHE_VERSION as u64);
    put_string(&mut b, key);
    put_string(&mut b, &s.symbol);
    let l = &s.ledger;
    for v in [l.input_rows, l.unique_rows, l.partial_dropped, l.partial_kept, l.zero_volume] {
        put_u64(&mut b, v as u64);
    }
    for v in l.units {
        put_u64(&mut b, v as u64);
    }
    put_u64(&mut b, l.conflicts.len() as u64);
    for (t, n) in &l.conflicts {
        put_u64(&mut b, *t as u64);
        put_u64(&mut b, *n as u64);
    }
    put_u64(&mut b, l.invalid.len() as u64);
    for i in &l.invalid {
        put_string(&mut b, &i.path);
        put_string(&mut b, i.reason);
        put_u64(&mut b, i.time as u64);
        for v in [i.open, i.high, i.low, i.close] {
            put_u64(&mut b, v.to_bits());
        }
    }
    put_u64(&mut b, s.rows.len() as u64);
    for r in &s.rows {
        put_u64(&mut b, r.time as u64);
        for v in [
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
            r.quote_volume,
            r.trades,
            r.taker_buy_base,
            r.taker_buy_quote,
        ] {
            put_u64(&mut b, v.to_bits());
        }
        put_u64(
            &mut b,
            (r.cov15m as u64) | ((r.unit as u64) << 8) | ((r.traded as u64) << 16) | ((r.flags as u64) << 24),
        );
        put_u64(&mut b, (r.gap as u64) | ((r.dup_rows as u64) << 32));
    }
    b
}

fn static_reason(s: &str) -> &'static str {
    match s {
        "non_finite" => "non_finite",
        "non_positive_price" => "non_positive_price",
        "high_below_low" => "high_below_low",
        "high_below_body" => "high_below_body",
        _ => "low_above_body",
    }
}

fn decode_symbol(bytes: &[u8], want_key: &str) -> Result<BaseSymbol, String> {
    let mut c = Cursor { b: bytes, at: 0 };
    if c.take(8)? != CACHE_MAGIC || c.u64()? != CACHE_VERSION as u64 {
        return Err("not a base cache".to_string());
    }
    if c.string()? != want_key {
        return Err("base cache key mismatch".to_string());
    }
    let symbol = c.string()?;
    let mut l = SymbolLedger::default();
    l.input_rows = c.u64()? as usize;
    l.unique_rows = c.u64()? as usize;
    l.partial_dropped = c.u64()? as usize;
    l.partial_kept = c.u64()? as usize;
    l.zero_volume = c.u64()? as usize;
    for k in 0..3 {
        l.units[k] = c.u64()? as usize;
    }
    for _ in 0..c.u64()? {
        let t = c.u64()? as i64;
        l.conflicts.push((t, c.u64()? as u32));
    }
    for _ in 0..c.u64()? {
        let path = c.string()?;
        let reason = static_reason(&c.string()?);
        let time = c.u64()? as i64;
        let (open, high, low, close) = (c.f64()?, c.f64()?, c.f64()?, c.f64()?);
        l.invalid.push(InvalidBar { symbol: symbol.clone(), path, time, reason, open, high, low, close });
    }
    let n = c.u64()? as usize;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let time = c.u64()? as i64;
        let mut v = [0f64; 9];
        for x in &mut v {
            *x = c.f64()?;
        }
        let packed = c.u64()?;
        let gd = c.u64()?;
        rows.push(BaseRow {
            time,
            open: v[0],
            high: v[1],
            low: v[2],
            close: v[3],
            volume: v[4],
            quote_volume: v[5],
            trades: v[6],
            taker_buy_base: v[7],
            taker_buy_quote: v[8],
            cov15m: (packed & 0xff) as u8,
            unit: ((packed >> 8) & 0xff) as u8,
            traded: (packed >> 16) & 0xff != 0,
            flags: (packed >> 24) as u32,
            gap: (gd & 0xffff_ffff) as u32,
            dup_rows: (gd >> 32) as u32,
        });
    }
    Ok(BaseSymbol { symbol, rows, ledger: l })
}

fn cache_path(dir: &str, symbol: &str) -> PathBuf {
    let safe: String = symbol
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    Path::new(dir).join(format!("{}.base", safe))
}

/// What a base build did, for the CLI line.
#[derive(Clone, Debug, Default)]
pub struct BaseSummary {
    pub rows: usize,
    pub symbols: usize,
    pub reused: usize,
    pub manifest: String,
    pub route: DeviceRoute,
}

/// Where the 15m to 1h fold ran, with every reason the device was not used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceRoute {
    /// `auto`, `on` or `off`.
    pub requested: &'static str,
    pub device: Option<String>,
    pub symbols_device: usize,
    pub symbols_cpu: usize,
    pub rows_device: usize,
    pub rows_cpu: usize,
    /// Device launches (one per wave of symbols).
    pub passes: usize,
    /// Labelled reasons the CPU ran a wave, with the symbols it covered.
    pub fallbacks: Vec<(String, usize)>,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_s: f64,
    pub wall_s: f64,
}

impl DeviceRoute {
    fn new(mode: GpuMode) -> Self {
        DeviceRoute {
            requested: match mode {
                GpuMode::Auto => "auto",
                GpuMode::On => "on",
                GpuMode::Off => "off",
            },
            ..Default::default()
        }
    }

    fn fall_back(&mut self, reason: String, symbols: usize) {
        match self.fallbacks.iter_mut().find(|f| f.0 == reason) {
            Some(f) => f.1 += symbols,
            None => self.fallbacks.push((reason, symbols)),
        }
    }

    fn json(&self) -> String {
        let fallbacks: Vec<String> = self
            .fallbacks
            .iter()
            .map(|(r, n)| format!("{{\"reason\":{},\"symbols\":{}}}", jstr(r), n))
            .collect();
        format!(
            "{{\"requested\":{},\"device\":{},\"symbols_device\":{},\"symbols_cpu\":{},\"rows_device\":{},\"rows_cpu\":{},\"passes\":{},\"h2d_bytes\":{},\"d2h_bytes\":{},\"fallbacks\":[{}]}}",
            jstr(self.requested),
            self.device.as_deref().map(jstr).unwrap_or_else(|| "null".to_string()),
            self.symbols_device,
            self.symbols_cpu,
            self.rows_device,
            self.rows_cpu,
            self.passes,
            self.h2d_bytes,
            self.d2h_bytes,
            fallbacks.join(",")
        )
    }
}

/// The planner: whether one wave of `rows` decoded 15m bars runs on the device.
/// `Err` is the labelled reason for the CPU. `--gpu on` forces the device; `auto`
/// needs at least [`crate::enrich::GPU_MIN_ROWS`] rows, below which the upload
/// and launch cost more than the CPU fold.
pub fn plan_device(mode: GpuMode, rows: usize, probe: &crate::backend::Probe) -> Result<(), String> {
    match mode {
        GpuMode::Off => Err("gpu_off".to_string()),
        _ if !probe.available() => Err(format!(
            "no_device: {}",
            probe.reason.as_deref().unwrap_or("no usable CUDA device")
        )),
        GpuMode::On => Ok(()),
        GpuMode::Auto if rows >= crate::enrich::GPU_MIN_ROWS => Ok(()),
        GpuMode::Auto => Err(format!(
            "too_few_rows: {} < {}",
            rows,
            crate::enrich::GPU_MIN_ROWS
        )),
    }
}

/// One symbol decoded and waiting for its fold.
struct Fresh {
    symbol: String,
    key: String,
    hashes: Vec<(String, String, u64)>,
    raws: Vec<Raw>,
    invalid: Vec<InvalidBar>,
    units: [usize; 3],
}

enum Loaded {
    Cached(BaseSymbol, Vec<(String, String, u64)>),
    Fresh(Fresh),
}

fn load_symbol(
    opts: &BaseOptions,
    root: &Path,
    settings: &str,
    symbol: String,
    srcs: Vec<prep::Source>,
    reused: &AtomicUsize,
) -> Result<Loaded, String> {
    let mut blobs = Vec::new();
    let mut hashes = Vec::new();
    for s in &srcs {
        let bytes = std::fs::read(&s.path).map_err(|e| format!("{}: {}", s.path.display(), e))?;
        hashes.push((relative(root, &s.path), manifest::sha256_hex(&bytes), bytes.len() as u64));
        blobs.push(bytes);
    }
    let mut key = String::from(settings);
    for (p, h, n) in &hashes {
        let _ = write!(key, ";{}:{}:{}", p, h, n);
    }
    let key = manifest::sha256_hex(key.as_bytes());
    if let Some(dir) = &opts.resume {
        let path = cache_path(dir, &symbol);
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(sym) = decode_symbol(&bytes, &key) {
                reused.fetch_add(1, Ordering::Relaxed);
                return Ok(Loaded::Cached(sym, hashes));
            }
        }
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut invalid = Vec::new();
    let mut units = [0usize; 3];
    for (s, bytes) in srcs.iter().zip(blobs) {
        let mut d = decode_bytes(&s.path, &symbol, bytes)?;
        for b in &mut d.invalid {
            b.path = relative(root, &s.path);
        }
        raws.extend(d.rows);
        invalid.extend(d.invalid);
        for k in 0..3 {
            units[k] += d.units[k];
        }
    }
    Ok(Loaded::Fresh(Fresh { symbol, key, hashes, raws, invalid, units }))
}

fn finish_symbol(
    opts: &BaseOptions,
    f: Fresh,
    rows: Vec<BaseRow>,
    ledger: SymbolLedger,
) -> Result<(BaseSymbol, Vec<(String, String, u64)>), String> {
    let sym = BaseSymbol { symbol: f.symbol.clone(), rows, ledger };
    if let Some(dir) = &opts.resume {
        publish_bytes(&cache_path(dir, &f.symbol), &encode_symbol(&f.key, &sym))?;
    }
    Ok((sym, f.hashes))
}

fn fold_fresh_cpu(
    opts: &BaseOptions,
    mut f: Fresh,
) -> Result<(BaseSymbol, Vec<(String, String, u64)>), String> {
    let invalid = std::mem::take(&mut f.invalid);
    let (rows, ledger) = fold_cpu(&f.symbol, &f.raws, invalid, f.units, opts.keep_partial)?;
    finish_symbol(opts, f, rows, ledger)
}

/// One device pass over the wave's symbols as segments of one block. Returns
/// each symbol's buckets and conflicts, in input order.
#[cfg(feature = "cuda")]
fn device_prepare(
    fresh: &[Fresh],
    keep_partial: bool,
    route: &mut DeviceRoute,
) -> Result<Vec<panel_prep::PrepOutput>, String> {
    let total: usize = fresh.iter().map(|f| f.raws.len()).sum();
    if total > i32::MAX as usize {
        return Err(format!("too_many_rows: {}", total));
    }
    let mut bars: Vec<PrepBar> = Vec::with_capacity(total);
    let mut segments = Vec::new();
    let mut owner = Vec::new();
    for (i, f) in fresh.iter().enumerate() {
        if f.raws.is_empty() {
            continue;
        }
        segments.push(PrepSegment { start: bars.len() as u32, len: f.raws.len() as u32 });
        owner.push(i);
        bars.extend(f.raws.iter().map(|r| r.prep_bar()));
    }
    let mut per = vec![panel_prep::PrepOutput::default(); fresh.len()];
    if bars.is_empty() {
        return Ok(per);
    }
    let (out, stats) = crate::panel_prep_gpu::prepare(&bars, &segments, IN_STEP, OUT_STEP, keep_partial)
        .ok_or_else(|| "launch: the device prep pass failed".to_string())?;
    for b in out.buckets {
        let i = *owner
            .get(b.segment as usize)
            .ok_or_else(|| format!("launch: bucket in segment {}", b.segment))?;
        per[i].buckets.push(b);
    }
    for c in out.conflicts {
        let i = *owner
            .get(c.segment as usize)
            .ok_or_else(|| format!("launch: conflict in segment {}", c.segment))?;
        per[i].conflicts.push(c);
    }
    route.passes += 1;
    route.h2d_bytes += stats.h2d_bytes;
    route.d2h_bytes += stats.d2h_bytes;
    route.kernel_s += stats.kernel_s;
    route.wall_s += stats.wall_s;
    Ok(per)
}

#[cfg(not(feature = "cuda"))]
fn device_prepare(
    _fresh: &[Fresh],
    _keep_partial: bool,
    _route: &mut DeviceRoute,
) -> Result<Vec<panel_prep::PrepOutput>, String> {
    Err("no_device: built without --features cuda".to_string())
}

/// Fold one wave of freshly decoded symbols where the planner chooses.
fn fold_wave(
    opts: &BaseOptions,
    fresh: Vec<Fresh>,
    probe: &crate::backend::Probe,
    route: &mut DeviceRoute,
) -> Result<Vec<(BaseSymbol, Vec<(String, String, u64)>)>, String> {
    let rows: usize = fresh.iter().map(|f| f.raws.len()).sum();
    let n = fresh.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let outputs = match plan_device(opts.gpu, rows, probe) {
        Ok(()) => match device_prepare(&fresh, opts.keep_partial, route) {
            Ok(o) => Some(o),
            Err(e) => {
                route.fall_back(e, n);
                None
            }
        },
        Err(reason) => {
            route.fall_back(reason, n);
            None
        }
    };
    let Some(outputs) = outputs else {
        route.symbols_cpu += n;
        route.rows_cpu += rows;
        return parallel(fresh, opts.jobs, |f| fold_fresh_cpu(opts, f));
    };
    route.symbols_device += n;
    route.rows_device += rows;
    route.device = probe.device.clone();
    let items: Vec<(Fresh, panel_prep::PrepOutput)> = fresh.into_iter().zip(outputs).collect();
    parallel(items, opts.jobs, |(mut f, out)| {
        let invalid = std::mem::take(&mut f.invalid);
        let (rows, ledger) = fold_output(&f.raws, invalid, f.units, opts.keep_partial, out);
        finish_symbol(opts, f, rows, ledger)
    })
}

/// P1: decode the archives, apply the duplicate and full-bucket policy and
/// write the symbol-major hourly base store plus `<out>.manifest.json`.
pub fn build_base(opts: &BaseOptions) -> Result<BaseSummary, String> {
    let root = PathBuf::from(&opts.input);
    let sources = discover(&opts.input)?;
    let guard = Guard::new(opts.allow_lockbox, None);
    guard_archive_names(&sources, &guard, "panel base")?;
    if opts.interval_seconds != IN_STEP {
        return Err(format!(
            "--interval {}s is not supported: the hourly fold is defined for 15m bars",
            opts.interval_seconds
        ));
    }
    let mut by_symbol: BTreeMap<String, Vec<prep::Source>> = BTreeMap::new();
    for s in sources {
        by_symbol.entry(s.symbol.clone()).or_default().push(s);
    }
    if let Some(dir) = &opts.resume {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir, e))?;
    }
    let settings = format!("interval={};keep_partial={}", IN_STEP, opts.keep_partial);
    let mut groups: Vec<(String, Vec<prep::Source>)> = by_symbol.into_iter().collect();
    let reused = AtomicUsize::new(0);
    let mut route = DeviceRoute::new(opts.gpu);
    let done: Vec<(BaseSymbol, Vec<(String, String, u64)>)> = if opts.gpu == GpuMode::Off {
        parallel(groups, opts.jobs, |(symbol, srcs)| {
            match load_symbol(opts, &root, &settings, symbol, srcs, &reused)? {
                Loaded::Cached(sym, hashes) => Ok((sym, hashes)),
                Loaded::Fresh(f) => fold_fresh_cpu(opts, f),
            }
        })?
    } else {
        let probe = crate::backend::probe();
        let wave = opts.jobs.max(1) * 8;
        let mut done = Vec::new();
        while !groups.is_empty() {
            let rest = groups.split_off(wave.min(groups.len()));
            let chunk = std::mem::replace(&mut groups, rest);
            let loaded = parallel(chunk, opts.jobs, |(symbol, srcs)| {
                load_symbol(opts, &root, &settings, symbol, srcs, &reused)
            })?;
            let mut fresh = Vec::new();
            for l in loaded {
                match l {
                    Loaded::Cached(sym, hashes) => done.push((sym, hashes)),
                    Loaded::Fresh(f) => fresh.push(f),
                }
            }
            done.extend(fold_wave(opts, fresh, &probe, &mut route)?);
        }
        done
    };
    let mut syms: Vec<BaseSymbol> = Vec::new();
    let mut inputs: Vec<(String, String, u64)> = Vec::new();
    for (s, h) in done {
        syms.push(s);
        inputs.extend(h);
    }
    syms.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    inputs.sort();
    let times: Vec<i64> = syms.iter().flat_map(|s| s.rows.iter().map(|r| r.time)).collect();
    let overlap = guard.check_times("panel base", &times)?;
    let table = base_table(&syms);
    if table.len() == 0 {
        return Err("panel base: no complete hourly bucket was produced".to_string());
    }
    store::write_store_with_time(&opts.out, &table, opts.group_rows, Some(0))?;
    let text = base_manifest(opts, &syms, &inputs, &overlap, guard.allow, &route);
    let manifest_path = format!("{}.manifest.json", opts.out);
    std::fs::write(&manifest_path, text).map_err(|e| format!("{}: {}", manifest_path, e))?;
    Ok(BaseSummary {
        rows: table.len(),
        symbols: syms.len(),
        reused: reused.load(Ordering::Relaxed),
        manifest: manifest_path,
        route,
    })
}

fn publish_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {}", tmp.display(), e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
}

fn f64_column(values: Vec<f64>) -> Column {
    Column { values, ints: None, dict: None }
}

fn symbol_column(codes: Vec<u32>, levels: Vec<String>) -> Column {
    Column { values: Vec::new(), ints: None, dict: Some(DictColumn { codes, levels }) }
}

fn base_table(syms: &[BaseSymbol]) -> Table {
    let n: usize = syms.iter().map(|s| s.rows.len()).sum();
    let mut ts = Vec::with_capacity(n);
    let mut codes = Vec::with_capacity(n);
    let mut levels = Vec::new();
    let mut cols: Vec<Vec<f64>> = (0..15).map(|_| Vec::with_capacity(n)).collect();
    for s in syms {
        if s.rows.is_empty() {
            continue;
        }
        let code = levels.len() as u32;
        levels.push(s.symbol.clone());
        for r in &s.rows {
            ts.push(r.time);
            codes.push(code);
            let v = [
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.quote_volume,
                r.trades,
                r.taker_buy_base,
                r.taker_buy_quote,
                r.cov15m as f64,
                r.gap as f64,
                r.flags as f64,
                r.traded as u8 as f64,
                r.dup_rows as f64,
                r.unit as f64,
            ];
            for (c, x) in cols.iter_mut().zip(v) {
                c.push(x);
            }
        }
    }
    let mut columns = Vec::with_capacity(BASE_COLUMNS.len());
    let mut numeric = Vec::with_capacity(BASE_COLUMNS.len());
    columns.push(Column { values: ts.iter().map(|t| *t as f64).collect(), ints: Some(ts), dict: None });
    numeric.push(true);
    columns.push(symbol_column(codes, levels));
    numeric.push(false);
    for c in cols {
        columns.push(f64_column(c));
        numeric.push(true);
    }
    Table::from_parts(BASE_COLUMNS.iter().map(|s| s.to_string()).collect(), numeric, columns, n, 0)
}

fn base_manifest(
    opts: &BaseOptions,
    syms: &[BaseSymbol],
    inputs: &[(String, String, u64)],
    overlap: &crate::lockbox::Overlap,
    allowed: bool,
    route: &DeviceRoute,
) -> String {
    let mut input_rows = 0usize;
    let mut unique_rows = 0usize;
    let mut partial_dropped = 0usize;
    let mut partial_kept = 0usize;
    let mut zero_volume = 0usize;
    let mut rows_out = 0usize;
    let mut cov = [0usize; 5];
    let mut conflicts = Vec::new();
    let mut invalid = Vec::new();
    let mut units = Vec::new();
    for s in syms {
        let l = &s.ledger;
        input_rows += l.input_rows;
        unique_rows += l.unique_rows;
        partial_dropped += l.partial_dropped;
        partial_kept += l.partial_kept;
        zero_volume += l.zero_volume;
        rows_out += s.rows.len();
        for r in &s.rows {
            cov[(r.cov15m as usize).min(4)] += 1;
        }
        for (t, len) in &l.conflicts {
            conflicts.push(format!("{{\"symbol\":{},\"ts\":{},\"date\":{},\"rows\":{}}}", jstr(&s.symbol), t, jstr(&date(*t)), len));
        }
        for b in &l.invalid {
            invalid.push(format!(
                "{{\"symbol\":{},\"path\":{},\"ts\":{},\"reason\":{},\"open\":{},\"high\":{},\"low\":{},\"close\":{}}}",
                jstr(&b.symbol), jstr(&b.path), b.time, jstr(b.reason), b.open, b.high, b.low, b.close
            ));
        }
        units.push(format!("{{\"symbol\":{},\"units\":{}}}", jstr(&s.symbol), units_json(&l.units)));
    }
    let duplicate_rows = input_rows - unique_rows;
    let conflict_rows: usize = syms
        .iter()
        .flat_map(|s| s.ledger.conflicts.iter())
        .map(|(_, len)| *len as usize - 1)
        .sum();
    let invalid_n: usize = syms.iter().map(|s| s.ledger.invalid.len()).sum();
    let conflict_n: usize = syms.iter().map(|s| s.ledger.conflicts.len()).sum();
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"tool\":\"rcsr panel base\",\n  \"schema\":\"panel-base-1\",");
    let _ = writeln!(
        s,
        "  \"resample\":{{\"in_step\":{},\"out_step\":{},\"require_full_bucket\":{},\"keep_partial\":{}}},",
        IN_STEP, OUT_STEP, !opts.keep_partial, opts.keep_partial
    );
    if opts.gpu != GpuMode::Off {
        let _ = writeln!(s, "  \"backend\":{},", route.json());
    }
    let _ = writeln!(s, "  \"columns\":[{}],", BASE_COLUMNS.iter().map(|c| jstr(c)).collect::<Vec<_>>().join(","));
    let _ = writeln!(s, "  \"inputs\":{{\"files\":{},\"archives\":[", inputs.len());
    for (i, (p, h, n)) in inputs.iter().enumerate() {
        let _ = writeln!(
            s,
            "    {{\"path\":{},\"sha256\":{},\"bytes\":{}}}{}",
            jstr(p),
            jstr(h),
            n,
            if i + 1 == inputs.len() { "" } else { "," }
        );
    }
    let _ = writeln!(s, "  ]}},");
    let _ = writeln!(
        s,
        "  \"quality\":{{\"input_rows\":{},\"unique_rows\":{},\"duplicate_rows\":{},\"identical_collapsed\":{},\"duplicate_conflicts\":{},\"invalid_ohlc\":{},\"partial_dropped\":{},\"partial_kept\":{},\"zero_volume\":{},\"rows_out\":{},\"symbols\":{},\"cov15m\":[{},{},{},{},{}]}},",
        input_rows,
        unique_rows,
        duplicate_rows,
        duplicate_rows - conflict_rows,
        conflict_n,
        invalid_n,
        partial_dropped,
        partial_kept,
        zero_volume,
        rows_out,
        syms.len(),
        cov[0],
        cov[1],
        cov[2],
        cov[3],
        cov[4]
    );
    let _ = writeln!(s, "  \"conflicts\":[{}],", conflicts.join(","));
    let _ = writeln!(s, "  \"invalid_ohlc\":[{}],", invalid.join(","));
    let _ = writeln!(s, "  \"unit_histogram\":[{}],", units.join(","));
    let lb = Lockbox::crypto_2026h2();
    let _ = writeln!(s, "  \"lockbox\":{},", lb.json(overlap, allowed, false, false));
    let sha = manifest::sha256_file(&opts.out).unwrap_or_else(|_| "unknown".to_string());
    let bytes = std::fs::metadata(&opts.out).map(|m| m.len()).unwrap_or(0);
    let _ = writeln!(s, "  \"output\":{{\"path\":{},\"sha256\":{},\"bytes\":{}}}", jstr(&opts.out), jstr(&sha), bytes);
    s.push_str("}\n");
    s
}

// ---------------------------------------------------------------------------
// P2/P4/P5: the time-major panel
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PanelOptions {
    pub base: String,
    pub membership: String,
    pub out: String,
    pub group_rows: usize,
    pub allow_lockbox: bool,
    pub snapshot: Option<String>,
    /// A gap longer than this many days between two base bars of one symbol
    /// starts a new listing.
    pub listing_gap_days: i64,
    /// Declared listing breaks: (symbol, UTC instant).
    pub relists: Vec<(String, i64)>,
    /// An undeclared open/previous-close ratio beyond this factor fails the build.
    pub jump_factor: f64,
    /// Share of the cross-section missing at one hour that makes it a venue outage.
    pub outage_share: f64,
    pub outage_min_symbols: usize,
    pub gpu: GpuMode,
}

impl PanelOptions {
    pub fn new(base: &str, membership: &str, out: &str) -> Self {
        PanelOptions {
            base: base.to_string(),
            membership: membership.to_string(),
            out: out.to_string(),
            group_rows: store::DEFAULT_GROUP_ROWS,
            allow_lockbox: false,
            snapshot: None,
            listing_gap_days: 30,
            relists: Vec::new(),
            jump_factor: 5.0,
            outage_share: 0.5,
            outage_min_symbols: 3,
            gpu: GpuMode::Auto,
        }
    }
}

/// `SYMBOL@YYYY-MM-DD,...`
pub fn parse_relists(s: &str) -> Result<Vec<(String, i64)>, String> {
    let mut out = Vec::new();
    for item in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
        let (sym, d) = item
            .split_once('@')
            .ok_or_else(|| format!("relist '{}' must be SYMBOL@DATE", item))?;
        let t = crate::lockbox::parse_time(d).ok_or_else(|| format!("relist '{}': bad date", item))?;
        out.push((sym.trim().to_ascii_uppercase(), t));
    }
    Ok(out)
}

/// Listing ordinal of every bar: a new listing starts after a gap longer than
/// `gap_days` or at a declared relist instant.
pub fn listing_ordinals(times: &[i64], relists: &[i64], gap_days: i64) -> Vec<u32> {
    let mut out = Vec::with_capacity(times.len());
    let mut ord = 0u32;
    for (i, &t) in times.iter().enumerate() {
        if i > 0 {
            let p = times[i - 1];
            if t - p > gap_days * 86_400 || relists.iter().any(|&r| p < r && r <= t) {
                ord += 1;
            }
        }
        out.push(ord);
    }
    out
}

/// One base bar with its quality fields.
#[derive(Clone, Copy, Debug)]
struct Aux {
    cov15m: u8,
    flags: u32,
    dup_rows: u32,
    unit: u8,
    traded: bool,
}

/// A panel row with its quality fields.
#[derive(Clone, Debug)]
pub struct V2Row {
    pub row: PanelRow,
    pub listing_id: u32,
    pub cov15m: u8,
    pub flags: u32,
    pub dup_rows: u32,
    pub unit: u8,
    pub traded: bool,
}

fn read_base(path: &str) -> Result<BTreeMap<String, (Vec<Bar>, Vec<Aux>)>, String> {
    let table = store::FileStore::open(path)?.to_table()?;
    let idx = |n: &str| table.index_of(n).map_err(|_| format!("{}: not a panel base (no column '{}')", path, n));
    let (ts, sym) = (idx("ts")?, idx("symbol")?);
    let c: Vec<usize> = [
        "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "cov15m", "flags", "traded", "dup_rows", "archive_unit",
    ]
    .iter()
    .map(|n| idx(n))
    .collect::<Result<_, _>>()?;
    let mut out: BTreeMap<String, Vec<(Bar, Aux)>> = BTreeMap::new();
    for r in 0..table.len() {
        let t = table.ints(ts).map(|v| v[r]).unwrap_or_else(|| table.value(r, ts) as i64);
        let name = table.text(r, sym).map(|s| s.to_string()).unwrap_or_default();
        let v = |k: usize| table.value(r, c[k]);
        let cov = v(9) as u8;
        let flags = v(10) as u32;
        let traded = v(11) != 0.0;
        let bar = Bar {
            time: t,
            open: v(0),
            high: v(1),
            low: v(2),
            close: v(3),
            volume: v(4),
            quote_volume: v(5),
            trades: v(6),
            taker_buy_base: v(7),
            taker_buy_quote: v(8),
            complete: cov == 4 && flags & FLAG_PARTIAL_BUCKET == 0,
            traded,
        };
        out.entry(name).or_default().push((
            bar,
            Aux { cov15m: cov, flags, dup_rows: v(12) as u32, unit: v(13) as u8, traded },
        ));
    }
    let mut res = BTreeMap::new();
    for (name, mut rows) in out {
        rows.sort_by_key(|x| x.0.time);
        for w in rows.windows(2) {
            if w[0].0.time == w[1].0.time {
                return Err(format!("{}: duplicate base bar {} at {}", path, name, date(w[0].0.time)));
            }
        }
        let (bars, aux) = rows.into_iter().unzip();
        res.insert(name, (bars, aux));
    }
    Ok(res)
}

/// An undeclared level jump inside one listing.
#[derive(Clone, Debug, PartialEq)]
pub struct Jump {
    pub symbol: String,
    pub at: i64,
    pub ratio: f64,
}

struct Assembled {
    rows: Vec<V2Row>,
    last: BTreeMap<String, i64>,
    jumps: Vec<Jump>,
    listings: BTreeMap<String, u32>,
    declared_breaks: usize,
}

fn assemble(
    series_in: &BTreeMap<String, (Vec<Bar>, Vec<Aux>)>,
    membership: &Membership,
    opts: &PanelOptions,
    cutoff: Option<i64>,
) -> Assembled {
    let index = panel::membership_index(membership);
    let mut rows: Vec<V2Row> = Vec::new();
    let mut jumps = Vec::new();
    let mut last = BTreeMap::new();
    let mut listings = BTreeMap::new();
    let mut next_listing = 0u32;
    let mut declared_breaks = 0usize;
    for (symbol, (bars_all, aux_all)) in series_in {
        let keep = match cutoff {
            Some(t) => bars_all.partition_point(|b| b.time <= t),
            None => bars_all.len(),
        };
        let bars = &bars_all[..keep];
        if bars.is_empty() {
            continue;
        }
        let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
        let relists: Vec<i64> = opts
            .relists
            .iter()
            .filter(|(s, _)| s == symbol)
            .map(|(_, t)| *t)
            .collect();
        let ord = listing_ordinals(&times, &relists, opts.listing_gap_days);
        let count = ord.last().copied().unwrap_or(0) + 1;
        declared_breaks += count as usize - 1;
        listings.insert(symbol.clone(), count);
        last.insert(symbol.clone(), *times.last().unwrap());
        let decisions = index.get(symbol.as_str()).map(|v| v.as_slice());
        let part = panel::symbol_panel_rows(symbol, bars, decisions, OUT_STEP);
        let mut prev: Option<(usize, f64)> = None;
        for pr in part {
            let k = times.partition_point(|t| *t < pr.bar.time);
            let a = aux_all[k];
            if let Some((pk, pclose)) = prev {
                if pk + 1 == k && ord[pk] == ord[k] && pclose > 0.0 && pr.bar.open > 0.0 {
                    let ratio = pr.bar.open / pclose;
                    if ratio > opts.jump_factor || ratio < 1.0 / opts.jump_factor {
                        jumps.push(Jump { symbol: symbol.clone(), at: pr.bar.time, ratio });
                    }
                }
            }
            prev = Some((k, pr.bar.close));
            rows.push(V2Row {
                row: pr,
                listing_id: next_listing + ord[k],
                cov15m: a.cov15m,
                flags: a.flags,
                dup_rows: a.dup_rows,
                unit: a.unit,
                traded: a.traded,
            });
        }
        next_listing += count;
    }
    rows.sort_by(|a, b| a.row.bar.time.cmp(&b.row.bar.time).then_with(|| a.row.symbol.cmp(&b.row.symbol)));
    Assembled { rows, last, jumps, listings, declared_breaks }
}

fn v2_table(rows: &[V2Row]) -> Table {
    let n = rows.len();
    let mut ts = Vec::with_capacity(n);
    let mut codes = Vec::with_capacity(n);
    let mut levels: Vec<String> = Vec::new();
    let mut level_of: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut cols: Vec<Vec<f64>> = (0..18).map(|_| Vec::with_capacity(n)).collect();
    for r in rows {
        let b = &r.row.bar;
        ts.push(b.time);
        let code = *level_of.entry(r.row.symbol.as_str()).or_insert_with(|| {
            levels.push(r.row.symbol.clone());
            (levels.len() - 1) as u32
        });
        codes.push(code);
        let vwap = if b.volume > 0.0 { b.quote_volume / b.volume } else { f64::NAN };
        let v = [
            r.listing_id as f64,
            r.row.as_of as f64,
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.quote_volume,
            b.trades,
            b.taker_buy_base,
            b.taker_buy_quote,
            vwap,
            r.row.gap_buckets as f64,
            r.cov15m as f64,
            r.flags as f64,
            r.traded as u8 as f64,
            r.dup_rows as f64,
            r.unit as f64,
        ];
        for (c, x) in cols.iter_mut().zip(v) {
            c.push(x);
        }
    }
    let mut columns = vec![
        Column { values: ts.iter().map(|t| *t as f64).collect(), ints: Some(ts), dict: None },
        symbol_column(codes, levels),
    ];
    let mut numeric = vec![true, false];
    for c in cols {
        columns.push(f64_column(c));
        numeric.push(true);
    }
    Table::from_parts(PANEL_COLUMNS.iter().map(|s| s.to_string()).collect(), numeric, columns, n, 0)
}

/// Quality counters and outage classification of the retained rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Quality {
    pub duplicate_buckets: usize,
    pub duplicate_extra_rows: usize,
    pub duplicate_conflicts: usize,
    pub partial_buckets: usize,
    pub zero_volume: usize,
    pub venue_outage_hours: usize,
    pub venue_outage_symbol_buckets: usize,
    pub symbol_outage_buckets: usize,
    pub max_gap: usize,
    pub venue_outages: Vec<(i64, i64, usize)>,
    pub cov15m: [usize; 5],
}

pub fn quality(rows: &[V2Row], opts: &PanelOptions) -> Quality {
    let mut q = Quality::default();
    let mut present: BTreeMap<i64, usize> = BTreeMap::new();
    let mut missing: BTreeMap<i64, usize> = BTreeMap::new();
    for r in rows {
        *present.entry(r.row.bar.time).or_insert(0) += 1;
        q.cov15m[(r.cov15m as usize).min(4)] += 1;
        if r.dup_rows > r.cov15m as u32 {
            q.duplicate_buckets += 1;
            q.duplicate_extra_rows += (r.dup_rows - r.cov15m as u32) as usize;
        }
        if r.flags & FLAG_DUPLICATE_CONFLICT != 0 {
            q.duplicate_conflicts += 1;
        }
        if r.flags & FLAG_PARTIAL_BUCKET != 0 {
            q.partial_buckets += 1;
        }
        if !r.traded {
            q.zero_volume += 1;
        }
        let g = r.row.gap_buckets;
        q.max_gap = q.max_gap.max(g);
        for k in 1..=g as i64 {
            *missing.entry(r.row.bar.time - k * OUT_STEP).or_insert(0) += 1;
        }
    }
    let mut run: Option<(i64, i64, usize)> = None;
    for (&t, &m) in &missing {
        let p = present.get(&t).copied().unwrap_or(0);
        let share = m as f64 / (m + p) as f64;
        if share >= opts.outage_share && m >= opts.outage_min_symbols {
            q.venue_outage_hours += 1;
            q.venue_outage_symbol_buckets += m;
            run = match run {
                Some((a, b, mx)) if t == b + OUT_STEP => Some((a, t, mx.max(m))),
                Some(done) => {
                    q.venue_outages.push(done);
                    Some((t, t, m))
                }
                None => Some((t, t, m)),
            };
        } else {
            q.symbol_outage_buckets += m;
        }
    }
    if let Some(done) = run {
        q.venue_outages.push(done);
    }
    q
}

fn int_field(obj: &str, key: &str) -> usize {
    let needle = format!("\"{}\":", key);
    obj.find(&needle)
        .map(|i| {
            obj[i + needle.len()..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

/// What a panel build did, for the CLI line.
#[derive(Clone, Debug, Default)]
pub struct PanelSummary {
    pub rows: usize,
    pub symbols: usize,
    pub delisted: usize,
    pub gaps: usize,
    pub listings: usize,
    pub manifest: String,
    pub route: DeviceRoute,
}

/// P4/P5: assemble the time-major panel and write the store and manifest.
pub fn build_panel(opts: &PanelOptions) -> Result<PanelSummary, String> {
    let membership = panel::read_membership(&opts.membership)?;
    let series = read_base(&opts.base)?;
    let guard = Guard::new(opts.allow_lockbox, None);
    let all_times: Vec<i64> = series.values().flat_map(|(b, _)| b.iter().map(|x| x.time)).collect();
    guard.check_times("panel build", &all_times)?;
    let asm = assemble(&series, &membership, opts, None);
    if !asm.jumps.is_empty() {
        let mut msg = format!(
            "{} undeclared level jump(s) (open/previous close beyond {}x inside one listing); \
             declare a listing break with --relist SYMBOL@DATE or split at the outage:",
            asm.jumps.len(),
            opts.jump_factor
        );
        for j in asm.jumps.iter().take(20) {
            let _ = write!(msg, " {} at {} ratio {:.4};", j.symbol, date(j.at), j.ratio);
        }
        return Err(msg);
    }
    let q = quality(&asm.rows, opts);
    let panel_rows: Vec<PanelRow> = asm.rows.iter().map(|r| r.row.clone()).collect();
    let panel: Panel = panel::assemble_panel_with_last(panel_rows, &asm.last)?;
    debug_assert!(panel.rows.iter().zip(&asm.rows).all(|(a, b)| a == &b.row));
    let table = v2_table(&asm.rows);
    store::write_store_with_time(&opts.out, &table, opts.group_rows, Some(0))?;
    crate::registry::write_schema(&opts.out, crate::registry::DEFAULT_PANEL_VERSION)?;
    let ledger_misses = ledger_violations(&asm.rows, &membership);
    let snapshot_text = match &opts.snapshot {
        Some(p) => {
            let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {}", p, e))?;
            if !(t.trim().starts_with('{') && t.trim().ends_with('}')) {
                return Err(format!("snapshot '{}' is not a JSON object", p));
            }
            Some(t)
        }
        None => None,
    };
    let base_manifest_path = format!("{}.manifest.json", opts.base);
    let base_quality = std::fs::read_to_string(&base_manifest_path)
        .ok()
        .and_then(|t| manifest::object(&t, "quality").map(|s| s.to_string()));
    let invalid_ohlc = base_quality.as_deref().map(|b| int_field(b, "invalid_ohlc")).unwrap_or(0);
    let mut text = panel::provenance_json_with(
        &[&opts.base, &opts.membership],
        &BTreeMap::new(),
        &membership,
        &panel,
        &opts.out,
        opts.allow_lockbox,
        snapshot_text.as_deref(),
    );
    let block = v2_block(opts, &asm, &q, invalid_ohlc, base_quality.as_deref(), ledger_misses);
    let marker = "  \"output\":";
    let at = text.rfind(marker).ok_or("manifest has no output block")?;
    text.insert_str(at, &format!("  \"v2\":{},\n", block));
    let manifest_path = format!("{}.manifest.json", opts.out);
    std::fs::write(&manifest_path, text).map_err(|e| format!("{}: {}", manifest_path, e))?;
    Ok(PanelSummary {
        rows: panel.rows.len(),
        symbols: panel.symbols.len(),
        delisted: panel.delisted.len(),
        gaps: panel.gaps,
        listings: asm.listings.values().map(|c| *c as usize).sum(),
        manifest: manifest_path,
        route: DeviceRoute::new(opts.gpu),
    })
}

/// Rows whose `member_asof` is not the latest membership decision for the
/// symbol at that instant, or whose decision is not a selection.
pub fn ledger_violations(rows: &[V2Row], membership: &Membership) -> usize {
    let index = panel::membership_index(membership);
    rows.iter()
        .filter(|r| {
            let Some(v) = index.get(r.row.symbol.as_str()) else { return true };
            let pos = v.partition_point(|d| d.as_of <= r.row.bar.time);
            match pos.checked_sub(1).map(|i| v[i]) {
                Some(d) => !(d.member && d.as_of == r.row.as_of),
                None => true,
            }
        })
        .count()
}

fn v2_block(
    opts: &PanelOptions,
    asm: &Assembled,
    q: &Quality,
    invalid_ohlc: usize,
    base_quality: Option<&str>,
    ledger_misses: usize,
) -> String {
    let multi: Vec<String> = asm
        .listings
        .iter()
        .filter(|(_, c)| **c > 1)
        .map(|(s, c)| format!("{{\"symbol\":{},\"listings\":{}}}", jstr(s), c))
        .collect();
    let outages: Vec<String> = q
        .venue_outages
        .iter()
        .map(|(a, b, m)| {
            format!(
                "{{\"start\":{},\"end\":{},\"hours\":{},\"symbols_missing_max\":{}}}",
                jstr(&format!("{} {:02}:00", date(*a), a.rem_euclid(86_400) / 3600)),
                jstr(&format!("{} {:02}:00", date(*b), b.rem_euclid(86_400) / 3600)),
                (b - a) / OUT_STEP + 1,
                m
            )
        })
        .collect();
    let mut s = String::from("{");
    let _ = write!(
        s,
        "\"panel_version\":2,\"columns\":[{}],\"derivatives\":{},",
        PANEL_COLUMNS.iter().map(|c| jstr(c)).collect::<Vec<_>>().join(","),
        crate::derivatives::manifest_json(None)
    );
    let _ = write!(
        s,
        "\"resample\":{{\"in_step\":{},\"out_step\":{}}},\"listing\":{{\"gap_days\":{},\"relists\":[{}],\"total\":{},\"multi_listing_symbols\":[{}]}},",
        IN_STEP,
        OUT_STEP,
        opts.listing_gap_days,
        opts.relists
            .iter()
            .map(|(s, t)| format!("{{\"symbol\":{},\"date\":{}}}", jstr(s), jstr(&date(*t))))
            .collect::<Vec<_>>()
            .join(","),
        asm.listings.values().map(|c| *c as usize).sum::<usize>(),
        multi.join(",")
    );
    let _ = write!(
        s,
        "\"jump_check\":{{\"factor\":{},\"undeclared\":{},\"listing_breaks\":{}}},",
        opts.jump_factor,
        asm.jumps.len(),
        asm.declared_breaks
    );
    let _ = write!(
        s,
        "\"quality\":{{\"duplicate_buckets\":{},\"duplicate_extra_rows\":{},\"duplicate_conflicts\":{},\"invalid_ohlc\":{},\"partial_buckets\":{},\"zero_volume\":{},\"venue_outage_hours\":{},\"venue_outage_symbol_buckets\":{},\"symbol_outage_buckets\":{},\"max_gap\":{},\"cov15m\":[{},{},{},{},{}]}},",
        q.duplicate_buckets,
        q.duplicate_extra_rows,
        q.duplicate_conflicts,
        invalid_ohlc,
        q.partial_buckets,
        q.zero_volume,
        q.venue_outage_hours,
        q.venue_outage_symbol_buckets,
        q.symbol_outage_buckets,
        q.max_gap,
        q.cov15m[0],
        q.cov15m[1],
        q.cov15m[2],
        q.cov15m[3],
        q.cov15m[4]
    );
    let _ = write!(
        s,
        "\"outage_rule\":{{\"share\":{},\"min_symbols\":{}}},\"venue_outages\":[{}],\"membership_ledger\":{{\"rows_not_in_force\":{}}}",
        opts.outage_share,
        opts.outage_min_symbols,
        outages.join(","),
        ledger_misses
    );
    if let Some(b) = base_quality {
        let _ = write!(s, ",\"base_quality\":{}", b);
    }
    s.push('}');
    s
}

// ---------------------------------------------------------------------------
// P6: validation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub id: &'static str,
    pub name: &'static str,
    /// "pass", "fail", "skipped" or "not_applicable".
    pub status: &'static str,
    pub violations: usize,
    pub detail: String,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn pass(&self) -> bool {
        self.checks.iter().all(|c| c.status != "fail")
    }

    pub fn json(&self) -> String {
        let mut s = String::from("{\"tool\":\"rcsr panel validate\",");
        let _ = write!(s, "\"pass\":{},\"checks\":[", self.pass());
        for (i, c) in self.checks.iter().enumerate() {
            let _ = write!(
                s,
                "{}{{\"id\":{},\"name\":{},\"status\":{},\"violations\":{},\"detail\":{}}}",
                if i == 0 { "" } else { "," },
                jstr(c.id),
                jstr(c.name),
                jstr(c.status),
                c.violations,
                jstr(&c.detail)
            );
        }
        s.push_str("]}");
        s
    }
}

#[derive(Clone, Debug)]
pub struct ValidateOptions {
    pub panel: String,
    pub base: Option<String>,
    pub membership: Option<String>,
    pub replay: usize,
    pub panel_options: PanelOptions,
}

fn check(id: &'static str, name: &'static str, violations: usize, detail: impl Into<String>) -> Check {
    Check { id, name, status: if violations == 0 { "pass" } else { "fail" }, violations, detail: detail.into() }
}

fn skipped(id: &'static str, name: &'static str, why: &str) -> Check {
    Check { id, name, status: "skipped", violations: 0, detail: why.to_string() }
}

/// V1-V12 of the panel. A failed check does not stop the others.
pub fn validate(opts: &ValidateOptions) -> Report {
    let mut rep = Report::default();
    let table = match store::FileStore::open(&opts.panel).and_then(|s| s.to_table()) {
        Ok(t) => t,
        Err(e) => {
            rep.checks.push(check("V1", "header and layout", 1, e));
            return rep;
        }
    };
    let missing: Vec<&str> = PANEL_COLUMNS.iter().copied().filter(|c| table.index_of(c).is_err()).collect();
    if !missing.is_empty() || table.len() == 0 {
        rep.checks.push(check("V1", "header and layout", 1, format!("missing columns {:?}", missing)));
        return rep;
    }
    let col = |n: &str| table.index_of(n).unwrap();
    let n = table.len();
    let ts: Vec<i64> = {
        let c = col("ts");
        table.ints(c).map(|v| v.to_vec()).unwrap_or_else(|| (0..n).map(|r| table.value(r, c) as i64).collect())
    };
    let sym: Vec<String> = (0..n).map(|r| table.text(r, col("symbol")).unwrap_or("").to_string()).collect();
    let v = |name: &str| -> Vec<f64> {
        let c = col(name);
        (0..n).map(|r| table.value(r, c)).collect()
    };
    let (open, high, low, close) = (v("open"), v("high"), v("low"), v("close"));
    let (volume, qv, trades) = (v("volume"), v("quote_volume"), v("trades"));
    let (tb, tq, vwap) = (v("taker_buy_base"), v("taker_buy_quote"), v("vwap"));
    let (gap, cov, flags, listing, asof) = (v("gap"), v("cov15m"), v("flags"), v("listing_id"), v("member_asof"));
    rep.checks.push(check("V1", "header and layout", 0, format!("{} rows, {} columns", n, table.n_cols())));

    // V2
    let mut v2 = 0usize;
    let mut last_by: BTreeMap<&str, i64> = BTreeMap::new();
    for r in 0..n {
        if ts[r] % OUT_STEP != 0 {
            v2 += 1;
        }
        if r > 0 && (ts[r], sym[r].as_str()) <= (ts[r - 1], sym[r - 1].as_str()) {
            v2 += 1;
        }
        match last_by.insert(sym[r].as_str(), ts[r]) {
            Some(p) if p >= ts[r] => v2 += 1,
            _ => {}
        }
    }
    rep.checks.push(check("V2", "unique ordered (ts, symbol) on the hour grid", v2, ""));

    // V3
    let base_series = opts.base.as_deref().map(read_base);
    match &base_series {
        Some(Ok(series)) => {
            let times: BTreeMap<&str, BTreeSet<i64>> =
                series.iter().map(|(s, (b, _))| (s.as_str(), b.iter().map(|x| x.time).collect())).collect();
            let bad = (0..n)
                .filter(|&r| !times.get(sym[r].as_str()).map(|t| t.contains(&ts[r])).unwrap_or(false))
                .count();
            rep.checks.push(check("V3", "every row is a real base bar (no pre-listing row, no synthetic tail)", bad, ""));
        }
        Some(Err(e)) => rep.checks.push(check("V3", "every row is a real base bar", 1, e.clone())),
        None => rep.checks.push(skipped("V3", "every row is a real base bar", "needs --base")),
    }

    // V4
    let mut v4 = 0usize;
    for r in 0..n {
        let ok = low[r] > 0.0
            && low[r] <= open[r].min(close[r])
            && high[r] >= open[r].max(close[r])
            && high[r] >= low[r]
            && volume[r] >= 0.0
            && trades[r] >= 0.0
            && tb[r] <= volume[r] * (1.0 + 1e-9)
            && tq[r] <= qv[r] * (1.0 + 1e-9)
            && (volume[r] <= 0.0 || (vwap[r] >= low[r] * (1.0 - 1e-9) && vwap[r] <= high[r] * (1.0 + 1e-9)));
        if !ok {
            v4 += 1;
        }
    }
    rep.checks.push(check("V4", "OHLC and flow invariants", v4, ""));

    // V5
    let v5 = (0..n)
        .filter(|&r| {
            let partial = flags[r] as u32 & FLAG_PARTIAL_BUCKET != 0;
            (cov[r] != 4.0) != partial
        })
        .count();
    rep.checks.push(check("V5", "cov15m == 4 unless PARTIAL_BUCKET", v5, ""));

    // V6
    let mut v6 = 0usize;
    let mut prev: BTreeMap<&str, i64> = BTreeMap::new();
    for r in 0..n {
        if let Some(p) = prev.insert(sym[r].as_str(), ts[r]) {
            let d = (ts[r] - p) / OUT_STEP - 1;
            if gap[r] != 0.0 && gap[r] != d as f64 {
                v6 += 1;
            }
            if d == 0 && gap[r] != 0.0 {
                v6 += 1;
            }
        } else if gap[r] != 0.0 {
            v6 += 1;
        }
    }
    rep.checks.push(check("V6", "gap matches recomputation", v6, ""));

    // V7
    match (&base_series, &opts.membership) {
        (Some(Ok(series)), Some(mp)) if opts.replay > 0 => match panel::read_membership(mp) {
            Ok(m) => {
                let mut bad = 0usize;
                let distinct: Vec<i64> = ts.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
                for k in 1..=opts.replay {
                    let cut = distinct[(distinct.len() - 1) * k / (opts.replay + 1)];
                    let asm = assemble(series, &m, &opts.panel_options, Some(cut));
                    let upto = ts.partition_point(|t| *t <= cut);
                    if asm.rows.len() != upto {
                        bad += 1;
                        continue;
                    }
                    for (i, r) in asm.rows.iter().enumerate() {
                        let b = &r.row.bar;
                        let same = b.time == ts[i]
                            && r.row.symbol == sym[i]
                            && b.open.to_bits() == open[i].to_bits()
                            && b.close.to_bits() == close[i].to_bits()
                            && b.volume.to_bits() == volume[i].to_bits()
                            && r.row.gap_buckets as f64 == gap[i]
                            && r.row.as_of as f64 == asof[i]
                            && r.cov15m as f64 == cov[i];
                        if !same {
                            bad += 1;
                        }
                    }
                }
                rep.checks.push(check("V7", "prefix causality replay", bad, format!("{} cuts", opts.replay)));
            }
            Err(e) => rep.checks.push(check("V7", "prefix causality replay", 1, e)),
        },
        _ => rep.checks.push(skipped("V7", "prefix causality replay", "needs --base and --membership")),
    }

    // V8
    rep.checks.push(Check {
        id: "V8",
        name: "cross-sectional ranks",
        status: "not_applicable",
        violations: 0,
        detail: "the v2 panel carries no rank columns".to_string(),
    });

    // V9
    let mut v9 = 0usize;
    let mut prev: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for r in 0..n {
        if let Some((l, c)) = prev.insert(sym[r].as_str(), (listing[r], close[r])) {
            if l == listing[r] && c > 0.0 {
                let ratio = open[r] / c;
                if ratio > opts.panel_options.jump_factor || ratio < 1.0 / opts.panel_options.jump_factor {
                    v9 += 1;
                }
            }
        }
    }
    rep.checks.push(check("V9", "no undeclared level jump inside a listing", v9, ""));

    // V10
    let manifest_path = format!("{}.manifest.json", opts.panel);
    match std::fs::read_to_string(&manifest_path) {
        Ok(t) => {
            let want = manifest::object(&t, "output").and_then(|o| manifest::string_field(o, "sha256"));
            let have = manifest::sha256_file(&opts.panel).ok();
            let bad = (want.is_none() || want != have) as usize;
            rep.checks.push(check("V10", "store SHA-256 matches the manifest", bad, want.unwrap_or_default()));
        }
        Err(_) => rep.checks.push(skipped("V10", "store SHA-256 matches the manifest", "no manifest beside the store")),
    }

    // V11
    let lb = Lockbox::crypto_2026h2();
    let v11 = ts.iter().filter(|t| lb.contains(**t)).count();
    rep.checks.push(check("V11", "zero lockbox rows", v11, ""));

    // V12
    match &opts.membership {
        Some(mp) => match panel::read_membership(mp) {
            Ok(m) => {
                let index = panel::membership_index(&m);
                let bad = (0..n)
                    .filter(|&r| {
                        let Some(v) = index.get(sym[r].as_str()) else { return true };
                        let pos = v.partition_point(|d| d.as_of <= ts[r]);
                        match pos.checked_sub(1).map(|i| v[i]) {
                            Some(d) => !(d.member && d.as_of as f64 == asof[r]),
                            None => true,
                        }
                    })
                    .count();
                let mut rows_per_year: BTreeMap<i32, usize> = BTreeMap::new();
                for t in &ts {
                    *rows_per_year.entry(ohlcv::civil_from_days(t.div_euclid(86_400)).0).or_insert(0) += 1;
                }
                let mut extra = 0usize;
                if let Ok(t) = std::fs::read_to_string(&manifest_path) {
                    for (y, c) in &rows_per_year {
                        let needle = format!("{{\"year\":{},\"rows\":{}}}", y, c);
                        if !t.contains(&needle) {
                            extra += 1;
                        }
                    }
                }
                rep.checks.push(check("V12", "rows match the membership ledger and per-year counts", bad + extra, ""));
            }
            Err(e) => rep.checks.push(check("V12", "rows match the membership ledger", 1, e)),
        },
        None => rep.checks.push(skipped("V12", "rows match the membership ledger", "needs --membership")),
    }
    rep
}

// ---------------------------------------------------------------------------
// CLI glue
// ---------------------------------------------------------------------------

/// True when `path` is a store with the v2 base columns.
pub fn is_base_store(path: &str) -> bool {
    store::is_store_path(path)
        && store::FileStore::open(path)
            .map(|s| s.header().names().iter().any(|n| n == "cov15m"))
            .unwrap_or(false)
}

fn cli_fail(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(3);
}

fn cli_usage(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(2);
}

/// `rcsr panel snapshot | base | build | validate` for the v2 panel.
pub fn cli(cmd: &str, args: &[String]) -> ! {
    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut data: Option<String> = None;
    let mut membership: Option<String> = None;
    let mut snapshot: Option<String> = None;
    let mut base: Option<String> = None;
    let mut jobs = 1usize;
    let mut keep_partial = false;
    let mut allow_lockbox = false;
    let mut resume: Option<String> = None;
    let mut group_rows = store::DEFAULT_GROUP_ROWS;
    let mut interval = "15m".to_string();
    let mut relists = Vec::new();
    let mut listing_gap_days = 30i64;
    let mut jump_factor = 5.0f64;
    let mut outage_share = 0.5f64;
    let mut replay = 3usize;
    let mut gpu: Option<GpuMode> = None;
    let mut i = 0usize;
    while i < args.len() {
        let num = |s: String, name: &str| -> f64 {
            s.parse::<f64>().unwrap_or_else(|_| cli_usage(&format!("{}: '{}' is not a number", name, s)))
        };
        match args[i].as_str() {
            "--input" => input = Some(value(args, &mut i)),
            "--out" => out = Some(value(args, &mut i)),
            "--data" => data = Some(value(args, &mut i)),
            "--membership" => membership = Some(value(args, &mut i)),
            "--snapshot" => snapshot = Some(value(args, &mut i)),
            "--base" => base = Some(value(args, &mut i)),
            "--resume" => resume = Some(value(args, &mut i)),
            "--interval" => interval = value(args, &mut i),
            "--jobs" => jobs = num(value(args, &mut i), "--jobs") as usize,
            "--group-rows" => group_rows = num(value(args, &mut i), "--group-rows") as usize,
            "--listing-gap-days" => listing_gap_days = num(value(args, &mut i), "--listing-gap-days") as i64,
            "--jump-factor" => jump_factor = num(value(args, &mut i), "--jump-factor"),
            "--outage-share" => outage_share = num(value(args, &mut i), "--outage-share"),
            "--replay" => replay = num(value(args, &mut i), "--replay") as usize,
            "--relist" => match parse_relists(&value(args, &mut i)) {
                Ok(r) => relists.extend(r),
                Err(e) => cli_usage(&e),
            },
            "--bar-seconds" => {
                if value(args, &mut i).trim() != "3600" {
                    cli_usage("the v2 panel is hourly: --bar-seconds must be 3600");
                }
            }
            "--gpu" => {
                let mode = match args.get(i + 1).map(|s| s.as_str()) {
                    Some(m @ ("auto" | "on" | "off" | "cpu" | "force")) => {
                        i += 1;
                        GpuMode::parse(m).unwrap_or_else(|e| cli_usage(&e))
                    }
                    _ => GpuMode::On,
                };
                gpu = Some(mode);
            }
            "--no-gpu" | "--cpu" => gpu = Some(GpuMode::Off),
            "--keep-partial" => keep_partial = true,
            "--require-full-bucket" => keep_partial = false,
            "--allow-lockbox" => allow_lockbox = true,
            other => cli_usage(&format!("unknown option '{}' for `rcsr panel {}` (hourly v2)", other, cmd)),
        }
        i += 1;
    }
    if gpu.is_some() && cmd != "base" {
        cli_usage(&format!("--gpu applies to `rcsr panel base`, not `{}`", cmd));
    }
    let mut po = |membership: &str, out: &str| -> PanelOptions {
        let mut p = PanelOptions::new(data.as_deref().unwrap_or(""), membership, out);
        p.group_rows = group_rows;
        p.allow_lockbox = allow_lockbox;
        p.snapshot = snapshot.clone();
        p.listing_gap_days = listing_gap_days;
        p.relists = std::mem::take(&mut relists);
        p.jump_factor = jump_factor;
        p.outage_share = outage_share;
        p
    };
    match cmd {
        "snapshot" => {
            let input = input.unwrap_or_else(|| cli_usage("snapshot needs --input <archive dir>"));
            let out = out.unwrap_or_else(|| "snapshot.json".to_string());
            let text = snapshot_text(&input, jobs).unwrap_or_else(|e| cli_fail(&e));
            if let Err(e) = std::fs::write(&out, &text) {
                cli_fail(&format!("{}: {}", out, e));
            }
            eprintln!("panel snapshot: {} bytes -> {}", text.len(), out);
        }
        "base" => {
            let input = input.unwrap_or_else(|| cli_usage("base needs --input <archive dir>"));
            let out = out.unwrap_or_else(|| "base_h1.rcsb".to_string());
            let mut o = BaseOptions::new(&input, &out);
            o.keep_partial = keep_partial;
            o.jobs = jobs;
            o.allow_lockbox = allow_lockbox;
            o.resume = resume;
            o.group_rows = group_rows;
            o.gpu = gpu.unwrap_or(GpuMode::Auto);
            o.interval_seconds = match interval.as_str() {
                "15m" => IN_STEP,
                "1m" => 60,
                other => cli_usage(&format!("unknown --interval '{}' (15m or 1m)", other)),
            };
            match build_base(&o) {
                Ok(s) => {
                    eprintln!(
                        "panel base: {} rows, {} symbols, {} cached -> {} (manifest {})",
                        s.rows, s.symbols, s.reused, out, s.manifest
                    );
                    let r = &s.route;
                    if r.requested != "off" {
                        eprintln!(
                            "panel base: gpu={} device={} cpu={} symbols, {} passes, kernel {:.3}s{}",
                            r.requested,
                            r.symbols_device,
                            r.symbols_cpu,
                            r.passes,
                            r.kernel_s,
                            r.fallbacks
                                .iter()
                                .map(|(why, n)| format!("; cpu fallback ({}): {} symbols", why, n))
                                .collect::<String>()
                        );
                    }
                }
                Err(e) => cli_fail(&e),
            }
        }
        "build" => {
            if resume.is_some() {
                cli_usage("--resume is a `panel base` option in the v2 pipeline");
            }
            let m = membership.unwrap_or_else(|| "membership.csv".to_string());
            let out = out.unwrap_or_else(|| "panel_hourly_default.rcsb".to_string());
            let p = po(&m, &out);
            match build_panel(&p) {
                Ok(s) => eprintln!(
                    "panel build: {} rows, {} symbols, {} delisted, {} gaps, {} listings -> {} (manifest {})",
                    s.rows, s.symbols, s.delisted, s.gaps, s.listings, out, s.manifest
                ),
                Err(e) => cli_fail(&e),
            }
        }
        "validate" => {
            let panel_path = data.clone().unwrap_or_else(|| cli_usage("validate needs --data <panel.rcsb>"));
            let m = membership.clone();
            let mut p = po(m.as_deref().unwrap_or(""), "");
            p.base = base.clone().unwrap_or_default();
            let report = validate(&ValidateOptions {
                panel: panel_path,
                base,
                membership: m,
                replay,
                panel_options: p,
            });
            println!("{}", report.json());
            if !report.pass() {
                std::process::exit(3);
            }
        }
        other => cli_usage(&format!("unknown panel command '{}'", other)),
    }
    std::process::exit(0);
}

fn value(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| cli_usage(&format!("{} requires a value", args[*i - 1])))
}

/// A level jump at `at` is declared when it follows a gap longer than
/// `gap_days` (a new listing) or crosses a declared relist instant.
pub fn jump_declared(prev: Option<i64>, at: i64, relists: &[i64], gap_days: i64) -> bool {
    match prev {
        Some(p) => at - p > gap_days * 86_400 || relists.iter().any(|&r| p < r && r <= at),
        None => true,
    }
}
