//! Point-in-time panel construction and honest backtest primitives (#145).
//!
//! The market-data audit found three structural flaws in the old panel:
//!
//! 1. the universe was selected from **2026** volume, so every symbol was a
//!    survivor and no delisted coin could enter;
//! 2. level/EWM features bridged listing, delisting and migration gaps;
//! 3. costs were fees-only, buy-and-hold was free and the final exit was not
//!    charged.
//!
//! This module implements the rebuild:
//!
//! * [`build_membership`] ranks by trailing liquidity **as of** each rebalance
//!   and never reads a later bar; a symbol that later delists stays in the
//!   sample while it is alive;
//! * [`build_panel`] keeps only real bars, never forward-fills across a
//!   listing/delisting/migration boundary, and reports per-year symbol counts;
//! * verified splices use [`crate::ohlcv::splice`], so a wrong `k` is a hard
//!   error and the LUNA-style level jump is reported as a canary;
//! * [`CostModel`], [`net_returns`], [`buy_and_hold`] and
//!   [`matched_exposure_random_long`] charge both sides and the final exit;
//! * [`crate::lockbox`] refuses 2026H2 for every runner;
//! * [`provenance_json`] records snapshots, membership as-of dates, checksums,
//!   per-year counts, delisted names and gaps.

use crate::data::{Column, DictColumn, Table};
use crate::lockbox::Lockbox;
use crate::ohlcv::{self, Bar, Migration};
use crate::{manifest, store};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// One point-in-time membership decision.
#[derive(Clone, Debug, PartialEq)]
pub struct MembershipRow {
    /// Rebalance instant (UTC seconds, 00:00).
    pub as_of: i64,
    pub symbol: String,
    /// Selected for this period.
    pub member: bool,
    /// Trailing average daily quote volume, as of `as_of`.
    pub adv_usd: f64,
    /// Bars available in the trailing lookback.
    pub n_bars: usize,
}

#[derive(Clone, Debug)]
pub struct MembershipConfig {
    /// Days between rebalances (30 = monthly, 90 = quarterly).
    pub rebalance_days: i64,
    /// Trailing liquidity window.
    pub lookback_days: i64,
    /// Minimum age before a symbol may be selected.
    pub min_history_days: i64,
    /// Minimum bars in the lookback window.
    pub min_bars: usize,
    /// Maximum names selected per period.
    pub top: usize,
    /// Minimum trailing ADV in quote currency (0 = no floor).
    pub min_adv_usd: f64,
    /// First rebalance instant (inclusive). `None` starts at the first bar.
    pub start: Option<i64>,
    /// Last rebalance instant (exclusive). `None` runs to the last bar.
    pub end: Option<i64>,
}

impl Default for MembershipConfig {
    fn default() -> Self {
        MembershipConfig {
            rebalance_days: 30,
            lookback_days: 90,
            min_history_days: 90,
            min_bars: 60,
            top: 100,
            min_adv_usd: 0.0,
            start: None,
            end: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Membership {
    pub rows: Vec<MembershipRow>,
    pub as_of: Vec<i64>,
    /// Symbols with at least one selected period.
    pub selected: BTreeSet<String>,
}

impl Membership {
    /// The latest decision for `symbol` at or before `ts`.
    pub fn at(&self, symbol: &str, ts: i64) -> Option<&MembershipRow> {
        self.rows
            .iter()
            .filter(|r| r.symbol == symbol && r.as_of <= ts)
            .max_by_key(|r| r.as_of)
    }

    pub fn is_member(&self, symbol: &str, ts: i64) -> bool {
        self.at(symbol, ts).map(|r| r.member).unwrap_or(false)
    }

    /// Rebalance instants, ascending.
    pub fn rebalances(&self) -> &[i64] {
        &self.as_of
    }
}

/// A symbol/bar pair before the panel table is built.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelRow {
    pub symbol: String,
    pub bar: Bar,
    /// Membership decision in force for this row.
    pub as_of: i64,
    /// Missing buckets between this bar and the previous bar **of the same
    /// symbol** (0 = contiguous). New listings and migrations reset the chain.
    pub gap_buckets: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Panel {
    pub rows: Vec<PanelRow>,
    /// Symbols actually present in the rebuilt panel.
    pub symbols: BTreeSet<String>,
    /// Symbols that stop before the panel's last date.
    pub delisted: BTreeSet<String>,
    /// Total missing daily buckets inside continued series.
    pub gaps: usize,
    pub first_ts: i64,
    pub last_ts: i64,
}

impl Panel {
    /// Restrict the panel to `[start, end)` and recompute the summary fields
    /// that a manifest reports. Used by `panel build --start/--end`, so an
    /// explicit evaluated range is both filtered and checked against the
    /// lockbox.
    pub fn retain_range(&mut self, start: Option<i64>, end: Option<i64>) {
        if start.is_none() && end.is_none() {
            return;
        }
        self.rows.retain(|r| {
            start.map_or(true, |s| r.bar.time >= s) && end.map_or(true, |e| r.bar.time < e)
        });
        // Rows are ordered by time, so the last write per symbol is its latest
        // bar; borrowing the symbol avoids 4.7M string clones on a big panel.
        let mut last: BTreeMap<&str, i64> = BTreeMap::new();
        let mut first_ts = i64::MAX;
        let mut last_ts = i64::MIN;
        let mut gaps = 0usize;
        for r in &self.rows {
            first_ts = first_ts.min(r.bar.time);
            last_ts = last_ts.max(r.bar.time);
            gaps += r.gap_buckets;
            last.insert(r.symbol.as_str(), r.bar.time);
        }
        self.symbols = last.keys().map(|s| s.to_string()).collect();
        self.first_ts = if self.rows.is_empty() { 0 } else { first_ts };
        self.last_ts = if self.rows.is_empty() { 0 } else { last_ts };
        self.gaps = gaps;
        let end = self.last_ts;
        self.delisted = last
            .into_iter()
            .filter(|(_, t)| *t < end - 30 * 86_400)
            .map(|(s, _)| s.to_string())
            .collect();
    }

    /// Symbol count per UTC year.
    pub fn counts_per_year(&self) -> BTreeMap<i32, (usize, usize)> {
        let mut rows: BTreeMap<i32, BTreeSet<&str>> = BTreeMap::new();
        for r in &self.rows {
            let (y, _, _) = ohlcv::civil_from_days(r.bar.time.div_euclid(86_400));
            rows.entry(y).or_default().insert(r.symbol.as_str());
        }
        rows.into_iter()
            .map(|(y, s)| (y, (s.len(), s.len())))
            .collect()
    }

    pub fn rows_per_year(&self) -> BTreeMap<i32, usize> {
        let mut out = BTreeMap::new();
        for r in &self.rows {
            let (y, _, _) = ohlcv::civil_from_days(r.bar.time.div_euclid(86_400));
            *out.entry(y).or_insert(0) += 1;
        }
        out
    }
}

/// First day of the UTC month containing `ts`.
fn month_start(ts: i64) -> i64 {
    let (y, m, _) = ohlcv::civil_from_days(ts.div_euclid(86_400));
    ohlcv::days_from_civil(y, m, 1) * 86_400
}

/// Build point-in-time membership. No decision reads a bar at or after its
/// `as_of`; a trailing ADV spike after the rebalance cannot change the result.
pub fn build_membership(
    series: &BTreeMap<String, Vec<Bar>>,
    cfg: &MembershipConfig,
) -> Result<Membership, String> {
    if cfg.top == 0 || cfg.lookback_days <= 0 || cfg.rebalance_days <= 0 {
        return Err("membership top/lookback/rebalance must be positive".to_string());
    }
    let mut all_times: Vec<i64> = series
        .values()
        .filter_map(|v| v.first().map(|b| b.time))
        .collect();
    let last_ts = series
        .values()
        .filter_map(|v| v.last().map(|b| b.time))
        .max()
        .unwrap_or(0);
    if all_times.is_empty() {
        return Err("membership: no bars".to_string());
    }
    all_times.sort_unstable();
    // #145: a PIT rebuild may be scoped to an evaluation span (2021-01 ..
    // 2026-07) while still reading the full trailing lookback before it.
    let first = cfg.start.map(|s| month_start(s)).unwrap_or_else(|| month_start(all_times[0]));
    let stop = cfg.end.unwrap_or(last_ts + 1);
    if stop <= first {
        return Err("membership: --end must be after --start".to_string());
    }
    let mut as_of = first;
    let mut decisions = Vec::new();
    let mut as_ofs = Vec::new();
    while as_of < stop && as_of <= last_ts {
        as_ofs.push(as_of);
        let lookback_start = as_of - cfg.lookback_days * 86_400;
        let mut candidates: Vec<(String, f64, usize)> = Vec::new();
        for (symbol, bars) in series {
            // Strictly before the decision instant.
            let first = bars.first().map(|b| b.time).unwrap_or(i64::MAX);
            if first > as_of - cfg.min_history_days * 86_400 {
                continue;
            }
            let mut sum = 0.0;
            let mut n = 0usize;
            for b in bars.iter().rev() {
                if b.time >= as_of {
                    continue;
                }
                if b.time < lookback_start {
                    break;
                }
                sum += if b.quote_volume > 0.0 {
                    b.quote_volume
                } else {
                    b.close * b.volume
                };
                n += 1;
            }
            if n < cfg.min_bars {
                continue;
            }
            let adv = sum / (cfg.lookback_days as f64);
            if adv < cfg.min_adv_usd {
                continue;
            }
            candidates.push((symbol.clone(), adv, n));
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let keep: BTreeSet<String> = candidates
            .iter()
            .take(cfg.top)
            .map(|(s, _, _)| s.clone())
            .collect();
        for (symbol, adv, n) in candidates {
            decisions.push(MembershipRow {
                as_of,
                symbol: symbol.clone(),
                member: keep.contains(&symbol),
                adv_usd: adv,
                n_bars: n,
            });
        }
        as_of += cfg.rebalance_days * 86_400;
    }
    let selected = decisions
        .iter()
        .filter(|r| r.member)
        .map(|r| r.symbol.clone())
        .collect();
    Ok(Membership {
        rows: decisions,
        as_of: as_ofs,
        selected,
    })
}

/// Rebuild a panel from point-in-time membership.
///
/// Bars are kept exactly as traded: a delisted symbol stops at its last bar, a
/// new listing starts at its first bar, and no row is manufactured across a
/// gap. `gap_buckets` records missing days inside a continued series.
pub fn build_panel(
    series: &BTreeMap<String, Vec<Bar>>,
    membership: &Membership,
) -> Result<Panel, String> {
    build_panel_with_step(series, membership, 86_400)
}

/// [`build_panel`] for a bar spacing other than one day (for example 3600 for
/// an hourly panel). `gap_buckets` counts missing buckets of this size.
pub fn build_panel_with_step(
    series: &BTreeMap<String, Vec<Bar>>,
    membership: &Membership,
    step_seconds: i64,
) -> Result<Panel, String> {
    let step = step_seconds.max(1);
    let index = membership_index(membership);
    let mut rows = Vec::new();
    for (symbol, bars) in series {
        rows.extend(symbol_panel_rows(
            symbol,
            bars,
            index.get(symbol.as_str()).map(|v| v.as_slice()),
            step,
        ));
    }
    assemble_panel(rows, series)
}

pub(crate) type MemberIndex<'a> = BTreeMap<&'a str, Vec<&'a MembershipRow>>;

pub(crate) fn membership_index(membership: &Membership) -> MemberIndex<'_> {
    // One sorted index per symbol: the naive `at()` scan is O(decisions) per
    // bar and turns a 665k-row build into minutes.
    let mut index: MemberIndex<'_> = BTreeMap::new();
    for r in &membership.rows {
        index.entry(r.symbol.as_str()).or_default().push(r);
    }
    for v in index.values_mut() {
        v.sort_by_key(|r| r.as_of);
    }
    index
}

pub(crate) fn symbol_panel_rows(
    symbol: &str,
    bars: &[Bar],
    decisions: Option<&[&MembershipRow]>,
    step: i64,
) -> Vec<PanelRow> {
    let mut out = Vec::new();
    let mut prev: Option<i64> = None;
    for b in bars {
        let decision = decisions.and_then(|v| {
            let pos = v.partition_point(|r| r.as_of <= b.time);
            pos.checked_sub(1).map(|i| v[i])
        });
        let Some(decision) = decision else {
            prev = None;
            continue;
        };
        if !decision.member {
            // A membership exclusion starts a new chain: it is not a data gap
            // and must not be counted as one.
            prev = None;
            continue;
        }
        let missing = match prev {
            Some(p) if b.time > p => ((b.time - p) / step - 1).max(0) as usize,
            _ => 0,
        };
        prev = Some(b.time);
        out.push(PanelRow {
            symbol: symbol.to_string(),
            bar: *b,
            as_of: decision.as_of,
            gap_buckets: missing,
        });
    }
    out
}

fn assemble_panel(
    rows: Vec<PanelRow>,
    series: &BTreeMap<String, Vec<Bar>>,
) -> Result<Panel, String> {
    let last: BTreeMap<String, i64> = series
        .iter()
        .filter_map(|(s, v)| v.last().map(|b| (s.clone(), b.time)))
        .collect();
    assemble_panel_with_last(rows, &last)
}

/// Assemble rows when the end of each symbol's input series is already known.
/// A warm resumed build never reads the series, so it takes them from the part
/// headers; the delisted set is therefore identical to the cold build's.
pub(crate) fn assemble_panel_with_last(
    mut rows: Vec<PanelRow>,
    last: &BTreeMap<String, i64>,
) -> Result<Panel, String> {
    if rows.is_empty() {
        return Err("panel: membership selects no rows".to_string());
    }
    rows.sort_unstable_by(|a, b| a.bar.time.cmp(&b.bar.time).then_with(|| a.symbol.cmp(&b.symbol)));
    let symbols: BTreeSet<String> = rows.iter().map(|r| r.symbol.clone()).collect();
    let first_ts = rows.iter().map(|r| r.bar.time).min().unwrap_or(0);
    let last_ts = rows.iter().map(|r| r.bar.time).max().unwrap_or(0);
    let gaps = rows.iter().map(|r| r.gap_buckets).sum();
    // A symbol whose last actual bar is well before the panel end is delisted,
    // not forward-filled.
    let delisted: BTreeSet<String> = symbols
        .iter()
        .filter(|s| last.get(*s).map(|t| *t < last_ts - 30 * 86_400).unwrap_or(true))
        .cloned()
        .collect();
    Ok(Panel {
        rows,
        symbols,
        delisted,
        gaps,
        first_ts,
        last_ts,
    })
}

fn part_name(symbol: &str) -> String {
    let safe: String = symbol
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    format!("{}.part", safe)
}

// ---------------------------------------------------------------------------
// Binary panel parts (issue #145, warm-rebuild budget)
// ---------------------------------------------------------------------------
//
// A text part costs more to re-read than the whole 1.5 GB prep store costs to
// parse, because every row is re-parsed from a 14-field line. A part is a
// fixed-size binary record instead: reading one is `fs::read` plus a copy, with
// no parsing and no float-formatting dependence.

const PANEL_CACHE_VERSION: &str = "rcsr-panel-cache-2";
const MEMBERSHIP_CACHE_VERSION: &str = "rcsr-membership-cache-1";
const PART_MAGIC: &[u8; 8] = b"RCSRPART";
const PART_VERSION: u32 = 2;
/// magic 8 + version 4 + step 8 + series_last 8 + rows 8 + checksum 8.
const PART_HEADER: usize = 44;
/// ts 8 + as_of 8 + gap 4 + flags 4 + nine f64.
const PART_RECORD: usize = 96;

/// FNV-1a over the payload: cheap, and enough to catch a truncated, edited or
/// half-written part. The store SHA-256 in `state.txt` pins the inputs.
fn part_checksum(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn write_symbol_part(
    path: &std::path::Path,
    rows: &[PanelRow],
    series_last: i64,
    step: i64,
) -> Result<u64, String> {
    let mut buf = Vec::with_capacity(PART_HEADER + rows.len() * PART_RECORD);
    buf.extend_from_slice(PART_MAGIC);
    buf.extend_from_slice(&PART_VERSION.to_le_bytes());
    buf.extend_from_slice(&step.to_le_bytes());
    buf.extend_from_slice(&series_last.to_le_bytes());
    buf.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    for r in rows {
        let b = r.bar;
        buf.extend_from_slice(&b.time.to_le_bytes());
        buf.extend_from_slice(&r.as_of.to_le_bytes());
        buf.extend_from_slice(&(r.gap_buckets.min(u32::MAX as usize) as u32).to_le_bytes());
        let flags = (b.complete as u32) | ((b.traded as u32) << 1);
        buf.extend_from_slice(&flags.to_le_bytes());
        for v in [
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.quote_volume,
            b.trades,
            b.taker_buy_base,
            b.taker_buy_quote,
        ] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    debug_assert_eq!(buf.len(), PART_HEADER + rows.len() * PART_RECORD);
    let sum = part_checksum(&buf[PART_HEADER..]);
    buf[36..44].copy_from_slice(&sum.to_le_bytes());
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, &buf).map_err(|e| format!("{}: {}", tmp.display(), e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))?;
    Ok(sum)
}

struct PartHeader {
    step: i64,
    series_last: i64,
    rows: usize,
    checksum: u64,
}

fn parse_part_header(path: &std::path::Path, bytes: &[u8]) -> Result<PartHeader, String> {
    if bytes.len() < PART_HEADER {
        return Err(format!("{}: truncated panel part", path.display()));
    }
    if &bytes[0..8] != PART_MAGIC {
        return Err(format!(
            "{}: not a binary panel part; use a new --resume directory",
            path.display()
        ));
    }
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let i64_at = |o: usize| i64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    if u32_at(8) != PART_VERSION {
        return Err(format!(
            "{}: panel part version {} (expected {})",
            path.display(),
            u32_at(8),
            PART_VERSION
        ));
    }
    let rows = u64_at(28) as usize;
    let want = PART_HEADER + rows * PART_RECORD;
    if bytes.len() != want {
        return Err(format!(
            "{}: {} bytes, expected {} for {} rows",
            path.display(),
            bytes.len(),
            want,
            rows
        ));
    }
    let checksum = u64_at(36);
    if part_checksum(&bytes[PART_HEADER..]) != checksum {
        return Err(format!("{}: panel part checksum mismatch", path.display()));
    }
    Ok(PartHeader {
        step: i64_at(12),
        series_last: i64_at(20),
        rows,
        checksum,
    })
}

/// Read one part: `(series end, checksum, rows)`.
fn read_symbol_part(
    path: &std::path::Path,
    symbol: &str,
    step: i64,
) -> Result<(i64, u64, Vec<PanelRow>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let h = parse_part_header(path, &bytes)?;
    if h.step != step {
        return Err(format!(
            "{}: part was built with --bar-seconds {}, not {}",
            path.display(),
            h.step,
            step
        ));
    }
    let mut out = Vec::with_capacity(h.rows);
    for rec in bytes[PART_HEADER..].chunks_exact(PART_RECORD) {
        let i64_at = |o: usize| i64::from_le_bytes(rec[o..o + 8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
        let f64_at = |o: usize| f64::from_le_bytes(rec[o..o + 8].try_into().unwrap());
        let flags = u32_at(20);
        let mut bar =
            Bar::new(i64_at(0), f64_at(24), f64_at(32), f64_at(40), f64_at(48), f64_at(56));
        bar.quote_volume = f64_at(64);
        bar.trades = f64_at(72);
        bar.taker_buy_base = f64_at(80);
        bar.taker_buy_quote = f64_at(88);
        bar.complete = flags & 1 != 0;
        bar.traded = flags & 2 != 0;
        out.push(PanelRow {
            symbol: symbol.to_string(),
            bar,
            as_of: i64_at(8),
            gap_buckets: u32_at(16) as usize,
        });
    }
    Ok((h.series_last, h.checksum, out))
}

/// One cache entry: the symbol's last bar in the input series plus the part's
/// row count and checksum. A warm build uses it to detect a deleted, replaced
/// or truncated part without reading the 1.5 GB input store.
struct PartEntry {
    series_last: i64,
    rows: usize,
    checksum: u64,
}

fn index_path(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join("parts.txt")
}

/// Publish a small file atomically (`tmp` + rename).
pub(crate) fn publish_text(path: &std::path::Path, text: &str) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {}", tmp.display(), e))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
}

fn read_index(dir: &std::path::Path) -> Result<Vec<(String, PartEntry)>, String> {
    let path = index_path(dir);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let line_no = i + 1;
        if f.len() != 4 {
            return Err(format!(
                "{}:{}: expected symbol, series_last, rows, checksum",
                path.display(),
                line_no
            ));
        }
        let bad = |what: &str| format!("{}:{}: bad {}", path.display(), line_no, what);
        out.push((
            f[0].to_string(),
            PartEntry {
                series_last: f[1].parse().map_err(|_| bad("series_last"))?,
                rows: f[2].parse::<usize>().map_err(|_| bad("row count"))?,
                checksum: f[3].parse().map_err(|_| bad("checksum"))?,
            },
        ));
    }
    Ok(out)
}

fn write_index(dir: &std::path::Path, index: &[(String, PartEntry)]) -> Result<(), String> {
    let mut text = String::with_capacity(index.len() * 48 + 40);
    text.push_str("# symbol\tseries_last\trows\tchecksum\n");
    for (symbol, e) in index {
        let _ = writeln!(text, "{}\t{}\t{}\t{}", symbol, e.series_last, e.rows, e.checksum);
    }
    publish_text(&index_path(dir), &text)
}

/// What a cache directory is pinned to. `--jobs` is deliberately absent: it
/// changes nothing about the rows.
fn cache_state_text(step: i64, store_sha256: &str, membership_key: &str, settings: &str) -> String {
    format!(
        "version={}\nstep={}\nstore_sha256={}\nmembership_key={}\nsettings={}\n",
        PANEL_CACHE_VERSION, step, store_sha256, membership_key, settings
    )
}

/// The SHA-256 of the input store, taken from the manifest that `rcsr prep`
/// published beside it when that still describes the file.
///
/// Hashing a 1.5 GB store costs ~9 s on the reference host and a warm pipeline
/// would pay it twice (membership and build), so the recorded checksum is
/// preferred. A missing manifest, a non-hex value or a byte count that disagrees
/// with the file falls back to hashing the store. An in-place edit that keeps
/// both the byte length and the manifest is not detected; delete the manifest to
/// force a hash.
pub fn store_key(path: &str) -> String {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(u64::MAX);
    let manifest_path = format!("{}.manifest.json", path);
    if let Ok(text) = std::fs::read_to_string(&manifest_path) {
        let sha = json_field(&text, "output_sha256");
        let recorded = json_field(&text, "output_bytes").and_then(|v| v.parse::<u64>().ok());
        if let Some(sha) = sha {
            let hex = sha.len() == 64 && sha.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
            if hex && recorded == Some(bytes) {
                return sha;
            }
        }
    }
    manifest::sha256_file(path).unwrap_or_default()
}

/// Minimal `"key":value` extractor. The manifests are written by this crate and
/// the crate is std-only, so a scan of the first occurrence is enough.
fn json_field(text: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\":", key);
    let start = text.find(&pat)? + pat.len();
    let rest = text[start..].trim_start();
    if let Some(r) = rest.strip_prefix('"') {
        return Some(r[..r.find('"')?].to_string());
    }
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '-'))
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

/// The membership cache key: the input store checksum plus every ranking
/// setting that changes a decision.
pub fn membership_cache_key(store_sha256: &str, cfg: &MembershipConfig) -> String {
    format!(
        "version={}\nstore_sha256={}\nrebalance={}\nlookback={}\nmin_history={}\n\
         min_bars={}\ntop={}\nmin_adv={}\nstart={}\nend={}\n",
        MEMBERSHIP_CACHE_VERSION,
        store_sha256,
        cfg.rebalance_days,
        cfg.lookback_days,
        cfg.min_history_days,
        cfg.min_bars,
        cfg.top,
        cfg.min_adv_usd,
        cfg.start.unwrap_or(0),
        cfg.end.unwrap_or(0)
    )
}

/// A stable digest of the loaded membership **decisions** (not the CSV bytes),
/// so a cache stays valid when the same table is re-serialised.
pub fn membership_key(membership: &Membership) -> String {
    let mut text = String::with_capacity(membership.rows.len() * 48);
    for r in &membership.rows {
        let _ = writeln!(
            text,
            "{} {} {} {:.6} {}",
            r.as_of, r.symbol, r.member as u8, r.adv_usd, r.n_bars
        );
    }
    for as_of in &membership.as_of {
        let _ = writeln!(text, "@{}", as_of);
    }
    manifest::sha256_hex(text.as_bytes())
}

/// Is `state_dir` a warm cache for exactly these inputs?
///
/// * `Ok(true)`: every part is present and indexed, so the input need not be
///   read at all;
/// * `Ok(false)`: no usable cache yet (none, or one built for other inputs);
///   build one from the input.
pub fn panel_cache_ready(
    state_dir: &str,
    step_seconds: i64,
    store_sha256: &str,
    membership: &Membership,
    settings: &str,
) -> Result<bool, String> {
    let dir = std::path::Path::new(state_dir);
    let wanted = cache_state_text(
        step_seconds.max(1),
        store_sha256,
        &membership_key(membership),
        settings,
    );
    match std::fs::read_to_string(dir.join("state.txt")) {
        Ok(got) if got == wanted => Ok(index_path(dir).exists()),
        Ok(_) => Ok(false),
        Err(_) => Ok(false),
    }
}

/// Resumable panel build (issue #145). One part per symbol is written
/// atomically; a re-run trusts an existing part and skips that symbol. The
/// state file refuses a resumed directory built with a different step or
/// membership so a stale part can never enter a new panel.
pub fn build_panel_resumable(
    series: Option<&BTreeMap<String, Vec<Bar>>>,
    membership: &Membership,
    step_seconds: i64,
    state_dir: &str,
    store_sha256: &str,
    settings: &str,
) -> Result<Panel, String> {
    let step = step_seconds.max(1);
    std::fs::create_dir_all(state_dir).map_err(|e| format!("{}: {}", state_dir, e))?;
    let dir = std::path::Path::new(state_dir);
    let state_path = dir.join("state.txt");
    let wanted = cache_state_text(step, store_sha256, &membership_key(membership), settings);
    let reuse = match std::fs::read_to_string(&state_path) {
        Ok(got) if got == wanted => true,
        _ => {
            // The directory holds a cache for other inputs (or an older
            // format). With the input in hand the cache is rebuilt in place:
            // every part is dropped first, so no stale row can enter the new
            // panel. Without the input, a mismatch is refused rather than
            // guessed at.
            if series.is_none() {
                return Err(format!(
                    "{} was built with a different bar step, input checksum, membership or \
                     settings; remove it or use a new --resume directory",
                    state_dir
                ));
            }
            if let Ok(it) = std::fs::read_dir(dir) {
                for entry in it.filter_map(|e| e.ok()) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".part")
                        || name.ends_with(".tmp")
                        || name == "parts.txt"
                        || name == "state.txt"
                    {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
            eprintln!("panel build: cache holds other inputs or an older format; rebuilding it");
            publish_text(&state_path, &wanted)?;
            false
        }
    };
    let mut last: BTreeMap<String, i64> = BTreeMap::new();
    let mut rows: Vec<PanelRow> = Vec::new();
    match series {
        None => {
            // Warm path: the index pins the symbol set, so a deleted or edited
            // part is an error rather than a quietly smaller panel.
            if !reuse {
                return Err(format!(
                    "{} has no completed panel cache; run once with --data to build it",
                    state_dir
                ));
            }
            let index = read_index(dir)?;
            if index.is_empty() {
                return Err(format!("{}: panel part index is empty", state_dir));
            }
            for (symbol, entry) in &index {
                let path = dir.join(part_name(symbol));
                let (series_last, checksum, part) = read_symbol_part(&path, symbol, step)
                    .map_err(|e| format!("{} (remove the directory and rebuild)", e))?;
                if part.len() != entry.rows || checksum != entry.checksum {
                    return Err(format!(
                        "{} does not match the cache index; remove the directory and rebuild",
                        path.display()
                    ));
                }
                if series_last != entry.series_last {
                    return Err(format!(
                        "{} ends at {} but the index records {}; remove the directory and rebuild",
                        path.display(),
                        series_last,
                        entry.series_last
                    ));
                }
                last.insert(symbol.clone(), series_last);
                rows.extend(part);
            }
            eprintln!(
                "panel build: resume {}/{} symbols (100.0%), {} part(s) reused, 0 parsed",
                index.len(),
                index.len(),
                index.len()
            );
        }
        Some(series) => {
            let index = membership_index(membership);
            let total = series.len();
            let mut reused = 0usize;
            let mut new_index: Vec<(String, PartEntry)> = Vec::with_capacity(total);
            for (done, (symbol, bars)) in series.iter().enumerate() {
                let path = dir.join(part_name(symbol));
                let series_last = bars.last().map(|b| b.time).unwrap_or(0);
                // A part is reused only when its recorded series end still
                // matches the input, so a lengthened series is rebuilt.
                let cached = if reuse && path.exists() {
                    match read_symbol_part(&path, symbol, step) {
                        Ok((part_last, checksum, part)) if part_last == series_last => {
                            Some((checksum, part))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                let (checksum, part) = match cached {
                    Some((checksum, part)) => {
                        reused += 1;
                        (checksum, part)
                    }
                    None => {
                        let part = symbol_panel_rows(
                            symbol,
                            bars,
                            index.get(symbol.as_str()).map(|v| v.as_slice()),
                            step,
                        );
                        let checksum = write_symbol_part(&path, &part, series_last, step)?;
                        (checksum, part)
                    }
                };
                last.insert(symbol.clone(), series_last);
                new_index.push((
                    symbol.clone(),
                    PartEntry { series_last, rows: part.len(), checksum },
                ));
                rows.extend(part);
                if (done + 1) % 25 == 0 || done + 1 == total {
                    eprintln!(
                        "panel build: resume {}/{} symbols ({:.1}%), {} part(s) reused",
                        done + 1,
                        total,
                        100.0 * (done + 1) as f64 / total.max(1) as f64,
                        reused
                    );
                }
            }
            write_index(dir, &new_index)?;
        }
    }
    assemble_panel_with_last(rows, &last)
}

/// A canonical panel table: `ts, symbol, open..taker_buy_quote, gap, member_asof`.
pub fn panel_table(panel: &Panel) -> Table {
    let names = [
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
        "gap",
        "member_asof",
    ];
    let n = panel.rows.len();
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n); names.len()];
    let mut times = Vec::with_capacity(n);
    let mut sym_codes = Vec::with_capacity(n);
    let mut sym_levels: Vec<String> = Vec::new();
    let mut level_of: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for r in &panel.rows {
        times.push(r.bar.time);
        cols[0].push(r.bar.time as f64);
        // Level codes are assigned in first-appearance order; a hash lookup
        // keeps a 4.7M-row panel linear instead of scanning 445 levels per row.
        let code = match level_of.get(r.symbol.as_str()) {
            Some(c) => *c,
            None => {
                sym_levels.push(r.symbol.clone());
                let c = (sym_levels.len() - 1) as u32;
                level_of.insert(r.symbol.as_str(), c);
                c
            }
        };
        sym_codes.push(code);
        cols[2].push(r.bar.open);
        cols[3].push(r.bar.high);
        cols[4].push(r.bar.low);
        cols[5].push(r.bar.close);
        cols[6].push(r.bar.volume);
        cols[7].push(r.bar.quote_volume);
        cols[8].push(r.bar.trades);
        cols[9].push(r.bar.taker_buy_base);
        cols[10].push(r.bar.taker_buy_quote);
        cols[11].push(r.gap_buckets as f64);
        cols[12].push(r.as_of as f64);
    }
    let mut columns = Vec::with_capacity(names.len());
    let mut numeric = Vec::with_capacity(names.len());
    for (j, name) in names.iter().enumerate() {
        if j == 1 {
            numeric.push(false);
            columns.push(Column {
                values: Vec::new(),
                ints: None,
                dict: Some(DictColumn {
                    codes: sym_codes.clone(),
                    levels: sym_levels.clone(),
                }),
            });
        } else {
            numeric.push(true);
            let ints = if j == 0 { Some(times.clone()) } else { None };
            columns.push(Column {
                values: std::mem::take(&mut cols[j]),
                ints,
                dict: None,
            });
        }
        let _ = name;
    }
    Table::from_parts(
        names.iter().map(|s| s.to_string()).collect(),
        numeric,
        columns,
        n,
        0,
    )
}

/// The build-window guard shared by the CLI and tests. A full build (no
/// explicit start/end) is provenance and is not checked; an explicit range is.
pub fn check_build_window(
    first_ts: i64,
    last_ts: i64,
    step_seconds: i64,
    allow_lockbox: bool,
) -> Result<(), String> {
    crate::lockbox::Guard::new(allow_lockbox, None).check(
        "panel build range",
        first_ts,
        last_ts.saturating_add(step_seconds.max(1)),
    )
}

/// Write a panel store plus a provenance manifest.
pub fn write_panel_panel(
    out: &str,
    panel: &Panel,
    membership: &Membership,
    inputs: &[&str],
    group_rows: usize,
    allow_lockbox: bool,
    snapshot: Option<&str>,
) -> Result<String, String> {
    let table = panel_table(panel);
    store::write_store_with_time(out, &table, group_rows, Some(0))?;
    let text = provenance_json(inputs, membership, panel, out, allow_lockbox, snapshot);
    let manifest_path = format!("{}.manifest.json", out);
    std::fs::write(&manifest_path, text).map_err(|e| format!("{}: {}", manifest_path, e))?;
    Ok(manifest_path)
}

/// [`write_panel_panel`] with the SHA-256 of `inputs[0]` already known. A warm
/// rebuild computes the store checksum for the cache key, and re-hashing 1.5 GB
/// only to write the manifest would undo part of the saving.
pub fn write_panel_panel_hashed(
    out: &str,
    panel: &Panel,
    membership: &Membership,
    inputs: &[&str],
    first_sha256: &str,
    group_rows: usize,
    allow_lockbox: bool,
    snapshot: Option<&str>,
) -> Result<String, String> {
    let table = panel_table(panel);
    store::write_store_with_time(out, &table, group_rows, Some(0))?;
    let mut known: BTreeMap<String, String> = BTreeMap::new();
    if let Some(first) = inputs.first() {
        known.insert((*first).to_string(), first_sha256.to_string());
    }
    let text = provenance_json_with(inputs, &known, membership, panel, out, allow_lockbox, snapshot);
    let manifest_path = format!("{}.manifest.json", out);
    std::fs::write(&manifest_path, text).map_err(|e| format!("{}: {}", manifest_path, e))?;
    Ok(manifest_path)
}

/// JSON provenance: inputs, as-of dates, counts, delisted names, gaps and
/// checksums.
pub fn provenance_json(
    inputs: &[&str],
    membership: &Membership,
    panel: &Panel,
    output: &str,
    allow_lockbox: bool,
    snapshot: Option<&str>,
) -> String {
    provenance_json_with(
        inputs,
        &BTreeMap::new(),
        membership,
        panel,
        output,
        allow_lockbox,
        snapshot,
    )
}

/// [`provenance_json`] with some input checksums already computed.
pub fn provenance_json_with(
    inputs: &[&str],
    known: &BTreeMap<String, String>,
    membership: &Membership,
    panel: &Panel,
    output: &str,
    allow_lockbox: bool,
    snapshot: Option<&str>,
) -> String {
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"tool\":\"rcsr panel\",");
    let _ = writeln!(s, "  \"inputs\":[");
    for (i, p) in inputs.iter().enumerate() {
        let sha = known
            .get(*p)
            .cloned()
            .or_else(|| manifest::sha256_file(p).ok())
            .unwrap_or_else(|| "unknown".to_string());
        let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let comma = if i + 1 == inputs.len() { "" } else { "," };
        let _ = writeln!(
            s,
            "    {{\"path\":{},\"sha256\":{},\"bytes\":{}}}{}",
            jstr(p),
            jstr(&sha),
            bytes,
            comma
        );
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(
        s,
        "  \"membership\":{{\"rebalances\":[{}],\"decisions\":{},\"selected\":{}}},",
        membership
            .as_of
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        membership.rows.len(),
        membership.selected.len()
    );
    let counts: Vec<String> = panel
        .counts_per_year()
        .into_iter()
        .map(|(y, (n, _))| format!("{{\"year\":{},\"symbols\":{}}}", y, n))
        .collect();
    let rows: Vec<String> = panel
        .rows_per_year()
        .into_iter()
        .map(|(y, n)| format!("{{\"year\":{},\"rows\":{}}}", y, n))
        .collect();
    let _ = writeln!(
        s,
        "  \"panel\":{{\"rows\":{},\"symbols\":{},\"delisted\":{},\"gaps\":{},\"first_ts\":{},\"last_ts\":{},\"first_date\":{},\"last_date\":{}}},",
        panel.rows.len(),
        panel.symbols.len(),
        panel.delisted.len(),
        panel.gaps,
        panel.first_ts,
        panel.last_ts,
        jstr(&date(panel.first_ts)),
        jstr(&date(panel.last_ts))
    );
    let _ = writeln!(s, "  \"symbols_per_year\":[{}],", counts.join(","));
    let _ = writeln!(s, "  \"rows_per_year\":[{}],", rows.join(","));
    let _ = writeln!(
        s,
        "  \"delisted\":[{}],",
        panel
            .delisted
            .iter()
            .map(|x| jstr(x))
            .collect::<Vec<_>>()
            .join(",")
    );
    // #145: record what the builder retained. A panel may contain the holdout
    // for provenance; the flag says whether an explicit final test permitted
    // it, and `evaluated` stays false because building is not scoring.
    let lb = Lockbox::crypto_2026h2();
    let overlap = lb.overlap(panel.rows.iter().map(|r| r.bar.time));
    let _ = writeln!(
        s,
        "  \"lockbox\":{},",
        lb.json(&overlap, allow_lockbox, false, false)
    );
    if let Some(snapshot) = snapshot {
        // The snapshot file is produced by the fetch/verify step; the panel
        // records it verbatim so the vendor checksums travel with the panel.
        let _ = writeln!(s, "  \"snapshot\":{},", snapshot.trim());
    }
    let sha = manifest::sha256_file(output).unwrap_or_else(|_| "unknown".to_string());
    let bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    let _ = writeln!(
        s,
        "  \"output\":{{\"path\":{},\"sha256\":{},\"bytes\":{}}}",
        jstr(output),
        jstr(&sha),
        bytes
    );
    s.push_str("}\n");
    s
}

pub(crate) fn date(ts: i64) -> String {
    let (y, m, d) = ohlcv::civil_from_days(ts.div_euclid(86_400));
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ---------------------------------------------------------------------------
// Costs, exits and the matched-exposure benchmark
// ---------------------------------------------------------------------------

/// All-in symmetric cost. The label subtracts `2 * per_side_bps`; the evaluator
/// charges every position change and the final exit at the same rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub fee_bps: f64,
    pub half_spread_bps: f64,
    pub impact_bps: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel {
            fee_bps: 10.0,
            half_spread_bps: 0.0,
            impact_bps: 0.0,
        }
    }
}

impl CostModel {
    pub fn per_side_bps(&self) -> f64 {
        self.fee_bps + self.half_spread_bps + self.impact_bps
    }

    pub fn round_trip_bps(&self) -> f64 {
        2.0 * self.per_side_bps()
    }

    /// Label cost fraction (one round trip).
    pub fn label_cost(&self) -> f64 {
        self.round_trip_bps() / 10_000.0
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostLedger {
    pub entry_bps: f64,
    pub exit_bps: f64,
    pub entries: usize,
    pub exits: usize,
}

impl CostLedger {
    pub fn total_bps(&self) -> f64 {
        self.entry_bps + self.exit_bps
    }
}

/// Net returns for a long/flat position series. `groups` identifies one series
/// per row; every group starts flat and is charged its final exit.
///
/// The returned ledger counts the entry and the final exit separately, which
/// is what makes a one-trade fixture agree with the label's round trip.
pub fn net_returns(
    gross: &[f64],
    positions: &[f64],
    groups: &[u64],
    cost: &CostModel,
    charge_final_exit: bool,
) -> (Vec<f64>, CostLedger) {
    let side = cost.per_side_bps() / 10_000.0;
    let mut net = vec![0.0f64; gross.len()];
    let mut ledger = CostLedger::default();
    let mut prev: BTreeMap<u64, f64> = BTreeMap::new();
    for i in 0..gross.len() {
        let p = positions[i];
        let old = prev.get(&groups[i]).copied().unwrap_or(0.0);
        let delta = (p - old).abs();
        let c = side * delta;
        net[i] = p * gross[i] - c;
        if delta > 0.0 {
            if p > old {
                ledger.entry_bps += c * 10_000.0;
                ledger.entries += 1;
            } else {
                ledger.exit_bps += c * 10_000.0;
                ledger.exits += 1;
            }
        }
        prev.insert(groups[i], p);
    }
    if charge_final_exit {
        // Charge the exit on the last row of each group.
        let mut last: BTreeMap<u64, usize> = BTreeMap::new();
        for (i, g) in groups.iter().enumerate() {
            last.insert(*g, i);
        }
        let mut order: Vec<(usize, u64)> = last.iter().map(|(g, i)| (*i, *g)).collect();
        order.sort_by_key(|(i, _)| *i);
        for (i, g) in order {
            let p = prev.get(&g).copied().unwrap_or(0.0);
            if p != 0.0 {
                let c = side * p.abs();
                net[i] -= c;
                ledger.exit_bps += c * 10_000.0;
                ledger.exits += 1;
            }
        }
    }
    (net, ledger)
}

/// Buy-and-hold under the same cost model: one entry, one charged exit.
pub fn buy_and_hold(
    gross: &[f64],
    groups: &[u64],
    cost: &CostModel,
) -> (Vec<f64>, CostLedger) {
    let positions = vec![1.0f64; gross.len()];
    net_returns(gross, &positions, groups, cost, true)
}

/// A random long/flat benchmark with the **same exposure** and the same number
/// of entries per group. Deterministic for a seed.
pub fn matched_exposure_random_long(
    positions: &[f64],
    groups: &[u64],
    seed: u64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; positions.len()];
    // Permute within each group so the count of long rows is identical.
    let mut by_group: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        by_group.entry(*g).or_default().push(i);
    }
    let mut state = seed | 1;
    for (_g, idx) in by_group {
        let mut vals: Vec<f64> = idx.iter().map(|&i| positions[i]).collect();
        // xorshift64* and Fisher-Yates; no dependency on the search RNG.
        for i in (1..vals.len()).rev() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let j = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % (i as u64 + 1)) as usize;
            vals.swap(i, j);
        }
        for (k, &i) in idx.iter().enumerate() {
            out[i] = vals[k];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Canaries
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct SpliceCanary {
    pub old: String,
    pub new: String,
    pub measured_ratio: f64,
    pub declared_k: f64,
    pub pass: bool,
    pub reason: String,
}

/// Check every declared migration against the measured boundary ratio.
pub fn splice_canaries(series: &BTreeMap<String, Vec<Bar>>, ms: &[Migration]) -> Vec<SpliceCanary> {
    let mut out = Vec::new();
    for m in ms {
        let (Some(old), Some(new)) = (series.get(&m.old), series.get(&m.new)) else {
            out.push(SpliceCanary {
                old: m.old.clone(),
                new: m.new.clone(),
                measured_ratio: f64::NAN,
                declared_k: m.k,
                pass: false,
                reason: "leg missing from the panel".to_string(),
            });
            continue;
        };
        let (Some(a), Some(b)) = (old.last(), new.first()) else {
            continue;
        };
        let ratio = ohlcv::splice_ratio(a, b);
        let err = (ratio / m.k).ln().abs();
        out.push(SpliceCanary {
            old: m.old.clone(),
            new: m.new.clone(),
            measured_ratio: ratio,
            declared_k: m.k,
            pass: err <= m.tolerance,
            reason: if err <= m.tolerance {
                "ratio within tolerance".to_string()
            } else {
                format!("|ln(ratio/k)| = {:.5} > tolerance {:.5}", err, m.tolerance)
            },
        });
    }
    out
}

/// A level jump between adjacent bars of one symbol, as a fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelJump {
    pub symbol: String,
    pub at: i64,
    pub ratio: f64,
}

/// Report consecutive-bar level jumps above `factor` (e.g. 2x). An undeclared
/// migration such as LUNA's 20,000x gap appears here instead of silently
/// contaminating level features.
pub fn level_jumps(series: &BTreeMap<String, Vec<Bar>>, factor: f64) -> Vec<LevelJump> {
    let mut out = Vec::new();
    for (symbol, bars) in series {
        for w in bars.windows(2) {
            if w[0].close > 0.0 && w[1].open > 0.0 {
                let ratio = w[1].open / w[0].close;
                if ratio > factor || ratio < 1.0 / factor {
                    out.push(LevelJump {
                        symbol: symbol.clone(),
                        at: w[1].time,
                        ratio,
                    });
                }
            }
        }
    }
    out
}

/// Consecutive-bar gaps longer than `days`, with the number of missing days.
pub fn long_gaps(series: &BTreeMap<String, Vec<Bar>>, days: i64) -> Vec<(String, i64, usize)> {
    let mut out = Vec::new();
    for (symbol, bars) in series {
        for w in bars.windows(2) {
            let missing = ((w[1].time - w[0].time) / 86_400 - 1).max(0) as usize;
            if missing as i64 > days {
                out.push((symbol.clone(), w[0].time, missing));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CSV/table input
// ---------------------------------------------------------------------------

fn find(columns: &[String], aliases: &[&str]) -> Option<usize> {
    columns.iter().position(|c| {
        let l = c.to_ascii_lowercase();
        aliases.iter().any(|a| l == *a)
    })
}

/// Read `ts,symbol,open,high,low,close,volume[,quote_volume,...]` from a table.
pub fn series_from_table(table: &Table) -> Result<BTreeMap<String, Vec<Bar>>, String> {
    let names = table.names.clone();
    let time = find(&names, &["ts", "timestamp", "time", "date", "datetime"])
        .ok_or("table has no time column")?;
    let symbol = find(&names, &["symbol", "ticker", "asset", "pair"])
        .ok_or("table has no symbol column")?;
    let open = find(&names, &["open", "o"]).ok_or("table has no open column")?;
    let high = find(&names, &["high", "h"]).ok_or("table has no high column")?;
    let low = find(&names, &["low", "l"]).ok_or("table has no low column")?;
    let close = find(&names, &["close", "c"]).ok_or("table has no close column")?;
    let volume = find(&names, &["volume", "vol", "v"]);
    let quote = find(&names, &["quote_volume", "quotevolume", "qv"]);
    let trades = find(&names, &["trades", "n_trades"]);
    let tb = find(&names, &["taker_buy_base", "takerbuybase"]);
    let tq = find(&names, &["taker_buy_quote", "takerbuyquote"]);
    let mut out: BTreeMap<String, Vec<Bar>> = BTreeMap::new();
    for r in 0..table.len() {
        let sym = table
            .text(r, symbol)
            .map(|s| s.to_string())
            .unwrap_or_else(|| table.cell_text(r, symbol));
        if sym.is_empty() {
            continue;
        }
        let t = table
            .ints(time)
            .map(|v| v[r])
            .unwrap_or_else(|| table.value(r, time).round() as i64);
        let get = |c: Option<usize>| c.map(|c| table.value(r, c)).unwrap_or(0.0);
        let mut bar = Bar::new(
            t,
            table.value(r, open),
            table.value(r, high),
            table.value(r, low),
            table.value(r, close),
            get(volume),
        )
        .with_flow(get(quote), get(trades));
        bar.taker_buy_base = get(tb);
        bar.taker_buy_quote = get(tq);
        out.entry(sym).or_default().push(bar);
    }
    for bars in out.values_mut() {
        bars.sort_by_key(|b| b.time);
        bars.dedup_by_key(|b| b.time);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

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

pub(crate) fn read_table(path: &str) -> Result<Table, String> {
    if store::is_store_path(path) {
        store::FileStore::open(path)?.to_table()
    } else {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        crate::data::parse_table(&text)
    }
}

pub fn run(args: &[String]) -> ! {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{}", usage());
        std::process::exit(0);
    }
    let cmd = args[0].as_str();
    // #259: the hourly v2 pipeline. `build` switches to it when `--data` is a
    // v2 base store (it carries `cov15m`).
    let v2_build = cmd == "build"
        && args
            .windows(2)
            .any(|w| w[0] == "--data" && crate::panel_v2::is_base_store(&w[1]));
    if matches!(cmd, "snapshot" | "base" | "validate") || v2_build {
        crate::panel_v2::cli(cmd, &args[1..]);
    }
    if matches!(cmd, "schema" | "append") {
        crate::panel_append::cli(cmd, &args[1..]);
    }
    let mut data = None;
    let mut membership_path = None;
    let mut out = None;
    let mut cfg = MembershipConfig::default();
    let mut migrations_src: Option<String> = None;
    let mut group_rows = store::DEFAULT_GROUP_ROWS;
    let mut bar_seconds = 86_400i64;
    let mut start: Option<i64> = None;
    let mut end: Option<i64> = None;
    let mut allow_lockbox = false;
    let mut resume: Option<String> = None;
    let mut membership_cache: Option<String> = None;
    let mut snapshot_path: Option<String> = None;
    let mut strict = false;
    let mut relists: Vec<(String, i64)> = Vec::new();
    let mut jump_factor = 5.0f64;
    let mut listing_gap_days = 30i64;
    let mut i = 1usize;
    while i < args.len() {
        let val = |i: usize| -> String {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: {} requires a value", args[i]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--data" => {
                data = Some(val(i));
                i += 1;
            }
            "--membership" => {
                membership_path = Some(val(i));
                i += 1;
            }
            "--out" => {
                out = Some(val(i));
                i += 1;
            }
            "--top" => {
                cfg.top = val(i).parse().unwrap_or(cfg.top);
                i += 1;
            }
            "--lookback" => {
                cfg.lookback_days = val(i).parse().unwrap_or(cfg.lookback_days);
                i += 1;
            }
            "--rebalance" => {
                cfg.rebalance_days = val(i).parse().unwrap_or(cfg.rebalance_days);
                i += 1;
            }
            "--min-history" => {
                cfg.min_history_days = val(i).parse().unwrap_or(cfg.min_history_days);
                i += 1;
            }
            "--min-bars" => {
                cfg.min_bars = val(i).parse().unwrap_or(cfg.min_bars);
                i += 1;
            }
            "--min-adv" => {
                cfg.min_adv_usd = val(i).parse().unwrap_or(cfg.min_adv_usd);
                i += 1;
            }
            "--migrations" => {
                migrations_src = Some(val(i));
                i += 1;
            }
            "--group-rows" => {
                group_rows = val(i).parse().unwrap_or(group_rows);
                i += 1;
            }
            "--bar-seconds" => {
                bar_seconds = val(i).parse().unwrap_or(bar_seconds);
                i += 1;
            }
            "--start" => {
                start = crate::lockbox::parse_time(&val(i));
                i += 1;
            }
            "--end" => {
                end = crate::lockbox::parse_time(&val(i));
                i += 1;
            }
            "--allow-lockbox" => allow_lockbox = true,
            "--resume" => {
                resume = Some(val(i));
                i += 1;
            }
            "--cache" => {
                membership_cache = Some(val(i));
                i += 1;
            }
            "--strict" => strict = true,
            "--relist" => {
                relists.extend(crate::panel_v2::parse_relists(&val(i)).unwrap_or_else(|e| fail(&e)));
                i += 1;
            }
            "--jump-factor" => {
                jump_factor = val(i).parse().unwrap_or(jump_factor);
                i += 1;
            }
            "--listing-gap-days" => {
                listing_gap_days = val(i).parse().unwrap_or(listing_gap_days);
                i += 1;
            }
            "--snapshot" => {
                snapshot_path = Some(val(i));
                i += 1;
            }
            other => {
                eprintln!("error: unknown panel option '{}'\n\n{}", other, usage());
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let data = match data {
        Some(d) => d,
        None => {
            eprintln!("error: panel needs --data <csv|rcsb>");
            std::process::exit(2);
        }
    };
    let migrations = match migrations_src.as_deref().map(parse_migrations).transpose() {
        Ok(m) => m.unwrap_or_default(),
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };
    // Reading and reshaping the 1.5 GB prep store into per-symbol bars costs
    // ~15 s. A warm `build` must not pay it, so the table is loaded lazily.
    let load_series = || -> BTreeMap<String, Vec<Bar>> {
        let table = match read_table(&data) {
            Ok(t) => t,
            Err(e) => fail(&e),
        };
        let mut series = match series_from_table(&table) {
            Ok(s) => s,
            Err(e) => fail(&e),
        };
        if !migrations.is_empty() {
            series = match ohlcv::apply_migrations(&series, &migrations) {
                Ok(s) => s,
                Err(e) => fail(&e),
            };
        }
        series
    };
    // `--start/--end` scope the rebalance decisions as well as the output.
    cfg.start = start;
    cfg.end = end;
    match cmd {
        "membership" => {
            // #145: the membership CSV is a pure function of the store and the
            // ranking settings, and costs ~10 s because it reshapes the whole
            // store. `--cache` keys it by the store SHA-256 so a warm pipeline
            // does not recompute it.
            let out = out.unwrap_or_else(|| "membership.csv".to_string());
            let store_sha = store_key(&data);
            let key = membership_cache_key(&store_sha, &cfg);
            if let Some(dir) = &membership_cache {
                let dir = std::path::Path::new(dir);
                let state = dir.join("state.txt");
                let csv = dir.join("membership.csv");
                if std::fs::read_to_string(&state).map(|got| got == key).unwrap_or(false)
                    && csv.exists()
                {
                    let text = std::fs::read_to_string(&csv)
                        .unwrap_or_else(|e| fail(&format!("{}: {}", csv.display(), e)));
                    publish_text(std::path::Path::new(&out), &text)
                        .unwrap_or_else(|e| fail(&e));
                    eprintln!(
                        "panel membership: cached (store sha256 {}), {} as-of dates -> {}",
                        &store_sha[..store_sha.len().min(12)],
                        text.lines().count().saturating_sub(1),
                        out
                    );
                    std::process::exit(0);
                }
                std::fs::create_dir_all(dir)
                    .unwrap_or_else(|e| fail(&format!("{}: {}", dir.display(), e)));
                // Publish the settings only with the table, so a crash leaves a
                // stale state.txt pointing at no CSV (a miss), never a CSV that
                // belongs to other settings.
                let _ = std::fs::remove_file(&state);
            }
            let series = load_series();
            let m = match build_membership(&series, &cfg) {
                Ok(m) => m,
                Err(e) => fail(&e),
            };
            let mut text = String::from("as_of,symbol,member,adv_usd,n_bars\n");
            for r in &m.rows {
                let _ = writeln!(
                    text,
                    "{},{},{},{:.4},{}",
                    r.as_of, r.symbol, r.member as u8, r.adv_usd, r.n_bars
                );
            }
            if let Err(e) = std::fs::write(&out, &text) {
                fail(&format!("{}: {}", out, e));
            }
            if let Some(dir) = &membership_cache {
                let dir = std::path::Path::new(dir);
                publish_text(&dir.join("membership.csv"), &text).unwrap_or_else(|e| fail(&e));
                publish_text(&dir.join("state.txt"), &key).unwrap_or_else(|e| fail(&e));
            }
            eprintln!(
                "panel membership: {} decisions, {} as-of dates, {} selected -> {}",
                m.rows.len(),
                m.as_of.len(),
                m.selected.len(),
                out
            );
        }
        "build" => {
            let mp = membership_path.unwrap_or_else(|| "membership.csv".to_string());
            let m = match read_membership(&mp) {
                Ok(m) => m,
                Err(e) => fail(&e),
            };
            // The cache key is the input store checksum, the decisions and the
            // settings that change a row. A mismatch is an error, not a mix.
            let store_sha = store_key(&data);
            let settings = migrations_src.clone().unwrap_or_default();
            let mut panel = match &resume {
                Some(dir) => {
                    let warm = panel_cache_ready(dir, bar_seconds, &store_sha, &m, &settings)
                        .unwrap_or_else(|e| fail(&e));
                    let panel = if warm {
                        build_panel_resumable(None, &m, bar_seconds, dir, &store_sha, &settings)
                    } else {
                        let series = load_series();
                        build_panel_resumable(
                            Some(&series),
                            &m,
                            bar_seconds,
                            dir,
                            &store_sha,
                            &settings,
                        )
                    };
                    panel.unwrap_or_else(|e| fail(&e))
                }
                None => {
                    let series = load_series();
                    build_panel_with_step(&series, &m, bar_seconds).unwrap_or_else(|e| fail(&e))
                }
            };
            panel.retain_range(start, end);
            if panel.rows.is_empty() {
                fail("panel build range selects no rows");
            }
            // Building the full panel may include the lockbox: provenance needs
            // the retained rows. An explicit evaluated range or a runner must
            // still clear the one central guard.
            if start.is_some() || end.is_some() {
                if let Err(e) =
                    check_build_window(panel.first_ts, panel.last_ts, bar_seconds, allow_lockbox)
                {
                    fail(&e);
                }
            }
            let snapshot_text = match &snapshot_path {
                Some(p) => match std::fs::read_to_string(p) {
                    Ok(t) if t.trim().starts_with('{') && t.trim().ends_with('}') => Some(t),
                    Ok(_) => fail(&format!("snapshot '{}' is not a JSON object", p)),
                    Err(e) => fail(&format!("{}: {}", p, e)),
                },
                None => None,
            };
            let out = out.unwrap_or_else(|| "panel.rcsb".to_string());
            match write_panel_panel_hashed(
                &out,
                &panel,
                &m,
                &[&data, &mp],
                &store_sha,
                group_rows,
                allow_lockbox,
                snapshot_text.as_deref(),
            ) {
                Ok(manifest) => eprintln!(
                    "panel build: {} rows, {} symbols, {} delisted, {} gaps -> {} (manifest {})",
                    panel.rows.len(),
                    panel.symbols.len(),
                    panel.delisted.len(),
                    panel.gaps,
                    out,
                    manifest
                ),
                Err(e) => fail(&e),
            }
        }
        "audit" => {
            let series = load_series();
            let m = match build_membership(&series, &cfg) {
                Ok(m) => m,
                Err(e) => fail(&e),
            };
            let mut panel = match build_panel_with_step(&series, &m, bar_seconds) {
                Ok(p) => p,
                Err(e) => fail(&e),
            };
            panel.retain_range(start, end);
            if panel.rows.is_empty() {
                fail("panel audit range selects no rows");
            }
            let lb = Lockbox::crypto_2026h2();
            let coverage = lb.overlap(panel.rows.iter().map(|r| r.bar.time));
            println!(
                "# panel: rows={} symbols={} delisted={} gaps={} span={}..{}",
                panel.rows.len(),
                panel.symbols.len(),
                panel.delisted.len(),
                panel.gaps,
                date(panel.first_ts),
                date(panel.last_ts)
            );
            println!(
                "# panel: lockbox rows={} windows={} permitted={} complete=false",
                coverage.rows,
                coverage.windows.len(),
                allow_lockbox
            );
            let mut per_period: BTreeMap<i64, usize> = BTreeMap::new();
            for r in &m.rows {
                if r.member {
                    *per_period.entry(r.as_of).or_insert(0) += 1;
                }
            }
            println!(
                "# panel: rebalances={} members_min={} members_max={}",
                m.as_of.len(),
                per_period.values().copied().min().unwrap_or(0),
                per_period.values().copied().max().unwrap_or(0)
            );
            for (y, n) in panel.counts_per_year() {
                println!("# panel: year={} symbols={}", y, n.0);
            }
            if !panel.delisted.is_empty() {
                println!("# panel: delisted={}", panel.delisted.iter().cloned().collect::<Vec<_>>().join(","));
            }
            for c in splice_canaries(&series, &migrations) {
                println!(
                    "# panel: splice {}->{} ratio={:.6} k={:.6} pass={} ({})",
                    c.old, c.new, c.measured_ratio, c.declared_k, c.pass, c.reason
                );
            }
            let mut undeclared = 0usize;
            for j in level_jumps(&series, jump_factor) {
                let wanted: Vec<i64> = relists.iter().filter(|(s, _)| *s == j.symbol).map(|(_, t)| *t).collect();
                let prev = series.get(&j.symbol).and_then(|b| {
                    let k = b.partition_point(|x| x.time < j.at);
                    k.checked_sub(1).map(|k| b[k].time)
                });
                let declared = crate::panel_v2::jump_declared(prev, j.at, &wanted, listing_gap_days);
                undeclared += !declared as usize;
                println!(
                    "# panel: jump {} at {} ratio={:.4} declared={}",
                    j.symbol,
                    date(j.at),
                    j.ratio,
                    declared
                );
            }
            for (s, t, n) in long_gaps(&series, 7) {
                println!("# panel: gap {} at {} missing_days={}", s, date(t), n);
            }
            if strict && undeclared > 0 {
                fail(&format!(
                    "{} undeclared level jump(s); declare the break with --relist SYMBOL@DATE",
                    undeclared
                ));
            }
        }
        "cost" => {
            // A one-round-trip fixture: entry and final exit are both charged.
            let cost = CostModel {
                fee_bps: 10.0,
                half_spread_bps: 0.0,
                impact_bps: 0.0,
            };
            let gross = [0.01];
            let positions = [1.0];
            let groups = [0u64];
            let (net, ledger) = net_returns(&gross, &positions, &groups, &cost, true);
            println!(
                "cost: per_side={}bp round_trip={}bp net={:.6} entry={}bp exit={}bp",
                cost.per_side_bps(),
                cost.round_trip_bps(),
                net[0],
                ledger.entry_bps,
                ledger.exit_bps
            );
            if (ledger.total_bps() - cost.round_trip_bps()).abs() > 1e-9 {
                fail("the cost ledger does not charge both sides");
            }
        }
        other => fail(&format!("unknown panel command '{}'", other)),
    }
    std::process::exit(0);
}

pub(crate) fn read_membership(path: &str) -> Result<Membership, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut rows = Vec::new();
    let mut as_ofs = BTreeSet::new();
    let mut selected = BTreeSet::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 5 {
            return Err(format!("{}:{}: expected 5 columns", path, i + 1));
        }
        let as_of: i64 = f[0].parse().map_err(|_| format!("{}:{}: bad as_of", path, i + 1))?;
        let member: bool = f[2].trim() != "0";
        let adv: f64 = f[3].parse().unwrap_or(0.0);
        let n: usize = f[4].parse().unwrap_or(0);
        as_ofs.insert(as_of);
        if member {
            selected.insert(f[1].to_string());
        }
        rows.push(MembershipRow {
            as_of,
            symbol: f[1].to_string(),
            member,
            adv_usd: adv,
            n_bars: n,
        });
    }
    Ok(Membership {
        rows,
        as_of: as_ofs.into_iter().collect(),
        selected,
    })
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(3);
}

pub(crate) fn jstr(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn usage() -> String {
    "rcsr panel — point-in-time universe and panel builder (#145)

USAGE:
  rcsr panel membership --data daily.csv --out membership.csv [options]
  rcsr panel build --data daily.csv --membership membership.csv --out panel.rcsb [options]
  rcsr panel audit --data daily.csv [options]
  rcsr panel cost

MEMBERSHIP:
  --top <n>               names selected per rebalance (default 100)
  --lookback <days>       trailing liquidity window (default 90)
  --rebalance <days>      30 = monthly, 90 = quarterly (default 30)
  --min-history <days>    minimum age before selection (default 90)
  --min-bars <n>          minimum trailing bars (default 60)
  --min-adv <usd>         minimum trailing ADV (default 0)
  --cache <dir>           cache the membership table keyed by the store SHA-256
                          and the ranking settings; a warm run copies it out

BUILD / AUDIT:
  --data <csv|rcsb>       daily OHLCV with a symbol column
  --membership <csv>      point-in-time membership from `membership`
  --migrations <table>    OLD->NEW:k[:tol],... verified before the build
  --out <path>            membership CSV or panel .rcsb
  --group-rows <n>        column-store group size (default 65536)
  --bar-seconds <n>       bar spacing for gap counts (default 86400; use 3600
                          for an hourly panel, 900 for 15m)
  --start <epoch|date>    build window start (default: panel first bar)
  --end <epoch|date>      build window end (default: panel last bar)
  --allow-lockbox         permit a window overlapping 2026H2 (declared final test only)
  --locked-test <START,END>
                          build (hourly v2): declare the protocol locked test (UTC,
                          half-open) in <out>.manifest.json and <out>.schema.json;
                          append and enrich copy it, and the search, walkforward,
                          ladder and ablation score refuse it
  --resume <dir>          resumable build: one atomic binary part per symbol,
                          keyed by the store SHA-256, the membership decisions
                          and the settings; a fully warm re-run reads the parts
                          only (no 1.5 GB parse) and prints the resume line
  --snapshot <json>       embed a verified archive-snapshot manifest verbatim
                          under the panel manifest's `snapshot` block
  --strict               audit: exit 3 on a level jump that is not declared (--relist) or
                          a listing break (gap over --listing-gap-days, default 30)

`build` refuses 2026H2 by default. Every row is a real bar: delisted symbols
stop at their last bar, listings start at their first, and no value is
forward-filled across a listing/delisting/migration boundary. `audit` reports
splices, level jumps, long gaps, per-year symbol counts and delisted names.

HOURLY V2 PANEL (#259):
  rcsr panel snapshot --input raw_15m_all --out snapshot.json [--jobs n]
  rcsr panel base --input raw_15m_all --out base_h1.rcsb [options]
  rcsr panel build --data base_h1.rcsb --membership membership.csv --out panel.rcsb [options]
  rcsr panel validate --data panel.rcsb [--base base_h1.rcsb] [--membership membership.csv]
  rcsr panel schema --data panel.rcsb [--panel-version V] [--verify]
  rcsr panel append --data panel.rcsb --family symbol_major.rcsb --columns a,b --out out.rcsb [--scratch dir]
  --input <dir>           archive directory (SYMBOL/SYMBOL-15m-YYYY-MM.zip); no network
  --jobs <n>              parallel symbols for snapshot and base (default 1)
  --gpu <auto|on|off>     base: run the 15m to 1h fold on the CUDA device (auto: from 1M decoded rows;
                          on: always; off: the CPU path). build: run the membership as-of join,
                          gap tags and chain starts on the device (auto: from 1M base rows).
                          Same store bytes either way; a CPU wave or join is labelled in
                          <out>.manifest.json (backend). --no-gpu/--cpu = off
  --require-full-bucket   base: drop an hour with fewer than four 15m bars (default)
  --keep-partial          base: keep it with cov15m 1-3 and the PARTIAL_BUCKET flag
  --interval <15m|1m>     base: archive interval; 1m only reaches the lockbox guard
  --symbols <file>        snapshot, base: keep only the listed symbols (one per line; # comments
                          and blank lines ignored; an empty list is a usage error). A listed
                          symbol with no directory (missing) or an empty directory (empty)
                          adds no rows and is not an error. The `universe` block of
                          snapshot.json and <out>.manifest.json records the file SHA-256,
                          listed, kept, dropped, missing and empty. A changed list rebuilds
                          the base --resume parts
  --resume <dir>          base: per-symbol cache keyed by the archive SHA-256
  --relist SYMBOL@DATE    build: declare a listing break (repeatable, comma list)
  --listing-gap-days <n>  build: a longer gap between two bars starts a new listing (30)
  --jump-factor <x>       build: undeclared open/previous-close ratio that fails (5)
  --outage-share <x>      build: share of the cross-section missing for a venue outage (0.5)
  --replay <n>            validate: prefix-causality cuts (3)
  `build` uses the v2 builder when --data has a cov15m column. `validate` exits 3
  when a check fails. A 2026H2 archive or row is refused without --allow-lockbox.

REGISTRY SCHEMA AND FAMILY APPEND (#260):
  `build` and `append` write <panel>.schema.json: one entry per column with family,
  timeframe, source columns, transform, causal offset, warm-up bars, dtype, unit,
  availability start and end, PIT rule, semantic cost, status and the SHA-256 of the
  stored values. `schema` rewrites it; `schema --verify` re-hashes the store and exits
  3 on a mismatch. A column that is not in the feature registry is a hard error.
  `append` carries registered columns of a symbol-major store (computed by `rcsr
  enrich`) into the time-major panel by (symbol, ts) key. Existing blocks are copied
  unchanged, the input file is untouched, a panel row without a family row is NaN.
"
        .to_string()
}
