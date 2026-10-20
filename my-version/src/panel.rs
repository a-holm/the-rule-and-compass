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
    let mut as_of = month_start(all_times[0]);
    let mut decisions = Vec::new();
    let mut as_ofs = Vec::new();
    while as_of <= last_ts {
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
    let mut rows = Vec::new();
    let mut symbols = BTreeSet::new();
    let mut gaps = 0usize;
    let mut first_ts = i64::MAX;
    let mut last_ts = i64::MIN;
    // One sorted index per symbol: the naive `at()` scan is O(decisions) per
    // bar and turns a 665k-row build into minutes.
    let mut index: BTreeMap<&str, Vec<&MembershipRow>> = BTreeMap::new();
    for r in &membership.rows {
        index.entry(r.symbol.as_str()).or_default().push(r);
    }
    for v in index.values_mut() {
        v.sort_by_key(|r| r.as_of);
    }
    for (symbol, bars) in series {
        let decisions = index.get(symbol.as_str());
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
                // A membership exclusion starts a new chain: it is not a data
                // gap and must not be counted as one.
                prev = None;
                continue;
            }
            let missing = match prev {
                Some(p) if b.time > p => ((b.time - p) / step - 1).max(0) as usize,
                _ => 0,
            };
            gaps += missing;
            prev = Some(b.time);
            first_ts = first_ts.min(b.time);
            last_ts = last_ts.max(b.time);
            symbols.insert(symbol.clone());
            rows.push(PanelRow {
                symbol: symbol.clone(),
                bar: *b,
                as_of: decision.as_of,
                gap_buckets: missing,
            });
        }
    }
    if rows.is_empty() {
        return Err("panel: membership selects no rows".to_string());
    }
    rows.sort_by(|a, b| a.bar.time.cmp(&b.bar.time).then(a.symbol.cmp(&b.symbol)));
    // A symbol whose last actual bar is well before the panel end is delisted,
    // not forward-filled.
    let delisted: BTreeSet<String> = symbols
        .iter()
        .filter(|s| {
            series
                .get(*s)
                .and_then(|v| v.last())
                .map(|b| b.time < last_ts - 30 * 86_400)
                .unwrap_or(true)
        })
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
    for r in &panel.rows {
        times.push(r.bar.time);
        cols[0].push(r.bar.time as f64);
        let code = match sym_levels.iter().position(|s| s == &r.symbol) {
            Some(c) => c as u32,
            None => {
                sym_levels.push(r.symbol.clone());
                (sym_levels.len() - 1) as u32
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

/// Write a panel store plus a provenance manifest.
pub fn write_panel_panel(
    out: &str,
    panel: &Panel,
    membership: &Membership,
    inputs: &[&str],
    group_rows: usize,
) -> Result<String, String> {
    let table = panel_table(panel);
    store::write_store_with_time(out, &table, group_rows, Some(0))?;
    let text = provenance_json(inputs, membership, panel, out);
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
) -> String {
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"tool\":\"rcsr panel\",");
    let _ = writeln!(s, "  \"inputs\":[");
    for (i, p) in inputs.iter().enumerate() {
        let sha = manifest::sha256_file(p).unwrap_or_else(|_| "unknown".to_string());
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

fn date(ts: i64) -> String {
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

fn read_table(path: &str) -> Result<Table, String> {
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
    let table = match read_table(&data) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    };
    let mut series = match series_from_table(&table) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    };
    let migrations = match migrations_src.as_deref().map(parse_migrations).transpose() {
        Ok(m) => m.unwrap_or_default(),
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };
    if !migrations.is_empty() {
        match ohlcv::apply_migrations(&series, &migrations) {
            Ok(s) => series = s,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(3);
            }
        }
    }
    match cmd {
        "membership" => {
            let m = match build_membership(&series, &cfg) {
                Ok(m) => m,
                Err(e) => fail(&e),
            };
            let out = out.unwrap_or_else(|| "membership.csv".to_string());
            let mut text = String::from("as_of,symbol,member,adv_usd,n_bars\n");
            for r in &m.rows {
                let _ = writeln!(
                    text,
                    "{},{},{},{:.4},{}",
                    r.as_of, r.symbol, r.member as u8, r.adv_usd, r.n_bars
                );
            }
            if let Err(e) = std::fs::write(&out, text) {
                fail(&format!("{}: {}", out, e));
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
            let panel = match build_panel_with_step(&series, &m, bar_seconds) {
                Ok(p) => p,
                Err(e) => fail(&e),
            };
            // Building may include the lockbox: provenance needs the full
            // panel. A *runner* refuses it; an explicit evaluated range is
            // checked here as well.
            if !allow_lockbox {
                if let (Some(a), Some(b)) = (start, end) {
                    if let Err(e) = Lockbox::crypto_2026h2().check(a, b) {
                        fail(&e);
                    }
                }
            }
            let out = out.unwrap_or_else(|| "panel.rcsb".to_string());
            match write_panel_panel(&out, &panel, &m, &[&data, &mp], group_rows) {
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
            let m = match build_membership(&series, &cfg) {
                Ok(m) => m,
                Err(e) => fail(&e),
            };
            let panel = match build_panel_with_step(&series, &m, bar_seconds) {
                Ok(p) => p,
                Err(e) => fail(&e),
            };
            println!(
                "# panel: rows={} symbols={} delisted={} gaps={} span={}..{}",
                panel.rows.len(),
                panel.symbols.len(),
                panel.delisted.len(),
                panel.gaps,
                date(panel.first_ts),
                date(panel.last_ts)
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
            for j in level_jumps(&series, 5.0) {
                println!("# panel: jump {} at {} ratio={:.4}", j.symbol, date(j.at), j.ratio);
            }
            for (s, t, n) in long_gaps(&series, 7) {
                println!("# panel: gap {} at {} missing_days={}", s, date(t), n);
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

fn read_membership(path: &str) -> Result<Membership, String> {
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

fn jstr(s: &str) -> String {
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

`build` refuses 2026H2 by default. Every row is a real bar: delisted symbols
stop at their last bar, listings start at their first, and no value is
forward-filled across a listing/delisting/migration boundary. `audit` reports
splices, level jumps, long gaps, per-year symbol counts and delisted names.
"
        .to_string()
}
