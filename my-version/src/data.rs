//! Data loading, splitting and normalisation.
//!
//! Rows are split into train/test in one of four ways:
//!
//! * `random` — shuffled i.i.d. holdout (the default; only correct when rows are
//!   exchangeable);
//! * `temporal` — chronological holdout: train on the past, test on the future,
//!   optionally with an **embargo** ([`SplitSpec::gap`] rows) between the two so
//!   that autocorrelation cannot leak the future into the training set;
//! * `group` — whole groups (a stock, a subject, a sensor) are assigned to one
//!   side only, so the test error measures generalisation to *unseen* groups;
//! * `temporal_group` — a chronological holdout performed inside every group:
//!   train on the past of every stock and test on its future, which is the
//!   realistic setting for panel data (OHLC for many symbols, over one period).
//!
//! Normalisation statistics are always computed from the **training rows only**.
//! Using the whole table would leak the test set into the model's input scaling.
//!
//! Time columns may be numbers or dates (`2024-01-31`, `2024/01/31`,
//! `2024-01-31T09:30:00`); anything unparseable falls back to a lexicographic
//! ordering, so the split is still well defined. Group columns may be text
//! (ticker symbols) — they are never used as model inputs, only for splitting.

use crate::rng::Mt;
use std::collections::HashMap;

#[derive(Clone, Default)]
pub struct Sample {
    /// x[row][variable]
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl Sample {
    pub fn len(&self) -> usize {
        self.y.len()
    }
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
}

/// A parsed table. Numeric cells appear in `rows`; the original text of every
/// cell is kept in `raw` so date and label columns remain usable.
#[derive(Clone)]
pub struct Table {
    pub names: Vec<String>,
    /// `numeric[j]` is true when every non-empty cell in column `j` parses as a
    /// number.
    pub numeric: Vec<bool>,
    /// Numeric view; non-numeric cells are `NaN`.
    pub rows: Vec<Vec<f64>>,
    /// Text view of every cell.
    pub raw: Vec<Vec<String>>,
}

impl Table {
    pub fn index_of(&self, name: &str) -> Result<usize, String> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("column '{}' not found in {:?}", name, self.names))
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of distinct values in a column (used by auto-detection).
    pub fn distinct_count(&self, col: usize) -> usize {
        let mut seen: Vec<&str> = Vec::new();
        for r in &self.raw {
            let s = r[col].trim();
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen.len()
    }
}

pub fn read_table(path: &str) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    parse_table(&text)
}

/// Parse a delimited table (header row required, `,`/tab/whitespace detected).
pub fn parse_table(text: &str) -> Result<Table, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| "empty input".to_string())?;
    let delim = if header.contains('\t') {
        '\t'
    } else if header.contains(',') {
        ','
    } else {
        ' '
    };
    let split = |l: &str| -> Vec<String> {
        if delim == ' ' {
            l.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            l.split(delim).map(|s| s.trim().to_string()).collect()
        }
    };
    let names = split(header);
    let mut raw: Vec<Vec<String>> = Vec::new();
    for l in lines {
        let parts = split(l);
        if parts.len() != names.len() {
            continue;
        }
        raw.push(parts);
    }
    if raw.is_empty() {
        return Err("no data rows".to_string());
    }

    // A column is numeric when every non-empty cell parses as a number.
    let mut numeric = vec![false; names.len()];
    for (j, slot) in numeric.iter_mut().enumerate() {
        let mut any = false;
        let mut all = true;
        for r in &raw {
            let cell = r[j].trim();
            if cell.is_empty() {
                continue;
            }
            any = true;
            if cell.parse::<f64>().is_err() {
                all = false;
                break;
            }
        }
        *slot = any && all;
    }

    let rows: Vec<Vec<f64>> = raw
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(j, c)| {
                    if numeric[j] {
                        c.trim().parse::<f64>().unwrap_or(f64::NAN)
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();

    Ok(Table {
        names,
        numeric,
        rows,
        raw,
    })
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitMode {
    Random,
    Temporal,
    Group,
    TemporalGroup,
}

impl SplitMode {
    pub fn name(self) -> &'static str {
        match self {
            SplitMode::Random => "random",
            SplitMode::Temporal => "temporal",
            SplitMode::Group => "group",
            SplitMode::TemporalGroup => "temporal-group",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(SplitMode::Random),
            "temporal" | "time" => Ok(SplitMode::Temporal),
            "group" | "grouped" => Ok(SplitMode::Group),
            "temporal-group" | "temporal_group" | "panel" => Ok(SplitMode::TemporalGroup),
            other => Err(format!(
                "unknown split mode '{}' (expected random, temporal, group or temporal-group)",
                other
            )),
        }
    }
}

#[derive(Clone)]
pub struct SplitSpec {
    pub mode: SplitMode,
    /// Index of the time/ordering column (`temporal*` modes).
    pub time_col: Option<usize>,
    /// Index of the group column (`group` and `temporal_group` modes).
    pub group_col: Option<usize>,
    pub test_frac: f64,
    /// Rows dropped immediately after the train/test boundary (embargo).
    pub gap: usize,
    /// Number of contiguous evaluation blocks in the test region.
    pub folds: usize,
}

impl Default for SplitSpec {
    fn default() -> Self {
        SplitSpec {
            mode: SplitMode::Random,
            time_col: None,
            group_col: None,
            test_frac: 0.25,
            gap: 0,
            folds: 1,
        }
    }
}

/// What the split actually did, for reporting.
#[derive(Clone, Default)]
pub struct SplitReport {
    pub mode: &'static str,
    pub rows_total: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    pub folds: usize,
    pub gap: usize,
    pub groups_total: usize,
    pub groups_train: usize,
    pub groups_test: usize,
    pub train_time_max: Option<f64>,
    pub test_time_min: Option<f64>,
    pub notes: Vec<String>,
}

pub struct Prepared {
    pub train: Sample,
    /// All test folds concatenated, in fold order (what the search evaluates on).
    pub test: Sample,
    /// One sample per evaluation block. With `folds == 1` this is a single
    /// sample identical to `test`.
    pub fold_samples: Vec<Sample>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub report: SplitReport,
}

/// Build normalised train/test samples. Returns (train, test, means, stds).
///
/// Backwards-compatible wrapper around [`prepare_full`] using a random split.
pub fn prepare(
    t: &Table,
    vars: &[String],
    target: &str,
    test_frac: f64,
    rng: &mut Mt,
) -> Result<(Sample, Sample, Vec<f64>, Vec<f64>), String> {
    let spec = SplitSpec {
        test_frac,
        ..Default::default()
    };
    let p = prepare_full(t, vars, target, &spec, rng)?;
    Ok((p.train, p.test, p.means, p.stds))
}

pub fn prepare_full(
    t: &Table,
    vars: &[String],
    target: &str,
    spec: &SplitSpec,
    rng: &mut Mt,
) -> Result<Prepared, String> {
    let var_idx: Vec<usize> = vars
        .iter()
        .map(|v| {
            let j = t.index_of(v)?;
            if !t.numeric[j] {
                return Err(format!(
                    "column '{}' is not numeric and cannot be used as an input variable",
                    v
                ));
            }
            if v == target {
                return Err(format!("column '{}' is both the target and an input", v));
            }
            Ok(j)
        })
        .collect::<Result<_, _>>()?;
    let tgt_idx = t.index_of(target)?;
    if !t.numeric[tgt_idx] {
        return Err(format!("target column '{}' is not numeric", target));
    }
    let nv = var_idx.len();

    let x_raw: Vec<Vec<f64>> = t
        .rows
        .iter()
        .map(|r| var_idx.iter().map(|&i| r[i]).collect())
        .collect();
    let y: Vec<f64> = t.rows.iter().map(|r| r[tgt_idx]).collect();

    let (train_idx, fold_idx, report) = build_split(t, spec, rng, &x_raw)?;
    if train_idx.is_empty() {
        return Err("training split is empty (reduce --test-frac)".to_string());
    }

    // Normalisation statistics come from the training rows alone.
    let mut means = vec![0.0; nv];
    let mut stds = vec![1.0; nv];
    for j in 0..nv {
        let m = train_idx.iter().map(|&i| x_raw[i][j]).sum::<f64>() / train_idx.len() as f64;
        let var = train_idx
            .iter()
            .map(|&i| (x_raw[i][j] - m).powi(2))
            .sum::<f64>()
            / train_idx.len() as f64;
        let s = var.sqrt();
        means[j] = m;
        stds[j] = if s < 1e-12 { 1.0 } else { s };
    }
    let scale = |i: usize| -> Vec<f64> {
        (0..nv).map(|j| (x_raw[i][j] - means[j]) / stds[j]).collect()
    };
    let make = |idx: &[usize]| -> Sample {
        let mut s = Sample::default();
        for &i in idx {
            s.x.push(scale(i));
            s.y.push(y[i]);
        }
        s
    };

    let train = make(&train_idx);
    let fold_samples: Vec<Sample> = fold_idx.iter().map(|f| make(f)).collect();
    let mut test = Sample::default();
    for f in &fold_samples {
        test.x.extend(f.x.iter().cloned());
        test.y.extend(f.y.iter().copied());
    }

    Ok(Prepared {
        train,
        test,
        fold_samples,
        means,
        stds,
        report,
    })
}

/// Decide where each row goes. Returns `(train indices, test fold indices, report)`.
fn build_split(
    t: &Table,
    spec: &SplitSpec,
    rng: &mut Mt,
    x_raw: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<Vec<usize>>, SplitReport), String> {
    let n = t.rows.len();
    let folds = spec.folds.max(1);
    let mut notes: Vec<String> = Vec::new();
    let mut report = SplitReport {
        mode: spec.mode.name(),
        rows_total: n,
        folds,
        gap: spec.gap,
        ..Default::default()
    };

    let (train, fold_idx) = match spec.mode {
        SplitMode::Random => {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.usize_below(i + 1);
                idx.swap(i, j);
            }
            let test_n = ((n as f64) * spec.test_frac).round() as usize;
            let test_n = test_n.clamp(if spec.test_frac > 0.0 { 1 } else { 0 }, n.saturating_sub(1));
            let test = idx[..test_n].to_vec();
            let train = idx[test_n..].to_vec();
            (train, split_blocks(&test, folds))
        }
        SplitMode::Temporal => {
            let time_col = spec
                .time_col
                .ok_or("temporal split needs a time column (--time <col>)")?;
            let keys = time_keys(t, time_col);
            let order = sorted_by(&keys);
            match split_ordered(&order, spec, folds) {
                Some((tr, folds_idx)) => {
                    report.train_time_max = order
                        .get(tr.len().saturating_sub(1))
                        .map(|&i| keys[i]);
                    report.test_time_min = folds_idx
                        .iter()
                        .flatten()
                        .map(|&i| keys[i])
                        .fold(None, |acc: Option<f64>, v| {
                            Some(acc.map_or(v, |a| a.min(v)))
                        });
                    (tr, folds_idx)
                }
                None => {
                    notes.push(format!(
                        "too few rows for a temporal split{}; all rows used for training",
                        if spec.gap > 0 {
                            " with this gap"
                        } else {
                            ""
                        }
                    ));
                    (order, Vec::new())
                }
            }
        }
        SplitMode::Group => {
            let group_col = spec
                .group_col
                .ok_or("group split needs a group column (--group-by <col>)")?;
            let (ids, names) = group_ids(t, group_col);
            report.groups_total = names.len();
            // Deterministic, balanced assignment: order groups by a hash of the
            // name, then move whole groups to the test side until it is big
            // enough. No group ever appears on both sides.
            let mut order: Vec<usize> = (0..names.len()).collect();
            let hashes: Vec<u64> = names.iter().map(|s| hash_str(s, 0x5eed_5eed)).collect();
            order.sort_by_key(|&g| hashes[g]);
            let target_test = ((n as f64) * spec.test_frac).round() as usize;
            let mut is_test = vec![false; names.len()];
            let mut acc = 0usize;
            let counts: Vec<usize> = (0..names.len())
                .map(|g| ids.iter().filter(|&&i| i == g).count())
                .collect();
            for &g in &order {
                if acc >= target_test {
                    break;
                }
                is_test[g] = true;
                acc += counts[g];
            }
            if is_test.iter().all(|&b| b) {
                if let Some(&g) = order.first() {
                    is_test[g] = false;
                    notes.push("only one group; kept it in training".to_string());
                }
            }
            if is_test.iter().all(|&b| !b) {
                if let Some(&g) = order.last() {
                    is_test[g] = true;
                    notes.push("kept at least one group for testing".to_string());
                }
            }
            let train: Vec<usize> = (0..n).filter(|&i| !is_test[ids[i]]).collect();
            let test_groups: Vec<usize> = order.iter().copied().filter(|&g| is_test[g]).collect();
            let ids = &ids;
            let test: Vec<usize> = test_groups
                .iter()
                .flat_map(|&g| (0..n).filter(move |&i| ids[i] == g))
                .collect();
            report.groups_train = names.len() - test_groups.len();
            report.groups_test = test_groups.len();
            (train, split_blocks(&test, folds))
        }
        SplitMode::TemporalGroup => {
            let time_col = spec
                .time_col
                .ok_or("temporal-group split needs a time column (--time <col>)")?;
            let group_col = spec
                .group_col
                .ok_or("temporal-group split needs a group column (--group-by <col>)")?;
            let keys = time_keys(t, time_col);
            let (ids, names) = group_ids(t, group_col);
            report.groups_total = names.len();
            let mut train: Vec<usize> = Vec::new();
            let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); folds];
            let mut groups_train = 0usize;
            for g in 0..names.len() {
                let mut rows: Vec<usize> = (0..n).filter(|&i| ids[i] == g).collect();
                rows.sort_by(|&a, &b| {
                    keys[a]
                        .partial_cmp(&keys[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                match split_ordered(&rows, spec, folds) {
                    Some((tr, folds_idx)) => {
                        if !tr.is_empty() {
                            groups_train += 1;
                        }
                        train.extend(tr);
                        for (k, blk) in folds_idx.into_iter().enumerate() {
                            if k < blocks.len() {
                                blocks[k].extend(blk);
                            }
                        }
                    }
                    None => {
                        notes.push(format!(
                            "group '{}' has too few rows for a temporal split; all of it is in training",
                            names[g]
                        ));
                        train.extend(rows);
                        groups_train += 1;
                    }
                }
            }
            let blocks: Vec<Vec<usize>> = blocks.into_iter().filter(|b| !b.is_empty()).collect();
            report.groups_train = groups_train;
            report.groups_test = names.len() - groups_train;
            report.folds = blocks.len().max(1);
            (train, blocks)
        }
    };

    report.train_rows = train.len();
    report.test_rows = fold_idx.iter().map(|f| f.len()).sum();
    report.folds = fold_idx.len();
    report.notes = notes;
    let _ = x_raw;
    Ok((train, fold_idx, report))
}

/// Split one time-ordered sequence into a training prefix and `folds`
/// contiguous test blocks, leaving `gap` rows as an embargo after the cut.
fn split_ordered(order: &[usize], spec: &SplitSpec, folds: usize) -> Option<(Vec<usize>, Vec<Vec<usize>>)> {
    let m = order.len();
    if m < 2 {
        return None;
    }
    let test_n = ((m as f64) * spec.test_frac).round() as usize;
    let test_n = test_n.clamp(1, m - 1);
    let train_n = m - test_n;
    let from = (train_n + spec.gap).min(m - 1);
    let tail = &order[from..];
    if tail.is_empty() {
        return None;
    }
    Some((order[..train_n].to_vec(), split_blocks(tail, folds)))
}

/// Chop a sequence into `folds` contiguous, near-equal, non-empty blocks.
fn split_blocks(idx: &[usize], folds: usize) -> Vec<Vec<usize>> {
    let folds = folds.max(1).min(idx.len().max(1));
    let mut out = Vec::with_capacity(folds);
    for k in 0..folds {
        let a = idx.len() * k / folds;
        let b = idx.len() * (k + 1) / folds;
        if b > a {
            out.push(idx[a..b].to_vec());
        }
    }
    if out.is_empty() {
        out.push(idx.to_vec());
    }
    out
}

fn sorted_by(keys: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .partial_cmp(&keys[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Comparable ordering key for every row of a time column.
///
/// Numbers and dates are parsed to seconds; if any value cannot be parsed the
/// whole column falls back to lexicographic order, which is still a valid total
/// order (and the right one for zero-padded `YYYY-MM-DD`).
pub fn time_keys(t: &Table, col: usize) -> Vec<f64> {
    let n = t.rows.len();
    let parsed: Option<Vec<f64>> = (0..n).map(|r| parse_time(&t.raw[r][col])).collect();
    if let Some(v) = parsed {
        return v;
    }
    let mut pairs: Vec<(String, usize)> = (0..n)
        .map(|r| (t.raw[r][col].trim().to_string(), r))
        .collect();
    pairs.sort();
    let mut key = vec![0.0; n];
    for (rank, (_s, r)) in pairs.into_iter().enumerate() {
        key[r] = rank as f64;
    }
    key
}

/// Parse a timestamp to seconds: accepts plain numbers, `YYYY-MM-DD`,
/// `YYYY/MM/DD` and an optional `T`/space separated `HH:MM[:SS]`.
pub fn parse_time(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let (date, clock) = match s.split_once(['T', ' ']) {
        Some((d, c)) => (d, Some(c)),
        None => (s, None),
    };
    let norm: String = date
        .chars()
        .map(|c| if c == '/' { '-' } else { c })
        .collect();
    let parts: Vec<&str> = norm.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y: i64 = parts[0].parse().ok()?;
    let mo: i64 = parts[1].parse().ok()?;
    let d: i64 = parts[2].parse().ok()?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, mo, d) as f64 * 86400.0;
    if let Some(c) = clock {
        let tp: Vec<&str> = c.split(':').collect();
        let hh: f64 = tp.first().and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        let mm: f64 = tp.get(1).and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        let ss: f64 = tp.get(2).and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        secs += hh * 3600.0 + mm * 60.0 + ss;
    }
    Some(secs)
}

/// Days since the Unix epoch (Howard Hinnant's `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Stable group ids (0..g) from a label or numeric column, plus the group names.
pub fn group_ids(t: &Table, col: usize) -> (Vec<usize>, Vec<String>) {
    let mut map: HashMap<&str, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut ids = Vec::with_capacity(t.rows.len());
    for r in &t.raw {
        let key = r[col].trim();
        let id = match map.get(key) {
            Some(&i) => i,
            None => {
                let i = names.len();
                names.push(key.to_string());
                map.insert(key, i);
                i
            }
        };
        ids.push(id);
    }
    (ids, names)
}

// ---------------------------------------------------------------------------
// History and delayed variables (time-delay embedding)
// ---------------------------------------------------------------------------

/// Options for turning a table into a time-delay embedding.
#[derive(Clone, Default)]
pub struct LagSpec {
    /// Number of delayed copies added per input variable (`x_lag1 .. x_lagN`).
    pub lags: usize,
    /// Fraction of each series kept as *history*: its leading rows are dropped
    /// from the scored data but still supply the past for later rows.
    pub history_frac: f64,
    /// Start a new series when the time step exceeds this multiple of the
    /// series' median step (`0` disables gap splitting).
    pub gap_break: f64,
}

#[derive(Clone, Default)]
pub struct LagReport {
    pub lags: usize,
    pub series: usize,
    pub rows_in: usize,
    pub rows_kept: usize,
    pub history_rows: usize,
    pub breaks: usize,
    pub notes: Vec<String>,
}

/// Append `x_lag1 .. x_lagK` columns for every input variable.
///
/// The past of a row is its predecessor **inside the same series**, where a
/// series is a group (when `--group-by` is set) broken further at time gaps.
/// The first `max(K, history_frac · series_length)` rows of every series are
/// dropped: they cannot be scored without looking before the start of the
/// series, and the rest of the history reserve is exactly the "Maximum History
/// Data" idea — rows that only exist to give later rows a past.
pub fn add_lags(
    t: &Table,
    vars: &[String],
    lag: &LagSpec,
    time_col: Option<usize>,
    group_col: Option<usize>,
) -> Result<(Table, Vec<String>, LagReport), String> {
    let n = t.rows.len();
    let mut report = LagReport {
        lags: lag.lags,
        rows_in: n,
        ..Default::default()
    };
    if lag.lags == 0 && lag.history_frac <= 0.0 {
        report.rows_kept = n;
        return Ok((t.clone(), vars.to_vec(), report));
    }
    let var_idx: Vec<usize> = vars
        .iter()
        .map(|v| t.index_of(v))
        .collect::<Result<_, _>>()?;

    // Series id per row, and the ordering key used inside a series.
    let (gid, group_count) = match group_col {
        Some(c) => {
            let (ids, names) = group_ids(t, c);
            (ids, names.len())
        }
        None => (vec![0usize; n], 1),
    };
    let keys: Vec<f64> = match time_col {
        Some(c) => time_keys(t, c),
        None => (0..n).map(|i| i as f64).collect(),
    };

    let mut by_series: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    for i in 0..n {
        by_series[gid[i]].push(i);
    }

    // Walk each series in time order, cutting it into segments at big steps,
    // and record each row's segment and position inside it.
    let mut seg_id = vec![usize::MAX; n];
    let mut seg_pos = vec![0usize; n];
    let mut seg_rows: Vec<Vec<usize>> = Vec::new();
    for rows in by_series.iter_mut() {
        rows.sort_by(|&a, &b| {
            keys[a]
                .partial_cmp(&keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if rows.is_empty() {
            continue;
        }
        let dts: Vec<f64> = if lag.gap_break > 0.0 && rows.len() > 1 {
            rows.windows(2).map(|w| keys[w[1]] - keys[w[0]]).collect()
        } else {
            Vec::new()
        };
        let threshold = {
            let mut sorted = dts.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            match sorted.get(sorted.len() / 2) {
                Some(&med) if med > 0.0 => Some(lag.gap_break * med),
                _ => None,
            }
        };
        let mut cur = usize::MAX;
        for (k, &r) in rows.iter().enumerate() {
            let starts = k == 0
                || match threshold {
                    Some(th) if dts[k - 1] > th => {
                        report.breaks += 1;
                        true
                    }
                    _ => false,
                };
            if starts {
                cur = seg_rows.len();
                seg_rows.push(Vec::new());
            }
            seg_id[r] = cur;
            seg_pos[r] = seg_rows[cur].len();
            seg_rows[cur].push(r);
        }
    }
    report.series = seg_rows.len();

    // Keep a row only once its segment has enough past for every lag and the
    // requested history reserve.
    let mut keep = vec![false; n];
    for i in 0..n {
        let len = seg_rows[seg_id[i]].len() as f64;
        let reserve = lag.lags.max((lag.history_frac * len).ceil() as usize);
        if seg_pos[i] >= reserve {
            keep[i] = true;
        } else {
            report.history_rows += 1;
        }
    }

    // Names for the new columns, in the same variable order.
    let mut names = t.names.clone();
    let mut new_vars = vars.to_vec();
    for v in vars {
        for k in 1..=lag.lags {
            let name = format!("{}_lag{}", v, k);
            names.push(name.clone());
            new_vars.push(name);
        }
    }
    let mut numeric = t.numeric.clone();
    numeric.extend(std::iter::repeat(true).take(names.len() - numeric.len()));

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut raw: Vec<Vec<String>> = Vec::new();
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let mut r = t.rows[i].clone();
        let mut rr = t.raw[i].clone();
        for &j in &var_idx {
            for k in 1..=lag.lags {
                let v = if seg_pos[i] >= k {
                    let src = seg_rows[seg_id[i]][seg_pos[i] - k];
                    t.rows[src][j]
                } else {
                    f64::NAN
                };
                r.push(v);
                rr.push(format!("{}", v));
            }
        }
        rows.push(r);
        raw.push(rr);
    }

    report.rows_kept = rows.len();
    if report.rows_kept == 0 {
        return Err("history/delays removed every row (reduce --lags/--history-frac)".to_string());
    }
    if report.breaks > 0 {
        report.notes.push(format!(
            "{} time gap(s) treated as series boundaries",
            report.breaks
        ));
    }
    report.notes.push(format!(
        "{} series, {} history row(s) dropped, {} delayed column(s) added",
        report.series,
        report.history_rows,
        lag.lags * vars.len()
    ));

    Ok((
        Table {
            names,
            numeric,
            rows,
            raw,
        },
        new_vars,
        report,
    ))
}

fn hash_str(s: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// True when column `j` looks like a time/ordering axis (by name, or because it
/// is monotone non-decreasing and almost all distinct).
pub fn is_time_like(t: &Table, j: usize) -> bool {
    let name = t.names[j].to_lowercase();
    if ["time", "date", "timestamp", "datetime", "ts", "day", "dt"]
        .iter()
        .any(|k| name == *k || name.contains(k))
    {
        return true;
    }
    if !t.numeric[j] {
        return false;
    }
    let col: Vec<f64> = t.rows.iter().map(|r| r[j]).collect();
    let mono = col.windows(2).all(|w| w[0] <= w[1]);
    mono && t.distinct_count(j) * 5 >= col.len() * 4
}

/// True when column `j` looks like a group label: text, or a low-cardinality
/// numeric column.
pub fn is_group_like(t: &Table, j: usize) -> bool {
    let d = t.distinct_count(j);
    if !t.numeric[j] {
        return d >= 2;
    }
    d >= 2 && d * 5 <= t.rows.len().max(5) + 1
}

/// First column (excluding `exclude`) that looks like a time axis.
pub fn find_time_col(t: &Table, exclude: &[usize]) -> Option<usize> {
    (0..t.names.len())
        .find(|&j| !exclude.contains(&j) && is_time_like(t, j))
}

/// First column (excluding `exclude`) that looks like a group label.
pub fn find_group_col(t: &Table, exclude: &[usize]) -> Option<usize> {
    (0..t.names.len())
        .find(|&j| !exclude.contains(&j) && is_group_like(t, j))
}

/// Pick a split mode from the shape of the table.
///
/// Returns `(mode, time column, group column, notes)`. The caller decides
/// whether to accept it; nothing is changed silently.
pub fn suggest_split(
    t: &Table,
    target_idx: usize,
    var_idx: &[usize],
) -> (SplitMode, Option<usize>, Option<usize>, Vec<String>) {
    let mut notes: Vec<String> = Vec::new();
    let mut exclude: Vec<usize> = vec![target_idx];
    exclude.extend_from_slice(var_idx);

    let time_col = find_time_col(t, &exclude);
    if let Some(j) = time_col {
        exclude.push(j);
    }
    let group_col = find_group_col(t, &exclude);

    let mode = match (time_col, group_col) {
        (Some(_), Some(_)) => {
            notes.push(
                "found both a time-like and a group-like column; using a temporal split inside each group"
                    .to_string(),
            );
            SplitMode::TemporalGroup
        }
        (Some(_), None) => {
            notes.push("found a time-like column; using a chronological split".to_string());
            SplitMode::Temporal
        }
        (None, Some(_)) => {
            notes.push("found a group-like column; holding out whole groups".to_string());
            SplitMode::Group
        }
        (None, None) => {
            notes.push("no time or group column found; using a random split".to_string());
            SplitMode::Random
        }
    };
    (mode, time_col, group_col, notes)
}
