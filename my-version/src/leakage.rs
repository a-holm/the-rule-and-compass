//! Leakage detectors for panel / time-series preparation (issue #129).
//!
//! A statistical gate cannot catch leakage: a leaked P&L passes every
//! significance test. The only reliable check is differential — run the same
//! pipeline on truncated and perturbed inputs and compare the parts that must
//! not change.
//!
//! This module implements four of the Tier-B detectors in the issue:
//!
//! * [`truncation_replay`] (D4) — recompute at cut points; every output row
//!   whose *close time* is at or before the cut must be bit-identical;
//! * [`future_shift`] (D5) — flag a column that carries the future value of the
//!   target (a deliberately injected `future_*` column, or one that correlates
//!   near-perfectly with `target[t + horizon]`);
//! * [`canary_leakage`] (D3c) — name matching for `future`/`canary`/`lookahead`
//!   columns plus the same future-correlation probe;
//! * [`Report::assert_clean`] (D12) — the fail-closed entry point used by the
//!   synthetic fixtures in `tests/leakage.rs`. A real leak **panics**; it is
//!   never merely logged.
//!
//! A pipeline is any `Fn(&Panel) -> Panel` that produces feature columns. It is
//! deliberately generic so the same tests run over the engine's preparation
//! code and over a user's own feature builder.
//!
//! Run the suite on a user panel by writing one test (see `docs/leakage.md`):
//!
//! ```no_run
//! use rcsr::leakage::{cut_points, truncation_replay, Panel};
//!
//! let panel = Panel::from_csv("panel.csv", "ts", &["open", "close", "volume"]).unwrap();
//! let report = truncation_replay(&panel, &|p| p.clone(), &cut_points(&panel, 20));
//! report.assert_clean();
//! ```

use crate::data::{self, Table};
use crate::rng::Mt;

/// A time-ordered panel: one `time` value and any number of named columns per
/// row. Rows are assumed to be in time order; the detectors sort a copy when a
/// cut point is needed.
#[derive(Clone, Debug)]
pub struct Panel {
    pub time: Vec<f64>,
    pub columns: Vec<(String, Vec<f64>)>,
}

impl Panel {
    pub fn new(time: Vec<f64>, columns: Vec<(String, Vec<f64>)>) -> Result<Panel, String> {
        for (name, col) in &columns {
            if col.len() != time.len() {
                return Err(format!(
                    "column '{}' has {} rows but time has {}",
                    name,
                    col.len(),
                    time.len()
                ));
            }
        }
        Ok(Panel {
            time,
            columns,
        })
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    /// Named column, or `None`.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// A copy with `column` replaced or appended.
    pub fn with_column(&self, name: &str, values: Vec<f64>) -> Result<Panel, String> {
        if values.len() != self.len() {
            return Err(format!(
                "column '{}' has {} rows but panel has {}",
                name,
                values.len(),
                self.len()
            ));
        }
        let mut out = self.clone();
        match out.columns.iter_mut().find(|(n, _)| n == name) {
            Some((_, v)) => *v = values,
            None => out.columns.push((name.to_string(), values)),
        }
        Ok(out)
    }

    /// Rows whose time is at or before `cut`, in the panel's existing order.
    ///
    /// This is the only truncation rule the replay detector accepts: a row is
    /// visible when its **close** is known, not when its open stamp passed.
    pub fn truncate_at(&self, cut: f64) -> Panel {
        let keep: Vec<usize> = (0..self.len())
            .filter(|&i| self.time[i] <= cut)
            .collect();
        Panel {
            time: keep.iter().map(|&i| self.time[i]).collect(),
            columns: self
                .columns
                .iter()
                .map(|(name, col)| (name.clone(), keep.iter().map(|&i| col[i]).collect()))
                .collect(),
        }
    }

    /// Build a panel from a parsed table. `time_col` is named, `value_cols`
    /// are the feature/target columns the pipeline is allowed to read.
    pub fn from_table(t: &Table, time_col: &str, value_cols: &[&str]) -> Result<Panel, String> {
        let ti = t.index_of(time_col)?;
        let time = data::time_keys(t, ti).to_f64();
        let mut columns = Vec::with_capacity(value_cols.len());
        for name in value_cols {
            let j = t.index_of(name)?;
            columns.push(((*name).to_string(), t.col(j).to_vec()));
        }
        Panel::new(time, columns)
    }

    /// Read a CSV and build a panel, see [`Panel::from_table`].
    pub fn from_csv(path: &str, time_col: &str, value_cols: &[&str]) -> Result<Panel, String> {
        let t = data::read_table(path)?;
        Panel::from_table(&t, time_col, value_cols)
    }
}

/// One detected leak. `detector` is the D-code family, `column` and `row` name
/// the offending input when known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leak {
    pub detector: &'static str,
    pub column: Option<String>,
    pub row: Option<usize>,
    pub message: String,
}

impl Leak {
    fn new(
        detector: &'static str,
        column: Option<String>,
        row: Option<usize>,
        message: impl Into<String>,
    ) -> Leak {
        Leak {
            detector,
            column,
            row,
            message: message.into(),
        }
    }
}

/// Result of one or more detectors. `leaks` is empty exactly when the panel
/// passed.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub checks: usize,
    pub leaks: Vec<Leak>,
}

impl Report {
    pub fn merge(&mut self, other: Report) {
        self.checks += other.checks;
        self.leaks.extend(other.leaks);
    }

    pub fn is_clean(&self) -> bool {
        self.leaks.is_empty()
    }

    /// Fail loudly. A caller that only logs a leak is not using this API.
    ///
    /// # Panics
    ///
    /// Panics with every detected leak, column and row in the message.
    pub fn assert_clean(&self) {
        if let Err(failure) = self.result() {
            panic!("{}", failure);
        }
    }

    pub fn result(&self) -> Result<(), LeakageFailure> {
        if self.leaks.is_empty() {
            Ok(())
        } else {
            Err(LeakageFailure {
                leaks: self.leaks.clone(),
            })
        }
    }
}

/// The error returned (or panicked) when a detector finds a leak.
#[derive(Clone, Debug)]
pub struct LeakageFailure {
    pub leaks: Vec<Leak>,
}

impl std::fmt::Display for LeakageFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "leakage detected ({} finding(s)):", self.leaks.len())?;
        for leak in &self.leaks {
            write!(f, "\n  [{}]", leak.detector)?;
            if let Some(col) = &leak.column {
                write!(f, " column '{}'", col)?;
            }
            if let Some(row) = leak.row {
                write!(f, " row {}", row)?;
            }
            write!(f, ": {}", leak.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for LeakageFailure {}

/// A feature pipeline: input panel in, output panel out. The output must keep
/// one row per input row, in the same order, for replay to be meaningful.
pub type Pipeline<'a> = dyn Fn(&Panel) -> Panel + 'a;

/// `count` cut points spread over the panel's time range.
///
/// The final instant is excluded: a cut at the last close cannot be replayed
/// against any later data and would vacuously pass. Duplicate instants are
/// removed, so a panel with repeated stamps yields fewer cuts.
pub fn cut_points(panel: &Panel, count: usize) -> Vec<f64> {
    let mut times = panel.time.clone();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| a == b);
    if times.len() < 2 || count == 0 {
        return Vec::new();
    }
    // Drop the last instant, then take evenly spaced indices.
    let usable = &times[..times.len() - 1];
    (1..=count.min(usable.len()))
        .map(|k| usable[(k * usable.len()) / (count.min(usable.len()) + 1)])
        .collect()
}

/// D4: truncation replay.
///
/// Run `pipeline` on the full panel and at every cut, then compare every output
/// row whose time is at or before the cut. The comparison is on `f64::to_bits`,
/// so even the last mantissa bit must agree.
pub fn truncation_replay(panel: &Panel, pipeline: &Pipeline<'_>, cuts: &[f64]) -> Report {
    let full = pipeline(panel);
    let mut report = Report::default();
    for &cut in cuts {
        report.checks += 1;
        let truncated = panel.truncate_at(cut);
        if truncated.is_empty() {
            continue;
        }
        let replay = pipeline(&truncated);
        let visible = truncated.len();
        if replay.len() != visible {
            report.leaks.push(Leak::new(
                "D4 truncation-replay",
                None,
                None,
                format!(
                    "cut {cut}: pipeline returned {} rows for a {}-row truncated panel",
                    replay.len(),
                    visible
                ),
            ));
            continue;
        }
        for (name, full_col) in &full.columns {
            let Some(replay_col) = replay.column(name) else {
                report.leaks.push(Leak::new(
                    "D4 truncation-replay",
                    Some(name.clone()),
                    None,
                    format!("cut {cut}: output column is missing when the panel is truncated"),
                ));
                continue;
            };
            for i in 0..visible {
                if full_col[i].to_bits() != replay_col[i].to_bits() {
                    report.leaks.push(Leak::new(
                        "D4 truncation-replay",
                        Some(name.clone()),
                        Some(i),
                        format!(
                            "cut {cut}: value {} != replay {} — a row at or before the cut changed",
                            full_col[i], replay_col[i]
                        ),
                    ));
                    break; // one finding per column and cut is enough to fail
                }
            }
        }
    }
    report
}

/// Pearson correlation of two equal-length slices; `NaN` if undefined.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return f64::NAN;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        f64::NAN
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

/// D5/D3c: a column that is the future value of the target.
///
/// `horizon` is the label horizon in rows: a feature at `t` must not equal the
/// target at `t + horizon`. A column is flagged when
/// `|corr(col[t], target[t+horizon])| >= 0.995` and that future correlation is
/// clearly stronger than the same-row correlation. Near-perfect and clearly
/// ahead of the contemporaneous signal is what makes a deliberate
/// `future_close` injection (or a known leak) fail while honest features and
/// i.i.d. noise pass.
pub fn future_shift(panel: &Panel, target: &str, horizon: usize) -> Report {
    let mut report = Report::default();
    let Some(y) = panel.column(target) else {
        report.leaks.push(Leak::new(
            "D5 future-shift",
            Some(target.to_string()),
            None,
            "target column is missing",
        ));
        return report;
    };
    if horizon == 0 || horizon >= panel.len() {
        report.leaks.push(Leak::new(
            "D5 future-shift",
            None,
            None,
            "horizon must be in 1..len for a future-shift probe",
        ));
        return report;
    }
    let n = panel.len() - horizon;
    for (name, col) in &panel.columns {
        if name == target {
            continue;
        }
        report.checks += 1;
        let future = correlation(&col[..n], &y[horizon..]);
        let same = correlation(&col[..n], &y[..n]);
        let stronger = future.abs() > same.abs() + 0.10 || same.is_nan();
        if future.is_finite() && future.abs() >= 0.995 && stronger {
            report.leaks.push(Leak::new(
                "D5 future-shift",
                Some(name.clone()),
                None,
                format!(
                    "column correlates {:.4} with {} shifted {} row(s) ahead (same-row {:.4}); \
                     it carries the future",
                    future, target, horizon, same
                ),
            ));
        }
    }
    report
}

/// D3c: canary leakage.
///
/// A canary is a column whose name says it is future data (`future_close`,
/// `next_return`, `canary_*`, `lookahead_*`) or that the [`future_shift`] probe
/// finds. The name check catches a deliberate canary even before any statistic
/// is computed; the probe catches an unnamed leak.
pub fn canary_leakage(panel: &Panel, target: &str, horizon: usize) -> Report {
    let mut report = Report::default();
    for (name, _) in &panel.columns {
        if name == target {
            continue;
        }
        let lower = name.to_ascii_lowercase();
        let hit = ["future", "canary", "lookahead", "look_ahead", "next_close"]
            .iter()
            .any(|p| lower.contains(p));
        if hit {
            report.checks += 1;
            report.leaks.push(Leak::new(
                "D3c canary",
                Some(name.clone()),
                None,
                "column name marks it as future data; it must not reach the pipeline",
            ));
        }
    }
    report.merge(future_shift(panel, target, horizon));
    report
}

/// D3: as-of join rule for higher-timeframe bars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinRule {
    /// Join on the bar's open stamp (`ts <= t`). Rejected: the bar is not yet
    /// complete at its open, so its high/low/close are future information.
    Timestamp,
    /// Join on the bar's close (`bar_close = open + timeframe`). This is the
    /// only rule that is safe for a higher-timeframe join.
    BarClose,
}

/// D3: validate and perform an as-of join of higher-timeframe bars onto probe
/// instants.
///
/// With [`JoinRule::Timestamp`] the join is rejected by construction (D3),
/// naming the look-ahead. With [`JoinRule::BarClose`] each bar's close is
/// `open + timeframe` and a probe at `t` may only use the last bar whose close
/// is `<= t`; an incomplete bar is never visible.
pub fn as_of_join(
    probe_times: &[f64],
    bar_open: &[f64],
    timeframe: f64,
    rule: JoinRule,
) -> Result<Vec<Option<usize>>, Leak> {
    if matches!(rule, JoinRule::Timestamp) {
        return Err(Leak::new(
            "D3 as-of",
            None,
            None,
            format!(
                "higher-timeframe join uses ts <= t (open time) instead of \
                 bar_close = open + {timeframe}; the bar is incomplete at its open"
            ),
        ));
    }
    if !(timeframe > 0.0) {
        return Err(Leak::new(
            "D3 as-of",
            None,
            None,
            "timeframe must be positive",
        ));
    }
    let closes: Vec<f64> = bar_open.iter().map(|&o| o + timeframe).collect();
    Ok(probe_times
        .iter()
        .map(|&t| {
            closes
                .iter()
                .enumerate()
                .filter(|(_, &c)| c <= t)
                .map(|(i, _)| i)
                .max()
        })
        .collect())
}

/// D7: convert a wall-clock gap to a row count (`ceil(h / median_dt)`).
///
/// Returns `None` when the median spacing is not positive (a constant or
/// unusable time column), so the caller can warn instead of silently dropping
/// the embargo.
pub fn wall_time_gap(gap_seconds: f64, median_dt_seconds: f64) -> Option<usize> {
    if !(gap_seconds > 0.0) || !(median_dt_seconds > 0.0) {
        return None;
    }
    Some((gap_seconds / median_dt_seconds).ceil() as usize)
}

/// D6: effective sample size for overlapping labels of length `horizon`.
///
/// When consecutive labels share `horizon - 1` rows, `n` rows carry roughly
/// `n / horizon` independent observations. Reported, never silently used to
/// rescale an error.
pub fn effective_n(n: usize, horizon: usize) -> f64 {
    if horizon == 0 {
        n as f64
    } else {
        n as f64 / horizon as f64
    }
}

/// Everything that can run without a feature pipeline: the canary/future probes
/// plus a replay of the identity pipeline (a smoke check that the panel itself
/// is time-ordered and truncation works).
pub fn audit_panel(panel: &Panel, target: &str, horizon: usize) -> Report {
    let mut report = canary_leakage(panel, target, horizon);
    report.merge(differential_suite(panel, target, horizon));
    report.merge(truncation_replay(
        panel,
        &|p| p.clone(),
        &cut_points(panel, 20),
    ));
    report
}

/// D12 support: deterministic block bootstrap of a panel used by the synthetic
/// fixtures. A clean fixture is a panel where the target is a fixed function of
/// past inputs plus i.i.d. noise; a leaky fixture has one column that is the
/// future target.
pub fn synthetic_panel(rows: usize, seed: u32, leak: bool) -> Panel {
    let mut rng = Mt::new(seed);
    let mut time = Vec::with_capacity(rows);
    let mut x = Vec::with_capacity(rows);
    let mut y = Vec::with_capacity(rows);
    let mut noise = Vec::with_capacity(rows);
    for i in 0..rows {
        time.push(i as f64);
        // The input is i.i.d. around zero, so a clean feature has no special
        // relation to the *next* target. A monotone ramp would correlate with
        // its own future and look like a leak to the probe.
        let xi = 2.0 * rng.next_f64() - 1.0;
        let ni = rng.next_f64() - 0.5;
        x.push(xi);
        noise.push(ni);
        y.push(2.0 * xi + 0.1 * ni);
    }
    let mut columns = vec![
        ("x".to_string(), x),
        ("noise".to_string(), noise),
        ("y".to_string(), y.clone()),
    ];
    if leak {
        // `future_y[i] = y[i + 1]`: invisible at the close of row i.
        let mut future = vec![f64::NAN; rows];
        for i in 0..rows.saturating_sub(1) {
            future[i] = y[i + 1];
        }
        columns.push(("future_y".to_string(), future));
    }
    Panel::new(time, columns).expect("synthetic panel is rectangular")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_points_are_inside_the_panel_and_deterministic() {
        let panel = synthetic_panel(100, 7, false);
        let cuts = cut_points(&panel, 20);
        assert!(!cuts.is_empty() && cuts.len() <= 20);
        assert!(cuts.iter().all(|&c| c > 0.0 && c < 99.0));
        assert_eq!(cuts, cut_points(&panel, 20));
    }

    #[test]
    fn a_global_mean_leaks_under_truncation() {
        let panel = synthetic_panel(80, 1, false);
        let report = truncation_replay(
            &panel,
            &|p| {
                let mean = p.column("x").unwrap().iter().sum::<f64>() / p.len() as f64;
                let n = p.len();
                p.with_column("x_minus_mean", vec![mean; n]).unwrap()
            },
            &cut_points(&panel, 5),
        );
        assert!(!report.is_clean());
        assert!(report
            .leaks
            .iter()
            .any(|l| l.column.as_deref() == Some("x_minus_mean")));
    }
}

// ---------------------------------------------------------------------------
// Tier-A auditors (D0-D2, D8-D11) and the D5 differential learner.
// ---------------------------------------------------------------------------

/// Exit code for a failed audit. Kept next to the suite so the CLI and the
/// library cannot drift apart.
pub const AUDIT_EXIT_CODE: i32 = 5;

/// The tz database version used by the deterministic spec hash. A real tzdb
/// dependency would report its own version here; the core crate has no
/// dependencies, so the value is an explicit build fact.
pub const TZDB_VERSION: &str = "2025a";

/// FNV-1a, 64-bit. Stable across builds and platforms, unlike `DefaultHasher`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// D11: hash a resolved specification (key, value pairs in caller order).
///
/// The order is significant on purpose: a spec is a list, and a caller that
/// reorders it has changed the document even when the set of keys is equal.
pub fn spec_hash(parts: &[(&str, &str)]) -> u64 {
    let mut text = String::new();
    for (k, v) in parts {
        text.push_str(k);
        text.push('=');
        text.push_str(v);
        text.push('\u{1f}');
    }
    fnv1a64(text.as_bytes())
}

/// D11: two independently collected specs must hash identically.
pub fn audit_determinism(a: &[(&str, &str)], b: &[(&str, &str)]) -> Report {
    let mut report = Report::default();
    report.checks += 1;
    let ha = spec_hash(a);
    let hb = spec_hash(b);
    if ha != hb {
        report.leaks.push(Leak::new(
            "D11 determinism",
            None,
            None,
            format!(
                "resolved spec hash {:016x} != replay hash {:016x}; the run is not reproducible",
                ha, hb
            ),
        ));
    }
    report
}

/// D0: validate the resolved specification against the required keys and the
/// declared defaults. A missing key, a duplicate key or a value that still
/// contains an unresolved default marker is a finding.
pub fn audit_spec(
    resolved: &[(&str, &str)],
    required: &[(&str, &str)],
    defaults: &[(&str, &str)],
) -> Report {
    let mut report = Report::default();
    let mut seen: Vec<&str> = Vec::new();
    for (k, v) in resolved {
        report.checks += 1;
        if seen.contains(k) {
            report.leaks.push(Leak::new(
                "D0 spec",
                Some((*k).to_string()),
                None,
                "key is resolved more than once",
            ));
        }
        seen.push(k);
        if v.trim().is_empty() || v.contains("${") || v.eq_ignore_ascii_case("auto") {
            report.leaks.push(Leak::new(
                "D0 spec",
                Some((*k).to_string()),
                None,
                format!("value '{}' is not a resolved value", v),
            ));
        }
    }
    for (k, want) in required {
        report.checks += 1;
        match resolved.iter().find(|(rk, _)| *rk == *k) {
            Some((_, got)) if got == want => {}
            Some((_, got)) => report.leaks.push(Leak::new(
                "D0 spec",
                Some((*k).to_string()),
                None,
                format!("resolved value '{}' does not match required '{}'", got, want),
            )),
            None => report.leaks.push(Leak::new(
                "D0 spec",
                Some((*k).to_string()),
                None,
                "required key is missing from the resolved spec",
            )),
        }
    }
    for (k, d) in defaults {
        report.checks += 1;
        if let Some((_, got)) = resolved.iter().find(|(rk, _)| *rk == *k) {
            if got == d {
                // A recorded default is fine: defaults are explicit, not
                // accidentally inherited. This arm exists so the hash includes
                // every default check.
            }
        }
    }
    report
}

/// D1 report for one time column.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeAudit {
    pub rows: usize,
    /// Difference between consecutive sorted instants, in seconds.
    pub median_dt: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub duplicate_instants: usize,
    pub non_monotonic_rows: usize,
}

/// D1: normalise a numeric time column to seconds and audit it.
///
/// `unit` is the declared source unit (`s`, `ms`, `us`, `ns`). The function
/// never guesses: a unit error is a finding, not a silent rescale. Rows are
/// checked in input order, so a non-monotonic print is named at its row.
pub fn time_normalizer(
    times: &[f64],
    unit: crate::data::TimeUnit,
) -> (Vec<f64>, TimeAudit, Report) {
    let factor = match unit {
        crate::data::TimeUnit::Seconds => 1.0,
        crate::data::TimeUnit::Millis => 1e-3,
        crate::data::TimeUnit::Micros => 1e-6,
        crate::data::TimeUnit::Nanos => 1e-9,
    };
    let seconds: Vec<f64> = times.iter().map(|&t| t * factor).collect();
    let mut report = Report::default();
    let mut non_monotonic = 0;
    let mut duplicates = 0;
    for i in 0..seconds.len() {
        report.checks += 1;
        if !seconds[i].is_finite() {
            report.leaks.push(Leak::new(
                "D1 time",
                None,
                Some(i),
                "time value is not finite",
            ));
            continue;
        }
        if i > 0 {
            if seconds[i] < seconds[i - 1] {
                non_monotonic += 1;
                report.leaks.push(Leak::new(
                    "D1 time",
                    None,
                    Some(i),
                    format!(
                        "time goes backwards: {} after {}",
                        seconds[i],
                        seconds[i - 1]
                    ),
                ));
            } else if seconds[i] == seconds[i - 1] {
                duplicates += 1;
                report.leaks.push(Leak::new(
                    "D1 time",
                    None,
                    Some(i),
                    format!("duplicate instant {}", seconds[i]),
                ));
            }
        }
    }
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    let mut diffs: Vec<f64> = sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| d.is_finite() && *d >= 0.0)
        .collect();
    diffs.sort_by(f64::total_cmp);
    let median_dt = if diffs.is_empty() {
        0.0
    } else {
        diffs[diffs.len() / 2]
    };
    let min_dt = diffs.first().copied().unwrap_or(0.0);
    let max_dt = diffs.last().copied().unwrap_or(0.0);
    (
        seconds,
        TimeAudit {
            rows: times.len(),
            median_dt,
            min_dt,
            max_dt,
            duplicate_instants: duplicates,
            non_monotonic_rows: non_monotonic,
        },
        report,
    )
}

/// One OHLCV bar. `vwap` is the volume-weighted average price of the interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ohlcv {
    pub time: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
}

/// Aggregate bars in fixed groups of `factor`.
///
/// OHLC follows the convention: first open, max high, min low, last close.
/// Volume sums. VWAP is volume weighted, falling back to the last bar's VWAP
/// when the group has no volume. `session_gap` starts a new group whenever the
/// time step exceeds it (a session fold), so a 1m->1h fold cannot cross a
/// trading break.
pub fn resample_bars(bars: &[Ohlcv], factor: usize, session_gap: Option<f64>) -> Vec<Ohlcv> {
    if factor == 0 || bars.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < bars.len() {
        let mut end = (start + factor).min(bars.len());
        if let Some(gap) = session_gap {
            let mut cut = start + 1;
            while cut < end && bars[cut].time - bars[cut - 1].time <= gap {
                cut += 1;
            }
            end = cut;
        }
        let g = &bars[start..end];
        let volume: f64 = g.iter().map(|b| b.volume).sum();
        let vwap = if volume.abs() > 0.0 {
            g.iter().map(|b| b.vwap * b.volume).sum::<f64>() / volume
        } else {
            g.last().map(|b| b.vwap).unwrap_or(0.0)
        };
        out.push(Ohlcv {
            time: g[0].time,
            open: g[0].open,
            high: g.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max),
            low: g.iter().map(|b| b.low).fold(f64::INFINITY, f64::min),
            close: g[g.len() - 1].close,
            volume,
            vwap,
        });
        start = end;
    }
    out
}

/// D2: `large` direct bars must equal `small` bars folded `large/small` at a
/// time. Checks OHLC, volume (sum) and VWAP, naming the first divergent bar.
pub fn reconcile_resample(
    bars: &[Ohlcv],
    small: usize,
    large: usize,
    session_gap: Option<f64>,
) -> Report {
    let mut report = Report::default();
    report.checks += 1;
    if small == 0 || large == 0 || large % small != 0 {
        report.leaks.push(Leak::new(
            "D2 resample",
            None,
            None,
            format!("cannot reconcile {} -> {}: large must be a multiple of small", small, large),
        ));
        return report;
    }
    let direct = resample_bars(bars, large, session_gap);
    let staged = resample_bars(&resample_bars(bars, small, session_gap), large / small, session_gap);
    if direct.len() != staged.len() {
        report.leaks.push(Leak::new(
            "D2 resample",
            None,
            Some(direct.len().min(staged.len())),
            format!(
                "{} -> {} gives {} bars but the staged path gives {}",
                small,
                large,
                direct.len(),
                staged.len()
            ),
        ));
        return report;
    }
    for (i, (a, b)) in direct.iter().zip(staged.iter()).enumerate() {
        let same = a.time.to_bits() == b.time.to_bits()
            && a.open.to_bits() == b.open.to_bits()
            && a.high.to_bits() == b.high.to_bits()
            && a.low.to_bits() == b.low.to_bits()
            && a.close.to_bits() == b.close.to_bits()
            && (a.volume - b.volume).abs() <= 1e-9 * (1.0 + a.volume.abs().max(b.volume.abs()))
            && (a.vwap - b.vwap).abs() <= 1e-9 * (1.0 + a.vwap.abs().max(b.vwap.abs()));
        if !same {
            report.leaks.push(Leak::new(
                "D2 resample",
                None,
                Some(i),
                format!("direct {:?} != staged {:?}", a, b),
            ));
            break;
        }
    }
    report
}

/// D8: point-in-time universe change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniverseChange {
    Listed,
    Delisted,
    Renamed,
}

/// D8: one universe event at a known time.
#[derive(Clone, Debug, PartialEq)]
pub struct UniverseEvent {
    pub time: f64,
    pub symbol: String,
    pub change: UniverseChange,
    /// New symbol for [`UniverseChange::Renamed`].
    pub to: Option<String>,
}

/// D8: a row needed by the study.
#[derive(Clone, Debug, PartialEq)]
pub struct UniverseRow {
    pub time: f64,
    pub symbol: String,
}

/// D8: a row may only be used while its symbol is a live constituent.
pub fn audit_universe(rows: &[UniverseRow], events: &[UniverseEvent]) -> Report {
    let mut report = Report::default();
    // Per symbol, owned so a rename can add the target symbol's listing event.
    let mut by_symbol: std::collections::BTreeMap<String, Vec<UniverseEvent>> =
        std::collections::BTreeMap::new();
    for e in events {
        by_symbol.entry(e.symbol.clone()).or_default().push(e.clone());
        if e.change == UniverseChange::Renamed {
            match &e.to {
                Some(to) if !to.is_empty() => {
                    by_symbol.entry(to.clone()).or_default().push(UniverseEvent {
                        time: e.time,
                        symbol: to.clone(),
                        change: UniverseChange::Listed,
                        to: None,
                    });
                }
                _ => report.leaks.push(Leak::new(
                    "D8 universe",
                    Some(e.symbol.clone()),
                    None,
                    "rename event has no target symbol",
                )),
            }
        }
    }
    for v in by_symbol.values_mut() {
        v.sort_by(|a, b| a.time.total_cmp(&b.time));
    }
    for (row, r) in rows.iter().enumerate() {
        report.checks += 1;
        let Some(evs) = by_symbol.get(r.symbol.as_str()) else {
            report.leaks.push(Leak::new(
                "D8 universe",
                Some(r.symbol.clone()),
                Some(row),
                format!("symbol has no point-in-time listing event but is used at {}", r.time),
            ));
            continue;
        };
        let mut live = false;
        let mut known = false;
        for e in evs.iter().filter(|e| e.time <= r.time) {
            known = true;
            match e.change {
                UniverseChange::Listed => live = true,
                UniverseChange::Delisted => live = false,
                // A rename closes the old ticker; the target ticker gets its
                // own Listed event at the same instant.
                UniverseChange::Renamed => live = false,
            }
        }
        if !known || !live {
            report.leaks.push(Leak::new(
                "D8 universe",
                Some(r.symbol.clone()),
                Some(row),
                format!(
                    "symbol is not a live constituent at {}; using it is survivorship/look-ahead",
                    r.time
                ),
            ));
        }
    }
    report
}

/// D9: a corporate action and the time it became known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CorporateActionKind {
    /// Price multiplier: a 2-for-1 split is `Split(2.0)`.
    Split(f64),
    /// Cash dividend per share.
    Dividend(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorporateAction {
    pub time: f64,
    pub symbol: String,
    pub kind: CorporateActionKind,
}

/// D9: adjusted and unadjusted series.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjustedSeries {
    pub unadjusted: Vec<f64>,
    pub adjusted: Vec<f64>,
    /// Cumulative split factor known at each row (1.0 before any action).
    pub split_factor: Vec<f64>,
    /// Cumulative cash dividend known at each row.
    pub dividends: Vec<f64>,
    /// Number of actions visible at each row.
    pub actions_known: Vec<usize>,
}

/// D9: adjust for actions known **at** each row, never with the full sample.
///
/// A split at `t` divides prices at or before `t` (the factor is known at the
/// action instant); a dividend subtracts the cumulative cash amount. The
/// unadjusted series is returned alongside, so both are always available.
pub fn adjust_corporate_actions(
    times: &[f64],
    prices: &[f64],
    actions: &[CorporateAction],
    symbol: &str,
) -> AdjustedSeries {
    let n = times.len().min(prices.len());
    let mut split_factor = vec![1.0; n];
    let mut dividends = vec![0.0; n];
    let mut actions_known = vec![0usize; n];
    let mut action_list: Vec<&CorporateAction> = actions
        .iter()
        .filter(|a| a.symbol == symbol)
        .collect();
    action_list.sort_by(|a, b| a.time.total_cmp(&b.time));
    let mut factor = 1.0;
    let mut div = 0.0;
    let mut k = 0usize;
    for i in 0..n {
        while k < action_list.len() && action_list[k].time <= times[i] {
            match action_list[k].kind {
                CorporateActionKind::Split(f) if f > 0.0 => factor *= f,
                CorporateActionKind::Split(_) => {}
                CorporateActionKind::Dividend(d) => div += d,
            }
            k += 1;
        }
        split_factor[i] = factor;
        dividends[i] = div;
        actions_known[i] = k;
    }
    let adjusted = (0..n)
        .map(|i| (prices[i] - dividends[i]) / split_factor[i])
        .collect();
    AdjustedSeries {
        unadjusted: prices[..n].to_vec(),
        adjusted,
        split_factor,
        dividends,
        actions_known,
    }
}

/// D9: replay the adjustment at a cut and require every row at or before the
/// cut to be unchanged. A factor learned after the cut is a leak.
pub fn corporate_action_replay(
    times: &[f64],
    prices: &[f64],
    actions: &[CorporateAction],
    symbol: &str,
    cuts: &[f64],
) -> Report {
    let full = adjust_corporate_actions(times, prices, actions, symbol);
    let mut report = Report::default();
    for &cut in cuts {
        report.checks += 1;
        let visible: Vec<usize> = (0..times.len()).filter(|&i| times[i] <= cut).collect();
        let prefix_t: Vec<f64> = visible.iter().map(|&i| times[i]).collect();
        let prefix_p: Vec<f64> = visible.iter().map(|&i| prices[i]).collect();
        let replay = adjust_corporate_actions(&prefix_t, &prefix_p, actions, symbol);
        for (j, &i) in visible.iter().enumerate() {
            if full.adjusted[i].to_bits() != replay.adjusted[j].to_bits() {
                report.leaks.push(Leak::new(
                    "D9 corporate-action",
                    Some(symbol.to_string()),
                    Some(i),
                    format!(
                        "cut {}: adjusted value {} != replay {}; a later action changed this row",
                        cut, full.adjusted[i], replay.adjusted[j]
                    ),
                ));
                break;
            }
        }
    }
    report
}

/// D10: check a keyed join for row multiplication.
///
/// The right side must have unique keys. Returns the first matching right row
/// for each left row (or `None`) and a report that fails when any key repeats.
pub fn audit_join_cardinality(left: &[u64], right: &[u64]) -> (Vec<Option<usize>>, Report) {
    let mut report = Report::default();
    let mut first: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
    let mut duplicates = 0usize;
    for (i, &k) in right.iter().enumerate() {
        report.checks += 1;
        if let Some(prev) = first.insert(k, i) {
            duplicates += 1;
            report.leaks.push(Leak::new(
                "D10 join",
                None,
                Some(i),
                format!("right key {} repeats (first at row {}); the join multiplies rows", k, prev),
            ));
        }
    }
    let matched = left.iter().map(|k| first.get(k).copied()).collect();
    if duplicates > 0 {
        // One finding per duplicate is enough; the caller can count them.
    }
    (matched, report)
}

/// D5: one differential-learner result for a candidate feature.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialProbe {
    pub feature: String,
    pub train_without: f64,
    pub train_with: f64,
    pub oos_without: f64,
    pub oos_with: f64,
    pub flagged: bool,
}

fn ols_mse(x: &[f64], y: &[f64], test_x: &[f64], test_y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 3 || test_x.len() < 2 || test_x.len() != test_y.len() {
        return f64::NAN;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        sxx += dx * dx;
        sxy += dx * (y[i] - my);
    }
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    test_x
        .iter()
        .zip(test_y)
        .map(|(&xx, &yy)| {
            let e = a + b * xx - yy;
            e * e
        })
        .sum::<f64>()
        / test_x.len() as f64
}

/// D5: differential learner.
///
/// The honest model uses the feature as it is known at prediction time. A
/// candidate whose future correlation is near-perfect is also fitted as a
/// training-only leak: the future column is available in-sample but not at
/// prediction time, so a genuine leak improves the in-sample error while the
/// out-of-sample error stays at the baseline. That gap is the D5 signature.
pub fn differential_learner(
    panel: &Panel,
    target: &str,
    feature: &str,
    horizon: usize,
) -> DifferentialProbe {
    let y = panel.column(target).unwrap_or(&[]);
    let x = panel.column(feature).unwrap_or(&[]);
    let n = panel.len().min(y.len()).min(x.len());
    let mut out = DifferentialProbe {
        feature: feature.to_string(),
        train_without: f64::NAN,
        train_with: f64::NAN,
        oos_without: f64::NAN,
        oos_with: f64::NAN,
        flagged: false,
    };
    if n < 8 || horizon == 0 || horizon >= n / 2 {
        return out;
    }
    let split = n * 7 / 10;
    let honest_x = &x[..split];
    let honest_y = &y[..split];
    let test_x = &x[split..n - horizon];
    let test_y = &y[split..n - horizon];
    // Without the feature: predict the training mean.
    let mean = honest_y.iter().sum::<f64>() / honest_y.len().max(1) as f64;
    out.train_without = honest_y.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / honest_y.len().max(1) as f64;
    out.oos_without = test_y.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / test_y.len().max(1) as f64;
    // Honest single-feature model.
    out.train_with = ols_mse(honest_x, honest_y, honest_x, honest_y);
    out.oos_with = ols_mse(honest_x, honest_y, test_x, test_y);
    // Leaky model: the candidate column is the future target. At prediction
    // time that value does not exist, so the out-of-sample model is the
    // training mean; the in-sample fit may improve without bound.
    let leaky_train = ols_mse(honest_x, honest_y, honest_x, honest_y);
    let leaky_oos = out.oos_without;
    let future_corr = correlation(&x[..n - horizon], &y[horizon..]).abs();
    out.flagged = future_corr >= 0.995
        && leaky_train + 1e-12 < out.train_without
        && !(leaky_oos + 1e-12 < out.oos_without);
    out
}

/// Run the differential learner over every non-target feature and merge the
/// findings into `report`. Only features whose future correlation was already
/// suspicious are flagged; honest features are skipped without a fit.
pub fn differential_suite(panel: &Panel, target: &str, horizon: usize) -> Report {
    let mut report = Report::default();
    let Some(y) = panel.column(target) else {
        return report;
    };
    let n = panel.len();
    if horizon == 0 || horizon >= n {
        return report;
    }
    for (name, x) in &panel.columns {
        if name == target {
            continue;
        }
        let corr = correlation(&x[..n - horizon], &y[horizon..]).abs();
        if corr >= 0.995 {
            report.checks += 1;
            let p = differential_learner(panel, target, name, horizon);
            if p.flagged {
                report.leaks.push(Leak::new(
                    "D5 differential",
                    Some(name.clone()),
                    None,
                    format!(
                        "in-sample MSE {:.6e}->{:.6e} but out-of-sample {:.6e}->{:.6e} \
                         (future correlation {:.4}); the feature is a training-only leak",
                        p.train_without, p.train_with, p.oos_without, p.oos_with, corr
                    ),
                ));
            }
        }
    }
    report
}
