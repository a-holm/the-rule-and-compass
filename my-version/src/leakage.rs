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
