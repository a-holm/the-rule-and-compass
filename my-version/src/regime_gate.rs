//! #1180: the causal regime gate.
//!
//! A gate is an expression of columns of the row, read with hysteresis and a
//! minimum dwell per group. Its state per row is the regime mask of
//! `--pnl-mask`; the composition and the gate search read the same function, so
//! the switch a specialist is trained under is the switch the composite uses.
//!
//! Causality: the value of row `t` reads only the columns of row `t`, and the
//! state of row `t` reads only the values of rows `0..=t` of its group. An
//! aggregate, meta or building-block operator is refused: an aggregate reads a
//! whole column, the future of every row included.
//!
//! Book semantics at the mask edges (see [`crate::trading::FitnessBook`]): a row
//! with state off is flat; every maximal run of on rows of a group is its own
//! book that starts flat and pays its exit at the run's last row.

use crate::data::TradingSlice;
use crate::expr::{self, Node};
use std::collections::HashMap;

/// A gate: expression source, the columns it reads, hysteresis and dwell.
#[derive(Clone, Debug)]
pub struct GateSpec {
    pub source: String,
    /// Column names in the order of the expression's inputs.
    pub columns: Vec<String>,
    pub tree: Node,
    /// Hysteresis half-width: on above `+band`, off below `-band`, else the
    /// previous state. 0 is the plain sign gate.
    pub band: f64,
    /// Minimum rows per state, per group, from the group's first row (>= 1).
    pub dwell: usize,
}

/// Names of the columns an expression reads, in order of first use. A name
/// followed by `(` is a function; `pi`, `e` and the reserved words are not
/// columns.
pub fn referenced_columns(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_digit() || (c == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                    j += 1;
                }
                if j < b.len() && b[j].is_ascii_digit() {
                    while j < b.len() && b[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                }
            }
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let name = &src[start..i];
            let mut j = i;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            let call = b.get(j) == Some(&b'(');
            let reserved = matches!(name, "pi" | "e" | "and" | "or" | "xor" | "not");
            if !call && !reserved && !out.iter().any(|n| n == name) {
                out.push(name.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn reads_whole_columns(n: &Node) -> bool {
    n.op.is_agg() || n.op.is_meta() || n.op.is_custom() || n.kids.iter().any(reads_whole_columns)
}

impl GateSpec {
    pub fn parse(source: &str, band: f64, dwell: usize) -> Result<GateSpec, String> {
        if !band.is_finite() || band < 0.0 {
            return Err(format!("the gate band must be finite and >= 0, got {}", band));
        }
        if dwell == 0 {
            return Err("the gate dwell must be at least 1 row".to_string());
        }
        let columns = referenced_columns(source);
        let tree = crate::parse::parse(source, &columns).map_err(|e| format!("cannot parse the gate expression '{}': {}", source, e))?;
        if reads_whole_columns(&tree) {
            return Err(format!(
                "the gate expression '{}' uses an aggregate, meta or building-block operator: an aggregate reads the whole column, future rows included, so the gate would not be causal",
                source
            ));
        }
        Ok(GateSpec { source: source.to_string(), columns, tree, band, dwell })
    }

    /// Text that identifies the gate in a book hash.
    pub fn identity(&self) -> String {
        format!("{}|{:016x}|{}", self.source, self.band.to_bits(), self.dwell)
    }

    /// The gate value of every row from the column values (one vector per
    /// [`GateSpec::columns`] entry, all of length `n`).
    pub fn values(&self, columns: &[Vec<f64>], n: usize) -> Result<Vec<f64>, String> {
        if columns.len() != self.columns.len() || columns.iter().any(|c| c.len() != n) {
            return Err("the gate columns do not match the rows".to_string());
        }
        let mut row = vec![0.0f64; columns.len()];
        Ok((0..n)
            .map(|i| {
                for (k, c) in columns.iter().enumerate() {
                    row[k] = c[i];
                }
                expr::eval(&self.tree, &row)
            })
            .collect())
    }
}

/// The gate state of every row. Rows of one group must be in time order;
/// groups may interleave. State and dwell start at the group's first row.
pub fn gate_states(spec: &GateSpec, group: &[u64], time: &[f64], columns: &[Vec<f64>]) -> Result<Vec<bool>, String> {
    let n = group.len();
    if time.len() != n {
        return Err(format!("{} times for {} rows", time.len(), n));
    }
    let values = spec.values(columns, n)?;
    let mut last: HashMap<u64, f64> = HashMap::new();
    for i in 0..n {
        if let Some(prev) = last.insert(group[i], time[i]) {
            if time[i] < prev {
                return Err(format!("rows of group {} are not in time order at row {}", group[i], i));
            }
        }
    }
    Ok(hysteresis_states(&values, group, spec.band, spec.dwell))
}

/// Hysteresis and dwell over a value series, state per group.
///
/// Raw target: with `band == 0`, on iff value > 0; otherwise on above `+band`,
/// off below `-band`, else the previous state. A non-finite value is off. A
/// group starts off; its first row takes the raw target at once. After that a
/// state change waits until the current state has held `dwell` rows (the entry
/// row counts).
pub fn hysteresis_states(values: &[f64], group: &[u64], band: f64, dwell: usize) -> Vec<bool> {
    let mut state: HashMap<u64, (bool, usize)> = HashMap::new();
    let mut out = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let entry = state.get(&group[i]).copied();
        let prev = entry.map_or(false, |(s, _)| s);
        let target = if !v.is_finite() {
            false
        } else if band == 0.0 {
            v > 0.0
        } else if v > band {
            true
        } else if v < -band {
            false
        } else {
            prev
        };
        let next = match entry {
            None => (target, 1),
            Some((s, age)) if target != s && age >= dwell => (target, 1),
            Some((s, age)) => (s, age + 1),
        };
        state.insert(group[i], next);
        out.push(next.0);
    }
    out
}

/// A distinct id for every maximal run of on rows of a group and for every off
/// row, in row order. Rows of one group may interleave with other groups.
pub fn run_ids(group: &[u64], on: &[bool]) -> Vec<u64> {
    let mut open: HashMap<u64, bool> = HashMap::new();
    let mut current: HashMap<u64, u64> = HashMap::new();
    let mut next = 0u64;
    let mut out = Vec::with_capacity(group.len());
    for i in 0..group.len() {
        let continues = on[i] && open.get(&group[i]).copied().unwrap_or(false);
        if !continues {
            current.insert(group[i], next);
            next += 1;
        }
        open.insert(group[i], on[i]);
        out.push(current[&group[i]]);
    }
    out
}

/// Rows on, and maximal on-runs per group.
pub fn run_count(group: &[u64], on: &[bool]) -> (usize, usize) {
    let ids = run_ids(group, on);
    let mut seen = std::collections::HashSet::new();
    let mut rows = 0;
    for i in 0..on.len() {
        if on[i] {
            rows += 1;
            seen.insert(ids[i]);
        }
    }
    (rows, seen.len())
}

/// State changes of a state series and the length of its runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Switches {
    pub periods: usize,
    pub switches: usize,
    /// `switches` per year of the `periods - 1` intervals between the states.
    pub per_year: f64,
    pub mean_on_run: f64,
    pub mean_off_run: f64,
}

/// Switch statistics of a period state series in time order. The observation
/// span is the `periods - 1` intervals between the states, so a gate that
/// flips every 24 hourly periods over 8760 intervals switches 365 times a year.
pub fn switch_stats(states: &[bool], periods_per_year: f64) -> Switches {
    let n = states.len();
    let switches = states.windows(2).filter(|w| w[0] != w[1]).count();
    let per_year = if n >= 2 && periods_per_year > 0.0 {
        switches as f64 * periods_per_year / (n - 1) as f64
    } else {
        f64::NAN
    };
    let (mut on_runs, mut on_len, mut off_runs, mut off_len) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && states[j] == states[i] {
            j += 1;
        }
        if states[i] {
            on_runs += 1;
            on_len += j - i;
        } else {
            off_runs += 1;
            off_len += j - i;
        }
        i = j;
    }
    let mean = |len: usize, runs: usize| if runs == 0 { f64::NAN } else { len as f64 / runs as f64 };
    Switches { periods: n, switches, per_year, mean_on_run: mean(on_len, on_runs), mean_off_run: mean(off_len, off_runs) }
}

/// The slice priced under a mask: every on-run is its own group (flat start,
/// exit paid at the run's last row) and the return of an off row is NaN, so the
/// pricing drops it as an unusable row.
pub fn masked_slice(slice: &TradingSlice, on: &[bool]) -> TradingSlice {
    let mut out = slice.clone();
    if on.iter().all(|&o| o) {
        return out;
    }
    out.rows.group = run_ids(&slice.rows.group, on);
    for (r, &keep) in out.ret.iter_mut().zip(on) {
        if !keep {
            *r = f64::NAN;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_names_skip_functions_numbers_and_constants() {
        assert_eq!(referenced_columns("btc_r672"), vec!["btc_r672"]);
        assert_eq!(referenced_columns("0 - btc_r672"), vec!["btc_r672"]);
        assert_eq!(referenced_columns("greater(a, 1e-3) and not b"), vec!["a", "b"]);
        assert_eq!(referenced_columns("max(x1, x2) + pi * e"), vec!["x1", "x2"]);
        assert!(referenced_columns("1").is_empty());
    }

    #[test]
    fn the_gate_refuses_aggregates() {
        assert!(GateSpec::parse("mean(a)", 0.0, 1).is_err());
        assert!(GateSpec::parse("a - mean(a)", 0.0, 1).is_err());
        assert!(GateSpec::parse("a", -1.0, 1).is_err());
        assert!(GateSpec::parse("a", 0.0, 0).is_err());
        assert!(GateSpec::parse("a +", 0.0, 1).is_err());
    }

    #[test]
    fn gate_hysteresis_and_dwell() {
        let g = vec![0u64; 12];
        // band 0.02, dwell 3. Row 0 starts on at once; row 1 (-0.5) wants off
        // but the state has held 1 row; rows 1 and 2 hold; row 3 flips off.
        let v = [0.5, -0.5, -0.5, -0.5, 0.01, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.0];
        let s = hysteresis_states(&v, &g, 0.02, 3);
        let expect = [true, true, true, false, false, false, true, true, true, false, false, false];
        assert_eq!(s, expect);
        // no dwell, no band: the plain sign, non-finite off
        let v = [1.0, -1.0, 0.0, f64::NAN, 2.0];
        assert_eq!(hysteresis_states(&v, &[0; 5], 0.0, 1), vec![true, false, false, false, true]);
        // the band keeps the previous state inside +-band
        let v = [0.5, 0.01, -0.01, -0.5, -0.01, 0.01, 0.5];
        assert_eq!(hysteresis_states(&v, &[0; 7], 0.02, 1), vec![true, true, true, false, false, false, true]);
    }

    #[test]
    fn dwell_and_state_are_per_group() {
        let g = [1u64, 2, 1, 2, 1, 2];
        let v = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        // dwell 2: each group keeps its first state for 2 rows and flips at its third row
        let s = hysteresis_states(&v, &g, 0.0, 2);
        assert_eq!(s, vec![true, false, true, false, false, true]);
    }

    #[test]
    fn the_state_of_a_row_reads_only_earlier_rows() {
        let spec = GateSpec::parse("a - b", 0.05, 4).unwrap();
        let n = 300;
        let group: Vec<u64> = (0..n).map(|i| (i % 3) as u64).collect();
        let time: Vec<f64> = (0..n).map(|i| (i / 3) as f64).collect();
        let a: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i * 13 % 97) as f64 - 48.0) / 80.0).collect();
        let full = gate_states(&spec, &group, &time, &[a.clone(), b.clone()]).unwrap();
        for k in [1usize, 7, 100, 299] {
            let part = gate_states(&spec, &group[..k], &time[..k], &[a[..k].to_vec(), b[..k].to_vec()]).unwrap();
            assert_eq!(part, full[..k].to_vec(), "prefix {}", k);
        }
    }

    #[test]
    fn a_group_out_of_time_order_is_refused() {
        let spec = GateSpec::parse("a", 0.0, 1).unwrap();
        let e = gate_states(&spec, &[0, 0], &[2.0, 1.0], &[vec![1.0, 1.0]]).unwrap_err();
        assert!(e.contains("time order"), "{}", e);
    }

    #[test]
    fn runs_split_at_off_rows_and_group_changes() {
        let g = [1u64, 1, 1, 1, 2, 2];
        let on = [true, true, false, true, true, true];
        let ids = run_ids(&g, &on);
        assert_eq!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[2], ids[3]);
        assert_ne!(ids[3], ids[4]);
        assert_eq!(ids[4], ids[5]);
        assert_eq!(run_count(&g, &on), (5, 3));
    }

    #[test]
    fn switch_count_of_a_daily_flip() {
        let states: Vec<bool> = (0..=24 * 365).map(|i| (i / 24) % 2 == 0).collect();
        let s = switch_stats(&states, 8760.0);
        assert_eq!(s.switches, 365);
        assert!((s.per_year - 365.0).abs() < 1e-9, "{}", s.per_year);
        assert!((s.mean_on_run - 24.0).abs() < 0.2 && (s.mean_off_run - 24.0).abs() < 0.2);
        let flat = switch_stats(&[true, true, true], 8760.0);
        assert_eq!((flat.switches, flat.per_year), (0, 0.0));
        assert!(flat.mean_off_run.is_nan());
        assert!(switch_stats(&[true], 8760.0).per_year.is_nan());
    }
}
