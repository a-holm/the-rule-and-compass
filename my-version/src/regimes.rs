//! #1175: regime strata of the per-period series of a report.
//!
//! A stratum is a set of periods: the periods whose bucket start lies in one
//! row of a regime calendar, or the periods whose causal regime expression is
//! on (or off). Every stratum reports the same figures as the pooled report
//! (periods, net, Sharpe, drawdown, exposure, alpha over exposure-matched
//! buy-and-hold) computed on its periods concatenated.
//!
//! A calendar is an evaluation input only. A causal expression is evaluated by
//! [`crate::regime_gate`] from columns at or before each row.

use crate::inference;
use crate::regime_gate::{self, GateSpec, Switches};
use crate::walkforward::{equity_metrics, jnum, jstr, sharpe};
use std::collections::HashMap;

pub const UNDATED: &str = "undated";
pub const CALENDAR_HEADER: &str = "label,start,end";

#[derive(Clone, Debug, PartialEq)]
pub struct CalendarRow {
    pub label: String,
    pub start: f64,
    pub end: f64,
}

/// A regime calendar: rows `label,start,end`, UTC half-open, not overlapping.
/// Lines starting with `#` are comments. Any label set is accepted.
#[derive(Clone, Debug)]
pub struct Calendar {
    pub rows: Vec<CalendarRow>,
    pub sha256: String,
}

impl Calendar {
    pub fn parse(text: &str) -> Result<Calendar, String> {
        let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
        match lines.next() {
            Some((_, h)) if h.trim() == CALENDAR_HEADER => {}
            _ => return Err(format!("the calendar needs the header '{}'", CALENDAR_HEADER)),
        }
        let mut rows = Vec::new();
        for (n, line) in lines {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() != 3 || f[0].is_empty() {
                return Err(format!("calendar line {}: expected label,start,end", n + 1));
            }
            if f[0] == UNDATED {
                return Err(format!("calendar line {}: '{}' is reserved for periods outside every row", n + 1, UNDATED));
            }
            let time = |s: &str, what: &str| {
                crate::data::parse_time(s).filter(|t| t.is_finite()).ok_or_else(|| format!("calendar line {}: cannot read {} '{}'", n + 1, what, s))
            };
            let (start, end) = (time(f[1], "start")?, time(f[2], "end")?);
            if !(end > start) {
                return Err(format!("calendar line {}: end must be after start", n + 1));
            }
            rows.push(CalendarRow { label: f[0].to_string(), start, end });
        }
        rows.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
        for w in rows.windows(2) {
            if w[1].start < w[0].end {
                return Err(format!("calendar rows overlap: '{}' and '{}'", w[0].label, w[1].label));
            }
        }
        Ok(Calendar { rows, sha256: crate::manifest::sha256_hex(text.as_bytes()) })
    }

    pub fn load(path: &str) -> Result<Calendar, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        Calendar::parse(&text).map_err(|e| format!("{}: {}", path, e))
    }

    /// The label of the row that contains `t`.
    pub fn label_at(&self, t: f64) -> Option<&str> {
        let k = self.rows.partition_point(|r| r.start <= t);
        k.checked_sub(1).map(|i| &self.rows[i]).filter(|r| t < r.end).map(|r| r.label.as_str())
    }

    /// Labels in order of first appearance.
    pub fn labels(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for r in &self.rows {
            if !out.contains(&r.label) {
                out.push(r.label.clone());
            }
        }
        out
    }

    /// Stratum names (the calendar labels, then `undated` when a period falls
    /// outside every row) and the stratum index of every period.
    pub fn assign(&self, stratum_time: &[f64]) -> (Vec<String>, Vec<usize>) {
        let mut names = self.labels();
        let mut idx = Vec::with_capacity(stratum_time.len());
        let mut undated = None;
        for &t in stratum_time {
            idx.push(match self.label_at(t) {
                Some(l) => names.iter().position(|n| n == l).expect("calendar label"),
                None => *undated.get_or_insert_with(|| {
                    names.push(UNDATED.to_string());
                    names.len() - 1
                }),
            });
        }
        (names, idx)
    }
}

/// The per-period series a stratum is computed from. Only `net` is required;
/// a missing series leaves its figures null.
#[derive(Clone, Copy, Debug, Default)]
pub struct Series<'a> {
    pub net: &'a [f64],
    pub gross: Option<&'a [f64]>,
    pub exposure: Option<&'a [f64]>,
    pub entries: Option<&'a [f64]>,
    pub bh: Option<&'a [f64]>,
}

#[derive(Clone, Copy, Debug)]
pub struct StatOpts {
    pub periods_per_year: f64,
    /// Circular block bootstrap draws of the mean; 0 leaves the interval null.
    pub draws: usize,
    pub block_len: Option<usize>,
    pub seed: u32,
}

#[derive(Clone, Debug)]
pub struct Stratum {
    pub label: String,
    pub source: String,
    pub periods: usize,
    pub years: f64,
    pub net_sum: f64,
    pub cost_sum: Option<f64>,
    pub net_mean: f64,
    pub net_ann: f64,
    pub net_ann_2x: Option<f64>,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown: f64,
    pub exposure: Option<f64>,
    pub entries: Option<f64>,
    pub bh_ann: Option<f64>,
    pub alpha_bh_ann: Option<f64>,
    pub mean_ci: Option<(f64, f64)>,
}

fn pick(v: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| v[i]).collect()
}

/// The figures of one stratum. `idx` lists its periods in time order. A
/// stratum with fewer than 2 periods keeps its counts and has null statistics.
pub fn stratum(label: &str, source: &str, idx: &[usize], s: &Series, o: &StatOpts) -> Stratum {
    let n = idx.len();
    let ppy = o.periods_per_year;
    let net = pick(s.net, idx);
    let enough = n >= 2;
    let nan = f64::NAN;
    let mean = |v: &[f64]| if v.is_empty() { nan } else { v.iter().sum::<f64>() / v.len() as f64 };
    let optional = |v: Option<&[f64]>| v.map(|v| pick(v, idx));
    let gross = optional(s.gross);
    let exposure = optional(s.exposure);
    let bh = optional(s.bh);
    let net_mean = if enough { mean(&net) } else { nan };
    let (max_drawdown, sortino) = if enough {
        (equity_metrics(&net, ppy).2, crate::risk::risk_stats(&net, ppy, crate::risk::DEFAULT_SORTINO_MAR).sortino)
    } else {
        (nan, nan)
    };
    let gate = |x: f64| if enough { Some(x) } else { None };
    let net_ann_2x = gross.as_ref().and_then(|g| {
        let two: Vec<f64> = net.iter().zip(g).map(|(n, g)| 2.0 * n - g).collect();
        gate(mean(&two) * ppy)
    });
    let alpha_bh_ann = match (&exposure, &bh) {
        (Some(e), Some(b)) => {
            let a: Vec<f64> = net.iter().zip(e.iter().zip(b)).map(|(n, (e, b))| n - e * b).collect();
            gate(mean(&a) * ppy)
        }
        _ => None,
    };
    let mean_ci = (enough && o.draws > 0).then(|| {
        let block = o.block_len.unwrap_or_else(|| inference::auto_block_len(n, 0, 1));
        let d = inference::bootstrap_draws(&net, block, o.draws, o.seed);
        let i = inference::percentile_interval(&d.means, 0.95);
        (i.lo, i.hi)
    });
    Stratum {
        label: label.to_string(),
        source: source.to_string(),
        periods: n,
        years: if ppy > 0.0 { n as f64 / ppy } else { nan },
        net_sum: net.iter().sum(),
        cost_sum: gross.as_ref().map(|g| g.iter().zip(&net).map(|(g, n)| g - n).sum()),
        net_mean,
        net_ann: net_mean * ppy,
        net_ann_2x,
        sharpe: if enough { sharpe(&net, ppy) } else { nan },
        sortino,
        max_drawdown,
        exposure: exposure.as_ref().and_then(|e| gate(mean(e))),
        entries: optional(s.entries).map(|e| e.iter().sum()),
        bh_ann: bh.as_ref().and_then(|b| gate(mean(b) * ppy)),
        alpha_bh_ann,
        mean_ci,
    }
}

/// One stratum per name, from the stratum index of every period.
pub fn strata(names: &[String], source: &str, assign: &[usize], s: &Series, o: &StatOpts) -> Vec<Stratum> {
    names
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let idx: Vec<usize> = (0..assign.len()).filter(|&i| assign[i] == k).collect();
            stratum(name, source, &idx, s, o)
        })
        .collect()
}

fn opt(v: Option<f64>) -> String {
    v.map_or("null".to_string(), jnum)
}

impl Stratum {
    pub fn json(&self) -> String {
        format!(
            "{{\"label\":{},\"source\":{},\"periods\":{},\"years\":{},\"net_sum\":{},\"cost_sum\":{},\"net_mean\":{},\
\"net_ann\":{},\"net_ann_2x\":{},\"sharpe\":{},\"sortino\":{},\"max_drawdown\":{},\"exposure\":{},\"entries\":{},\
\"bh_ann\":{},\"alpha_bh_ann\":{},\"mean_ci\":{}}}",
            jstr(&self.label),
            jstr(&self.source),
            self.periods,
            jnum(self.years),
            jnum(self.net_sum),
            opt(self.cost_sum),
            jnum(self.net_mean),
            jnum(self.net_ann),
            opt(self.net_ann_2x),
            jnum(self.sharpe),
            jnum(self.sortino),
            jnum(self.max_drawdown),
            opt(self.exposure),
            opt(self.entries),
            opt(self.bh_ann),
            opt(self.alpha_bh_ann),
            self.mean_ci.map_or("null".to_string(), |(lo, hi)| format!("[{},{}]", jnum(lo), jnum(hi)))
        )
    }
}

pub fn strata_json(v: &[Stratum]) -> String {
    format!("[{}]", v.iter().map(Stratum::json).collect::<Vec<_>>().join(","))
}

fn cell(v: f64) -> String {
    if v.is_finite() { format!("{:.4}", v) } else { "n/a".to_string() }
}

fn cell_opt(v: Option<f64>) -> String {
    v.map_or("n/a".to_string(), cell)
}

/// Markdown table of the strata of one source.
pub fn markdown_table(v: &[Stratum]) -> String {
    let mut s = String::from(
        "| stratum | periods | years | net ann | net ann 2x | Sharpe | Sortino | max drawdown | exposure | entries | buy-and-hold ann | alpha ann | mean CI |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for x in v {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            x.label,
            x.periods,
            cell(x.years),
            cell(x.net_ann),
            cell_opt(x.net_ann_2x),
            cell(x.sharpe),
            cell(x.sortino),
            cell(x.max_drawdown),
            cell_opt(x.exposure),
            x.entries.map_or("n/a".to_string(), |e| format!("{}", e)),
            cell_opt(x.bh_ann),
            cell_opt(x.alpha_bh_ann),
            x.mean_ci.map_or("n/a".to_string(), |(lo, hi)| format!("[{:.6}, {:.6}]", lo, hi)),
        ));
    }
    s
}

/// A causal regime expression with its name.
#[derive(Clone, Debug)]
pub struct ExprSpec {
    pub name: String,
    pub gate: GateSpec,
}

/// Row counts behind the period states of one expression.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowCounts {
    pub rows: usize,
    pub rows_on: usize,
    pub nonfinite: usize,
}

impl RowCounts {
    pub fn add(&mut self, o: RowCounts) {
        self.rows += o.rows;
        self.rows_on += o.rows_on;
        self.nonfinite += o.nonfinite;
    }
}

/// The state of every period from the rows of its bucket: on when more than
/// half of the rows are on. `row_key` is the bucket key of each priced row
/// (the same key the period grid uses), `on` the gate state of the row and
/// `finite` whether the gate value of the row was finite. A period with no row
/// is off.
pub fn period_states(row_key: &[f64], on: &[bool], finite: &[bool], period_key: &[f64]) -> (Vec<bool>, RowCounts) {
    let bits = |k: f64| if k == 0.0 { 0.0f64.to_bits() } else { k.to_bits() };
    let mut count: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut c = RowCounts::default();
    for i in 0..row_key.len() {
        let e = count.entry(bits(row_key[i])).or_insert((0, 0));
        e.1 += 1;
        c.rows += 1;
        if on[i] {
            e.0 += 1;
            c.rows_on += 1;
        }
        if !finite[i] {
            c.nonfinite += 1;
        }
    }
    let states = period_key.iter().map(|&k| count.get(&bits(k)).is_some_and(|&(o, n)| o * 2 > n)).collect();
    (states, c)
}

/// One expression block of the report.
#[derive(Clone, Debug)]
pub struct ExprBlock {
    pub name: String,
    pub source: String,
    pub band: f64,
    pub dwell: usize,
    pub counts: RowCounts,
    pub switches: Switches,
    pub strata: Vec<Stratum>,
}

/// The `regimes` member of a report.
#[derive(Clone, Debug, Default)]
pub struct RegimeReport {
    pub calendar_sha256: Option<String>,
    pub calendar_rows: usize,
    pub calendar: Vec<Stratum>,
    pub exprs: Vec<ExprBlock>,
}

pub fn expr_block(spec: &ExprSpec, states: &[bool], counts: RowCounts, s: &Series, o: &StatOpts) -> ExprBlock {
    let names = vec!["on".to_string(), "off".to_string()];
    let assign: Vec<usize> = states.iter().map(|&on| usize::from(!on)).collect();
    ExprBlock {
        name: spec.name.clone(),
        source: spec.gate.source.clone(),
        band: spec.gate.band,
        dwell: spec.gate.dwell,
        counts,
        switches: regime_gate::switch_stats(states, o.periods_per_year),
        strata: strata(&names, &format!("expr:{}", spec.name), &assign, s, o),
    }
}

impl RegimeReport {
    pub fn json(&self) -> String {
        let cal = match &self.calendar_sha256 {
            Some(h) => format!(
                "{{\"sha256\":{},\"rows\":{},\"strata\":{}}}",
                jstr(h),
                self.calendar_rows,
                strata_json(&self.calendar)
            ),
            None => "null".to_string(),
        };
        let exprs: Vec<String> = self
            .exprs
            .iter()
            .map(|e| {
                format!(
                    "{{\"name\":{},\"expr\":{},\"band\":{},\"dwell\":{},\"rows\":{},\"share_rows_on\":{},\"nonfinite_rows\":{},\
\"switches\":{},\"switches_per_year\":{},\"mean_state_periods\":{{\"on\":{},\"off\":{}}},\"strata\":{}}}",
                    jstr(&e.name),
                    jstr(&e.source),
                    jnum(e.band),
                    e.dwell,
                    e.counts.rows,
                    jnum(if e.counts.rows == 0 { f64::NAN } else { e.counts.rows_on as f64 / e.counts.rows as f64 }),
                    e.counts.nonfinite,
                    e.switches.switches,
                    jnum(e.switches.per_year),
                    jnum(e.switches.mean_on_run),
                    jnum(e.switches.mean_off_run),
                    strata_json(&e.strata)
                )
            })
            .collect();
        format!("{{\"mode\":\"strata_of_the_report\",\"calendar\":{},\"expressions\":[{}]}}", cal, exprs.join(","))
    }

    pub fn markdown(&self) -> String {
        let mut s = String::from("## Regime strata\n\n");
        if self.calendar_sha256.is_some() {
            s.push_str(&format!(
                "Calendar rows: {} (sha256 {}). A period belongs to the row that contains its bucket start; periods outside every row are `undated`.\n\n",
                self.calendar_rows,
                self.calendar_sha256.as_deref().unwrap_or("")
            ));
            s.push_str(&markdown_table(&self.calendar));
            s.push('\n');
        }
        for e in &self.exprs {
            s.push_str(&format!(
                "Expression `{}` as `{}` (band {}, dwell {}): {} switches ({} per year), {} rows, {} non-finite.\n\n",
                e.name,
                e.source,
                e.band,
                e.dwell,
                e.switches.switches,
                cell(e.switches.per_year),
                e.counts.rows,
                e.counts.nonfinite
            ));
            s.push_str(&markdown_table(&e.strata));
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISO: &str = "label,start,end\nbear,2018-01-01T00:00:00Z,2018-07-01T00:00:00Z\nbull,2018-07-01T00:00:00Z,2019-01-01T00:00:00Z\n";

    #[test]
    fn calendar_lookup_is_half_open() {
        let c = Calendar::parse(ISO).unwrap();
        let t = |s: &str| crate::data::parse_time(s).unwrap();
        assert_eq!(c.label_at(t("2018-06-30T23:59:59Z")), Some("bear"));
        assert_eq!(c.label_at(t("2018-07-01T00:00:00Z")), Some("bull"));
        assert_eq!(c.label_at(t("2019-01-01T00:00:00Z")), None);
        assert_eq!(c.label_at(t("2017-12-31T00:00:00Z")), None);
    }

    #[test]
    fn overlapping_rows_and_bad_headers_are_refused() {
        let e = Calendar::parse("label,start,end\na,2018-01-01,2018-03-01\nb,2018-02-01,2018-04-01\n").unwrap_err();
        assert!(e.contains("overlap"), "{}", e);
        assert!(Calendar::parse("a,b,c\n").is_err());
        assert!(Calendar::parse("label,start,end\nundated,2018-01-01,2018-02-01\n").is_err());
        assert!(Calendar::parse("label,start,end\na,2018-02-01,2018-01-01\n").is_err());
    }

    #[test]
    fn any_label_set_is_a_calendar() {
        let c = Calendar::parse("# a comment\nlabel,start,end\ny2024,2024-01-01,2025-01-01\ny2023,2023-01-01,2024-01-01\n").unwrap();
        assert_eq!(c.labels(), vec!["y2023", "y2024"]);
    }

    #[test]
    fn periods_outside_every_row_are_undated() {
        let c = Calendar::parse(ISO).unwrap();
        let t = |s: &str| crate::data::parse_time(s).unwrap();
        let (names, idx) = c.assign(&[t("2017-06-01"), t("2018-02-01"), t("2018-08-01"), t("2020-01-01")]);
        assert_eq!(names, vec!["bear", "bull", "undated"]);
        assert_eq!(idx, vec![2, 0, 1, 2]);
    }

    #[test]
    fn a_short_stratum_keeps_counts_and_nulls_the_statistics() {
        let net = [0.01, 0.02, 0.03];
        let o = StatOpts { periods_per_year: 365.0, draws: 10, block_len: None, seed: 1 };
        let s = stratum("x", "calendar", &[1], &Series { net: &net, ..Default::default() }, &o);
        assert_eq!(s.periods, 1);
        assert!(s.sharpe.is_nan() && s.mean_ci.is_none() && s.net_mean.is_nan());
        assert!(s.json().contains("\"sharpe\":null"));
    }

    #[test]
    fn period_state_is_the_majority_of_rows() {
        let keys = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let on = [true, true, false, true, false, false];
        let finite = [true, true, true, true, true, false];
        let (st, c) = period_states(&keys, &on, &finite, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(st, vec![true, false, false, false]);
        assert_eq!(c, RowCounts { rows: 6, rows_on: 3, nonfinite: 1 });
    }
}
