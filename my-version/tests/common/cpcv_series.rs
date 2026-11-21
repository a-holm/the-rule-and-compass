//! #1236: synthetic CPCV(4, 2) split reports with the per-period series of
//! `walkforward --emit-series`, for the path assembly and the candidate table.
#![allow(dead_code)]

use rcsr::cpcv::{self, Group, Locked, Plan, Scheme};
use rcsr::devlog::Json;
use rcsr::intervals::EmbargoSource;
use std::path::Path;

pub const H: i64 = 3600;

/// The series a path carries, in the order of the walk-forward pooled block.
pub const SERIES: [&str; 7] =
    ["period_gross", "period_exposure", "period_entries", "period_bh", "period_cost", "period_entry_stamp", "period_bh_net"];

/// What one synthetic split report holds.
#[derive(Clone, Copy)]
pub struct Carry {
    /// `pooled.<SERIES[k]>` is written when `series[k]`.
    pub series: [bool; 7],
    pub pooled_entry_times: bool,
    pub window_entry_times: bool,
    /// One series written one value short.
    pub short: Option<&'static str>,
    /// `pooled.grid.rule` target with `pooled.period_rows`.
    pub target: bool,
}

pub const ALL: Carry = Carry { series: [true; 7], pooled_entry_times: true, window_entry_times: true, short: None, target: false };
pub const NONE: Carry = Carry { series: [false; 7], pooled_entry_times: false, window_entry_times: false, short: None, target: false };

/// `ALL` without one series.
pub fn without(name: &str) -> Carry {
    let mut c = ALL;
    c.series[SERIES.iter().position(|s| *s == name).unwrap()] = false;
    c
}

pub fn day(s: &str) -> i64 {
    rcsr::data::parse_time(s).unwrap() as i64
}

pub const START: &str = "2021-01-01";
pub const END: &str = "2021-01-11";

/// CPCV(4, 2) from 2021-01-01 to 2021-01-11 around a locked range on 2021-01-04: G2 has
/// two pieces, so every split that tests G2 has three windows. Writes `splits.csv`.
pub fn write_table(dir: &Path) -> (String, Scheme) {
    let lock = Locked { start: day("2021-01-04") + 12 * H, end: day("2021-01-04") + 18 * H, embargo: 0 };
    let plan = Plan {
        start: day(START),
        end: day(END),
        groups: 4,
        test_groups: 2,
        purge: H,
        embargo: H,
        embargo_source: EmbargoSource::Flag,
        embargo_column: None,
        validation_share: 0.0,
        locked: vec![lock],
    };
    let text = cpcv::table_csv(&cpcv::generate(&plan).unwrap().splits);
    let path = dir.join("splits.csv");
    std::fs::write(&path, &text).unwrap();
    (path.to_string_lossy().into_owned(), cpcv::read_scheme(&text).unwrap())
}

/// A calendar with `bull` before 2021-01-07 and `bear` from then on.
pub fn write_calendar(dir: &Path) -> String {
    let path = dir.join("calendar.csv");
    std::fs::write(&path, format!("label,start,end\nbull,{},{}\nbear,{},{}\n", day(START), day("2021-01-07"), day("2021-01-07"), day(END))).unwrap();
    path.to_string_lossy().into_owned()
}

/// The design 4.2 rule of the broad pilot without `null_q` (it needs `--null-wf`).
pub const PILOT_RULE: &str = "{\"schema\":\"rcsr-trials-rule/1\",\"cost_multiples\":[1,2],\
\"gates\":{\"entry_times\":[\">=\",150],\"exposure\":[\"between\",0.01,0.90],\"net20\":[\">\",0],\"bull_net\":[\">\",0],\
\"bear_excess\":[\">=\",0.25],\"alpha_bh:bear\":[\">=\",0],\"alpha_bh\":[\">\",0],\"groups_pos\":[\">=\",4]},\
\"relaxed_gates\":{\"net10\":[\">\",0],\"bear_excess\":[\">=\",0]},\
\"pareto\":{\"net20\":\"max\",\"sharpe\":\"max\",\"max_drawdown\":\"max\",\"worst_stratum\":\"max\"},\
\"rank\":[\"net20\",\"sharpe\",\"sortino\",\"calmar\",\"worst_stratum\",\"alpha_bh\"],\"pick\":4,\"diversity_rho\":0.7,\"min_front\":4}";

/// The gate metrics of `PILOT_RULE` that sit in a row's `metrics`; `alpha_bh:bear` is the
/// `alpha_bh` of the row's `bear` stratum.
pub const PILOT_METRICS: [&str; 7] = ["entry_times", "exposure", "net20", "bull_net", "bear_excess", "alpha_bh", "groups_pos"];

fn list(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

/// The split report of split `s` over its test intervals `test`, written to `<dir>/<name>`.
/// Every value depends on the split, so a path that takes a group from another split differs.
pub fn write_report(dir: &Path, name: &str, s: usize, test: &[(i64, i64)], carry: &Carry) -> String {
    let k = s as u64 + 1;
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); SERIES.len()];
    let (mut net, mut time, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    let mut windows = Vec::new();
    let mut defs = Vec::new();
    let (mut entries_all, mut active_all, mut times_all) = (0.0, 0.0, 0.0);
    for (w, &(a, b)) in test.iter().enumerate() {
        let (mut stamps, mut entries) = (0.0, 0.0);
        let mut t = a;
        while t < b {
            let h = (t / H) as u64;
            let edge = t == a || t + H >= b;
            let empty = carry.target && h % 23 == 0;
            let exposure = 0.05 * (1 + (h * (k + 2)) % 17) as f64;
            let enter = if (h + k) % 5 == 0 { (1 + h % 3) as f64 } else { 0.0 };
            let bh = 1e-3 * (((h * 7 + 3 * k) % 13) as f64 - 6.0);
            let gross = exposure * bh + 1e-4 * (((h * (k + 3)) % 11) as f64 - 3.0);
            let cost = 2.5e-4 * enter + if edge { 1e-4 } else { 0.0 };
            let stamp = if enter > 0.0 { 1.0 } else { 0.0 };
            let bh_net = bh - if edge { 1e-3 } else { 0.0 };
            let v = [gross, exposure, enter, bh, cost, stamp, bh_net];
            for (c, x) in cols.iter_mut().zip(v) {
                c.push(if empty { 0.0 } else { x });
            }
            net.push(if empty { 0.0 } else { gross - cost });
            time.push(t as f64);
            rows.push(if empty { 0.0 } else { 3.0 });
            stamps += stamp;
            entries += enter;
            t += H;
        }
        let count = stamps + ((s + w) % 3) as f64;
        let active = (10 * (w + 1) + s) as f64;
        entries_all += entries;
        active_all += active;
        times_all += count;
        defs.push(format!("{{\"label\":\"w{w}\",\"train_start\":0,\"train_end\":1,\"test_start\":{a},\"test_end\":{b}}}"));
        windows.push(format!(
            "{{\"label\":\"w{w}\",\"expression\":\"x + {s}\",\"metrics\":{{\"entries\":{entries},\"active_rows\":{active}}}{}}}",
            if carry.window_entry_times { format!(",\"entry_times\":{count}") } else { String::new() }
        ));
    }
    let mut pooled = format!("\"periods\":{},\"entries\":{entries_all},\"active_rows\":{active_all}", net.len());
    if carry.target {
        let empty = rows.iter().filter(|r| **r == 0.0).count();
        pooled.push_str(&format!(",\"grid\":{{\"rule\":\"target\",\"periods\":{},\"empty_periods\":{empty}}}", net.len()));
    }
    pooled.push_str(&format!(",\"period_net\":[{}],\"period_time\":[{}]", list(&net), list(&time)));
    if carry.target {
        pooled.push_str(&format!(",\"period_rows\":[{}]", list(&rows)));
    }
    for (i, key) in SERIES.iter().enumerate() {
        if carry.series[i] {
            let v = &cols[i];
            let v = if carry.short == Some(*key) { &v[..v.len() - 1] } else { &v[..] };
            pooled.push_str(&format!(",\"{key}\":[{}]", list(v)));
        }
    }
    if carry.pooled_entry_times {
        pooled.push_str(&format!(",\"entry_times\":{times_all}"));
    }
    let text = format!(
        "{{\"mode\":\"walkforward\",\"window_definition\":[{}],\"purge\":{{\"horizon_seconds\":3600,\"embargo_seconds\":3600}},\
\"cost_model\":{{\"per_side_bps\":10,\"round_trip_bps\":20}},\"cost\":{{\"per_side_bps\":10,\"source\":\"flag\"}},\
\"policy\":{{\"name\":\"threshold\",\"band_width\":0,\"min_hold\":1}},\"threshold\":{{\"quantile\":0.5,\"fixed\":null}},\
\"annualization\":{{\"periods_per_year\":8760,\"method\":\"explicit\"}},\"book\":{{\"missing_score\":\"flat\"}},\
\"pooled_seam\":{{\"reset_flat\":true,\"seam_cost\":0,\"seam_trades\":0}},\"windows\":[{}],\"pooled\":{{{}}}}}\n",
        defs.join(","),
        windows.join(","),
        pooled
    );
    std::fs::write(dir.join(name), text).unwrap();
    dir.join(name).to_string_lossy().into_owned()
}

/// One report per split of `scheme`, in split order, named `<prefix>_<split id>.json`.
pub fn write_reports(dir: &Path, prefix: &str, scheme: &Scheme, carry: &dyn Fn(usize) -> Carry) -> Vec<String> {
    (0..scheme.ids.len()).map(|s| write_report(dir, &format!("{prefix}_{}.json", scheme.ids[s]), s, &scheme.tests[s], &carry(s))).collect()
}

fn floats(j: &Json, path: &[&str]) -> Vec<f64> {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}"));
    }
    match cur {
        Json::Arr(v) => v.iter().map(|x| x.as_f64().unwrap()).collect(),
        other => panic!("{path:?} is not an array: {other:?}"),
    }
}

/// The values of `pooled.<key>` of a split report on the periods of `group`.
pub fn group_values(report: &Json, group: &Group, key: &str) -> Vec<f64> {
    let time = floats(report, &["pooled", "period_time"]);
    let v = floats(report, &["pooled", key]);
    time.iter().zip(v).filter(|(t, _)| group.contains(**t)).map(|(_, x)| x).collect()
}

/// The sum of `windows[].entry_times` over the windows of a split report inside `group`.
pub fn group_entry_times(report: &Json, group: &Group) -> f64 {
    let (Some(Json::Arr(defs)), Some(Json::Arr(ws))) = (report.get("window_definition"), report.get("windows")) else {
        panic!("a split report without windows")
    };
    defs.iter()
        .zip(ws)
        .filter(|(d, _)| group.contains(d.get("test_start").and_then(Json::as_f64).unwrap()))
        .map(|(_, w)| w.get("entry_times").and_then(Json::as_f64).unwrap())
        .sum()
}

/// The concatenated group series of path `j`: `pooled.<key>` of the split that path `j`
/// takes for each group, in group order.
pub fn path_values(reports: &[Json], scheme: &Scheme, j: usize, key: &str) -> Vec<f64> {
    let assign = cpcv::assignment(&scheme.test_sets, scheme.groups.len());
    (0..scheme.groups.len()).flat_map(|g| group_values(&reports[assign[g][j]], &scheme.groups[g], key)).collect()
}

/// The entry times of path `j`: the sum over its groups of the chosen split's window counts.
pub fn path_entry_times(reports: &[Json], scheme: &Scheme, j: usize) -> f64 {
    let assign = cpcv::assignment(&scheme.test_sets, scheme.groups.len());
    (0..scheme.groups.len()).map(|g| group_entry_times(&reports[assign[g][j]], &scheme.groups[g])).sum()
}
