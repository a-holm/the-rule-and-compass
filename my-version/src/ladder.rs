//! Long-only baseline ladder on one panel, one set of folds and one cost model.
//!
//! Every rung is a `{0, 1}` position series (long or flat) or cash. All rungs
//! see the same rows, the same folds and the same pricing:
//!
//! * a fold fits on `[train_start, validation_start - purge)`, selects
//!   hyperparameters and the score threshold on `[validation_start,
//!   train_end)`, and is scored once on `[test_start, test_end)`;
//! * a score rung is long iff `score > theta`, where `theta` is a fit-row
//!   quantile and the quantile is chosen on validation net P&L;
//! * every block starts flat and pays its final exit, through
//!   [`crate::baselines::net_rows`];
//! * rows of one timestamp are equal-weighted into a period series.
//!
//! The rungs, in report order: `cash`, `buy_and_hold`, `equal_weight_universe`,
//! `constant_long_flat`, `zero_param`, `ols_ridge`, `best_single_feature`,
//! `vol_binned20`, `naive_reversal`, `top_tercile_long`, `feature_threshold`,
//! `gbm`, `symbolic_regression`, `matched_exposure_random_long`. A rung whose
//! input column is not given is listed as `not_run` with the reason.
//!
//! OLS, GBM and the symbolic-regression score share the projection, the folds
//! and the threshold protocol. The GBM tuning budget is a tree count; the
//! report states the tree budget and wall seconds beside the symbolic
//! regression's candidate evaluations and wall seconds, as given by the caller.

use crate::baselines;
use crate::data::{self, Table};
use crate::devlog::num;
use crate::gpu_stats::{RunLedger, Stage};
use crate::inference::{self, Bh, Dsr, Pbo, Spa};
use crate::protocol::{self, WindowRow};
use crate::risk::{risk_stats, BetaAlpha, DEFAULT_SORTINO_MAR};
use crate::target::TargetSpec;
use crate::target_structure;
use crate::walkforward::{quantile, SECONDS_PER_YEAR};
use std::time::Instant;

pub const SCHEMA: &str = "rcsr-ladder/1";
pub const DEFAULT_COST_BPS: f64 = 12.0;
pub const DEFAULT_COST_GRID: [f64; 3] = [12.0, 20.0, 40.0];
pub const DEFAULT_QUANTILES: [f64; 5] = [0.5, 0.6, 0.7, 0.8, 0.9];
pub const DEFAULT_PURGE_SECONDS: f64 = protocol::DEFAULT_PURGE_SECONDS;
pub const DEFAULT_GBM_TREES: usize = 240;
const RIDGE_LAMBDAS: [f64; 5] = [1e-8, 1e-4, 1e-2, 1.0, 100.0];
const GBM_CONFIGS: [(usize, f64); 6] = [(2, 0.1), (3, 0.1), (4, 0.1), (2, 0.05), (3, 0.05), (4, 0.05)];
const GBM_BINS: usize = 32;
const GBM_CHECK_EVERY: usize = 5;
const GBM_MIN_LEAF: usize = 20;
const MAX_FIT_ROWS: usize = 200_000;

pub(crate) fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub(crate) fn jnum(v: f64) -> String {
    if v.is_finite() {
        num(v)
    } else {
        "null".to_string()
    }
}

fn jopt(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_string(), jnum)
}

fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(","))
}

// ---------------------------------------------------------------------------
// Panel and folds
// ---------------------------------------------------------------------------

/// Rows sorted by `(group, time)`. Feature columns are column-major.
#[derive(Clone, Debug)]
pub struct Panel {
    pub names: Vec<String>,
    pub x: Vec<Vec<f64>>,
    pub time: Vec<f64>,
    pub group: Vec<u64>,
    pub y: Vec<f64>,
    pub ret: Vec<f64>,
    pub extra: Vec<(String, Vec<f64>)>,
}

impl Panel {
    pub fn new(
        names: Vec<String>,
        x: Vec<Vec<f64>>,
        time: Vec<f64>,
        group: Vec<u64>,
        y: Vec<f64>,
        ret: Vec<f64>,
        extra: Vec<(String, Vec<f64>)>,
    ) -> Result<Panel, String> {
        let n = time.len();
        let ok = group.len() == n
            && y.len() == n
            && ret.len() == n
            && x.iter().all(|c| c.len() == n)
            && extra.iter().all(|(_, c)| c.len() == n)
            && names.len() == x.len();
        if !ok {
            return Err("panel columns differ in length".to_string());
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            group[a]
                .cmp(&group[b])
                .then_with(|| time[a].total_cmp(&time[b]))
                .then_with(|| a.cmp(&b))
        });
        let take = |v: &Vec<f64>| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        Ok(Panel {
            names,
            x: x.iter().map(take).collect(),
            time: take(&time),
            group: order.iter().map(|&i| group[i]).collect(),
            y: take(&y),
            ret: take(&ret),
            extra: extra.iter().map(|(k, v)| (k.clone(), take(v))).collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    /// A feature or an extra column by name.
    pub fn col(&self, name: &str) -> Option<&[f64]> {
        if let Some(j) = self.names.iter().position(|n| n == name) {
            return Some(&self.x[j]);
        }
        self.extra.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_slice())
    }
}

fn numeric_column(t: &Table, name: &str) -> Result<Vec<f64>, String> {
    let j = t.index_of(name)?;
    if !t.numeric[j] {
        return Err(format!("column '{}' must be numeric", name));
    }
    Ok(t.col(j).to_vec())
}

fn time_column(t: &Table, name: &str) -> Result<Vec<f64>, String> {
    let j = t.index_of(name)?;
    if t.numeric[j] {
        return Ok(t.col(j).to_vec());
    }
    let d = t
        .dict(j)
        .ok_or_else(|| format!("time column '{}' is neither numeric nor a date", name))?;
    (0..t.len())
        .map(|r| {
            data::parse_time(d.text(r)).ok_or_else(|| format!("row {}: cannot parse time '{}'", r + 1, d.text(r)))
        })
        .collect()
}

fn group_column(t: &Table, name: Option<&str>) -> Result<Vec<u64>, String> {
    let Some(name) = name else {
        return Ok(vec![0; t.len()]);
    };
    let j = t.index_of(name)?;
    if let Some(d) = t.dict(j) {
        let mut ids: Vec<&str> = Vec::new();
        let mut out = Vec::with_capacity(t.len());
        for r in 0..t.len() {
            let s = d.text(r);
            let k = match ids.iter().position(|x| *x == s) {
                Some(k) => k,
                None => {
                    ids.push(s);
                    ids.len() - 1
                }
            };
            out.push(k as u64);
        }
        return Ok(out);
    }
    Ok(t.col(j).iter().map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() }).collect())
}

#[derive(Clone, Debug)]
pub struct Fold {
    pub label: String,
    pub fit: Vec<usize>,
    pub valid: Vec<usize>,
    pub test: Vec<usize>,
}

/// Row ranges per fold. Rows need a finite time and return. The fit range ends
/// `purge_seconds` before the validation start so a fit label never reads into
/// the validation slice.
/// A fold whose fit, validation or test range touches `locked` (a declared
/// locked test, half-open seconds) is refused.
pub fn make_folds(
    p: &Panel,
    windows: &[WindowRow],
    purge_seconds: f64,
    locked: Option<(f64, f64)>,
) -> Result<Vec<Fold>, String> {
    let mut folds = Vec::with_capacity(windows.len());
    for w in windows {
        let win = &w.window;
        if let Some((ls, le)) = locked {
            let side = if win.train_start < le && win.train_end > ls {
                Some("fit or validation")
            } else if win.test_start < le && win.test_end > ls {
                Some("test")
            } else {
                None
            };
            if let Some(side) = side {
                return Err(format!(
                    "fold {}: the {} range touches the declared locked test {}..{}; \
                     read it only with --locked-test --ledger --sealed",
                    win.label,
                    side,
                    protocol::iso(ls),
                    protocol::iso(le)
                ));
            }
        }
        let fit_end = w.validation_start - purge_seconds;
        let mut f = Fold {
            label: win.label.clone(),
            fit: Vec::new(),
            valid: Vec::new(),
            test: Vec::new(),
        };
        for i in 0..p.len() {
            let t = p.time[i];
            if !t.is_finite() || !p.ret[i].is_finite() {
                continue;
            }
            if t >= win.train_start && t < fit_end {
                f.fit.push(i);
            } else if t >= w.validation_start && t < win.train_end {
                f.valid.push(i);
            } else if t >= win.test_start && t < win.test_end {
                f.test.push(i);
            }
        }
        if f.fit.is_empty() || f.valid.is_empty() || f.test.is_empty() {
            return Err(format!(
                "fold {}: fit {} / validation {} / test {} rows; every range needs rows",
                f.label,
                f.fit.len(),
                f.valid.len(),
                f.test.len()
            ));
        }
        folds.push(f);
    }
    if folds.is_empty() {
        return Err("no folds".to_string());
    }
    Ok(folds)
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Options {
    pub cost_bps: f64,
    pub cost_grid: Vec<f64>,
    pub quantiles: Vec<f64>,
    pub purge_seconds: f64,
    /// Label horizon in seconds; sets the bootstrap block floor.
    pub horizon_seconds: f64,
    pub vol_col: Option<String>,
    pub reversal_col: Option<String>,
    pub signal_col: Option<String>,
    pub sr_score_col: Option<String>,
    /// Candidate evaluations and wall seconds of the symbolic-regression run.
    pub sr_evals: Option<u64>,
    pub sr_wall_seconds: Option<f64>,
    pub spec: Option<TargetSpec>,
    pub gbm_trees: usize,
    pub seed: u32,
    pub draws: usize,
    pub block_len: Option<usize>,
    pub fdr_q: f64,
    pub pbo_blocks: usize,
    /// Cap on rows used to fit the GBM and the ridge, drawn by stride.
    pub max_fit_rows: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            cost_bps: DEFAULT_COST_BPS,
            cost_grid: DEFAULT_COST_GRID.to_vec(),
            quantiles: DEFAULT_QUANTILES.to_vec(),
            purge_seconds: DEFAULT_PURGE_SECONDS,
            horizon_seconds: 0.0,
            vol_col: None,
            reversal_col: None,
            signal_col: None,
            sr_score_col: None,
            sr_evals: None,
            sr_wall_seconds: None,
            spec: None,
            gbm_trees: DEFAULT_GBM_TREES,
            seed: 0,
            draws: 2000,
            block_len: None,
            fdr_q: 0.10,
            pbo_blocks: 8,
            max_fit_rows: MAX_FIT_ROWS,
        }
    }
}

// ---------------------------------------------------------------------------
// Pooled test rows and pricing
// ---------------------------------------------------------------------------

struct Pool {
    idx: Vec<usize>,
    /// `[start, end)` of each fold inside `idx`.
    ranges: Vec<(usize, usize)>,
    /// Pool rows in time order and the bucket boundaries of equal times.
    by_time: Vec<usize>,
    bounds: Vec<usize>,
    times: Vec<f64>,
    ppy: f64,
    step: Option<f64>,
}

impl Pool {
    fn new(p: &Panel, folds: &[Fold]) -> Pool {
        let mut idx = Vec::new();
        let mut ranges = Vec::new();
        for f in folds {
            let s = idx.len();
            idx.extend_from_slice(&f.test);
            ranges.push((s, idx.len()));
        }
        let mut by_time: Vec<usize> = (0..idx.len()).collect();
        by_time.sort_by(|&a, &b| p.time[idx[a]].total_cmp(&p.time[idx[b]]).then_with(|| a.cmp(&b)));
        let mut bounds = vec![0usize];
        let mut times = Vec::new();
        for k in 0..by_time.len() {
            let t = p.time[idx[by_time[k]]];
            if k == 0 || t != *times.last().unwrap() {
                if k > 0 {
                    bounds.push(k);
                }
                times.push(t);
            }
        }
        bounds.push(by_time.len());
        let mut steps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 1e-9).collect();
        steps.sort_by(|a, b| a.total_cmp(b));
        let step = steps.get(steps.len() / 2).copied().filter(|dt| *dt >= 1.0 && *dt <= SECONDS_PER_YEAR);
        let ppy = step.map_or(365.0, |dt| SECONDS_PER_YEAR / dt);
        Pool {
            idx,
            ranges,
            by_time,
            bounds,
            times,
            ppy,
            step,
        }
    }

    fn len(&self) -> usize {
        self.idx.len()
    }

    /// Equal-weight mean of a per-row vector for each timestamp.
    fn periods(&self, v: &[f64]) -> Vec<f64> {
        (0..self.times.len())
            .map(|b| {
                let (s, e) = (self.bounds[b], self.bounds[b + 1]);
                self.by_time[s..e].iter().map(|&k| v[k]).sum::<f64>() / (e - s) as f64
            })
            .collect()
    }

}

/// Per-row net and gross of `pos` (aligned with the pool) at `side` per side.
fn price(p: &Panel, pool: &Pool, pos: &[f64], side: f64) -> (Vec<f64>, Vec<f64>) {
    let mut net = Vec::with_capacity(pool.len());
    let mut gross = Vec::with_capacity(pool.len());
    for &(s, e) in &pool.ranges {
        let rows = &pool.idx[s..e];
        let r: Vec<f64> = rows.iter().map(|&i| p.ret[i]).collect();
        let g: Vec<u64> = rows.iter().map(|&i| p.group[i]).collect();
        net.extend(baselines::net_rows(&pos[s..e], &r, &g, side));
        gross.extend(pos[s..e].iter().zip(&r).map(|(a, b)| a * b));
    }
    (net, gross)
}

fn net_sum_slice(p: &Panel, rows: &[usize], pos: &[f64], side: f64) -> f64 {
    let r: Vec<f64> = rows.iter().map(|&i| p.ret[i]).collect();
    let g: Vec<u64> = rows.iter().map(|&i| p.group[i]).collect();
    baselines::net_rows(pos, &r, &g, side).iter().sum()
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

fn finite_quantile(v: &[f64], q: f64) -> f64 {
    quantile(v, q)
}

/// Fit-row stride sample, at most `cap` rows, order preserved.
fn thin(rows: &[usize], cap: usize) -> Vec<usize> {
    if cap == 0 || rows.len() <= cap {
        return rows.to_vec();
    }
    let step = rows.len() as f64 / cap as f64;
    (0..cap).map(|k| rows[(k as f64 * step) as usize]).collect()
}

struct Ridge {
    mean: Vec<f64>,
    sd: Vec<f64>,
    ymean: f64,
    coef: Vec<f64>,
}

impl Ridge {
    fn predict(&self, cols: &[&[f64]], row: usize) -> f64 {
        let mut s = self.ymean;
        for (j, c) in cols.iter().enumerate() {
            let v = c[row];
            if v.is_finite() {
                s += self.coef[j] * (v - self.mean[j]) / self.sd[j];
            }
        }
        s
    }
}

fn cholesky_solve(a: &[Vec<f64>], b: &[f64], ridge: f64) -> Vec<f64> {
    let p = b.len();
    let trace: f64 = (0..p).map(|i| a[i][i]).sum::<f64>().max(1e-12);
    let mut jitter = 0.0f64;
    for _ in 0..12 {
        let mut l = vec![vec![0.0f64; p]; p];
        let mut ok = true;
        'outer: for i in 0..p {
            for j in 0..=i {
                let mut s = a[i][j] + if i == j { ridge + jitter } else { 0.0 };
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 1e-14 * trace / p.max(1) as f64 {
                        ok = false;
                        break 'outer;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        if ok {
            let mut z = vec![0.0f64; p];
            for i in 0..p {
                let mut s = b[i];
                for k in 0..i {
                    s -= l[i][k] * z[k];
                }
                z[i] = s / l[i][i];
            }
            let mut x = vec![0.0f64; p];
            for i in (0..p).rev() {
                let mut s = z[i];
                for k in i + 1..p {
                    s -= l[k][i] * x[k];
                }
                x[i] = s / l[i][i];
            }
            return x;
        }
        jitter = if jitter == 0.0 { 1e-10 * trace / p.max(1) as f64 } else { jitter * 10.0 };
    }
    vec![0.0; p]
}

/// One pass of normal equations, one Cholesky solve per lambda.
fn fit_ridge_path(p: &Panel, feats: &[usize], rows: &[usize], lambdas: &[f64]) -> Vec<Ridge> {
    let k = feats.len();
    let use_rows: Vec<usize> = rows.iter().copied().filter(|&i| p.y[i].is_finite()).collect();
    let n = use_rows.len().max(1) as f64;
    let mut mean = vec![0.0f64; k];
    let mut sd = vec![1.0f64; k];
    for (j, &f) in feats.iter().enumerate() {
        let (mut s, mut c) = (0.0, 0.0);
        for &i in &use_rows {
            let v = p.x[f][i];
            if v.is_finite() {
                s += v;
                c += 1.0;
            }
        }
        mean[j] = if c > 0.0 { s / c } else { 0.0 };
        let mut ss = 0.0;
        for &i in &use_rows {
            let v = p.x[f][i];
            if v.is_finite() {
                ss += (v - mean[j]).powi(2);
            }
        }
        let var = if c > 1.0 { ss / c } else { 0.0 };
        sd[j] = if var > 1e-300 { var.sqrt() } else { 1.0 };
    }
    let ymean = use_rows.iter().map(|&i| p.y[i]).sum::<f64>() / n;
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    let mut z = vec![0.0f64; k];
    for &i in &use_rows {
        for (j, &f) in feats.iter().enumerate() {
            let v = p.x[f][i];
            z[j] = if v.is_finite() { (v - mean[j]) / sd[j] } else { 0.0 };
        }
        let yc = p.y[i] - ymean;
        for a in 0..k {
            xty[a] += z[a] * yc;
            for b in 0..=a {
                xtx[a][b] += z[a] * z[b];
            }
        }
    }
    for a in 0..k {
        xty[a] /= n;
        for b in 0..=a {
            xtx[a][b] /= n;
            xtx[b][a] = xtx[a][b];
        }
    }
    lambdas
        .iter()
        .map(|&l| Ridge {
            mean: mean.clone(),
            sd: sd.clone(),
            ymean,
            coef: cholesky_solve(&xtx, &xty, l),
        })
        .collect()
}

fn mse_of(pred: &[f64], rows: &[usize], y: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (k, &i) in rows.iter().enumerate() {
        if pred[k].is_finite() && y[i].is_finite() {
            s += (pred[k] - y[i]).powi(2);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        f64::INFINITY
    }
}

// --- gradient boosting -----------------------------------------------------

#[derive(Clone, Debug)]
enum Node {
    Leaf(f64),
    Split { feat: usize, bin: u8, left: usize, right: usize },
}

#[derive(Clone, Debug)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict(&self, bins: &[Vec<u8>], row: usize) -> f64 {
        let mut at = 0usize;
        loop {
            match &self.nodes[at] {
                Node::Leaf(v) => return *v,
                Node::Split { feat, bin, left, right } => {
                    at = if bins[*feat][row] <= *bin { *left } else { *right };
                }
            }
        }
    }
}

struct Binner {
    edges: Vec<Vec<f64>>,
}

impl Binner {
    fn fit(p: &Panel, feats: &[usize], rows: &[usize]) -> Binner {
        let edges = feats
            .iter()
            .map(|&f| {
                let vals: Vec<f64> = rows.iter().map(|&i| p.x[f][i]).filter(|v| v.is_finite()).collect();
                let mut e: Vec<f64> = (1..GBM_BINS)
                    .map(|b| finite_quantile(&vals, b as f64 / GBM_BINS as f64))
                    .filter(|v| v.is_finite())
                    .collect();
                e.dedup();
                e
            })
            .collect();
        Binner { edges }
    }

    /// Bin index of every requested row. A non-finite value goes to bin 0.
    fn apply(&self, p: &Panel, feats: &[usize], rows: &[usize]) -> Vec<Vec<u8>> {
        feats
            .iter()
            .enumerate()
            .map(|(j, &f)| {
                rows.iter()
                    .map(|&i| {
                        let v = p.x[f][i];
                        if v.is_finite() {
                            self.edges[j].partition_point(|e| *e < v) as u8
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

fn grow_tree(bins: &[Vec<u8>], resid: &[f64], depth: usize, lr: f64) -> Tree {
    let mut nodes: Vec<Node> = vec![Node::Leaf(0.0)];
    let all: Vec<usize> = (0..resid.len()).collect();
    let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(0, all, 0)];
    while let Some((at, rows, d)) = stack.pop() {
        let sum: f64 = rows.iter().map(|&i| resid[i]).sum();
        let cnt = rows.len() as f64;
        let leaf = lr * sum / cnt.max(1.0);
        nodes[at] = Node::Leaf(leaf);
        if d >= depth || rows.len() < 2 * GBM_MIN_LEAF {
            continue;
        }
        let mut best: Option<(f64, usize, u8)> = None;
        for (f, col) in bins.iter().enumerate() {
            let mut hs = [0.0f64; 256];
            let mut hc = [0.0f64; 256];
            let mut max_bin = 0usize;
            for &i in &rows {
                let b = col[i] as usize;
                hs[b] += resid[i];
                hc[b] += 1.0;
                if b > max_bin {
                    max_bin = b;
                }
            }
            let (mut ls, mut lc) = (0.0, 0.0);
            for b in 0..max_bin {
                ls += hs[b];
                lc += hc[b];
                let (rs, rc) = (sum - ls, cnt - lc);
                if lc < GBM_MIN_LEAF as f64 || rc < GBM_MIN_LEAF as f64 {
                    continue;
                }
                let gain = ls * ls / lc + rs * rs / rc - sum * sum / cnt;
                if gain > best.map_or(1e-18, |x| x.0) {
                    best = Some((gain, f, b as u8));
                }
            }
        }
        if let Some((_, f, b)) = best {
            let (l, r): (Vec<usize>, Vec<usize>) = rows.iter().partition(|&&i| bins[f][i] <= b);
            let li = nodes.len();
            nodes.push(Node::Leaf(0.0));
            let ri = nodes.len();
            nodes.push(Node::Leaf(0.0));
            nodes[at] = Node::Split {
                feat: f,
                bin: b,
                left: li,
                right: ri,
            };
            stack.push((li, l, d + 1));
            stack.push((ri, r, d + 1));
        }
    }
    Tree { nodes }
}

struct GbmFit {
    base: f64,
    trees: Vec<Tree>,
    binner: Binner,
    configs: usize,
    trees_grown: usize,
    depth: usize,
    lr: f64,
}

/// Boost each config for its share of the tree budget, keep the config and the
/// tree count with the lowest validation MSE.
fn fit_gbm(p: &Panel, feats: &[usize], fit: &[usize], valid: &[usize], budget: usize, cap: usize) -> GbmFit {
    let rows: Vec<usize> = thin(fit, cap).into_iter().filter(|&i| p.y[i].is_finite()).collect();
    let binner = Binner::fit(p, feats, &rows);
    let fb = binner.apply(p, feats, &rows);
    let vb = binner.apply(p, feats, valid);
    let y: Vec<f64> = rows.iter().map(|&i| p.y[i]).collect();
    let base = y.iter().sum::<f64>() / y.len().max(1) as f64;
    let per = (budget / GBM_CONFIGS.len()).max(GBM_CHECK_EVERY);
    let mut best: Option<(f64, GbmFit)> = None;
    let mut grown = 0usize;
    for &(depth, lr) in GBM_CONFIGS.iter() {
        let mut pred = vec![base; rows.len()];
        let mut vpred = vec![base; valid.len()];
        let mut trees: Vec<Tree> = Vec::new();
        let mut local_best = (f64::INFINITY, 0usize);
        for t in 1..=per {
            let resid: Vec<f64> = (0..rows.len()).map(|i| y[i] - pred[i]).collect();
            let tree = grow_tree(&fb, &resid, depth, lr);
            for i in 0..rows.len() {
                pred[i] += tree.predict(&fb, i);
            }
            for i in 0..valid.len() {
                vpred[i] += tree.predict(&vb, i);
            }
            trees.push(tree);
            grown += 1;
            if t % GBM_CHECK_EVERY == 0 {
                let m = mse_of(&vpred, valid, &p.y);
                if m < local_best.0 {
                    local_best = (m, t);
                }
            }
        }
        if local_best.1 > 0 && best.as_ref().map_or(true, |b| local_best.0 < b.0) {
            trees.truncate(local_best.1);
            best = Some((
                local_best.0,
                GbmFit {
                    base,
                    trees,
                    binner: Binner { edges: binner.edges.clone() },
                    configs: 0,
                    trees_grown: 0,
                    depth,
                    lr,
                },
            ));
        }
    }
    let mut fit = match best {
        Some((_, f)) => f,
        None => GbmFit {
            base,
            trees: Vec::new(),
            binner,
            configs: 0,
            trees_grown: 0,
            depth: 0,
            lr: 0.0,
        },
    };
    fit.configs = GBM_CONFIGS.len();
    fit.trees_grown = grown;
    fit
}

fn gbm_predict(m: &GbmFit, p: &Panel, feats: &[usize], rows: &[usize]) -> Vec<f64> {
    let b = m.binner.apply(p, feats, rows);
    (0..rows.len())
        .map(|k| m.base + m.trees.iter().map(|t| t.predict(&b, k)).sum::<f64>())
        .collect()
}

// ---------------------------------------------------------------------------
// Rungs
// ---------------------------------------------------------------------------

struct Rung {
    name: &'static str,
    kind: &'static str,
    role: &'static str,
    /// Positions aligned with the pool; empty when the rung did not run.
    pos: Vec<f64>,
    /// Scores aligned with the pool, when the rung predicts something.
    score: Vec<f64>,
    /// Scores are predictions of the target (so an MSE applies).
    predicts_target: bool,
    params: String,
    fits: u64,
    trials: u64,
    wall: f64,
    note: Option<String>,
    status: &'static str,
}

impl Rung {
    fn new(name: &'static str, kind: &'static str) -> Rung {
        Rung {
            name,
            kind,
            role: "",
            pos: Vec::new(),
            score: Vec::new(),
            predicts_target: false,
            params: String::new(),
            fits: 0,
            trials: 1,
            wall: 0.0,
            note: None,
            status: "ok",
        }
    }

    fn not_run(name: &'static str, kind: &'static str, why: &str) -> Rung {
        let mut r = Rung::new(name, kind);
        r.status = "not_run";
        r.note = Some(why.to_string());
        r
    }
}

type Scorer<'a> = dyn FnMut(&Fold) -> (Vec<f64>, Vec<f64>, Vec<f64>) + 'a;

/// Threshold rung: per fold, `theta` is a fit-score quantile, the quantile is
/// the validation-net argmax over the grid.
fn score_rung(
    p: &Panel,
    folds: &[Fold],
    pool: &Pool,
    opts: &Options,
    mut r: Rung,
    scorer: &mut Scorer,
) -> Rung {
    let started = Instant::now();
    let side = opts.cost_bps * 1e-4;
    r.pos = vec![0.0; pool.len()];
    r.score = vec![f64::NAN; pool.len()];
    let mut chosen = Vec::new();
    for (fi, fold) in folds.iter().enumerate() {
        let (fit_s, valid_s, test_s) = scorer(fold);
        let mut best = (f64::NEG_INFINITY, f64::NAN, opts.quantiles.first().copied().unwrap_or(0.7));
        for &q in &opts.quantiles {
            let theta = finite_quantile(&fit_s, q);
            if !theta.is_finite() {
                continue;
            }
            let pos: Vec<f64> = valid_s.iter().map(|&s| if s > theta { 1.0 } else { 0.0 }).collect();
            let net = net_sum_slice(p, &fold.valid, &pos, side);
            if net > best.0 {
                best = (net, theta, q);
            }
        }
        let (s, e) = pool.ranges[fi];
        for (k, &sc) in test_s.iter().enumerate() {
            r.score[s + k] = sc;
            r.pos[s + k] = if best.1.is_finite() && sc > best.1 { 1.0 } else { 0.0 };
        }
        chosen.push(best.2);
        let _ = e;
    }
    r.trials = opts.quantiles.len() as u64;
    r.params = format!("quantile_by_fold={}", jarr(&chosen));
    r.wall = started.elapsed().as_secs_f64();
    r
}

fn rows_of(col: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&i| col[i]).collect()
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            n += 1.0;
            sa += x;
            sb += y;
            saa += x * x;
            sbb += y * y;
            sab += x * y;
        }
    }
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    if n > 2.0 && va > 0.0 && vb > 0.0 {
        cov / (va * vb).sqrt()
    } else {
        f64::NAN
    }
}

fn top_tercile(pool: &Pool, score: &[f64]) -> Option<Vec<f64>> {
    let mut pos = vec![0.0; pool.len()];
    let mut applicable = false;
    for b in 0..pool.times.len() {
        let rows = &pool.by_time[pool.bounds[b]..pool.bounds[b + 1]];
        let mut ranked: Vec<usize> = rows.iter().copied().filter(|&k| score[k].is_finite()).collect();
        if ranked.len() < 3 {
            continue;
        }
        applicable = true;
        ranked.sort_by(|&a, &c| score[c].total_cmp(&score[a]).then_with(|| a.cmp(&c)));
        for &k in ranked.iter().take(ranked.len() / 3) {
            pos[k] = 1.0;
        }
    }
    applicable.then_some(pos)
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Row {
    pub name: String,
    pub kind: String,
    pub status: String,
    pub role: String,
    pub note: Option<String>,
    pub params: String,
    pub fits: u64,
    pub trials: u64,
    pub wall_seconds: f64,
    pub exposure: f64,
    pub beta: f64,
    /// `(cost_bps, sum of period net returns)`.
    pub net_by_cost: Vec<(f64, f64)>,
    pub mean_period_net: f64,
    pub mean_ci: (f64, f64),
    pub sharpe: f64,
    pub p_mean_positive: f64,
    pub bh_rejected: Option<bool>,
    pub bh_adjusted: f64,
    pub turnover: f64,
    pub trades: usize,
    pub break_even_bps: f64,
    pub hit_rate: f64,
    pub return_mse: f64,
    pub rank_ic: f64,
    pub rank_ic_ci: (f64, f64),
    pub fold_sign_share: f64,
    pub group_sign_share: f64,
    pub excess_over_cash: f64,
    pub beats_zero_param: Option<bool>,
    pub series: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub rows: Vec<Row>,
    pub folds: Vec<String>,
    pub periods: usize,
    pub rows_scored: usize,
    pub periods_per_year: f64,
    pub block_len: usize,
    pub cost_bps: f64,
    pub n_trials: u64,
    pub benchmark: String,
    pub best_long_flat_rule: String,
    pub bh: Option<Bh>,
    pub spa: Option<Spa>,
    pub spa_error: Option<String>,
    pub dsr: Option<(String, Dsr)>,
    pub pbo: Option<Pbo>,
    pub pbo_error: Option<String>,
    pub draws: usize,
    pub seed: u32,
    pub fdr_q: f64,
    pub gbm_beats_sr: Option<bool>,
    pub sr_evals: Option<u64>,
    pub sr_wall_seconds: Option<f64>,
    pub gbm_trees: usize,
    pub notes: Vec<String>,
    /// #284: run-scoped phase ledger (`io`, `pnl`, `host_other`); empty until a run fills it.
    pub ledger: RunLedger,
}

fn rank_vec(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut k = 0;
    while k < order.len() {
        let mut e = k;
        while e + 1 < order.len() && v[order[e + 1]] == v[order[k]] {
            e += 1;
        }
        let avg = (k + e) as f64 / 2.0 + 1.0;
        for &o in &order[k..=e] {
            r[o] = avg;
        }
        k = e + 1;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    corr(&rank_vec(a), &rank_vec(b))
}

/// Rank IC per timestamp when a cross-section has at least 5 rows, else per
/// chunk of consecutive timestamps. Returns the IC series.
fn ic_series(p: &Panel, pool: &Pool, score: &[f64]) -> Vec<f64> {
    let wide = (0..pool.times.len()).any(|b| pool.bounds[b + 1] - pool.bounds[b] >= 5);
    let target = |k: usize| p.ret[pool.idx[k]];
    let mut out = Vec::new();
    if wide {
        for b in 0..pool.times.len() {
            let rows = &pool.by_time[pool.bounds[b]..pool.bounds[b + 1]];
            let (s, t): (Vec<f64>, Vec<f64>) = rows
                .iter()
                .filter(|&&k| score[k].is_finite())
                .map(|&k| (score[k], target(k)))
                .unzip();
            if s.len() >= 5 {
                let ic = spearman(&s, &t);
                if ic.is_finite() {
                    out.push(ic);
                }
            }
        }
    } else {
        let chunk = 48usize;
        let mut b = 0;
        while b < pool.times.len() {
            let e = (b + chunk).min(pool.times.len());
            let (s, t): (Vec<f64>, Vec<f64>) = pool.by_time[pool.bounds[b]..pool.bounds[e]]
                .iter()
                .filter(|&&k| score[k].is_finite())
                .map(|&k| (score[k], target(k)))
                .unzip();
            if s.len() >= 8 {
                let ic = spearman(&s, &t);
                if ic.is_finite() {
                    out.push(ic);
                }
            }
            b = e;
        }
    }
    out
}

fn sign_share(values: &[f64]) -> f64 {
    let f: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if f.is_empty() {
        return f64::NAN;
    }
    f.iter().filter(|v| **v > 0.0).count() as f64 / f.len() as f64
}

fn build_rows(
    p: &Panel,
    pool: &Pool,
    opts: &Options,
    block: usize,
    rungs: Vec<Rung>,
    market: &[f64],
    zero_param_net: Option<f64>,
) -> Vec<Row> {
    let side = opts.cost_bps * 1e-4;
    let mut out = Vec::with_capacity(rungs.len());
    for r in rungs {
        let mut row = Row {
            name: r.name.to_string(),
            kind: r.kind.to_string(),
            status: r.status.to_string(),
            role: r.role.to_string(),
            note: r.note.clone(),
            params: r.params.clone(),
            fits: r.fits,
            trials: r.trials,
            wall_seconds: r.wall,
            exposure: f64::NAN,
            beta: f64::NAN,
            net_by_cost: Vec::new(),
            mean_period_net: f64::NAN,
            mean_ci: (f64::NAN, f64::NAN),
            sharpe: f64::NAN,
            p_mean_positive: f64::NAN,
            bh_rejected: None,
            bh_adjusted: f64::NAN,
            turnover: f64::NAN,
            trades: 0,
            break_even_bps: f64::NAN,
            hit_rate: f64::NAN,
            return_mse: f64::NAN,
            rank_ic: f64::NAN,
            rank_ic_ci: (f64::NAN, f64::NAN),
            fold_sign_share: f64::NAN,
            group_sign_share: f64::NAN,
            excess_over_cash: f64::NAN,
            beats_zero_param: None,
            series: Vec::new(),
        };
        if r.pos.is_empty() {
            out.push(row);
            continue;
        }
        let free = r.name == "equal_weight_universe";
        let (net, gross) = price(p, pool, &r.pos, if free { 0.0 } else { side });
        let net_p = pool.periods(&net);
        let gross_p = pool.periods(&gross);
        let n = r.pos.len() as f64;
        row.exposure = r.pos.iter().sum::<f64>() / n;
        for &c in &opts.cost_grid {
            let (nc, _) = price(p, pool, &r.pos, if free { 0.0 } else { c * 1e-4 });
            row.net_by_cost.push((c, pool.periods(&nc).iter().sum()));
        }
        let rs = risk_stats(&net_p, pool.ppy, DEFAULT_SORTINO_MAR);
        row.mean_period_net = rs.mean;
        row.sharpe = rs.sharpe;
        if net_p.len() >= 3 {
            let bs = inference::block_bootstrap(&net_p, block, opts.draws, opts.seed, pool.ppy, 0.95);
            row.mean_ci = (bs.mean_ci.lo, bs.mean_ci.hi);
            row.p_mean_positive = bs.p_mean_positive;
        }
        let mut ba = BetaAlpha::identity();
        for (x, y) in market.iter().zip(&net_p) {
            ba.push(*x, *y);
        }
        row.beta = ba.beta();
        row.excess_over_cash = net_p.iter().sum();
        let mut dp = 0.0;
        let mut entries = 0usize;
        let mut long_rows = 0.0;
        let mut hits = 0.0;
        for &(s, e) in &pool.ranges {
            let mut prev_g = u64::MAX;
            let mut prev = 0.0;
            for k in s..e {
                let g = p.group[pool.idx[k]];
                if g != prev_g {
                    prev = 0.0;
                    prev_g = g;
                }
                let cur = r.pos[k];
                dp += (cur - prev).abs();
                if cur > 0.0 && prev == 0.0 {
                    entries += 1;
                }
                if cur > 0.0 {
                    long_rows += 1.0;
                    if p.ret[pool.idx[k]] > 0.0 {
                        hits += 1.0;
                    }
                }
                prev = cur;
            }
        }
        row.turnover = dp / n;
        row.trades = entries;
        row.hit_rate = if long_rows > 0.0 { hits / long_rows } else { f64::NAN };
        row.break_even_bps = inference::break_even_bps(&net_p, &gross_p, opts.cost_bps);
        let fold_net: Vec<f64> = pool
            .ranges
            .iter()
            .map(|&(s, e)| net[s..e].iter().sum::<f64>())
            .collect();
        row.fold_sign_share = sign_share(&fold_net);
        let mut by_group: Vec<(u64, f64)> = Vec::new();
        for k in 0..pool.len() {
            let g = p.group[pool.idx[k]];
            match by_group.last_mut() {
                Some(last) if last.0 == g => last.1 += net[k],
                _ => by_group.push((g, net[k])),
            }
        }
        row.group_sign_share = sign_share(&by_group.iter().map(|x| x.1).collect::<Vec<_>>());
        if !r.score.is_empty() {
            let ics = ic_series(p, pool, &r.score);
            if !ics.is_empty() {
                row.rank_ic = inference::mean(&ics);
                if ics.len() >= 3 {
                    let bs = inference::block_bootstrap(&ics, 1, opts.draws, opts.seed, 1.0, 0.95);
                    row.rank_ic_ci = (bs.mean_ci.lo, bs.mean_ci.hi);
                }
            }
            if r.predicts_target {
                row.return_mse = return_space_mse(p, pool, opts, &r.score);
            }
        }
        if r.status == "ok" && r.kind == "model" && r.name != "zero_param" {
            if let Some(z) = zero_param_net {
                row.beats_zero_param = Some(row.excess_over_cash > z);
            }
        }
        row.series = net_p;
        out.push(row);
    }
    out
}

fn return_space_mse(p: &Panel, pool: &Pool, opts: &Options, score: &[f64]) -> f64 {
    let scale = opts
        .spec
        .as_ref()
        .and_then(|s| s.scale_column().map(|c| (s, c)))
        .and_then(|(s, c)| p.col(c).map(|col| (s, col)));
    let (mut s, mut c) = (0.0, 0.0);
    for k in 0..pool.len() {
        let i = pool.idx[k];
        let pred = match &scale {
            Some((spec, g)) => spec.return_from_target(score[k], g[i]),
            None => score[k],
        };
        if pred.is_finite() && p.ret[i].is_finite() {
            s += (pred - p.ret[i]).powi(2);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        f64::NAN
    }
}

/// Run the whole ladder.
pub fn run_ladder(p: &Panel, folds: &[Fold], opts: &Options) -> Result<Report, String> {
    let started = Instant::now();
    let mut report = run_ladder_rungs(p, folds, opts)?;
    let wall = started.elapsed().as_secs_f64();
    report.ledger.record_stage(Stage::Pnl, wall);
    report.ledger.set_wall(wall);
    Ok(report)
}

fn run_ladder_rungs(p: &Panel, folds: &[Fold], opts: &Options) -> Result<Report, String> {
    if opts.quantiles.is_empty() {
        return Err("--quantiles needs at least one value".to_string());
    }
    let pool = Pool::new(p, folds);
    let feats: Vec<usize> = (0..p.names.len()).collect();
    let mut rungs: Vec<Rung> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let n = pool.len();

    let mut cash = Rung::new("cash", "control");
    cash.pos = vec![0.0; n];
    rungs.push(cash);
    let mut bh = Rung::new("buy_and_hold", "control");
    bh.pos = vec![1.0; n];
    rungs.push(bh);
    let mut ew = Rung::new("equal_weight_universe", "control");
    ew.pos = vec![1.0; n];
    ew.note = Some("every row long, no cost".to_string());
    rungs.push(ew);

    let mut constant = Rung::new("constant_long_flat", "control");
    constant.pos = vec![0.0; n];
    for (fi, f) in folds.iter().enumerate() {
        let ys: Vec<f64> = f.fit.iter().map(|&i| p.y[i]).filter(|v| v.is_finite()).collect();
        let long = baselines::constant_is_long(ys.iter().sum::<f64>() / ys.len().max(1) as f64);
        let (s, e) = pool.ranges[fi];
        for v in &mut constant.pos[s..e] {
            *v = if long { 1.0 } else { 0.0 };
        }
    }
    constant.params = "long iff the fit-row mean target is strictly positive".to_string();
    rungs.push(constant);

    // zero-parameter target structure
    match &opts.spec {
        Some(spec) => match spec.scale_column().and_then(|c| p.col(c)) {
            Some(g) => {
                let s_all = target_structure::structure_values(spec, g);
                let mut r = score_rung(
                    p,
                    folds,
                    &pool,
                    opts,
                    Rung::new("zero_param", "model"),
                    &mut |f: &Fold| (rows_of(&s_all, &f.fit), rows_of(&s_all, &f.valid), rows_of(&s_all, &f.test)),
                );
                r.predicts_target = true;
                r.note = Some("construction identity: the label at a zero return".to_string());
                rungs.push(r);
            }
            None => rungs.push(Rung::not_run(
                "zero_param",
                "model",
                "the target spec has no scale column in the panel",
            )),
        },
        None => rungs.push(Rung::not_run("zero_param", "model", "needs --target-spec")),
    }

    // OLS / ridge on the full projected view
    let started = Instant::now();
    let mut ols_pool_score: Vec<f64> = Vec::new();
    let mut ols_rung = {
        let mut lambdas_chosen = Vec::new();
        let mut scores: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
        for f in folds {
            let fit_rows = thin(&f.fit, opts.max_fit_rows);
            let path = fit_ridge_path(p, &feats, &fit_rows, &RIDGE_LAMBDAS);
            let cols: Vec<&[f64]> = p.x.iter().map(|c| c.as_slice()).collect();
            let eval = |m: &Ridge, rows: &[usize]| -> Vec<f64> { rows.iter().map(|&i| m.predict(&cols, i)).collect() };
            let mut best = (f64::INFINITY, 0usize);
            for (li, m) in path.iter().enumerate() {
                let v = mse_of(&eval(m, &f.valid), &f.valid, &p.y);
                if v < best.0 {
                    best = (v, li);
                }
            }
            let m = &path[best.1];
            lambdas_chosen.push(RIDGE_LAMBDAS[best.1]);
            scores.push((eval(m, &f.fit), eval(m, &f.valid), eval(m, &f.test)));
        }
        let mut k = 0usize;
        let mut r = score_rung(p, folds, &pool, opts, Rung::new("ols_ridge", "model"), &mut |_f: &Fold| {
            let s = scores[k].clone();
            k += 1;
            s
        });
        r.predicts_target = true;
        r.fits = (folds.len() * RIDGE_LAMBDAS.len()) as u64;
        r.trials = (RIDGE_LAMBDAS.len() * opts.quantiles.len()) as u64;
        r.params = format!("{}; lambda_by_fold={}", r.params, jarr(&lambdas_chosen));
        r.wall = started.elapsed().as_secs_f64();
        r
    };
    ols_rung.note = Some(format!("{} features, standardised on fit rows", feats.len()));
    ols_pool_score.extend_from_slice(&ols_rung.score);
    let ols_positions = ols_rung.pos.clone();

    // best single feature, selected on validation
    let started = Instant::now();
    let best_single = {
        let mut names_by_fold: Vec<String> = Vec::new();
        let mut r = Rung::new("best_single_feature", "model");
        r.pos = vec![0.0; n];
        r.score = vec![f64::NAN; n];
        let side = opts.cost_bps * 1e-4;
        for (fi, f) in folds.iter().enumerate() {
            let mut best: Option<(f64, usize, f64, f64)> = None;
            for &j in &feats {
                let col = &p.x[j];
                let fy: Vec<f64> = f.fit.iter().map(|&i| p.y[i]).collect();
                let fx: Vec<f64> = f.fit.iter().map(|&i| col[i]).collect();
                let c = corr(&fx, &fy);
                if !c.is_finite() {
                    continue;
                }
                let sgn = if c >= 0.0 { 1.0 } else { -1.0 };
                let fit_s: Vec<f64> = fx.iter().map(|v| sgn * v).collect();
                for &q in &opts.quantiles {
                    let theta = finite_quantile(&fit_s, q);
                    if !theta.is_finite() {
                        continue;
                    }
                    let pos: Vec<f64> = f.valid.iter().map(|&i| if sgn * col[i] > theta { 1.0 } else { 0.0 }).collect();
                    let net = net_sum_slice(p, &f.valid, &pos, side);
                    if best.map_or(true, |b| net > b.0) {
                        best = Some((net, j, sgn, theta));
                    }
                }
            }
            let (s, _) = pool.ranges[fi];
            if let Some((_, j, sgn, theta)) = best {
                names_by_fold.push(format!("{}{}", if sgn < 0.0 { "-" } else { "" }, p.names[j]));
                for (k, &i) in f.test.iter().enumerate() {
                    let sc = sgn * p.x[j][i];
                    r.score[s + k] = sc;
                    r.pos[s + k] = if sc > theta { 1.0 } else { 0.0 };
                }
            } else {
                names_by_fold.push("none".to_string());
            }
        }
        r.trials = (feats.len() * opts.quantiles.len()) as u64;
        r.fits = (feats.len() * folds.len()) as u64;
        r.params = format!("feature_by_fold={}", jarr_str(&names_by_fold));
        r.note = Some(format!("selected on validation over {} features; trials counted", feats.len()));
        r.wall = started.elapsed().as_secs_f64();
        r
    };

    // 20-bin E[y | vol]
    let vol_rung = match opts.vol_col.as_deref().and_then(|c| p.col(c).map(|v| (c, v))) {
        Some((_, vol)) => {
            let mut scores: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
            for f in folds {
                let s = rows_of(vol, &thin(&f.fit, opts.max_fit_rows));
                let y: Vec<f64> = thin(&f.fit, opts.max_fit_rows).iter().map(|&i| p.y[i]).collect();
                let m = target_structure::fit(&s, &s, &y, &[]);
                let pred = |rows: &[usize]| {
                    let v = rows_of(vol, rows);
                    target_structure::predict(&m, &v, &v).binned20
                };
                scores.push((pred(&f.fit), pred(&f.valid), pred(&f.test)));
            }
            let mut k = 0usize;
            let mut r = score_rung(p, folds, &pool, opts, Rung::new("vol_binned20", "model"), &mut |_f: &Fold| {
                let s = scores[k].clone();
                k += 1;
                s
            });
            r.predicts_target = true;
            r.fits = folds.len() as u64;
            r
        }
        None => Rung::not_run("vol_binned20", "model", "needs --vol-col"),
    };

    // naive long-only reversal: long after a down move
    let reversal = match opts.reversal_col.as_deref().and_then(|c| p.col(c)) {
        Some(col) => {
            let mut r = Rung::new("naive_reversal", "model");
            r.pos = pool.idx.iter().map(|&i| if col[i] < 0.0 { 1.0 } else { 0.0 }).collect();
            r.score = pool.idx.iter().map(|&i| -col[i]).collect();
            r.params = "long iff the lagged return column is negative".to_string();
            r
        }
        None => Rung::not_run("naive_reversal", "model", "needs --reversal-col"),
    };

    let tercile = match top_tercile(&pool, &ols_pool_score) {
        Some(pos) => {
            let mut r = Rung::new("top_tercile_long", "model");
            r.pos = pos;
            r.score = ols_pool_score.clone();
            r.params = "long the top tercile of the ols_ridge score per timestamp, cash otherwise".to_string();
            r
        }
        None => Rung::not_run("top_tercile_long", "model", "fewer than 3 symbols per timestamp"),
    };

    // 1[f > tau] with turnover cost
    let feature_threshold = match opts.signal_col.as_deref().and_then(|c| p.col(c)) {
        Some(col) => {
            let all: Vec<f64> = col.to_vec();
            score_rung(p, folds, &pool, opts, Rung::new("feature_threshold", "model"), &mut |f: &Fold| {
                (rows_of(&all, &f.fit), rows_of(&all, &f.valid), rows_of(&all, &f.test))
            })
        }
        None => Rung::not_run("feature_threshold", "model", "needs --signal-col"),
    };

    // tuned GBM
    let started = Instant::now();
    let gbm = {
        let mut scores: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut shapes = Vec::new();
        let mut grown = 0u64;
        for f in folds {
            let m = fit_gbm(p, &feats, &f.fit, &f.valid, opts.gbm_trees, opts.max_fit_rows);
            grown += m.trees_grown as u64;
            shapes.push(format!("d{}/lr{}/t{}", m.depth, m.lr, m.trees.len()));
            scores.push((
                gbm_predict(&m, p, &feats, &thin(&f.fit, opts.max_fit_rows)),
                gbm_predict(&m, p, &feats, &f.valid),
                gbm_predict(&m, p, &feats, &f.test),
            ));
        }
        // The threshold is fitted on the thinned fit scores, same rule as the rest.
        let mut k = 0usize;
        let mut r = score_rung(p, folds, &pool, opts, Rung::new("gbm", "model"), &mut |_f: &Fold| {
            let s = scores[k].clone();
            k += 1;
            s
        });
        r.predicts_target = true;
        r.fits = grown;
        r.trials = (GBM_CONFIGS.len() * (opts.gbm_trees / GBM_CONFIGS.len() / GBM_CHECK_EVERY).max(1) * opts.quantiles.len()) as u64;
        r.params = format!("{}; chosen={}; tree_budget_per_fold={}", r.params, jarr_str(&shapes), opts.gbm_trees);
        r.wall = started.elapsed().as_secs_f64();
        r
    };

    // symbolic regression, scored by the caller's out-of-fold column
    let sr = match opts.sr_score_col.as_deref().and_then(|c| p.col(c)) {
        Some(col) => {
            let all: Vec<f64> = col.to_vec();
            let mut r = score_rung(p, folds, &pool, opts, Rung::new("symbolic_regression", "model"), &mut |f: &Fold| {
                (rows_of(&all, &f.fit), rows_of(&all, &f.valid), rows_of(&all, &f.test))
            });
            r.fits = opts.sr_evals.unwrap_or(0);
            r.wall = opts.sr_wall_seconds.unwrap_or(0.0);
            r.predicts_target = true;
            r
        }
        None => Rung::not_run("symbolic_regression", "model", "needs --sr-score (runs in the pilot and campaign)"),
    };

    // matched-exposure random long, matched to the strongest searched or linear book
    let reference: &[f64] = if sr.status == "ok" && !sr.pos.is_empty() { &sr.pos } else { &ols_positions };
    let mut matched = Rung::new("matched_exposure_random_long", "control");
    matched.pos = vec![0.0; n];
    for &(s, e) in &pool.ranges {
        let g: Vec<u64> = pool.idx[s..e].iter().map(|&i| p.group[i]).collect();
        let m = baselines::matched_exposure_positions(&reference[s..e], &g, opts.seed as u64)?;
        matched.pos[s..e].copy_from_slice(&m);
    }
    matched.params = if sr.status == "ok" {
        "run lengths of the symbolic_regression book shuffled per symbol".to_string()
    } else {
        "run lengths of the ols_ridge book shuffled per symbol".to_string()
    };

    rungs.push(ols_rung);
    rungs.push(best_single);
    rungs.push(vol_rung);
    rungs.push(reversal);
    rungs.push(tercile);
    rungs.push(feature_threshold);
    rungs.push(gbm);
    rungs.push(sr);
    rungs.push(matched);

    // market series and statistics
    let market_rows: Vec<f64> = pool.idx.iter().map(|&i| p.ret[i]).collect();
    let market = pool.periods(&market_rows);
    let step = pool.step;
    let horizon_periods = match step {
        Some(dt) if opts.horizon_seconds > 0.0 => (opts.horizon_seconds / dt).ceil() as usize,
        _ => 1,
    };
    let block = opts
        .block_len
        .unwrap_or_else(|| inference::auto_block_len(market.len(), horizon_periods, 1));

    let mut sr_zero = None;
    let zero_net = {
        let z = rungs.iter().find(|r| r.name == "zero_param" && !r.pos.is_empty());
        z.map(|r| {
            let (net, _) = price(p, &pool, &r.pos, opts.cost_bps * 1e-4);
            pool.periods(&net).iter().sum::<f64>()
        })
    };
    if let Some(z) = zero_net {
        sr_zero = Some(z);
    }
    let rows = build_rows(p, &pool, opts, block, rungs, &market, sr_zero);
    finish_report(p, &pool, opts, folds, rows, &market, block, &mut notes)
}

fn jarr_str(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(","))
}

fn finish_report(
    _p: &Panel,
    pool: &Pool,
    opts: &Options,
    folds: &[Fold],
    mut rows: Vec<Row>,
    market: &[f64],
    block: usize,
    notes: &mut Vec<String>,
) -> Result<Report, String> {
    let ran = |r: &Row| r.status == "ok";
    let mean_of = |name: &str, rows: &[Row]| rows.iter().find(|r| r.name == name && ran(r)).map(|r| r.mean_period_net);
    // Benchmark: best of cash, buy-and-hold and the constant rule by mean net.
    let mut benchmark = "cash".to_string();
    let mut best = mean_of("cash", &rows).unwrap_or(0.0);
    for name in ["buy_and_hold", "constant_long_flat"] {
        if let Some(m) = mean_of(name, &rows) {
            if m > best {
                best = m;
                benchmark = name.to_string();
            }
        }
    }
    let rule_names = [
        "constant_long_flat",
        "zero_param",
        "ols_ridge",
        "best_single_feature",
        "vol_binned20",
        "naive_reversal",
        "top_tercile_long",
        "feature_threshold",
    ];
    let mut best_rule = String::new();
    let mut best_rule_mean = f64::NEG_INFINITY;
    for name in rule_names {
        if let Some(m) = mean_of(name, &rows) {
            if m > best_rule_mean {
                best_rule_mean = m;
                best_rule = name.to_string();
            }
        }
    }

    // BH-FDR on the one-sided p-values of the non-control rungs that ran.
    let tested: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| ran(r) && r.kind == "model" && r.p_mean_positive.is_finite())
        .map(|(i, _)| i)
        .collect();
    let bh = if tested.is_empty() {
        None
    } else {
        let ps: Vec<f64> = tested.iter().map(|&i| rows[i].p_mean_positive).collect();
        let b = inference::bh_fdr(&ps, opts.fdr_q);
        for (k, &i) in tested.iter().enumerate() {
            rows[i].bh_rejected = Some(b.rejected[k]);
            rows[i].bh_adjusted = b.adjusted[k];
        }
        Some(b)
    };

    // SPA of every model rung against the benchmark.
    let bench_series: Vec<f64> = rows
        .iter()
        .find(|r| r.name == benchmark)
        .map(|r| r.series.clone())
        .unwrap_or_default();
    let d: Vec<Vec<f64>> = tested
        .iter()
        .map(|&i| rows[i].series.iter().zip(&bench_series).map(|(a, b)| a - b).collect())
        .collect();
    let (spa, spa_error) = if d.is_empty() {
        (None, Some("no model rung ran".to_string()))
    } else {
        match inference::spa_test(&d, block, opts.draws, opts.seed) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(e)),
        }
    };

    let n_trials: u64 = rows.iter().filter(|r| ran(r)).map(|r| r.trials).sum();

    // DSR for the best model rung by Sharpe, deflated by the whole ladder's trials.
    let dsr = {
        let sharpes: Vec<f64> = tested.iter().map(|&i| inference::sharpe_per_period(&rows[i].series)).collect();
        let finite: Vec<f64> = sharpes.iter().copied().filter(|s| s.is_finite()).collect();
        let var_sr = {
            let sd = inference::sample_sd(&finite);
            sd * sd
        };
        tested
            .iter()
            .zip(&sharpes)
            .filter(|(_, s)| s.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&i, &s)| {
                let x = &rows[i].series;
                (
                    rows[i].name.clone(),
                    inference::deflated_sharpe(
                        s,
                        x.len() as f64,
                        inference::skewness(x),
                        inference::kurtosis(x),
                        n_trials as f64,
                        var_sr,
                    ),
                )
            })
    };

    let cols: Vec<Vec<f64>> = tested.iter().map(|&i| rows[i].series.clone()).collect();
    let (pbo, pbo_error) = if cols.len() < 2 {
        (None, Some("PBO needs at least two model rungs".to_string()))
    } else {
        match inference::pbo_cscv(&cols, opts.pbo_blocks) {
            Ok(p) => (Some(p), None),
            Err(e) => (None, Some(e)),
        }
    };

    let net_of = |name: &str| rows.iter().find(|r| r.name == name && ran(r)).map(|r| r.excess_over_cash);
    let gbm_beats_sr = match (net_of("gbm"), net_of("symbolic_regression")) {
        (Some(g), Some(s)) => Some(g > s),
        _ => None,
    };
    for r in rows.iter_mut() {
        if r.name == "symbolic_regression" && ran(r) {
            r.role = if gbm_beats_sr == Some(true) {
                "interpretability_only".to_string()
            } else {
                "competitive".to_string()
            };
        }
    }
    notes.push(
        "selection on the fit and validation slices is repeated inside every fold; the test slice is scored once"
            .to_string(),
    );
    notes.push(format!(
        "the benchmark for SPA is {} (best of cash, buy_and_hold and constant_long_flat by mean period net)",
        benchmark
    ));
    let _ = market;
    Ok(Report {
        rows,
        folds: folds.iter().map(|f| f.label.clone()).collect(),
        periods: pool.times.len(),
        rows_scored: pool.len(),
        periods_per_year: pool.ppy,
        block_len: block,
        cost_bps: opts.cost_bps,
        n_trials,
        benchmark,
        best_long_flat_rule: best_rule,
        bh,
        spa,
        spa_error,
        dsr,
        pbo,
        pbo_error,
        draws: opts.draws,
        seed: opts.seed,
        fdr_q: opts.fdr_q,
        gbm_beats_sr,
        sr_evals: opts.sr_evals,
        sr_wall_seconds: opts.sr_wall_seconds,
        gbm_trees: opts.gbm_trees,
        notes: std::mem::take(notes),
        ledger: RunLedger::new(),
    })
}

impl Report {
    pub fn row(&self, name: &str) -> Option<&Row> {
        self.rows.iter().find(|r| r.name == name)
    }

    pub fn json(&self) -> String {
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                let nets: Vec<String> = r
                    .net_by_cost
                    .iter()
                    .map(|(c, v)| format!("{{\"cost_bps\":{},\"net\":{}}}", jnum(*c), jnum(*v)))
                    .collect();
                format!(
                    "{{\"name\":{},\"kind\":{},\"status\":{},\"role\":{},\"note\":{},\"params\":{},\
\"candidate_evaluations\":{},\"trials\":{},\"wall_seconds\":{},\"exposure\":{},\"beta\":{},\
\"net_by_cost\":[{}],\"mean_period_net\":{},\"mean_ci\":[{},{}],\"sharpe\":{},\"p_mean_positive\":{},\
\"bh_rejected\":{},\"bh_adjusted\":{},\"turnover_per_row\":{},\"entries\":{},\"break_even_bps\":{},\
\"hit_rate\":{},\"return_space_mse\":{},\"rank_ic\":{},\"rank_ic_ci\":[{},{}],\
\"fold_sign_share\":{},\"symbol_sign_share\":{},\"net_over_cash\":{},\"beats_zero_param\":{}}}",
                    jstr(&r.name),
                    jstr(&r.kind),
                    jstr(&r.status),
                    jstr(&r.role),
                    r.note.as_deref().map_or("null".to_string(), jstr),
                    jstr(&r.params),
                    r.fits,
                    r.trials,
                    jnum(r.wall_seconds),
                    jnum(r.exposure),
                    jnum(r.beta),
                    nets.join(","),
                    jnum(r.mean_period_net),
                    jnum(r.mean_ci.0),
                    jnum(r.mean_ci.1),
                    jnum(r.sharpe),
                    jnum(r.p_mean_positive),
                    r.bh_rejected.map_or("null".to_string(), |b| b.to_string()),
                    jnum(r.bh_adjusted),
                    jnum(r.turnover),
                    r.trades,
                    jnum(r.break_even_bps),
                    jnum(r.hit_rate),
                    jnum(r.return_mse),
                    jnum(r.rank_ic),
                    jnum(r.rank_ic_ci.0),
                    jnum(r.rank_ic_ci.1),
                    jnum(r.fold_sign_share),
                    jnum(r.group_sign_share),
                    jnum(r.excess_over_cash),
                    r.beats_zero_param.map_or("null".to_string(), |b| b.to_string()),
                )
            })
            .collect();
        let spa = match &self.spa {
            Some(s) => format!(
                "{{\"models\":{},\"periods\":{},\"block_len\":{},\"draws\":{},\"statistic\":{},\"p_consistent\":{},\"p_lower\":{},\"p_upper\":{}}}",
                s.models,
                s.n,
                s.block_len,
                s.draws,
                jnum(s.statistic),
                jnum(s.p_consistent),
                jnum(s.p_lower),
                jnum(s.p_upper)
            ),
            None => format!("{{\"error\":{}}}", jstr(self.spa_error.as_deref().unwrap_or(""))),
        };
        let bh = match &self.bh {
            Some(b) => format!("{{\"q\":{},\"rejected\":{}}}", jnum(b.q), b.n_rejected),
            None => "null".to_string(),
        };
        let dsr = match &self.dsr {
            Some((name, d)) => format!(
                "{{\"rung\":{},\"sharpe_per_period\":{},\"periods\":{},\"n_trials\":{},\"var_sr\":{},\"sr0\":{},\"dsr\":{}}}",
                jstr(name),
                jnum(d.sharpe),
                jnum(d.t),
                jnum(d.n_trials),
                jnum(d.var_sr),
                jnum(d.sr0),
                jnum(d.dsr)
            ),
            None => "null".to_string(),
        };
        let pbo = match &self.pbo {
            Some(p) => format!(
                "{{\"candidates\":{},\"blocks\":{},\"combinations\":{},\"pbo\":{}}}",
                p.candidates,
                p.blocks,
                p.combinations,
                jnum(p.pbo)
            ),
            None => format!("{{\"error\":{}}}", jstr(self.pbo_error.as_deref().unwrap_or(""))),
        };
        format!(
            "{{\"schema\":{},\"folds\":{},\"periods\":{},\"rows_scored\":{},\"periods_per_year\":{},\"block_len\":{},\
\"cost_bps\":{},\"n_trials\":{},\"benchmark\":{},\"best_long_flat_rule\":{},\"equal_cost\":{{\"gbm_tree_budget_per_fold\":{},\
\"sr_candidate_evaluations\":{},\"sr_wall_seconds\":{},\"gbm_beats_sr\":{}}},\"multiple_testing\":{{\"draws\":{},\"seed\":{},\
\"fdr\":{},\"spa\":{},\"dsr\":{},\"pbo\":{}}},\"notes\":{},\"rungs\":[{}],\"ledger\":{}}}",
            jstr(SCHEMA),
            jarr_str(&self.folds),
            self.periods,
            self.rows_scored,
            jnum(self.periods_per_year),
            self.block_len,
            jnum(self.cost_bps),
            self.n_trials,
            jstr(&self.benchmark),
            jstr(&self.best_long_flat_rule),
            self.gbm_trees,
            self.sr_evals.map_or("null".to_string(), |v| v.to_string()),
            jopt(self.sr_wall_seconds),
            self.gbm_beats_sr.map_or("null".to_string(), |b| b.to_string()),
            self.draws,
            self.seed,
            bh,
            spa,
            dsr,
            pbo,
            jarr_str(&self.notes),
            rows.join(","),
            if self.ledger.wall_s() > 0.0 {
                self.ledger.phase_json(
                    "ladder",
                    &[Stage::Vary, Stage::Eval, Stage::Fit, Stage::Select, Stage::Archive, Stage::Features],
                )
            } else {
                "null".to_string()
            }
        )
    }

    pub fn markdown(&self) -> String {
        let f = |v: f64, d: usize| {
            if v.is_finite() {
                format!("{:.*}", d, v)
            } else {
                "-".to_string()
            }
        };
        let mut s = String::new();
        s.push_str(&format!(
            "# Baseline ladder\n\n{} folds, {} periods, {} test rows, cost {} bp per side, N_trials {}.\n\n",
            self.folds.len(),
            self.periods,
            self.rows_scored,
            self.cost_bps,
            self.n_trials
        ));
        s.push_str("| rung | status | exposure | beta | net over cash |");
        if let Some(r) = self.rows.first() {
            for (c, _) in &r.net_by_cost {
                s.push_str(&format!(" net @{} bp |", c));
            }
        }
        s.push_str(" mean net [95% CI] | Sharpe | break-even bp | hit rate | rank IC | MSE | BH | beats zero-param | evals | wall s |\n");
        s.push_str("|---|---|---|---|---|");
        if let Some(r) = self.rows.first() {
            for _ in &r.net_by_cost {
                s.push_str("---|");
            }
        }
        s.push_str("---|---|---|---|---|---|---|---|---|---|\n");
        for r in &self.rows {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} |",
                r.name,
                if r.role.is_empty() { r.status.clone() } else { format!("{} ({})", r.status, r.role) },
                f(r.exposure, 3),
                f(r.beta, 3),
                f(r.excess_over_cash, 5)
            ));
            for (_, v) in &r.net_by_cost {
                s.push_str(&format!(" {} |", f(*v, 5)));
            }
            s.push_str(&format!(
                " {} [{}, {}] | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                f(r.mean_period_net, 7),
                f(r.mean_ci.0, 7),
                f(r.mean_ci.1, 7),
                f(r.sharpe, 2),
                f(r.break_even_bps, 1),
                f(r.hit_rate, 3),
                f(r.rank_ic, 4),
                f(r.return_mse, 8),
                r.bh_rejected.map_or("-".to_string(), |b| if b { "reject".into() } else { "keep".into() }),
                r.beats_zero_param.map_or("-".to_string(), |b| b.to_string()),
                if r.fits > 0 { r.fits.to_string() } else { "-".to_string() },
                f(r.wall_seconds, 2),
            ));
        }
        s.push_str(&format!(
            "\nBenchmark {}; best long/flat rule {}.\n",
            self.benchmark, self.best_long_flat_rule
        ));
        if let Some(sp) = &self.spa {
            s.push_str(&format!(
                "\nSPA against the benchmark: p {} (consistent), {} (lower), {} (upper), {} models, block {}.\n",
                f(sp.p_consistent, 4),
                f(sp.p_lower, 4),
                f(sp.p_upper, 4),
                sp.models,
                sp.block_len
            ));
        }
        if let Some((n, d)) = &self.dsr {
            s.push_str(&format!(
                "\nDSR of {}: {} with N_trials {} (expected null Sharpe {}).\n",
                n,
                f(d.dsr, 4),
                d.n_trials,
                f(d.sr0, 5)
            ));
        }
        if let Some(p) = &self.pbo {
            s.push_str(&format!("\nPBO {} over {} rungs.\n", f(p.pbo, 4), p.candidates));
        }
        if self.gbm_beats_sr == Some(true) {
            s.push_str("\nThe tuned GBM beats symbolic regression on net over cash; symbolic regression is interpretability only.\n");
        }
        for n in &self.notes {
            s.push_str(&format!("\n{}.\n", n));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Loading and CLI
// ---------------------------------------------------------------------------

pub struct Input {
    pub data: String,
    pub target: String,
    pub ret: Option<String>,
    pub time: String,
    pub group: Option<String>,
    pub features: Vec<String>,
}

/// Load the columns the ladder needs from a CSV or a `.rcsb` store.
pub fn load_panel(inp: &Input, opts: &Options) -> Result<Panel, String> {
    let ret_name = inp.ret.clone().unwrap_or_else(|| inp.target.clone());
    let mut want: Vec<String> = vec![inp.target.clone(), ret_name.clone(), inp.time.clone()];
    if let Some(g) = &inp.group {
        want.push(g.clone());
    }
    let mut extras: Vec<String> = Vec::new();
    for c in [&opts.vol_col, &opts.reversal_col, &opts.signal_col, &opts.sr_score_col]
        .into_iter()
        .flatten()
    {
        extras.push(c.clone());
    }
    if let Some(s) = &opts.spec {
        if let Some(c) = s.scale_column() {
            extras.push(c.to_string());
        }
    }
    want.extend(extras.iter().cloned());
    want.extend(inp.features.iter().cloned());
    want.sort();
    want.dedup();
    let table = if crate::store::is_store_path(&inp.data) {
        let fs = crate::store::FileStore::open(&inp.data)?;
        let names: Vec<String> = want.clone();
        let proj = crate::store::Projection::resolve_owned(&fs, &names)?;
        fs.to_table_projected(&proj)?
    } else {
        data::read_table(&inp.data)?
    };
    panel_from_table(&table, inp, &ret_name, &extras)
}

pub fn panel_from_table(table: &Table, inp: &Input, ret_name: &str, extras: &[String]) -> Result<Panel, String> {
    let time = time_column(table, &inp.time)?;
    let group = group_column(table, inp.group.as_deref())?;
    let y = numeric_column(table, &inp.target)?;
    let ret = numeric_column(table, ret_name)?;
    let feats: Vec<String> = if inp.features.is_empty() {
        table
            .names
            .iter()
            .enumerate()
            .filter(|(j, n)| {
                table.numeric[*j]
                    && **n != inp.target
                    && **n != ret_name
                    && **n != inp.time
                    && Some(n.as_str()) != inp.group.as_deref()
                    && !extras.contains(n)
            })
            .map(|(_, n)| n.clone())
            .collect()
    } else {
        inp.features.clone()
    };
    if feats.is_empty() {
        return Err("no feature columns: pass --features or add numeric columns".to_string());
    }
    let mut x = Vec::new();
    for f in &feats {
        x.push(numeric_column(table, f)?);
    }
    let mut extra = Vec::new();
    for e in extras {
        if !feats.contains(e) {
            extra.push((e.clone(), numeric_column(table, e)?));
        }
    }
    Panel::new(feats, x, time, group, y, ret, extra)
}

fn usage() -> &'static str {
    "usage: rcsr ladder --data <csv|rcsb> --target <col> --time <col> --windows <protocol/windows.csv> [options]\n\
\n\
Long-only baseline ladder on one panel, one set of folds and one cost model.\n\
\n\
  --data <path>             CSV or .rcsb store (the pinned panel projection)\n\
  --target <col>            fit target (the label); the return is --return-col\n\
  --return-col <col>        realised forward return of the row (default: --target)\n\
  --time <col>              epoch seconds or ISO dates\n\
  --group <col>             symbol column (default: one group)\n\
  --features <a,b,...>      feature columns (default: every other numeric column)\n\
  --windows <file>          protocol windows table (rcsr protocol windows)\n\
  --scheme <name>           expanding | rolling | locked (default expanding)\n\
  --purge-seconds <s>       gap between the fit range and the validation slice (default 219600)\n\
  --horizon-seconds <s>     label horizon; floors the bootstrap block (default 0)\n\
  --cost-bps <x>            per-side cost of the headline book (default 12)\n\
  --cost-grid <a,b,...>     per-side costs reported per rung (default 12,20,40)\n\
  --quantiles <a,b,...>     score-threshold grid chosen on validation (default 0.5,0.6,0.7,0.8,0.9)\n\
  --vol-col <col>           volatility column for the 20-bin E[y|sigma] rung\n\
  --reversal-col <col>      lagged return column for the naive reversal rung\n\
  --signal-col <col>        feature for the 1[f > tau] rung\n\
  --target-spec <file>      target spec file for the zero-parameter rung (scale column must be in the panel)\n\
  --sr-score <col>          out-of-fold symbolic-regression score column\n\
  --sr-evals <n>            candidate evaluations of the symbolic-regression run\n\
  --sr-wall <s>             wall seconds of the symbolic-regression run\n\
  --gbm-trees <n>           GBM tuning budget in trees per fold (default 240)\n\
  --max-fit-rows <n>        cap on fit rows for ridge, GBM and bins (default 200000)\n\
  --draws <n>               bootstrap and SPA draws (default 2000)\n\
  --block-len <n>           block length in periods (default max(horizon, cbrt n))\n\
  --fdr-q <q>               BH-FDR level (default 0.10)\n\
  --pbo-blocks <even n>     CSCV blocks (default 8)\n\
  --seed <n>                seed of every draw (default 0)\n\
  --lockbox <START,END>     locked test to guard (default: the declaration stored with --data);\n\
                            a fold touching it is refused (exit 3)\n\
  --locked-test             read the locked test: needs --scheme locked, --ledger and --sealed;\n\
                            appends a baseline_read to the ledger, a second read is refused\n\
  --ledger <file>           candidate ledger (with --locked-test)\n\
  --sealed <file>           sealed candidate list recorded in the ledger (with --locked-test)\n\
  --out <file>              write the JSON report (ends with a ledger object: io and\n\
                            pnl seconds on the run wall)\n\
  --markdown <file|->       write the Markdown table\n\
  --json                    print the JSON report\n"
}

fn parse_list(s: &str, what: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|v| v.trim().parse::<f64>().map_err(|_| format!("{}: cannot parse '{}'", what, v)))
        .collect()
}

pub fn run(args: &[String]) -> ! {
    match run_inner(args) {
        Ok(()) => std::process::exit(0),
        Err((code, msg)) => {
            eprintln!("error: {}", msg);
            std::process::exit(code)
        }
    }
}

fn run_inner(args: &[String]) -> Result<(), (i32, String)> {
    let t_run = Instant::now();
    let usage_err = |m: String| (2, m);
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return Ok(());
    }
    let mut opts = Options::default();
    let mut data_path = None;
    let mut target = None;
    let mut ret = None;
    let mut time = None;
    let mut group = None;
    let mut features: Vec<String> = Vec::new();
    let mut windows = None;
    let mut scheme = "expanding".to_string();
    let mut lockbox_arg: Option<String> = None;
    let mut locked_test = false;
    let mut ledger_path: Option<String> = None;
    let mut sealed_path: Option<String> = None;
    let mut spec_path: Option<String> = None;
    let mut out = None;
    let mut markdown = None;
    let mut print_json = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || -> Result<String, (i32, String)> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| usage_err(format!("{} requires a value", a)))
        };
        let num_err = |a: &str| usage_err(format!("{}: invalid number", a));
        match a {
            "--data" => data_path = Some(val()?),
            "--target" => target = Some(val()?),
            "--return-col" => ret = Some(val()?),
            "--time" => time = Some(val()?),
            "--group" => group = Some(val()?),
            "--features" => features = val()?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            "--windows" => windows = Some(val()?),
            "--scheme" => scheme = val()?,
            "--purge-seconds" => opts.purge_seconds = val()?.parse().map_err(|_| num_err(a))?,
            "--horizon-seconds" => opts.horizon_seconds = val()?.parse().map_err(|_| num_err(a))?,
            "--cost-bps" => opts.cost_bps = val()?.parse().map_err(|_| num_err(a))?,
            "--cost-grid" => opts.cost_grid = parse_list(&val()?, a).map_err(usage_err)?,
            "--quantiles" => opts.quantiles = parse_list(&val()?, a).map_err(usage_err)?,
            "--vol-col" => opts.vol_col = Some(val()?),
            "--reversal-col" => opts.reversal_col = Some(val()?),
            "--signal-col" => opts.signal_col = Some(val()?),
            "--target-spec" => spec_path = Some(val()?),
            "--sr-score" => opts.sr_score_col = Some(val()?),
            "--sr-evals" => opts.sr_evals = Some(val()?.parse().map_err(|_| num_err(a))?),
            "--sr-wall" => opts.sr_wall_seconds = Some(val()?.parse().map_err(|_| num_err(a))?),
            "--gbm-trees" => opts.gbm_trees = val()?.parse().map_err(|_| num_err(a))?,
            "--max-fit-rows" => opts.max_fit_rows = val()?.parse().map_err(|_| num_err(a))?,
            "--draws" => opts.draws = val()?.parse().map_err(|_| num_err(a))?,
            "--block-len" => opts.block_len = Some(val()?.parse().map_err(|_| num_err(a))?),
            "--fdr-q" => opts.fdr_q = val()?.parse().map_err(|_| num_err(a))?,
            "--pbo-blocks" => opts.pbo_blocks = val()?.parse().map_err(|_| num_err(a))?,
            "--seed" => opts.seed = val()?.parse().map_err(|_| num_err(a))?,
            "--lockbox" => lockbox_arg = Some(val()?),
            "--locked-test" => locked_test = true,
            "--ledger" => ledger_path = Some(val()?),
            "--sealed" => sealed_path = Some(val()?),
            "--out" => out = Some(val()?),
            "--markdown" => markdown = Some(val()?),
            "--json" => print_json = true,
            other => return Err(usage_err(format!("unknown argument '{}'", other))),
        }
        i += 1;
    }
    let need = |v: Option<String>, n: &str| v.ok_or_else(|| usage_err(format!("{} is required", n)));
    let data_path = need(data_path, "--data")?;
    let target = need(target, "--target")?;
    let time = need(time, "--time")?;
    let windows = need(windows, "--windows")?;
    if opts.cost_grid.is_empty() {
        opts.cost_grid = vec![opts.cost_bps];
    }
    if opts.pbo_blocks % 2 != 0 || opts.pbo_blocks < 2 {
        return Err(usage_err("--pbo-blocks must be an even number of at least 2".to_string()));
    }
    let input_err = |m: String| (3, m);
    let t_io = Instant::now();
    if let Some(path) = &spec_path {
        let text = std::fs::read_to_string(path).map_err(|e| input_err(format!("{}: {}", path, e)))?;
        let specs = crate::target::parse_file(&text).map_err(|e| input_err(format!("{}: {}", path, e)))?;
        opts.spec = specs.into_iter().find(|s| s.column == target || s.name == target);
        if opts.spec.is_none() {
            return Err(input_err(format!("{}: no entry for target '{}'", path, target)));
        }
    }
    let text = std::fs::read_to_string(&windows).map_err(|e| input_err(format!("{}: {}", windows, e)))?;
    let rows = protocol::parse_windows_csv(&text, Some(&scheme)).map_err(input_err)?;
    let declared = crate::lockbox::resolve_declared(lockbox_arg.as_deref(), &data_path).map_err(|e| usage_err(format!("--lockbox: {}", e)))?;
    if !locked_test && (ledger_path.is_some() || sealed_path.is_some()) {
        return Err(usage_err("--ledger and --sealed apply to --locked-test only".to_string()));
    }
    let mut read: Option<(protocol::Ledger, String)> = None;
    if locked_test {
        let (Some(ledger), Some(sealed)) = (&ledger_path, &sealed_path) else {
            return Err(usage_err("--locked-test needs --ledger and --sealed".to_string()));
        };
        if scheme != "locked" {
            return Err(usage_err("--locked-test reads the locked scheme only: use --scheme locked".to_string()));
        }
        let Some(d) = &declared else {
            return Err(usage_err("--locked-test needs a declared locked test: --lockbox or a store declaration".to_string()));
        };
        let (ls, le) = (d.lockbox.start as f64, d.lockbox.end as f64);
        for r in &rows {
            let w = &r.window;
            if w.train_start < le && w.train_end > ls {
                return Err(input_err(format!("window '{}' trains on the locked test", w.label)));
            }
            if !(w.test_start >= ls && w.test_end <= le) {
                return Err(input_err(format!("window '{}' tests outside the locked test", w.label)));
            }
        }
        let l = protocol::Ledger::new(ledger);
        l.authorize_baseline_read(std::path::Path::new(sealed)).map_err(input_err)?;
        read = Some((l, crate::manifest::sha256_file(&data_path).map_err(input_err)?));
    }
    let guard = if locked_test {
        None
    } else {
        declared.as_ref().map(|d| (d.lockbox.start as f64, d.lockbox.end as f64))
    };
    let inp = Input {
        data: data_path,
        target,
        ret,
        time,
        group,
        features,
    };
    let panel = load_panel(&inp, &opts).map_err(input_err)?;
    let folds = make_folds(&panel, &rows, opts.purge_seconds, guard).map_err(input_err)?;
    let io_s = t_io.elapsed().as_secs_f64();
    let mut report = run_ladder(&panel, &folds, &opts).map_err(input_err)?;
    let mut ledger = RunLedger::new();
    ledger.record_stage(Stage::Io, io_s);
    ledger.record_stage(Stage::Pnl, report.ledger.stage_s(Stage::Pnl));
    let wall = t_run.elapsed().as_secs_f64();
    ledger.record_stage(Stage::HostOther, wall - ledger.stage_sum_s());
    ledger.set_wall(wall);
    report.ledger = ledger;
    let mut json = report.json();
    if let Some(d) = &declared {
        let in_scope = if locked_test { folds.iter().map(|f| f.test.len()).sum() } else { 0 };
        json = d.splice_report(&json, in_scope, locked_test, true);
    }
    if let Some((ledger, data_sha)) = &read {
        let report_sha = crate::manifest::sha256_hex(json.as_bytes());
        ledger.record_baseline_read(data_sha, &report_sha).map_err(input_err)?;
    }
    if let Some(p) = &out {
        std::fs::write(p, &json).map_err(|e| input_err(format!("{}: {}", p, e)))?;
    }
    if let Some(p) = &markdown {
        let md = report.markdown();
        if p == "-" {
            print!("{}", md);
        } else {
            std::fs::write(p, md).map_err(|e| input_err(format!("{}: {}", p, e)))?;
        }
    }
    if print_json || (out.is_none() && markdown.is_none()) {
        println!("{}", json);
    }
    Ok(())
}

/// Equal-weight period net returns of `pos` on `rows` (one block, in
/// `(group, time)` order): the block starts flat and pays its final exit.
pub fn period_net(p: &Panel, rows: &[usize], pos: &[f64], side: f64) -> Vec<f64> {
    let r: Vec<f64> = rows.iter().map(|&i| p.ret[i]).collect();
    let g: Vec<u64> = rows.iter().map(|&i| p.group[i]).collect();
    let net = baselines::net_rows(pos, &r, &g, side);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| p.time[rows[a]].total_cmp(&p.time[rows[b]]).then_with(|| a.cmp(&b)));
    let mut out: Vec<f64> = Vec::new();
    let (mut sum, mut cnt, mut last) = (0.0, 0usize, f64::NAN);
    for &k in &order {
        let t = p.time[rows[k]];
        if cnt > 0 && t != last {
            out.push(sum / cnt as f64);
            sum = 0.0;
            cnt = 0;
        }
        sum += net[k];
        cnt += 1;
        last = t;
    }
    if cnt > 0 {
        out.push(sum / cnt as f64);
    }
    out
}

/// Mean squared error in return space of predictions on `rows`.
pub fn return_space_mse_rows(p: &Panel, rows: &[usize], score: &[f64], opts: &Options) -> f64 {
    let scale = opts
        .spec
        .as_ref()
        .and_then(|s| s.scale_column().map(|c| (s, c)))
        .and_then(|(s, c)| p.col(c).map(|col| (s, col)));
    let (mut s, mut c) = (0.0, 0.0);
    for (k, &i) in rows.iter().enumerate() {
        let pred = match &scale {
            Some((spec, g)) => spec.return_from_target(score[k], g[i]),
            None => score[k],
        };
        if pred.is_finite() && p.ret[i].is_finite() {
            s += (pred - p.ret[i]).powi(2);
            c += 1.0;
        }
    }
    if c > 0.0 {
        s / c
    } else {
        f64::NAN
    }
}
