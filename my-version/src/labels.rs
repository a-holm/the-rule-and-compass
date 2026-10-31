//! Long-only horizon, action and triple-barrier labels (exact CPU f64).
//!
//! For a row at bar `t`, horizon `h` bars and delay `d` bars:
//!
//! ```text
//! P0        = open[t+1+d]                         entry
//! fwd       = ln(close[t+h+d] / P0)               long gross log return
//! sigma_H   = max(s_W, floor) * sqrt(h)           s_W: population sd of the
//!                                                 last W one-bar log returns
//! y         = clip(fwd / sigma_H, -clip, +clip)   gross regression target
//! long_net  = fwd - c                             c: long round trip + funding
//! action    = 1 if fwd > c else 0                 {0, +1}, never -1
//! ```
//!
//! The triple barrier is long side only: profit `P0 * exp(+u * sigma_H)`, stop
//! `P0 * exp(-d * sigma_H)` (exit to flat) and the vertical barrier at bar
//! `t+h+d`. A bar that touches both barriers exits at the stop. Barrier levels
//! are the fill prices. `t1` is the end of the exit bar and never exceeds the
//! end of the vertical bar.
//!
//! A row is `NaN` when any bar from `t` to `t+h+d` is missing, partial or has a
//! non-positive price. Nothing bridges a gap and nothing reads after `t+h+d`.
//! There is no short label, barrier or borrow/funding term.

use crate::devlog::{num, text};
use crate::manifest::sha256_hex;
use crate::ohlcv::Bar;
use crate::target::{Scale, TargetSpec};

/// Version of the label definitions; part of the label hash.
pub const VERSION: &str = "rcsr-long-labels/1";

const BP: f64 = 1e-4;

/// Horizons from this length on need `--allow-overlap` to run at a smaller stride.
pub const OVERLAP_GUARD_SECONDS: i64 = 4 * 3600;

/// Exit kind of the barrier label.
pub const EXIT_NONE: u8 = 0;
pub const EXIT_PROFIT: u8 = 1;
pub const EXIT_STOP: u8 = 2;
pub const EXIT_VERTICAL: u8 = 3;

/// Long round-trip cost components in basis points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub fee_bps: f64,
    pub spread_bps: f64,
    pub slippage_bps: f64,
}

impl CostModel {
    /// `c_rt = 2 * fee + spread + slippage`, as a log-return fraction.
    pub fn round_trip(&self) -> f64 {
        (2.0 * self.fee_bps + self.spread_bps + self.slippage_bps) * BP
    }

    /// The same in basis points.
    pub fn round_trip_bps(&self) -> f64 {
        2.0 * self.fee_bps + self.spread_bps + self.slippage_bps
    }
}

/// Funding paid by a held position. Only a long position pays; the default cost
/// model has no short borrow or funding term.
pub fn funding_charge(position: i8, accrued: f64) -> f64 {
    if position == 1 {
        accrued
    } else {
        0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongLabelSpec {
    pub name: String,
    pub bar_seconds: i64,
    /// Bars from the entry bar to the exit bar, inclusive of both.
    pub horizon_bars: usize,
    /// Extra bars between the decision and the entry.
    pub delay_bars: usize,
    /// Volatility window `W` in bars.
    pub sigma_window: usize,
    /// Floor of `s_W`, basis points.
    pub sigma_floor_bps: f64,
    pub clip: f64,
    pub cost: CostModel,
    /// Profit barrier multiple `u` of `sigma_H`.
    pub barrier_up: f64,
    /// Stop barrier multiple `d` of `sigma_H`.
    pub barrier_down: f64,
    /// Keep every `stride`-th row (by row index in the series).
    pub stride: usize,
}

impl LongLabelSpec {
    pub fn new(name: &str, bar_seconds: i64, horizon_bars: usize) -> Self {
        LongLabelSpec {
            name: name.to_string(),
            bar_seconds,
            horizon_bars,
            delay_bars: 0,
            sigma_window: 96,
            sigma_floor_bps: 5.0,
            clip: 3.0,
            cost: CostModel { fee_bps: 0.0, spread_bps: 0.0, slippage_bps: 0.0 },
            barrier_up: 1.0,
            barrier_down: 1.0,
            stride: 1,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.bar_seconds <= 0 {
            return Err("label: bar_seconds must be positive".to_string());
        }
        if self.horizon_bars == 0 {
            return Err("label: horizon must be at least one bar".to_string());
        }
        if self.sigma_window < 2 {
            return Err("label: sigma window must be at least 2 bars".to_string());
        }
        if self.stride == 0 {
            return Err("label: stride must be >= 1".to_string());
        }
        for (n, v) in [
            ("fee", self.cost.fee_bps),
            ("spread", self.cost.spread_bps),
            ("slippage", self.cost.slippage_bps),
            ("sigma floor", self.sigma_floor_bps),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("label: {} must be finite and non-negative", n));
            }
        }
        for (n, v) in [("clip", self.clip), ("profit multiple", self.barrier_up), ("stop multiple", self.barrier_down)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("label: {} must be finite and positive", n));
            }
        }
        Ok(())
    }

    pub fn entry_offset_seconds(&self) -> i64 {
        self.bar_seconds
    }

    pub fn delay_seconds(&self) -> i64 {
        self.delay_bars as i64 * self.bar_seconds
    }

    pub fn horizon_seconds(&self) -> i64 {
        self.horizon_bars as i64 * self.bar_seconds
    }

    /// `entry_offset + horizon + delay`. Derived from the spec, never from a name.
    pub fn purge_seconds(&self) -> i64 {
        self.entry_offset_seconds() + self.horizon_seconds() + self.delay_seconds()
    }

    /// A stride below the horizon overlaps labels. From four hours on it needs
    /// an explicit opt-in; shorter horizons are covered by the uniqueness weights.
    pub fn check_overlap(&self, allow_overlap: bool) -> Result<(), String> {
        if self.stride < self.horizon_bars
            && self.horizon_seconds() >= OVERLAP_GUARD_SECONDS
            && !allow_overlap
        {
            return Err(format!(
                "label {}: stride {} bar(s) is below the {}-bar horizon, so labels overlap; \
                 use stride >= {} or pass --allow-overlap (uniqueness weights are stored either way)",
                self.name, self.stride, self.horizon_bars, self.horizon_bars
            ));
        }
        Ok(())
    }

    fn sigma_column(&self) -> String {
        format!("vol{}", self.sigma_window)
    }

    /// The `rcsr-target-spec/1` entry of the gross target `column`. Its window
    /// is `[t + entry + delay, t + entry + delay + horizon]`, so the split code
    /// purges `purge_seconds()` around every validation and test row.
    pub fn target_spec(&self, column: &str, return_column: &str) -> TargetSpec {
        TargetSpec {
            column: column.to_string(),
            name: self.name.clone(),
            kind: "forward_return".to_string(),
            deprecated: false,
            return_column: Some(return_column.to_string()),
            return_units: "log".to_string(),
            horizon_seconds: self.horizon_seconds() as f64,
            entry_offset_seconds: self.entry_offset_seconds() as f64,
            delay_seconds: self.delay_seconds() as f64,
            entry_price: "open[t+1]".to_string(),
            exit_price: "close[t+h]".to_string(),
            offset: 0.0,
            scale: Scale::Column {
                column: self.sigma_column(),
                multiplier: (self.horizon_bars as f64).sqrt(),
                window_seconds: self.sigma_window as f64 * self.bar_seconds as f64,
                ends_at_offset_seconds: 0.0,
            },
            clip: Some((-self.clip, self.clip)),
            units: format!("log_return/(sqrt({})*{})", self.horizon_bars, self.sigma_column()),
            cost_bps: Some(self.cost.round_trip_bps()),
            cost_in_label: false,
            construction_columns: vec![return_column.to_string(), self.sigma_column()],
            long_only: true,
            note: None,
        }
    }

    fn canonical(&self) -> String {
        format!(
            "{{\"version\":{},\"name\":{},\"bar_seconds\":{},\"horizon_bars\":{},\"delay_bars\":{},\"sigma_window\":{},\"sigma_floor_bps\":{},\"clip\":{},\"cost_bps\":{},\"barrier_up\":{},\"barrier_down\":{},\"stride\":{},\"long_only\":true}}",
            text(VERSION),
            text(&self.name),
            self.bar_seconds,
            self.horizon_bars,
            self.delay_bars,
            self.sigma_window,
            num(self.sigma_floor_bps),
            num(self.clip),
            num(self.cost.round_trip_bps()),
            num(self.barrier_up),
            num(self.barrier_down),
            self.stride
        )
    }
}

/// Per-row label columns, aligned with the input bars.
#[derive(Clone, Debug)]
pub struct LongLabels {
    pub spec: LongLabelSpec,
    /// `s_W * sqrt(h)` after the floor; the decision-time scale.
    pub sigma_h: Vec<f64>,
    pub fwd: Vec<f64>,
    pub y: Vec<f64>,
    pub long_net: Vec<f64>,
    /// `{0, +1}` or `NaN`.
    pub action: Vec<f64>,
    /// Barrier exit return, gross.
    pub exit_ret: Vec<f64>,
    /// `1[exit_ret > cost]`.
    pub m_net: Vec<f64>,
    /// `1[profit barrier first]`.
    pub m_barrier: Vec<f64>,
    /// End of the exit bar, epoch seconds; the vertical end where no barrier
    /// level is available. Set on every row with a forward return.
    pub t1: Vec<f64>,
    pub exit_kind: Vec<u8>,
    /// Uniqueness weights, mean 1 over the valid rows.
    pub weight: Vec<f64>,
    /// Average uniqueness of each valid row.
    pub uniqueness: Vec<f64>,
    cost_total: Vec<f64>,
    end_idx: Vec<usize>,
}

fn contiguous_window(bars: &[Bar], from: usize, to: usize, step: i64) -> bool {
    (from..=to).all(|k| {
        let b = &bars[k];
        b.complete && b.time == bars[from].time + (k - from) as i64 * step
    })
}

fn priced(b: &Bar) -> bool {
    [b.open, b.high, b.low, b.close].iter().all(|v| v.is_finite() && *v > 0.0)
}

/// Population sd of the last `w` one-bar log returns ending at bar `t`.
fn sigma_at(bars: &[Bar], t: usize, w: usize, step: i64) -> Option<f64> {
    if t < w || !contiguous_window(bars, t - w, t, step) {
        return None;
    }
    let mut r = Vec::with_capacity(w);
    for k in t + 1 - w..=t {
        if !(priced(&bars[k]) && priced(&bars[k - 1])) {
            return None;
        }
        r.push((bars[k].close / bars[k - 1].close).ln());
    }
    let m = r.iter().sum::<f64>() / w as f64;
    Some((r.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / w as f64).sqrt())
}

/// Build the long-only label columns. `funding` is the per-bar funding accrual
/// of a long perp position in log-return units; pass `None` without a verified
/// funding column.
pub fn build(bars: &[Bar], spec: &LongLabelSpec, funding: Option<&[f64]>) -> Result<LongLabels, String> {
    spec.validate()?;
    if let Some(f) = funding {
        if f.len() != bars.len() {
            return Err("label: funding column length differs from the bars".to_string());
        }
    }
    let n = bars.len();
    let step = spec.bar_seconds;
    let (h, d) = (spec.horizon_bars, spec.delay_bars);
    let nan = vec![f64::NAN; n];
    let mut out = LongLabels {
        spec: spec.clone(),
        sigma_h: nan.clone(),
        fwd: nan.clone(),
        y: nan.clone(),
        long_net: nan.clone(),
        action: nan.clone(),
        exit_ret: nan.clone(),
        m_net: nan.clone(),
        m_barrier: nan.clone(),
        t1: nan.clone(),
        exit_kind: vec![EXIT_NONE; n],
        weight: nan.clone(),
        uniqueness: nan.clone(),
        cost_total: nan,
        end_idx: vec![0; n],
    };
    let c_rt = spec.cost.round_trip();
    let floor = spec.sigma_floor_bps * BP;
    let root_k = (h as f64).sqrt();
    for t in 0..n {
        let (e, last) = (t + 1 + d, t + h + d);
        if last >= n || t % spec.stride != 0 {
            continue;
        }
        if !contiguous_window(bars, t, last, step) || !(e..=last).all(|k| priced(&bars[k])) {
            continue;
        }
        let mut cost = c_rt;
        if let Some(f) = funding {
            let accrued: f64 = f[e..=last].iter().sum();
            if !accrued.is_finite() {
                continue;
            }
            cost += funding_charge(1, accrued);
        }
        let p0 = bars[e].open;
        let fwd = (bars[last].close / p0).ln();
        out.fwd[t] = fwd;
        out.long_net[t] = fwd - cost;
        out.action[t] = if fwd > cost { 1.0 } else { 0.0 };
        out.cost_total[t] = cost;
        out.t1[t] = (bars[last].time + step) as f64;
        out.end_idx[t] = last;
        let Some(s) = sigma_at(bars, t, spec.sigma_window, step) else {
            continue;
        };
        let sigma_h = s.max(floor) * root_k;
        out.sigma_h[t] = sigma_h;
        out.y[t] = (fwd / sigma_h).max(-spec.clip).min(spec.clip);
        let profit = p0 * (spec.barrier_up * sigma_h).exp();
        let stop = p0 * (-spec.barrier_down * sigma_h).exp();
        let mut exit = (last, EXIT_VERTICAL, bars[last].close);
        for j in e..=last {
            if bars[j].low <= stop {
                exit = (j, EXIT_STOP, stop);
                break;
            }
            if bars[j].high >= profit {
                exit = (j, EXIT_PROFIT, profit);
                break;
            }
        }
        out.end_idx[t] = exit.0;
        out.exit_kind[t] = exit.1;
        out.t1[t] = (bars[exit.0].time + step) as f64;
        out.exit_ret[t] = (exit.2 / p0).ln();
        out.m_net[t] = if out.exit_ret[t] > cost { 1.0 } else { 0.0 };
        out.m_barrier[t] = if exit.1 == EXIT_PROFIT { 1.0 } else { 0.0 };
    }
    (out.uniqueness, out.weight) = uniqueness_weights(&out, d);
    Ok(out)
}

/// `u_i = mean over the label's bars of 1 / c_t`, `w_i = u_i * n / sum(u)`,
/// with `c_t` the number of labels whose span covers bar `t`. A span runs from
/// the entry bar to the exit bar.
fn uniqueness_weights(l: &LongLabels, d: usize) -> (Vec<f64>, Vec<f64>) {
    let n = l.fwd.len();
    let rows: Vec<usize> = (0..n).filter(|&t| l.fwd[t].is_finite()).collect();
    let mut w = vec![f64::NAN; n];
    if rows.is_empty() {
        return (w.clone(), w);
    }
    let mut conc = vec![0i64; n + 1];
    for &t in &rows {
        conc[t + 1 + d] += 1;
        conc[l.end_idx[t] + 1] -= 1;
    }
    let mut prefix = vec![0.0; n + 1];
    let mut c = 0i64;
    for k in 0..n {
        c += conc[k];
        prefix[k + 1] = prefix[k] + if c > 0 { 1.0 / c as f64 } else { 0.0 };
    }
    let mut u = vec![f64::NAN; n];
    let mut total = 0.0;
    for &t in &rows {
        let (a, b) = (t + 1 + d, l.end_idx[t]);
        u[t] = (prefix[b + 1] - prefix[a]) / (b - a + 1) as f64;
        total += u[t];
    }
    for &t in &rows {
        w[t] = u[t] * rows.len() as f64 / total;
    }
    (u, w)
}

/// Meta-label: `1[long exit net > 0]` on rows where the primary rule is long,
/// `NaN` elsewhere. `primary` holds out-of-fold `{0, +1}` decisions; the caller
/// owns that out-of-fold guarantee.
pub fn meta_labels(primary: &[f64], labels: &LongLabels) -> Result<Vec<f64>, String> {
    if primary.len() != labels.fwd.len() {
        return Err("meta label: primary length differs from the labels".to_string());
    }
    Ok((0..primary.len())
        .map(|t| {
            if primary[t] == 1.0 && labels.exit_ret[t].is_finite() {
                if labels.exit_ret[t] - labels.cost_total[t] > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                f64::NAN
            }
        })
        .collect())
}

/// Label production report.
#[derive(Clone, Debug)]
pub struct LabelReport {
    pub rows: usize,
    pub valid_rows: usize,
    pub barrier_rows: usize,
    pub stride_rows: usize,
    pub profit_exits: usize,
    pub stop_exits: usize,
    pub vertical_exits: usize,
    pub action_long_share: f64,
    pub clipped_share: f64,
    pub sign_pos_share: f64,
    pub sign_neg_share: f64,
    pub sign_zero_share: f64,
    pub fwd_mean: f64,
    /// Quantiles of `fwd` at 1, 5, 25, 50, 75, 95 and 99 percent.
    pub fwd_quantiles: [f64; 7],
    /// Round-trip cost over the mean absolute forward move.
    pub cost_to_move: f64,
    pub average_uniqueness: f64,
    pub effective_n: f64,
    pub effective_n_hours: f64,
    pub purge_seconds: i64,
    pub label_hash: String,
}

const QUANTILES: [f64; 7] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn share(count: usize, of: usize) -> f64 {
    if of == 0 {
        f64::NAN
    } else {
        count as f64 / of as f64
    }
}

impl LongLabels {
    /// Hash of the definition version, the spec and every column value.
    pub fn label_hash(&self) -> String {
        let mut bytes = self.spec.canonical().into_bytes();
        for col in [
            &self.fwd, &self.y, &self.long_net, &self.action, &self.exit_ret, &self.m_net,
            &self.m_barrier, &self.t1, &self.weight,
        ] {
            for v in col.iter() {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        bytes.extend_from_slice(&self.exit_kind);
        sha256_hex(&bytes)
    }

    pub fn report(&self) -> LabelReport {
        let rows = self.fwd.len();
        let valid: Vec<usize> = (0..rows).filter(|&t| self.fwd[t].is_finite()).collect();
        let nv = valid.len();
        let with_y: Vec<usize> = valid.iter().copied().filter(|&t| self.y[t].is_finite()).collect();
        let kind = |k: u8| valid.iter().filter(|&&t| self.exit_kind[t] == k).count();
        let mut fwd: Vec<f64> = valid.iter().map(|&t| self.fwd[t]).collect();
        fwd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_abs = if nv == 0 { f64::NAN } else { fwd.iter().map(|v| v.abs()).sum::<f64>() / nv as f64 };
        let mut qs = [f64::NAN; 7];
        for (slot, q) in qs.iter_mut().zip(QUANTILES) {
            *slot = quantile(&fwd, q);
        }
        let u_sum: f64 = if nv == 0 { 0.0 } else { valid.iter().map(|&t| self.uniqueness[t]).sum() };
        let hours = self.spec.horizon_seconds() as f64 / 3600.0;
        LabelReport {
            rows,
            valid_rows: nv,
            barrier_rows: with_y.len(),
            stride_rows: (0..rows).filter(|t| t % self.spec.stride != 0).count(),
            profit_exits: kind(EXIT_PROFIT),
            stop_exits: kind(EXIT_STOP),
            vertical_exits: kind(EXIT_VERTICAL),
            action_long_share: share(valid.iter().filter(|&&t| self.action[t] == 1.0).count(), nv),
            clipped_share: share(
                with_y.iter().filter(|&&t| self.y[t].abs() >= self.spec.clip).count(),
                with_y.len(),
            ),
            sign_pos_share: share(valid.iter().filter(|&&t| self.fwd[t] > 0.0).count(), nv),
            sign_neg_share: share(valid.iter().filter(|&&t| self.fwd[t] < 0.0).count(), nv),
            sign_zero_share: share(valid.iter().filter(|&&t| self.fwd[t] == 0.0).count(), nv),
            fwd_mean: if nv == 0 { f64::NAN } else { fwd.iter().sum::<f64>() / nv as f64 },
            fwd_quantiles: qs,
            cost_to_move: self.spec.cost.round_trip() / mean_abs,
            average_uniqueness: if nv == 0 { f64::NAN } else { u_sum / nv as f64 },
            effective_n: u_sum,
            effective_n_hours: u_sum * hours,
            purge_seconds: self.spec.purge_seconds(),
            label_hash: self.label_hash(),
        }
    }
}

impl LabelReport {
    pub fn json(&self) -> String {
        let q = self
            .fwd_quantiles
            .iter()
            .zip(QUANTILES)
            .map(|(v, p)| format!("\"p{}\":{}", num(p * 100.0), num(*v)))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"version\":{},\"long_only\":true,\"rows\":{},\"valid_rows\":{},\"barrier_rows\":{},\"stride_rows\":{},\"exits\":{{\"profit\":{},\"stop\":{},\"vertical\":{}}},\"action_long_share\":{},\"clipped_share\":{},\"sign_share\":{{\"positive\":{},\"negative\":{},\"zero\":{}}},\"fwd_mean\":{},\"fwd_quantiles\":{{{}}},\"cost_to_move\":{},\"average_uniqueness\":{},\"effective_n\":{},\"effective_n_hours\":{},\"purge_seconds\":{},\"label_hash\":{}}}",
            text(VERSION),
            self.rows,
            self.valid_rows,
            self.barrier_rows,
            self.stride_rows,
            self.profit_exits,
            self.stop_exits,
            self.vertical_exits,
            num(self.action_long_share),
            num(self.clipped_share),
            num(self.sign_pos_share),
            num(self.sign_neg_share),
            num(self.sign_zero_share),
            num(self.fwd_mean),
            q,
            num(self.cost_to_move),
            num(self.average_uniqueness),
            num(self.effective_n),
            num(self.effective_n_hours),
            self.purge_seconds,
            text(&self.label_hash)
        )
    }
}

/// The only side the label builder supports.
pub fn check_side(side: &str) -> Result<(), String> {
    match side {
        "long" => Ok(()),
        "short" => Err(
            "--side short is refused: labels are long-only (action {0,+1}, long-side barriers); \
             a short arm is not part of this engine"
                .to_string(),
        ),
        other => Err(format!("unknown --side '{}' (expected long)", other)),
    }
}

/// `--allow-short` is refused with the same message as `--side short`.
pub fn refuse_allow_short() -> String {
    check_side("short").unwrap_err().replace("--side short", "--allow-short")
}
