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
use crate::lockbox::Lockbox;
use crate::manifest::sha256_hex;
use crate::ohlcv::{Bar, Timeframe};
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
    /// Row stamps (epoch seconds) of the input bars.
    pub time: Vec<i64>,
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
        time: bars.iter().map(|b| b.time).collect(),
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

/// Number of largest absolute forward moves listed in a label report.
pub const TOP_MOVES: usize = 10;

/// Store column prefix of the long-only label family.
pub const COLUMN_PREFIX: &str = "lbl_";

/// Label column kinds in store order. The flag is true where the column needs
/// the volatility window `s_W` and is `NaN` during its warm-up.
pub const COLUMN_KINDS: [(&str, bool); 9] = [
    ("fwd", false),
    ("y", true),
    ("long_net", false),
    ("action", false),
    ("t1", false),
    ("exit_ret", true),
    ("m_net", true),
    ("m_barrier", true),
    ("weight", false),
];

/// `lbl_<label>_<kind>`.
pub fn column_name(label: &str, kind: &str) -> String {
    format!("{}{}_{}", COLUMN_PREFIX, label, kind)
}

/// Split a stored label column into its label name and kind.
pub fn parse_column(name: &str) -> Option<(&str, &'static str)> {
    let rest = name.strip_prefix(COLUMN_PREFIX)?;
    for (kind, _) in COLUMN_KINDS {
        if let Some(label) = rest.strip_suffix(kind).and_then(|l| l.strip_suffix('_')) {
            if !label.is_empty() {
                return Some((label, kind));
            }
        }
    }
    None
}

/// `lbl_<label>_xs`: the cross-sectional rank column written by `rcsr panel rank`.
pub fn is_xs_column(name: &str) -> bool {
    name.strip_prefix(COLUMN_PREFIX)
        .and_then(|rest| rest.strip_suffix("_xs"))
        .is_some_and(|label| !label.is_empty())
}

/// The refusal for a `long:` label without `fee=`.
pub fn missing_fee_message(src: &str) -> String {
    format!(
        "label '{}': 'fee=' is required: state the per-side fee in basis points (fee=0 for none)",
        src
    )
}

impl LongLabelSpec {
    /// Parse `long:<horizon>[,name=ID][,delay=BARS][,fee=BP][,spread=BP]
    /// [,slip=BP][,vol=BARS][,floor=BP][,clip=C][,up=U][,down=D][,stride=N]`.
    /// The horizon is a bar count or a timeframe name resolved against
    /// `bar_seconds`. The default name is `h<bars>`.
    pub fn parse(src: &str, bar_seconds: i64) -> Result<Self, String> {
        let bad = |part: &str| format!("label '{}': bad option '{}'", src, part);
        let mut spec = LongLabelSpec::new("", bar_seconds, 0);
        let mut fee_set = false;
        let mut named = false;
        for part in src.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if let Some(h) = part.strip_prefix("long:") {
                spec.horizon_bars = match h.trim().parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => {
                        let secs = Timeframe::parse(h.trim())?.seconds();
                        if bar_seconds <= 0 || secs < bar_seconds || secs % bar_seconds != 0 {
                            return Err(format!(
                                "label '{}': horizon {} is not a whole number of {}s bars",
                                src, h.trim(), bar_seconds
                            ));
                        }
                        (secs / bar_seconds) as usize
                    }
                };
                continue;
            }
            let (key, value) = part.split_once('=').ok_or_else(|| bad(part))?;
            let value = value.trim();
            let float = || value.parse::<f64>().map_err(|_| bad(part));
            let count = || value.parse::<usize>().map_err(|_| bad(part));
            match key.trim() {
                "name" => {
                    if value.is_empty()
                        || !value.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
                    {
                        return Err(format!("label '{}': name must be lower-case ASCII letters, digits and '_'", src));
                    }
                    spec.name = value.to_string();
                    named = true;
                }
                "delay" => spec.delay_bars = count()?,
                "fee" => {
                    spec.cost.fee_bps = float()?;
                    fee_set = true;
                }
                "spread" => spec.cost.spread_bps = float()?,
                "slip" => spec.cost.slippage_bps = float()?,
                "vol" => spec.sigma_window = count()?,
                "floor" => spec.sigma_floor_bps = float()?,
                "clip" => spec.clip = float()?,
                "up" => spec.barrier_up = float()?,
                "down" => spec.barrier_down = float()?,
                "stride" => spec.stride = count()?,
                "side" => check_side(value)?,
                _ => return Err(bad(part)),
            }
        }
        if spec.horizon_bars == 0 {
            return Err(format!("label '{}': needs 'long:<horizon>' with at least one bar", src));
        }
        if !fee_set {
            return Err(missing_fee_message(src));
        }
        if !named {
            spec.name = format!("h{}", spec.horizon_bars);
        }
        spec.validate()?;
        Ok(spec)
    }

    /// Whether a `long:` label spec states `fee=`.
    pub fn declares_fee(src: &str) -> bool {
        src.split(',').any(|p| p.trim().split_once('=').is_some_and(|(k, _)| k.trim() == "fee"))
    }

    /// Volatility window of a `long:` label spec, read without resolving the horizon.
    pub fn declared_sigma_window(src: &str) -> usize {
        src.split(',')
            .find_map(|p| p.trim().strip_prefix("vol=")?.trim().parse().ok())
            .unwrap_or(LongLabelSpec::new("", 1, 1).sigma_window)
    }

    /// Stored column of `kind` for this label.
    pub fn column(&self, kind: &str) -> String {
        column_name(&self.name, kind)
    }

    /// Leading rows of `kind` that are `NaN` because `s_W` is not available.
    pub fn column_warmup(&self, kind: &str) -> usize {
        match COLUMN_KINDS.iter().find(|(k, _)| *k == kind) {
            Some((_, true)) => self.sigma_window,
            _ => 0,
        }
    }
}

/// One of the largest absolute forward moves.
#[derive(Clone, Debug, PartialEq)]
pub struct TopMove {
    pub symbol: Option<String>,
    /// Row stamp (the decision bar), epoch seconds.
    pub time: i64,
    pub fwd: f64,
}

/// Label production report.
#[derive(Clone, Debug)]
pub struct LabelReport {
    pub name: String,
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
    /// The `TOP_MOVES` largest `|fwd|` rows, largest first.
    pub top_moves: Vec<TopMove>,
    pub average_uniqueness: f64,
    pub effective_n: f64,
    pub effective_n_hours: f64,
    pub purge_seconds: i64,
    /// The declared locked test the statistics leave out, with the number of
    /// valid rows left out. `None` when no range applies.
    pub locked_test: Option<LockedLeftOut>,
    pub label_hash: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LockedLeftOut {
    pub range: Lockbox,
    pub rows_left_out: usize,
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

/// Counts and values behind a report; mergeable across series.
#[derive(Clone, Debug, Default)]
struct Tally {
    rows: usize,
    stride_rows: usize,
    fwd: Vec<f64>,
    barrier_rows: usize,
    clipped: usize,
    exits: [usize; 3],
    action_long: usize,
    uniqueness_sum: f64,
    top: Vec<TopMove>,
    left_out: usize,
}

fn move_order(a: &TopMove, b: &TopMove) -> std::cmp::Ordering {
    b.fwd
        .abs()
        .partial_cmp(&a.fwd.abs())
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.time.cmp(&b.time))
        .then(a.symbol.cmp(&b.symbol))
}

impl Tally {
    fn merge(&mut self, other: Tally) {
        self.rows += other.rows;
        self.stride_rows += other.stride_rows;
        self.fwd.extend(other.fwd);
        self.barrier_rows += other.barrier_rows;
        self.clipped += other.clipped;
        for k in 0..3 {
            self.exits[k] += other.exits[k];
        }
        self.action_long += other.action_long;
        self.uniqueness_sum += other.uniqueness_sum;
        self.left_out += other.left_out;
        self.top.extend(other.top);
        self.top.sort_by(move_order);
        self.top.truncate(TOP_MOVES);
    }

    fn into_report(mut self, spec: &LongLabelSpec, range: Option<Lockbox>, label_hash: String) -> LabelReport {
        let nv = self.fwd.len();
        self.fwd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_abs = if nv == 0 { f64::NAN } else { self.fwd.iter().map(|v| v.abs()).sum::<f64>() / nv as f64 };
        let mut qs = [f64::NAN; 7];
        for (slot, q) in qs.iter_mut().zip(QUANTILES) {
            *slot = quantile(&self.fwd, q);
        }
        let hours = spec.horizon_seconds() as f64 / 3600.0;
        LabelReport {
            name: spec.name.clone(),
            rows: self.rows,
            valid_rows: nv,
            barrier_rows: self.barrier_rows,
            stride_rows: self.stride_rows,
            profit_exits: self.exits[0],
            stop_exits: self.exits[1],
            vertical_exits: self.exits[2],
            action_long_share: share(self.action_long, nv),
            clipped_share: share(self.clipped, self.barrier_rows),
            sign_pos_share: share(self.fwd.iter().filter(|&&v| v > 0.0).count(), nv),
            sign_neg_share: share(self.fwd.iter().filter(|&&v| v < 0.0).count(), nv),
            sign_zero_share: share(self.fwd.iter().filter(|&&v| v == 0.0).count(), nv),
            fwd_mean: if nv == 0 { f64::NAN } else { self.fwd.iter().sum::<f64>() / nv as f64 },
            fwd_quantiles: qs,
            cost_to_move: spec.cost.round_trip() / mean_abs,
            top_moves: self.top,
            average_uniqueness: if nv == 0 { f64::NAN } else { self.uniqueness_sum / nv as f64 },
            effective_n: self.uniqueness_sum,
            effective_n_hours: self.uniqueness_sum * hours,
            purge_seconds: spec.purge_seconds(),
            locked_test: range.map(|range| LockedLeftOut { range, rows_left_out: self.left_out }),
            label_hash,
        }
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

    /// Values of the stored column `kind` (see [`COLUMN_KINDS`]).
    pub fn column(&self, kind: &str) -> Option<&[f64]> {
        Some(match kind {
            "fwd" => &self.fwd,
            "y" => &self.y,
            "long_net" => &self.long_net,
            "action" => &self.action,
            "t1" => &self.t1,
            "exit_ret" => &self.exit_ret,
            "m_net" => &self.m_net,
            "m_barrier" => &self.m_barrier,
            "weight" => &self.weight,
            _ => return None,
        })
    }

    fn tally(&self, symbol: Option<&str>, keep: &dyn Fn(i64) -> bool) -> Tally {
        let rows = self.fwd.len();
        let finite = (0..rows).filter(|&t| self.fwd[t].is_finite()).count();
        let valid: Vec<usize> = (0..rows).filter(|&t| self.fwd[t].is_finite() && keep(self.time[t])).collect();
        let with_y: Vec<usize> = valid.iter().copied().filter(|&t| self.y[t].is_finite()).collect();
        let kind = |k: u8| valid.iter().filter(|&&t| self.exit_kind[t] == k).count();
        let mut top: Vec<TopMove> = valid
            .iter()
            .map(|&t| TopMove { symbol: symbol.map(str::to_string), time: self.time[t], fwd: self.fwd[t] })
            .collect();
        top.sort_by(move_order);
        top.truncate(TOP_MOVES);
        Tally {
            rows,
            stride_rows: (0..rows).filter(|t| t % self.spec.stride != 0).count(),
            fwd: valid.iter().map(|&t| self.fwd[t]).collect(),
            barrier_rows: with_y.len(),
            clipped: with_y.iter().filter(|&&t| self.y[t].abs() >= self.spec.clip).count(),
            exits: [kind(EXIT_PROFIT), kind(EXIT_STOP), kind(EXIT_VERTICAL)],
            action_long: valid.iter().filter(|&&t| self.action[t] == 1.0).count(),
            uniqueness_sum: valid.iter().map(|&t| self.uniqueness[t]).sum(),
            top,
            left_out: finite - valid.len(),
        }
    }

    /// Report of this one series. Top moves carry no symbol.
    pub fn report(&self) -> LabelReport {
        self.tally(None, &|_| true).into_report(&self.spec, None, self.label_hash())
    }
}

/// Builds one report over several series (one per symbol). The panel label
/// hash is the SHA-256 of the spec, then each series' symbol and series hash in
/// feed order, so it changes when any symbol's labels change.
#[derive(Clone, Debug)]
pub struct LabelAccumulator {
    spec: LongLabelSpec,
    range: Option<Lockbox>,
    tally: Tally,
    chain: Vec<u8>,
}

impl LabelAccumulator {
    pub fn new(spec: &LongLabelSpec) -> Self {
        Self::with_range(spec, None)
    }

    /// Rows whose label window reaches `range` (entry within `purge_seconds`
    /// before its start, or inside it) are left out of every statistic. The
    /// label hash still covers every row.
    pub fn with_range(spec: &LongLabelSpec, range: Option<Lockbox>) -> Self {
        LabelAccumulator { spec: spec.clone(), range, tally: Tally::default(), chain: spec.canonical().into_bytes() }
    }

    pub fn add(&mut self, symbol: &str, labels: &LongLabels) {
        let symbol = (!symbol.is_empty()).then_some(symbol);
        let purge = self.spec.purge_seconds();
        let range = self.range;
        let keep = move |t: i64| range.map_or(true, |r| t + purge <= r.start || t >= r.end);
        self.tally.merge(labels.tally(symbol, &keep));
        self.chain.extend_from_slice(symbol.unwrap_or("").as_bytes());
        self.chain.push(0);
        self.chain.extend_from_slice(labels.label_hash().as_bytes());
    }

    pub fn finish(self) -> LabelReport {
        let hash = sha256_hex(&self.chain);
        self.tally.into_report(&self.spec, self.range, hash)
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
        let top = self
            .top_moves
            .iter()
            .map(|m| {
                format!(
                    "{{\"symbol\":{},\"time\":{},\"time_iso\":{},\"fwd\":{}}}",
                    m.symbol.as_deref().map(text).unwrap_or_else(|| "null".to_string()),
                    m.time,
                    text(&crate::enrich::format_utc_seconds(m.time)),
                    num(m.fwd)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let locked = self
            .locked_test
            .map(|l| {
                format!(
                    ",\"locked_test\":{{\"start_date\":{},\"end_date\":{},\"purge_seconds\":{},\"rows_left_out\":{}}}",
                    text(&crate::lockbox::date(l.range.start)),
                    text(&crate::lockbox::date(l.range.end)),
                    self.purge_seconds,
                    l.rows_left_out
                )
            })
            .unwrap_or_default();
        format!(
            "{{\"version\":{},\"name\":{},\"long_only\":true,\"rows\":{},\"valid_rows\":{},\"barrier_rows\":{},\"stride_rows\":{},\"exits\":{{\"profit\":{},\"stop\":{},\"vertical\":{}}},\"action_long_share\":{},\"clipped_share\":{},\"sign_share\":{{\"positive\":{},\"negative\":{},\"zero\":{}}},\"fwd_mean\":{},\"fwd_quantiles\":{{{}}},\"cost_to_move\":{},\"top_abs_moves\":{{\"n\":{},\"moves\":[{}]}},\"average_uniqueness\":{},\"effective_n\":{},\"effective_n_hours\":{},\"purge_seconds\":{}{},\"label_hash\":{}}}",
            text(VERSION),
            text(&self.name),
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
            TOP_MOVES,
            top,
            num(self.average_uniqueness),
            num(self.effective_n),
            num(self.effective_n_hours),
            self.purge_seconds,
            locked,
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

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_700_000_000 - 1_700_000_000 % 3600;
    const N: usize = 400;
    const RANGE: Lockbox = Lockbox { start: T0 + 200 * 3600, end: T0 + 300 * 3600 };

    fn series() -> Vec<Bar> {
        (0..N)
            .map(|i| {
                let x = i as f64;
                let mut o = 100.0 + 3.0 * (x * 0.37).sin();
                let mut c = 100.0 + 3.0 * (x * 0.37 + 0.4).sin();
                if i == 250 {
                    c = o * 1.5;
                }
                if i == 251 {
                    o = 150.0;
                    c = 150.0;
                }
                Bar::new(T0 + i as i64 * 3600, o, o.max(c) + 0.3, o.min(c) - 0.3, c, 1.0)
            })
            .collect()
    }

    fn spec(name: &str, h: usize, stride: usize) -> LongLabelSpec {
        let mut s = LongLabelSpec::new(name, 3600, h);
        s.sigma_window = 5;
        s.stride = stride;
        s
    }

    fn leaves_out(h: usize, stride: usize) {
        let s = spec("x", h, stride);
        let l = build(&series(), &s, None).unwrap();
        let purge = s.purge_seconds();
        let kept = |t: i64| t + purge <= RANGE.start || t >= RANGE.end;
        let mut acc = LabelAccumulator::with_range(&s, Some(RANGE));
        acc.add("A", &l);
        let r = acc.finish();

        let valid: Vec<usize> = (0..N).filter(|&t| l.fwd[t].is_finite()).collect();
        let kept_fwd: Vec<f64> = valid.iter().filter(|&&t| kept(l.time[t])).map(|&t| l.fwd[t]).collect();
        let out = valid.iter().filter(|&&t| l.time[t] + purge > RANGE.start && l.time[t] < RANGE.end).count();
        assert!(out > 0);
        assert_eq!(r.rows, N);
        assert_eq!(r.valid_rows, kept_fwd.len());
        assert_eq!(r.locked_test.unwrap().rows_left_out, out);
        assert_eq!(r.valid_rows + out, valid.len());
        let mean = kept_fwd.iter().sum::<f64>() / kept_fwd.len() as f64;
        assert!((r.fwd_mean - mean).abs() < 1e-12);
        for m in &r.top_moves {
            assert!(kept(m.time), "top move at {} reaches the locked range", m.time);
        }
        if h == 1 {
            assert!(l.fwd.iter().any(|v| v.abs() > 0.4), "the fixture holds the large move");
            assert!(r.top_moves.iter().all(|m| m.fwd.abs() < 0.4));
        }
        assert!(r.json().contains(&format!("\"rows_left_out\":{}", out)));

        let mut plain = LabelAccumulator::new(&s);
        plain.add("A", &l);
        let p = plain.finish();
        assert_eq!(p.valid_rows, valid.len());
        assert_eq!(p.label_hash, r.label_hash);
        assert!(!p.json().contains("locked_test"));
    }

    #[test]
    fn report_leaves_out_the_locked_range() {
        leaves_out(1, 1);
    }

    #[test]
    fn report_counts_left_out_rows_for_h4_stride_4() {
        leaves_out(4, 4);
    }
}
