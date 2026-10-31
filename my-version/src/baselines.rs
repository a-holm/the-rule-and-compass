//! #310: long-only baseline ladder: cash, buy-and-hold, constant long/flat and
//! matched-exposure random long.
//!
//! Every control is a `{0, 1}` position series over the same valid rows,
//! priced by the #304 reduction ([`crate::trading_gpu`]) and summarised with
//! the #307 risk monoids ([`crate::risk`]). There is no short control.
//!
//! * `cash`: `p = 0`. Net 0, cost 0, Sharpe `NaN` (zero variance).
//! * `buy_and_hold`: `p = 1`. One entry and one charged exit per group,
//!   `net = sum(r) - 2s * groups`.
//! * `constant`: one fixed side for the whole block, chosen from the training
//!   rows only: long iff the mean training target is strictly positive.
//! * `matched_exposure_random_long`: the strategy's own long/flat runs, with
//!   the order of the long runs and of the flat runs shuffled inside each
//!   group. Long rows, entries, exits and final exits are preserved exactly.
//!   The shuffle uses a counter-based RNG keyed by `(seed, group, stream,
//!   index)`, so a group's result does not depend on any other group, on the
//!   batch composition or on the thread count.
//!
//! Reported values are CPU `f64`. [`device_check`] prices the same position
//! vectors through the device kernel (or the canonical CPU reduction when no
//! device exists) as a cross-check.

use crate::risk::{risk_stats, BetaAlpha, DEFAULT_SORTINO_MAR};
use crate::trading_gpu::{
    buy_and_hold_totals, cash_totals, check_group_contiguous, check_long_only, reduce, PnlParams,
    PnlTotals,
};

/// Seed used when the caller does not pass `--control-seed`.
pub const DEFAULT_CONTROL_SEED: u64 = 0;

/// Score threshold that separates the `{0, 1}` position encoding.
const POSITION_THRESHOLD: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Cash,
    BuyAndHold,
    Constant,
    MatchedRandomLong,
}

impl ControlKind {
    pub const ALL: [ControlKind; 4] = [
        ControlKind::Cash,
        ControlKind::BuyAndHold,
        ControlKind::Constant,
        ControlKind::MatchedRandomLong,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ControlKind::Cash => "cash",
            ControlKind::BuyAndHold => "buy_and_hold",
            ControlKind::Constant => "constant",
            ControlKind::MatchedRandomLong => "matched_exposure_random_long",
        }
    }
}

/// The constant control's fixed rule: long iff the training mean is a strictly
/// positive finite number.
pub fn constant_is_long(train_mean: f64) -> bool {
    train_mean.is_finite() && train_mean > 0.0
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Counter-based 64-bit draw: a pure function of its four arguments.
pub fn counter_draw(seed: u64, group: u64, stream: u64, index: u64) -> u64 {
    mix64(mix64(mix64(mix64(seed) ^ group) ^ stream) ^ index)
}

/// Fisher-Yates with counter draws; the draw for step `i` is
/// `counter_draw(seed, group, stream, i)` mapped to `[0, i]` by a widening
/// multiply (no modulo bias).
fn shuffle(values: &mut [u64], seed: u64, group: u64, stream: u64) {
    for i in (1..values.len()).rev() {
        let u = counter_draw(seed, group, stream, i as u64);
        let j = ((u as u128 * (i as u128 + 1)) >> 64) as usize;
        values.swap(i, j);
    }
}

/// Matched-exposure random long positions.
///
/// `positions` is a `{0, 1}` series in group-contiguous order. Inside each
/// group the series is cut into maximal runs; the run lengths of the long runs
/// are shuffled among the long slots and the flat run lengths among the flat
/// slots. The slot pattern (and therefore the first and last state, the entry
/// count and the exit count) is unchanged.
pub fn matched_exposure_positions(
    positions: &[f64],
    groups: &[u64],
    seed: u64,
) -> Result<Vec<f64>, String> {
    if positions.len() != groups.len() {
        return Err(format!(
            "length mismatch: positions {}, groups {}",
            positions.len(),
            groups.len()
        ));
    }
    check_long_only(positions)?;
    check_group_contiguous(groups)?;
    let n = positions.len();
    let mut out = vec![0.0f64; n];
    let mut lo = 0usize;
    while lo < n {
        let mut hi = lo + 1;
        while hi < n && groups[hi] == groups[lo] {
            hi += 1;
        }
        let mut long_runs: Vec<u64> = Vec::new();
        let mut flat_runs: Vec<u64> = Vec::new();
        let mut slots: Vec<bool> = Vec::new();
        let mut start = lo;
        while start < hi {
            let state = positions[start] != 0.0;
            let mut end = start + 1;
            while end < hi && (positions[end] != 0.0) == state {
                end += 1;
            }
            slots.push(state);
            if state {
                long_runs.push((end - start) as u64);
            } else {
                flat_runs.push((end - start) as u64);
            }
            start = end;
        }
        shuffle(&mut long_runs, seed, groups[lo], 0);
        shuffle(&mut flat_runs, seed, groups[lo], 1);
        let (mut li, mut fi) = (0usize, 0usize);
        let mut at = lo;
        for state in slots {
            let len = if state {
                li += 1;
                long_runs[li - 1]
            } else {
                fi += 1;
                flat_runs[fi - 1]
            } as usize;
            if state {
                for p in &mut out[at..at + len] {
                    *p = 1.0;
                }
            }
            at += len;
        }
        lo = hi;
    }
    Ok(out)
}

/// Per-row net return of a `{0, 1}` book: `p * r - s|dp|`, plus `s * p` on the
/// last row of each group. Same arithmetic as the #304 row term.
pub fn net_rows(positions: &[f64], returns: &[f64], groups: &[u64], side: f64) -> Vec<f64> {
    let n = returns.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = positions[i];
        let prev = if i > 0 && groups[i] == groups[i - 1] {
            positions[i - 1]
        } else {
            0.0
        };
        let mut cost = side * (p - prev).abs();
        if (i + 1 == n || groups[i + 1] != groups[i]) && p != 0.0 {
            cost += side * p;
        }
        out.push(p * returns[i] - cost);
    }
    out
}

/// CPU `f64` totals for an explicit `{0, 1}` position vector, through the #304
/// canonical reduction.
pub fn position_totals(
    positions: &[f64],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Result<PnlTotals, String> {
    check_long_only(positions)?;
    let p = PnlParams {
        threshold: POSITION_THRESHOLD,
        ..*params
    };
    Ok(reduce(positions, returns, groups, &p)?.strategy)
}

/// The position vector of each control, in [`ControlKind::ALL`] order.
pub fn control_positions(
    strategy_positions: &[f64],
    groups: &[u64],
    constant_long: bool,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let n = groups.len();
    Ok(vec![
        vec![0.0; n],
        vec![1.0; n],
        vec![if constant_long { 1.0 } else { 0.0 }; n],
        matched_exposure_positions(strategy_positions, groups, seed)?,
    ])
}

/// One control beside its risk numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlRow {
    pub kind: ControlKind,
    pub totals: PnlTotals,
    /// Long rows over valid rows.
    pub exposure: f64,
    /// `sum |position change|`, final exit excluded.
    pub turnover: f64,
    /// Beta of the control's period series on the buy-and-hold period series.
    pub beta: f64,
    /// `net - cash.net`, over the whole block.
    pub excess_over_cash: f64,
    /// Annualised Sharpe of the control's period series (`NaN` for cash).
    pub sharpe: f64,
    pub mean_return: f64,
    pub periods: u64,
}

/// The four controls for one block, plus how they were produced.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlSet {
    pub seed: u64,
    /// Equal-weight net period series of each control, in [`ControlKind::ALL`]
    /// order, on the block's valid periods.
    pub series: Vec<Vec<f64>>,
    /// The constant control's side (`true` = long).
    pub constant_long: bool,
    pub train_mean: f64,
    pub rows: Vec<ControlRow>,
    pub device: Option<DeviceCheck>,
}

impl ControlSet {
    pub fn get(&self, kind: ControlKind) -> &ControlRow {
        self.rows
            .iter()
            .find(|r| r.kind == kind)
            .expect("all four controls are always present")
    }
}

/// Device versus CPU `f64` agreement for the four controls.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceCheck {
    /// `cuda` when the kernel ran, `cpu-canonical` otherwise.
    pub backend: String,
    /// Largest absolute difference over `gross`, `cost`, `turnover`, `net`.
    pub max_abs_delta: f64,
    /// Integer counters (rows, long rows, entries, exits, trades, hits) agree.
    pub counts_equal: bool,
}

fn totals_delta(a: &PnlTotals, b: &PnlTotals) -> f64 {
    [
        (a.gross - b.gross).abs(),
        (a.cost - b.cost).abs(),
        (a.turnover - b.turnover).abs(),
        (a.net - b.net).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn counts_equal(a: &PnlTotals, b: &PnlTotals) -> bool {
    a.rows == b.rows
        && a.long_rows == b.long_rows
        && a.entries == b.entries
        && a.exits == b.exits
        && a.trades == b.trades
        && a.hits == b.hits
        && a.final_exits == b.final_exits
}

/// Price the control position vectors on the device in one batch (candidate
/// tile of at most [`crate::trading_gpu::MAX_TILE_CANDIDATES`]). `None` when
/// this build or machine has no device.
#[cfg(feature = "cuda")]
pub fn device_totals(
    positions: &[Vec<f64>],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Option<Vec<PnlTotals>> {
    if !crate::trading_gpu::device_available() || returns.is_empty() {
        return None;
    }
    let slices: Vec<&[f64]> = positions.iter().map(|p| p.as_slice()).collect();
    let thresholds = vec![POSITION_THRESHOLD; positions.len()];
    let reports = crate::trading_gpu::reduce_device_batch(
        &slices, returns, groups, &thresholds, params, 256, 65_536,
    )?;
    Some(reports.into_iter().map(|r| r.strategy).collect())
}

#[cfg(not(feature = "cuda"))]
pub fn device_totals(
    _positions: &[Vec<f64>],
    _returns: &[f64],
    _groups: &[u64],
    _params: &PnlParams,
) -> Option<Vec<PnlTotals>> {
    None
}

/// Cross-check the CPU totals of `positions` against the device (or the
/// canonical CPU path when no device exists).
pub fn device_check(
    positions: &[Vec<f64>],
    cpu: &[PnlTotals],
    returns: &[f64],
    groups: &[u64],
    params: &PnlParams,
) -> Result<DeviceCheck, String> {
    let (backend, other) = match device_totals(positions, returns, groups, params) {
        Some(t) => ("cuda".to_string(), t),
        None => {
            let mut v = Vec::with_capacity(positions.len());
            for p in positions {
                v.push(position_totals(p, returns, groups, params)?);
            }
            ("cpu-canonical".to_string(), v)
        }
    };
    let mut max_abs_delta = 0.0f64;
    let mut all_equal = true;
    for (a, b) in cpu.iter().zip(&other) {
        max_abs_delta = max_abs_delta.max(totals_delta(a, b));
        all_equal &= counts_equal(a, b);
    }
    Ok(DeviceCheck {
        backend,
        max_abs_delta,
        counts_equal: all_equal,
    })
}

/// Build the four controls for one block.
///
/// `returns`/`groups` are the valid rows in group-contiguous order and
/// `strategy_positions` is the strategy's own `{0, 1}` book on those rows.
/// `period_series` turns a per-row net vector (aligned with the inputs) into
/// the block's equal-weight period series, so the controls use exactly the
/// strategy's own aggregation. `check_device` adds the device cross-check.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_controls(
    strategy_positions: &[f64],
    returns: &[f64],
    groups: &[u64],
    side: f64,
    train_mean: f64,
    seed: u64,
    periods_per_year: f64,
    check_device: bool,
    period_series: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Result<ControlSet, String> {
    let params = PnlParams {
        per_side_bps: side * 10_000.0,
        ..PnlParams::default()
    };
    let constant_long = constant_is_long(train_mean);
    let positions = control_positions(strategy_positions, groups, constant_long, seed)?;
    let cash = cash_totals(returns, params.cash_return);
    let bh = buy_and_hold_totals(returns, groups, side);
    let totals = [
        cash,
        bh,
        if constant_long { bh } else { cash },
        position_totals(&positions[3], returns, groups, &params)?,
    ];
    let series: Vec<Vec<f64>> = positions
        .iter()
        .map(|p| period_series(&net_rows(p, returns, groups, side)))
        .collect();
    let bh_series = &series[1];
    let mut rows = Vec::with_capacity(4);
    for (k, kind) in ControlKind::ALL.into_iter().enumerate() {
        let mut ba = BetaAlpha::identity();
        for (x, y) in bh_series.iter().zip(&series[k]) {
            ba.push(*x, *y);
        }
        let stats = risk_stats(&series[k], periods_per_year, DEFAULT_SORTINO_MAR);
        rows.push(ControlRow {
            kind,
            totals: totals[k],
            exposure: totals[k].exposure(),
            turnover: totals[k].turnover,
            beta: ba.beta(),
            excess_over_cash: totals[k].net - cash.net,
            sharpe: stats.sharpe,
            mean_return: stats.mean,
            periods: stats.n,
        });
    }
    let device = if check_device {
        Some(device_check(&positions, &totals, returns, groups, &params)?)
    } else {
        None
    };
    Ok(ControlSet {
        seed,
        series,
        constant_long,
        train_mean,
        rows,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(p: &[f64]) -> (usize, usize) {
        let mut entries = 0;
        let mut exits = 0;
        let mut prev = 0.0;
        for &x in p {
            if x > prev {
                entries += 1;
            }
            if x < prev {
                exits += 1;
            }
            prev = x;
        }
        (entries, exits)
    }

    #[test]
    fn matched_preserves_rows_entries_exits_per_group() {
        let p = [
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, //
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let g = [0u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let m = matched_exposure_positions(&p, &g, 7).unwrap();
        assert_eq!(
            p[..10].iter().sum::<f64>(),
            m[..10].iter().sum::<f64>()
        );
        assert_eq!(runs(&p[..10]), runs(&m[..10]));
        assert_eq!(runs(&p[10..]), runs(&m[10..]));
        assert_eq!(m[9], p[9]);
        assert_eq!(m, matched_exposure_positions(&p, &g, 7).unwrap());
    }

    #[test]
    fn counter_draw_is_a_pure_function() {
        assert_eq!(counter_draw(1, 2, 3, 4), counter_draw(1, 2, 3, 4));
        assert_ne!(counter_draw(1, 2, 3, 4), counter_draw(1, 2, 3, 5));
        assert_ne!(counter_draw(1, 2, 3, 4), counter_draw(2, 2, 3, 4));
    }

    #[test]
    fn short_positions_are_rejected() {
        assert!(matched_exposure_positions(&[0.0, -1.0], &[0, 0], 0).is_err());
    }
}
