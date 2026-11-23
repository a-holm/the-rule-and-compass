//! #302: point-in-time as-of join for panel membership.
//!
//! This module is the exact CPU reference for [`crate::panel_pit_gpu`] /
//! `src/cuda/pit.cu`. The membership **decisions** (ranking, ADV tie-breaks,
//! `member`) are made by [`crate::panel`] on the CPU; this join only maps every
//! base row to the latest decision that was visible at its timestamp.
//!
//! A decision is visible to a row at time `t` when `as_of <= t`, so a decision at
//! `as_of` is visible to a row at `as_of`.
//!
//! Base rows and decisions are non-decreasing inside one symbol segment. Ties in
//! the decision list resolve to the **last** equal key, like
//! `partition_point(|r| r.as_of <= t)`.
//!
//! Outputs per base row:
//!
//! * `leaf`: index of the visible decision in the global decision list, or `-1`;
//! * `found`: a decision is visible;
//! * `keep`: a decision is visible and has `member = 1`;
//! * `as_of`: the `as_of` of the visible decision (0 when none);
//! * `gap`: `max(0, (t[i] - t[i-1]) / step - 1)` when row `i` and row `i - 1`
//!   are both kept, in one segment and in one input chain; else 0;
//! * `chain_id`: running number of chain starts, counted from 1 inside each
//!   segment so no other symbol or later row can change it. A kept row starts a chain
//!   unless it continues the previous row. `member = 0`, a missing decision,
//!   a segment change (listing/delisting) and a change of the input chain
//!   (migration) therefore start a new chain. The value is only meaningful on
//!   kept rows and feeds `PrepBar.chain` of [`crate::panel_prep`];
//! * `kept_rows`: ascending indices of the kept rows.

use crate::ohlcv::Bar;
use crate::panel::Membership;
use std::collections::BTreeMap;

/// One symbol: a contiguous base row range and a contiguous leaf range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitSegment {
    pub start: u32,
    pub len: u32,
    pub leaf_start: u32,
    pub leaf_len: u32,
}

/// A decoded join input. All vectors are symbol-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PitBlock {
    pub times: Vec<i64>,
    /// Input chain id per row (migration); equal ids may be bridged by `gap`.
    pub chain: Vec<i64>,
    pub segments: Vec<PitSegment>,
    pub keys: Vec<i64>,
    /// `member` flag per leaf.
    pub member: Vec<u8>,
}

/// Join parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitParams {
    /// Gap bucket size in seconds.
    pub step: i64,
}

impl PitParams {
    /// Membership join: `<=` at `as_of`.
    pub fn membership(step: i64) -> Self {
        PitParams { step }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PitOutput {
    pub leaf: Vec<i32>,
    pub keep: Vec<u8>,
    pub found: Vec<u8>,
    pub as_of: Vec<i64>,
    pub gap: Vec<u32>,
    pub chain_id: Vec<i64>,
    pub kept_rows: Vec<u32>,
    /// Look-ahead violations of this join's own `leaf` column (always 0).
    pub lookahead_violations: u64,
}

impl PitBlock {
    pub fn validate(&self) -> Result<(), String> {
        let n = self.times.len();
        if self.chain.len() != n {
            return Err(format!("chain has {} rows, times {}", self.chain.len(), n));
        }
        if self.member.len() != self.keys.len() {
            return Err("member and keys differ in length".to_string());
        }
        let mut want_row = 0u64;
        let mut want_leaf = 0u64;
        for (i, s) in self.segments.iter().enumerate() {
            if s.start as u64 != want_row || s.leaf_start as u64 != want_leaf {
                return Err(format!("segment {} is not contiguous", i));
            }
            want_row += s.len as u64;
            want_leaf += s.leaf_len as u64;
            let rows = &self.times[s.start as usize..(s.start + s.len) as usize];
            if rows.windows(2).any(|w| w[1] < w[0]) {
                return Err(format!("segment {}: base times are not sorted", i));
            }
            let keys = &self.keys[s.leaf_start as usize..(s.leaf_start + s.leaf_len) as usize];
            if keys.windows(2).any(|w| w[1] < w[0]) {
                return Err(format!("segment {}: leaf keys are not sorted", i));
            }
        }
        if want_row != n as u64 {
            return Err(format!("segments cover {} rows, input has {}", want_row, n));
        }
        if want_leaf != self.keys.len() as u64 {
            return Err(format!(
                "segments cover {} leaves, input has {}",
                want_leaf,
                self.keys.len()
            ));
        }
        Ok(())
    }

    /// Membership join input from the CPU decisions: one segment per symbol of
    /// `series` (map order), leaves sorted by `as_of` (stable, as
    /// `panel::membership_index`).
    pub fn from_membership(series: &BTreeMap<String, Vec<Bar>>, membership: &Membership) -> Self {
        Self::from_membership_slices(series.iter().map(|(s, b)| (s.as_str(), b.as_slice())), membership)
    }

    /// [`Self::from_membership`] over borrowed bar slices, one segment each in
    /// iteration order.
    pub fn from_membership_slices<'a>(
        series: impl Iterator<Item = (&'a str, &'a [Bar])>,
        membership: &Membership,
    ) -> Self {
        let mut by_symbol: BTreeMap<&str, Vec<(i64, u8)>> = BTreeMap::new();
        for r in &membership.rows {
            by_symbol.entry(r.symbol.as_str()).or_default().push((r.as_of, r.member as u8));
        }
        for v in by_symbol.values_mut() {
            v.sort_by_key(|d| d.0);
        }
        let mut block = PitBlock::default();
        for (symbol, bars) in series {
            let start = block.times.len() as u32;
            let leaf_start = block.keys.len() as u32;
            for b in bars {
                block.times.push(b.time);
                block.chain.push(0);
            }
            if let Some(v) = by_symbol.get(symbol) {
                for (as_of, m) in v {
                    block.keys.push(*as_of);
                    block.member.push(*m);
                }
            }
            block.segments.push(PitSegment {
                start,
                len: bars.len() as u32,
                leaf_start,
                leaf_len: block.keys.len() as u32 - leaf_start,
            });
        }
        block
    }
}

/// The exact CPU reference join.
pub fn join(block: &PitBlock, p: PitParams) -> Result<PitOutput, String> {
    block.validate()?;
    if p.step <= 0 {
        return Err("pit: step must be positive".to_string());
    }
    let n = block.times.len();
    let mut out = PitOutput {
        leaf: vec![-1; n],
        keep: vec![0; n],
        found: vec![0; n],
        as_of: vec![0; n],
        gap: vec![0; n],
        chain_id: vec![0; n],
        kept_rows: Vec::new(),
        lookahead_violations: 0,
    };
    for s in &block.segments {
        let mut ord = 0i64;
        let ls = s.leaf_start as usize;
        let le = ls + s.leaf_len as usize;
        let keys = &block.keys[ls..le];
        for r in s.start as usize..(s.start + s.len) as usize {
            let t = block.times[r];
            let pos = keys.partition_point(|k| *k <= t);
            if pos > 0 {
                let j = ls + pos - 1;
                let key = block.keys[j];
                out.leaf[r] = j as i32;
                out.found[r] = 1;
                out.keep[r] = block.member[j];
                out.as_of[r] = key;
            }
            let cont = out.keep[r] == 1
                && r > s.start as usize
                && out.keep[r - 1] == 1
                && block.chain[r] == block.chain[r - 1];
            if cont {
                let d = (t - block.times[r - 1]) / p.step - 1;
                out.gap[r] = d.max(0) as u32;
            }
            if out.keep[r] == 1 && !cont {
                ord += 1;
            }
            out.chain_id[r] = ord;
            if out.keep[r] == 1 {
                out.kept_rows.push(r as u32);
            }
        }
    }
    out.lookahead_violations = lookahead_violations(block, &out.leaf);
    Ok(out)
}

/// Count rows whose leaf index is outside the row's leaf range or not yet
/// visible (`as_of > t`). A future-shifted column is reported here.
pub fn lookahead_violations(block: &PitBlock, leaf: &[i32]) -> u64 {
    let mut bad = 0u64;
    for s in &block.segments {
        let ls = s.leaf_start as i64;
        let le = ls + s.leaf_len as i64;
        for r in s.start as usize..(s.start + s.len) as usize {
            let j = leaf[r] as i64;
            if j < 0 {
                continue;
            }
            if j < ls || j >= le || block.keys[j as usize] > block.times[r] {
                bad += 1;
            }
        }
    }
    bad
}
