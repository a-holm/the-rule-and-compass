//! #302: CPU reference for the PIT membership as-of join.
//!
//! The reference is checked against the production CPU path it mirrors
//! (`panel::build_panel_with_step`) on boundary fixtures
//! and a random block, then against its own prefix replay and a future-shift
//! canary. The device pass is compared against this reference in
//! `tests/panel_gpu.rs`.

#[path = "common/pit.rs"]
mod pit;

use pit::*;
use rcsr::ohlcv::Bar;
use rcsr::panel::{self, Membership, MembershipRow};
use rcsr::panel_pit::{self, PitBlock, PitParams};
use rcsr::panel_prep::{prepare, PrepBar, PrepSegment};
use std::collections::{BTreeMap, BTreeSet};

fn bar(t: i64) -> Bar {
    Bar::new(t, 10.0, 11.0, 9.0, 10.5, 1.0)
}

fn membership(rows: &[(i64, &str, bool)]) -> Membership {
    let mut m = Membership::default();
    let mut as_ofs = BTreeSet::new();
    for (as_of, symbol, member) in rows {
        as_ofs.insert(*as_of);
        if *member {
            m.selected.insert(symbol.to_string());
        }
        m.rows.push(MembershipRow {
            as_of: *as_of,
            symbol: symbol.to_string(),
            member: *member,
            adv_usd: 0.0,
            n_bars: 0,
        });
    }
    m.as_of = as_ofs.into_iter().collect();
    m
}

#[test]
fn membership_boundary_matches_build_panel() {
    // Rows one second before, at and after each decision.
    let times = [999, 1000, 1001, 1999, 2000, 2001, 2999, 3000, 3001, 3600];
    let mut series = BTreeMap::new();
    series.insert("A".to_string(), times.iter().map(|t| bar(*t)).collect::<Vec<_>>());
    series.insert("B".to_string(), times.iter().map(|t| bar(*t + 1)).collect::<Vec<_>>());
    let m = membership(&[
        (1000, "A", true),
        (2000, "A", false),
        (3000, "A", true),
        (1500, "B", true),
        (3001, "B", false),
    ]);
    let step = 1;
    let panel = panel::build_panel_with_step(&series, &m, step).unwrap();
    let block = PitBlock::from_membership(&series, &m);
    let out = panel_pit::join(&block, PitParams::membership(step)).unwrap();

    // Boundary: a row at as_of sees the decision; one second before does not.
    let a_at = |t: i64| {
        let r = block.times.iter().position(|x| *x == t).unwrap();
        (out.found[r], out.keep[r], out.as_of[r])
    };
    assert_eq!(a_at(999).0, 0);
    assert_eq!(a_at(1000), (1, 1, 1000));
    assert_eq!(a_at(1999), (1, 1, 1000));
    assert_eq!(a_at(2000), (1, 0, 2000));
    assert_eq!(a_at(3000), (1, 1, 3000));

    let mut want: Vec<(String, i64, i64, usize)> = panel
        .rows
        .iter()
        .map(|r| (r.symbol.clone(), r.bar.time, r.as_of, r.gap_buckets))
        .collect();
    want.sort();
    let mut got = Vec::new();
    for (si, (sym, _)) in series.iter().enumerate() {
        let s = block.segments[si];
        for r in s.start as usize..(s.start + s.len) as usize {
            if out.keep[r] == 1 {
                got.push((sym.clone(), block.times[r], out.as_of[r], out.gap[r] as usize));
            }
        }
    }
    got.sort();
    assert_eq!(got, want);
    assert_eq!(out.lookahead_violations, 0);
}

#[test]
fn membership_matches_build_panel_on_a_random_block() {
    let block = pit::membership_block(7, 6, 4000);
    // Rebuild bars and membership from the block, then compare with the panel.
    let mut series: BTreeMap<String, Vec<Bar>> = BTreeMap::new();
    let mut rows = Vec::new();
    for (i, s) in block.segments.iter().enumerate() {
        let sym = format!("S{i}");
        let mut seen = BTreeSet::new();
        let bars: Vec<Bar> = (s.start as usize..(s.start + s.len) as usize)
            .filter(|r| seen.insert(block.times[*r]))
            .map(|r| bar(block.times[r]))
            .collect();
        series.insert(sym.clone(), bars);
        for j in s.leaf_start as usize..(s.leaf_start + s.leaf_len) as usize {
            rows.push((block.keys[j], sym.clone(), block.member[j] == 1));
        }
    }
    let refs: Vec<(i64, &str, bool)> = rows.iter().map(|(a, s, m)| (*a, s.as_str(), *m)).collect();
    let m = membership(&refs);
    let step = 900;
    let panel = panel::build_panel_with_step(&series, &m, step).unwrap();
    let pb = PitBlock::from_membership(&series, &m);
    let out = panel_pit::join(&pb, PitParams::membership(step)).unwrap();
    let mut want: Vec<(String, i64, i64, usize)> = panel
        .rows
        .iter()
        .map(|r| (r.symbol.clone(), r.bar.time, r.as_of, r.gap_buckets))
        .collect();
    want.sort();
    let mut got = Vec::new();
    for (si, sym) in series.keys().enumerate() {
        let s = pb.segments[si];
        for r in s.start as usize..(s.start + s.len) as usize {
            if out.keep[r] == 1 {
                got.push((sym.clone(), pb.times[r], out.as_of[r], out.gap[r] as usize));
            }
        }
    }
    got.sort();
    assert_eq!(got.len(), want.len());
    assert_eq!(got, want);
}

#[test]
fn membership_exclusion_and_chain_reset_do_not_bridge_a_gap() {
    // 15m bars for 10 hours, excluded from hour 4 up to hour 6.
    let hour = 3600i64;
    let times: Vec<i64> = (0..40).map(|i| i * 900).collect();
    let series: BTreeMap<String, Vec<Bar>> =
        [("A".to_string(), times.iter().map(|t| bar(*t)).collect())].into();
    let m = membership(&[(0, "A", true), (4 * hour, "A", false), (6 * hour, "A", true)]);
    let block = PitBlock::from_membership(&series, &m);
    let out = panel_pit::join(&block, PitParams::membership(900)).unwrap();
    assert_eq!(out.kept_rows.len(), 32);
    // The first kept row after the exclusion has gap 0 and a new chain.
    let first_back = out.kept_rows.iter().position(|r| block.times[*r as usize] == 6 * hour).unwrap();
    let r = out.kept_rows[first_back] as usize;
    assert_eq!(out.gap[r], 0);
    assert_ne!(out.chain_id[r], out.chain_id[r - 9]);
    assert_eq!(out.chain_id[r - 9], out.chain_id[0]);

    // Feeding the chain ids to the #301 prep keeps the gap tag at 0 at the
    // reset; a single constant chain would bridge two missing hours.
    let feed = |ids: bool| {
        let bars: Vec<PrepBar> = out
            .kept_rows
            .iter()
            .map(|r| {
                let mut b = PrepBar::new(block.times[*r as usize], 10.0, 11.0, 9.0, 10.5)
                    .with_flow(1.0, 1.0, 1.0);
                b.chain = if ids { out.chain_id[*r as usize] } else { 0 };
                b
            })
            .collect();
        let segs = [PrepSegment { start: 0, len: bars.len() as u32 }];
        prepare(&bars, &segs, 900, 3600, false).unwrap()
    };
    let with_ids = feed(true);
    let without = feed(false);
    let gap_at_6 = |o: &rcsr::panel_prep::PrepOutput| {
        o.buckets.iter().find(|b| b.time == 6 * hour).unwrap().gap
    };
    assert_eq!(gap_at_6(&with_ids), 0);
    assert_eq!(gap_at_6(&without), 2);
}

#[test]
fn prefix_replay_at_twenty_cuts_is_bit_identical() {
    for (name, block, params) in [("membership", pit::membership_block(11, 5, 3000), PitParams::membership(900))] {
        let full = panel_pit::join(&block, params).unwrap();
        let lo = *block.times.iter().min().unwrap();
        let hi = *block.times.iter().max().unwrap();
        for k in 0..20 {
            let cut = lo + (hi - lo) * (k + 1) / 21;
            let pre_block = pit::prefix(&block, cut);
            let pre = panel_pit::join(&pre_block, params).unwrap();
            assert_prefix_equal(&format!("{name} cut {k}"), &block, &full, &pre_block, &pre);
        }
    }
}

#[test]
fn future_shifted_column_fails_lookahead_violations() {
    for (block, params) in [(pit::membership_block(21, 3, 2000), PitParams::membership(900))] {
        let out = panel_pit::join(&block, params).unwrap();
        assert_eq!(out.lookahead_violations, 0);
        // Shift every visible leaf one step into the future.
        let mut shifted = out.leaf.clone();
        for s in &block.segments {
            let le = (s.leaf_start + s.leaf_len) as i32;
            for r in s.start as usize..(s.start + s.len) as usize {
                if shifted[r] >= 0 && shifted[r] + 1 < le {
                    shifted[r] += 1;
                }
            }
        }
        let bad = panel_pit::lookahead_violations(&block, &shifted);
        assert!(bad > 0, "a future-shifted column must be reported");
    }
}

#[test]
fn unsorted_input_is_rejected() {
    let mut b = pit::membership_block(3, 1, 50);
    b.times.swap(3, 40);
    assert!(panel_pit::join(&b, PitParams::membership(900)).is_err());
}
