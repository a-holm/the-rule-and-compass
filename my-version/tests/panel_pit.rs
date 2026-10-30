//! #302: CPU reference for the PIT membership / MTF as-of joins.
//!
//! The reference is checked against the production CPU paths it mirrors
//! (`panel::build_panel_with_step`, `ohlcv::join_causal`) on boundary fixtures
//! and a random block, then against its own prefix replay and a future-shift
//! canary. The device pass is compared against this reference in
//! `tests/panel_gpu.rs`.

#[path = "common/pit.rs"]
mod pit;

use pit::*;
use rcsr::ohlcv::{self, Bar, Timeframe};
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
fn mtf_matches_join_causal_age_and_staleness() {
    let tf = Timeframe::parse("1h").unwrap();
    let base: Vec<Bar> = (0..40).map(|i| bar(7200 + i * 900)).collect();
    // The 09:00 and 10:00 hourly buckets (relative) are missing.
    let higher: Vec<Bar> = [0, 3600, 7200, 14400, 18000].iter().map(|t| bar(*t)).collect();
    let block = PitBlock::from_mtf(&[&base], &[&higher]);
    let out = panel_pit::join(&block, PitParams::mtf(tf.seconds(), 900)).unwrap();
    let joined = ohlcv::join_causal(&base, &higher, tf).unwrap();
    for (i, slot) in joined.iter().enumerate() {
        match slot {
            Some(h) => {
                assert_eq!(block.keys[out.leaf[i] as usize], h.time);
                let age = base[i].time - (h.time + tf.seconds());
                assert_eq!(out.age[i], age as f64);
                let expected = (base[i].time / 3600) * 3600 - 3600;
                assert_eq!(out.fresh[i] == 1, h.time == expected, "row {i}");
            }
            None => {
                assert_eq!(out.leaf[i], -1);
                assert!(out.age[i].is_nan());
            }
        }
    }
    // MTF exactly at t: the bar [0,3600) is visible at 3600, not at 3599.
    let b2 = [bar(3599), bar(3600)];
    let h2 = [bar(0)];
    let blk = PitBlock::from_mtf(&[&b2], &[&h2]);
    let o = panel_pit::join(&blk, PitParams::mtf(3600, 900)).unwrap();
    assert_eq!(o.leaf, vec![-1, 0]);
    assert_eq!(o.age[1], 0.0);
    // A stale bucket keeps a finite age but is not fresh.
    let stale = base.iter().position(|b| b.time >= 25_200).unwrap();
    assert!(out.age[stale].is_finite());
    assert_eq!(out.fresh[stale], 0);
    assert!(out.age[0].is_finite());
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
    // reset; a single constant chain would bridge two hours.
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
    assert_eq!(gap_at_6(&without), 1);
}

#[test]
fn prefix_replay_at_twenty_cuts_is_bit_identical() {
    for (name, block, params) in [
        ("membership", pit::membership_block(11, 5, 3000), PitParams::membership(900)),
        ("mtf1h", pit::mtf_block(12, 5, 3000, 3600), PitParams::mtf(3600, 900)),
        ("mtf1d", pit::mtf_block(13, 5, 3000, 86_400), PitParams::mtf(86_400, 900)),
    ] {
        let full = panel_pit::join(&block, params).unwrap();
        let lo = *block.times.iter().min().unwrap();
        let hi = *block.times.iter().max().unwrap();
        for k in 0..20 {
            let cut = lo + (hi - lo) * (k + 1) / 21;
            let pre_block = pit::prefix(&block, cut, params.delay);
            let pre = panel_pit::join(&pre_block, params).unwrap();
            assert_prefix_equal(&format!("{name} cut {k}"), &block, &full, &pre_block, &pre);
        }
    }
}

#[test]
fn future_shifted_column_fails_lookahead_violations() {
    for (block, params) in [
        (pit::membership_block(21, 3, 2000), PitParams::membership(900)),
        (pit::mtf_block(22, 3, 2000, 3600), PitParams::mtf(3600, 900)),
    ] {
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
        let bad = panel_pit::lookahead_violations(&block, &shifted, params.delay);
        assert!(bad > 0, "a future-shifted column must be reported");
    }
}

#[test]
fn future_shift_count_equals_the_production_ohlcv_check() {
    let block = pit::mtf_block(23, 1, 3000, 3600);
    let out = panel_pit::join(&block, PitParams::mtf(3600, 900)).unwrap();
    let mut shifted = out.leaf.clone();
    let le = block.keys.len() as i32;
    for v in shifted.iter_mut() {
        if *v >= 0 && *v + 1 < le {
            *v += 1;
        }
    }
    let tf = Timeframe::parse("1h").unwrap();
    let base: Vec<Bar> = block.times.iter().map(|t| bar(*t)).collect();
    let joined: Vec<Option<Bar>> = shifted
        .iter()
        .map(|j| (*j >= 0).then(|| bar(block.keys[*j as usize])))
        .collect();
    let want = ohlcv::lookahead_violations(&base, &joined, tf).len() as u64;
    assert!(want > 0);
    assert_eq!(panel_pit::lookahead_violations(&block, &shifted, 3600), want);
}

#[test]
fn unsorted_input_is_rejected() {
    let mut b = pit::membership_block(3, 1, 50);
    b.times.swap(3, 40);
    assert!(panel_pit::join(&b, PitParams::membership(900)).is_err());
}
