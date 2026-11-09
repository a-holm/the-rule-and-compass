//! #928: the robust finalist rule on planted books.

use rcsr::selection::{
    choose_robust, circular_block_se, select_robust, RobustMember, RobustMeta, RobustParams, ROBUST_BLOCK_LEN,
};
use rcsr::trading::{blocked_alpha, evaluate, TradingRows, TradingSpec};

const PERIODS: usize = 160;

/// One group, 4 blocks of 40 periods. Rows of class `t % 4`:
/// class 0 earns steadily; class 1 earns only in block 0 (class 2 pays it back
/// there, so buy-and-hold stays small); classes 2 and 3 are noise-free filler.
fn returns() -> Vec<f64> {
    (0..PERIODS)
        .map(|t| {
            let block0 = t < 40;
            match (t % 4, block0) {
                (0, _) => 0.01,
                (1, true) => 0.05,
                (1, false) => -0.0005,
                (2, true) => -0.05,
                (2, false) => 0.0002,
                (_, true) => 0.0,
                (_, false) => -0.0002,
            }
        })
        .collect()
}

fn rows() -> TradingRows {
    TradingRows { time: (0..PERIODS).map(|t| t as f64).collect(), group: vec![0; PERIODS], funding: Vec::new() }
}

fn long_on(class: usize) -> Vec<f64> {
    (0..PERIODS).map(|t| if t % 4 == class { 1.0 } else { -1.0 }).collect()
}

struct Planted {
    preds: Vec<Vec<f64>>,
    metas: Vec<RobustMeta>,
}

/// Rank 1 flat, rank 2 steady (complexity 3), rank 3 spiky (complexity 9),
/// rank 4 always long (complexity 2).
fn planted() -> Planted {
    let meta = |archive_rank, complexity| RobustMeta { archive_rank, complexity, usable: true, objective: 0.0 };
    Planted {
        preds: vec![vec![-1.0; PERIODS], long_on(0), long_on(1), vec![1.0; PERIODS]],
        metas: vec![meta(1, 1.0), meta(2, 3.0), meta(3, 9.0), meta(4, 2.0)],
    }
}

fn params(blocks: usize, se_z: f64) -> RobustParams {
    RobustParams { blocks, se_z, block_len: ROBUST_BLOCK_LEN }
}

fn select(p: &Planted, params: RobustParams) -> rcsr::selection::RobustSelection {
    let ret = returns();
    select_robust(&p.metas, params, &TradingSpec::default(), &ret, &rows(), &mut |i| p.preds[i].clone()).unwrap()
}

#[test]
fn robust_selection_prefers_steady_member() {
    let p = planted();
    let ret = returns();
    let net = |i: usize| evaluate(&p.preds[i], &ret, &rows(), &TradingSpec::default()).unwrap().net_pnl;
    assert!(net(2) > net(1), "validation_objective would take the spiky member over the steady one");
    let sel = select(&p, params(4, 1.0));
    assert_eq!(sel.chosen, Some(1), "{:?}", sel.members);
    let steady = &sel.members[1];
    assert!(steady.eligible && steady.block_alpha.iter().all(|&a| a > 0.0), "{steady:?}");
    let spiky = &sel.members[2];
    assert!(!spiky.eligible);
    assert_eq!(spiky.block_alpha.iter().filter(|&&a| a > 0.0).count(), 1, "{spiky:?}");
}

#[test]
fn robust_selection_removes_beta() {
    let p = planted();
    let sel = select(&p, params(4, 1.0));
    let beta = &sel.members[3];
    assert!(beta.pooled_alpha.abs() < 1e-12, "{beta:?}");
    assert!(beta.block_alpha.iter().all(|a| a.abs() < 1e-12));
    assert!(!beta.eligible);
    assert!(!sel.members[0].eligible, "the flat member has no exposure");
}

#[test]
fn no_eligible_member_is_no_signal() {
    let p = Planted { preds: vec![vec![-1.0; PERIODS], vec![1.0; PERIODS]], metas: planted().metas[..2].to_vec() };
    let sel = select(&p, params(4, 1.0));
    assert_eq!(sel.chosen, None);
    assert!(sel.fields_json().contains("\"verdict\":\"no_signal\""));
    assert!(sel.fields_json().contains("\"chosen_rank\":null"));
}

#[test]
fn unusable_member_is_never_eligible() {
    let mut p = planted();
    p.metas[1].usable = false;
    let sel = select(&p, params(4, 1.0));
    assert!(!sel.members[1].eligible);
    assert_ne!(sel.chosen, Some(1));
}

#[test]
fn selection_reads_only_the_rows_it_is_given() {
    let p = planted();
    let a = select(&p, params(4, 1.0));
    let b = select(&p, params(4, 1.0));
    assert_eq!(a, b);
}

#[test]
fn blocks_need_enough_timestamps() {
    let short = TradingRows { time: vec![0.0, 1.0, 2.0], group: vec![0; 3], funding: Vec::new() };
    let e = blocked_alpha(&[1.0; 3], &[0.01; 3], &short, &TradingSpec::default(), 4).unwrap_err();
    assert!(e.contains("4 time blocks"), "{e}");
    let ret = [0.01; 3];
    let e = select_robust(&planted().metas, params(4, 1.0), &TradingSpec::default(), &ret, &short, &mut |_| vec![1.0; 3]).unwrap_err();
    assert!(e.contains("--select-blocks 4 needs at least 4 distinct validation timestamps, found 3"), "{e}");
}

#[test]
fn alpha_is_per_period_over_groups() {
    // Two groups, the second earns twice the first: the period mean is 1.5x.
    let time: Vec<f64> = (0..8).flat_map(|t| [t as f64, t as f64]).collect();
    let group: Vec<u64> = (0..8).flat_map(|_| [0u64, 1]).collect();
    let rows = TradingRows { time, group, funding: Vec::new() };
    let ret: Vec<f64> = (0..8).flat_map(|t| if t % 2 == 0 { [0.01, 0.02] } else { [-0.01, -0.02] }).collect();
    let pred: Vec<f64> = (0..8).flat_map(|t| if t % 2 == 0 { [1.0, 1.0] } else { [-1.0, -1.0] }).collect();
    let a = blocked_alpha(&pred, &ret, &rows, &TradingSpec::default(), 2).unwrap();
    assert_eq!(a.series.len(), 8);
    // net per period: 0.015 on even periods, 0 on odd; exposure 0.5; bh 0.015 or -0.015.
    // alpha per period = net - e * bh with e the block exposure 0.5.
    let want: Vec<f64> = (0..8).map(|t| if t % 2 == 0 { 0.015 - 0.5 * 0.015 } else { 0.0 + 0.5 * 0.015 }).collect();
    for (g, w) in a.series.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "{:?} vs {:?}", a.series, want);
    }
}

#[test]
fn circular_block_se_is_zero_for_a_constant_and_reproducible() {
    assert!(circular_block_se(&[0.3; 50], 7, 200, 1) < 1e-12);
    let series: Vec<f64> = (0..60).map(|t| ((t * 7919) % 13) as f64 / 13.0 - 0.5).collect();
    let a = circular_block_se(&series, 5, 300, 11);
    assert!(a > 0.0);
    assert_eq!(a, circular_block_se(&series, 5, 300, 11));
    assert_ne!(a, circular_block_se(&series, 5, 300, 12));
}

#[test]
fn circular_block_se_grows_with_block_length_for_a_trending_series() {
    let series: Vec<f64> = (0..120).map(|t| if (t / 20) % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let short = circular_block_se(&series, 1, 400, 3);
    let long = circular_block_se(&series, 20, 400, 3);
    assert!(long > 2.0 * short, "{short} {long}");
}

fn member(archive_rank: usize, complexity: f64, trades: usize, pooled: f64, se: f64, eligible: bool) -> RobustMember {
    RobustMember { archive_rank, complexity, trades, block_alpha: vec![pooled], pooled_alpha: pooled, se, eligible }
}

#[test]
fn parsimony_takes_the_simplest_within_the_standard_error() {
    let m = [member(1, 9.0, 10, 1.0, 0.5, true), member(2, 3.0, 10, 0.6, 0.4, true), member(3, 1.0, 10, 0.4, 0.4, true)];
    assert_eq!(choose_robust(&m, 1.0), Some(1), "floor 0.5 admits the 0.6 member, not the 0.4 member");
    assert_eq!(choose_robust(&m, 0.0), Some(0));
    assert_eq!(choose_robust(&m, 2.0), Some(2));
}

#[test]
fn parsimony_ties_go_to_fewer_trades_then_rank() {
    let m = [member(5, 3.0, 20, 1.0, 0.1, true), member(2, 3.0, 10, 1.0, 0.1, true), member(4, 3.0, 10, 1.0, 0.1, true)];
    assert_eq!(choose_robust(&m, 1.0), Some(1));
    let ineligible = [member(1, 1.0, 1, 5.0, 0.1, false), member(2, 3.0, 10, 0.2, 0.1, true)];
    assert_eq!(choose_robust(&ineligible, 1.0), Some(1));
    assert_eq!(choose_robust(&[member(1, 1.0, 1, 5.0, 0.1, false)], 1.0), None);
}

#[test]
fn params_are_validated() {
    assert!(params(0, 1.0).validate().is_err());
    assert!(params(9, 1.0).validate().is_err());
    assert!(params(4, -1.0).validate().is_err());
    assert!(RobustParams { blocks: 4, se_z: 1.0, block_len: 0 }.validate().is_err());
    assert!(params(1, 0.0).validate().is_ok());
}

#[test]
fn one_block_without_se_ranks_by_the_validation_objective() {
    let mut p = planted();
    let alpha = select(&p, params(1, 0.0));
    let by_alpha = (0..4).max_by(|&a, &b| alpha.members[a].pooled_alpha.total_cmp(&alpha.members[b].pooled_alpha)).unwrap();
    let worst_alpha = (0..4).min_by(|&a, &b| alpha.members[a].pooled_alpha.total_cmp(&alpha.members[b].pooled_alpha)).unwrap();
    assert_ne!(by_alpha, worst_alpha);
    // The objective orders the members the other way round (lower is better).
    for (i, m) in p.metas.iter_mut().enumerate() {
        m.objective = 10.0 - alpha.members[i].pooled_alpha;
    }
    let sel = select(&p, params(1, 0.0));
    assert_eq!(sel.chosen, Some(worst_alpha), "K=1, Z=0 takes the validation_objective choice, not the best alpha");
    assert_ne!(sel.chosen, Some(by_alpha));
    let with_se = select(&p, params(1, 1.0));
    assert_ne!(with_se.chosen, Some(worst_alpha), "any other setting keeps the alpha rule");
    p.metas[worst_alpha].objective = f64::INFINITY;
    let next = select(&p, params(1, 0.0)).chosen.unwrap();
    assert_ne!(next, worst_alpha);
}
