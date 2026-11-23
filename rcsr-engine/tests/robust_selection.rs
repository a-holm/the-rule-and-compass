//! #928: the robust finalist rule on planted books.

use rcsr::selection::{choose_robust, circular_block_se, select_robust, RobustMember, RobustMeta, RobustParams};
use rcsr::trading::{blocked_alpha, evaluate, TradingRows, TradingSpec};

const PERIODS: usize = 160;
const BLOCK_LEN: usize = 21;

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
    RobustParams { blocks, se_z, block_len: Some(BLOCK_LEN) }
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
    RobustMember { archive_rank, complexity, trades, block_alpha: vec![pooled], pooled_alpha: pooled, se, eligible, block_len: 1 }
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
    assert!(RobustParams { blocks: 4, se_z: 1.0, block_len: Some(0) }.validate().is_err());
    assert!(params(1, 0.0).validate().is_ok());
}

#[test]
fn one_block_without_se_ranks_by_the_validation_objective() {
    let mut p = planted();
    let alpha = select(&p, params(1, 0.0));
    let by_alpha = (0..4).max_by(|&a, &b| alpha.members[a].pooled_alpha.total_cmp(&alpha.members[b].pooled_alpha)).unwrap();
    let worst_alpha = (0..4).min_by(|&a, &b| alpha.members[a].pooled_alpha.total_cmp(&alpha.members[b].pooled_alpha)).unwrap();
    assert_ne!(by_alpha, worst_alpha);
    // Lower objective is better, so the worst-alpha member has the best objective.
    for (i, m) in p.metas.iter_mut().enumerate() {
        m.objective = alpha.members[i].pooled_alpha;
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

#[test]
fn edge_gate_rejects_every_fitness_not_below_one() {
    use rcsr::selection::edge_rejected;
    for (fitness, rejected) in [(0.99, false), (1.0, true), (1.01, true), (f64::NAN, true), (f64::INFINITY, true)] {
        assert_eq!(edge_rejected(true, fitness), rejected, "fitness {fitness}");
        assert!(!edge_rejected(false, fitness), "no gate without the flag: fitness {fitness}");
    }
}

fn meta(archive_rank: usize, complexity: f64) -> RobustMeta {
    RobustMeta { archive_rank, complexity, usable: true, objective: 0.0 }
}

fn one_group(periods: usize) -> TradingRows {
    TradingRows { time: (0..periods).map(|t| t as f64).collect(), group: vec![0; periods], funding: Vec::new() }
}

fn default_params() -> RobustParams {
    RobustParams { blocks: 4, se_z: 1.0, block_len: None }
}

/// Mixed-sign returns with no period structure.
fn noisy_returns(periods: usize) -> Vec<f64> {
    (0..periods).map(|t| (((t * 7919) % 13) as f64 / 13.0 - 0.4) * 0.01).collect()
}

fn run_default(periods: usize, pred: Vec<f64>) -> rcsr::selection::RobustSelection {
    let ret = noisy_returns(periods);
    select_robust(&[meta(1, 1.0)], default_params(), &TradingSpec::default(), &ret, &one_group(periods), &mut |_| pred.clone())
        .unwrap()
}

#[test]
fn circular_block_se_caps_the_block_at_half_the_series() {
    let series: Vec<f64> = (0..12).map(|t| ((t * 5) % 7) as f64 - 3.0).collect();
    let se = circular_block_se(&series, 12, 500, 9);
    assert!(se > 0.0, "{se}");
    assert_eq!(se, circular_block_se(&series, 6, 500, 9));
    assert_eq!(se, circular_block_se(&series, 100, 500, 9));
}

/// 12 periods in 4 blocks of 3. Per block the first period earns `g`, the second
/// `k g`, the third nothing. Member B (complexity 2) is long on the first
/// period, member A (complexity 9) on the first two; `k` makes A's pooled alpha
/// 1.05 times B's.
fn h336() -> (Vec<f64>, Vec<Vec<f64>>) {
    let g = [0.02, 0.03, 0.025, 0.04];
    let k = 1.1 / 2.05;
    let ret: Vec<f64> = (0..12).map(|t| match t % 3 { 0 => g[t / 3], 1 => k * g[t / 3], _ => 0.0 }).collect();
    let long = |upto: usize| -> Vec<f64> { (0..12).map(|t| if t % 3 < upto { 1.0 } else { -1.0 }).collect() };
    (ret, vec![long(2), long(1)])
}

#[test]
fn robust_h336_fixture_prefers_the_simpler_member_within_one_se() {
    let (ret, preds) = h336();
    let metas = [meta(1, 9.0), meta(2, 2.0)];
    let sel = select_robust(&metas, default_params(), &TradingSpec::default(), &ret, &one_group(12), &mut |i| preds[i].clone()).unwrap();
    let (a, b) = (&sel.members[0], &sel.members[1]);
    assert!((a.pooled_alpha / b.pooled_alpha - 1.05).abs() < 1e-9, "{a:?} {b:?}");
    assert!(b.eligible && b.block_alpha.iter().all(|&v| v > 0.0), "{b:?}");
    assert!(a.se > 0.0 && b.se > 0.0, "{a:?} {b:?}");
    assert_eq!((a.block_len, b.block_len), (3, 3));
    assert_eq!(sel.chosen, Some(1), "{:?}", sel.members);
    assert!(sel.fields_json().contains("\"block_len\":null"));
}

#[test]
fn robust_default_block_len_follows_the_period_count() {
    for (periods, want) in [(12, 3), (25, 3), (59, 4), (4287, 17)] {
        let pred: Vec<f64> = (0..periods).map(|t| if t % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let sel = run_default(periods, pred);
        assert_eq!(sel.members[0].block_len, want, "{periods} periods");
    }
}

#[test]
fn robust_default_block_len_covers_the_hold_up_to_a_quarter() {
    let runs = |periods: usize, on: usize, off: usize| -> Vec<f64> {
        (0..periods).map(|t| if t % (on + off) < on { 1.0 } else { -1.0 }).collect()
    };
    assert_eq!(run_default(4287, runs(4287, 60, 60)).members[0].block_len, 60);
    // Hold 10 on 25 periods wants 10; the cap is 25 / 4 = 6.
    let pred: Vec<f64> = (0..25).map(|t| if t < 10 || t >= 15 { 1.0 } else { -1.0 }).collect();
    assert_eq!(run_default(25, pred).members[0].block_len, 6);
}

#[test]
fn explicit_block_len_is_used_for_every_member_and_capped_at_half() {
    let (ret, preds) = h336();
    let metas = [meta(1, 9.0), meta(2, 2.0)];
    let run = |len: usize| {
        select_robust(&metas, RobustParams { blocks: 4, se_z: 1.0, block_len: Some(len) }, &TradingSpec::default(), &ret, &one_group(12), &mut |i| {
            preds[i].clone()
        })
    };
    let sel = run(6).unwrap();
    assert!(sel.members.iter().all(|m| m.block_len == 6));
    assert!(sel.fields_json().contains("\"block_len\":6,\"verdict\""));
    let e = run(7).unwrap_err();
    assert!(e.contains("--select-block-len 7") && e.contains("12 validation periods (6)"), "{e}");
}

/// Reference series: the mean of `net - exposure x bh` per key, blocks cut over
/// the distinct keys.
fn reference_series(keys: &[f64], pos: &[f64], ret: &[f64], blocks: usize) -> Vec<f64> {
    let mut distinct: Vec<f64> = keys.to_vec();
    distinct.sort_by(|a, b| a.total_cmp(b));
    distinct.dedup();
    let per = distinct.len() / blocks;
    let mean_at = |k: f64, f: &dyn Fn(usize) -> f64| {
        let idx: Vec<usize> = (0..keys.len()).filter(|&i| keys[i] == k).collect();
        idx.iter().map(|&i| f(i)).sum::<f64>() / idx.len() as f64
    };
    let block_of = |p: usize| (p / per).min(blocks - 1);
    let exposure: Vec<f64> = (0..blocks)
        .map(|b| {
            let ks: Vec<f64> = (0..distinct.len()).filter(|&p| block_of(p) == b).map(|p| mean_at(distinct[p], &|i| pos[i])).collect();
            ks.iter().sum::<f64>() / ks.len() as f64
        })
        .collect();
    (0..distinct.len())
        .map(|p| {
            let e = exposure[block_of(p)];
            let net = mean_at(distinct[p], &|i| pos[i] * ret[i]);
            let bh = mean_at(distinct[p], &|i| ret[i]);
            net - e * bh
        })
        .collect()
}

#[test]
fn blocked_alpha_series_is_one_entry_per_period_bucket() {
    const DAY: f64 = 86_400.0;
    let days = 40usize;
    let mut time = Vec::new();
    let mut group = Vec::new();
    let mut ret = Vec::new();
    let mut pred = Vec::new();
    for d in 0..days {
        for g in 0..2usize {
            time.push(d as f64 * DAY + g as f64 * 12.0 * 3600.0);
            group.push(g as u64);
            ret.push((((d * 7 + g * 5) % 11) as f64 - 4.0) * 0.002);
            pred.push(if (d + g) % 2 == 0 { 1.0 } else { -1.0 });
        }
    }
    let rows = TradingRows { time: time.clone(), group, funding: Vec::new() };
    let pos: Vec<f64> = pred.iter().map(|&p| if p > 0.0 { 1.0 } else { 0.0 }).collect();
    let bucketed = TradingSpec { period_seconds: Some(DAY), ..TradingSpec::default() };
    let a = blocked_alpha(&pred, &ret, &rows, &bucketed, 4).unwrap();
    assert_eq!(a.series.len(), 40);
    let day_keys: Vec<f64> = time.iter().map(|t| (t / DAY).floor()).collect();
    let want = reference_series(&day_keys, &pos, &ret, 4);
    let mean = want.iter().sum::<f64>() / want.len() as f64;
    assert!((a.pooled_alpha - mean).abs() < 1e-15, "{} {}", a.pooled_alpha, mean);
    for (g, w) in a.series.iter().zip(&want) {
        assert!((g - w).abs() < 1e-15, "{g} {w}");
    }
    let plain = blocked_alpha(&pred, &ret, &rows, &TradingSpec::default(), 4).unwrap();
    assert_eq!(plain.series.len(), 80);
    let want = reference_series(&time, &pos, &ret, 4);
    for (g, w) in plain.series.iter().zip(&want) {
        assert!((g - w).abs() < 1e-15, "{g} {w}");
    }
}
