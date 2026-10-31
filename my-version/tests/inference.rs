//! Block uncertainty and multiple-testing statistics, on fixtures whose answer
//! is known by hand or analytically.

use rcsr::inference::*;
use rcsr::rng::Mt;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn normal(rng: &mut Mt) -> f64 {
    let u1 = (rng.next_f64()).max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn ar1(n: usize, phi: f64, seed: u32) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let mut x = 0.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n + 200 {
        x = phi * x + normal(&mut rng);
        if i >= 200 {
            out.push(x);
        }
    }
    out
}

/// Overlapping labels: `y_t = e_t + ... + e_{t+h-1}` on iid innovations.
fn overlapping(n: usize, h: usize, seed: u32) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let e: Vec<f64> = (0..n + h).map(|_| normal(&mut rng)).collect();
    (0..n).map(|t| e[t..t + h].iter().sum()).collect()
}

#[test]
fn cvar_is_the_mean_of_the_worst_tail() {
    let r = [-0.05, -0.03, 0.01, 0.02, 0.04, -0.01, 0.03, 0.0, 0.02, 0.01];
    assert!(close(cvar(&r, 0.95), -0.05, 1e-15));
    assert!(close(cvar(&r, 0.90), -0.05, 1e-15));
    assert!(close(cvar(&r, 0.80), -0.04, 1e-15));
    assert!(close(cvar(&r, 0.70), (-0.05 - 0.03 - 0.01) / 3.0, 1e-15));
}

#[test]
fn calmar_is_cagr_over_drawdown() {
    assert!(close(calmar(0.30, -0.15), 2.0, 1e-15));
    assert!(calmar(0.30, 0.0).is_nan());
}

#[test]
fn break_even_and_cost_ladder_match_a_hand_fixture() {
    let gross = [0.02, 0.0, 0.01, -0.01];
    let cost = [0.0005, 0.0005, 0.0, 0.0005];
    let net: Vec<f64> = gross.iter().zip(&cost).map(|(g, c)| g - c).collect();
    // mean gross 0.005, mean cost 0.000375: break-even is 10 * 0.005 / 0.000375.
    assert!(close(break_even_bps(&net, &gross, 10.0), 133.333_333_333_333_3, 1e-9));
    let ladder = cost_ladder(&net, &gross, 10.0, &[0.5, 1.0, 2.0, 3.0], 4.0);
    let expect = [0.0048125, 0.004625, 0.00425, 0.003875];
    for (row, e) in ladder.iter().zip(expect) {
        assert!(close(row.mean_return, e, 1e-15), "{:?}", row);
    }
    assert_eq!(ladder[3].cost_bps_per_side, 30.0);
    // At the break-even cost the mean net return is zero.
    let at_break_even = cost_ladder(&net, &gross, 10.0, &[13.333_333_333_333_33], 4.0);
    assert!(at_break_even[0].mean_return.abs() < 1e-15);
}

#[test]
fn benjamini_hochberg_step_up() {
    let p = [0.04, 0.001, 0.5, 0.02];
    let bh = bh_fdr(&p, 0.10);
    // sorted 0.001, 0.02, 0.04, 0.5 against 0.025, 0.05, 0.075, 0.10
    assert_eq!(bh.rejected, vec![true, true, false, true]);
    assert_eq!(bh.n_rejected, 3);
    let adj = [0.16 / 3.0, 0.004, 0.5, 0.04];
    for (a, e) in bh.adjusted.iter().zip(adj) {
        assert!(close(*a, e, 1e-15), "{} vs {}", a, e);
    }
    assert_eq!(bh_fdr(&[0.2, 0.3], 0.10).n_rejected, 0);
}

#[test]
fn null_p_uses_one_plus_count_over_b_plus_one() {
    let null = [1.0, 2.0, 5.0, 6.0, 7.0];
    assert!(close(null_p(5.0, &null), 4.0 / 6.0, 1e-15));
    assert!(close(null_p(100.0, &null), 1.0 / 6.0, 1e-15));
    assert!(close(null_p(-1.0, &null), 1.0, 1e-15));
}

#[test]
fn participation_ratio_counts_independent_trials() {
    let identical = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
    assert!(close(participation_ratio(&identical), 1.0, 1e-15));
    let identity = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    assert!(close(participation_ratio(&identity), 3.0, 1e-15));
    let rho = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    assert!(close(participation_ratio(&rho), 4.0 / 2.5, 1e-15));
    let cols = vec![
        vec![1.0, -1.0, 1.0, -1.0],
        vec![1.0, 1.0, -1.0, -1.0],
        vec![2.0, -2.0, 2.0, -2.0],
    ];
    let c = correlation_matrix(&cols);
    assert!(close(c[0][2], 1.0, 1e-12) && close(c[0][1], 0.0, 1e-12));
    // columns 0 and 2 are one trial, column 1 another
    assert!(close(participation_ratio(&c), 9.0 / 5.0, 1e-12));
}

#[test]
fn deflated_sharpe_matches_hand_values() {
    // One trial: PSR against zero. z = 0.1 * sqrt(100) / sqrt(1 + 0.1^2 * 2/4) = 1 / 1.0025.
    let psr = deflated_sharpe(0.1, 101.0, 0.0, 3.0, 1.0, 1.0);
    assert_eq!(psr.sr0, 0.0);
    assert!(close(psr.dsr, 0.840_740, 2e-5), "{}", psr.dsr);
    // 100 trials, unit trial variance: Phi^-1(0.99) = 2.3263479, Phi^-1(1 - 1/(100 e)) = 2.6807.
    let sr0 = expected_max_sharpe(100.0, 1.0);
    let hand = (1.0 - EULER_GAMMA) * 2.326_347_874 + EULER_GAMMA * 2.6807;
    assert!(close(sr0, hand, 2e-3), "{} vs {}", sr0, hand);
    // Deflation lowers the DSR and a larger trial count lowers it more.
    let a = deflated_sharpe(0.3, 250.0, -0.2, 4.5, 10.0, 0.01);
    let b = deflated_sharpe(0.3, 250.0, -0.2, 4.5, 100.0, 0.01);
    assert!(b.sr0 > a.sr0 && b.dsr < a.dsr);
    assert_eq!((a.t, a.skew, a.kurtosis, a.n_trials), (250.0, -0.2, 4.5, 10.0));
}

fn cols_from(a: &[f64], blocks: usize) -> usize {
    a.len() / blocks
}

#[test]
fn pbo_is_one_for_mirrored_candidates_and_zero_for_a_dominant_one() {
    // Candidate A trends down block by block; B = -A. The in-sample winner is
    // always the out-of-sample loser.
    let rows = 6;
    let blocks = 4;
    let level = [4.0, 1.0, -2.0, -3.0];
    let mut a = Vec::new();
    for l in level {
        for i in 0..rows {
            a.push(l + if i % 2 == 0 { 0.5 } else { -0.5 });
        }
    }
    assert_eq!(cols_from(&a, blocks), rows);
    let b: Vec<f64> = a.iter().map(|v| -v).collect();
    let pbo = pbo_cscv(&[a.clone(), b], blocks).unwrap();
    assert_eq!(pbo.combinations, 6);
    assert_eq!(pbo.pbo, 1.0);
    // rank 1 of 2: omega = 1/3, logit = ln(0.5)
    assert!(pbo.logits.iter().all(|&l| close(l, 0.5f64.ln(), 1e-12)));

    // A dominates B by a constant in every block.
    let c: Vec<f64> = a.iter().map(|v| v - 0.25).collect();
    let dom = pbo_cscv(&[a, c], blocks).unwrap();
    assert_eq!(dom.pbo, 0.0);
    assert!(dom.logits.iter().all(|&l| close(l, 2.0f64.ln(), 1e-12)));
}

#[test]
fn pbo_rejects_a_bad_block_count() {
    let c = vec![vec![0.0; 40], vec![0.0; 40]];
    assert!(pbo_cscv(&c, 3).is_err());
    assert!(pbo_cscv(&c[..1], 4).is_err());
}

#[test]
fn spa_separates_signal_from_noise() {
    let mut rng = Mt::new(11);
    let n = 600;
    let noise: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let mut rng2 = Mt::new(12);
    let noise2: Vec<f64> = (0..n).map(|_| normal(&mut rng2)).collect();
    let signal: Vec<f64> = noise2.iter().map(|v| v + 0.5).collect();
    let null_only = spa_test(&[noise.clone(), noise2.clone()], 5, 2000, 3).unwrap();
    assert!(null_only.p_consistent > 0.10, "{:?}", null_only);
    let with_signal = spa_test(&[noise, signal], 5, 2000, 3).unwrap();
    assert!(with_signal.p_consistent <= 0.01, "{:?}", with_signal);
    assert_eq!(with_signal.best_model, Some(1));
    assert!(with_signal.p_lower <= with_signal.p_consistent + 1e-12);
    assert!(with_signal.p_upper >= with_signal.p_consistent - 1e-12);
    // identical across runs
    let again = spa_test(&[vec![0.1, -0.2, 0.3, 0.0, 0.1, 0.2, -0.1, 0.05]], 2, 500, 9).unwrap();
    let twice = spa_test(&[vec![0.1, -0.2, 0.3, 0.0, 0.1, 0.2, -0.1, 0.05]], 2, 500, 9).unwrap();
    assert_eq!((again.statistic, again.p_consistent), (twice.statistic, twice.p_consistent));
}

#[test]
fn bootstrap_is_reproducible_and_matches_the_sample_mean() {
    let x = ar1(500, 0.3, 5);
    let a = block_bootstrap(&x, 10, 500, 7, 365.0, 0.95);
    let b = block_bootstrap(&x, 10, 500, 7, 365.0, 0.95);
    assert_eq!(a.mean_ci.lo, b.mean_ci.lo);
    assert_eq!(a.design_effect, b.design_effect);
    assert!(close(a.mean, x.iter().sum::<f64>() / 500.0, 1e-15));
    assert!(a.mean_ci.lo < a.mean && a.mean < a.mean_ci.hi);
    assert!(close(a.sharpe_annual, a.sharpe_per_period * 365f64.sqrt(), 1e-12));
}

#[test]
fn design_effect_recovers_the_ar1_variance_inflation() {
    // Var(mean) inflates by (1 + phi) / (1 - phi) = 17/3 for phi = 0.7.
    // The estimate of one series is noisy (about 10% at this length), so the
    // test averages eight independent series.
    let mut total = 0.0;
    for k in 0..8 {
        let x = ar1(40_000, 0.7, 21 + k);
        let s = block_bootstrap(&x, 300, 2000, 4 + k, 8760.0, 0.95);
        assert!(close(s.effective_n, 40_000.0 / s.design_effect, 1e-9));
        total += s.design_effect;
    }
    let de = total / 8.0;
    assert!((5.1..=6.2).contains(&de), "mean DE {}", de);
}

#[test]
fn design_effect_recovers_the_overlapping_label_inflation() {
    // Labels spanning h periods inflate Var(mean) by h against an iid view.
    let h = 12;
    let mut total = 0.0;
    for k in 0..8 {
        let x = overlapping(40_000, h, 31 + k);
        total += block_bootstrap(&x, 240, 2000, 4 + k, 8760.0, 0.95).design_effect;
    }
    let de = total / 8.0;
    assert!((10.8..=13.2).contains(&de), "mean DE {}", de);
    // The auto block length respects the label span and the holding period.
    assert_eq!(auto_block_len(40_000, h, 3), 35);
    assert_eq!(auto_block_len(1000, 5, 40), 40);
    assert_eq!(auto_block_len(8, 100, 100), 8);
}

#[test]
fn iid_over_rejects_on_ar1_and_the_block_does_not() {
    // The true mean is zero. A 95% interval should exclude zero about 5% of
    // the time.
    let trials = 200;
    let (mut iid_reject, mut block_reject) = (0, 0);
    for k in 0..trials {
        let x = ar1(400, 0.7, 1000 + k as u32);
        let iid = bootstrap_draws(&x, 1, 400, 17);
        let blk = bootstrap_draws(&x, 40, 400, 17);
        let excludes = |d: &Draws| {
            let mut m = d.means.clone();
            m.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = quantile_sorted(&m, 0.025);
            let hi = quantile_sorted(&m, 0.975);
            lo > 0.0 || hi < 0.0
        };
        iid_reject += excludes(&iid) as usize;
        block_reject += excludes(&blk) as usize;
    }
    let iid_rate = iid_reject as f64 / trials as f64;
    let block_rate = block_reject as f64 / trials as f64;
    assert!(iid_rate >= 0.30, "iid rejection rate {}", iid_rate);
    assert!(block_rate <= 0.20, "block rejection rate {}", block_rate);
    assert!(block_rate + 0.15 < iid_rate);
}
