//! Public acceptance for the risk monoids (issue #307).
//!
//! These use the five-row net-return fixture from the GPU coverage note
//! (`03-pnl-trading.md` §1) plus a chunked parity check. The serial CPU
//! `walkforward::sharpe`/`equity_metrics` parity lives in `src/risk.rs` where
//! the private reference is reachable.

use rcsr::risk::{
    compound_blocks, risk_stats, AnnualisationMode, BetaAlpha, RiskAccumulator,
    CANONICAL_SEGMENT, DEFAULT_SORTINO_MAR,
};
use std::time::Instant;

fn fixture_returns() -> Vec<f64> {
    // Net returns from the worked 5-row long/flat fixture at 10 bp/side:
    // [0.009, -0.001, 0.029, -0.001, 0.013], sum 0.049.
    vec![0.009, -0.001, 0.029, -0.001, 0.013]
}

#[test]
fn five_row_fixture_compounds_and_logs() {
    let r = fixture_returns();
    assert!((r.iter().sum::<f64>() - 0.049).abs() < 1e-15);
    let s = risk_stats(&r, 365.0, DEFAULT_SORTINO_MAR);
    let equity = r.iter().fold(1.0f64, |e, &x| e * (1.0 + x));
    assert!((s.total_return - (equity - 1.0)).abs() < 1e-15);
    assert!((s.log_growth_per_period - (equity.ln() / 5.0)).abs() < 1e-15);
    // The note's rounded value (the exact value is pinned above).
    assert!((s.log_growth_per_period - 0.009697).abs() < 1e-5);
    assert_eq!(s.n, 5);
}

#[test]
fn chunk_merge_is_partition_independent() {
    let r = fixture_returns();
    let whole = risk_stats(&r, 8760.0, 0.0);
    for parts in [&[1usize][..], &[2, 3][..], &[5][..]] {
        let mut acc = RiskAccumulator::new(0.0);
        let mut at = 0;
        for &n in parts.iter().cycle() {
            if at >= r.len() {
                break;
            }
            let end = (at + n).min(r.len());
            acc.push_chunk(&r[at..end]);
            at = end;
        }
        assert_eq!(acc.finish(8760.0), whole, "parts {parts:?}");
    }
}

#[test]
fn beta_alpha_public_tuple() {
    let x = [0.01, 0.02, -0.01, 0.03];
    let y = [0.02, 0.01, 0.00, 0.04];
    let mut ba = BetaAlpha::identity();
    ba.push(x[0], y[0]);
    ba.push(x[1], y[1]);
    ba.push(x[2], y[2]);
    ba.push(x[3], y[3]);
    assert_eq!(ba.n, 4);
    assert!(ba.beta().is_finite());
    assert!(ba.annualised_alpha(365.0).is_finite());
}

#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
fn three_point_nine_million_row_risk_pass() {
    let n = 3_943_856usize;
    let mut s = 0x307u64;
    let mut returns = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        returns.push((u - 0.5) * 0.04);
    }
    let start = Instant::now();
    let stats = risk_stats(&returns, 8760.0, DEFAULT_SORTINO_MAR);
    let elapsed = start.elapsed();
    let ns_per_row = elapsed.as_nanos() as f64 / n as f64;
    println!(
        "risk pass: rows={n} ns/row={ns_per_row:.3} wall={:?} segments={} merge_order=canonical-{CANONICAL_SEGMENT}-left-to-right sharpe={:.9} max_dd={:.9}",
        elapsed,
        n.div_ceil(CANONICAL_SEGMENT),
        stats.sharpe,
        stats.max_drawdown
    );
    assert_eq!(stats.n as usize, n);
}

#[test]
fn daily_mode_and_cash_are_explicit() {
    let hourly = [0.01, -0.005, 0.002, 0.0];
    let daily = compound_blocks(&hourly, 2);
    assert_eq!(daily.len(), 2);
    let ppy = AnnualisationMode::DailyCompounded.periods_per_year(24.0);
    assert_eq!(ppy, 365.0);
    let s = risk_stats(&[0.0; 32], ppy, 0.0);
    assert!(s.sharpe.is_nan(), "cash Sharpe must be NaN, not 0");
    assert!(s.sortino.is_nan());
}
