//! Exact long/flat P&L evaluator: hand fixtures, cash, exits, refusals and
//! bit-for-bit agreement with the walk-forward scalar evaluator.

use rcsr::trading::{self, Objective, TradingRows, TradingSpec};
use rcsr::walkforward;

const DAY: f64 = 86_400.0;

fn rows(n: usize) -> TradingRows {
    TradingRows {
        time: (0..n).map(|i| i as f64 * DAY).collect(),
        group: vec![0; n],
        funding: Vec::new(),
    }
}

fn spec(cost_bps: f64) -> TradingSpec {
    TradingSpec {
        cost_bps,
        ..Default::default()
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn five_row_fixture_matches_hand_arithmetic() {
    // positions 1,1,0,1,1 at 10 bp per side; the last row is held, so it also
    // pays the exit.
    let pred = [1.0, 1.0, -1.0, 1.0, 1.0];
    let r = [0.01, -0.02, 0.03, 0.04, -0.01];
    let c = 0.001;
    let net = [0.01 - c, -0.02, -c, 0.04 - c, -0.01 - c];
    let s = trading::evaluate(&pred, &r, &rows(5), &spec(10.0)).unwrap();

    close(s.gross_pnl, 0.01 - 0.02 + 0.04 - 0.01);
    close(s.net_pnl, net.iter().sum::<f64>());
    close(s.total_cost, 4.0 * c);
    assert_eq!((s.trades, s.entries, s.exits), (3, 2, 1));
    close(s.exposure, 0.8);
    close(s.hit_rate, 0.5);

    let total: f64 = net.iter().fold(1.0, |e, x| e * (1.0 + x)) - 1.0;
    close(s.total_return, total);
    close(s.log_growth, net.iter().map(|x| x.ln_1p()).sum::<f64>());

    let mean = net.iter().sum::<f64>() / 5.0;
    let var = net.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 4.0;
    close(s.sharpe, mean / var.sqrt() * 365f64.sqrt());
    assert_eq!(s.annualisation, "per_period");
    close(s.periods_per_year, 365.0);

    let mut eq = 1.0f64;
    let mut peak = 1.0f64;
    let mut dd = 0.0f64;
    for x in net {
        eq *= 1.0 + x;
        peak = peak.max(eq);
        dd = dd.min(eq / peak - 1.0);
    }
    close(s.max_drawdown, dd);
}

#[test]
fn break_even_cost_zeroes_the_net_pnl() {
    let pred = [1.0, 1.0, -1.0, 1.0, 1.0];
    let r = [0.01, -0.02, 0.03, 0.04, -0.01];
    let s = trading::evaluate(&pred, &r, &rows(5), &spec(10.0)).unwrap();
    // gross 0.02 over 4 units of |delta p| including the exit: 50 bp.
    close(s.break_even_bps, 50.0);
    let at = trading::evaluate(&pred, &r, &rows(5), &spec(s.break_even_bps)).unwrap();
    assert!(at.net_pnl.abs() < 1e-12, "{}", at.net_pnl);
    let ladder_10 = s.ladder.iter().find(|l| l.cost_bps == 10.0).unwrap();
    close(ladder_10.net_pnl, s.net_pnl);
}

#[test]
fn a_held_book_pays_one_entry_and_one_exit_not_a_charge_per_row() {
    let n = 17;
    let s = trading::evaluate(&vec![1.0; n], &vec![0.0; n], &rows(n), &spec(100.0)).unwrap();
    close(s.total_cost, 0.02);
    close(s.net_pnl, -0.02);
    assert_eq!((s.trades, s.entries, s.exits), (1, 1, 1));
}

#[test]
fn the_final_sell_to_flat_charges_the_exit() {
    let held = trading::evaluate(&[1.0, 1.0, 1.0], &[0.0; 3], &rows(3), &spec(50.0)).unwrap();
    close(held.net_pnl, -0.01);
    // Selling on the last row costs the same: one entry, one sell.
    let sold = trading::evaluate(&[1.0, 1.0, -1.0], &[0.0; 3], &rows(3), &spec(50.0)).unwrap();
    close(sold.net_pnl, -0.01);
    assert_eq!((sold.trades, sold.exits), (2, 0));
}

#[test]
fn cash_earns_the_declared_return_and_costs_nothing() {
    let mut sp = spec(25.0);
    sp.cash_return = 0.001;
    sp.min_trades = 0;
    let flat = trading::evaluate(&[-1.0; 5], &[0.05; 5], &rows(5), &sp).unwrap();
    close(flat.net_pnl, 0.005);
    assert_eq!((flat.trades, flat.exits), (0, 0));
    close(flat.total_cost, 0.0);
    close(flat.cash.net_pnl, 0.005);

    // Long on rows 1-2, flat on 3-5: cash is earned only while flat.
    let mixed = trading::evaluate(&[1.0, 1.0, -1.0, -1.0, -1.0], &[0.01; 5], &rows(5), &sp).unwrap();
    close(mixed.gross_pnl, 0.02 + 3.0 * 0.001);
    close(mixed.net_pnl, 0.02 + 3.0 * 0.001 - 2.0 * 0.0025);
}

#[test]
fn funding_is_paid_only_while_long() {
    let mut r = rows(4);
    r.funding = vec![0.001; 4];
    let s = trading::evaluate(&[1.0, 1.0, -1.0, -1.0], &[0.0; 4], &r, &spec(0.0)).unwrap();
    close(s.funding_paid, 0.002);
    close(s.net_pnl, -0.002);
}

#[test]
fn a_negative_position_is_a_hard_error() {
    let e = trading::evaluate_positions(&[1.0, -1.0, 0.0], &[0.0; 3], &rows(3), &spec(0.0)).unwrap_err();
    assert!(e.contains("negative position"), "{e}");
    let e = trading::evaluate_positions(&[0.5, 1.5, 0.0], &[0.0; 3], &rows(3), &spec(0.0)).unwrap_err();
    assert!(e.contains("exceeds max_position"), "{e}");
    let ok = trading::evaluate_positions(&[0.0, 0.5, 1.0], &[0.01; 3], &rows(3), &spec(0.0)).unwrap();
    close(ok.gross_pnl, 0.015);
}

#[test]
fn sizing_map_stays_in_the_unit_interval() {
    let mut sp = spec(0.0);
    sp.tau = Some(2.0);
    sp.threshold = 1.0;
    for (pred, want) in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5), (3.0, 1.0), (9.0, 1.0)] {
        close(trading::position_of(pred, &sp), want);
    }
    assert!(trading::position_of(f64::NAN, &sp).is_nan());
}

#[test]
fn min_holding_delays_the_exit() {
    let mut sp = spec(0.0);
    sp.min_holding = 3;
    let s = trading::evaluate(&[1.0, -1.0, -1.0, -1.0, 1.0, -1.0], &[1.0; 6], &rows(6), &sp);
    // Long on rows 1-3 (held three rows), flat on 4, long on 5-6.
    close(s.unwrap().exposure, 5.0 / 6.0);
}

#[test]
fn non_finite_predictions_and_returns_are_dropped() {
    let pred = [1.0, f64::NAN, 1.0, 1.0];
    let r = [0.01, 0.5, f64::NAN, 0.02];
    let s = trading::evaluate(&pred, &r, &rows(4), &spec(0.0)).unwrap();
    assert_eq!(s.rows, 2);
    close(s.gross_pnl, 0.03);
}

#[test]
fn sub_daily_periods_are_compounded_per_day() {
    // 3 days of hourly rows, always long, +0.1 % an hour.
    let n = 72;
    let rows = TradingRows {
        time: (0..n).map(|i| i as f64 * 3600.0).collect(),
        group: vec![0; n],
        funding: Vec::new(),
    };
    let r: Vec<f64> = (0..n).map(|i| if i % 24 < 12 { 0.002 } else { 0.0 }).collect();
    let s = trading::evaluate(&vec![1.0; n], &r, &rows, &spec(0.0)).unwrap();
    assert_eq!(s.annualisation, "daily_compounded");
    close(s.periods_per_year, 365.0);
    assert_eq!(s.periods, 72);
    // Identical days: zero variance, so the daily Sharpe is undefined, not 0.
    assert!(s.sharpe.is_nan());
}

#[test]
fn objective_scores_are_lower_is_better_and_nan_is_unusable() {
    let pred = [1.0, 1.0, -1.0, 1.0, 1.0];
    let r = [0.01, -0.02, 0.03, 0.04, -0.01];
    let s = trading::evaluate(&pred, &r, &rows(5), &spec(10.0)).unwrap();
    close(Objective::Pnl.score(&s, 1), -s.net_pnl);
    close(Objective::Sharpe.score(&s, 1), -s.sharpe);
    close(Objective::LogGrowth.score(&s, 1), -s.log_growth);
    assert!(Objective::Pnl.score(&s, 4).is_infinite(), "min_trades");
    let flat = trading::evaluate(&[-1.0; 5], &r, &rows(5), &TradingSpec { min_trades: 0, ..spec(0.0) }).unwrap();
    assert!(flat.sharpe.is_nan());
    assert!(Objective::Sharpe.score(&flat, 0).is_infinite());
    assert_eq!(Objective::parse("mse").unwrap(), Objective::Metric);
    assert!(Objective::parse("sortino").is_err());
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

#[test]
fn long_only_matches_the_walkforward_evaluator_bit_for_bit() {
    let mut st = 7u64;
    for cost_bps in [0.0, 7.5] {
        let n = 90;
        let mut time = Vec::new();
        let mut group = Vec::new();
        let mut pred = Vec::new();
        let mut r = Vec::new();
        for g in 0..2u64 {
            for i in 0..n {
                time.push(i as f64 * DAY);
                group.push(g);
                // Group 0 ends flat, so only group 1 pays an exit.
                let last = i == n - 1;
                let p = lcg(&mut st) - 0.5;
                pred.push(if last && g == 0 { -1.0 } else if last { 1.0 } else { p });
                r.push(0.02 * (lcg(&mut st) - 0.5));
            }
        }
        let want = walkforward::reference_metrics(&time, &group, &r, &pred, 0.0, cost_bps * 1e-4, 365.0);
        let got = trading::evaluate(
            &pred,
            &r,
            &TradingRows { time, group, funding: Vec::new() },
            &spec(cost_bps),
        )
        .unwrap();
        assert_eq!(format!("{:?}", got.period), format!("{:?}", want), "cost {cost_bps}");
        assert!(got.period.trades > 10);
    }
}
