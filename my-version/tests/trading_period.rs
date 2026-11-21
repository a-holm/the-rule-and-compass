//! #1016: the search-side book uses the same period rule as walk-forward.

use rcsr::trading::{self, TradingRows, TradingSpec};

const HOUR: f64 = 3600.0;
const DAY: f64 = 86_400.0;
const T0: f64 = 1_609_459_200.0;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn ret(i: usize, g: usize) -> f64 {
    0.01 * (frac(i as f64 * 0.754_877_666_246_692_7 + g as f64 * 0.31) - 0.5)
}

#[test]
fn series_of_staggered_stride_is_daily() {
    let (mut time, mut group, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..30usize {
        for (g, phase) in [0.0, 7.0, 15.0].into_iter().enumerate() {
            time.push(T0 + DAY * d as f64 + HOUR * phase);
            group.push(g as u64);
            r.push(ret(d, g));
        }
    }
    let rows = TradingRows { time, group, funding: Vec::new() };
    let pred = vec![1.0; r.len()];
    let daily = TradingSpec { period_seconds: Some(DAY), ..Default::default() };
    let s = trading::evaluate(&pred, &r, &rows, &daily).unwrap();
    assert_eq!(s.annualisation, "per_period");
    assert_eq!(s.periods_per_year, 365.0);
    assert_eq!(s.periods, 30);
    let mut want = 1.0f64;
    for d in 0..30usize {
        want *= 1.0 + (0..3usize).map(|g| ret(d, g)).sum::<f64>() / 3.0;
    }
    assert!((s.total_return - (want - 1.0)).abs() < 1e-12, "{} vs {}", s.total_return, want - 1.0);
    let plain = trading::evaluate(&pred, &r, &rows, &TradingSpec::default()).unwrap();
    assert_eq!(s.net_pnl.to_bits(), plain.net_pnl.to_bits());
}

#[test]
fn validation_net_per_row_h1() {
    let (mut time, mut group, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..120usize {
        for g in 0..2usize {
            time.push(T0 + HOUR * i as f64);
            group.push(g as u64);
            r.push(ret(i, g));
        }
    }
    let rows = TradingRows { time, group, funding: Vec::new() };
    let pred: Vec<f64> = (0..r.len()).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
    let spec = TradingSpec { cost_bps: 5.0, period_seconds: Some(HOUR), ..Default::default() };
    let s = trading::evaluate(&pred, &r, &rows, &spec).unwrap();
    assert_eq!(s.periods_per_year, 8760.0);
    assert_eq!(s.annualisation, "per_period");
    let want = s.net_pnl / s.rows as f64 * 8760.0;
    assert!((s.net_per_row_annualised - want).abs() <= 1e-12, "{} vs {}", s.net_per_row_annualised, want);
    assert!(s.json().contains("\"net_per_row_annualised\":"));
}

#[test]
fn series_of_stride_above_horizon() {
    let (mut time, mut group, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..30usize {
        for g in 0..3usize {
            time.push(T0 + DAY * d as f64);
            group.push(g as u64);
            r.push(ret(d, g));
        }
    }
    let rows = TradingRows { time, group, funding: Vec::new() };
    let pred = vec![1.0; r.len()];
    let label = rcsr::labels::LongLabelSpec::parse("long:1,name=h1s24,fee=0,stride=24,anchor=2021-01-04", 3600).unwrap();
    let target = label.target_spec("lbl_y", "lbl_fwd");
    assert_eq!(target.horizon_seconds, HOUR);
    let spec = TradingSpec { period_seconds: Some(target.period_width_seconds()), ..Default::default() };
    let s = trading::evaluate(&pred, &r, &rows, &spec).unwrap();
    assert_eq!(s.periods_per_year, 365.0);
    assert_eq!(s.periods, 30);
    let horizon_only = TradingSpec { period_seconds: Some(target.horizon_seconds), ..Default::default() };
    let wrong = trading::evaluate(&pred, &r, &rows, &horizon_only).unwrap();
    assert_eq!(wrong.periods_per_year, 8760.0);
    assert_eq!(s.net_pnl.to_bits(), wrong.net_pnl.to_bits());
}
