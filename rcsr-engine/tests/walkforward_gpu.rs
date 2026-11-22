//! Walk-forward device P&L check (#124, #304).
//!
//! The `--device-pnl` block recomputes the long-only book through the fused
//! canonical reduction. A cuda build runs it on the device; these tests live
//! here so the pure `walkforward` binary never opens the device.

use rcsr::data;
use rcsr::walkforward::{self, EvalSpec, Window};

fn table(rows: &[(f64, &str, f64, f64)]) -> data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    for (ts, sym, x, y) in rows {
        s.push_str(&format!("{},{},{},{}\n", ts, sym, x, y));
    }
    data::parse_table(&s).unwrap()
}

fn single(rows: &[(f64, f64, f64)]) -> data::Table {
    let rows: Vec<(f64, &str, f64, f64)> = rows
        .iter()
        .map(|&(ts, x, y)| (ts, "A", x, y))
        .collect();
    table(&rows)
}

fn window(label: &str, a: f64, b: f64, c: f64, d: f64) -> Window {
    Window {
        label: label.to_string(),
        train_start: a,
        train_end: b,
        test_start: c,
        test_end: d,
    }
}

fn spec(expr: &str, windows: Vec<Window>) -> EvalSpec {
    EvalSpec {
        expression: expr.to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: None,
        windows,
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.70,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(4.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    }
}
#[cfg(feature = "cuda")]
fn device_ready() -> bool {
    let present = rcsr::gpu::probe().available();
    if !present {
        rcsr::backend::skip_without_device("skipping: no usable CUDA device");
    }
    present
}

#[test]
fn device_pnl_check_matches_the_scalar_book() {
    #[cfg(feature = "cuda")]
    if !device_ready() {
        return;
    }
    let t = single(&[
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
        (4.0, 4.0, 0.01),
        (5.0, 5.0, -0.02),
        (6.0, 6.0, 0.03),
        (7.0, 7.0, -0.01),
    ]);
    let mut s = spec("x", vec![window("w", 0.0, 4.0, 4.0, 8.0)]);
    s.cost_bps = 100.0;
    s.threshold_quantile = 0.5;
    s.device_pnl = true;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let d = r.windows[0].device_pnl.as_ref().expect("device P&L block");
    #[cfg(feature = "cuda")]
    assert_eq!(d.backend, "cuda");
    let m = &r.windows[0].metrics;
    assert_eq!(d.strategy.rows, m.rows as u64);
    assert_eq!(d.strategy.entries, m.entries as u64);
    assert_eq!(d.strategy.trades, m.trades as u64);
    assert!((d.strategy.net - d.cpu_net).abs() <= 1e-12);
    assert!((d.strategy.turnover - m.total_abs_delta).abs() <= 1e-12);
    assert_eq!(d.cash.net, 0.0);
    assert_eq!(d.cash.cost, 0.0);
    let summary = r.device_pnl.as_ref().expect("report summary");
    assert_eq!(summary.windows, 1);
    assert_eq!(summary.rows, m.rows as u64);
    assert!(summary.max_abs_net_delta <= 1e-12);
    assert!(r.json().contains("\"device_pnl\""));
    assert!(r.json().contains("\"buy_and_hold\""));
}

#[test]
fn device_pnl_check_is_clean_on_an_underflowing_score() {
    #[cfg(feature = "cuda")]
    if !device_ready() {
        return;
    }
    // gauss(0.5) is normal, gauss(11) is below the f32 range, gauss(30) is
    // exactly 0 in f64.
    let xs = [0.5, 11.0, 30.0];
    let mut rows: Vec<(f64, f64, f64)> = Vec::new();
    for i in 0..4 {
        rows.push((i as f64, 30.0, 0.0));
    }
    let mut flips = 0u64;
    for i in 0..24 {
        let x = xs[i % 3];
        if x == 11.0 {
            flips += 1;
        }
        rows.push((4.0 + i as f64, x, 0.002 * ((i % 5) as f64) - 0.003));
    }
    let t = single(&rows);
    let mut s = spec("gauss(x)", vec![window("w", 0.0, 4.0, 4.0, 28.0)]);
    s.threshold_fixed = Some(0.0);
    s.device_pnl = true;
    let r = walkforward::run_table(&t, &s, "test").unwrap();
    let summary = r.device_pnl.as_ref().expect("report summary");
    assert!(summary.max_abs_net_delta <= 1e-12, "{}", summary.max_abs_net_delta);
    assert_eq!(summary.bit_mismatch_windows, 0);
    assert_eq!(summary.count_mismatch_windows, 0);
    assert_eq!(summary.f32_flip_rows, flips);
    let d = r.windows[0].device_pnl.as_ref().expect("device P&L block");
    #[cfg(feature = "cuda")]
    assert_eq!(d.backend, "cuda");
    assert_eq!(d.f32_flip_rows, flips);
    assert_eq!(d.strategy.long_rows, 16);
}
