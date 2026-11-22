//! Deterministic synthetic hourly OHLCV panel for the fitness A/B (#483).
#![allow(dead_code)]

use std::fmt::Write as _;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn norm(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// First bar time: 2020-01-01T00:00:00Z.
pub const SYNTH_T0: i64 = 1_577_836_800;

/// CSV `ts,symbol,open,high,low,close,volume`, symbol-major then time. A shared
/// market factor, GARCH-like variance, a weak 24-bar momentum and a 3-bar
/// reversal drift. The same `(symbols, bars, seed)` gives the same bytes.
pub fn synth_panel_csv(symbols: usize, bars: usize, seed: u64) -> Vec<u8> {
    let mut mrng = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
    let market: Vec<f64> = (0..bars).map(|_| mrng.norm()).collect();
    let mut out = String::with_capacity(symbols * bars * 70);
    out.push_str("ts,symbol,open,high,low,close,volume\n");
    for s in 0..symbols {
        let mut rng = Rng(seed.wrapping_mul(1_000_003).wrapping_add(s as u64 * 7919 + 1));
        let mut price = 100.0 * (1.0 + s as f64 * 0.1);
        let mut var = 0.007f64 * 0.007;
        let mut hist: Vec<f64> = Vec::with_capacity(bars);
        for t in 0..bars {
            let sigma = var.sqrt();
            let k = hist.len();
            let m24: f64 = hist[k.saturating_sub(24)..].iter().sum();
            let m3: f64 = hist[k.saturating_sub(3)..].iter().sum();
            let z24 = m24 / (sigma * 24f64.sqrt());
            let z3 = m3 / (sigma * 3f64.sqrt());
            let mu = sigma * (0.06 * z24.tanh() - 0.04 * z3.tanh());
            let eps = 0.6 * market[t] + 0.8 * rng.norm();
            let r = mu + sigma * eps;
            var = (0.000002 + 0.05 * (r * r) + 0.93 * var).clamp(1e-6, 4e-4);
            let open = price * (1.0 + 0.0002 * rng.norm());
            let close = open * r.exp();
            let high = open.max(close) * (1.0 + (rng.norm().abs() * sigma * 0.5));
            let low = open.min(close) * (1.0 - (rng.norm().abs() * sigma * 0.5));
            let vol = 1000.0 * (0.5 * rng.norm() + 3.0 * (r.abs() / sigma).min(5.0) * 0.2).exp();
            hist.push(r);
            price = close;
            writeln!(
                out,
                "{},S{:02}USDT,{:.8},{:.8},{:.8},{:.8},{:.4}",
                SYNTH_T0 + 3600 * t as i64,
                s,
                open,
                high,
                low,
                close,
                vol
            )
            .unwrap();
        }
    }
    out.into_bytes()
}
