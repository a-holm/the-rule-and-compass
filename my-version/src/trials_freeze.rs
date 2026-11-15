//! #1237: frozen sealed-read thresholds (stub for the red run).

use crate::risk::Drawdown;

pub fn windows(time: &[f64], path_end: f64, seconds: f64) -> Vec<(usize, usize)> {
    let _ = (time, path_end, seconds);
    Vec::new()
}

pub struct DrawdownTree {
    n: usize,
}

impl DrawdownTree {
    pub fn new(x: &[f64]) -> DrawdownTree {
        DrawdownTree { n: x.len() }
    }

    pub fn query(&self, a: usize, b: usize) -> Drawdown {
        let _ = (self.n, a, b);
        Drawdown::identity()
    }
}

pub struct RollingWindow {
    pub start: usize,
    pub end: usize,
    pub mean: f64,
    pub drawdown: f64,
}

pub fn rolling_windows(time: &[f64], x: &[f64], path_end: f64, days: u32) -> Vec<RollingWindow> {
    let _ = (time, x, path_end, days);
    Vec::new()
}
