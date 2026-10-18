//! Error metrics.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Absolute,
    Squared,
    Correlation,
}

impl Metric {
    pub fn parse(s: &str) -> Metric {
        match s.to_ascii_lowercase().as_str() {
            "squared" | "square" | "mse" => Metric::Squared,
            "correlation" | "corr" | "r" => Metric::Correlation,
            _ => Metric::Absolute,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Metric::Absolute => "absolute",
            Metric::Squared => "squared",
            Metric::Correlation => "correlation",
        }
    }
}

/// Lower is better. Non-finite predictions are a hard failure.
pub fn score(pred: &[f64], y: &[f64], m: Metric) -> f64 {
    let n = pred.len();
    if n == 0 || y.len() != n {
        return f64::INFINITY;
    }
    for &p in pred {
        if !p.is_finite() {
            return f64::INFINITY;
        }
    }
    match m {
        Metric::Absolute => {
            pred.iter().zip(y).map(|(p, t)| (p - t).abs()).sum::<f64>() / n as f64
        }
        Metric::Squared => {
            pred.iter().zip(y).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64
        }
        Metric::Correlation => 1.0 - correlation(pred, y).abs(),
    }
}

pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        sa += da * da;
        sb += db * db;
    }
    if sa <= 0.0 || sb <= 0.0 {
        0.0
    } else {
        sab / (sa.sqrt() * sb.sqrt())
    }
}
