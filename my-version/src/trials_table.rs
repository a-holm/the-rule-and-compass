//! #1177: the multi-metric candidate table of `rcsr trials --table` (WIP stub).

pub struct Friedman {
    pub chi2: f64,
    pub df: usize,
    pub p: f64,
}

pub fn pareto_fronts(values: &[Vec<f64>]) -> Vec<usize> {
    vec![0; values.len()]
}

pub fn column_ranks(col: &[f64]) -> Vec<f64> {
    vec![f64::NAN; col.len()]
}

pub fn average_ranks(values: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (vec![Vec::new(); values.len()], vec![f64::NAN; values.len()])
}

pub fn friedman(_ranks: &[Vec<f64>]) -> Friedman {
    Friedman { chi2: f64::NAN, df: 0, p: f64::NAN }
}
