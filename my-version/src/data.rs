//! Data loading, normalisation and train/test split.

use crate::rng::Mt;

#[derive(Clone, Default)]
pub struct Sample {
    /// x[row][variable]
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl Sample {
    pub fn len(&self) -> usize {
        self.y.len()
    }
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
}

pub struct Table {
    pub names: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

pub fn read_table(path: &str) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| "empty file".to_string())?;
    let delim = if header.contains('\t') {
        '\t'
    } else if header.contains(',') {
        ','
    } else {
        ' '
    };
    let split = |l: &str| -> Vec<String> {
        if delim == ' ' {
            l.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            l.split(delim).map(|s| s.trim().to_string()).collect()
        }
    };
    let names = split(header);
    let mut rows = Vec::new();
    for l in lines {
        let parts = split(l);
        if parts.len() != names.len() {
            continue;
        }
        let vals: Result<Vec<f64>, _> = parts.iter().map(|p| p.parse::<f64>()).collect();
        if let Ok(v) = vals {
            rows.push(v);
        }
    }
    if rows.is_empty() {
        return Err("no numeric rows".to_string());
    }
    Ok(Table { names, rows })
}

fn index_of(names: &[String], want: &str) -> Result<usize, String> {
    names
        .iter()
        .position(|n| n == want)
        .ok_or_else(|| format!("column '{}' not found in {:?}", want, names))
}

/// Build normalised train/test samples. Returns (train, test, means, stds).
pub fn prepare(
    t: &Table,
    vars: &[String],
    target: &str,
    test_frac: f64,
    rng: &mut Mt,
) -> Result<(Sample, Sample, Vec<f64>, Vec<f64>), String> {
    let var_idx: Vec<usize> = vars
        .iter()
        .map(|v| index_of(&t.names, v))
        .collect::<Result<_, _>>()?;
    let tgt_idx = index_of(&t.names, target)?;
    let n = t.rows.len();
    let nv = var_idx.len();

    // All rows -> matrix, taget vector.
    let mut x: Vec<Vec<f64>> = t
        .rows
        .iter()
        .map(|r| var_idx.iter().map(|&i| r[i]).collect())
        .collect();
    let y: Vec<f64> = t.rows.iter().map(|r| r[tgt_idx]).collect();

    // Normalise inputs (z-score).
    let mut means = vec![0.0; nv];
    let mut stds = vec![1.0; nv];
    for j in 0..nv {
        let m = x.iter().map(|r| r[j]).sum::<f64>() / n as f64;
        let var = x.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / n as f64;
        let s = var.sqrt();
        means[j] = m;
        stds[j] = if s < 1e-12 { 1.0 } else { s };
        for r in x.iter_mut() {
            r[j] = (r[j] - m) / stds[j];
        }
    }

    // Deterministic shuffle then split.
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.usize_below(i + 1);
        idx.swap(i, j);
    }
    let test_n = ((n as f64) * test_frac).round() as usize;
    let test_n = test_n.min(n.saturating_sub(1));

    let mut train = Sample::default();
    let mut test = Sample::default();
    for (k, &i) in idx.iter().enumerate() {
        if k < test_n {
            test.x.push(x[i].clone());
            test.y.push(y[i]);
        } else {
            train.x.push(x[i].clone());
            train.y.push(y[i]);
        }
    }
    if train.is_empty() {
        return Err("training split is empty".to_string());
    }
    Ok((train, test, means, stds))
}
