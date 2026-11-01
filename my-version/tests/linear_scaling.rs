//! #470: `--linear-scaling` is opt-in. With the flag off the archive is the
//! pre-change archive bit for bit; with it on, the archive holds wrapped
//! formulas that fit a scale-and-offset target.

use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2);
    }
    Sample::new(x, y)
}

fn archive_hash(out: &Outcome) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for s in &out.archive.items {
        eat(&s.error.to_bits().to_le_bytes());
        eat(&(s.complexity as u64).to_le_bytes());
        eat(rcsr::expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]).as_bytes());
    }
    h
}

fn config(strategy: FitStrategy) -> Config {
    Config {
        generations: 12,
        islands: 2,
        pop: 60,
        max_complexity: 14,
        fit_iters: 8,
        patience: 0,
        seed: 5,
        fit_strategy: strategy,
        metric: Metric::Squared,
        gpu: false,
        gpu_notes: false,
        ..Default::default()
    }
}

/// Archive hash of the pre-change CPU path for `config` on `sample(300, 5)`.
const GOLDEN: &[(FitStrategy, u64)] = &[
    (FitStrategy::Exact, 0xd7f8_c657_7ac8_8687),
    (FitStrategy::Auto, 0x2915_2e01_f9dc_b6a2),
];

#[test]
fn linear_scaling_off_is_bit_identical_cpu() {
    let train = sample(300, 5);
    let test = sample(100, 6);
    let hashes: Vec<u64> = GOLDEN
        .iter()
        .map(|(strategy, _)| archive_hash(&search::run(&config(*strategy), &train, &test)))
        .collect();
    eprintln!("hashes {hashes:x?}");
    for ((strategy, want), h) in GOLDEN.iter().zip(hashes) {
        assert_eq!(h, *want, "{strategy:?}: archive differs from the pre-change path ({h:#018x})");
    }
}

#[test]
fn linear_scaling_fits_a_scale_and_offset_target() {
    let train = sample(300, 5);
    let test = sample(100, 6);
    let mut cfg = config(FitStrategy::Exact);
    cfg.linear_scaling = true;
    let out = search::run(&cfg, &train, &test);
    assert!(!out.archive.items.is_empty());
    for m in &out.archive.items {
        let sse: f64 = train
            .x
            .iter()
            .zip(&train.y)
            .map(|(r, y)| (rcsr::expr::eval(&m.tree, r) - y).powi(2))
            .sum();
        let mse = sse / train.y.len() as f64;
        if m.error.is_finite() {
            assert!((mse - m.error).abs() <= 1e-9 * (1.0 + mse), "archive error {} vs rescored {}", m.error, mse);
        }
    }
}
