#![cfg(feature = "cuda")]
use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        let v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
        y.push(if i % 7 == 3 { 0.0 } else { v });
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

#[test]
fn dump_archives() {
    let train = sample(2_048, 5);
    let test = sample(256, 6);
    for metric in [
        Metric::Squared,
        Metric::Absolute,
        Metric::RankCorrelation,
        Metric::Median,
        Metric::Interquartile,
        Metric::Maximum,
    ] {
        let cfg = Config {
            generations: 4,
            islands: 1,
            pop: 20,
            max_complexity: 10,
            fit_iters: 4,
            patience: 0,
            seed: 11,
            fit_strategy: FitStrategy::Gpu,
            metric,
            gpu: true,
            dispatch: rcsr::dispatch::DispatchMode::Device,
            gpu_notes: false,
            ..Default::default()
        };
        let out = search::run(&cfg, &train, &test);
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
        println!("ARCHIVE {:?} {:#018x} items {}", metric, h, out.archive.items.len());
    }
}
