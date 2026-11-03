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
fn dump_squared_archive() {
    let train = sample(2_048, 5);
    let test = sample(256, 6);
    for gens in [1usize, 2, 3, 4] {
        let cfg = Config {
            generations: gens,
            islands: 1,
            pop: 20,
            max_complexity: 10,
            fit_iters: 4,
            patience: 0,
            seed: 11,
            fit_strategy: FitStrategy::Gpu,
            metric: Metric::Squared,
            gpu: true,
            dispatch: rcsr::dispatch::DispatchMode::Device,
            gpu_notes: false,
            ..Default::default()
        };
        let out = search::run(&cfg, &train, &test);
        for s in &out.archive.items {
            println!(
                "gen{} c{} err {:016x} {}",
                gens,
                s.complexity,
                s.error.to_bits(),
                rcsr::expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()])
            );
        }
    }
}
