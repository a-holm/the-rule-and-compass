//! #589: a device `+inf` (status 2, the f32 program overflowed) is checked on
//! the CPU right after its batch. After every generation, each individual with
//! error `+inf` scores `+inf` on the CPU too, for the squared metric and for
//! `--fitness pnl`. Device tests skip cleanly without CUDA.

#![cfg(feature = "cuda")]

use rcsr::checkpoint::Checkpoint;
use rcsr::data::{Sample, TradingSlice};
use rcsr::expr::{self, Node, Op};
use rcsr::fit::{FitAdmission, FitStrategy};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Fitness};
use rcsr::trading::{FitnessBook, TradingRows, TradingSpec};
use std::sync::Arc;

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

const ROWS: usize = 2_400;

/// No trade floor: a status-0 `+inf` of the floor follows the f32 contract and is
/// not a range failure. `x0` carries the return signal, `x1` is wide enough for `exp(cube(x1))` to
/// overflow f32 while f64 stays finite.
fn fixture() -> (Sample, Sample, Arc<FitnessBook>) {
    let mut rng = Lcg(17);
    let mut rows = TradingRows::default();
    let mut x = Vec::with_capacity(ROWS);
    let mut ret = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let g = i / 600;
        rows.group.push(g as u64);
        rows.time.push(1.6e9 + 900.0 * (i - 600 * g) as f64);
        let signal = rng.unit() - 0.5;
        let wide = if i % 400 < 3 { 4.9 } else { rng.unit() * 6.0 - 5.5 };
        x.push(vec![signal, wide]);
        ret.push(0.004 * signal + 0.003 * (rng.unit() - 0.5));
    }
    let slice = TradingSlice { rows, ret: ret.clone() };
    let spec = TradingSpec { cost_bps: 2.0, min_trades: 0, ..Default::default() };
    let book = FitnessBook::new(&slice, &spec, 900.0).expect("a profitable fixture book");
    let y_sq = x.iter().map(|r| 0.5 * r[0] + 0.1 * r[1].tanh()).collect();
    let squared = Sample { x: x.clone(), y: y_sq, w: Vec::new(), ys: Vec::new(), pairs: None };
    let pnl = Sample { x, y: ret, w: Vec::new(), ys: Vec::new(), pairs: None };
    (squared, pnl, Arc::new(book))
}

fn config(seed: u32, fitness: Fitness) -> Config {
    let mut ops = Op::default_set();
    for op in [Op::Sinh, Op::Cosh, Op::Exp, Op::Cube, Op::Logistic, Op::Ln] {
        if !ops.contains(&op) {
            ops.push(op);
        }
    }
    Config {
        generations: 10,
        islands: 2,
        pop: 64,
        max_complexity: 14,
        fit_iters: 6,
        patience: 0,
        seed,
        ops,
        fit_strategy: FitStrategy::Gpu,
        fit_admission: FitAdmission::All,
        metric: Metric::Squared,
        fitness,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        pipeline_depth: 0,
        ..Default::default()
    }
}

fn cpu_error(tree: &Node, train: &Sample, cfg: &Config) -> f64 {
    match &cfg.fitness {
        Fitness::LongPnl(book) => {
            let pred: Vec<f64> = train.x.iter().map(|r| expr::eval(tree, r)).collect();
            book.regret(&pred)
        }
        _ => search::eval_error(tree, train, cfg.metric),
    }
}

/// Count of `+inf` individuals seen, and of those the CPU scores finite.
fn unchecked_inf(cfg: &Config, train: &Sample) -> (usize, Vec<String>) {
    let names = ["x0".to_string(), "x1".to_string()];
    let mut inf = 0usize;
    let mut wrong = Vec::new();
    search::run_resumable(cfg, train, train, None, 1, |cp: &Checkpoint| {
        for island in &cp.islands {
            for ind in &island.population {
                if ind.error == f64::INFINITY {
                    inf += 1;
                    let exact = cpu_error(&ind.tree, train, cfg);
                    if exact != f64::INFINITY {
                        wrong.push(format!(
                            "generation {}: device +inf, CPU {exact:e}: {}",
                            cp.generations,
                            expr::to_string(&ind.tree, &names)
                        ));
                    }
                }
            }
        }
    })
    .expect("search runs");
    (inf, wrong)
}

fn device_available() -> bool {
    let ok = rcsr::gpu::probe().available();
    if !ok {
        eprintln!("skipping GPU test: no usable CUDA device");
    }
    ok
}

#[test]
fn device_search_keeps_no_unchecked_inf_squared() {
    if !device_available() {
        return;
    }
    let (train, _, _) = fixture();
    let mut all = Vec::new();
    for seed in [11, 12] {
        let (inf, wrong) = unchecked_inf(&config(seed, Fitness::Metric), &train);
        eprintln!("squared seed {seed}: {inf} +inf individuals, {} unchecked", wrong.len());
        all.extend(wrong);
    }
    assert!(all.is_empty(), "{} device +inf the CPU scores finite: {:?}", all.len(), &all[..all.len().min(5)]);
}

#[test]
fn device_search_keeps_no_unchecked_inf_pnl() {
    if !device_available() {
        return;
    }
    let (_, train, book) = fixture();
    let mut all = Vec::new();
    for seed in [11, 12] {
        let (inf, wrong) = unchecked_inf(&config(seed, Fitness::LongPnl(book.clone())), &train);
        eprintln!("pnl seed {seed}: {inf} +inf individuals, {} unchecked", wrong.len());
        all.extend(wrong);
    }
    assert!(all.is_empty(), "{} device +inf the CPU scores finite: {:?}", all.len(), &all[..all.len().min(5)]);
}
