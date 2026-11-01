//! #481: the device long/flat regret scorer (`ScoreObjective::LongPnl`).
//!
//! The contract is stated on the device's own `f32` predictions: the device
//! totals and regret equal `FitnessBook::totals`/`regret_of` applied to the
//! `rc_predict` output of the same program, bit for bit. Run under
//! `compute-sanitizer --tool memcheck` first; the fault-injection tests run
//! with `--test-threads=1` (#456).

#![cfg(feature = "cuda")]

use rcsr::data::{Sample, TradingSlice};
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::gpu::{GpuEvaluator, Population};
use rcsr::metrics::{Metric, ScoreObjective, ScoreStatus};
use rcsr::search::{self, Config, Fitness};
use rcsr::trading::{FitnessBook, FitnessTotals, TradingRows, TradingSpec};
use std::sync::{Arc, Mutex};

const VARS: usize = 4;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Rows in canonical (group, time) order. Group `g` starts at `starts[g]`;
/// the last group runs to `n`.
struct Fixture {
    n: usize,
    cols: Vec<f32>,
    y: Vec<f32>,
    slice: TradingSlice,
}

fn fixture(starts: &[usize], n: usize, seed: u64) -> Fixture {
    assert_eq!(starts[0], 0);
    assert!(starts.windows(2).all(|w| w[0] < w[1]) && *starts.last().unwrap() < n);
    let mut rng = Lcg(seed);
    let mut rows = TradingRows::default();
    let mut ret = Vec::with_capacity(n);
    let mut cols = vec![0.0f32; n * VARS];
    let mut g = 0usize;
    let mut first = 0usize;
    for i in 0..n {
        if g + 1 < starts.len() && i == starts[g + 1] {
            g += 1;
            first = i;
        }
        rows.group.push(g as u64);
        rows.time.push(1.6e9 + 900.0 * (i - first) as f64);
        let signal = rng.unit() - 0.5;
        for v in 0..VARS {
            let x = if v == 0 { signal } else { rng.unit() * 4.0 - 2.0 };
            cols[v * n + i] = x as f32;
        }
        ret.push(0.004 * signal + 0.003 * (rng.unit() - 0.5));
    }
    let y = ret.iter().map(|&r| r as f32).collect();
    Fixture {
        n,
        cols,
        y,
        slice: TradingSlice { rows, ret },
    }
}

fn spec(min_trades: usize) -> TradingSpec {
    TradingSpec {
        cost_bps: 2.0,
        min_trades,
        ..Default::default()
    }
}

fn book(f: &Fixture, min_trades: usize) -> FitnessBook {
    FitnessBook::new(&f.slice, &spec(min_trades), 900.0).expect("a profitable fixture book")
}

fn evaluator(f: &Fixture, b: &FitnessBook) -> Option<GpuEvaluator> {
    let Some(mut ev) = GpuEvaluator::new(&f.cols, f.n, VARS, &f.y) else {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    };
    ev.set_pnl_book(b).expect("the book uploads");
    Some(ev)
}

fn random_tree(rng: &mut Lcg, depth: usize) -> Node {
    if depth == 0 || rng.below(4) == 0 {
        return if rng.below(3) == 0 {
            constant(((rng.unit() * 8.0).floor() - 4.0) / 2.0 + 0.25)
        } else {
            input(rng.below(VARS))
        };
    }
    const UNARY: [Op; 9] = [Op::Sin, Op::Cos, Op::Tanh, Op::Exp, Op::Ln, Op::Sqrt, Op::Logistic, Op::Square, Op::Neg];
    const BINARY: [Op; 4] = [Op::Add, Op::Sub, Op::Mul, Op::Div];
    if rng.below(3) == 0 {
        node(UNARY[rng.below(UNARY.len())], vec![random_tree(rng, depth - 1)])
    } else {
        let a = random_tree(rng, depth - 1);
        let b = random_tree(rng, depth - 1);
        node(BINARY[rng.below(BINARY.len())], vec![a, b])
    }
}

/// `count` programs: a few hand-written ones (a near-threshold signal, an
/// all-flat and an always-long rule, a NaN on half the rows), then random trees.
fn programs(count: usize, seed: u64) -> Vec<Node> {
    let mut out = vec![
        input(0),
        node(Op::Sub, vec![input(0), constant(0.1)]),
        node(Op::Tanh, vec![node(Op::Mul, vec![constant(1.5), input(1)])]),
        constant(-1.0),
        constant(1.0),
        node(Op::Ln, vec![input(0)]),
    ];
    let mut rng = Lcg(seed);
    while out.len() < count {
        out.push(random_tree(&mut rng, 4));
    }
    out.truncate(count);
    out
}

fn compiled(trees: &[Node]) -> Population {
    let pop = Population::from_trees(trees.iter());
    assert_eq!(pop.len(), trees.len(), "every fixture tree compiles");
    pop
}

/// The device value and totals of every program against the CPU book applied
/// to the device's own predictions.
fn assert_exact(f: &Fixture, b: &FitnessBook, trees: &[Node]) -> Vec<FitnessTotals> {
    let Some(mut ev) = evaluator(f, b) else {
        return Vec::new();
    };
    let pop = compiled(trees);
    let (prog, off) = pop.programs();
    let (batch, totals) = ev.long_pnl_totals_for_tests(prog, off).expect("the totals launch runs");
    let scored = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    assert!(scored.ms >= 0.0 && scored.values.len() == trees.len());
    let mut cpu_totals = Vec::with_capacity(trees.len());
    for k in 0..trees.len() {
        let words = &prog[off[k] as usize..off[k + 1] as usize];
        let pred32 = ev.predict_one_for_tests(words).expect("the device predicts");
        let pred: Vec<f64> = pred32.iter().map(|&p| p as f64).collect();
        let cpu = b.totals(&pred);
        let want = b.regret_of(&cpu);
        let dev = totals[k];
        assert_eq!(dev.flips, cpu.flips, "program {k} flips");
        assert_eq!(dev.exits, cpu.exits, "program {k} exits");
        assert_eq!(dev.long_rows, cpu.long_rows, "program {k} long rows");
        assert_eq!(dev.nonfinite, cpu.nonfinite, "program {k} non-finite");
        assert_eq!(dev.gross.to_bits(), cpu.gross.to_bits(), "program {k} gross {} vs {}", dev.gross, cpu.gross);
        assert_eq!(batch.values[k].to_bits(), want.to_bits(), "program {k} regret {} vs {}", batch.values[k], want);
        assert_eq!(scored.values[k].to_bits(), want.to_bits(), "program {k} score_batch");
        let status = if cpu.nonfinite { ScoreStatus::NonFinite } else { ScoreStatus::Ok };
        assert_eq!(batch.status[k], status, "program {k} status");
        assert_eq!(scored.status[k], status, "program {k} score_batch status");
        cpu_totals.push(cpu);
    }
    cpu_totals
}

/// Group starts on, before and after warp-step (32) and segment (8192)
/// boundaries, one-row groups, and one group (20000..40000) over three
/// segments.
const STARTS_100K: [usize; 37] = [
    0, 1, 2, 31, 32, 33, 100, 8191, 8192, 8193, 8250, 16383, 16384, 16385, 20000, 40000, 40959,
    40960, 40961, 50000, 57343, 57344, 57345, 60000, 65535, 65536, 65537, 70000, 73727, 73728,
    80000, 81919, 81920, 90000, 98303, 98304, 100002,
];

/// The small book for memcheck: two segments and a partial third. The first
/// segment boundary (8192) is inside a group, the second (16384) starts one;
/// one-row groups sit on and around both warp-step and segment boundaries.
const STARTS_SMALL: [usize; 12] = [0, 1, 31, 32, 33, 4000, 9000, 16383, 16384, 16385, 16399, 16400];

#[test]
fn device_totals_equal_the_cpu_tree_on_a_small_book() {
    let f = fixture(&STARTS_SMALL, 16_401, 11);
    let b = book(&f, 1);
    let cpu = assert_exact(&f, &b, &programs(24, 5));
    if cpu.is_empty() {
        return;
    }
    assert!(cpu.iter().any(|t| t.nonfinite), "a program is non-finite somewhere");
    assert!(cpu.iter().any(|t| !t.nonfinite && t.flips > 2 && t.exits > 0), "a program trades");
}

#[test]
fn device_totals_equal_the_cpu_tree_on_device_predictions() {
    let f = fixture(&STARTS_100K, 100_003, 7);
    let b = book(&f, 1);
    assert_eq!(b.groups(), 37);
    let cpu = assert_exact(&f, &b, &programs(64, 9));
    if cpu.is_empty() {
        return;
    }
    let finite = cpu.iter().filter(|t| !t.nonfinite).count();
    assert!(finite >= 32, "{finite} finite programs");
    assert!(cpu.iter().any(|t| !t.nonfinite && t.exits > 0 && t.flips > 100));
}

#[test]
fn non_finite_trade_floor_and_flat_programs_have_exact_values() {
    let f = fixture(&STARTS_SMALL, 16_401, 3);
    let trees = vec![
        node(Op::Ln, vec![input(0)]),
        node(Op::Div, vec![input(1), constant(0.0)]),
        constant(-1.0),
        constant(0.0),
        input(0),
    ];
    let open = book(&f, 0);
    let Some(mut ev) = evaluator(&f, &open) else {
        return;
    };
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let b = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    assert!(b.ms >= 0.0);
    for k in 0..2 {
        assert_eq!(b.values[k], f64::INFINITY, "program {k}");
        assert_eq!(b.status[k], ScoreStatus::NonFinite, "program {k}");
    }
    // All flat at the threshold too: `0 > 0` is false.
    for k in 2..4 {
        assert_eq!(b.values[k].to_bits(), 1.0f64.to_bits(), "program {k} is exactly 1");
        assert_eq!(b.status[k], ScoreStatus::Ok);
    }
    assert!(b.values[4].is_finite() && b.values[4] < 1.0);

    let floor = book(&f, 1_000_000);
    ev.set_pnl_book(&floor).expect("a second book replaces the first");
    let b = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    for k in 2..5 {
        assert_eq!(b.values[k], f64::INFINITY, "program {k} is under the trade floor");
        assert_eq!(b.status[k], ScoreStatus::Ok, "the floor is a value, not a device status");
    }
}

#[test]
fn values_do_not_depend_on_the_batch() {
    let f = fixture(&STARTS_SMALL, 16_401, 21);
    let b = book(&f, 1);
    let Some(mut ev) = evaluator(&f, &b) else {
        return;
    };
    let trees = programs(256, 13);
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let zeros = vec![0; trees.len()];
    let all = ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros);
    assert!(all.ms >= 0.0 && all.values.len() == 256);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for _ in 0..3 {
        let again = ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros);
        assert_eq!(bits(&again.values), bits(&all.values), "a repeated launch");
    }
    let seven = compiled(&trees[..7]);
    let (p7, o7) = seven.programs();
    let b7 = ev.score_batch(p7, o7, ScoreObjective::LongPnl, &[0; 7]);
    assert_eq!(bits(&b7.values), bits(&all.values[..7]), "a batch of 7");
    for k in [0usize, 5, 100, 255] {
        let one = compiled(&trees[k..k + 1]);
        let (p1, o1) = one.programs();
        let b1 = ev.score_batch(p1, o1, ScoreObjective::LongPnl, &[0]);
        assert_eq!(b1.values[0].to_bits(), all.values[k].to_bits(), "program {k} alone");
    }
    // The search's fixed-width pool path addresses the same programs by slot.
    let width = rcsr::genome::width_for(64);
    let mut pool = rcsr::genome::GenomePool::new(trees.len(), width);
    pool.begin_generation();
    for k in 0..trees.len() {
        let words = &prog[off[k] as usize..off[k + 1] as usize];
        assert!(pool.push(words, rcsr::genome::GenomeMeta::default()), "program {k} fits the pool");
    }
    let pooled = ev.score_pool(&pool, ScoreObjective::LongPnl, &zeros);
    assert_eq!(bits(&pooled.values), bits(&all.values), "the pooled path");
}

#[test]
fn a_context_without_a_book_or_with_a_refused_book_is_refused() {
    let f = fixture(&STARTS_SMALL, 16_401, 4);
    let b = book(&f, 1);
    let Some(mut ev) = GpuEvaluator::new(&f.cols, f.n, VARS, &f.y) else {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    };
    let trees = programs(4, 1);
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let none = ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0; 4]);
    assert!(none.ms < 0.0, "no book, no long_pnl launch");
    assert!(none.status.iter().all(|s| *s == ScoreStatus::DeviceError));

    rcsr::enrich_gpu::set_alloc_fault_for_tests(1);
    let refused = ev.set_pnl_book(&b);
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    let why = refused.expect_err("the armed allocation is refused");
    assert!(why.contains("pnl_book") && why.contains("long_pnl fitness book"), "{why}");
    let after = ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0; 4]);
    assert!(after.ms < 0.0, "a refused book leaves no book behind");

    ev.set_pnl_book(&b).expect("the unarmed upload works");
    let ok = ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0; 4]);
    assert!(ok.ms >= 0.0);

    let other = fixture(&STARTS_SMALL[..5], 100, 4);
    let short = FitnessBook::new(&other.slice, &spec(1), 900.0).unwrap();
    assert!(ev.set_pnl_book(&short).unwrap_err().contains("rows"));
}

// ---- search level ----

/// Four groups of 600 rows. `x0` carries the return, `x1` is noise.
fn search_fixture() -> (Sample, Arc<FitnessBook>) {
    let f = fixture(&[0, 600, 1200, 1800], 2_400, 17);
    let x = (0..f.n)
        .map(|i| (0..2).map(|v| f.cols[v * f.n + i] as f64).collect())
        .collect();
    let sample = Sample {
        x,
        y: f.slice.ret.clone(),
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    let b = FitnessBook::new(&f.slice, &spec(1), 900.0).unwrap();
    (sample, Arc::new(b))
}

type Archived = Arc<Mutex<Vec<(Node, f64)>>>;

fn show(tree: &Node) -> String {
    rcsr::expr::to_string(tree, &["x0".to_string(), "x1".to_string()])
}

fn config(b: &Arc<FitnessBook>, seen: &Archived) -> Config {
    let seen = seen.clone();
    Config {
        generations: 8,
        pop: 48,
        islands: 2,
        seed: 5,
        metric: Metric::Squared,
        fitness: Fitness::LongPnl(b.clone()),
        gpu: true,
        gpu_notes: false,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        pipeline_depth: 2,
        front_observer: Some(Arc::new(move |_, _, archive: &rcsr::pareto::Archive| {
            let mut s = seen.lock().unwrap();
            for item in &archive.items {
                s.push((item.tree.clone(), item.error));
            }
        })),
        ..Default::default()
    }
}

/// Every in-run archive value against a fresh device score of its tree.
fn assert_archived_values_are_device_scores(sample: &Sample, b: &FitnessBook, seen: &Archived) {
    let cols: Vec<f32> = (0..2)
        .flat_map(|v| sample.x.iter().map(move |r| r[v] as f32))
        .collect();
    let y: Vec<f32> = sample.y.iter().map(|&v| v as f32).collect();
    let mut ev = GpuEvaluator::new(&cols, sample.len(), 2, &y).expect("a fresh evaluator");
    ev.set_pnl_book(b).unwrap();
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    let mut checked = 0usize;
    for (tree, error) in seen.iter() {
        let pop = Population::from_trees(std::iter::once(tree));
        let want = if pop.len() == 1 {
            let (prog, off) = pop.programs();
            let s = ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0]);
            assert!(s.ms >= 0.0);
            checked += 1;
            if s.status[0].is_ok() {
                s.values[0]
            } else {
                f64::INFINITY
            }
        } else {
            let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(tree, r)).collect();
            b.regret(&pred)
        };
        assert_eq!(error.to_bits(), want.to_bits(), "{} archived {error}, rescored {want}", show(tree));
    }
    assert!(checked > 0);
}

fn assert_device_scored(out: &search::Outcome) {
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{:?}", out.backend_reason);
    let gpu = out.gpu.as_ref().expect("the GPU report");
    assert_eq!(gpu.score_objective, Some(ScoreObjective::LongPnl));
    assert!(gpu.accounting.programs > 0, "the device scored programs");
}

#[test]
fn pipelined_search_archives_device_long_pnl_scores() {
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cfg = config(&b, &seen);
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    assert_archived_values_are_device_scores(&sample, &b, &seen);
    // The reported front is the exact CPU f64 regret.
    for s in &out.archive.items {
        let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(&s.tree, r)).collect();
        assert_eq!(s.error.to_bits(), b.regret(&pred).to_bits(), "{}", show(&s.tree));
    }
}

#[test]
fn pipelined_device_fit_archives_device_long_pnl_scores() {
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cfg = Config {
        fit_strategy: rcsr::fit::FitStrategy::Gpu,
        fit_admission: rcsr::fit::FitAdmission::All,
        ..config(&b, &seen)
    };
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    assert!(out.gpu.as_ref().unwrap().fit_gpu > 0, "the device fitted constants");
    assert_archived_values_are_device_scores(&sample, &b, &seen);
}

#[test]
fn a_refused_book_allocation_falls_back_to_the_cpu_scorer() {
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cols: Vec<f32> = (0..2)
        .flat_map(|v| sample.x.iter().map(move |r| r[v] as f32))
        .collect();
    let y: Vec<f32> = sample.y.iter().map(|&v| v as f32).collect();
    // The device probe allocates once per process; take it before counting.
    let _ = rcsr::dispatch_gpu::device_rates();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(u64::MAX);
    let probe = GpuEvaluator::new(&cols, sample.len(), 2, &y);
    let rows_allocs = rcsr::enrich_gpu::alloc_count_for_tests();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    if probe.is_none() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    drop(probe);

    let cfg = config(&b, &seen);
    rcsr::enrich_gpu::set_alloc_fault_for_tests(rows_allocs + 1);
    let out = search::run(&cfg, &sample, &sample);
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    assert_eq!(out.backend, rcsr::backend::Kind::Cpu);
    let why = out.backend_reason.clone().unwrap_or_default();
    assert!(why.contains("pnl_book") && why.contains("long_pnl fitness book"), "{why}");

    let cpu = search::run(&Config { gpu: false, front_observer: None, ..config(&b, &seen) }, &sample, &sample);
    let key = |o: &search::Outcome| {
        o.archive
            .items
            .iter()
            .map(|s| (show(&s.tree), s.error.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&out), key(&cpu), "the fallback is the CPU search");
}

#[test]
fn tiled_residency_names_its_refusal() {
    assert_eq!(
        rcsr::gpu::LONG_PNL_TILED,
        "long_pnl fitness needs resident training rows (tiled residency: #293)"
    );
    let f = fixture(&STARTS_SMALL, 16_401, 8);
    let b = book(&f, 1);
    let Some(mut ev) = GpuEvaluator::new_tiled(f.cols.clone(), f.n, VARS, &f.y, 4096, None, None) else {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    };
    assert_eq!(ev.set_pnl_book(&b).unwrap_err(), rcsr::gpu::LONG_PNL_TILED);
    let pop = compiled(&programs(2, 1));
    let (prog, off) = pop.programs();
    assert!(ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0, 0]).ms < 0.0);
}

/// The 1,000,000-row, 256-program throughput of the long-P&L batch against
/// the squared batch on the same programs. Numbers go to the issue and
/// `docs/gpu-backend.md`.
#[test]
#[ignore]
fn long_pnl_throughput_against_squared() {
    let n = 1_000_000usize;
    let starts: Vec<usize> = (0..100).map(|g| g * 10_000).collect();
    let f = fixture(&starts, n, 29);
    let b = book(&f, 1);
    let Some(mut ev) = evaluator(&f, &b) else {
        return;
    };
    let mut rng = Lcg(31);
    let trees: Vec<Node> = (0..256)
        .map(|k| {
            // Eight-variable-style trees with tanh and logistic, depth 4.
            let t = random_tree(&mut rng, 4);
            if k % 2 == 0 {
                node(Op::Tanh, vec![t])
            } else {
                node(Op::Logistic, vec![t])
            }
        })
        .collect();
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let zeros = vec![0; trees.len()];
    let time = |ev: &mut GpuEvaluator, o: ScoreObjective| {
        for _ in 0..2 {
            assert!(ev.score_batch(prog, off, o, &zeros).ms >= 0.0);
        }
        let reps = 10;
        let t = std::time::Instant::now();
        let mut kernel_ms = 0.0f64;
        for _ in 0..reps {
            let s = ev.score_batch(prog, off, o, &zeros);
            assert!(s.ms >= 0.0);
            kernel_ms += s.ms as f64;
        }
        let wall = t.elapsed().as_secs_f64() / reps as f64;
        let evals = (n * trees.len()) as f64;
        (evals / wall, evals / (kernel_ms / reps as f64 / 1000.0))
    };
    let (sq_wall, sq_kernel) = time(&mut ev, ScoreObjective::Squared);
    let (pnl_wall, pnl_kernel) = time(&mut ev, ScoreObjective::LongPnl);
    println!(
        "rows {n} programs {} squared {:.3e} row-evals/s (kernel {:.3e}) long_pnl {:.3e} row-evals/s (kernel {:.3e}) ratio {:.3} (kernel {:.3})",
        trees.len(),
        sq_wall,
        sq_kernel,
        pnl_wall,
        pnl_kernel,
        pnl_wall / sq_wall,
        pnl_kernel / sq_kernel
    );
    assert!(pnl_wall >= 0.75 * sq_wall, "long_pnl {pnl_wall:.3e} vs squared {sq_wall:.3e}");
}

// ---- CLI ----

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// `r = 0.01 x1 + noise` is the realised return; `y = r + 0.05 x2` the label.
fn write_cli_fixture() -> (String, String) {
    let dir = std::env::temp_dir().join(format!("rcsr-pnl-fitness-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..600 {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", 1_600_000_000 + 900 * i, x1, x2, r, r + 0.05 * x2));
    }
    let data = dir.join("d.csv");
    std::fs::write(&data, text).unwrap();
    let spec = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":900,\"delay_seconds\":0,\
         \"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let spec_path = dir.join("s.json");
    std::fs::write(&spec_path, spec).unwrap();
    (data.to_string_lossy().into_owned(), spec_path.to_string_lossy().into_owned())
}

fn cli(data: &str, spec: &str, backend: &str) -> rcsr::devlog::Json {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", spec, backend, "--no-dev-log", "--seed", "3", "--generations", "10",
            "--pop", "48", "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
            "--cost-bps", "5", "--validation-frac", "0.3", "--fitness", "pnl", "--dispatch", "device",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    rcsr::devlog::Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
}

#[test]
fn cli_names_the_device_scorer_and_reports_exact_values() {
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (data, spec) = write_cli_fixture();
    let gpu = cli(&data, &spec, "--gpu");
    assert_eq!(gpu.get("backend").and_then(|j| j.as_str()), Some("gpu"));
    let f = gpu.get("search_fitness").expect("search_fitness");
    let field = |k: &str| f.get(k).and_then(|j| j.as_str()).map(str::to_string);
    assert_eq!(field("kind").as_deref(), Some("long_pnl_regret"));
    assert_eq!(field("backend").as_deref(), Some("cpu_f64"));
    assert_eq!(field("scorer").as_deref(), Some("long_pnl_regret"));
    assert_eq!(field("scorer_backend").as_deref(), Some("cuda"));
    assert_eq!(field("scorer_values").as_deref(), Some("screening_f32_predictions"));

    let cpu = cli(&data, &spec, "--cpu");
    let f = cpu.get("search_fitness").expect("search_fitness");
    assert_eq!(f.get("scorer_backend").and_then(|j| j.as_str()), Some("cpu"));
    assert_eq!(f.get("scorer_values").and_then(|j| j.as_str()), Some("exact_f64"));
}
