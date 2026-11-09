//! #481: the device long/flat regret scorer (`ScoreObjective::LongPnl`).
//!
//! The contract is stated on the device's own `f32` predictions: the device
//! totals and regret equal `FitnessBook::totals`/`regret_of` applied to the
//! `rc_predict` output of the same program, bit for bit. Run under
//! `compute-sanitizer --tool memcheck` first; the fault-injection tests run
//! with `--test-threads=1` (#456).

#![cfg(feature = "cuda")]

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::data::{Sample, TradingSlice};
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::gpu::{GpuEvaluator, Population};
use rcsr::metrics::{Metric, ScoreObjective, ScoreStatus};
use rcsr::search::{self, Config, Fitness};
use rcsr::trading::{FitnessBook, FitnessTotals, TradingRows, TradingSpec};
use std::sync::{Arc, Mutex};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

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
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_SMALL, 16_401, 4);
    let b = book(&f, 1);
    let Some(mut ev) = GpuEvaluator::new(&f.cols, f.n, VARS, &f.y) else {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
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

// ---- #484: time blocks ----

fn book_blocks(f: &Fixture, min_trades: usize, blocks: usize, lambda: f64) -> FitnessBook {
    FitnessBook::with_blocks(&f.slice, &spec(min_trades), 900.0, blocks, lambda).expect("a profitable blocked book")
}

/// The device value and per-block totals of every program against the CPU
/// block tree applied to the device's own predictions.
fn assert_exact_blocks(f: &Fixture, b: &FitnessBook, trees: &[Node]) -> usize {
    let Some(mut ev) = evaluator(f, b) else {
        return 0;
    };
    let pop = compiled(trees);
    let (prog, off) = pop.programs();
    let (batch, blocks) = ev
        .long_pnl_block_totals_for_tests(prog, off, b.blocks())
        .expect("the totals launch runs");
    let scored = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    assert!(scored.ms >= 0.0 && scored.values.len() == trees.len());
    let mut finite = 0usize;
    for k in 0..trees.len() {
        let words = &prog[off[k] as usize..off[k + 1] as usize];
        let pred: Vec<f64> = ev
            .predict_one_for_tests(words)
            .expect("the device predicts")
            .iter()
            .map(|&p| p as f64)
            .collect();
        let cpu = b.block_totals(&pred);
        let want = b.fitness_of(&cpu);
        assert_eq!(blocks[k].len(), cpu.len());
        for (j, (d, c)) in blocks[k].iter().zip(&cpu).enumerate() {
            assert_eq!(d.flips, c.flips, "program {k} block {j} flips");
            assert_eq!(d.exits, c.exits, "program {k} block {j} exits");
            assert_eq!(d.nonfinite, c.nonfinite, "program {k} block {j} non-finite");
            assert_eq!(d.gross.to_bits(), c.gross.to_bits(), "program {k} block {j} gross {} vs {}", d.gross, c.gross);
        }
        assert_eq!(batch.values[k].to_bits(), want.to_bits(), "program {k} fitness {} vs {}", batch.values[k], want);
        assert_eq!(scored.values[k].to_bits(), want.to_bits(), "program {k} score_batch");
        assert_eq!(want.to_bits(), b.regret(&pred).to_bits());
        let bad = cpu.iter().any(|t| t.nonfinite);
        let status = if bad { ScoreStatus::NonFinite } else { ScoreStatus::Ok };
        assert_eq!(batch.status[k], status, "program {k} status");
        if want.is_finite() {
            finite += 1;
        }
    }
    finite
}

#[test]
fn device_block_totals_equal_the_cpu_tree_on_a_small_book() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_SMALL, 16_401, 11);
    for blocks in [2usize, 5, 8] {
        let b = book_blocks(&f, 1, blocks, 1.5);
        assert_eq!(b.blocks(), blocks);
        let finite = assert_exact_blocks(&f, &b, &programs(24, 5));
        assert!(finite == 0 || finite >= 8, "{finite} finite programs at {blocks} blocks");
    }
}

/// Group 0 and group 1 each have 16384 rows, so with two blocks the cut falls
/// on the 8192-row segment boundary inside both groups, and a one-row group
/// follows.
#[test]
fn device_block_totals_equal_the_cpu_tree_when_a_cut_meets_a_segment_boundary() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&[0, 16_384, 32_768], 32_769, 19);
    for (blocks, lambda) in [(2usize, 1.0), (4, 0.0), (3, 2.5)] {
        let b = book_blocks(&f, 1, blocks, lambda);
        let finite = assert_exact_blocks(&f, &b, &programs(24, 7));
        assert!(finite == 0 || finite >= 8, "{finite} finite programs at {blocks} blocks");
    }
}

#[test]
fn device_block_totals_equal_the_cpu_tree_on_device_predictions() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_100K, 100_003, 7);
    for blocks in [2usize, 5, 8] {
        let b = book_blocks(&f, 1, blocks, 1.0);
        let finite = assert_exact_blocks(&f, &b, &programs(64, 9));
        assert!(finite == 0 || finite >= 32, "{finite} finite programs at {blocks} blocks");
    }
}

#[test]
fn block_trade_floor_and_flat_programs_have_exact_values() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_SMALL, 16_401, 3);
    let trees = vec![constant(-1.0), constant(0.0), input(0), node(Op::Ln, vec![input(0)])];
    let open = book_blocks(&f, 0, 4, 1.0);
    let Some(mut ev) = evaluator(&f, &open) else {
        return;
    };
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let b = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    // Every block is flat: regret 1 each, sd 0, so the fitness is exactly 1.
    for k in 0..2 {
        assert_eq!(b.values[k].to_bits(), 1.0f64.to_bits(), "program {k}");
    }
    assert!(b.values[2].is_finite() && b.values[2] < 1.0);
    assert_eq!(b.values[3], f64::INFINITY);
    assert_eq!(b.status[3], ScoreStatus::NonFinite);

    let floor = book_blocks(&f, 1_000_000, 4, 1.0);
    ev.set_pnl_book(&floor).expect("a second book replaces the first");
    let b = ev.score_batch(prog, off, ScoreObjective::LongPnl, &vec![0; trees.len()]);
    for k in 0..3 {
        assert_eq!(b.values[k], f64::INFINITY, "program {k} is under the trade floor");
    }
}

// ---- search level ----

/// Four groups of 600 rows. `x0` carries the return, `x1` is noise.
fn search_fixture() -> (Sample, Arc<FitnessBook>) {
    search_fixture_blocks(1)
}

fn search_fixture_blocks(blocks: usize) -> (Sample, Arc<FitnessBook>) {
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
    let b = FitnessBook::with_blocks(&f.slice, &spec(1), 900.0, blocks, 1.0).unwrap();
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
                // #589: a status-2 +inf is replaced by the exact CPU score.
                let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(tree, r)).collect();
                b.regret(&pred)
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cfg = config(&b, &seen);
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cfg = Config {
        fit_strategy: rcsr::fit::FitStrategy::Gpu,
        fit_admission: rcsr::fit::FitAdmission::All,
        ..config(&b, &seen)
    };
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    assert!(out.gpu.as_ref().unwrap().fit_gpu > 0, "the device fitted constants");
    assert_archived_values_are_device_scores(&sample, &b, &seen);
}

/// The barrier path: screen, fit the admitted candidates on the device with
/// the squared LM, rescore the fitted trees.
#[test]
fn synchronous_device_fit_archives_device_long_pnl_scores() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let (sample, b) = search_fixture();
    let seen: Archived = Arc::default();
    let cfg = Config {
        fit_strategy: rcsr::fit::FitStrategy::Gpu,
        pipeline_depth: 0,
        ..config(&b, &seen)
    };
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    let gpu = out.gpu.as_ref().unwrap();
    assert!(gpu.fit_gpu > 0 && gpu.fit_screened > 0, "screened and fitted on the device");
    assert_archived_values_are_device_scores(&sample, &b, &seen);
}

#[test]
fn a_refused_book_allocation_falls_back_to_the_cpu_scorer() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
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
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        rcsr::gpu::LONG_PNL_TILED,
        "long_pnl fitness needs resident training rows (tiled residency: #293)"
    );
    let f = fixture(&STARTS_SMALL, 16_401, 8);
    let b = book(&f, 1);
    let Some(mut ev) = GpuEvaluator::new_tiled(f.cols.clone(), f.n, VARS, &f.y, 4096, None, None) else {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    };
    assert_eq!(ev.set_pnl_book(&b).unwrap_err(), rcsr::gpu::LONG_PNL_TILED);
    let pop = compiled(&programs(2, 1));
    let (prog, off) = pop.programs();
    assert!(ev.score_batch(prog, off, ScoreObjective::LongPnl, &[0, 0]).ms < 0.0);
}

#[test]
fn blocked_pipelined_search_archives_device_scores() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let (sample, b) = search_fixture_blocks(3);
    assert_eq!(b.blocks(), 3);
    let seen: Archived = Arc::default();
    let cfg = config(&b, &seen);
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    assert_archived_values_are_device_scores(&sample, &b, &seen);
    for s in &out.archive.items {
        let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(&s.tree, r)).collect();
        assert_eq!(s.error.to_bits(), b.regret(&pred).to_bits(), "{}", show(&s.tree));
    }
}

#[test]
fn blocked_synchronous_device_fit_archives_device_scores() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let (sample, b) = search_fixture_blocks(4);
    let seen: Archived = Arc::default();
    let cfg = Config {
        fit_strategy: rcsr::fit::FitStrategy::Gpu,
        pipeline_depth: 0,
        ..config(&b, &seen)
    };
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let out = search::run(&cfg, &sample, &sample);
    assert_device_scored(&out);
    assert!(out.gpu.as_ref().unwrap().fit_gpu > 0, "the device fitted constants");
    assert_archived_values_are_device_scores(&sample, &b, &seen);
}

/// Throughput of the blocked kernel against the unblocked one on the same
/// 1,000,000 rows and programs.
#[test]
#[ignore]
fn long_pnl_block_throughput_against_one_block() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let n = 1_000_000usize;
    let starts: Vec<usize> = (0..100).map(|g| g * 10_000).collect();
    let f = fixture(&starts, n, 29);
    let Some(mut ev) = evaluator(&f, &book(&f, 1)) else {
        return;
    };
    let mut rng = Lcg(31);
    let trees: Vec<Node> = (0..256)
        .map(|k| {
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
    let mut rates = Vec::new();
    for blocks in [1usize, 2, 5, 8, 1] {
        ev.set_pnl_book(&book_blocks(&f, 1, blocks, 1.0)).expect("the book uploads");
        for _ in 0..2 {
            assert!(ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros).ms >= 0.0);
        }
        let reps = 10;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            assert!(ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros).ms >= 0.0);
        }
        let rate = (n * trees.len()) as f64 / (t.elapsed().as_secs_f64() / reps as f64);
        println!("blocks {blocks} {rate:.3e} row-evals/s");
        rates.push(rate);
    }
    let base = rates[0].max(rates[4]);
    println!("ratios to one block: 2 {:.3} 5 {:.3} 8 {:.3}", rates[1] / base, rates[2] / base, rates[3] / base);
}

/// The 1,000,000-row, 256-program throughput of the long-P&L batch against
/// the squared batch on the same programs. Numbers go to the issue and
/// `docs/gpu-backend.md`.
#[test]
#[ignore]
fn long_pnl_throughput_against_squared() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
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
}

// ---- CLI ----

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// `r = 0.01 x1 + noise` is the realised return; `y = r + 0.05 x2` the label.
fn write_cli_fixture() -> (Scratch, String, String) {
    let scratch = Scratch::new("pnl-fitness-gpu");
    let dir = scratch.path().to_path_buf();
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
    (scratch, data.to_string_lossy().into_owned(), spec_path.to_string_lossy().into_owned())
}

fn cli(data: &str, spec: &str, backend: &[&str]) -> rcsr::devlog::Json {
    let out = rcsr_cmd::rcsr_for(backend)
        .args(backend)
        .args([
            "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", spec, "--no-dev-log", "--seed", "3", "--generations", "10",
            "--pop", "48", "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
            "--cost-bps", "5", "--validation-frac", "0.3", "--fitness", "pnl",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    rcsr::devlog::Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
}

#[test]
fn cli_names_the_device_scorer_and_reports_exact_values() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let (_scratch, data, spec) = write_cli_fixture();
    let gpu = cli(&data, &spec, &["--gpu", "--dispatch", "device"]);
    assert_eq!(gpu.get("backend").and_then(|j| j.as_str()), Some("gpu"));
    let f = gpu.get("search_fitness").expect("search_fitness");
    let field = |k: &str| f.get(k).and_then(|j| j.as_str()).map(str::to_string);
    assert_eq!(field("kind").as_deref(), Some("long_pnl_regret"));
    assert_eq!(field("backend").as_deref(), Some("cpu_f64"));
    assert_eq!(field("scorer").as_deref(), Some("long_pnl_regret"));
    assert_eq!(field("scorer_backend").as_deref(), Some("cuda"));
    assert_eq!(field("scorer_values").as_deref(), Some("screening_f32_predictions"));

    let cpu = cli(&data, &spec, &["--cpu"]);
    let f = cpu.get("search_fitness").expect("search_fitness");
    assert_eq!(f.get("scorer_backend").and_then(|j| j.as_str()), Some("cpu"));
    assert_eq!(f.get("scorer_values").and_then(|j| j.as_str()), Some("exact_f64"));
}

/// #848: the same check as the CPU test, with the device ranking the candidates.
#[test]
fn reject_fixed_position_runs_on_the_device() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let (scratch, data, spec) = write_cli_fixture();
    let log = scratch.path().join("fixed.jsonl");
    let out = rcsr_cmd::rcsr_on_device()
        .args([
            "--gpu", "--dispatch", "device", "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t",
            "--split", "temporal", "--target-spec", &spec, "--seed", "11", "--generations", "20", "--pop", "48",
            "--ops", "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss", "--metric", "squared", "--json",
            "--quiet", "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--fitness", "pnl",
            "--reject-fixed-position", "--dev-log", &log.to_string_lossy(), "--dev-log-level", "full",
            "--dev-log-every", "1",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let run = rcsr::devlog::Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"));
    assert_eq!(run.get("backend").and_then(|j| j.as_str()), Some("gpu"));
    let rejected: f64 = std::fs::read_to_string(&log)
        .expect("dev log")
        .lines()
        .filter_map(rcsr::devlog::Json::parse)
        .filter(|e| e.get("event").and_then(|j| j.as_str()) == Some("generation"))
        .filter_map(|e| e.get("variation")?.get("fixed_position_rejected")?.as_f64())
        .sum();
    assert!(rejected > 0.0, "the filter rejected nothing on the device route");
    let vars = ["x1".to_string(), "x2".to_string()];
    let Some(rcsr::devlog::Json::Arr(front)) = run.get("pareto") else { panic!("no pareto") };
    assert!(!front.is_empty());
    for member in front {
        let exact = member.get("expression_exact").and_then(|j| j.as_str()).unwrap();
        let tree = rcsr::parse::parse(exact, &vars).unwrap_or_else(|e| panic!("{exact}: {e:?}"));
        assert!(rcsr::expr::fixed_position(&tree, 0.0).is_none(), "fixed member on the front: {exact}");
    }
}

// ---- #604: an explicit --gpu never completes on the CPU ----

/// Tests that run a device search hold the read side; a test that arms a fault holds the write side.
static FAULTS: std::sync::RwLock<()> = std::sync::RwLock::new(());

struct Faults(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

impl Faults {
    fn arm(sites: &str) -> Self {
        let guard = FAULTS.write().unwrap_or_else(|e| e.into_inner());
        rcsr::gpu::set_injected_faults_for_tests(sites);
        Faults(guard)
    }
}

impl Drop for Faults {
    fn drop(&mut self) {
        rcsr::gpu::set_injected_faults_for_tests("");
    }
}

fn device_present() -> bool {
    let present = GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_some();
    if !present {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
    }
    present
}

fn required_config(b: &Arc<FitnessBook>, required: bool) -> Config {
    Config {
        gpu_required: required,
        dispatch: if required {
            rcsr::dispatch::DispatchMode::Device
        } else {
            rcsr::dispatch::DispatchMode::Auto
        },
        ..config(b, &Archived::default())
    }
}

fn assert_fails_closed(site: &str, needle: &str) {
    if !device_present() {
        return;
    }
    let (sample, b) = search_fixture();
    let _faults = Faults::arm(site);
    let err = search::run_seeded_checked(&required_config(&b, true), &sample, &sample, &[])
        .err()
        .unwrap_or_else(|| panic!("{site}: a required GPU run must fail"));
    assert!(err.starts_with(search::GPU_REQUIRED_PREFIX), "{site}: {err}");
    assert!(err.contains(needle), "{site}: {err}");
}

fn assert_auto_falls_back(site: &str, needle: &str, cpu_backend: bool) {
    if !device_present() {
        return;
    }
    let (sample, b) = search_fixture();
    let _faults = Faults::arm(site);
    let out = search::run_seeded_checked(&required_config(&b, false), &sample, &sample, &[])
        .unwrap_or_else(|e| panic!("{site}: auto mode must fall back: {e}"));
    if cpu_backend {
        assert_eq!(out.backend, rcsr::backend::Kind::Cpu, "{site}");
    }
    let why = out.backend_reason.expect("the fallback is labelled");
    assert!(why.contains(needle), "{site}: {why}");
    assert!(!out.archive.items.is_empty(), "{site}: the CPU run finished");
}

#[test]
fn a_refused_book_fails_a_required_gpu_run() {
    assert_fails_closed("pnl_book", "long_pnl fitness book");
}

#[test]
fn a_refused_book_falls_back_labelled_in_auto_mode() {
    assert_auto_falls_back("pnl_book", "long_pnl fitness book", true);
}

#[test]
fn a_failed_batch_fails_a_required_gpu_run() {
    assert_fails_closed("batch", "generation batch");
}

#[test]
fn a_failed_batch_falls_back_labelled_in_auto_mode() {
    assert_auto_falls_back("batch", "generation batch", false);
}

#[test]
fn a_failed_pipeline_stage_fails_a_required_gpu_run() {
    assert_fails_closed("pipeline", "pipeline batch");
}

#[test]
fn a_failed_pipeline_stage_falls_back_labelled_in_auto_mode() {
    assert_auto_falls_back("pipeline", "pipeline batch", false);
}

#[test]
fn every_device_side_book_refusal_names_its_cause() {
    let f = fixture(&STARTS_SMALL, 16_401, 4);
    let b = book(&f, 1);
    let Some(mut ev) = GpuEvaluator::new(&f.cols, f.n, VARS, &f.y) else {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    };
    let _faults = Faults::arm("");
    let bound = b.block_bounds()[0];
    let nine = [bound; 9];
    let zero = [0.0];
    let cases: [(&str, Result<(), String>); 6] = [
        (
            "the row count differs",
            ev.upload_pnl_book(&b.excess()[..100], &b.flags()[..100], b.side(), b.threshold(), &[bound], 1.0, 1),
        ),
        (
            "a book array is null",
            ev.upload_pnl_book(b.excess(), b.flags(), b.side(), b.threshold(), &[], 1.0, 1),
        ),
        (
            "the block count is outside 1 to 8",
            ev.upload_pnl_book(b.excess(), b.flags(), b.side(), b.threshold(), &nine, 1.0, 1),
        ),
        (
            "the block spread weight",
            ev.upload_pnl_book(b.excess(), b.flags(), b.side(), b.threshold(), &[bound], -1.0, 1),
        ),
        (
            "the block spread weight",
            ev.upload_pnl_book(b.excess(), b.flags(), b.side(), b.threshold(), &[bound], f64::NAN, 1),
        ),
        (
            "a block hindsight bound is not positive",
            ev.upload_pnl_book(b.excess(), b.flags(), b.side(), b.threshold(), &zero, 1.0, 1),
        ),
    ];
    for (needle, result) in cases {
        let why = result.expect_err(needle);
        assert!(why.starts_with("the device refused the long_pnl fitness book: "), "{why}");
        assert!(why.contains(needle), "want '{needle}' in: {why}");
    }
    ev.set_pnl_book(&b).expect("a valid book still uploads after the refusals");
}

// ---- #925: the band and the minimum hold ----

use rcsr::trading::{BandSpec, BandUnits};

const BAND_PAIRS: [(f64, usize); 3] = [(0.3, 1), (0.0, 8), (0.5, 24)];

fn band_spec(min_trades: usize, band: f64, units: BandUnits, hold: usize) -> TradingSpec {
    TradingSpec {
        band: BandSpec { units, resolved: if units == BandUnits::Score { band } else { 0.0 }, ..BandSpec::score(band) },
        min_holding: hold,
        ..spec(min_trades)
    }
}

fn band_book(f: &Fixture, band: f64, units: BandUnits, hold: usize, blocks: usize) -> FitnessBook {
    FitnessBook::with_blocks(&f.slice, &band_spec(1, band, units, hold), 900.0, blocks, 1.0).expect("a profitable band book")
}

/// Persistent near-threshold programs, so the band and the hold change the
/// positions, followed by the random programs.
fn band_programs(count: usize, seed: u64) -> Vec<Node> {
    let mut out = vec![
        node(Op::Mul, vec![constant(0.75), input(0)]),
        node(Op::Sub, vec![node(Op::Tanh, vec![input(1)]), constant(0.05)]),
        node(Op::Mul, vec![constant(0.3), node(Op::Add, vec![input(0), input(2)])]),
    ];
    out.extend(programs(count.saturating_sub(3), seed));
    out.truncate(count);
    out
}

/// Every (B, K) pair, score and sd units, one and three blocks: device
/// totals and values against the CPU book on the device predictions.
fn assert_band_books(f: &Fixture, trees: &[Node], pairs: &[(f64, usize, BandUnits)], blocks: &[usize]) -> usize {
    let mut finite = 0usize;
    for &(band, hold, units) in pairs {
        for &nb in blocks {
            let b = band_book(f, band, units, hold, nb);
            assert!(!b.is_plain());
            let n = if nb == 1 {
                let cpu = assert_exact(f, &b, trees);
                if cpu.is_empty() {
                    return 0;
                }
                cpu.iter().filter(|t| !t.nonfinite && t.flips > 0).count()
            } else {
                assert_exact_blocks(f, &b, trees)
            };
            assert!(n > 0, "band {band} hold {hold} {units:?} blocks {nb}: no finite trading program");
            finite += n;
        }
    }
    finite
}

/// The memcheck book: two segments and a partial third (one round of eight
/// warps), the three acceptance pairs, an sd-unit band, a 32-row hold (the
/// 33rd automaton state) and three time blocks.
#[test]
fn device_band_totals_equal_the_cpu_on_a_small_book() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_SMALL, 16_401, 11);
    let mut pairs: Vec<(f64, usize, BandUnits)> = BAND_PAIRS.iter().map(|&(b, k)| (b, k, BandUnits::Score)).collect();
    pairs.push((1.0, 12, BandUnits::Sd));
    pairs.push((0.2, 32, BandUnits::Score));
    assert_band_books(&f, &band_programs(16, 5), &pairs, &[1, 3]);
}

/// Thirteen segments: two rounds, segment seams inside groups and on group
/// starts, and one group over three segments.
#[test]
fn device_band_totals_equal_the_cpu_over_two_rounds() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_100K, 100_003, 7);
    let pairs = [(0.3, 1, BandUnits::Score), (0.5, 24, BandUnits::Score), (0.5, 3, BandUnits::Sd)];
    assert_band_books(&f, &band_programs(8, 9), &pairs, &[1, 2]);
}

#[test]
fn device_band_totals_equal_the_cpu_at_a_million_rows() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let starts: Vec<usize> = (0..100).map(|g| g * 10_000).collect();
    let f = fixture(&starts, 1_000_000, 23);
    let pairs: Vec<(f64, usize, BandUnits)> = BAND_PAIRS.iter().map(|&(b, k)| (b, k, BandUnits::Score)).collect();
    let n = assert_band_books(&f, &band_programs(24, 13), &pairs, &[1]);
    assert!(n == 0 || n >= 3 * 8, "{n} trading programs");
}

/// The #925 training-score sd: the mean, then the centred sum of squares,
/// each summed per lane, lane tree and warp tree in the device order.
fn two_pass_tree_sd(x: &[f64]) -> Option<f64> {
    use rcsr::trading::{FITNESS_LANES, FITNESS_SEGMENT_ROWS, FITNESS_WARPS};
    fn fold(v: &mut [f64]) {
        let mut half = v.len() / 2;
        while half >= 1 {
            for i in 0..half {
                v[i] += v[i + half];
            }
            half /= 2;
        }
    }
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let (lo, hi) = finite.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    if finite.len() < 2 || lo == hi {
        return None;
    }
    let tree = |term: &dyn Fn(f64) -> f64| {
        let mut warps = [0.0f64; FITNESS_WARPS];
        for (seg, chunk) in x.chunks(FITNESS_SEGMENT_ROWS).enumerate() {
            let mut lanes = [0.0f64; FITNESS_LANES];
            for (k, &v) in chunk.iter().enumerate() {
                if v.is_finite() {
                    lanes[k % FITNESS_LANES] += term(v);
                }
            }
            fold(&mut lanes);
            warps[seg % FITNESS_WARPS] += lanes[0];
        }
        fold(&mut warps);
        warps[0]
    };
    let mean = tree(&|v| v) / finite.len() as f64;
    let sd = (tree(&|v| (v - mean) * (v - mean)) / (finite.len() - 1) as f64).sqrt();
    (sd.is_finite() && sd > 0.0).then_some(sd)
}

/// #925, #945: the device sd, half-width and count equal `trading::score_sd`
/// bit for bit, and the one-pass sd is within 1e-12 relative of the #925
/// two-pass sd on the same predictions.
#[test]
fn band_sd_device_matches_cpu() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let starts: Vec<usize> = (0..100).map(|g| g * 10_000).collect();
    let f = fixture(&starts, 1_000_000, 37);
    let b = band_book(&f, 1.0, BandUnits::Sd, 12, 1);
    let Some(mut ev) = evaluator(&f, &b) else {
        return;
    };
    let mut trees = band_programs(24, 21);
    trees.push(constant(0.5));
    let pop = compiled(&trees);
    let (prog, off) = pop.programs();
    let (batch, totals) = ev.long_pnl_totals_for_tests(prog, off).expect("the totals launch runs");
    let sds = ev.long_pnl_band_sd_for_tests(trees.len()).expect("the sd download");
    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for k in 0..trees.len() {
        let words = &prog[off[k] as usize..off[k + 1] as usize];
        let pred: Vec<f64> = ev.predict_one_for_tests(words).expect("the device predicts").iter().map(|&p| p as f64).collect();
        let cpu = rcsr::trading::score_sd(&pred);
        let (dev_sd, dev_band, dev_rows) = sds[k];
        assert_eq!(dev_rows as usize, cpu.rows, "program {k} rows");
        assert_eq!(cpu.sd.is_some(), two_pass_tree_sd(&pred).is_some(), "program {k} degenerate");
        match cpu.sd {
            Some(sd) => {
                assert_eq!(dev_sd.to_bits(), sd.to_bits(), "program {k} sd {dev_sd} vs {sd}");
                assert_eq!(dev_band.to_bits(), (1.0 * sd).to_bits(), "program {k} band");
                let two = two_pass_tree_sd(&pred).unwrap();
                let rel = (sd - two).abs() / two;
                assert!(rel <= 1e-12, "program {k}: one-pass {sd} against two-pass {two}: {rel:e}");
                worst = worst.max(rel);
                checked += 1;
            }
            None => {
                assert!(dev_sd.is_nan(), "program {k}: degenerate sd {dev_sd}");
                assert_eq!(dev_band, 0.0);
            }
        }
        assert_eq!(dev_band.to_bits(), b.resolve_band(&pred).to_bits(), "program {k} resolved band");
        let t = b.totals(&pred);
        assert_eq!((totals[k].flips, totals[k].exits, totals[k].long_rows), (t.flips, t.exits, t.long_rows), "program {k}");
        assert_eq!(totals[k].gross.to_bits(), t.gross.to_bits(), "program {k} gross");
        assert_eq!(batch.values[k].to_bits(), b.regret_of(&t).to_bits(), "program {k} regret");
    }
    assert!(checked >= 16, "{checked} programs with a finite sd");
    println!("one-pass against two-pass sd: largest relative difference {worst:e} over {checked} programs");
    let (_, band, rows) = sds[trees.len() - 1];
    assert_eq!((band, rows), (0.0, 1_000_000), "the constant program is degenerate");
}

#[test]
fn a_hold_above_32_is_refused_by_the_device() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let f = fixture(&STARTS_SMALL, 16_401, 3);
    let b = band_book(&f, 0.1, BandUnits::Score, 33, 1);
    let Some(mut ev) = GpuEvaluator::new(&f.cols, f.n, VARS, &f.y) else {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    };
    let why = ev.set_pnl_book(&b).unwrap_err();
    assert!(why.contains("at most 32"), "{why}");
    assert!(ev.set_pnl_band(0.1, false, 32).is_ok());
    assert!(ev.set_pnl_band(-0.1, false, 2).unwrap_err().contains("band"));
}

fn band_search_fixture(band: f64, units: BandUnits, hold: usize) -> (Sample, Arc<FitnessBook>) {
    let (sample, _) = search_fixture();
    let f = fixture(&[0, 600, 1200, 1800], 2_400, 17);
    (sample, Arc::new(band_book(&f, band, units, hold, 1)))
}

/// Every in-run archive value of a device band search against an independent
/// reference: the CPU band book applied to the member's `f32` predictions from
/// a fresh device interpreter (`rc_predict`, no device book). A slot the search
/// settles on the CPU takes the exact `f64` regret: a tree without a device
/// program, a non-finite `f32` prediction (#589) and a device `+inf` trade
/// floor (#640). The plain book on the same predictions must differ for some
/// member, so a device path that drops the band cannot pass.
fn assert_archived_values_are_the_cpu_band_book(sample: &Sample, b: &FitnessBook, plain: &FitnessBook, seen: &Archived) {
    let cols: Vec<f32> = (0..2)
        .flat_map(|v| sample.x.iter().map(move |r| r[v] as f32))
        .collect();
    let y: Vec<f32> = sample.y.iter().map(|&v| v as f32).collect();
    let mut ev = GpuEvaluator::new(&cols, sample.len(), 2, &y).expect("a fresh evaluator");
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    let (mut on_device, mut band_matters) = (0usize, 0usize);
    for (tree, error) in seen.iter() {
        let pred64: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(tree, r)).collect();
        let exact = b.regret(&pred64);
        let pop = Population::from_trees(std::iter::once(tree));
        let want = if pop.len() == 1 {
            let (prog, off) = pop.programs();
            let words = &prog[off[0] as usize..off[1] as usize];
            let pred32: Vec<f64> = ev.predict_one_for_tests(words).expect("the device predicts").iter().map(|&p| p as f64).collect();
            let device = b.regret(&pred32);
            if pred32.iter().any(|p| !p.is_finite()) || device.is_infinite() {
                exact
            } else {
                on_device += 1;
                if plain.regret(&pred32).to_bits() != device.to_bits() {
                    band_matters += 1;
                }
                device
            }
        } else {
            exact
        };
        assert_eq!(error.to_bits(), want.to_bits(), "{} archived {error}, CPU band book {want}", show(tree));
    }
    assert!(on_device > 0, "no archived value came from the device");
    assert!(band_matters > 0, "the band changes no archived value");
}

#[test]
fn band_pipelined_search_archives_device_scores() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    if GpuEvaluator::new(&[0.0], 1, 1, &[0.0]).is_none() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    for (band, units, hold) in [(0.2, BandUnits::Score, 4), (0.5, BandUnits::Sd, 3)] {
        let (sample, b) = band_search_fixture(band, units, hold);
        let plain = book(&fixture(&[0, 600, 1200, 1800], 2_400, 17), 1);
        let seen: Archived = Arc::default();
        let out = search::run(&config(&b, &seen), &sample, &sample);
        assert_device_scored(&out);
        assert_archived_values_are_the_cpu_band_book(&sample, &b, &plain, &seen);
        for s in &out.archive.items {
            let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(&s.tree, r)).collect();
            assert_eq!(s.error.to_bits(), b.regret(&pred).to_bits(), "{}", show(&s.tree));
        }
    }
}

/// Throughput of the band kernels against the plain kernel on the same
/// 1,000,000 rows and programs; the sd units add the reduction kernel.
#[test]
#[ignore]
fn long_pnl_band_throughput_against_plain() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    let n = 1_000_000usize;
    let starts: Vec<usize> = (0..100).map(|g| g * 10_000).collect();
    let f = fixture(&starts, n, 29);
    let Some(mut ev) = evaluator(&f, &book(&f, 1)) else {
        return;
    };
    let mut rng = Lcg(31);
    let trees: Vec<Node> = (0..256)
        .map(|k| {
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
    let books: Vec<(String, FitnessBook)> = vec![
        ("plain".to_string(), book(&f, 1)),
        ("band 0.3 hold 1".to_string(), band_book(&f, 0.3, BandUnits::Score, 1, 1)),
        ("band 0 hold 8".to_string(), band_book(&f, 0.0, BandUnits::Score, 8, 1)),
        ("band 0.5 hold 24".to_string(), band_book(&f, 0.5, BandUnits::Score, 24, 1)),
        ("band 1 sd hold 12".to_string(), band_book(&f, 1.0, BandUnits::Sd, 12, 1)),
        ("plain again".to_string(), book(&f, 1)),
    ];
    let mut rates = Vec::new();
    for (name, b) in &books {
        ev.set_pnl_book(b).expect("the book uploads");
        for _ in 0..2 {
            assert!(ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros).ms >= 0.0);
        }
        let reps = 10;
        let t = std::time::Instant::now();
        let mut kernel_ms = 0.0f64;
        for _ in 0..reps {
            let s = ev.score_batch(prog, off, ScoreObjective::LongPnl, &zeros);
            assert!(s.ms >= 0.0);
            kernel_ms += s.ms as f64;
        }
        let rate = (n * trees.len()) as f64 / (t.elapsed().as_secs_f64() / reps as f64);
        let kernel = (n * trees.len()) as f64 / (kernel_ms / reps as f64 / 1000.0);
        println!("{name}: {rate:.3e} row-evals/s (kernel {kernel:.3e})");
        rates.push(rate);
    }
    let base = rates[0].max(rates[5]);
    for (k, (name, _)) in books.iter().enumerate().take(5).skip(1) {
        println!("ratio to plain: {name} {:.3}", rates[k] / base);
    }
}

/// `--gpu` refuses a hold the device cannot carry (exit 2); the automatic
/// backend scores it on the CPU and names the reason.
#[test]
fn a_hold_above_32_exits_2_on_gpu_and_falls_back_in_auto() {
    let _shared = FAULTS.read().unwrap_or_else(|e| e.into_inner());
    if !device_present() {
        return;
    }
    let (_scratch, data, spec) = write_cli_fixture();
    let gpu = rcsr_cmd::rcsr_on_device()
        .args([
            "--gpu", "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", &spec, "--no-dev-log", "--seed", "3", "--generations", "4", "--pop", "32",
            "--metric", "squared", "--json", "--quiet", "--objective", "pnl", "--cost-bps", "5",
            "--validation-frac", "0.3", "--fitness", "pnl", "--min-holding", "33",
        ])
        .output()
        .expect("run rcsr");
    assert_eq!(gpu.status.code(), Some(2), "{}", String::from_utf8_lossy(&gpu.stderr));
    assert!(String::from_utf8_lossy(&gpu.stderr).contains("at most 32"));
    let (sample, _) = search_fixture();
    let f = fixture(&[0, 600, 1200, 1800], 2_400, 17);
    let b = Arc::new(band_book(&f, 0.1, BandUnits::Score, 33, 1));
    let out = search::run_seeded_checked(&required_config(&b, false), &sample, &sample, &[]).expect("auto mode falls back");
    assert_eq!(out.backend, rcsr::backend::Kind::Cpu);
    let why = out.backend_reason.clone().unwrap_or_default();
    assert!(why.contains("at most 32"), "{why}");
    for s in &out.archive.items {
        let pred: Vec<f64> = sample.x.iter().map(|r| rcsr::expr::eval(&s.tree, r)).collect();
        assert_eq!(s.error.to_bits(), b.regret(&pred).to_bits(), "{}", show(&s.tree));
    }
}
