//! Known-truth benchmark suite.
//!
//! A symbolic search can always drive the training error down; the only honest
//! question is whether it *recovers the law that generated the data* rather than
//! memorising the sample. This module supplies a reproducible suite of small,
//! closed-form problems — Feynman-style physics laws and elementary dynamical
//! systems — together with the machinery to run the engine on every
//! problem × noise level and to score whether the true law was found.
//!
//! # What a problem is
//!
//! Every [`Problem`] fixes four things so a run is comparable across seeds:
//!
//! * a **name** and the **input variable names** (the search sees the raw units);
//! * a **truth formula** as an [`Node`] tree over those inputs;
//! * a deterministic **data generator**: a fixed number of training and test
//!   rows drawn uniformly from a fixed input range (the data seed is constant,
//!   so the *same points* are used for every search seed);
//! * a **noise level**: Gaussian noise scaled by the standard deviation of the
//!   noiseless target, so `0.01` is 1 % relative noise and `0.0` is exact.
//!
//! The test sample is always **fresh and noiseless**, drawn from the same range
//! as the training sample but from independent points. A model that merely
//! interpolates the training rows cannot score well on it.
//!
//! # The recovery criterion
//!
//! For every run the engine's Pareto front (all non-dominated
//! error/complexity solutions) is scored on the fresh noiseless test set. The
//! **selected** solution is the front member with the highest test `R²` (ties
//! go to the least complex one, because the front is scanned in complexity
//! order). A run counts as **recovered** when that `R²` is at least
//! [`DEFAULT_THRESHOLD`] (`0.99`). In words: *the front contained an expression
//! that explains at least 99 % of the variance of the true law at new points*.
//!
//! `R² = 1 - SSE/SST` is computed against the mean of the fresh test target; it
//! is `-inf` when any prediction is non-finite. Selecting the best front member
//! on the test set is a deliberate, documented choice: it asks whether the
//! engine *offered* the law, not whether it ranked it first on the noisy
//! training data.
//!
//! See `docs/benchmarks.md` for the recorded results and the full problem table.

use crate::bench_stats::{self as stats, front_hv};
use crate::bench_suites::{self, SignalTask, SIGNAL_THRESHOLD};
use crate::data::Sample;
use crate::expr::{self, constant, input, node, Node, Op};
use crate::metrics::Metric;
use crate::pareto::{Archive, Sol};
use crate::rng::Mt;
use crate::search::{self, Config};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Seed for the data generator. Fixed so the same problem always yields the
/// same points, independent of the search seed; `--seed` drives the engine only.
pub(crate) const DATA_SEED: u32 = 0x0BEE_F001;
/// Seed for the noise stream, kept separate from the point stream.
const NOISE_SEED: u32 = 0x5EED_5EED;

/// Default recovery criterion: the selected solution must reach this test `R²`.
pub const DEFAULT_THRESHOLD: f64 = 0.99;

/// Training rows per problem.
const N_TRAIN: usize = 60;
/// Fresh noiseless test rows per problem (used only for scoring).
const N_TEST: usize = 200;

// ---------------------------------------------------------------------------
// Problem definition
// ---------------------------------------------------------------------------

/// One known-truth problem: a truth formula, its input ranges and a fixed
/// deterministic generator for `(train, noiseless test)` samples.
#[derive(Clone)]
pub struct Problem {
    /// Short, stable identifier (kebab/lower-snake), used in reports.
    pub name: &'static str,
    /// Input variable names, in the order the truth formula indexes them.
    pub vars: Vec<String>,
    /// The generating law, in raw input units.
    pub truth: Node,
    /// Inclusive `(lo, hi)` sampling range per input variable.
    pub ranges: Vec<(f64, f64)>,
    /// Number of training rows.
    pub n_train: usize,
    /// Number of fresh noiseless test rows.
    pub n_test: usize,
    /// Whether the problem belongs to the fast `--tiny` subset.
    pub tiny: bool,
    /// Suite the problem belongs to.
    pub suite: &'static str,
    /// Node count of the truth tree under the default complexity model.
    pub truth_cx: usize,
    /// Input indexes the truth does not use.
    pub distractors: Vec<usize>,
}

impl Problem {
    /// Build a problem from a truth tree and its input ranges.
    pub fn new(
        name: &'static str,
        vars: &[&str],
        truth: Node,
        ranges: &[(f64, f64)],
        tiny: bool,
    ) -> Self {
        Problem {
            name,
            vars: vars.iter().map(|s| s.to_string()).collect(),
            truth_cx: expr::complexity(&truth),
            truth,
            ranges: ranges.to_vec(),
            n_train: N_TRAIN,
            n_test: N_TEST,
            tiny,
            suite: "known-truth",
            distractors: Vec::new(),
        }
    }

    /// Move the problem into another suite with its own row counts.
    pub fn in_suite(mut self, suite: &'static str, n_train: usize, n_test: usize) -> Self {
        self.suite = suite;
        self.n_train = n_train;
        self.n_test = n_test;
        self
    }

    pub fn with_truth_cx(mut self, cx: usize) -> Self {
        self.truth_cx = cx;
        self
    }

    pub fn with_distractors(mut self, d: &[usize]) -> Self {
        self.distractors = d.to_vec();
        self
    }

    /// The truth formula as a printable string.
    pub fn truth_string(&self) -> String {
        expr::to_string(&self.truth, &self.vars)
    }

    /// Deterministic `(train, test)` pair for this problem.
    ///
    /// Both samples are drawn from the problem's ranges with a fixed data seed;
    /// the training target carries `noise` worth of relative Gaussian noise and
    /// the test target is always noiseless.
    pub fn generate(&self, noise: f64) -> (Sample, Sample) {
        let seed = fnv(self.name.as_bytes());
        let mut rng = Mt::new(DATA_SEED ^ seed);
        let mut nrng = Mt::new(NOISE_SEED ^ seed);
        let train = draw(self, self.n_train, noise, &mut rng, &mut nrng);
        let test = draw(self, self.n_test, 0.0, &mut rng, &mut nrng);
        (train, test)
    }

    /// A noiseless sample of `n` fresh points (mainly for tests).
    pub fn noiseless(&self, n: usize) -> Sample {
        let seed = fnv(self.name.as_bytes());
        let mut rng = Mt::new(DATA_SEED ^ seed);
        let mut nrng = Mt::new(NOISE_SEED ^ seed);
        draw(self, n, 0.0, &mut rng, &mut nrng)
    }
}

/// Draw `n` rows from the problem's ranges, appending the truth target. When
/// `noise > 0` the target gets Gaussian noise scaled by its own standard
/// deviation, so a level is comparable across problems of different magnitude.
fn draw(p: &Problem, n: usize, noise: f64, rng: &mut Mt, nrng: &mut Mt) -> Sample {
    let mut s = Sample {
        x: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    for _ in 0..n {
        let row: Vec<f64> = p.ranges.iter().map(|&(lo, hi)| rng.range(lo, hi)).collect();
        let t = expr::eval(&p.truth, &row);
        s.y.push(t);
        s.x.push(row);
    }
    if noise > 0.0 {
        let m = s.y.iter().sum::<f64>() / n as f64;
        let var = s.y.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        for v in s.y.iter_mut() {
            *v += sd * noise * gaussian(nrng);
        }
    }
    s
}

/// Standard normal draw (Box–Muller), from a uniform `[0, 1)` stream.
pub(crate) fn gaussian(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// FNV-1a over the problem name, so every problem gets its own stream.
pub(crate) fn fnv(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

pub(crate) fn c(v: f64) -> Node {
    constant(v)
}
pub(crate) fn x(i: usize) -> Node {
    input(i)
}
pub(crate) fn u1(op: Op, a: Node) -> Node {
    node(op, vec![a])
}
pub(crate) fn u2(op: Op, a: Node, b: Node) -> Node {
    node(op, vec![a, b])
}

/// The 25 known-truth problems. All truth formulas are finite over their ranges.
pub fn known_truth() -> Vec<Problem> {
    use Op::*;
    let p = |name, vars: &[&str], truth, ranges: &[(f64, f64)], tiny| {
        Problem::new(name, vars, truth, ranges, tiny)
    };
    vec![
        p("line", &["x"], u2(Add, u2(Mul, c(2.0), x(0)), c(1.0)), &[(-2.0, 2.0)], true),
        p("quadratic", &["x"], u1(Square, x(0)), &[(-2.0, 2.0)], true),
        p("cubic", &["x"], u1(Cube, x(0)), &[(-1.5, 1.5)], false),
        p("inverse", &["x"], u2(Div, c(1.0), x(0)), &[(0.5, 2.5)], true),
        p("product", &["x1", "x2"], u2(Mul, x(0), x(1)), &[(-2.0, 2.0), (-2.0, 2.0)], true),
        p("ratio", &["x1", "x2"], u2(Div, x(0), x(1)), &[(-2.0, 2.0), (0.5, 2.5)], true),
        p("sqrt_law", &["x"], u1(Sqrt, x(0)), &[(0.25, 4.0)], true),
        p("exp_decay", &["x"], u1(Exp, u1(Neg, x(0))), &[(0.0, 2.0)], true),
        p("log_law", &["x"], u1(Ln, x(0)), &[(0.2, 4.0)], true),
        p("sine_law", &["x"], u1(Sin, x(0)), &[(-std::f64::consts::PI, std::f64::consts::PI)], true),
        p("gaussian", &["x"], u1(Exp, u1(Neg, u1(Square, x(0)))), &[(-2.5, 2.5)], false),
        p("logistic_law", &["x"], u1(Logistic, x(0)), &[(-4.0, 4.0)], false),
        p("tanh_law", &["x"], u1(Tanh, x(0)), &[(-3.0, 3.0)], false),
        p(
            "harmonic_sum",
            &["x1", "x2"],
            u2(Add, u2(Div, c(1.0), x(0)), u2(Div, c(1.0), x(1))),
            &[(0.5, 2.5), (0.5, 2.5)],
            false,
        ),
        p(
            "reciprocal_sum",
            &["x1", "x2"],
            u2(Div, c(1.0), u2(Add, x(0), x(1))),
            &[(0.5, 3.0), (0.5, 3.0)],
            false,
        ),
        p(
            "pythagoras",
            &["x1", "x2"],
            u1(Sqrt, u2(Add, u1(Square, x(0)), u1(Square, x(1)))),
            &[(-3.0, 3.0), (-3.0, 3.0)],
            false,
        ),
        p("kinetic_energy", &["m", "v"], u2(Mul, c(0.5), u2(Mul, x(0), u1(Square, x(1)))), &[(0.5, 3.0), (-3.0, 3.0)], false),
        p(
            "coulomb",
            &["q1", "q2", "r"],
            u2(Div, u2(Mul, x(0), x(1)), u1(Square, x(2))),
            &[(0.5, 3.0), (0.5, 3.0), (0.5, 3.0)],
            false,
        ),
        p(
            "ideal_gas",
            &["n", "T", "V"],
            u2(Div, u2(Mul, x(0), x(1)), x(2)),
            &[(0.5, 3.0), (0.5, 3.0), (0.5, 3.0)],
            false,
        ),
        p("spring_period", &["m", "k"], u1(Sqrt, u2(Div, x(0), x(1))), &[(0.5, 3.0), (0.5, 3.0)], false),
        p(
            "velocity_addition",
            &["u", "v"],
            u2(Div, u2(Add, x(0), x(1)), u2(Add, c(1.0), u2(Mul, x(0), x(1)))),
            &[(-0.8, 0.8), (-0.8, 0.8)],
            false,
        ),
        p("damped_oscillator", &["x"], u2(Mul, u1(Exp, u1(Neg, x(0))), u1(Sin, x(0))), &[(0.0, 4.0)], false),
        p(
            "relativistic",
            &["beta"],
            u2(Div, c(1.0), u1(Sqrt, u2(Sub, c(1.0), u1(Square, x(0))))),
            &[(-0.9, 0.9)],
            false,
        ),
        p("sine_sum", &["x1", "x2"], u1(Sin, u2(Add, x(0), x(1))), &[(-2.0, 2.0), (-2.0, 2.0)], false),
        p("power_law", &["x"], u2(Pow, x(0), c(1.5)), &[(0.25, 4.0)], false),
    ]
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Coefficient of determination `1 - SSE/SST` of `pred` against `y`, using the
/// mean of `y` as the null model. `-inf` when the inputs are mismatched or any
/// prediction is non-finite; a constant target is scored exactly (1 if the
/// prediction matches, else 0).
pub fn r_squared(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || y.is_empty() {
        return f64::NEG_INFINITY;
    }
    if pred.iter().any(|p| !p.is_finite()) {
        return f64::NEG_INFINITY;
    }
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sst: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    if sst <= 0.0 {
        let ok = pred
            .iter()
            .zip(y)
            .all(|(p, t)| (p - t).abs() <= 1e-12 * (1.0 + t.abs()));
        return if ok { 1.0 } else { 0.0 };
    }
    let sse: f64 = pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum();
    1.0 - sse / sst
}

/// Root-mean-square error; `+inf` on mismatched or non-finite input.
pub fn rmse(pred: &[f64], y: &[f64]) -> f64 {
    if pred.len() != y.len() || y.is_empty() || pred.iter().any(|p| !p.is_finite()) {
        return f64::INFINITY;
    }
    let sse: f64 = pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum();
    (sse / y.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Running the suite
// ---------------------------------------------------------------------------

/// Version of the data generators and criteria. Two result files with a
/// different value are never compared.
pub const GENERATOR_VERSION: &str = "bench-v2";

/// Which evaluator backend a run asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendChoice {
    /// The engine default: the device when the build has one, else the CPU.
    Auto,
    Cpu,
    Gpu,
}

impl BackendChoice {
    pub fn name(self) -> &'static str {
        match self {
            BackendChoice::Auto => "auto",
            BackendChoice::Cpu => "cpu",
            BackendChoice::Gpu => "gpu",
        }
    }
}

/// How the bench picks the constant-fit arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FitChoice {
    Strategy(crate::fit::FitStrategy),
    /// Resolve like the CLI does when `--fit-strategy` is absent.
    Default,
}

/// Search knobs that are copied into the engine configuration. `None` keeps the
/// engine default (and, for the fit arm, the bench default `Exact`).
#[derive(Clone, Debug, Default)]
pub struct Knobs {
    pub fit_strategy: Option<FitChoice>,
    pub fit_iters: Option<usize>,
    /// Comma list as given on the command line (`--ops`).
    pub ops: Option<String>,
    pub max_complexity: Option<usize>,
    pub patience: Option<usize>,
    pub migration_interval: Option<usize>,
    pub restart_interval: Option<usize>,
    pub variation: Option<crate::search::VariationMode>,
    pub dispatch: Option<crate::dispatch::DispatchMode>,
    /// #466: `Some(false)` turns the front prune off (`--no-prune`).
    pub prune: Option<bool>,
}

/// How to run the suite: noise levels, search budget, seed and criterion.
#[derive(Clone)]
pub struct SuiteConfig {
    /// Relative noise levels to test every problem at.
    pub noise: Vec<f64>,
    pub generations: usize,
    pub islands: usize,
    pub pop: usize,
    pub seed: u32,
    pub metric: Metric,
    /// Restrict the known-truth suite to the fast [`Problem::tiny`] subset.
    pub tiny: bool,
    /// Recovery threshold on the test `R²`.
    pub threshold: f64,
    /// Optional per-run wall-clock budget.
    pub time_budget: Option<f64>,
    /// #76: let the Pareto front supply 15 % of parents.
    pub front_parents: bool,
    /// Suites to run; empty means `known-truth`.
    pub suites: Vec<String>,
    /// Run only these problem names; empty means every problem.
    pub problems: Vec<String>,
    /// Engine seeds; empty means the single [`seed`](Self::seed).
    pub seeds: Vec<u32>,
    pub backend: BackendChoice,
    pub knobs: Knobs,
    /// Runs in flight at once (CPU only).
    pub jobs: usize,
    /// #463: worker threads for fitting and scoring inside an island.
    pub threads: usize,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        SuiteConfig {
            noise: vec![0.0, 0.01, 0.1],
            generations: 150,
            islands: 2,
            pop: 200,
            seed: 0,
            metric: Metric::Squared,
            tiny: false,
            threshold: DEFAULT_THRESHOLD,
            time_budget: None,
            front_parents: true, // #76
            suites: Vec::new(),
            problems: Vec::new(),
            seeds: Vec::new(),
            backend: BackendChoice::Auto,
            knobs: Knobs::default(),
            jobs: 1,
            threads: 1,
        }
    }
}

impl SuiteConfig {
    fn suite_names(&self) -> Vec<String> {
        if self.suites.is_empty() {
            vec!["known-truth".to_string()]
        } else {
            self.suites.clone()
        }
    }

    fn seed_list(&self) -> Vec<u32> {
        if self.seeds.is_empty() {
            vec![self.seed]
        } else {
            self.seeds.clone()
        }
    }
}

/// The outcome of one problem × noise × seed run.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub suite: String,
    pub problem: String,
    /// `None` for signal tasks, which have no noise sweep.
    pub noise: Option<f64>,
    pub seed: u32,
    /// Selected solution's test `R²` ≥ threshold.
    pub recovered: bool,
    /// Test `R²` of the selected solution on fresh noiseless data. For signal
    /// tasks: `R²` of the validation-selected member against the noiseless test
    /// signal.
    pub test_r2: f64,
    /// Training error of the selected solution, in the search metric.
    pub train_error: f64,
    /// RMSE of the selected solution on fresh noiseless data.
    pub test_rmse: f64,
    /// Node count of the selected solution.
    pub complexity: usize,
    /// Printed form of the selected solution.
    pub expression: String,
    /// Wall-clock seconds for this run.
    pub seconds: f64,
    pub truth_cx: usize,
    pub generations_run: usize,
    pub stop_reason: String,
    pub backend: String,
    /// Test `R²` of the truth structure refit on the training rows.
    pub oracle_r2: Option<f64>,
    pub oracle_recovered: Option<bool>,
    pub exact_cx: Option<usize>,
    pub excess_cx: Option<i64>,
    pub front_hv: f64,
    pub hit_gen: Option<usize>,
    pub hit_s: Option<f64>,
    pub hit_evals: Option<usize>,
    pub oracle_hit_gen: Option<usize>,
    pub oracle_hit_s: Option<f64>,
    pub oracle_hit_evals: Option<usize>,
    pub signal_r2: Option<f64>,
    pub signal_recovered: Option<bool>,
    pub selected_cx: Option<usize>,
    pub distractors_used: Option<usize>,
    pub fit_attempts: Option<usize>,
    pub fit_gpu: Option<usize>,
    pub fit_fallback: Option<usize>,
    /// #466: seconds the front prune took (0 with `--no-prune`).
    pub prune_s: f64,
}

/// Mean-adjusted summary of one `(suite, noise)` group.
#[derive(Clone, Debug)]
pub struct GroupSummary {
    pub suite: String,
    pub noise: Option<f64>,
    pub runs: usize,
    pub recovered: usize,
    pub oracle_recovered: usize,
    /// Runs where the oracle criterion is defined.
    pub oracle_defined: usize,
    pub oracle_failed: usize,
    pub signal_recovered: usize,
    pub signal_defined: usize,
    pub median_excess_cx: Option<f64>,
    pub median_hit_s: Option<f64>,
    pub seconds: f64,
}

/// Aggregate statistics for a group of runs.
#[derive(Clone, Debug)]
pub struct Summary {
    pub runs: usize,
    pub recovered: usize,
    /// `recovered / runs` (0 when empty).
    pub rate: f64,
    /// Median complexity of the selected solution over the group's runs.
    pub median_complexity: f64,
}

/// One noise level's worth of statistics.
#[derive(Clone, Debug)]
pub struct NoiseSummary {
    pub noise: f64,
    pub runs: usize,
    pub recovered: usize,
    pub rate: f64,
    pub median_complexity: f64,
}

/// Everything a suite run produced.
#[derive(Clone, Debug)]
pub struct Report {
    /// Number of problems actually run (after the filters).
    pub problems: usize,
    pub results: Vec<RunResult>,
    pub summary: Summary,
    pub by_noise: Vec<NoiseSummary>,
    pub groups: Vec<GroupSummary>,
}

impl Report {
    /// Results for one noise level, in problem order.
    pub fn for_noise(&self, noise: f64) -> Vec<&RunResult> {
        self.results
            .iter()
            .filter(|r| r.noise.is_some_and(|n| (n - noise).abs() <= 1e-12))
            .collect()
    }
}

fn stats(results: &[RunResult]) -> Summary {
    let runs = results.len();
    let recovered = results.iter().filter(|r| r.recovered).count();
    let mut cx: Vec<f64> = results.iter().map(|r| r.complexity as f64).collect();
    cx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_complexity = median(&cx);
    Summary {
        runs,
        recovered,
        rate: if runs == 0 {
            0.0
        } else {
            recovered as f64 / runs as f64
        },
        median_complexity,
    }
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// The oracle criterion: the member's error is at most twice the error of the
/// truth structure refit on the same training rows.
pub fn oracle_ok(r2: f64, oracle_r2: f64) -> bool {
    (1.0 - r2) <= 2.0 * (1.0 - oracle_r2) + 1e-9
}

/// Test `R²` of the truth structure `c0 + c1 * truth`, every constant free,
/// refit on the training rows (50 LM iterations from the true constants).
pub fn oracle_r2(p: &Problem, train: &Sample, test: &Sample) -> f64 {
    let mut tree = u2(Op::Add, c(0.0), u2(Op::Mul, c(1.0), p.truth.clone()));
    crate::fit::fit_weighted(&mut tree, &train.x, &train.y, &[], 50);
    let pred: Vec<f64> = test.x.iter().map(|row| expr::eval(&tree, row)).collect();
    r_squared(&pred, &test.y)
}

/// What a run asks the front observer to score.
enum Scoring {
    Problem {
        test: Arc<Sample>,
        oracle_r2: Option<f64>,
        threshold: f64,
    },
    Signal {
        val: Arc<Sample>,
        s_test: Arc<Vec<f64>>,
        test_x: Arc<Sample>,
    },
}

/// Scores new front members once per structural hash and records the first
/// generation at which the hit criteria hold.
struct Tracker {
    scoring: Scoring,
    /// problem: `(test R², _)`; signal: `(validation error, signal R²)`.
    cache: HashMap<u64, (f64, f64)>,
    evals_per_gen: usize,
    hit: Option<(usize, f64)>,
    oracle_hit: Option<(usize, f64)>,
}

impl Tracker {
    fn observe(&mut self, gen: usize, secs: f64, archive: &Archive) {
        match &self.scoring {
            Scoring::Problem {
                test,
                oracle_r2,
                threshold,
            } => {
                let (test, oracle_r2, threshold) = (test.clone(), *oracle_r2, *threshold);
                let mut best = f64::NEG_INFINITY;
                let mut oracle = false;
                for sol in &archive.items {
                    let h = expr::structural_hash(&sol.tree);
                    let r2 = self
                        .cache
                        .entry(h)
                        .or_insert_with(|| {
                            let pred: Vec<f64> =
                                test.x.iter().map(|row| expr::eval(&sol.tree, row)).collect();
                            (r_squared(&pred, &test.y), 0.0)
                        })
                        .0;
                    best = best.max(r2);
                    if oracle_r2.is_some_and(|o| oracle_ok(r2, o)) {
                        oracle = true;
                    }
                }
                if self.hit.is_none() && best >= threshold {
                    self.hit = Some((gen, secs));
                }
                if self.oracle_hit.is_none() && oracle {
                    self.oracle_hit = Some((gen, secs));
                }
            }
            Scoring::Signal { val, s_test, test_x } => {
                let (val, s_test, test_x) = (val.clone(), s_test.clone(), test_x.clone());
                let mut chosen: Option<(f64, f64, f64)> = None;
                for sol in &archive.items {
                    let h = expr::structural_hash(&sol.tree);
                    let (verr, sr2) = *self.cache.entry(h).or_insert_with(|| {
                        let (verr, sr2) = score_signal(&sol.tree, &val, &test_x, &s_test);
                        (verr, sr2)
                    });
                    let better = match chosen {
                        None => true,
                        Some((e, _, cost)) => verr < e || (verr == e && sol.cost < cost),
                    };
                    if better {
                        chosen = Some((verr, sr2, sol.cost));
                    }
                }
                if self.hit.is_none() && chosen.is_some_and(|(_, r2, _)| r2 >= SIGNAL_THRESHOLD) {
                    self.hit = Some((gen, secs));
                }
            }
        }
    }
}

/// `(mean squared error on the validation rows, R² against the noiseless test
/// signal)` of a tree. A non-finite prediction gives `(+inf, -inf)`.
fn score_signal(tree: &Node, val: &Sample, test: &Sample, s_test: &[f64]) -> (f64, f64) {
    let pv: Vec<f64> = val.x.iter().map(|row| expr::eval(tree, row)).collect();
    let verr = if pv.iter().all(|p| p.is_finite()) {
        pv.iter().zip(&val.y).map(|(p, t)| (p - t) * (p - t)).sum::<f64>() / pv.len() as f64
    } else {
        f64::INFINITY
    };
    let pt: Vec<f64> = test.x.iter().map(|row| expr::eval(tree, row)).collect();
    (verr, r_squared(&pt, s_test))
}

/// Engine configuration for one run.
fn base_config(cfg: &SuiteConfig, seed: u32) -> Config {
    let mut sc = Config::default();
    sc.generations = cfg.generations.max(1);
    sc.islands = cfg.islands.max(1);
    sc.pop = cfg.pop.max(2);
    sc.metric = cfg.metric;
    sc.seed = seed;
    sc.time_budget = cfg.time_budget;
    sc.front_parents = cfg.front_parents; // #76
    sc.threads = cfg.threads.max(1); // #463
    match cfg.backend {
        BackendChoice::Auto => {}
        BackendChoice::Cpu => sc.gpu = false,
        BackendChoice::Gpu => sc.gpu = true,
    }
    let k = &cfg.knobs;
    if let Some(FitChoice::Strategy(s)) = k.fit_strategy {
        sc.fit_strategy = s;
    }
    if let Some(v) = k.fit_iters {
        sc.fit_iters = v;
    }
    if let Some(ops) = &k.ops {
        if let Ok(list) = Op::parse_list_checked(ops) {
            sc.ops = list;
        }
    }
    if let Some(v) = k.max_complexity {
        sc.max_complexity = v;
    }
    if let Some(v) = k.patience {
        sc.patience = v;
    }
    if let Some(v) = k.migration_interval {
        sc.migration_interval = v;
    }
    if let Some(v) = k.restart_interval {
        sc.restart_interval = v;
    }
    if let Some(v) = k.variation {
        sc.variation = v;
    }
    if let Some(v) = k.dispatch {
        sc.dispatch = v;
    }
    if let Some(v) = k.prune {
        sc.prune = v;
    }
    sc
}

/// Apply `--fit-strategy default` the way the CLI does.
fn resolve_default_fit(cfg: &SuiteConfig, sc: &mut Config, train: &Sample) {
    if cfg.knobs.fit_strategy == Some(FitChoice::Default) {
        let r = search::resolve_fit_strategy(sc, train, crate::fit::FitRequest::Unset);
        sc.fit_strategy = r.strategy;
        sc.fit_admission = r.admission;
    }
}

fn fit_counters(out: &search::Outcome) -> (Option<usize>, Option<usize>, Option<usize>) {
    match &out.gpu {
        Some(g) => (Some(g.fit_attempts), Some(g.fit_gpu), Some(g.fit_fallback)),
        None => (None, None, None),
    }
}

fn empty_result(suite: &str, problem: &str, noise: Option<f64>, seed: u32, truth_cx: usize) -> RunResult {
    RunResult {
        suite: suite.to_string(),
        problem: problem.to_string(),
        noise,
        seed,
        recovered: false,
        test_r2: f64::NEG_INFINITY,
        train_error: f64::INFINITY,
        test_rmse: f64::INFINITY,
        complexity: 0,
        expression: String::new(),
        seconds: 0.0,
        truth_cx,
        generations_run: 0,
        stop_reason: String::new(),
        backend: String::new(),
        oracle_r2: None,
        oracle_recovered: None,
        exact_cx: None,
        excess_cx: None,
        front_hv: 0.0,
        hit_gen: None,
        hit_s: None,
        hit_evals: None,
        oracle_hit_gen: None,
        oracle_hit_s: None,
        oracle_hit_evals: None,
        signal_r2: None,
        signal_recovered: None,
        selected_cx: None,
        distractors_used: None,
        fit_attempts: None,
        fit_gpu: None,
        fit_fallback: None,
        prune_s: 0.0,
    }
}

fn install_hits(r: &mut RunResult, t: &Tracker) {
    let evals = |g: usize| g * t.evals_per_gen;
    if let Some((g, s)) = t.hit {
        r.hit_gen = Some(g);
        r.hit_s = Some(s);
        r.hit_evals = Some(evals(g));
    }
    if let Some((g, s)) = t.oracle_hit {
        r.oracle_hit_gen = Some(g);
        r.oracle_hit_s = Some(s);
        r.oracle_hit_evals = Some(evals(g));
    }
}

/// Run one problem at one noise level with the configuration's seed.
pub fn run_one(p: &Problem, noise: f64, cfg: &SuiteConfig) -> RunResult {
    run_problem(p, noise, cfg.seed, cfg)
}

/// Run one problem at one noise level and score recovery.
pub fn run_problem(p: &Problem, noise: f64, seed: u32, cfg: &SuiteConfig) -> RunResult {
    let (train, test) = p.generate(noise);
    let oracle = oracle_r2(p, &train, &test);
    let oracle_valid = oracle.is_finite() && oracle >= 0.0;

    let mut sc = base_config(cfg, seed);
    resolve_default_fit(cfg, &mut sc, &train);
    let test = Arc::new(test);
    let tracker = Arc::new(Mutex::new(Tracker {
        scoring: Scoring::Problem {
            test: test.clone(),
            oracle_r2: oracle_valid.then_some(oracle),
            threshold: cfg.threshold,
        },
        cache: HashMap::new(),
        evals_per_gen: sc.islands * sc.pop,
        hit: None,
        oracle_hit: None,
    }));
    let observer = tracker.clone();
    sc.front_observer = Some(Arc::new(move |g, s, a| {
        observer.lock().unwrap().observe(g, s, a)
    }));
    let max_cx = sc.max_complexity as f64;

    let t0 = Instant::now();
    let out = search::run(&sc, &train, &test);
    let seconds = t0.elapsed().as_secs_f64();

    // Scan the Pareto front in complexity order and keep the member with the
    // highest test R²; ties therefore go to the simplest expression.
    let mut r = empty_result(p.suite, p.name, Some(noise), seed, p.truth_cx);
    r.seconds = seconds;
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut have_best = false;
    for sol in out.archive.sorted() {
        let pred: Vec<f64> = test.x.iter().map(|row| expr::eval(&sol.tree, row)).collect();
        let r2 = r_squared(&pred, &test.y);
        points.push((r2, sol.cost));
        if oracle_valid && r.exact_cx.is_none() && oracle_ok(r2, oracle) {
            r.exact_cx = Some(sol.complexity);
        }
        if !have_best || r2 > r.test_r2 {
            have_best = true;
            r.test_r2 = r2;
            r.train_error = sol.error;
            r.test_rmse = rmse(&pred, &test.y);
            r.complexity = sol.complexity;
            r.expression = expr::to_string(&sol.tree, &p.vars);
        }
    }
    r.recovered = r.test_r2 >= cfg.threshold;
    r.generations_run = out.generations;
    r.stop_reason = out.stop_reason.to_string();
    r.backend = out.backend.name().to_string();
    (r.fit_attempts, r.fit_gpu, r.fit_fallback) = fit_counters(&out);
    r.prune_s = out.prune.seconds;
    r.front_hv = front_hv(&points, max_cx);
    r.oracle_r2 = Some(oracle);
    if oracle_valid {
        r.oracle_recovered = Some(oracle_ok(r.test_r2, oracle));
        r.excess_cx = r.exact_cx.map(|e| e as i64 - p.truth_cx as i64);
    }
    install_hits(&mut r, &tracker.lock().unwrap());
    r
}

/// Run one signal task with one engine seed.
pub fn run_signal(task: &SignalTask, seed: u32, cfg: &SuiteConfig) -> RunResult {
    let data = task.generate();
    let mut sc = base_config(cfg, seed);
    resolve_default_fit(cfg, &mut sc, &data.train);
    let val = Arc::new(data.val.clone());
    let test = Arc::new(data.test.clone());
    let s_test = Arc::new(data.s_test.clone());
    let tracker = Arc::new(Mutex::new(Tracker {
        scoring: Scoring::Signal {
            val: val.clone(),
            s_test: s_test.clone(),
            test_x: test.clone(),
        },
        cache: HashMap::new(),
        evals_per_gen: sc.islands * sc.pop,
        hit: None,
        oracle_hit: None,
    }));
    let observer = tracker.clone();
    sc.front_observer = Some(Arc::new(move |g, s, a| {
        observer.lock().unwrap().observe(g, s, a)
    }));
    let max_cx = sc.max_complexity as f64;

    let t0 = Instant::now();
    let out = search::run(&sc, &data.train, &val);
    let seconds = t0.elapsed().as_secs_f64();

    let mut r = empty_result("signal", task.name, None, seed, task.truth_cx);
    r.seconds = seconds;
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut chosen: Option<(f64, Sol)> = None;
    for sol in out.archive.sorted() {
        let (verr, sr2) = score_signal(&sol.tree, &val, &test, &s_test);
        points.push((sr2, sol.cost));
        if chosen.as_ref().map_or(true, |(e, _)| verr < *e) {
            chosen = Some((verr, sol));
        }
    }
    if let Some((_, sol)) = chosen {
        let pred: Vec<f64> = test.x.iter().map(|row| expr::eval(&sol.tree, row)).collect();
        let sr2 = r_squared(&pred, &s_test);
        r.test_r2 = sr2;
        r.train_error = sol.error;
        r.test_rmse = rmse(&pred, &s_test);
        r.complexity = sol.complexity;
        r.expression = expr::to_string(&sol.tree, &task.vars);
        r.signal_r2 = Some(sr2);
        r.signal_recovered = Some(sr2 >= SIGNAL_THRESHOLD);
        r.selected_cx = Some(sol.complexity);
        r.distractors_used = Some(
            task.distractors
                .iter()
                .filter(|&&d| uses_input(&sol.tree, d))
                .count(),
        );
    } else {
        r.signal_r2 = Some(f64::NEG_INFINITY);
        r.signal_recovered = Some(false);
    }
    r.recovered = r.test_r2 >= cfg.threshold;
    r.generations_run = out.generations;
    r.stop_reason = out.stop_reason.to_string();
    r.backend = out.backend.name().to_string();
    (r.fit_attempts, r.fit_gpu, r.fit_fallback) = fit_counters(&out);
    r.prune_s = out.prune.seconds;
    r.front_hv = front_hv(&points, max_cx);
    install_hits(&mut r, &tracker.lock().unwrap());
    r
}

/// Whether the tree reads input column `idx`.
pub fn uses_input(n: &Node, idx: usize) -> bool {
    (n.op == Op::Input && n.val as usize == idx) || n.kids.iter().any(|k| uses_input(k, idx))
}

/// One planned run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunSpec {
    pub suite: String,
    pub problem: String,
    pub noise: Option<f64>,
    pub seed: u32,
}

pub const SUITES: [&str; 3] = ["known-truth", "hard", "signal"];

/// Problems of a suite (not the signal tasks).
pub fn suite_problems(suite: &str) -> Vec<Problem> {
    match suite {
        "known-truth" => known_truth(),
        "hard" => bench_suites::hard(),
        _ => Vec::new(),
    }
}

/// Expand `all` and validate suite names.
pub fn resolve_suites(names: &[String]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for n in names {
        let list: Vec<String> = if n == "all" {
            SUITES.iter().map(|s| s.to_string()).collect()
        } else if SUITES.contains(&n.as_str()) {
            vec![n.clone()]
        } else {
            return Err(format!(
                "unknown suite '{}' (try known-truth, hard, signal or all)",
                n
            ));
        };
        for s in list {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    Ok(out)
}

/// The runs a configuration expands to, in execution order.
pub fn plan(cfg: &SuiteConfig) -> Result<Vec<RunSpec>, String> {
    let suites = resolve_suites(&cfg.suite_names())?;
    let mut known: Vec<String> = Vec::new();
    for s in &suites {
        if s == "signal" {
            known.extend(bench_suites::signal().iter().map(|t| t.name.to_string()));
        } else {
            known.extend(suite_problems(s).iter().map(|p| p.name.to_string()));
        }
    }
    for want in &cfg.problems {
        if !known.contains(want) {
            return Err(format!("--problems: unknown problem '{}'", want));
        }
    }
    let wanted = |name: &str| cfg.problems.is_empty() || cfg.problems.iter().any(|p| p == name);
    let mut specs = Vec::new();
    for suite in &suites {
        for seed in cfg.seed_list() {
            if suite == "signal" {
                for t in bench_suites::signal() {
                    if wanted(t.name) {
                        specs.push(RunSpec {
                            suite: suite.clone(),
                            problem: t.name.to_string(),
                            noise: None,
                            seed,
                        });
                    }
                }
            } else {
                let problems: Vec<Problem> = suite_problems(suite)
                    .into_iter()
                    .filter(|p| wanted(p.name) && (!cfg.tiny || suite != "known-truth" || p.tiny))
                    .collect();
                for &noise in &cfg.noise {
                    for p in &problems {
                        specs.push(RunSpec {
                            suite: suite.clone(),
                            problem: p.name.to_string(),
                            noise: Some(noise),
                            seed,
                        });
                    }
                }
            }
        }
    }
    Ok(specs)
}

/// Run one planned run.
pub fn run_spec(spec: &RunSpec, cfg: &SuiteConfig) -> RunResult {
    if spec.suite == "signal" {
        let task = bench_suites::signal()
            .into_iter()
            .find(|t| t.name == spec.problem)
            .expect("planned signal task exists");
        run_signal(&task, spec.seed, cfg)
    } else {
        let p = suite_problems(&spec.suite)
            .into_iter()
            .find(|p| p.name == spec.problem)
            .expect("planned problem exists");
        run_problem(&p, spec.noise.unwrap_or(0.0), spec.seed, cfg)
    }
}

/// Run every planned run. `on_result` is called as each run finishes (in
/// completion order when `jobs > 1`); the returned vector is in plan order.
pub fn run_plan(
    specs: &[RunSpec],
    cfg: &SuiteConfig,
    on_result: &mut (dyn FnMut(&RunResult) + Send),
) -> Vec<RunResult> {
    let jobs = cfg.jobs.max(1).min(specs.len().max(1));
    if jobs == 1 {
        let mut out = Vec::with_capacity(specs.len());
        for s in specs {
            let r = run_spec(s, cfg);
            on_result(&r);
            out.push(r);
        }
        return out;
    }
    let next = AtomicUsize::new(0);
    let shared: Mutex<(Vec<Option<RunResult>>, &mut (dyn FnMut(&RunResult) + Send))> =
        Mutex::new((vec![None; specs.len()], on_result));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= specs.len() {
                    break;
                }
                let r = run_spec(&specs[i], cfg);
                let mut g = shared.lock().unwrap();
                (g.1)(&r);
                g.0[i] = Some(r);
            });
        }
    });
    shared
        .into_inner()
        .unwrap()
        .0
        .into_iter()
        .map(|r| r.expect("every planned run finished"))
        .collect()
}

fn group_summaries(results: &[RunResult]) -> Vec<GroupSummary> {
    let mut keys: Vec<(String, Option<f64>)> = Vec::new();
    for r in results {
        let k = (r.suite.clone(), r.noise);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.into_iter()
        .map(|(suite, noise)| {
            let sub: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.suite == suite && r.noise == noise)
                .collect();
            let excess: Vec<f64> = sub
                .iter()
                .filter_map(|r| r.excess_cx.map(|e| e as f64))
                .collect();
            let hits: Vec<f64> = sub.iter().filter_map(|r| r.hit_s).collect();
            GroupSummary {
                suite,
                noise,
                runs: sub.len(),
                recovered: sub.iter().filter(|r| r.recovered).count(),
                oracle_recovered: sub.iter().filter(|r| r.oracle_recovered == Some(true)).count(),
                oracle_defined: sub.iter().filter(|r| r.oracle_recovered.is_some()).count(),
                oracle_failed: sub
                    .iter()
                    .filter(|r| r.suite != "signal" && r.oracle_recovered.is_none())
                    .count(),
                signal_recovered: sub.iter().filter(|r| r.signal_recovered == Some(true)).count(),
                signal_defined: sub.iter().filter(|r| r.signal_recovered.is_some()).count(),
                median_excess_cx: stats::median_of(&excess),
                median_hit_s: stats::median_of(&hits),
                seconds: sub.iter().map(|r| r.seconds).sum(),
            }
        })
        .collect()
}

/// Build the report from finished runs.
pub fn build_report(results: Vec<RunResult>, cfg: &SuiteConfig) -> Report {
    let summary = stats(&results);
    let by_noise = cfg
        .noise
        .iter()
        .map(|&nv| {
            let sub: Vec<RunResult> = results
                .iter()
                .filter(|r| r.noise.is_some_and(|n| (n - nv).abs() <= 1e-12))
                .cloned()
                .collect();
            let s = stats(&sub);
            NoiseSummary {
                noise: nv,
                runs: s.runs,
                recovered: s.recovered,
                rate: s.rate,
                median_complexity: s.median_complexity,
            }
        })
        .collect();
    let mut names: Vec<(&str, &str)> = results
        .iter()
        .map(|r| (r.suite.as_str(), r.problem.as_str()))
        .collect();
    names.sort();
    names.dedup();
    let groups = group_summaries(&results);
    Report {
        problems: names.len(),
        results,
        summary,
        by_noise,
        groups,
    }
}

/// Run every selected problem at every noise level and seed.
pub fn run_suite(cfg: &SuiteConfig) -> Report {
    let specs = plan(cfg).unwrap_or_else(|e| panic!("{}", e));
    let results = run_plan(&specs, cfg, &mut |_| {});
    build_report(results, cfg)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn fmt_opt_f(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.*}", prec, x),
        _ => "-".to_string(),
    }
}

/// A human-readable report: one block per suite and noise level, then the
/// summary.
pub fn report_text(report: &Report, cfg: &SuiteConfig) -> String {
    let mut out = String::new();
    let suites: Vec<String> = {
        let mut s: Vec<String> = Vec::new();
        for r in &report.results {
            if !s.contains(&r.suite) {
                s.push(r.suite.clone());
            }
        }
        s
    };
    let seeds = cfg.seed_list();
    out.push_str(&format!(
        "rcsr-bench {}  suite={}  problems={}  seeds={}  metric={}\n",
        env!("CARGO_PKG_VERSION"),
        suites.join(","),
        report.problems,
        seeds
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(","),
        cfg.metric.name()
    ));
    out.push_str(&format!(
        "recovery criterion: test R\u{00b2} of the best front solution on fresh noiseless data >= {:.2}\n",
        cfg.threshold
    ));
    out.push_str(
        "oracle criterion: 1 - R\u{00b2} <= 2 (1 - oracle R\u{00b2}) + 1e-9, the oracle being the truth structure refit on the training rows\n",
    );
    if cfg.jobs > 1 {
        out.push_str(&format!(
            "note: --jobs {} runs share the CPU, so seconds and hit_s are contended\n",
            cfg.jobs
        ));
    }
    for g in &report.groups {
        match g.noise {
            Some(noise) => out.push_str(&format!(
                "\n-- {} noise {:.3} ({})\n",
                g.suite,
                noise,
                describe_noise(noise)
            )),
            None => out.push_str(&format!(
                "\n-- {} (signal tasks, no noise sweep; selection on validation rows)\n",
                g.suite
            )),
        }
        let rows: Vec<&RunResult> = report
            .results
            .iter()
            .filter(|r| r.suite == g.suite && r.noise == g.noise)
            .collect();
        if g.noise.is_some() {
            out.push_str(&format!(
                "{:<18} {:>4} {:>4} {:>4} {:>8} {:>4} {:>5} {:>8} {:>8} {:>12} {:>8}\n",
                "problem", "seed", "rec", "orc", "R2", "cx", "exact", "hit_s", "front_hv", "train_error", "sec"
            ));
            for r in rows {
                out.push_str(&format!(
                    "{:<18} {:>4} {:>4} {:>4} {:>8.4} {:>4} {:>5} {:>8} {:>8.4} {:>12.3e} {:>8.2}\n",
                    r.problem,
                    r.seed,
                    if r.recovered { "yes" } else { "no" },
                    match r.oracle_recovered {
                        Some(true) => "yes",
                        Some(false) => "no",
                        None => "-",
                    },
                    r.test_r2,
                    r.complexity,
                    r.exact_cx.map_or("-".to_string(), |c| c.to_string()),
                    fmt_opt_f(r.hit_s, 2),
                    r.front_hv,
                    r.train_error,
                    r.seconds
                ));
            }
        } else {
            out.push_str(&format!(
                "{:<18} {:>4} {:>4} {:>8} {:>4} {:>7} {:>8} {:>8}\n",
                "task", "seed", "sig", "signalR2", "cx", "distrct", "hit_s", "sec"
            ));
            for r in rows {
                out.push_str(&format!(
                    "{:<18} {:>4} {:>4} {:>8.4} {:>4} {:>7} {:>8} {:>8.2}\n",
                    r.problem,
                    r.seed,
                    if r.signal_recovered == Some(true) { "yes" } else { "no" },
                    r.signal_r2.unwrap_or(f64::NEG_INFINITY),
                    r.selected_cx.map_or("-".to_string(), |c| c.to_string()),
                    r.distractors_used.map_or("-".to_string(), |c| c.to_string()),
                    fmt_opt_f(r.hit_s, 2),
                    r.seconds
                ));
            }
        }
    }
    out.push_str("\n== summary ==\n");
    out.push_str(&format!(
        "recovered {}/{} ({:.1}%)   median complexity {:.0}\n",
        report.summary.recovered,
        report.summary.runs,
        100.0 * report.summary.rate,
        report.summary.median_complexity
    ));
    out.push_str("by suite and noise level (Wilson 95% intervals):\n");
    for g in &report.groups {
        let label = match g.noise {
            Some(n) => format!("{} {:.3}", g.suite, n),
            None => g.suite.clone(),
        };
        let ci = |k: usize, n: usize| {
            let (lo, hi) = stats::wilson(k, n);
            format!("{}/{} [{:.3}, {:.3}]", k, n, lo, hi)
        };
        out.push_str(&format!("  {:<18} runs {:>4}  recovered {}", label, g.runs, ci(g.recovered, g.runs)));
        if g.oracle_defined > 0 {
            out.push_str(&format!("  oracle {}", ci(g.oracle_recovered, g.oracle_defined)));
        }
        if g.oracle_failed > 0 {
            out.push_str(&format!("  oracle_failed {}", g.oracle_failed));
        }
        if g.signal_defined > 0 {
            out.push_str(&format!("  signal {}", ci(g.signal_recovered, g.signal_defined)));
        }
        out.push_str(&format!(
            "  median excess_cx {}  median hit_s {}  seconds {:.1}\n",
            fmt_opt_f(g.median_excess_cx, 1),
            fmt_opt_f(g.median_hit_s, 2),
            g.seconds
        ));
    }
    out
}

fn describe_noise(n: f64) -> String {
    if n <= 0.0 {
        "noiseless".to_string()
    } else {
        format!("{:.0}% relative noise", 100.0 * n)
    }
}

fn json_number(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

fn json_opt<T: ToString>(v: Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

fn json_opt_f(v: Option<f64>) -> String {
    v.map_or("null".to_string(), json_number)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn git_describe() -> String {
    static GIT: OnceLock<String> = OnceLock::new();
    GIT.get_or_init(|| {
        let dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["describe", "--always", "--dirty"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
    .clone()
}

/// The `config` object: everything that must match for two files to be
/// comparable, then the search knobs and the build, which may differ.
pub fn config_json(cfg: &SuiteConfig) -> String {
    let k = &cfg.knobs;
    let fit = match k.fit_strategy {
        None => "exact".to_string(),
        Some(FitChoice::Default) => "default".to_string(),
        Some(FitChoice::Strategy(s)) => s.name().to_string(),
    };
    let opt_u = |v: Option<usize>| json_opt(v);
    format!(
        "{{\"generator\":\"{}\",\"threshold\":{},\"signal_threshold\":{},\"knobs\":{{\"generations\":{},\"islands\":{},\"pop\":{},\"metric\":\"{}\",\"time_budget\":{},\"front_parents\":{},\"fit_strategy\":\"{}\",\"fit_iters\":{},\"ops\":{},\"max_complexity\":{},\"patience\":{},\"migration_interval\":{},\"restart_interval\":{},\"variation\":{},\"dispatch\":{},\"prune\":{}}},\"build\":{{\"version\":\"{}\",\"cuda\":{},\"backend\":\"{}\",\"jobs\":{},\"threads\":{},\"git\":\"{}\"}}}}",
        GENERATOR_VERSION,
        json_number(cfg.threshold),
        json_number(SIGNAL_THRESHOLD),
        cfg.generations,
        cfg.islands,
        cfg.pop,
        cfg.metric.name(),
        json_opt_f(cfg.time_budget),
        cfg.front_parents,
        fit,
        opt_u(k.fit_iters),
        k.ops.as_ref().map_or("null".to_string(), |o| format!("\"{}\"", json_escape(o))),
        opt_u(k.max_complexity),
        opt_u(k.patience),
        opt_u(k.migration_interval),
        opt_u(k.restart_interval),
        k.variation.map_or("null".to_string(), |v| format!("\"{}\"", v.as_str())),
        k.dispatch.map_or("null".to_string(), |v| format!("\"{}\"", v.name())),
        k.prune.map_or("null".to_string(), |v| v.to_string()),
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "cuda"),
        cfg.backend.name(),
        cfg.jobs,
        cfg.threads,
        json_escape(&git_describe()),
    )
}

/// The fields of one run, without braces.
fn run_fields(r: &RunResult) -> String {
    format!(
        "\"suite\": \"{}\", \"problem\": \"{}\", \"noise\": {}, \"seed\": {}, \"recovered\": {}, \"test_r2\": {}, \"complexity\": {}, \"train_error\": {}, \"test_rmse\": {}, \"seconds\": {}, \"expression\": \"{}\", \"truth_cx\": {}, \"generations_run\": {}, \"stop_reason\": \"{}\", \"backend\": \"{}\", \"oracle_r2\": {}, \"oracle_recovered\": {}, \"exact_cx\": {}, \"excess_cx\": {}, \"front_hv\": {}, \"hit_gen\": {}, \"hit_s\": {}, \"hit_evals\": {}, \"oracle_hit_gen\": {}, \"oracle_hit_s\": {}, \"oracle_hit_evals\": {}, \"signal_r2\": {}, \"signal_recovered\": {}, \"selected_cx\": {}, \"distractors_used\": {}, \"fit_attempts\": {}, \"fit_gpu\": {}, \"fit_fallback\": {}, \"prune_s\": {}",
        json_escape(&r.suite),
        json_escape(&r.problem),
        json_opt_f(r.noise),
        r.seed,
        r.recovered,
        json_number(r.test_r2),
        r.complexity,
        json_number(r.train_error),
        json_number(r.test_rmse),
        json_number(r.seconds),
        json_escape(&r.expression),
        r.truth_cx,
        r.generations_run,
        json_escape(&r.stop_reason),
        json_escape(&r.backend),
        json_opt_f(r.oracle_r2),
        json_opt(r.oracle_recovered),
        json_opt(r.exact_cx),
        json_opt(r.excess_cx),
        json_number(r.front_hv),
        json_opt(r.hit_gen),
        json_opt_f(r.hit_s),
        json_opt(r.hit_evals),
        json_opt(r.oracle_hit_gen),
        json_opt_f(r.oracle_hit_s),
        json_opt(r.oracle_hit_evals),
        json_opt_f(r.signal_r2),
        json_opt(r.signal_recovered),
        json_opt(r.selected_cx),
        json_opt(r.distractors_used),
        json_opt(r.fit_attempts),
        json_opt(r.fit_gpu),
        json_opt(r.fit_fallback),
        json_number(r.prune_s),
    )
}

/// One NDJSON run line (no trailing newline).
pub fn run_ndjson(r: &RunResult, cfg: &SuiteConfig) -> String {
    format!(
        "{{\"type\": \"run\", {}, \"config\": {}}}",
        run_fields(r),
        config_json(cfg)
    )
}

fn group_json(g: &GroupSummary) -> String {
    let ci = |k: usize, n: usize| {
        let (lo, hi) = stats::wilson(k, n);
        format!("{{\"hits\": {}, \"of\": {}, \"wilson_lo\": {}, \"wilson_hi\": {}}}", k, n, lo, hi)
    };
    format!(
        "{{\"suite\": \"{}\", \"noise\": {}, \"runs\": {}, \"recovered\": {}, \"oracle_recovered\": {}, \"oracle_failed\": {}, \"signal_recovered\": {}, \"median_excess_cx\": {}, \"median_hit_s\": {}, \"seconds\": {}}}",
        json_escape(&g.suite),
        json_opt_f(g.noise),
        g.runs,
        ci(g.recovered, g.runs),
        ci(g.oracle_recovered, g.oracle_defined),
        g.oracle_failed,
        ci(g.signal_recovered, g.signal_defined),
        json_opt_f(g.median_excess_cx),
        json_opt_f(g.median_hit_s),
        json_number(g.seconds)
    )
}

/// The closing NDJSON line.
pub fn summary_ndjson(report: &Report, cfg: &SuiteConfig) -> String {
    let groups: Vec<String> = report.groups.iter().map(group_json).collect();
    format!(
        "{{\"type\": \"summary\", \"runs\": {}, \"recovered\": {}, \"groups\": [{}], \"config\": {}}}",
        report.summary.runs,
        report.summary.recovered,
        groups.join(", "),
        config_json(cfg)
    )
}

/// The same report as a single JSON object (stdout-friendly).
pub fn report_json(report: &Report, cfg: &SuiteConfig) -> String {
    let mut out = String::new();
    let suites: Vec<String> = cfg.suite_names();
    out.push_str("{\n");
    out.push_str("\"tool\": \"rcsr-bench\",\n");
    out.push_str(&format!("\"version\": \"{}\",\n", env!("CARGO_PKG_VERSION")));
    out.push_str(&format!("\"suite\": \"{}\",\n", json_escape(&suites.join(","))));
    out.push_str(&format!("\"config\": {},\n", config_json(cfg)));
    out.push_str(&format!("\"tiny\": {},\n", cfg.tiny));
    out.push_str(&format!("\"seed\": {},\n", cfg.seed));
    out.push_str(&format!(
        "\"seeds\": [{}],\n",
        cfg.seed_list()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!("\"metric\": \"{}\",\n", cfg.metric.name()));
    out.push_str(&format!(
        "\"recovery_threshold\": {},\n",
        json_number(cfg.threshold)
    ));
    out.push_str(&format!("\"problems\": {},\n", report.problems));
    out.push_str(&format!("\"runs\": {},\n", report.summary.runs));
    out.push_str("\"noise_levels\": [");
    for (i, nv) in cfg.noise.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&json_number(*nv));
    }
    out.push_str("],\n");

    out.push_str("\"results\": [\n");
    for (i, r) in report.results.iter().enumerate() {
        out.push_str(&format!(
            "  {{{}}}{}\n",
            run_fields(r),
            if i + 1 < report.results.len() { "," } else { "" }
        ));
    }
    out.push_str("],\n");

    out.push_str("\"summary\": {\n");
    out.push_str(&format!(
        "  \"runs\": {}, \"recovered\": {}, \"rate\": {}, \"median_complexity\": {},\n",
        report.summary.runs,
        report.summary.recovered,
        json_number(report.summary.rate),
        json_number(report.summary.median_complexity)
    ));
    out.push_str("  \"by_noise\": [\n");
    for (i, s) in report.by_noise.iter().enumerate() {
        out.push_str(&format!(
            "    {{\"noise\": {}, \"runs\": {}, \"recovered\": {}, \"rate\": {}, \"median_complexity\": {}}}{}\n",
            json_number(s.noise),
            s.runs,
            s.recovered,
            json_number(s.rate),
            json_number(s.median_complexity),
            if i + 1 < report.by_noise.len() { "," } else { "" }
        ));
    }
    out.push_str("  ],\n");
    out.push_str("  \"groups\": [\n");
    for (i, g) in report.groups.iter().enumerate() {
        out.push_str(&format!(
            "    {}{}\n",
            group_json(g),
            if i + 1 < report.groups.len() { "," } else { "" }
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n}\n");
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Op::{Add, Square};

    #[test]
    fn suite_has_at_least_twenty_problems() {
        let ps = known_truth();
        assert!(ps.len() >= 20, "only {} problems", ps.len());
        // Names are unique and every problem has one range per variable.
        let mut names: Vec<&str> = ps.iter().map(|p| p.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate problem names");
        for p in &ps {
            assert_eq!(
                p.vars.len(),
                p.ranges.len(),
                "{}: {} vars vs {} ranges",
                p.name,
                p.vars.len(),
                p.ranges.len()
            );
        }
    }

    #[test]
    fn tiny_subset_is_fast_and_small() {
        let tiny: Vec<Problem> = known_truth().into_iter().filter(|p| p.tiny).collect();
        assert!(!tiny.is_empty(), "no tiny problems");
        assert!(tiny.len() < known_truth().len());
    }

    /// Every generator yields finite data of the stated shape, and the truth
    /// formula reproduces the noiseless target to machine precision.
    #[test]
    fn generators_are_finite_and_match_truth() {
        for p in known_truth() {
            let s = p.noiseless(p.n_train);
            assert_eq!(s.len(), p.n_train, "{}: row count", p.name);
            for row in &s.x {
                assert_eq!(row.len(), p.vars.len(), "{}: input width", p.name);
            }
            assert!(s.x.iter().flatten().all(|v| v.is_finite()), "{}: x", p.name);
            assert!(s.y.iter().all(|v| v.is_finite()), "{}: y", p.name);
            for (row, &y) in s.x.iter().zip(&s.y) {
                let t = expr::eval(&p.truth, row);
                assert!(t.is_finite(), "{}: truth not finite", p.name);
                assert!(
                    (t - y).abs() <= 1e-12 * (1.0 + t.abs()),
                    "{}: truth {t} != target {y}",
                    p.name
                );
            }
        }
    }

    #[test]
    fn noise_changes_the_target_but_keeps_finite_data() {
        for p in known_truth() {
            let (clean, _) = p.generate(0.0);
            let (noisy, test) = p.generate(0.1);
            assert!(noisy.y.iter().all(|v| v.is_finite()), "{}", p.name);
            assert!(test.y.iter().all(|v| v.is_finite()), "{}", p.name);
            // The test target is always noiseless: it matches the truth exactly.
            for (row, &y) in test.x.iter().zip(&test.y) {
                let t = expr::eval(&p.truth, row);
                assert!(
                    (t - y).abs() <= 1e-12 * (1.0 + t.abs()),
                    "{}: test target is not noiseless",
                    p.name
                );
            }
            let differs = clean.y.iter().zip(&noisy.y).any(|(a, b)| (a - b).abs() > 1e-15);
            assert!(differs, "{}: noise had no effect", p.name);
        }
    }

    #[test]
    fn data_is_reproducible() {
        let a = Problem::new(
            "tmp",
            &["x"],
            u2(Add, u1(Square, x(0)), c(1.0)),
            &[(-1.0, 1.0)],
            true,
        );
        let (t1, e1) = a.generate(0.05);
        let (t2, e2) = a.generate(0.05);
        assert_eq!(t1.x, t2.x);
        assert_eq!(t1.y, t2.y);
        assert_eq!(e1.y, e2.y);
    }

    #[test]
    fn r_squared_matches_hand_computation() {
        let y = [1.0, 2.0, 3.0, 4.0];
        assert!((r_squared(&y, &y) - 1.0).abs() < 1e-12);
        // Predicting the mean gives R² = 0.
        let m = [2.5, 2.5, 2.5, 2.5];
        assert!(r_squared(&m, &y).abs() < 1e-12);
        // A perfect prediction on a constant target is 1, else 0.
        assert_eq!(r_squared(&[2.0, 2.0], &[2.0, 2.0]), 1.0);
        assert_eq!(r_squared(&[3.0, 3.0], &[2.0, 2.0]), 0.0);
        assert!(r_squared(&[f64::NAN], &[1.0]).is_infinite());
    }

    #[test]
    fn parse_and_run_tiny_end_to_end() {
        let cfg = SuiteConfig {
            noise: vec![0.0],
            generations: 40,
            islands: 1,
            pop: 100,
            seed: 1,
            tiny: true,
            ..Default::default()
        };
        let report = run_suite(&cfg);
        assert_eq!(report.results.len(), report.problems);
        assert!(report.problems >= 1);
        assert!(report.summary.runs == report.results.len());
        assert!(report.summary.rate >= 0.0 && report.summary.rate <= 1.0);
        // The suite's easiest law must be found on noiseless data.
        let line = report
            .results
            .iter()
            .find(|r| r.problem == "line")
            .expect("line problem missing");
        assert!(line.recovered, "line R² = {}", line.test_r2);
        assert!(line.test_r2 >= DEFAULT_THRESHOLD);
    }

    #[test]
    fn bench_truth_cx_matches_tree() {
        let mut n = 0;
        for p in known_truth().into_iter().chain(bench_suites::hard()) {
            assert_eq!(p.truth_cx, expr::complexity(&p.truth), "{}", p.name);
            n += 1;
        }
        assert_eq!(n, 45);
        assert_eq!(bench_suites::hard().len(), 20);
        for t in bench_suites::signal() {
            assert_eq!(t.truth_cx, expr::complexity(&t.truth), "{}", t.name);
        }
    }

    #[test]
    fn bench_oracle_is_exact_at_noise_zero() {
        for p in known_truth().into_iter().chain(bench_suites::hard()) {
            let (train, test) = p.generate(0.0);
            let o = oracle_r2(&p, &train, &test);
            assert!(o >= 1.0 - 1e-12, "{}: oracle R2 {}", p.name, o);
        }
    }

    #[test]
    fn hard_data_shape_and_distractors() {
        for p in bench_suites::hard() {
            let (train, test) = p.generate(0.0);
            assert_eq!((train.len(), test.len()), (100, 500), "{}", p.name);
            assert!(train.y.iter().chain(&test.y).all(|v| v.is_finite()), "{}", p.name);
            for d in &p.distractors {
                assert!(!uses_input(&p.truth, *d), "{}: distractor {} is used", p.name, d);
            }
        }
    }

    #[test]
    fn bench_signal_oracle_r2() {
        for t in bench_suites::signal() {
            let d = t.generate();
            assert_eq!(d.train.len(), 2000);
            assert_eq!(d.val.len(), 1000);
            assert_eq!(d.test.len(), 2000);
            assert_eq!(d.train.x[0].len(), t.vars.len(), "{}", t.name);
            let ratio = d.train_signal_ratio();
            assert!(
                (ratio - t.oracle_r2).abs() < 0.02,
                "{}: realised {} vs {}",
                t.name,
                ratio,
                t.oracle_r2
            );
            let again = t.generate();
            assert_eq!(d.train.y, again.train.y, "{}: not reproducible", t.name);
            for (row, &s) in d.test.x.iter().zip(&d.s_test) {
                assert!((expr::eval(&t.truth, row) - s).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn signal_ar_rows_stay_in_time_order() {
        let t = bench_suites::signal().into_iter().find(|t| t.name == "sig_ar").unwrap();
        assert!(t.ordered);
        let d = t.generate();
        let lag1 = |col: usize, phi_hint: f64| {
            let xs: Vec<f64> = d.train.x.iter().map(|r| r[col]).collect();
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            let num: f64 = xs.windows(2).map(|w| (w[0] - m) * (w[1] - m)).sum();
            let den: f64 = xs.iter().map(|v| (v - m) * (v - m)).sum();
            assert!((num / den - phi_hint).abs() < 0.1, "col {}: {}", col, num / den);
        };
        lag1(0, 0.5);
        lag1(1, 0.95);
    }

    #[test]
    fn plan_expands_suites_seeds_and_filters() {
        let cfg = SuiteConfig {
            noise: vec![0.0, 0.1],
            suites: vec!["hard".to_string(), "signal".to_string()],
            problems: vec!["nguyen9".to_string(), "sig_ar".to_string()],
            seeds: vec![1, 2],
            ..Default::default()
        };
        let specs = plan(&cfg).unwrap();
        assert_eq!(specs.len(), 2 * 2 + 2);
        assert!(specs.iter().filter(|s| s.suite == "signal").all(|s| s.noise.is_none()));
        let bad = SuiteConfig {
            problems: vec!["nope".to_string()],
            ..Default::default()
        };
        assert!(plan(&bad).is_err());
        let all = SuiteConfig {
            suites: vec!["all".to_string()],
            noise: vec![0.0],
            ..Default::default()
        };
        assert_eq!(plan(&all).unwrap().len(), 25 + 20 + 3);
    }

    #[test]
    fn run_problem_reports_hits_and_oracle_fields() {
        let cfg = SuiteConfig {
            noise: vec![0.0],
            generations: 40,
            islands: 1,
            pop: 100,
            seed: 1,
            backend: BackendChoice::Cpu,
            ..Default::default()
        };
        let p = known_truth().into_iter().find(|p| p.name == "line").unwrap();
        let r = run_problem(&p, 0.0, 1, &cfg);
        assert!(r.recovered && r.oracle_recovered == Some(true));
        assert_eq!(r.backend, "cpu");
        assert!(r.hit_gen.is_some() && r.hit_s.is_some());
        assert_eq!(r.hit_evals, r.hit_gen.map(|g| g * 100));
        assert!(r.oracle_hit_gen.is_some());
        assert!(r.exact_cx.is_some() && r.excess_cx.is_some());
        assert!(r.front_hv > 0.0);
        assert!(r.generations_run >= 1);
        let line = run_ndjson(&r, &cfg);
        assert!(!line.contains('\n'));
        let j = stats::parse_json(&line).unwrap();
        assert_eq!(j.get("type").and_then(|t| t.as_str()), Some("run"));
        assert!(j.get("config").and_then(|c| c.get("knobs")).is_some());
    }

    #[test]
    fn run_signal_reports_selection_fields() {
        let cfg = SuiteConfig {
            generations: 5,
            islands: 1,
            pop: 40,
            seed: 1,
            backend: BackendChoice::Cpu,
            ..Default::default()
        };
        let t = bench_suites::signal().remove(0);
        let r = run_signal(&t, 1, &cfg);
        assert_eq!(r.suite, "signal");
        assert!(r.noise.is_none());
        assert!(r.signal_r2.is_some() && r.signal_recovered.is_some());
        assert!(r.selected_cx.is_some() && r.distractors_used.is_some());
        assert!(r.oracle_recovered.is_none());
    }
}
