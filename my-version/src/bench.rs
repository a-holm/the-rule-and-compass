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

use crate::data::Sample;
use crate::expr::{self, constant, input, node, Node, Op};
use crate::metrics::Metric;
use crate::rng::Mt;
use crate::search::{self, Config};
use std::time::Instant;

/// Seed for the data generator. Fixed so the same problem always yields the
/// same points, independent of the search seed; `--seed` drives the engine only.
const DATA_SEED: u32 = 0x0BEE_F001;
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
            truth,
            ranges: ranges.to_vec(),
            n_train: N_TRAIN,
            n_test: N_TEST,
            tiny,
        }
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
fn gaussian(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// FNV-1a over the problem name, so every problem gets its own stream.
fn fnv(bytes: &[u8]) -> u32 {
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

fn c(v: f64) -> Node {
    constant(v)
}
fn x(i: usize) -> Node {
    input(i)
}
fn u1(op: Op, a: Node) -> Node {
    node(op, vec![a])
}
fn u2(op: Op, a: Node, b: Node) -> Node {
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
    /// Restrict to the fast [`Problem::tiny`] subset.
    pub tiny: bool,
    /// Recovery threshold on the test `R²`.
    pub threshold: f64,
    /// Optional per-run wall-clock budget.
    pub time_budget: Option<f64>,
    /// #76: let the Pareto front supply 15 % of parents.
    pub front_parents: bool,
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
        }
    }
}

/// The outcome of one problem × noise run.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub problem: String,
    pub noise: f64,
    /// Selected solution's test `R²` ≥ threshold.
    pub recovered: bool,
    /// Test `R²` of the selected solution on fresh noiseless data.
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
    /// Number of problems actually run (after the `tiny` filter).
    pub problems: usize,
    pub results: Vec<RunResult>,
    pub summary: Summary,
    pub by_noise: Vec<NoiseSummary>,
}

impl Report {
    /// Results for one noise level, in problem order.
    pub fn for_noise(&self, noise: f64) -> Vec<&RunResult> {
        self.results
            .iter()
            .filter(|r| (r.noise - noise).abs() <= 1e-12)
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

/// Run one problem at one noise level and score recovery.
pub fn run_one(p: &Problem, noise: f64, cfg: &SuiteConfig) -> RunResult {
    let (train, test) = p.generate(noise);

    let mut sc = Config::default();
    sc.generations = cfg.generations.max(1);
    sc.islands = cfg.islands.max(1);
    sc.pop = cfg.pop.max(2);
    sc.metric = cfg.metric;
    sc.seed = cfg.seed;
    sc.time_budget = cfg.time_budget;
    sc.front_parents = cfg.front_parents; // #76

    let t0 = Instant::now();
    let out = search::run(&sc, &train, &test);
    let seconds = t0.elapsed().as_secs_f64();

    // Scan the Pareto front in complexity order and keep the member with the
    // highest test R²; ties therefore go to the simplest expression.
    let mut best: Option<RunResult> = None;
    for sol in out.archive.sorted() {
        let pred: Vec<f64> = test.x.iter().map(|row| expr::eval(&sol.tree, row)).collect();
        let r2 = r_squared(&pred, &test.y);
        let improve = best.as_ref().map(|b| r2 > b.test_r2).unwrap_or(true);
        if improve {
            best = Some(RunResult {
                problem: p.name.to_string(),
                noise,
                recovered: false,
                test_r2: r2,
                train_error: sol.error,
                test_rmse: rmse(&pred, &test.y),
                complexity: sol.complexity,
                expression: expr::to_string(&sol.tree, &p.vars),
                seconds,
            });
        }
    }
    let mut r = best.unwrap_or(RunResult {
        problem: p.name.to_string(),
        noise,
        recovered: false,
        test_r2: f64::NEG_INFINITY,
        train_error: f64::INFINITY,
        test_rmse: f64::INFINITY,
        complexity: 0,
        expression: String::new(),
        seconds,
    });
    r.recovered = r.test_r2 >= cfg.threshold;
    r
}

/// Run every selected problem at every noise level.
pub fn run_suite(cfg: &SuiteConfig) -> Report {
    let problems: Vec<Problem> = known_truth()
        .into_iter()
        .filter(|p| !cfg.tiny || p.tiny)
        .collect();
    let mut results = Vec::with_capacity(problems.len() * cfg.noise.len());
    for &noise in &cfg.noise {
        for p in &problems {
            results.push(run_one(p, noise, cfg));
        }
    }
    let summary = stats(&results);
    let by_noise = cfg
        .noise
        .iter()
        .map(|&nv| {
            let sub: Vec<RunResult> = results
                .iter()
                .filter(|r| (r.noise - nv).abs() <= 1e-12)
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
    Report {
        problems: problems.len(),
        results,
        summary,
        by_noise,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// A human-readable report: one block per noise level, then the summary.
pub fn report_text(report: &Report, cfg: &SuiteConfig) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "rcsr-bench {}  suite=known-truth  problems={}  seed={}  metric={}\n",
        env!("CARGO_PKG_VERSION"),
        report.problems,
        cfg.seed,
        cfg.metric.name()
    ));
    out.push_str(&format!(
        "recovery criterion: test R\u{00b2} of the best front solution on fresh noiseless data >= {:.2}\n",
        cfg.threshold
    ));
    for &noise in &cfg.noise {
        out.push_str(&format!(
            "\n-- noise {:.3} ({})\n",
            noise,
            describe_noise(noise)
        ));
        out.push_str(&format!(
            "{:<20} {:>4} {:>4} {:>10} {:>4} {:>12} {:>8}\n",
            "problem", "rec", "R2", "test_rmse", "cx", "train_error", "sec"
        ));
        for r in report.for_noise(noise) {
            out.push_str(&format!(
                "{:<20} {:>4} {:>4.3} {:>10.3e} {:>4} {:>12.3e} {:>8.2}\n",
                r.problem,
                if r.recovered { "yes" } else { "no" },
                r.test_r2,
                r.test_rmse,
                r.complexity,
                r.train_error,
                r.seconds
            ));
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
    out.push_str("by noise level:\n");
    for s in &report.by_noise {
        out.push_str(&format!(
            "  {:<5.3}  {:>3} runs  {:>3} recovered ({:>5.1}%)  median complexity {:.0}\n",
            s.noise,
            s.runs,
            s.recovered,
            100.0 * s.rate,
            s.median_complexity
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

/// The same report as a single JSON object (stdout-friendly).
pub fn report_json(report: &Report, cfg: &SuiteConfig) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("\"tool\": \"rcsr-bench\",\n");
    out.push_str(&format!("\"version\": \"{}\",\n", env!("CARGO_PKG_VERSION")));
    out.push_str("\"suite\": \"known-truth\",\n");
    out.push_str(&format!("\"tiny\": {},\n", cfg.tiny));
    out.push_str(&format!("\"seed\": {},\n", cfg.seed));
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
            "  {{\"problem\": \"{}\", \"noise\": {}, \"recovered\": {}, \"test_r2\": {}, \"complexity\": {}, \"train_error\": {}, \"test_rmse\": {}, \"seconds\": {}, \"expression\": \"{}\"}}{}\n",
            json_escape(&r.problem),
            json_number(r.noise),
            r.recovered,
            json_number(r.test_r2),
            r.complexity,
            json_number(r.train_error),
            json_number(r.test_rmse),
            json_number(r.seconds),
            json_escape(&r.expression),
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
}
