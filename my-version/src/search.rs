//! Island-model, multi-objective evolutionary search.
//!
//! A population of expression individuals evolves on separate islands, tracked
//! against a global Pareto front over (error, complexity), with generations,
//! migration and restarts.
//!
//! Optimisation notes (see `docs/optimisation.md`):
//! * offspring constants are fitted with Levenberg–Marquardt, not perturbed
//!   randomly, so a good *shape* is immediately usable;
//! * each island evolves on its own thread, so islands scale across cores;
//! * structurally identical offspring are discarded before evaluation, so the
//!   engine never pays twice for the same expression;
//! * selection uses lexicographic parsimony pressure (error first, size as a
//!   tie-break) so bloat is controlled without a hand-tuned penalty;
//! * the loop stops early when it has converged, or when a time budget is hit.

use crate::checkpoint::{Checkpoint, Individual, IslandState};
use crate::complexity::ComplexityModel;
use crate::data::Sample;
use crate::expr::{
    self, crossover_with, eval, mutate_with, random_tree_with, AggTable, Node, Op,
};
use crate::metrics::{score_weighted, Metric};
// #15: multi-output combined error and per-output constant fitting.
use crate::multi;
use crate::pareto::{Archive, Sol};
// #18: convergence statistics live in their own module; the search only calls
// `ProgressStats::observe` once per generation.
use crate::progress_stats::{Convergence, ProgressStats};
use crate::rng::Mt;
// #25: the dimensional algebra lives in its own module; the search only asks
// `Config::accepts_dimensions`.
use crate::units::Units;
use std::collections::HashSet;

#[derive(Clone)]
pub struct Config {
    pub generations: usize,
    pub islands: usize,
    pub pop: usize,
    /// Maximum configured complexity (see [`ComplexityModel`]).
    pub max_complexity: usize,
    /// How a candidate's complexity is measured and ranked on the front.
    pub complexity_model: ComplexityModel,
    pub metric: Metric,
    pub seed: u32,
    pub ops: Vec<Op>,
    pub migration_interval: usize,
    pub restart_interval: usize,
    pub init_depth: usize,
    /// Levenberg–Marquardt iterations for constant fitting (0 disables).
    pub fit_iters: usize,
    /// Discard structurally duplicated offspring before evaluating them.
    pub dedup: bool,
    /// Wall-clock budget in seconds; `None` means no limit.
    pub time_budget: Option<f64>,
    /// Stop after this many generations with no improvement (0 disables).
    pub patience: usize,
    /// Number of training rows used to score and fit candidates in a
    /// generation. `None`, or any value at least the training size, scores the
    /// full training sample. Values below 2 are rejected by the CLI.
    pub predictor_samples: Option<usize>,
    /// #25: declared units for the inputs (in input order) and the target.
    /// `None` disables dimensional analysis entirely.
    pub units: Option<Units>,
    /// #25: reject a candidate whose root dimension differs from the target's
    /// (infinite error) instead of merely noting the mismatch.
    pub require_dimensions: bool,
    /// Optional progress callback, invoked once per generation.
    pub progress: Option<ProgressFn>,
    // #27: use the SoA/AVX evaluator for candidate predictions. This is a
    // path choice, not a search setting: both paths must produce the same
    // errors, so it is ignored by `search_setting_differences`.
    pub simd: bool,
    /// #26: allow the batched GPU evaluator when the crate is built with
    /// `--features cuda` and a device is present. This is a performance choice
    /// only: it is not part of the checkpoint settings, and the numbers the
    /// user reads are always re-scored on the CPU in `f64`.
    pub gpu: bool,
    /// #26: print the one-line `# gpu: ...` note. The CLI clears this with
    /// `--quiet`, which promises an empty stderr for machine-readable output.
    pub gpu_notes: bool,
}

/// A per-generation progress report.
#[derive(Clone, Debug)]
pub struct Progress {
    pub generation: usize,
    pub total_generations: usize,
    pub best_error: f64,
    pub pareto_members: usize,
    // #18: confidence, stability, maturity and percent_converged.
    pub convergence: Convergence,
}

/// Progress callback type. `Send + Sync` so islands can report concurrently.
pub type ProgressFn = std::sync::Arc<dyn Fn(Progress) + Send + Sync>;

impl Default for Config {
    fn default() -> Self {
        Config {
            generations: 200,
            islands: 1,
            pop: 200,
            max_complexity: 30,
            complexity_model: ComplexityModel::default(),
            metric: Metric::Absolute,
            seed: 0,
            ops: Op::default_set(),
            migration_interval: 25,
            restart_interval: 60,
            init_depth: 3,
            fit_iters: 8,
            dedup: true,
            time_budget: None,
            patience: 40,
            predictor_samples: None,
            units: None,
            require_dimensions: false,
            progress: None,
            simd: true, // #27
            // #26: the library defaults to the CPU so every existing caller and
            // test keeps its trajectory; the `rcsr` CLI opts in automatically.
            // `--no-gpu` turns it off again.
            gpu: false,
            gpu_notes: true,
        }
    }
}

impl Config {
    /// The names of search settings that differ between two configurations.
    ///
    /// `generations` is deliberately excluded: it is a **budget**, not a
    /// property of the trajectory. Resuming may extend it (never shorten it),
    /// and every other setting must match the checkpoint exactly. `progress`
    /// is a reporting callback, not search state, and is ignored.
    pub fn search_setting_differences(&self, other: &Config) -> Vec<&'static str> {
        let mut diff = Vec::new();
        macro_rules! compare {
            ($field:ident, $name:expr) => {
                if self.$field != other.$field {
                    diff.push($name);
                }
            };
        }
        compare!(islands, "islands");
        compare!(pop, "pop");
        compare!(max_complexity, "max_complexity");
        compare!(complexity_model, "complexity model");
        compare!(metric, "metric");
        compare!(seed, "seed");
        compare!(ops, "ops");
        compare!(migration_interval, "migration_interval");
        compare!(restart_interval, "restart_interval");
        compare!(init_depth, "init_depth");
        compare!(fit_iters, "fit_iters");
        compare!(dedup, "dedup");
        if self.time_budget.map(f64::to_bits) != other.time_budget.map(f64::to_bits) {
            diff.push("time_budget");
        }
        compare!(patience, "patience");
        compare!(predictor_samples, "predictor_samples");
        // #25: units only change the trajectory when they are required; the
        // declaration itself is still part of the saved configuration.
        compare!(units, "units");
        compare!(require_dimensions, "require_dimensions");
        diff
    }

    /// True when both configurations describe the same search trajectory.
    /// The generation budget is excluded (see [`Config::search_setting_differences`]).
    pub fn same_search_settings(&self, other: &Config) -> bool {
        self.search_setting_differences(other).is_empty()
    }

    /// #25: the only dimensional question the search asks.
    ///
    /// A tree is accepted when no units are declared, when dimensional
    /// constraints are not required, or when its inferred root dimension
    /// equals the target's. Everything else is rejected.
    pub fn accepts_dimensions(&self, tree: &Node) -> bool {
        match &self.units {
            Some(units) if self.require_dimensions => units.accepts(tree),
            _ => true,
        }
    }
}

#[derive(Clone)]
struct Ind {
    tree: Node,
    error: f64,
    /// Configured complexity from [`Config::complexity_model`].
    complexity: f64,
}

impl Ind {
    // #15: `fit_iters` lets a multi-output sample fit one constant vector per
    // output. #25: a dimensionally inconsistent candidate is rejected outright
    // when units are required (the predicate is a no-op otherwise).
    // #27: `train` carries the sample and its cached SoA columns.
    fn evaluated(tree: Node, train: &EvalSample, cfg: &Config) -> Self {
        let tree = expr::simplify(&tree);
        let complexity = cfg.complexity_model.complexity(&tree);
        let error = if cfg.accepts_dimensions(&tree) {
            score_tree(&tree, train, cfg) // #27
        } else {
            f64::INFINITY
        };
        Ind {
            tree,
            error,
            complexity,
        }
    }

    /// #26: prepare an individual for a batched GPU score.
    ///
    /// Everything that depends on the CPU (simplification, complexity, the
    /// dimensional filter) happens here; only the fitness is left `NaN` for
    /// [`GpuSearch::score`](GpuSearch) to fill in. A candidate rejected by the
    /// dimensional filter keeps an infinite error, so the batching never
    /// resurrects it. NaN is the explicit "pending" marker used by the search.
    fn pending(tree: Node, cfg: &Config) -> Self {
        let tree = expr::simplify(&tree);
        let complexity = cfg.complexity_model.complexity(&tree);
        let error = if cfg.accepts_dimensions(&tree) {
            f64::NAN
        } else {
            f64::INFINITY
        };
        Ind {
            tree,
            error,
            complexity,
        }
    }

    fn reeval(&mut self, train: &EvalSample, cfg: &Config) {
        self.tree = expr::simplify(&self.tree);
        self.complexity = cfg.complexity_model.complexity(&self.tree);
        // #25: re-evaluation must not resurrect a rejected candidate.
        self.error = if cfg.accepts_dimensions(&self.tree) {
            score_tree(&self.tree, train, cfg) // #27
        } else {
            f64::INFINITY
        };
    }

    fn to_checkpoint(&self) -> Individual {
        Individual {
            tree: self.tree.clone(),
            error: self.error,
            complexity: self.complexity,
        }
    }

    fn from_checkpoint(ind: Individual) -> Self {
        Ind {
            tree: ind.tree,
            error: ind.error,
            complexity: ind.complexity,
        }
    }
}

/// #26: the optional batched evaluator used by the search loop.
///
/// The dataset is uploaded once per run (the full training sample), and one
/// [`Population`](crate::gpu::Population) is scored per generation in a single
/// kernel launch. Constants are still fitted on the CPU *before* the batch: the
/// GPU only supplies the fitness that selection consumes.
///
/// The kernel sums squared errors in `f32`; [`GpuSearch::score`] divides by the
/// row count so the search sees a mean squared error. For `--metric squared`
/// that is the configured objective (up to `f32` rounding); for the other
/// regression metrics it is a monotone ranking proxy. Either way it is only
/// used to guide the search: the archive and every reported number are rebuilt
/// on the full training sample with the CPU `f64` evaluator.
#[cfg(feature = "cuda")]
struct GpuSearch {
    ev: crate::gpu::GpuEvaluator,
    n_rows: usize,
}

#[cfg(feature = "cuda")]
impl GpuSearch {
    fn new(train: &Sample) -> Result<Self, String> {
        // A test/diagnostic switch: `RCSR_GPU=0` makes a CUDA build behave as
        // if no device were present, so the fallback path can be exercised on
        // a machine that has one.
        if matches!(
            std::env::var("RCSR_GPU").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("no")
        ) {
            return Err("RCSR_GPU=0 (device use disabled)".to_string());
        }
        let n_rows = train.len();
        let n_vars = train.x.first().map(|r| r.len()).unwrap_or(0);
        if n_rows == 0 || n_vars == 0 {
            return Err("empty training sample".to_string());
        }
        // The kernel wants column-major f32; the sample is row-major f64.
        let mut cols = vec![0.0f32; n_rows * n_vars];
        for (r, row) in train.x.iter().enumerate() {
            for (v, &x) in row.iter().enumerate().take(n_vars) {
                cols[v * n_rows + r] = x as f32;
            }
        }
        let y: Vec<f32> = train.y.iter().map(|&v| v as f32).collect();
        match crate::gpu::GpuEvaluator::new(&cols, n_rows, n_vars, &y) {
            Some(ev) => Ok(GpuSearch { ev, n_rows }),
            None => Err(format!(
                "no usable CUDA device ({})",
                crate::gpu::GpuEvaluator::device_name()
            )),
        }
    }

    /// Score every pending (`error.is_nan()`) individual in every island in one
    /// batched call. Candidates the kernel cannot compile (an operator with no
    /// GPU opcode) are scored on the CPU instead, so they never lose fitness.
    ///
    /// Any CUDA failure returns `Err`; the caller switches the backend off and
    /// re-scores the whole batch on the CPU, so a failed batch can never leave
    /// a wrong number behind.
    fn score(
        &mut self,
        islands: &mut [Vec<Ind>],
        train: &Sample,
        cfg: &Config,
    ) -> Result<(), String> {
        let pending: Vec<(usize, usize)> = islands
            .iter()
            .enumerate()
            .flat_map(|(i, isl)| {
                isl.iter()
                    .enumerate()
                    .filter(|(_, ind)| ind.error.is_nan())
                    .map(move |(j, _)| (i, j))
            })
            .collect();
        if pending.is_empty() {
            return Ok(());
        }

        let mut pop = crate::gpu::Population::new();
        let mut slots: Vec<(usize, usize)> = Vec::with_capacity(pending.len());
        let mut on_cpu: Vec<(usize, usize)> = Vec::new();
        for &(i, j) in &pending {
            if pop.add(&islands[i][j].tree) {
                slots.push((i, j));
            } else {
                on_cpu.push((i, j));
            }
        }

        if !pop.is_empty() {
            let (sse, ms) = self.ev.eval_population(&pop);
            if ms < 0.0 || sse.len() != pop.len() {
                return Err("CUDA batch evaluation failed".to_string());
            }
            let n = self.n_rows as f64;
            for (&(i, j), &s) in slots.iter().zip(sse.iter()) {
                islands[i][j].error = if s.is_finite() {
                    (s as f64) / n
                } else {
                    f64::INFINITY
                };
            }
        }
        for (i, j) in on_cpu {
            islands[i][j].reeval(train, cfg); // #25
        }
        Ok(())
    }
}

/// #26: score the pending individuals, falling back to the CPU on any error.
///
/// Returns whether the GPU remains usable. On failure the whole pending batch
/// is re-scored by [`Ind::reeval`], so the search continues with correct `f64`
/// numbers and a single visible `# gpu: unavailable (...)` note.
#[cfg(feature = "cuda")]
fn score_generation(
    gpu: &mut Option<GpuSearch>,
    used: &mut bool,
    islands: &mut [Vec<Ind>],
    train: &Sample,
    cfg: &Config,
) -> bool {
    let Some(g) = gpu.as_mut() else {
        return false;
    };
    match g.score(islands, train, cfg) {
        Ok(()) => {
            *used = true;
            true
        }
        Err(e) => {
            if cfg.gpu_notes {
                eprintln!("# gpu: unavailable ({})", e);
            }
            *gpu = None;
            for pop in islands.iter_mut() {
                for ind in pop.iter_mut() {
                    if ind.error.is_nan() {
                        ind.reeval(train, cfg); // #25
                    }
                }
            }
            false
        }
    }
}

/// #26: whether a run may use the GPU at all.
///
/// Restricted to the single-output, unweighted, full-sample regression path:
/// multi-output and implicit objectives need more than one prediction per row,
/// and `--predictor-samples` changes the rows every generation, so each of
/// those keeps the CPU evaluator.
#[cfg(feature = "cuda")]
fn gpu_blocker(cfg: &Config, train: &Sample) -> Option<&'static str> {
    if train.is_multi() {
        return Some("multi-output objective");
    }
    if !train.w.is_empty() {
        return Some("weighted rows");
    }
    if cfg.metric.is_implicit() {
        return Some("implicit objective");
    }
    if cfg.metric.is_classification() {
        return Some("classification metric");
    }
    if cfg.metric.is_parameter_aware() {
        return Some("parameter-aware metric");
    }
    if subsampling_active(cfg, train) {
        return Some("--predictor-samples changes the rows per generation");
    }
    None
}

pub fn eval_error(tree: &Node, s: &Sample, metric: Metric) -> f64 {
    // #27: keep the public entry point on the fast path; `--no-simd` goes
    // through a scalar `EvalSample` inside the search.
    eval_error_with(tree, s, metric, true)
}

/// #27: [`eval_error`] with an explicit evaluator choice.
///
/// `simd` compiles the tree once and evaluates whole SoA columns, falling back
/// to the scalar walk when [`crate::eval_soa::compile`] rejects an operator.
/// `simd = false` is exactly the historical per-row [`expr::eval`] walk. Both
/// paths return bit-identical errors for a supported tree; the choice cannot
/// change the search trajectory.
fn eval_error_with(tree: &Node, s: &Sample, metric: Metric, simd: bool) -> f64 {
    let data = EvalSample {
        sample: s,
        cols: if simd {
            Some(crate::eval_soa::columns(&s.x))
        } else {
            None
        },
    };
    eval_error_on(tree, &data, metric)
}

/// #27: a sample plus its SoA transpose.
///
/// The transpose is O(rows × vars) and is paid **once per sample**, not once
/// per candidate: every individual scored on the same table shares it. `cols`
/// is `None` when `--no-simd` asks for the scalar walk.
struct EvalSample<'a> {
    sample: &'a Sample,
    cols: Option<Vec<Vec<f64>>>,
}

impl<'a> EvalSample<'a> {
    fn new(sample: &'a Sample, cfg: &Config) -> Self {
        EvalSample {
            sample,
            cols: if cfg.simd {
                Some(crate::eval_soa::columns(&sample.x))
            } else {
                None
            },
        }
    }
}

fn eval_error_on(tree: &Node, data: &EvalSample, metric: Metric) -> f64 {
    let s = data.sample;
    if s.is_empty() {
        return f64::INFINITY;
    }
    // #14: an implicit metric measures the distance to the level set
    // |f| / ||grad f||. That needs f and its gradient at each row, so it is
    // scored from the tree and the row instead of a prediction vector. Every
    // other metric takes the original path, bit for bit.
    if metric.is_implicit() {
        return crate::implicit::score(tree, s, metric);
    }
    let pred = predictions(tree, data);
    // The parameter count lets the information criteria (AIC/AICc/BIC) work as
    // search metrics; it is ignored by every other metric. The weight vector is
    // empty for an unweighted sample, which takes the identical unweighted path.
    score_weighted(&pred, &s.y, &s.w, metric, expr::count_constants(tree))
}

/// #27: one candidate's prediction vector.
///
/// The fast path compiles the tree once and evaluates all rows in batches over
/// the sample's cached SoA columns. An unsupported operator anywhere in the
/// tree makes `compile` return `None`, and this call then takes the scalar
/// walk — transparently, for that tree only.
fn predictions(tree: &Node, data: &EvalSample) -> Vec<f64> {
    if let Some(cols) = &data.cols {
        if let Some(program) = crate::eval_soa::compile(tree) {
            let rows = data.sample.x.len();
            let mut pred = vec![0.0; rows];
            crate::eval_soa::eval_batch(&program, cols, rows, &mut pred);
            return pred;
        }
    }
    data.sample.x.iter().map(|row| eval(tree, row)).collect()
}

/// #14: fit a candidate's constants on the objective its metric actually
/// scores.
///
/// The implicit-derivative metric is scale-free. Fitting `f` against the zero
/// target with ordinary least squares would shrink every multiplicative
/// constant towards zero, collapsing the level set into the trivial `f == 0`;
/// the guard rightly rejects that candidate, so a good shape would be destroyed
/// before it could be scored. In this one case [`crate::implicit::fit`]
/// minimises the signed geometric residual `f / ||grad f||` instead. Every
/// other metric (including `implicit_residual`, which by definition scores
/// `|f|`) keeps the historical `fit_weighted` path bit for bit.
fn fit_constants(tree: &mut Node, cfg: &Config, train: &Sample) {
    if cfg.metric.is_implicit_derivative() {
        crate::implicit::fit(tree, &train.x, &train.w, cfg.fit_iters);
    } else {
        crate::fit::fit_weighted(tree, &train.x, &train.y, &train.w, cfg.fit_iters);
    }
}

/// #15: score a candidate on `s`, dispatching to the shared-shape multi-output
/// objective when more than one target is present. Single-output samples take
/// the SoA/scalar pair in [`eval_error_on`].
// #27: `data` carries the evaluator choice and the cached SoA columns; the
// multi-output path has its own per-output fit/metric and keeps its existing
// scalar walk.
fn score_tree(tree: &Node, data: &EvalSample, cfg: &Config) -> f64 {
    if data.sample.is_multi() {
        multi::combined_error(tree, data.sample, cfg.metric, cfg.fit_iters)
    } else {
        eval_error_on(tree, data, cfg.metric)
    }
}

/// True when candidate scoring should use a per-generation subset of the
/// training rows. A request at least as large as the training sample is a
/// no-op: the full sample is used, in its original order, so the search is
/// bit-for-bit identical to leaving the option unset.
fn subsampling_active(cfg: &Config, train: &Sample) -> bool {
    matches!(cfg.predictor_samples, Some(n) if n < train.len())
}

/// Seed of the dedicated RNG stream for one generation's row subset.
///
/// Kept separate from every other RNG in the engine: turning subsampling on or
/// off therefore changes only how candidates are scored, never the search's own
/// mutation, crossover or restart decisions.
fn subset_seed(seed: u32, generation: usize) -> u32 {
    seed ^ 0x9E37_79B9u32.wrapping_mul((generation as u32).wrapping_add(0x7F4A_7C15))
}

/// Draw `n` distinct row indices from `0 .. rows` without replacement.
///
/// The subset is a pure function of `(seed, generation)`: the same arguments
/// always return the same sorted indices, and consecutive generations draw a
/// fresh subset. A partial Fisher–Yates shuffle keeps the cost at O(n) in the
/// number of rows actually scored.
///
/// This is public so the sampling contract (distinct rows, deterministic for a
/// given seed and generation) can be tested directly.
pub fn generation_subset(rows: usize, n: usize, seed: u32, generation: usize) -> Vec<usize> {
    let n = n.min(rows);
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..rows).collect();
    let mut rng = Mt::new(subset_seed(seed, generation));
    for i in (rows - n..rows).rev() {
        let j = rng.usize_below(i + 1);
        idx.swap(i, j);
    }
    let mut chosen = idx[rows - n..].to_vec();
    chosen.sort_unstable();
    chosen
}

/// Build the sample a generation evolves on, or `None` when the full training
/// sample must be used. The caller passes the full sample in that case, which
/// preserves the original row order and floating-point summation.
fn training_for_generation(cfg: &Config, train: &Sample, generation: usize) -> Option<Sample> {
    if !subsampling_active(cfg, train) {
        return None;
    }
    let n = cfg.predictor_samples.unwrap_or(train.len());
    let idx = generation_subset(train.len(), n, cfg.seed, generation);
    Some(Sample {
        x: idx.iter().map(|&i| train.x[i].clone()).collect(),
        y: idx.iter().map(|&i| train.y[i]).collect(),
        w: if train.w.is_empty() {
            Vec::new()
        } else {
            idx.iter().map(|&i| train.w[i]).collect()
        },
        // #15: subset every output column, not just the first, so a
        // multi-output generation scores the same rows for every target.
        ys: if train.ys.is_empty() {
            Vec::new()
        } else {
            train
                .ys
                .iter()
                .map(|col| idx.iter().map(|&i| col[i]).collect())
                .collect()
        },
    })
}

/// Rebuild an archive from its members and the final populations, scoring every
/// expression on the **full** training sample. Used only when the generations
/// ran on subsets, so the returned front and every error derived from it are
/// full-data even though fitness was estimated.
fn full_data_archive(
    archive: &Archive,
    islands: &[Vec<Ind>],
    train: &EvalSample, // #27 sample + cached SoA columns
    cfg: &Config,       // #15 refit per-output constants; #25 dimensional filter
) -> Archive {
    let trees: Vec<&Node> = archive
        .items
        .iter()
        .map(|s| &s.tree)
        .chain(islands.iter().flat_map(|pop| pop.iter().map(|ind| &ind.tree)))
        .collect();
    let mut out = Archive::new(archive.cap);
    let mut seen: HashSet<u64> = HashSet::new();
    for tree in trees {
        if !seen.insert(expr::structural_hash(tree)) {
            continue;
        }
        let tree = expr::simplify(tree);
        // #15: combined error on a multi-output sample; #25: a rejected
        // candidate must not sneak back in on the full-data rebuild.
        let error = if cfg.accepts_dimensions(&tree) {
            score_tree(&tree, train, cfg) // #27
        } else {
            f64::INFINITY
        };
        out.try_insert(Sol {
            complexity: expr::complexity(&tree),
            cost: cfg.complexity_model.complexity(&tree),
            tree,
            error,
        });
    }
    out
}

/// Lexicographic parsimony pressure: better error wins; if the errors are equal
/// to within a relative tolerance, the smaller expression wins. This controls
/// bloat without a tunable penalty coefficient (Luke & Panait 2006).
fn better(a: &Ind, b: &Ind) -> bool {
    let tol = 1e-9 * (1.0 + a.error.abs().max(b.error.abs()));
    if a.error < b.error - tol {
        return true;
    }
    if b.error < a.error - tol {
        return false;
    }
    a.complexity < b.complexity
}

fn tournament(island: &[Ind], rng: &mut Mt, k: usize) -> usize {
    let n = island.len();
    if n == 0 {
        return 0;
    }
    let mut best = rng.usize_below(n);
    for _ in 1..k {
        let c = rng.usize_below(n);
        if better(&island[c], &island[best]) {
            best = c;
        }
    }
    best
}

// #77: seed the per-generation duplicate set with the elite (`next`) and every
// parent (`isl`), so a child that is an unchanged parent is rejected before the
// expensive fit and score.
fn seed_seen(seen: &mut HashSet<u64>, next: &[Ind], parents: &[Ind]) {
    for ind in next.iter().chain(parents) {
        seen.insert(expr::structural_hash(&ind.tree));
    }
}

/// Evolve one island by a single generation. Pure with respect to the rest of
/// the world, so it can run on its own thread.
fn evolve_island(
    isl: &mut Vec<Ind>,
    rng: &mut Mt,
    cfg: &Config,
    train: &EvalSample, // #27 sample + cached SoA columns
    nvars: usize,
    aggs: &AggTable,
    // #26: leave the fitness pending when the caller will batch it on the GPU.
    defer_gpu: bool,
) {
    for ind in isl.iter_mut() {
        if !ind.error.is_finite() {
            ind.reeval(train, cfg); // #25
        }
    }

    let pop = cfg.pop;
    let mut next: Vec<Ind> = Vec::with_capacity(pop);

    // Elitism: carry the best individual over unchanged.
    if let Some(best) = isl
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
    {
        next.push(best.clone());
    }

    let mut seen: HashSet<u64> = HashSet::with_capacity(pop * 2);
    // #77: parents are candidates too; seeding them here rejects an unchanged
    // child before it is fitted and scored.
    seed_seen(&mut seen, &next, isl);
    let mut attempts = 0usize;
    let max_attempts = pop * 50 + 50;

    while next.len() < pop && attempts < max_attempts {
        attempts += 1;
        let p1 = tournament(isl, rng, 4);
        let mut child = isl[p1].tree.clone();
        if rng.bool(0.5) {
            let p2 = tournament(isl, rng, 4);
            let parent2 = isl[p2].tree.clone();
            crossover_with(&mut child, &parent2, rng, Some(aggs));
        }
        mutate_with(&mut child, rng, nvars, &cfg.ops, Some(aggs));
        if rng.bool(0.5) {
            mutate_with(&mut child, rng, nvars, &cfg.ops, Some(aggs));
        }

        let mut tree = expr::simplify(&child);
        if cfg.complexity_model.complexity(&tree) > cfg.max_complexity as f64 {
            continue;
        }
        if cfg.dedup && !seen.insert(expr::structural_hash(&tree)) {
            continue;
        }

        // #15: a multi-output shape keeps the constant slots that only some
        // outputs need; it is fitted per output inside the evaluator.
        // #14: otherwise fit on the objective the metric actually scores.
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fit_constants(&mut tree, cfg, train.sample);
            tree = expr::simplify(&tree);
        }
        // #26: with the GPU batch active the fitness is filled in after every
        // island has been generated; the CPU still owns simplification,
        // fitting, de-duplication and the complexity/dimension checks.
        let ind = if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            Ind::evaluated(tree, train, cfg) // #25
        };
        if ind.complexity > cfg.max_complexity as f64 {
            continue;
        }
        next.push(ind);
    }

    // Fill any remainder with fresh random trees.
    while next.len() < pop {
        let mut tree = expr::simplify(&random_tree_with(
            rng,
            nvars,
            &cfg.ops,
            cfg.init_depth,
            Some(aggs),
        ));
        // #15: multi-output fits per output inside the evaluator.
        // #14: otherwise fit on the objective the metric actually scores.
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fit_constants(&mut tree, cfg, train.sample);
            tree = expr::simplify(&tree);
        }
        // #26: same deferral for the random fill.
        next.push(if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            Ind::evaluated(tree, train, cfg) // #25
        });
    }

    *isl = next;
}

/// A user-provided starting expression, after simplification, constant fitting
/// and evaluation on the training sample.
///
/// The tree is in the search's own (normalised input) space. Callers that show
/// formulas to a person should denormalise it first, exactly as they do for an
/// archive entry.
#[derive(Clone, Debug)]
pub struct SeedResult {
    pub tree: Node,
    /// Configured complexity from [`Config::complexity_model`].
    pub complexity: f64,
    /// Training error under [`Config::metric`].
    pub error: f64,
}

pub struct Outcome {
    pub archive: Archive,
    pub test_errors: Vec<f64>,
    pub generations: usize,
    pub converged_early: bool,
    /// The seeds that were accepted, in command-line order. Empty when the run
    /// was not seeded.
    pub seeds: Vec<SeedResult>,
    // #18: convergence statistics at the end of the run.
    pub convergence: Convergence,
}

/// Run a search once, from a fresh start.
///
/// This is the historical entry point. It delegates to [`run_resumable`] with
/// no checkpoint and no callback, so its behaviour is unchanged bit for bit.
pub fn run(cfg: &Config, train: &Sample, test: &Sample) -> Outcome {
    run_search(cfg, train, test, None, 0, &[], None)
        .unwrap_or_else(|e| panic!("search failed: {}", e))
}

/// Run a search from a fresh start with prior solutions injected.
///
/// Every seed is simplified, its constants are fitted on the training sample
/// and its error is evaluated before the first generation. It is then inserted
/// into the archive and into the initial population of **every** island, so an
/// island keeps the seed until migration even when it would not have found it
/// itself. Structurally identical seeds are inserted once.
///
/// A seed is an elite: it is inserted even when its complexity exceeds
/// `cfg.max_complexity`, which otherwise limits generated offspring only.
pub fn run_seeded(cfg: &Config, train: &Sample, test: &Sample, seeds: &[Node]) -> Outcome {
    run_search(cfg, train, test, None, 0, seeds, None)
        .unwrap_or_else(|e| panic!("search failed: {}", e))
}

/// Run a search, optionally continuing a checkpoint, and report snapshots.
///
/// A snapshot is handed to `on_checkpoint` after every `save_every`
/// generations (when non-zero) and once more before the final archive is
/// rebuilt, so a save always includes the state at a generation boundary.
/// The callback must not mutate the search; saving therefore cannot change the
/// result.
///
/// When `resume` is `Some`, the checkpoint's data fingerprint is checked
/// against `train`/`test` and its search settings must match `cfg`; the
/// generation budget in `cfg` may be larger than the saved one (to continue a
/// run), but never smaller. Any mismatch is a descriptive `Err`.
pub fn run_resumable(
    cfg: &Config,
    train: &Sample,
    test: &Sample,
    resume: Option<Checkpoint>,
    save_every: usize,
    on_checkpoint: impl FnMut(&Checkpoint),
) -> Result<Outcome, String> {
    run_resumable_seeded(
        cfg,
        train,
        test,
        resume,
        save_every,
        &[],
        on_checkpoint,
    )
}

/// [`run_resumable`] with prior solutions injected, exactly as
/// [`run_seeded`] describes.
///
/// Seeds are not part of the checkpoint format: on resume they are added to
/// the loaded populations and archive, so a caller that wants the identical
/// resumed trajectory should not pass them again.
pub fn run_resumable_seeded(
    cfg: &Config,
    train: &Sample,
    test: &Sample,
    resume: Option<Checkpoint>,
    save_every: usize,
    seeds: &[Node],
    mut on_checkpoint: impl FnMut(&Checkpoint),
) -> Result<Outcome, String> {
    run_search(
        cfg,
        train,
        test,
        resume,
        save_every,
        seeds,
        Some(&mut on_checkpoint),
    )
}

type CheckpointFn<'a> = &'a mut dyn FnMut(&Checkpoint);

#[allow(clippy::too_many_arguments)]
fn snapshot(
    cfg: &Config,
    gens_done: usize,
    stopped_early: bool,
    best_prev: f64,
    no_improve: usize,
    archive: &Archive,
    islands: &[Vec<Ind>],
    rngs: &[Mt],
    stagnant: &[usize],
    main_rng: &Mt,
    train: &Sample,
    test: &Sample,
) -> Checkpoint {
    let (words, idx) = main_rng.state();
    Checkpoint {
        config: cfg.clone(),
        generations: gens_done,
        stopped_early,
        best_prev,
        no_improve,
        train: crate::checkpoint::fingerprint(train),
        test: crate::checkpoint::fingerprint(test),
        archive: Archive {
            items: archive.items.clone(),
            cap: archive.cap,
        },
        islands: islands
            .iter()
            .zip(rngs.iter())
            .map(|(pop, rng)| {
                let (words, idx) = rng.state();
                IslandState {
                    rng: Mt::from_state(*words, idx),
                    population: pop.iter().map(Ind::to_checkpoint).collect(),
                }
            })
            .collect(),
        stagnant: stagnant.to_vec(),
        main_rng: Mt::from_state(*words, idx),
    }
}

fn emit(on_checkpoint: &mut Option<CheckpointFn<'_>>, cp: &Checkpoint) {
    if let Some(cb) = on_checkpoint.as_deref_mut() {
        cb(cp);
    }
}

/// Simplify each seed, fit its constants on the training sample and evaluate
/// it, in that order. A seed that is structurally identical to an earlier one
/// (after simplification and again after fitting) is dropped: the archive and
/// every island need it once.
fn prepare_seeds(
    cfg: &Config,
    train: &EvalSample, // #27 sample + cached SoA columns
    seeds: &[Node],
    aggs: &AggTable,
) -> Vec<Ind> {
    let mut out: Vec<Ind> = Vec::new();
    // Two sets: one for the simplified input form (so a repeated line is
    // dropped even before fitting) and one for the fitted tree (so two
    // different starts that converge to the same expression count once).
    let mut seen_input: HashSet<u64> = HashSet::new();
    let mut seen_fitted: HashSet<u64> = HashSet::new();
    for seed in seeds {
        let mut tree = expr::simplify(seed);
        // A parsed seed carries unresolved aggregates (`val = NaN`); resolve
        // them against the training sample before anything evaluates it.
        if expr::bind_aggregates(&mut tree, aggs).is_err() {
            continue;
        }
        if !seen_input.insert(expr::structural_hash(&tree)) {
            continue;
        }
        // #15: multi-output fits per output inside the evaluator.
        // #14: otherwise fit on the objective the metric actually scores.
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fit_constants(&mut tree, cfg, train.sample);
            tree = expr::simplify(&tree);
        }
        if !seen_fitted.insert(expr::structural_hash(&tree)) {
            continue;
        }
        out.push(Ind::evaluated(tree, train, cfg)); // #25
    }
    out
}

fn run_search(
    cfg: &Config,
    train: &Sample,
    test: &Sample,
    resume: Option<Checkpoint>,
    save_every: usize,
    seeds: &[Node],
    mut on_checkpoint: Option<CheckpointFn<'_>>,
) -> Result<Outcome, String> {
    let nvars = train.x.first().map(|r| r.len()).unwrap_or(0);
    // #26: a checkpoint written during a GPU run stores the f32 proxy fitness,
    // so a resumed run must rebuild the reported front on the CPU even when it
    // is resumed with `--no-gpu`.
    let resumed = resume.is_some();
    // The aggregates are a property of the training rows, computed once. They
    // are deliberately **not** rebuilt for a generation's row subset and never
    // read the test sample: a subset changes which rows score a candidate, not
    // what the dataset's statistics are.
    let aggs = AggTable::from_sample(train);
    let t0 = std::time::Instant::now();
    // #10: user-defined blocks are thread-local. Snapshot the registry here so
    // every island thread (and the initial population below) sees the same
    // definitions; the list is empty unless --function was supplied.
    let blocks = expr::current_blocks();

    // #26: one resident evaluator for the full training sample, used for whole
    // generations. Constants were already fitted on the CPU; this only chooses
    // where the tree evaluation runs. A missing feature, a missing device or an
    // unsupported objective leaves `defer_gpu` false and the CPU path runs.
    #[cfg(feature = "cuda")]
    let mut gpu: Option<GpuSearch> = if !cfg.gpu {
        None
    } else {
        match gpu_blocker(cfg, train) {
            Some(reason) => {
                if cfg.gpu_notes {
                    eprintln!("# gpu: off ({})", reason);
                }
                None
            }
            None => match GpuSearch::new(train) {
                Ok(g) => {
                    if cfg.gpu_notes {
                        eprintln!(
                            "# gpu: {} (one batch per generation; f32 fitness, reported numbers stay f64)",
                            crate::gpu::GpuEvaluator::device_name()
                        );
                    }
                    Some(g)
                }
                Err(e) => {
                    if cfg.gpu_notes {
                        eprintln!("# gpu: unavailable ({})", e);
                    }
                    None
                }
            },
        }
    };
    #[cfg(not(feature = "cuda"))]
    if cfg.gpu && cfg.gpu_notes {
        eprintln!("# gpu: unavailable (built without --features cuda)");
    }
    #[cfg(feature = "cuda")]
    let mut defer_gpu = gpu.is_some();
    #[cfg(not(feature = "cuda"))]
    let defer_gpu = false;
    #[cfg(feature = "cuda")]
    let mut gpu_used = false;
    #[cfg(not(feature = "cuda"))]
    let gpu_used = false;

    let mut rng: Mt;
    let mut archive: Archive;
    let mut rngs: Vec<Mt>;
    let mut islands: Vec<Vec<Ind>>;
    let mut stagnant: Vec<usize>;
    let mut gens_done: usize;
    let mut converged_early: bool;
    let mut best_prev: f64;
    let mut no_improve: usize;

    if let Some(cp) = resume {
        cp.check_data(train, test)?;
        if !cfg.same_search_settings(&cp.config) {
            let diff = cfg.search_setting_differences(&cp.config);
            return Err(format!(
                "checkpoint search settings do not match ({})",
                diff.join(", ")
            ));
        }
        if cfg.generations < cp.config.generations || cfg.generations < cp.generations {
            return Err(format!(
                "cannot resume: {} generation(s) are complete and the saved budget is {}, \
                 but the requested budget is {} (the budget may be raised, not lowered)",
                cp.generations, cp.config.generations, cfg.generations
            ));
        }
        let Checkpoint {
            generations,
            stopped_early,
            best_prev: saved_best,
            no_improve: saved_no_improve,
            archive: saved_archive,
            islands: saved_islands,
            stagnant: saved_stagnant,
            main_rng,
            ..
        } = cp;
        gens_done = generations;
        converged_early = stopped_early;
        best_prev = saved_best;
        no_improve = saved_no_improve;
        archive = saved_archive;
        stagnant = saved_stagnant;
        rng = main_rng;
        rngs = Vec::with_capacity(saved_islands.len());
        islands = Vec::with_capacity(saved_islands.len());
        for isl in saved_islands {
            rngs.push(isl.rng);
            islands.push(
                isl.population
                    .into_iter()
                    .map(Ind::from_checkpoint)
                    .collect(),
            );
        }
        let want = cfg.islands.max(1);
        if islands.len() != want || rngs.len() != want || stagnant.len() != want {
            return Err(format!(
                "checkpoint has {} island(s), {} rng state(s) and {} counter(s), \
                 but this configuration needs {}",
                islands.len(),
                rngs.len(),
                stagnant.len(),
                want
            ));
        }
        if islands.iter().any(|pop| pop.len() != cfg.pop) {
            return Err(format!(
                "checkpoint population size does not match the saved pop {}",
                cfg.pop
            ));
        }
    } else {
        rng = Mt::new(cfg.seed);
        archive = Archive::new(4000);

        // Distinct, reproducible streams per island.
        rngs = (0..cfg.islands.max(1))
            .map(|i| Mt::new(cfg.seed ^ (0x9E37_79B9u32.wrapping_mul(i as u32 + 1))))
            .collect();

        islands = Vec::with_capacity(cfg.islands.max(1));
        stagnant = vec![0; cfg.islands.max(1)];
        // Generation 0's subset also scores the initial population, so nothing
        // is evaluated on the full table before the first generation.
        let init_sample = training_for_generation(cfg, train, 0);
        let init_train = init_sample.as_ref().unwrap_or(train);
        let init_data = EvalSample::new(init_train, cfg); // #27
        for isl in 0..cfg.islands.max(1) {
            let depth = cfg.init_depth.min(1 + isl % 3);
            let mut pop = Vec::with_capacity(cfg.pop);
            for _ in 0..cfg.pop {
                let tree =
                    random_tree_with(&mut rng, nvars, &cfg.ops, depth, Some(&aggs));
                // #26: generation 0 is a whole generation too; defer it to the
                // same batch as the first real generation.
                pop.push(if defer_gpu {
                    Ind::pending(tree, cfg)
                } else {
                    Ind::evaluated(tree, &init_data, cfg) // #25
                });
            }
            islands.push(pop);
        }

        gens_done = 0;
        converged_early = false;
        best_prev = f64::INFINITY;
        no_improve = 0;
    }

    // #26: a checkpoint from a GPU run carries the f32 proxy, and a run forced
    // onto the CPU must not select against the wrong objective. Re-scoring the
    // loaded populations is deterministic, so an ordinary CPU resume keeps the
    // exact same trajectory.
    if resumed && !defer_gpu {
        // #27: re-scoring goes through the same SoA view as the live search.
        let train_view = EvalSample::new(train, cfg);
        for pop in islands.iter_mut() {
            for ind in pop.iter_mut() {
                ind.reeval(&train_view, cfg); // #25
            }
        }
    }

    // ---- seeded solutions (issue #16) ---------------------------------
    // Process every seed once: simplify, fit constants on the training
    // sample, evaluate its error, then insert it into the archive and into
    // every island's initial population. Seeds are elites: they are inserted
    // even above `max_complexity`, which only limits generated offspring.
    let mut seed_results: Vec<SeedResult> = Vec::new();
    // #27: seeds are fitted on the full training sample and re-scored (when a
    // generation-0 subset exists) on the same rows as the initial population.
    let train_data = EvalSample::new(train, cfg);
    if !seeds.is_empty() {
        // The population is being inserted before generation 0, whose subset
        // (if any) already scores the initial individuals; use the same sample
        // so every individual competes on identical rows.
        let init_sample = training_for_generation(cfg, train, 0);
        let init_train = init_sample.as_ref().unwrap_or(train);
        let init_data = EvalSample::new(init_train, cfg); // #27
        for (k, ind) in prepare_seeds(cfg, &train_data, seeds, &aggs)
            .into_iter()
            .enumerate()
        {
            archive.try_insert(Sol {
                complexity: expr::complexity(&ind.tree),
                cost: ind.complexity,
                tree: ind.tree.clone(),
                error: ind.error,
            });
            if k < cfg.pop {
                let in_pop = if init_sample.is_some() {
                    Ind::evaluated(ind.tree.clone(), &init_data, cfg) // #25
                } else if defer_gpu {
                    // #26: the archive keeps the CPU error; the population copy
                    // is scored with the rest of generation 0.
                    Ind::pending(ind.tree.clone(), cfg)
                } else {
                    ind.clone()
                };
                for pop in islands.iter_mut() {
                    if k < pop.len() {
                        pop[k] = in_pop.clone();
                    }
                }
            }
            seed_results.push(SeedResult {
                tree: ind.tree,
                complexity: ind.complexity,
                error: ind.error,
            });
        }
    }

    // #26: score generation 0 (and any seed population copies) in one batch
    // before the first selection step reads an error.
    #[cfg(feature = "cuda")]
    {
        defer_gpu = score_generation(&mut gpu, &mut gpu_used, &mut islands, train, cfg);
    }

    // #18: one tracker, updated once per generation from the archive. It holds
    // no population state, so it cannot affect the search trajectory.
    let mut progress_stats = ProgressStats::new(crate::progress_stats::DEFAULT_WINDOW);
    let mut stats_observed = false;

    if !converged_early {
        for gen in gens_done..cfg.generations {
            gens_done = gen + 1;

            // One fixed subset for the whole generation; a fresh draw next time.
            // `None` means the full training sample, exactly as before.
            let gen_sample = training_for_generation(cfg, train, gen);
            let gen_train = gen_sample.as_ref().unwrap_or(train);
            let gen_data = EvalSample::new(gen_train, cfg); // #27

            // With subsampling active, re-score the carried-over population on
            // this generation's subset so every individual competes on the same
            // rows. (Inactive runs keep the original errors untouched.)
            if gen > 0 && gen_sample.is_some() {
                for pop in islands.iter_mut() {
                    for ind in pop.iter_mut() {
                        ind.reeval(&gen_data, cfg); // #25
                    }
                }
            }

            // ---- parallel generation across islands ---------------------
            let aggs_ref = &aggs;
            let gen_view = &gen_data; // #27
            std::thread::scope(|s| {
                let handles: Vec<_> = islands
                    .iter_mut()
                    .zip(rngs.iter_mut())
                    .map(|(isl, r)| {
                        // #10: carry the block registry into the island thread.
                        let blocks = blocks.clone();
                        s.spawn(move || {
                            expr::set_blocks(blocks);
                            evolve_island(isl, r, cfg, &gen_view, nvars, aggs_ref, defer_gpu)
                        })
                    })
                    .collect();
                for h in handles {
                    let _ = h.join();
                }
            });

            // #26: one batched call covers every candidate of the generation.
            #[cfg(feature = "cuda")]
            {
                defer_gpu =
                    score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg);
            }

            // ---- archive + convergence bookkeeping ----------------------
            let mut gen_best = f64::INFINITY;
            for pop in &islands {
                for ind in pop {
                    if ind.error < gen_best {
                        gen_best = ind.error;
                    }
                    archive.try_insert(Sol {
                        complexity: expr::complexity(&ind.tree),
                        cost: ind.complexity,
                        tree: ind.tree.clone(),
                        error: ind.error,
                    });
                }
            }

            if let Some(budget) = cfg.time_budget {
                if t0.elapsed().as_secs_f64() >= budget {
                    converged_early = true;
                    break;
                }
            }

            // #18: convergence is a pure reading of the archive; O(front size).
            let convergence = progress_stats.observe(gens_done, gen_best, &archive);
            stats_observed = true;

            if let Some(report) = &cfg.progress {
                report(Progress {
                    generation: gens_done,
                    total_generations: cfg.generations,
                    best_error: gen_best,
                    pareto_members: archive.items.len(),
                    // #18: every progress event carries the four numbers.
                    convergence,
                });
            }

            if gen_best.is_finite() {
                if gen_best <= 1e-24 {
                    converged_early = true;
                    break;
                }
                let rel = (best_prev - gen_best) / (1.0 + best_prev.abs());
                if rel <= 1e-9 {
                    no_improve += 1;
                } else {
                    no_improve = 0;
                }
                best_prev = best_prev.min(gen_best);
                if cfg.patience > 0 && no_improve >= cfg.patience {
                    converged_early = true;
                    break;
                }
            }

            // ---- migration between islands ------------------------------
            if cfg.islands > 1
                && cfg.migration_interval > 0
                && (gen + 1) % cfg.migration_interval == 0
            {
                let bests: Vec<Option<Node>> = islands
                    .iter()
                    .map(|pop| {
                        pop.iter()
                            .min_by(|a, b| {
                                a.error
                                    .partial_cmp(&b.error)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|i| i.tree.clone())
                    })
                    .collect();
                for ii in 0..islands.len() {
                    let src = (ii + 1) % islands.len();
                    if let Some(t) = &bests[src] {
                        if let Some(worst) = islands[ii].iter_mut().max_by(|a, b| {
                            a.error
                                .partial_cmp(&b.error)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }) {
                            // #26: batch the migrant with the next GPU call.
                            *worst = if defer_gpu {
                                Ind::pending(t.clone(), cfg)
                            } else {
                                Ind::evaluated(t.clone(), &gen_data, cfg) // #25
                            };
                        }
                    }
                }
            }

            // ---- restart stagnant islands -------------------------------
            if cfg.restart_interval > 0 && (gen + 1) % cfg.restart_interval == 0 {
                let arch_best = archive
                    .items
                    .iter()
                    .map(|s| s.error)
                    .fold(f64::INFINITY, f64::min);
                for ii in 0..islands.len() {
                    let cur_best = islands[ii]
                        .iter()
                        .map(|i| i.error)
                        .fold(f64::INFINITY, f64::min);
                    if cur_best > arch_best + 1e-9 {
                        stagnant[ii] += 1;
                    } else {
                        stagnant[ii] = 0;
                    }
                    if stagnant[ii] >= 2 {
                        let mut r = rngs[ii].clone();
                        for ind in islands[ii].iter_mut() {
                            let mut t = random_tree_with(
                                &mut r,
                                nvars,
                                &cfg.ops,
                                cfg.init_depth,
                                Some(&aggs),
                            );
                            // #15: multi-output fits per output inside the evaluator.
                            if cfg.fit_iters > 0 && !gen_train.is_multi() {
                                fit_constants(&mut t, cfg, gen_train);
                            }
                            // #26: restart trees are batched with the next GPU
                            // call too, before the checkpoint is taken.
                            *ind = if defer_gpu {
                                Ind::pending(t, cfg)
                            } else {
                                Ind::evaluated(t, &gen_data, cfg)
                            };
                        }
                        rngs[ii] = r;
                        stagnant[ii] = 0;
                    }
                }
            }

            // #26: migration and restarts may have added pending individuals;
            // score them before the checkpoint so a saved state is never NaN.
            #[cfg(feature = "cuda")]
            {
                defer_gpu =
                    score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg);
            }

            // A checkpoint is taken only at a generation boundary: all of the
            // generation's work, including migration and restarts, is done.
            if save_every > 0 && gens_done % save_every == 0 {
                let cp = snapshot(
                    cfg,
                    gens_done,
                    converged_early,
                    best_prev,
                    no_improve,
                    &archive,
                    &islands,
                    &rngs,
                    &stagnant,
                    &rng,
                    train,
                    test,
                );
                emit(&mut on_checkpoint, &cp);
            }
        }
    }

    // Write on exit (including a run that stopped early) before the final
    // full-data archive rebuild, so the saved archive is the search archive at
    // the last completed generation.
    if on_checkpoint.is_some() {
        let cp = snapshot(
            cfg,
            gens_done,
            converged_early,
            best_prev,
            no_improve,
            &archive,
            &islands,
            &rngs,
            &stagnant,
            &rng,
            train,
            test,
        );
        emit(&mut on_checkpoint, &cp);
    }

    if subsampling_active(cfg, train) || gpu_used || resumed {
        // Generation fitness was estimated on subsets, or on the GPU in f32, so
        // the front is rebuilt from the surviving trees and scored on every
        // training row with the CPU in f64. From here on the archive and every
        // number derived from it are full-data and match the CPU backend.
        archive = full_data_archive(&archive, &islands, &train_data, cfg); // #25
    } else {
        for pop in &islands {
            for ind in pop {
                archive.try_insert(Sol {
                    complexity: expr::complexity(&ind.tree),
                    cost: ind.complexity,
                    tree: ind.tree.clone(),
                    error: ind.error,
                });
            }
        }
    }
    let sorted = archive.sorted();
    // #27: one transpose for the whole test sample, shared by every front
    // entry. Multi-output keeps its own scalar path.
    let test_data = EvalSample::new(test, cfg);
    let test_errors: Vec<f64> = sorted
        .iter()
        .map(|s| {
            // #15: fit the per-output constants on train, then score test.
            if train.is_multi() {
                multi::combined_test_error(&s.tree, train, test, cfg.metric, cfg.fit_iters)
            } else {
                eval_error_on(&s.tree, &test_data, cfg.metric)
            }
        })
        .collect();
    let mut final_archive = Archive::new(archive.cap);
    final_archive.items = sorted;

    // #18: the last in-loop reading describes the final generation. If no
    // generation ran (a resumed run that had already converged, or a zero
    // budget), take one reading of the final archive so the JSON is populated.
    let mut convergence = progress_stats.current();
    // #26: a GPU-guided run's in-loop readings are on the f32 proxy; take the
    // final reading from the rebuilt f64 archive so the reported statistics
    // describe the numbers the user sees.
    if !stats_observed || gpu_used || resumed {
        let best = final_archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        convergence = progress_stats.observe(gens_done, best, &final_archive);
    }

    Ok(Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early,
        seeds: seed_results,
        // #18: the four convergence numbers for the final --json output.
        convergence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    /// `y = x0` on a small grid, two inputs.
    fn sample() -> Sample {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..21 {
            let a = -1.0 + 2.0 * i as f64 / 20.0;
            x.push(vec![a, 0.5 * a + 1.0]);
            y.push(a);
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new() }
    }

    #[test]
    fn prepare_seeds_simplifies_fits_and_dedupes() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.fit_iters = 8;
        // A deliberately wrong constant on the right shape.
        let seed = node(Op::Mul, vec![constant(3.0), input(0)]);
        let aggs = AggTable::from_sample(&s);
        let data = EvalSample::new(&s, &cfg);
        let accepted = prepare_seeds(&cfg, &data, &[seed.clone(), seed], &aggs);
        assert_eq!(accepted.len(), 1, "a repeated seed is inserted once");
        assert!(
            accepted[0].error < 1e-12,
            "the constant must be fitted, error = {}",
            accepted[0].error
        );
        let shown = expr::to_string(&accepted[0].tree, &["x".to_string()]);
        assert!(
            !shown.contains("3 *"),
            "the fitted constant must replace 3, got {}",
            shown
        );
    }

    #[test]
    fn prepare_seeds_respects_no_fit() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.fit_iters = 0;
        let seed = node(Op::Mul, vec![constant(3.0), input(0)]);
        let aggs = AggTable::from_sample(&s);
        let data = EvalSample::new(&s, &cfg);
        let accepted = prepare_seeds(&cfg, &data, &[seed], &aggs);
        assert!(accepted[0].error > 1e-3, "an unfitted 3*x is not exact");
    }

    /// #77: a child identical to a parent must be caught by the dedup set, not
    /// fitted and scored as a new individual.
    #[test]
    fn parent_clones_are_rejected_before_evaluation() {
        let s = sample();
        let cfg = Config::default();
        let parent = Ind::evaluated(node(Op::Mul, vec![input(0), constant(2.0)]), &s, &cfg);
        let mut seen = HashSet::new();
        seed_seen(&mut seen, &[], std::slice::from_ref(&parent));
        assert!(
            !seen.insert(expr::structural_hash(&parent.tree)),
            "a clone of a parent must be a duplicate"
        );
    }
}
