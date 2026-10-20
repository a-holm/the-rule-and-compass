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

use crate::checkpoint::{Backend, Checkpoint, Individual, IslandState};
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
use std::collections::{BTreeMap, HashSet};

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
    /// generation. `None` scores the full training sample. #86: a request at
    /// least the training size is an exact no-op, and a request above the
    /// measured break-even (`3/5` of the rows) is widened to the full sample
    /// because the per-generation rebuild and carried-individual refit cost
    /// more than the rows saved. Values below 2 are rejected by the CLI.
    pub predictor_samples: Option<usize>,
    /// #85: number of rows used to **fit** constants in a generation. `0`
    /// disables screening (today's behaviour); any value at least the
    /// generation's sample size is a no-op. Candidates are still scored on the
    /// generation's full sample and the surviving archive is refit on every
    /// training row before it is reported. Values below 2 are rejected by the
    /// CLI.
    pub fit_samples: usize,
    /// #76: let the global Pareto front supply a fraction of parents. `false`
    /// is the pre-#76 algorithm, bit for bit; `true` (the default when it is
    /// measured to help) draws a parent from the front with probability
    /// [`FRONT_PARENT_P`] instead of the island tournament.
    pub front_parents: bool,
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
            fit_samples: 0, // #85: off by default
            front_parents: true, // #76: measured on the known-truth suite
            units: None,
            require_dimensions: false,
            progress: None,
            simd: true, // #27
            // #141: the GPU is the default backend when the binary is built
            // with CUDA and a device is available. The probe happens once at
            // startup; a non-CUDA build, a missing device or an unsupported
            // objective falls back to the CPU. `--no-gpu`/`--cpu` is the
            // deliberate override and `--gpu` asserts that a device exists.
            gpu: cfg!(feature = "cuda"),
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
        compare!(fit_samples, "fit_samples"); // #85
        compare!(front_parents, "front_parents"); // #76
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
        Ind::evaluated_fitted(tree, None, train, cfg)
    }

    /// #87: [`Ind::evaluated`] with the fitter's own SSE available.
    ///
    /// When the metric is `squared`, the fit ran on the same rows that score
    /// the candidate, and simplification folded nothing, the fitted SSE *is*
    /// the score (divided by the row count, exactly as `mean_sq` does), so the
    /// separate scoring sweep is skipped. Every other case keeps the ordinary
    /// evaluation path bit for bit.
    fn evaluated_fitted(
        tree: Node,
        fitted: Option<FitResult>,
        train: &EvalSample,
        cfg: &Config,
    ) -> Self {
        let tree = expr::simplify(&tree);
        let hash = expr::structural_hash(&tree);
        let complexity = cfg.complexity_model.complexity(&tree);
        let error = if cfg.accepts_dimensions(&tree) {
            match fitted {
                Some(fit) if can_reuse_sse(cfg, train, &fit, hash) => {
                    fit.sse / train.sample.len() as f64
                }
                _ => score_tree(&tree, train, cfg), // #27
            }
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

    // #86: re-fit the constants on the current generation's fit rows *before*
    // scoring, so a carried individual follows the same `fit(S_g) -> score(S_g)`
    // path as an offspring. Re-scoring alone left it as `fit(S_{g-1}) ->
    // score(S_g)`: stale constants were compared against freshly fitted ones,
    // and the elite — the longest-lived individual — was never refitted at all.
    // The fit sample always equals the scoring sample in this path, so the
    // final-SSE reuse stays exact (see `fit_constants`).
    fn refit_and_reeval(&mut self, train: &EvalSample, fit_sample: &Sample, cfg: &Config) {
        let mut tree = expr::simplify(&self.tree);
        let fitted = if cfg.fit_iters > 0 && !train.sample.is_multi() {
            // One warm-started LM iteration is enough to move constants that
            // were already fitted to a different subset; a cold full fit here
            // would double the generation's fit cost and erase the 750/1500
            // win. Offspring below keep the full `cfg.fit_iters` budget.
            fit_constants_with_iters(&mut tree, cfg, fit_sample, train.sample, 1)
        } else {
            None
        };
        *self = Ind::evaluated_fitted(tree, fitted, train, cfg); // #25 #87
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
    /// #122: the fixed-capacity, double-buffered postfix arena. It is allocated
    /// once (grown only if a later batch is larger) and reused every generation,
    /// so a generation never calls `cudaMalloc` for its programs.
    pool: Option<crate::genome::GenomePool>,
    /// #131: per-generation opcode/fallback counts.
    stats: BTreeMap<usize, GpuGenStats>,
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
            Some(ev) => Ok(GpuSearch {
                ev,
                n_rows,
                pool: None,
                stats: BTreeMap::new(),
            }),
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
        generation: usize,
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

        let mut slots: Vec<(usize, usize)> = Vec::with_capacity(pending.len());
        let mut on_cpu: Vec<(usize, usize)> = Vec::new();
        // #122: one fixed-width slot per genome. A tree whose postfix exceeds
        // the configured bound is scored on the CPU (the existing fallback), so
        // a wide seed can never truncate or corrupt the arena.
        let width = crate::genome::width_for(cfg.max_complexity);
        let reuse = self
            .pool
            .as_ref()
            .map_or(false, |p| p.capacity() >= pending.len() && p.width() == width);
        if !reuse {
            self.pool = Some(crate::genome::GenomePool::new(pending.len().max(1), width));
        }
        let pool = self.pool.as_mut().expect("a pool was just allocated");
        pool.begin_generation();
        for &(i, j) in &pending {
            let mut buf = Vec::new();
            let meta = crate::genome::GenomeMeta {
                island: i as u16,
                tag: j as u32,
                ..Default::default()
            };
            if crate::gpu::compile(&islands[i][j].tree, &mut buf)
                && buf.len() <= width
                && pool.push(&buf, meta)
            {
                slots.push((i, j));
            } else {
                on_cpu.push((i, j));
            }
        }

        // #131: record what the batch actually used. Several calls can belong
        // to one generation (initial screening, refit, restarts); merge them.
        let stats = self.stats.entry(generation).or_default();
        stats.generation = generation;
        stats.programs += pool.len();
        stats.fallbacks += on_cpu.len();

        if !pool.is_empty() {
            let (sse, ms) = self.ev.eval_pool(pool);
            if ms < 0.0 || sse.len() != pool.len() {
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
            // The generation is complete: the just-evaluated arena becomes the
            // readable front, and the next generation writes into the other one.
            pool.commit();
        }
        if !on_cpu.is_empty() {
            // #27: the CPU fallback uses the same SoA view as the live search.
            let train_view = EvalSample::new(train, cfg);
            // #122: unsupported programs used to be scored serially after the
            // kernel (the measured 2 s tail at P=10 000, 5 % `if`). Score them
            // on a bounded CPU pool instead; each tree is independent, so the
            // result is bit-identical to the serial walk.
            rescore_pending_cpu(islands, &train_view, cfg);
        }
        Ok(())
    }

    /// #131: the per-generation report, in generation order.
    fn run_stats(&self) -> GpuRunStats {
        GpuRunStats {
            generations: self
                .stats
                .values()
                .cloned()
                .collect(),
        }
    }
}

/// Score every still-pending (`error.is_nan()`) individual on the CPU.
///
/// Work is split into contiguous slices of each island's population and scored
/// on up to [`fallback_workers`] threads. A tree's error depends only on the
/// tree and the sample, never on a neighbour or on completion order, so the
/// parallel and serial paths return the same bits. Small batches stay serial:
/// the thread round trip would cost more than it saves.
fn rescore_pending_cpu(islands: &mut [Vec<Ind>], train: &EvalSample, cfg: &Config) {
    rescore_pending_cpu_with(islands, train, cfg, fallback_workers());
}

/// The worker count is a parameter so the determinism test can compare one
/// thread with many without touching the process environment.
fn rescore_pending_cpu_with(
    islands: &mut [Vec<Ind>],
    train: &EvalSample,
    cfg: &Config,
    workers: usize,
) {
    let pending: usize = islands
        .iter()
        .map(|pop| pop.iter().filter(|ind| ind.error.is_nan()).count())
        .sum();
    if pending == 0 {
        return;
    }
    let workers = workers.min(pending);
    if workers <= 1 {
        for pop in islands.iter_mut() {
            for ind in pop.iter_mut() {
                if ind.error.is_nan() {
                    ind.reeval(train, cfg); // #25
                }
            }
        }
        return;
    }
    let islands_n = islands.len().max(1);
    let per_island = (workers / islands_n).max(1);
    let blocks = expr::current_blocks();
    std::thread::scope(|scope| {
        for pop in islands.iter_mut() {
            let per = pop.len().div_ceil(per_island).max(1);
            for chunk in pop.chunks_mut(per) {
                let blocks = blocks.clone();
                scope.spawn(move || {
                    // #10: custom blocks are thread-local, exactly as in an
                    // island's evolution thread.
                    expr::set_blocks(blocks);
                    for ind in chunk.iter_mut() {
                        if ind.error.is_nan() {
                            ind.reeval(train, cfg); // #25
                        }
                    }
                });
            }
        }
    });
}

/// #122: size of the bounded CPU fallback pool.
fn fallback_workers() -> usize {
    const MAX: usize = 8;
    if let Ok(v) = std::env::var("RCSR_CPU_FALLBACK_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.clamp(1, 64);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(MAX)
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
    backend_reason: &mut Option<String>,
    generation: usize,
) -> bool {
    let Some(g) = gpu.as_mut() else {
        return false;
    };
    match g.score(islands, train, cfg, generation) {
        Ok(()) => {
            *used = true;
            true
        }
        Err(e) => {
            if cfg.gpu_notes {
                eprintln!("# gpu: unavailable ({})", e);
            }
            // #141: the backend report must record the runtime fallback, not
            // just the startup choice.
            *backend_reason = Some(e);
            *gpu = None;
            // #27: re-scoring the pending batch goes through the SoA view.
            let train_view = EvalSample::new(train, cfg);
            // #122: the same bounded CPU pool as the per-batch fallback.
            rescore_pending_cpu(islands, &train_view, cfg);
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
    // #85: the screened path scores offspring on a per-generation subset, which
    // the resident full-sample GPU evaluator cannot represent.
    if cfg.fit_iters > 0 && cfg.fit_samples >= 2 && cfg.fit_samples < train.len() {
        return Some("--fit-samples scores offspring on a screening subset");
    }
    None
}

/// The backend this configuration and sample will use in this build
/// (issue #140).
///
/// This mirrors the capability checks in [`run_search`] and is used by the CLI
/// to refuse a backend-changing `--resume` before the search starts. An actual
/// device failure still falls back to the CPU inside the search.
pub fn effective_backend(cfg: &Config, train: &Sample) -> Backend {
    #[cfg(feature = "cuda")]
    {
        if cfg.gpu
            && !matches!(
                std::env::var("RCSR_GPU").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no")
            )
            && gpu_blocker(cfg, train).is_none()
        {
            return Backend::Gpu;
        }
    }
    let _ = (cfg, train);
    Backend::Cpu
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
        #[cfg(test)]
        note_transpose();
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

// #133: test-only count of SoA transposes on this thread. The search builds
// these views on the main thread (island threads share the view), so a unit
// test can assert "once per distinct sample" instead of "once per use".
#[cfg(test)]
thread_local! {
    static TRANSPOSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_transpose() {
    TRANSPOSES.with(|n| n.set(n.get() + 1));
}

#[cfg(test)]
fn transpose_count() -> usize {
    TRANSPOSES.with(|n| n.get())
}

#[cfg(test)]
fn reset_transpose_count() {
    TRANSPOSES.with(|n| n.set(0));
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
///
/// #85: `fit` is the (possibly screening) sample the constants are fitted on;
/// `score` is the sample that will score the candidate. They are the same
/// reference in the default path.
fn fit_constants(
    tree: &mut Node,
    cfg: &Config,
    fit: &Sample,
    score: &Sample,
) -> Option<FitResult> {
    fit_constants_with_iters(tree, cfg, fit, score, cfg.fit_iters)
}

/// #86: [`fit_constants`] with an explicit iteration budget. Carried
/// individuals are warm-started from constants already fitted on the previous
/// subset, so one LM iteration refreshes them on the new rows; offspring keep
/// the full `--fit-iters` budget.
fn fit_constants_with_iters(
    tree: &mut Node,
    cfg: &Config,
    fit: &Sample,
    score: &Sample,
    iters: usize,
) -> Option<FitResult> {
    if cfg.metric.is_implicit_derivative() {
        crate::implicit::fit(tree, &fit.x, &fit.w, iters);
        None
    } else {
        let sse = crate::fit::fit_weighted(tree, &fit.x, &fit.y, &fit.w, iters);
        // #87: remember the tree the SSE belongs to. Simplification after the
        // fit may fold constants, in which case the SSE is stale and the
        // caller must score normally.
        Some(FitResult {
            sse,
            hash: expr::structural_hash(tree),
            // #85: an SSE fitted on a screening subset is not the score of the
            // full sample, so the reuse below must stay off.
            same_rows: std::ptr::eq(fit, score),
        })
    }
}

/// #87: the fitter's own residual sum of squares at the constants it wrote
/// back, plus the structural hash of the tree at that moment.
struct FitResult {
    sse: f64,
    hash: u64,
    /// #85: whether the fit used the same rows that will score the candidate.
    same_rows: bool,
}

/// #87: whether `fit`'s SSE is provably the score of `tree` on `train`.
///
/// The fitter sums `Σ (f(xᵢ) − yᵢ)²` in row order; `Metric::Squared` on an
/// unweighted, single-output sample divides the same sum by the row count. A
/// different metric, weights (whose `sqrt(w)` residual squares differently),
/// a multi-output objective, an empty sample, or a simplification that folded
/// anything all keep the ordinary evaluation path.
fn can_reuse_sse(cfg: &Config, train: &EvalSample, fit: &FitResult, hash: u64) -> bool {
    cfg.metric == Metric::Squared
        && fit.same_rows // #85
        && fit.hash == hash
        && fit.sse.is_finite()
        && !train.sample.is_multi()
        && train.sample.w.is_empty()
        && !train.sample.is_empty()
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

/// #86: the largest fraction of the training rows a screened generation must
/// save before the subset is activated, as `NUM / DEN` = 3/5.
///
/// The fitter-only break-even `(24 - 1/G) / 25 ~ 0.96` assumed that only the
/// scoring sweep was saved for the carried population. Refitting those
/// individuals (one warm-started LM iteration plus the four-iteration polish,
/// about six sweeps) changes the model to roughly `(26 + 6) m` against `26 n`
/// per generation, i.e. `m/n < 0.81` before the per-generation sample rebuild
/// and the extra final archive pass. On `w2.csv` (1500 train rows, 10
/// generations, `--fit-iters 8`) `900/1500 = 0.60` still wins but `1000/1500 =
/// 0.67` is slower than full data, so the measured rule is 3/5. `750/1500`
/// stays active; `1400/1500` is widened to the full sample.
const PREDICTOR_BREAK_EVEN_NUM: usize = 3;
const PREDICTOR_BREAK_EVEN_DEN: usize = 5;

/// #86: the effective per-generation subset size, or `None` when the full
/// training sample must be used.
///
/// `n >= train.len()` is an exact no-op (the full sample, original order,
/// bit-identical to leaving the option unset). A request above the measured
/// break-even is widened to the full sample for the same reason, with a note
/// printed by the CLI; this keeps the near-full regime from being a
/// pessimization while still accepting the measured wins at 750/1500 and below.
pub fn effective_predictor_samples(cfg: &Config, train: &Sample) -> Option<usize> {
    let requested = cfg.predictor_samples?;
    let rows = train.len();
    // The CLI rejects values below 2; a direct `Config` sees them as "off"
    // rather than as a one-row sample.
    if rows == 0 || requested < 2 || requested >= rows {
        return None;
    }
    if requested * PREDICTOR_BREAK_EVEN_DEN > rows * PREDICTOR_BREAK_EVEN_NUM {
        return None;
    }
    Some(requested)
}

/// True when candidate scoring should use a per-generation subset of the
/// training rows.
fn subsampling_active(cfg: &Config, train: &Sample) -> bool {
    effective_predictor_samples(cfg, train).is_some()
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

/// Copy the rows named by `idx` into a fresh sample, preserving their order.
fn subset_sample(train: &Sample, idx: &[usize]) -> Sample {
    Sample {
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
    }
}

/// Build the sample a generation evolves on, or `None` when the full training
/// sample must be used. The caller passes the full sample in that case, which
/// preserves the original row order and floating-point summation.
fn training_for_generation(cfg: &Config, train: &Sample, generation: usize) -> Option<Sample> {
    // #86: use the effective size, not the raw request, so a near-full request
    // resolves to the full sample here as well.
    let n = effective_predictor_samples(cfg, train)?;
    let idx = generation_subset(train.len(), n, cfg.seed, generation);
    Some(subset_sample(train, &idx))
}

/// #85: salt that separates the screening stream from `--predictor-samples`
/// (and from every other RNG), so asking for a fit subset never changes which
/// rows a generation is scored on.
const FIT_SUBSET_SALT: u32 = 0x85F1_7A5B;

/// #85: true when constants are fitted on fewer rows than the generation's
/// scoring sample. A request at least as large as the sample is a no-op.
fn fit_sampling_active(cfg: &Config, sample: &Sample) -> bool {
    cfg.fit_iters > 0 && cfg.fit_samples >= 2 && cfg.fit_samples < sample.len()
}

/// #85: the rows used to fit constants in one generation.
///
/// The subset is drawn fresh per generation from the current generation's
/// sample with [`generation_subset`], using a salted seed so it is independent
/// of `--predictor-samples`. `None` means "fit on the whole sample", which
/// keeps the default path bit for bit.
fn fit_for_generation(cfg: &Config, sample: &Sample, generation: usize) -> Option<Sample> {
    if !fit_sampling_active(cfg, sample) {
        return None;
    }
    let idx = generation_subset(
        sample.len(),
        cfg.fit_samples,
        cfg.seed ^ FIT_SUBSET_SALT,
        generation,
    );
    Some(subset_sample(sample, &idx))
}

/// #85: refit one screening survivor on the full training sample and score it
/// there, ready to enter the archive.
fn refit_survivor(tree: &Node, train: &EvalSample, cfg: &Config) -> Sol {
    let mut tree = expr::simplify(tree);
    if cfg.fit_iters > 0 && !train.sample.is_multi() {
        fit_constants(&mut tree, cfg, train.sample, train.sample);
        tree = expr::simplify(&tree);
    }
    let error = if cfg.accepts_dimensions(&tree) {
        score_tree(&tree, train, cfg)
    } else {
        f64::INFINITY
    };
    Sol {
        complexity: expr::complexity(&tree),
        cost: cfg.complexity_model.complexity(&tree),
        tree,
        error,
    }
}

/// #85: successive halving promotes one offspring in this many. With the
/// default `--pop 100` that is the top quarter; the fraction is deliberately
/// fixed (no Hyperband-style schedule across generations, because candidates
/// are replaced rather than trained longer).
const PROMOTE_ONE_IN: usize = 16;

/// #85: how many of `pop` screened offspring advance to a full-data refit.
fn promoted_count(pop: usize) -> usize {
    (pop / PROMOTE_ONE_IN).max(1).min(pop)
}

/// #85: keep the best `pop` full-data candidates. The population keeps its
/// fixed size (checkpoints and the next generation assume `pop`); structural
/// duplicates are removed by the archive, not here.
fn select_population(candidates: &mut Vec<Ind>, pop: usize) {
    candidates.sort_by(|a, b| {
        a.error
            .partial_cmp(&b.error)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.complexity
                    .partial_cmp(&b.complexity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    candidates.truncate(pop);
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

// #76: fraction of parent draws that come from the global Pareto front.
//
// The front is non-dominated, so every member is the best expression found at
// some complexity. Drawing uniformly over it keeps that spread: a complex but
// much more accurate shape is as likely as a small one, and the existing
// `max_complexity` check, simplification and de-duplication still reject a
// bloated offspring. Biasing toward low complexity would instead duplicate what
// the island tournament already does.
const FRONT_PARENT_P: f64 = 0.15;

/// #76: optionally draw a parent tree from the front.
///
/// Returns `None` without touching the RNG when the feature is off (`p == 0`)
/// or the front is empty, so an off run is bit-identical to the pre-#76 search.
/// The stored `Sol.error` is deliberately ignored: it may come from a
/// different row subset (`--predictor-samples`), and the child is always scored
/// on the generation's own sample anyway.
fn front_parent<'a>(front: &'a [Sol], rng: &mut Mt, p: f64) -> Option<&'a Node> {
    if p > 0.0 && !front.is_empty() && rng.bool(p) {
        Some(&front[rng.usize_below(front.len())].tree)
    } else {
        None
    }
}

/// #76: one parent, from the front with probability `p`, else the island
/// tournament. The draw uses the island's own RNG, so a run remains
/// deterministic for a given seed.
fn pick_parent<'a>(island: &'a [Ind], front: &'a [Sol], rng: &mut Mt, p: f64) -> &'a Node {
    match front_parent(front, rng, p) {
        Some(tree) => tree,
        None => &island[tournament(island, rng, 4)].tree,
    }
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
    // #85: the rows constants are fitted on (the screening subset, or the
    // whole scoring sample when screening is off).
    fit_sample: &Sample,
    nvars: usize,
    aggs: &AggTable,
    // #85: successive-halving phase one. Generate `pop` offspring and score
    // them on `train` (the screening sample); the caller promotes the top
    // fraction and selects against the full-data parents.
    screened: bool,
    // #26: leave the fitness pending when the caller will batch it on the GPU.
    defer_gpu: bool,
    // #76: the previous generation's Pareto front. Read-only and borrowed for
    // the duration of the island thread; nothing is cloned and the islands
    // still run in parallel. `front_p == 0.0` makes it a no-op.
    front: &[Sol],
    front_p: f64,
) {
    if !screened {
        for ind in isl.iter_mut() {
            if !ind.error.is_finite() {
                ind.reeval(train, cfg); // #25
            }
        }
    }

    let pop = cfg.pop;
    let mut next: Vec<Ind> = Vec::with_capacity(pop);

    // Elitism: carry the best individual over unchanged. In the screened phase
    // the full-data parents are kept by the caller and compete at selection, so
    // the offspring vector stays a full `pop`.
    if !screened {
        if let Some(best) = isl
            .iter()
            .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
        {
            next.push(best.clone());
        }
    }

    let mut seen: HashSet<u64> = HashSet::with_capacity(pop * 2);
    // #77: parents are candidates too; seeding them here rejects an unchanged
    // child before it is fitted and scored.
    seed_seen(&mut seen, &next, isl);
    let mut attempts = 0usize;
    let max_attempts = pop * 50 + 50;

    while next.len() < pop && attempts < max_attempts {
        attempts += 1;
        let p1 = pick_parent(isl, front, rng, front_p); // #76
        let mut child = p1.clone();
        if rng.bool(0.5) {
            let parent2 = pick_parent(isl, front, rng, front_p).clone(); // #76
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
        let mut fitted = None;
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fitted = fit_constants(&mut tree, cfg, fit_sample, train.sample); // #85 #87
            tree = expr::simplify(&tree);
        }
        // #26: with the GPU batch active the fitness is filled in after every
        // island has been generated; the CPU still owns simplification,
        // fitting, de-duplication and the complexity/dimension checks.
        let ind = if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            Ind::evaluated_fitted(tree, fitted, train, cfg) // #25 #87
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
        let mut fitted = None;
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fitted = fit_constants(&mut tree, cfg, fit_sample, train.sample); // #85 #87
            tree = expr::simplify(&tree);
        }
        // #26: same deferral for the random fill.
        next.push(if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            Ind::evaluated_fitted(tree, fitted, train, cfg) // #25 #87
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

/// #131: one GPU scoring batch, reported in `--json` under `gpu.generations`.
#[derive(Clone, Debug, Default)]
pub struct GpuGenStats {
    /// Generation number this batch belongs to.
    pub generation: usize,
    /// Operator occurrences in the programs that ran on the GPU.
    pub opcodes: Vec<(String, usize)>,
    /// Programs compiled into the batch.
    pub programs: usize,
    /// Candidates sent to the CPU fallback in the same batch.
    pub fallbacks: usize,
}

/// #131: aggregate GPU usage for a run.
#[derive(Clone, Debug, Default)]
pub struct GpuRunStats {
    pub generations: Vec<GpuGenStats>,
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
    /// #85: effective rows per generation used to fit constants, or `0` when
    /// screening is off. Never larger than the training sample.
    pub fit_samples: usize,
    /// #85: whether the reported archive was refit on the full training sample.
    pub archive_refit: bool,
    /// #86: effective rows per generation used for scoring and fitting, or `0`
    /// when the full training sample was used (off, exact no-op, or a request
    /// widened above the break-even).
    pub predictor_samples: usize,
    /// #141: the backend that actually guided the generations, recorded from
    /// what ran rather than inferred from the flags. Reported errors are
    /// always rebuilt on the CPU in `f64` regardless of this value.
    pub backend: crate::backend::Kind,
    /// #141: why the CPU ran instead of the GPU (missing feature, no device,
    /// unsupported objective or a CUDA error).
    pub backend_reason: Option<String>,
    /// #141: the device that guided the search, when `backend` is `gpu`.
    pub device: Option<String>,
    /// #131: which native opcodes and how many CPU fallbacks each GPU batch
    /// used. `None` when the run did not use the GPU.
    pub gpu: Option<GpuRunStats>,
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
    backend: Backend,
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
        backend,
        config_hash: crate::checkpoint::config_hash(cfg),
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
        let mut fitted = None;
        if cfg.fit_iters > 0 && !train.sample.is_multi() {
            fitted = fit_constants(&mut tree, cfg, train.sample, train.sample); // #85 #87
            tree = expr::simplify(&tree);
        }
        if !seen_fitted.insert(expr::structural_hash(&tree)) {
            continue;
        }
        out.push(Ind::evaluated_fitted(tree, fitted, train, cfg)); // #25 #87
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

    // #26/#141: one resident evaluator for the full training sample, used for
    // whole generations. Constants were already fitted on the CPU; this only
    // chooses where the tree evaluation runs. The backend report below records
    // what actually ran: a missing feature, a missing device, an unsupported
    // objective or a runtime CUDA error leaves the CPU path and a reason.
    #[cfg(feature = "cuda")]
    let mut backend_device: Option<String> = None;
    #[cfg(not(feature = "cuda"))]
    let backend_device: Option<String> = None;
    let mut backend_reason: Option<String> = None;
    #[cfg(feature = "cuda")]
    let mut gpu: Option<GpuSearch> = if !cfg.gpu {
        None
    } else {
        match gpu_blocker(cfg, train) {
            Some(reason) => {
                if cfg.gpu_notes {
                    eprintln!("# gpu: off ({})", reason);
                }
                backend_reason = Some(reason.to_string());
                None
            }
            None => match GpuSearch::new(train) {
                Ok(g) => {
                    let device = crate::gpu::GpuEvaluator::device_name();
                    if cfg.gpu_notes {
                        eprintln!(
                            "# gpu: {} (one batch per generation; f32 fitness, reported numbers stay f64)",
                            device
                        );
                    }
                    backend_device = Some(device);
                    Some(g)
                }
                Err(e) => {
                    if cfg.gpu_notes {
                        eprintln!("# gpu: unavailable ({})", e);
                    }
                    backend_reason = Some(e);
                    None
                }
            },
        }
    };
    #[cfg(not(feature = "cuda"))]
    {
        if cfg.gpu {
            backend_reason = Some("built without --features cuda".to_string());
            if cfg.gpu_notes {
                eprintln!("# gpu: unavailable (built without --features cuda)");
            }
        }
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
    // #85: did any generation fit on a screening subset? Only then may the
    // final archive be refit; a request at least as large as the sample, or
    // `--no-fit`, stays an exact no-op.
    let mut fit_screened = resumed && fit_sampling_active(cfg, train);
    // #27/#85: the full-data view is needed by the seeds, the screened
    // generation's promotion step and the final rebuild. Build it once.
    let train_data = EvalSample::new(train, cfg);

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
        // #133: with no predictor subset, generation 0 scores the full training
        // sample whose SoA view already exists; do not transpose it again.
        let init_data_owned;
        let init_data = match &init_sample {
            Some(sample) => {
                init_data_owned = EvalSample::new(sample, cfg);
                &init_data_owned
            }
            None => &train_data,
        };
        // #85: with screening on, the initial population is the first set of
        // full-data parents, so it must not start on a predictor subset.
        let init_eval = if fit_sampling_active(cfg, train) {
            &train_data
        } else {
            init_data
        };
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
                    Ind::evaluated(tree, init_eval, cfg) // #25 #85
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
        // #27/#133: re-scoring shares the SoA view built once above.
        for pop in islands.iter_mut() {
            for ind in pop.iter_mut() {
                ind.reeval(&train_data, cfg); // #25
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
    if !seeds.is_empty() {
        // The population is being inserted before generation 0, whose subset
        // (if any) already scores the initial individuals; use the same sample
        // so every individual competes on identical rows.
        let seed_sample = training_for_generation(cfg, train, 0);
        // #133: no subset means the full-data view already exists.
        let seed_data_owned;
        let seed_data = match &seed_sample {
            Some(sample) => {
                seed_data_owned = EvalSample::new(sample, cfg);
                &seed_data_owned
            }
            None => &train_data,
        };
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
                let in_pop = if seed_sample.is_some() {
                    Ind::evaluated(ind.tree.clone(), seed_data, cfg) // #25
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
        defer_gpu = score_generation(&mut gpu, &mut gpu_used, &mut islands, train, cfg, &mut backend_reason, gens_done);
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
            // #133: transpose once per distinct sample, not once per use. With
            // no predictor subset the generation shares the full-data view.
            let gen_data_owned;
            let gen_data: &EvalSample = match &gen_sample {
                Some(sample) => {
                    gen_data_owned = EvalSample::new(sample, cfg);
                    &gen_data_owned
                }
                None => &train_data,
            };
            // #85: a fresh screening subset for constant fitting, drawn from
            // this generation's sample. `None` keeps the whole sample.
            let fit_sample = fit_for_generation(cfg, gen_train, gen);
            let fit_train = fit_sample.as_ref().unwrap_or(gen_train);
            let screened = fit_sample.is_some();
            if screened {
                fit_screened = true;
            }
            // #85: successive halving phase one scores offspring on the
            // screening sample; phase two promotes the top fraction to a
            // full-data refit and selects against the full-data parents.
            let screen_data = if screened {
                Some(EvalSample::new(fit_train, cfg))
            } else {
                None
            };
            let eval_data = screen_data.as_ref().unwrap_or(gen_data);

            // #86: with predictor subsampling active (and no screening), refit
            // the carried-over population on this generation's fit rows, then
            // score it on the same rows. Re-scoring alone compared constants
            // fitted on the previous generation's subset against freshly fitted
            // offspring and never refitted the elite. In the screened path the
            // parents are already full-data and stay so.
            if gen > 0 && gen_sample.is_some() && !screened {
                for pop in islands.iter_mut() {
                    for ind in pop.iter_mut() {
                        ind.refit_and_reeval(gen_data, fit_train, cfg); // #25
                    }
                }
            }

            // #85: keep the full-data parents for the selection step.
            let parents: Option<Vec<Vec<Ind>>> = if screened {
                Some(islands.clone())
            } else {
                None
            };

            // ---- parallel generation across islands ---------------------
            let aggs_ref = &aggs;
            let eval_view = eval_data; // #27
            // #76: the front from the previous generation is borrowed, not
            // copied, and the scope ends before this generation updates the
            // archive below. At generation 0 (or with the feature off) it is
            // empty or unused and the search is unchanged.
            let front: &[Sol] = &archive.items;
            let front_p = if cfg.front_parents {
                FRONT_PARENT_P
            } else {
                0.0
            };
            std::thread::scope(|s| {
                let handles: Vec<_> = islands
                    .iter_mut()
                    .zip(rngs.iter_mut())
                    .map(|(isl, r)| {
                        // #10: carry the block registry into the island thread.
                        let blocks = blocks.clone();
                        s.spawn(move || {
                            expr::set_blocks(blocks);
                            evolve_island(
                                isl,
                                r,
                                cfg,
                                &eval_view,
                                fit_train,
                                nvars,
                                aggs_ref,
                                screened,
                                defer_gpu && !screened,
                                front,   // #76
                                front_p, // #76
                            ) // #85
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
                if !screened {
                    defer_gpu =
                        score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen);
                }
            }

            // #85: successive halving phase two. Promote the top fraction of
            // the screening offspring, refit and re-score them on every
            // training row, then select the next population against the
            // full-data parents. The archive only ever sees full-data scores.
            if let Some(parents) = parents {
                let promote = promoted_count(cfg.pop);
                for (pop, parents) in islands.iter_mut().zip(parents) {
                    let mut offspring = std::mem::take(pop);
                    offspring.sort_by(|a, b| {
                        a.error
                            .partial_cmp(&b.error)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut promoted: Vec<Ind> = Vec::with_capacity(promote);
                    for mut ind in offspring.into_iter().take(promote) {
                        let sol = refit_survivor(&ind.tree, &train_data, cfg);
                        ind.tree = sol.tree;
                        ind.error = sol.error;
                        ind.complexity = sol.cost;
                        promoted.push(ind);
                    }
                    let mut combined = parents;
                    combined.extend(promoted);
                    select_population(&mut combined, cfg.pop);
                    *pop = combined;
                }
            }

            // ---- archive + convergence bookkeeping ----------------------
            let mut gen_best = f64::INFINITY;
            for pop in &islands {
                for ind in pop {
                    if ind.error < gen_best {
                        gen_best = ind.error;
                    }
                    archive.try_insert_view(
                        &ind.tree,
                        ind.error,
                        expr::complexity(&ind.tree),
                        ind.complexity,
                    );
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
                                Ind::evaluated(t.clone(), gen_data, cfg) // #25
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
                            let mut fitted = None;
                            if cfg.fit_iters > 0 && !gen_train.is_multi() {
                                fitted = fit_constants(&mut t, cfg, fit_train, gen_train); // #85 #87
                            }
                            // #26: restart trees are batched with the next GPU
                            // call too, before the checkpoint is taken.
                            *ind = if defer_gpu {
                                Ind::pending(t, cfg)
                            } else {
                                Ind::evaluated_fitted(t, fitted, gen_data, cfg) // #87
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
                    score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen);
            }

            // A checkpoint is taken only at a generation boundary: all of the
            // generation's work, including migration and restarts, is done.
            if save_every > 0 && gens_done % save_every == 0 {
                let backend = if defer_gpu || gpu_used {
                    Backend::Gpu
                } else {
                    Backend::Cpu
                };
                let cp = snapshot(
                    cfg,
                    backend,
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
        let backend = if defer_gpu || gpu_used {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        let cp = snapshot(
            cfg,
            backend,
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

    if subsampling_active(cfg, train) || gpu_used || resumed || fit_screened {
        // Generation fitness was estimated on subsets, or on the GPU in f32, so
        // the front is rebuilt from the surviving trees and scored on every
        // training row with the CPU in f64. From here on the archive and every
        // number derived from it are full-data and match the CPU backend.
        archive = full_data_archive(&archive, &islands, &train_data, cfg); // #25 #85
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

    // #131: the GPU report is emitted only for a run that actually used the
    // device, so a CPU run's JSON is byte-identical to before.
    #[cfg(feature = "cuda")]
    let gpu_stats = if gpu_used { gpu.as_ref().map(|g| g.run_stats()) } else { None };
    #[cfg(not(feature = "cuda"))]
    let gpu_stats: Option<GpuRunStats> = None;

    Ok(Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early,
        seeds: seed_results,
        // #18: the four convergence numbers for the final --json output.
        convergence,
        // #85: the CLI states both numbers; `0` keeps the old output.
        fit_samples: if fit_screened {
            cfg.fit_samples.min(train.len())
        } else {
            0
        },
        archive_refit: fit_screened,
        // #86: the effective subset actually used, after the break-even rule.
        predictor_samples: effective_predictor_samples(cfg, train).unwrap_or(0),
        backend: if gpu_used {
            crate::backend::Kind::Gpu
        } else {
            crate::backend::Kind::Cpu
        },
        backend_reason,
        device: if gpu_used { backend_device } else { None },
        gpu: gpu_stats,
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

    /// #141: the library default follows the build: a CUDA binary prefers the
    /// GPU, every other build is CPU. Selection still depends on the probe.
    #[test]
    fn gpu_default_follows_the_cuda_feature() {
        assert_eq!(Config::default().gpu, cfg!(feature = "cuda"));
    }

    /// #141: a CPU outcome reports the reason and never a stale device. The
    /// CPU is forced here so this test is deterministic on a GPU machine.
    #[test]
    fn cpu_outcome_reports_backend_and_reason() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.gpu = false;
        cfg.generations = 2;
        cfg.pop = 12;
        cfg.patience = 0;
        let out = run(&cfg, &s, &s);
        assert_eq!(out.backend, crate::backend::Kind::Cpu);
        assert!(out.device.is_none());
    }

    /// #133: the full training view is built once for the run and reused by
    /// generation 0 and every generation that draws no predictor subset. The
    /// pooled test view is built once at the end.
    #[test]
    fn the_full_train_view_is_transposed_once_per_run() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.generations = 4;
        cfg.pop = 8;
        cfg.fit_iters = 0;
        cfg.islands = 1;
        cfg.predictor_samples = None;
        assert!(cfg.simd);
        reset_transpose_count();
        let _ = run(&cfg, &s, &s);
        assert_eq!(
            transpose_count(),
            2,
            "expected one full-train and one test transpose, got {}",
            transpose_count()
        );
    }

    // #76: two distinguishable front members.
    fn two_member_front() -> Vec<Sol> {
        vec![
            Sol {
                tree: input(0),
                error: 0.1,
                complexity: 1,
                cost: 1.0,
            },
            Sol {
                tree: constant(2.0),
                error: 0.2,
                complexity: 1,
                cost: 1.0,
            },
        ]
    }

    /// #76: at p = 1 every parent comes from the front; at p = 0 every parent
    /// comes from the island. The front is non-dominated, so the draw is
    /// uniform and both members must appear.
    #[test]
    fn archive_parent_draws_from_front() {
        let front = two_member_front();
        // A tree that is in neither front member.
        let island = vec![Ind {
            tree: node(Op::Sin, vec![input(0)]),
            error: 1.0,
            complexity: 1.0,
        }];

        let mut rng = Mt::new(11);
        let mut seen = [0usize; 2];
        for _ in 0..100 {
            let t = pick_parent(&island, &front, &mut rng, 1.0);
            let k = if t == &front[0].tree {
                0
            } else if t == &front[1].tree {
                1
            } else {
                panic!("p = 1.0 drew a parent that is not on the front");
            };
            seen[k] += 1;
        }
        assert!(seen[0] > 0 && seen[1] > 0, "uniform front draw: {seen:?}");

        let mut rng = Mt::new(11);
        for _ in 0..100 {
            assert!(
                pick_parent(&island, &front, &mut rng, 0.0) == &island[0].tree,
                "p = 0.0 must use the island tournament"
            );
        }

        assert!(
            front_parent(&[], &mut Mt::new(1), 1.0).is_none(),
            "an empty front is a no-op"
        );
    }

    /// #76: with the feature off, a populated front changes nothing — same
    /// population and same RNG state as an empty front.
    #[test]
    fn front_snapshot_is_inert_when_off() {
        let s = sample();
        let aggs = AggTable::from_sample(&s);
        let mut cfg = Config::default();
        cfg.pop = 8;
        cfg.fit_iters = 0;
        let data = EvalSample::new(&s, &cfg);

        let base: Vec<Ind> = (0..8)
            .map(|i| {
                Ind::evaluated(
                    if i % 2 == 0 {
                        input(0)
                    } else {
                        node(Op::Add, vec![input(0), constant(i as f64)])
                    },
                    &data,
                    &cfg,
                )
            })
            .collect();

        let run = |front: &[Sol]| {
            let mut isl = base.clone();
            let mut rng = Mt::new(7);
            evolve_island(
                &mut isl,
                &mut rng,
                &cfg,
                &data,
                &s,
                1,
                &aggs,
                false,
                false,
                front,
                0.0,
            );
            let trees: Vec<Node> = isl.iter().map(|i| i.tree.clone()).collect();
            let errors: Vec<u64> = isl.iter().map(|i| i.error.to_bits()).collect();
            (trees, errors, *rng.state().0, rng.state().1)
        };

        let empty = run(&[]);
        let filled = run(&two_member_front());
        assert_eq!(empty.0, filled.0, "the front changed the population");
        assert_eq!(empty.1, filled.1, "the front changed the errors");
        assert_eq!(empty.2, filled.2, "the front consumed RNG draws");
        assert_eq!(empty.3, filled.3, "the front moved the RNG index");
    }

    /// #76: the feedback path runs end to end and never archives a non-finite
    /// error.
    #[test]
    fn front_feedback_run_is_finite() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let v = i as f64 / 39.0;
            x.push(vec![v]);
            y.push(2.0 * v + 1.0);
        }
        let train = Sample { x, y, w: Vec::new(), ys: Vec::new() };
        let test = train.clone();
        let cfg = Config {
            generations: 4,
            pop: 20,
            islands: 1,
            seed: 3,
            ..Config::default()
        };
        assert!(cfg.front_parents, "the feedback path must be on");
        let out = run(&cfg, &train, &test);
        assert!(!out.archive.items.is_empty(), "the archive is non-empty");
        assert!(
            out.archive.items.iter().all(|s| s.error.is_finite()),
            "every archived error is finite"
        );
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

    /// #86: a carried individual must be refitted on the generation's rows
    /// before it is scored, not merely re-scored with stale constants.
    #[test]
    fn carried_individual_is_refitted_on_the_new_subset() {
        fn slope_sample(slope: f64, n: usize) -> Sample {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for i in 0..n {
                let a = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
                x.push(vec![a]);
                y.push(slope * a);
            }
            Sample { x, y, w: Vec::new(), ys: Vec::new() }
        }

        let mut cfg = Config::default();
        cfg.fit_iters = 8;
        // Slopes deliberately away from 1 so simplification cannot fold the
        // fitted `c * x` into the bare input between refits.
        let first = slope_sample(2.0, 40);
        let second = slope_sample(5.0, 40);
        let first_data = EvalSample::new(&first, &cfg);
        let second_data = EvalSample::new(&second, &cfg);
        let mut ind = Ind {
            tree: node(Op::Mul, vec![constant(2.0), input(0)]),
            error: f64::INFINITY,
            complexity: 0.0,
        };

        ind.refit_and_reeval(&first_data, &first, &cfg);
        let text_first = expr::to_string(&ind.tree, &["x".to_string()]);
        assert!(ind.error < 1e-12, "first refit error = {}", ind.error);

        ind.refit_and_reeval(&second_data, &second, &cfg);
        let text_second = expr::to_string(&ind.tree, &["x".to_string()]);
        assert!(ind.error < 1e-12, "second refit error = {}", ind.error);
        assert_ne!(
            text_first, text_second,
            "the carried constants must move with the subset"
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

    /// #87: on an unweighted `squared` fit the fitter's SSE is the score, bit
    /// for bit, so the separate scoring sweep can be skipped.
    #[test]
    fn fitted_sse_matches_the_squared_score() {
        // 200 rows, as in the issue's measurement.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let a = -1.0 + 2.0 * i as f64 / 199.0;
            x.push(vec![a]);
            y.push(0.7 * a);
        }
        let s = Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
        };
        let mut cfg = Config::default();
        cfg.metric = Metric::Squared;
        cfg.fit_iters = 8;
        let data = EvalSample::new(&s, &cfg);

        let mut tree = node(Op::Mul, vec![constant(3.0), input(0)]);
        let fit = fit_constants(&mut tree, &cfg, &s, &s).expect("the ordinary fit path");
        let hash = expr::structural_hash(&tree);
        assert!(
            can_reuse_sse(&cfg, &data, &fit, hash),
            "a squared fit on the scoring rows with nothing folded must be reusable"
        );
        let fitted_error = fit.sse / s.len() as f64;

        let reused = Ind::evaluated_fitted(tree.clone(), Some(fit), &data, &cfg);
        let scored = Ind::evaluated(tree, &data, &cfg);
        assert_eq!(
            reused.error.to_bits(),
            scored.error.to_bits(),
            "the reused SSE must equal the fresh score bit for bit"
        );
        assert_eq!(fitted_error.to_bits(), scored.error.to_bits());
    }

    /// #87: when simplification folds a fitted constant, the stored SSE no
    /// longer describes the tree that is scored, so evaluation is used.
    #[test]
    fn folded_constants_fall_back_to_evaluation() {
        let s = sample(); // y = x0, so the additive constant fits to 0
        let mut cfg = Config::default();
        cfg.metric = Metric::Squared;
        cfg.fit_iters = 8;
        let data = EvalSample::new(&s, &cfg);

        let mut tree = node(Op::Add, vec![input(0), constant(3.0)]);
        let fit = fit_constants(&mut tree, &cfg, &s, &s).expect("the ordinary fit path");
        let simplified = expr::simplify(&tree);
        assert!(
            !can_reuse_sse(&cfg, &data, &fit, expr::structural_hash(&simplified)),
            "a simplification that folded the constant must force a fresh score"
        );

        let fitted = Ind::evaluated_fitted(tree.clone(), Some(fit), &data, &cfg);
        let scored = Ind::evaluated(tree, &data, &cfg);
        assert_eq!(fitted.error.to_bits(), scored.error.to_bits());
    }

    /// #85: the screening subset is redrawn every generation, not reused.
    #[test]
    fn fit_screen_is_redrawn_each_generation() {
        let s = sample();
        let cfg = Config {
            fit_samples: 5,
            ..Default::default()
        };
        let a = fit_for_generation(&cfg, &s, 0).expect("generation 0 screens");
        let b = fit_for_generation(&cfg, &s, 1).expect("generation 1 screens");
        assert_eq!(a.len(), 5);
        assert_ne!(a.x, b.x, "consecutive generations must draw fresh rows");
    }

    /// #85: a survivor is refitted and scored on the full sample before it can
    /// enter the archive.
    #[test]
    fn refit_survivor_scores_on_the_full_sample() {
        let s = sample(); // y = x0
        let cfg = Config {
            metric: Metric::Squared,
            fit_iters: 8,
            ..Default::default()
        };
        // Fit on the first five rows only; the full-data refit must recover
        // the exact constant.
        let screening = subset_sample(&s, &[0, 1, 2, 3, 4]);
        let mut tree = node(Op::Mul, vec![constant(3.0), input(0)]);
        fit_constants(&mut tree, &cfg, &screening, &screening);
        let view = EvalSample::new(&s, &cfg);
        let sol = refit_survivor(&tree, &view, &cfg);
        let full = eval_error_with(&sol.tree, &s, Metric::Squared, true);
        assert_eq!(sol.error.to_bits(), full.to_bits());
        assert!(sol.error < 1e-12, "full-data error = {}", sol.error);
    }

    /// #87: guarded operators must not leave a finite fit SSE behind. The
    /// fitted value is infinite exactly when the scorer rejects the tree, so
    /// the reuse is refused and both paths agree bit for bit.
    #[test]
    fn guarded_trees_do_not_reuse_a_finite_fit_sse() {
        let cfg = Config {
            metric: Metric::Squared,
            fit_iters: 8,
            ..Default::default()
        };

        // 1/(x-c) at the data row: the denominator is zero.
        let s = Sample {
            x: vec![vec![0.5]],
            y: vec![0.0],
            w: Vec::new(),
            ys: Vec::new(),
        };
        let data = EvalSample::new(&s, &cfg);
        let mut tree = node(
            Op::Div,
            vec![
                constant(1.0),
                node(Op::Sub, vec![input(0), constant(0.5)]),
            ],
        );
        let fit = fit_constants(&mut tree, &cfg, &s, &s).expect("ordinary fit path");
        assert!(!fit.sse.is_finite(), "the guard must fire in the fitter");
        let reused = Ind::evaluated_fitted(tree.clone(), Some(fit), &data, &cfg);
        let scored = Ind::evaluated(tree, &data, &cfg);
        assert_eq!(reused.error.to_bits(), scored.error.to_bits());
        assert!(reused.error.is_infinite());

        // exp(701*x) at x=1: `expr::eval` says infinity, not exp(701).
        let s = Sample {
            x: vec![vec![1.0]],
            y: vec![0.0],
            w: Vec::new(),
            ys: Vec::new(),
        };
        let data = EvalSample::new(&s, &cfg);
        let mut tree = node(
            Op::Exp,
            vec![node(Op::Mul, vec![constant(701.0), input(0)])],
        );
        let fit = fit_constants(&mut tree, &cfg, &s, &s).expect("ordinary fit path");
        assert!(!fit.sse.is_finite(), "the guard must fire in the fitter");
        let reused = Ind::evaluated_fitted(tree.clone(), Some(fit), &data, &cfg);
        let scored = Ind::evaluated(tree, &data, &cfg);
        assert_eq!(reused.error.to_bits(), scored.error.to_bits());
        assert!(reused.error.is_infinite());
    }

    /// #77: a child identical to a parent must be caught by the dedup set, not
    /// fitted and scored as a new individual.
    #[test]
    fn parent_clones_are_rejected_before_evaluation() {
        let s = sample();
        let cfg = Config::default();
        // #27: evaluation goes through the SoA view.
        let view = EvalSample::new(&s, &cfg);
        let parent = Ind::evaluated(node(Op::Mul, vec![input(0), constant(2.0)]), &view, &cfg);
        let mut seen = HashSet::new();
        seed_seen(&mut seen, &[], std::slice::from_ref(&parent));
        assert!(
            !seen.insert(expr::structural_hash(&parent.tree)),
            "a clone of a parent must be a duplicate"
        );
    }

    /// #122: the bounded CPU fallback must be a pure parallelisation, not a
    /// change of result. One thread and four threads score the same unsupported
    /// (`if`) and supported trees to the same bits.
    #[test]
    fn cpu_fallback_is_identical_for_one_and_many_threads() {
        let s = sample();
        let cfg = Config::default();
        let view = EvalSample::new(&s, &cfg);
        let trees = vec![
            node(
                Op::If,
                vec![
                    node(Op::Mul, vec![input(0), constant(2.0)]),
                    input(1),
                    constant(0.5),
                ],
            ),
            node(Op::Sin, vec![input(0)]),
            node(Op::Div, vec![constant(1.0), input(0)]),
            node(
                Op::Add,
                vec![node(Op::Square, vec![input(1)]), constant(1.0)],
            ),
        ];
        let make = || {
            vec![
                trees
                    .iter()
                    .cloned()
                    .map(|t| Ind::pending(t, &cfg))
                    .collect::<Vec<_>>(),
                trees
                    .iter()
                    .cloned()
                    .map(|t| Ind::pending(t, &cfg))
                    .collect::<Vec<_>>(),
            ]
        };
        let mut one = make();
        let mut many = make();
        rescore_pending_cpu_with(&mut one, &view, &cfg, 1);
        rescore_pending_cpu_with(&mut many, &view, &cfg, 4);
        for (a, b) in one.iter().flatten().zip(many.iter().flatten()) {
            assert_eq!(a.error.to_bits(), b.error.to_bits());
            assert_eq!(a.complexity.to_bits(), b.complexity.to_bits());
            assert!(expr::same(&a.tree, &b.tree));
        }
        assert!(one[0][0].error.is_finite(), "the unsupported tree was scored");
    }
}
