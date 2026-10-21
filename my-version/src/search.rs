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
#[cfg(feature = "cuda")]
use crate::postfix::{Arena, FrontParents, Variation};
// #18: convergence statistics live in their own module; the search only calls
// `ProgressStats::observe` once per generation.
use crate::progress_stats::{Convergence, ProgressStats};
// #120: the opt-in quality-diversity archive.
use crate::qd::{self, QdArchive, QdReport};
use crate::rng::Mt;
// #25: the dimensional algebra lives in its own module; the search only asks
// `Config::accepts_dimensions`.
use crate::units::Units;
use std::collections::BTreeMap;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VariationMode {
    /// Use the flat postfix engine when the run qualifies, else the `Node` path.
    #[default]
    Auto,
    /// Always use the historical owned-tree loop.
    Node,
    /// Require the flat postfix engine; refuse the run when it does not qualify.
    Postfix,
}

impl VariationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            VariationMode::Auto => "auto",
            VariationMode::Node => "node",
            VariationMode::Postfix => "postfix",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(VariationMode::Auto),
            "node" => Ok(VariationMode::Node),
            "postfix" => Ok(VariationMode::Postfix),
            other => Err(format!(
                "variation must be 'auto', 'node' or 'postfix', got '{}'",
                other
            )),
        }
    }
}

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
    /// #116: exact constant-fitting arm. `Exact` is the historical CPU LM;
    /// `Auto`/`Linear` use the closed-form linear least-squares arm for trees
    /// affine in every constant and fall back to LM otherwise. Changing it
    /// changes the trajectory, so it is a saved search setting.
    pub fit_strategy: crate::fit::FitStrategy,
    /// #116: print one `# fit:` line with the arm counts. Reporting only; it is
    /// not saved in a checkpoint and is ignored by the search settings check.
    pub fit_report: bool,
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
    /// #116: `--fit-rows N`. Like [`fit_samples`](Self::fit_samples), fit
    /// constants on a deterministic `N`-row subsample and score on all rows;
    /// `0` keeps the historical path and `N` at least the sample size is an
    /// exact no-op. When both are set, `fit_rows` wins and is reported as such.
    pub fit_rows: usize,
    /// #116: disk-backed rows for the streaming fit arm. `Some` only when the
    /// CLI resolved a resident `.rcsb` store whose prepared training rows can be
    /// re-read exactly; it is read by `--fit-strategy stream` and never changes
    /// selection or scoring. Like [`simd`](Self::simd), it is a path choice and
    /// is not saved in a checkpoint.
    pub fit_stream_source: Option<std::sync::Arc<crate::fit::StoreSpec>>,
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
    /// #120: bins per descriptor axis for the opt-in quality-diversity archive.
    /// `0` disables it (the default), so every existing trajectory and pinned
    /// archive is unchanged.
    pub qd_bins: usize,
    /// #120: maximum local Pareto front per descriptor cell.
    pub qd_cell_cap: usize,
    /// #120: use a CVT-MAP-Elites archive (K centroids) instead of the
    /// rectangular grid. `qd_bins` is then the number of centroids.
    pub qd_cvt: bool,
    /// #120: feed the archive back into parent selection, migration and
    /// restarts (UCB1, stratified cells, least-occupied reseeding). Off means
    /// the layer is passive and the trajectory is unchanged.
    pub qd_policy: bool,
    /// #120 control: `0` is the UCB1 mix, `1` selects cells uniformly and `2`
    /// hashes descriptors to random cells. Only the first is the proposed
    /// method; the other two are equal-budget controls.
    pub qd_control: u8,
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
    /// #119: bounded producer/scorer overlap depth in batches (`0` disables).
    /// While the GPU scores a batch, the CPU keeps generating and fitting the
    /// next candidates. This is a scheduling choice only: the scored values,
    /// candidate order and archive are identical to the synchronous path, so it
    /// is not saved in a checkpoint.
    pub pipeline_depth: usize,
    /// #122: which variation engine to use. `auto` selects the flat postfix
    /// engine only for a large, fit-free GPU run.
    pub variation: VariationMode,
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
            fit_strategy: crate::fit::FitStrategy::Exact, // #116
            fit_report: false,
            dedup: true,
            time_budget: None,
            patience: 40,
            predictor_samples: None,
            fit_samples: 0, // #85: off by default
            fit_rows: 0,    // #116: off by default
            fit_stream_source: None,
            front_parents: true, // #76: measured on the known-truth suite
            units: None,
            require_dimensions: false,
            qd_bins: 0, // #120: off
            qd_cell_cap: 4,
            qd_cvt: false,
            qd_policy: false,
            qd_control: 0,
            progress: None,
            simd: true, // #27
            // #141: the GPU is the default backend when the binary is built
            // with CUDA and a device is available. The probe happens once at
            // startup; a non-CUDA build, a missing device or an unsupported
            // objective falls back to the CPU. `--no-gpu`/`--cpu` is the
            // deliberate override and `--gpu` asserts that a device exists.
            gpu: cfg!(feature = "cuda"),
            gpu_notes: true,
            pipeline_depth: 2, // #119: only used when the GPU path is active
            variation: VariationMode::Auto,
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
        compare!(fit_strategy, "fit_strategy"); // #116
        compare!(dedup, "dedup");
        if self.time_budget.map(f64::to_bits) != other.time_budget.map(f64::to_bits) {
            diff.push("time_budget");
        }
        compare!(patience, "patience");
        compare!(predictor_samples, "predictor_samples");
        compare!(fit_samples, "fit_samples"); // #85
        compare!(fit_rows, "fit_rows"); // #116
        compare!(front_parents, "front_parents"); // #76
        compare!(qd_bins, "qd_bins"); // #120
        compare!(qd_cell_cap, "qd_cell_cap"); // #120
        compare!(qd_cvt, "qd_cvt"); // #120
        compare!(qd_policy, "qd_policy"); // #120
        compare!(qd_control, "qd_control"); // #120
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
    /// #120: behaviour descriptor accumulated from the same prediction vector
    /// that produced `error`. `None` when the QD layer is off, when the tree
    /// was scored on the GPU, on a row subset, with a metric that has no
    /// prediction vector, or before it has been evaluated. The QD archive also
    /// cross-checks a descriptor against the sample it was computed on before
    /// trusting it.
    descriptor: Option<qd::Descriptor>,
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
        // #120: when the QD layer is on, obtain the descriptor from the same
        // prediction sweep that scores the candidate. If the fitter's SSE is
        // reused, that sweep does not happen; the error is still reused
        // bit-for-bit (the passive layer must not change the trajectory) and
        // the descriptor costs one separate prediction pass, exactly the old
        // fallback. The folded path applies whenever scoring already produces
        // predictions, which is every metric except the squared reuse case.
        let want_descriptor = cfg.qd_bins > 0 && !train.sample.is_multi();
        let (error, descriptor) = if cfg.accepts_dimensions(&tree) {
            match fitted {
                Some(fit) if can_reuse_sse(cfg, train, &fit, hash) => {
                    let d = want_descriptor.then(|| qd::descriptor(&tree, train.sample));
                    (fit.sse / train.sample.len() as f64, d)
                }
                _ => eval_error_and_descriptor(&tree, train, cfg.metric, want_descriptor), // #27
            }
        } else {
            (f64::INFINITY, None)
        };
        Ind {
            tree,
            error,
            complexity,
            descriptor,
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
            // A GPU batch returns the error, not the prediction vector, so the
            // descriptor is filled by the archive's fallback path.
            descriptor: None,
        }
    }

    fn reeval(&mut self, train: &EvalSample, cfg: &Config) {
        self.tree = expr::simplify(&self.tree);
        self.complexity = cfg.complexity_model.complexity(&self.tree);
        // #25: re-evaluation must not resurrect a rejected candidate.
        let want_descriptor = cfg.qd_bins > 0 && !train.sample.is_multi();
        let (error, descriptor) = if cfg.accepts_dimensions(&self.tree) {
            eval_error_and_descriptor(&self.tree, train, cfg.metric, want_descriptor) // #27
        } else {
            (f64::INFINITY, None)
        };
        self.error = error;
        self.descriptor = descriptor;
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
            // A checkpoint stores the tree and score, not the descriptor; the
            // archive recomputes it on the resumed run like any other candidate.
            descriptor: None,
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
    /// #121: persistent bounded CPU pool for unsupported programs.
    fallback: Option<FallbackPool>,
    /// #169: batched GPU-fit outcomes for this run.
    fit_attempts: usize,
    fit_gpu: usize,
    fit_fallback: usize,
    /// #169: pending offspring screened before the fit batch (admission).
    fit_screened: usize,
    /// #169: CPU fallback counts labelled by cause.
    fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #169: wall time spent in the batched fit kernels, milliseconds.
    fit_ms: f64,
    /// #169: wall time spent in the whole fit stage (kernel, launch, CPU
    /// fallback and write-back), milliseconds.
    fit_host_ms: f64,
}

#[cfg(feature = "cuda")]
impl GpuSearch {
    fn new(train: &Sample, fit_data: bool) -> Result<Self, String> {
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
            Some(mut ev) => {
                // #169: the batched fit's final polish runs on the exact f64
                // rows, so its constants are not limited by the f32 evaluator
                // copy. Only the GPU-fit arm pays this second resident copy.
                if fit_data {
                    let mut cols64 = vec![0.0f64; n_rows * n_vars];
                    for (r, row) in train.x.iter().enumerate() {
                        for (v, &x) in row.iter().enumerate().take(n_vars) {
                            cols64[v * n_rows + r] = x;
                        }
                    }
                    let _ = ev.set_fit_data(&cols64, &train.y);
                }
                Ok(GpuSearch {
                    ev,
                    n_rows,
                    pool: None,
                    stats: BTreeMap::new(),
                    fallback: Some(FallbackPool::new(fallback_workers(), fallback_workers())),
                    fit_attempts: 0,
                    fit_gpu: 0,
                    fit_fallback: 0,
                    fit_screened: 0,
                    fit_fallback_by_cause: BTreeMap::new(),
                    fit_ms: 0.0,
                    fit_host_ms: 0.0,
                })
            }
            None => Err(format!(
                "no usable CUDA device ({})",
                crate::gpu::GpuEvaluator::device_name()
            )),
        }
    }

    /// #169: move the constants of the **admitted** pending candidates with
    /// the batched GPU Levenberg–Marquardt kernel.
    ///
    /// The caller screens every unfitted offspring on the GPU with its seed
    /// constants and promotes only the best fraction of each island, so a
    /// generation no longer pays a fitted LM attempt for candidates that lose
    /// the screening round. A candidate is offered to the kernel only when its
    /// tree has an analytic GPU derivative, at most
    /// [`crate::fit::GPU_FIT_MAX_CONSTANTS`] constants, at most
    /// [`crate::fit::GPU_FIT_MAX_NODES`] nodes, and at least `32p` rows.
    /// Everything else — and every result the kernel marks ill-conditioned or
    /// non-finite — is refit by the exact CPU f64 arm, seeded with whatever
    /// constants the kernel produced. Each CPU fallback is labelled by cause
    /// (`unsupported_plan`, `too_few_rows`, `too_many_constants`,
    /// `ill_conditioned`, `non_finite`, `launch`).
    fn fit_pending(
        &mut self,
        islands: &mut [Vec<Ind>],
        admitted: &[(usize, usize)],
        train: &Sample,
        cfg: &Config,
        generation: usize,
    ) {
        let mut jobs: Vec<(usize, usize, crate::fit::GpuFitPlan)> = Vec::new();
        let mut fallbacks: Vec<(usize, usize, FitFallbackCause)> = Vec::new();
        for &(i, j) in admitted {
            match crate::fit::gpu_fit_plan_reason(&islands[i][j].tree) {
                Ok(plan) if train.len() >= 32 * plan.nconst => jobs.push((i, j, plan)),
                Ok(_) => fallbacks.push((i, j, FitFallbackCause::TooFewRows)),
                Err(reason) => fallbacks.push((i, j, plan_fallback_cause(reason))),
            }
        }
        self.fit_attempts += admitted.len();
        let stats = self.stats.entry(generation).or_default();
        stats.generation = generation;
        stats.fit_attempts += admitted.len();
        // A candidate the plan cannot offer is a labelled CPU fallback, so the
        // process-wide `--fit-report` counter agrees with the run ledger.
        for &(_, _, cause) in &fallbacks {
            self.note_fit_fallback(generation, cause);
            crate::fit::note_gpu_fit(false);
        }

        if !jobs.is_empty() {
            let plans: Vec<&crate::fit::GpuFitPlan> = jobs.iter().map(|(_, _, p)| p).collect();
            let (out, ms) = self.ev.fit_batch(&plans, cfg.fit_iters);
            if ms > 0.0 {
                self.fit_ms += ms as f64;
            }
            if out.len() == jobs.len() {
                for ((i, j, plan), res) in jobs.iter().zip(out) {
                    let finite = res.c.iter().all(|v| v.is_finite());
                    if finite {
                        plan.apply(&mut islands[*i][*j].tree, &res.c);
                    }
                    if res.status == 0 && finite {
                        self.fit_gpu += 1;
                        self.stats.entry(generation).or_default().fit_gpu += 1;
                        crate::fit::note_gpu_fit(true);
                    } else {
                        let cause = if finite {
                            status_fallback_cause(res.status)
                        } else {
                            FitFallbackCause::NonFinite
                        };
                        fallbacks.push((*i, *j, cause));
                        self.note_fit_fallback(generation, cause);
                        crate::fit::note_gpu_fit(false);
                    }
                }
            } else {
                // The launch failed: every offered candidate goes to the CPU.
                for (i, j, _) in &jobs {
                    fallbacks.push((*i, *j, FitFallbackCause::Launch));
                    self.note_fit_fallback(generation, FitFallbackCause::Launch);
                    crate::fit::note_gpu_fit(false);
                }
            }
        }

        // Exact CPU fallback, seeded with the GPU constants when there were any.
        for &(i, j, _) in &fallbacks {
            let tree = &mut islands[i][j].tree;
            crate::fit::fit_weighted_metric(
                tree,
                &train.x,
                &train.y,
                &train.w,
                cfg.fit_iters,
                cfg.metric,
            );
        }
    }

    /// #169: record one CPU fallback for this generation and the whole run.
    fn note_fit_fallback(&mut self, generation: usize, cause: FitFallbackCause) {
        self.fit_fallback += 1;
        *self
            .fit_fallback_by_cause
            .entry(cause.label().to_string())
            .or_default() += 1;
        let stats = self.stats.entry(generation).or_default();
        stats.fit_fallback += 1;
        *stats
            .fit_fallback_by_cause
            .entry(cause.label().to_string())
            .or_default() += 1;
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

        if defers_fit_to_gpu(cfg, true) {
            // #169 admission: screen every unfitted offspring with its seed
            // constants first, then fit only the best fraction of each island.
            // The non-promoted offspring keep this screening score, so the
            // generation evaluates every candidate once and only survivors a
            // second time after the fit.
            self.fit_screened += pending.len();
            self.stats.entry(generation).or_default().fit_screened += pending.len();
            self.evaluate_pending(islands, &pending, train, cfg, generation)?;
            let admitted = select_promoted(islands, &pending, train.len());
            let fit_start = std::time::Instant::now();
            self.fit_pending(islands, &admitted, train, cfg, generation);
            self.fit_host_ms += fit_start.elapsed().as_secs_f64() * 1000.0;
            // Re-score the survivors so selection sees their fitted constants.
            self.evaluate_pending(islands, &admitted, train, cfg, generation)?;
            return Ok(());
        }

        self.evaluate_pending(islands, &pending, train, cfg, generation)
    }

    /// #169: compile and score one explicit set of pending candidates in a
    /// single GPU batch, with their current constants. Candidates the kernel
    /// cannot compile are scored on the CPU instead, so they never lose
    /// fitness.
    ///
    /// Any CUDA failure returns `Err`; the caller switches the backend off and
    /// re-scores the pending batch on the CPU, so a failed batch can never
    /// leave a wrong number behind.
    fn evaluate_pending(
        &mut self,
        islands: &mut [Vec<Ind>],
        pending: &[(usize, usize)],
        train: &Sample,
        cfg: &Config,
        generation: usize,
    ) -> Result<(), String> {
        let mut slots: Vec<(usize, usize)> = Vec::with_capacity(pending.len());
        let mut on_cpu: Vec<(usize, usize)> = Vec::new();
        let mut lowered = 0usize;
        // #131: record what the batch actually emitted. Several calls can belong
        // to one generation (initial screening, refit, restarts); merge them.
        // The counts come from `compile_program`, i.e. from the words that were
        // uploaded to the device, so an opcode is reported only if it ran.
        let stats = self.stats.entry(generation).or_default();
        stats.generation = generation;
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
        for &(i, j) in pending {
            let meta = crate::genome::GenomeMeta {
                island: i as u16,
                tag: j as u32,
                ..Default::default()
            };
            // #131: one guarded entry point (lower + stack bound) for the
            // synchronous pool path; #121's lowering/stack split is kept.
            let compiled = crate::gpu::compile_program(&islands[i][j].tree).ok();
            let pushed = compiled
                .as_ref()
                .map_or(false, |c| c.words().len() <= width && pool.push(c.words(), meta));
            match compiled {
                Some(c) if pushed => {
                    if c.lowered() {
                        lowered += 1;
                    }
                    for (name, n) in c.opcodes() {
                        *stats.opcodes.entry(name.clone()).or_default() += *n;
                    }
                    slots.push((i, j));
                }
                _ => on_cpu.push((i, j)),
            }
        }

        stats.programs += pool.len();
        stats.native += pool.len() - lowered;
        stats.lowered += lowered;
        stats.fallbacks += on_cpu.len();
        stats.unsupported_programs += on_cpu.len();
        stats.fallback_workers = self
            .fallback
            .as_ref()
            .expect("a GPU search owns its fallback pool")
            .workers();

        // #121: hand the unsupported trees to the worker pool *before* the
        // kernel launch, so CPU fallback and GPU evaluation overlap. The view
        // must outlive the ticket; `finish` below waits for every chunk before
        // it is dropped.
        let cpu_view = if on_cpu.is_empty() {
            None
        } else {
            Some(EvalSample::new(train, cfg))
        };
        let ticket = cpu_view.as_ref().map(|view| {
            let jobs: Vec<(usize, Node)> = on_cpu
                .iter()
                .map(|&(i, j)| (0, islands[i][j].tree.clone()))
                .collect();
            self.fallback
                .as_ref()
                .expect("a GPU search owns its fallback pool")
                .submit(jobs, view, cfg)
        });

        let mut gpu_error: Option<String> = None;
        if !pool.is_empty() {
            let (sse, ms) = self.ev.eval_pool(pool);
            if ms < 0.0 || sse.len() != pool.len() {
                // Do not return before the submitted fallback batch is drained:
                // a stale result would be read by the next call.
                gpu_error = Some("CUDA batch evaluation failed".to_string());
            } else {
                let n = self.n_rows as f64;
                for (&(i, j), &s) in slots.iter().zip(sse.iter()) {
                    islands[i][j].error = if s.is_finite() {
                        (s as f64) / n
                    } else {
                        f64::INFINITY
                    };
                }
                // The generation is complete: the just-evaluated arena becomes
                // the readable front, and the next generation writes into the
                // other one.
                pool.commit();
            }
        }
        if let Some(ticket) = ticket {
            // #121: the persistent bounded pool scored these off this thread;
            // errors are merged in candidate order, independent of completion
            // order and worker count. The results are discarded on a failed
            // batch, which the caller re-scores in full.
            let (errors, seconds) = self
                .fallback
                .as_ref()
                .expect("a GPU search owns its fallback pool")
                .finish(ticket);
            if gpu_error.is_none() {
                for (&(i, j), &e) in on_cpu.iter().zip(errors.iter()) {
                    islands[i][j].error = e;
                }
            }
            let stats = self.stats.entry(generation).or_default();
            stats.fallback_seconds += seconds;
        }
        if let Some(e) = gpu_error {
            return Err(e);
        }
        Ok(())
    }

    /// #131: the per-generation report, in generation order. #118 adds the
    /// accumulated full-path accounting from the same evaluator.
    fn run_stats(&self) -> GpuRunStats {
        GpuRunStats {
            generations: self
                .stats
                .values()
                .cloned()
                .collect(),
            accounting: self.ev.stats(),
            fit_attempts: self.fit_attempts,
            fit_gpu: self.fit_gpu,
            fit_fallback: self.fit_fallback,
            fit_screened: self.fit_screened,
            fit_fallback_by_cause: self.fit_fallback_by_cause.clone(),
            fit_refits: 0,
            fit_ms: self.fit_ms,
            fit_host_ms: self.fit_host_ms,
        }
    }

    /// #121: move the fallback pool out before the evaluator is dropped, so a
    /// failed CUDA batch can still use it for the CPU rescore.
    fn take_fallback(&mut self) -> FallbackPool {
        self.fallback
            .take()
            .expect("the fallback pool is taken at most once")
    }
}

/// Score every still-pending (`error.is_nan()`) individual on the CPU.
///
/// A tree's error depends only on the tree and the sample, never on a neighbour
/// or on completion order, so the parallel and serial paths return the same
/// bits. The worker count is a function parameter in the unit tests so one and
/// many threads can be compared without touching the process environment.
#[cfg(feature = "cuda")]
fn rescore_pending_cpu(
    islands: &mut [Vec<Ind>],
    train: &EvalSample,
    cfg: &Config,
    pool: &FallbackPool,
) -> f64 {
    let jobs: Vec<(usize, Node)> = islands
        .iter()
        .flat_map(|pop| {
            pop.iter()
                .filter(|ind| ind.error.is_nan())
                .map(|ind| ind.tree.clone())
        })
        .enumerate()
        .collect();
    if jobs.is_empty() {
        return 0.0;
    }
    let (errors, seconds) = pool.score(jobs, train, cfg);
    // Deterministic merge: `errors` is indexed by the (island, slot) order the
    // jobs were collected in, independent of which worker finished first or
    // how many workers there were. The tree and complexity were already set by
    // `Ind::pending`; only the pending fitness is filled in, exactly as the
    // serial `reeval` path would.
    let mut k = 0;
    for pop in islands.iter_mut() {
        for ind in pop.iter_mut() {
            if ind.error.is_nan() {
                ind.error = errors[k];
                k += 1;
            }
        }
    }
    debug_assert_eq!(k, errors.len());
    seconds
}

/// #121: a persistent, bounded CPU fallback pool.
///
/// Unsupported programs used to be scored serially on the calling (broker)
/// thread after every GPU batch; one `if` stalled the whole generation. This
/// pool is created once per GPU run, never evaluates on the calling thread, and
/// keeps at most `workers` chunks queued (a `sync_channel`), so a burst cannot
/// grow memory without bound.
///
/// Jobs carry owned, simplified trees. Results come back tagged with their
/// absolute index and the caller merges them in index order, so the archive and
/// every error bit are identical to the serial walk for any worker count.
#[cfg(feature = "cuda")]
struct FallbackPool {
    tx: std::sync::mpsc::SyncSender<FallbackChunk>,
    done: std::sync::mpsc::Receiver<FallbackResult>,
    workers: usize,
    queue: usize,
    /// Number of worker threads that have executed at least one chunk. Used by
    /// the tests to prove the scoring ran on the pool, not the caller.
    busy: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

/// Borrowed scoring context for one [`FallbackPool::score`] call. The raw
/// pointers are only dereferenced by workers between `score` sending the chunks
/// and receiving the last result; `score` blocks on that result before the
/// borrow ends, so they cannot outlive `train`/`cfg`.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct ScoreContext(*const EvalSample<'static>, *const Config);

// SAFETY: the pointees are read-only for the duration of the call and the
// caller joins every chunk before returning; see the type comment.
#[cfg(feature = "cuda")]
unsafe impl Send for ScoreContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for ScoreContext {}

#[cfg(feature = "cuda")]
struct FallbackChunk {
    start: usize,
    jobs: Vec<(usize, Node)>,
    ctx: ScoreContext,
    blocks: Vec<expr::Block>,
}

#[cfg(feature = "cuda")]
struct FallbackResult {
    start: usize,
    errors: Vec<f64>,
}

#[cfg(feature = "cuda")]
impl FallbackPool {
    /// Spawn `workers` named worker threads. `queue` bounds the number of
    /// chunks waiting in the channel; with `queue == workers` at most
    /// `2 * workers` chunks are in flight (one running per worker, one queued).
    fn new(workers: usize, queue: usize) -> Self {
        let workers = workers.max(1);
        let queue = queue.max(1);
        let (tx, rx) = std::sync::mpsc::sync_channel::<FallbackChunk>(queue);
        let (done_tx, done) = std::sync::mpsc::channel::<FallbackResult>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for i in 0..workers {
            let rx = std::sync::Arc::clone(&rx);
            let done_tx = done_tx.clone();
            let busy = std::sync::Arc::clone(&busy);
            std::thread::Builder::new()
                .name(format!("rcsr-fallback-{i}"))
                .spawn(move || {
                    let mut counted = false;
                    loop {
                        let chunk = {
                            let guard = rx.lock().expect("the fallback queue is not poisoned");
                            guard.recv()
                        };
                        let Ok(chunk) = chunk else { break };
                        if !counted {
                            busy.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            counted = true;
                        }
                        // #10: custom blocks are thread-local.
                        expr::set_blocks(chunk.blocks);
                        // SAFETY: the context borrows `train`/`cfg` from
                        // `score`, which is blocked waiting for this result.
                        let (train, cfg) = unsafe { (&*chunk.ctx.0, &*chunk.ctx.1) };
                        let mut errors = Vec::with_capacity(chunk.jobs.len());
                        for (_, tree) in &chunk.jobs {
                            errors.push(score_tree(tree, train, cfg));
                        }
                        if done_tx
                            .send(FallbackResult {
                                start: chunk.start,
                                errors,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .expect("spawn a CPU fallback worker");
        }
        FallbackPool {
            tx,
            done,
            workers,
            queue,
            busy,
        }
    }

    /// Send `jobs` to the worker pool and return a ticket. Sending blocks only
    /// while the bounded channel is full, so this returns as soon as every
    /// chunk is queued or running; [`FallbackPool::finish`] waits for the
    /// results. Splitting submit from finish lets the caller overlap the CPU
    /// fallback with the GPU kernel instead of paying a serial tail.
    fn submit(&self, jobs: Vec<(usize, Node)>, train: &EvalSample, cfg: &Config) -> FallbackTicket {
        debug_assert!(self.queue >= 1, "the fallback queue is bounded but non-empty");
        debug_assert!(
            self.busy.load(std::sync::atomic::Ordering::Relaxed) <= self.workers,
            "more worker threads ran than the pool owns"
        );
        let total = jobs.len();
        let started = std::time::Instant::now();
        // Contiguous, fixed-size chunks: ~4 chunks per worker keeps the queue
        // fed without making the channel the bottleneck.
        let chunk_len = total.div_ceil(self.workers * 4).max(1);
        let blocks = expr::current_blocks();
        let ctx = ScoreContext(
            train as *const EvalSample as *const EvalSample<'static>,
            cfg as *const Config,
        );
        let mut chunks = 0usize;
        let mut pending = jobs.into_iter().peekable();
        while pending.peek().is_some() {
            let start = chunks;
            let jobs: Vec<(usize, Node)> = pending.by_ref().take(chunk_len).collect();
            self.tx
                .send(FallbackChunk {
                    start,
                    jobs,
                    ctx,
                    blocks: blocks.clone(),
                })
                .expect("the fallback pool is alive");
            chunks += 1;
        }
        FallbackTicket {
            chunks,
            chunk_len,
            total,
            started,
        }
    }

    /// Wait for a submitted batch and return errors in job order plus the
    /// wall-clock seconds since submission.
    fn finish(&self, ticket: FallbackTicket) -> (Vec<f64>, f64) {
        let mut out = vec![f64::NAN; ticket.total];
        let mut received = 0;
        while received < ticket.chunks {
            let result = self
                .done
                .recv()
                .expect("the fallback pool returned a result");
            for (offset, error) in result.errors.into_iter().enumerate() {
                // `start` is the chunk ordinal; the absolute job index is
                // recomputed from the chunk's own position in the job list.
                let idx = result.start * ticket.chunk_len + offset;
                if idx < ticket.total {
                    out[idx] = error;
                }
            }
            received += 1;
        }
        (out, ticket.started.elapsed().as_secs_f64())
    }

    /// Score `jobs` on the pool and return errors in the same order, plus the
    /// wall-clock fallback seconds.
    fn score(&self, jobs: Vec<(usize, Node)>, train: &EvalSample, cfg: &Config) -> (Vec<f64>, f64) {
        let ticket = self.submit(jobs, train, cfg);
        self.finish(ticket)
    }

    fn workers(&self) -> usize {
        self.workers
    }
}

/// #121: a submitted fallback batch, waiting to be collected.
#[cfg(feature = "cuda")]
struct FallbackTicket {
    chunks: usize,
    chunk_len: usize,
    total: usize,
    started: std::time::Instant,
}

/// #121: size of the bounded CPU fallback pool.
#[cfg(feature = "cuda")]
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
            // #121: keep the fallback pool alive for the rescore below.
            let fallback = g.take_fallback();
            *gpu = None;
            // #169 admission: if the CUDA batch failed after the screening pass,
            // the pending offspring may not have been fitted yet. Fit them with
            // the exact CPU arm before re-scoring so the fallback never silently
            // selects on seed constants. A rescore failure after the batch fit
            // simply warm-starts the same CPU LM again.
            if defers_fit_to_gpu(cfg, true) {
                for pop in islands.iter_mut() {
                    for ind in pop.iter_mut() {
                        if ind.error.is_nan() {
                            crate::fit::fit_weighted_metric(
                                &mut ind.tree,
                                &train.x,
                                &train.y,
                                &train.w,
                                cfg.fit_iters,
                                cfg.metric,
                            );
                        }
                    }
                }
            }
            // #27: re-scoring the pending batch goes through the SoA view.
            let train_view = EvalSample::new(train, cfg);
            // #121: the same bounded worker pool as the per-batch fallback.
            rescore_pending_cpu(islands, &train_view, cfg, &fallback);
            false
        }
    }
}

/// #119: score one overlap batch, writing each result into its fixed
/// `(island, slot)` destination. Candidates the kernel cannot compile are left
/// `None` and scored by the normal barrier path afterwards. A CUDA failure is
/// returned to the caller, which stops the GPU and re-scores the whole
/// generation on the CPU.
#[cfg(feature = "cuda")]
fn score_pipeline_batch(
    jobs: Vec<PipelineJob>,
    gpu: &mut Option<GpuSearch>,
    results: &mut [Vec<Option<f64>>],
    n_rows: usize,
    generation: usize,
) -> Result<(), String> {
    let Some(g) = gpu.as_mut() else {
        return Err("pipeline scoring without a GPU evaluator".to_string());
    };
    let mut pop = crate::gpu::Population::new();
    let mut slots: Vec<(usize, usize)> = Vec::with_capacity(jobs.len());
    for (i, j, tree) in jobs {
        if pop.add(&tree) {
            slots.push((i, j));
        }
    }
    // #121: the overlap path uses the same lowering and reports the same
    // supported/lowered/unsupported split as the synchronous path.
    // #131: it also reports the opcodes of the programs actually uploaded.
    let stats = g.stats.entry(generation).or_default();
    stats.generation = generation;
    stats.programs += pop.len();
    stats.native += pop.native();
    stats.lowered += pop.lowered();
    stats.fallbacks += pop.unsupported();
    stats.unsupported_programs += pop.unsupported();
    for (name, n) in pop.opcode_usage() {
        *stats.opcodes.entry(name).or_default() += n;
    }
    if pop.is_empty() {
        return Ok(());
    }
    let (sse, ms) = g.ev.eval_population(&pop);
    if ms < 0.0 || sse.len() != pop.len() {
        return Err("CUDA batch evaluation failed".to_string());
    }
    let n = n_rows as f64;
    for (&(i, j), &s) in slots.iter().zip(&sse) {
        results[i][j] = Some(if s.is_finite() {
            (s as f64) / n
        } else {
            f64::INFINITY
        });
    }
    Ok(())
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
    // #85/#116: the screened path scores offspring on a per-generation subset,
    // which the resident full-sample GPU evaluator cannot represent.
    let fit_rows = effective_fit_rows(cfg, train);
    if fit_rows > 0 && fit_rows < train.len() {
        return Some("--fit-samples/--fit-rows scores offspring on a screening subset");
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
    eval_error_and_descriptor(tree, data, metric, false).0
}

/// #120: score a candidate and, when asked, accumulate its behaviour
/// descriptor from the *same* prediction vector.
///
/// This is the one-pass contract: the scoring reduction already visits every
/// row of `pred`, so the usage count (a tree walk) and the prediction moments
/// (`PredictionStats`) are computed without a second tree evaluation. The QD
/// layer therefore costs the search nothing beyond the statistics themselves
/// on the CPU scoring path.
///
/// Two shapes deliberately have no descriptor: the implicit metric evaluates
/// per row through the gradient (the fallback evaluates once more), and a
/// multi-output objective has one prediction vector per output and keeps its
/// own path. Both are reported by the QD archive's fallback.
fn eval_error_and_descriptor(
    tree: &Node,
    data: &EvalSample,
    metric: Metric,
    want_descriptor: bool,
) -> (f64, Option<qd::Descriptor>) {
    let s = data.sample;
    if s.is_empty() {
        return (f64::INFINITY, None);
    }
    let nvars = s.x.first().map(Vec::len).unwrap_or(0);
    // #14: an implicit metric measures the distance to the level set
    // |f| / ||grad f||. That needs f and its gradient at each row, so it is
    // scored from the tree and the row instead of a prediction vector. Every
    // other metric takes the original path, bit for bit.
    if metric.is_implicit() {
        let error = crate::implicit::score(tree, s, metric);
        let descriptor = want_descriptor.then(|| qd::descriptor(tree, s));
        return (error, descriptor);
    }
    let pred = predictions(tree, data);
    // The parameter count lets the information criteria (AIC/AICc/BIC) work as
    // search metrics; it is ignored by every other metric. The weight vector is
    // empty for an unweighted sample, which takes the identical unweighted path.
    let error = score_weighted(
        &pred,
        &s.y,
        &s.w,
        metric,
        expr::count_constants(tree),
    );
    let descriptor =
        want_descriptor.then(|| qd::descriptor_from_prediction(tree, &pred, nvars));
    (error, descriptor)
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

/// #14, #117: fit a candidate's constants on the objective its metric actually
/// scores.
///
/// The implicit-derivative metric is scale-free. Fitting `f` against the zero
/// target with ordinary least squares would shrink every multiplicative
/// constant towards zero, collapsing the level set into the trivial `f == 0`;
/// the guard rightly rejects that candidate, so a good shape would be destroyed
/// before it could be scored. In this one case [`crate::implicit::fit`]
/// minimises the signed geometric residual `f / ||grad f||` instead.
///
/// Every explicit metric goes through [`crate::fit::fit_weighted_metric`], which
/// keeps `squared`/AIC/AICc/BIC on the exact historical least-squares path and
/// fits the configured loss for the others (IRLS, logistic IRLS, or a
/// documented coordinate polish for rank/order losses).
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
        // #116: when the CLI resolved a disk-backed source for exactly these
        // training rows, the stream arm reads them group by group instead of
        // the resident `Sample`. This is the same exact f64 sufficient-
        // statistic merge; a read/conditioning failure falls through to the
        // resident strategy below.
        let squared_family = matches!(
            cfg.metric,
            Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic
        );
        let full_training_rows = std::ptr::eq(fit, score)
            && cfg.fit_samples == 0
            && cfg.fit_rows == 0
            && cfg.predictor_samples.is_none();
        if cfg.fit_strategy == crate::fit::FitStrategy::Stream
            && squared_family
            && full_training_rows
        {
            if let Some(spec) = &cfg.fit_stream_source {
                let source = crate::fit::StoreRows::new(spec.clone());
                if let Some(sse) = crate::fit::fit_stream_source(tree, &source, iters) {
                    return Some(FitResult {
                        sse,
                        hash: expr::structural_hash(tree),
                        same_rows: true,
                    });
                }
            }
        }
        // #116 + #117: strategy and metric together.
        let sse = crate::fit::fit_weighted_strategy_metric(
            tree,
            &fit.x,
            &fit.y,
            &fit.w,
            iters,
            cfg.fit_strategy,
            cfg.metric,
        );
        Some(FitResult {
            sse,
            hash: expr::structural_hash(tree),
            // #85: an SSE fitted on a screening subset is not the score of the
            // full sample, so the reuse below must stay off.
            same_rows: std::ptr::eq(fit, score),
        })
    }
}

/// #87: the fitter's own value at the constants it wrote back (the residual
/// sum of squares for the squared family, the configured loss otherwise), plus
/// the structural hash of the tree at that moment.
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

#[cfg(test)]
thread_local! {
    /// #85: how many times [`refit_survivor`] reused the fitter's final SSE
    /// instead of scoring the promoted tree again.
    static REFIT_SSE_REUSED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

/// #85/#116: rows requested for the fit screen. `--fit-rows` takes precedence
/// when both are set (the CLI says so), and both ignore a request at or above
/// the sample size so that `--fit-rows N` with `N >= train` is an exact full
/// fit.
fn effective_fit_rows(cfg: &Config, sample: &Sample) -> usize {
    let request = if cfg.fit_rows >= 2 {
        cfg.fit_rows
    } else {
        cfg.fit_samples
    };
    if cfg.fit_iters == 0 || request < 2 || request >= sample.len() {
        0
    } else {
        request
    }
}

/// #169: why an admitted candidate still needed the exact CPU fit arm.
///
/// The labels are stable: they appear in `gpu.fit.fallback_by_cause`, in each
/// generation's ledger and in `--fit-report`/docs. Structural rejections are
/// grouped into `unsupported_plan` except the explicit `p > 16` cap, which has
/// its own label because it is the only limit the kernel could lift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitFallbackCause {
    /// No analytic GPU derivative, too many nodes, shared block constants, or
    /// no constants at all.
    UnsupportedPlan,
    /// Fewer than `32p` rows, so the normal-equation system is underdetermined.
    TooFewRows,
    /// More than [`crate::fit::GPU_FIT_MAX_CONSTANTS`] constants (`p > 16`).
    TooManyConstants,
    /// The kernel's conditioning gate rejected the normal equations (status 1).
    IllConditioned,
    /// A non-finite value, derivative or residual (status 2).
    NonFinite,
    /// The CUDA fit launch failed; the whole batch went to the CPU.
    Launch,
}

impl FitFallbackCause {
    /// Every cause, in reporting order.
    pub const ALL: [FitFallbackCause; 6] = [
        FitFallbackCause::UnsupportedPlan,
        FitFallbackCause::TooFewRows,
        FitFallbackCause::TooManyConstants,
        FitFallbackCause::IllConditioned,
        FitFallbackCause::NonFinite,
        FitFallbackCause::Launch,
    ];

    /// Stable JSON label.
    pub fn label(self) -> &'static str {
        match self {
            FitFallbackCause::UnsupportedPlan => "unsupported_plan",
            FitFallbackCause::TooFewRows => "too_few_rows",
            FitFallbackCause::TooManyConstants => "too_many_constants",
            FitFallbackCause::IllConditioned => "ill_conditioned",
            FitFallbackCause::NonFinite => "non_finite",
            FitFallbackCause::Launch => "launch",
        }
    }
}

/// #169: map a structural fit-plan rejection to its reported cause.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn plan_fallback_cause(reason: crate::fit::GpuFitReject) -> FitFallbackCause {
    match reason {
        crate::fit::GpuFitReject::TooManyConstants => FitFallbackCause::TooManyConstants,
        _ => FitFallbackCause::UnsupportedPlan,
    }
}

/// #169: map the kernel's runtime status to its reported cause.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn status_fallback_cause(status: i32) -> FitFallbackCause {
    match status {
        1 => FitFallbackCause::IllConditioned,
        2 => FitFallbackCause::NonFinite,
        _ => FitFallbackCause::UnsupportedPlan,
    }
}

/// #169: training size at which the seed-constant screen stops paying for
/// itself.
///
/// Below this many rows the screen is a useful proxy: it ranks offspring by
/// their inherited/parent-fitted constants, which is close to what the LM will
/// produce, and cutting the fitted fraction keeps the fit volume down. At
/// `N >= 1 048 576` the same screen becomes almost uninformative while the
/// exact CPU fit dominates the generation: the 3.9M hourly probe admitted only
/// `pop/4` and reached best error 0.4753 against 0.1612 for the exact arm at
/// equal evaluations. Widening admission to every candidate with constants
/// restores the exact arm's fit coverage; the batched kernel is still 17-25x
/// faster per candidate, so the arm remains a speed-up. The threshold is a row
/// count, never a timing measurement, so the admission set stays reproducible.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const WIDE_ADMISSION_ROWS: usize = 1 << 20;

/// #169: how many offspring of one island are admitted to the fitted LM batch.
///
/// Below [`WIDE_ADMISSION_ROWS`] the fixed fraction `pop/4` is floored at 32 so
/// a small island still admits the candidates the kernel is best at, and capped
/// at the island size so the default and the tiny test configurations are exact
/// no-ops. At and above the threshold every candidate with constants is
/// admitted, because the seed-constant screen no longer predicts the fitted
/// ranking well enough to leave three quarters of the offspring unfitted (the
/// 3.9M hourly cell failed its quality gate that way). The choice uses only the
/// population size and the row count, never timing, so it is reproducible.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn admission_count(island_pop: usize, train_rows: usize) -> usize {
    if train_rows >= WIDE_ADMISSION_ROWS {
        return island_pop.max(1);
    }
    ((island_pop / 4).max(32)).min(island_pop).max(1)
}

/// #169: deterministic admission of pending offspring to the fitted LM batch.
///
/// Every pending (`error.is_nan()`) individual is screened with its seed
/// constants first. Per island, the best `admission_count(island_pop,
/// train_rows)` are promoted; ties break on complexity and then on the
/// population index, so the set depends only on the screened errors, the row
/// count and the fixed island/population order. Non-promoted offspring keep
/// their screening score. At `train_rows >= WIDE_ADMISSION_ROWS` every
/// candidate with constants is promoted, so the screen can only exclude trees
/// that have nothing to fit.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn select_promoted(
    islands: &[Vec<Ind>],
    pending: &[(usize, usize)],
    train_rows: usize,
) -> Vec<(usize, usize)> {
    let mut promoted = Vec::with_capacity(pending.len());
    for (i, pop) in islands.iter().enumerate() {
        let mut ranked: Vec<(usize, f64, f64)> = pending
            .iter()
            .filter(|(island, _)| *island == i)
            .map(|&(_, j)| (j, pop[j].error, pop[j].complexity))
            // #169: a candidate with no constants has nothing to fit. Keep its
            // screening score and do not spend an admission slot or bill a
            // meaningless CPU fallback for it.
            .filter(|&(j, _, _)| {
                !matches!(
                    crate::fit::gpu_fit_plan_reason(&pop[j].tree),
                    Err(crate::fit::GpuFitReject::NoConstants)
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.2.partial_cmp(&b.2)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(a.0.cmp(&b.0))
        });
        let take = admission_count(pop.len(), train_rows).min(ranked.len());
        promoted.extend(ranked[..take].iter().map(|&(j, _, _)| (i, j)));
    }
    // Fitting order is deterministic; the candidates are independent, but a
    // stable order keeps the ledger and any debugging output reproducible.
    promoted.sort_unstable();
    promoted
}

/// #169: true when this generation's offspring are left unfitted for the
/// batched GPU LM arm. The kernel minimises squared error, so only that metric
/// qualifies; every other configuration keeps the CPU fit path it had before.
fn defers_fit_to_gpu(cfg: &Config, defer_gpu: bool) -> bool {
    defer_gpu
        && cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && matches!(
            cfg.metric,
            Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic
        )
}

/// #85: true when constants are fitted on fewer rows than the generation's
/// scoring sample. A request at least as large as the sample is a no-op.
fn fit_sampling_active(cfg: &Config, sample: &Sample) -> bool {
    effective_fit_rows(cfg, sample) > 0
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
        effective_fit_rows(cfg, sample),
        cfg.seed ^ FIT_SUBSET_SALT,
        generation,
    );
    Some(subset_sample(sample, &idx))
}

/// #85: refit one screening survivor on the full training sample and score it
/// there, ready to enter the archive.
fn refit_survivor(tree: &Node, train: &EvalSample, cfg: &Config) -> Sol {
    let mut tree = expr::simplify(tree);
    // #85/#87: the promoted tree is fitted on exactly the rows that will score
    // it, so for the squared metric the fitter's final SSE is the score.
    // Reusing it removes one full-data sweep per survivor; the score is the
    // same bit pattern (see `Ind::evaluated_fitted`).
    let mut fitted = None;
    if cfg.fit_iters > 0 && !train.sample.is_multi() {
        fitted = fit_constants(&mut tree, cfg, train.sample, train.sample);
        tree = expr::simplify(&tree);
    }
    let hash = expr::structural_hash(&tree);
    let error = if cfg.accepts_dimensions(&tree) {
        match fitted {
            Some(fit) if can_reuse_sse(cfg, train, &fit, hash) => {
                #[cfg(test)]
                REFIT_SSE_REUSED.with(|n| n.set(n.get() + 1));
                fit.sse / train.sample.len() as f64
            }
            _ => score_tree(&tree, train, cfg), // #27
        }
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
    // #169: refit every finalist with the exact CPU f64 arm because the search
    // used the batched GPU constants.
    refit_gpu_fit: bool,
) -> (Archive, usize) {
    let trees: Vec<&Node> = archive
        .items
        .iter()
        .map(|s| &s.tree)
        .chain(islands.iter().flat_map(|pop| pop.iter().map(|ind| &ind.tree)))
        .collect();
    let mut unique: Vec<Node> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for tree in trees {
        if seen.insert(expr::structural_hash(tree)) {
            unique.push(expr::simplify(tree));
        }
    }
    let mut refits = 0usize;
    // #169: reported constants must come from the exact CPU f64 arm, not from
    // the f32 row arithmetic of the batched kernel. This is a warm start from
    // the GPU constants on the full training sample. Trees are independent, so
    // the refit is split across workers; the result order is unchanged.
    if refit_gpu_fit && cfg.fit_iters > 0 && !train.sample.is_multi() {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8)
            .min(unique.len().max(1));
        let blocks = expr::current_blocks();
        if workers > 1 && unique.len() > 1 {
            let per = unique.len().div_ceil(workers);
            std::thread::scope(|scope| {
                for chunk in unique.chunks_mut(per) {
                    let blocks = blocks.clone();
                    scope.spawn(move || {
                        expr::set_blocks(blocks);
                        for t in chunk.iter_mut() {
                            crate::fit::fit_weighted_metric(
                                t,
                                &train.sample.x,
                                &train.sample.y,
                                &train.sample.w,
                                cfg.fit_iters,
                                cfg.metric,
                            );
                            *t = expr::simplify(t);
                        }
                    });
                }
            });
        } else {
            for t in unique.iter_mut() {
                crate::fit::fit_weighted_metric(
                    t,
                    &train.sample.x,
                    &train.sample.y,
                    &train.sample.w,
                    cfg.fit_iters,
                    cfg.metric,
                );
                *t = expr::simplify(t);
            }
        }
        for _ in 0..unique.len() {
            crate::fit::note_gpu_refit();
        }
        refits = unique.len();
    }
    let mut out = Archive::new(archive.cap);
    for tree in unique {
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
    (out, refits)
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

/// #120: one parent under the QD policy or the original #76 draw.
///
/// The QD branch reads the archive snapshot only; it never mutates the archive
/// inside a generation. `front_p` keeps the 5% global-front draw that the
/// issue reserves for the diversity policy.
fn pick_parent_policy(
    island: &[Ind],
    front: &[Sol],
    qd: Option<&QdArchive>,
    rng: &mut Mt,
    front_p: f64,
    qd_policy: bool,
    qd_control: u8,
) -> Node {
    if qd_policy {
        // The issue reserves 5 % of QD parent draws for the global front.
        if let Some(tree) = front_parent(front, rng, 0.05) {
            return tree.clone();
        }
        let parent = qd.and_then(|q| {
            if qd_control == 1 {
                q.select_parent_uniform(rng)
            } else {
                q.select_parent(rng)
            }
        });
        if let Some(tree) = parent {
            return tree.clone();
        }
    }
    pick_parent(island, front, rng, front_p).clone()
}

// #77: seed the per-generation duplicate set with the elite (`next`) and every
// parent (`isl`), so a child that is an unchanged parent is rejected before the
// expensive fit and score.
fn seed_seen(seen: &mut HashSet<u64>, next: &[Ind], parents: &[Ind]) {
    for ind in next.iter().chain(parents) {
        seen.insert(expr::structural_hash(&ind.tree));
    }
}

/// #119: one candidate job for the overlap queue: island, destination slot in
/// that island's `next` vector, and the tree to score.
type PipelineJob = (usize, usize, Node);

/// #119: offer a freshly generated candidate to the overlap queue. Sending is
/// best-effort by design: the consumer drains until every sender is dropped, so
/// a send can only fail if the whole pipeline has already stopped (it never
/// does while producers run). The candidate itself stays in the producer's
/// `next`, so a dropped job is simply scored at the barrier.
fn offer(sink: &Option<(&std::sync::mpsc::SyncSender<PipelineJob>, usize)>, slot: usize, ind: &Ind) {
    if let Some((tx, island)) = sink {
        let _ = tx.send((*island, slot, ind.tree.clone()));
    }
}

/// Evolve one island by a single generation. Pure with respect to the rest of
/// the world, so it can run on its own thread.
#[allow(clippy::too_many_arguments)]
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
    // #120: the QD archive at the start of this generation (read-only).
    qd: Option<&QdArchive>,
    qd_policy: bool,
    qd_control: u8,
    // #119: when present, every pending candidate is also offered to the
    // bounded GPU queue as soon as it is fitted, so scoring overlaps the rest
    // of this generation. `None` is the synchronous path exactly.
    sink: Option<(&std::sync::mpsc::SyncSender<PipelineJob>, usize)>,
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
        // #120: with the policy on, a UCB/stratified cell elite supplies most
        // parents; 5% still come from the global Pareto front. With the policy
        // off this is exactly the #76 draw.
        let p1 = pick_parent_policy(isl, front, qd, rng, front_p, qd_policy, qd_control);
        let mut child = p1;
        if rng.bool(0.5) {
            let parent2 = pick_parent_policy(isl, front, qd, rng, front_p, qd_policy, qd_control);
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
        // #169: with the GPU-fit arm active the constants are moved by the
        // batched kernel before scoring; fitting them here would hide the
        // offload behind a CPU LM pass.
        if cfg.fit_iters > 0 && !train.sample.is_multi() && !defers_fit_to_gpu(cfg, defer_gpu) {
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
        let slot = next.len();
        next.push(ind);
        offer(&sink, slot, &next[slot]); // #119
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
        if cfg.fit_iters > 0 && !train.sample.is_multi() && !defers_fit_to_gpu(cfg, defer_gpu) {
            fitted = fit_constants(&mut tree, cfg, fit_sample, train.sample); // #85 #87
            tree = expr::simplify(&tree);
        }
        // #26: same deferral for the random fill.
        let slot = next.len();
        next.push(if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            Ind::evaluated_fitted(tree, fitted, train, cfg) // #25 #87
        });
        offer(&sink, slot, &next[slot]); // #119
    }

    // #120: a small niche-immigrant stream. Without it mutation keeps the
    // island in the same descriptor region even when parents are drawn from
    // the archive. The immigrants are aimed at the least-visited cell and are
    // scored exactly like any other offspring (the GPU batch defers them too).
    if qd_policy {
        if let Some(target) = qd.and_then(|q| q.least_occupied_descriptor()) {
            let immigrants = (pop / 10).max(1).min(pop);
            for k in 0..immigrants {
                let mut t = random_tree_with(rng, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                let mut best_score = {
                    let d = qd::descriptor(&t, train.sample);
                    (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2)
                };
                for _ in 0..3 {
                    let cand = random_tree_with(rng, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                    let d = qd::descriptor(&cand, train.sample);
                    let score = (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2);
                    if score < best_score {
                        t = cand;
                        best_score = score;
                    }
                }
                let mut tree = expr::simplify(&t);
                let mut fitted = None;
                if cfg.fit_iters > 0 && !train.sample.is_multi() {
                    fitted = fit_constants(&mut tree, cfg, fit_sample, train.sample);
                    tree = expr::simplify(&tree);
                }
                let slot = pop - immigrants + k;
                next[slot] = if defer_gpu {
                    Ind::pending(tree, cfg)
                } else {
                    Ind::evaluated_fitted(tree, fitted, train, cfg)
                };
                offer(&sink, slot, &next[slot]);
            }
        }
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
    /// Operator occurrences in the programs that ran on the GPU, sorted by
    /// opcode name. Empty only when the batch contained no operator words.
    pub opcodes: BTreeMap<String, usize>,
    /// Programs compiled into the batch.
    pub programs: usize,
    /// #121: compiled programs the kernel accepted verbatim.
    pub native: usize,
    /// #121: compiled programs that needed the exact lowering pass.
    pub lowered: usize,
    /// Candidates sent to the CPU fallback in the same batch.
    pub fallbacks: usize,
    /// #121: alias of `fallbacks`, matching the issue's JSON name.
    pub unsupported_programs: usize,
    /// #121: wall-clock seconds the CPU fallback pool spent in this batch.
    pub fallback_seconds: f64,
    /// #121: worker threads in the fallback pool.
    pub fallback_workers: usize,
    /// #169: candidates offered to the batched GPU fit.
    pub fit_attempts: usize,
    /// #169: candidates the batched GPU fit returned usable constants for.
    pub fit_gpu: usize,
    /// #169: candidates refit by the exact CPU arm (unsupported, ill-conditioned
    /// or non-finite).
    pub fit_fallback: usize,
    /// #169: pending offspring screened before admission.
    pub fit_screened: usize,
    /// #169: CPU fallback counts labelled by cause.
    pub fit_fallback_by_cause: BTreeMap<String, usize>,
}

/// #131: aggregate GPU usage for a run.
#[derive(Clone, Debug, Default)]
pub struct GpuRunStats {
    pub generations: Vec<GpuGenStats>,
    /// #118: measured full-path accounting for every GPU batch in the run.
    /// Unlike `generations`, this is present even when the pipeline path (not
    /// the per-generation barrier) did the scoring.
    pub accounting: crate::gpu_stats::GpuStats,
    /// #169: batched GPU-fit totals for the run.
    pub fit_attempts: usize,
    pub fit_gpu: usize,
    pub fit_fallback: usize,
    /// #169: pending offspring screened before admission.
    pub fit_screened: usize,
    /// #169: CPU fallback counts labelled by cause.
    pub fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #169: exact CPU f64 refits of finalists before reporting.
    pub fit_refits: usize,
    /// #169: wall time spent in the batched fit kernels, milliseconds.
    pub fit_ms: f64,
    /// #169: wall time spent in the whole fit stage, milliseconds.
    pub fit_host_ms: f64,
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
    /// #116: effective `--fit-rows` request used to fit constants, or `0` when
    /// the full training sample was used. This is the fit-only screen; every
    /// reported error and every archived constant is refit on all rows.
    pub fit_rows: usize,
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
    /// #120: quality-diversity coverage report, or `None` when the layer is off.
    pub qd: Option<QdReport>,
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

/// #197: how close to `0` or `1` a fitted constant must be to count as a
/// neutral identity. This is the explicit seed-canonicalisation tolerance.
const SEED_IDENTITY_TOL: f64 = 1e-12;

/// #197: remove neutral identity nodes from a fitted seed.
///
/// A domain-safe simplifier may keep `0 + x`, `x - 0`, `1 * x`, `x / 1`,
/// `x ^ 1` and `-(-x)` because it cannot prove the operand finite, or because
/// it is preserving the sign of a zero. For a user-supplied seed those nodes
/// are noise: they make the seed's structural hash and complexity differ from
/// the identical expression the search itself generates. The search then finds
/// the smaller tree, which correctly dominates and removes the seed from the
/// Pareto front, and the seed appears to have been lost.
///
/// Seeds are canonicalised here, after fitting, so that `2 * sqrt(m / k) + 0`
/// is the same tree as `sqrt(m / k)` regardless of how conservative the
/// simplifier is. A neutral constant is one within [`SEED_IDENTITY_TOL`] of
/// `0` or `1`, because the fitter lands on `6e-18` rather than exact zero for
/// an offset it has driven to zero. That tolerance is an explicit, documented
/// approximation: the dropped term changes the seed's value by less than
/// `1e-12`, far below the error the search compares. The rewrite otherwise
/// preserves the value for finite inputs, `NaN` and the infinities.
fn canonicalize_seed(n: &Node) -> Node {
    use Op::*;
    let mut m = Node {
        op: n.op,
        val: n.val,
        kids: n.kids.iter().map(canonicalize_seed).collect(),
        block: n.block.clone(),
    };
    if m.op == Const || m.op == Input {
        return m;
    }
    if m.op.is_custom() {
        if let Some(block) = &mut m.block {
            block.body = canonicalize_seed(&block.body);
        }
        return m;
    }
    let zero = |k: &Node| k.op == Const && k.val.abs() < SEED_IDENTITY_TOL;
    let one = |k: &Node| k.op == Const && (k.val - 1.0).abs() < SEED_IDENTITY_TOL;
    match m.op {
        Add => {
            if zero(&m.kids[0]) {
                return m.kids[1].clone();
            }
            if zero(&m.kids[1]) {
                return m.kids[0].clone();
            }
        }
        Sub => {
            if zero(&m.kids[1]) {
                return m.kids[0].clone();
            }
        }
        Mul => {
            if one(&m.kids[0]) {
                return m.kids[1].clone();
            }
            if one(&m.kids[1]) {
                return m.kids[0].clone();
            }
        }
        Div => {
            if one(&m.kids[1]) {
                return m.kids[0].clone();
            }
        }
        Pow => {
            if one(&m.kids[1]) {
                return m.kids[0].clone();
            }
        }
        Neg => {
            if m.kids[0].op == Neg && m.kids[0].kids.len() == 1 {
                return m.kids[0].kids[0].clone();
            }
        }
        _ => {}
    }
    m
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
        // #197: make the fitted seed the same canonical tree the search would
        // generate, so a neutral term in the user's text cannot make the seed
        // structurally distinct (and therefore dominated) later.
        tree = canonicalize_seed(&tree);
        tree = expr::simplify(&tree);
        if !seen_fitted.insert(expr::structural_hash(&tree)) {
            continue;
        }
        out.push(Ind::evaluated_fitted(tree, fitted, train, cfg)); // #25 #87
    }
    out
}


/// #122: whether a run can use the flat postfix variation engine.
///
/// The engine owns variation, structural hashing and duplicate rejection. It
/// deliberately refuses configurations whose semantics it cannot reproduce
/// exactly, rather than silently changing them:
///
/// * constant fitting (`--fit-iters > 0`) stays on the owned-tree path (#116);
/// * aggregates, custom blocks and dimension constraints need the full `Node`
///   machinery and fall back to it as a whole;
/// * a custom complexity model or a population below the 4 096-genome gate
///   keeps the historical loop, where it is already cheap.
#[cfg(feature = "cuda")]
fn postfix_variation_for(cfg: &Config, train: &Sample) -> Result<Variation, String> {
    if cfg.fit_iters != 0 {
        return Err("constant fitting is enabled".to_string());
    }
    if cfg.islands.max(1) * cfg.pop < 4096 {
        return Err(format!(
            "population {} is below the 4096-genome gate",
            cfg.islands.max(1) * cfg.pop
        ));
    }
    if train.is_multi() {
        return Err("multi-output objective".to_string());
    }
    if cfg.complexity_model != crate::complexity::ComplexityModel::default() {
        return Err("custom complexity model".to_string());
    }
    if cfg.units.is_some() || cfg.require_dimensions {
        return Err("dimension constraints".to_string());
    }
    if cfg.fit_samples > 0 || cfg.predictor_samples.is_some() {
        return Err("screening or predictor subsampling".to_string());
    }
    if cfg.qd_bins > 0 {
        return Err("quality-diversity archive".to_string());
    }
    if !expr::current_blocks().is_empty() {
        return Err("custom building blocks".to_string());
    }
    let width = crate::genome::width_for(cfg.max_complexity);
    Variation::new(
        width,
        train.x.first().map(|r| r.len()).unwrap_or(0),
        cfg.max_complexity,
        cfg.seed,
        &cfg.ops,
        cfg.init_depth,
        cfg.dedup,
    )
}

/// #122: the generation loop over the flat postfix arena.
///
/// The device evaluates the whole arena in one upload and one launch; the host
/// only varies compact arrays, tests the archive with `(error, cost)` and
/// rebuilds an owned tree for a candidate that actually enters the front.
/// Fitting, simplification of reported expressions, checkpoints and every
/// reported number remain on the CPU in `f64`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_postfix_search(
    cfg: &Config,
    train: &Sample,
    test: &Sample,
    train_data: &EvalSample,
    variation: Variation,
    mut gpu: GpuSearch,
    mut archive: Archive,
    islands: Vec<Vec<Ind>>,
    rngs: Vec<Mt>,
    main_rng: Mt,
    mut stagnant: Vec<usize>,
    mut gens_done: usize,
    mut converged_early: bool,
    mut best_prev: f64,
    mut no_improve: usize,
    save_every: usize,
    mut on_checkpoint: Option<CheckpointFn<'_>>,
    backend_reason: Option<String>,
    device: Option<String>,
) -> Result<Outcome, String> {
    let pop = cfg.pop;
    let n_islands = cfg.islands.max(1);
    let trees: Vec<Vec<crate::expr::Node>> = islands
        .iter()
        .map(|p| p.iter().map(|ind| ind.tree.clone()).collect())
        .collect();
    let errors: Vec<Vec<f64>> = islands
        .iter()
        .map(|p| p.iter().map(|ind| ind.error).collect())
        .collect();
    let mut pool = Arena::from_trees(&trees, &errors, variation.width)?;
    let mut progress_stats = ProgressStats::new(crate::progress_stats::DEFAULT_WINDOW);
    let mut gpu_stats: Vec<GpuGenStats> = Vec::new();
    let mut restart_flags = vec![false; n_islands];
    let t0 = std::time::Instant::now();

    for gen in gens_done..cfg.generations {
        // A checkpoint saved after an early stop stays stopped even when the
        // generation budget is raised, exactly like the owned-tree loop.
        if converged_early {
            break;
        }
        gens_done = gen + 1;
        // #76/#122: compile the previous generation's front once, then let the
        // flat engine draw 15 % of parents from it. With the feature off the
        // pool is empty and `front_p` is 0.0, so the draw consumes no RNG and
        // the trajectory is exactly the pre-#76 one.
        let front = if cfg.front_parents {
            FrontParents::from_front(&archive.items, variation.width, cfg.max_complexity)
        } else {
            FrontParents::empty()
        };
        let front_p = if cfg.front_parents {
            FRONT_PARENT_P
        } else {
            0.0
        };
        let mut next = variation.next_generation_with_front(
            &pool,
            gen as u32,
            &restart_flags,
            &front,
            front_p,
        );
        next.evaluate(&mut gpu.ev, train.len())?;
        restart_flags = vec![false; n_islands];

        // Archive: test `(error, cost)` first and rebuild a tree only for a
        // candidate that survives. This is the survivor-only hand-off.
        let mut gen_best = f64::INFINITY;
        for i in 0..n_islands {
            for j in 0..pop {
                let idx = next.slot_index(i, j);
                let error = next.err[idx];
                if error < gen_best {
                    gen_best = error;
                }
                let cost = next.cost[idx] as f64;
                if !archive.admits(error, cost) {
                    continue;
                }
                let Some(tree) = next.decode(idx) else { continue };
                archive.try_insert_view(&tree, error, expr::complexity(&tree), cost);
            }
        }
        gpu_stats.push(GpuGenStats {
            generation: gen,
            opcodes: Default::default(),
            programs: n_islands * pop,
            native: n_islands * pop,
            lowered: 0,
            fallbacks: 0,
            unsupported_programs: 0,
            fallback_seconds: 0.0,
            fallback_workers: 0,
            fit_attempts: 0,
            fit_gpu: 0,
            fit_fallback: 0,
            fit_screened: 0,
            fit_fallback_by_cause: Default::default(),
        });

        if let Some(budget) = cfg.time_budget {
            if t0.elapsed().as_secs_f64() >= budget {
                converged_early = true;
            }
        }

        let convergence = progress_stats.observe(gens_done, gen_best, &archive);
        if let Some(report) = &cfg.progress {
            report(Progress {
                generation: gens_done,
                total_generations: cfg.generations,
                best_error: gen_best,
                pareto_members: archive.items.len(),
                convergence,
            });
        }

        if gen_best.is_finite() {
            if gen_best <= 1e-24 {
                converged_early = true;
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
            }
        }

        if n_islands > 1 && cfg.migration_interval > 0 && (gen + 1) % cfg.migration_interval == 0
        {
            next.migrate();
        }

        // Stagnation and restart, exactly as the owned-tree loop: measured at
        // the generation boundary, applied before the checkpoint so a saved
        // state never carries the pre-restart population.
        if cfg.restart_interval > 0 && (gen + 1) % cfg.restart_interval == 0 {
            let arch_best = archive
                .items
                .iter()
                .map(|s| s.error)
                .fold(f64::INFINITY, f64::min);
            let mut restarted = false;
            for i in 0..n_islands {
                let cur_best = (0..pop)
                    .map(|j| next.err[next.slot_index(i, j)])
                    .fold(f64::INFINITY, f64::min);
                if cur_best > arch_best + 1e-9 {
                    stagnant[i] += 1;
                } else {
                    stagnant[i] = 0;
                }
                if stagnant[i] >= 2 {
                    variation.restart_island(&mut next, i, gen as u32);
                    stagnant[i] = 0;
                    restarted = true;
                }
            }
            if restarted {
                next.evaluate(&mut gpu.ev, train.len())?;
            }
        }

        if save_every > 0 && gens_done % save_every == 0 {
            let trees = next
                .to_trees()
                .ok_or_else(|| "postfix pool could not be decoded for a checkpoint".to_string())?;
            let islands_now: Vec<Vec<Ind>> = trees
                .into_iter()
                .enumerate()
                .map(|(i, pop_trees)| {
                    pop_trees
                        .into_iter()
                        .enumerate()
                        .map(|(j, tree)| Ind {
                            error: next.err[next.slot_index(i, j)],
                            complexity: next.cost[next.slot_index(i, j)] as f64,
                            tree,
                            descriptor: None,
                        })
                        .collect()
                })
                .collect();
            let cp = snapshot(
                cfg,
                Backend::Gpu,
                gens_done,
                converged_early,
                best_prev,
                no_improve,
                &archive,
                &islands_now,
                &rngs,
                &stagnant,
                &main_rng,
                train,
                test,
            );
            emit(&mut on_checkpoint, &cp);
        }

        pool = next;
        if converged_early {
            break;
        }
    }

    if on_checkpoint.is_some() {
        let trees = pool
            .to_trees()
            .ok_or_else(|| "postfix pool could not be decoded for the final checkpoint".to_string())?;
        let islands_now: Vec<Vec<Ind>> = trees
            .into_iter()
            .enumerate()
            .map(|(i, pop_trees)| {
                pop_trees
                    .into_iter()
                    .enumerate()
                    .map(|(j, tree)| Ind {
                        error: pool.err[pool.slot_index(i, j)],
                        complexity: pool.cost[pool.slot_index(i, j)] as f64,
                        tree,
                        descriptor: None,
                    })
                    .collect()
            })
            .collect();
        let cp = snapshot(
            cfg,
            Backend::Gpu,
            gens_done,
            converged_early,
            best_prev,
            no_improve,
            &archive,
            &islands_now,
            &rngs,
            &stagnant,
            &main_rng,
            train,
            test,
        );
        emit(&mut on_checkpoint, &cp);
    }

    // Exact f64 rebuild over the surviving front only. The generation errors
    // were the f32 kernel proxy; every number the user reads comes from here.
    let mut final_archive = Archive::new(archive.cap);
    for sol in std::mem::take(&mut archive.items) {
        let tree = expr::simplify(&sol.tree);
        let error = if cfg.accepts_dimensions(&tree) {
            score_tree(&tree, train_data, cfg)
        } else {
            f64::INFINITY
        };
        final_archive.try_insert(Sol {
            complexity: expr::complexity(&tree),
            cost: cfg.complexity_model.complexity(&tree),
            tree,
            error,
        });
    }
    final_archive.canonicalize();
    let sorted = final_archive.sorted();
    let test_data = EvalSample::new(test, cfg);
    let test_errors: Vec<f64> = sorted
        .iter()
        .map(|s| eval_error_on(&s.tree, &test_data, cfg.metric))
        .collect();
    let mut final_archive = Archive::new(final_archive.cap);
    final_archive.items = sorted;

    let best = final_archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min);
    let convergence = progress_stats.observe(gens_done, best, &final_archive);

    Ok(Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early,
        seeds: Vec::new(),
        convergence,
        fit_samples: 0,
        fit_rows: 0,
        archive_refit: false,
        predictor_samples: 0,
        backend: crate::backend::Kind::Gpu,
        backend_reason,
        device,
        gpu: Some(GpuRunStats {
            generations: gpu_stats,
            // #118: the flat path goes through `GpuEvaluator::eval_arena`,
            // which records the same full-path ledger as the packed batch.
            accounting: gpu.ev.stats(),
            fit_attempts: gpu.fit_attempts,
            fit_gpu: gpu.fit_gpu,
            fit_fallback: gpu.fit_fallback,
            fit_screened: gpu.fit_screened,
            fit_fallback_by_cause: gpu.fit_fallback_by_cause.clone(),
            fit_refits: 0,
            fit_ms: gpu.fit_ms,
            fit_host_ms: gpu.fit_host_ms,
        }),
        qd: None,
    })
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
    // whole generations. With the default `exact` arm constants are fitted on
    // the CPU before the batch; `--fit-strategy gpu` adds a fit batch (#169). The backend report below records
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
            None => match GpuSearch::new(train, cfg.fit_strategy.defers_to_gpu()) {
                Ok(g) => {
                    let device = crate::gpu::GpuEvaluator::device_name();
                    if cfg.gpu_notes {
                        // #169: the GPU-fit arm runs at the generation
                        // barrier, so the overlap pipeline is off for it.
                        let scheduling = if cfg.fit_strategy.defers_to_gpu()
                            && defers_fit_to_gpu(cfg, true)
                        {
                            "GPU-fit batches (synchronous)".to_string()
                        } else if cfg.variation != VariationMode::Node
                            && postfix_variation_for(cfg, train).is_ok()
                        {
                            "flat postfix variation".to_string()
                        } else if cfg.pipeline_depth == 0 {
                            "synchronous batches".to_string()
                        } else {
                            format!(
                                "overlap depth {} (batches of {})",
                                cfg.pipeline_depth,
                                (cfg.pop / cfg.pipeline_depth.max(1)).max(1)
                            )
                        };
                        eprintln!(
                            "# gpu: {} ({}; f32 fitness, reported numbers stay f64)",
                            crate::gpu::GpuEvaluator::device_name(),
                            scheduling
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
        if cfg.variation == VariationMode::Postfix {
            return Err("--variation postfix needs a build with --features cuda".to_string());
        }
        if cfg.gpu {
            backend_reason = Some("built without --features cuda".to_string());
            if cfg.gpu_notes {
                eprintln!("# gpu: unavailable (built without --features cuda)");
            }
        }
    }
    #[cfg(feature = "cuda")]
    if cfg.gpu_notes && cfg.fit_strategy.defers_to_gpu() && !defers_fit_to_gpu(cfg, true) {
        eprintln!(
            "# gpu-fit: off (the batched kernel minimises squared error; \
             --metric must be squared, aic, aicc or bic)"
        );
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
                if cfg.qd_policy {
                    // #120: a seed occupies one niche rather than being cloned
                    // into every island. The QD layer later records it at its
                    // own descriptor.
                    let ii = k % islands.len();
                    if k < islands[ii].len() {
                        islands[ii][k] = in_pop;
                    }
                } else {
                    for pop in islands.iter_mut() {
                        if k < pop.len() {
                            pop[k] = in_pop.clone();
                        }
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

    // #122: the flat postfix generation loop. It is selected before the first
    // generation and never changes mid-run; everything it cannot represent
    // stays on the historical owned-tree path.
    #[cfg(feature = "cuda")]
    if cfg.variation != VariationMode::Node {
        if let Some(g) = gpu.take() {
            match postfix_variation_for(cfg, train) {
                Ok(variation) if seeds.is_empty() => {
                    return run_postfix_search(
                        cfg,
                        train,
                        test,
                        &train_data,
                        variation,
                        g,
                        archive,
                        islands,
                        rngs,
                        rng,
                        stagnant,
                        gens_done,
                        converged_early,
                        best_prev,
                        no_improve,
                        save_every,
                        on_checkpoint,
                        backend_reason,
                        backend_device.clone(),
                    );
                }
                Ok(_) => {
                    gpu = Some(g);
                    if cfg.variation == VariationMode::Postfix {
                        return Err(
                            "--variation postfix cannot carry --seed-expr/--seed-file; \
                             use --variation auto"
                                .to_string(),
                        );
                    }
                    if cfg.gpu_notes {
                        eprintln!("# gpu: node variation (seeded start)");
                    }
                }
                Err(reason) => {
                    if cfg.variation == VariationMode::Postfix {
                        return Err(format!("--variation postfix: {}", reason));
                    }
                    gpu = Some(g);
                    if cfg.gpu_notes {
                        eprintln!("# gpu: node variation ({})", reason);
                    }
                }
            }
        } else if cfg.variation == VariationMode::Postfix {
            return Err("--variation postfix needs a usable CUDA device".to_string());
        }
    }

    // #18: one tracker, updated once per generation from the archive. It holds
    // no population state, so it cannot affect the search trajectory.
    let mut progress_stats = ProgressStats::new(crate::progress_stats::DEFAULT_WINDOW);
    let mut stats_observed = false;
    // #120: passive descriptor archive. It observes evaluated individuals and
    // never feeds back into selection, so turning it on cannot change the
    // trajectory or the Pareto front.
    let mut qd_archive = if cfg.qd_bins > 0 {
        let archive = if cfg.qd_control == 2 {
            QdArchive::new_random(cfg.qd_bins, cfg.qd_cell_cap)
        } else if cfg.qd_cvt {
            QdArchive::new_cvt(cfg.qd_bins, cfg.qd_cell_cap)
        } else {
            QdArchive::new(cfg.qd_bins, cfg.qd_cell_cap)
        };
        Some(archive.with_nvars(nvars))
    } else {
        None
    };

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
            // #120: an `Ind` descriptor is accumulated from the prediction that
            // scored `eval_data`. It describes `gen_train` only when those are
            // the same rows: no per-generation subset and no screening subset.
            let cached_descriptors_valid = gen_sample.is_none() && !screened;

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
            // #120: the archive as of the previous generation, borrowed exactly
            // like the Pareto front. It is updated only after the barrier.
            let qd_snapshot = qd_archive.as_ref();
            // #119: overlap the GPU with the rest of this generation. The CPU
            // threads keep generating and fitting; the main thread drains a
            // bounded queue and scores batches. Selection still waits for every
            // result at the barrier, so the trajectory is unchanged.
            #[cfg(feature = "cuda")]
            // #169: the overlap pipeline scores a tree as soon as it is offered,
            // before the barrier fit batch runs, so it cannot carry GPU-fitted
            // constants. The GPU-fit arm uses the synchronous batch instead.
            let pipeline_active = !screened
                && defer_gpu
                && cfg.pipeline_depth > 0
                && !defers_fit_to_gpu(cfg, defer_gpu);
            #[cfg(not(feature = "cuda"))]
            let pipeline_active = false;
            // One batch is `pop / depth` candidates, so `depth` batches queue
            // ahead of the scorer; live memory is at most `(depth + 1) * batch`
            // candidate trees plus their compiled programs.
            let batch_size = (cfg.pop / cfg.pipeline_depth.max(1)).max(1);
            let queue_capacity = cfg.pipeline_depth.max(1) * batch_size;
            #[cfg(not(feature = "cuda"))]
            let _ = (pipeline_active, batch_size, queue_capacity);
            #[cfg(feature = "cuda")]
            let mut pipeline_results: Vec<Vec<Option<f64>>> =
                vec![vec![None; cfg.pop.max(1)]; islands.len()];
            #[cfg(feature = "cuda")]
            let mut pipeline_error: Option<String> = None;
            #[cfg(feature = "cuda")]
            let mut pipeline_scored = false;
            std::thread::scope(|s| {
                // The queue lives in the scope: producers take a clone of the
                // sender, and `drain` consumes the original so it can end.
                #[cfg(feature = "cuda")]
                let queue = if pipeline_active {
                    Some(crate::pipeline::JobQueue::new(queue_capacity))
                } else {
                    None
                };
                #[cfg(feature = "cuda")]
                let senders: Vec<Option<std::sync::mpsc::SyncSender<PipelineJob>>> =
                    (0..islands.len())
                        .map(|_| queue.as_ref().map(|q| q.sender()))
                        .collect();
                #[cfg(not(feature = "cuda"))]
                let senders: Vec<Option<std::sync::mpsc::SyncSender<PipelineJob>>> =
                    vec![None; islands.len()];

                let handles: Vec<_> = islands
                    .iter_mut()
                    .zip(rngs.iter_mut())
                    .zip(senders.into_iter())
                    .enumerate()
                    .map(|(i, ((isl, r), sender))| {
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
                                qd_snapshot, // #120
                                cfg.qd_policy, // #120
                                cfg.qd_control, // #120 control
                                sender.as_ref().map(|t| (t, i)), // #119
                            ) // #85
                        })
                    })
                    .collect();

                #[cfg(feature = "cuda")]
                if let Some(queue) = queue {
                    pipeline_error = queue.drain(batch_size, |jobs| {
                        pipeline_scored = true;
                        score_pipeline_batch(
                            jobs,
                            &mut gpu,
                            &mut pipeline_results,
                            gen_train.len(),
                            gen,
                        )
                    });
                }

                for h in handles {
                    let _ = h.join();
                }
            });

            #[cfg(feature = "cuda")]
            {
                if pipeline_active {
                    // Results are keyed by (island, slot), so the arrival order
                    // across concurrent island threads cannot change them.
                    for (i, slots) in pipeline_results.iter().enumerate() {
                        for (j, value) in slots.iter().enumerate() {
                            if let Some(e) = value {
                                islands[i][j].error = *e;
                            }
                        }
                    }
                    if pipeline_error.is_none() {
                        if pipeline_scored {
                            gpu_used = true;
                        }
                        // Score anything the kernel could not compile (left
                        // pending) through the normal CPU fallback.
                        defer_gpu =
                            score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen);
                    } else {
                        // Stop the GPU and re-score every still-pending
                        // candidate in f64 on the CPU; a failed batch must
                        // never leave a wrong (or missing) number behind.
                        if cfg.gpu_notes {
                            if let Some(e) = &pipeline_error {
                                eprintln!("# gpu: unavailable ({})", e);
                            }
                        }
                        // #121: keep the pool for the rescore, then drop the
                        // evaluator; a failed batch also scores via the pool.
                        let fallback = gpu
                            .as_mut()
                            .map(|g| g.take_fallback())
                            .unwrap_or_else(|| {
                                FallbackPool::new(fallback_workers(), fallback_workers())
                            });
                        gpu = None;
                        let train_view = EvalSample::new(train, cfg);
                        rescore_pending_cpu(&mut islands, &train_view, cfg, &fallback);
                        defer_gpu = false;
                    }
                } else if !screened {
                    // #26: one batched call covers every candidate of the
                    // generation (the pre-#119 synchronous path).
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
                        // #120: stratified migration replaces the individual in
                        // the least-visited descriptor cell, not the globally
                        // worst error. With QD off this is the old best-to-worst
                        // ring migration bit for bit.
                        let victim = if cfg.qd_policy {
                            match qd_archive
                                .as_ref()
                                .and_then(|q| q.least_occupied_descriptor())
                            {
                                Some(target) => {
                                    // #120: reuse the descriptors the scoring
                                    // pass already accumulated when they
                                    // describe `gen_train`; only the GPU or
                                    // subset paths pay for one here.
                                    let descriptors: Vec<qd::Descriptor> = islands[ii]
                                        .iter()
                                        .map(|ind| {
                                            if cached_descriptors_valid {
                                                if let Some(d) = ind.descriptor {
                                                    return d;
                                                }
                                            }
                                            qd::descriptor(&ind.tree, gen_train)
                                        })
                                        .collect();
                                    descriptors
                                        .iter()
                                        .enumerate()
                                        .min_by(|(_, a), (_, b)| {
                                            let score = |d: &qd::Descriptor| {
                                                (d.usage - target.usage).powi(2)
                                                    + (d.skew - target.skew).powi(2)
                                            };
                                            score(a)
                                                .partial_cmp(&score(b))
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                        .map(|(i, _)| i)
                                }
                                None => None,
                            }
                        } else {
                            None
                        };
                        let slot = victim.or_else(|| {
                            islands[ii]
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| {
                                    a.error
                                        .partial_cmp(&b.error)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i)
                        });
                        if let Some(worst) = slot.and_then(|i| islands[ii].get_mut(i)) {
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
                        // #120: aim the restart at the least-visited region by
                        // rejection sampling a few cheap descriptor probes.
                        let qd_target = if cfg.qd_policy {
                            qd_archive
                                .as_ref()
                                .and_then(|q| q.least_occupied_descriptor())
                        } else {
                            None
                        };
                        for ind in islands[ii].iter_mut() {
                            let mut t = random_tree_with(
                                &mut r,
                                nvars,
                                &cfg.ops,
                                cfg.init_depth,
                                Some(&aggs),
                            );
                            if let Some(target) = qd_target {
                                let mut best = t.clone();
                                let mut best_d = qd::descriptor(&t, gen_train);
                                let mut best_score = (best_d.usage - target.usage).powi(2)
                                    + (best_d.skew - target.skew).powi(2);
                                for _ in 0..3 {
                                    let cand = random_tree_with(
                                        &mut r,
                                        nvars,
                                        &cfg.ops,
                                        cfg.init_depth,
                                        Some(&aggs),
                                    );
                                    let d = qd::descriptor(&cand, gen_train);
                                    let score = (d.usage - target.usage).powi(2)
                                        + (d.skew - target.skew).powi(2);
                                    if score < best_score {
                                        best = cand;
                                        best_d = d;
                                        best_score = score;
                                    }
                                }
                                let _ = best_d;
                                t = best;
                            }
                            // #15: multi-output fits per output inside the evaluator.
                            let mut fitted = None;
                            if cfg.fit_iters > 0
                                && !gen_train.is_multi()
                                && !defers_fit_to_gpu(cfg, defer_gpu)
                            {
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

            // #120: record this generation's evaluated behaviours. The order
            // is fixed (island 0..I, population order, then the archive), so
            // the archive is a deterministic function of the trajectory.
            if let Some(qd) = qd_archive.as_mut() {
                let mut items = Vec::new();
                for pop in &islands {
                    for ind in pop {
                        if ind.error.is_finite() {
                            // #120: the descriptor the scoring pass already
                            // accumulated. `None` means the archive falls back
                            // to its known-descriptor cache or one evaluation
                            // (GPU-scored, subsampled or checkpointed trees).
                            let d = if cached_descriptors_valid {
                                ind.descriptor
                            } else {
                                None
                            };
                            items.push((ind.tree.clone(), ind.error, ind.complexity, d));
                        }
                    }
                }
                for s in &archive.items {
                    items.push((s.tree.clone(), s.error, s.cost, None));
                }
                qd.record_with_descriptors(items, gen_train);
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

    // #120: promoted cell elites are rebuilt exactly on every training row in
    // f64 before they are reported. Screening and row subsets may guide the
    // search, but the QD report is never a subset estimate.
    if cfg.qd_policy {
        if let Some(qd) = qd_archive.as_mut() {
            qd.refit_all(|tree| {
                let sol = refit_survivor(tree, &train_data, cfg);
                (sol.error, sol.cost)
            });
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

    // #169: with the GPU-fit arm every finalist is also refit by the exact CPU
    // f64 LM before it is scored on all training rows.
    let gpu_fit_used = cfg.fit_strategy.defers_to_gpu() && gpu_used;
    let mut gpu_fit_refits = 0usize;
    if subsampling_active(cfg, train) || gpu_used || resumed || fit_screened {
        // Generation fitness was estimated on subsets, or on the GPU in f32, so
        // the front is rebuilt from the surviving trees and scored on every
        // training row with the CPU in f64. From here on the archive and every
        // number derived from it are full-data and match the CPU backend.
        let rebuilt = full_data_archive(&archive, &islands, &train_data, cfg, gpu_fit_used);
        archive = rebuilt.0;
        gpu_fit_refits = rebuilt.1;
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
    let gpu_stats = if gpu_used {
        gpu.as_ref().map(|g| {
            let mut s = g.run_stats();
            s.fit_refits = gpu_fit_refits;
            s
        })
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_stats: Option<GpuRunStats> = {
        let _ = gpu_fit_refits;
        None
    };

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
            let rows = effective_fit_rows(cfg, train);
            if rows > 0 { rows.min(train.len()) } else { cfg.fit_samples.min(train.len()) }
        } else {
            0
        },
        fit_rows: if fit_screened && cfg.fit_rows >= 2 {
            cfg.fit_rows.min(train.len())
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
        // #120: coverage of the descriptor grid for this run.
        qd: qd_archive.as_ref().map(|q| q.report()),
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

    fn pending_ind(error: f64, complexity: f64) -> Ind {
        Ind {
            // One fittable constant, so admission does not filter the fixture
            // as a no-constant tree.
            tree: node(Op::Add, vec![input(0), constant(1.0)]),
            error,
            complexity,
            descriptor: None,
        }
    }

    /// #169: the admission fraction is a pure function of the population size
    /// and the training row count, and widens to every candidate at large `N`.
    #[test]
    fn admission_count_is_the_documented_fraction() {
        let small = WIDE_ADMISSION_ROWS - 1;
        assert_eq!(admission_count(1, small), 1);
        assert_eq!(admission_count(24, small), 24, "small islands admit everyone");
        assert_eq!(admission_count(64, small), 32, "max(32, pop/4)");
        assert_eq!(admission_count(100, small), 32);
        assert_eq!(admission_count(128, small), 32);
        assert_eq!(admission_count(256, small), 64);
        // 154k, the pinned fit-heavy cell: the documented fraction is unchanged.
        assert_eq!(admission_count(256, 154_000), 64);
        // The 3.9M hourly cell: every candidate with constants is admitted.
        assert_eq!(admission_count(128, 3_900_000), 128);
        assert_eq!(admission_count(256, WIDE_ADMISSION_ROWS), 256);
        assert_eq!(admission_count(1, WIDE_ADMISSION_ROWS), 1);
    }

    /// #169: promotion is deterministic, per island, and ranks on the same
    /// (error, complexity, index) key the selector uses. Non-promoted
    /// offspring keep their screening score because they are not returned.
    #[test]
    fn promotion_is_deterministic_and_selects_the_screened_best() {
        let n = 128;
        let mut islands = vec![Vec::new(), Vec::new()];
        for (i, pop) in islands.iter_mut().enumerate() {
            for j in 0..n {
                // A mix of pending (one of every 17) and already-scored
                // individuals. The finite, non-pending ones must never enter
                // the promotion set.
                let error = if j % 17 == 0 {
                    1e9 + j as f64
                } else {
                    (((j * 37 + i * 11) % n) as f64) + 0.5
                };
                pop.push(pending_ind(error, (j % 5) as f64));
            }
        }
        let pending: Vec<(usize, usize)> = (0..2)
            .flat_map(|i| (0..n).filter(|j| j % 17 != 0).map(move |j| (i, j)))
            .collect();
        let a = select_promoted(&islands, &pending, 240);
        let b = select_promoted(&islands, &pending, 240);
        assert_eq!(a, b, "the same screen must yield the same admission set");
        assert_eq!(a.len(), 64, "32 per island");

        for i in 0..2 {
            let mut expected: Vec<usize> = pending
                .iter()
                .filter(|(island, _)| *island == i)
                .map(|&(_, j)| j)
                .collect();
            expected.sort_by(|&x, &y| {
                islands[i][x]
                    .error
                    .partial_cmp(&islands[i][y].error)
                    .unwrap()
                    .then(
                        islands[i][x]
                            .complexity
                            .partial_cmp(&islands[i][y].complexity)
                            .unwrap(),
                    )
                    .then(x.cmp(&y))
            });
            let got: Vec<usize> = a
                .iter()
                .filter(|(island, _)| *island == i)
                .map(|&(_, j)| j)
                .collect();
            let mut expected_top = expected[..32].to_vec();
            expected_top.sort_unstable();
            assert_eq!(got, expected_top, "island {}", i);
        }
    }

    /// #169: equal screening errors break on complexity and then on the fixed
    /// population index, so a shuffled completion order cannot change admission.
    #[test]
    fn promotion_breaks_ties_by_complexity_then_index() {
        let mut islands = vec![Vec::new()];
        for j in 0..64 {
            islands[0].push(pending_ind(f64::NAN, if j < 40 { 2.0 } else { 1.0 }));
        }
        // Screen every candidate to the same error.
        for ind in islands[0].iter_mut() {
            ind.error = 5.0;
        }
        let pending: Vec<(usize, usize)> = (0..64).map(|j| (0, j)).collect();
        let got = select_promoted(&islands, &pending, 240);
        let mut wanted: Vec<(usize, usize)> = (0..8).map(|j| (0, j)).collect();
        wanted.extend((40..64).map(|j| (0, j)));
        assert_eq!(got, wanted, "lower complexity wins an error tie");
    }

    /// #169: a tree with no constants is never admitted: there is nothing to
    /// fit, so it keeps its screening score and does not consume a slot.
    #[test]
    fn promotion_skips_trees_with_no_constants() {
        let mut islands = vec![Vec::new()];
        for j in 0..64 {
            let mut ind = pending_ind(1.0 + j as f64, 1.0);
            if j % 2 == 0 {
                ind.tree = input(0); // no constants
            }
            islands[0].push(ind);
        }
        let pending: Vec<(usize, usize)> = (0..64).map(|j| (0, j)).collect();
        let got = select_promoted(&islands, &pending, 240);
        assert_eq!(got.len(), 32);
        assert!(
            got.iter().all(|&(_, j)| j % 2 == 1),
            "only trees with constants may be admitted: {:?}",
            got
        );
    }

    /// #169: at and above `WIDE_ADMISSION_ROWS` the seed-constant ranking must
    /// not leave a fittable offspring unfitted. Every candidate with constants
    /// is promoted, regardless of its screened error, while a tree with no
    /// constants is still skipped (there is nothing for the LM to fit).
    #[test]
    fn large_cells_admit_every_candidate_with_constants() {
        let mut islands = vec![Vec::new()];
        for j in 0..128 {
            let mut ind = pending_ind(1.0 + j as f64, 1.0);
            if j % 16 == 0 {
                ind.tree = input(0); // no constants
            }
            islands[0].push(ind);
        }
        let pending: Vec<(usize, usize)> = (0..128).map(|j| (0, j)).collect();
        let got = select_promoted(&islands, &pending, WIDE_ADMISSION_ROWS);
        let wanted: Vec<(usize, usize)> = (0..128).filter(|j| j % 16 != 0).map(|j| (0, j)).collect();
        assert_eq!(
            got, wanted,
            "every fittable candidate must be admitted at large N"
        );
        // Just below the threshold the old fraction still applies.
        let narrow = select_promoted(&islands, &pending, WIDE_ADMISSION_ROWS - 1);
        assert_eq!(
            narrow.len(),
            32,
            "below the threshold the pop/4 floor is unchanged"
        );
    }

    /// #169: every CPU fallback cause has a stable label, and both the plan
    /// rejection and the kernel status map into the documented set.
    #[test]
    fn fallback_causes_are_labelled_and_classified() {
        use crate::fit::GpuFitReject;
        let labels: Vec<&str> = FitFallbackCause::ALL.iter().map(|c| c.label()).collect();
        assert_eq!(
            labels,
            vec![
                "unsupported_plan",
                "too_few_rows",
                "too_many_constants",
                "ill_conditioned",
                "non_finite",
                "launch",
            ]
        );
        assert_eq!(
            plan_fallback_cause(GpuFitReject::TooManyConstants),
            FitFallbackCause::TooManyConstants
        );
        for reason in [
            GpuFitReject::TooManyNodes,
            GpuFitReject::NoConstants,
            GpuFitReject::SharedConstants,
            GpuFitReject::UnsupportedOp,
        ] {
            assert_eq!(
                plan_fallback_cause(reason),
                FitFallbackCause::UnsupportedPlan,
                "{}",
                reason.label()
            );
        }
        assert_eq!(
            status_fallback_cause(1),
            FitFallbackCause::IllConditioned
        );
        assert_eq!(status_fallback_cause(2), FitFallbackCause::NonFinite);
        assert_eq!(
            status_fallback_cause(0),
            FitFallbackCause::UnsupportedPlan
        );
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
            descriptor: None,
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
                None,  // #120: no QD snapshot in this test
                false, // #120
                0,     // #120 control
                None,  // #119: synchronous test path
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
            descriptor: None,
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

    /// #116: `--fit-rows N` screens the fit only; `N >= train rows` is an
    /// exact no-op and `--fit-rows` wins over `--fit-samples`.
    #[test]
    fn fit_rows_is_a_full_fit_at_or_above_the_sample_size() {
        let s = sample(); // 21 rows
        let mut cfg = Config {
            fit_rows: s.len(),
            ..Default::default()
        };
        assert!(fit_for_generation(&cfg, &s, 0).is_none(), "N == rows is a full fit");
        cfg.fit_rows = s.len() + 25;
        assert!(fit_for_generation(&cfg, &s, 0).is_none(), "N > rows is a full fit");
        cfg.fit_rows = 8;
        let sub = fit_for_generation(&cfg, &s, 0).expect("8 rows screens");
        assert_eq!(sub.len(), 8);
        cfg.fit_samples = 5;
        let sub = fit_for_generation(&cfg, &s, 0).expect("fit_rows wins");
        assert_eq!(sub.len(), 8);
        assert_eq!(effective_fit_rows(&cfg, &s), 8);
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

    /// #85: the promoted tree is fitted on exactly the rows that score it, so
    /// the squared path reuses the fitter's final SSE instead of walking every
    /// row a second time. A metric with a different loss still scores afresh.
    #[test]
    fn refit_reuses_the_fit_sse_only_on_the_exact_squared_path() {
        let s = sample(); // y = x0
        let view = EvalSample::new(
            &s,
            &Config {
                metric: Metric::Squared,
                fit_iters: 8,
                ..Default::default()
            },
        );
        let mut squared = Config {
            metric: Metric::Squared,
            fit_iters: 8,
            ..Default::default()
        };
        // A constant under `sin` cannot fold into the expression, so the hash
        // at the fit and after simplification agree.
        let tree = node(
            Op::Sin,
            vec![node(Op::Mul, vec![constant(1.0), input(0)])],
        );
        let before = REFIT_SSE_REUSED.with(|n| n.get());
        refit_survivor(&tree, &view, &squared);
        assert_eq!(
            REFIT_SSE_REUSED.with(|n| n.get()),
            before + 1,
            "the squared refit must reuse the final SSE"
        );

        // The configured loss for `absolute` is not the fitter's squared SSE.
        squared.metric = Metric::Absolute;
        let before = REFIT_SSE_REUSED.with(|n| n.get());
        let view = EvalSample::new(&s, &squared);
        refit_survivor(&tree, &view, &squared);
        assert_eq!(
            REFIT_SSE_REUSED.with(|n| n.get()),
            before,
            "a non-squared metric must never reuse the squared SSE"
        );
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

    /// #122/#121: the bounded CPU fallback must be a pure parallelisation, not
    /// a change of result. One thread and four threads score the same
    /// unsupported (`if`) and supported trees to the same bits.
    #[cfg(feature = "cuda")]
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
        let single = FallbackPool::new(1, 1);
        let quad = FallbackPool::new(4, 4);
        rescore_pending_cpu(&mut one, &view, &cfg, &single);
        rescore_pending_cpu(&mut many, &view, &cfg, &quad);
        for (a, b) in one.iter().flatten().zip(many.iter().flatten()) {
            assert_eq!(a.error.to_bits(), b.error.to_bits());
            assert_eq!(a.complexity.to_bits(), b.complexity.to_bits());
            assert!(expr::same(&a.tree, &b.tree));
        }
        // The serial reference: the same trees scored in the calling thread.
        for (k, tree) in trees.iter().enumerate() {
            let serial = score_tree(tree, &view, &cfg);
            assert_eq!(
                one[k / 4][k % 4].error.to_bits(),
                serial.to_bits(),
                "worker result differs from the serial walk at job {k}"
            );
        }
        assert!(one[0][0].error.is_finite(), "the unsupported tree was scored");
        // The work ran on the pool's threads, not the caller.
        assert!(
            single.busy.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "the fallback pool never ran a job"
        );
        assert!(
            quad.busy.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "the four-worker pool never ran a job"
        );
    }

    /// #121: the fallback queue is bounded by construction and the merge order
    /// does not depend on which worker finished first. Also checks that already
    /// scored individuals are skipped without shifting the pending results.
    #[cfg(feature = "cuda")]
    #[test]
    fn fallback_pool_is_bounded_and_merges_in_candidate_order() {
        let s = sample();
        let cfg = Config::default();
        let view = EvalSample::new(&s, &cfg);
        // 97 distinct trees: `sin(k*x)` scores differently for every k.
        let trees: Vec<Node> = (0..97)
            .map(|k| node(Op::Sin, vec![node(Op::Mul, vec![constant(k as f64), input(0)])]))
            .collect();
        let pool = FallbackPool::new(2, 2);
        assert_eq!(pool.workers(), 2);
        assert_eq!(pool.queue, 2, "the queue holds at most one chunk per worker");

        // Islands with every third individual already scored, so the pending
        // set is non-contiguous and a naive index would drift.
        let mut islands: Vec<Vec<Ind>> = trees
            .chunks(17)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        let mut ind = Ind::pending(t.clone(), &cfg);
                        if i % 3 == 1 {
                            ind.error = -1.0;
                        }
                        ind
                    })
                    .collect()
            })
            .collect();
        rescore_pending_cpu(&mut islands, &view, &cfg, &pool);
        let mut pending = 0usize;
        for pop in &islands {
            for ind in pop {
                if ind.error == -1.0 {
                    continue;
                }
                let want = score_tree(&ind.tree, &view, &cfg);
                assert_eq!(
                    ind.error.to_bits(),
                    want.to_bits(),
                    "the merged result shifted relative to its candidate"
                );
                pending += 1;
            }
        }
        assert!(pending > 60, "the fixture must actually exercise the pool");
        assert!(
            pool.busy.load(std::sync::atomic::Ordering::Relaxed) <= 2,
            "the pool ran more threads than it owns"
        );
    }

    /// #120: scoring a candidate with the QD layer on must not require the
    /// evaluating descriptor fallback, and must not change the error. This is
    /// the one-pass contract for the CPU scoring path.
    #[test]
    fn qd_descriptor_is_folded_into_the_scoring_pass() {
        let data = sample();
        let trees = [
            node(Op::Add, vec![input(0), constant(0.25)]),
            node(Op::Mul, vec![input(0), node(Op::Add, vec![input(1), constant(0.5)])]),
            node(Op::Sin, vec![node(Op::Mul, vec![constant(1.3), input(0)])]),
        ];
        let mut off = Config::default();
        off.fit_iters = 0;
        let mut on = off.clone();
        on.qd_bins = 16;
        let mut view_cfg = Config::default();
        view_cfg.fit_iters = 0;
        let off_data = EvalSample::new(&data, &view_cfg);
        let on_data = EvalSample::new(&data, &on);

        let before = crate::qd::fallback_descriptor_count();
        for t in &trees {
            let plain = Ind::evaluated(t.clone(), &off_data, &off);
            let qd = Ind::evaluated(t.clone(), &on_data, &on);
            assert_eq!(
                plain.error.to_bits(),
                qd.error.to_bits(),
                "the QD fold changed the reported error"
            );
            let want = crate::qd::descriptor_from_prediction(
                t,
                &data.x.iter().map(|row| expr::eval(t, row)).collect::<Vec<_>>(),
                2,
            );
            let got = qd.descriptor.expect("QD scoring must attach a descriptor");
            assert_eq!(got, want, "folded descriptor differs from the evaluating path");
            assert_eq!(plain.descriptor, None, "QD off must not accumulate moments");
        }
        assert_eq!(
            crate::qd::fallback_descriptor_count(),
            before,
            "QD scoring evaluated a tree a second time for its descriptor"
        );
    }
}
