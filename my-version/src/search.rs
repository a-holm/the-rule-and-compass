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

use crate::afp;
use crate::checkpoint::{Backend, Checkpoint, Individual, IslandState};
use crate::mix::{MixCounts, MixWindow};
use crate::complexity::ComplexityModel;
use crate::data::Sample;
use crate::expr::{
    self, crossover_with_opts, eval, mutate_with_opts, random_tree_opts, AggTable, Node, Op,
};
use crate::metrics::{score_weighted, Metric};
// #15: multi-output combined error and per-output constant fitting.
use crate::multi;
use crate::pareto::{Archive, Sol};
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
use crate::postfix::{unique_slots, Arena, FrontParents, Variation};
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

/// #465: how an island chooses its next population.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Selection {
    /// Generational replacement with lexicographic parsimony tournaments.
    #[default]
    Lexicographic,
    /// Age-fitness Pareto survivor selection over parents and offspring.
    Afp,
}

impl Selection {
    pub fn as_str(self) -> &'static str {
        match self {
            Selection::Lexicographic => "lexicographic",
            Selection::Afp => "afp",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "lexicographic" => Ok(Selection::Lexicographic),
            "afp" => Ok(Selection::Afp),
            other => Err(format!(
                "selection must be 'lexicographic' or 'afp', got '{}'",
                other
            )),
        }
    }
}

/// #471: which edit kinds the node-path mutation draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MutationMix {
    /// Point 25 %, terminal 35 %, subtree 40 %.
    #[default]
    Classic,
    /// Adds insert, delete, swap and rotate.
    Extended,
}

impl MutationMix {
    pub fn as_str(self) -> &'static str {
        match self {
            MutationMix::Classic => "classic",
            MutationMix::Extended => "extended",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "classic" => Ok(MutationMix::Classic),
            "extended" => Ok(MutationMix::Extended),
            other => Err(format!(
                "mutation mix must be 'classic' or 'extended', got '{}'",
                other
            )),
        }
    }

    pub fn weights(self) -> expr::MixWeights {
        match self {
            MutationMix::Classic => expr::CLASSIC_MIX,
            MutationMix::Extended => expr::EXTENDED_MIX,
        }
    }
}

/// #471: how random trees draw an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TreeArity {
    /// Uniform over `ops`.
    #[default]
    Uniform,
    /// Arity class first (binary 0.5, unary 0.4, ternary 0.1), then uniform.
    Balanced,
}

impl TreeArity {
    pub fn as_str(self) -> &'static str {
        match self {
            TreeArity::Uniform => "uniform",
            TreeArity::Balanced => "balanced",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "uniform" => Ok(TreeArity::Uniform),
            "balanced" => Ok(TreeArity::Balanced),
            other => Err(format!(
                "tree arity must be 'uniform' or 'balanced', got '{}'",
                other
            )),
        }
    }
}

/// #480: what ranks a candidate. Constants are always fitted with the configured
/// metric; this chooses only the value the archive and selection compare.
#[derive(Clone, Default)]
pub enum Fitness {
    /// The configured `--metric` on the training sample.
    #[default]
    Metric,
    /// The exact long/flat net regret on the canonical training rows.
    LongPnl(std::sync::Arc<crate::trading::FitnessBook>),
    /// Read from a checkpoint: only the book hash is known. The CLI replaces
    /// it with the rebuilt book before the search starts.
    SavedLongPnl(u64),
}

impl Fitness {
    /// The comparison and checkpoint key: kind plus book hash.
    pub fn key(&self) -> String {
        match self {
            Fitness::Metric => "metric".to_string(),
            Fitness::LongPnl(book) => format!("long_pnl:{:016x}", book.hash()),
            Fitness::SavedLongPnl(hash) => format!("long_pnl:{:016x}", hash),
        }
    }

    pub fn is_metric(&self) -> bool {
        matches!(self, Fitness::Metric)
    }

    fn book(&self) -> Option<&crate::trading::FitnessBook> {
        match self {
            Fitness::LongPnl(book) => Some(book),
            _ => None,
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
    /// #480: the value candidates are ranked by. Saved search state.
    pub fitness: Fitness,
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
    /// #496: arithmetic of the device fit kernel's polish. Saved search state.
    pub fit_polish: crate::fit::FitPolish,
    /// #223: which offspring the deferred GPU fit admits. A resolved default
    /// or `auto` GPU arm admits every candidate with constants; an explicit
    /// `gpu` arm keeps the screened fraction. Saved search state.
    pub fit_admission: crate::fit::FitAdmission,
    /// #116: print one `# fit:` line with the arm counts. Reporting only; it is
    /// not saved in a checkpoint and is ignored by the search settings check.
    pub fit_report: bool,
    /// Discard structurally duplicated offspring before evaluating them.
    pub dedup: bool,
    /// #477: reject offspring and random trees with no row-dependent input
    /// before the constant fit, except the first constant of an island.
    /// Saved search state.
    pub reject_constant_offspring: bool,
    /// Wall-clock budget in seconds; `None` means no limit.
    pub time_budget: Option<f64>,
    /// Stop after this many generations with no improvement (0 disables).
    pub patience: usize,
    /// #464: scale-free patience. `Some(r)` counts a generation as "no
    /// improvement" when `(best_prev - best) / |best_prev| <= r`; `None` keeps
    /// the `1 + |best_prev|` denominator and the 1e-9 threshold. Saved search state.
    pub patience_tol: Option<f64>,
    /// #464: `--stop-on front:<W>`. `Some(w)` stops after `w` consecutive
    /// generations with a settled front; `Some(0)` takes the window from
    /// `patience`. Saved search state.
    pub stop_front: Option<usize>,
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
    /// #255: hard bound on how many admitted candidates one generation may
    /// refit with the exact CPU fallback arm. The first `N` fallbacks in the
    /// deterministic admission order are refit; any remainder keeps the finite
    /// constants the device produced and is reported in `fit_fallback_bounded`.
    /// `0` disables the bound. The default is high enough not to fire on the
    /// recorded cell, so normal runs are bit-identical to the serial loop.
    pub fit_fallback_max: usize,
    /// #495: finalist refit band. After a device-fit search only trees whose
    /// full-data error is within this relative distance of the lowest error at
    /// no greater cost, plus every reported front member, get the exact CPU
    /// refit. `inf` refits every unique tree.
    pub refit_band: f64,
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
    /// #211: optional development-log callback, invoked once per generation
    /// with the measured record. Like [`progress`](Self::progress) it is a
    /// reporting hook: it is not search state, is ignored by
    /// [`search_setting_differences`](Self::search_setting_differences), and
    /// `None` (the default) skips every measurement the log would need.
    pub dev_log: Option<DevFn>,
    /// #462: optional front observer, invoked once per generation after the
    /// archive update with the 1-based generation, the elapsed search seconds
    /// and the archive. Reporting only: not search state, not saved in a
    /// checkpoint, ignored by [`search_setting_differences`](Self::search_setting_differences).
    pub front_observer: Option<FrontFn>,
    /// #466: prune neutral code from the front members after the generation
    /// loop. It runs after the loop, so it is not search state: it is not
    /// saved in a checkpoint and is ignored by
    /// [`search_setting_differences`](Self::search_setting_differences).
    pub prune: bool,
    /// #466: training rows the prune fits on; a larger sample is subsampled.
    pub prune_rows: usize,
    /// #466: cap on candidate fits per run (0 disables the prune).
    pub prune_budget: usize,
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
    /// #293: `--gpu-memory`, an upper bound on the device memory the run may
    /// plan on. Not a search setting: it is ignored by the checkpoint check.
    pub gpu_memory: Option<u64>,
    /// #438: `--dispatch`. `Device` skips the per-stage timing comparison only;
    /// refusals and CPU-contract stages still route to the CPU. Not a search
    /// setting.
    pub dispatch: crate::dispatch::DispatchMode,
    /// #119: bounded producer/scorer overlap depth in batches (`0` disables).
    /// While the GPU scores a batch, the CPU keeps generating and fitting the
    /// next candidates. This is a scheduling choice only: the scored values,
    /// candidate order and archive are identical to the synchronous path, so it
    /// is not saved in a checkpoint.
    pub pipeline_depth: usize,
    /// #463: worker threads for fitting and scoring offspring. Scheduling
    /// only: results are bit-identical for every value, so it is not saved in
    /// a checkpoint. `1` runs every phase serially on the island thread.
    pub threads: usize,
    /// #122: which variation engine to use. `auto` selects the flat postfix
    /// engine only for a large, fit-free GPU run.
    pub variation: VariationMode,
    /// #465: survivor and parent selection inside an island.
    pub selection: Selection,
    /// #470: wrap every candidate shape in a fitted affine output map
    /// `a + b*f(x)`. Changes the trajectory, so it is a saved search setting.
    pub linear_scaling: bool,
    /// #471: the node-path edit weights. Changes the trajectory, so it is a
    /// saved search setting.
    pub mutation_mix: MutationMix,
    /// #471: operator draw in random trees.
    pub tree_arity: TreeArity,
    /// #471: probability that a crossover cut point is an internal node.
    pub crossover_internal: f64,
    /// #471: per-island edit weights learned from success rates.
    pub adaptive_mix: bool,
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

/// #211: development-log callback type. Called once per completed generation on
/// the search's main thread with a fully measured
/// [`GenRecord`](crate::devlog::GenRecord); the CLI formats it as one NDJSON
/// line. Not called at all when no log is attached.
pub type DevFn = std::sync::Arc<dyn Fn(&crate::devlog::GenRecord) + Send + Sync>;

/// #462: front observer callback type. Called once per generation with the
/// 1-based generation, the elapsed search seconds and the archive.
pub type FrontFn = std::sync::Arc<dyn Fn(usize, f64, &Archive) + Send + Sync>;

/// #211: per-generation counters shared by the island threads.
///
/// Every increment is a relaxed atomic add on a counter that only exists when a
/// development log is attached, so the search pays one predictable branch per
/// counted event and nothing else. Nothing here feeds back into variation,
/// fitting or selection: the counters are write-only for the whole generation.
#[derive(Debug, Default)]
pub struct GenCounters {
    pub attempts: std::sync::atomic::AtomicU64,
    pub duplicates: std::sync::atomic::AtomicU64,
    pub too_complex: std::sync::atomic::AtomicU64,
    pub offspring: std::sync::atomic::AtomicU64,
    pub fitted: std::sync::atomic::AtomicU64,
    /// Constant-fitting CPU time, nanoseconds, summed over threads.
    pub fit_ns: std::sync::atomic::AtomicU64,
    /// Candidate-evaluation CPU time, nanoseconds, summed over threads.
    pub eval_ns: std::sync::atomic::AtomicU64,
    /// #463: wall time of the parallel fit-and-score phase, nanoseconds,
    /// summed over islands.
    pub batch_ns: std::sync::atomic::AtomicU64,
    /// #463: candidates dropped after the fit because they exceeded the
    /// complexity cap; each one re-enters variation.
    pub post_fit_drops: std::sync::atomic::AtomicU64,
    /// #500: carried individuals re-scored at the start of a generation.
    pub reevals: std::sync::atomic::AtomicU64,
    /// #477: offspring rejected before the fit because no input reaches the
    /// rows outside an aggregate or meta operator.
    pub constant: std::sync::atomic::AtomicU64,
    /// Whether the fit, eval and batch timers read the clock. Off outside a
    /// development log; the counts are kept either way.
    pub timed: bool,
}

impl GenCounters {
    fn add(slot: &std::sync::atomic::AtomicU64, by: u64) {
        slot.fetch_add(by, std::sync::atomic::Ordering::Relaxed);
    }

    fn get(slot: &std::sync::atomic::AtomicU64) -> u64 {
        slot.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Increment `field` on `self` when a log is attached.
    fn count(c: Option<&GenCounters>, field: fn(&GenCounters) -> &std::sync::atomic::AtomicU64) {
        if let Some(c) = c {
            Self::add(field(c), 1);
        }
    }

    /// Total of `field` across the generation's threads.
    fn total(
        c: Option<&GenCounters>,
        field: fn(&GenCounters) -> &std::sync::atomic::AtomicU64,
    ) -> u64 {
        c.map(|c| Self::get(field(c))).unwrap_or(0)
    }

    /// Total of `field` in seconds.
    fn seconds(
        c: Option<&GenCounters>,
        field: fn(&GenCounters) -> &std::sync::atomic::AtomicU64,
    ) -> f64 {
        Self::total(c, field) as f64 / 1e9
    }
}

/// #211: `fit_constants`, timed and counted only when a log is attached.
#[inline]
fn fit_counted(
    counters: Option<&GenCounters>,
    tree: &mut Node,
    cfg: &Config,
    fit: &Sample,
    score: &Sample,
) -> Option<FitResult> {
    let t = counters.filter(|c| c.timed).map(|_| std::time::Instant::now());
    let out = fit_constants(tree, cfg, fit, score);
    if let (Some(c), Some(t)) = (counters, t) {
        GenCounters::add(&c.fitted, 1);
        GenCounters::add(&c.fit_ns, t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
    out
}

/// #211: an evaluated individual, timed only when a log is attached.
#[inline]
fn eval_counted(counters: Option<&GenCounters>, make: impl FnOnce() -> Ind) -> Ind {
    let t = counters.filter(|c| c.timed).map(|_| std::time::Instant::now());
    let ind = make();
    if let (Some(c), Some(t)) = (counters, t) {
        GenCounters::add(&c.eval_ns, t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
    ind
}

/// #463: candidates per worker below which a batch of cheap fits runs
/// serially, because spawning the workers costs more than the fits.
const MIN_BATCH_PER_WORKER: usize = 16;

/// #463: training rows from which one fit is expensive enough that a worker
/// pays off for a single candidate.
const HEAVY_FIT_ROWS: usize = 2048;

/// #463: workers worth spawning for `n` candidates of `rows` rows.
fn batch_workers(workers: usize, n: usize, rows: usize) -> usize {
    let per = if rows >= HEAVY_FIT_ROWS { 1 } else { MIN_BATCH_PER_WORKER };
    workers.min(n / per).max(1)
}

/// #463: map `f` over `items` on up to `workers` threads and return the
/// results in input order. Work is handed out one item at a time, so uneven
/// fit costs balance. `f` must not touch shared mutable state that changes
/// the result, so the output is independent of `workers`.
fn par_map<T: Send, R: Send>(workers: usize, items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let n = items.len();
    let workers = workers.min(n);
    if workers <= 1 {
        return items.into_iter().map(f).collect();
    }
    let slots: Vec<Mutex<Option<T>>> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
    let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let blocks = expr::current_blocks();
    let run = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        let item = slots[i].lock().unwrap().take().expect("each slot is taken once");
        let r = f(item);
        *out[i].lock().unwrap() = Some(r);
    };
    std::thread::scope(|s| {
        for _ in 1..workers {
            let blocks = blocks.clone();
            let run = &run;
            s.spawn(move || {
                expr::set_blocks(blocks);
                run();
            });
        }
        run();
    });
    out.into_iter()
        .map(|m| m.into_inner().unwrap().expect("every slot is filled"))
        .collect()
}

/// #463: phase B of an island generation. Fit (when `fit`), simplify and score
/// `trees` on `workers` threads; results keep the input order. Neither the fit
/// nor the score draws from the RNG, so the output does not depend on
/// `workers`.
fn fit_and_score(
    trees: Vec<Node>,
    workers: usize,
    cfg: &Config,
    train: &EvalSample,
    fit_sample: &Sample,
    fit: bool,
    defer_gpu: bool,
    counters: Option<&GenCounters>,
) -> Vec<Ind> {
    let t = counters.filter(|c| c.timed).map(|_| std::time::Instant::now());
    let workers = batch_workers(workers, trees.len(), train.sample.len());
    let out = par_map(workers, trees, |mut tree| {
        if cfg.linear_scaling {
            tree = crate::scale::ensure(tree, 0.0, 1.0); // #470
        }
        let mut fitted = None;
        if fit {
            fitted = fit_counted(counters, &mut tree, cfg, fit_sample, train.sample); // #85 #87 #211
            tree = expr::simplify(&tree);
        }
        // #26: with the GPU batch active the fitness is filled in later.
        if defer_gpu {
            Ind::pending(tree, cfg)
        } else {
            eval_counted(counters, || Ind::evaluated_fitted(tree, fitted, train, cfg)) // #25 #87 #211
        }
    });
    if let (Some(c), Some(t)) = (counters, t) {
        GenCounters::add(&c.batch_ns, t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
    out
}

/// #255: default per-generation cap on exact CPU fallback refits. It is a
/// safety bound, not a tuning knob: a generation would have to send more than
/// this many candidates to the CPU before it changes any result. The recorded
/// cell admitted at most 540 per generation.
pub const DEFAULT_FIT_FALLBACK_MAX: usize = 4096;

/// #466: default `--prune-rows`.
pub const DEFAULT_PRUNE_ROWS: usize = 20_000;
/// #466: default `--prune-budget`.
pub const DEFAULT_PRUNE_BUDGET: usize = 20_000;

mod prune;
pub use prune::PruneReport;
/// #495: default relative band of the finalist refit.
pub const DEFAULT_REFIT_BAND: f64 = 0.001;

impl Default for Config {
    fn default() -> Self {
        Config {
            generations: 200,
            islands: 1,
            pop: 200,
            max_complexity: 30,
            complexity_model: ComplexityModel::default(),
            metric: Metric::Absolute,
            fitness: Fitness::Metric,
            seed: 0,
            ops: Op::default_set(),
            migration_interval: 25,
            restart_interval: 60,
            init_depth: 3,
            fit_iters: 8,
            fit_strategy: crate::fit::FitStrategy::Exact, // #116
            fit_polish: crate::fit::FitPolish::F64,
            fit_admission: crate::fit::FitAdmission::Screened,
            fit_report: false,
            dedup: true,
            reject_constant_offspring: false,
            time_budget: None,
            patience: 40,
            patience_tol: None,
            stop_front: None,
            predictor_samples: None,
            fit_samples: 0, // #85: off by default
            fit_rows: 0,    // #116: off by default
            fit_fallback_max: DEFAULT_FIT_FALLBACK_MAX, // #255
            refit_band: DEFAULT_REFIT_BAND, // #495
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
            dev_log: None, // #211: off unless --dev-log is given
            front_observer: None, // #462
            prune: false,         // #466: opt in with --prune
            prune_rows: DEFAULT_PRUNE_ROWS,
            prune_budget: DEFAULT_PRUNE_BUDGET,
            simd: true, // #27
            // #141: the GPU is the default backend when the binary is built
            // with CUDA and a device is available. The probe happens once at
            // startup; a non-CUDA build, a missing device or an unsupported
            // objective falls back to the CPU. `--no-gpu`/`--cpu` is the
            // deliberate override and `--gpu` asserts that a device exists.
            gpu: cfg!(feature = "cuda"),
            gpu_notes: true,
            gpu_memory: None,
            dispatch: crate::dispatch::DispatchMode::Auto,
            pipeline_depth: 2, // #119: only used when the GPU path is active
            threads: 1,
            variation: VariationMode::Auto,
            selection: Selection::Lexicographic,
            linear_scaling: false,
            mutation_mix: MutationMix::Classic,
            tree_arity: TreeArity::Uniform,
            crossover_internal: 0.0,
            adaptive_mix: false,
        }
    }
}

impl Config {
    /// #464: relative tolerance of the front stop.
    fn front_tol(&self) -> f64 {
        self.patience_tol
            .unwrap_or(crate::progress_stats::FRONT_STOP_TOL)
    }

    /// #464: consecutive stable generations that end a run; never reached when 0.
    fn front_window(&self) -> usize {
        let w = match self.stop_front {
            Some(0) => self.patience,
            Some(w) => w,
            None => 0,
        };
        if w == 0 {
            usize::MAX
        } else {
            w
        }
    }

    /// The names of search settings that differ between two configurations.
    ///
    /// `generations` is deliberately excluded: it is a **budget**, not a
    /// property of the trajectory. Resuming may extend it (never shorten it),
    /// and every other setting must match the checkpoint exactly. `progress`
    /// and `dev_log` (#211) are reporting callbacks, not search state, and are
    /// ignored.
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
        if self.fitness.key() != other.fitness.key() {
            diff.push("fitness"); // #480
        }
        compare!(seed, "seed");
        compare!(ops, "ops");
        compare!(migration_interval, "migration_interval");
        compare!(restart_interval, "restart_interval");
        compare!(reject_constant_offspring, "reject_constant_offspring"); // #477
        compare!(init_depth, "init_depth");
        compare!(fit_iters, "fit_iters");
        compare!(fit_strategy, "fit_strategy"); // #116
        compare!(fit_polish, "fit_polish");
        compare!(fit_admission, "fit_admission");
        compare!(linear_scaling, "linear_scaling"); // #470
        compare!(mutation_mix, "mutation_mix"); // #471
        compare!(tree_arity, "tree_arity"); // #471
        compare!(crossover_internal, "crossover_internal"); // #471
        compare!(adaptive_mix, "adaptive_mix"); // #471
        compare!(dedup, "dedup");
        // #256: `time_budget` is a per-process limit, not search state; a resumed
        // run may set a new one.
        compare!(patience, "patience");
        compare!(patience_tol, "patience_tol"); // #464
        compare!(stop_front, "stop_front"); // #464
        compare!(predictor_samples, "predictor_samples");
        compare!(fit_samples, "fit_samples"); // #85
        compare!(fit_rows, "fit_rows"); // #116
        compare!(fit_fallback_max, "fit_fallback_max"); // #255
        compare!(refit_band, "refit_band"); // #495
        compare!(front_parents, "front_parents"); // #76
        compare!(selection, "selection"); // #465
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

    /// #470: the complexity cap for a generated candidate. The affine wrapper is
    /// free, so it extends the cap by exactly what it costs.
    fn complexity_cap(&self) -> f64 {
        let extra = if self.linear_scaling {
            crate::scale::overhead(&self.complexity_model)
        } else {
            0.0
        };
        self.max_complexity as f64 + extra
    }

    /// #471: the variation options of the node path.
    pub fn variation_opts(&self) -> expr::VariationOpts {
        expr::VariationOpts {
            mix: self.mutation_mix.weights(),
            balanced: self.tree_arity == TreeArity::Balanced,
            crossover_internal: self.crossover_internal,
        }
    }

    /// #471: the name of a node-path variation flag that is set. The postfix
    /// engine has its own edits and refuses these.
    pub fn node_variation_flag(&self) -> Option<&'static str> {
        if self.mutation_mix != MutationMix::Classic {
            Some("--mutation-mix")
        } else if self.tree_arity != TreeArity::Uniform {
            Some("--tree-arity")
        } else if self.crossover_internal > 0.0 {
            Some("--crossover-internal")
        } else if self.adaptive_mix {
            Some("--adaptive-mix")
        } else {
            None
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
    /// #465: generations since the lineage began; only `--selection afp` reads it.
    age: u32,
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
                _ => fitness_error_and_descriptor(&tree, train, cfg, want_descriptor), // #27 #480
            }
        } else {
            (f64::INFINITY, None)
        };
        Ind {
            tree,
            error,
            complexity,
            descriptor,
            age: 0,
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
            age: 0,
        }
    }

    fn reeval(&mut self, train: &EvalSample, cfg: &Config) {
        self.tree = expr::simplify(&self.tree);
        self.complexity = cfg.complexity_model.complexity(&self.tree);
        self.score_in_place(train, cfg);
    }

    fn score_in_place(&mut self, train: &EvalSample, cfg: &Config) {
        // #25: re-evaluation must not resurrect a rejected candidate.
        let want_descriptor = cfg.qd_bins > 0 && !train.sample.is_multi();
        let (error, descriptor) = if cfg.accepts_dimensions(&self.tree) {
            fitness_error_and_descriptor(&self.tree, train, cfg, want_descriptor) // #27 #480
        } else {
            (f64::INFINITY, None)
        };
        self.error = error;
        self.descriptor = descriptor;
    }

    /// #500: `reeval` for an individual with a non-finite error. A `+inf` error
    /// stays `+inf`, without a pass over every row, when the dimension filter
    /// rejects the tree or a leading slice of the rows already holds a
    /// non-finite prediction. Returns whether a full scoring pass ran.
    fn reeval_probing(&mut self, train: &EvalSample, cfg: &Config) -> bool {
        self.tree = expr::simplify(&self.tree);
        self.complexity = cfg.complexity_model.complexity(&self.tree);
        if self.error == f64::INFINITY
            && (!cfg.accepts_dimensions(&self.tree) || prefix_is_non_finite(&self.tree, train, cfg))
        {
            self.descriptor = None;
            return false;
        }
        self.score_in_place(train, cfg);
        true
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
        let age = self.age;
        *self = Ind::evaluated_fitted(tree, fitted, train, cfg); // #25 #87
        self.age = age;
    }

    fn to_checkpoint(&self) -> Individual {
        Individual {
            tree: self.tree.clone(),
            error: self.error,
            complexity: self.complexity,
            age: self.age,
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
            age: ind.age,
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
/// The kernel accumulates one objective per row (`f32` per row, `f64`
/// accumulator) and returns the sum; the search divides it by `Σw` (or the row
/// count) so it sees the configured metric's mean. Native objectives are
/// squared, log_squashed (#229), absolute (#222), and the label-aware
/// classification family log_loss/Brier/hinge/hinge_ordinary/hinge_squared
/// (#289); every other metric keeps the CPU evaluator with a named reason
/// (#236). The value is only used to guide the search: the archive and every
/// reported number are rebuilt on the full training sample with the CPU `f64`
/// evaluator.
#[cfg(feature = "cuda")]
struct GpuSearch {
    ev: crate::gpu::GpuEvaluator,
    /// #265: the fixed fit subsample, its own device context (f32 rows, f64
    /// twin, weights) and the row hash. `None` fits on the rows of `ev`.
    fit_sub: Option<FitSubsample>,
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
    /// #448: candidates with constants admitted without a screening score (wide admission).
    fit_admitted_wide: usize,
    /// #169: CPU fallback counts labelled by cause.
    fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #265: fallbacks whose CPU refit lowered the exact error, by cause.
    fit_fallback_improved_by_cause: BTreeMap<String, usize>,
    /// #286: finite device answers grouped by label (`rank_deficient`,
    /// `recovered`, `no_improvement`, `start_unusable`,
    /// `recovered_skipped`). None of these is a CPU fallback.
    fit_status_by_label: BTreeMap<String, usize>,
    /// #169: wall time spent in the batched fit kernels, milliseconds.
    fit_ms: f64,
    /// #496: the polish-phase part of `fit_ms`, apportioned by cycle ratio.
    fit_polish_ms: f64,
    /// #601: device fit launches, candidates they carried, longest launch (ms).
    fit_calls: usize,
    fit_call_candidates: usize,
    fit_call_ms_max: f64,
    /// #169: wall time spent in the whole fit stage (kernel, launch, CPU
    /// fallback and write-back), milliseconds.
    fit_host_ms: f64,
    /// #284/#255: exact CPU fallback seconds inside the fit stage, milliseconds.
    /// This is a subset of `fit_host_ms`, never added to it twice.
    fit_fallback_ms: f64,
    fit_gate_ms: f64,
    /// #600: main-thread milliseconds of the overlap drain spent in scoring and
    /// in the fit stage. The fit part is a subset of `fit_host_ms`.
    drain_eval_ms: f64,
    drain_fit_ms: f64,
    /// #255: candidates left at their finite device constants because the
    /// per-generation fallback bound was reached.
    fit_fallback_bounded: usize,
    /// #255: largest number of scoped workers the fallback batch used.
    fit_fallback_workers: usize,
    /// #284: run clock for device intervals.
    ledger_origin: std::time::Instant,
    /// #284: device intervals and fit sub-ledger. The search adds the exclusive
    /// main-thread stages; this ledger owns what only the evaluator/fitter sees.
    ledger: crate::gpu_stats::RunLedger,
    /// #230: true only when the CUDA evaluator scores this run's metric
    /// itself through the objective kernel. False for an exact order/rank run,
    /// which is scored through [`crate::gpu::GpuEvaluator::score_stat_batch`]
    /// instead (#288), and for a fit-only arm (log-loss fit, rank polish on a
    /// small sample), which the CPU scores; either way the population is never
    /// ranked by a proxy.
    native_metric: bool,
    /// #288: the exact order-statistic/rank statistic this run is scored on,
    /// or `None` when the run uses the objective kernel (`native_metric`) or
    /// is fit-only. The #287 dispatch table names it.
    native_stat: Option<crate::metrics::ScoreStat>,
    /// #287: the exact native statistic the objective kernel computes. `None`
    /// for a fit-only arm and for an order/rank run. `score_pool` uses this,
    /// never the fit objective, so a `correlation` run cannot be ranked by SSE.
    score: Option<crate::metrics::ScoreObjective>,
    /// #290: true once at least one batch actually ran on the device. The
    /// route planner can keep a tiny batch on the CPU even in a GPU run, so the
    /// backend claim must come from what ran, not from the capability flag.
    device_eval_used: bool,
    /// #290: true once at least one fit batch actually ran on the device.
    device_fit_used: bool,
    /// #293: the device memory plan this evaluator was built under.
    memory_plan: Option<crate::memory::DevicePlan>,
    /// #293: the tile stream of a tiled run, `None` when the rows are resident.
    tile_stream: Option<TileStream>,
}

/// #293: what one scored batch streams to the device in a tiled run.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct TileStream {
    tile_bytes: u64,
    tiles: u64,
    /// Candidates per wave.
    wave: usize,
}

/// Column-major `f64` copy of the feature rows, `cols[var * rows + row]`.
#[cfg(feature = "cuda")]
fn column_major(sample: &Sample, n_vars: usize) -> Vec<f64> {
    let n = sample.len();
    let mut cols = vec![0.0f64; n * n_vars];
    for (r, row) in sample.x.iter().enumerate() {
        for (v, &x) in row.iter().enumerate().take(n_vars) {
            cols[v * n + r] = x;
        }
    }
    cols
}

/// A fixed fit subsample resident on the device next to the full-row scorer.
#[cfg(feature = "cuda")]
struct FitSubsample {
    /// The subsample rows, for the exact checks and CPU fallbacks that must
    /// see the rows the device fitted.
    sample: Sample,
    /// Its own context: f32 rows, the f64 twin and the weights of these rows.
    ev: crate::gpu::GpuEvaluator,
    /// FNV-1a of the ascending row indices.
    hash: u64,
}

#[cfg(feature = "cuda")]
impl FitSubsample {
    fn new(
        train: &Sample,
        rows: &[usize],
        n_vars: usize,
        loss: crate::fit::DeviceLoss,
        logistic_fit: bool,
        budget: Option<u64>,
        run_start: std::time::Instant,
    ) -> Result<Self, String> {
        let sample = subset_sample(train, rows);
        let n = sample.len();
        let cols64 = column_major(&sample, n_vars);
        let cols32: Vec<f32> = cols64.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = sample.y.iter().map(|&v| v as f32).collect();
        let Some(mut ev) =
            crate::gpu::GpuEvaluator::with_budget(&cols32, n, n_vars, &y32, budget)
        else {
            return Err(device_refusal("the fit subsample rows"));
        };
        ev.set_run_start(run_start);
        ev.set_loss(loss);
        if logistic_fit || loss == crate::fit::DeviceLoss::Logistic {
            let sign = crate::fit::classification_signs_weighted(&sample.y, &sample.w);
            if !ev.set_fit_sign(&sign) {
                return Err(device_refusal("the fit subsample label sign"));
            }
        }
        if !ev.set_fit_data(&cols64, &sample.y, &sample.w) {
            return Err(device_refusal("the f64 fit subsample"));
        }
        Ok(FitSubsample {
            sample,
            ev,
            hash: subsample_hash(rows),
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuSearch {
    /// #293: host-to-device bytes one batch of `candidates` streams in a tiled
    /// run. Every wave sweeps all tiles, twice for the correlation family.
    fn streamed_bytes(&self, candidates: usize) -> u64 {
        let Some(t) = self.tile_stream else {
            return 0;
        };
        let passes = match self.score {
            Some(
                crate::metrics::ScoreObjective::Correlation
                | crate::metrics::ScoreObjective::CorrelationSigned,
            ) => 2,
            _ => 1,
        };
        let waves = candidates.div_ceil(t.wave).max(1) as u64;
        waves * passes * t.tiles * t.tile_bytes
    }

    fn new(
        train: &Sample,
        fit_data: bool,
        loss: crate::fit::DeviceLoss,
        native_metric: bool,
        // #288: the exact order-statistic/rank statistic this run is scored on.
        native_stat: Option<crate::metrics::ScoreStat>,
        score: Option<crate::metrics::ScoreObjective>,
        // #234/#289: the device fit or classification score needs the label
        // sign for a log-loss run.
        logistic_fit: bool,
        // #284: the run clock every device interval is placed on.
        run_start: std::time::Instant,
        // #293: the device plan: its budget is the ceiling on the live device
        // allocations, and a tiled plan streams the rows instead of holding them.
        // #265: the fixed rows the device fit runs on (ascending training row
        // indices), or empty to fit on every row.
        fit_rows: &[usize],
        plan: Option<&crate::memory::DevicePlan>,
        // #481: the long/flat book `ScoreObjective::LongPnl` reads.
        pnl_book: Option<&crate::trading::FitnessBook>,
    ) -> Result<Self, String> {
        let budget = plan.map(|p| p.budget);
        let tiled = plan.filter(|p| p.mode == crate::memory::Residency::Tiled);
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
        let created = match tiled {
            Some(p) => {
                if fit_data {
                    return Err("device_memory: tiled residency has no device fit or exact statistic scorer".to_string());
                }
                if score == Some(crate::metrics::ScoreObjective::Slope) {
                    return Err("device_memory: tiled residency refuses slope scoring: row r - 1 of a tile's first row is not resident".to_string());
                }
                if score == Some(crate::metrics::ScoreObjective::LongPnl) {
                    return Err(crate::gpu::LONG_PNL_TILED.to_string());
                }
                crate::gpu::GpuEvaluator::new_tiled(
                    cols,
                    n_rows,
                    n_vars,
                    &y,
                    p.tile_rows,
                    budget,
                    Some(p.partials_cap_bytes()),
                )
            }
            None => crate::gpu::GpuEvaluator::with_budget(&cols, n_rows, n_vars, &y, budget),
        };
        match created {
            Some(mut ev) => {
                // #284: one clock for evaluator and fit intervals.
                ev.set_run_start(run_start);
                // #229: the legacy evaluator accumulates the fit objective the
                // run ranks on; `gpu_blocker` has already refused any metric
                // without a native kernel objective.
                ev.set_loss(loss);
                // #481: the book is the first allocation after the rows, once
                // per run.
                let long_pnl = score == Some(crate::metrics::ScoreObjective::LongPnl);
                if long_pnl {
                    let Some(book) = pnl_book else {
                        return Err("the long_pnl fitness book was not rebuilt".to_string());
                    };
                    ev.set_pnl_book(book)?;
                }
                // #289: a classification score uploads its label sign once per
                // sample, whether or not the device also fits constants. The
                // fit-only log-loss arm needs the same vector for the fit
                // kernel, so it is uploaded once and shared.
                if (score.is_some() && !long_pnl) || logistic_fit {
                    let sign =
                        crate::fit::classification_signs_weighted(&train.y, &train.w);
                    if !ev.set_fit_sign(&sign) {
                        return Err(device_refusal("the classification label sign"));
                    }
                }
                // #169: the batched fit's final polish runs on the exact f64
                // rows, so its constants are not limited by the f32 evaluator
                // copy. Only the GPU-fit arm pays this second resident copy.
                //
                // With a fit subsample the twin lives in a second context that
                // holds only those rows; this context keeps a full twin just
                // for the exact order/rank scorer.
                let sub = !fit_rows.is_empty() && fit_rows.len() < n_rows && fit_data;
                if fit_data && (!sub || native_stat.is_some()) {
                    let cols64 = column_major(train, n_vars);
                    if !ev.set_fit_data(&cols64, &train.y, &train.w) {
                        return Err(device_refusal("the f64 fit rows"));
                    }
                } else if !train.w.is_empty() {
                    // #289: a weighted run whose fit stays on the CPU still
                    // needs the per-row weights for the native score.
                    if !ev.set_score_weights(&train.w) {
                        return Err(device_refusal("the row weights"));
                    }
                }
                // #508: a panel's slope pairs skip the group boundaries; the
                // device reads them as a per-row mask.
                let mask = slope_pair_mask(train).unwrap_or(SlopePairMask { bits: Vec::new(), pairs: 0 });
                if !ev.set_slope_pairs(&mask.bits, mask.pairs) {
                    return Err(device_refusal("the slope pair mask"));
                }
                let fit_sub = if sub {
                    Some(FitSubsample::new(
                        train, fit_rows, n_vars, loss, logistic_fit, budget, run_start,
                    )?)
                } else {
                    None
                };
                Ok(GpuSearch {
                    ev,
                    fit_sub,
                    pool: None,
                    stats: BTreeMap::new(),
                    fallback: Some(FallbackPool::new(fallback_workers(), fallback_workers())),
                    fit_attempts: 0,
                    fit_gpu: 0,
                    fit_fallback: 0,
                    fit_screened: 0,
                    fit_admitted_wide: 0,
                    fit_fallback_by_cause: BTreeMap::new(),
                    fit_fallback_improved_by_cause: BTreeMap::new(),
                    fit_status_by_label: BTreeMap::new(),
                    fit_ms: 0.0,
                    fit_polish_ms: 0.0,
                    fit_calls: 0,
                    fit_call_candidates: 0,
                    fit_call_ms_max: 0.0,
                    fit_host_ms: 0.0,
                    fit_fallback_ms: 0.0,
                    fit_gate_ms: 0.0,
                    drain_eval_ms: 0.0,
                    drain_fit_ms: 0.0,
                    fit_fallback_bounded: 0,
                    fit_fallback_workers: 0,
                    ledger_origin: run_start,
                    ledger: crate::gpu_stats::RunLedger::new(),
                    native_metric,
                    native_stat,
                    score,
                    device_eval_used: false,
                    device_fit_used: false,
                    memory_plan: None,
                    tile_stream: tiled.map(|p| TileStream {
                        tile_bytes: p.tile_rows as u64 * n_vars as u64 * 4,
                        tiles: n_rows.div_ceil(p.tile_rows.max(1)) as u64,
                        wave: p.wave_candidates.max(1),
                    }),
                })
            }
            None => Err(match crate::gpu::last_alloc_failure() {
                Some(bytes) => format!(
                    "device_memory: the device refused an allocation of {} bytes{} for the resident rows ({})",
                    bytes,
                    alloc_tag_note(),
                    crate::memory::human(bytes)
                ),
                None => format!(
                    "no usable CUDA device ({})",
                    crate::gpu::GpuEvaluator::device_name()
                ),
            }),
        }
    }

    /// #288: whether the run's fitness comes from this evaluator, so the
    /// backend may report `gpu`. True for both scoring shapes: the objective
    /// kernel (`native_metric`) and the exact order/rank batch scorer.
    fn claims_device(&self) -> bool {
        self.native_metric || self.native_stat.is_some()
    }

    /// #284: current position on the run clock, for device intervals.
    fn ledger_offset_s(&self) -> f64 {
        self.ledger_origin.elapsed().as_secs_f64()
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
        dispatcher: &mut crate::dispatch::Dispatcher,
    ) {
        let (fit_sample, mut fit_ev, fit_hash) = match self.fit_sub.take() {
            Some(FitSubsample { sample, ev, hash }) => (Some(sample), Some(ev), hash),
            None => (None, None, 0),
        };
        // Everything that judges or refits a device proposal sees the rows the
        // device fitted; the caller re-scores on every row afterwards.
        let fit_train: &Sample = fit_sample.as_ref().unwrap_or(train);
        let mut planned: Vec<(usize, usize, crate::fit::GpuFitPlan)> = Vec::new();
        let mut fallbacks: Vec<(usize, usize, FitFallbackCause)> = Vec::new();
        // #229: the objective the batched kernel minimises. #230: the rank
        // family has no LM loss but has its own exact device polish, so it is
        // admitted through `rank_fit_only`. Anything else reaching here is
        // labelled rather than guessed.
        let fit_loss = crate::fit::fit_loss(cfg.metric);
        let rank_loss = crate::fit::rank_fit_only(cfg.metric);
        // #226/#232: `median` and `interquartile` have no LM loss but their own
        // exact device polish on the same squared start and ladder.
        let order_stat_arm = crate::fit::order_stat_fit_only(cfg.metric);
        // #227: `maximum` likewise — an exact maximum-error polish, no LM
        // loss, no order statistic. #287 scores maximum natively, but the fit
        // arm keeps its measured cost model: below the crossover the exact CPU
        // polish runs instead of paying a device launch.
        let maximum_arm = crate::fit::maximum_fit_only(cfg.metric)
            && crate::fit::maximum_gpu_fit_worthwhile(fit_train.len(), 1, cfg.fit_iters);
        // #234: a single-class fit sample has no finite log-loss optimum. The
        // counts come from the same midpoint rule the CPU arm uses, so the
        // device never fits a sign the sample does not support.
        // #289: the predicate is the *fit* objective, not the fit-only split:
        // `log_loss` now has a native scorer, but its device fit still needs
        // both classes to have a finite optimum. The hinge codes read the same
        // sign and are declined the same way.
        let single_class = crate::fit::fit_loss(cfg.metric)
            .is_some_and(|loss| loss.needs_label_sign())
            && {
                let (pos, neg) = crate::fit::class_counts_weighted(&fit_train.y, &fit_train.w);
                pos == 0 || neg == 0
            };
        for &(i, j) in admitted {
            if single_class {
                fallbacks.push((i, j, FitFallbackCause::SingleClass));
                continue;
            }
            if matches!(cfg.metric, Metric::Correlation | Metric::CorrelationSigned)
                && crate::fit::is_affine_correlation_tree(&islands[i][j].tree)
            {
                fallbacks.push((i, j, FitFallbackCause::AffineClosedForm));
                continue;
            }
            match crate::fit::gpu_fit_plan_reason(&islands[i][j].tree) {
                _ if fit_loss.is_none()
                    && rank_loss.is_none()
                    && !order_stat_arm
                    && !maximum_arm =>
                {
                    fallbacks.push((i, j, FitFallbackCause::MetricNotNative))
                }
                Ok(plan) => planned.push((i, j, plan)),
                Err(reason) => fallbacks.push((i, j, plan_fallback_cause(reason))),
            }
        }
        // #453: the device runs the admitted candidates as one batch, so the
        // cost model plans the batch once, not each candidate alone.
        let batch = plan_fit_batch(dispatcher, fit_train.len(), cfg.fit_iters, fit_loss, planned);
        let batch_bytes = batch.bytes;
        let batch_decision = batch.decision;
        fallbacks.extend(batch.refused);
        let jobs = batch.jobs;
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
            // #230: the rank family runs its own device arm — the same squared
            // LM start as `fit_weighted_metric`, followed by the batched exact
            // rank polish under the same first-improver ladder. #226/#232:
            // `median` and `interquartile` do the same on their own exact order
            // statistic. Every other metric stays on the LM kernel of
            // #169/#229.
            let want = crate::fit::OrderStat::for_metric(cfg.metric);
            let fit_device_start = self.ledger_offset_s();
            // The fit subsample has its own context; without one the fit runs
            // on the scoring context's rows.
            let dev = match fit_ev.as_mut() {
                Some(ev) => ev,
                None => &mut self.ev,
            };
            dev.set_fit_polish(cfg.fit_polish);
            let (out, ms) = if maximum_arm {
                // #227: the exact maximum-error polish on the squared start.
                dev.max_fit_batch(&plans, cfg.fit_iters)
            } else if let Some(want) = want {
                dev.order_stat_fit_batch(&plans, cfg.fit_iters, want)
            } else {
                match crate::fit::rank_metric_signed(cfg.metric) {
                    Some(signed) => {
                        dev.rank_fit_batch(&plans, cfg.fit_iters, &fit_train.y, signed)
                    }
                    // #236: the squared warm start, then the differenced IRLS,
                    // keeping the incoming constants unless it improves on them.
                    None if fit_loss == Some(crate::fit::DeviceLoss::Slope) => {
                        dev.slope_fit_batch(&plans, cfg.fit_iters)
                    }
                    None => {
                        let loss = fit_loss.expect(
                            "jobs is only filled when a native, rank, order-statistic or \
                             maximum arm exists",
                        );
                        dev.fit_batch(&plans, cfg.fit_iters, loss)
                    }
                }
            };
            if ms > 0.0 {
                self.fit_ms += ms as f64;
                self.fit_polish_ms += dev.take_polish_ms(ms);
                self.fit_calls += 1;
                self.fit_call_candidates += plans.len();
                self.fit_call_ms_max = self.fit_call_ms_max.max(ms as f64);
                // #284: the fit kernel is one device interval on the run clock;
                // its duration is the raw event time, not the host envelope.
                match dev.take_span_start_s(crate::gpu::SPAN_FIT) {
                    Some(start_s) => self
                        .ledger
                        .record_fit_kernel_event(0, start_s, ms as f64 / 1000.0),
                    None => self
                        .ledger
                        .record_fit_kernel_interval(0, fit_device_start, ms as f64 / 1000.0),
                }
            }
            if out.len() == jobs.len() {
                // #290: a successful batch is what makes this run device-used.
                self.device_fit_used = true;
                // #234: the log-loss arm judges the proposal by the configured
                // objective in exact f64, against the exact value at the
                // starting constants. The kernel's acceptance is therefore a
                // suggestion, never the final word. The other LM arms keep
                // their historical trajectory (their proposal is the kernel's
                // own optimum of the same objective).
                let gated = crate::fit::device_fit_is_gated(cfg.metric);
                // #498: the exact judgements are independent, so they run on
                // the worker pool; the verdicts are applied below in admission
                // order.
                let mut verdicts: Vec<Option<(crate::fit::Verdict, Node)>> = Vec::new();
                if gated {
                    let gate_start = std::time::Instant::now();
                    let gate_jobs: Vec<(usize, Node, Vec<f64>)> = jobs
                        .iter()
                        .zip(&out)
                        .enumerate()
                        .filter(|(_, (_, res))| {
                            crate::fit::fit_status_usable(res.status)
                                && res.c.iter().all(|v| v.is_finite())
                        })
                        .map(|(k, ((i, j, _), res))| {
                            (k, islands[*i][*j].tree.clone(), res.c.clone())
                        })
                        .collect();
                    verdicts.resize_with(jobs.len(), || None);
                    for (k, judged) in gate_proposals(gate_jobs, fit_train, cfg.metric, cfg.threads)
                    {
                        verdicts[k] = Some(judged);
                    }
                    let gate_seconds = gate_start.elapsed().as_secs_f64();
                    self.fit_gate_ms += gate_seconds * 1000.0;
                    self.stats.entry(generation).or_default().fit_gate_seconds += gate_seconds;
                }
                for (k, ((i, j, plan), res)) in jobs.iter().zip(out).enumerate() {
                    let finite = res.c.iter().all(|v| v.is_finite());
                    let usable = crate::fit::fit_status_usable(res.status);
                    if gated && usable && finite {
                        let (verdict, judged) =
                            verdicts[k].take().expect("every gated job has a verdict");
                        if matches!(verdict, crate::fit::Verdict::Accepted(_)) {
                            islands[*i][*j].tree = judged;
                        }
                        let cause = match verdict {
                            crate::fit::Verdict::Accepted(_) => {
                                self.fit_gpu += 1;
                                self.stats.entry(generation).or_default().fit_gpu += 1;
                                crate::fit::note_gpu_fit(true);
                                continue;
                            }
                            crate::fit::Verdict::NonFinite => FitFallbackCause::NonFinite,
                            crate::fit::Verdict::Rejected if cfg.metric == Metric::LogLoss => {
                                FitFallbackCause::LogLossWorse
                            }
                            crate::fit::Verdict::Rejected => FitFallbackCause::Worse,
                            crate::fit::Verdict::NotIdentifiable => {
                                FitFallbackCause::NotIdentifiable
                            }
                        };
                        fallbacks.push((*i, *j, cause));
                        self.note_fit_fallback(generation, cause);
                        crate::fit::note_gpu_fit(false);
                        continue;
                    }
                    if finite {
                        plan.apply(&mut islands[*i][*j].tree, &res.c);
                    }
                    if usable && finite {
                        self.fit_gpu += 1;
                        self.stats.entry(generation).or_default().fit_gpu += 1;
                        if let Some(label) = gpu_status_label(res.status) {
                            *self
                                .fit_status_by_label
                                .entry(label.to_string())
                                .or_default() += 1;
                        }
                        crate::fit::note_gpu_fit(true);
                    } else {
                        let cause = if !finite {
                            FitFallbackCause::NonFinite
                        } else if let Some(want) = want {
                            order_stat_status_fallback_cause(res.status, want)
                        } else if maximum_arm {
                            maximum_status_fallback_cause(res.status)
                        } else {
                            status_fallback_cause(res.status)
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
            // #290: attribute the measured batch time to the one planned batch
            // decision (#453).
            if let Some(decision) = batch_decision {
                dispatcher.record(
                    decision,
                    ms.max(0.0) as f64 / 1000.0,
                    jobs.len(),
                    batch_bytes,
                );
            }
        }

        // #255: exact CPU fallback, seeded with the GPU constants when there
        // were any. The candidates are independent and their admission order is
        // fixed, so the batch is split across scoped workers exactly like the
        // finalist refits in `full_data_archive`; the results are written back
        // in the original order and are bit-identical to the serial loop.
        let fallback_start = std::time::Instant::now();
        let (refit, bounded) = fallback_batch_split(fallbacks.len(), cfg.fit_fallback_max);
        if bounded > 0 {
            self.fit_fallback_bounded += bounded;
            let stats = self.stats.entry(generation).or_default();
            stats.fit_fallback_bounded += bounded;
        }
        let mut workers_used = 0usize;
        if refit > 0 {
            let mut jobs: Vec<FallbackFitJob> = fallbacks[..refit]
                .iter()
                .map(|&(island, slot, _)| {
                    FallbackFitJob::new(island, slot, islands[island][slot].tree.clone())
                })
                .collect();
            workers_used = fit_fallback_jobs(
                &mut jobs,
                &fit_train.x,
                &fit_train.y,
                &fit_train.w,
                cfg.fit_iters,
                cfg.metric,
                fallback_workers(),
            );
            for (&(_, _, cause), job) in fallbacks[..refit].iter().zip(&jobs) {
                if job.improved {
                    self.note_fit_fallback_improved(generation, cause);
                }
            }
            for job in jobs {
                islands[job.island][job.slot].tree = job.tree;
            }
        }
        // #284: fallback seconds are measured around the CPU arm itself, not
        // inferred from a count. This is a leaf of the fit host window.
        let fallback_seconds = fallback_start.elapsed().as_secs_f64();
        self.fit_fallback_ms += fallback_seconds * 1000.0;
        self.fit_fallback_workers = self.fit_fallback_workers.max(workers_used);
        let stats = self.stats.entry(generation).or_default();
        stats.fit_fallback_seconds += fallback_seconds;
        stats.fit_fallback_workers = stats.fit_fallback_workers.max(workers_used);
        if let (Some(sample), Some(ev)) = (fit_sample, fit_ev) {
            self.fit_sub = Some(FitSubsample {
                sample,
                ev,
                hash: fit_hash,
            });
        }
        for &(i, j) in admitted {
            refresh_after_fit(&mut islands[i][j], cfg);
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

    /// #265: record that the CPU refit of a fallback of `cause` lowered the
    /// exact error.
    fn note_fit_fallback_improved(&mut self, generation: usize, cause: FitFallbackCause) {
        count_improved(&mut self.fit_fallback_improved_by_cause, cause);
        let stats = self.stats.entry(generation).or_default();
        count_improved(&mut stats.fit_fallback_improved_by_cause, cause);
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
        dispatcher: &mut crate::dispatch::Dispatcher,
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

        // #288: an exact order-statistic/rank run is scored by the batch
        // scorer over flattened plans, not by the objective (SSE/absolute/
        // log-squashed) kernel. Routing it through `eval_pool` would rank the
        // population by a proxy the metric is not defined on — the #236 bug.
        if defers_fit_to_gpu(cfg, true) {
            // #169 admission: screen every unfitted offspring with its seed
            // constants first, then fit only the best fraction of each island.
            // The non-promoted offspring keep this screening score, so the
            // generation evaluates every candidate once and only survivors a
            // second time after the fit.
            let wide = admits_all(cfg, train.len());
            // #393: with wide admission every candidate with constants is
            // fitted and scored once after the fit, so only the constant-free
            // ones need the screening score.
            let screened: Vec<(usize, usize)> = if wide {
                pending
                    .iter()
                    .copied()
                    .filter(|&(i, j)| {
                        matches!(
                            crate::fit::gpu_fit_plan_reason(&islands[i][j].tree),
                            Err(crate::fit::GpuFitReject::NoConstants)
                        )
                    })
                    .collect()
            } else {
                pending.clone()
            };
            self.fit_screened += screened.len();
            self.stats.entry(generation).or_default().fit_screened += screened.len();
            let unscreened = pending.len() - screened.len();
            self.fit_admitted_wide += unscreened;
            self.stats.entry(generation).or_default().fit_admitted_wide += unscreened;
            if !screened.is_empty() {
                self.evaluate_the_batch(islands, &screened, train, cfg, generation, dispatcher)?;
            }
            let admitted = select_promoted(islands, &pending, wide);
            let fit_start = std::time::Instant::now();
            self.fit_pending(islands, &admitted, train, cfg, generation, dispatcher);
            self.fit_host_ms += fit_start.elapsed().as_secs_f64() * 1000.0;
            // Re-score the survivors so selection sees their fitted constants.
            self.evaluate_the_batch(islands, &admitted, train, cfg, generation, dispatcher)?;
            return Ok(());
        }

        self.evaluate_the_batch(islands, &pending, train, cfg, generation, dispatcher)
    }

    /// #288: dispatch one batch to the scorer that can rank this run's metric.
    ///
    /// Both arms return the same value contract: the configured objective's
    /// mean/statistic, and `+inf` for a candidate with a non-finite prediction.
    /// A device decline is an `Err`, which turns the whole generation back to
    /// the exact CPU scorer rather than leaving a proxy or a missing value.
    fn evaluate_the_batch(
        &mut self,
        islands: &mut [Vec<Ind>],
        pending: &[(usize, usize)],
        train: &Sample,
        cfg: &Config,
        generation: usize,
        dispatcher: &mut crate::dispatch::Dispatcher,
    ) -> Result<(), String> {
        if self.native_stat.is_some() {
            self.evaluate_pending_stat(islands, pending, train, cfg, generation, dispatcher)
        } else {
            self.evaluate_pending(islands, pending, train, cfg, generation, dispatcher)
        }
    }

    /// #288: score one explicit set of pending candidates through the exact
    /// order-statistic/rank batch entry point.
    ///
    /// Each candidate is flattened to a [`crate::fit::GpuFitPlan`] with its
    /// current constants. A tree the flattening cannot represent is scored by
    /// the bounded CPU pool, so it never loses fitness; its cause is the same
    /// conservative `unsupported_plan` family the fit arm uses. The GPU and
    /// the CPU fallback overlap exactly as in [`Self::evaluate_pending`].
    fn evaluate_pending_stat(
        &mut self,
        islands: &mut [Vec<Ind>],
        pending: &[(usize, usize)],
        train: &Sample,
        cfg: &Config,
        generation: usize,
        dispatcher: &mut crate::dispatch::Dispatcher,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();
        let nodes = pending
            .iter()
            .map(|&(i, j)| islands[i][j].complexity.max(1.0))
            .fold(1.0f64, f64::max) as usize;
        let eval_workload = crate::dispatch::Workload::eval(pending.len(), train.len(), nodes);
        let eval_decision = dispatcher.plan(crate::dispatch::Stage::Eval, &eval_workload);
        let force_cpu = eval_decision.route == crate::dispatch::Route::Cpu;
        let stat = self
            .native_stat
            .expect("evaluate_pending_stat is only called for a native score stat");
        let mut plans: Vec<crate::fit::GpuFitPlan> = Vec::with_capacity(pending.len());
        let mut slots: Vec<(usize, usize)> = Vec::with_capacity(pending.len());
        let mut on_cpu: Vec<(usize, usize)> = Vec::new();
        let stats = self.stats.entry(generation).or_default();
        stats.generation = generation;
        for &(i, j) in pending {
            if force_cpu {
                on_cpu.push((i, j));
                continue;
            }
            match crate::fit::gpu_fit_plan_reason(&islands[i][j].tree) {
                Ok(plan) => {
                    plans.push(plan);
                    slots.push((i, j));
                }
                Err(_) => on_cpu.push((i, j)),
            }
        }
        stats.programs += plans.len();
        stats.native += plans.len();
        stats.fallbacks += on_cpu.len();
        stats.unsupported_programs += on_cpu.len();
        stats.fallback_workers = self
            .fallback
            .as_ref()
            .expect("a GPU search owns its fallback pool")
            .workers();

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
        if !plans.is_empty() {
            let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().collect();
            let constants: Vec<Vec<f64>> = refs.iter().map(|p| p.c0.clone()).collect();
            match self.ev.score_stat_batch(&refs, &constants, stat) {
                Some(values) if values.len() == refs.len() => {
                    for (&(i, j), value) in slots.iter().zip(values) {
                        islands[i][j].error = value.unwrap_or(f64::INFINITY);
                    }
                    self.device_eval_used = true;
                }
                _ => {
                    gpu_error = Some(format!(
                        "CUDA {} batch scoring failed",
                        stat.name()
                    ));
                }
            }
        }
        if let Some(ticket) = ticket {
            let (errors, seconds) = self
                .fallback
                .as_ref()
                .expect("a GPU search owns its fallback pool")
                .finish(ticket);
            if gpu_error.is_none() {
                for (&(i, j), e) in on_cpu.iter().zip(errors.iter()) {
                    islands[i][j].error = *e;
                }
            }
            let stats = self.stats.entry(generation).or_default();
            stats.fallback_seconds += seconds;
            self.ledger
                .record_eval_fallback(on_cpu.len() as u64, seconds);
        }
        dispatcher.record(
            eval_decision,
            stage_start.elapsed().as_secs_f64(),
            pending.len(),
            eval_workload.bytes,
        );
        if let Some(e) = gpu_error {
            return Err(e);
        }
        Ok(())
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
        dispatcher: &mut crate::dispatch::Dispatcher,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();
        let nodes = pending
            .iter()
            .map(|&(i, j)| islands[i][j].complexity.max(1.0))
            .fold(1.0f64, f64::max) as usize;
        let eval_workload = crate::dispatch::Workload::eval(pending.len(), train.len(), nodes);
        let eval_workload = eval_workload
            .with_bytes(eval_workload.bytes + self.streamed_bytes(pending.len()));
        // #290: a fit-only device run has no native scorer, so the evaluator is
        // CPU by contract and the plan says so. A native run lets the model
        // choose; a small batch keeps the CPU when launch dominates.
        let eval_decision = if self.native_metric {
            dispatcher.plan(crate::dispatch::Stage::Eval, &eval_workload)
        } else {
            dispatcher.plan(
                crate::dispatch::Stage::Eval,
                &eval_workload.with_refusal(crate::dispatch::FallbackCause::MetricNotNative),
            )
        };
        let force_cpu = eval_decision.route == crate::dispatch::Route::Cpu;
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
        let mut pcount: Vec<i32> = Vec::with_capacity(pending.len());
        for &(i, j) in pending {
            let meta = crate::genome::GenomeMeta {
                island: i as u16,
                tag: j as u32,
                ..Default::default()
            };
            // #230: the rank family is fitted on the device but scored on the
            // CPU, so its fitness never comes from the squared-error kernel.
            let compiled = if self.native_metric && !force_cpu {
                crate::gpu::compile_program(&islands[i][j].tree).ok()
            } else {
                None
            };
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
                    pcount.push(crate::expr::count_constants(&islands[i][j].tree) as i32);
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
            // #290: a whole batch routed to the CPU by the cost model does not
            // need another SoA transpose. The scalar view scores exactly the
            // same rows without paying (or counting) a full-train transpose;
            // the SoA view is kept for the mixed supported/unsupported case
            // that already needed one before this routing existed.
            if force_cpu && slots.is_empty() {
                Some(EvalSample {
                    sample: train,
                    cols: None,
                })
            } else {
                Some(EvalSample::new(train, cfg))
            }
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
            let objective = self
                .score
                .expect("a native GPU search owns a ScoreObjective");
            let batch = self.ev.score_pool(pool, objective, &pcount);
            if batch.ms < 0.0 || !batch.is_consistent() || batch.values.len() != pool.len() {
                // Do not return before the submitted fallback batch is drained:
                // a stale result would be read by the next call.
                gpu_error = Some("CUDA batch evaluation failed".to_string());
            } else {
                for (&(i, j), (&value, &status)) in slots
                    .iter()
                    .zip(batch.values.iter().zip(batch.status.iter()))
                {
                    // #287/#289: a non-finite row is `+inf` with a status; an
                    // Ok value is already the configured statistic (mean, RSS +
                    // penalty, max, 1-rho or a weighted classification mean),
                    // never an SSE to divide by n.
                    islands[i][j].error = if status.is_ok() {
                        value
                    } else {
                        f64::INFINITY
                    };
                }
                // The generation is complete: the just-evaluated arena becomes
                // the readable front, and the next generation writes into the
                // other one.
                pool.commit();
                // #290: only a batch that actually ran on the device sets this.
                self.device_eval_used = true;
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
            // #284: evaluator fallback is a separate measured leaf; its programs
            // are already counted in `stats.fallbacks`.
            self.ledger
                .record_eval_fallback(on_cpu.len() as u64, seconds);
        }
        if let Some(e) = gpu_error {
            // Record the failed call before returning: a failed launch is still
            // a measured stage with a route and an actual, not a silent gap.
            dispatcher.record(
                eval_decision,
                stage_start.elapsed().as_secs_f64(),
                pending.len(),
                eval_workload.bytes,
            );
            return Err(e);
        }
        dispatcher.record(
            eval_decision,
            stage_start.elapsed().as_secs_f64(),
            pending.len(),
            eval_workload.bytes,
        );
        Ok(())
    }

    /// #131: the per-generation report, in generation order. #118 adds the
    /// accumulated full-path accounting from the same evaluator.
    fn run_stats(&self) -> GpuRunStats {
        // #284: the evaluator/fitter side of the run ledger. The exclusive
        // main-thread stages are added by the caller; this half owns the device
        // intervals and the fit leaves.
        let mut ledger = self.ev.ledger().clone();
        ledger.set_eval_window(self.ev.stats().wall_s);
        ledger.absorb(&self.ledger);
        // The fit kernels run in the subsample context when there is one.
        let fit_ev = self.fit_sub.as_ref().map_or(&self.ev, |s| &s.ev);
        let arena = fit_ev.arena_report();
        ledger.set_fit_alloc(crate::gpu_stats::FitAlloc {
            alloc_calls: arena.fit_alloc_calls,
            growths: arena.fit_growths,
            steady_alloc_calls: arena.fit_steady_alloc_calls,
            batches: arena.fit_batches,
        });
        ledger.record_fit(
            self.fit_host_ms / 1000.0,
            self.fit_ms / 1000.0,
            self.fit_fallback_ms / 1000.0,
        );
        ledger.record_fit_polish_split(self.fit_polish_ms / 1000.0);
        ledger.record_fit_gate(self.fit_gate_ms / 1000.0);
        ledger.record_fit_counters(
            self.fit_attempts as u64,
            self.fit_gpu as u64,
            self.fit_fallback as u64,
            self.fit_fallback_by_cause.clone(),
            self.fit_fallback_improved_by_cause.clone(),
        );
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
            fit_admitted_wide: self.fit_admitted_wide,
            fit_fallback_by_cause: self.fit_fallback_by_cause.clone(),
            fit_fallback_improved_by_cause: self.fit_fallback_improved_by_cause.clone(),
            fit_status_by_label: self.fit_status_by_label.clone(),
            fit_refits: 0,
            fit_refit_candidates: 0,
            fit_refit_s: 0.0,
            fit_rescore_s: 0.0,
            fit_ms: self.fit_ms,
            fit_polish_ms: self.fit_polish_ms,
            fit_calls: self.fit_calls,
            fit_call_candidates: self.fit_call_candidates,
            fit_call_ms_max: self.fit_call_ms_max,
            fit_host_ms: self.fit_host_ms,
            fit_fallback_ms: self.fit_fallback_ms,
            fit_gate_ms: self.fit_gate_ms,
fit_fallback_bounded: self.fit_fallback_bounded,
            fit_fallback_workers: self.fit_fallback_workers,
            fit_fallback_max: DEFAULT_FIT_FALLBACK_MAX,
            fit_rows: fit_ev.fit_twin().0 as usize,
            score_rows: self.ev.rows(),
            fit_twin_bytes: fit_ev.fit_twin().1,
            fit_subsample_hash: self.fit_sub.as_ref().map(|s| s.hash),
            fit_polish: crate::fit::FitPolish::F64,
            fit_admission: crate::fit::FitAdmission::Screened,
            ledger,
            memory_plan: self.memory_plan.clone(),
            observed: Some(self.ev.alloc_report().observed()),
            score_objective: self.score,
        }
    }

    /// #211: a cheap snapshot of this evaluator's ledger for the development
    /// log. The run-level counters are monotone; the caller subtracts the
    /// previous generation's snapshot. The per-batch fields come from the stats
    /// this generation recorded, so they are already that generation's alone.
    fn dev_snapshot(&self, generation: usize) -> GpuSnapshot {
        let batch = self.stats.get(&generation);
        GpuSnapshot {
            accounting: self.ev.stats(),
            fit_attempts: self.fit_attempts,
            fit_gpu: self.fit_gpu,
            fit_fallback: self.fit_fallback,
            fit_fallback_by_cause: self.fit_fallback_by_cause.clone(),
            fit_fallback_improved_by_cause: self.fit_fallback_improved_by_cause.clone(),
            fallback_s: batch.map(|b| b.fallback_seconds).unwrap_or(0.0),
            fallback_workers: batch.map(|b| b.fallback_workers).unwrap_or(0),
            programs: batch.map(|b| b.programs).unwrap_or(0),
            native: batch.map(|b| b.native).unwrap_or(0),
            lowered: batch.map(|b| b.lowered).unwrap_or(0),
            fallbacks: batch.map(|b| b.fallbacks).unwrap_or(0),
            fit_host_s: self.fit_host_ms / 1000.0,
            fit_fallback_s: self.fit_fallback_ms / 1000.0,
            fit_gate_s: self.fit_gate_ms / 1000.0,
            fit_fallback_workers: self.fit_fallback_workers,
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

/// The configured metric at the tree's current constants, in exact f64.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn exact_error(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64], metric: Metric) -> f64 {
    let pred: Vec<f64> = x.iter().map(|row| expr::eval(tree, row)).collect();
    score_weighted(&pred, y, w, metric, expr::count_constants(tree))
}

/// #265: run the exact CPU fit and report whether it lowered the exact error
/// compared with the constants the tree held on entry (the device constants, or
/// the incoming ones when the device produced none).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn refit_lowers_error(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
) -> bool {
    let before = exact_error(tree, x, y, w, metric);
    crate::fit::fit_weighted_metric(tree, x, y, w, iters, metric);
    let after = exact_error(tree, x, y, w, metric);
    after.is_finite() && (before.is_nan() || after < before)
}

/// #265: count one improved refit under its fallback cause.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn count_improved(map: &mut BTreeMap<String, usize>, cause: FitFallbackCause) {
    *map.entry(cause.label().to_string()).or_default() += 1;
}

/// #255: split a fallback batch into the part that is refit and the part held
/// at its finite device constants by the per-generation bound.
///
/// A cap of `0` means unbounded. The split is a pure function of the admitted
/// count and the configured cap, so the same seed and configuration always
/// choose the same prefix; the caller keeps the first `refit` entries in the
/// deterministic admission order.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn fallback_batch_split(total: usize, cap: usize) -> (usize, usize) {
    if cap == 0 {
        (total, 0)
    } else {
        (total.min(cap), total.saturating_sub(cap))
    }
}

/// #255: one exact CPU fallback fit, owned so the batch can be split across
/// scoped workers without aliasing the islands.
#[cfg(feature = "cuda")]
struct FallbackFitJob {
    island: usize,
    slot: usize,
    tree: Node,
    improved: bool,
}

#[cfg(feature = "cuda")]
impl FallbackFitJob {
    fn new(island: usize, slot: usize, tree: Node) -> Self {
        FallbackFitJob {
            island,
            slot,
            tree,
            improved: false,
        }
    }
}

/// #255: run the exact CPU fallback fits across at most `workers` scoped
/// threads, in fixed order. Each candidate is independent and writes only its
/// own tree, so the outcome does not depend on which worker finishes first.
/// Returns the number of workers that actually received a chunk.
#[cfg(feature = "cuda")]
fn fit_fallback_jobs(
    jobs: &mut [FallbackFitJob],
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
    metric: Metric,
    workers: usize,
) -> usize {
    if jobs.is_empty() {
        return 0;
    }
    let workers = workers.max(1).min(jobs.len());
    if workers == 1 {
        for job in jobs.iter_mut() {
            job.improved = refit_lowers_error(&mut job.tree, x, y, w, iters, metric);
        }
        return 1;
    }
    let per = jobs.len().div_ceil(workers);
    let chunks = jobs.len().div_ceil(per);
    let blocks = expr::current_blocks();
    std::thread::scope(|scope| {
        for chunk in jobs.chunks_mut(per) {
            let blocks = blocks.clone();
            scope.spawn(move || {
                expr::set_blocks(blocks);
                for job in chunk.iter_mut() {
                    job.improved = refit_lowers_error(&mut job.tree, x, y, w, iters, metric);
                }
            });
        }
    });
    chunks
}

/// #498: judge device proposals with the exact gate on up to `workers`
/// threads. Each job is `(position, tree, proposal)`; the result keeps the job
/// order and carries the tree as the gate left it, so the outcome does not
/// depend on `workers`.
#[cfg(feature = "cuda")]
fn gate_proposals(
    jobs: Vec<(usize, Node, Vec<f64>)>,
    sample: &Sample,
    metric: Metric,
    workers: usize,
) -> Vec<(usize, (crate::fit::Verdict, Node))> {
    par_map(workers.max(1), jobs, |(k, mut tree, proposal)| {
        let verdict = crate::fit::gate_device_proposal(
            &mut tree, &proposal, &sample.x, &sample.y, &sample.w, metric,
        );
        (k, (verdict, tree))
    })
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
    dispatcher: &mut crate::dispatch::Dispatcher,
) -> bool {
    let Some(g) = gpu.as_mut() else {
        return false;
    };
    match g.score(islands, train, cfg, generation, dispatcher) {
        Ok(()) => {
            // #230: a fit-only evaluator scores on the CPU, so it must not make
            // the run claim `backend = gpu`. The device use is reported through
            // `fit_only_device` and the `gpu.fit` counters instead. #290: a
            // native run claims the GPU only when a batch actually ran there;
            // the cost model may have kept every small batch on the CPU.
            // #288: an exact order/rank run is scored on the device too.
            // Preserve a `used` already set by the overlap pipeline: this call
            // only scores the leftovers and must not clear a prior device use.
            *used = *used || (g.claims_device() && (g.device_eval_used || g.device_fit_used));
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
fn eval_pipeline_jobs(
    g: &mut GpuSearch,
    jobs: &[PipelineJob],
    generation: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut pop = crate::gpu::Population::new();
    let mut kept: Vec<usize> = Vec::with_capacity(jobs.len());
    let mut pcount: Vec<i32> = Vec::with_capacity(jobs.len());
    for (k, (_, _, tree)) in jobs.iter().enumerate() {
        if pop.add(tree) {
            kept.push(k);
            pcount.push(crate::expr::count_constants(tree) as i32);
        }
    }
    if pop.is_empty() {
        return Ok((Vec::new(), Vec::new()));
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
    let objective = g
        .score
        .ok_or_else(|| "pipeline scoring without a native ScoreObjective".to_string())?;
    let batch = g.ev.score_population(&pop, objective, &pcount);
    if batch.ms < 0.0 || !batch.is_consistent() || batch.values.len() != pop.len() {
        return Err("CUDA batch evaluation failed".to_string());
    }
    let errs = batch
        .values
        .iter()
        .zip(batch.status.iter())
        .map(|(&value, &status)| {
            if status.is_ok() {
                value
            } else {
                f64::INFINITY
            }
        })
        .collect();
    Ok((kept, errs))
}

#[cfg(feature = "cuda")]
fn score_pipeline_batch(
    jobs: Vec<PipelineJob>,
    gpu: &mut Option<GpuSearch>,
    results: &mut [Vec<Option<PipelineResult>>],
    train: &Sample,
    generation: usize,
    fit: Option<PipelineFit>,
    pending: &mut PipelineFitQueue,
) -> Result<(), String> {
    let Some(g) = gpu.as_mut() else {
        return Err("pipeline scoring without a GPU evaluator".to_string());
    };
    let offered_hash: Vec<u64> = jobs.iter().map(|(_, _, t)| expr::structural_hash(t)).collect();
    let eval_start = std::time::Instant::now();
    let (kept, errs) = eval_pipeline_jobs(g, &jobs, generation)?;
    g.drain_eval_ms += eval_start.elapsed().as_secs_f64() * 1000.0;
    if !kept.is_empty() {
        // #290: the overlap pipeline is a real device batch too; record it so
        // the route ledger and the backend claim come from what ran.
        g.device_eval_used = true;
    }
    for (&k, &e) in kept.iter().zip(&errs) {
        let (i, j) = (jobs[k].0, jobs[k].1);
        results[i][j] = Some(PipelineResult {
            error: e,
            offered: offered_hash[k],
            tree: None,
        });
    }
    let Some(fit) = fit else {
        return Ok(());
    };

    // #601: the candidates with constants wait in the queue until a device
    // call is big enough to fill the GPU (`FIT_BATCH_MIN`).
    for (k, job) in jobs.into_iter().enumerate() {
        match crate::fit::gpu_fit_plan_reason(&job.2) {
            Err(crate::fit::GpuFitReject::NoConstants) => {}
            plan => pending.push(job, offered_hash[k], plan),
        }
    }
    if pending.planned() >= FIT_BATCH_MIN {
        fit_pipeline_queue(pending, gpu, results, train, generation, fit)?;
    }
    Ok(())
}

/// #601: fit everything the queue holds in one device call, then score the
/// fitted trees. Called when the queue reaches `FIT_BATCH_MIN` planned
/// candidates and once more when the generation's queue closes.
#[cfg(feature = "cuda")]
fn fit_pipeline_queue(
    pending: &mut PipelineFitQueue,
    gpu: &mut Option<GpuSearch>,
    results: &mut [Vec<Option<PipelineResult>>],
    train: &Sample,
    generation: usize,
    fit: PipelineFit,
) -> Result<(), String> {
    let Some(g) = gpu.as_mut() else {
        return Err("pipeline scoring without a GPU evaluator".to_string());
    };
    let (mut jobs, offered_hash, entries) = pending.take();
    let mut plans: Vec<(usize, crate::fit::GpuFitPlan)> = Vec::new();
    let mut fallback_jobs: Vec<usize> = Vec::new();
    let mut fallback_cause: BTreeMap<usize, FitFallbackCause> = BTreeMap::new();
    for (k, entry) in entries.into_iter().enumerate() {
        match entry {
            Ok(plan) => plans.push((k, plan)),
            Err(_) => fallback_jobs.push(k),
        }
    }
    let offered = plans.len() + fallback_jobs.len();
    if offered == 0 {
        return Ok(());
    }
    let fit_start = std::time::Instant::now();
    g.fit_attempts += offered;
    {
        let stats = g.stats.entry(generation).or_default();
        stats.generation = generation;
        stats.fit_attempts += offered;
    }
    for &k in &fallback_jobs {
        let (i, j) = (jobs[k].0, jobs[k].1);
        let cause = match crate::fit::gpu_fit_plan_reason(&jobs[k].2) {
            Err(crate::fit::GpuFitReject::TooManyConstants) => FitFallbackCause::TooManyConstants,
            _ => FitFallbackCause::UnsupportedPlan,
        };
        fallback_cause.insert(k, cause);
        g.note_fit_fallback(generation, cause);
        crate::fit::note_gpu_fit(false);
        results[i][j] = Some(PipelineResult {
            error: f64::NAN,
            offered: offered_hash[k],
            tree: None,
        });
    }

    let mut fitted: Vec<usize> = Vec::new();
    if !plans.is_empty() {
        let refs: Vec<&crate::fit::GpuFitPlan> = plans.iter().map(|(_, p)| p).collect();
        // #236: slope always starts from the squared warm start and keeps the
        // incoming constants unless it improves on them.
        g.ev.set_fit_polish(fit.polish);
        let fit_device_start = g.ledger_offset_s();
        let (out, ms) = if fit.loss == crate::fit::DeviceLoss::Slope {
            g.ev.slope_fit_batch(&refs, fit.iters)
        } else {
            g.ev.fit_batch(&refs, fit.iters, fit.loss)
        };
        if ms > 0.0 {
            g.fit_ms += ms as f64;
            g.fit_polish_ms += g.ev.take_polish_ms(ms);
            g.fit_calls += 1;
            g.fit_call_candidates += plans.len();
            g.fit_call_ms_max = g.fit_call_ms_max.max(ms as f64);
            // #600: the fit kernel is one device interval on the run clock.
            match g.ev.take_span_start_s(crate::gpu::SPAN_FIT) {
                Some(start_s) => g.ledger.record_fit_kernel_event(0, start_s, ms as f64 / 1000.0),
                None => g.ledger.record_fit_kernel_interval(
                    0,
                    fit_device_start,
                    ms as f64 / 1000.0,
                ),
            }
        }
        if ms < 0.0 || out.len() != plans.len() {
            for (k, _) in &plans {
                fallback_jobs.push(*k);
                let (i, j) = (jobs[*k].0, jobs[*k].1);
                fallback_cause.insert(*k, FitFallbackCause::Launch);
                g.note_fit_fallback(generation, FitFallbackCause::Launch);
                crate::fit::note_gpu_fit(false);
                results[i][j] = Some(PipelineResult {
                    error: f64::NAN,
                    offered: offered_hash[*k],
                    tree: None,
                });
            }
        } else {
            for ((k, plan), res) in plans.iter().zip(out) {
                let finite = res.c.iter().all(|v| v.is_finite());
                if crate::fit::fit_status_usable(res.status) && finite {
                    plan.apply(&mut jobs[*k].2, &res.c);
                    if let Some(label) = gpu_status_label(res.status) {
                        *g.fit_status_by_label.entry(label.to_string()).or_default() += 1;
                    }
                    g.fit_gpu += 1;
                    g.stats.entry(generation).or_default().fit_gpu += 1;
                    crate::fit::note_gpu_fit(true);
                    fitted.push(*k);
                } else {
                    let (i, j) = (jobs[*k].0, jobs[*k].1);
                    let cause = if !finite {
                        FitFallbackCause::NonFinite
                    } else {
                        status_fallback_cause(res.status)
                    };
                    fallback_cause.insert(*k, cause);
                    g.note_fit_fallback(generation, cause);
                    crate::fit::note_gpu_fit(false);
                    results[i][j] = Some(PipelineResult {
                        error: f64::NAN,
                        offered: offered_hash[*k],
                        tree: None,
                    });
                    fallback_jobs.push(*k);
                }
            }
        }
    }

    // The exact CPU arm keeps the same constants a synchronous run would have
    // produced. These are the structural/rare arithmetic residuals only; the
    // device answers above are never refit.
    let mut fb: Vec<(usize, FallbackFitJob)> = fallback_jobs
        .iter()
        .filter_map(|&k| {
            let (i, j) = (jobs[k].0, jobs[k].1);
            results[i][j]
                .as_ref()
                .is_some_and(|r| r.error.is_nan())
                .then(|| {
                    (k, FallbackFitJob::new(i, j, jobs[k].2.clone()))
                })
        })
        .collect();
    if !fb.is_empty() {
        let mut owned: Vec<FallbackFitJob> = fb
            .iter()
            .map(|(_, job)| FallbackFitJob::new(job.island, job.slot, job.tree.clone()))
            .collect();
        let fallback_start = std::time::Instant::now();
        fit_fallback_jobs(
            &mut owned,
            &train.x,
            &train.y,
            &train.w,
            fit.iters,
            fit.metric,
            fallback_workers(),
        );
        let fallback_seconds = fallback_start.elapsed().as_secs_f64();
        g.fit_fallback_ms += fallback_seconds * 1000.0;
        g.stats.entry(generation).or_default().fit_fallback_seconds += fallback_seconds;
        for ((k, job), done) in fb.iter_mut().zip(owned) {
            if done.improved {
                let cause = fallback_cause
                    .get(k)
                    .copied()
                    .unwrap_or(FitFallbackCause::UnsupportedPlan);
                g.note_fit_fallback_improved(generation, cause);
            }
            job.tree = done.tree.clone();
            jobs[*k].2 = done.tree;
            fitted.push(*k);
        }
    }

    let fit_ms_elapsed = fit_start.elapsed().as_secs_f64() * 1000.0;
    g.fit_host_ms += fit_ms_elapsed;
    g.drain_fit_ms += fit_ms_elapsed;
    if !fitted.is_empty() {
        // #560: score the fitted tree after the post-fit simplify, as the
        // synchronous path does, and hand that tree to the barrier.
        let refit: Vec<PipelineJob> = fitted
            .iter()
            .map(|&k| {
                let (i, j, tree) = &jobs[k];
                (*i, *j, expr::simplify(tree))
            })
            .collect();
        let rescore_start = std::time::Instant::now();
        let (kept2, errs2) = eval_pipeline_jobs(g, &refit, generation)?;
        g.drain_eval_ms += rescore_start.elapsed().as_secs_f64() * 1000.0;
        for (&pos, &err) in kept2.iter().zip(&errs2) {
            let (i, j, tree) = &refit[pos];
            results[*i][*j] = Some(PipelineResult {
                error: err,
                offered: offered_hash[fitted[pos]],
                tree: Some(tree.clone()),
            });
        }
    }
    Ok(())
}

/// #26: whether a run may use the GPU at all.
///
/// Restricted to the single-output, full-sample path with a **native device
/// objective**: multi-output and implicit objectives need more than one
/// prediction per row, `--predictor-samples` changes the rows every generation,
/// and every metric without a native kernel objective would be ranked by a
/// proxy it is not defined on. Each of those keeps the CPU evaluator.
///
/// The native scorers carry the per-row weights, so `--weight` is admitted for
/// every native metric with a weighted form (`metrics::weights_supported`):
/// the regression family and the classification objectives. `maximum` refuses
/// weights.
///
/// #288: the order/rank family and `auc` are `ScoreDispatch::Exact` rows scored
/// by the plan batch entry point; this gate admits them from their measured row
/// floor. Below the floor the **fit** stage is admitted separately by
/// [`rank_fit_only`], which creates a fit-only evaluator (`native_metric =
/// false`) whose every fitness value is scored by the CPU.
#[cfg(feature = "cuda")]
fn gpu_blocker(cfg: &Config, train: &Sample) -> Option<&'static str> {
    if !cfg.fitness.is_metric() {
        return long_pnl_blocker(cfg, train); // #481
    }
    if train.is_multi() {
        return Some("multi-output objective");
    }
    // Weighted rows run on the device only through a native scorer that has a
    // weighted form; every other metric keeps the exact CPU evaluator rather
    // than dropping the weights.
    if !train.w.is_empty() {
        match crate::metrics::native_score_objective(cfg.metric) {
            Some(crate::metrics::ScoreObjective::Maximum) => {
                return Some("weighted rows: maximum has no weighted form");
            }
            Some(_) if crate::metrics::weights_supported(cfg.metric) => {}
            _ => return Some("weighted rows need a native scorer with a weighted form"),
        }
    }
    // #236/#287/#288: a metric may use the device only when the score-dispatch
    // table names a native statistic for it; otherwise the *named* refusal is
    // the reason, and no run can be silently ranked by MSE.
    if let Some(reason) = crate::metrics::score_refusal(cfg.metric) {
        return Some(reason);
    }
    if cfg.metric == crate::metrics::Metric::ImplicitResidual
        && train.y.iter().any(|&v| v != 0.0)
    {
        return Some("implicit_residual needs an all-zero training target");
    }
    // #236: the device differences row r against row r - 1 of the resident
    // sample. A pair list with any other shape (a shuffled split, a group
    // boundary, a row missing from the side) keeps the exact CPU scorer.
    if cfg.metric == crate::metrics::Metric::Slope && slope_pair_mask(train).is_none() {
        return Some("slope pairs are not the consecutive rows of the training sample");
    }
    // A fit subsample keeps only the pairs whose two rows it drew, and those
    // are not its consecutive rows.
    if cfg.metric == crate::metrics::Metric::Slope && effective_fit_rows(cfg, train) > 0 {
        return Some("slope pairs are not the consecutive rows of a --fit-rows/--fit-samples subsample");
    }
    // #288: AUC needs exactly two distinct target values; the label rule is
    // the CPU midpoint rule, so any other target is refused, never collapsed.
    if cfg.metric == crate::metrics::Metric::Auc
        && crate::metrics::classification_target_error(&train.y).is_some()
    {
        return Some("auc needs a two-valued target");
    }
    // #288: below the measured crossover the exact CPU statistic is cheaper
    // than the device launch and sort. The run keeps the CPU scorer (and may
    // still fit constants on the device through the fit-only arm); the reason
    // is named, never silent.
    if let Some(stat) = crate::metrics::ScoreStat::for_metric(cfg.metric) {
        if train.len() < stat.min_rows() {
            return Some(stat.small_sample_reason());
        }
    }
    if subsampling_active(cfg, train) {
        return Some("--predictor-samples changes the rows per generation");
    }
    // #85/#116: the CPU screened path scores offspring on a per-generation
    // subset, which the resident full-sample GPU evaluator cannot represent.
    // The device fit arm instead fits on a fixed subsample and scores every
    // row, so only a run without that arm is refused.
    if effective_fit_rows(cfg, train) > 0 && !defers_fit_to_gpu(cfg, true) {
        return Some(
            "--fit-samples/--fit-rows scores on all rows only with the device fit arm (--fit-strategy gpu)",
        );
    }
    None
}

/// #481: the objective the device ranks candidates by: the long/flat regret
/// under `--fitness pnl`, else the metric's native objective. Every device
/// scoring path (`GpuSearch::score`, the overlap pipeline, the flat postfix
/// arena) reads it through `GpuSearch::score`.
#[cfg(feature = "cuda")]
fn search_score_objective(cfg: &Config) -> Option<crate::metrics::ScoreObjective> {
    match cfg.fitness {
        Fitness::LongPnl(_) => Some(crate::metrics::ScoreObjective::LongPnl),
        Fitness::SavedLongPnl(_) => None,
        Fitness::Metric => crate::metrics::native_score_objective(cfg.metric),
    }
}

/// #481: [`gpu_blocker`] under `--fitness pnl`. The device scores the
/// long/flat regret itself and `--metric` only fits constants, so the metric's
/// score refusals do not apply; what remains are the sample shapes the
/// resident book cannot represent and the fit arm's own row contract.
#[cfg(feature = "cuda")]
fn long_pnl_blocker(cfg: &Config, train: &Sample) -> Option<&'static str> {
    let Some(book) = cfg.fitness.book() else {
        return Some("the long_pnl fitness book was not rebuilt");
    };
    if train.is_multi() {
        return Some("multi-output objective");
    }
    if book.len() != train.len() {
        return Some("the long_pnl fitness book does not cover the training rows");
    }
    if subsampling_active(cfg, train) {
        return Some("--predictor-samples changes the rows per generation");
    }
    if effective_fit_rows(cfg, train) > 0 && !defers_fit_to_gpu(cfg, true) {
        return Some(
            "--fit-samples/--fit-rows scores on all rows only with the device fit arm (--fit-strategy gpu)",
        );
    }
    if cfg.metric == crate::metrics::Metric::Slope
        && defers_fit_to_gpu(cfg, true)
        && (slope_pair_mask(train).is_none() || effective_fit_rows(cfg, train) > 0)
    {
        return Some("slope pairs are not the consecutive rows of the device fit sample");
    }
    None
}

/// #508: the slope pair mask of a training sample (see [`slope_pair_mask`]).
#[cfg(feature = "cuda")]
pub(crate) struct SlopePairMask {
    /// `ceil(n / 32)` words, bit `r` set when row `r` pairs with row `r - 1`.
    /// Empty when every row `r >= 1` pairs with its predecessor.
    pub(crate) bits: Vec<u32>,
    pub(crate) pairs: usize,
}

/// #236/#508: the device differences row `r` against row `r - 1` of the
/// resident sample, so the sample's slope pairs must be a set of `[k - 1, k]`
/// pairs with distinct `k` in `1..n`. A missing pair list is the full
/// consecutive set (empty mask). A pair list of any other shape (a shuffled
/// split, a repeated, backward or non-adjacent pair) is `None` and keeps the
/// exact CPU scorer.
#[cfg(feature = "cuda")]
pub(crate) fn slope_pair_mask(train: &Sample) -> Option<SlopePairMask> {
    let Some(pairs) = &train.pairs else {
        return Some(SlopePairMask { bits: Vec::new(), pairs: train.len().saturating_sub(1) });
    };
    let n = train.len();
    let mut bits = vec![0u32; n.div_ceil(32)];
    for &[a, b] in pairs {
        let (a, b) = (a as usize, b as usize);
        if b == 0 || b >= n || a + 1 != b || bits[b / 32] >> (b % 32) & 1 == 1 {
            return None;
        }
        bits[b / 32] |= 1 << (b % 32);
    }
    if n >= 2 && pairs.len() == n - 1 {
        bits.clear();
    }
    Some(SlopePairMask { bits, pairs: pairs.len() })
}

/// #230: `Some(signed)` when the device may run this run's **fit stage only**,
/// even though [`gpu_blocker`] refuses its *scoring* objective.
///
/// The rank family has an exact device fit (squared LM start plus the batched
/// integer rank polish) and no native device scorer yet, so this returns
/// `Some` only when every other blocker already allows the resident full-sample
/// path. The evaluator it authorises has `native_metric = false`: the CPU scores
/// every candidate, and the backend report stays `cpu` with the open-scorer
/// reason.
#[cfg(feature = "cuda")]
fn rank_fit_only_arm(cfg: &Config, train: &Sample) -> Option<bool> {
    let signed = crate::fit::rank_fit_only(cfg.metric)?;
    if !cfg.fit_strategy.defers_to_gpu() || cfg.fit_iters == 0 {
        return None;
    }
    if train.is_multi()
        || !train.w.is_empty()
        || cfg.metric.is_implicit()
        || cfg.metric.is_classification()
        || cfg.metric.is_parameter_aware()
        || subsampling_active(cfg, train)
    {
        return None;
    }
    Some(signed)
}

/// #226/#232: `Some(reason)` when a `--metric median` or `--metric
/// interquartile` run with `--fit-strategy gpu` must keep the exact CPU fit arm
/// for this run.
///
/// The device order-statistic polish is a **fit-only** arm (`native_metric =
/// false`: the evaluator cannot score a median or an interquartile mean), so it
/// reuses the rank arm's gates and adds one measured cost-model check per
/// metric: below [`crate::fit::MEDIAN_GPU_MIN_ROWS`] (median) or
/// [`crate::fit::INTERQUARTILE_GPU_MIN_ROWS`] (interquartile) the per-call
/// latency of the batched selection is not repaid by a small sample, and the
/// CPU polish stays. The reason is recorded in the backend note and the run's
/// `backend_reason`, so a CPU fallback here is labelled, never silent. `None`
/// for a run that is not an order-statistic GPU-fit run at all.
#[cfg(feature = "cuda")]
fn order_stat_fit_only_blocker(cfg: &Config, train: &Sample) -> Option<&'static str> {
    let want = match crate::fit::OrderStat::for_metric(cfg.metric) {
        Some(want) if cfg.fit_strategy.defers_to_gpu() && cfg.fit_iters > 0 => want,
        _ => return None,
    };
    if train.is_multi() {
        return Some("multi-output objective");
    }
    if !train.w.is_empty() {
        return Some("weighted rows");
    }
    if subsampling_active(cfg, train) {
        return Some("--predictor-samples changes the rows per generation");
    }
    let fit_n = device_fit_rows(cfg, train);
    let (worthwhile, why) = match want {
        crate::fit::OrderStat::Median => (
            crate::fit::median_gpu_fit_worthwhile(fit_n, 1, cfg.fit_iters),
            "median fit arm: the sample is too small for the device polish",
        ),
        crate::fit::OrderStat::Interquartile => (
            crate::fit::interquartile_gpu_fit_worthwhile(fit_n, 1, cfg.fit_iters),
            "interquartile fit arm: the sample is too small for the device polish",
        ),
    };
    if !worthwhile {
        return Some(why);
    }
    None
}

/// #227: `Some(reason)` when a `--metric maximum` run with
/// `--fit-strategy gpu` must keep the exact CPU fit arm for this run.
///
/// The device maximum-error polish is a **fit-only** arm (`native_metric =
/// false`: the evaluator cannot score a maximum), so it reuses the rank and
/// order-statistic gates and adds one measured cost-model check: below
/// [`crate::fit::MAXIMUM_GPU_MIN_ROWS`] the per-call latency of the bisection
/// on the error level is not repaid by a small sample, and the CPU polish
/// stays. The reason is recorded in the backend note and the run's
/// `backend_reason`, so this CPU fallback is labelled, never silent. `None`
/// for a run that is not a maximum GPU-fit run at all.
#[cfg(feature = "cuda")]
fn maximum_fit_only_blocker(cfg: &Config, train: &Sample) -> Option<&'static str> {
    if !crate::fit::maximum_fit_only(cfg.metric)
        || !cfg.fit_strategy.defers_to_gpu()
        || cfg.fit_iters == 0
    {
        return None;
    }
    if train.is_multi() {
        return Some("multi-output objective");
    }
    if !train.w.is_empty() {
        return Some("weighted rows");
    }
    if subsampling_active(cfg, train) {
        return Some("--predictor-samples changes the rows per generation");
    }
    if !crate::fit::maximum_gpu_fit_worthwhile(device_fit_rows(cfg, train), 1, cfg.fit_iters) {
        return Some("maximum fit arm: the sample is too small for the device polish");
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

/// #236: the labelled cause a requested GPU run will actually be scored on the
/// CPU, or `None` when the device is usable (or `--gpu` was not requested).
///
/// This is the same decision [`effective_backend`] makes. The CLI prints it in
/// the resource line and records it in the development log, so a run whose
/// objective the kernel cannot score never claims the GPU in its own evidence
/// record.
pub fn backend_blocker_reason(cfg: &Config, train: &Sample) -> Option<&'static str> {
    #[cfg(feature = "cuda")]
    {
        if cfg.gpu
            && !matches!(
                std::env::var("RCSR_GPU").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no")
            )
        {
            return gpu_blocker(cfg, train);
        }
    }
    let _ = (cfg, train);
    None
}

/// #230/#234/#288: which device shapes this run can use. One decision shared
/// by the evaluator construction and the memory plan.
#[cfg(feature = "cuda")]
struct GpuPlanShape {
    score_stat: Option<crate::metrics::ScoreStat>,
    /// The exact order/rank scorer runs on the device.
    score_admitted: bool,
    /// The device moves constants only; the CPU scores.
    fit_only: bool,
    logistic_fit: bool,
}

#[cfg(feature = "cuda")]
fn gpu_plan_shape(cfg: &Config, train: &Sample) -> GpuPlanShape {
    let order_stat_blocker = order_stat_fit_only_blocker(cfg, train);
    // #227: `--metric maximum` uses the same fit-only shape with the batched
    // exact maximum-error polish and its own measured cost-model gate.
    let maximum_blocker = maximum_fit_only_blocker(cfg, train);
    // `--no-fit` (or a non-gpu strategy) has nothing to fit on the device,
    // so the fit-only evaluator must not be created for it.
    let order_stat_fit = cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && crate::fit::order_stat_fit_only(cfg.metric);
    // #234: `log_loss` has a native *fit* objective (the batched LM
    // kernel) and no native fitness, so it uses the same fit-only shape:
    // the device moves constants, the exact CPU log-loss scores.
    let logistic_fit = cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && crate::fit::fit_only_loss(cfg.metric).is_some();
    let maximum_fit = cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && crate::fit::maximum_fit_only(cfg.metric);
    // #288: an exact order-statistic/rank metric has a native *scoring*
    // entry point named by the #287 dispatch table, so it is admitted
    // through the same gate as the objective kernel and is not fit-only.
    // #481: under the long/flat fitness the metric only fits, so no exact
    // statistic scores the run.
    let score_stat = crate::metrics::ScoreStat::for_metric(cfg.metric)
        .filter(|_| cfg.fitness.is_metric());
    let score_admitted = score_stat.is_some() && gpu_blocker(cfg, train).is_none();
    // #287: maximum has a native objective, so it is never fit-only.
    let maximum_fit_only = maximum_fit
        && maximum_blocker.is_none()
        && crate::metrics::native_score_objective(cfg.metric).is_none();
    let fit_only = !score_admitted
        && cfg.fitness.is_metric() // #480: the CPU scores the long_pnl fitness
        && (rank_fit_only_arm(cfg, train).is_some()
            || (order_stat_fit && order_stat_blocker.is_none())
            || logistic_fit
            || maximum_fit_only);
    GpuPlanShape {
        score_stat,
        score_admitted,
        fit_only,
        logistic_fit,
    }
}

/// #293: the device memory plan for this run, or `None` when the run does not
/// build a device evaluator or the device cannot be queried.
#[cfg(feature = "cuda")]
pub fn device_plan_for(cfg: &Config, train: &Sample) -> Option<crate::memory::DevicePlan> {
    if !cfg.gpu
        || matches!(
            std::env::var("RCSR_GPU").as_deref(),
            Ok("0") | Ok("false") | Ok("off") | Ok("no")
        )
    {
        return None;
    }
    let shape = gpu_plan_shape(cfg, train);
    if !(shape.score_admitted || shape.fit_only || gpu_blocker(cfg, train).is_none()) {
        return None;
    }
    let n_vars = train.x.first().map(|r| r.len()).unwrap_or(0);
    if train.is_empty() || n_vars == 0 {
        return None;
    }
    let info = crate::gpu::device_info()?;
    let device_fit = cfg.fit_strategy.defers_to_gpu() && cfg.fit_iters > 0;
    // #481: the long/flat scorer reads its book, not the label sign.
    let long_pnl = !cfg.fitness.is_metric();
    let label_sign = (!long_pnl && !shape.score_admitted && !shape.fit_only) || shape.logistic_fit;
    let workload = crate::memory::DeviceWorkload {
        rows: train.len(),
        vars: n_vars,
        candidates: cfg.pop.saturating_mul(cfg.islands.max(1)),
        program_words: crate::genome::width_for(cfg.max_complexity),
        device_fit,
        fit_rows: if device_fit { effective_fit_rows(cfg, train) } else { 0 },
        stat: shape.score_admitted,
        weights: !train.w.is_empty(),
        label_sign,
        pnl_book: long_pnl,
    };
    let caps = crate::memory::DeviceCaps::from_env(cfg.gpu_memory);
    Some(crate::memory::device_plan(&workload, &info, &caps))
}

/// #223: resolve the fit arm of a run that did not name one (`Unset`) or named
/// `auto`. Run once before generation 0, never on a resume. The CPU arm it
/// keeps when the device is not admissible is the arm the request names, so a
/// CPU-only run is unchanged.
pub fn resolve_fit_strategy(
    cfg: &Config,
    train: &Sample,
    request: crate::fit::FitRequest,
) -> crate::fit::FitResolution {
    // The device arm is what the run would switch to, so a fit subsample is
    // judged as the device arm fits it.
    let mut probe = cfg.clone();
    probe.fit_strategy = crate::fit::FitStrategy::Gpu;
    let backend_ok = if effective_backend(&probe, train) == Backend::Gpu {
        Ok(())
    } else if !cfg.gpu {
        Err("--cpu/--no-gpu".to_string())
    } else if let Some(why) = backend_blocker_reason(&probe, train) {
        Err(why.to_string())
    } else if cfg!(feature = "cuda") {
        Err("no usable CUDA device".to_string())
    } else {
        Err("built without --features cuda".to_string())
    };
    let memory_ok = fit_memory_check(cfg, train);
    let cost_ok = fit_cost_check(cfg);
    crate::fit::resolve_fit_arm(
        request,
        cfg.metric,
        cfg.fit_iters,
        backend_ok,
        memory_ok,
        cost_ok,
    )
}

#[cfg(feature = "cuda")]
fn fit_memory_check(cfg: &Config, train: &Sample) -> Result<(), String> {
    let mut probe = cfg.clone();
    probe.fit_strategy = crate::fit::FitStrategy::Gpu;
    match device_plan_for(&probe, train) {
        Some(plan) if plan.refusal.as_ref().map_or(false, |r| r.stage == "fit") => {
            Err("device_memory".to_string())
        }
        _ => Ok(()),
    }
}

#[cfg(not(feature = "cuda"))]
fn fit_memory_check(_cfg: &Config, _train: &Sample) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "cuda")]
fn fit_cost_check(cfg: &Config) -> bool {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let full = crate::fit::gpu_fit_op_coverage(&cfg.ops, !expr::current_blocks().is_empty());
    crate::fit::ls_gpu_fit_worthwhile(cfg.islands.max(1), cores, fallback_workers(), full)
}

#[cfg(not(feature = "cuda"))]
fn fit_cost_check(_cfg: &Config) -> bool {
    false
}

/// #293: why a device plan cannot run, or `None` when it can. A stat scorer
/// routed to the CPU leaves the run without a device scorer.
#[cfg(feature = "cuda")]
fn plan_blocker(plan: &crate::memory::DevicePlan) -> Option<String> {
    if let Some(reason) = plan.refusal_reason() {
        return Some(reason);
    }
    if plan.workload.stat && plan.stat_route != crate::memory::StageRoute::Device {
        return Some(format!(
            "device_memory: the exact {} scorer needs {} bytes of device memory, budget {} bytes{}",
            "order/rank",
            (plan.workload.rows as u64)
                .saturating_mul(plan.workload.vars as u64 + 1)
                .saturating_mul(4)
                .saturating_add(plan.terms.side_vectors + plan.terms.buffers),
            plan.budget,
            if plan.mode == crate::memory::Residency::Tiled {
                format!("; the rows stream in tiles of {}, so the scorer runs on the CPU", plan.tile_rows)
            } else {
                String::new()
            }
        ));
    }
    None
}

#[cfg(feature = "cuda")]
fn device_refusal(what: &str) -> String {
    match crate::gpu::last_alloc_failure() {
        Some(bytes) => format!(
            "device_memory: the device refused an allocation of {} bytes{} for {}",
            bytes,
            alloc_tag_note(),
            what
        ),
        None => format!("the device refused {}", what),
    }
}

/// ` (tag cols_f64)` for the most recent refused allocation, or nothing.
#[cfg(feature = "cuda")]
fn alloc_tag_note() -> String {
    match crate::gpu::last_alloc_failure_tag() {
        Some(tag) => format!(" (tag {})", tag),
        None => String::new(),
    }
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

/// #480: [`eval_error_and_descriptor`] under the configured fitness. The long/flat
/// regret reads the same prediction vector the descriptor does.
fn fitness_error_and_descriptor(
    tree: &Node,
    data: &EvalSample,
    cfg: &Config,
    want_descriptor: bool,
) -> (f64, Option<qd::Descriptor>) {
    let Some(book) = cfg.fitness.book() else {
        return eval_error_and_descriptor(tree, data, cfg.metric, want_descriptor);
    };
    let s = data.sample;
    if s.is_empty() || s.len() != book.len() {
        debug_assert!(false, "fitness book has {} rows, the sample {}", book.len(), s.len());
        return (f64::INFINITY, None);
    }
    let nvars = s.x.first().map(Vec::len).unwrap_or(0);
    with_predictions(tree, data, |pred| {
        let error = book.regret(pred);
        let descriptor = want_descriptor.then(|| qd::descriptor_from_prediction(tree, pred, nvars));
        (error, descriptor)
    })
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
    // The parameter count lets the information criteria (AIC/AICc/BIC) work as
    // search metrics; it is ignored by every other metric. The weight vector is
    // empty for an unweighted sample, which takes the identical unweighted path.
    with_predictions(tree, data, |pred| {
        let error = crate::metrics::score_sample(pred, s, metric, expr::count_constants(tree));
        let descriptor = want_descriptor.then(|| qd::descriptor_from_prediction(tree, pred, nvars));
        (error, descriptor)
    })
}

/// #500: rows probed for a non-finite prediction before a full re-score.
const INF_PROBE_ROWS: usize = 2048;

/// #500: whether the first rows already hold a non-finite prediction, which
/// makes the full score `+inf` for the plain metrics. Shapes whose score does
/// not follow that rule (implicit, weighted, multi-output, descriptor, P&L
/// book) report `false` and take the full pass.
fn prefix_is_non_finite(tree: &Node, data: &EvalSample, cfg: &Config) -> bool {
    let s = data.sample;
    if s.is_empty()
        || s.is_multi()
        || !s.w.is_empty()
        || cfg.metric.is_implicit()
        || cfg.qd_bins > 0
        || cfg.fitness.book().is_some()
    {
        return false;
    }
    let rows = s.x.len().min(INF_PROBE_ROWS);
    if let Some(cols) = &data.cols {
        if let Some(program) = crate::eval_soa::compile(tree) {
            let mut pred = vec![0.0; rows];
            crate::eval_soa::eval_batch(&program, cols, rows, &mut pred);
            return pred.iter().any(|p| !p.is_finite());
        }
    }
    s.x.iter().take(rows).any(|row| !eval(tree, row).is_finite())
}

/// #475: one candidate's prediction vector, owned.
///
/// The fast path compiles the tree once and evaluates all rows in tiles over
/// the sample's cached SoA columns. An unsupported operator anywhere in the
/// tree makes `compile` return `None`, and this call then takes the scalar
/// walk, transparently, for that tree only.
fn predictions(tree: &Node, data: &EvalSample) -> Vec<f64> {
    let mut buf = PredBuf::default();
    buf.fill(tree, data);
    buf.pred
}

/// #475: the prediction buffer and the evaluator scratch of one worker.
#[derive(Default)]
struct PredBuf {
    pred: Vec<f64>,
    scratch: crate::eval_soa::Scratch,
}

impl PredBuf {
    fn fill(&mut self, tree: &Node, data: &EvalSample) {
        let rows = data.sample.x.len();
        #[cfg(test)]
        if self.pred.capacity() < rows {
            PRED_ALLOCS.with(|n| n.set(n.get() + 1));
        }
        if let Some(cols) = &data.cols {
            if let Some(program) = crate::eval_soa::compile(tree) {
                self.pred.clear();
                self.pred.resize(rows, 0.0);
                crate::eval_soa::eval_batch_into(&program, cols, rows, &mut self.pred, &mut self.scratch);
                return;
            }
        }
        self.pred.clear();
        self.pred.extend(data.sample.x.iter().map(|row| eval(tree, row)));
    }
}

thread_local! {
    static PRED_BUF: std::cell::RefCell<PredBuf> = std::cell::RefCell::new(PredBuf::default());
}

#[cfg(test)]
thread_local! {
    static PRED_ALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// #475: run `f` on a candidate's predictions, held in this thread's reusable
/// buffer. A nested call on the same thread gets its own buffer.
fn with_predictions<R>(tree: &Node, data: &EvalSample, f: impl FnOnce(&[f64]) -> R) -> R {
    PRED_BUF.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            buf.fill(tree, data);
            f(&buf.pred)
        }
        Err(_) => f(&predictions(tree, data)),
    })
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
    // #470: start the map of a wrapped tree at the closed-form least squares
    // on the fit rows; the configured fitter then refines every constant.
    if cfg.linear_scaling && cfg.metric.is_least_squares() {
        crate::scale::restart(tree, &fit.x, &fit.y, &fit.w);
    }
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
        if let (Metric::Slope, Some(pairs)) = (cfg.metric, &fit.pairs) {
            let sse = crate::fit::fit_slope_pairs(tree, &fit.x, &fit.y, pairs, iters);
            return Some(FitResult {
                sse,
                hash: expr::structural_hash(tree),
                same_rows: std::ptr::eq(fit, score),
            });
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
        && cfg.fitness.is_metric() // #480: the SSE is not the long/flat regret
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
        fitness_error_and_descriptor(tree, data, cfg, false).0
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
        pairs: train.pairs.as_ref().map(|pairs| {
            let mut at = vec![u32::MAX; train.len()];
            for (k, &i) in idx.iter().enumerate() {
                at[i] = k as u32;
            }
            pairs
                .iter()
                .filter_map(|&[a, b]| {
                    let (a, b) = (at[a as usize], at[b as usize]);
                    (a != u32::MAX && b != u32::MAX).then_some([a, b])
                })
                .collect()
        }),
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

/// Rows the device fit arm sees: the fixed fit subsample when one is requested,
/// every training row otherwise. Scoring always uses every row.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn device_fit_rows(cfg: &Config, sample: &Sample) -> usize {
    match effective_fit_rows(cfg, sample) {
        0 => sample.len(),
        n => n,
    }
}

/// True when a requested fit subsample is fitted on the device (fixed rows,
/// uploaded once) while the evaluator scores every row.
fn fit_subsample_on_device(cfg: &Config, sample: &Sample) -> bool {
    effective_fit_rows(cfg, sample) > 0 && defers_fit_to_gpu(cfg, true)
}

/// The fixed row indices of the device fit subsample, ascending.
fn device_fit_indices(cfg: &Config, sample: &Sample) -> Vec<usize> {
    generation_subset(
        sample.len(),
        effective_fit_rows(cfg, sample),
        cfg.seed ^ FIT_SUBSET_SALT,
        0,
    )
}

/// FNV-1a over the little-endian `u64` row indices.
fn subsample_hash(indices: &[usize]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &i in indices {
        for b in (i as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The run's fit subsample as `(rows, hash, fixed)`, or `None` for a full fit.
/// `fixed` is true when the device arm fits the same rows for the whole run.
/// The CPU screened path draws a fresh subset each generation, so its hash is
/// that of generation 0.
pub fn fit_subsample_record(cfg: &Config, sample: &Sample) -> Option<(usize, u64, bool)> {
    let rows = effective_fit_rows(cfg, sample);
    if rows == 0 {
        return None;
    }
    let fixed =
        effective_backend(cfg, sample) == Backend::Gpu && fit_subsample_on_device(cfg, sample);
    let draw = device_fit_indices(cfg, sample);
    Some((rows, subsample_hash(&draw), fixed))
}

/// #169: why an admitted candidate still needed the exact CPU fit arm.
///
/// The labels are stable: they appear in `gpu.fit.fallback_by_cause`, in each
/// generation's ledger and in `--fit-report`/docs. Structural rejections are
/// grouped into `unsupported_plan` except the explicit `p > 20` cap, which has
/// its own label because it is the only limit the kernel could lift.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitFallbackCause {
    /// No analytic GPU derivative, too many nodes, shared block constants, or
    /// no constants at all.
    UnsupportedPlan,
    /// Fewer than `32p` rows, so the normal-equation system is underdetermined.
    TooFewRows,
    /// More than [`crate::fit::GPU_FIT_MAX_CONSTANTS`] constants (`p > 20`).
    TooManyConstants,
    /// #229/#222/#234: the metric has no native device fit objective (only
    /// `squared`, `log_squashed`, `absolute` and the fit-only `log_loss` do),
    /// so the exact CPU arm owns the fit.
    MetricNotNative,
    /// The kernel's conditioning gate rejected the normal equations (status 1).
    IllConditioned,
    /// A non-finite value, derivative or residual (status 2).
    NonFinite,
    /// The CUDA fit launch failed; the whole batch went to the CPU.
    Launch,
    /// #230: the device rank stage declined (allocation, launch or an
    /// inconsistent score) and the exact CPU rank arm took over.
    RankDeclined,
    /// #226: the device median stage declined (allocation, launch or an
    /// inconsistent score) and the exact CPU median arm took over.
    MedianDeclined,
    /// #232: the device interquartile stage declined (allocation, launch or an
    /// inconsistent score) and the exact CPU interquartile arm took over.
    InterquartileDeclined,
    /// #234: the fit sample holds one class, so the unconstrained log-loss has
    /// no finite minimiser. The exact CPU arm keeps the candidate rather than
    /// fitting a sign no row supports.
    SingleClass,
    /// #234: the kernel's log-loss constants scored strictly worse than the
    /// starting constants in exact `f64` `log_loss`, so the start was kept.
    LogLossWorse,
    /// #393: the exact gate judged the device proposal on the configured metric
    /// and it scored worse (or non-finite) than the starting constants, so the
    /// start was kept and the exact CPU arm refits.
    Worse,
    /// #393: `signed` only. No constant can move the mean, so the proposal has
    /// no justification on the bias; the exact CPU arm refits.
    NotIdentifiable,
    /// A correlation tree that is affine in every constant has a global closed
    /// form (weighted least squares). It is a route, not a failure: the CPU arm
    /// solves it directly and the device never sees it.
    AffineClosedForm,
    /// #227: the device maximum-error stage declined (allocation, launch or an
    /// inconsistent scan) and the exact CPU maximum arm took over.
    MaximumDeclined,
    /// #290: the measured dispatch model predicted the CPU as cheaper for this
    /// candidate's fit shape. This is a policy route, not an arithmetic
    /// failure, and it carries its own label so it is never confused with
    /// `ill_conditioned` or `non_finite`.
    DispatchCost,
}

impl FitFallbackCause {
    /// Every cause, in reporting order.
    pub const ALL: [FitFallbackCause; 17] = [
        FitFallbackCause::UnsupportedPlan,
        FitFallbackCause::TooFewRows,
        FitFallbackCause::TooManyConstants,
        FitFallbackCause::MetricNotNative,
        FitFallbackCause::IllConditioned,
        FitFallbackCause::NonFinite,
        FitFallbackCause::Launch,
        FitFallbackCause::RankDeclined,
        FitFallbackCause::MedianDeclined,
        FitFallbackCause::InterquartileDeclined,
        FitFallbackCause::SingleClass,
        FitFallbackCause::LogLossWorse,
        FitFallbackCause::MaximumDeclined,
        FitFallbackCause::DispatchCost,
        FitFallbackCause::Worse,
        FitFallbackCause::NotIdentifiable,
        FitFallbackCause::AffineClosedForm,
    ];

    /// Stable JSON label.
    pub fn label(self) -> &'static str {
        match self {
            FitFallbackCause::UnsupportedPlan => "unsupported_plan",
            FitFallbackCause::TooFewRows => "too_few_rows",
            FitFallbackCause::TooManyConstants => "too_many_constants",
            FitFallbackCause::MetricNotNative => "metric_not_native",
            FitFallbackCause::IllConditioned => "ill_conditioned",
            FitFallbackCause::NonFinite => "non_finite",
            FitFallbackCause::Launch => "launch",
            FitFallbackCause::RankDeclined => "rank_declined",
            FitFallbackCause::MedianDeclined => "median_declined",
            FitFallbackCause::InterquartileDeclined => "interquartile_declined",
            FitFallbackCause::SingleClass => "single_class",
            FitFallbackCause::LogLossWorse => "log_loss_worse",
            FitFallbackCause::MaximumDeclined => "maximum_declined",
            FitFallbackCause::DispatchCost => "dispatch_cost",
            FitFallbackCause::Worse => "worse",
            FitFallbackCause::NotIdentifiable => "not_identifiable",
            FitFallbackCause::AffineClosedForm => "affine_closed_form",
        }
    }
}

/// #453: the outcome of planning one fit batch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct FitBatch {
    /// Candidates the device will fit together.
    jobs: Vec<(usize, usize, crate::fit::GpuFitPlan)>,
    /// The batch decision; `None` when no candidate passed the row floor.
    decision: Option<crate::dispatch::Decision>,
    /// Workload bytes of the batch decision.
    bytes: u64,
    /// Candidates refused, with the cause each reports.
    refused: Vec<(usize, usize, FitFallbackCause)>,
}

/// #453: plan the fit stage once for the whole batch.
///
/// Each candidate keeps its structural row floor (`32 * nconst` rows). The
/// candidates that pass it form one workload of `W` fits with the largest
/// constant count, because the device runs them as one launch with one
/// transfer. The decision applies to every candidate in the batch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn plan_fit_batch(
    dispatcher: &mut crate::dispatch::Dispatcher,
    rows: usize,
    sweeps: usize,
    loss: Option<crate::fit::DeviceLoss>,
    plans: Vec<(usize, usize, crate::fit::GpuFitPlan)>,
) -> FitBatch {
    let mut refused = Vec::new();
    let mut valid = Vec::with_capacity(plans.len());
    for (i, j, plan) in plans {
        if rows < 32 * plan.nconst {
            refused.push((i, j, FitFallbackCause::TooFewRows));
        } else {
            valid.push((i, j, plan));
        }
    }
    if valid.is_empty() {
        return FitBatch { jobs: valid, decision: None, bytes: 0, refused };
    }
    let widest = valid.iter().map(|(_, _, p)| p.nconst).max().unwrap_or(1);
    let workload = crate::dispatch::Workload::fit(valid.len(), rows, widest, sweeps)
        .with_min_rows(32 * widest);
    let workload = match loss {
        Some(loss) => workload.with_objective(loss),
        None => workload,
    };
    let decision = dispatcher.plan(crate::dispatch::Stage::Fit, &workload);
    if decision.route == crate::dispatch::Route::Device {
        return FitBatch { jobs: valid, decision: Some(decision), bytes: workload.bytes, refused };
    }
    // A cost-model refusal is a policy route, not an arithmetic fallback:
    // label it `dispatch_cost`.
    let cause = match decision.reason {
        crate::dispatch::Reason::TooFewRows => FitFallbackCause::TooFewRows,
        crate::dispatch::Reason::TooManyConstants => FitFallbackCause::TooManyConstants,
        crate::dispatch::Reason::UnsupportedPlan => FitFallbackCause::UnsupportedPlan,
        _ => FitFallbackCause::DispatchCost,
    };
    refused.extend(valid.iter().map(|(i, j, _)| (*i, *j, cause)));
    dispatcher.record(decision, 0.0, valid.len(), workload.bytes);
    FitBatch { jobs: Vec::new(), decision: Some(decision), bytes: workload.bytes, refused }
}

/// #169: map a structural fit-plan rejection to its reported cause.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn plan_fallback_cause(reason: crate::fit::GpuFitReject) -> FitFallbackCause {
    match reason {
        crate::fit::GpuFitReject::TooManyConstants => FitFallbackCause::TooManyConstants,
        _ => FitFallbackCause::UnsupportedPlan,
    }
}

/// #286: the label a finite, device-handled fit carries in the ledger.
///
/// These statuses are **not** fallbacks: the exact CPU finalist refit still
/// owns every reported number, and the search's post-fit re-score owns
/// selection. Status 0 is the unlabelled success and returns `None`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn gpu_status_label(status: i32) -> Option<&'static str> {
    match status {
        4 => Some("rank_deficient"),
        5 => Some("recovered"),
        6 => Some("no_improvement"),
        7 => Some("start_unusable"),
        8 => Some("recovered_skipped"),
        _ => None,
    }
}

/// #169: map the kernel's runtime status to its reported cause.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn status_fallback_cause(status: i32) -> FitFallbackCause {
    match status {
        1 => FitFallbackCause::IllConditioned,
        2 => FitFallbackCause::NonFinite,
        3 => FitFallbackCause::RankDeclined,
        _ => FitFallbackCause::UnsupportedPlan,
    }
}

/// #226/#232: the order-statistic arm's status 3 is its own labelled cause, so
/// a declined median polish is never reported as a declined rank polish, and a
/// declined interquartile polish is never reported as a declined median one.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn order_stat_status_fallback_cause(
    status: i32,
    want: crate::fit::OrderStat,
) -> FitFallbackCause {
    match status {
        1 => FitFallbackCause::IllConditioned,
        2 => FitFallbackCause::NonFinite,
        3 => match want {
            crate::fit::OrderStat::Median => FitFallbackCause::MedianDeclined,
            crate::fit::OrderStat::Interquartile => FitFallbackCause::InterquartileDeclined,
        },
        _ => FitFallbackCause::UnsupportedPlan,
    }
}

/// #227: the maximum arm's status 3 is its own labelled cause, so a declined
/// minimax polish is never reported as a declined rank or order-statistic one.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn maximum_status_fallback_cause(status: i32) -> FitFallbackCause {
    match status {
        1 => FitFallbackCause::IllConditioned,
        2 => FitFallbackCause::NonFinite,
        3 => FitFallbackCause::MaximumDeclined,
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

/// #393: the CPU path simplifies the tree and recomputes its complexity after a
/// fit (`evolve_island`). `fit_pending` calls this for every candidate whose
/// constants a device fit or a CPU fallback replaced, so tie-breaks and the
/// in-search archive see the fitted tree; the overlap pipeline simplifies before
/// it scores (`score_pipeline_batch`). A candidate the dimensional filter
/// rejects keeps an infinite error, as in [`Ind::pending`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn refresh_after_fit(ind: &mut Ind, cfg: &Config) {
    ind.tree = expr::simplify(&ind.tree);
    ind.complexity = cfg.complexity_model.complexity(&ind.tree);
    if !cfg.accepts_dimensions(&ind.tree) {
        ind.error = f64::INFINITY;
    }
}

/// #393: true when every candidate with constants is fitted: the row count is at
/// or above [`WIDE_ADMISSION_ROWS`], or the metric is gated and fitted by the CPU
/// arm on every candidate ([`crate::fit::admits_every_candidate`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn wide_admission(metric: Metric, train_rows: usize) -> bool {
    train_rows >= WIDE_ADMISSION_ROWS || crate::fit::admits_every_candidate(metric)
}

/// #223: true when this run's deferred fit admits every candidate with
/// constants: wide rows, a gated metric or a resolved `All` admission.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn admits_all(cfg: &Config, train_rows: usize) -> bool {
    wide_admission(cfg.metric, train_rows) || cfg.fit_admission == crate::fit::FitAdmission::All
}

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
fn admission_count(island_pop: usize, wide: bool) -> usize {
    if wide {
        return island_pop.max(1);
    }
    ((island_pop / 4).max(32)).min(island_pop).max(1)
}

/// #169: deterministic admission of pending offspring to the fitted LM batch.
///
/// Every pending (`error.is_nan()`) individual is screened with its seed
/// constants first. Per island, the best `admission_count(island_pop,
/// wide)` are promoted; ties break on complexity and then on the
/// population index, so the set depends only on the screened errors, the row
/// count and the fixed island/population order. Non-promoted offspring keep
/// their screening score. When `wide` is set (see [`wide_admission`]) every
/// candidate with constants is promoted, so the screen can only exclude trees
/// that have nothing to fit.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn select_promoted(
    islands: &[Vec<Ind>],
    pending: &[(usize, usize)],
    wide: bool,
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
        let take = admission_count(pop.len(), wide).min(ranked.len());
        promoted.extend(ranked[..take].iter().map(|&(j, _, _)| (i, j)));
    }
    // Fitting order is deterministic; the candidates are independent, but a
    // stable order keeps the ledger and any debugging output reproducible.
    promoted.sort_unstable();
    promoted
}

/// #169: true when this generation's offspring are left unfitted for the
/// batched GPU LM arm. The kernel minimises one objective per launch, so only a
/// metric with a native device objective qualifies (#229: `squared`, the
/// information criteria via [`Metric::is_least_squares`], and `log_squashed`
/// with its IRLS weights; #222: `absolute` with L1/IRLS weights and true-MAE
/// acceptance); every other configuration keeps the CPU fit path it had before.
///
/// #228: `signed` is deliberately not deferred — [`crate::fit::fit_loss`] is
/// `None` for it. Its fit arm (the exact `f64` constrained-LS projection) is a
/// host step over the same rows, and #236's correctness gate keeps every
/// non-squared metric on the CPU evaluator until the device has a native,
/// tested scorer for it. Deferring the fit while the offspring are ranked by
/// MSE is exactly the silent substitution that gate exists to prevent; on main
/// this configuration is already refused up front by
/// [`crate::fit::gpu_fit_metric_error`], and `--metric signed` uses the exact
/// CPU arm (including the projection). The device-warm-start form of the same
/// arm (`BatchedGpu` + [`crate::fit::fit_batch_verified`]) is tested directly; it
/// becomes reachable from the search as soon as a native signed loss lands.
fn defers_fit_to_gpu(cfg: &Config, defer_gpu: bool) -> bool {
    defer_gpu
        && cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && (crate::fit::fit_loss(cfg.metric).is_some()
            || crate::fit::rank_fit_only(cfg.metric).is_some()
            || crate::fit::order_stat_fit_only(cfg.metric)
            || crate::fit::maximum_fit_only(cfg.metric))
}

/// #451: why `--fit-strategy gpu` does not defer the fit to the device, or
/// `None` when it does (or the strategy is not `gpu`).
#[cfg(any(feature = "cuda", test))]
fn gpu_fit_off_reason(cfg: &Config) -> Option<String> {
    if !cfg.fit_strategy.defers_to_gpu() || defers_fit_to_gpu(cfg, true) {
        return None;
    }
    if cfg.fit_iters == 0 {
        return Some("--fit-iters 0: nothing to fit".to_string());
    }
    crate::fit::gpu_fit_metric_error(cfg.metric)
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

/// The canonical front of a scored candidate set (#330). Epsilon-dominance is
/// not transitive, so arrival order would decide membership. The candidates are
/// first put in the canonical `(cost, error, structural hash)` order, which
/// makes the insertion sequence, and so the front, a function of the set.
fn canonical_front(cap: usize, mut scored: Vec<Sol>) -> Archive {
    scored.sort_by(|a, b| {
        a.cost
            .total_cmp(&b.cost)
            .then(a.error.total_cmp(&b.error))
            .then(expr::structural_hash(&a.tree).cmp(&expr::structural_hash(&b.tree)))
    });
    let mut out = Archive::new(cap);
    for sol in scored {
        out.try_insert(sol);
    }
    out.canonicalize();
    out
}

/// #495: what the finalist rebuild did, for the ledger and `--json`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default)]
struct RefitReport {
    /// Unique trees of the archive and the final populations.
    candidates: usize,
    /// Exact CPU f64 refits performed.
    refits: usize,
    /// Wall seconds in the exact refit windows.
    refit_s: f64,
    /// Wall seconds in full-data scoring of the unique trees.
    rescore_s: f64,
}

/// #495: indices of the trees worth an exact refit. `pre` are the pre-refit
/// full-data errors and `costs` the complexity costs. A tree qualifies when its
/// error is within `band` (relative) of the lowest finite error among trees of
/// no greater cost. The set depends only on errors and costs.
fn refit_band_selection(pre: &[f64], costs: &[f64], band: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pre.len()).collect();
    order.sort_by(|&a, &b| costs[a].total_cmp(&costs[b]).then(a.cmp(&b)));
    let mut picked = Vec::new();
    let mut env = f64::INFINITY;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && costs[order[end]] == costs[order[start]] {
            end += 1;
        }
        for &i in &order[start..end] {
            if pre[i].is_finite() && pre[i] < env {
                env = pre[i];
            }
        }
        for &i in &order[start..end] {
            if pre[i].is_finite() && pre[i] <= env + band * env.abs() {
                picked.push(i);
            }
        }
        start = end;
    }
    picked.sort_unstable();
    picked
}

/// #169: exact CPU f64 refit of `trees` on all training rows, a warm start from
/// the carried constants. Trees are independent, so the work is split across
/// workers; the result order is unchanged.
fn refit_exact(trees: &mut [Node], train: &EvalSample, cfg: &Config) {
    let fit_one = |t: &mut Node| {
        crate::fit::fit_weighted_metric(
            t,
            &train.sample.x,
            &train.sample.y,
            &train.sample.w,
            cfg.fit_iters,
            cfg.metric,
        );
        *t = expr::simplify(t);
        crate::fit::note_gpu_refit();
    };
    let workers = cfg.threads.max(1).min(trees.len().max(1)); // #463
    let blocks = expr::current_blocks();
    if workers > 1 && trees.len() > 1 {
        let per = trees.len().div_ceil(workers);
        std::thread::scope(|scope| {
            for chunk in trees.chunks_mut(per) {
                let blocks = blocks.clone();
                let fit_one = &fit_one;
                scope.spawn(move || {
                    expr::set_blocks(blocks);
                    for t in chunk.iter_mut() {
                        fit_one(t);
                    }
                });
            }
        });
    } else {
        for t in trees.iter_mut() {
            fit_one(t);
        }
    }
}

/// #15/#25/#27: full-data error of one tree; a dimensionally rejected
/// candidate must not sneak back in on the rebuild.
fn full_data_error(tree: &Node, train: &EvalSample, cfg: &Config) -> f64 {
    if cfg.accepts_dimensions(tree) {
        score_tree(tree, train, cfg)
    } else {
        f64::INFINITY
    }
}

fn full_data_sol(tree: Node, error: f64, cfg: &Config) -> Sol {
    Sol {
        complexity: expr::complexity(&tree),
        cost: cfg.complexity_model.complexity(&tree),
        tree,
        error,
    }
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
    // #169: refit the finalists with the exact CPU f64 arm because the search
    // used the batched GPU constants.
    refit_gpu_fit: bool,
) -> (Archive, RefitReport) {
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
    let mut report = RefitReport {
        candidates: unique.len(),
        ..RefitReport::default()
    };
    let refit_on = refit_gpu_fit && cfg.fit_iters > 0 && !train.sample.is_multi();
    let workers = |n: usize| batch_workers(cfg.threads, n, train.sample.len());
    // #463: scoring is independent per tree; results keep the order.
    let score_all = |trees: Vec<Node>| -> Vec<Sol> {
        par_map(workers(trees.len()), trees, |tree| {
            let error = full_data_error(&tree, train, cfg);
            full_data_sol(tree, error, cfg)
        })
    };

    if !refit_on || cfg.refit_band.is_infinite() {
        // #284: only the exact CPU refit window, not the whole archive rebuild.
        if refit_on {
            let start = std::time::Instant::now();
            refit_exact(&mut unique, train, cfg);
            report.refits = unique.len();
            report.refit_s = start.elapsed().as_secs_f64();
        }
        let start = std::time::Instant::now();
        let scored = score_all(unique);
        report.rescore_s = start.elapsed().as_secs_f64();
        return (canonical_front(archive.cap, scored), report);
    }

    // #495: score with the carried constants, refit only the band around the
    // front, then refit every reported member that is still unrefit.
    let start = std::time::Instant::now();
    let scored = score_all(unique);
    report.rescore_s += start.elapsed().as_secs_f64();
    let costs: Vec<f64> = scored.iter().map(|s| s.cost).collect();
    let pre: Vec<f64> = scored.iter().map(|s| s.error).collect();
    let first = refit_band_selection(&pre, &costs, cfg.refit_band);
    let (front, refits) = refit_to_fixpoint(archive.cap, scored, first, |batch: Vec<Node>| {
        let start = std::time::Instant::now();
        let mut batch = batch;
        refit_exact(&mut batch, train, cfg);
        report.refit_s += start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let rescored = score_all(batch);
        report.rescore_s += start.elapsed().as_secs_f64();
        rescored
    });
    report.refits = refits;
    (front, report)
}

/// #495: refit the trees at `todo`, rebuild the front, and refit every front
/// member that was not refit, until none is left. `refit` maps trees to their
/// refit, scored solutions in the same order. Each round refits at least one
/// new tree, so the loop ends. Returns the front and the number of refits.
fn refit_to_fixpoint(
    cap: usize,
    mut scored: Vec<Sol>,
    mut todo: Vec<usize>,
    mut refit: impl FnMut(Vec<Node>) -> Vec<Sol>,
) -> (Archive, usize) {
    let mut done = vec![false; scored.len()];
    let mut refits = 0;
    loop {
        if !todo.is_empty() {
            let batch: Vec<Node> = todo.iter().map(|&i| scored[i].tree.clone()).collect();
            refits += batch.len();
            for (&i, sol) in todo.iter().zip(refit(batch)) {
                scored[i] = sol;
                done[i] = true;
            }
        }
        let front = canonical_front(cap, scored.clone());
        let refit_hashes: HashSet<u64> = scored
            .iter()
            .zip(&done)
            .filter(|(_, &d)| d)
            .map(|(s, _)| expr::structural_hash(&s.tree))
            .collect();
        todo = Vec::new();
        for member in &front.items {
            let h = expr::structural_hash(&member.tree);
            if refit_hashes.contains(&h) {
                continue;
            }
            if let Some(i) =
                (0..scored.len()).find(|&i| !done[i] && expr::structural_hash(&scored[i].tree) == h)
            {
                todo.push(i);
            }
        }
        if todo.is_empty() {
            return (front, refits);
        }
    }
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

// #465: age-fitness Pareto survivor selection (`--selection afp`).

/// #465: random immigrants added to every island each generation.
fn afp_immigrants(pop: usize) -> usize {
    (pop / 20).max(1)
}

/// #465: the minimised objectives of a population. A non-finite error sorts
/// last, so an all-infinite island falls back to `(complexity, age)`.
fn afp_points(inds: &[Ind]) -> Vec<afp::Point> {
    inds.iter()
        .map(|i| {
            let e = if i.error.is_nan() { f64::INFINITY } else { i.error };
            [e, i.complexity, i.age as f64]
        })
        .collect()
}

/// #465: survivors of parents + offspring + immigrants (in that order, which
/// is the tie-break order). Every survivor is one generation older.
fn afp_select(candidates: Vec<Ind>, pop: usize) -> Vec<Ind> {
    let keep = afp::survivors(&afp_points(&candidates), pop);
    let mut wanted = vec![false; candidates.len()];
    for i in keep {
        wanted[i] = true;
    }
    candidates
        .into_iter()
        .zip(wanted)
        .filter_map(|(mut ind, keep)| {
            keep.then(|| {
                ind.age += 1;
                ind
            })
        })
        .collect()
}

/// #465: the island member migration replaces: highest Pareto rank, then the
/// smallest crowding distance, then the highest index.
fn afp_worst(island: &[Ind]) -> Option<usize> {
    let (rank, crowd) = afp::rank_and_crowd(&afp_points(island));
    (0..island.len()).max_by(|&a, &b| {
        rank[a]
            .cmp(&rank[b])
            .then(crowd[b].total_cmp(&crowd[a]))
            .then(a.cmp(&b))
    })
}

/// #465: one parent and its age. Without AFP this is [`pick_parent_policy`]
/// with age 0, draw for draw. With AFP a front parent counts as age 0 and the
/// island tournament runs on rank and crowding distance.
#[allow(clippy::too_many_arguments)]
fn pick_parent_aged(
    island: &[Ind],
    front: &[Sol],
    qd: Option<&QdArchive>,
    rng: &mut Mt,
    front_p: f64,
    qd_policy: bool,
    qd_control: u8,
    afp: bool,
    rank: &[u32],
    crowd: &[f64],
) -> (Node, u32) {
    if !afp || qd_policy {
        return (pick_parent_policy(island, front, qd, rng, front_p, qd_policy, qd_control), 0);
    }
    match front_parent(front, rng, front_p) {
        Some(tree) => (tree.clone(), 0),
        None => {
            let i = afp::tournament(rank, crowd, rng, 4);
            (island[i].tree.clone(), island[i].age)
        }
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

/// #119: one candidate job for the overlap queue: island, destination slot in
/// that island's `next` vector, and the tree to score.
type PipelineJob = (usize, usize, Node);

/// #286: one candidate's result from an overlap batch. #560: `error` belongs
/// to `tree`, the fitted and simplified tree the barrier stores, or to the
/// offered tree when `tree` is `None`. `offered` is the structural hash of the
/// offered tree: a QD immigrant can replace an offered candidate in the same
/// slot, and a result must never land on a tree it was not computed for.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Debug)]
struct PipelineResult {
    error: f64,
    offered: u64,
    tree: Option<Node>,
}

/// #601: the fewest planned candidates one overlap fit call carries. A fit
/// block owns a candidate for the whole pass over the rows, so a call with
/// fewer blocks than the device holds takes as long as a full one.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const FIT_BATCH_MIN: usize = 128;

/// #601: candidates of the overlap queue that wait for a device fit call.
/// Each entry is the offered job, its structural hash and the fit plan or the
/// reason it has none.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Default)]
struct PipelineFitQueue {
    jobs: Vec<PipelineJob>,
    offered: Vec<u64>,
    entries: Vec<Result<crate::fit::GpuFitPlan, crate::fit::GpuFitReject>>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl PipelineFitQueue {
    fn push(
        &mut self,
        job: PipelineJob,
        offered: u64,
        plan: Result<crate::fit::GpuFitPlan, crate::fit::GpuFitReject>,
    ) {
        self.jobs.push(job);
        self.offered.push(offered);
        self.entries.push(plan);
    }

    fn planned(&self) -> usize {
        self.entries.iter().filter(|e| e.is_ok()).count()
    }

    fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    #[allow(clippy::type_complexity)]
    fn take(
        &mut self,
    ) -> (
        Vec<PipelineJob>,
        Vec<u64>,
        Vec<Result<crate::fit::GpuFitPlan, crate::fit::GpuFitReject>>,
    ) {
        (
            std::mem::take(&mut self.jobs),
            std::mem::take(&mut self.offered),
            std::mem::take(&mut self.entries),
        )
    }
}

/// #286: the device fit arm carried by the overlap queue when the admission
/// set is every candidate with constants (wide panels). The fit of an earlier
/// candidate then runs while the island threads keep generating later ones,
/// instead of a single evaluate → fit → re-evaluate barrier.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
struct PipelineFit {
    loss: crate::fit::DeviceLoss,
    iters: usize,
    metric: Metric,
    polish: crate::fit::FitPolish,
}

/// #286: overlap-with-fit is legal only where the serial admission set is
/// already "every pending candidate with constants". That is exactly
/// [`WIDE_ADMISSION_ROWS`], and only for the LM family (the rank/order/
/// maximum/logistic arms have their own gated policies and stay synchronous).
/// The label-sign codes stay synchronous too: the overlap batch has no
/// single-class decline.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn pipeline_fit_safe(cfg: &Config, train_rows: usize, screened: bool) -> bool {
    // The overlap queue fits on the scoring context's rows; a fit subsample
    // has its own context and takes the synchronous batches.
    let request = if cfg.fit_rows >= 2 { cfg.fit_rows } else { cfg.fit_samples };
    let fit_subsample = request >= 2 && request < train_rows;
    !screened
        && !fit_subsample
        && cfg.pipeline_depth > 0
        && defers_fit_to_gpu(cfg, true)
        && (train_rows >= WIDE_ADMISSION_ROWS || cfg.fit_admission == crate::fit::FitAdmission::All)
        && cfg.fit_iters > 0
        && crate::fit::fit_loss(cfg.metric).is_some_and(|loss| !loss.needs_label_sign())
        && !crate::fit::device_fit_is_gated(cfg.metric)
}

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

/// #477: draw `make()` again (up to 8 times) while it returns a tree with no
/// row-dependent input and `have_const` says the island already holds one. The
/// first constant tree is kept and sets `have_const`. Off, or with no input
/// variable, it is a single draw.
fn draw_row_dependent(
    cfg: &Config,
    nvars: usize,
    have_const: &mut bool,
    mut make: impl FnMut() -> Node,
) -> Node {
    let mut t = make();
    if !cfg.reject_constant_offspring || nvars == 0 {
        return t;
    }
    for _ in 0..8 {
        if expr::row_dependent(&t) {
            return t;
        }
        if !*have_const {
            *have_const = true;
            return t;
        }
        t = make();
    }
    t
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
    // #211: per-generation development-log counters. `None` (the default) makes
    // every count and timer below a single branch with no effect.
    counters: Option<&GenCounters>,
    // #463: worker threads for the fit-and-score phase of this island.
    workers: usize,
    // #471: this island's window of edit success counts.
    mix: &mut MixWindow,
) {
    if !screened {
        for ind in isl.iter_mut() {
            if !ind.error.is_finite() && ind.reeval_probing(train, cfg) {
                GenCounters::count(counters, |c| &c.reevals); // #25 #500
            }
        }
    }

    let afp = cfg.selection == Selection::Afp;
    // #465: Pareto rank and crowding of the parents drive the tournament.
    let (par_rank, par_crowd) = if afp {
        afp::rank_and_crowd(&afp_points(isl))
    } else {
        (Vec::new(), Vec::new())
    };
    let pop = cfg.pop;
    let mut next: Vec<Ind> = Vec::with_capacity(pop);

    // Elitism: carry the best individual over unchanged. In the screened phase
    // the full-data parents are kept by the caller and compete at selection, so
    // the offspring vector stays a full `pop`.
    if !screened && !afp {
        if let Some(best) = isl
            .iter()
            .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
        {
            next.push(best.clone());
        }
    }

    // #477: whether `next` already holds a constant individual. Only the
    // first constant of an island is kept when the filter is on.
    let filter_constant = cfg.reject_constant_offspring && nvars > 0;
    let mut have_const = filter_constant && next.iter().any(|i| !expr::row_dependent(&i.tree));
    let mut seen: HashSet<u64> = HashSet::with_capacity(pop * 2);
    // #77: parents are candidates too; seeding them here rejects an unchanged
    // child before it is fitted and scored.
    seed_seen(&mut seen, &next, isl);
    let mut attempts = 0usize;
    let max_attempts = pop * 50 + 50;

    // #471: edit weights and the adaptive window. Learning needs the child's
    // error now, which a deferred GPU batch or a screening subset does not give.
    let base_opts = cfg.variation_opts();
    let mut opts = base_opts;
    if cfg.adaptive_mix {
        opts.mix = mix.weights(&base_opts.mix);
    }
    let learn = cfg.adaptive_mix && !screened && !defer_gpu;
    let parent_errors: std::collections::HashMap<u64, f64> = if learn {
        isl.iter()
            .map(|i| (expr::structural_hash(&i.tree), i.error))
            .chain(front.iter().map(|s| (expr::structural_hash(&s.tree), s.error)))
            .collect()
    } else {
        std::collections::HashMap::new()
    };
    let mut counts: MixCounts = [(0, 0); expr::EDIT_KINDS];

    let fit = cfg.fit_iters > 0 && !train.sample.is_multi() && !defers_fit_to_gpu(cfg, defer_gpu);

    // #463: variation is serial and owns the RNG (phase A); constant fitting
    // and scoring never draw from it, so they run on the worker pool in draw
    // order (phase B); the results are committed in draw order (phase C). The
    // trajectory is the serial loop's, whatever `workers` is.
    while next.len() < pop && attempts < max_attempts {
        let want = pop - next.len();
        let mut trees: Vec<Node> = Vec::with_capacity(want);
        let mut meta: Vec<Option<(f64, u8)>> = Vec::with_capacity(want);
        let mut ages: Vec<u32> = Vec::with_capacity(want);
        while trees.len() < want && attempts < max_attempts {
            attempts += 1;
            GenCounters::count(counters, |c| &c.attempts);
            // #120: with the policy on, a UCB/stratified cell elite supplies most
            // parents; 5% still come from the global Pareto front. With the policy
            // off this is exactly the #76 draw.
            let (p1, mut age) = pick_parent_aged(
                isl, front, qd, rng, front_p, qd_policy, qd_control, afp, &par_rank, &par_crowd,
            );
            let parent_err = if learn {
                parent_errors.get(&expr::structural_hash(&p1)).copied()
            } else {
                None
            };
            let mut used = 0u8; // #471: bit per edit kind applied to this child
            // #470: variation edits the bare shape; the child inherits the first
            // parent's map as its warm start.
            let (scale_a, scale_b, p1) = if cfg.linear_scaling {
                crate::scale::split(&p1)
            } else {
                (0.0, 1.0, p1)
            };
            let mut child = p1;
            if rng.bool(0.5) {
                let (parent2, age2) = pick_parent_aged(
                    isl, front, qd, rng, front_p, qd_policy, qd_control, afp, &par_rank, &par_crowd,
                );
                age = age.max(age2);
                let parent2 = if cfg.linear_scaling {
                    crate::scale::split(&parent2).2
                } else {
                    parent2
                };
                crossover_with_opts(&mut child, &parent2, rng, Some(aggs), opts.crossover_internal);
            }
            let edit = mutate_with_opts(&mut child, rng, nvars, &cfg.ops, Some(aggs), &opts);
            used |= edit.map_or(0, |k| 1 << k.index());
            if rng.bool(0.5) {
                let edit = mutate_with_opts(&mut child, rng, nvars, &cfg.ops, Some(aggs), &opts);
                used |= edit.map_or(0, |k| 1 << k.index());
            }

            let mut tree = expr::simplify(&child);
            if cfg.linear_scaling {
                tree = expr::simplify(&crate::scale::wrap(scale_a, scale_b, tree));
            }
            if cfg.complexity_model.complexity(&tree) > cfg.complexity_cap() {
                GenCounters::count(counters, |c| &c.too_complex); // #211
                continue;
            }
            let is_const = filter_constant && !expr::row_dependent(&tree); // #477
            if is_const && have_const {
                GenCounters::count(counters, |c| &c.constant);
                continue;
            }
            if cfg.dedup && !seen.insert(expr::structural_hash(&tree)) {
                GenCounters::count(counters, |c| &c.duplicates); // #211
                continue;
            }
            have_const |= is_const;
            trees.push(tree);
            meta.push(parent_err.map(|e| (e, used)));
            ages.push(age);
        }
        let scored = fit_and_score(trees, workers, cfg, train, fit_sample, fit, defer_gpu, counters);
        for (k, (mut ind, age)) in scored.into_iter().zip(ages).enumerate() {
            ind.age = age;
            if ind.complexity > cfg.complexity_cap() {
                GenCounters::count(counters, |c| &c.post_fit_drops); // #463
                continue;
            }
            if let Some((parent, used)) = meta[k] {
                let won = ind.error.is_finite() && (!parent.is_finite() || ind.error < parent);
                for kind in expr::EditKind::ALL {
                    if used & (1 << kind.index()) != 0 {
                        counts[kind.index()].0 += 1;
                        counts[kind.index()].1 += won as u64;
                    }
                }
            }
            GenCounters::count(counters, |c| &c.offspring); // #211
            let slot = next.len();
            next.push(ind);
            offer(&sink, slot, &next[slot]); // #119
        }
    }

    if learn {
        mix.push(counts);
    }

    // Fill any remainder with fresh random trees.
    if next.len() < pop {
        let trees: Vec<Node> = (next.len()..pop)
            .map(|_| {
                draw_row_dependent(cfg, nvars, &mut have_const, || {
                    expr::simplify(&random_tree_opts(
                        rng,
                        nvars,
                        &cfg.ops,
                        cfg.init_depth,
                        Some(aggs),
                        base_opts.balanced,
                    ))
                })
            })
            .collect();
        for ind in fit_and_score(trees, workers, cfg, train, fit_sample, fit, defer_gpu, counters) {
            let slot = next.len();
            GenCounters::count(counters, |c| &c.offspring); // #211
            next.push(ind);
            offer(&sink, slot, &next[slot]); // #119
        }
    }

    // #120: a small niche-immigrant stream. Without it mutation keeps the
    // island in the same descriptor region even when parents are drawn from
    // the archive. The immigrants are aimed at the least-visited cell and are
    // scored exactly like any other offspring (the GPU batch defers them too).
    if qd_policy {
        if let Some(target) = qd.and_then(|q| q.least_occupied_descriptor()) {
            let immigrants = (pop / 10).max(1).min(pop);
            let mut trees: Vec<Node> = Vec::with_capacity(immigrants);
            let mut have_const_imm = filter_constant
                && next[..pop - immigrants].iter().any(|i| !expr::row_dependent(&i.tree)); // #477
            for _ in 0..immigrants {
                let hc = have_const_imm;
                let t = draw_row_dependent(cfg, nvars, &mut have_const_imm, || {
                    let mut t = random_tree_opts(
                        rng, nvars, &cfg.ops, cfg.init_depth, Some(aggs), base_opts.balanced,
                    );
                    let mut best_score = if hc && !expr::row_dependent(&t) {
                        f64::INFINITY
                    } else {
                        let d = qd::descriptor(&t, train.sample);
                        (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2)
                    };
                    for _ in 0..3 {
                        let cand = random_tree_opts(
                            rng, nvars, &cfg.ops, cfg.init_depth, Some(aggs), base_opts.balanced,
                        );
                        let d = qd::descriptor(&cand, train.sample);
                        let score = (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2);
                        if score < best_score && !(hc && !expr::row_dependent(&cand)) {
                            t = cand;
                            best_score = score;
                        }
                    }
                    expr::simplify(&t)
                });
                trees.push(t);
            }
            // Immigrants are fitted even when the batch defers scoring.
            let fit_immigrants = cfg.fit_iters > 0 && !train.sample.is_multi();
            let inds = fit_and_score(
                trees, workers, cfg, train, fit_sample, fit_immigrants, defer_gpu, counters,
            );
            for (k, ind) in inds.into_iter().enumerate() {
                let slot = pop - immigrants + k;
                GenCounters::count(counters, |c| &c.offspring); // #211
                next[slot] = ind;
                offer(&sink, slot, &next[slot]);
            }
        }
    }

    // #465: fresh lineages at age 0 compete with the older ones. They are scored
    // with the rest of the generation but never offered to the overlap queue,
    // whose result slots cover the first `pop` candidates only.
    if afp {
        let k = afp_immigrants(pop);
        let mut have_const_afp =
            filter_constant && next.iter().any(|i| !expr::row_dependent(&i.tree)); // #477
        let trees: Vec<Node> = (0..k)
            .map(|_| {
                draw_row_dependent(cfg, nvars, &mut have_const_afp, || {
                    expr::simplify(&random_tree_opts(rng, nvars, &cfg.ops, cfg.init_depth, Some(aggs), base_opts.balanced))
                })
            })
            .collect();
        next.extend(fit_and_score(trees, workers, cfg, train, fit_sample, fit, defer_gpu, counters));
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
    /// #169: candidates sent to the exact CPU fallback arm (unsupported,
    /// ill-conditioned or non-finite). #255: the ones beyond
    /// `fit_fallback_bounded` keep their finite device constants and are not
    /// refit, but they are still fallbacks by cause.
    pub fit_fallback: usize,
    /// #169: pending offspring screened before admission.
    pub fit_screened: usize,
    /// #448: candidates with constants fitted without a screening score under
    /// wide admission; `fit_screened + fit_admitted_wide >= fit_attempts`.
    pub fit_admitted_wide: usize,
    /// #169: CPU fallback counts labelled by cause.
    pub fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #265: fallbacks whose CPU refit lowered the exact error, by cause.
    pub fit_fallback_improved_by_cause: BTreeMap<String, usize>,
    /// #255: wall seconds this generation spent in the exact CPU fallback
    /// fits. A subset of the run's `fit.host_seconds`.
    pub fit_fallback_seconds: f64,
    /// #498: wall seconds this generation spent in the exact host gate of
    /// device proposals. A subset of the run's `fit.host_seconds`.
    pub fit_gate_seconds: f64,
    /// #255: fallbacks held at their finite device constants because the
    /// per-generation bound was reached.
    pub fit_fallback_bounded: usize,
    /// #255: workers the fallback batch used.
    pub fit_fallback_workers: usize,
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
    /// #448: candidates with constants fitted without a screening score under
    /// wide admission; `fit_screened + fit_admitted_wide >= fit_attempts`.
    pub fit_admitted_wide: usize,
    /// #169: CPU fallback counts labelled by cause.
    pub fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #265: fallbacks whose CPU refit lowered the exact error, by cause.
    pub fit_fallback_improved_by_cause: BTreeMap<String, usize>,
    /// #286: finite device answers by label; every entry is a non-fallback.
    pub fit_status_by_label: BTreeMap<String, usize>,
    /// #169: exact CPU f64 refits of finalists before reporting.
    pub fit_refits: usize,
    /// #495: unique trees considered for the finalist refit.
    pub fit_refit_candidates: usize,
    /// #495: wall seconds of the exact finalist refit windows.
    pub fit_refit_s: f64,
    /// #495: wall seconds of full-data scoring in the finalist rebuild.
    pub fit_rescore_s: f64,
    /// #169: wall time spent in the batched fit kernels, milliseconds.
    pub fit_ms: f64,
    /// #496: the polish-phase part of `fit_ms`, apportioned by per-block cycles.
    pub fit_polish_ms: f64,
    /// #601: device fit launches, the candidates they carried, longest launch (ms).
    pub fit_calls: usize,
    pub fit_call_candidates: usize,
    pub fit_call_ms_max: f64,
    /// #496: arithmetic of the device fit polish.
    pub fit_polish: crate::fit::FitPolish,
    /// #169: wall time spent in the whole fit stage, milliseconds.
    pub fit_host_ms: f64,
/// #284/#255: wall time spent in the exact CPU fallback fits, milliseconds.
    /// Already included in `fit_host_ms`; report it, never add it again.
    pub fit_fallback_ms: f64,
    /// #498: wall time of the exact gate of device proposals, milliseconds.
    /// Already included in `fit_host_ms`; report it, never add it again.
    pub fit_gate_ms: f64,
    /// #255: fallbacks held at their finite device constants by the
    /// per-generation bound.
    pub fit_fallback_bounded: usize,
    /// #255: largest worker count the fallback batch used.
    pub fit_fallback_workers: usize,
    /// #255: configured per-generation fallback cap (`0` = unbounded).
    pub fit_fallback_max: usize,
    /// Rows of the device fit twin: the fit subsample, else every row; `0`
    /// without a device fit.
    pub fit_rows: usize,
    /// Rows the device evaluator scores.
    pub score_rows: usize,
    /// Bytes of the f64 fit twin resident on the device; `0` without a
    /// device fit.
    pub fit_twin_bytes: u64,
    /// Hash of the fixed fit subsample rows, or `None` for a full fit.
    pub fit_subsample_hash: Option<u64>,
    /// #223: the admission the deferred fit ran under.
    pub fit_admission: crate::fit::FitAdmission,
    /// #284: run-scoped stage/device ledger. `Default` is an empty ledger, so
    /// every existing fixture that builds `GpuRunStats` still compiles.
    pub ledger: crate::gpu_stats::RunLedger,
    /// #293: the device memory plan the run was built under.
    pub memory_plan: Option<crate::memory::DevicePlan>,
    /// #293: the device allocation ledger at the end of the run.
    pub observed: Option<crate::memory::DeviceObserved>,
    /// #481: the objective the device scored candidates by, `None` for a
    /// fit-only or exact-statistic evaluator.
    pub score_objective: Option<crate::metrics::ScoreObjective>,
}

pub struct Outcome {
    pub archive: Archive,
    pub test_errors: Vec<f64>,
    pub generations: usize,
    /// #256: true only when the engine declared the run finished early (the
    /// `solved` or `patience` stop) **and** the final `percent_converged` is at
    /// least [`crate::progress_stats::CONVERGED_THRESHOLD`]. A `time_budget`
    /// stop is never convergence; read [`Outcome::stop_reason`] for why the
    /// loop ended.
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
    /// #290: calibrated per-stage cost model and the route ledger for this run.
    pub dispatch: crate::dispatch::DispatchReport,
    /// #120: quality-diversity coverage report, or `None` when the layer is off.
    pub qd: Option<QdReport>,
    /// #211: why the generation loop stopped: `solved` (an error at or below
    /// the solved floor), `patience` (no improvement for `--patience`
    /// generations), `front_stable` (#464), `time_budget`, or `generations` (the
    /// budget ran out). Read
    /// by the development log's `run_end` event; it is not printed, so the
    /// default output is unchanged.
    pub stop_reason: &'static str,
    /// #466: what the front prune did; all zero unless `--prune` is given.
    pub prune: PruneReport,
    /// #482: offspring scored over the run, summed over islands and
    /// generations, including generations before a resume.
    pub candidates_evaluated: u64,
}

/// #256: the reported `converged_early`. `stopped` is the loop-control flag
/// (`solved`, `patience` or a terminal checkpoint); it stays separate so those
/// stops remain terminal in a checkpoint even when the run did not converge.
fn reported_converged_early(stopped: bool, convergence: &Convergence) -> bool {
    stopped && convergence.percent_converged >= crate::progress_stats::CONVERGED_THRESHOLD
}

impl Outcome {
    /// #256: the loop ended before the generation budget ran out, for any
    /// reason (`solved`, `patience`, `time_budget`, `already_stopped`).
    pub fn stopped_early(&self) -> bool {
        self.stop_reason != "generations"
    }
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

/// #256: a checkpoint is also taken when this many wall seconds have passed
/// since the last one, so a run whose generations take minutes does not lose
/// hours to a crash between two `--save-every` boundaries. `RCSR_CHECKPOINT_INTERVAL_S`
/// overrides it (the tests use a tiny value).
const CHECKPOINT_MAX_INTERVAL_S: f64 = 900.0;

fn checkpoint_interval_s() -> f64 {
    std::env::var("RCSR_CHECKPOINT_INTERVAL_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(CHECKPOINT_MAX_INTERVAL_S)
}

/// Whether a boundary checkpoint is due after `gens_done` generations.
fn checkpoint_due(save_every: usize, gens_done: usize, since_last: f64, interval_s: f64) -> bool {
    save_every > 0 && (gens_done % save_every == 0 || since_last >= interval_s)
}

#[allow(clippy::too_many_arguments)]
fn snapshot(
    cfg: &Config,
    backend: Backend,
    gens_done: usize,
    stopped_early: bool,
    best_prev: f64,
    no_improve: usize,
    front_stable: usize,
    archive: &Archive,
    islands: &[Vec<Ind>],
    rngs: &[Mt],
    stagnant: &[usize],
    mixes: &[MixWindow],
    main_rng: &Mt,
    train: &Sample,
    test: &Sample,
    final_archive: Option<&Archive>,
    candidates_evaluated: u64,
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
        front_stable,
        train: crate::checkpoint::fingerprint(train),
        test: crate::checkpoint::fingerprint(test),
        archive: Archive {
            items: archive.items.clone(),
            cap: archive.cap,
        },
        islands: islands
            .iter()
            .zip(rngs.iter())
            .enumerate()
            .map(|(i, (pop, rng))| {
                let (words, idx) = rng.state();
                IslandState {
                    rng: Mt::from_state(*words, idx),
                    mix: mixes.get(i).cloned().unwrap_or_default(),
                    population: pop.iter().map(Ind::to_checkpoint).collect(),
                }
            })
            .collect(),
        stagnant: stagnant.to_vec(),
        main_rng: Mt::from_state(*words, idx),
        // #256: the search state is f32 proxy fitness on the GPU. The refit
        // archive the run reports travels beside it, not instead of it.
        error_kind: crate::checkpoint::ErrorKind::for_backend(backend),
        final_archive: final_archive.map(|a| Archive {
            items: a.items.clone(),
            cap: a.cap,
        }),
        identity: Some(crate::provenance::BuildIdentity::current(
            backend == Backend::Gpu,
        )),
        target_spec: None,
        candidates_evaluated,
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
    if let Some(flag) = cfg.node_variation_flag() {
        return Err(format!("{} is a node-path setting", flag));
    }
    if cfg.selection == Selection::Afp {
        return Err("--selection afp runs on the node path".to_string());
    }
    if cfg.fit_iters != 0 {
        return Err("constant fitting is enabled".to_string());
    }
    // #288: the flat arena is scored by the objective kernel through
    // `eval_pool`; an exact order-statistic/rank run must use the tree scorer
    // over flattened plans, so it stays on the owned-tree loop rather than
    // being ranked by an SSE proxy.
    if crate::metrics::ScoreStat::for_metric(cfg.metric).is_some() {
        return Err("exact order/rank scoring uses the tree scorer".to_string());
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
    // #287: the flat arena carries programs, not the per-candidate constant
    // count the AIC family's penalty needs. Keep that family on the owned-tree
    // path, where `count_constants(tree)` is available for every candidate.
    if search_score_objective(cfg).is_some_and(|o| o.needs_constants()) {
        return Err("parameter-aware metric needs per-candidate constants".to_string());
    }
    // The pass-through path must never score a metric without a native
    // objective (the #236 defect). `gpu` is only built when this holds, but the
    // gate is asserted here so a future caller cannot bypass it. #481: under
    // `--fitness pnl` the objective is the long/flat regret.
    if search_score_objective(cfg).is_none() {
        return Err("metric has no native score objective".to_string());
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

/// #211: a cheap copy of the GPU ledger, used to take a per-generation delta.
///
/// The run-level counters are monotone, so subtracting a snapshot from the one
/// taken at the next generation boundary gives that generation's share without
/// adding a second accounting path. `programs`/`native`/`lowered`/`fallbacks`
/// come from the batch stats keyed by generation and are therefore not
/// subtracted.
#[derive(Clone, Debug, Default)]
pub struct GpuSnapshot {
    pub accounting: crate::gpu_stats::GpuStats,
    pub fit_attempts: usize,
    pub fit_gpu: usize,
    pub fit_fallback: usize,
    pub fit_fallback_by_cause: BTreeMap<String, usize>,
    /// #265: fallbacks whose CPU refit lowered the exact error, by cause.
    pub fit_fallback_improved_by_cause: BTreeMap<String, usize>,
    pub fallback_s: f64,
    pub fallback_workers: usize,
    pub programs: usize,
    pub native: usize,
    pub lowered: usize,
    pub fallbacks: usize,
    /// #256: whole fit stage wall (device call, CPU fallback, write-back).
    pub fit_host_s: f64,
    /// #256: exact CPU fit-fallback wall, a subset of `fit_host_s`.
    pub fit_fallback_s: f64,
    /// #498: exact host gate wall, a subset of `fit_host_s`.
    pub fit_gate_s: f64,
    pub fit_fallback_workers: usize,
}

impl GpuSnapshot {
    /// Everything that happened after `earlier`.
    pub fn since(&self, earlier: &GpuSnapshot) -> GpuSnapshot {
        let mut causes = self.fit_fallback_by_cause.clone();
        for (name, count) in &earlier.fit_fallback_by_cause {
            let entry = causes.entry(name.clone()).or_insert(0);
            *entry = entry.saturating_sub(*count);
        }
        causes.retain(|_, v| *v > 0);
        let mut improved = self.fit_fallback_improved_by_cause.clone();
        for (name, count) in &earlier.fit_fallback_improved_by_cause {
            let entry = improved.entry(name.clone()).or_insert(0);
            *entry = entry.saturating_sub(*count);
        }
        improved.retain(|_, v| *v > 0);
        GpuSnapshot {
            accounting: self.accounting.since(&earlier.accounting),
            fit_attempts: self.fit_attempts.saturating_sub(earlier.fit_attempts),
            fit_gpu: self.fit_gpu.saturating_sub(earlier.fit_gpu),
            fit_fallback: self.fit_fallback.saturating_sub(earlier.fit_fallback),
            fit_fallback_by_cause: causes,
            fit_fallback_improved_by_cause: improved,
            fallback_s: (self.fallback_s - earlier.fallback_s).max(0.0),
            fallback_workers: self.fallback_workers.max(earlier.fallback_workers),
            programs: self.programs,
            native: self.native,
            lowered: self.lowered,
            fallbacks: self.fallbacks,
            fit_host_s: (self.fit_host_s - earlier.fit_host_s).max(0.0),
            fit_fallback_s: (self.fit_fallback_s - earlier.fit_fallback_s).max(0.0),
            fit_gate_s: (self.fit_gate_s - earlier.fit_gate_s).max(0.0),
            fit_fallback_workers: self.fit_fallback_workers.max(earlier.fit_fallback_workers),
        }
    }
}

/// #284: read the always-on stage clock and reset the mark.
fn stage_lap(mark: &mut std::time::Instant) -> f64 {
    let now = std::time::Instant::now();
    let d = now.duration_since(*mark).as_secs_f64();
    *mark = now;
    d
}

/// #256: take `fit_s` seconds of fit-stage wall out of the windows it ran in,
/// in the given order (`[eval, prep, search]`), never below zero. Returns the
/// reduced windows and the fit seconds actually placed, so the windows plus the
/// fit phase still add up to what they added up to before.
fn split_fit_window(mut windows: [f64; 3], fit_s: f64) -> ([f64; 3], f64) {
    let mut placed = 0.0;
    let mut remaining = if fit_s.is_finite() { fit_s.max(0.0) } else { 0.0 };
    for w in windows.iter_mut() {
        let take = remaining.min(*w);
        *w -= take;
        placed += take;
        remaining -= take;
    }
    (windows, placed)
}

/// #211: read the clock and reset the mark. A no-op (0.0) with no log attached.
fn dev_lap(mark: &mut Option<std::time::Instant>) -> f64 {
    match mark.as_mut() {
        Some(t) => {
            let now = std::time::Instant::now();
            let d = now.duration_since(*t).as_secs_f64();
            *t = now;
            d
        }
        None => 0.0,
    }
}

/// #211: hash of the island RNG states at a generation boundary.
///
/// Two runs that agree on this key at the same generation saw the same random
/// stream; a comparison between logs can therefore tell "different data" from
/// "different randomness".
fn dev_rng_key(rngs: &[Mt], main: &Mt) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |w: u32| {
        h ^= w as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for r in rngs {
        let (words, idx) = r.state();
        mix(idx as u32);
        for &w in words.iter() {
            mix(w);
        }
    }
    let (words, idx) = main.state();
    mix(idx as u32);
    for &w in words.iter() {
        mix(w);
    }
    h
}

/// #211: assemble one generation's development record.
///
/// Called only when a development log is attached, so the quantile sorts below
/// never run on a normal search. `errors` are this generation's finite
/// population errors and `island_best` its per-island minima, both measured by
/// the caller (the owned-tree and flat-arena loops read different state).
#[allow(clippy::too_many_arguments)]
fn dev_generation(
    generation: usize,
    cfg: &Config,
    errors: Vec<f64>,
    island_best: Vec<f64>,
    archive: &Archive,
    convergence: Convergence,
    rng_key: u64,
    t: f64,
    phases: [f64; 5],
    counters: Option<&GenCounters>,
    offers: usize,
    admissions: usize,
    refits: usize,
    gpu: Option<GpuSnapshot>,
    qd: Option<crate::devlog::QdGen>,
) -> crate::devlog::GenRecord {
    let q = |p: f64| crate::diagnostics::quantile(&errors, p);
    let best_error = errors.iter().copied().fold(f64::INFINITY, f64::min);
    let points: Vec<(f64, f64)> = archive.items.iter().map(|s| (s.cost, s.error)).collect();
    let (fit, fallback_s, fallback_workers) = match &gpu {
        Some(g) => (
            crate::devlog::FitGen {
                attempts: g.fit_attempts,
                gpu: g.fit_gpu,
                fallback: g.fit_fallback,
                refits,
                fallback_by_cause: g
                    .fit_fallback_by_cause
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect(),
                fallback_improved_by_cause: g
                    .fit_fallback_improved_by_cause
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect(),
                host_seconds: g.fit_host_s,
                fallback_seconds: g.fit_fallback_s,
                gate_seconds: g.fit_gate_s,
                fallback_workers: g.fit_fallback_workers,
            },
            g.fallback_s,
            g.fallback_workers,
        ),
        None => (
            crate::devlog::FitGen {
                refits,
                ..Default::default()
            },
            0.0,
            0,
        ),
    };
    let gpu_rec = gpu.as_ref().map(|g| crate::devlog::GpuGen {
        batches: g.accounting.batches,
        programs: g.programs.max(g.accounting.programs as usize) as u64,
        kernel_s: g.accounting.kernel_s,
        wall_s: g.accounting.wall_s,
        duty: g.accounting.duty_cycle(),
        native: g.native,
        lowered: g.lowered,
        fallbacks: g.fallbacks,
        fallback_s: g.fallback_s,
        fallback_workers: g.fallback_workers,
    });
    // #256: the GPU fit stage runs inside the evaluator window (and, with
    // screening, inside the carried-refit and island windows). It is its own
    // phase, taken out of those windows in that order, so the phases still sum
    // to the generation wall and `eval` is evaluation alone.
    let (windows, fit_s) = split_fit_window([phases[2], phases[0], phases[1]], fit.host_seconds);
    crate::devlog::GenRecord {
        generation,
        total_generations: cfg.generations,
        t,
        wall_s: phases.iter().sum(),
        prep_s: windows[1],
        search_s: windows[2],
        eval_s: windows[0],
        select_s: phases[3],
        archive_s: phases[4],
        fit_s,
        // Island-thread fitting plus the exact CPU fallback fits, which run on
        // the pool and are otherwise invisible to the counters (wall seconds).
        fit_cpu_s: GenCounters::seconds(counters, |c| &c.fit_ns) + fit.fallback_seconds,
        // Candidate scoring plus the evaluator's CPU fallback pool (wall).
        eval_cpu_s: GenCounters::seconds(counters, |c| &c.eval_ns) + fallback_s,
        phase_b_s: GenCounters::seconds(counters, |c| &c.batch_ns),
        post_fit_drops: GenCounters::total(counters, |c| &c.post_fit_drops) as usize,
        reevals: GenCounters::total(counters, |c| &c.reevals) as usize,
        constant_rejected: GenCounters::total(counters, |c| &c.constant) as usize,
        best_error,
        p05: q(0.05),
        p25: q(0.25),
        median: q(0.5),
        p75: q(0.75),
        p95: q(0.95),
        pareto_size: archive.items.len(),
        hypervolume: crate::qd::fixed_reference_hypervolume(&points, 1.0),
        archive_hash: archive.hash(),
        offers,
        admissions,
        attempts: GenCounters::total(counters, |c| &c.attempts) as usize,
        duplicates: GenCounters::total(counters, |c| &c.duplicates) as usize,
        too_complex: GenCounters::total(counters, |c| &c.too_complex) as usize,
        offspring: GenCounters::total(counters, |c| &c.offspring) as usize,
        fitted: GenCounters::total(counters, |c| &c.fitted) as usize,
        refits,
        gpu: gpu_rec,
        fit,
        fallback_s,
        fallback_workers,
        qd,
        confidence: convergence.confidence,
        stability: convergence.stability,
        maturity: convergence.maturity,
        percent_converged: convergence.percent_converged,
        island_best,
        rng_key,
    }
}

/// #122: the generation loop over the flat postfix arena.
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
    mut front_stable: usize,
    mut candidates: u64,
    save_every: usize,
    mut on_checkpoint: Option<CheckpointFn<'_>>,
    backend_reason: Option<String>,
    device: Option<String>,
    dispatcher: &mut crate::dispatch::Dispatcher,
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
    // #284: run clock and ledger for the flat path.
    let mut run_ledger = crate::gpu_stats::RunLedger::new();
    run_ledger.record_stage(
        crate::gpu_stats::Stage::HostOther,
        t0.elapsed().as_secs_f64(),
    );
    let mut eval_stage_total = 0.0f64;
    let mut io_stage_total = 0.0f64;
    let mut last_save = std::time::Instant::now();
    let save_interval_s = checkpoint_interval_s();
    // #211: why the loop ends, for the development log's run_end event.
    let mut stop_reason: &'static str = if converged_early {
        "already_stopped"
    } else {
        "generations"
    };

    let mut front_stop = cfg
        .stop_front
        .map(|_| crate::progress_stats::FrontStop::new(&archive, front_stable));

    for gen in gens_done..cfg.generations {
        // A checkpoint saved after an early stop stays stopped even when the
        // generation budget is raised, exactly like the owned-tree loop.
        if converged_early {
            break;
        }
        gens_done = gen + 1;
        // #211: development-log clock and counters. `None` without --dev-log, so
        // this loop is the one it always was.
        let mut dev_mark = cfg.dev_log.is_some().then(std::time::Instant::now);
        let mut stage_mark = std::time::Instant::now();
        let mut dev_offers = 0usize;
        let mut dev_admissions = 0usize;
        let dev_gpu_before = if dev_mark.is_some() {
            Some(gpu.dev_snapshot(gen))
        } else {
            None
        };
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
        let dev_generate = dev_lap(&mut dev_mark); // #211
        let stage_generate = stage_lap(&mut stage_mark); // #284
        next.evaluate(
            &mut gpu.ev,
            gpu.score
                .expect("the pass-through postfix path only runs native score objectives"),
        )?;
        let dev_eval = dev_lap(&mut dev_mark); // #211
        let stage_eval = stage_lap(&mut stage_mark); // #284
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
                if dev_mark.is_some() {
                    dev_offers += 1; // #211
                }
                if !archive.admits(error, cost) {
                    continue;
                }
                if dev_mark.is_some() {
                    dev_admissions += 1; // #211
                }
                let Some(tree) = next.decode(idx) else { continue };
                archive.try_insert_view(&tree, error, expr::complexity(&tree), cost);
            }
        }
        let dev_archive = dev_lap(&mut dev_mark); // #211
        let stage_archive = stage_lap(&mut stage_mark); // #284
        run_ledger.record_stage(crate::gpu_stats::Stage::Vary, stage_generate);
        run_ledger.record_stage(crate::gpu_stats::Stage::Archive, stage_archive);
        eval_stage_total += stage_eval;
        candidates += (n_islands * pop) as u64;
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
            fit_admitted_wide: 0,
            fit_fallback_by_cause: Default::default(),
            fit_fallback_improved_by_cause: Default::default(),
            fit_fallback_seconds: 0.0,
            fit_gate_seconds: 0.0,
            fit_fallback_bounded: 0,
            fit_fallback_workers: 0,
        });

        // #256: a time-budget stop is not convergence and not terminal. The
        // generation completes (progress event, bookkeeping, checkpoint) and
        // the loop ends after it.
        let budget_hit = cfg
            .time_budget
            .map_or(false, |budget| t0.elapsed().as_secs_f64() >= budget);

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
        // #211: same boundary as `--progress`. The flat arena has its own
        // de-duplication pass, so the duplicate count is read from the arena
        // (programs that share a canonical postfix hash) instead of a rejected
        // attempt counter; `fit_*` stays zero because this path is fit-free by
        // construction.
        if let Some(observe) = &cfg.front_observer {
            observe(gens_done, t0.elapsed().as_secs_f64(), &archive);
        }
        if let Some(report) = &cfg.dev_log {
            let mut dev_errors: Vec<f64> = Vec::with_capacity(n_islands * pop);
            let mut dev_island_best: Vec<f64> = Vec::with_capacity(n_islands);
            for i in 0..n_islands {
                let mut best = f64::INFINITY;
                for j in 0..pop {
                    let e = next.err[next.slot_index(i, j)];
                    if e.is_finite() {
                        dev_errors.push(e);
                        best = best.min(e);
                    }
                }
                dev_island_best.push(best);
            }
            let unique = unique_slots(&next).len();
            let programs = n_islands * pop;
            let gpu_now = gpu.dev_snapshot(gen);
            let counters = GenCounters::default();
            GenCounters::add(&counters.attempts, programs as u64);
            GenCounters::add(&counters.offspring, programs as u64);
            GenCounters::add(&counters.duplicates, programs.saturating_sub(unique) as u64);
            let mut rec = dev_generation(
                gens_done,
                cfg,
                dev_errors,
                dev_island_best,
                &archive,
                convergence,
                {
                    let mut h = 0xcbf2_9ce4_8422_2325u64;
                    for w in [cfg.seed, gen as u32] {
                        h ^= w as u64;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                    h
                },
                t0.elapsed().as_secs_f64(),
                [0.0, dev_generate, dev_eval, 0.0, dev_archive],
                Some(&counters),
                dev_offers,
                dev_admissions,
                0,
                dev_gpu_before.map(|b| gpu_now.since(&b)),
                None,
            );
            // The flat arena compiles the whole batch for the kernel and
            // records its own synthesis of the per-generation counters; that
            // entry, not the (empty) node-path one, describes this batch.
            if let (Some(g), Some(gs)) = (rec.gpu.as_mut(), gpu_stats.last()) {
                g.native = gs.native;
                g.lowered = gs.lowered;
                g.fallbacks = gs.fallbacks;
            }
            report(&rec);
        }

        if gen_best.is_finite() {
            if cfg.fitness.is_metric() && cfg.metric.is_solved(gen_best) {
                converged_early = true;
                stop_reason = "solved"; // #211
            }
            if crate::progress_stats::no_improvement(best_prev, gen_best, cfg.patience_tol) {
                no_improve += 1;
            } else {
                no_improve = 0;
            }
            best_prev = best_prev.min(gen_best);
            if cfg.patience > 0 && no_improve >= cfg.patience {
                converged_early = true;
                stop_reason = "patience"; // #211
            }
        }
        if let Some(fs) = front_stop.as_mut() {
            front_stable = fs.observe(&archive, cfg.front_tol());
            if gen_best.is_finite() && !converged_early && front_stable >= cfg.front_window() {
                converged_early = true;
                stop_reason = "front_stable"; // #464
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
                next.evaluate(
            &mut gpu.ev,
            gpu.score
                .expect("the pass-through postfix path only runs native score objectives"),
        )?;
            }
        }

        if checkpoint_due(save_every, gens_done, last_save.elapsed().as_secs_f64(), save_interval_s) {
            last_save = std::time::Instant::now();
            let io_mark = std::time::Instant::now();
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
                            age: 0,
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
                front_stable,
                &archive,
                &islands_now,
                &rngs,
                &stagnant,
                &[],
                &main_rng,
                train,
                test,
                None,
                candidates,
            );
            emit(&mut on_checkpoint, &cp);
            io_stage_total += io_mark.elapsed().as_secs_f64();
        }

        pool = next;
        if converged_early {
            break;
        }
        if budget_hit {
            stop_reason = "time_budget"; // #211
            break;
        }
    }

    // #284: close measured generation stages to the run clock. The evaluator
    // window joins the ledger first so the fit reclassification can subtract
    // from it rather than add a second stage.
    run_ledger.record_stage(crate::gpu_stats::Stage::Io, io_stage_total);
    run_ledger.record_stage(crate::gpu_stats::Stage::Eval, eval_stage_total);
    let tail_start = std::time::Instant::now();
    let pre_tail_elapsed = tail_start.duration_since(t0).as_secs_f64();
    let accounted_before_tail = run_ledger.stage_sum_s();
    if pre_tail_elapsed > accounted_before_tail {
        run_ledger.record_stage(
            crate::gpu_stats::Stage::HostOther,
            pre_tail_elapsed - accounted_before_tail,
        );
    }

    // #256: the exit checkpoint is emitted after the exact f64 rebuild so it
    // carries the archive the run reports.
    let mut io_final_s = 0.0f64;
    let io_mark = std::time::Instant::now();
    let pending_cp: Option<Checkpoint> = if on_checkpoint.is_some() {
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
                        age: 0,
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
            front_stable,
            &archive,
            &islands_now,
            &rngs,
            &stagnant,
            &[],
            &main_rng,
            train,
            test,
            None,
            candidates,
        );
        Some(cp)
    } else {
        None
    };
    io_final_s += io_mark.elapsed().as_secs_f64();

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
    let (final_archive, prune_report) = prune::prune_archive(final_archive, train, train_data, cfg);
    let sorted = final_archive.sorted();
    let test_data = EvalSample::new(test, cfg);
    let test_errors: Vec<f64> = sorted
        .iter()
        .map(|s| eval_error_on(&s.tree, &test_data, cfg.metric))
        .collect();
    let mut final_archive = Archive::new(final_archive.cap);
    final_archive.items = sorted;
    if let Some(mut cp) = pending_cp {
        cp.final_archive = Some(Archive {
            items: final_archive.items.clone(),
            cap: final_archive.cap,
        });
        let io_mark = std::time::Instant::now();
        emit(&mut on_checkpoint, &cp);
        io_final_s += io_mark.elapsed().as_secs_f64();
    }

    let best = final_archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min);
    let convergence = progress_stats.observe(gens_done, best, &final_archive);

    // #284: the remaining tail is one measured host window (final f64 rebuild,
    // test scoring and archive canonicalisation).
    let tail_total = (t0.elapsed().as_secs_f64() - pre_tail_elapsed).max(0.0);
    run_ledger.record_stage(crate::gpu_stats::Stage::Io, io_final_s);
    run_ledger.record_stage(
        crate::gpu_stats::Stage::HostOther,
        (tail_total - io_final_s).max(0.0),
    );
    run_ledger.set_wall(t0.elapsed().as_secs_f64());

    let mut stats = gpu.run_stats();
    let mut ledger = run_ledger;
    ledger.absorb(&stats.ledger);
    // The flat path is fit-free (no per-candidate constants), but keep the
    // same reclassification rule for a future fit arm.
    ledger.reclassify_eval_to_fit(stats.fit_host_ms / 1000.0);
    ledger.set_wall(t0.elapsed().as_secs_f64());
    stats.ledger = ledger;
    stats.fit_fallback_max = cfg.fit_fallback_max;
    stats.fit_admission = cfg.fit_admission;
    stats.fit_polish = cfg.fit_polish;

    Ok(Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early: reported_converged_early(converged_early, &convergence),
        seeds: Vec::new(),
        convergence,
        fit_samples: 0,
        fit_rows: 0,
        archive_refit: false,
        predictor_samples: 0,
        backend: crate::backend::Kind::Gpu,
        backend_reason,
        device,
        gpu: Some(stats),
        dispatch: dispatcher.snapshot(),
        qd: None,
        stop_reason,
        prune: prune_report,
        candidates_evaluated: candidates,
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
    // #284: one run clock and one ledger for the search phase. The ledger is
    // filled by the exclusive main-thread windows below; the device intervals
    // and fit leaves are absorbed from `GpuSearch` at the end.
    let mut run_ledger = crate::gpu_stats::RunLedger::new();
    // #10: user-defined blocks are thread-local. Snapshot the registry here so
    // every island thread (and the initial population below) sees the same
    // definitions; the list is empty unless --function was supplied.
    let blocks = expr::current_blocks();

    // #290: one calibration and one route ledger per run. The probe is the
    // startup cost model; contract stages (variation, selection, archive, I/O,
    // features, P&L and exact refit) are planned once and recorded so the
    // ledger answers for every stage, not only the two that can move.
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut calibration = crate::dispatch::Calibration::probe();
    #[cfg(feature = "cuda")]
    if cfg.gpu {
        if let Some(rates) = crate::dispatch_gpu::device_rates() {
            calibration = calibration.with_device_rates(rates);
        }
    }
    let mut dispatcher =
        crate::dispatch::Dispatcher::new(calibration).with_mode(cfg.dispatch);
    for stage in [
        crate::dispatch::Stage::Vary,
        crate::dispatch::Stage::Select,
        crate::dispatch::Stage::Archive,
        crate::dispatch::Stage::Features,
        crate::dispatch::Stage::Pnl,
        crate::dispatch::Stage::Io,
        crate::dispatch::Stage::Refit,
    ] {
        let decision = dispatcher.plan(stage, &crate::dispatch::Workload::contract());
        dispatcher.record_plan(decision, 1, 0);
    }

    // #26/#141: one resident evaluator for the full training sample, used for
    // whole generations. With the default `exact` arm constants are fitted on
    // the CPU before the batch; `--fit-strategy gpu` adds a fit batch (#169). The backend report below records
    // what actually ran: a missing feature, a missing device, an unsupported
    // objective or a runtime CUDA error leaves the CPU path and a reason.
    #[cfg(feature = "cuda")]
    let mut backend_device: Option<String> = None;
    #[cfg(not(feature = "cuda"))]
    let backend_device: Option<String> = None;
    // #230: the device exists only because the rank **fit** arm needs it; the
    // fitness is still CPU f64, so `backend` stays `cpu` but the gpu report and
    // the finalist rebuild must still happen.
    #[cfg(feature = "cuda")]
    let mut fit_only_device = false;
    #[cfg(not(feature = "cuda"))]
    let fit_only_device = false;
    let mut backend_reason: Option<String> = None;
    #[cfg(feature = "cuda")]
    let mut gpu: Option<GpuSearch> = if !cfg.gpu {
        None
    } else {
        // #230: the evaluator still has no native rank scorer, but the exact
        // rank polish is a device *fit* objective. When that is the only thing
        // standing between the run and the device, build a **fit-only**
        // evaluator: it moves constants on the GPU, and every fitness value is
        // scored by the CPU (`native_metric = false`). The backend itself stays
        // `cpu`, and `backend_reason` names the open native scorer.
        //
        // #226/#232: `--metric median` and `--metric interquartile` use the
        // same fit-only shape with the batched exact order-statistic polish.
        // Their measured cost-model gate can decline the arm for a small
        // sample; that refusal is recorded as the reason.
        let GpuPlanShape {
            score_stat,
            score_admitted,
            fit_only,
            logistic_fit,
        } = gpu_plan_shape(cfg, train);
        let plan: Option<(
            crate::fit::DeviceLoss,
            bool,
            Option<crate::metrics::ScoreStat>,
        )> = if score_admitted {
            Some((crate::fit::DeviceLoss::Squared, false, score_stat))
        } else if fit_only {
            // #288: a fit-only evaluator for an order/rank metric exists
            // because the *scoring* floor (or another named gate) refused the
            // native scorer; record that reason rather than the generic one.
            let floor_reason = score_stat
                .filter(|s| train.len() < s.min_rows())
                .map(|s| s.small_sample_reason().to_string());
            backend_reason = Some(if let Some(reason) = floor_reason {
                reason
            } else if logistic_fit {
                format!(
                    "{} is scored exactly on the CPU; the device fits constants only",
                    cfg.metric.name()
                )
            } else {
                "metric is not scored natively on the GPU".to_string()
            });
            Some((crate::fit::DeviceLoss::Squared, false, None))
        } else {
            match gpu_blocker(cfg, train) {
                Some(reason) => {
                    if cfg.gpu_notes {
                        eprintln!("# gpu: off ({})", reason);
                    }
                    backend_reason = Some(reason.to_string());
                    None
                }
                None => {
                    // #287/#289: the score gate is independent of the fit
                    // objective. The evaluator gets a `ScoreObjective`; `loss`
                    // only seeds the legacy fit/eval field and is `Squared`
                    // when the metric has no device LM loss (correlation,
                    // signed, maximum, Brier, hinge).
                    let _objective = search_score_objective(cfg)
                        .expect("gpu_blocker admits only native score objectives");
                    let loss = crate::fit::fit_loss(cfg.metric)
                        .unwrap_or(crate::fit::DeviceLoss::Squared);
                    Some((loss, true, None))
                }
            }
        };
        match plan {
            None => None,
            Some((loss, native, stat)) => {
                let score = if native {
                    Some(
                        search_score_objective(cfg)
                            .expect("native GPU plan owns a score objective"),
                    )
                } else {
                    None
                };
                let fit_rows = if fit_subsample_on_device(cfg, train) {
                    device_fit_indices(cfg, train)
                } else {
                    Vec::new()
                };
                // #293: the plan runs before the first allocation.
                let memory_plan = device_plan_for(cfg, train);
                let created = match memory_plan.as_ref().and_then(plan_blocker) {
                    Some(reason) => Err(reason),
                    None => GpuSearch::new(
                        train,
                        (cfg.fit_strategy.defers_to_gpu() && cfg.fit_iters > 0) || stat.is_some(),
                        loss,
                        native,
                        stat,
                        score,
                        logistic_fit,
                        t0,
                        &fit_rows,
                        memory_plan.as_ref(),
                        cfg.fitness.book(),
                    ),
                };
                match created {
                    Ok(mut g) => {
                        g.memory_plan = memory_plan;
                        let device = crate::gpu::GpuEvaluator::device_name();
                        if cfg.gpu_notes {
                            // #169: the GPU-fit arm runs at the generation
                            // barrier, so the overlap pipeline is off for it.
                            let scheduling = if stat.is_some() {
                                "synchronous exact plan batches (#288)".to_string()
                            } else if cfg.fit_strategy.defers_to_gpu()
                                && defers_fit_to_gpu(cfg, true)
                            {
                                if pipeline_fit_safe(cfg, train.len(), false) {
                                    format!(
                                        "GPU-fit overlap depth {} (batches of {})",
                                        cfg.pipeline_depth,
                                        (cfg.pop / cfg.pipeline_depth.max(1)).max(1)
                                    )
                                } else {
                                    "GPU-fit batches (synchronous)".to_string()
                                }
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
                            if native {
                                let fitness = if !cfg.fitness.is_metric() {
                                    "long_pnl regret of the f32 predictions, reported numbers stay f64"
                                } else if cfg.metric == crate::metrics::Metric::Squared {
                                    "f32 fitness, reported numbers stay f64"
                                } else {
                                    "native device loss, f32 fitness"
                                };
                                eprintln!("# gpu: {} ({}; {})", device, scheduling, fitness);
                            } else if let Some(stat) = stat {
                                eprintln!(
                                    "# gpu: {} ({}; exact {} device statistic, f64 fitness \
                                     and finalists, #288)",
                                    device,
                                    scheduling,
                                    stat.name()
                                );
                            } else if logistic_fit {
                                eprintln!(
                                    "# gpu-fit: {} ({}; exact log-loss fit, CPU f64 fitness and \
                                     finalists — the evaluator has no label semantics, #234)",
                                    device, scheduling
                                );
                            } else if let Some(want) =
                                crate::fit::OrderStat::for_metric(cfg.metric)
                            {
                                eprintln!(
                                    "# gpu-fit: {} ({}; exact {} polish, CPU f64 \
                                     fitness — native {} scoring is open, #{})",
                                    device,
                                    scheduling,
                                    want.label(),
                                    want.label(),
                                    if want == crate::fit::OrderStat::Median {
                                        "226"
                                    } else {
                                        "232"
                                    }
                                );
                            } else if crate::fit::maximum_fit_only(cfg.metric) {
                                eprintln!(
                                    "# gpu-fit: {} ({}; exact maximum-error polish, CPU f64 \
                                     fitness — native maximum scoring is open, #227)",
                                    device, scheduling
                                );
                            } else {
                                eprintln!(
                                    "# gpu-fit: {} ({}; exact rank polish, CPU f64 \
                                     fitness — native rank scoring is open, #230)",
                                    device, scheduling
                                );
                            }
                        }
                        if cfg.gpu_notes {
                            if let Some(p) = g
                                .memory_plan
                                .as_ref()
                                .filter(|p| p.mode == crate::memory::Residency::Tiled)
                            {
                                eprintln!(
                                    "# gpu: tiled residency (tile {} rows, {} wave(s) of up to {} candidates; \
                                     the rows stream from the host, fit and exact statistics stay on the CPU, device_memory)",
                                    p.tile_rows, p.waves, p.wave_candidates
                                );
                            }
                        }
                        if !native && stat.is_none() {
                            fit_only_device = true;
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
                }
            }
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
    if cfg.gpu_notes {
        if let Some(why) = gpu_fit_off_reason(cfg) {
            eprintln!("# gpu-fit: off ({})", why);
        }
    }
    // #226/#232: only a run that actually asked for the device fit arm can be
    // refused by it. `--no-fit` (or a CPU strategy) has nothing to fit, so it
    // must not borrow this label.
    #[cfg(feature = "cuda")]
    if cfg.gpu_notes
        && cfg.fit_strategy.defers_to_gpu()
        && cfg.fit_iters > 0
        && crate::fit::order_stat_fit_only(cfg.metric)
        && gpu.is_none()
    {
        let why = order_stat_fit_only_blocker(cfg, train).unwrap_or("no usable device");
        let stat = crate::fit::OrderStat::for_metric(cfg.metric)
            .map(|w| w.label())
            .unwrap_or("median");
        eprintln!("# gpu-fit: off ({}); the exact CPU {} arm runs", why, stat);
        if backend_reason.is_none() {
            backend_reason = Some(why.to_string());
        }
    }
    #[cfg(feature = "cuda")]
    if cfg.gpu_notes && crate::fit::maximum_fit_only(cfg.metric) && gpu.is_none() {
        let why = maximum_fit_only_blocker(cfg, train).unwrap_or("no usable device");
        eprintln!("# gpu-fit: off ({}); the exact CPU maximum arm runs", why);
        if backend_reason.is_none() {
            backend_reason = Some(why.to_string());
        }
    }
    #[cfg(feature = "cuda")]
    let gpu_available = gpu.is_some();
    #[cfg(not(feature = "cuda"))]
    let gpu_available = false;
    if !gpu_available {
        // No device: the two movable stages still answer for themselves with
        // the named cause, instead of looking like CPU-contract stages.
        let cause = if backend_reason
            .as_deref()
            .map_or(false, |r| r.starts_with("device_memory"))
        {
            crate::dispatch::FallbackCause::DeviceMemory
        } else {
            crate::dispatch::FallbackCause::NoDevice
        };
        let eval = crate::dispatch::Workload::eval(
            cfg.pop.saturating_mul(cfg.islands.max(1)),
            train.len(),
            cfg.max_complexity.max(1),
        )
        .with_refusal(cause);
        let decision = dispatcher.plan(crate::dispatch::Stage::Eval, &eval);
        dispatcher.record_plan(decision, eval.candidates, eval.bytes);
        let fit = crate::dispatch::Workload::fit(
            cfg.pop.saturating_mul(cfg.islands.max(1)),
            train.len(),
            4,
            cfg.fit_iters.max(1),
        )
        .with_refusal(cause);
        let decision = dispatcher.plan(crate::dispatch::Stage::Fit, &fit);
        dispatcher.record_plan(decision, fit.candidates, fit.bytes);
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
    let mut mixes: Vec<MixWindow>;
    let mut gens_done: usize;
    let mut converged_early: bool;
    let mut best_prev: f64;
    let mut no_improve: usize;
    let mut front_stable: usize;
    let mut candidates: u64;
    // #85: did any generation fit on a screening subset? Only then may the
    // final archive be refit; a request at least as large as the sample, or
    // `--no-fit`, stays an exact no-op.
    // The device fit arm fits a fixed subsample while the evaluator scores every
    // row, so the CPU screening draw below stays off for it.
    let device_fit_sub = defer_gpu && fit_subsample_on_device(cfg, train);
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
            front_stable: saved_front_stable,
            archive: saved_archive,
            islands: saved_islands,
            stagnant: saved_stagnant,
            candidates_evaluated: saved_candidates,
            main_rng,
            ..
        } = cp;
        gens_done = generations;
        candidates = saved_candidates;
        converged_early = stopped_early;
        best_prev = saved_best;
        no_improve = saved_no_improve;
        front_stable = saved_front_stable;
        archive = saved_archive;
        stagnant = saved_stagnant;
        rng = main_rng;
        rngs = Vec::with_capacity(saved_islands.len());
        islands = Vec::with_capacity(saved_islands.len());
        mixes = Vec::with_capacity(saved_islands.len());
        for isl in saved_islands {
            rngs.push(isl.rng);
            mixes.push(isl.mix);
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
        mixes = vec![MixWindow::new(); cfg.islands.max(1)];
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
        let init_eval = if fit_sampling_active(cfg, train) && !device_fit_sub {
            &train_data
        } else {
            init_data
        };
        // #463: the trees are drawn serially (the RNG order is fixed); scoring
        // them does not touch the RNG and runs on the worker pool.
        let mut init_trees: Vec<Node> = Vec::new();
        for isl in 0..cfg.islands.max(1) {
            let depth = cfg.init_depth.min(1 + isl % 3);
            for _ in 0..cfg.pop {
                init_trees.push(random_tree_opts(
                    &mut rng,
                    nvars,
                    &cfg.ops,
                    depth,
                    Some(&aggs),
                    cfg.tree_arity == TreeArity::Balanced,
                ));
            }
        }
        let mut init_inds = par_map(
            batch_workers(cfg.threads, init_trees.len(), init_eval.sample.len()),
            init_trees,
            |tree| {
                let tree = if !cfg.linear_scaling {
                    tree
                } else if defer_gpu {
                    crate::scale::wrap(0.0, 1.0, tree) // #470
                } else {
                    let s = init_eval.sample;
                    crate::scale::wrap_closed_form(tree, &s.x, &s.y, &s.w)
                };
                // #26: generation 0 is a whole generation too; defer it to the
                // same batch as the first real generation.
                if defer_gpu {
                    Ind::pending(tree, cfg)
                } else {
                    Ind::evaluated(tree, init_eval, cfg) // #25 #85
                }
            },
        )
        .into_iter();
        for _ in 0..cfg.islands.max(1) {
            islands.push(init_inds.by_ref().take(cfg.pop).collect());
        }

        gens_done = 0;
        converged_early = false;
        best_prev = f64::INFINITY;
        no_improve = 0;
        front_stable = 0;
        candidates = 0;
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
        defer_gpu = score_generation(&mut gpu, &mut gpu_used, &mut islands, train, cfg, &mut backend_reason, gens_done, &mut dispatcher);
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
                        front_stable,
                        candidates,
                        save_every,
                        on_checkpoint,
                        backend_reason,
                        backend_device.clone(),
                        &mut dispatcher,
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

    // #211: why the generation loop ends, for the development log's run_end.
    let mut stop_reason: &'static str = if converged_early {
        "already_stopped"
    } else {
        "generations"
    };

    // #284: everything before the first generation is an exclusive host window:
    // evaluator construction, initial population, seeded solutions and the
    // generation-0 screening/fit. Any fit inside it is moved from host_other to
    // the fit stage, never counted twice.
    let setup_s = t0.elapsed().as_secs_f64();
    #[cfg(feature = "cuda")]
    let setup_fit_s = gpu.as_ref().map(|g| g.fit_host_ms / 1000.0).unwrap_or(0.0);
    #[cfg(not(feature = "cuda"))]
    let setup_fit_s = 0.0;
    run_ledger.record_stage(crate::gpu_stats::Stage::Fit, setup_fit_s);
    run_ledger.record_stage(
        crate::gpu_stats::Stage::HostOther,
        (setup_s - setup_fit_s).max(0.0),
    );
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables, unused_assignments))]
    let mut io_stage_total = 0.0f64;
    let mut eval_stage_total = 0.0f64;
    let mut last_save = std::time::Instant::now();
    let save_interval_s = checkpoint_interval_s();
    if !converged_early {
        let mut front_stop = cfg
            .stop_front
            .map(|_| crate::progress_stats::FrontStop::new(&archive, front_stable));

        for gen in gens_done..cfg.generations {
            gens_done = gen + 1;

            // #211: development-log scaffolding. With no log attached `dev_mark`
            // is `None`, the counter block is `None`, and every `dev_lap` below
            // returns 0.0 without reading a clock, so the generation is the one
            // it always was.
            let mut dev_mark = cfg.dev_log.is_some().then(std::time::Instant::now);
            let mut stage_mark = std::time::Instant::now();
            let dev_counters = Some(GenCounters {
                timed: cfg.dev_log.is_some() || cfg.gpu,
                ..GenCounters::default()
            });
            // The island threads take `Option<&GenCounters>`, which is `Copy`;
            // the owning `Option` stays here for the readout below.
            let dev_counters_ref = dev_counters.as_ref();
            let mut dev_refits = 0usize;
            let mut dev_offers = 0usize;
            let mut dev_admissions = 0usize;
            #[cfg(feature = "cuda")]
            let dev_gpu_before = if dev_mark.is_some() {
                gpu.as_ref().map(|g| g.dev_snapshot(gen))
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let dev_gpu_before: Option<GpuSnapshot> = None;

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
            let device_fit_sub = defer_gpu && fit_subsample_on_device(cfg, train);
            let fit_sample = if device_fit_sub {
                fit_screened = true;
                None
            } else {
                fit_for_generation(cfg, gen_train, gen)
            };
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
                        if dev_mark.is_some() {
                            dev_refits += 1; // #211
                        }
                    }
                }
            }

            // #85: keep the full-data parents for the selection step.
            let parents: Option<Vec<Vec<Ind>>> = if screened || cfg.selection == Selection::Afp {
                Some(islands.clone())
            } else {
                None
            };
            let dev_prep = dev_lap(&mut dev_mark); // #211
            let stage_prep = stage_lap(&mut stage_mark); // #284

            // ---- parallel generation across islands ---------------------
            let aggs_ref = &aggs;
            let eval_view = eval_data; // #27
            // #463: islands share the thread budget.
            let island_workers = (cfg.threads / islands.len().max(1)).max(1);
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
            // #286: the overlap queue now carries the fit arm too, but only on
            // the wide panels where the serial admission set is every pending
            // candidate with constants. Then fitting an earlier candidate
            // overlaps the island threads' generation of later ones, instead
            // of one evaluate → fit → re-evaluate barrier per generation.
            let pipeline_fit = pipeline_fit_safe(cfg, train.len(), screened).then(|| PipelineFit {
                loss: crate::fit::fit_loss(cfg.metric)
                    .expect("pipeline_fit_safe admits only LM objectives"),
                iters: cfg.fit_iters,
                metric: cfg.metric,
                polish: cfg.fit_polish,
            });
            #[cfg(feature = "cuda")]
            let pipeline_active = !screened
                && defer_gpu
                && cfg.pipeline_depth > 0
                && (!defers_fit_to_gpu(cfg, defer_gpu) || pipeline_fit.is_some())
                // #288: the overlap batches score compiled programs through the
                // objective kernel; an exact order/rank run must use the plan
                // scorer, so it takes the synchronous (tree) batch instead.
                && crate::metrics::ScoreStat::for_metric(cfg.metric).is_none();
            #[cfg(not(feature = "cuda"))]
            let pipeline_active = false;
            #[cfg(not(feature = "cuda"))]
            #[allow(unused_variables)]
            let pipeline_fit: Option<PipelineFit> = None;
            // One batch is `pop / depth` candidates, so `depth` batches queue
            // ahead of the scorer; live memory is at most `(depth + 1) * batch`
            // candidate trees plus their compiled programs.
            let batch_size = (cfg.pop / cfg.pipeline_depth.max(1)).max(1);
            let queue_capacity = cfg.pipeline_depth.max(1) * batch_size;
            #[cfg(not(feature = "cuda"))]
            let _ = (pipeline_active, batch_size, queue_capacity);
            #[cfg(feature = "cuda")]
            let mut pipeline_results: Vec<Vec<Option<PipelineResult>>> =
                vec![vec![None; cfg.pop.max(1)]; islands.len()];
            #[cfg(feature = "cuda")]
            let mut pipeline_error: Option<String> = None;
            #[cfg(feature = "cuda")]
            let drain_before = gpu
                .as_ref()
                .map_or((0.0, 0.0), |g| (g.drain_eval_ms, g.drain_fit_ms));
            #[cfg(feature = "cuda")]
            let mut pipeline_scored = false;
            #[cfg(feature = "cuda")]
            let mut fit_queue = PipelineFitQueue::default();
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
                    .zip(mixes.iter_mut())
                    .zip(senders.into_iter())
                    .enumerate()
                    .map(|(i, (((isl, r), mix), sender))| {
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
                                dev_counters_ref, // #211
                                island_workers, // #463
                                mix, // #471
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
                            gen_train,
                            gen,
                            pipeline_fit,
                            &mut fit_queue,
                        )
                    });
                    if pipeline_error.is_none() {
                        if let (Some(fit), false) = (pipeline_fit, fit_queue.is_empty()) {
                            pipeline_error = fit_pipeline_queue(
                                &mut fit_queue,
                                &mut gpu,
                                &mut pipeline_results,
                                gen_train,
                                gen,
                                fit,
                            )
                            .err();
                        }
                    }
                }

                for h in handles {
                    let _ = h.join();
                }
            });
            let dev_search = dev_lap(&mut dev_mark); // #211: island scope
            candidates += GenCounters::total(dev_counters_ref, |c| &c.offspring);
            let stage_search = stage_lap(&mut stage_mark); // #284: see note below

            #[cfg(feature = "cuda")]
            {
                if pipeline_active {
                    // Results are keyed by (island, slot), so the arrival order
                    // across concurrent island threads cannot change them.
                    // #560: a result for a replaced candidate is dropped, which
                    // leaves the slot pending for `score_generation` below; a
                    // rejected dimension keeps its infinite error.
                    for (i, slots) in pipeline_results.iter_mut().enumerate() {
                        for (j, value) in slots.iter_mut().enumerate() {
                            let Some(r) = value.take() else { continue };
                            let ind = &mut islands[i][j];
                            if expr::structural_hash(&ind.tree) != r.offered {
                                continue;
                            }
                            if let Some(tree) = r.tree {
                                ind.complexity = cfg.complexity_model.complexity(&tree);
                                ind.tree = tree;
                            }
                            ind.error = if cfg.accepts_dimensions(&ind.tree) {
                                r.error
                            } else {
                                f64::INFINITY
                            };
                        }
                    }
                    if pipeline_error.is_none() {
                        if pipeline_scored {
                            gpu_used = true;
                        }
                        // Score anything the kernel could not compile (left
                        // pending) through the normal CPU fallback.
                        defer_gpu =
                            score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen, &mut dispatcher);
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
                        score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen, &mut dispatcher);
                }
            }
            let dev_eval = dev_lap(&mut dev_mark); // #211: evaluator batch
            let stage_eval = stage_lap(&mut stage_mark); // #284

            // #85: successive halving phase two. Promote the top fraction of
            // the screening offspring, refit and re-score them on every
            // training row, then select the next population against the
            // full-data parents. The archive only ever sees full-data scores.
            if let Some(parents) = parents {
                let promote = promoted_count(cfg.pop);
                for (pop, parents) in islands.iter_mut().zip(parents) {
                    if !screened {
                        // #465: survivors of parents + offspring + immigrants.
                        let mut combined = parents;
                        combined.extend(std::mem::take(pop));
                        *pop = afp_select(combined, cfg.pop);
                        continue;
                    }
                    let mut offspring = std::mem::take(pop);
                    offspring.sort_by(|a, b| {
                        a.error
                            .partial_cmp(&b.error)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top: Vec<Ind> = offspring.into_iter().take(promote).collect();
                    // #463: the refits are independent; results keep the order.
                    let promoted: Vec<Ind> = par_map(
                        batch_workers(cfg.threads, top.len(), train_data.sample.len()),
                        top,
                        |mut ind| {
                            let sol = refit_survivor(&ind.tree, &train_data, cfg);
                            ind.tree = sol.tree;
                            ind.error = sol.error;
                            ind.complexity = sol.cost;
                            ind
                        },
                    );
                    if dev_mark.is_some() {
                        dev_refits += promoted.len(); // #211
                    }
                    let mut combined = parents;
                    combined.extend(promoted);
                    if cfg.selection == Selection::Afp {
                        combined = afp_select(combined, cfg.pop);
                    } else {
                        select_population(&mut combined, cfg.pop);
                    }
                    *pop = combined;
                }
            }
            let dev_select = dev_lap(&mut dev_mark); // #211
            let stage_select = stage_lap(&mut stage_mark); // #284

            // ---- archive + convergence bookkeeping ----------------------
            let mut gen_best = f64::INFINITY;
            for pop in &islands {
                for ind in pop {
                    if ind.error < gen_best {
                        gen_best = ind.error;
                    }
                    if dev_mark.is_some() {
                        // #211: "admission" is the archive's own dominance test,
                        // asked before the insert so the count is a property of
                        // the offer, not of the insertion order.
                        dev_offers += 1;
                        if archive.admits(ind.error, ind.complexity) {
                            dev_admissions += 1;
                        }
                    }
                    archive.try_insert_view(
                        &ind.tree,
                        ind.error,
                        expr::complexity(&ind.tree),
                        ind.complexity,
                    );
                }
            }
            let dev_archive = dev_lap(&mut dev_mark); // #211
            let stage_archive = stage_lap(&mut stage_mark); // #284

            #[cfg(feature = "cuda")]
            let (drain_eval_s, drain_fit_s) = gpu.as_ref().map_or((0.0, 0.0), |g| {
                (
                    (g.drain_eval_ms - drain_before.0) / 1000.0,
                    (g.drain_fit_ms - drain_before.1) / 1000.0,
                )
            });
            #[cfg(not(feature = "cuda"))]
            let (drain_eval_s, drain_fit_s) = (0.0f64, 0.0f64);
            // #284: exclusive main-thread stages. `prep` is the generation
            // setup/carried refit window, which has no dedicated name in the
            // stage enum, so it is `host_other`. The evaluator window is
            // accumulated and partitioned with the nested fit window after the
            // loop, so fit is never counted twice.
            run_ledger.record_stage(crate::gpu_stats::Stage::HostOther, stage_prep);
            run_ledger.record_island_window_overlap(
                stage_search,
                GenCounters::seconds(dev_counters_ref, |c| &c.fit_ns),
                islands.len().max(1) * island_workers,
                drain_eval_s,
                drain_fit_s,
            );
            run_ledger.record_stage(crate::gpu_stats::Stage::Select, stage_select);
            run_ledger.record_stage(crate::gpu_stats::Stage::Archive, stage_archive);
            eval_stage_total += stage_eval;

            // #256: a time-budget stop is not convergence and not terminal.
            // The generation finishes (progress event, bookkeeping, migration,
            // checkpoint) and the loop ends after it.
            let budget_hit = cfg
                .time_budget
                .map_or(false, |budget| t0.elapsed().as_secs_f64() >= budget);

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

            // #211: one development-log record per completed generation, written
            // at the same boundary `--progress` reports. Everything here is a
            // reading of state the search already holds; nothing is drawn from
            // the RNG and nothing is written back.
            if let Some(observe) = &cfg.front_observer {
                observe(gens_done, t0.elapsed().as_secs_f64(), &archive);
            }
            if let Some(report) = &cfg.dev_log {
                #[cfg(feature = "cuda")]
                let gpu_now = gpu.as_ref().map(|g| g.dev_snapshot(gen));
                #[cfg(not(feature = "cuda"))]
                let gpu_now: Option<GpuSnapshot> = None;
                // The population error vector and per-island best are read here,
                // once, rather than inside the helper: the log pays for them
                // only when it exists.
                let mut dev_errors: Vec<f64> = Vec::with_capacity(cfg.pop * islands.len());
                let mut dev_island_best: Vec<f64> = Vec::with_capacity(islands.len());
                for pop in &islands {
                    let mut best = f64::INFINITY;
                    for ind in pop {
                        if ind.error.is_finite() {
                            dev_errors.push(ind.error);
                            best = best.min(ind.error);
                        }
                    }
                    dev_island_best.push(best);
                }
                let rec = dev_generation(
                    gens_done,
                    cfg,
                    dev_errors,
                    dev_island_best,
                    &archive,
                    convergence,
                    dev_rng_key(&rngs, &rng),
                    t0.elapsed().as_secs_f64(),
                    [dev_prep, dev_search, dev_eval, dev_select, dev_archive],
                    dev_counters_ref,
                    dev_offers,
                    dev_admissions,
                    dev_refits,
                    dev_gpu_before.zip(gpu_now).map(|(b, n)| n.since(&b)),
                    qd_archive.as_ref().map(|q| {
                        let r = q.report();
                        crate::devlog::QdGen {
                            occupied: r.occupied_reachable,
                            reachable: r.reachable,
                            coverage: r.reachable_coverage,
                            entropy: r.occupancy_entropy,
                        }
                    }),
                );
                report(&rec);
            }

            if gen_best.is_finite() {
                if cfg.fitness.is_metric() && cfg.metric.is_solved(gen_best) {
                    converged_early = true;
                    stop_reason = "solved"; // #211
                    break;
                }
                if crate::progress_stats::no_improvement(best_prev, gen_best, cfg.patience_tol) {
                    no_improve += 1;
                } else {
                    no_improve = 0;
                }
                best_prev = best_prev.min(gen_best);
                if cfg.patience > 0 && no_improve >= cfg.patience {
                    converged_early = true;
                    stop_reason = "patience"; // #211
                    break;
                }
            }

            if let Some(fs) = front_stop.as_mut() {
                front_stable = fs.observe(&archive, cfg.front_tol());
                if gen_best.is_finite() && front_stable >= cfg.front_window() {
                    converged_early = true;
                    stop_reason = "front_stable"; // #464
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
                let best_ages: Vec<u32> = islands
                    .iter()
                    .map(|pop| {
                        pop.iter()
                            .min_by(|a, b| {
                                a.error
                                    .partial_cmp(&b.error)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map_or(0, |i| i.age)
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
                            if cfg.selection == Selection::Afp {
                                return afp_worst(&islands[ii]);
                            }
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
                            worst.age = best_ages[src];
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
                        let mut have_const = false; // #477
                        for ind in islands[ii].iter_mut() {
                            let hc = have_const; // #477
                            let mut t = draw_row_dependent(cfg, nvars, &mut have_const, || {
                                let mut t = random_tree_opts(
                                    &mut r,
                                    nvars,
                                    &cfg.ops,
                                    cfg.init_depth,
                                    Some(&aggs),
                                    cfg.tree_arity == TreeArity::Balanced,
                                );
                                if let Some(target) = qd_target {
                                    let mut best = t.clone();
                                    let mut best_d = qd::descriptor(&t, gen_train);
                                    let mut best_score = if hc && !expr::row_dependent(&t) {
                                        f64::INFINITY
                                    } else {
                                        (best_d.usage - target.usage).powi(2) + (best_d.skew - target.skew).powi(2)
                                    };
                                    for _ in 0..3 {
                                        let cand = random_tree_opts(
                                            &mut r,
                                            nvars,
                                            &cfg.ops,
                                            cfg.init_depth,
                                            Some(&aggs),
                                            cfg.tree_arity == TreeArity::Balanced,
                                        );
                                        let d = qd::descriptor(&cand, gen_train);
                                        let score = (d.usage - target.usage).powi(2)
                                            + (d.skew - target.skew).powi(2);
                                        if score < best_score && !(hc && !expr::row_dependent(&cand)) {
                                            best = cand;
                                            best_d = d;
                                            best_score = score;
                                        }
                                    }
                                    let _ = best_d;
                                    t = best;
                                }
                                t
                            });
                            if cfg.linear_scaling {
                                t = crate::scale::ensure(t, 0.0, 1.0); // #470
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
                    score_generation(&mut gpu, &mut gpu_used, &mut islands, gen_train, cfg, &mut backend_reason, gen, &mut dispatcher);
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
            if checkpoint_due(save_every, gens_done, last_save.elapsed().as_secs_f64(), save_interval_s) {
                last_save = std::time::Instant::now();
                let io_mark = std::time::Instant::now();
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
                    front_stable,
                    &archive,
                    &islands,
                    &rngs,
                    &stagnant,
                    &mixes,
                    &rng,
                    train,
                    test,
                    None,
                    candidates,
                );
                emit(&mut on_checkpoint, &cp);
                io_stage_total += io_mark.elapsed().as_secs_f64();
            }

            if budget_hit {
                stop_reason = "time_budget"; // #211
                break;
            }
        }
    }

    // #284: in a non-CUDA build the evaluator window is not emitted (there is
    // no GPU report), but the stage clock still runs so the code is identical.
    #[cfg(not(feature = "cuda"))]
    let _ = eval_stage_total;
    // The measured evaluator windows join the ledger before the closure check;
    // the fit part is reclassified below after the gadget's totals are known.
    run_ledger.record_stage(crate::gpu_stats::Stage::Io, io_stage_total);
    run_ledger.record_stage(crate::gpu_stats::Stage::Eval, eval_stage_total);

    // #284: close the generation loop's measured stages to the run clock. Any
    // gap between the last stage boundary and this mark is still host time.
    let tail_start = std::time::Instant::now();
    let pre_tail_elapsed = tail_start.duration_since(t0).as_secs_f64();
    let accounted_before_tail = run_ledger.stage_sum_s();
    if pre_tail_elapsed > accounted_before_tail {
        run_ledger.record_stage(
            crate::gpu_stats::Stage::HostOther,
            pre_tail_elapsed - accounted_before_tail,
        );
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

    // Capture the exit state before the final full-data archive rebuild: the
    // saved search archive is the one at the last completed generation. It is
    // emitted after the rebuild (#256) so it also carries the archive the run
    // reports.
    let mut io_final_s = 0.0f64;
    let io_mark = std::time::Instant::now();
    let pending_cp: Option<Checkpoint> = if on_checkpoint.is_some() {
        let backend = if (defer_gpu && !fit_only_device) || gpu_used {
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
            front_stable,
            &archive,
            &islands,
            &rngs,
            &stagnant,
            &mixes,
            &rng,
            train,
            test,
            None,
            candidates,
        );
        Some(cp)
    } else {
        None
    };
    io_final_s += io_mark.elapsed().as_secs_f64();

    // #169: with the GPU-fit arm every finalist is also refit by the exact CPU
    // f64 LM before it is scored on all training rows.
    let gpu_fit_used =
        cfg.fit_strategy.defers_to_gpu() && (gpu_used || fit_only_device);
    let mut refit_report = RefitReport::default();
    if subsampling_active(cfg, train) || gpu_used || fit_only_device || resumed || fit_screened {
        // Generation fitness was estimated on subsets, or on the GPU in f32, so
        // the front is rebuilt from the surviving trees and scored on every
        // training row with the CPU in f64. From here on the archive and every
        // number derived from it are full-data and match the CPU backend.
        let rebuilt = full_data_archive(&archive, &islands, &train_data, cfg, gpu_fit_used);
        archive = rebuilt.0;
        refit_report = rebuilt.1;
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
    let (archive, prune_report) = prune::prune_archive(archive, train, &train_data, cfg);
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
    if let Some(mut cp) = pending_cp {
        cp.final_archive = Some(Archive {
            items: final_archive.items.clone(),
            cap: final_archive.cap,
        });
        let io_mark = std::time::Instant::now();
        emit(&mut on_checkpoint, &cp);
        io_final_s += io_mark.elapsed().as_secs_f64();
    }

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

    // #284: the exact finalist refit is a fit-stage leaf, and the remaining tail
    // (test scoring, report bookkeeping) is a measured host window.
    let gpu_fit_refits = refit_report.refits;
    let refit_tail_s = refit_report.refit_s;
    run_ledger.record_stage(crate::gpu_stats::Stage::Fit, refit_tail_s);
    run_ledger.record_refit(gpu_fit_refits as u64, refit_tail_s);
    let tail_total = (t0.elapsed().as_secs_f64() - pre_tail_elapsed).max(0.0);
    run_ledger.record_stage(crate::gpu_stats::Stage::Io, io_final_s);
    run_ledger.record_stage(
        crate::gpu_stats::Stage::HostOther,
        (tail_total - refit_tail_s - io_final_s).max(0.0),
    );
    run_ledger.set_wall(t0.elapsed().as_secs_f64());

    // #131: the GPU report is emitted only for a run that actually used the
    // device, so a CPU run's JSON is byte-identical to before.
    #[cfg(feature = "cuda")]
    let gpu_stats = if gpu_used || fit_only_device {
        gpu.as_ref().map(|g| {
            let mut s = g.run_stats();
            s.fit_refits = gpu_fit_refits;
            s.fit_refit_candidates = refit_report.candidates;
            s.fit_refit_s = refit_report.refit_s;
            s.fit_rescore_s = refit_report.rescore_s;
            s.fit_admission = cfg.fit_admission;
            s.fit_polish = cfg.fit_polish;
// #284: absorb the device intervals and fit leaves, then add the
            // exclusive main-thread stages. Fit is a sub-window of the measured
            // eval window, so it is partitioned out exactly once.
            let mut ledger = run_ledger.clone();
            ledger.absorb(&s.ledger);
            // Fit is a sub-window of the measured eval window. The part before
            // the loop (generation 0) is already booked as Fit; only the loop's
            // share is reclassified, so the stage sum never grows.
            let fit_loop_s = ((s.fit_host_ms - g.drain_fit_ms) / 1000.0 - setup_fit_s).max(0.0);
            ledger.reclassify_eval_to_fit(fit_loop_s);
            ledger.set_wall(t0.elapsed().as_secs_f64());
            s.ledger = ledger;
            s.fit_fallback_max = cfg.fit_fallback_max;
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
        converged_early: reported_converged_early(converged_early, &convergence),
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
        backend_reason: if backend_reason.is_none() && dispatcher.ledger().cost_model_kept_cpu() {
            Some("dispatch: cost model kept every stage on the cpu".to_string())
        } else {
            backend_reason
        },
        device: if gpu_used || fit_only_device {
            backend_device
        } else {
            None
        },
        gpu: gpu_stats,
        // #290: calibrated model plus the per-stage route ledger.
        dispatch: dispatcher.snapshot(),
        // #120: coverage of the descriptor grid for this run.
        qd: qd_archive.as_ref().map(|q| q.report()),
        stop_reason, // #211
        prune: prune_report,
        candidates_evaluated: candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #453: `n` eligible `c0 * x0 + c1` plans, one constant pair each.
    fn affine_plans(n: usize) -> Vec<(usize, usize, crate::fit::GpuFitPlan)> {
        use crate::expr::{constant, input, node, Op};
        (0..n)
            .map(|k| {
                let tree = node(
                    Op::Add,
                    vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.5)],
                );
                (0, k, crate::fit::gpu_fit_plan_reason(&tree).expect("an eligible plan"))
            })
            .collect()
    }

    /// A launch so expensive that one fit cannot amortise it but 64 can.
    fn costly_launch() -> crate::dispatch::Dispatcher {
        let mut cal = crate::dispatch::Calibration::default();
        cal.launch_s = 5.0;
        crate::dispatch::Dispatcher::new(cal)
    }

    /// #453: one candidate alone routes to the CPU, 64 together route to the
    /// device, and the admission sends all 64 to the device.
    #[test]
    fn fit_batch_plans_the_batch_not_each_candidate() {
        let rows = 2_456_306;
        let alone = plan_fit_batch(&mut costly_launch(), rows, 8, None, affine_plans(1));
        assert!(alone.jobs.is_empty(), "a batch of one pays the whole launch");
        assert_eq!(alone.refused.len(), 1);
        assert_eq!(alone.refused[0].2, FitFallbackCause::DispatchCost);

        let batch = plan_fit_batch(&mut costly_launch(), rows, 8, None, affine_plans(64));
        assert_eq!(batch.jobs.len(), 64);
        assert!(batch.refused.is_empty());
        let decision = batch.decision.expect("a batch decision");
        assert_eq!(decision.route, crate::dispatch::Route::Device);
    }

    /// #453: a CPU batch labels every candidate `dispatch_cost` and the ledger
    /// holds one fit call carrying all of them.
    #[test]
    fn fit_batch_ledger_has_one_decision_with_all_candidates() {
        let rows = 2_456_306;
        let mut d = crate::dispatch::Dispatcher::new(crate::dispatch::Calibration {
            launch_s: 1.0e9,
            ..crate::dispatch::Calibration::default()
        });
        let cpu = plan_fit_batch(&mut d, rows, 8, None, affine_plans(64));
        assert_eq!(cpu.refused.len(), 64);
        assert!(cpu.refused.iter().all(|r| r.2 == FitFallbackCause::DispatchCost));
        let entry = d.ledger().entry(crate::dispatch::Stage::Fit).expect("fit row");
        assert_eq!((entry.calls, entry.candidates), (1, 64));

        let mut d = costly_launch();
        let batch = plan_fit_batch(&mut d, rows, 8, None, affine_plans(64));
        d.record(batch.decision.unwrap(), 0.0, batch.jobs.len(), batch.bytes);
        let entry = d.ledger().entry(crate::dispatch::Stage::Fit).expect("fit row");
        assert_eq!((entry.calls, entry.candidates), (1, 64));
    }

    /// #453: a candidate below its own `32 * nconst` row floor keeps its
    /// structural `too_few_rows` cause and does not join the batch.
    #[test]
    fn fit_batch_keeps_the_per_candidate_row_floor() {
        let batch = plan_fit_batch(&mut costly_launch(), 40, 8, None, affine_plans(3));
        assert!(batch.jobs.is_empty());
        assert_eq!(batch.refused.len(), 3);
        assert!(batch.refused.iter().all(|r| r.2 == FitFallbackCause::TooFewRows));
        assert!(batch.decision.is_none());
    }

    #[test]
    fn gpu_fit_off_reason_follows_the_fit_decision() {
        use crate::fit::FitStrategy;
        use crate::metrics::Metric;
        for &metric in Metric::all() {
            let cfg = Config {
                metric,
                fit_strategy: FitStrategy::Gpu,
                fit_iters: 5,
                ..Config::default()
            };
            let reason = gpu_fit_off_reason(&cfg);
            assert_eq!(reason.is_none(), defers_fit_to_gpu(&cfg, true), "{}", metric.name());
            if crate::fit::gpu_fit_metric_error(metric).is_none() {
                assert_eq!(reason, None, "{}", metric.name());
            } else {
                assert_eq!(reason, crate::fit::gpu_fit_metric_error(metric));
            }
        }
        let cfg = Config {
            metric: Metric::Squared,
            fit_strategy: FitStrategy::Gpu,
            fit_iters: 0,
            ..Config::default()
        };
        assert_eq!(
            gpu_fit_off_reason(&cfg).as_deref(),
            Some("--fit-iters 0: nothing to fit")
        );
        let cfg = Config { fit_strategy: FitStrategy::Exact, ..cfg };
        assert_eq!(gpu_fit_off_reason(&cfg), None);
    }

    /// #256: the fit stage leaves the eval window first, then prep, then
    /// search, and the phases keep their total.
    #[test]
    fn the_fit_window_is_taken_out_of_eval_first() {
        let (w, fit) = split_fit_window([5.5, 0.5, 1.0], 5.0);
        assert_eq!(fit, 5.0);
        assert_eq!(w, [0.5, 0.5, 1.0]);
        let (w, fit) = split_fit_window([5.5, 0.5, 1.0], 6.25);
        assert_eq!(fit, 6.25);
        assert_eq!(w, [0.0, 0.0, 0.75]);
        let (w, fit) = split_fit_window([1.0, 1.0, 1.0], 10.0);
        assert_eq!((w, fit), ([0.0, 0.0, 0.0], 3.0));
        assert_eq!(split_fit_window([1.0, 2.0, 3.0], f64::NAN).1, 0.0);
    }
    use crate::expr::{constant, input, node, Op};
    use crate::metrics::Metric;

    /// `y = x0` on a small grid, two inputs.
    fn sample() -> Sample {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..21 {
            let a = -1.0 + 2.0 * i as f64 / 20.0;
            x.push(vec![a, 0.5 * a + 1.0]);
            y.push(a);
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    /// `n` rows of a smooth one-variable target, for the large-sample gates.
    #[cfg(feature = "cuda")]
    fn sample_rows(n: usize) -> Sample {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            x.push(vec![a, 0.5 * a + 1.0]);
            y.push(a);
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    /// #288: the dispatch gate admits exactly the metrics with an exact
    /// order/rank scorer and names every refusal.
    #[cfg(feature = "cuda")]
    #[test]
    fn order_and_rank_metrics_pass_the_native_stat_gate() {
        use crate::metrics::Metric;
        let mut s = sample_rows(crate::fit::MEDIAN_GPU_MIN_ROWS);
        for metric in [
            Metric::Median,
            Metric::Interquartile,
            Metric::Maximum,
            Metric::RankCorrelation,
            Metric::RankCorrelationSigned,
        ] {
            let cfg = Config { metric, ..Config::default() };
            assert_eq!(
                gpu_blocker(&cfg, &s),
                None,
                "{} must be admitted",
                metric.name()
            );
        }
        // AUC is the one classification metric with a native exact scorer; the
        // midpoint rule needs a two-valued target.
        for (i, y) in s.y.iter_mut().enumerate() {
            *y = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let cfg = Config { metric: Metric::Auc, ..Config::default() };
        assert_eq!(gpu_blocker(&cfg, &s), None);
        s.y[0] = 2.0;
        s.y[1] = -2.0;
        assert_eq!(gpu_blocker(&cfg, &s), Some("auc needs a two-valued target"));
        // A metric outside the table keeps its named refusal, never MSE.
        let cfg = Config { metric: Metric::ImplicitDerivative, ..Config::default() };
        assert_eq!(
            gpu_blocker(&cfg, &s),
            crate::metrics::score_refusal(Metric::ImplicitDerivative)
        );
        assert!(gpu_blocker(&cfg, &s).is_some());
    }

    /// #288: below the measured crossover the exact CPU statistic stays, and
    /// the reason says so; the fit-only arm may still move constants.
    #[cfg(feature = "cuda")]
    #[test]
    fn order_and_rank_scoring_refuses_below_the_row_floor() {
        use crate::metrics::Metric;
        for (metric, stat) in [
            (Metric::Median, crate::metrics::ScoreStat::Median),
            (Metric::Interquartile, crate::metrics::ScoreStat::Interquartile),
            (Metric::RankCorrelation, crate::metrics::ScoreStat::RankCorrelation),
        ] {
            let cfg = Config { metric, ..Config::default() };
            let small = sample_rows(crate::fit::MEDIAN_GPU_MIN_ROWS - 1);
            assert_eq!(gpu_blocker(&cfg, &small), Some(stat.small_sample_reason()));
        }
        // #287: maximum is an objective-kernel row of the table, with no floor.
        let cfg = Config { metric: Metric::Maximum, ..Config::default() };
        assert_eq!(gpu_blocker(&cfg, &sample_rows(64)), None);
    }

    fn pending_ind(error: f64, complexity: f64) -> Ind {
        Ind {
            // One fittable constant, so admission does not filter the fixture
            // as a no-constant tree.
            tree: node(Op::Add, vec![input(0), constant(1.0)]),
            error,
            complexity,
            descriptor: None,
            age: 0,
        }
    }

    /// #169: the admission fraction is a pure function of the population size
    /// and the training row count, and widens to every candidate at large `N`.
    #[test]
    fn admission_count_is_the_documented_fraction() {
        assert_eq!(admission_count(1, false), 1);
        assert_eq!(admission_count(24, false), 24, "small islands admit everyone");
        assert_eq!(admission_count(64, false), 32, "max(32, pop/4)");
        assert_eq!(admission_count(100, false), 32);
        assert_eq!(admission_count(128, false), 32);
        assert_eq!(admission_count(256, false), 64);
        // 154k, the pinned fit-heavy cell: the documented fraction is unchanged.
        assert_eq!(admission_count(256, false), 64);
        // The 3.9M hourly cell: every candidate with constants is admitted.
        assert_eq!(admission_count(128, true), 128);
        assert_eq!(admission_count(256, true), 256);
        assert_eq!(admission_count(1, true), 1);
    }

    /// #223: a resolved `All` admission fits every candidate with constants at
    /// any row count; the default and an explicit `gpu` arm stay screened.
    #[test]
    fn admit_all_admits_every_candidate_with_constants() {
        let screened = Config { metric: Metric::Squared, ..Config::default() };
        let all = Config { fit_admission: crate::fit::FitAdmission::All, ..screened.clone() };
        for rows in [10, 1_500, 115_500, WIDE_ADMISSION_ROWS - 1] {
            assert!(!admits_all(&screened, rows), "{rows}");
            assert!(admits_all(&all, rows), "{rows}");
        }
        assert!(admits_all(&screened, WIDE_ADMISSION_ROWS));
        assert_eq!(admission_count(256, admits_all(&screened, 1_500)), 64);
        assert_eq!(admission_count(256, admits_all(&all, 1_500)), 256);
    }

    /// #223: with the CPU evaluator the resolver keeps the CPU arm the request
    /// names and labels why.
    #[test]
    fn resolver_keeps_the_cpu_arm_without_a_device() {
        use crate::fit::{FitRequest, FitStrategy};
        let sample = Sample::new(vec![vec![1.0], vec![2.0]], vec![1.0, 2.0]);
        let cpu = Config { metric: Metric::Squared, gpu: false, ..Config::default() };
        let r = resolve_fit_strategy(&cpu, &sample, FitRequest::Unset);
        assert_eq!((r.strategy, r.reason.as_str()), (FitStrategy::Exact, "--cpu/--no-gpu"));
        let r = resolve_fit_strategy(&cpu, &sample, FitRequest::Auto);
        assert_eq!((r.strategy, r.reason.as_str()), (FitStrategy::Auto, "--cpu/--no-gpu"));
        let abs = Config { metric: Metric::Absolute, ..cpu.clone() };
        let r = resolve_fit_strategy(&abs, &sample, FitRequest::Unset);
        assert_eq!(r.reason, "metric_not_least_squares");
        let nofit = Config { fit_iters: 0, ..cpu };
        let r = resolve_fit_strategy(&nofit, &sample, FitRequest::Unset);
        assert_eq!(r.reason, "no_fit");
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
        let a = select_promoted(&islands, &pending, false);
        let b = select_promoted(&islands, &pending, false);
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
        let got = select_promoted(&islands, &pending, false);
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
        let got = select_promoted(&islands, &pending, false);
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
        let got = select_promoted(&islands, &pending, wide_admission(Metric::Squared, WIDE_ADMISSION_ROWS));
        let wanted: Vec<(usize, usize)> = (0..128).filter(|j| j % 16 != 0).map(|j| (0, j)).collect();
        assert_eq!(
            got, wanted,
            "every fittable candidate must be admitted at large N"
        );
        // Just below the threshold the old fraction still applies.
        let narrow = select_promoted(&islands, &pending, wide_admission(Metric::Squared, WIDE_ADMISSION_ROWS - 1));
        assert_eq!(
            narrow.len(),
            32,
            "below the threshold the pop/4 floor is unchanged"
        );
    }

    /// #393: the deferred path simplifies and recomputes complexity after a fit
    /// write-back, exactly as the CPU path does.
    #[test]
    fn refresh_after_fit_matches_the_cpu_write_back() {
        let cfg = Config::default();
        let tree = node(
            Op::Add,
            vec![
                constant(0.0),
                node(Op::Mul, vec![constant(1.0), node(Op::Sin, vec![input(0)])]),
            ],
        );
        let before = cfg.complexity_model.complexity(&tree);
        let mut ind = Ind {
            tree: tree.clone(),
            error: f64::NAN,
            complexity: before,
            descriptor: None,
            age: 0,
        };
        refresh_after_fit(&mut ind, &cfg);
        let want = expr::simplify(&tree);
        assert_eq!(expr::structural_hash(&ind.tree), expr::structural_hash(&want));
        assert_eq!(ind.complexity, cfg.complexity_model.complexity(&want));
        assert!(ind.complexity < before, "dead 0+ and 1* nodes carry no complexity");
    }

    /// `--fit-strategy gpu --metric signed` is accepted and defers the fit: the
    /// kernel is the squared warm start of the CPU arm.
    #[test]
    fn signed_defers_to_the_device_as_a_squared_warm_start() {
        assert!(crate::fit::gpu_fit_metric_error(Metric::Signed).is_none());
        assert_eq!(
            crate::fit::fit_loss(Metric::Signed),
            Some(crate::fit::DeviceLoss::Squared)
        );
        let cfg = Config {
            fit_strategy: crate::fit::FitStrategy::Gpu,
            fit_iters: 8,
            metric: Metric::Signed,
            ..Default::default()
        };
        assert!(defers_fit_to_gpu(&cfg, true));
        assert!(!defers_fit_to_gpu(&Config { fit_iters: 0, ..cfg }, true));
    }

    /// #393: the metrics whose proposals need the exact gate defer without the
    /// ungated pipeline write-back, and are fitted for every candidate.
    #[test]
    fn gated_arms_defer_and_stay_out_of_the_pipeline() {
        for metric in [
            Metric::Correlation,
            Metric::CorrelationSigned,
            Metric::Signed,
            Metric::LogLoss,
        ] {
            assert!(crate::fit::device_fit_is_gated(metric), "{metric:?}");
            let cfg = Config {
                fit_strategy: crate::fit::FitStrategy::Gpu,
                fit_iters: 8,
                metric,
                pipeline_depth: 2,
                ..Default::default()
            };
            assert!(!pipeline_fit_safe(&cfg, WIDE_ADMISSION_ROWS, false), "{metric:?}");
        }
        for metric in [Metric::Squared, Metric::LogSquashed, Metric::Absolute] {
            assert!(!crate::fit::device_fit_is_gated(metric), "{metric:?}");
        }
    }

    /// #393: the gated arms fit every candidate with constants at any row count,
    /// like the CPU arm; squared keeps `max(32, pop/4)` below the threshold.
    #[test]
    fn gated_arms_admit_every_candidate_with_constants() {
        let mut islands = vec![Vec::new()];
        for j in 0..128 {
            let mut ind = pending_ind(1.0 + j as f64, 1.0);
            if j % 16 == 0 {
                ind.tree = input(0);
            }
            islands[0].push(ind);
        }
        let pending: Vec<(usize, usize)> = (0..128).map(|j| (0, j)).collect();
        let wanted: Vec<(usize, usize)> =
            (0..128).filter(|j| j % 16 != 0).map(|j| (0, j)).collect();
        let rows = WIDE_ADMISSION_ROWS - 1;
        for metric in [Metric::Correlation, Metric::CorrelationSigned, Metric::Signed] {
            assert!(crate::fit::admits_every_candidate(metric));
            let got = select_promoted(&islands, &pending, wide_admission(metric, rows));
            assert_eq!(got, wanted, "{metric:?}");
        }
        let squared = select_promoted(&islands, &pending, wide_admission(Metric::Squared, rows));
        assert_eq!(squared.len(), 32);
        assert!(!crate::fit::admits_every_candidate(Metric::LogLoss));
    }

    /// #265: a forced fallback refit that lowers the exact error counts as
    /// improved; one that starts at the optimum does not.
    #[test]
    fn fallback_refit_counts_only_a_lowered_error() {
        let s = sample();
        let line = |a: f64| node(Op::Mul, vec![constant(a), input(0)]);
        let mut bad = line(0.25);
        assert!(refit_lowers_error(&mut bad, &s.x, &s.y, &s.w, 8, Metric::Squared));
        let mut exact = line(1.0);
        assert!(!refit_lowers_error(&mut exact, &s.x, &s.y, &s.w, 8, Metric::Squared));
    }

    /// #265: improvements are tallied under the cause that sent the candidate
    /// to the CPU arm, and a non-improving fallback adds nothing.
    #[test]
    fn improved_fallbacks_are_counted_per_cause() {
        let s = sample();
        let mut map = BTreeMap::new();
        let runs = [
            (FitFallbackCause::IllConditioned, 0.25),
            (FitFallbackCause::IllConditioned, 1.0),
            (FitFallbackCause::NonFinite, 0.5),
            (FitFallbackCause::IllConditioned, -2.0),
        ];
        for (cause, start) in runs {
            let mut t = node(Op::Mul, vec![constant(start), input(0)]);
            if refit_lowers_error(&mut t, &s.x, &s.y, &s.w, 8, Metric::Squared) {
                count_improved(&mut map, cause);
            }
        }
        assert_eq!(map.get("ill_conditioned"), Some(&2));
        assert_eq!(map.get("non_finite"), Some(&1));
        assert_eq!(map.values().sum::<usize>(), 3);
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
                "metric_not_native",
                "ill_conditioned",
                "non_finite",
                "launch",
                "rank_declined",
                "median_declined",
                "interquartile_declined",
                "single_class",
                "log_loss_worse",
                "maximum_declined",
                "dispatch_cost",
                "worse",
                "not_identifiable",
                "affine_closed_form",
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
            status_fallback_cause(3),
            FitFallbackCause::RankDeclined
        );
        assert_eq!(
            status_fallback_cause(0),
            FitFallbackCause::UnsupportedPlan
        );
        // #226/#232: each order-statistic arm's declined polish has its own
        // label and never borrows the rank or the other statistic's one.
        let median = crate::fit::OrderStat::Median;
        let iqr = crate::fit::OrderStat::Interquartile;
        assert_eq!(
            order_stat_status_fallback_cause(3, median),
            FitFallbackCause::MedianDeclined
        );
        assert_eq!(
            order_stat_status_fallback_cause(3, median).label(),
            "median_declined"
        );
        assert_eq!(
            order_stat_status_fallback_cause(3, iqr),
            FitFallbackCause::InterquartileDeclined
        );
        assert_eq!(
            order_stat_status_fallback_cause(3, iqr).label(),
            "interquartile_declined"
        );
        assert_ne!(
            order_stat_status_fallback_cause(3, iqr),
            order_stat_status_fallback_cause(3, median),
            "the two declined causes must stay distinguishable"
        );
        assert_eq!(
            order_stat_status_fallback_cause(1, iqr),
            FitFallbackCause::IllConditioned
        );
        assert_eq!(
            order_stat_status_fallback_cause(2, iqr),
            FitFallbackCause::NonFinite
        );
        assert_eq!(
            order_stat_status_fallback_cause(1, median),
            FitFallbackCause::IllConditioned
        );
        assert_eq!(
            order_stat_status_fallback_cause(2, median),
            FitFallbackCause::NonFinite
        );
        // #227: the maximum arm's declined polish has its own label too.
        assert_eq!(
            maximum_status_fallback_cause(3),
            FitFallbackCause::MaximumDeclined
        );
        assert_eq!(
            maximum_status_fallback_cause(3).label(),
            "maximum_declined"
        );
        assert_eq!(
            maximum_status_fallback_cause(1),
            FitFallbackCause::IllConditioned
        );
        assert_eq!(
            maximum_status_fallback_cause(2),
            FitFallbackCause::NonFinite
        );
    }

    /// #286: the overlap queue may carry the GPU fit arm only where serial
    /// admission already promotes every candidate with constants. Below that
    /// row floor, and for the metric-specific gated arms, the serial batch is
    /// the only trajectory-preserving schedule.
    #[test]
    fn pipeline_fit_only_where_admission_is_unconditional() {
        let cfg = Config {
            fit_strategy: crate::fit::FitStrategy::Gpu,
            fit_iters: 8,
            metric: Metric::Squared,
            pipeline_depth: 2,
            ..Default::default()
        };
        assert!(!pipeline_fit_safe(&cfg, WIDE_ADMISSION_ROWS - 1, false));
        assert!(pipeline_fit_safe(&cfg, WIDE_ADMISSION_ROWS, false));
        assert!(!pipeline_fit_safe(&cfg, WIDE_ADMISSION_ROWS, true));
        assert!(!pipeline_fit_safe(&Config { pipeline_depth: 0, ..cfg.clone() }, WIDE_ADMISSION_ROWS, false));
        assert!(!pipeline_fit_safe(
            &Config { metric: Metric::LogLoss, ..cfg.clone() },
            WIDE_ADMISSION_ROWS,
            false
        ));
        assert!(!pipeline_fit_safe(
            &Config { metric: Metric::RankCorrelation, ..cfg },
            WIDE_ADMISSION_ROWS,
            false
        ));
    }

    /// #286: statuses 4–8 are labels on a finite answer, never CPU fallbacks.
    #[test]
    fn finite_fit_statuses_are_labels_not_fallbacks() {
        for status in [4, 5, 6, 7, 8] {
            assert!(crate::fit::fit_status_usable(status));
            assert!(gpu_status_label(status).is_some(), "status {status}");
        }
        assert!(crate::fit::fit_status_usable(0));
        assert!(gpu_status_label(0).is_none());
        assert!(!crate::fit::fit_status_usable(1));
        assert!(!crate::fit::fit_status_usable(2));
        assert_eq!(gpu_status_label(4), Some("rank_deficient"));
        assert_eq!(gpu_status_label(5), Some("recovered"));
        assert_eq!(gpu_status_label(6), Some("no_improvement"));
        assert_eq!(gpu_status_label(7), Some("start_unusable"));
    }

    /// #236/#508: the device slope scorer differences row r against row r - 1
    /// under a per-row pair mask, so a pair list of any other shape keeps the
    /// CPU with a named reason.
    #[cfg(feature = "cuda")]
    #[test]
    fn slope_device_admits_only_consecutive_pairs() {
        let mut s = sample();
        let n = s.len();
        let cfg = Config { metric: Metric::Slope, ..Config::default() };
        assert_eq!(gpu_blocker(&cfg, &s), None, "no pair list is the consecutive set");
        s.pairs = Some(crate::fit::consecutive_pairs(n));
        assert_eq!(gpu_blocker(&cfg, &s), None);
        let reason = Some("slope pairs are not the consecutive rows of the training sample");
        let mut boundary = crate::fit::consecutive_pairs(n);
        boundary.remove(9);
        s.pairs = Some(boundary);
        assert_eq!(gpu_blocker(&cfg, &s), None, "#508: a group boundary is a masked pair");
        let mask = slope_pair_mask(&s).expect("a masked pair list");
        assert_eq!(mask.pairs, n - 2);
        assert_eq!(mask.bits[0] & (1 << 10), 0, "row 10 has no pair");
        assert_eq!(mask.bits[0] & (1 << 9), 1 << 9);
        assert_eq!(mask.bits[0] & 1, 0, "row 0 never pairs");
        let mut shuffled = crate::fit::consecutive_pairs(n);
        shuffled[3] = [2, 5];
        s.pairs = Some(shuffled);
        assert_eq!(gpu_blocker(&cfg, &s), reason, "a non-adjacent pair");
        let mut doubled = crate::fit::consecutive_pairs(n);
        doubled[4] = doubled[5];
        s.pairs = Some(doubled);
        assert_eq!(gpu_blocker(&cfg, &s), reason, "a repeated pair");
        let mut reversed = crate::fit::consecutive_pairs(n);
        reversed[0] = [1, 0];
        s.pairs = Some(reversed);
        assert_eq!(gpu_blocker(&cfg, &s), reason, "a backward pair");
        s.pairs = Some(crate::fit::consecutive_pairs(n));
        let subsample = Config {
            fit_rows: 10,
            fit_strategy: crate::fit::FitStrategy::Gpu,
            fit_iters: 4,
            ..cfg.clone()
        };
        assert_eq!(
            gpu_blocker(&subsample, &s),
            Some("slope pairs are not the consecutive rows of a --fit-rows/--fit-samples subsample")
        );
        s.w = vec![1.0; n];
        assert!(gpu_blocker(&cfg, &s).is_some(), "slope has no weighted form");
    }

    /// #229: a metric without a native device objective is labelled
    /// `metric_not_native`, and the gate keeps such a run off the fit path in
    /// the first place. `log_squashed` passes because the kernel fits it.
    #[test]
    fn the_fit_gate_admits_only_native_device_objectives() {
        assert!(FitFallbackCause::ALL.contains(&FitFallbackCause::MetricNotNative));
        assert_eq!(FitFallbackCause::MetricNotNative.label(), "metric_not_native");
        let gpu_fit = Config {
            fit_strategy: crate::fit::FitStrategy::Gpu,
            fit_iters: 8,
            metric: Metric::LogSquashed,
            ..Default::default()
        };
        assert!(defers_fit_to_gpu(&gpu_fit, true), "log_squashed is native");
        let absolute = Config {
            metric: Metric::Absolute,
            ..gpu_fit.clone()
        };
        assert!(defers_fit_to_gpu(&absolute, true), "#222: absolute is native");
        for metric in [Metric::Squared, Metric::Aic, Metric::Aicc, Metric::Bic] {
            let cfg = Config {
                metric,
                ..gpu_fit.clone()
            };
            assert!(defers_fit_to_gpu(&cfg, true), "{metric:?} fits by least squares");
        }
        // #236: slope fits on the device from the squared warm start.
        let slope = Config {
            metric: Metric::Slope,
            ..gpu_fit.clone()
        };
        assert!(defers_fit_to_gpu(&slope, true), "#236: slope is native");
        // #430: the correlation family defers its fit to the device.
        let correlation = Config {
            metric: Metric::Correlation,
            ..gpu_fit.clone()
        };
        assert!(defers_fit_to_gpu(&correlation, true), "#430: correlation is native");
        let implicit = Config {
            metric: Metric::ImplicitDerivative,
            ..gpu_fit.clone()
        };
        assert!(
            !defers_fit_to_gpu(&implicit, true),
            "implicit has no native device loss"
        );
        // #226: the median arm has its own exact device polish, so the fit is
        // deferred exactly like the rank family's.
        let median = Config {
            metric: Metric::Median,
            ..gpu_fit.clone()
        };
        assert!(defers_fit_to_gpu(&median, true), "median has an exact device polish");
        // #227: the maximum arm has its own exact device polish.
        let maximum = Config {
            metric: Metric::Maximum,
            ..gpu_fit.clone()
        };
        assert!(
            defers_fit_to_gpu(&maximum, true),
            "maximum has an exact device polish"
        );
        // `--no-fit` never reaches the kernel, whatever the metric.
        let no_fit = Config {
            fit_iters: 0,
            ..gpu_fit.clone()
        };
        assert!(!defers_fit_to_gpu(&no_fit, true));
        assert!(!defers_fit_to_gpu(&no_fit, false));
    }

    /// #169: the kernel's constant cap is 20, so the `too_many_constants`
    /// boundary is `p = 21`. Nothing else is labelled with that cause: `p <= 20`
    /// must be offered to the kernel instead of silently falling back.
    #[test]
    fn the_gpu_fit_constant_cap_is_twenty_at_the_search_boundary() {
        fn many_constants(p: usize) -> crate::expr::Node {
            let mut t = node(Op::Mul, vec![constant(0.1), input(0)]);
            for k in 1..p {
                t = node(
                    Op::Add,
                    vec![
                        t,
                        node(Op::Mul, vec![constant(0.1 + 0.01 * k as f64), input(k)]),
                    ],
                );
            }
            t
        }
        assert_eq!(crate::fit::GPU_FIT_MAX_CONSTANTS, 20);
        for p in 17..=20usize {
            assert!(
                crate::fit::gpu_fit_plan_reason(&many_constants(p)).is_ok(),
                "p={} must be fitted on the device, not silently fallen back",
                p
            );
        }
        for p in 21..=22usize {
            let reason = match crate::fit::gpu_fit_plan_reason(&many_constants(p)) {
                Err(reason) => reason,
                Ok(_) => panic!("p={} has no kernel instantiation", p),
            };
            assert_eq!(plan_fallback_cause(reason), FitFallbackCause::TooManyConstants);
        }
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

    /// #475: scoring many candidates on one thread reuses one prediction
    /// buffer instead of allocating a vector per candidate.
    #[test]
    fn scoring_candidates_reuses_one_prediction_buffer() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.simd = true;
        let view = EvalSample::new(&s, &cfg);
        let trees = [
            node(Op::Add, vec![input(0), constant(1.0)]),
            node(Op::Mul, vec![input(0), input(0)]),
            input(0),
            constant(2.0),
        ];
        let expected: Vec<u64> = trees
            .iter()
            .map(|t| eval_error_on(t, &view, Metric::Squared).to_bits())
            .collect();
        PRED_ALLOCS.with(|n| n.set(0));
        for _ in 0..25 {
            for (t, want) in trees.iter().zip(&expected) {
                assert_eq!(eval_error_on(t, &view, Metric::Squared).to_bits(), *want);
            }
        }
        assert_eq!(PRED_ALLOCS.with(|n| n.get()), 0, "the buffer is already sized");
        let fresh = std::thread::spawn({
            let s = s.clone();
            move || {
                let cfg = Config::default();
                let view = EvalSample::new(&s, &cfg);
                for _ in 0..25 {
                    let _ = eval_error_on(&input(0), &view, Metric::Squared);
                }
                PRED_ALLOCS.with(|n| n.get())
            }
        })
        .join()
        .unwrap();
        assert_eq!(fresh, 1, "one buffer per worker thread");
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
            age: 0,
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
                None,  // #211: no development log in this test
                1,     // #463
                &mut MixWindow::new(), // #471
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
        let train = Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None };
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

    fn stop_run(metric: crate::metrics::Metric, law: bool) -> Outcome {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..40 {
            let a = -1.0 + 2.0 * i as f64 / 39.0;
            x.push(vec![a]);
            y.push(if law { 2.0 * a + 1.0 } else { a });
        }
        let train = Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None };
        let cfg = Config {
            metric,
            generations: 6,
            pop: 30,
            islands: 1,
            seed: 5,
            patience: 0,
            gpu: false,
            ..Config::default()
        };
        run(&cfg, &train, &train.clone())
    }

    /// #462: the front observer sees every generation and cannot change the
    /// trajectory.
    #[test]
    fn front_observer_does_not_change_the_archive() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..40 {
            let a = -1.0 + 2.0 * i as f64 / 39.0;
            x.push(vec![a]);
            y.push((2.0 * a).sin() + a);
        }
        let train = Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None };
        let plain = Config {
            generations: 8,
            pop: 40,
            islands: 2,
            seed: 9,
            patience: 0,
            gpu: false,
            ..Config::default()
        };
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let last = std::sync::Arc::new(AtomicUsize::new(0));
        let (c2, l2) = (calls.clone(), last.clone());
        let observed = Config {
            front_observer: Some(std::sync::Arc::new(move |g, secs, archive: &Archive| {
                assert!(secs >= 0.0 && !archive.items.is_empty());
                c2.fetch_add(1, Ordering::SeqCst);
                l2.store(g, Ordering::SeqCst);
            })),
            ..plain.clone()
        };
        let a = run(&plain, &train, &train.clone());
        let b = run(&observed, &train, &train.clone());
        assert_eq!(a.archive.hash(), b.archive.hash());
        assert_eq!(calls.load(Ordering::SeqCst), b.generations);
        assert_eq!(last.load(Ordering::SeqCst), b.generations);
        assert!(plain.search_setting_differences(&observed).is_empty());
    }

    /// #402: a negative information criterion is a good fit, not a solved run.
    #[test]
    fn information_criteria_run_every_generation() {
        for metric in [Metric::Aic, Metric::Aicc, Metric::Bic] {
            let out = stop_run(metric, true);
            assert_eq!(out.stop_reason, "generations", "{}", metric.name());
            assert_eq!(out.generations, 6, "{}", metric.name());
        }
    }

    /// #402: squared and absolute still stop when the error reaches zero.
    #[test]
    fn squared_and_absolute_still_stop_at_zero_error() {
        for metric in [Metric::Squared, Metric::Absolute] {
            let out = stop_run(metric, false);
            assert_eq!(out.stop_reason, "solved", "{}", metric.name());
            assert!(out.generations < 6, "{}", metric.name());
        }
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
            Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
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
            age: 0,
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
            pairs: None,
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

    /// #480: under the long/flat fitness a squared fit must not hand its SSE
    /// to the score, and the score is the regret of the same predictions.
    fn pnl_book(s: &Sample) -> std::sync::Arc<crate::trading::FitnessBook> {
        let rows = crate::trading::TradingRows {
            time: (0..s.len()).map(|i| 900.0 * i as f64).collect(),
            group: vec![0; s.len()],
            funding: Vec::new(),
        };
        let slice = crate::data::TradingSlice { rows, ret: s.y.clone() };
        let spec = crate::trading::TradingSpec {
            cost_bps: 1.0,
            min_trades: 0,
            ..Default::default()
        };
        std::sync::Arc::new(crate::trading::FitnessBook::new(&slice, &spec, 900.0).unwrap())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn long_pnl_fitness_names_why_the_device_is_off() {
        let s = sample();
        // #481: the device scores the regret; the metric only fits, so a
        // metric without a native scorer does not block it.
        let cfg = Config {
            fitness: Fitness::LongPnl(pnl_book(&s)),
            metric: Metric::Median,
            ..Default::default()
        };
        assert_eq!(gpu_blocker(&cfg, &s), None);
        assert_eq!(search_score_objective(&cfg), Some(crate::metrics::ScoreObjective::LongPnl));
        let shape = gpu_plan_shape(&cfg, &s);
        assert!(shape.score_stat.is_none() && !shape.score_admitted && !shape.fit_only);
        let saved = Config {
            fitness: Fitness::SavedLongPnl(7),
            ..Default::default()
        };
        assert_eq!(gpu_blocker(&saved, &s), Some("the long_pnl fitness book was not rebuilt"));
        assert_eq!(search_score_objective(&saved), None);
        let other_rows = Config {
            fitness: Fitness::LongPnl(pnl_book(&sample_rows(s.len() + 3))),
            ..Default::default()
        };
        assert_eq!(
            gpu_blocker(&other_rows, &s),
            Some("the long_pnl fitness book does not cover the training rows")
        );
    }

    #[test]
    fn long_pnl_fitness_never_reuses_the_fit_sse() {
        let s = sample(); // y = x0
        let book = pnl_book(&s);
        let cfg = Config {
            metric: Metric::Squared,
            fit_iters: 8,
            fitness: Fitness::LongPnl(book.clone()),
            ..Default::default()
        };
        let view = EvalSample::new(&s, &cfg);
        let tree = node(
            Op::Sin,
            vec![node(Op::Mul, vec![constant(1.0), input(0)])],
        );
        let before = REFIT_SSE_REUSED.with(|n| n.get());
        let sol = refit_survivor(&tree, &view, &cfg);
        assert_eq!(REFIT_SSE_REUSED.with(|n| n.get()), before);
        let pred = predictions(&sol.tree, &view);
        assert_eq!(sol.error.to_bits(), book.regret(&pred).to_bits());
        assert!(sol.error < 0.5, "x0 is the return, the fitted sign rule should be near the bound: {}", sol.error);
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
            pairs: None,
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
            pairs: None,
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

    fn perms3<T: Clone>(v: &[T; 3]) -> Vec<[T; 3]> {
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
            .iter()
            .map(|p| [v[p[0]].clone(), v[p[1]].clone(), v[p[2]].clone()])
            .collect()
    }

    fn fake_sol(cost: f64, error: f64, k: f64) -> Sol {
        Sol {
            tree: node(Op::Add, vec![input(0), constant(k)]),
            error,
            complexity: cost as usize,
            cost,
        }
    }

    fn item_bits(a: &Archive) -> Vec<(u64, u64, u64)> {
        a.items
            .iter()
            .map(|s| (s.cost.to_bits(), s.error.to_bits(), expr::structural_hash(&s.tree)))
            .collect()
    }

    /// #330: a non-transitive epsilon-dominance triple gives one front for all
    /// six arrival orders on the rebuild path, while plain insertion does not.
    #[test]
    fn rebuild_front_is_order_independent_for_a_non_transitive_triple() {
        // a dominates b and b dominates c, but a does not dominate c.
        let triple = [
            fake_sol(1.0, 1.000_003_0, 1.0),
            fake_sol(2.0, 1.000_001_5, 2.0),
            fake_sol(3.0, 1.000_000_0, 3.0),
        ];
        let mut inserted = std::collections::BTreeSet::new();
        let mut rebuilt: Option<(Vec<(u64, u64, u64)>, u64)> = None;
        for order in perms3(&triple) {
            let mut plain = Archive::new(100);
            for s in order.iter().cloned() {
                plain.try_insert(s);
            }
            inserted.insert(item_bits(&plain));
            let front = canonical_front(100, order.to_vec());
            let got = (item_bits(&front), front.hash());
            match &rebuilt {
                None => rebuilt = Some(got),
                Some(first) => assert_eq!(first, &got),
            }
        }
        assert!(inserted.len() > 1, "the triple must expose the order dependence");
        assert_eq!(rebuilt.unwrap().0.len(), 2);
    }

    /// #495: the envelope is the lowest finite error at no greater cost; a tree
    /// is selected when it is within the relative band of it.
    #[test]
    fn refit_band_selection_uses_the_lower_envelope() {
        let costs = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let pre = [10.0, 8.0, 8.02, 8.5, f64::NAN, f64::INFINITY];
        assert_eq!(refit_band_selection(&pre, &costs, 0.001), vec![0, 1]);
        assert_eq!(refit_band_selection(&pre, &costs, 0.01), vec![0, 1, 2]);
        assert_eq!(refit_band_selection(&pre, &costs, 0.0), vec![0, 1]);
        // A tree cheaper than the others sets the envelope for the dearer ones.
        let costs = [2.0, 1.0];
        assert_eq!(refit_band_selection(&[5.0, 1.0], &costs, 0.5), vec![1]);
        // A negative error (aic) is its own envelope.
        assert_eq!(refit_band_selection(&[-4.0, -3.0], &[1.0, 2.0], 0.001), vec![0]);
    }

    /// #495: a tree outside the band that enters the front after the band
    /// refits is refit before it is reported.
    #[test]
    fn fixpoint_refits_every_reported_member() {
        let scored = vec![
            fake_sol(1.0, 1.0, 1.0),
            fake_sol(2.0, 2.0, 2.0),
            fake_sol(3.0, 2.5, 3.0),
        ];
        let costs: Vec<f64> = scored.iter().map(|s| s.cost).collect();
        let pre: Vec<f64> = scored.iter().map(|s| s.error).collect();
        let first = refit_band_selection(&pre, &costs, 0.001);
        assert_eq!(first, vec![0], "only the cheapest tree is inside the band");
        let mut seen: Vec<Vec<u64>> = Vec::new();
        // The refit of the first tree worsens its error (a simplification that
        // changes the value), which uncovers the second.
        let (front, refits) = refit_to_fixpoint(10, scored, first, |batch| {
            seen.push(batch.iter().map(expr::structural_hash).collect());
            batch
                .into_iter()
                .map(|tree| {
                    let k = tree.kids[1].val;
                    let error = if k == 1.0 { 3.0 } else { 2.0 };
                    let tree = node(Op::Add, vec![input(0), constant(k + 0.5)]);
                    Sol { tree, error, complexity: 1, cost: k }
                })
                .collect()
        });
        assert_eq!(refits, 2);
        assert_eq!(seen.len(), 2, "the second round refits the exposed member");
        assert_eq!(front.items.len(), 2);
        let refit_ks: Vec<f64> = front.items.iter().map(|s| s.tree.kids[1].val).collect();
        assert_eq!(refit_ks, vec![1.5, 2.5], "every reported member is a refit");
    }

    fn refit_band_trees() -> Vec<Node> {
        let lin = |c: f64, d: f64| {
            node(Op::Add, vec![node(Op::Mul, vec![constant(c), input(0)]), constant(d)])
        };
        vec![
            input(0),
            node(Op::Mul, vec![constant(1.7), input(0)]),
            lin(1.9, 0.4),
            lin(2.0, 0.6),
            lin(0.3, -1.0),
            node(Op::Add, vec![input(0), constant(0.2)]),
            node(Op::Mul, vec![constant(0.4), input(1)]),
            node(Op::Sin, vec![node(Op::Mul, vec![constant(1.2), input(0)])]),
            node(Op::Add, vec![lin(2.0, 0.5), node(Op::Mul, vec![constant(0.01), input(1)])]),
            node(Op::Mul, vec![input(0), input(1)]),
        ]
    }

    fn refit_band_sample() -> Sample {
        let mut s = sample();
        for (row, y) in s.x.iter().zip(s.y.iter_mut()) {
            *y = 2.0 * row[0] + 0.5;
        }
        s
    }

    fn rebuild_with_band(band: f64) -> (Archive, RefitReport) {
        let s = refit_band_sample();
        let cfg = Config { refit_band: band, threads: 1, ..Config::default() };
        let view = EvalSample::new(&s, &cfg);
        let mut a = Archive::new(100);
        for t in refit_band_trees() {
            a.items.push(Sol {
                complexity: expr::complexity(&t),
                cost: cfg.complexity_model.complexity(&t),
                error: 0.0,
                tree: t,
            });
        }
        full_data_archive(&a, &[], &view, &cfg, true)
    }

    /// #495: `inf` is the historical path: refit every unique tree, score,
    /// take the canonical front.
    #[test]
    fn refit_band_inf_matches_the_full_refit() {
        let s = refit_band_sample();
        let cfg = Config { threads: 1, ..Config::default() };
        let view = EvalSample::new(&s, &cfg);
        let scored: Vec<Sol> = refit_band_trees()
            .into_iter()
            .map(|mut t| {
                crate::fit::fit_weighted_metric(
                    &mut t,
                    &view.sample.x,
                    &view.sample.y,
                    &view.sample.w,
                    cfg.fit_iters,
                    cfg.metric,
                );
                let t = expr::simplify(&t);
                let error = score_tree(&t, &view, &cfg);
                Sol {
                    complexity: expr::complexity(&t),
                    cost: cfg.complexity_model.complexity(&t),
                    error,
                    tree: t,
                }
            })
            .collect();
        let want = canonical_front(100, scored);
        let (got, report) = rebuild_with_band(f64::INFINITY);
        assert_eq!(item_bits(&got), item_bits(&want));
        assert_eq!(got.hash(), want.hash());
        assert_eq!(report.refits, report.candidates);
    }

    /// #495: the default band reports the same front costs and errors as the
    /// full refit with fewer refits, and every reported member is a refit.
    #[test]
    fn default_band_front_matches_the_full_refit() {
        let (want, full) = rebuild_with_band(f64::INFINITY);
        let (got, banded) = rebuild_with_band(DEFAULT_REFIT_BAND);
        assert_eq!(got.items.len(), want.items.len());
        for (g, w) in got.items.iter().zip(&want.items) {
            assert_eq!(g.cost.to_bits(), w.cost.to_bits());
            assert!((g.error - w.error).abs() <= 1e-9 * (1.0 + w.error.abs()), "{} vs {}", g.error, w.error);
        }
        assert_eq!(banded.candidates, full.candidates);
        assert!(banded.refits < full.refits, "{} of {}", banded.refits, full.refits);
        assert!(banded.refits >= got.items.len());
    }

    /// #495: the saved setting round-trips through the checkpoint config.
    #[test]
    fn refit_band_is_a_saved_search_setting() {
        let a = Config::default();
        let b = Config { refit_band: f64::INFINITY, ..Config::default() };
        assert_eq!(a.search_setting_differences(&b), vec!["refit_band"]);
        assert!(a.search_setting_differences(&a).is_empty());
    }

    /// #330: `full_data_archive` returns the same canonical archive hash for
    /// forward and reverse member order.
    #[test]
    fn full_data_archive_hash_is_independent_of_member_order() {
        let s = sample();
        let cfg = Config::default();
        let view = EvalSample::new(&s, &cfg);
        let trees = vec![
            input(0),
            node(Op::Add, vec![input(0), constant(0.0)]),
            node(Op::Mul, vec![input(0), constant(2.0)]),
            node(Op::Sin, vec![input(0)]),
            node(Op::Add, vec![input(1), constant(0.25)]),
            node(Op::Sub, vec![input(0), node(Op::Mul, vec![input(1), constant(1e-9)])]),
        ];
        let build = |order: &[Node]| {
            let mut a = Archive::new(100);
            for t in order {
                a.items.push(Sol {
                    tree: t.clone(),
                    error: 0.0,
                    complexity: expr::complexity(t),
                    cost: cfg.complexity_model.complexity(t),
                });
            }
            full_data_archive(&a, &[], &view, &cfg, false).0
        };
        let forward = build(&trees);
        let rev: Vec<Node> = trees.iter().rev().cloned().collect();
        let reverse = build(&rev);
        assert!(!forward.items.is_empty());
        assert_eq!(item_bits(&forward), item_bits(&reverse));
        assert_eq!(forward.hash(), reverse.hash());
        let mut again = Archive::new(forward.cap);
        again.items = forward.items.clone();
        again.canonicalize();
        assert_eq!(item_bits(&again), item_bits(&forward), "already canonical");
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

    /// #255: the per-generation bound is a deterministic prefix of the
    /// admission order: the first `cap` fallbacks are refit, the remainder are
    /// counted, and `0` means unbounded.
    #[test]
    fn fallback_batch_bound_is_a_deterministic_prefix() {
        assert_eq!(fallback_batch_split(0, 16), (0, 0));
        assert_eq!(fallback_batch_split(5, 16), (5, 0));
        assert_eq!(fallback_batch_split(16, 16), (16, 0));
        assert_eq!(fallback_batch_split(21, 16), (16, 5));
        assert_eq!(fallback_batch_split(21, 0), (21, 0));
        // The recorded cell's largest generation is far below the default, so
        // the bound never changes its result.
        assert_eq!(fallback_batch_split(540, DEFAULT_FIT_FALLBACK_MAX), (540, 0));
        for total in [1usize, 7, 64, 4096, 4097] {
            let (refit, bounded) = fallback_batch_split(total, DEFAULT_FIT_FALLBACK_MAX);
            assert_eq!(refit + bounded, total);
            assert_eq!(refit, total.min(DEFAULT_FIT_FALLBACK_MAX));
        }
    }

    /// #255: the exact CPU fallback batch is a pure parallelisation. One worker
    /// and eight workers must produce the same tree bits, in the same order,
    /// including the seeded constants the device already applied.
    #[cfg(feature = "cuda")]
    #[test]
    fn fit_fallback_is_identical_for_one_and_many_workers() {
        let s = sample();
        let cfg = Config {
            fit_iters: 8,
            metric: Metric::Squared,
            ..Config::default()
        };
        // Distinct starting constants and shapes, so every worker has real work.
        let trees: Vec<Node> = (0..24u32)
            .map(|k| {
                let a = (k as f64 + 1.0) / 7.0;
                let b = 1.0 + (k % 5) as f64;
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(a), input(0)]),
                        node(Op::Sin, vec![node(Op::Mul, vec![constant(b), input(1)])]),
                    ],
                )
            })
            .collect();
        let make = || -> Vec<FallbackFitJob> {
            trees
                .iter()
                .enumerate()
                .map(|(k, t)| FallbackFitJob::new(k % 2, k, t.clone()))
                .collect()
        };
        let mut one = make();
        let mut many = make();
        let used_one = fit_fallback_jobs(
            &mut one,
            &s.x,
            &s.y,
            &s.w,
            cfg.fit_iters,
            cfg.metric,
            1,
        );
        let used_many = fit_fallback_jobs(
            &mut many,
            &s.x,
            &s.y,
            &s.w,
            cfg.fit_iters,
            cfg.metric,
            8,
        );
        assert_eq!(used_one, 1);
        assert_eq!(used_many, 8);
        assert_eq!(one.len(), many.len());
        for (a, b) in one.iter().zip(&many) {
            assert_eq!((a.island, a.slot), (b.island, b.slot));
            assert!(
                expr::same(&a.tree, &b.tree),
                "worker result differs at island {} slot {}",
                a.slot,
                b.slot
            );
        }
        // The parallel result is also the serial reference for each candidate.
        for (k, job) in many.iter().enumerate() {
            let mut serial = trees[k].clone();
            crate::fit::fit_weighted_metric(
                &mut serial,
                &s.x,
                &s.y,
                &s.w,
                cfg.fit_iters,
                cfg.metric,
            );
            assert!(
                expr::same(&job.tree, &serial),
                "parallel fit differs from the serial arm at job {k}"
            );
        }
    }

    /// #498: the exact host gate is a pure parallelisation. One worker and
    /// eight workers return the same verdict list and the same trees, with a
    /// mix of accepted, rejected and non-finite proposals.
    #[cfg(feature = "cuda")]
    #[test]
    fn gate_is_identical_for_one_and_many_workers() {
        let mut s = sample();
        for (i, y) in s.y.iter_mut().enumerate() {
            *y = if i % 3 == 0 { 0.0 } else { 1.0 };
        }
        let jobs = |metric: Metric| -> Vec<(usize, Node, Vec<f64>)> {
            (0..64usize)
                .map(|k| {
                    let start = 0.2 + (k % 7) as f64 / 10.0;
                    let tree = node(
                        Op::Add,
                        vec![node(Op::Mul, vec![constant(start), input(0)]), constant(0.1)],
                    );
                    let proposal = match k % 4 {
                        0 => vec![start + 1.0, 0.1],
                        1 => vec![-3.0 * start, 0.1],
                        2 => vec![f64::NAN, 0.1],
                        _ => vec![start, 0.1 + (k as f64) / 50.0],
                    };
                    let _ = metric;
                    (k, tree, proposal)
                })
                .collect()
        };
        for metric in [Metric::LogLoss, Metric::Correlation, Metric::Signed] {
            let one = gate_proposals(jobs(metric), &s, metric, 1);
            let many = gate_proposals(jobs(metric), &s, metric, 8);
            assert_eq!(one.len(), 64);
            assert_eq!(one.len(), many.len());
            for ((ka, (va, ta)), (kb, (vb, tb))) in one.iter().zip(&many) {
                assert_eq!(ka, kb);
                assert_eq!(va, vb, "{:?} verdict differs at {}", metric, ka);
                assert!(expr::same(ta, tb), "{:?} tree differs at {}", metric, ka);
            }
            let accepted = one
                .iter()
                .filter(|(_, (v, _))| matches!(v, crate::fit::Verdict::Accepted(_)))
                .count();
            assert!(accepted > 0 && accepted < 64, "{:?}: {} accepted", metric, accepted);
            assert!(one.iter().any(|(_, (v, _))| *v == crate::fit::Verdict::NonFinite));
        }
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

    fn aged_ind(tree: Node, error: f64, complexity: f64, age: u32) -> Ind {
        Ind { tree, error, complexity, descriptor: None, age }
    }

    /// #465: the lowest-error individual always survives, whatever its age or
    /// size.
    #[test]
    fn afp_keeps_best_error() {
        let mut all: Vec<Ind> = Vec::new();
        for i in 0..8 {
            // Parents: old and mid-sized, errors 1.0 .. 8.0.
            all.push(aged_ind(input(0), 1.0 + i as f64, 3.0 + i as f64, 20));
        }
        for i in 0..8 {
            // Offspring: young and small, errors 2.0 .. 9.0 except one best.
            let error = if i == 5 { 0.25 } else { 2.0 + i as f64 };
            let complexity = if i == 5 { 40.0 } else { 1.0 };
            all.push(aged_ind(input(0), error, complexity, if i == 5 { 30 } else { 0 }));
        }
        let kept = afp_select(all, 8);
        assert_eq!(kept.len(), 8);
        let best = kept.iter().map(|i| i.error).fold(f64::INFINITY, f64::min);
        assert_eq!(best, 0.25);
    }

    /// #465: survivors age by one; offspring inherit the older parent's age;
    /// immigrants and rebuilt individuals start at 0.
    #[test]
    fn afp_age_rules() {
        let kept = afp_select(
            vec![aged_ind(input(0), 1.0, 1.0, 4), aged_ind(input(0), 2.0, 2.0, 0)],
            2,
        );
        assert_eq!(kept.iter().map(|i| i.age).collect::<Vec<_>>(), vec![5, 1]);

        let s = sample();
        let aggs = AggTable::from_sample(&s);
        let mut cfg = Config::default();
        cfg.pop = 40;
        cfg.fit_iters = 0;
        cfg.selection = Selection::Afp;
        let data = EvalSample::new(&s, &cfg);
        let mut isl: Vec<Ind> = (0..40)
            .map(|i| {
                let mut ind = Ind::evaluated(
                    node(Op::Add, vec![input(0), constant(i as f64)]),
                    &data,
                    &cfg,
                );
                ind.age = if i % 2 == 0 { 2 } else { 9 };
                ind
            })
            .collect();
        let mut rng = Mt::new(11);
        evolve_island(
            &mut isl, &mut rng, &cfg, &data, &s, 1, &aggs, false, false, &[], 0.0, None, false,
            0, None, None, 1, &mut MixWindow::new(),
        );
        let immigrants = afp_immigrants(cfg.pop);
        assert_eq!(isl.len(), cfg.pop + immigrants);
        let (offspring, fresh) = isl.split_at(cfg.pop);
        assert!(fresh.iter().all(|i| i.age == 0), "immigrants start at age 0");
        assert!(offspring.iter().all(|i| i.age == 2 || i.age == 9));
        assert!(offspring.iter().any(|i| i.age == 9), "the older parent's age is inherited");

        let rebuilt = Ind::pending(input(0), &cfg);
        assert_eq!(rebuilt.age, 0);
        assert_eq!(Ind::evaluated(input(0), &data, &cfg).age, 0);
    }

    /// #500: trees whose predictions are non-finite on every row, one of them
    /// with a subtree that `expr::simplify` rewrites, mixed with finite ones.
    fn inf_population(data: &EvalSample, cfg: &Config) -> Vec<Ind> {
        (0..40)
            .map(|i| {
                let tree = match i % 4 {
                    0 => node(Op::Div, vec![input(0), constant(0.0)]),
                    1 => node(
                        Op::Add,
                        vec![
                            node(Op::Div, vec![input(0), constant(0.0)]),
                            node(Op::Add, vec![constant(1.0), constant(2.0)]),
                        ],
                    ),
                    _ => node(Op::Add, vec![input(0), constant(i as f64)]),
                };
                Ind::evaluated(tree, data, cfg)
            })
            .collect()
    }

    fn same_state(a: &Ind, b: &Ind) -> bool {
        a.tree == b.tree
            && a.error.to_bits() == b.error.to_bits()
            && a.complexity.to_bits() == b.complexity.to_bits()
            && a.descriptor == b.descriptor
    }

    /// #500: probing a `+inf` individual leaves exactly the state a full
    /// re-score leaves, and runs no scoring pass for a tree that is non-finite
    /// on the probed rows.
    #[test]
    fn skipping_inf_reeval_keeps_the_cpu_trajectory() {
        let s = sample();
        let cfg = Config::default();
        let data = EvalSample::new(&s, &cfg);
        let mut ran = 0;
        for ind in inf_population(&data, &cfg) {
            let mut full = ind.clone();
            full.reeval(&data, &cfg);
            let mut probed = ind.clone();
            if probed.reeval_probing(&data, &cfg) {
                ran += 1;
            }
            assert!(same_state(&full, &probed));
        }
        assert_eq!(ran, 20, "only the finite trees are scored again");
    }

    /// #500: a `+inf` error that the exact CPU score does not reproduce (a
    /// device overflow) and a pending `NaN` are scored in full.
    #[test]
    fn inf_reeval_runs_when_the_prediction_is_finite() {
        let s = sample();
        let cfg = Config::default();
        let data = EvalSample::new(&s, &cfg);
        let mut spurious = Ind::evaluated(input(0), &data, &cfg);
        spurious.error = f64::INFINITY;
        assert!(spurious.reeval_probing(&data, &cfg));
        assert!(spurious.error.is_finite());
        let mut pending = Ind::evaluated(input(0), &data, &cfg);
        pending.error = f64::NAN;
        assert!(pending.reeval_probing(&data, &cfg));
        assert!(pending.error.is_finite());
    }

    /// #500: shapes whose `+inf` rule differs from the plain metrics take the
    /// full pass, and the island counts the passes it runs.
    #[test]
    fn inf_probe_is_limited_to_the_plain_metrics() {
        let s = sample();
        let mut weighted = sample();
        weighted.w = vec![1.0; weighted.y.len()];
        let cfg = Config::default();
        let tree = node(Op::Div, vec![input(0), constant(0.0)]);
        let data = EvalSample::new(&s, &cfg);
        assert!(prefix_is_non_finite(&tree, &data, &cfg));
        assert!(!prefix_is_non_finite(&tree, &EvalSample::new(&weighted, &cfg), &cfg));
        let mut qd = Config::default();
        qd.qd_bins = 4;
        assert!(!prefix_is_non_finite(&tree, &EvalSample::new(&s, &qd), &qd));

        let aggs = AggTable::from_sample(&s);
        let mut cfg = Config::default();
        cfg.pop = 40;
        let data = EvalSample::new(&s, &cfg);
        let mut isl = inf_population(&data, &cfg);
        let counters = GenCounters::default();
        evolve_island(
            &mut isl, &mut Mt::new(5), &cfg, &data, &s, 1, &aggs, false, false, &[], 0.0, None,
            false, 0, None, Some(&counters), 1, &mut MixWindow::new(),
        );
        assert_eq!(GenCounters::total(Some(&counters), |c| &c.reevals), 0);
    }

    /// #465: migration replaces the member of the highest Pareto rank.
    #[test]
    fn afp_worst_is_the_highest_rank() {
        let island = vec![
            aged_ind(input(0), 1.0, 1.0, 0),
            aged_ind(input(0), 5.0, 9.0, 7),
            aged_ind(input(0), 2.0, 2.0, 1),
        ];
        assert_eq!(afp_worst(&island), Some(1));
    }

    fn affine_sample() -> Sample {
        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64 * 0.1 - 3.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 1.0).collect();
        Sample::new(x, y)
    }

    fn scaling_cfg(strategy: crate::fit::FitStrategy) -> Config {
        let mut cfg = Config::default();
        cfg.metric = Metric::Squared;
        cfg.fit_iters = 8;
        cfg.fit_strategy = strategy;
        cfg.linear_scaling = true;
        cfg
    }

    /// #470: a constant-free shape reaches the exact affine map in one fit call.
    #[test]
    fn linear_scaling_recovers_affine_exactly() {
        let s = affine_sample();
        for strategy in [crate::fit::FitStrategy::Exact, crate::fit::FitStrategy::Auto] {
            let cfg = scaling_cfg(strategy);
            let mut tree = crate::scale::wrap(0.0, 1.0, expr::input(0));
            let fit = fit_constants(&mut tree, &cfg, &s, &s).expect("a fit result");
            assert!(fit.sse <= 1e-20, "{:?}: sse {}", strategy, fit.sse);
            let sse: f64 = s.x.iter().zip(&s.y).map(|(r, y)| (eval(&tree, r) - y).powi(2)).sum();
            assert!(sse <= 1e-20, "{:?}: phenotype sse {}", strategy, sse);
        }
    }

    /// #470: wrap, fit, split and re-wrap keep the constants bit for bit.
    #[test]
    fn linear_scaling_split_roundtrip() {
        let s = affine_sample();
        let cfg = scaling_cfg(crate::fit::FitStrategy::Exact);
        let shape = expr::node(Op::Sin, vec![expr::input(0)]);
        let mut tree = crate::scale::wrap(0.3, 0.7, shape);
        fit_constants(&mut tree, &cfg, &s, &s);
        let (a, b, f) = crate::scale::split(&tree);
        let again = crate::scale::wrap(a, b, f);
        let (a2, b2, _) = crate::scale::split(&again);
        assert_eq!(a.to_bits(), a2.to_bits());
        assert_eq!(b.to_bits(), b2.to_bits());
        assert_eq!(again, tree);
    }

    /// #470: a short search with the map finds `2x + 1` and reports the wrapped
    /// formula; the same search without it does not change.
    #[test]
    fn linear_scaling_search_reports_the_wrapped_formula() {
        let s = affine_sample();
        let mut cfg = scaling_cfg(crate::fit::FitStrategy::Exact);
        cfg.generations = 6;
        cfg.pop = 40;
        cfg.seed = 3;
        cfg.gpu = false;
        let out = run(&cfg, &s, &s);
        let best = out
            .archive
            .items
            .iter()
            .map(|m| m.error)
            .fold(f64::INFINITY, f64::min);
        assert!(best <= 1e-18, "best error {}", best);
    }
}
